# -*- coding: utf-8 -*-
# encodermap/misc/backmapping.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade, Tobias Lemke
#
# Encodermap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
# This package is distributed in the hope that it will be useful to other
# researches. IT DOES NOT COME WITH ANY WARRANTY WHATSOEVER; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# See <http://www.gnu.org/licenses/>.
################################################################################
"""Backmapping functions to create new atomistic conformations from intrinsic
coordinates.

"""

################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import os.path
from contextlib import contextmanager
from copy import deepcopy
from math import pi
from pathlib import Path

# Third Party Imports
import numpy as np
import tensorflow as tf
import transformations
from optional_imports import _optional_import
from tqdm import tqdm as normal_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from transformations import rotation_matrix as transformations_rotation_matrix

# Encodermap imports
from encodermap.loading import features
from encodermap.misc.misc import _is_notebook
from encodermap.misc.rotate import _dihedral, _get_near_and_far_networkx
from encodermap.trajinfo.info_all import TrajEnsemble
from encodermap.trajinfo.info_single import SingleTraj


# numba to accelerate
try:
    # Third Party Imports
    from numba import jit

    _NUMBA_AVAILABLE: bool = True
except ImportError:
    _NUMBA_AVAILABLE: bool = False


################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")
xr = _optional_import("xarray")
mda = _optional_import("MDAnalysis")
AnalysisFromFunction = _optional_import(
    "MDAnalysis", "analysis.base.AnalysisFromFunction"
)
MemoryReader = _optional_import("MDAnalysis", "coordinates.memory.MemoryReader")
jit = _optional_import("numba", "jit")
nb = _optional_import("numba")
nx = _optional_import("networkx")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, overload


if TYPE_CHECKING:
    # Standard Library Imports
    from io import BytesIO

    # Third Party Imports
    import MDAnalysis as mda
    import mdtraj as md
    import networkx as nx
    from MDAnalysis.analysis.base import AnalysisFromFunction
    from MDAnalysis.coordinates.memory import MemoryReader

    # Local Folder Imports
    from ..trajinfo import SingleTraj, TrajEnsemble


################################################################################
# Globals
################################################################################


__all__: list[str] = ["backbone_hydrogen_oxygen_crossproduct", "mdtraj_backmapping"]


################################################################################
# Helpers
################################################################################


@contextmanager
def temp_seed(seed):
    """Within this context manager, the random state of NumPy is fixed."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _raise_components_exception(
    components: Sequence[nx.Graph],
    trajs: TrajEnsemble,
    top: Optional[md.Topology] = None,
    remove_component_size: int = 0,
) -> None:
    """Raises a descriptive error message for the components upon breaking the
    connection graph of a protein topology."""
    smallest_component = sorted(components, key=lambda x: len(x))[0]
    largest_component = sorted(components, key=lambda x: len(x))[1]
    if top is None:
        _str = f"from the provided {trajs.__class__.__name__}"
    elif isinstance(top, (int, np.int64)):
        _str = f"from `SingleTraj` number {top} of the provided `TrajEnsemble`."
    else:
        _str = f"from the provided {top} file"
    msg = (
        f"The protein {_str} is disconnected. Changing dihedrals "
        f"in multiple disconnected chains is currently not possible. You can also "
        f"use `em.plot.plot_ball_and_stick(traj highlight='bonds')` to see which "
        f"bonds are present in the topology. If you are sure "
        f"your protein is just one chain you can try to load a custom topology "
        f"or provide a topology with manually fixed bonds. I got {len(components)} "
        f"disconnected components. The smallest component contains these atoms: "
        f"{smallest_component=} {largest_component=}. You can try and add "
        f"these bonds with the `custom_aas` keyword to this function call. If "
        f"these components contain unwanted residues like solvent or salt-ions, "
        f"you can set `remove_component_size` to a number representing the "
        f"sizes of these components (i.e. 3 for three atom water) to remove "
        f"these components from the trajectory."
    )
    if remove_component_size > 0:
        msg += (
            f" Your currently chosen `remove_component_size`={remove_component_size} "
            f"is not large enough to reduce the system to only one connected component."
        )
    raise Exception(msg)


################################################################################
# Public Functions
################################################################################


def split_and_reverse_dihedrals(x: tf.Tensor) -> tf.Tensor:
    """Splits dihedrals in BackMapping model into left (reversed) and right part.
    These dihedrals are then used to bring the chain_in_plane into 3D.

    Args:
        x (tf.Tensor): The dihedrals with shape (None, n_reisudes * 3 - 3)

    Examples:
        >>> from encodermap.misc.backmapping import split_and_reverse_dihedrals
        >>> import numpy as np

        >>> # create dihedrals for protein with 3 resiudes, i.e. 3*3 - 3  = 6 central dihedral angles
        >>> # single sample will be used -> shape = (1, 6)
        >>> np.random.seed(20)
        >>> dihedrals = np.random.random((1, 6)) * 2 * np.pi
        >>> print(dihedrals)
        [[3.69533481 5.64050171 5.60165278 5.12605805 0.22550092 4.34644107]]

        >>> dihedrals_left, dihedrals_right = split_and_reverse_dihedrals(dihedrals)
        >>> print(dihedrals_left, dihedrals_right)
        [[5.60165278 5.64050171 3.69533481]] [[5.12605805 0.22550092 4.34644107]]

    """
    middle = int(int(x.shape[1]) / 2)
    cond = tf.math.equal(tf.math.mod(x.shape[1], 2), 0)
    return tf.cond(
        cond,
        true_fn=lambda: (
            x[:, middle - 1 :: -1],
            x[:, middle:],
        ),  # , middle, middle),
        false_fn=lambda: (
            x[:, middle::-1],
            x[:, middle + 1 :],
        ),  # , middle + 1, middle),
    )


def split_and_reverse_cartesians(x):
    """Splits cartesians and returns a left (reversed) right part.

    Because dihedrals are made up from 4 atoms, three atoms are
    identical in the left and right part of the list. This holds true:
    left[0] = right[2]
    left[1] = right[1]
    left[2] = right[0]

    Args:
        x (tf.Tensor): The cartesians with shape (None, n_reisudes * 3, 3)

    Examples:
        >>> from encodermap.misc.backmapping import split_and_reverse_cartesians
        >>> import numpy as np

        >>> # create cartesians for protein with 3 resiudes, i.e. 9
        >>> # single sample will be used -> shape = (1, 9, 3)
        >>> np.random.seed(20)
        >>> cartesians = np.random.random((1, 9, 3)) * 10

        >>> cartesians_left, cartesians_right = split_and_reverse_cartesians(cartesians)

        >>> print(cartesians_left.shape, cartesians_right.shape)
        (1, 6, 3) (1, 6, 3)

        >>> print(cartesians_left[:,0] == cartesians_right[:,2])
        [[ True  True  True]]

        >>> print(cartesians_left[:,1] == cartesians_right[:,1])
        [[ True  True  True]]

        >>> print(cartesians_left[:,2] == cartesians_right[:,0])
        [[ True  True  True]]



    """
    split = int(int(x.shape[1]) / 2)
    return x[:, split + 1 :: -1], x[:, split - 1 :]


def dihedrals_to_cartesian_tf_layers(
    dihedrals: tf.Tensor,
    cartesians: tf.Tensor,
    left_iteration_counter: int,
    right_iteration_counter: int,
) -> tf.Tensor:
    """Calculates dihedrals to cartesians in Graph/Layer execution.

    Args:
        dihedrals (tf.Tensor): The dihedrals of shape (None, n_resides * 3 - 3)
        cartesians (tf.Tensor): The cartesians of shape (None, n_residues * 3, 3).
        left_iteration_counter (int): The range(left_iteration_counter) for
            iteration over the left split of the cartesians chain. Needs to
            be supplied, because the shape of the SymbolicTensor is not known
            to tensorflow.
        right_iteration_counter (int): The range(right_iteration_counter) for
            iteration over the left split of the cartesians chain. Needs to
            be supplied, because the shape of the SymbolicTensor is not known
            to tensorflow.

    Returns:
        tf.Tensor: The finished 3d chain.

    """
    if len(cartesians.get_shape()) == 2:
        # if a single line of cartesians is passed, it is repeated to match the number of dihedrals
        cartesians = tf.tile(
            tf.expand_dims(cartesians, axis=0), [tf.shape(dihedrals[0], 1, 1)]
        )

    # split and reverse so that the center of the molecule stays on the 2D plane
    # and the left and right ends curl into the 3rd dimension
    cartesians_left, cartesians_right = split_and_reverse_cartesians(cartesians)
    dihedrals_left, dihedrals_right = split_and_reverse_dihedrals(dihedrals)

    new_cartesians_left = dihedral_to_cartesian_tf_one_way_layers(
        dihedrals=dihedrals_left,
        cartesian=cartesians_left,
        n=left_iteration_counter,
    )
    new_cartesians_right = dihedral_to_cartesian_tf_one_way_layers(
        dihedrals=dihedrals_right,
        cartesian=cartesians_right,
        n=right_iteration_counter,
    )

    new_cartesians = tf.concat(
        [new_cartesians_left[:, ::-1], new_cartesians_right[:, 3:]], axis=1
    )

    return new_cartesians


@jit(nopython=True)
def _displacement_jit(xyz: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Faster nopython numba-jit compiled version of calculating displacements.

    Args:
        xyz (np.ndarray): Numpy array with shape (n_atoms, 3).
        index (np.ndarray): Numpy array with shape (n_pairs, 2) and of type int,
            indexing the displacements calculated between two atoms.

    Returns:
        np.ndarray: The displacements of shape (n_pairs, 1).

    """
    return xyz[index[1]] - xyz[index[0]]


@jit(nopython=True)
def _dihedral_jit(
    xyz: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Faster nopython numba-jit compiled implementation of calculating a dihedral.

    Args:
        xyz (np.ndarray): Numpy array with shape (n_atoms, 3).
        indices (np.ndarray): Numpy array with shape (n_dihedrals, 4) and of type int,
            indexing the dihedrals defined between four atoms.

    Returns:
        np.ndarray: The displacements of shape (n_dihedrals, 1).

    """
    b1 = _displacement_jit(xyz, indices[0:2])
    b2 = _displacement_jit(xyz, indices[1:3])
    b3 = _displacement_jit(xyz, indices[2:4])
    c1 = np.cross(b2, b3)
    c2 = np.cross(b1, b2)
    p1 = (b1 * c1).sum(-1)
    p1 *= (b2 * b2).sum(-1) ** 0.5
    p2 = (c1 * c2).sum(-1)
    return np.arctan2(p1, p2)


@jit(nopython=True)
def _rotmat_jit(
    angle: np.float32,
    direction: np.ndarray,
    pivot_point: np.ndarray,
) -> np.ndarray:
    """Same as :func:`rotation_matrix`, but jit compiled."""
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction_unit = direction / (direction**2).sum() ** 0.5
    R = np.identity(3, dtype="float32")
    R *= cosa
    R += np.outer(direction_unit, direction_unit) * (1.0 - cosa)
    direction_unit *= sina
    R += np.array(
        [
            [0.0, -direction_unit[2], direction_unit[1]],
            [direction_unit[2], 0.0, -direction_unit[0]],
            [-direction_unit[1], direction_unit[0], 0.0],
        ],
        dtype="float32",
    )
    M = np.identity(4, dtype="float32")
    M[:3, :3] = R
    M[:3, 3] = pivot_point - np.dot(R, pivot_point)
    return M


@jit(nopython=True, parallel=True)
def parallel_rotation_application(
    xyz: np.ndarray,
    dihedral_indices: np.ndarray,
    dihedrals: np.ndarray,
    new_and_far_sides: np.ndarray,
) -> None:
    for j in range(
        dihedrals.shape[1]
    ):  # cannot be parallelized because the later angles depend on the previous
        for i in nb.prange(
            dihedrals.shape[0]
        ):  # can be parallelized because every frame can be treated separately
            target_angle = dihedrals[i, j]
            dihedral_index = dihedral_indices[i]
            current_angle = _dihedral_jit(xyz[i], dihedral_index)
            angle = target_angle - current_angle
            direction = xyz[i, dihedral_index[2]] - xyz[i, dihedral_index[1]]
            pivot_point = xyz[i, dihedral_index[0]]
            M = _rotmat_jit(angle, direction, pivot_point)
            padded = np.ones((len(xyz[i][~new_and_far_sides[i]]), 4), dtype="float32")
            padded[:, :3] = xyz[i][~new_and_far_sides[i]]
            xyz[i][~new_and_far_sides[i]] = M.dot(padded.T).T[:, :3]


@overload
def _full_backmapping_np(
    feature_description: dict[int, dict[int, Sequence[int]]],
    *inputs: np.ndarray,
    return_indices: Literal[False],
) -> tuple[np.ndarray, BytesIO]: ...


@overload
def _full_backmapping_np(
    feature_description: dict[int, dict[int, Sequence[int]]],
    *inputs: np.ndarray,
    return_indices: Literal[True],
) -> tuple[np.ndarray, BytesIO, dict[str, np.ndarray]]: ...


def _full_backmapping_np(
    feature_description: dict[int, dict[int, Sequence[int]]],
    *inputs: np.ndarray,
    return_indices: bool = False,
) -> Union[
    tuple[np.ndarray, BytesIO], tuple[np.ndarray, BytesIO, dict[str, np.ndarray]]
]:
    """Also BackMaps sidechains. For that, we need a way to know which
    distances, angles, dihedrals belong to the backbone, and which belong to
    a sidechain. In contrast to how cartesians are normally represented in
    MD trajectories, the cartesians in this class are ordered by first all
    central cartesians, then all sidechain cartesians. Consider a theoretical
    protein with three residues: MET-LYS-GLY. The protein's backbone has
    9 cartesian coordiantes, 8 distances, 7 angles, and 6 dihedrals.
    Methionine has 3 sidechain angles (ch1, chi2, chi3), Lysine has 4 sidechain
    angles (chi1, ..., chi4), and Glycine has no sidechain angles. The
    feature_description dict for this protein should be:
    {-1: {1: 3, 2: 4, 3: 0}.

    * Cartesians:
        The cartesian coordinates of the sidechain atoms is an array with shape
        `(sum([v + 1 for v in feature_description[-1].values() if v > 0])`. So for
        this theoretical protein, we have 4 sidechain coordinates for MET and 5 for LYS.
        The combined cartesians array first contains all backbones, and then follows
        up with the sidechain positions as defined in the feature_description dict.
    * Distances:
        The distances between the sidechain atoms include the CA of the backbone,
        thus we have `(sum([v + 1 for v in feature_description[-1].values() if v > 0])`
        sidechain distances. MET has 4 sidechain distances, LYS has 5.
    * Angles:
        The angles between the sidechain atoms include the CA and N of the backbone.
        We have `(sum([v + 1 for v in feature_description[-1].values() if v > 0])`
        angles. MET has 4 sidechain angles (N-CA-CB, CA-CB-CG, CB-CG-SD, CG-SD-CE)
        and LYS has 5 sidechain angles.
    * Dihedrals:
        The dihedrals between the sidechain atoms include the CA and N of the
        backbone, so that we have `(sum(list(feature_description[-1].values()))`
        sidechain dihedrals.

    Indexing these features is done via boolean arrays. True values are kept
    stationary. False values are allowed to move. Indexing the central distances
    can be done with a triangular matrix with 8 rows, corresponding to the
    8 distances in our example protein: ::

          METN,  METCA, METC,  LYSN,  LYSCA, LYSC,  GLYN,  GLYCA, GLYC
        0 True,  False, False, False, False, False, False, False, False
        1 True,  True,  False, False, False, False, False, False, False
        2 True,  True,  True,  False, False, False, False, False, False
        3 True,  True,  True,  True,  False, False, False, False, False
        4 True,  True,  True,  True,  True,  False, False, False, False
        5 True,  True,  True,  True,  True,  True,  False, False, False
        6 True,  True,  True,  True,  True,  True,  True,  False, False
        7 True,  True,  True,  True,  True,  True,  True,  True,  False

    Indexing the sidechain distances can be done with a matrix with small
    triangular matrices like this. For our example protein we have 9 such
    indices for the 9 sidechain distances ::

           METCA, METCB, METCG, METSD, METCE, LYSCA, LYSCB, LYSCG, LYSCD, LYSCE, LYSNZ
        0  True,  False, False, False, False, False, False, False, False, False, False
        1  True,  True,  False, False, False, False, False, False, False, False, False
        2  True,  True,  True,  False, False, False, False, False, False, False, False
        3  True,  True,  True,  False, False, False, False, False, False, False, False
        4  True,  True,  True,  True,  False, False, False, False, False, False, False
        5  False, False, False, False, False, True,  False, False, False, False, False
        6  False, False, False, False, False, True,  True,  False, False, False, False
        7  False, False, False, False, False, True,  True,  True,  False, False, False
        8  False, False, False, False, False, True,  True,  True,  True,  False, False
        9  False, False, False, False, False, True,  True,  True,  True,  True,  False
        10 False, False, False, False, False, True,  True,  True,  True,  True,  True

    However, to keep the CA-atoms from appearing twice, the resulting array needs to
    ditch the CA columns of the side_distances and add True, where needed. The algorithm
    for the right side of the central_distances is row 0: all false, every three rows
    add True for the number of sidechain atoms. The last row is just True. The left
    side for the sidechain distances (without CA) are just True. We keep the
    chain immovable for adjusting these lengths. For
    our protein the resulting array looks like this: ::

          0,     1,     2,     3,     4,     5,     6,     7,     8,     9,     10,    11,    12,    13,    14,    15,    16,    17
          METN,  METCA, METC,  LYSN,  LYSCA, LYSC,  GLYN,  GLYCA, GLYC,  METCB, METCG, METSD, METCE, LYSCB, LYSCG, LYSCD, LYSCE, LYSNZ
        0 True,  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False
        1 True,  True,  False, False, False, False, False, False, False, True,  True,  True,  True,  False, False, False, False, False
        2 True,  True,  True,  False, False, False, False, False, False, True,  True,  True,  True,  False, False, False, False, False
        3 True,  True,  True,  True,  False, False, False, False, False, True,  True,  True,  True,  False, False, False, False, False
        4 True,  True,  True,  True,  True,  False, False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True
        5 True,  True,  True,  True,  True,  True,  False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True
        6 True,  True,  True,  True,  True,  True,  True,  False, False, True,  True,  True,  True,  True,  True,  True,  True,  True
        7 True,  True,  True,  True,  True,  True,  True,  True,  False, True,  True,  True,  True,  True,  True,  True,  True,  True
        -----------------------
        8 True,  True,  True,  True,  True,  True,  True,  True,  True,  False, False, False, False, True,  True,  True,  True,  True
        9 True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  False, False, False, True,  True,  True,  True,  True
       10 True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  False, False, True,  True,  True,  True,  True
       11 True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  False, True,  True,  True,  True,  True
       12 True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False
       13 True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  False, False, False, False
       14 True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  False, False, False
       15 True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  False, False
       16 True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  False

    Indexing the angles is similar to the distances. For an angle, we need
    to index the left side (including pivot point) and the right side
    (excluding the pivot point). For the central chain, this can be done by
    omitting the first row of the distance indices. The sidechain indices can
    be left as is. They contain the correct indices for rotation. Take the angle
    METN-METCA-METCB as an example. We need the whole central chain to stay still,
    while METCB, METCG, METSD, METCE are moving. That is exactly row 8 of
    the array above. The rotation axis for this chain, which is currently just
    in 2D is the z-axis: (0, 0, 1).

    Indexing the dihedrals is similar to the angles. For a dihedral, we need to
    define a pivot point: The first atom of the bond (the second atom of the four-tuple
    defining the dihedral). Also, we need a rotation axis, which is the vector
    connecting the two atoms. Whether the second atom is part of the movable
    group is trivial, as it won't be rotated because it lies within the rotation axis.
    The indices for the central dihedrals are just the same indices as the
    distances omitting the first and last rows of the distance indices. This
    coincides with the indices for the sidechain atoms. The first dihedral
    angle of any protein is the psi1 angle between the N-terminal nitrogen
    N-CA-C-N. In this case, the rotational axis is between CA-C and thus, the
    sidechain (CB, etc.) needs to stay still for this dihedral. This is exactly
    what row 1 of our index array describes. The next dihedrals, omega1, and phi1,
    keep these indices of the sidechains, while the indices increase in the
    central part of the indices (rows 2 and 3). The next dihedral, psi2 will keep
    the sidechain dihedrals of the next residue stationary, as now the CA is on
    the far side of the rotation. The sidechain dihedral rotations are similar,
    however, the last row for every residue's sidechain must be omitted.

    Args:
        feature_description (dict[int, dict[int, Sequence[int]]]): A dictrionary
            that defines the topological hull of the proteins.
        *inputs (np.ndarray): The inputs in the order central_distances,
            central_angles, central_dihedrals, side_distances, side_angles,
            side_dihedrals.
        return_indices (bool): Whether to also return a dict of np.ndarray, to be
            used as indices.

    Returns:
        Union[tuple[np.ndarray, BytesIO], tuple[np.ndarray, BytesIO, dict[str, np.ndarray]]]:
            Either a tuple of the reconstructed coordinates in a numpy array and
            a BytesIO object containing png data. Or a tuple of the former two,
            with a index dictionary.

    """
    ############################################################################
    # Init
    ############################################################################

    # Imports
    # Standard Library Imports
    import itertools
    from io import BytesIO

    # Third Party Imports
    import matplotlib.pyplot as plt
    from scipy.linalg import block_diag
    from transformations import rotation_matrix

    # Encodermap imports
    from encodermap.misc.rotate import _dihedral

    # Definitions and Tests
    n_residues: int = max(list(feature_description[-1].keys()))
    assert np.array_equal(
        np.arange(1, n_residues + 1),
        np.sort(np.asarray(list(feature_description[-1].keys()))),
    ), (
        f"Currently the `feature_indices[-1]` dict needs to contain monotonous "
        f"increasing keys. Starting from 1 {feature_description[-1].keys()=}"
    )
    n_sidechains: int = sum([v + 1 for v in feature_description[-1].values() if v > 0])
    sum_sidechains = sum(list(feature_description[-1].values()))

    # this can be defined beforehand and then stacked as often, as a batch needs it
    init_xyz: np.ndarray = np.zeros(
        shape=(1, n_residues * 3 + n_sidechains, 3),
        dtype="float32",
    )

    # first we create the central_distance indices
    central_distance_indices = np.tri(
        N=n_residues * 3 - 1,
        M=n_residues * 3,
        k=0,
    ).astype(bool)
    right_side_central_distance_indices = [
        np.full(shape=(1, n_sidechains), fill_value=False, dtype=bool)
    ]
    count = 0  # starts at the first atom of the central chan
    count2 = n_residues * 3 + 1  # starts at the first atom of the sidechain
    sidechain_cartesians_ind = []
    sidechain_positions_indices = []
    central_distance_index_duplets = np.vstack(
        [
            np.arange(0, n_residues * 3)[:-1],
            np.arange(0, n_residues * 3)[1:],
        ]
    ).T.tolist()
    sidechain_distance_index_duplets = []
    central_angle_index_triplets = np.vstack(
        [
            np.arange(0, n_residues * 3)[:-2],
            np.arange(0, n_residues * 3)[1:-1],
            np.arange(0, n_residues * 3)[2:],
        ]
    ).T.tolist()
    sidechain_angle_index_triplets = []
    central_dihedral_index_quadruplets = np.vstack(
        [
            np.arange(0, n_residues * 3)[:-3],
            np.arange(0, n_residues * 3)[1:-2],
            np.arange(0, n_residues * 3)[2:-1],
            np.arange(0, n_residues * 3)[3:],
        ]
    ).T.tolist()
    sidechain_dihedral_index_quadruplets = []

    # iterate over feature_description[-1] to get all indices and the right side
    # of the central cartesians
    for i, (residue, n_sidechains_in_residue) in zip(
        itertools.count(1, 3), feature_description[-1].items()
    ):
        if n_sidechains_in_residue == 0:
            if residue == 1 or residue == n_residues:
                continue
            else:
                right_side_central_distance_indices.append(t)
        else:
            sidechain_cartesians_ind.append(
                np.arange(count, count + n_sidechains_in_residue)
            )
            sidechain_positions_indices.append(
                [i] + np.arange(count2 - 1, count2 + n_sidechains_in_residue).tolist()
            )
            for sidechain_i in range(n_sidechains_in_residue + 1):
                if sidechain_i == 0:
                    # adds CA-CB
                    sidechain_distance_index_duplets.append(
                        [(residue - 1) * 3 + 1, count2 - 1]
                    )
                    # adds N-CA-CB
                    sidechain_angle_index_triplets.append(
                        [(residue - 1) * 3, (residue - 1) * 3 + 1, count2 - 1]
                    )
                    # adds N-CA-CB-CG
                    sidechain_dihedral_index_quadruplets.append(
                        [(residue - 1) * 3, (residue - 1) * 3 + 1, count2 - 1, count2]
                    )
                elif sidechain_i == 1:
                    # adds CB-CG
                    sidechain_distance_index_duplets.append([count2 - 1, count2])
                    # adds CA-CB-CG
                    sidechain_angle_index_triplets.append(
                        [(residue - 1) * 3 + 1, count2 - 1, count2]
                    )
                    # adds CA-CB-CG-CD
                    if sidechain_i < n_sidechains_in_residue:
                        sidechain_dihedral_index_quadruplets.append(
                            [(residue - 1) * 3 + 1, count2 - 1, count2, count2 + 1]
                        )
                else:
                    # adds CG-CD and so on
                    sidechain_distance_index_duplets.append(
                        [count2 + sidechain_i - 2, count2 + sidechain_i - 1]
                    )
                    # adds CB-CG-CD and so on
                    sidechain_angle_index_triplets.append(
                        [
                            count2 + sidechain_i - 3,
                            count2 + sidechain_i - 2,
                            count2 + sidechain_i - 1,
                        ]
                    )
                    if sidechain_i < n_sidechains_in_residue:
                        sidechain_dihedral_index_quadruplets.append(
                            [
                                count2 + sidechain_i - 3,
                                count2 + sidechain_i - 2,
                                count2 + sidechain_i - 1,
                                count2 + sidechain_i,
                            ]
                        )
            count += n_sidechains_in_residue + 1
            count2 += n_sidechains_in_residue + 1
            t = np.zeros(
                shape=(3, n_sidechains),
                dtype=bool,
            )
            t[:, :count] = True
            right_side_central_distance_indices.append(t)
    assert len(sidechain_angle_index_triplets) == n_sidechains
    assert len(sidechain_dihedral_index_quadruplets) == sum_sidechains, (
        f"I could not reconstruct the correct number of sidechain dihedral "
        f"quadruplets. The number of sidechain dihedrals requires the list "
        f"to have length {sum_sidechains}, but I created a list with "
        f"{len(sidechain_dihedral_index_quadruplets)}. The input has shape "
        f"{inputs[5].shape}."
    )
    right_side_central_distance_indices.append(
        np.full(shape=(1, n_sidechains), fill_value=True, dtype=bool)
    )
    right_side_central_distance_indices = np.vstack(right_side_central_distance_indices)
    angle_index_triplets = np.vstack(
        central_angle_index_triplets + sidechain_angle_index_triplets
    )
    dihedral_index_quadruplets = np.vstack(
        central_dihedral_index_quadruplets + sidechain_dihedral_index_quadruplets
    )
    if sidechain_cartesians_ind != []:  # if sidechains
        _use_sidechains = True
        sidechain_cartesians_ind = np.concatenate(sidechain_cartesians_ind)
        central_distance_indices = np.hstack(
            [central_distance_indices, right_side_central_distance_indices]
        )
        side_distance_indices = [
            (np.tri(N=i + 1, M=i + 2, k=0) + 1)[:, 1:]
            for i in feature_description[-1].values()
            if i > 0
        ]
        side_distance_indices = (block_diag(*side_distance_indices) % 2) == 0
        left_side_side_distance_indices = (
            np.full(  # all atoms in the central chain are True
                shape=(len(side_distance_indices), n_residues * 3),
                fill_value=True,
                dtype=bool,
            )
        )
        side_distance_indices = np.hstack(
            [left_side_side_distance_indices, side_distance_indices]
        )
        distance_indices = np.vstack([central_distance_indices, side_distance_indices])
    else:  # if no sidechains
        _use_sidechains = False
        distance_indices = central_distance_indices
    assert distance_indices.shape == (
        n_residues * 3 - 1 + n_sidechains,
        init_xyz.shape[1],
    ), (
        f"The shape of the distance index after stacking is unexpected.\n"
        f"Expected: {(n_residues * 3 - 1 + n_sidechains, init_xyz.shape[1])}\n"
        f"Actual: {distance_indices.shape}"
    )

    # now the angles
    central_angle_indices = central_distance_indices[1:]
    if _use_sidechains:  # if sidechains
        angle_indices = np.vstack([central_distance_indices[1:], side_distance_indices])
        side_angle_indices = side_distance_indices
    else:  # no sidechains
        angle_indices = central_distance_indices[1:]
    assert len(angle_indices) == len(distance_indices) - 1

    # and the dihedrals
    if _use_sidechains:  # if sidechains
        dihedral_indices = np.vstack(
            [
                central_distance_indices[1:-1],
                side_distance_indices[sidechain_cartesians_ind],
            ]
        )
        corrector = np.count_nonzero(
            list(feature_description[-1].values())
        )  # per reisude with sidechain dihedrals one less
    else:
        dihedral_indices = central_distance_indices[1:-1]
        corrector = 0
    assert len(dihedral_indices) == len(distance_indices) - 2 - corrector
    assert angle_index_triplets.shape[0] == angle_indices.shape[0]
    assert dihedral_index_quadruplets.shape[0] == dihedral_indices.shape[0], (
        f"The number of dihedral indices ({len(distance_indices)}) and quadruplets "
        f"does not match ({len(dihedral_index_quadruplets)}). I get "
        f"{inputs[2].shape[1] + inputs[5].shape[1]} in inputs."
    )

    ############################################################################
    # Call
    ############################################################################

    (
        central_distances,
        central_angles,
        central_dihedrals,
        side_distances,
        side_angles,
        side_dihedrals,
    ) = inputs

    assert all(
        len(i) == len(inputs[0]) for i in inputs[1:]
    ), f"Inhomogeneous input lengths: {[len(i) for i in inputs]}"
    distances = np.hstack([central_distances, side_distances])
    angles = np.hstack([central_angles, side_angles])
    dihedrals = np.hstack([central_dihedrals, side_dihedrals])
    assert distance_indices.shape[0] == distances.shape[1]
    assert angle_indices.shape[0] == angles.shape[1]
    assert dihedral_indices.shape[0] == dihedrals.shape[1]

    # for debug
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, subplot_kw={"projection": "3d"}, figsize=(20, 8)
    )
    buf = BytesIO()

    # copy the predefined array and make an array of quarternions
    if len(angles) > 1:
        xyz_out = np.repeat(init_xyz, len(angles), axis=0)
    else:
        xyz_out = init_xyz.copy()
    xyz_out = np.pad(
        xyz_out, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=1
    )

    # distances in more tensor-flow-friendly implementation
    # by creating a list and concatenating we can forego assignments whicha are
    # not supported for symbolic tensors
    xs_central = [np.zeros((len(angles),))]
    ys_central = [np.zeros((len(angles),))]
    xs_side = []
    ys_side = []

    residue = 0  # residue here is again, 0-based
    idx = 0
    j = 0
    n_sidechains_in_residue = np.array(
        [feature_description[-1][k] for k in sorted(feature_description[-1].keys())]
    ).astype(np.int32)
    for i in range(len(central_distance_indices)):
        assert np.all(central_distances[:, i] > 0)
        xs_central.append(xs_central[-1] + central_distances[:, i])
        ys_central.append(np.zeros((len(angles),)))
        if idx == 0 and _use_sidechains:
            n_sidechains = n_sidechains_in_residue[residue]
            if n_sidechains > 0:
                for n in range(n_sidechains + 1):
                    xs_side.append(xs_central[-1])
                    dists = side_distances[:, j - n : j + 1]
                    assert np.all(dists > 0), (
                        f"Side distances at (0-based) residue {residue} are smaller than 0. "
                        f"This is the {n} sidechain distance of a total of "
                        f"{n_sidechains +  1} sidechain distances. For that, I index "
                        f"the sidechain dists array for all frames from {j-n=} to "
                        f"{j+1=}, which gives {side_distances[:, j-n:j+1]} for the "
                        f"first 5 frames.."
                    )
                    _ = np.sum(dists, axis=1)
                    if np.any(np.isnan(_)):
                        raise Exception(
                            f"At index {i=} of the central distances and {j=} of "
                            f"the sidechain distances, the value of the y coordinate "
                            f"became NaN. The shapes of the arrays are:"
                            f"{central_distances.shape=} {side_distances.shape=} "
                            f"{np.any(np.isnan(central_distances))=} "
                            f"{np.any(np.isnan(side_distances))=}"
                        )
                    ys_side.append(_)
                    j += 1
        idx += 1
        if idx >= 3:
            residue += 1
            idx = 0
    xs = np.stack(xs_central + xs_side, axis=1)
    ys = np.stack(ys_central + ys_side, axis=1)
    xyz_out = np.stack([xs, ys, np.zeros(xs.shape), np.ones(xs.shape)], axis=2)
    assert not np.any(
        np.isnan(xyz_out)
    ), f"After fixing dists, some values in `xyz_out` are NaN."

    for frame, (
        central_frame_dists,
        side_frame_dists,
        central_frame_angles,
        side_frame_angles,
        frame_dihedrals,
    ) in enumerate(
        zip(central_distances, side_distances, central_angles, side_angles, dihedrals)
    ):
        # plot
        if frame == 0:
            ax1.plot(*xyz_out[0, : n_residues * 3, :3].T, "bo-")
            for ind in sidechain_positions_indices:
                ax1.plot(*xyz_out[0, ind, :3].T, "bo-")

        # angles
        for i, (ang, ind, angle_index) in enumerate(
            zip(
                central_frame_angles,
                central_angle_indices,
                central_angle_index_triplets,
            )
        ):
            direction = np.array([0, 0, 1]).astype("float32")
            pivot_point = xyz_out[frame, angle_index[1], :3]
            a, b, c = xyz_out[frame, angle_index, :3]
            ba = a - b
            bc = c - b
            prod = np.linalg.norm(ba) * np.linalg.norm(bc)
            cosine_angle = np.clip(np.dot(ba, bc) / prod, -1, 1)
            current_angle = np.arccos(cosine_angle)
            angle = np.abs(ang - current_angle)
            rotmat = rotation_matrix(
                angle=angle, direction=direction, point=pivot_point
            )
            rotated = rotmat.dot(xyz_out[frame, ~ind].T).T[:, :3]
            xyz_out[frame, ~ind, :3] = rotated

        if _use_sidechains:
            for i, (ang, ind, angle_index) in enumerate(
                zip(
                    side_frame_angles,
                    side_angle_indices,
                    sidechain_angle_index_triplets,
                )
            ):
                direction = np.array([0, 0, -1]).astype("float32")
                pivot_point = xyz_out[frame, angle_index[1], :3]
                a, b, c = xyz_out[frame, angle_index, :3]
                ba = a - b
                bc = c - b
                cosine_angle = np.clip(
                    np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1
                )
                current_angle = np.arccos(cosine_angle)
                angle = np.abs(ang - current_angle)
                rotmat = rotation_matrix(
                    angle=angle, direction=direction, point=pivot_point
                )
                rotated = rotmat.dot(xyz_out[frame, ~ind].T).T[:, :3]
                xyz_out[frame, ~ind, :3] = rotated

        # plot
        if frame == 0:
            ax2.plot(*xyz_out[0, : n_residues * 3, :3].T, "bo-")
            for ind in sidechain_positions_indices:
                ax2.plot(*xyz_out[0, ind, :3].T, "bo-")

        # dihedrals
        for i, (d_ang, ind, dihedral_index) in enumerate(
            zip(frame_dihedrals, dihedral_indices, dihedral_index_quadruplets)
        ):
            bond = dihedral_index[[1, 2]]
            a, b, c, d = xyz_out[frame, dihedral_index, :3]
            direction = np.diff(xyz_out[frame, bond, :3], axis=0).flatten()
            pivot_point = xyz_out[frame, bond[0], :3]
            current_dihedral = _dihedral(xyz_out[frame, :, :3], dihedral_index)[0][0]
            if np.any(np.isnan(current_dihedral)):
                current_dihedral = 0.0
            dihedral = d_ang - current_dihedral
            rotmat = rotation_matrix(
                angle=dihedral, direction=direction, point=pivot_point
            )
            xyz_out[frame, ~ind, :3] = rotmat.dot(xyz_out[frame, ~ind].T).T[:, :3]

        # plot
        if frame == 0:
            ax3.plot(*xyz_out[0, : n_residues * 3, :3].T, "bo-")
            for ind in sidechain_positions_indices:
                ax3.plot(*xyz_out[0, ind, :3].T, "bo-")

    fig.savefig(buf, format="png")
    buf.seek(0)
    if not return_indices:
        return xyz_out[..., :3], buf
    else:
        return (
            xyz_out[..., :3],
            buf,
            {
                "central_distance_indices": np.asarray(central_distance_index_duplets),
                "side_distance_indices": np.asarray(sidechain_distance_index_duplets),
                "central_angles_indices": np.asarray(central_angle_index_triplets),
                "side_angles_indices": np.asarray(sidechain_angle_index_triplets),
                "central_dihedrals_indices": np.asarray(
                    central_dihedral_index_quadruplets
                ),
                "side_dihedrals_indices": np.asarray(
                    sidechain_dihedral_index_quadruplets
                ),
            },
        )


def _residue_number() -> Iterator[int]:
    i = 0
    atoms = 0
    while True:
        yield i
        atoms += 1
        if atoms > 2:
            atoms = 0
            i += 1


def _alternating() -> Iterator[int]:
    i = 0
    while True:
        if i % 2 == 0:
            yield 1
        else:
            yield -1
        i += 1


@overload
def mdtraj_backmapping(
    top: Optional[Union[Path, str, int, md.Topology]],
    dihedrals: Optional[np.ndarray],
    sidechain_dihedrals: Optional[np.ndarray],
    trajs: Optional[Union[TrajEnsemble, SingleTraj]],
    remove_component_size: int,
    verify_every_rotation: bool,
    angle_type: Literal["degree", "radian"],
    omega: bool,
    guess_amid_atoms: bool,
    return_indices: Literal[False],
    parallel: bool,
    progbar: Optional[Any],
) -> md.Trajectory: ...


@overload
def mdtraj_backmapping(
    top: Optional[Union[Path, str, int, md.Topology]],
    dihedrals: Optional[np.ndarray],
    sidechain_dihedrals: Optional[np.ndarray],
    trajs: Optional[Union[TrajEnsemble, SingleTraj]],
    remove_component_size: int,
    verify_every_rotation: bool,
    angle_type: Literal["degree", "radian"],
    omega: bool,
    guess_amid_atoms: bool,
    return_indices: Literal[True],
    parallel: bool,
    progbar: Optional[Any],
) -> tuple[md.Trajectory, dict[str, np.ndarray]]: ...


def mdtraj_backmapping(
    top: Optional[Union[Path, str, int, md.Topology]] = None,
    dihedrals: Optional[np.ndarray] = None,
    sidechain_dihedrals: Optional[np.ndarray] = None,
    trajs: Optional[Union[TrajEnsemble, SingleTraj]] = None,
    remove_component_size: int = 0,
    verify_every_rotation: bool = False,
    angle_type: Literal["degree", "radian"] = "radian",
    omega: bool = True,
    guess_sp2_atoms: bool = True,
    return_indices: bool = False,
    parallel: bool = False,
    progbar: Optional[Any] = None,
) -> Union[md.Trajectory, tuple[md.Trajectory, dict[str, np.ndarray]]]:
    """Uses MDTraj and Christoph Gohlke's transformations.py to rotate the
    bonds in the provided topology.

    Todo:
        * Make this faster. Maybe write a C or FORTRAN implementation.

    General procedure:
        * Decide on which topology to use (if different topologies are in the
            `TrajEnsemble` class, the `dihedrals` and `sidechain_dihedrals` arrays
            need to be altered so that the correct dihedrals are used.
            Because EncoderMap is trained on a full input `dihedrals` and
            `sidechain_dihedrals` contain the dihedrals for the topology
            in `TrajEnsemble` with most of such angles. Some SingleTraj
            classes in TrajEnsemble might not contain all these angles if, for
            example, an amino acid has been modified the mutant contains more
            sidechain dihedrals than the wt. So the correct sidechain dihedrals
            for the wildtype need to be selected.
        * Get the indices of the far sides of the rotations. The graph is
            gradually broken apart and the longer sub-graphs are kept.
        * Extend the trajectory. The lengths of dihedrals and sidechain_dihedrals
            should match. The frame given by top will be duplicated
            len(dihedrals)-times.
        * Get the current angles. We know what the final angles should be,
            but now how far to rotate the bonds. This can be done by getting
            the difference between current and target angle.
        * Rotate the bonds. Using Christoph Gohlke's transformations.py,
            the rotation matrix is constructed and the array is padded
            with zeros to resemble an array of quaternions.

    Args:
        top (Optional[str]): The topology file to use.
        dihedrals (Optional[np.ndarray]): The dihedrals to put onto the trajectory.
            `len(dihedrals)` is number of frames of output trajectory.
            `dihedrals.shape[1]` needs to be the same as the number of dihedrals
            in the topology. Can be None, in which case dihedrals and
            sidechain dihedrals will be faked.
        sidechain_dihedrals (Optional[np.ndarray]):
            The sidechain dihedrals to put onto the trajectory.
            If None is provided, the sidechains are kept like they were in
            the topology. Defaults to None.
        trajs (Optional[em.TrajEnsemble, em.SingleTraj]): Encodermap TrajEnsemble
            class. It Can accelerate the loading of current dihedral angles.
            Checks if provided topology is part of trajs. Defaults to None.
        verify_every_rotation (bool): Whether the rotation succeeded.
        angle_type (Literal["degree", "radians"]): Whether input is in degrees. Input will be
            converted to radians. Defaults to False.
        omega (bool): Whether your input backbone dihedrals contain the omega angle.
        return_indices (bool): Whether to not only return the back-mapped
            trajectory, but also a dict of labels. This dict contains the keys:
                * 'dihedrals_labels'
                * 'generic_dihedrals_labels'
                * 'side_dihedrals_labels'
                * 'generic_side_dihedrals_labels'
            Which matches the indices of the returned dihedrals with the input
            MD structures in `top` and/or `trajs`. This can be useful to make
            sure that input dihedrals match output dihedrals. Why? Because there
            are some proline dihedrals that cannot be adjusted. They are filtered
            out before doing backmapping, and the indices give the names of all
            dihedrals that were adjusted. See the Example below.

    Examples:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> import encodermap as em
        >>> from pprint import pprint
        >>> output_dir = Path(
        ...     em.get_from_kondata(
        ...         "OTU11",
        ...         mk_parentdir=True,
        ...         silence_overwrite_message=True,
        ...     ),
        ... )
        >>> # assign how many backbone angles we need
        >>> traj = em.load(output_dir / "OTU11_wt_only_prot.pdb")
        >>> traj.load_CV("central_dihedrals")
        >>> n_angles = traj.central_dihedrals.shape[-1]
        >>> n_angles
        732
        >>> # create some fake dihedrals with a uniform distribution between -pi and pi
        >>> dihedrals = np.random.uniform(low=-np.pi, high=np.pi, size=(5, n_angles))
        >>> out, index = em.misc.backmapping.mdtraj_backmapping(
        ...     top=output_dir / "OTU11_wt_only_prot.pdb",
        ...     dihedrals=dihedrals,
        ...     remove_component_size=10,
        ...     return_indices=True,
        ... )
        >>> out = em.SingleTraj(out)
        >>> out.load_CV("central_dihedrals")
        >>> # Here you will see, what indicies were automatically dropped during backmapping
        >>> # They will be proline phi angles, as these angles can not be
        >>> # freely rotated
        >>> all_coords = set(out._CVs.coords["CENTRAL_DIHEDRALS"].values)
        >>> indexed_coords = set(index['dihedrals_labels'])
        >>> pprint(all_coords - indexed_coords)
        {'CENTERDIH PHI   RESID  PRO:   8 CHAIN 0',
         'CENTERDIH PHI   RESID  PRO:  70 CHAIN 0',
         'CENTERDIH PHI   RESID  PRO:  73 CHAIN 0',
         'CENTERDIH PHI   RESID  PRO:  80 CHAIN 0',
         'CENTERDIH PHI   RESID  PRO: 151 CHAIN 0',
         'CENTERDIH PHI   RESID  PRO: 200 CHAIN 0',
         'CENTERDIH PHI   RESID  PRO: 205 CHAIN 0',
         'CENTERDIH PHI   RESID  PRO: 231 CHAIN 0',
         'CENTERDIH PHI   RESID  PRO: 234 CHAIN 0',
         'CENTERDIH PHI   RESID  PRO: 238 CHAIN 0'}


    Raises:
        Exception: If the input seems like it is in degrees.
        Exception: If top is not part of the TrajEnsemble class provided in argument `trajs`.

    Returns:
        mdtraj.Trajectory: An MDTraj trajectory with the correct dihedrals/side-dihedrals.

    """
    # Third Party Imports
    import networkx as nx

    # if `dihedrals` is None we sample them from a random uniform distribution
    if dihedrals is None and sidechain_dihedrals is None:
        if trajs is not None:
            if top is None:
                top = 0
            if not hasattr(trajs, "central_dihedrals"):
                trajs.load_CV("central_dihedrals")
            if not hasattr(trajs, "side_dihedrals"):
                trajs.load_CV("side_dihedrals")
            with temp_seed(1):
                dihedrals = np.random.uniform(
                    low=-np.pi,
                    high=np.pi,
                    size=(10, trajs[top].central_dihedrals.shape[-1]),
                )
                sidechain_dihedrals = np.random.uniform(
                    low=-np.pi,
                    high=np.pi,
                    size=(10, trajs[top].side_dihedrals.shape[-1]),
                )
        elif top is not None and trajs is None:
            assert not isinstance(top, int) or isinstance(top, md.Topology), (
                f"When providing no `dihedrals` to sample fake dihedrals from "
                f"a random uniform distribution, you can't provide int or "
                f"md.Topology for `top`. Please provide a str or Path."
            )
            trajs = TrajEnsemble([top])
            top = 0
            trajs.load_CVs(["central_dihedrals", "side_dihedrals"])
            dihedrals = np.random.uniform(
                low=-np.pi,
                high=np.pi,
                size=(10, trajs[top].central_dihedrals.shape[-1]),
            )
            sidechain_dihedrals = np.random.uniform(
                low=-np.pi, high=np.pi, size=(10, trajs[top].side_dihedrals.shape[-1])
            )
        else:
            raise Exception(f"Please provide either a `top` or `trajs` argument.")

    # change and check the angles
    if angle_type == "radian":
        if np.any(dihedrals > np.pi):
            raise Exception(
                f"The argument `angle_type` is meant to specify, what angles "
                f"(radian or degree) are provided for the argument `dihedrals`. "
                f"This allows you to provide either to this function by just "
                f"specifying this argument. You specified {angle_type} but some "
                f"of your dihedrals are greater than pi."
            )
        if sidechain_dihedrals is not None:
            if np.any(sidechain_dihedrals > np.pi):
                raise Exception(
                    f"The argument `angle_type` is meant to specify, what angles "
                    f"(radian or degree) are provided for the argument `sidechain_dihedrals`. "
                    f"This allows you to provide either to this function by just "
                    f"specifying this argument. You specified {angle_type} but some "
                    f"of your sidechain dihedrals are greater than pi."
                )
    elif angle_type == "degree":
        if np.all(dihedrals <= np.pi):
            raise Exception(
                f"The argument `angle_type` is meant to specify, what angles "
                f"(radian or degree) are provided for the argument `dihedrals`. "
                f"This allows you to provide either to this function by just "
                f"specifying this argument. You specified {angle_type} but none "
                f"of your dihedrals were greater than pi: {dihedrals}"
            )
        dihedrals = np.deg2rad(dihedrals)
        if sidechain_dihedrals is not None:
            if np.all(sidechain_dihedrals <= np.pi):
                raise Exception(
                    f"The argument `angle_type` is meant to specify, what angles "
                    f"(radian or degree) are provided for the argument `sidechain_dihedrals`. "
                    f"This allows you to provide either to this function by just "
                    f"specifying this argument. You specified {angle_type} but none "
                    f"of your sidechain dihedrals were greater than pi."
                )
            sidechain_dihedrals = np.deg2rad(sidechain_dihedrals)
    else:
        raise Exception(
            f"Argument `angle_type` must be either 'radian' or 'degree', "
            f"you supplied: {angle_type}"
        )

    # make sure the input has the same shape along the "frame" axis, that will be created.
    if sidechain_dihedrals is not None:
        assert len(dihedrals) == len(sidechain_dihedrals), (
            f"The number of provided dihedrals ({len(dihedrals)}) and "
            f"sidechain dihedrals ({len(sidechain_dihedrals)}) must be the same."
        )

    # either top or trajs has to be not None
    if trajs is None:  # pragma: no cover
        if top is None:
            raise Exception(
                f"Please provide the path to a topology file"
                f"(.pdb, .gro) to use for backmapping."
            )
        elif isinstance(top, (str, Path)):
            inp_trajs = TrajEnsemble([top])
            inp_trajs.load_CVs(["central_dihedrals", "side_dihedrals"])
        elif isinstance(top, int):
            raise Exception(
                f"When providing an int for `top`, pleas also provide a `em.TrajEnsemble` "
                f"for argument `trajs.`"
            )
        else:
            raise ValueError(
                f"Argument `top` must be of type str, int, or None, "
                f"you provided: {type(top)}."
            )
    elif isinstance(trajs, SingleTraj) or trajs.__class__.__name__ == "SingleTraj":
        if isinstance(top, (int, Path, str)):
            print(
                "When providing `em.SingleTraj` for argument `trajs`, the argument "
                "`top` will be ignored."
            )
        assert "central_dihedrals" in trajs._CVs, (
            f"The provided traj, doesn't have the collective variable 'central_dihedrals' "
            f"loaded. Please load them by calling: `traj.load_CVs('all')."
        )
        if sidechain_dihedrals is not None:
            assert "side_dihedrals" in trajs._CVs, (
                f"The provided traj, doesn't have the collective variable 'central_dihedrals' "
                f"loaded. Please load them by calling: `traj.load_CVs('all')."
            )
        inp_trajs = trajs._gen_ensemble()
    elif isinstance(trajs, TrajEnsemble) or trajs.__class__.__name__ == "TrajEnsemble":
        assert "central_dihedrals" in trajs._CVs, (
            f"The provided traj, doesn't have the collective variable 'central_dihedrals' "
            f"loaded. Please load them by calling: `traj.load_CVs('all')."
        )
        if sidechain_dihedrals is not None:
            assert "side_dihedrals" in trajs._CVs, (
                f"The provided traj, doesn't have the collective variable 'central_dihedrals' "
                f"loaded. Please load them by calling: `traj.load_CVs('all')."
            )
        if isinstance(top, (str, Path)) and os.path.isfile(top):
            print(
                "When providing `em.TrajEnsemble` for argument `trajs`, the argument "
                "`top` will be ignored if it points to a file."
            )
        elif isinstance(top, str) and top in trajs.common_str:
            _trajs_index = trajs.trajs_by_common_str[top][0].traj_num
        elif top is None:
            _trajs_index = 0
        elif isinstance(top, (int, np.int64)):
            # _trajs_index = top
            _trajs_index = trajs.trajs_by_top[trajs.top[top]][0].traj_num
        else:
            raise ValueError(
                f"Argument `top` must be of type str, int, md.Topology or None, "
                f"you provided: {type(top)}."
            )
        inp_trajs = trajs.tsel[_trajs_index]._gen_ensemble()
    else:
        raise ValueError(
            f"Argument `trajs` must be of type `em.SingleTraj`, `em.TrajEnsemble`, or None, "
            f"you provided: {type(trajs)}."
        )

    if trajs is None:
        trajs = inp_trajs

    # now we match the names of the featurizer
    all_central_indices = trajs._CVs.central_dihedrals.coords[
        trajs._CVs.central_dihedrals.attrs["feature_axis"]
    ]
    central_indices = all_central_indices[
        np.all(~np.isnan(inp_trajs._CVs.central_dihedrals.values[0]), axis=0)
    ]
    if sidechain_dihedrals is not None:
        all_side_indices = trajs._CVs.side_dihedrals.coords[
            trajs._CVs.side_dihedrals.attrs["feature_axis"]
        ]
        side_indices = all_side_indices[
            np.all(~np.isnan(inp_trajs._CVs.side_dihedrals.values[0]), axis=0)
        ]
    if not omega:
        central_indices = central_indices[
            ~central_indices.str.lower().str.contains("omega")
        ]
        if sidechain_dihedrals is not None:
            side_indices = side_indices[~side_indices.str.lower().str.contains("omega")]

    generic_labels = not any(
        central_indices.coords["CENTRAL_DIHEDRALS"].str.contains("RESID")
    )

    if generic_labels:
        # we have generic labels and add non-generic ones
        _back_labels = {
            "generic_dihedrals_labels": central_indices.values,
            "dihedrals_labels": np.asarray(
                features.CentralDihedrals(inp_trajs[0], omega=omega).describe()
            ),
        }
        if sidechain_dihedrals is not None:
            _back_labels |= {
                "generic_side_dihedrals_labels": side_indices.values,
                "side_dihedrals_labels": np.asarray(
                    features.SideChainDihedrals(inp_trajs[0]).describe()
                ),
            }
    else:
        # we have non-generic labels and build generic ones
        _back_labels = {
            "dihedrals_labels": central_indices.values,
            "generic_dihedrals_labels": np.asarray(
                features.CentralDihedrals(
                    inp_trajs[0], omega=omega, generic_labels=True
                ).describe()
            ),
        }
        if sidechain_dihedrals is not None:
            _back_labels |= {
                "side_dihedrals_labels": side_indices.values,
                "generic_side_dihedrals_labels": np.asarray(
                    features.SideChainDihedrals(
                        inp_trajs[0], generic_labels=True
                    ).describe()
                ),
            }

    # check that all indices are present
    if len(all_central_indices) >= len(central_indices):
        if dihedrals.shape[1] == len(all_central_indices):
            dih_indices = np.arange(len(all_central_indices))[
                np.in1d(all_central_indices, central_indices)
            ]
            dihedrals = dihedrals[:, dih_indices]
        elif dihedrals.shape[1] == len(central_indices):
            dih_indices = np.arange(dihedrals.shape[1])
            dihedrals = dihedrals[:, dih_indices]
        else:
            raise Exception(
                f"The shape of the provided `dihedrals` is wrong, either provide "
                f"an array with shape[1] = {len(central_indices)}, or "
                f"{len(all_central_indices)}, "
                f"your array has the shape {dihedrals.shape[1]}."
            )
    else:
        raise NotImplementedError

    if sidechain_dihedrals is not None:
        if len(all_side_indices) >= len(side_indices):
            if sidechain_dihedrals.shape[1] == len(all_side_indices):
                _side_indices_out = side_indices.copy()
                side_indices = np.in1d(all_side_indices, side_indices)
                msg = (
                    f"Your supplied `sidechain_dihedrals` are misshaped. They are"
                    f"expected to have either shape (n, {len(side_indices)}, which"
                    f"matches the number of sidechain dihedrals in the specified"
                    f"topology: {inp_trajs[0].top} or a shape of (n, "
                    f"{len(all_side_indices)}) which matches the total number of "
                    f"possible sidechain angles in the provided `TrajEnsemble` with "
                    f"{trajs.top} different toplogies."
                )
                assert side_indices.shape[0] == sidechain_dihedrals.shape[-1], msg
                sidechain_dihedrals = sidechain_dihedrals[:, side_indices]
            elif sidechain_dihedrals.shape[1] == len(side_indices):
                _side_indices_out = side_indices.copy()
                side_indices = np.arange(sidechain_dihedrals.shape[1])
                sidechain_dihedrals = sidechain_dihedrals[:, side_indices]
            else:
                raise Exception(
                    f"The shape of the provided `dihedrals` is wrong, either provide "
                    f"an array with shape[1] = {len(side_indices)}, or {len(all_side_indices)}, "
                    f"your array has the shape {sidechain_dihedrals.shape[1]}."
                )
        else:
            raise NotImplementedError

    dih_indices = inp_trajs[0]._CVs.central_dihedrals_feature_indices.values[0]
    if omega:
        idx = ~np.all(np.isnan(dih_indices), axis=1)
    else:
        idx = (
            ~np.all(np.isnan(dih_indices), axis=1)
            & ~all_central_indices.str.lower().str.contains("omega")
        ).values
    dih_indices = dih_indices[idx]
    dih_indices = dih_indices.astype(int)
    _dih_indices = deepcopy(dih_indices)
    assert dih_indices.ndim == 2, f"Problem when calculating dihedrals {inp_trajs=}"
    side_indices = inp_trajs[0]._CVs.side_dihedrals_feature_indices.values[0]
    side_indices = side_indices[~np.all(np.isnan(side_indices), axis=1)]
    side_indices = side_indices.astype(int)
    _side_indices = deepcopy(side_indices)
    assert isinstance(inp_trajs, TrajEnsemble)

    # get indices of atoms for rotations
    g = inp_trajs.top[0].to_bondgraph()

    # can be used to visualize topology
    # nx.draw(g, pos=nx.spring_layout(g))
    if not nx.is_connected(g):
        # Third Party Imports
        from networkx import connected_components

        components = [*connected_components(g)]
        if remove_component_size > 0:
            component_sizes = sorted([len(c) for c in components])
            if any([i > remove_component_size for i in component_sizes[:-1]]):
                _raise_components_exception(
                    components, trajs, top, remove_component_size
                )
            offending_components = []
            for c in components:
                if len(c) <= remove_component_size:
                    offending_components.extend([a.index for a in list(c)])
            inp_trajs[0].atom_slice(offending_components, invert=True)
            g = inp_trajs.top[0].to_bondgraph()
        else:
            _raise_components_exception(components, trajs, top, remove_component_size)

    # at this point dih_bond_indices has been defined.
    dih_bond_indices = dih_indices[:, 1:3]
    assert (
        dih_bond_indices.shape[0] == _dih_indices.shape[0] == dihedrals.shape[1]
    ), f"{dih_bond_indices.shape=}, {_dih_indices.shape=}, {dihedrals.shape=} {omega=}"

    # filter out the proline angles
    dih_bond_atoms = np.dstack(
        [
            [inp_trajs[0].top.atom(a).__str__() for a in dih_bond_indices[:, 0]],
            [inp_trajs[0].top.atom(a).__str__() for a in dih_bond_indices[:, 1]],
        ]
    )[0]
    indices = np.full(dihedrals.shape[1], 1)
    assert indices.shape[0] == dihedrals.shape[1]
    assert (
        dihedrals[:, indices].shape == dihedrals.shape
    ), f"{dihedrals[:, indices].shape=} {dihedrals.shape=}"
    for i, bond in enumerate(dih_bond_atoms):
        if "PRO" in bond[0] and "PRO" in bond[1] and "N" in bond[0] and "CA" in bond[1]:
            indices[i] = 0
    indices = indices.astype(bool)

    # get rid of the proline dihedrals
    _back_labels["dihedrals_labels"] = _back_labels["dihedrals_labels"][indices]
    _back_labels["generic_dihedrals_labels"] = _back_labels["generic_dihedrals_labels"][
        indices
    ]
    dih_indices = dih_indices[indices]
    dih_bond_indices = dih_bond_indices[indices]
    dihedrals = dihedrals[:, indices]
    assert dihedrals.shape[1] == dih_indices.shape[0] == dih_bond_indices.shape[0]
    dih_near_sides, dih_far_sides = _get_near_and_far_networkx(
        g,
        dih_bond_indices,
        inp_trajs[0].top,
        parallel=parallel,
    )

    if sidechain_dihedrals is not None:
        side_bond_indices = side_indices[:, 1:3]
        assert (
            side_bond_indices.shape[0]
            == _side_indices.shape[0]
            == sidechain_dihedrals.shape[1]
        ), (
            f"{side_bond_indices.shape=}, {_side_indices.shape=}, "
            f"{sidechain_dihedrals.shape=}"
        )
        # filter out the proline angles
        side_bond_atoms = np.dstack(
            [
                [inp_trajs[0].top.atom(a).__str__() for a in side_bond_indices[:, 0]],
                [inp_trajs[0].top.atom(a).__str__() for a in side_bond_indices[:, 1]],
            ]
        )[0]
        indices = np.full(sidechain_dihedrals.shape[1], 1)
        assert indices.shape[0] == sidechain_dihedrals.shape[1]
        assert (
            sidechain_dihedrals[:, indices].shape == sidechain_dihedrals.shape
        ), f"{sidechain_dihedrals[:, indices].shape=} {sidechain_dihedrals.shape=}"
        for i, bond in enumerate(side_bond_atoms):
            if (
                "PRO" in bond[0]
                and "PRO" in bond[1]
                and "CA" in bond[0]
                and "CB" in bond[1]
            ):
                indices[i] = 0
            if (
                "PRO" in bond[0]
                and "PRO" in bond[1]
                and "CB" in bond[0]
                and "CG" in bond[1]
            ):
                indices[i] = 0
        indices = indices.astype(bool)
        _back_labels["side_dihedrals_labels"] = _back_labels["side_dihedrals_labels"][
            indices
        ]
        _back_labels["generic_side_dihedrals_labels"] = _back_labels[
            "generic_side_dihedrals_labels"
        ][indices]
        _side_indices_out = _side_indices_out[indices]
        side_indices = side_indices[indices]
        side_bond_indices = side_bond_indices[indices]
        sidechain_dihedrals = sidechain_dihedrals[:, indices]

        side_near_sides, side_far_sides = _get_near_and_far_networkx(
            g,
            side_bond_indices,
            inp_trajs[0].top,
            parallel=parallel,
        )

    # assert that the dihedrals and _back_labels have the correct shape
    # that way we can be sure to use the labels to index the correct dihedrals
    # after obtaining the finished trajectory
    assert (
        _back_labels["dihedrals_labels"].shape
        == _back_labels["generic_dihedrals_labels"].shape
    )
    assert _back_labels["dihedrals_labels"].shape[0] == dihedrals.shape[-1]
    if sidechain_dihedrals is not None:
        assert (
            _back_labels["side_dihedrals_labels"].shape
            == _back_labels["generic_side_dihedrals_labels"].shape
        )
        assert (
            _back_labels["side_dihedrals_labels"].shape[0]
            == sidechain_dihedrals.shape[-1]
        )

    # extend the traj
    for i in range(len(dihedrals)):
        if i == 0:
            out_traj = deepcopy(inp_trajs[0][0].traj)
        else:
            out_traj = out_traj.join(inp_trajs[0][0].traj)
    out_traj.top = inp_trajs[0].top

    # adjust the torsions
    new_xyz = np.ascontiguousarray(out_traj.xyz.copy().astype("float32"))
    new_xyz = np.pad(
        new_xyz, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=1
    )
    dihedrals = dihedrals.astype("float32")
    dih_indices = dih_indices.astype("int32")
    total_counts = dihedrals.shape[0] * dihedrals.shape[1]
    if sidechain_dihedrals is not None:
        total_counts += dihedrals.shape[0] * sidechain_dihedrals.shape[1]
        sidechain_dihedrals = sidechain_dihedrals.astype("float32")
    if guess_sp2_atoms:
        total_counts += _countprog(out_traj)
    if parallel:
        raise Exception(f"Parallel has not yet been tested.")
        parallel_rotation_application(
            new_xyz,
            dih_indices,
            dihedrals,
            dih_near_sides,
        )
    else:
        pbar = progbar
        if pbar is None:
            if _is_notebook():
                pbar = notebook_tqdm(
                    total=total_counts,
                    leave=False,
                    position=0,
                    desc="Backmapping...",
                )
            else:
                pbar = normal_tqdm(
                    total=total_counts,
                    leave=False,
                    position=0,
                    desc="Backmapping...",
                )
        else:
            if not isinstance(pbar, bool):
                pbar.reset(pbar.total + total_counts - 1)
            else:
                pbar = None

        for i in range(dihedrals.shape[0]):
            for j in range(dihedrals.shape[1]):
                # central_dihedrals
                near_side = dih_near_sides[j]
                far_side = dih_far_sides[j]
                dihedral = dih_indices[j]
                bond = dih_bond_indices[j]

                # define inputs
                target_angle = dihedrals[i, j]
                current_angle = _dihedral(new_xyz[i, :, :3], dihedral)[0][0]
                angle = target_angle - current_angle
                direction = np.diff(new_xyz[i, bond, :3], axis=0).flatten()
                pivot_point = new_xyz[i, bond[0], :3]

                # perform rotation
                rotmat = transformations_rotation_matrix(angle, direction, pivot_point)
                new_xyz[i, far_side, :3] = rotmat.dot(new_xyz[i, far_side].T).T[:, :3]

                # verify
                if verify_every_rotation:
                    _ = _dihedral(new_xyz[i, :, :3], dihedral)[0][0]
                    if not np.isclose(_, target_angle, atol=1e-3):
                        _resids = [
                            str(inp_trajs[0].top.atom(x).residue.index)
                            for x in dihedral
                        ]
                        s = (
                            f"Adjusting dihedral angle for atoms "
                            f"{[str(inp_trajs[0].top.atom(x)) for x in dihedral]} "
                            f"failed with an absolute tolerance of 1e-3. "
                            f"Residue indices are: "
                            f"{_resids}"
                            f"\nTarget angle was {target_angle} {angle_type}, "
                            f"but rotation yielded angle with {_} {angle_type}."
                            f"\nCurrent angle was {current_angle}. To reach "
                            f"target angle is a rotation of {angle} {angle_type} "
                            f"was carried out."
                            f"\nRotation axis was vector from "
                            f"{inp_trajs[0].top.atom(bond[0])} to "
                            f"{inp_trajs[0].top.atom(bond[1])}"
                            f"\nOnly these atoms should have been affected by "
                            f"rotation: {far_side}"
                            "\nBut somehow this method still crashed. Maybe "
                            "these prints will help."
                        )
                        raise Exception(s)
                if pbar is not None:
                    pbar.update()

            if sidechain_dihedrals is not None:
                for j in range(sidechain_dihedrals.shape[1]):
                    # central_dihedrals
                    near_side = side_near_sides[j]
                    far_side = side_far_sides[j]
                    dihedral = side_indices[j]
                    bond = side_bond_indices[j]

                    # define inputs
                    target_angle = sidechain_dihedrals[i, j]
                    current_angle = _dihedral(new_xyz[i, :, :3], dihedral)
                    angle = target_angle - current_angle
                    direction = np.diff(new_xyz[i, bond, :3], axis=0).flatten()
                    pivot_point = new_xyz[i, bond[0], :3]

                    # perform rotation
                    rotmat = transformations_rotation_matrix(
                        angle, direction, pivot_point
                    )
                    rotmat = transformations_rotation_matrix(
                        angle, direction, pivot_point
                    )
                    new_xyz[i, far_side, :3] = rotmat.dot(new_xyz[i, far_side].T).T[
                        :, :3
                    ]

                    if verify_every_rotation:
                        _ = _dihedral(new_xyz[i, :, :3], dihedral)[0][0]
                        if not np.isclose(_, target_angle, atol=1e-3):
                            _resids = [
                                str(inp_trajs[0].top.atom(x).residue.index)
                                for x in dihedral
                            ]
                            s = (
                                f"Adjusting dihedral angle for atoms "
                                f"{[str(inp_trajs[0].top.atom(x)) for x in dihedral]} "
                                f"failed with an absolute tolerance of 1e-3. "
                                f"Residue indices are: "
                                f"{_resids}"
                                f"\nTarget angle was {target_angle} {angle_type}, "
                                f"but rotation yielded angle with {_} {angle_type}."
                                f"\nCurrent angle was {current_angle}. To reach target "
                                f"angle is a rotation of {angle} {angle_type} was "
                                f"carried out.\nRotation axis was vector from "
                                f"{inp_trajs[0].top.atom(bond[0])} to "
                                f"{inp_trajs[0].top.atom(bond[1])}"
                                f"\nOnly these atoms should have been affected by "
                                f"rotation: {far_side}\nBut somehow this method "
                                f"still crashed. Maybe these prints will help."
                            )
                            raise Exception(s)
                    if pbar is not None:
                        pbar.update()

    # overwrite traj and return
    out_traj.xyz = new_xyz[..., :3]

    # fix the amide atoms
    if guess_sp2_atoms:
        _guess_sp2_atoms(out_traj, pbar)

    if not return_indices:
        return out_traj
    if return_indices:
        return out_traj, _back_labels


def _countprog(traj):
    total = 0
    for i, r in enumerate(traj.top.residues):
        for j in range(traj.n_frames):
            if r:
                continue
            if i > 0:
                total += 1
            if i < traj.n_residues - 2:
                total += 1
    return total


def _guess_sp2_atoms(
    traj: md.Trajectory,
    pbar: Optional[Any] = None,
) -> None:
    # Third Party Imports
    from scipy.spatial.transform import Rotation as R

    Ns = traj.top.select("name N")
    CAs = traj.top.select("name CA")
    Cs = traj.top.select("name C")
    assert len(Ns) == len(CAs) == len(Cs) == traj.n_residues, (
        f"I could not determine the correct number of backbone atoms for this "
        f"protein of {traj.n_residues} residues. I expected {traj.n_residues} "
        f"nitrogen atoms, but got {len(Ns)}. I expected {traj.n_residues} "
        f"alpha carbon atoms, but got {len(CAs)}. I expected {traj.n_residues} "
        f"carboxylic carbons, but got {len(Cs)}. Maybe your protein contains "
        f"non-standard residues."
    )
    for i, (r, N, CA, C) in enumerate(zip(traj.top.residues, Ns, CAs, Cs)):
        for j, frame in enumerate(traj):
            if r.name == "PRO":
                continue
            N_pos = frame.xyz[0, N]
            CA_pos = frame.xyz[0, CA]
            C_pos = frame.xyz[0, C]
            if i > 0:
                H = next(r.atoms_by_name("H")).index
                C_prev_pos = frame.xyz[0, Cs[i - 1]]
                v1 = CA_pos - N_pos
                v2 = C_prev_pos - N_pos
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                n *= 123 / 180 * np.pi
                M = R.from_rotvec(n)
                new_H_pos = v1 @ M.as_matrix()
                new_H_pos /= np.linalg.norm(new_H_pos)
                new_H_pos *= 0.11
                new_H_pos += N_pos
                traj.xyz[j, H] = new_H_pos
                if pbar is not None:
                    pbar.update()
            if i < traj.n_residues - 2:
                O = next(r.atoms_by_name("O")).index
                N_next_pos = frame.xyz[0, Ns[i + 1]]
                v1 = CA_pos - C_pos
                v2 = N_next_pos - C_pos
                n = np.cross(v1, v2)
                n /= np.linalg.norm(n)
                n *= 121 / 180 * np.pi
                M = R.from_rotvec(n)
                new_O_pos = v1 @ M.as_matrix()
                new_O_pos /= np.linalg.norm(new_O_pos)
                new_O_pos *= 0.124
                new_O_pos += C_pos
                traj.xyz[j, O] = new_O_pos
                # actual_distance = np.linalg.norm(frame.xyz[0, C] - frame.xyz[0, O])
                # u = CA_pos - C_pos
                # v = frame.xyz[0, O] - C_pos
                # actual_angle = np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
                # print(
                #     f"In the protein, the atoms {traj.top.atom(C)} and {traj.top.atom(O)} are "
                #     f"{actual_distance:.3f} nm apart with a CA-C-O angle of {actual_angle:.3f} rad. "
                #     f"Setting the angle to {121 / 180 * np.pi:.3f} and the distance "
                #     f"to 0.124 nm, I get the position of to be {new_O_pos=}, instead "
                #     f"of {frame.xyz[0, O]}."
                # )
                if pbar is not None:
                    pbar.update()


def dihedral_to_cartesian_tf_one_way_layers(
    dihedrals: tf.Tensor,
    cartesian: tf.Tensor,
    n: int,
) -> None:
    """Takes one of the cartesian sides (left, or right) and turns them into 3D.

    The left or right sides of the cartesian chain can have different number
    of atoms in them, depending on whether the protein has an even or odd
    number of central atoms.

    Args:
        dihedrals: tf.Tensor: The dihedrals with shape (batch, None)
        cartesian: tf.Tensor: The cartesians with shape (batch, None, 3).
        n: int: The range variable for iteration. Must be the same as dihedrals.shape[-1].

    Returns:
        tf.Tensor: The output.

    """
    tf.debugging.assert_rank(dihedrals, 2)
    dihedrals = -dihedrals

    rotated = cartesian[:, 1:]
    collected_cartesians = [cartesian[:, :1]]
    for i in range(n):
        collected_cartesians.append(rotated[:, 0:1])
        axis = rotated[:, 1] - rotated[:, 0]
        # axis /= tf.norm(axis, axis=1, keepdims=True)
        # numeric problems with tf.norm
        # see here:
        # https://datascience.stackexchange.com/q/80898
        axis /= tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(axis), axis=1)), axis=-1)
        offset = rotated[:, 1:2]
        rotated = offset + tf.matmul(
            rotated[:, 1:] - offset, rotation_matrix(axis, dihedrals[:, i])
        )
    collected_cartesians.append(rotated)
    collected_cartesians = tf.concat(collected_cartesians, axis=1)
    return collected_cartesians


def backbone_hydrogen_oxygen_crossproduct(backbone_positions):
    assert backbone_positions.shape[2] % 3 == 0  # C, CA, N atoms, multiple of three
    pass


def guess_sp2_atom(cartesians, indices, angle_to_previous, bond_length):
    added_cartesians = []
    for i in indices:
        prev_vec = cartesians[:, i - 1] - cartesians[:, i]
        try:
            next_vec = cartesians[:, i + 1] - cartesians[:, i]
        except tf.errors.InvalidArgumentError:
            next_vec = cartesians[:, i - 2] - cartesians[:, i]

        perpendicular_axis = tf.linalg.cross(prev_vec, next_vec)
        perpendicular_axis /= tf.norm(perpendicular_axis, axis=1, keepdims=True)
        bond_vec = tf.matmul(
            tf.expand_dims(prev_vec, 1),
            rotation_matrix(perpendicular_axis, angle_to_previous),
        )
        bond_vec = bond_vec[:, 0, :]
        bond_vec *= bond_length / tf.norm(bond_vec, axis=1, keepdims=True)
        added_cartesians.append(cartesians[:, i] + bond_vec)
    added_cartesians = tf.stack(added_cartesians, axis=1)
    return added_cartesians


def guess_amide_H(cartesians, N_indices):
    return guess_sp2_atom(cartesians, N_indices[1::], 123 / 180 * pi, 1.10)


def guess_amide_O(cartesians, C_indices):
    return guess_sp2_atom(cartesians, C_indices, 121 / 180 * pi, 1.24)


def rotation_matrix(axis_unit_vec, angle):
    angle = tf.expand_dims(tf.expand_dims(angle, axis=-1), axis=-1)
    i = tf.expand_dims(tf.eye(3), 0)
    zeros = tf.zeros(tf.shape(axis_unit_vec)[0])
    cross_prod_matrix = tf.convert_to_tensor(
        [
            [zeros, -axis_unit_vec[:, 2], axis_unit_vec[:, 1]],
            [axis_unit_vec[:, 2], zeros, -axis_unit_vec[:, 0]],
            [-axis_unit_vec[:, 1], axis_unit_vec[:, 0], zeros],
        ]
    )
    cross_prod_matrix = tf.transpose(cross_prod_matrix, [2, 0, 1])
    r = tf.cos(angle) * i
    r += tf.sin(angle) * cross_prod_matrix
    axis_unit_vec = tf.expand_dims(axis_unit_vec, 2)
    r += (1 - tf.cos(angle)) * tf.matmul(
        axis_unit_vec, tf.transpose(axis_unit_vec, [0, 2, 1])
    )
    return r


def merge_cartesians(
    central_cartesians, N_indices, O_indices, H_cartesians, O_cartesians
):
    cartesian = [central_cartesians[:, 0]]
    h_i = 0
    o_i = 0
    for i in range(1, central_cartesians.shape[1]):
        cartesian.append(central_cartesians[:, i])
        if i in N_indices[1::]:
            cartesian.append(H_cartesians[:, h_i])
            h_i += 1
        elif i in O_indices:
            cartesian.append(O_cartesians[:, o_i])
            o_i += 1
    cartesian = tf.stack(cartesian, axis=1)
    assert (
        cartesian.shape[1]
        == central_cartesians.shape[1] + H_cartesians.shape[1] + O_cartesians.shape[1]
    )
    return cartesian


def dihedral_backmapping(
    pdb_path, dihedral_trajectory, rough_n_points=-1, sidechains=None
):
    """
    Takes a pdb file with a peptide and creates a trajectory based on the dihedral angles given.
    It simply rotates around the dihedral angle axis. In the result side-chains might overlap but the backbone should
    turn out quite well.

    :param pdb_path: (str)
    :param dihedral_trajectory:
        array-like of shape (traj_length, number_of_dihedrals)
    :param rough_n_points: (int) a step_size to select a subset of values from dihedral_trajectory is calculated by
        max(1, int(len(dihedral_trajectory) / rough_n_points)) with rough_n_points = -1 all values are used.
    :return: (MDAnalysis.Universe)
    """
    step_size = max(1, int(len(dihedral_trajectory) / rough_n_points))
    dihedral_trajectory = dihedral_trajectory[::step_size]
    if sidechains is not None:
        sidechain_dihedral_trajectory = sidechains[::step_size]

    uni = mda.Universe(pdb_path, format="PDB")
    protein = uni.select_atoms("protein")

    dihedrals = []
    sidechain_dihedrals = []

    for residue in protein.residues:
        psi = residue.psi_selection()
        if psi:
            dihedrals.append(psi)

    for residue in protein.residues:
        omega = residue.omega_selection()
        if omega:
            dihedrals.append(omega)

    for residue in protein.residues:
        phi = residue.phi_selection()
        if phi:
            dihedrals.append(phi)

    if sidechains is not None:
        for residue in protein.residues:
            chi1 = residue.chi1_selection()
            if chi1:
                sidechain_dihedrals.append(chi1)

        for residue in protein.residues:
            if "chi2" in residue.__dir__():
                sidechain_dihedrals.append(residue.chi2_selection())

        for residue in protein.residues:
            if "chi3" in residue.__dir__():
                sidechain_dihedrals.append(residue.chi3_selection())

        for residue in protein.residues:
            if "chi4" in residue.__dir__():
                sidechain_dihedrals.append(residue.chi4_selection())

        for residue in protein.residues:
            if "chi5" in residue.__dir__():
                sidechain_dihedrals.append(residue.chi5_selection())

    if sidechains is not None:
        if sidechain_dihedral_trajectory.shape[1] == len(sidechain_dihedrals) * 2:
            sidechain_dihedral_trajectory = sidechain_dihedral_trajectory[:, ::2]

    _expand_universe(uni, len(dihedral_trajectory))

    if sidechains is None:
        for dihedral_values, step in zip(dihedral_trajectory, uni.trajectory):
            for dihedral, value in zip(dihedrals, dihedral_values):
                _set_dihedral(dihedral, protein, value / (2 * pi) * 360)
    else:
        for dihedral_values, sidechain_dihedral_values, step in zip(
            dihedral_trajectory, sidechain_dihedral_trajectory, uni.trajectory
        ):
            for dihedral, value in zip(dihedrals, dihedral_values):
                _set_dihedral(dihedral, protein, value / (2 * pi) * 360)
            for dihedral, value in zip(sidechain_dihedrals, sidechain_dihedral_values):
                _set_dihedral(dihedral, protein, value / (2 * pi) * 360)
    return uni


def _set_dihedral(dihedral, atoms, angle):
    current_angle = dihedral.dihedral.value()
    head = atoms[dihedral[2].id :]
    vec = dihedral[2].position - dihedral[1].position
    head.rotateby(angle - current_angle, vec, dihedral[2].position)


def _expand_universe(universe, length):
    coordinates = (
        AnalysisFromFunction(lambda ag: ag.positions.copy(), universe.atoms)
        .run()
        .results
    )["timeseries"]
    coordinates = np.tile(coordinates, (length, 1, 1))
    universe.load_new(coordinates, format=MemoryReader)

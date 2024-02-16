# -*- coding: utf-8 -*-
# encodermap/loading/featurizer.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade
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
"""EncoderMap featurization follows the example of the now deprecated PyEMMA package.

You can define your features in advance, inspect the expected output and then let
the computer do the number crunching afterwards. This can be done with either
PyEMMAs streamable featurization or **new** with dask and delayed on a dask-cluster
of your liking. Here are the basic concepts of EncoderMap's featurization.

"""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import itertools
import numbers
import re
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path
from sqlite3 import ProgrammingError

# Third Party Imports
import numpy as np
import pandas as pd
import rich.progress
from optional_imports import _optional_import

# Local Folder Imports
from ..loading import features
from ..misc.misc import FEATURE_NAMES, _validate_uri
from ..misc.xarray import get_indices_by_feature_dim, unpack_data_and_feature
from ..trajinfo.info_all import TrajEnsemble
from ..trajinfo.info_single import SingleTraj


################################################################################
# Optional Imports
################################################################################


featurizer = _optional_import("pyemma", "coordinates.featurizer")
source = _optional_import("pyemma", "coordinates.source")
load = _optional_import("pyemma", "coordinates.load")
xr = _optional_import("xarray")
md = _optional_import("mdtraj")
rich = _optional_import("rich")
Client = _optional_import("dask", "distributed.Client")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

# Local Folder Imports
from .._typing import CustomAAsDict
from .features import CustomFeature


if TYPE_CHECKING:
    # Third Party Imports
    import xarray as xr

    # Local Folder Imports
    from ..trajinfo.trajinfo_utils import CustomTopology
    from .features import AnyFeature


################################################################################
# Import tqdm which can be either the jupyter one or the plain one
################################################################################


def _is_notebook():  # pragma: no cover
    try:
        # Third Party Imports
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if _is_notebook():  # pragma: no cover
    # Third Party Imports
    from tqdm.notebook import tqdm
else:
    # Third Party Imports
    from tqdm import tqdm


################################################################################
# Globals
################################################################################


__all__: list[str] = ["Featurizer"]


UNDERSCORE_MAPPING: dict[str, str] = {
    "central_dihedrals": "CentralDihedrals",
    "all_cartesians": "AllCartesians",
    "all_distances": "AllBondDistances",
    "central_cartesians": "CentralCartesians",
    "central_distances": "CentralBondDistances",
    "central_angles": "CentralAngles",
    "side_cartesians": "SideChainCartesians",
    "side_distances": "SideChainBondDistances",
    "side_angles": "SideChainAngles",
    "side_dihedrals": "SideChainDihedrals",
}


################################################################################
# Utils
################################################################################


class CoordsLoad:
    pass


def is_iterable_of_types(l, supertype):
    """Checks whether all elements of l are of type `supertype`."""
    return is_iterable(l) and all(
        issubclass(t, supertype) for t, _ in itertools.groupby(l, type)
    )


def is_iterable_of_int(l):
    """Checks if l is iterable and contains only integral types."""
    return is_iterable_of_types(l, numbers.Integral)


def is_iterable(I):
    return isinstance(I, Iterable)


def _atoms_in_residues(
    traj: SingleTraj,
    residue_idxs: Sequence[int],
    subset_of_atom_idxs: Optional[np.ndarray] = None,
    fallback_to_full_residue: bool = True,
) -> list[np.ndarray]:
    """Returns a list of arrays containing the atom indices in each residue of `residue_idxs`

    Args:
        traj (SingleTraj): A `SingleTraj` instance.
        residue_idxs (Sequence[int]): List or ndarray (ndim=1) of integers.
        subset_of_atom_idxs (Optional[np.ndarray]): Iterable of atom_idxs to which the
            selection has to be restricted. If None, all atoms considered.
            Defaults to None.
        fallback_to_full_residue (bool): It is possible that some
            residues don't yield any atoms with some subsets. Take all atoms in
            that case. If False, then [] is returned for that residue.
            Defaults to None.

    """
    atoms_in_residues = []
    if subset_of_atom_idxs is None:
        subset_of_atom_idxs = np.arange(traj.top.n_atoms)
    special_residues = []
    for rr in traj.top.residues:
        if rr.index in residue_idxs:
            toappend = np.array(
                [aa.index for aa in rr.atoms if aa.index in subset_of_atom_idxs]
            )
            if len(toappend) == 0:
                special_residues.append(rr)
                if fallback_to_full_residue:
                    toappend = np.array([aa.index for aa in rr.atoms])

            atoms_in_residues.append(toappend)

    # Any special cases?
    if len(special_residues) != 0:
        if fallback_to_full_residue:
            msg = "the full residue"
        else:
            msg = "emtpy lists"
        warnings.warn(
            f"These residues yielded no atoms in the subset and were returned as "
            f"{msg} {[rr for rr in special_residues[-2:]]}"
        )

    return atoms_in_residues


def combinations(
    seq: Iterable,
    k: int,
) -> np.ndarray:
    """Return j length subsequences of elements from the input iterable.

    This version uses Numpy/Scipy and should be preferred over itertools. It avoids
    the creation of all intermediate Python objects.

    Examples:
        >>> import numpy as np
        >>> from itertools import combinations as iter_comb
        >>> x = np.arange(3)
        >>> c1 = combinations(x, 2)
        >>> print(c1)
        [[0 1]
         [0 2]
         [1 2]]
        >>> c2 = np.array(tuple(iter_comb(x, 2)))
        >>> print(c2)
        [[0 1]
         [0 2]
         [1 2]]
    """
    # Standard Library Imports
    from itertools import chain
    from itertools import combinations as _combinations

    # Third Party Imports
    from scipy.special import comb

    count = comb(len(seq), k, exact=True)
    res = np.fromiter(chain.from_iterable(_combinations(seq, k)), int, count=count * k)
    return res.reshape(-1, k)


def product(*arrays: np.ndarray) -> np.ndarray:
    """Generate a cartesian product of input arrays.

    Args:
        arrays (np.ndarray): 1-D arrays to form the cartesian product of.

    Returns:
        np.ndarray: 2-D array of shape (M, len(arrays)) containing cartesian
            products formed of input arrays.

    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    out = np.empty_like(ix, dtype=dtype)

    for n, _ in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def _parse_pairwise_input(
    indices1: Sequence[int],
    indices2: Sequence[int],
) -> np.ndarray:
    """For input of a pairwise type (distances, inverse distances, contacts) checks
    the type of input the user gave and formats it so that `DistanceFeature`,
    `InverseDistanceFeature`, and `ContactFeature` can work.

    In case the input isn't already a list of distances, this function will:
        - sort the indices1 array
        - check for duplicates within the indices1 array
        - sort the indices2 array
        - check for duplicates within the indices2 array
        - check for duplicates between the indices1 and indices2 array
        - if indices2 is     None, produce a list of pairs of indices in indices1, or
        - if indices2 is not None, produce a list of pairs of (i,j) where i comes from indices1, and j from indices2

    """

    if is_iterable_of_int(indices1):
        # Eliminate duplicates and sort
        indices1 = np.unique(indices1)

        # Intra-group distances
        if indices2 is None:
            atom_pairs = combinations(indices1, 2)

        # Inter-group distances
        elif is_iterable_of_int(indices2):
            # Eliminate duplicates and sort
            indices2 = np.unique(indices2)

            # Eliminate duplicates between indices1 and indices1
            uniqs = np.in1d(indices2, indices1, invert=True)
            indices2 = indices2[uniqs]
            atom_pairs = product(indices1, indices2)

    else:
        atom_pairs = indices1

    return atom_pairs


def pairs(
    sel: np.ndarray,
    excluded_neighbors: int = 0,
) -> np.ndarray:
    """Creates all pairs between indexes. Will exclude closest neighbors up to
    `excluded_neighbors` The self-pair (i,i) is always excluded.

    Args:
        sel (np.ndarray): Array with selected atom indexes.
        excluded_neighbors (int): Number of neighbors that will be excluded
            when creating the pairs. Defaults to 0.

    Returns:
        np.ndarray: A m x 2 array with all pair indexes between
            different atoms that are at least `excluded_neighbors` indexes
            apart, i.e. if i is the index of an atom, the pairs
            [i,i-2], [i,i-1], [i,i], [i,i+1], [i,i+2], will not be in `sel`
            (n=excluded_neighbors) if `excluded_neighbors` = 2. Moreover,
            the list is non-redundant,i.e. if [i,j] is in sel, then [j,i] is not.

    """
    assert isinstance(excluded_neighbors, int)

    p = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            # get ordered pair
            I = sel[i]
            J = sel[j]
            if I > J:
                I = sel[j]
                J = sel[i]
            # exclude 1 and 2 neighbors
            if J > I + excluded_neighbors:
                p.append([I, J])
    return np.array(p)


################################################################################
# Classes
################################################################################


class SingleTrajFeaturizer:
    def __init__(self, traj: SingleTraj) -> None:
        self.traj = traj
        self._n_custom_features = 0
        self._custom_feature_ids = []
        self.active_features = []

    def add_list_of_feats(
        self,
        which: Union[Literal["all"], Sequence[str]] = "all",
        deg: bool = False,
        omega: bool = True,
        check_aas: bool = True,
    ) -> None:
        """Adds features to the Featurizer to be loaded either in-memory. The
        argument `which` can be either 'all' or a list of the following strings:
            * 'AllCartesians': Cartesian coordinates of all atoms with
                shape (n_frames, n_atoms, 3).
            * 'AllBondDistances': Bond distances of all bonds recognized by
                mdtraj. Use top = md.Topology.from_openmm()if mdtraj does not
                recognize all bonds.
            * 'CentralCartesians': Cartesians of the N, C, CA atoms in the
                backbone with shape (n_frames, n_residues * 3, 3).
            * 'CentralBondDistances': The bond distances of the N, C, CA bonds
                with shape (n_frames, n_residues * 3 - 1).
            * 'CentralAngles': The angles between the backbone bonds with shape
                (n_frames, n_residues * 3 - 2).
            * 'CentralDihedrals': The dihedrals between the backbone atoms
                (omega, phi, psi). With shape (n_frames, n_residues * 3 - 3).
            * 'SideChainCartesians': Cartesians of the sidechain-atoms.
                Starting with CB, CG, ...
            * 'SideChainBondDistances': Bond distances between the
                sidechain atoms. starting with the CA-CG bond.
            * 'SideChainAngles': Angles between sidechain atoms. Starting with
                the C-CA-CB angle.
            * 'SideChainDihedrals': Dihedrals of the sidechains (chi1, chi2, chi3).
        If 'all' is provided for `which` the CentralCartesian, CentralDistances,
        CentralAngles, CentralDihedrals, SideChainDihedrals will be added.

        Args:
            which (Union[str, list], optional). Either add 'all' features or
                a list of features. See Above for possible features. Defaults
                to 'all'.
            deg (bool): Whether the output should be formatted in degrees.
            omega (bool): Whether to include the omega angles of the backbone.
            check_aas (bool): Whether to check if all residues in top are known. Helps with custom topology to not skip unkonw custom/non-natural amino acids.

        """
        recognized_str = list(UNDERSCORE_MAPPING.keys()) + list(
            UNDERSCORE_MAPPING.values()
        )
        if isinstance(which, str):
            if which == "all":
                which = [
                    "CentralCartesians",
                    "CentralBondDistances",
                    "CentralAngles",
                    "CentralDihedrals",
                    "SideChainDihedrals",
                ]
            else:
                if which not in recognized_str:
                    raise Exception(
                        f"Recognized arguments to which are 'all' or any of the "
                        f"following: {recognized_str}. The str you provided {which} "
                        f"did not match any."
                    )
                which = [which]
        elif isinstance(which, (list, tuple)):
            assert all([isinstance(i, str) for i in which]), (
                f"The argument `which` needs to be provided a sequence of str. "
                f"You have offending types in this argument."
            )
            diff = set(which) - set(recognized_str)
            if diff:
                raise Exception(
                    f"One or more of the str in `which` are not recognized. "
                    f"The argument `which` needs to be a sequence containing any "
                    f"of the following: {recognized_str}. The unrecognized str are: "
                    f"{diff}."
                )

        # add the features
        for cf in which:
            if cf in UNDERSCORE_MAPPING:
                cf = UNDERSCORE_MAPPING[cf]
            feature = getattr(features, cf)
            if not feature._use_angle and not feature._use_omega:
                feature = feature(self.traj, check_aas=True)
            elif feature._use_angle and not feature._use_omega:
                feature = feature(
                    self.traj,
                    deg=deg,
                    check_aas=check_aas,
                )
            elif feature._use_angle and feature._use_omega:
                feature = feature(self.traj, deg=deg, omega=omega, check_aas=check_aas)
            else:
                raise Exception(
                    f"Unknown combination of `_use_angle` and `_use_omega` in "
                    f"class attributes of {feature=}"
                )
            self.active_features.append(feature)

    def add_custom_feature(self, feature: AnyFeature) -> None:
        if not hasattr(feature, "id"):
            feature.id = self._n_custom_features
            self._custom_feature_ids.append(self._n_custom_features)
            self._n_custom_features += 1
        elif feature.id is None:
            feature.id = self._n_custom_features
            self._custom_feature_ids.append(self._n_custom_features)
            self._n_custom_features += 1
        else:
            assert feature.id not in self._custom_feature_ids, (
                f"A CustomFeature with the id {feature.id} already exists. "
                f"Please change the id of your CustomFeature."
            )
        self.active_features.append(feature)

    def add_distances_ca(
        self, periodic: bool = True, excluded_neighbors: int = 2
    ) -> None:
        """Adds the distances between all Ca's to the feature list.

        Args:
            periodic (bool): Use the minimum image convention when computing distances.
            excluded_neighbors (int): Number of exclusions when compiling the
                list of pairs. Two CA-atoms are considered neighbors if they
                belong to adjacent residues. Defaults to 2.

        """
        at_idxs_ca = self.select_Ca
        res_idxs_ca = [self.traj.top.atom(ca).residue.index for ca in at_idxs_ca]
        res_idxs_ca_pairs = pairs(res_idxs_ca, excluded_neighbors=excluded_neighbors)
        distance_indexes = []
        for ri, rj in res_idxs_ca_pairs:
            distance_indexes.append(
                [
                    self.traj.top.residue(ri).atom("CA").index,
                    self.traj.top.residue(rj).atom("CA").index,
                ]
            )
        distance_indexes = np.array(distance_indexes)

        self.add_distances(distance_indexes, periodic=periodic)

    def add_distances(
        self,
        indices: Union[np.ndarray, Sequence[int]],
        periodic: bool = True,
        indices2: Optional[Sequence[int]] = None,
    ) -> None:
        """Adds the distances between atoms to the feature list.

        Args:
            indices (Union[np.ndarray, Iterable[Sequence[int]]]): Can be one of to types:
                A numpy array of shape (n, 2) with the pairs of atoms between
                which the distances shall be computed. Or a sequence of integers
                which are the indices (not pairs of indices) of the atoms between
                which the distances shall be computed. In this case, the arg `indices2`
                needs to be supplied.

            periodic (bool): If periodic is True and the trajectory contains
                unitcell information, distances will be computed under the
                minimum image convention. Defaults to True.
            indices2 (Optional[Sequence[int]]): Only has effect if `indices` is
                a sequence of integers. Instead of the above behavior, only the
                distances between the atoms in indices` and `indices2` will be
                computed.

        Note:
            When using the iterable of integers input, `indices` and `indices2`
            will be sorted numerically and made unique before converting them to
            a pairlist. Please look carefully at the output of `self.describe()` to
            see what features exactly have been added.

        """
        # Local Folder Imports
        from .features import DistanceFeature

        atom_pairs = _parse_pairwise_input(indices, indices2)

        atom_pairs = self._check_indices(atom_pairs)
        f = DistanceFeature(self.traj, atom_pairs, periodic=periodic)
        self.active_features.append(f)

    def add_backbone_torsions(
        self,
        selstr: Optional[int] = None,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
    ) -> None:
        """Adds all backbone phi/psi angles or the ones specified in `selstr` to the feature list.

        Args:
            selstr (Optional[str]): If None, all phi/psi angles will be considered.
                Otherwise, can be a string specifying the atoms of specific
                backbone torsions (see example).
            deg (bool): Whether the output should be in degrees (True) or radians
                (False). Defaults to False.
            cossin (bool): Whether to return the angles (False) or tuples of their
                cos and sin values (True). Defaults to False.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.

        Examples:
            >>> import encodermap as em
            >>> import numpy as np
            >>> trajs = em.load_project("linear_dimers")
            >>> feat = em.Featurizer(trajs[0])
            >>> feat.add_backbone_torsions("resname PRO")
            >>> feat.describe()
            ['PHI 0 PRO 19',
             'PSI 0 PRO 19',
             'PHI 0 PRO 37',
             'PSI 0 PRO 37',
             'PHI 0 PRO 38',
             'PSI 0 PRO 38',
             'PHI 0 PRO 95',
             'PSI 0 PRO 95',
             'PHI 0 PRO 113',
             'PSI 0 PRO 113',
             'PHI 0 PRO 114',
             'PSI 0 PRO 114']
            >>> ds = feat.get_output()
            >>> da = ds.BackboneTorsionFeature
            >>> phi_indices = da.coords["BACKBONETORSIONFEATURE"].str.contains("PHI")
            >>> angles = np.rad2deg(da.sel(BACKBONETORSIONFEATURE=phi_indices).values[0])
            >>> np.min(angles)
            -103.39891
            >>> np.max(angles)
            -10.015779

        """
        # Local Folder Imports
        from .features import BackboneTorsionFeature

        f = BackboneTorsionFeature(
            self.traj, selstr=selstr, deg=deg, cossin=cossin, periodic=periodic
        )
        self.active_features.append(f)

    def add_angles(
        self,
        indexes: np.ndarray,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
    ) -> None:
        """Adds the list of angles to the feature list.

        Args:
            indexes (np.ndarray): An array with triplets of atom indices.
            deg (bool): Whether the output should be in degrees (True) or radians
                (False). Defaults to False.
            cossin (bool): Whether to return the angles (False) or tuples of their
                cos and sin values (True). Defaults to False.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.

        """
        # Local Folder Imports
        from .features import AngleFeature

        indexes = self._check_indices(indexes, pair_n=3)
        f = AngleFeature(
            self.traj,
            indexes,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
        )
        self.active_features.append(f)

    def add_all(
        self,
        reference: Optional[md.Trajectory] = None,
        atom_indices: Optional[np.ndarray] = None,
        ref_atom_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Adds all atom coordinates to the feature list.
        The coordinates are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

        Args:
            reference (Optional[md.Trajectory]). If different from None, all
                data is aligned using MDTraj's superpose. Defaults to None.
            atom_indices (Optional[np.ndarray]): The indices of atoms to superpose
                If None all atoms will be used. Defaults to None.
            ref_atom_indices (Optional[np.ndarray]): Use these atoms on the
                reference structure. If not supplied, the same atom indices
                will be used for this trajectory and the reference one.

        """
        self.add_selection(
            list(range(self.traj.n_atoms)),
            reference=reference,
            atom_indices=atom_indices,
            ref_atom_indices=ref_atom_indices,
        )

    def add_selection(
        self,
        indexes: np.ndarray,
        reference: Optional[np.ndarray] = None,
        atom_indices: Optional[np.ndarray] = None,
        ref_atom_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Adds the coordinates of the selected atom indexes to the feature list.
        The coordinates of the selection [1, 2, ...] are flattened as follows: [x1, y1, z1, x2, y2, z2, ...]

        Args:
            indexes (np.ndarray): Array with selected atom indexes.
            reference (Optional[md.Trajectory]). If different from None, all
                data is aligned using MDTraj's superpose. Defaults to None.
            atom_indices (Optional[np.ndarray]): The indices of atoms to superpose
                If None, all atoms will be used. Defaults to None.
            ref_atom_indices (Optional[np.ndarray]): Use these atoms on the
                reference structure. If not supplied, the same atom indices
                will be used for this trajectory and the reference one.

        """
        # Local Folder Imports
        from .features import AlignFeature, SelectionFeature

        if reference is None:
            f = SelectionFeature(self.traj, indexes)
        else:
            if not isinstance(reference, md.Trajectory):
                raise ValueError(
                    "reference is not a mdtraj.Trajectory object, but {}".format(
                        reference
                    )
                )
            f = AlignFeature(
                reference=reference,
                indexes=indexes,
                atom_indices=atom_indices,
                ref_atom_indices=ref_atom_indices,
            )
        self.active_features.append(f)

    def add_inverse_distances(
        self,
        indices: Union[np.ndarray, Sequence[int]],
        periodic: bool = True,
        indices2: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> None:
        """Adds the inverse distances between atoms to the feature list.

        Args:
            indices (Union[np.ndarray, Sequence[int]]): A array with shape (n, 2)
                giving the pairs of atoms between which the inverse distances
                shall be computed. Can also be a sequence of integers giving
                the first atoms in the distance calculations. In this case,
                `indices2` needs to be supplied.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            indices2 (Optional[Union[np.ndarray, Sequence[int]]]): If the argument
                `indices` is just a sequence of int (and not a (n, 2) np.ndarray),
                this argument needs to be provided.

        Note:
            When using the *iterable of integers* input, `indices` and `indices2`
            will be sorted numerically and made unique before converting them to
            a pairlist. Please look carefully at the output of `describe()` to
            see what features exactly have been added.

        """
        # Local Folder Imports
        from .features import InverseDistanceFeature

        atom_pairs = _parse_pairwise_input(
            indices,
            indices2,
        )

        atom_pairs = self._check_indices(atom_pairs)
        f = InverseDistanceFeature(self.traj, atom_pairs, periodic=periodic)
        self.active_features.append(f)

    def add_contacts(
        self,
        indices: Union[np.ndarray, Sequence[int]],
        indices2: Optional[Union[np.ndarray, Sequence[int]]] = None,
        threshold: float = 0.3,
        periodic: bool = True,
        count_contacts: bool = False,
    ) -> None:
        """Adds the contacts to the feature list.

        Args:
            indices (Union[np.ndarray, Sequence[int]]): A array with shape (n, 2)
                giving the pairs of atoms between which the inverse distances
                shall be computed. Can also be a sequence of integers giving
                the first atoms in the distance calculations. In this case,
                `indices2` needs to be supplied.
            indices2 (Optional[Union[np.ndarray, Sequence[int]]]): If the argument
                `indices` is just a sequence of int (and not a (n, 2) np.ndarray),
                this argument needs to be provided.
            threshold (float): Distance below this (in nanometer) are considered
                as contacts. The output will contain 1.0 for these contacts.
                Above this threshold, the output will contain 0.0. Defaults to 0.2.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            count_contacts (bool): If set to true, this feature will return
                the number of formed contacts (and not feature values with
                either 1.0 or 0). The output of this feature will be of shape
                (Nt,1), and not (Nt, nr_of_contacts). Defaults to False.

        Note:
            When using the *iterable of integers* input, `indices` and `indices2`
            will be sorted numerically and made unique before converting them
            to a pairlist. Please look carefully at the output of `describe()`
            to see what features exactly have been added.

        """
        # Local Folder Imports
        from .features import ContactFeature

        atom_pairs = _parse_pairwise_input(indices, indices2)
        atom_pairs = self._check_indices(atom_pairs)
        f = ContactFeature(self.traj, atom_pairs, threshold, periodic, count_contacts)
        self.active_features.append(f)

    def add_residue_mindist(
        self,
        residue_pairs: Union[Literal["all"], np.ndarray] = "all",
        scheme: Literal["ca", "closest", "closest-heavy"] = "closest-heavy",
        ignore_nonprotein: bool = True,
        threshold: Optional[float] = None,
        periodic: bool = True,
        count_contacts: bool = False,
    ) -> None:
        """Adds the minimum distance between residues to the feature list.
        See below how the minimum distance can be defined. If the topology
        generated out of `traj` contains information on periodic boundary
        conditions, the minimum image convention will be used when computing
        distances.

        Args:
            residue_pairs (Union[Literal["all"], np.ndarray]): Can be 'all', in
                which case, between all pairs of residues excluding first and
                second neighbor. If a np.array with shape (n ,2) is supplied, these
                residue indices (0-based) will be used to compute the mindists.
                Default to 'all'.
            scheme (Literal["ca", "closest", "closest-heavy"]): Within a residue,
                determines the sub-group atoms that will be considered when
                computing distances. Defaults to 'closest-heavy'.
            ignore_nonprotein (bool): Whether to ignore residues that are not
                of protein type (e.g. water molecules, post-translational modifications,
                non-standard residues, etc.). Defaults to True.
            threshold (float): Distances below this threshold (in nm) will
                result in a feature 1.0, the distances above will result in 0.0. If
                left to None, the numerical value will be returned. Defaults to None.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            count_contacts (bool): If set to true, this feature will return
                the number of formed contacts (and not feature values with
                either 1.0 or 0). The output of this feature will be of shape
                (Nt,1), and not (Nt, nr_of_contacts). Defaults to False.

        Note:
            Using `scheme` = 'closest' or 'closest-heavy' with
            `residue pairs` = 'all' will compute nearly all interatomic distances,
            for every frame, before extracting the closest pairs. This can be
            very time-consuming. Those schemes are intended to be used with a
            subset of residues chosen `residue_pairs`.


        """
        # Local Folder Imports
        from .features import ResidueMinDistanceFeature

        if scheme != "ca" and isinstance(residue_pairs, str):
            if residue_pairs == "all":
                warnings.warn(
                    "Using all residue pairs with schemes like closest or "
                    "closest-heavy is very time consuming. Consider reducing "
                    "the residue pairs"
                )

        f = ResidueMinDistanceFeature(
            self.traj,
            residue_pairs,
            scheme,
            ignore_nonprotein,
            threshold,
            periodic,
            count_contacts=count_contacts,
        )

        self.active_features.append(f)

    def add_group_COM(
        self,
        group_definitions: Sequence[int],
        ref_geom: Optional[md.Trajectory] = None,
        image_molecules: bool = False,
        mass_weighted: bool = True,
    ) -> None:
        """Adds the centers of mass (COM) in cartesian coordinates of a group or
        groups of atoms. If these group definitions coincide directly with
        residues, use `add_residue_COM` instead. No periodic boundaries are
        taken into account.

        Args:
            group_definitions (Sequence[int]): List of the groups of atom indices
                for which the COM will be computed. The atoms are zero-indexed.
            ref_geom (Optional[md.Trajectory]): If a md.Trajectory is provided,
                the coordinates of the provided traj will be centered using
                this reference, before computing COM. If None is provided,
                they won't be centered. Defaults to None.
            image_molecules (bool): The method traj.image_molecules will be
                called before computing averages. The method tries to correct
                for molecules broken across periodic boundary conditions, but
                can be time consuming. See
                http://mdtraj.org/latest/api/generated/mdtraj.Trajectory.html#mdtraj.Trajectory.image_molecules
                for more details. Defualts to False.
            mass_weighted (bool): Set to False if you want the geometric center
                and not the COM. Defaults to True.

        """
        # Local Folder Imports
        from .features import GroupCOMFeature

        f = GroupCOMFeature(
            self.traj,
            group_definitions,
            ref_geom=ref_geom,
            image_molecules=image_molecules,
            mass_weighted=mass_weighted,
        )
        self.active_features.append(f)

    def add_residue_COM(
        self,
        residue_indices: Sequence[int],
        scheme: Literal["all", "backbone", "sidechain"] = "all",
        ref_geom: Optional[md.Trajectory] = None,
        image_molecules: bool = False,
        mass_weighted: bool = True,
    ) -> None:
        """Adds a per-residue center of mass (COM) in cartesian coordinates.
        No periodic boundaries are taken into account.

        Args:
            residue_indices (Sequence[int]): List of the residue indices
                for which the COM will be computed. The atoms are zero-indexed.
            scheme (Literal["all", "backbone", "sidechain"]): What atoms
                contribute to the COM computation. If the scheme yields no
                atoms for some residue, the selection falls back to 'all' for
                that residue. Defaults to 'all'.
            ref_geom (Optional[md.Trajectory]): If a md.Trajectory is provided,
                the coordinates of the provided traj will be centered using
                this reference, before computing COM. If None is provided,
                they won't be centered. Defaults to None.
            image_molecules (bool): The method traj.image_molecules will be
                called before computing averages. The method tries to correct
                for molecules broken across periodic boundary conditions, but
                can be time consuming. See
                http://mdtraj.org/latest/api/generated/mdtraj.Trajectory.html#mdtraj.Trajectory.image_molecules
                for more details. Defualts to False.
            mass_weighted (bool): Set to False if you want the geometric center
                and not the COM. Defaults to True.

        """

        # Local Folder Imports
        from .features import ResidueCOMFeature

        assert scheme in ["all", "backbone", "sidechain"]

        residue_atoms = _atoms_in_residues(
            self.traj,
            residue_indices,
            subset_of_atom_idxs=self.traj.top.select(scheme),
        )

        f = ResidueCOMFeature(
            self.traj,
            np.asarray(residue_indices),
            residue_atoms,
            scheme,
            ref_geom=ref_geom,
            image_molecules=image_molecules,
            mass_weighted=mass_weighted,
        )

        self.active_features.append(f)

    def add_dihedrals(
        self,
        indexes: np.ndarray,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
    ) -> None:
        """Adds the list of dihedrals to the feature list

        Args:
            indexes (np.ndarray): An array with quadruplets of atom indices.
            deg (bool): Whether the output should be in degrees (True) or radians
                (False). Defaults to False.
            cossin (bool): Whether to return the angles (False) or tuples of their
                cos and sin values (True). Defaults to False.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.

        """
        # Local Folder Imports
        from .features import DihedralFeature

        indexes = self._check_indices(indexes, pair_n=4)
        f = DihedralFeature(
            self.traj,
            indexes,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
        )
        self.active_features.append(f)

    def add_sidechain_torsions(
        self,
        selstr: Optional[str] = None,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        which: Union[
            Literal["all"], Sequence[Literal["chi1", "chi2", "chi3", "chi4", "chi5"]]
        ] = "all",
    ) -> None:
        """Adds all side chain torsion angles or the ones specified in `selstr`
        to the feature list.

        Args:

            selstr (Optional[str]): Selection string specifying the atom
                selection used to specify a specific set of backbone angles.
                If None (default), all chi angles found in the topology will be
                computed. Defaults to None.
            deg (bool): Whether the output should be in degrees (True) or radians
                (False). Defaults to False.
            cossin (bool): Whether to return the angles (False) or tuples of their
                cos and sin values (True). Defaults to False.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            which (Union[Literal["all"], Sequence[Literal["chi1", "chi2", "chi3", "chi4", "chi5"]]]):
                Which angles to consider. Can be 'all' or any combination of
                ('all', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5').

        """
        # Local Folder Imports
        from .features import SideChainTorsions

        f = SideChainTorsions(
            self.traj,
            selstr=selstr,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
            which=which,
        )
        self.active_features.append(f)

    def add_minrmsd_to_ref(
        self,
        ref: Union[md.Trajectory, SingleTraj],
        ref_frame: int = 0,
        atom_indices: Optional[np.ndarray] = None,
        precentered: bool = False,
    ) -> None:
        """Adds the minimum root-mean-square-deviation (minrmsd)
        with respect to a reference structure to the feature list.

        Args:
            ref (Union[md.Trajectory, SingleTraj]): Reference structure for
                computing the minrmsd.
            ref_frame (int): Reference frame of the filename specified in `ref`.
                Defaults to 0.
            atom_indices (Optional[np.ndarray]): Atoms that will be used for:
                1. aligning the target and reference geometries.
                2. computing rmsd after the alignment.
                If left to None, all atoms of `ref` will be used.
            precentered (bool): Use this boolean at your own risk to let
                mdtraj know that the target conformations are already centered
                at the origin, i.e., their (uniformly weighted) center of
                mass lies at the origin. This will speed up the computation of
                the rmsd. Defaults to False
        """
        # Local Folder Imports
        from .features import MinRmsdFeature

        f = MinRmsdFeature(
            self.traj,
            ref,
            ref_frame=ref_frame,
            atom_indices=atom_indices,
            precentered=precentered,
        )
        self.active_features.append(f)

    @property
    def ndim(self) -> int:
        return self.dimension()

    @property
    def features(self) -> list[AnyFeature]:
        return self.active_features

    @property
    def select_Ca(self) -> np.ndarray:
        return self.traj.top.select("name CA")

    def _check_indices(self, pair_inds: np.ndarray, pair_n: int = 2) -> np.ndarray:
        """Ensure pairs are valid (shapes, all atom indices available?, etc.)"""

        pair_inds = np.array(pair_inds).astype(dtype=int, casting="safe")

        if pair_inds.ndim != 2:
            raise ValueError("pair indices has to be a matrix.")

        if pair_inds.shape[1] != pair_n:
            raise ValueError(f"pair indices shape has to be (x, {pair_n}).")

        if pair_inds.max() > self.traj.top.n_atoms:
            raise ValueError(
                f"index out of bounds: {pair_inds.max()}. Maximum atom index "
                f"available: {self.traj.top.n_atoms}"
            )

        return pair_inds

    def transform(self, p: Optional[tqdm] = None) -> np.ndarray:
        """Calls the `transform()` methods of the accumulated features.

        Args:
            p (Optional[tqdm]): If an instance of tqdm is provided, a progress
                it will be updated after every call of `feature.transform()`.
                If None is provided, no progress bar will be displayed.

        Returns:
            np.ndarray: A numpy array with the features in `self.active_features`,
                stacked along the feature dimension.

        """
        # if there are no features selected, return given trajectory
        if not self.active_features:
            warnings.warn(
                "You have not selected any features. Add features and call "
                "`transform` or `get_output` again.."
            )
            return

        # otherwise, build feature vector.
        feature_vec = []

        for f in self.active_features:
            # perform sanity checks for custom feature input
            if isinstance(f, CustomFeature):
                # NOTE: casting=safe raises in numpy>=1.9
                vec = f.transform(self.traj).astype(np.float32, casting="safe")
                if vec.shape[0] == 0:
                    vec = np.empty((0, f.dimension))

                if not isinstance(vec, np.ndarray):
                    raise ValueError(
                        f"Your custom feature {f.describe()} did not return a numpy.ndarray!"
                    )
                if not vec.ndim == 2:
                    raise ValueError(
                        f"Your custom feature {f.desccribe()} did not return a "
                        f"2d array. Shape was {vec.shape}"
                    )
                if not vec.shape[0] == self.traj.xyz.shape[0]:
                    raise ValueError(
                        f"Your custom feature {f.desccribe()} did not return as "
                        f"many frames, as it received. Input was {self.traj.xyz.shape[0]}, "
                        f"output was {vec.shape[0]}"
                    )
            else:
                vec = f.transform().astype(np.float32)
            feature_vec.append(vec)

            if p is not None:
                p.update()

        if len(feature_vec) > 1:
            res = np.hstack(feature_vec)
        else:
            res = feature_vec[0]

        return res

    def get_output(self, pbar: Optional[tqdm] = None) -> xr.Dataset:
        # Local Folder Imports
        from ..misc.xarray import unpack_data_and_feature

        if pbar is None:
            if self.traj.basename is None:
                desc = f"Getting output of {len(self.active_features)} features"
            else:
                desc = (
                    f"Getting output of {len(self.active_features)} features for "
                    f"{self.traj.basename}"
                )
            pbar = tqdm(
                total=len(self.active_features),
                desc=desc,
            )
        with pbar as p:
            out = self.transform(p=p)
        ds = unpack_data_and_feature(self, self.traj, out)
        return ds

    def describe(self) -> list[str]:
        all_labels = []
        for f in self.active_features:
            all_labels += f.describe()
        return all_labels

    def __repr__(self) -> str:
        # Standard Library Imports
        import pprint

        feat_str = pprint.pformat(self.describe()[:10])[:-1] + ", ...]"
        return f"EncoderMap Featurizer with features:\n{feat_str}"

    def __len__(self) -> int:
        return len(self.active_features)

    def dimension(self) -> int:
        return sum(f.dimension for f in self.active_features)


class Featurizer:
    def __new__(cls, traj: Union[SingleTraj, TrajEnsemble]):
        if isinstance(traj, SingleTraj):
            return SingleTrajFeaturizer(traj)
        else:
            return EnsembleFeaturizer(traj)


class AddSingleFeatureMethodsToClass(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)

        # iteratively add these functions
        _add_X_function_names = (
            "add_all",
            "add_selection",
            "add_distances_ca",
            "add_distances",
            "add_inverse_distances",
            "add_contacts",
            "add_residue_mindist",
            "add_group_COM",
            "add_residue_COM",
            "add_angles",
            "add_dihedrals",
            "add_minrmsd_to_ref",
        )
        for add_X_function_name in _add_X_function_names:
            # create a function with the corresponding add_X_function_name
            # IMPORTANT: keep this as a keyword argument, to prevent
            # python from late-binding
            def add_X_func(
                self, *args, add_x_name=add_X_function_name, **kwargs
            ) -> None:
                # iterate over the trajs in self.trajs
                for top, trajs in self.trajs.trajs_by_top.items():
                    # create a featurizer
                    f = SingleTrajFeaturizer(trajs[0])
                    # get the method defined by pyemma_function_name
                    func = getattr(f, add_x_name)
                    # call the method with *args and **kwargs, so that the
                    # correct feature is added
                    func(*args, **kwargs)
                    # this is the feature we are looking for.
                    feature = f.active_features[-1]
                    # add the feature
                    self.active_features.setdefault(top, []).append(feature)
                    if top not in self.feature_containers:
                        self.feature_containers[top] = FeatureContainer()
                        self.feature_containers[
                            top
                        ].active_features = self.active_features[top]

            # also add the docstring :)
            add_X_func.__doc__ = getattr(
                SingleTrajFeaturizer, add_X_function_name
            ).__doc__
            setattr(x, add_X_function_name, add_X_func)
        return x


class FeatureContainer:
    @property
    def features(self) -> list[AnyFeature]:
        return self.active_features


class EnsembleFeaturizer(metaclass=AddSingleFeatureMethodsToClass):
    def __init__(self, trajs: TrajEnsemble) -> None:
        self.trajs = trajs
        self.active_features = {}
        self.feature_containers = {}
        self.ensemble = False

    def add_list_of_feats(
        self,
        which: Union[Literal["all"], Sequence[str]] = "all",
        ensemble: bool = False,
        deg: bool = False,
        omega: bool = True,
        check_aas: bool = True,
    ) -> None:
        """Adds features to the Featurizer to be loaded either in-memory. The
        argument `which` can be either 'all' or a list of the following strings:
            * 'AllCartesians': Cartesian coordinates of all atoms with
                shape (n_frames, n_atoms, 3).
            * 'AllBondDistances': Bond distances of all bonds recognized by
                mdtraj. Use top = md.Topology.from_openmm()if mdtraj does not
                recognize all bonds.
            * 'CentralCartesians': Cartesians of the N, C, CA atoms in the
                backbone with shape (n_frames, n_residues * 3, 3).
            * 'CentralBondDistances': The bond distances of the N, C, CA bonds
                with shape (n_frames, n_residues * 3 - 1).
            * 'CentralAngles': The angles between the backbone bonds with shape
                (n_frames, n_residues * 3 - 2).
            * 'CentralDihedrals': The dihedrals between the backbone atoms
                (omega, phi, psi). With shape (n_frames, n_residues * 3 - 3).
            * 'SideChainCartesians': Cartesians of the sidechain-atoms.
                Starting with CB, CG, ...
            * 'SideChainBondDistances': Bond distances between the
                sidechain atoms. starting with the CA-CG bond.
            * 'SideChainAngles': Angles between sidechain atoms. Starting with
                the C-CA-CB angle.
            * 'SideChainDihedrals': Dihedrals of the sidechains (chi1, chi2, chi3).
        If 'all' is provided for `which` the CentralCartesian, CentralDistances,
        CentralAngles, CentralDihedrals, SideChainDihedrals will be added.

        Args:
            which (Union[str, list], optional). Either add 'all' features or
                a list of features. See Above for possible features. Defaults
                to 'all'.
            ensemble (bool): Whether the trajs in this class belong to an ensemble.
                This implies that they contain either the same topology or are
                very similar (think wt, and mutant). Setting this option True will
                try to match the CVs of the trajs onto the same dataset.
                If a VAL residue has been replaced by LYS in the mutant,
                the number of sidechain dihedrals will increase. The CVs of the
                trajs with VAL will thus contain some NaN values. Defaults to False.
            deg (bool): Whether the output should be formatted in degrees.
            omega (bool): Whether to include the omega angles of the backbone.
            check_aas (bool): Whether to check if all residues in top are known. Helps with custom topology to not skip unkonw custom/non-natural amino acids.

        """
        self.ensemble = ensemble
        recognized_str = list(UNDERSCORE_MAPPING.keys()) + list(
            UNDERSCORE_MAPPING.values()
        )
        for top, trajs in self.trajs.trajs_by_top.items():
            if isinstance(which, str):
                if which == "all":
                    which = [
                        "CentralCartesians",
                        "CentralBondDistances",
                        "CentralAngles",
                        "CentralDihedrals",
                        "SideChainDihedrals",
                    ]
                else:
                    if which not in recognized_str:
                        raise Exception(
                            f"Recognized arguments to which are 'all' or any of the "
                            f"following: {recognized_str}. The str you provided {which} "
                            f"did not match any."
                        )
                    which = [which]
            elif isinstance(which, (list, tuple)):
                assert all([isinstance(i, str) for i in which]), (
                    f"The argument `which` needs to be provided a sequence of str. "
                    f"You have offending types in this argument."
                )
                diff = set(which) - set(recognized_str)
                if diff:
                    raise Exception(
                        f"One or more of the str in `which` are not recognized. "
                        f"The argument `which` needs to be a sequence containing any "
                        f"of the following: {recognized_str}. The unrecognized str are: "
                        f"{diff}."
                    )

            # add the features
            for cf in which:
                if cf in UNDERSCORE_MAPPING:
                    cf = UNDERSCORE_MAPPING[cf]
                feature = getattr(features, cf)
                if not feature._use_angle and not feature._use_omega:
                    feature = feature(
                        trajs[0],
                        check_aas=True,
                        generic_labels=ensemble,
                    )
                elif feature._use_angle and not feature._use_omega:
                    feature = feature(
                        trajs[0],
                        deg=deg,
                        check_aas=check_aas,
                        generic_labels=ensemble,
                    )
                elif feature._use_angle and feature._use_omega:
                    feature = feature(
                        trajs[0],
                        deg=deg,
                        omega=omega,
                        check_aas=check_aas,
                        generic_labels=ensemble,
                    )
                else:
                    raise Exception(
                        f"Unkwon combination of `_use_angle` and `_use_omega` in "
                        f"class attributes of {feature=}"
                    )
                self.active_features.setdefault(top, []).append(feature)

            if top not in self.feature_containers:
                self.feature_containers[top] = FeatureContainer()
                self.feature_containers[top].active_features = self.active_features[top]

    @property
    def features(self) -> list[AnyFeature]:
        feats = []
        for features in self.active_features.items():
            feats.extend(list(features))
        return feats

    def transform(
        self,
        traj,
        outer_p: Optional[tqdm] = None,
        inner_p: Optional[tqdm] = None,
    ) -> np.ndarray:
        # otherwise, build feature vector.
        feature_vec = []

        for f in self.active_features[traj.top]:
            xyz = traj.xyz
            if traj._have_unitcell:
                unitcell_vectors = traj.unitcell_vectors
                unitcell_info = np.hstack([traj.unitcell_lengths, traj.unitcell_angles])
            else:
                unitcell_vectors = None
                unitcell_info = None
            if isinstance(f, CustomFeature):
                vec = f.transform(
                    traj,
                    xyz,
                    unitcell_vectors,
                    unitcell_info,
                ).astype(np.float32, casting="safe")
                if vec.shape[0] == 0:
                    vec = np.empty((0, f.dimension))

                if not isinstance(vec, np.ndarray):
                    raise ValueError(
                        "Your custom feature %s did not return"
                        " a numpy.ndarray!" % str(f.describe())
                    )
                if not vec.ndim == 2:
                    raise ValueError(
                        "Your custom feature %s did not return"
                        " a 2d array. Shape was %s"
                        % (str(f.describe()), str(vec.shape))
                    )
                if not vec.shape[0] == self.traj.xyz.shape[0]:
                    raise ValueError(
                        "Your custom feature %s did not return"
                        " as many frames as it received!"
                        "Input was %i, output was %i"
                        % (str(f.describe()), self.traj.xyz.shape[0], vec.shape[0])
                    )
            else:
                vec = f.transform(
                    xyz,
                    unitcell_vectors,
                    unitcell_info,
                ).astype(np.float32)
            feature_vec.append(vec)

            if outer_p is not None:
                if isinstance(outer_p, rich.progress.Progress):
                    outer_p.update(0, advance=1)
                else:
                    outer_p.update()

            if inner_p is not None:
                if isinstance(inner_p, rich.progress.Progress):
                    outer_p.update(traj.traj_num + 1, advance=1)
                else:
                    outer_p.update()

        if len(feature_vec) > 1:
            res = np.hstack(feature_vec)
        else:
            res = feature_vec[0]

        return res

    def n_features(self) -> int:
        for i, (key, val) in enumerate(self.active_features.items()):
            if i == 0:
                length = len(val)
            else:
                assert length == len(val), (
                    f"There are different number of features per topology in "
                    f"`self.active_features`. These features can't be transformed."
                )
        return length

    def get_output(
        self,
        pbar: Optional[tqdm] = None,
    ) -> xr.Dataset:
        DSs = []
        n_features = self.n_features()
        if pbar is None:
            with rich.progress.Progress() as progress:
                tasks = []
                overall_tasks = progress.add_task(
                    description=(
                        f"Getting output for an ensemble containing "
                        f"{self.trajs.n_trajs} trajs"
                    ),
                    total=n_features * self.trajs.n_trajs,
                )
                for i, traj in enumerate(self.trajs):
                    tasks.append(
                        progress.add_task(
                            description=(
                                f"Getting output of {n_features} features for "
                                f"{traj.basename}"
                            ),
                            total=n_features,
                        )
                    )
                for i, traj in enumerate(self.trajs):
                    out = self.transform(traj, progress, progress)
                    ds = unpack_data_and_feature(
                        self.feature_containers[traj.top], traj, out
                    )
                    DSs.append(ds)
        else:
            for i, traj in enumerate(self.trajs):
                out = self.transform(traj, pbar, None)
                ds = unpack_data_and_feature(
                    self.feature_containers[traj.top], traj, out
                )
                DSs.append(ds)
        return format_output(DSs)


def format_output(
    datasets: Sequence[xr.Dataset],
) -> xr.Dataset:
    """Concatenates multiple xr.Datasets and keeps coordinates in correct order.

    Iterates over the labels in the coords that are not `traj`, `time` and picks
    the one with the greatest dimension. These labels will be used as
    the column names, the non-defined values are np.nan.

    Args:
        datasets (Sequence[xr.Dataset]): The datasets to combine.

    Returns:
        xr.Dataset: The output dataset.

    """
    # make sure that all traj-nums are unique
    traj_nums = [ds.traj_num.values for ds in datasets]
    assert all([i.size == 1 for i in traj_nums])
    traj_nums = np.array(traj_nums)[:, 0]
    assert len(traj_nums) == len(np.unique(traj_nums)), (
        f"The sequence of datasets provided for arg `datasets` contains multiple "
        f"traj_nums: {traj_nums=}"
    )

    # create a large dataset
    out = xr.concat(datasets, dim="traj_num", fill_value=np.nan)

    # EncoderMap datasets
    encodermap_dataarrays = list(UNDERSCORE_MAPPING.keys())

    all_labels = {}
    for name, da in out.data_vars.items():
        if name not in encodermap_dataarrays:
            continue
        feature_axis = da.attrs["feature_axis"]
        labels = da.coords[feature_axis].values
        all_labels.setdefault(feature_axis, []).extend(labels)

    # side dihedrals and central dihedrals need some special sorting
    # the other generic labels can be sorted by their last int
    for key, val in all_labels.items():
        all_labels[key] = np.unique(all_labels[key])

        if key == "SIDE_DIHEDRALS":
            all_labels[key] = sorted(
                all_labels[key],
                key=lambda x: (
                    int(re.findall(r"\d+", x)[-1]),
                    int(re.findall(r"\d+", x)[0]),
                ),
            )
        elif key == "CENTRAL_DIHEDRALS":
            all_labels[key] = sorted(
                all_labels[key],
                key=lambda x: (
                    int(re.findall(r"\d+", x)[-1]),
                    1 if "PSI" in x else (2 if "OMEGA" in x else 3),
                ),
            )
        else:
            all_labels[key] = sorted(
                all_labels[key], key=lambda x: int(re.findall(r"\d+", x)[-1])
            )
    return out.reindex(all_labels, fill_value=np.nan)


################################################################################
# Deprecated Stuff, because PyEMMA got archived we do featurization on our own
################################################################################

# def format_output_deprecated(
#     inps: list["DataSource"],
#     feats: list["AnyFeature"],
#     trajs: list[TrajEnsemble],
# ) -> tuple[tuple[np.ndarray], tuple[Featurizer], tuple[SingleTraj]]:
#     """Formats the output of multiple topologies.
#
#     Iterates over the features in `feats` and looks for the feature
#     with the greatest dimension, i.e., the longest returned `describe()`. This
#     feature yields the column names, the non-defined values are np.nan.
#
#     Args:
#         inps (list[DataSource]): The list of inputs, that
#             return the values of the feats, when `get_output()` is called.
#         feats (list[encodermap.loading.Featurizer]: These featurizers collect the
#             features and will be used to determine the highest length of feats.
#         trajs (list[encodermap.TrajEnsemble]): List of trajs with
#             identical topologies.
#
#     Returns:
#         tuple[list[np.ndarray], list[Featurizer], list[SingleTraj]: The
#             data, that `TrajEnsemble` can work with.
#
#     """
#
#     class Featurizer_out:
#         def __init__(self):
#             self.indices_by_top = {}
#
#     # append to this
#     all_out = []
#
#     feat_out = Featurizer_out()
#     feat_out.features = []
#     max_feat_lengths = {}
#     labels = {}
#     added_trajs = 0
#     for feat_num, (feat, traj) in enumerate(zip(feats, trajs)):
#         assert len(traj.top) == 1
#         for i, f in enumerate(feat.feat.active_features):
#             name = f.__class__.__name__
#
#             if name not in max_feat_lengths:
#                 max_feat_lengths[name] = 0
#                 feat_out.features.append(
#                     EmptyFeature(name, len(f.describe()), f.describe(), f.indexes)
#                 )
#
#             if name == "SideChainDihedrals":
#                 if name not in labels:
#                     labels[name] = []
#                 labels[name].extend(f.describe())
#             else:
#                 if max_feat_lengths[name] < len(f.describe()):
#                     max_feat_lengths[name] = len(f.describe())
#                     labels[name] = f.describe()
#                     feat_out.features[i] = EmptyFeature(
#                         name, len(f.describe()), f.describe(), f.indexes
#                     )
#             feat_out.indices_by_top.setdefault(traj.top[0], {})[name] = f.indexes
#
#     # rejig the sidechain labels
#     side_key = "SideChainDihedrals"
#     if side_key in labels:
#         labels[side_key] = np.unique(labels[side_key])
#         labels[side_key] = sorted(
#             labels[side_key], key=lambda x: (int(x[-3:]), int(x[13]))
#         )
#         index_of_sidechain_dihedral_features = [
#             f.name == side_key for f in feat_out.features
#         ].index(True)
#         new_empty_feat = EmptyFeature(
#             side_key,
#             len(labels[side_key]),
#             labels[side_key],
#             None,
#         )
#         feat_out.features[index_of_sidechain_dihedral_features] = new_empty_feat
#
#     # after rejigging the sidechain labels,
#     # reset the sidechain indices
#     for (top, index_dict), feat, traj in zip(
#         feat_out.indices_by_top.items(), feats, trajs
#     ):
#         feat_labels = feat.features[-1].describe()
#         check = np.in1d(np.asarray(labels[side_key]), np.asarray(feat_labels))
#         labels_copy = index_dict[side_key].copy()
#         feat_out.indices_by_top[top][side_key] = np.full((len(check), 4), np.nan, float)
#         feat_out.indices_by_top[top][side_key][check] = labels_copy
#
#     for (k, v), f in zip(labels.items(), feat_out.features):
#         if not len(v) == len(f.describe()) == f._dim:
#             raise Exception(
#                 f"Could not consolidate the features of the {f.name} "
#                 f"feature. The `labels` dict, which dictates the size "
#                 f"of the resulting array with np.nan's defines a shape "
#                 f"of {len(v)}, but the feature defines a shape of {len(f.describe())} "
#                 f"(or `f._dim = {f._dim}`). The labels dict gives these labels:\n\n{v}"
#                 f"\n\n, the feature labels gives these labels:\n\n{f.describe()}."
#             )
#
#     # Flatten the labels. These will be the columns for a pandas dataframe.
#     # At the start, the dataframe will be full of np.nan.
#     # The values of inp.get_output() will then be used in conjunction with
#     # The labels of the features to fill this dataframe partially.
#     flat_labels = [item for sublist in labels.values() for item in sublist]
#     if not len(flat_labels) == sum([f._dim for f in feat_out.features]):
#         raise Exception(
#             f"The length of the generic CV labels ({len(flat_labels)} "
#             f"does not match the length of the labels of the generic features "
#             f"({[f._dim for f in feat_out.features]})."
#         )
#
#     # iterate over the sorted trajs, inps, and feats
#     for inp, feat, sub_trajs in zip(inps, feats, trajs):
#         # make a flat list for this specific feature space
#         assert isinstance(sub_trajs, TrajEnsemble)
#         describe_this_feature = []
#         for f in feat.feat.active_features:
#             # make sure generic labels are used
#             if f.describe.__func__.__name__ != "generic_describe":
#                 raise Exception(
#                     f"It seems like this feature: {f.__class__} does not return generic "
#                     f"feature names (i.e. labels), but topology-specific ones (generic: 'SIDECHDIH CHI1 1', "
#                     f"topology specific: 'SIDECHDIH CHI1 ASP1'). Normally, EncoderMap's "
#                     f"features can be instantiated with a `generic_labels=True` flag to "
#                     f"overwrite the features `describe()` method with a `generic_describe()` "
#                     f"method. This changes the `.__func__.__name__` of the `describe()` method "
#                     f"to 'generic_describe'. However the func name for this feature is "
#                     f"{f.describe.__func__.__name__}."
#                 )
#             describe_this_feature.extend(f.describe())
#
#         # use the output to fill a pandas dataframe with all labels
#         out = inp.get_output()
#
#         # case1: one traj in sub_trajs: output is a list of length 1
#         if sub_trajs.n_trajs == 1:
#             assert len(out) == 1
#             traj = sub_trajs[0]
#             o = out[0]
#             if traj.index != (None,):
#                 for ind in traj.index:
#                     if ind is None:
#                         continue
#                     o = o[ind]
#                 assert o.shape[0] == traj.n_frames, (
#                     f"Indexing output of featurizer for `TrajEnsemble` with single traj "
#                     f"did not return the correct shape: {o.shape[0]=}, {traj.n_frames=}."
#                 )
#             df = pd.DataFrame(np.nan, index=range(traj.n_frames), columns=flat_labels)
#             df = df.assign(**{k: v for k, v in zip(describe_this_feature, o.T)})
#             all_out.append((df.to_numpy(), feat_out, traj))
#             added_trajs += 1
#         elif sub_trajs.n_trajs > 1:
#             if len(out) != sub_trajs.n_trajs:
#                 out_dict = {k: v for k, v in zip(inp._filenames, out)}
#                 assert len(out_dict) == len(out)
#                 for i, traj in enumerate(sub_trajs):
#                     o = deepcopy(out_dict[traj.traj_file])
#                     o_orig_shape = deepcopy(out_dict[traj.traj_file].shape)
#                     if traj.index != (None,):
#                         for ind in traj.index:
#                             if ind is None:
#                                 continue
#                             o = o[ind]
#                         assert o.shape[0] == traj.n_frames, (
#                             f"Indexing output of featurizer for `TrajEnsemble` with multiple trajs, some "
#                             f"of which share the same file {sub_trajs.traj_files=} "
#                             f"did not return the correct shape: {o.shape[0]=}, {traj.n_frames=}, "
#                             f"{len(traj._original_frame_indices)=}, {o_orig_shape=},"
#                             f"{[v.shape for v in out_dict.values()]=} {inp._filenames=}, "
#                             f"{[j.shape for j in out]=}"
#                         )
#                     df = pd.DataFrame(
#                         np.nan, index=range(traj.n_frames), columns=flat_labels
#                     )
#                     df = df.assign(**{k: v for k, v in zip(describe_this_feature, o.T)})
#                     all_out.append((df.to_numpy(), feat_out, traj))
#                     added_trajs += 1
#             else:
#                 for o, traj in zip(out, sub_trajs):
#                     if traj.index != (None,):
#                         for ind in traj.index:
#                             if ind is None:
#                                 continue
#                             o = o[ind]
#                         assert o.shape[0] == traj.n_frames
#                     df = pd.DataFrame(
#                         np.nan, index=range(traj.n_frames), columns=flat_labels
#                     )
#                     df = df.assign(**{k: v for k, v in zip(describe_this_feature, o.T)})
#                     all_out.append((df.to_numpy(), feat_out, traj))
#                     added_trajs += 1
#         else:
#             raise Exception("Unknown case of shared topologies and files.")
#
#     # make sure the shapes of all df matches
#     shapes = [o[0].shape[1] for o in all_out]
#     if not len(list(set(shapes))) == 1:
#         raise Exception(
#             f"Alignment was not possible. Some values exhibit different shapes: "
#             f"{list(set(shapes))}. All shapes:\n\n{[o[0].shape[1] for o in all_out]}"
#         )
#     assert added_trajs == sum(
#         [t.n_trajs for t in trajs]
#     ), f"{added_trajs=}, {len(all_out)=}, {sum([t.n_trajs for t in trajs])=}"
#     return tuple(all_out)
#
#
# class PyEMMAFeaturizer_deprecated:
#     def __init__(
#         self,
#         trajs: TrajEnsemble,
#         custom_aas: Optional[Union["CustomTopology", CustomAAsDict]] = None,
#     ) -> None:
#         """Instantiate the Featurizer.
#
#         Can be supplied with custom and non-standard aminoacid definitions.
#
#         Args:
#             trajs: Union[em.SingleTraj, em.TrajEnsemble]: The trajs.
#             custom_aas: Optional[Union[CustomAminoAcids, CustomAAsDict]]: An instance of the
#                 `CustomAminoAcids` class or a dictionary that can be made into such.
#
#         """
#         # Local Folder Imports
#         from ..trajinfo.trajinfo_utils import CustomTopology
#
#         # decide on custom_aas
#         if isinstance(custom_aas, dict):
#             self.custom_aas = CustomTopology.from_dict(custom_aas)
#         elif custom_aas is None:
#             self.custom_aas = None
#         elif (
#             isinstance(custom_aas, CustomTopology)
#             or custom_aas.__class__.__name__ == "CustomAminoAcids"
#         ):
#             self.custom_aas = custom_aas
#         else:
#             raise ValueError(
#                 f"Argument `custom_aas` needs to be `dict` or `CustomAminoAcids` "
#                 f"instance. Received {type(custom_aas)}."
#             )
#
#         # set the trajs and align them if needed
#         self._can_load = True
#         self.trajs = trajs
#
#         # copy docstrings form pyemma to the various add_* methods
#         self._copy_docstrings_from_pyemma()
#
#     def _copy_docstrings_from_pyemma(self):
#         """Copies the docstrings of the add* methods from the pyemma featurizer."""
#         if isinstance(self.feat, list):
#             feat_ = self.feat[0]
#         else:
#             feat_ = self.feat
#
#         # fmt: off
#         self.add_all.__func__.__doc__ = feat_.add_all.__doc__
#         self.add_selection.__func__.__doc__ = feat_.add_selection.__doc__
#         self.add_distances.__func__.__doc__ = feat_.add_distances.__doc__
#         self.add_distances_ca.__func__.__doc__ = feat_.add_distances_ca.__doc__
#         self.add_inverse_distances.__func__.__doc__ = feat_.add_inverse_distances.__doc__
#         self.add_contacts.__func__.__doc__ = feat_.add_contacts.__doc__
#         self.add_residue_mindist.__func__.__doc__ = feat_.add_residue_mindist.__doc__
#         self.add_group_COM.__func__.__doc__ = feat_.add_group_COM.__doc__
#         self.add_residue_COM.__func__.__doc__ = feat_.add_residue_COM.__doc__
#         self.add_group_mindist.__func__.__doc__ = feat_.add_group_mindist.__doc__
#         self.add_angles.__func__.__doc__ = feat_.add_angles.__doc__
#         self.add_dihedrals.__func__.__doc__ = feat_.add_dihedrals.__doc__
#         self.add_backbone_torsions.__func__.__doc__ = feat_.add_backbone_torsions.__doc__
#         self.add_chi1_torsions.__func__.__doc__ = feat_.add_chi1_torsions.__doc__
#         self.add_sidechain_torsions.__func__.__doc__ = feat_.add_sidechain_torsions.__doc__
#         self.add_minrmsd_to_ref.__func__.__doc__ = feat_.add_minrmsd_to_ref.__doc__
#         # fmt: on
#
#     def get_output(self) -> xr.Dataset:
#         """Gets the output of the feat obj(s)."""
#         if self.mode == "single_top":
#             if len(self.feat.active_features) == 0:
#                 print("No features loaded. No output will be returned")
#                 return
#
#         if self.mode == "multiple_top":
#             if len(self.feat[0].features) == 0:
#                 print("No features loaded. No output will be returned")
#                 return
#
#         if self.mode == "single_top":
#             datasets = []
#             if self._can_load:
#                 out = self.inp.get_output()
#             else:
#                 with tempfile.TemporaryDirectory() as td:
#                     fnames = [
#                         str(Path(td) / f"file_{i}.xtc")
#                         for i in range(self.trajs.n_trajs)
#                     ]
#                     for fname, t in zip(fnames, self.trajs):
#                         t.traj.save_xtc(fname)
#                     out = load(fnames, features=self.feat)
#                     if isinstance(out, np.ndarray):
#                         out = [out]
#             for traj, o in zip(self.trajs, out):
#                 datasets.append(unpack_data_and_feature(self, traj, o))
#             if len(datasets) == 1:
#                 assert datasets[0].coords["traj_num"] == np.array(
#                     [self.trajs[0].traj_num]
#                 )
#                 return datasets[0]
#             else:
#                 out = xr.combine_nested(
#                     datasets, concat_dim="traj_num", fill_value=np.nan
#                 )
#                 if (
#                     len(out.coords["traj_num"]) != len(self.trajs)
#                     and len(out.coords["traj_num"]) != self.trajs.n_trajs
#                 ):
#                     raise Exception(
#                         f"The combine_nested xarray method returned "
#                         f"a bad dataset, which has {out.coords['traj_num']} "
#                         f"trajectories, but the featurizer has {self.trajs} "
#                         f"trajectories."
#                     )
#                 # out = xr.concat(datasets, dim='traj_num')
#         else:
#             datasets = []
#             out = format_output(self.inp, self.feat, self.sorted_trajs)
#             for i, (data, feat, traj) in enumerate(out):
#                 ds = unpack_data_and_feature(feat, traj, data)
#                 assert ds.coords["traj_num"].values.tolist() == [traj.traj_num]
#                 datasets.append(ds)
#             try:
#                 out = xr.concat(datasets, dim="traj_num", fill_value=np.nan)
#             except ValueError as e:
#                 if "index has duplicate values" in str(e):
#                     for traj in self.trajs:
#                         for ind in traj.index:
#                             if isinstance(ind, (np.ndarray, list)):
#                                 uniques, counts = np.unique(
#                                     np.asarray(ind), return_counts=True
#                                 )
#                     raise Exception(
#                         f"One of the `SingleTraj`s has duplicate frames. "
#                         f"The frame {uniques[np.argmax(counts)]} appears "
#                         f"{np.max(counts)} times. This can happen, when a"
#                         f"`SingleTraj` is indexed like so:"
#                         f"`SingleTraj[[0, 1, 2, 3, 0]]` in which case the "
#                         f"frame 0 occurs twice. This is not forbidden per se, but "
#                         f"Featurization won't work with this kind of `SingleTraj`."
#                     )
#                 else:
#                     raise e
#
#         return out
#
#     def add_list_of_feats(
#         self,
#         which: Union[Literal["all"], Sequence[str]] = "all",
#         deg: bool = False,
#         omega: bool = True,
#         check_aas: bool = True,
#     ) -> None:
#         """Adds features to the Featurizer to be loaded either in-memory or out-of-memory.
#         `which` can be either 'all' or a list of the following strings. 'all' will add all of these features:
#         * 'AllCartesians': Cartesian coordinates of all atoms with shape (n_frames, n_atoms, 3).
#         * 'AllBondDistances': Bond distances of all bonds recognized by mdtraj. Use top = md.Topology.from_openmm()
#             if mdtraj does not recognize all bonds.
#         * 'CentralCartesians': Cartesians of the N, C, CA atoms in the backbone with shape (n_frames, n_residues * 3, 3).
#         * 'CentralBondDistances': The bond distances of the N, C, CA bonds with shape (n_frames, n_residues * 3 - 1).
#         * 'CentralAngles': The angles between the backbone bonds with shape (n_frames, n_residues * 3 - 2).
#         * 'CentralDihedrals': The dihedrals between the backbone atoms (omega, phi, psi). With shape (n_frames,
#             n_residues * 3 - 3).
#         * 'SideChainCartesians': Cartesians of the sidechain-atoms. Starting with CB, CG, ...
#         * 'SideChainBondDistances': Bond distances between the sidechain atoms. starting with the CA-CG bond.
#         * 'SideChainAngles': Angles between sidechain atoms. Starting with the C-CA-CB angle.
#         * 'SideChainDihedrals': Dihedrals of the sidechains (chi1, chi2, chi3).
#
#         Args:
#             which (Union[str, list], optional). Either add 'all' features or a list of features. See Above for
#                 possible features. Defaults to 'all'.
#
#         """
#         if isinstance(which, str):
#             if which == "all":
#                 which = [
#                     "CentralCartesians",
#                     "CentralBondDistances",
#                     "CentralAngles",
#                     "CentralDihedrals",
#                     "SideChainDihedrals",
#                 ]
#         if not isinstance(which, list):
#             which = [which]
#         if self.mode == "single_top":
#             for cf in which:
#                 if cf in UNDERSCORE_MAPPING:
#                     cf = UNDERSCORE_MAPPING[cf]
#                 feature = getattr(features, cf)(
#                     self.top,
#                     check_aas=check_aas,
#                 )
#                 if hasattr(feature, "deg"):
#                     feature = getattr(features, cf)(
#                         self.top,
#                         deg=deg,
#                         check_aas=check_aas,
#                     )
#                 if hasattr(feature, "omega"):
#                     feature = getattr(features, cf)(
#                         self.top, deg=deg, omega=omega, check_aas=check_aas
#                     )
#                 self.feat.add_custom_feature(feature)
#         else:
#             for cf in which:
#                 if cf in UNDERSCORE_MAPPING:
#                     cf = UNDERSCORE_MAPPING[cf]
#                 for top, feat in zip(self.top, self.feat):
#                     feature = getattr(features, cf)(
#                         top,
#                         generic_labels=True,
#                         check_aas=check_aas,
#                     )
#                     if hasattr(feature, "deg"):
#                         feature = getattr(features, cf)(
#                             top, deg=deg, generic_labels=True, check_aas=check_aas
#                         )
#                     if hasattr(feature, "omega"):
#                         feature = getattr(features, cf)(
#                             top,
#                             deg=deg,
#                             omega=omega,
#                             generic_labels=True,
#                             check_aas=check_aas,
#                         )
#                     feat.add_custom_feature(feature)
#
#     def add_all(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_all(*args, **kwargs)
#
#     def add_selection(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_selection(*args, **kwargs)
#
#     def add_distances(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_distances(*args, **kwargs)
#
#     def add_distances_ca(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_distances_ca(*args, **kwargs)
#
#     def add_inverse_distances(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_inverse_distances(*args, **kwargs)
#
#     def add_contacts(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_contacts(*args, **kwargs)
#
#     def add_residue_mindist(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_residue_mindist(*args, **kwargs)
#
#     def add_group_COM(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_group_COM(*args, **kwargs)
#
#     def add_residue_COM(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_residue_COM(*args, **kwargs)
#
#     def add_group_mindist(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_group_mindist(*args, **kwargs)
#
#     def add_angles(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_angles(*args, **kwargs)
#
#     def add_dihedrals(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_dihedrals(*args, **kwargs)
#
#     def add_backbone_torsions(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_backbone_torsions(*args, **kwargs)
#
#     def add_chi1_torsions(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_sidechain_torsions(which=["chi1"], *args, **kwargs)
#
#     def add_sidechain_torsions(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_sidechain_torsions(*args, **kwargs)
#
#     def add_minrmsd_to_ref(self, *args, **kwargs):
#         if self.mode == "multiple_top":
#             raise Exception(
#                 "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
#             )
#         self.feat.add_minrmsd_to_ref(*args, **kwargs)
#
#     def add_custom_feature(self, feature):
#         self.feat.add_custom_feature(feature)
#
#     @property
#     def features(self) -> list["AnyFeature"]:
#         if self.mode == "single_top":
#             return self.feat.active_features
#         else:
#             return [f.features for f in self.feat]
#
#     @property
#     def sorted_info_single(self):
#         if self.mode == "single_top":
#             raise Exception(
#                 "Attribute is only accessible, when working with multiple topologies."
#             )
#         out = []
#         for info_all in self.sorted_trajs:
#             for traj in info_all:
#                 out.append(traj)
#         return out
#
#     @property
#     def sorted_featurizers(self):
#         if self.mode == "single_top":
#             raise Exception(
#                 "Attribute is only accessible, when working with multiple topologies."
#             )
#         out = []
#         for feat, info_all in zip(self.feat, self.sorted_trajs):
#             out.extend([feat for i in range(info_all.n_trajs)])
#         return out
#
#     @property
#     def trajs(self):
#         return self._trajs
#
#     def describe(self):
#         return self.feat.describe()
#
#     @trajs.setter
#     def trajs(self, trajs):
#         # a single traj
#         if isinstance(trajs, SingleTraj) or trajs.__class__.__name__ == "SingleTraj":
#             self._trajs = trajs._gen_ensemble()
#             self.top = trajs.top
#             self.sorted_trajs = None
#
#             # fix topologies and prepare featurization
#             if self.custom_aas is not None:
#                 # Local Folder Imports
#                 from ..misc.backmapping import _add_bonds
#
#                 _add_bonds(self.trajs, self.custom_aas)
#                 self.custom_aas.inject_topologies(self.trajs)
#                 self.custom_aas.add_definitions(features)
#                 self.top = trajs.top
#
#             # instantiate the featurizer and inform the user
#             # about problems using https://...pdb files
#             self.feat = featurizer(self.top)
#             if _validate_uri(trajs.traj_file):
#                 self.inp = source([trajs.xyz], features=self.feat)
#             else:
#                 try:
#                     self.inp = source([trajs.traj_file], features=self.feat)
#                 except (ValueError, ProgrammingError) as e:
#                     if "SQL" not in str(e) and "input files" not in str(e):
#                         raise e
#                     if "SQL" in str(e):
#                         self.inp = CoordsLoad()
#                         self.inp.get_output = lambda: load(
#                             [trajs.traj_file], features=self.feat
#                         )
#                     if "input files" in str(e):
#                         self._can_load = False
#                 except OSError as e:
#                     if "Could not determine delimiter" not in str(e):
#                         raise e
#                     self._can_load = False
#                     # with tempfile.NamedTemporaryFile(suffix=".pdb") as tf:
#                     #     trajs[0].traj.save_pdb(tf.name)
#                     #     assert Path(tf.name).is_file()
#                     #     self.inp = CoordsLoad()
#                     #     self.inp.get_output = lambda: load(
#                     #         [tf.name], features=self.feat
#                     #     )
#
#             # set the mode
#             self.mode = "single_top"
#
#         # multiple trajs
#         elif (
#             isinstance(trajs, TrajEnsemble)
#             or trajs.__class__.__name__ == "TrajEnsemble"
#         ):
#             # with differing
#             if len(trajs.top) > 1:
#                 self._trajs = trajs
#                 self.top = trajs.top
#                 self.sorted_trajs = list(self.trajs.trajs_by_top.values())
#
#                 # fix topologies and prepare featurization
#                 if self.custom_aas is not None:
#                     # Local Folder Imports
#                     from ..misc.backmapping import _add_bonds
#
#                     _add_bonds(self.trajs, self.custom_aas)
#                     self.custom_aas.inject_topologies(self.trajs)
#                     self.custom_aas.add_definitions(features)
#                     self.top = trajs.top
#
#                 # in this case, `self.feat` is a list of featurizer objects
#                 # that will produce their own outputs, which will
#                 # be combined afterward
#                 self.feat = [Featurizer(t) for t in self.sorted_trajs]
#
#                 # the input sources are also a list
#                 self.inp = [
#                     source([t.traj_file for t in t_subset], features=feat.feat)
#                     for t_subset, feat in zip(self.sorted_trajs, self.feat)
#                 ]
#
#                 # set the mode
#                 self.mode = "multiple_top"
#
#             # with the same topology
#             else:
#                 self._trajs = trajs
#                 self.top = trajs.top[0]
#
#                 # fix topologies and prepare featurization
#                 if self.custom_aas is not None:
#                     # Local Folder Imports
#                     from ..misc.backmapping import _add_bonds
#
#                     _add_bonds(self.trajs, self.custom_aas)
#                     self.custom_aas.inject_topologies(self.trajs)
#                     self.custom_aas.add_definitions(features)
#                     self.top = trajs.top[0]
#
#                 # self.featurizer is again a single featurizer
#                 self.feat = featurizer(self.top)
#
#                 # try to create a datasource, which can fail in case some
#                 # trajs are instantiated from an MDTRaj trajectory
#                 if all([_validate_uri(traj.traj_file) for traj in trajs]):
#                     self.inp = source(trajs.xtc, features=self.feat)
#                 else:
#                     try:
#                         self.inp = source(
#                             [traj.traj_file for traj in trajs], features=self.feat
#                         )
#                     except ValueError as e:
#                         if "did not exists" in str(e):
#                             if any(
#                                 [isinstance(t.__traj, md.Trajectory) for t in trajs]
#                             ):
#                                 raise NotImplementedError(
#                                     "One of your trajectories was "
#                                     "instantiated from `mdtraj.Trajectory`,"
#                                     "which is not compatible with PyEMMA's "
#                                     "featurization. In future we could add a"
#                                     "Featurizer that uses in-memory mdtrajs. "
#                                     "Let us know if you'd like to have this feature."
#                                 )
#                         raise Exception(
#                             f"{trajs=}, {trajs[0].basename=}, {[t.traj_file for t in trajs]=}"
#                         ) from e
#
#                 # set the mode
#                 self.mode = "single_top"
#         else:
#             raise TypeError(
#                 f"trajs must be {SingleTraj.__class__.__name__} or "
#                 f"{TrajEnsemble.__class__.__name__}, you provided {trajs.__class__.__name__}"
#             )
#
#     def __len__(self):
#         if self.mode == "single_top":
#             return len(self.feat.active_features)
#         else:
#             return len([f.features for f in self.feat])
#
#     def __str__(self):
#         if self.mode == "single_top":
#             return self.feat.__str__()
#         else:
#             return ", ".join([f.__str__() for f in self.feat])
#
#     def __repr__(self):
#         if self.mode == "single_top":
#             return self.feat.__repr__()
#         else:
#             return ", ".join([f.__repr__() for f in self.feat])
#
#
# class EmptyFeature:
#     """Class to fill with attributes to be read by encodermap.xarray.
#
#     This class will be used in multiple_top mode, where the attributes
#     _dim, describe and name will be overwritten with correct values to
#     build features that contain NaN values.
#
#     """
#
#     def __init__(self, name, _dim, description, indexes):
#         """Initialize the Empty feature.
#
#         Args:
#             name (str): The name of the feature.
#             _dim (int): The feature length of the feature shape=(n_frames, ferature).
#             description (list of str): The description for every feature.
#
#         """
#         self.name = name
#         self._dim = _dim
#         self.description = description
#         self.indexes = indexes
#
#     def describe(self):
#         return self.description
#
#
# class Topologiesdeprecated:
#     def __init__(self, tops, alignments=None):
#         self.tops = tops
#         if alignments is None:
#             alignments = [
#                 "side_dihedrals",
#                 "central_cartesians",
#                 "central_distances",
#                 "central_angles",
#                 "central_dihedrals",
#             ]
#         self.alignments = {k: {} for k in alignments}
#         self.compare_tops()
#         allowed_strings = list(
#             filter(
#                 lambda x: True if "side" in x else False,
#                 (k for k in UNDERSCORE_MAPPING.keys()),
#             )
#         )
#         if not all([i in allowed_strings for i in alignments]):
#             raise Exception(
#                 f"Invalid alignment string in `alignments`. Allowed strings are {allowed_strings}"
#             )
#
#     def compare_tops(self):
#         if not all([t.n_residues == self.tops[0].n_residues for t in self.tops]):
#             raise Exception(
#                 "Using Different Topologies currently only works if all contain the same number of residues."
#             )
#         generators = [t.residues for t in self.tops]
#         sidechains = [t.select("sidechain") for t in self.tops]
#         all_bonds = [
#             list(map(lambda x: (x[0].index, x[1].index), t.bonds)) for t in self.tops
#         ]
#
#         # iterate over residues of the sequences
#         n_res_max = max([t.n_residues for t in self.tops])
#         for i in range(n_res_max):
#             # get some info
#             residues = [next(g) for g in generators]
#             all_atoms = [[a.name for a in r.atoms] for r in residues]
#             atoms = [
#                 list(
#                     filter(
#                         lambda x: True
#                         if x.index in sel and "H" not in x.name and "OXT" not in x.name
#                         else False,
#                         r.atoms,
#                     )
#                 )
#                 for r, sel in zip(residues, sidechains)
#             ]
#             atoms_indices = [[a.index for a in atoms_] for atoms_ in atoms]
#             bonds = [
#                 list(
#                     filter(
#                         lambda bond: True if any([b in ai for b in bond]) else False, ab
#                     )
#                 )
#                 for ai, ab in zip(atoms_indices, all_bonds)
#             ]
#
#             # reduce the integers of atoms_indices and bonds, so that N is 0. That way, we can compare them, even, when
#             # two amino aicds in the chains are different
#             N_indices = [
#                 list(filter(lambda x: True if x.name == "N" else False, r.atoms))[
#                     0
#                 ].index
#                 for r in residues
#             ]
#
#             # align to respective N
#             atoms_indices = [
#                 [x - N for x in y] for y, N in zip(atoms_indices, N_indices)
#             ]
#             bonds = [
#                 [(x[0] - N, x[1] - N) for x in y] for y, N in zip(bonds, N_indices)
#             ]
#
#             chi1 = [
#                 any(set(l).issubset(set(a)) for l in features.CHI1_ATOMS)
#                 for a in all_atoms
#             ]
#             chi2 = [
#                 any(set(l).issubset(set(a)) for l in features.CHI2_ATOMS)
#                 for a in all_atoms
#             ]
#             chi3 = [
#                 any(set(l).issubset(set(a)) for l in features.CHI3_ATOMS)
#                 for a in all_atoms
#             ]
#             chi4 = [
#                 any(set(l).issubset(set(a)) for l in features.CHI4_ATOMS)
#                 for a in all_atoms
#             ]
#             chi5 = [
#                 any(set(l).issubset(set(a)) for l in features.CHI5_ATOMS)
#                 for a in all_atoms
#             ]
#             chi = np.array([chi1, chi2, chi3, chi4, chi5])
#
#             self.alignments["side_dihedrals"][f"residue_{i}"] = chi
#
#             if "side_cartesians" in self.alignments:
#                 raise NotImplementedError(
#                     "Cartesians between different topologies can currently not be aligned."
#                 )
#
#             if "side_distances" in self.alignments:
#                 raise NotImplementedError(
#                     "Distances between different topologies can currently not be aligned."
#                 )
#
#             if "side_angles" in self.alignments:
#                 raise NotImplementedError(
#                     "Angles between different topologies can currently not be aligned."
#                 )
#
#         self.drop_double_false()
#
#     def drop_double_false(self):
#         """Drops features that None of the topologies have.
#
#         For example: Asp and Glu. Asp has a chi1 and chi2 torsion. Glu has chi1, chi2 and chi3. Both
#         don't have chi4 or chi5. In self.compare_tops these dihedrals are still considered. In this
#         method they will be removed.
#
#         """
#         for alignment, value in self.alignments.items():
#             for residue, array in value.items():
#                 where = np.where(np.any(array, axis=1))[0]
#                 self.alignments[alignment][residue] = array[where]
#
#     def get_max_length(self, alignment):
#         """Maximum length that a feature should have given a certain axis.
#
#         Args:
#             alignment (str): The key for `self.alignments`.
#
#         """
#         alignment_dict = self.alignments[alignment]
#         stacked = np.vstack([v for v in alignment_dict.values()])
#         counts = np.count_nonzero(stacked, axis=0)  # Flase is 0
#         return np.max(counts)
#
#     def format_output(self, inputs, feats, sorted_trajs):
#         """Formats the output of an em.Featurizer object using the alignment info.
#
#         Args:
#             inputs (list): List of pyemma.coordinates.data.feature_reader.FeatureReader objects.
#             feats (list): List of encodermap.Featurizer objetcs.
#             sorted_trajs (list): List of em.TrajEnsemble objects sorted in the same way as `self.tops`.
#
#         """
#         out = []
#         for i, (inp, top, feat, trajs) in enumerate(
#             zip(inputs, self.tops, feats, sorted_trajs)
#         ):
#             value_dict = {}
#             for traj_ind, (data, traj) in enumerate(zip(inp.get_output(), trajs)):
#                 if any(
#                     [isinstance(o, EmptyFeature) for o in feat.feat.active_features]
#                 ):
#                     # Local Folder Imports
#                     from ..misc.xarray import add_one_by_one
#
#                     ffunc = lambda x: True if "NaN" not in x else False
#                     indices = [0] + add_one_by_one(
#                         [len(list(filter(ffunc, f.describe()))) for f in feat.features]
#                     )
#                 else:
#                     indices = get_indices_by_feature_dim(feat, traj, data.shape)
#
#                 # divide the values returned by PyEMMA
#                 for f, ind in zip(feat.features, indices):
#                     try:
#                         name = FEATURE_NAMES[f.name]
#                     except KeyError:
#                         name = f.__class__.__name__
#                         f.name = name
#                     except AttributeError:
#                         name = f.__class__.__name__
#                         f.name = name
#                     if traj_ind == 0:
#                         value_dict[name] = []
#                     value_dict[name].append(data[:, ind])
#
#             # stack along the frame axis, just like pyemma would
#             value_dict = {k: np.vstack(v) for k, v in value_dict.items()}
#
#             # put nans in all features specified by alignment
#             for alignment, alignment_dict in self.alignments.items():
#                 if alignment not in value_dict:
#                     continue
#                 max_length = self.get_max_length(alignment)
#                 new_values = np.full(
#                     shape=(value_dict[alignment].shape[0], max_length),
#                     fill_value=np.nan,
#                 )
#                 where = np.vstack([v for v in alignment_dict.values()])[:, i]
#                 new_values[:, where] = value_dict[alignment]
#                 value_dict[alignment] = new_values
#
#                 # find the index of the feature in feat.feat.active_features
#                 names = np.array(
#                     [f.__class__.__name__ for f in feat.feat.active_features]
#                 )
#                 index = np.where([n in FEATURE_NAMES for n in names])[0]
#                 index = index[
#                     np.where([FEATURE_NAMES[n] == alignment for n in names[index]])
#                 ]
#
#                 # get the old description and change it around
#                 assert len(index) == 1
#                 index = index[0]
#                 if not isinstance(feat.feat.active_features[index], EmptyFeature):
#                     old_desc = np.array(
#                         [i for i in feat.feat.active_features[index].describe()]
#                     )
#                     new_desc = np.array(
#                         [
#                             f"NaN due to ensemble with other topologies {i}"
#                             for i in range(max_length)
#                         ]
#                     )
#                     new_desc[where] = old_desc
#                     new_desc = new_desc.tolist()
#
#                     # get the old indexes and add the NaNs
#                     old_indexes = feat.feat.active_features[index].indexes
#                     new_indexes = np.full(
#                         shape=(max_length, old_indexes.shape[1]), fill_value=np.nan
#                     )
#                     new_indexes[where] = old_indexes
#
#                     # create empty feature
#                     new_class = EmptyFeature(
#                         alignment, max_length, new_desc, new_indexes
#                     )
#                     feat.feat.active_features[index] = new_class
#             assert isinstance(trajs, TrajEnsemble)
#             new_values = np.hstack([v for v in value_dict.values()])
#             out.append([new_values, feat, trajs])
#         return out
#
#     def __iter__(self):
#         self._index = 0
#         return self
#
#     def __next__(self):
#         if self._index >= len(self.tops):
#             raise StopIteration
#         else:
#             self._index += 1
#             return self.tops[self._index - 1]

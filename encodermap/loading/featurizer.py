# -*- coding: utf-8 -*-
# encodermap/loading/featurizer.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
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
import os
import re
import time
import warnings
from pathlib import Path

# Third Party Imports
import numpy as np
from optional_imports import _optional_import

# Encodermap imports
from encodermap.loading import features
from encodermap.loading.delayed import build_dask_xarray


################################################################################
# Optional Imports
################################################################################


xr = _optional_import("xarray")
md = _optional_import("mdtraj")
rich = _optional_import("rich")
Client = _optional_import("dask", "distributed.Client")
dask = _optional_import("dask")
Callback = _optional_import("dask", "callbacks.Callback")
dot_graph = _optional_import("dask", "dot.dot_graph")
progress = _optional_import("dask", "distributed.progress")
HDF5TrajectoryFile = _optional_import("mdtraj", "formats.HDF5TrajectoryFile")
_get_global_client = _optional_import("distributed", "client._get_global_client")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, Optional, Union


if TYPE_CHECKING:
    # Third Party Imports
    import dask
    import xarray as xr
    from dask import dot_graph
    from dask.callbacks import Callback
    from dask.distributed import Client, progress
    from distributed.client import _get_global_client

    # Encodermap imports
    from encodermap.loading.features import AnyFeature
    from encodermap.trajinfo.info_all import TrajEnsemble
    from encodermap.trajinfo.info_single import SingleTraj


################################################################################
# Import tqdm which can be either the jupyter one or the plain one
################################################################################


def _is_notebook() -> bool:  # pragma: no cover
    """Checks, whether code is currently executed in a notebook."""
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


__all__: list[str] = ["Featurizer", "DaskFeaturizer"]


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

_ADD_X_FUNCTION_NAMES: list[str] = [
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
    "add_backbone_torsions",
    "add_sidechain_torsions",
]


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

    Returns:
        list[np.ndarray]: The resulting list of arrays.

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


class Track(Callback):
    def __init__(
        self,
        path: str = "/tmp.json/dasks",
        save_every: int = 1,
    ) -> None:
        self.path = path
        self.save_every = save_every
        self.n = 0
        os.makedirs(path, exist_ok=True)

    def _plot(
        self,
        dsk,
        state,
    ) -> None:
        data = {}
        func = {}
        for key in state["released"]:
            data[key] = {"color": "blue"}
        for key in state["cache"]:
            data[key] = {"color": "red"}
        for key in state["finished"]:
            func[key] = {"color": "blue"}
        for key in state["running"]:
            func[key] = {"color": "red"}

        filename = os.path.join(self.path, "part_{:0>4d}".format(self.n))

        dot_graph(
            dsk,
            filename=filename,
            format="png",
            data_attributes=data,
            function_attributes=func,
        )

    def _pretask(
        self,
        key,
        dsk,
        state,
    ) -> None:
        if self.n % self.save_every == 0:
            self._plot(dsk, state)
        self.n += 1

    def _finish(
        self,
        dsk,
        state,
        errored,
    ) -> None:
        self._plot(dsk, state)
        self.n += 1


################################################################################
# Classes
################################################################################


class SingleTrajFeaturizer:
    def __init__(self, traj: SingleTraj, delayed: bool = False) -> None:
        self.traj = traj
        self.delayed = delayed
        self._n_custom_features = 0
        self._custom_feature_ids = []
        self.active_features = []

    def add_list_of_feats(
        self,
        which: Union[Literal["all", "full"], Sequence[str]] = "all",
        deg: bool = False,
        omega: bool = True,
        check_aas: bool = True,
        periodic: bool = True,
        delayed: bool = False,
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
            if which == "full":
                which = [
                    "CentralCartesians",
                    "CentralBondDistances",
                    "CentralAngles",
                    "CentralDihedrals",
                    "SideChainDihedrals",
                    "SideChainCartesians",
                    "SideChainAngles",
                    "AllCartesians",
                    "SideChainBondDistances",
                ]
            elif which == "all":
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
            if (
                not feature._use_periodic
                and not feature._use_angle
                and not feature._use_omega
            ):
                feature = feature(
                    self.traj,
                    check_aas=True,
                    delayed=delayed,
                )
            elif (
                feature._use_periodic
                and not feature._use_angle
                and not feature._use_omega
            ):
                feature = feature(
                    self.traj,
                    check_aas=True,
                    periodic=periodic,
                    delayed=delayed,
                )
            elif (
                feature._use_periodic and feature._use_angle and not feature._use_omega
            ):
                feature = feature(
                    self.traj,
                    deg=deg,
                    check_aas=check_aas,
                    periodic=periodic,
                    delayed=delayed,
                )
            elif feature._use_periodic and feature._use_angle and feature._use_omega:
                feature = feature(
                    self.traj,
                    deg=deg,
                    omega=omega,
                    check_aas=check_aas,
                    periodic=periodic,
                    delayed=delayed,
                )
            else:
                raise Exception(
                    f"Unknown combination of `_use_angle` and `_use_omega` in "
                    f"class attributes of {feature=}"
                )
            self._add_feature(feature)

    def add_custom_feature(self, feature: AnyFeature) -> None:
        # Encodermap imports
        from encodermap.loading.features import CustomFeature

        if not hasattr(feature, "name"):
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
            assert isinstance(feature, CustomFeature) or issubclass(
                feature.__class__, CustomFeature
            )
            feature.name = f"CustomFeature_{feature.id}"
        self._add_feature(feature)

    def _add_feature(self, feature: AnyFeature) -> None:
        """Adds any feature to the list of current features.

        Also checks whether the feature is already part of the active features.

        """
        assert feature.delayed == self.delayed, (
            f"In-memory featurizer {self.__class__} unexpectedly got a delayed "
            f"feature {feature}. {feature.delayed=} {self.delayed=}"
        )
        if feature.dimension == 0:
            warnings.warn(
                f"Given an empty feature (eg. due to an empty/ineffective "
                f"selection). Skipping it. Feature desc: {feature.describe()}"
            )
            return
        if feature not in self.active_features:
            self.active_features.append(feature)
        else:
            warnings.warn(
                f"Tried to re-add the same feature {feature.__class__.__name__} to "
                f"{self.active_features=}"
            )

    def add_distances_ca(
        self,
        periodic: bool = True,
        excluded_neighbors: int = 2,
        delayed: bool = False,
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

        self.add_distances(distance_indexes, periodic=periodic, delayed=delayed)

    def add_distances(
        self,
        indices: Union[np.ndarray, Sequence[int]],
        periodic: bool = True,
        indices2: Optional[Sequence[int]] = None,
        delayed: bool = False,
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
        f = DistanceFeature(self.traj, atom_pairs, periodic=periodic, delayed=delayed)
        self._add_feature(f)

    def add_backbone_torsions(
        self,
        selstr: Optional[int] = None,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        delayed: bool = False,
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
            >>> from pprint import pprint
            >>> trajs = em.load_project("linear_dimers")
            >>> feat = em.Featurizer(trajs[0])
            >>> feat.add_backbone_torsions("resname PRO")
            >>> pprint(feat.describe())
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
            self.traj,
            selstr=selstr,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
            delayed=delayed,
        )
        self._add_feature(f)

    def add_angles(
        self,
        indexes: np.ndarray,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        delayed: bool = False,
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
            delayed=delayed,
        )
        self._add_feature(f)

    def add_all(
        self,
        reference: Optional[md.Trajectory] = None,
        atom_indices: Optional[np.ndarray] = None,
        ref_atom_indices: Optional[np.ndarray] = None,
        delayed: bool = False,
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
            delayed=delayed,
        )

    def add_selection(
        self,
        indexes: np.ndarray,
        reference: Optional[np.ndarray] = None,
        atom_indices: Optional[np.ndarray] = None,
        ref_atom_indices: Optional[np.ndarray] = None,
        delayed: bool = False,
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
            f = SelectionFeature(self.traj, indexes, delayed=delayed)
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
                delayed=delayed,
            )
        self._add_feature(f)

    def add_inverse_distances(
        self,
        indices: Union[np.ndarray, Sequence[int]],
        periodic: bool = True,
        indices2: Optional[Union[np.ndarray, Sequence[int]]] = None,
        delayed: bool = False,
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
        f = InverseDistanceFeature(
            self.traj, atom_pairs, periodic=periodic, delayed=delayed
        )
        self._add_feature(f)

    def add_contacts(
        self,
        indices: Union[np.ndarray, Sequence[int]],
        indices2: Optional[Union[np.ndarray, Sequence[int]]] = None,
        threshold: float = 0.3,
        periodic: bool = True,
        count_contacts: bool = False,
        delayed: bool = False,
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
        f = ContactFeature(
            self.traj, atom_pairs, threshold, periodic, count_contacts, delayed=delayed
        )
        self._add_feature(f)

    def add_residue_mindist(
        self,
        residue_pairs: Union[Literal["all"], np.ndarray] = "all",
        scheme: Literal["ca", "closest", "closest-heavy"] = "closest-heavy",
        ignore_nonprotein: bool = True,
        threshold: Optional[float] = None,
        periodic: bool = True,
        count_contacts: bool = False,
        delayed: bool = False,
    ) -> None:
        """Adds the minimum distance between residues to the feature list.
        See below how the minimum distance can be defined. If the topology
        generated out of `traj` contains information on periodic boundary
        conditions, the minimum image convention will be used when computing
        distances.

        Args:
            residue_pairs (Union[Literal["all"], np.ndarray]): Can be 'all', in
                which case mindists will be calculated between all pairs of
                residues excluding first and second neighbor. If a np.array
                with shape (n ,2) is supplied, these residue indices (0-based)
                will be used to compute the mindists. Defaults to 'all'.
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
            delayed=delayed,
        )

        self._add_feature(f)

    def add_group_COM(
        self,
        group_definitions: Sequence[int],
        ref_geom: Optional[md.Trajectory] = None,
        image_molecules: bool = False,
        mass_weighted: bool = True,
        delayed: bool = False,
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
            delayed=delayed,
        )
        self._add_feature(f)

    def add_residue_COM(
        self,
        residue_indices: Sequence[int],
        scheme: Literal["all", "backbone", "sidechain"] = "all",
        ref_geom: Optional[md.Trajectory] = None,
        image_molecules: bool = False,
        mass_weighted: bool = True,
        delayed: bool = False,
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
            delayed=delayed,
        )

        self._add_feature(f)

    def add_dihedrals(
        self,
        indexes: np.ndarray,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        delayed: bool = False,
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
            delayed=delayed,
        )
        self._add_feature(f)

    def add_sidechain_torsions(
        self,
        selstr: Optional[str] = None,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        which: Union[
            Literal["all"], Sequence[Literal["chi1", "chi2", "chi3", "chi4", "chi5"]]
        ] = "all",
        delayed: bool = False,
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
            delayed=delayed,
        )
        self._add_feature(f)

    def add_minrmsd_to_ref(
        self,
        ref: Union[md.Trajectory, SingleTraj],
        ref_frame: int = 0,
        atom_indices: Optional[np.ndarray] = None,
        precentered: bool = False,
        delayed: bool = False,
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
            delayed=delayed,
        )
        self._add_feature(f)

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
        # Encodermap imports
        from encodermap.loading.features import CustomFeature

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
                        f"Your custom feature {f.describe()} did not return a "
                        f"2d array. Shape was {vec.shape}"
                    )
                if not vec.shape[0] == self.traj.xyz.shape[0]:
                    raise ValueError(
                        f"Your custom feature {f.describe()} did not return as "
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
        # Encodermap imports
        from encodermap.misc.xarray import unpack_data_and_feature

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
        return unpack_data_and_feature(self, self.traj, out)

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
    """EncoderMap's featurization revives the archived code from PyEMMA
    (https://github.com/markovmodel/PyEMMA).

    EncoderMap's Featurizer collects and computes collective variables (CVs).
    CVs are data that are aligned with MD trajectories on the frame/time axis.
    Trajectory data contains (besides the topology) an axis for atoms, and
    an axis for cartesian coordinate (x, y, z), so that a trajectory can be
    understood as an array with shape (n_frames, n_atoms, 3). A CV is an array
    that is aligned with the frame/time and has its own feature axis. If the
    trajectory in our example has 3 residues (MET, ALA, GLY), we can define
    6 dihedral angles along the backbone of this peptide. These angles are:

    * PSI1:   Between MET1-N  - MET1-CA - MET1-C  - ALA2-N
    * OMEGA1: Between MET1-CA - MET1-C  - ALA2-N  - ALA2-CA
    * PHI1:   Between MET1-C  - ALA2-N  - ALA2-CA - ALA2-C
    * PSI2:   Between ALA2-N  - ALA2-CA - ALA2-C  - GLY3-N
    * OMEGA2: Between ALA2-CA - ALA2-C  - GLY3-N  - GLY3-CA
    * PHI2:   Between ALA2-C  - GLY3-N  - GLY3-CA - GLY3-C

    Thus, the collective variable 'backbone-dihedrals' provides an array of
    shape (n_frames, 6) and is aligned with the frame/time axis of the trajectory.

    """

    def __new__(cls, traj: Union[SingleTraj, TrajEnsemble]):
        # Encodermap imports
        from encodermap.trajinfo.info_single import SingleTraj

        if isinstance(traj, SingleTraj):
            return SingleTrajFeaturizer(traj)
        else:
            return EnsembleFeaturizer(traj)


class AddSingleFeatureMethodsToClass(type):
    """Metaclass that programatically adds methods to the EnsembleFeaturizer."""

    def __new__(cls, name, bases, dct):  # pragma: no doccheck
        x = super().__new__(cls, name, bases, dct)

        # iteratively add these functions
        for add_X_function_name in _ADD_X_FUNCTION_NAMES:
            # create a function with the corresponding add_X_function_name
            # IMPORTANT: keep this as a keyword argument, to prevent
            # python from late-binding
            def add_X_func(
                self, *args, add_x_name=add_X_function_name, **kwargs
            ) -> None:
                # iterate over the trajs in self.trajs
                for top, trajs in self.trajs.trajs_by_top.items():
                    # create a featurizer
                    if top not in self.feature_containers:
                        f = SingleTrajFeaturizer(trajs[0], delayed=self.delayed)
                        self.feature_containers[top] = f
                    else:
                        f = self.feature_containers[top]
                    # get the method defined by pyemma_function_name
                    func = getattr(f, add_x_name)
                    # call the method with *args and **kwargs, so that the
                    # correct feature is added
                    func(*args, **kwargs)
                    # this is the feature we are looking for.
                    feature = f.active_features[-1]
                    # add the feature
                    if top in self.active_features:
                        if feature in self.active_features[top]:
                            continue
                        else:
                            self.active_features[top].append(feature)
                    else:
                        self.active_features.setdefault(top, []).append(feature)

            # also add the docstring :)
            add_X_func.__doc__ = getattr(
                SingleTrajFeaturizer, add_X_function_name
            ).__doc__
            setattr(x, add_X_function_name, add_X_func)
        return x


class DaskFeaturizerMeta(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)

        for add_X_function_name in _ADD_X_FUNCTION_NAMES + ["add_list_of_feats"]:

            def add_X_func(self, *args, add_x_name=add_X_function_name, **kwargs):
                # call the parents featurizer class add function
                assert self.feat.delayed, (
                    f"Programmatically added `add_X_func` got a featurizer with a"
                    f"wrong `delayed` variable: {id(self.feat)=} {self.feat.delayed=}"
                )
                getattr(self.feat, add_x_name)(*args, delayed=True, **kwargs)

            add_X_func.__doc__ = getattr(
                SingleTrajFeaturizer,
                add_X_function_name,
            ).__doc__
            setattr(x, add_X_function_name, add_X_func)
        return x


class EnsembleFeaturizer(metaclass=AddSingleFeatureMethodsToClass):
    """The EnsembleFeaturizer is a container of multiple SinlgeTrajFeaturizer.

    The `SingleTrajFeaturizer` are collected in a dict with the topologies
    of the sub-ensembles as keys.

    """

    def __init__(self, trajs: TrajEnsemble, delayed: bool = False) -> None:
        """Instantiates the `EnsembleFeaturizer`.

        Args:
            trajs (TrajEnsmble): The `TrajEnsemble` to featurizer.
            delayed (bool): Whether using dask to calculate features, or just do
                a regular featurization.

        """
        self.trajs = trajs
        self.delayed = delayed
        self.active_features = {}
        self.feature_containers = {}
        self.ensemble = False
        self._n_custom_features = 0
        self._custom_feature_ids = []

    def describe(self) -> dict[md.Topology, list[str]]:
        """Returns the labels of the feature output.

        Returns:
            dict[md.Topology, list[str]]: A dict where the keys are the
                topologies in the `TrajEnsemble` and the values are the
                `describe()` outputs of the `SingleTrajFeaturizer` classes.

        """
        out = {}
        for top, container in self.feature_containers.items():
            out[top] = container.describe()
        return out

    def __len__(self) -> int:
        lengths = [len(f) for f in self.feature_containers.values()]
        assert all(
            [lengths[0] == length for length in lengths]
        ), f"This `{self.__class__.__name__}` has uneven features per topology."
        if len(lengths) < 1:
            return 0
        return lengths[0]

    def _add_feature(
        self, f: AnyFeature, top: md.Topology, trajs: TrajEnsemble
    ) -> None:
        assert f.delayed == self.delayed, (
            f"In-memory featurizer {self.__class__} unexpectedly got a delayed "
            f"feature {f}. {f.delayed=} {self.delayed=}"
        )
        if top in self.feature_containers:
            feat = self.feature_containers[top]
        else:
            feat = SingleTrajFeaturizer(trajs[0], delayed=self.delayed)
            self.feature_containers[top] = feat
        feat._add_feature(f)
        self.active_features.setdefault(top, []).append(f)

    def add_custom_feature(self, feature: AnyFeature) -> None:
        # Encodermap imports
        from encodermap.loading.features import CustomFeature

        # decide on feature's id
        if feature.__class__.__name__ == "CustomFeature":
            if not hasattr(feature, "name"):
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
                assert (
                    isinstance(feature, CustomFeature)
                    or issubclass(feature.__class__, CustomFeature)
                    or hasattr(feature, "_is_custom")
                )
                feature.name = f"CustomFeature_{feature.id}"
        else:
            try:
                feature.name = feature.__class__.__name__
            except AttributeError:
                pass

        # add
        for top, trajs in self.trajs.trajs_by_top.items():
            self._add_feature(feature, top, trajs)

    def add_list_of_feats(
        self,
        which: Union[Literal["all"], Sequence[str]] = "all",
        ensemble: bool = False,
        deg: bool = False,
        omega: bool = True,
        check_aas: bool = True,
        periodic: bool = True,
        delayed: bool = False,
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
                if which == "full":
                    which = [
                        "CentralCartesians",
                        "CentralBondDistances",
                        "CentralAngles",
                        "CentralDihedrals",
                        "SideChainDihedrals",
                        "SideChainCartesians",
                        "SideChainAngles",
                        "AllCartesians",
                        "SideChainBondDistances",
                    ]
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
            # the _use_omega and _use_angle class attrs are added by the
            # `FeatureMeta` in `features.py` by inspecting a `Feature` subclass'
            # call signature
            for cf in which:
                if cf in UNDERSCORE_MAPPING:
                    cf = UNDERSCORE_MAPPING[cf]
                feature = getattr(features, cf)
                if (
                    not feature._use_periodic
                    and not feature._use_angle
                    and not feature._use_omega
                ):
                    feature = feature(
                        trajs[0],
                        check_aas=True,
                        generic_labels=ensemble,
                        delayed=delayed,
                    )
                elif (
                    feature._use_periodic
                    and not feature._use_angle
                    and not feature._use_omega
                ):
                    feature = feature(
                        trajs[0],
                        check_aas=True,
                        generic_labels=ensemble,
                        periodic=periodic,
                        delayed=delayed,
                    )
                elif (
                    feature._use_periodic
                    and feature._use_angle
                    and not feature._use_omega
                ):
                    feature = feature(
                        trajs[0],
                        deg=deg,
                        check_aas=check_aas,
                        generic_labels=ensemble,
                        periodic=periodic,
                        delayed=delayed,
                    )
                elif (
                    feature._use_periodic and feature._use_angle and feature._use_omega
                ):
                    feature = feature(
                        trajs[0],
                        deg=deg,
                        omega=omega,
                        check_aas=check_aas,
                        generic_labels=ensemble,
                        periodic=periodic,
                        delayed=delayed,
                    )
                else:
                    raise Exception(
                        f"Unknown combination of `_use_angle` and `_use_omega` in "
                        f"class attributes of {feature=}"
                    )
                if top in self.active_features:
                    if feature in self.active_features[top]:
                        warnings.warn(
                            f"Tried to re-add the same feature {feature.__class__.__name__} to "
                            f"{self.active_features=}"
                        )
                        continue
                    else:
                        self.active_features[top].append(feature)
                else:
                    self.active_features.setdefault(top, []).append(feature)

                if top in self.feature_containers:
                    f = self.feature_containers[top]
                else:
                    f = SingleTrajFeaturizer(trajs[0], delayed=self.delayed)
                    self.feature_containers[top] = f
                f._add_feature(feature)

        # after all is done, all tops should contain the same number of feats
        no_of_feats = set([len(v) for v in self.active_features.values()])
        assert len(no_of_feats) == 1, (
            f"I was not able to add the same number of features to the respective "
            f"topologies:\n{self.active_features=}\n{self.feature_containers=}"
        )

    @property
    def features(self) -> list[AnyFeature]:
        feats = []
        for features in self.active_features.items():
            feats.extend(list(features))
        return feats

    def transform(
        self,
        traj: Union[SingleTraj, md.Trajectory],
        outer_p: Optional[Union[tqdm, rich.progress.Progress]] = None,
        inner_p: Optional[Union[tqdm, rich.progress.Progress]] = None,
        inner_p_id: Optional[int] = None,
    ) -> np.ndarray:
        """Applies the features to the trajectory data.

        traj (Union[SingleTraj, md.Trajectory]): The trajectory which provides
            the data. Make sure, that the topology of this traj matches the
            topology used to initialize the features.
        outer_p (Optional[Union[tqdm, rich.progress.Progress]]): An object
            that supports `.update()` to advance a progress bar. The
            `rich.progress.Progress` is special, as it needs additional code
            to advance the multi-file progress bar dispolayed by the
            `EnsembleFeaturzier`. The `outer_p` represents the overall
            progress.
        inner_p (Optional[Union[tqdm, rich.progress.Progress]]): Same as `outer_p`,
            but the `inner_p` represents the progress per file.
        inner_p_id (Optional[int]): The id of the `inner_p`, which needs to be
            provided, if `outer_p` and `inner_p` are instances of
            `rich.progress.Progress`.

        """
        # Encodermap imports
        from encodermap.loading.features import CustomFeature

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
                if vec.ndim == 1:
                    vec = np.expand_dims(vec, -1)
                if vec.ndim == 3:
                    vec = vec.reshape(xyz.shape[0], -1)
                    f.atom_feature = True
                if not vec.shape[0] == traj.xyz.shape[0]:
                    raise ValueError(
                        "Your custom feature %s did not return"
                        " as many frames as it received!"
                        "Input was %i, output was %i"
                        % (str(f.describe()), traj.xyz.shape[0], vec.shape[0])
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
                    if inner_p_id is None:
                        inner_p_id = traj.traj_num + 1
                    outer_p.update(inner_p_id, advance=1)
                else:
                    outer_p.update()

        if len(feature_vec) > 1:
            res = np.hstack(feature_vec)
        else:
            res = feature_vec[0]

        # sleep half a second to let the progbars catch up
        if outer_p is not None or inner_p is not None:
            time.sleep(0.5)

        return res

    def n_features(self) -> int:
        for i, (key, val) in enumerate(self.active_features.items()):
            if i == 0:
                length = len(val)
            else:
                _debug = []
                for i, (key, val) in enumerate(self.active_features.items()):
                    _debug.append(f"Top: {key}\nValue: {val}")
                _debug = "\n\n".join(_debug)
                assert length == len(val), (
                    f"There are different number of features per topology in "
                    f"`self.active_features`. These features can't be transformed. "
                    f"Here are the features by topology:\n{_debug}"
                )
        return length

    def get_output(
        self,
        pbar: Optional[tqdm] = None,
    ) -> xr.Dataset:
        # Encodermap imports
        from encodermap.misc.xarray import unpack_data_and_feature

        if self.active_features == {}:
            print(f"First add some features before calling `get_output()`.")
            return
        DSs = []
        n_features = self.n_features()

        try:
            # Third Party Imports
            from rich.progress import Progress

            _rich_installed = True
        except ModuleNotFoundError:
            _rich_installed = False

        if pbar is None and _rich_installed:
            with Progress() as progress:
                tasks = []
                progress.add_task(
                    description=(
                        f"Getting output for an ensemble containing "
                        f"{self.trajs.n_trajs} trajs"
                    ),
                    total=n_features * self.trajs.n_trajs,
                )
                for i, traj in enumerate(self.trajs):
                    desc = traj.basename
                    if traj.basename == "trajs":
                        desc = f"trajectory {traj.traj_num}"
                    tasks.append(
                        progress.add_task(
                            description=(
                                f"Getting output of {n_features} features for "
                                f"{desc}"
                            ),
                            total=n_features,
                        )
                    )
                for i, traj in enumerate(self.trajs):
                    out = self.transform(traj, progress, progress, inner_p_id=i + 1)
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
    # Encodermap imports
    from encodermap.trajinfo.trajinfo_utils import trajs_combine_attrs

    # make sure that all traj-nums are unique
    traj_nums = [ds.traj_num.values for ds in datasets]
    assert all([i.size == 1 for i in traj_nums])
    traj_nums = np.array(traj_nums)[:, 0]
    assert len(traj_nums) == len(np.unique(traj_nums)), (
        f"The sequence of datasets provided for arg `datasets` contains multiple "
        f"traj_nums: {traj_nums=}"
    )

    # create a large dataset
    out = xr.concat(
        datasets,
        data_vars="all",
        # compat="broadcast_equals",
        # coords="all",
        # join="outer",
        dim="traj_num",
        fill_value=np.nan,
        combine_attrs=trajs_combine_attrs,
    )

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
        elif key == "ALLATOM":
            all_labels[key] = sorted(
                all_labels[key],
                key=lambda x: (
                    0 if x.endswith("c") else 1,
                    *map(int, re.findall(r"\d+", x)[::-1]),
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


class DaskFeaturizer(metaclass=DaskFeaturizerMeta):
    """Container for `SingleTrajFeaturizer` and `EnsembleFeaturizer`
    that implements delayed transforms.

    The DaskFeaturizer is similar to the other two featurizer classes and
    mostly implements the same API. However, instead of computing the
    transformations using in-memory computing, it prepares a `xarray.Dataset`,
    which contains `dask.Arrays`. This dataset can be lazily and distributively
    evaluated using dask.distributed clients and clusters.

    """

    def __init__(
        self,
        trajs: Union[SingleTraj, TrajEnsemble],
        n_workers: Union[str, int] = "cpu-2",
        client: Optional[Client] = None,
    ) -> None:

        if not hasattr(trajs, "itertrajs"):
            self.feat = SingleTrajFeaturizer(trajs, delayed=True)
        else:
            self.feat = EnsembleFeaturizer(trajs, delayed=True)

        if n_workers == "cpu-2":
            # Standard Library Imports
            from multiprocessing import cpu_count

            n_workers = cpu_count() - 2

        if n_workers == "max":
            # Standard Library Imports
            from multiprocessing import cpu_count

            n_workers = cpu_count()

        dask.config.set(scheduler="processes")

        if client is None:
            self.client = _get_global_client()
        else:
            self.client = client
        if self.client is None:
            self.client = Client(n_workers=n_workers)
            print(
                f"Created dask scheduler. Access the dashboard via: "
                f"{self.client.dashboard_link}"
            )
        else:
            print(
                f"Using existing dask scheduler. Access the dashboard via: "
                f"{self.client.dashboard_link}"
            )

    def add_custom_feature(self, feature):
        if not hasattr(feature, "delayed"):
            feature.delayed = True
        if not feature.delayed:
            feature.delayed = True
        self.feat.add_custom_feature(feature)
        if hasattr(self, "dataset"):
            warnings.warn(
                f"The compute graph has already been built. I will rebuild the "
                f"graph and add the feature as a transformer. Subsequent "
                f"calls to `.get_output()` will include this feature."
            )
            self.build_graph()

    def build_graph(
        self,
        traj: Optional[SingleTraj] = None,
        streamable: bool = False,
        return_delayeds: bool = False,
    ) -> None:
        """Prepares the dask graph.

        Args:
            with_trajectories (Optional[bool]): Whether to also compute xyz.
                This can be useful if you want to also save the trajectories to disk.

        """
        if self.feat.active_features == {} or self.feat.active_features == []:
            print(f"First add some features before calling `get_output()`.")
            return

        self.dataset, self.variables = build_dask_xarray(
            self,
            traj=traj,
            streamable=streamable,
            return_delayeds=return_delayeds,
        )

    def to_netcdf(
        self,
        filename: Union[Path, str],
        overwrite: bool = False,
        with_trajectories: bool = False,
    ) -> str:
        """Saves the dask tasks to a NetCDF4 formatted HDF5 file.

        Args:
            filename (Union[str, list[str]]): The filename to be used.
            overwrite (bool): Whether to overwrite the existing filename.
            with_trajectories (bool): Also save the trajectory data. The output
                file can be read with `encodermap.load(filename)` and rebuilds
                the trajectories complete with traj_nums, common_str, custom_top,
                and all CVs, that this featurizer calculates.

        Returns:
            str: Returns the filename of the created files.

        """
        # Standard Library Imports
        from pathlib import Path

        filename = Path(filename)
        if "dataset" in self.__dict__:
            raise Exception(f"Graph already built.")

        # allows multiple writes to netcdf4 files
        def set_env():
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

        self.client.run(set_env)

        if filename.is_file() and not overwrite:  # pragma: nocover
            raise Exception(
                f"File {filename} already exists. Set `overwrite=True` to overwrite."
            )
        if filename.is_file() and overwrite:
            filename.unlink()

        # build
        self.build_graph(return_delayeds=with_trajectories)

        if self.variables is not None:
            # Third Party Imports
            import h5py
            from xarray import conventions
            from xarray.backends.api import (
                _finalize_store,
                _validate_attrs,
                _validate_dataset_names,
            )
            from xarray.backends.common import ArrayWriter
            from xarray.backends.h5netcdf_ import H5NetCDFStore

            # use xarrays's to_netcdf code and add saving of delayed coordinates, etc.
            _validate_dataset_names(self.dataset)
            _validate_attrs(self.dataset, invalid_netcdf=False)
            store_open = H5NetCDFStore.open
            have_chunks = any(
                v.chunks is not None for v in self.dataset.variables.values()
            )
            autoclose = have_chunks
            store = store_open(
                filename=filename,
                mode="a",
                format="NETCDF4",
                autoclose=autoclose,
            )
            writer = ArrayWriter()
            try:
                # create dicts of data to write to store
                variables, attrs = conventions.encode_dataset_coordinates(self.dataset)
                variables |= self.variables
                store.store(variables, attrs, set(), writer, None)
                store.close()
                writes = writer.sync(compute=False)
            finally:
                store.close()

            # this runs the computation and displays a progress
            delayed = dask.delayed(_finalize_store)(writes, store)
            delayed = delayed.persist()
            progress(delayed)
            delayed.compute()

            # afterward, we remove the unwanted groups starting with md from the .h5 file
            # they are artifacts of hijacking xarray's `to_netcdf`
            # we also move all keys that are not part of the traj coords
            md_keys = ["coordinates", "time", "cell_lengths", "cell_angles"]
            with h5py.File(filename, "a") as f:
                keys = list(f.keys())
                for key in filter(lambda k: k.startswith("md"), keys):
                    del f[key]
                keys = list(f.keys())
                for key in keys:
                    if not any([m in key for m in md_keys]):
                        f.move(key, f"CVs/{key}")

            # and add common_str, custom_top, etc.
            self.feat.trajs.save(fname=filename, CVs=False, only_top=True)
        else:
            self.dataset.to_netcdf(
                filename,
                format="NETCDF4",
                group="CVs",
                engine="h5netcdf",
                invalid_netcdf=False,
                compute=True,
            )
        return str(filename)

    def get_output(
        self,
        make_trace: bool = False,
    ) -> xr.Dataset:
        """This function passes the trajs and the features of to dask to create a
        delayed xarray out of that."""
        if "dataset" not in self.__dict__:
            self.build_graph()
        if not make_trace:
            ds = self.dataset.compute()
            if not ds:
                raise Exception(
                    f"Computed dataset is empty. Maybe a computation failed in "
                    f"the dask-delayed dataset: {self.dataset}"
                )
            # future = client.submit(future)
            # out = self.client.compute(self.dataset)
            # progress(out)
            # return out.result()
        else:
            raise NotImplementedError("Currently not able to trace dask execution.")
        # else:
        #     with tempfile.TemporaryDirectory() as tmpdir:
        #         tmpdir = Path(tmpdir)
        #         with Track(path=str(tmpdir)):
        #             out = self.client.compute(self.dataset)
        #             progress(out)
        #             return out.result()
        #
        #     raise NotImplementedError(
        #         "gifsicle --delay 10 --loop=forever --colors 256 --scale=0.4 -O3 --merge dasks/part_*.png > output.gif"
        #     )
        return ds

    @property
    def feature_containers(self) -> dict[md.Topology, SingleTrajFeaturizer]:
        return self.feat.feature_containers

    @property
    def active_features(
        self,
    ) -> Union[list[AnyFeature], dict[md.Topology, list[AnyFeature]]]:
        return self.feat.active_features

    def __len__(self):
        return len(self.feat)

    def transform(
        self,
        traj_or_trajs: Optional[Union[SingleTraj, TrajEnsemble]] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return self.feat.transform(traj_or_trajs, *args, **kwargs)

    def describe(self) -> list[str]:
        return self.feat.describe()

    def dimension(self) -> int:
        return self.feat.dimension

    def visualize(self) -> None:
        return dask.visualize(self.dataset)

# -*- coding: utf-8 -*-
# encodermap/loading/features.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade, Patricia Schwarz
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
"""Features contain topological information of proteins and other biomolecules.

These topological information can be calculated once and then provided with
input coordinates to calculate frame-wise collective variables of MD simulations.

The features in this module used to inherit from PyEMMA's features
(https://github.com/markovmodel/PyEMMA), but PyEMMA has since been archived.

If using EncoderMap's featurization make sure to also cite PyEMMA, from which
a lot of this code was adopted::

   @article{scherer_pyemma_2015,
        author = {Scherer, Martin K. and Trendelkamp-Schroer, Benjamin
                  and Paul, Fabian and Pérez-Hernández, Guillermo and Hoffmann, Moritz and
                  Plattner, Nuria and Wehmeyer, Christoph and Prinz, Jan-Hendrik and Noé, Frank},
        title = {{PyEMMA} 2: {A} {Software} {Package} for {Estimation},
                 {Validation}, and {Analysis} of {Markov} {Models}},
        journal = {Journal of Chemical Theory and Computation},
        volume = {11},
        pages = {5525-5542},
        year = {2015},
        issn = {1549-9618},
        shorttitle = {{PyEMMA} 2},
        url = {http://dx.doi.org/10.1021/acs.jctc.5b00743},
        doi = {10.1021/acs.jctc.5b00743},
        urldate = {2015-10-19},
        month = oct,
   }

"""

################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import inspect
import itertools
import warnings
from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, TypeVar, Union

# Third Party Imports
import numpy as np
from optional_imports import _optional_import


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    # Third Party Imports
    import dask
    import mdtraj as md

    # Encodermap imports
    from encodermap.trajinfo.info_all import TrajEnsemble
    from encodermap.trajinfo.info_single import SingleTraj
    from encodermap.trajinfo.trajinfo_utils import _AMINO_ACID_CODES


AllCartesiansType = TypeVar("AllCartesians", bound="Parent")
AllBondDistancesType = TypeVar("AllBondDistances", bound="Parent")
CentralCartesiansType = TypeVar("CentralCartesians", bound="Parent")
CentralBondDistancesType = TypeVar("CentralBondDistances", bound="Parent")
CentralAnglesType = TypeVar("CentralAngles", bound="Parent")
CentralDihedralsType = TypeVar("CentralDihedrals", bound="Parent")
SideChainCartesiansType = TypeVar("SideChainCartesians", bound="Parent")
SideChainBondDistancesType = TypeVar("SideChainBondDistances", bound="Parent")
SideChainAnglesType = TypeVar("SideChainAngles", bound="Parent")
SideChainDihedralsType = TypeVar("SideChainDihedrals", bound="Parent")
CustomFeatureType = TypeVar("CustomFeature", bound="Parent")
SelectionFeatureType = TypeVar("SelectionFeature", bound="Parent")
AngleFeatureType = TypeVar("AngleFeature", bound="Parent")
DihedralFeatureType = TypeVar("DihedralFeature", bound="Parent")
DistanceFeatureType = TypeVar("DistanceFeature", bound="Parent")
AlignFeatureType = TypeVar("AlignFeature", bound="Parent")
InverseDistanceFeatureType = TypeVar("InverseDistanceFeature", bound="Parent")
ContactFeatureType = TypeVar("ContactFeature", bound="Parent")
BackboneTorsionFeatureType = TypeVar("BackboneTorsionFeature", bound="Parent")
ResidueMinDistanceFeatureType = TypeVar("ResidueMinDistanceFeature", bound="Parent")
GroupCOMFeatureType = TypeVar("GroupCOMFeature", bound="Parent")
ResidueCOMFeatureType = TypeVar("ResidueCOMFeature", bound="Parent")
SideChainTorsionsType = TypeVar("SideChainTorsions", bound="Parent")
MinRmsdFeatureType = TypeVar("MinRmsdFeature", bound="Parent")


AnyFeature = Union[
    AllCartesiansType,
    AllBondDistancesType,
    CentralCartesiansType,
    CentralBondDistancesType,
    CentralAnglesType,
    CentralDihedralsType,
    SideChainCartesiansType,
    SideChainBondDistancesType,
    SideChainAnglesType,
    SideChainDihedralsType,
    CustomFeatureType,
    SelectionFeatureType,
    AngleFeatureType,
    DihedralFeatureType,
    DistanceFeatureType,
    AlignFeatureType,
    InverseDistanceFeatureType,
    ContactFeatureType,
    BackboneTorsionFeatureType,
    ResidueMinDistanceFeatureType,
    GroupCOMFeatureType,
    ResidueCOMFeatureType,
    SideChainTorsionsType,
    MinRmsdFeatureType,
]


################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")
_dist_mic = _optional_import("mdtraj", "geometry._geometry._dist_mic")
_dist = _optional_import("mdtraj", "geometry._geometry._dist")
_dihedral_mic = _optional_import("mdtraj", "geometry._geometry._dihedral_mic")
_dihedral = _optional_import("mdtraj", "geometry._geometry._dihedral")
_angle_mic = _optional_import("mdtraj", "geometry._geometry._angle_mic")
_angle = _optional_import("mdtraj", "geometry._geometry._angle")
box_vectors_to_lengths_and_angles = _optional_import(
    "mdtraj", "utils.unitcell.box_vectors_to_lengths_and_angles"
)
dask = _optional_import("dask")


################################################################################
# Globals
################################################################################

__all__: list[str] = [
    "AllCartesians",
    "AllBondDistances",
    "CentralCartesians",
    "CentralBondDistances",
    "CentralAngles",
    "CentralDihedrals",
    "SideChainCartesians",
    "SideChainBondDistances",
    "SideChainAngles",
    "SideChainDihedrals",
    "CustomFeature",
    "SelectionFeature",
    "AngleFeature",
    "DihedralFeature",
    "DistanceFeature",
    "AlignFeature",
    "InverseDistanceFeature",
    "ContactFeature",
    "BackboneTorsionFeature",
    "ResidueMinDistanceFeature",
    "GroupCOMFeature",
    "ResidueCOMFeature",
    "SideChainTorsions",
    "MinRmsdFeature",
]


PERIODIC_WARNING: bool = False
PYEMMA_CITATION_WARNING: bool = False
PYEMMA_FEATURES: list[str] = [
    "SelectionFeature",
    "AngleFeature",
    "DihedralFeature",
    "DistanceFeature",
    "AlignFeature",
    "InverseDistanceFeature",
    "ContactFeature",
    "BackboneTorsionFeature",
    "ResidueMinDistanceFeature",
    "GroupCOMFeature",
    "ResidueCOMFeature",
    "SideChainTorsions",
    "MinRmsdFeature",
]


################################################################################
# Functions
################################################################################


def pair(*numbers: int) -> int:
    """ConvertGroup's (https://convertgroup.com/) implementation of
    Matthew Szudzik's pairing function (http://szudzik.com/ElegantPairing.pdf)

    Maps a pair of non-negative integers to a uniquely associated single non-negative integer.
    Pairing also generalizes for `n` non-negative integers, by recursively mapping the first pair.
    For example, to map the following tuple:

    Args:
        *numbers (int): Variable length integers.

    Returns:
        int: The paired integer.

    """
    if len(numbers) < 2:
        raise ValueError("Szudzik pairing function needs at least 2 numbers as input")
    elif any((n < 0) or (not isinstance(n, int)) for n in numbers):
        raise ValueError(
            f"Szudzik pairing function maps only non-negative integers. In your "
            f"input, there seems to be negative or non-integer values: {numbers=}"
        )

    numbers = deque(numbers)

    # fetch the first two numbers
    n1 = numbers.popleft()
    n2 = numbers.popleft()

    if n1 != max(n1, n2):
        mapping = pow(n2, 2) + n1
    else:
        mapping = pow(n1, 2) + n1 + n2

    mapping = int(mapping)

    if not numbers:
        # recursion concludes
        return mapping
    else:
        numbers.appendleft(mapping)
        return pair(*numbers)


def unpair(number: int, n: int = 2) -> list[int]:
    """ConvertGroup's (https://convertgroup.com/) implementation of
    Matthew Szudzik's pairing function (http://szudzik.com/ElegantPairing.pdf)

    The inverse function outputs the pair associated with a non-negative integer.
    Unpairing also generalizes by recursively unpairing a non-negative integer to
    `n` non-negative integers.

    For example, to associate a `number` with three non-negative
    integers n_1, n_2, n_3, such that:

    pairing(n_1, n_2, n_3) = `number`

    the `number` will first be unpaired to n_p, n_3, then the n_p will be unpaired to n_1, n_2,
    producing the desired n_1, n_2 and n_3.

    Args:
        number(int): The paired integer.
        n (int): How many integers are paired in `number`?

    Returns:
        list[int]: A list of length `n` with the constituting ints.

    """
    if (number < 0) or (not isinstance(number, int)):
        raise ValueError("Szudzik unpairing function requires a non-negative integer")

    if number - pow(np.floor(np.floor(number)), 2) < np.floor(np.floor(number)):

        n1 = number - pow(np.floor(np.floor(number)), 2)
        n2 = np.floor(np.floor(number))

    else:
        n1 = np.floor(np.floor(number))
        n2 = number - pow(np.floor(np.floor(number)), 2) - np.floor(np.floor(number))

    n1, n2 = int(n1), int(n2)

    if n > 2:
        return [unpair(n1, n - 1) + (n2,)]
    else:
        # recursion concludes
        return [n1, n2]


def _check_aas(traj: SingleTraj) -> None:
    r = set([r.name for r in traj.top.residues])
    diff = r - set(traj._custom_top.amino_acid_codes.keys())
    if diff:
        raise Exception(
            f"I don't recognize these residues: {diff}. "
            f"Either add them to the `SingleTraj` or `TrajEnsemble` via "
            f"`traj.load_custom_topology(custom_aas)` or "
            f"`trajs.load_custom_topology(custom_aas)` "
            f"Or remove them from your trajectory. See the documentation of the "
            f"`em.CustomTopology` class. Here are the recognized residues:\n\n"
            f"{traj._custom_top.amino_acid_codes.keys()}"
        )


def describe_last_feats(feat: AnyFeature, n: int = 5) -> None:
    """Prints the description of the last `n` features.

    Args:
        feat (encodermap.Featurizer): An instance of a featurizer.
        n (int): The number of last features to describe. Default is 5.

    """
    for i, lbl in enumerate(feat.describe()[-n:]):
        print(lbl)


def _describe_atom(
    topology: md.Topology,
    index: int,
) -> str:
    """
    Returns a string describing the given atom.

    Args:
        topology (md.Topology): An MDTraj Topology.
        index (str): The index of the atom.

    Return:
        str: A description of the atom.

    """
    atom = topology.atom(index)
    if topology.n_chains > 1:
        return f"{atom.residue.name} {atom.residue.resSeq} {atom.name} {atom.index} {atom.residue.chain.index}"
    else:
        return f"{atom.residue.name} {atom.residue.resSeq} {atom.name} {atom.index}"


################################################################################
# Parent Classes
################################################################################


class CitePYEMMAWarning(UserWarning):
    pass


class FeatureMeta(type):
    """Inspects the __init__ of classes and adds attributes to them based on
    their call signature.

    If a feature uses the arguments `deg` or `omega` in
    its call signature, the instance will have the CLASS attributes `_use_angle` and
    `_use_omega` set to True. Otherwise, the instance will have them set as False.

    This allows other functions that use these features to easily discern whether
    they need these arguments before instantiating the classes.

    Example:
        >>> from encodermap.loading import features
        >>> f_class = getattr(features, "SideChainDihedrals")
        >>> f_class._use_angle
        True
        >>> f_class._use_omega
        False

    """

    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        args = inspect.getfullargspec(x.__init__)
        if args.varargs is not None:
            raise Exception(f"{x.__init__=} {x=} {args=} {args=}")
        args = args.args
        if "deg" in args:
            x._use_angle = True
        else:
            x._use_angle = False
        if "omega" in args:
            x._use_omega = True
        else:
            x._use_omega = False
        if "periodic" in args:
            x._use_periodic = True
        else:
            x._use_periodic = False
        x.atom_feature = False
        x._raise_on_unitcell = False
        return x


class Feature(metaclass=FeatureMeta):
    """Parent class to all feature classes. Implements the FeatureMeta,
     the transform method, and checks for unknown amino acids..

    This class implements functionality, that holds true for all features.
    The `transform()` method can be used by subclasses in two ways:
        * Provide all args with None. In this case, the traj in `self.traj`
            will be used to calculate the transformation.
        * Provide custom `xyz`, `unitcell_vectors`, and `unitcell_info`. In this
            case,

    """

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        check_aas: bool = True,
        periodic: Optional[bool] = None,
        delayed: bool = False,
    ) -> None:
        self.traj = traj
        self._raise_on_unitcell = False
        if isinstance(self.traj.top, list):
            assert len(self.traj.top) == 1, (
                f"The trajs in the features seem to have multiple toplogies: "
                f"{self.traj.top_files}. Features can only work with single "
                f"topologies."
            )
            self.top = self.traj.top[0]
        else:
            self.top = self.traj.top
        if check_aas:
            _check_aas(traj)

        self.delayed = delayed

        if periodic is not None:
            if periodic and self.traj._have_unitcell:
                self.periodic = True
            elif periodic and not self.traj._have_unitcell:
                self.periodic = False
                self._raise_on_unitcell
                global PERIODIC_WARNING
                if not PERIODIC_WARNING:
                    warnings.warn(
                        f"You requested a `em.loading.features.Feature` to calculate "
                        f"features in a periodic box, using the minimum image convention, "
                        f"but the trajectory you provided does not have "
                        f"unitcell information. If this feature will later be supplied "
                        f"with trajectories with unitcell information, an Exception "
                        f"will be raised, to make sure distances/angles are calculated "
                        f"correctly.",
                        stacklevel=2,
                    )
                    PERIODIC_WARNING = True
            else:
                self.periodic = False

        global PYEMMA_CITATION_WARNING
        if not PYEMMA_CITATION_WARNING and self.__class__.__name__ in PYEMMA_FEATURES:
            warnings.warn(
                message=(
                    "EncoderMap's featurization uses code from the now deprecated "
                    "python package PyEMMA (https://github.com/markovmodel/PyEMMA). "
                    "Please make sure to also cite them, when using EncoderMap."
                ),
                category=CitePYEMMAWarning,
            )
            PYEMMA_CITATION_WARNING = True

    @property
    def dimension(self) -> int:
        """int: The dimension of the feature."""
        return self._dim

    @dimension.setter
    def dimension(self, val: Union[float, int]) -> None:
        self._dim = int(val)

    def __eq__(self, other: AnyFeature) -> bool:
        if not issubclass(other.__class__, Feature):
            return False
        if not isinstance(other, self.__class__):
            return False
        if self.dimension != other.dimension:
            return False
        if self.traj is not None:
            if self.traj.top != other.traj.top:
                return False
        if hasattr(self, "ref"):
            if not np.allclose(self.ref.xyz, other.ref.xyz, rtol=1e-4):
                return False
        if hasattr(self, "scheme"):
            if self.scheme != other.scheme:
                return False
        if hasattr(self, "ignore_nonprotein"):
            if self.ignore_nonprotein != other.ignore_nonprotein:
                return False
        if hasattr(self, "periodic"):
            if self.periodic != other.periodic:
                return False
        if hasattr(self, "threshold"):
            if self.threshold != other.threshold:
                return False
        if hasattr(self, "group_definitions"):
            for self_group_def, other_group_def in zip(
                self.group_definitions, other.group_definitions
            ):
                if not np.array_equal(self_group_def, other_group_def):
                    return False
        if hasattr(self, "group_pairs"):
            if not np.array_equal(self.group_pairs, other.group_pairs):
                return False
        if hasattr(self, "count_contacts"):
            if self.count_contacts != other.count_contacts:
                return False
        # Encodermap imports
        from encodermap.misc.xarray import _get_indexes_from_feat

        try:
            self_index = _get_indexes_from_feat(self, self.traj)
            other_index = _get_indexes_from_feat(other, other.traj)
            if not np.array_equal(self_index, other_index):
                return False
        except AttributeError:
            pass
        return True

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Carries out the computation of the CVs.

        For featurization of single trajs, all arguments can be left None,
        and the values of the `traj` at class instantiation will be
        returned by this method. For ensembles with a single topology, but
        multiple trajectories, the xyz, unitcell_vectors, and unitcell_info
        should be provided accordingly. This parent class' `transform` then
        carries out checks (do all arguments provide the same number of frames,
        does the xyz array have the same number of atoms as the `traj` at
        instantiation, do the unitcell_angles coincide with the one of the
        parent traj, ...). Thus, it is generally advised to call this method
        with super() to run these checks.

        Args:
            xyz (Optional[np.ndarray]): If None, the coordinates of the
                trajectory in provided as `traj`, when the feature was instantiated
                will be used.
            unitcell_vectors (Optional[np.ndarray]): If None, the unitcell vectors
                of the trajectory in provided as `traj`, when the feature was instantiated
                will be used. Unitcell_vectors are arrays with shape (n_frames, 3, 3),
                where the rows are the bravais vectors a, b, c.
            unitcell_info (Optional[np.ndarray]): If None, the unitcell info of
                the trajectory in provided as `traj`, when the feature was
                instantiated will be used. The unitcell_info is an array with
                shape (n_frames, 6), where the first three columns are the unitcell
                lengths in nm, the remaining columns are the unitcell angles in deg.

        Returns:
            tuple: A tuple containing three np.ndarrays:
                - The xyz coordinates.
                - The unitcell_vectors
                - The unitcell_info

        """
        if self._raise_on_unitcell and (
            unitcell_info is not None or unitcell_vectors is not None
        ):
            raise Exception(
                f"This feature was instantiated with the keyword argument `periodic=True`, "
                f"but the `SingleTraj` used for instantiation did not contain any unitcell "
                f"information. Now, unitcell_infos are fed into the `transform` "
                f"method of this feature. This behavior is not allowed. Make sure to "
                f"either specifically set `periodic=False` or fix the unitcells in "
                f"your trajectory files."
            )
        if xyz is not None:
            input_atoms = xyz.shape[1]
            try:
                self_atoms = self.traj.xyz.shape[1]
            except AttributeError as e:
                raise Exception(f"{self=}") from e
            if hasattr(self, "periodic"):
                if self.periodic:
                    assert unitcell_vectors is not None and unitcell_info is not None, (
                        f"When providing a `feature.transform` function with xyz "
                        f"data, and setting {self.periodic=} to True, please "
                        f"also provide `unitcell_vectors` and `unitcell_info` "
                        f"to calculate distances/angles/dihedrals in periodic space."
                    )
            assert input_atoms == self_atoms, (
                f"The shape of the input xyz coordinates is off from the expected "
                f"shape. The topology {self.top} defines {self_atoms} atoms. The "
                f"provided array has {xyz.shaope[1]=} atoms."
            )
        else:
            xyz = self.traj.xyz.copy()
        if unitcell_vectors is not None:
            assert len(unitcell_vectors) == len(xyz), (
                f"The shape of the provided `unitcell_vectors` is off from the "
                f"expected shape. The xyz data contains {len(xyz)=} frames, while "
                f"the `unitcell_vectors` contains {len(unitcell_vectors)=} frames."
            )
        else:
            if self.traj._have_unitcell:
                unitcell_vectors = self.traj.unitcell_vectors.copy()
            else:
                unitcell_vectors = None
        if unitcell_info is not None:
            assert len(unitcell_info) == len(xyz), (
                f"The shape of the provided `unitcell_info` is off from the "
                f"expected shape. The xyz data contains {len(xyz)=} frames, while "
                f"the `unitcell_info` contains {len(unitcell_info)=} frames."
            )
            provided_orthogonal = np.allclose(unitcell_info[:, 3:], 90)
            self_orthogonal = np.allclose(self.traj.unitcell_angles, 90)
            assert provided_orthogonal == self_orthogonal, (
                f"The trajectory you provided to `transform` and the one "
                f"this feature was instantiated with have different crystal "
                f"systems in their unitcells: {provided_orthogonal=} {self_orthogonal=}"
            )
        else:
            if self.traj._have_unitcell:
                unitcell_info = np.hstack(
                    [
                        self.traj.unitcell_lengths.copy(),
                        self.traj.unitcell_angles.copy(),
                    ]
                )
            else:
                unitcell_info = None
        return xyz, unitcell_vectors, unitcell_info


class CustomFeature(Feature):
    delayed: bool = False
    _nonstandard_transform_args: list[str] = [
        "top",
        "indexes",
        "delayed_call",
        "_fun",
        "_args",
        "_kwargs",
    ]
    _is_custom: Final[True] = True
    traj: Optional[SingleTraj] = None
    top: Optional[md.Topology] = None
    indexes: Optional[np.ndarray] = None
    _fun: Optional[Callable] = None
    _args: Optional[tuple[Any, ...]] = None
    _kwargs: Optional[dict[str, Any]] = None

    def __init__(
        self,
        fun: Callable,
        dim: int,
        traj: Optional[SingleTraj] = None,
        description: Optional[str] = None,
        fun_args: tuple[Any, ...] = tuple(),
        fun_kwargs: dict[str, Any] = None,
        delayed: bool = False,
    ) -> None:
        self.id = None
        self.traj = traj
        self.indexes = None
        if fun_kwargs is None:
            fun_kwargs = {}
        self._fun = fun
        self._args = fun_args
        self._kwargs = fun_kwargs
        self._dim = dim
        self.desc = description
        self.delayed = delayed
        assert self._dim > 0, f"Feature dimensions need to be greater than 0."

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        if isinstance(self.desc, str):
            desc = [self.desc]
        elif self.desc is None:
            arg_str = (
                f"{self._args}, {self._kwargs}" if self._kwargs else f"{self._args}"
            )
            desc = [f"CustomFeature_{self.id} calling {self._fun} with args {arg_str}"]
        elif self.desc and not (len(self.desc) == self._dim or len(self.desc) == 1):
            raise ValueError(
                f"to avoid confusion, ensure the lengths of 'description' "
                f"list matches dimension - or give a single element which will be repeated."
                f"Input was {self.desc}"
            )

        if len(desc) == 1 and self.dimension > 0:
            desc *= self.dimension

        return desc

    @property
    def dask_indices(self):
        """str: The name of the delayed transformation to carry out with this feature."""
        return "indexes"

    @staticmethod
    @dask.delayed
    def dask_transform(
        top: md.Topology,
        indexes: np.ndarray,
        delayed_call: Optional[Callable] = None,
        _fun: Optional[Callable] = None,
        _args: Optional[Sequence[Any]] = None,
        _kwargs: Optional[dict[str, Any]] = None,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """The CustomFeature dask transfrom is still under development."""
        if unitcell_info is not None:
            traj = md.Trajectory(
                xyz=xyz,
                topology=top,
                unitcell_lengths=unitcell_info[:, :3],
                unitcell_angles=unitcell_info[:, 3:],
            )
        else:
            traj = md.Trajectory(
                xyz=xyz,
                topology=top,
            )

        if _kwargs is None:
            _kwargs = {}

        if delayed_call is not None:
            return delayed_call(traj, indexes, **_kwargs)
        else:
            if _args is None:
                _args = tuple()
            return _fun(traj, *_args, **_kwargs)

    def transform(
        self,
        traj: Optional[md.Trajectory] = None,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        if xyz is not None:
            self.traj = traj
            xyz, unitcell_vectors, unitcell_info = super().transform(
                xyz, unitcell_vectors, unitcell_info
            )
            if unitcell_info is not None:
                traj = md.Trajectory(
                    xyz=xyz,
                    topology=self.traj.top,
                    unitcell_lengths=unitcell_info[:, :3],
                    unitcell_angles=unitcell_info[:, 3:],
                )
            else:
                traj = md.Trajectory(
                    xyz=xyz,
                    topology=self.traj.top,
                )
        if hasattr(self, "call"):
            if xyz is None:
                traj = md.Trajectory(
                    xyz=self.traj.xyz.copy(),
                    topology=self.traj.top,
                    unitcell_lengths=deepcopy(traj.traj.unitcell_lengths),
                    unitcell_angles=deepcopy(traj.traj.unitcell_angles),
                )
            return self.call(traj)
        feature = self._fun(traj, *self._args, **self._kwargs)
        if not isinstance(feature, np.ndarray):
            raise ValueError("Your function should return a NumPy array!")
        return feature


class SelectionFeature(Feature):
    prefix_label: str = "ATOM:"

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        indexes: Sequence[int],
        check_aas: bool = True,
        delayed: bool = False,
    ) -> None:
        super().__init__(traj, check_aas, delayed=delayed)
        self.indexes = np.asarray(indexes).astype("int32")
        if len(self.indexes) == 0:
            raise ValueError(f"Empty indices in {self.__class__.__name__}.")
        self.dimension = 3 * len(self.indexes)

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        labels = []
        for i in self.indexes:
            labels.append(f"{self.prefix_label}{_describe_atom(self.top, i)} x")
            labels.append(f"{self.prefix_label}{_describe_atom(self.top, i)} y")
            labels.append(f"{self.prefix_label}{_describe_atom(self.top, i)} z")
        return labels

    @property
    def dask_indices(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "indexes"

    @staticmethod
    @dask.delayed
    def dask_transform(
        indexes: np.ndarray,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> dask.delayed:
        """The same as `transform()` but without the need to pickle `traj`.

        When dask delayed concurrencies are distributed, required python objects
        are pickled. Thus, every feature needs to have its own pickled traj.
        That defeats the purpose of dask distributed. Thus, this method implements
        the same calculations as `transform` as a more barebones approach.
        It foregoes the checks for periodicity and unit-cell shape and just
        takes xyz, unitcell vectors, and unitcell info. Furthermore, it is a
        staticmethod, so it doesn't require `self` to function. However, it
        needs the indexes in `self.indexes`. That's why the `dask_indices`
        property informs the scheduler to also pickle and pass this object to
        the workers.

        Args:
            indexes (np.ndarray): A numpy array with shape (n, ) giving the
                0-based index of the atoms which positions should be returned.
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        """
        newshape = (xyz.shape[0], 3 * indexes.shape[0])
        result = np.reshape(xyz[:, indexes, :], newshape)
        return result

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        xyz, unitcell_vectors, unitcell_info = super().transform(
            xyz, unitcell_vectors, unitcell_info
        )
        newshape = (xyz.shape[0], 3 * self.indexes.shape[0])
        result = np.reshape(xyz[:, self.indexes, :], newshape)
        return result


class AngleFeature(Feature):
    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        angle_indexes: np.ndarray,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        check_aas: bool = True,
        delayed: bool = False,
    ) -> None:
        """Instantiate the `AngleFeature` class.

        Args:
            traj (Union[SingleTraj, TrajEnsemble]): The trajectory container
                which topological information will be used to build the angles.
            angle_indexes (np.ndarray): A numpy array with shape (n_dihedrals, 4),
                that indexes the 3-tuples of atoms that will be used for
                the angle calculation.
            deg (bool): Whether to return the dihedrals in degree (True) or
                in radian (False). Defaults to False.
            cossin (bool): Whether to return the angles (False) or tuples of their
                cos and sin values (True). Defaults to False.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            check_aas (bool): Whether to check if all aas in `traj.top` are
                recognized. Defaults to True.

        """
        self.angle_indexes = np.array(angle_indexes).astype("int32")
        if len(self.angle_indexes) == 0:
            raise ValueError("empty indices")
        self.deg = deg
        self.cossin = cossin
        self.dimension = len(self.angle_indexes)
        if cossin:
            self.dimension *= 2
        super().__init__(traj, check_aas, periodic=periodic, delayed=delayed)

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        if self.cossin:
            sin_cos = ("ANGLE: COS(%s - %s - %s)", "ANGLE: SIN(%s - %s - %s)")
            labels = [
                s
                % (
                    _describe_atom(self.top, triple[0]),
                    _describe_atom(self.top, triple[1]),
                    _describe_atom(self.top, triple[2]),
                )
                for triple in self.angle_indexes
                for s in sin_cos
            ]
        else:
            labels = [
                "ANGLE: %s - %s - %s "
                % (
                    _describe_atom(self.top, triple[0]),
                    _describe_atom(self.top, triple[1]),
                    _describe_atom(self.top, triple[2]),
                )
                for triple in self.angle_indexes
            ]
        return labels

    @property
    def dask_indices(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "angle_indexes"

    @staticmethod
    @dask.delayed
    def dask_transform(
        indexes: np.ndarray,
        periodic: bool,
        deg: bool,
        cossin: bool,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> dask.delayed:
        """The same as `transform()` but without the need to pickle `traj`.

        When dask delayed concurrencies are distributed, required python objects
        are pickled. Thus, every feature needs to have its own pickled traj.
        That defeats the purpose of dask distributed. Thus, this method implements
        the same calculations as `transform` as a more barebones approach.
        It foregoes the checks for periodicity and unit-cell shape and just
        takes xyz, unitcell vectors, and unitcell info. Furthermore, it is a
        staticmethod, so it doesn't require `self` to function. However, it
        needs the indexes in `self.indexes`. That's why the `dask_indices`
        property informs the scheduler to also pickle and pass this object to
        the workers.

        Args:
            indexes (np.ndarray): A numpy array with shape (n, ) giving the
                0-based index of the atoms which positions should be returned.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            deg (bool): Whether to return the result in degree (`deg=True`) or in
                radians (`deg=False`). Defaults to False (radians).
            cossin (bool): If True, each angle will be returned as a pair of
                (sin(x), cos(x)). This is useful, if you calculate the means
                (e.g. TICA/PCA, clustering) in that space. Defaults to False.
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        """
        if periodic:
            assert unitcell_vectors is not None
            if unitcell_info is None:
                # convert to angles
                unitcell_angles = []
                for fr_unitcell_vectors in unitcell_vectors:
                    _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                        fr_unitcell_vectors[0],
                        fr_unitcell_vectors[1],
                        fr_unitcell_vectors[2],
                    )
                    unitcell_angles.append(np.array([alpha, beta, gamma]))
            else:
                unitcell_angles = unitcell_info[:, 3:]
            # check for an orthogonal box
            orthogonal = np.allclose(unitcell_angles, 90)

            out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
            _angle_mic(
                xyz,
                indexes,
                unitcell_vectors.transpose(0, 2, 1).copy(),
                out,
                orthogonal,
            )
        else:
            out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
            _angle(xyz, indexes, out)
        if cossin:
            out = np.dstack((np.cos(out), np.sin(out)))
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
        if deg and not cossin:
            out = np.rad2deg(out)
        return out

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        if xyz is not None:
            periodic = self.periodic
        else:
            periodic = self.periodic and self.traj._have_unitcell
        xyz, unitcell_vectors, unitcell_info = super().transform(
            xyz, unitcell_vectors, unitcell_info
        )
        if periodic:
            assert unitcell_vectors is not None
            if unitcell_info is None:
                # convert to angles
                unitcell_angles = []
                for fr_unitcell_vectors in unitcell_vectors:
                    _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                        fr_unitcell_vectors[0],
                        fr_unitcell_vectors[1],
                        fr_unitcell_vectors[2],
                    )
                    unitcell_angles.append(np.array([alpha, beta, gamma]))
            else:
                unitcell_angles = unitcell_info[:, 3:]
            # check for an orthogonal box
            orthogonal = np.allclose(unitcell_angles, 90)

            out = np.empty(
                (xyz.shape[0], self.angle_indexes.shape[0]), dtype="float32", order="C"
            )
            _angle_mic(
                xyz,
                self.angle_indexes,
                unitcell_vectors.transpose(0, 2, 1).copy(),
                out,
                orthogonal,
            )
        else:
            out = np.empty(
                (xyz.shape[0], self.angle_indexes.shape[0]), dtype="float32", order="C"
            )
            _angle(xyz, self.angle_indexes, out)
        if self.cossin:
            out = np.dstack((np.cos(out), np.sin(out)))
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
        if self.deg and not self.cossin:
            out = np.rad2deg(out)
        return out


class DihedralFeature(AngleFeature):
    """Dihedrals are torsion angles defined by four atoms."""

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        dih_indexes: np.ndarray,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        check_aas: bool = True,
        delayed: bool = False,
    ) -> None:
        """Instantiate the `DihedralFeature` class.

        Args:
            traj (Union[SingleTraj, TrajEnsemble]): The trajectory container
                which topological information will be used to build the dihedrals.
            dih_indexes (np.ndarray): A numpy array with shape (n_dihedrals, 4),
                that indexes the 4-tuples of atoms that will be used for
                the dihedral calculation.
            deg (bool): Whether to return the dihedrals in degree (True) or
                in radian (False). Defaults to False.
            cossin (bool): Whether to return the angles (False) or tuples of their
                cos and sin values (True). Defaults to False.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            check_aas (bool): Whether to check if all aas in `traj.top` are
                recognized. Defaults to True.

        """
        super().__init__(
            traj=traj,
            angle_indexes=dih_indexes,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
            check_aas=check_aas,
            delayed=delayed,
        )

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature. The length
                is determined by the `dih_indexes` and the `cossin` argument
                in the `__init__()` method. If `cossin` is false, then
                `len(describe()) == self.angle_indexes[-1]`, else `len(describe())`
                is twice as long.

        """
        if self.cossin:
            sin_cos = ("DIH: COS(%s -  %s - %s - %s)", "DIH: SIN(%s -  %s - %s - %s)")
            labels = [
                s
                % (
                    _describe_atom(self.top, quad[0]),
                    _describe_atom(self.top, quad[1]),
                    _describe_atom(self.top, quad[2]),
                    _describe_atom(self.top, quad[3]),
                )
                for quad in self.angle_indexes
                for s in sin_cos
            ]
        else:
            labels = [
                "DIH: %s - %s - %s - %s "
                % (
                    _describe_atom(self.top, quad[0]),
                    _describe_atom(self.top, quad[1]),
                    _describe_atom(self.top, quad[2]),
                    _describe_atom(self.top, quad[3]),
                )
                for quad in self.angle_indexes
            ]
        return labels

    @staticmethod
    @dask.delayed
    def dask_transform(
        indexes: np.ndarray,
        periodic: bool,
        deg: bool,
        cossin: bool,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> dask.delayed:
        """The same as `transform()` but without the need to pickle `traj`.

        When dask delayed concurrencies are distributed, required python objects
        are pickled. Thus, every feature needs to have its own pickled traj.
        That defeats the purpose of dask distributed. Thus, this method implements
        the same calculations as `transform` as a more barebones approach.
        It foregoes the checks for periodicity and unit-cell shape and just
        takes xyz, unitcell vectors, and unitcell info. Furthermore, it is a
        staticmethod, so it doesn't require `self` to function. However, it
        needs the indexes in `self.indexes`. That's why the `dask_indices`
        property informs the scheduler to also pickle and pass this object to
        the workers.

        Args:
            indexes (np.ndarray): A numpy array with shape (n, ) giving the
                0-based index of the atoms which positions should be returned.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            deg (bool): Whether to return the result in degree (`deg=True`) or in
                radians (`deg=False`). Defaults to False (radians).
            cossin (bool): If True, each angle will be returned as a pair of
                (sin(x), cos(x)). This is useful, if you calculate the means
                (e.g. TICA/PCA, clustering) in that space. Defaults to False.
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        """
        if periodic:
            assert unitcell_vectors is not None
            # convert to angles
            if unitcell_info is None:
                unitcell_angles = []
                for fr_unitcell_vectors in unitcell_vectors:
                    _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                        fr_unitcell_vectors[0],
                        fr_unitcell_vectors[1],
                        fr_unitcell_vectors[2],
                    )
                    unitcell_angles.append(np.array([alpha, beta, gamma]))
            else:
                unitcell_angles = unitcell_info[:, 3:]

            # check for an orthogonal box
            orthogonal = np.allclose(unitcell_angles, 90)

            out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
            _dihedral_mic(
                xyz,
                indexes,
                unitcell_vectors.transpose(0, 2, 1).copy(),
                out,
                orthogonal,
            )

        else:
            out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
            _dihedral(xyz, indexes, out)

        if cossin:
            out = np.dstack((np.cos(out), np.sin(out)))
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])

        if deg:
            out = np.rad2deg(out)
        return out

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        if xyz is not None:
            periodic = self.periodic
        else:
            periodic = self.periodic and self.traj._have_unitcell
        xyz, unitcell_vectors, unitcell_info = Feature.transform(
            self, xyz, unitcell_vectors, unitcell_info
        )
        if periodic:
            assert unitcell_vectors is not None

            # convert to angles
            if unitcell_info is None:
                unitcell_angles = []
                for fr_unitcell_vectors in unitcell_vectors:
                    _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                        fr_unitcell_vectors[0],
                        fr_unitcell_vectors[1],
                        fr_unitcell_vectors[2],
                    )
                    unitcell_angles.append(np.array([alpha, beta, gamma]))
            else:
                unitcell_angles = unitcell_info[:, 3:]

            # check for an orthogonal box
            orthogonal = np.allclose(unitcell_angles, 90)

            out = np.empty(
                (xyz.shape[0], self.angle_indexes.shape[0]), dtype="float32", order="C"
            )
            _dihedral_mic(
                xyz,
                self.angle_indexes,
                unitcell_vectors.transpose(0, 2, 1).copy(),
                out,
                orthogonal,
            )

        else:
            out = np.empty(
                (xyz.shape[0], self.angle_indexes.shape[0]), dtype="float32", order="C"
            )
            _dihedral(xyz, self.angle_indexes, out)

        if self.cossin:
            out = np.dstack((np.cos(out), np.sin(out)))
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])

        if self.deg:
            out = np.rad2deg(out)
        return out


class DistanceFeature(Feature):
    prefix_label: str = "DIST:"

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        distance_indexes: np.ndarray,
        periodic: bool = True,
        dim: Optional[int] = None,
        check_aas: bool = True,
        delayed: bool = False,
    ) -> None:
        super().__init__(traj, check_aas, periodic=periodic, delayed=delayed)
        self.distance_indexes = np.array(distance_indexes)
        if len(self.distance_indexes) == 0:
            raise ValueError("empty indices")
        if dim is None:
            self._dim = len(distance_indexes)
        else:
            self._dim = dim

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        labels = [
            (
                f"{self.prefix_label} {_describe_atom(self.top, pair[0])} "
                f"{_describe_atom(self.top, pair[1])}"
            )
            for pair in self.distance_indexes
        ]
        return labels

    @property
    def dask_indices(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "distance_indexes"

    @staticmethod
    @dask.delayed
    def dask_transform(
        indexes: np.ndarray,
        periodic: bool,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> dask.delayed:
        """The same as `transform()` but without the need to pickle `traj`.

        When dask delayed concurrencies are distributed, required python objects
        are pickled. Thus, every feature needs to have its own pickled traj.
        That defeats the purpose of dask distributed. Thus, this method implements
        the same calculations as `transform` as a more barebones approach.
        It foregoes the checks for periodicity and unit-cell shape and just
        takes xyz, unitcell vectors, and unitcell info. Furthermore, it is a
        staticmethod, so it doesn't require `self` to function. However, it
        needs the indexes in `self.indexes`. That's why the `dask_indices`
        property informs the scheduler to also pickle and pass this object to
        the workers.

        Args:
            indexes (np.ndarray): A numpy array with shape (n, ) giving the
                0-based index of the atoms which positions should be returned.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        """
        if periodic:
            assert unitcell_vectors is not None
            # check for an orthogonal box
            if unitcell_info is None:
                # convert to angles
                unitcell_angles = []
                for fr_unitcell_vectors in unitcell_vectors:
                    _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                        fr_unitcell_vectors[0],
                        fr_unitcell_vectors[1],
                        fr_unitcell_vectors[2],
                    )
                    unitcell_angles.append(np.array([alpha, beta, gamma]))
            else:
                unitcell_angles = unitcell_info[:, 3:]

            # check for an orthogonal box
            orthogonal = np.allclose(unitcell_angles, 90)

            out = np.empty(
                (
                    xyz.shape[0],
                    indexes.shape[0],
                ),
                dtype="float32",
                order="C",
            )
            _dist_mic(
                xyz,
                indexes.astype("int32"),
                unitcell_vectors.transpose(0, 2, 1).copy(),
                out,
                orthogonal,
            )
        else:
            out = np.empty(
                (
                    xyz.shape[0],
                    indexes.shape[0],
                ),
                dtype="float32",
                order="C",
            )
            _dist(xyz, indexes.astype("int32"), out)
        return out

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        if xyz is not None:
            periodic = self.periodic
        else:
            periodic = self.periodic and self.traj._have_unitcell
        xyz, unitcell_vectors, unitcell_info = super().transform(
            xyz, unitcell_vectors, unitcell_info
        )
        if periodic:
            assert unitcell_info is not None

            # check for an orthogonal box
            if unitcell_info is None:
                # convert to angles
                unitcell_angles = []
                for fr_unitcell_vectors in unitcell_vectors:
                    _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                        fr_unitcell_vectors[0],
                        fr_unitcell_vectors[1],
                        fr_unitcell_vectors[2],
                    )
                    unitcell_angles.append(np.array([alpha, beta, gamma]))
            else:
                unitcell_angles = unitcell_info[:, 3:]
            # check for an orthogonal box
            orthogonal = np.allclose(unitcell_angles, 90)
            out = np.empty(
                (
                    xyz.shape[0],
                    self.distance_indexes.shape[0],
                ),
                dtype="float32",
                order="C",
            )
            _dist_mic(
                xyz,
                np.ascontiguousarray(self.distance_indexes.astype("int32")),
                unitcell_vectors.transpose(0, 2, 1).copy(),
                out,
                orthogonal,
            )
        else:
            out = np.empty(
                (
                    xyz.shape[0],
                    self.distance_indexes.shape[0],
                ),
                dtype="float32",
                order="C",
            )
            _dist(xyz, np.ascontiguousarray(self.distance_indexes.astype("int32")), out)
        return out


class AlignFeature(SelectionFeature):
    prefix_label: str = "aligned ATOM:"

    def __init__(
        self,
        traj: SingleTraj,
        reference: md.Trajectory,
        indexes: np.ndarray,
        atom_indices: Optional[np.ndarray] = None,
        ref_atom_indices: Optional[np.ndarray] = None,
        in_place: bool = False,
        delayed: bool = False,
    ) -> None:
        super(AlignFeature, self).__init__(traj=traj, indexes=indexes, delayed=delayed)
        self.ref = reference
        self.atom_indices = atom_indices
        self.ref_atom_indices = ref_atom_indices
        self.in_place = in_place

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns the aligned xyz coordinates."""
        if not self.in_place:
            traj = self.traj.traj.slice(slice(None), copy=True)
        else:
            traj = self.traj.traj
        traj.xyz = xyz
        aligned = traj.superpose(
            reference=self.ref,
            atom_indices=self.atom_indices,
            ref_atom_indices=self.ref_atom_indices,
        )
        # apply selection
        return super(AlignFeature, self).transform(
            aligned.xyz, unitcell_vectors, unitcell_info
        )


class InverseDistanceFeature(DistanceFeature):
    prefix_label: str = "INVDIST:"

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        distance_indexes: np.ndarray,
        periodic: bool = True,
        delayed: bool = False,
    ) -> None:
        DistanceFeature.__init__(
            self, traj, distance_indexes, periodic=periodic, delayed=delayed
        )

    @property
    def dask_indices(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "distance_indexes"

    @staticmethod
    @dask.delayed
    def dask_transform(
        indexes: np.ndarray,
        periodic: bool,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> dask.delayed:
        """The same as `transform()` but without the need to pickle `traj`.

        When dask delayed concurrencies are distributed, required python objects
        are pickled. Thus, every feature needs to have its own pickled traj.
        That defeats the purpose of dask distributed. Thus, this method implements
        the same calculations as `transform` as a more barebones approach.
        It foregoes the checks for periodicity and unit-cell shape and just
        takes xyz, unitcell vectors, and unitcell info. Furthermore, it is a
        staticmethod, so it doesn't require `self` to function. However, it
        needs the indexes in `self.indexes`. That's why the `dask_indices`
        property informs the scheduler to also pickle and pass this object to
        the workers.

        Args:
            indexes (np.ndarray): A numpy array with shape (n, ) giving the
                0-based index of the atoms which positions should be returned.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        """
        if periodic:
            assert unitcell_vectors is not None
            # check for an orthogonal box
            if unitcell_info is None:
                # convert to angles
                unitcell_angles = []
                for fr_unitcell_vectors in unitcell_vectors:
                    _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                        fr_unitcell_vectors[0],
                        fr_unitcell_vectors[1],
                        fr_unitcell_vectors[2],
                    )
                    unitcell_angles.append(np.array([alpha, beta, gamma]))
            else:
                unitcell_angles = unitcell_info[:, 3:]
            # check for an orthogonal box
            orthogonal = np.allclose(unitcell_angles, 90)

            out = np.empty(
                (
                    xyz.shape[0],
                    indexes.shape[0],
                ),
                dtype="float32",
                order="C",
            )
            _dist_mic(
                xyz,
                indexes.astype("int32"),
                unitcell_vectors.transpose(0, 2, 1).copy(),
                out,
                orthogonal,
            )
        else:
            out = np.empty(
                (
                    xyz.shape[0],
                    indexes.shape[0],
                ),
                dtype="float32",
                order="C",
            )
            _dist(xyz, indexes.astype("int32"), out)
        return 1 / out

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        return 1.0 / super().transform(xyz, unitcell_vectors, unitcell_info)


class ContactFeature(DistanceFeature):
    """Defines certain distances as contacts and returns a binary (0, 1) result.

    Instead of returning the binary result can also count contacts with the
    argument `count_contacts=True` provided at instantiation. In that case,
    every frame returns an integer number.

    """

    prefix_label: str = "CONTACT:"
    _nonstandard_transform_args: list[str] = ["threshold", "count_contacts"]

    def __init__(
        self,
        traj: SingleTraj,
        distance_indexes: np.ndarray,
        threshold: float = 5.0,
        periodic: bool = True,
        count_contacts: bool = False,
        delayed: bool = False,
    ) -> None:
        """Instantiate the contact feature.

        A regular contact feature yields a np.ndarray with zeros and ones.
        The zeros are no contact. The ones are contact.

        Args:
            traj (SingleTraj): An instance of `SingleTraj`.
            distance_indexes (np.ndarray): An np.ndarray with shape (n_dists, 2),
                where distance_indexes[:, 0] indexes the first atoms of the distance
                measurement, and distance_indexes[:, 1] indexes the second atoms of the
                distance measurement.
            threshold (float): The threshold in nm, under which a distance is
                considered to be a contact. Defaults to 5.0 nm.
            periodic (bool): Whether to use the minimum image convention when
                calculating distances. Defaults to True.
            count_contacts (bool): When True, return an integer of the number of
                contacts instead of returning the array of regular contacts.

        """
        super(ContactFeature, self).__init__(
            traj, distance_indexes, periodic=periodic, delayed=delayed
        )
        if count_contacts:
            self.prefix_label: str = "counted " + self.prefix_label
        self.threshold = threshold
        self.count_contacts = count_contacts
        if count_contacts:
            self.dimension = 1
        else:
            self.dimension = len(self.distance_indexes)

    @property
    def dask_indices(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "distance_indexes"

    @staticmethod
    @dask.delayed
    def dask_transform(
        indexes: np.ndarray,
        periodic: bool,
        threshold: float,
        count_contacts: bool,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> dask.delayed:
        """The same as `transform()` but without the need to pickle `traj`.

        When dask delayed concurrencies are distributed, required python objects
        are pickled. Thus, every feature needs to have its own pickled traj.
        That defeats the purpose of dask distributed. Thus, this method implements
        the same calculations as `transform` as a more barebones approach.
        It foregoes the checks for periodicity and unit-cell shape and just
        takes xyz, unitcell vectors, and unitcell info. Furthermore, it is a
        staticmethod, so it doesn't require `self` to function. However, it
        needs the indexes in `self.indexes`. That's why the `dask_indices`
        property informs the scheduler to also pickle and pass this object to
        the workers.

        Args:
            indexes (np.ndarray): A numpy array with shape (n, ) giving the
                0-based index of the atoms which positions should be returned.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            threshold (float): The threshold in nm, under which a distance is
                considered to be a contact. Defaults to 5.0 nm.
            count_contacts (bool): When True, return an integer of the number of contacts
                instead of returning the array of regular contacts.
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        """
        if periodic:
            assert unitcell_vectors is not None
            # check for an orthogonal box
            if unitcell_info is None:
                # convert to angles
                unitcell_angles = []
                for fr_unitcell_vectors in unitcell_vectors:
                    _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                        fr_unitcell_vectors[0],
                        fr_unitcell_vectors[1],
                        fr_unitcell_vectors[2],
                    )
                    unitcell_angles.append(np.array([alpha, beta, gamma]))
            else:
                unitcell_angles = unitcell_info[:, 3:]
            # check for an orthogonal box
            orthogonal = np.allclose(unitcell_angles, 90)

            out = np.empty(
                (
                    xyz.shape[0],
                    indexes.shape[0],
                ),
                dtype="float32",
                order="C",
            )
            _dist_mic(
                xyz,
                indexes.astype("int32"),
                unitcell_vectors.transpose(0, 2, 1).copy(),
                out,
                orthogonal,
            )
        else:
            out = np.empty(
                (
                    xyz.shape[0],
                    indexes.shape[0],
                ),
                dtype="float32",
                order="C",
            )
            _dist(xyz, indexes.astype("int32"), out)
        res = np.zeros((len(out), indexes.shape[0]), dtype=np.float32)
        I = np.argwhere(out <= threshold)  # noqa: E741
        res[I[:, 0], I[:, 1]] = 1.0
        if count_contacts:
            return res.sum(axis=1, keepdims=True)
        else:
            return res

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        dists = super(ContactFeature, self).transform(
            xyz, unitcell_vectors, unitcell_info
        )
        res = np.zeros(
            (len(self.traj), self.distance_indexes.shape[0]), dtype=np.float32
        )
        I = np.argwhere(dists <= self.threshold)  # noqa: E741
        res[I[:, 0], I[:, 1]] = 1.0
        if self.count_contacts:
            return res.sum(axis=1, keepdims=True)
        else:
            return res


class BackboneTorsionFeature(DihedralFeature):
    def __init__(
        self,
        traj: SingleTraj,
        selstr: Optional[str] = None,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        delayed: bool = False,
    ) -> None:
        self.traj = traj
        indices = self.traj.indices_phi
        self.selstr = selstr

        if not selstr:
            self._phi_inds = indices
        else:
            self._phi_inds = indices[
                np.in1d(indices[:, 1], self.traj.top.select(selstr), assume_unique=True)
            ]

        indices = self.traj.indices_psi
        if not selstr:
            self._psi_inds = indices
        else:
            self._psi_inds = indices[
                np.in1d(indices[:, 1], self.traj.top.select(selstr), assume_unique=True)
            ]

        # alternate phi, psi pairs (phi_1, psi_1, ..., phi_n, psi_n)
        dih_indexes = np.array(
            list(phi_psi for phi_psi in zip(self._phi_inds, self._psi_inds))
        ).reshape(-1, 4)

        super(BackboneTorsionFeature, self).__init__(
            self.traj,
            dih_indexes,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
            delayed=delayed,
        )

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature. The length
                is determined by the `dih_indexes` and the `cossin` argument
                in the `__init__()` method. If `cossin` is false, then
                `len(describe()) == self.angle_indexes[-1]`, else `len(describe())`
                is twice as long.

        """
        top = self.traj.top
        getlbl = lambda at: "%i %s %i" % (
            at.residue.chain.index,
            at.residue.name,
            at.residue.resSeq,
        )

        if self.cossin:
            sin_cos = ("COS(PHI %s)", "SIN(PHI %s)")
            labels_phi = [
                (
                    sin_cos[0] % getlbl(top.atom(ires[1])),
                    sin_cos[1] % getlbl(top.atom(ires[1])),
                )
                for ires in self._phi_inds
            ]
            sin_cos = ("COS(PSI %s)", "SIN(PSI %s)")
            labels_psi = [
                (
                    sin_cos[0] % getlbl(top.atom(ires[1])),
                    sin_cos[1] % getlbl(top.atom(ires[1])),
                )
                for ires in self._psi_inds
            ]
            # produce the same ordering as the given indices (phi_1, psi_1, ..., phi_n, psi_n)
            # or (cos(phi_1), sin(phi_1), cos(psi_1), sin(psi_1), ..., cos(phi_n), sin(phi_n), cos(psi_n), sin(psi_n))
            res = list(
                itertools.chain.from_iterable(
                    itertools.chain.from_iterable(zip(labels_phi, labels_psi))
                )
            )
        else:
            labels_phi = [
                "PHI %s" % getlbl(top.atom(ires[1])) for ires in self._phi_inds
            ]
            labels_psi = [
                "PSI %s" % getlbl(top.atom(ires[1])) for ires in self._psi_inds
            ]
            res = list(itertools.chain.from_iterable(zip(labels_phi, labels_psi)))
        return res


class ResidueMinDistanceFeature(DistanceFeature):
    _nonstandard_transform_args: list[str] = [
        "threshold",
        "count_contacts",
        "scheme",
        "top",
    ]

    def __init__(
        self,
        traj: SingleTraj,
        contacts: np.ndarray,
        scheme: Literal["ca", "closest", "closest-heavy"],
        ignore_nonprotein: bool,
        threshold: float,
        periodic: bool,
        count_contacts: bool = False,
        delayed: bool = False,
    ) -> None:
        if count_contacts and threshold is None:
            raise ValueError(
                "Cannot count contacts when no contact threshold is supplied."
            )

        self.contacts = contacts
        self.scheme = scheme
        self.threshold = threshold
        self.prefix_label: str = "RES_DIST (%s)" % scheme
        self.ignore_nonprotein = ignore_nonprotein

        if count_contacts:
            self.prefix_label: str = "counted " + self.prefix_label
        self.count_contacts = count_contacts

        dummy_traj = md.Trajectory(np.zeros((traj.top.n_atoms, 3)), traj.top)
        dummy_dist, dummy_pairs = md.compute_contacts(
            dummy_traj,
            contacts=contacts,
            scheme=scheme,
            periodic=periodic,
            ignore_nonprotein=ignore_nonprotein,
        )

        dim = 1 if count_contacts else dummy_dist.shape[1]
        super(ResidueMinDistanceFeature, self).__init__(
            distance_indexes=dummy_pairs,
            traj=traj,
            periodic=periodic,
            dim=dim,
            delayed=delayed,
        )

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        labels = []
        for a, b in self.distance_indexes:
            labels.append(
                f"{self.prefix_label} {self.traj.top.residue(a)} - {self.traj.top.residue(b)}"
            )
        return labels

    @property
    def dask_indices(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "contacts"

    @staticmethod
    @dask.delayed
    def dask_transform(
        indexes: np.ndarray,
        top: md.Topology,
        scheme: Literal["ca", "closest", "closest-heavy"],
        periodic: bool,
        threshold: float,
        count_contacts: bool,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> dask.delayed:
        """The same as `transform()` but without the need to pickle `traj`.

        When dask delayed concurrencies are distributed, required python objects
        are pickled. Thus, every feature needs to have its own pickled traj.
        That defeats the purpose of dask distributed. Thus, this method implements
        the same calculations as `transform` as a more barebones approach.
        It foregoes the checks for periodicity and unit-cell shape and just
        takes xyz, unitcell vectors, and unitcell info. Furthermore, it is a
        staticmethod, so it doesn't require `self` to function. However, it
        needs the indexes in `self.indexes`. That's why the `dask_indices`
        property informs the scheduler to also pickle and pass this object to
        the workers.

        Args:
            indexes (np.ndarray): For this special feature, the indexes argument
                in the @staticmethod dask_transform is `self.contacts`.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            threshold (float): The threshold in nm, under which a distance is
                considered to be a contact. Defaults to 5.0 nm.
            count_contacts (bool): When True, return an integer of the number of contacts
                instead of returning the array of regular contacts.
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        """
        # create a dummy traj, with the appropriate topology
        traj = md.Trajectory(
            xyz=xyz,
            topology=top,
            unitcell_lengths=unitcell_info[:, :3],
            unitcell_angles=unitcell_info[:, 3:],
        )

        # We let mdtraj compute the contacts with the input scheme
        D = md.compute_contacts(
            traj,
            contacts=indexes,
            scheme=scheme,
            periodic=periodic,
        )[0]

        res = np.zeros_like(D)
        # Do we want binary?
        if threshold is not None:
            I = np.argwhere(D <= threshold)
            res[I[:, 0], I[:, 1]] = 1.0
        else:
            res = D

        if count_contacts and threshold is not None:
            return res.sum(axis=1, keepdims=True)
        else:
            return res

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        (
            xyz,
            unitcell_vectors,
            unitcell_info,
        ) = Feature.transform(self, xyz, unitcell_vectors, unitcell_info)

        # create a dummy traj, with the appropriate topology
        traj = md.Trajectory(
            xyz=xyz,
            topology=self.traj.top,
            unitcell_lengths=unitcell_info[:, :3],
            unitcell_angles=unitcell_info[:, 3:],
        )

        # We let mdtraj compute the contacts with the input scheme
        D = md.compute_contacts(
            traj,
            contacts=self.contacts,
            scheme=self.scheme,
            periodic=self.periodic,
        )[0]

        res = np.zeros_like(D)
        # Do we want binary?
        if self.threshold is not None:
            I = np.argwhere(D <= self.threshold)
            res[I[:, 0], I[:, 1]] = 1.0
        else:
            res = D

        if self.count_contacts and self.threshold is not None:
            return res.sum(axis=1, keepdims=True)
        else:
            return res


class GroupCOMFeature(Feature):
    """Cartesian coordinates of the center-of-mass (COM) of atom groups.

    Groups can be defined as sequences of sequences of int. So a list of list of int
    can be used to define groups of various sizes. The resulting array will have
    the shape of (n_frames, n_groups ** 2). The xyz coordinates are flattended,
    so the array can be rebuilt with `np.dstack()`

    Examples:
        >>> import encodermap as em
        >>> import numpy as np
        >>> traj = em.SingleTraj.from_pdb_id("1YUG")
        >>> f = em.features.GroupCOMFeature(
        ...     traj=traj,
        ...     group_definitions=[
        ...         [0, 1, 2],
        ...         [3, 4, 5, 6, 7],
        ...         [8, 9, 10],
        ...     ]
        ... )
        >>> a = f.transform()
        >>> a.shape  # this array is flattened along the feature axis
        (15, 9)
        >>> a = np.dstack([
        ...     a[..., ::3],
        ...     a[..., 1::3],
        ...     a[..., 2::3],
        ... ])
        >>> a.shape  # now the z, coordinate of the 2nd center of mass is a[:, 1, -1]
        (15, 3, 3)

    Note:
        Centering (`ref_geom`) and imaging (`image_molecules=True`) can be time-
        consuming. Consider doing this to your trajectory files prior to featurization.

    """

    _nonstandard_transform_args: list[str] = [
        "top",
        "ref_geom",
        "image_molecules",
        "masses_in_groups",
    ]

    def __init__(
        self,
        traj: SingleTraj,
        group_definitions: Sequence[Sequence[int]],
        ref_geom: Optional[md.Trajectory] = None,
        image_molecules: bool = False,
        mass_weighted: bool = True,
        delayed: bool = False,
    ) -> None:
        """Instantiate the GroupCOMFeature.

        Args:
            traj (SingleTraj): An instance of `SingleTraj`.
            group_definitions (Sequence[Sequence[int]]): A sequence of sequences
                of int defining the groups of which the COM should be calculated.
                See the example for how to use this argument.
            ref_geom (Optional[md.Trajectory]): The coordinates can be centered
                to a reference geometry before computing the COM. Defaults to None.
            image_molecules (bool): The method traj.image_molecules will be
                called before computing averages. The method tries to correct
                for molecules broken across periodic boundary conditions,
                but can be time-consuming. See
                http://mdtraj.org/latest/api/generated/mdtraj.Trajectory.html#mdtraj.Trajectory.image_molecules
                for more details
            mass_weighted (bool): Whether the COM should be calculated mass-weighted.

        """
        self._raise_on_unitcell = False
        if not (ref_geom is None or isinstance(ref_geom, md.Trajectory)):
            raise ValueError(
                f"argument ref_geom has to be either None or and "
                f"mdtraj.Trajectory, got instead {type(ref_geom)}"
            )

        self.ref_geom = ref_geom
        self.traj = traj
        self.top = traj.top
        self.image_molecules = image_molecules
        self.group_definitions = [np.asarray(gf) for gf in group_definitions]
        self.atom_masses = np.array([aa.element.mass for aa in self.traj.top.atoms])

        if mass_weighted:
            self.masses_in_groups = [
                self.atom_masses[aa_in_rr] for aa_in_rr in self.group_definitions
            ]
        else:
            self.masses_in_groups = [
                np.ones_like(aa_in_rr) for aa_in_rr in self.group_definitions
            ]

        self.delayed = delayed

        # Prepare and store the description
        self._describe = []
        for group in self.group_definitions:
            for coor in "xyz":
                self._describe.append(
                    f"COM-{coor} of atom group [{group[:3]}..{group[-3:]}]"
                )
        self.dimension = 3 * len(self.group_definitions)

    @property
    def dask_indices(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "group_definitions"

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        return self._describe

    @staticmethod
    @dask.delayed
    def dask_transform(
        indexes: list[list[int]],
        top: md.Topology,
        ref_geom: Union[md.Trajectory, None],
        image_molecules: bool,
        masses_in_groups: list[float],
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> dask.delayed:
        """The same as `transform()` but without the need to pickle `traj`.

        When dask delayed concurrencies are distributed, required python objects
        are pickled. Thus, every feature needs to have its own pickled traj.
        That defeats the purpose of dask distributed. Thus, this method implements
        the same calculations as `transform` as a more barebones approach.
        It foregoes the checks for periodicity and unit-cell shape and just
        takes xyz, unitcell vectors, and unitcell info. Furthermore, it is a
        staticmethod, so it doesn't require `self` to function. However, it
        needs the indexes in `self.indexes`. That's why the `dask_indices`
        property informs the scheduler to also pickle and pass this object to
        the workers.

        Args:
            indexes (np.ndarray): For this special feature, the indexes argument
                in the @staticmethod dask_transform is `self.group_definitions`.
            periodic (bool): Whether to observe the minimum image convention
                and respect proteins breaking over the periodic boundary
                condition as a whole (True). In this case, the trajectory container
                in `traj` needs to have unitcell information. Defaults to True.
            threshold (float): The threshold in nm, under which a distance is
                considered to be a contact. Defaults to 5.0 nm.
            count_contacts (bool): When True, return an integer of the number of contacts
                instead of returning the array of regular contacts.
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        """
        # create a dummy traj, with the appropriate topology
        traj = md.Trajectory(
            xyz=xyz,
            topology=top,
            unitcell_lengths=unitcell_info[:, :3],
            unitcell_angles=unitcell_info[:, 3:],
        )
        COM_xyz = []
        if ref_geom is not None:
            traj = traj.superpose(ref_geom)
        if image_molecules:
            traj = traj.image_molecules()
        for aas, mms in zip(indexes, masses_in_groups):
            COM_xyz.append(
                np.average(
                    traj.xyz[
                        :,
                        aas,
                    ],
                    axis=1,
                    weights=mms,
                )
            )
        return np.hstack(COM_xyz)

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        (
            xyz,
            unitcell_vectors,
            unitcell_info,
        ) = Feature.transform(self, xyz, unitcell_vectors, unitcell_info)
        # create a dummy traj, with the appropriate topology
        traj = md.Trajectory(
            xyz=xyz,
            topology=self.traj.top,
            unitcell_lengths=(
                unitcell_info[:, :3] if unitcell_info is not None else None
            ),
            unitcell_angles=unitcell_info[:, 3:] if unitcell_info is not None else None,
        )
        COM_xyz = []
        if self.ref_geom is not None:
            traj = traj.superpose(self.ref_geom)
        if self.image_molecules:
            traj = traj.image_molecules()
        for aas, mms in zip(self.group_definitions, self.masses_in_groups):
            COM_xyz.append(
                np.average(
                    traj.xyz[
                        :,
                        aas,
                    ],
                    axis=1,
                    weights=mms,
                )
            )
        return np.hstack(COM_xyz)


class ResidueCOMFeature(GroupCOMFeature):
    def __init__(
        self,
        traj: SingleTraj,
        residue_indices: Sequence[int],
        residue_atoms: np.ndarray,
        scheme: Literal["all", "backbone", "sidechain"] = "all",
        ref_geom: Optional[md.Trajectory] = None,
        image_molecules: bool = False,
        mass_weighted: bool = True,
        delayed: bool = False,
    ) -> None:
        """Instantiate the ResidueCOMFeature.

        Args:
            residue_indices (Sequence[int]): The residue indices for which the
                COM will be computed. These are always zero-indexed that are not
                necessarily the residue sequence record of the topology (resSeq).
                resSeq indices start at least at 1 but can depend on the topology.
                Furthermore, resSeq numbers can be duplicated across chains; residue
                indices are always unique.


        """
        super(ResidueCOMFeature, self).__init__(
            traj,
            residue_atoms,
            mass_weighted=mass_weighted,
            ref_geom=ref_geom,
            image_molecules=image_molecules,
            delayed=delayed,
        )

        self.residue_indices = residue_indices
        self.scheme = scheme

        self._describe = []
        for ri in self.residue_indices:
            for coor in "xyz":
                self._describe.append(
                    f"{self.traj.top.residue(ri)} COM-{coor} ({self.scheme})"
                )


class SideChainTorsions(DihedralFeature):
    options = ("chi1", "chi2", "chi3", "chi4", "chi5")

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        selstr: Optional[str] = None,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        which: Union[
            Literal["all"], Sequence[Literal["chi1", "chi2", "chi3", "chi4", "chi5"]]
        ] = "all",
        delayed: bool = False,
    ) -> None:
        if not isinstance(which, (tuple, list)):
            which = [which]
        if not set(which).issubset(set(self.options) | {"all"}):
            raise ValueError(
                'Argument "which" should only contain one of {}, but was {}'.format(
                    ["all"] + list(self.options), which
                )
            )
        if "all" in which:
            which = self.options

        # get all dihedral index pairs
        indices_dict = {k: getattr(traj, "indices_%s" % k) for k in which}
        if selstr:
            selection = traj.top.select(selstr)
            truncated_indices_dict = {}
            for k, inds in indices_dict.items():
                mask = np.in1d(inds[:, 1], selection, assume_unique=True)
                truncated_indices_dict[k] = inds[mask]
            indices_dict = truncated_indices_dict

        valid = {k: indices_dict[k] for k in indices_dict if indices_dict[k].size > 0}
        if not valid:
            raise ValueError(
                "Could not determine any side chain dihedrals for your topology!"
            )
        self._prefix_label_lengths = np.array(
            [len(indices_dict[k]) if k in which else 0 for k in self.options]
        )
        indices = np.vstack(list(valid.values()))

        super(SideChainTorsions, self).__init__(
            traj=traj,
            dih_indexes=indices,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
            delayed=delayed,
        )

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature. The length
                is determined by the `dih_indexes` and the `cossin` argument
                in the `__init__()` method. If `cossin` is false, then
                `len(describe()) == self.angle_indexes[-1]`, else `len(describe())`
                is twice as long.

        """
        getlbl = lambda at: "%i %s %i" % (
            at.residue.chain.index,
            at.residue.name,
            at.residue.resSeq,
        )
        prefixes = []
        for lengths, label in zip(self._prefix_label_lengths, self.options):
            if self.cossin:
                lengths *= 2
            prefixes.extend([label.upper()] * lengths)

        if self.cossin:
            cossin = ("COS({dih} {res})", "SIN({dih} {res})")
            labels = [
                s.format(
                    dih=prefixes[j + len(cossin) * i],
                    res=getlbl(self.top.atom(ires[1])),
                )
                for i, ires in enumerate(self.angle_indexes)
                for j, s in enumerate(cossin)
            ]
        else:
            labels = [
                "{dih} {res}".format(
                    dih=prefixes[i], res=getlbl(self.top.atom(ires[1]))
                )
                for i, ires in enumerate(self.angle_indexes)
            ]

        return labels


class MinRmsdFeature(Feature):
    _nonstandard_transform_args: list[str] = [
        "top",
        "ref",
    ]

    def __init__(
        self,
        traj: SingleTraj,
        ref: Union[md.Trajectory, SingleTraj],
        ref_frame: int = 0,
        atom_indices: Optional[np.ndarray] = None,
        precentered: bool = False,
        delayed: bool = False,
    ) -> None:
        # Encodermap imports
        from encodermap.trajinfo.info_single import SingleTraj

        self._raise_on_unitcell = False
        self.traj = traj
        self.top = self.traj.top
        assert isinstance(
            ref_frame, int
        ), f"ref_frame has to be of type integer, and not {type(ref_frame)}"

        if isinstance(ref, (md.Trajectory, SingleTraj)):
            self.name = f"MinRmsdFeature_with_{ref.n_atoms}_atoms_in_reference"
        else:
            raise TypeError(
                f"input reference has to be either `encodermap.SingleTraj` or "
                f"a mdtraj.Trajectory object, and not of {ref}"
            )

        self.ref = ref
        self.ref_frame = ref_frame
        self.atom_indices = atom_indices
        self.precentered = precentered
        self.dimension = 1
        self.delayed = delayed

    @property
    def dask_indices(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "atom_indices"

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        label = "minrmsd to frame %u of %s" % (self.ref_frame, self.name)
        if self.precentered:
            label += ", precentered=True"
        if self.atom_indices is not None:
            label += ", subset of atoms  "
        return [label]

    @property
    def dask_indices(self):
        """str: The name of the delayed transformation to carry out with this feature."""
        return "atom_indices"

    @staticmethod
    @dask.delayed
    def dask_transform(
        indexes: np.ndarray,
        top: md.Topology,
        ref: md.Trajectory,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        # create a dummy traj, with the appropriate topology
        traj = md.Trajectory(
            xyz=xyz,
            topology=top,
            unitcell_lengths=unitcell_info[:, :3],
            unitcell_angles=unitcell_info[:, 3:],
        )

        return np.array(md.rmsd(traj, ref, atom_indices=indexes), ndmin=2).T

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_info: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_info` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided, the coordinates of `self.traj`
                will be used. Otherwise, the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_info (Optional[np.ndarray]): Basically identical to
                `unitcell_vectors`. A numpy array of shape (n_frames, 6), where
                the first three columns are the unitcell_lengths in nanometer.
                The other three columns are the unitcell_angles in degrees.

        Returns:
            np.ndarray: The result of the computation with shape (n_frames, n_indexes).

        """
        (
            xyz,
            unitcell_vectors,
            unitcell_info,
        ) = Feature.transform(self, xyz, unitcell_vectors, unitcell_info)

        # create a dummy traj, with the appropriate topology
        traj = md.Trajectory(
            xyz=xyz,
            topology=self.traj.top,
            unitcell_lengths=unitcell_info[:, :3],
            unitcell_angles=unitcell_info[:, 3:],
        )

        return np.array(
            md.rmsd(traj, self.ref, atom_indices=self.atom_indices), ndmin=2
        ).T


################################################################################
# EncoderMap features
################################################################################


class CentralDihedrals(DihedralFeature):
    """Feature that collects all dihedrals in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.

    """

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        selstr: Optional[str] = None,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        omega: bool = True,
        generic_labels: bool = False,
        check_aas: bool = True,
        delayed: bool = False,
    ):
        """Instantiate this feature class.

        Args:
            traj (em.SingleTraj): A topology to build features from.
            selstr (Optional[str]): A string, that limits the selection of dihedral angles.
                Only dihedral angles which atoms are represented by the `selstr` argument
                are considered. This selection string follows MDTraj's atom selection
                language: https://mdtraj.org/1.9.3/atom_selection.html. Can also
                be None, in which case all backbone dihedrals (also omega) are
                considered. Defaults to None.
            deg (bool): Whether to return the result in degree (`deg=True`) or in
                radians (`deg=False`). Defaults to False (radians).
            cossin (bool): If True, each angle will be returned as a pair of
                (sin(x), cos(x)). This is useful, if you calculate the means
                (e.g. TICA/PCA, clustering) in that space. Defaults to False.
            periodic (bool): Whether to recognize periodic boundary conditions and
                work under the minimum image convention. Defaults to True.

        """
        self.traj = traj
        self.selstr = selstr
        self.omega = omega

        indices = self.traj.indices_psi
        if not selstr:
            self._psi_inds = indices
        else:
            self._psi_inds = indices[
                np.in1d(indices[:, 1], self.top.select(selstr), assume_unique=True)
            ]

        self.omega = omega
        if self.omega:
            indices = self.traj.indices_omega
            if not selstr:
                self._omega_inds = indices
            else:
                self._omega_inds = indices[
                    np.in1d(indices[:, 1], self.top.select(selstr), assume_unique=True)
                ]

        indices = self.traj.indices_phi
        if not selstr:
            self._phi_inds = indices
        else:
            self._phi_inds = indices[
                np.in1d(indices[:, 1], self.top.select(selstr), assume_unique=True)
            ]

        if self.omega:
            zipped = list(zip(self._psi_inds, self._omega_inds, self._phi_inds))
        else:
            zipped = list(zip(self._psi_inds, self._phi_inds))

        # alternate phi, psi , omega pairs (phi_1, psi_1, omega_1..., phi_n, psi_n, omega_n)
        dih_indexes = np.array(zipped).reshape(-1, 4)

        # set generic_labels for xarray
        if generic_labels:
            self.describe = self.generic_describe

        super(CentralDihedrals, self).__init__(
            self.traj,
            dih_indexes,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
            check_aas=check_aas,
            delayed=delayed,
        )

    @property
    def name(self) -> str:
        """str: The name of the class: "CentralDihedrals"."""
        return "CentralDihedrals"

    @property
    def indexes(self) -> np.ndarray:
        """np.ndarray: A (n_angles, 4) shaped numpy array giving the atom indices
        of the dihedral angles to be calculated."""
        return self.angle_indexes.astype("int32")

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        if hasattr(self.traj, "clustal_w"):
            clustal_w = np.array([*self.traj.clustal_w])
            count = len(np.where(clustal_w != "-")[0])
            assert count == self.traj.n_residues, (
                f"Provided clustal W alignment {self.traj.clustal_w} does not "
                f"contain as many residues as traj {self.traj.n_residues}. Can not "
                f"use this alignment."
            )
            _psi_inds = (np.arange(len(clustal_w)) + 1)[clustal_w != "-"][
                :-1
            ]  # last residue can't have psi
            _phi_inds = (np.arange(len(clustal_w)) + 1)[clustal_w != "-"][
                1:
            ]  # first residue can't have phi
            if self.omega:
                _omega_inds = (np.arange(len(clustal_w)) + 1)[clustal_w != "-"][
                    :-1
                ]  # last residue can't have omega
            assert len(_psi_inds) == len(self._psi_inds)
            assert len(_phi_inds) == len(self._phi_inds)
            if self.omega:
                assert len(_omega_inds) == len(self._omega_inds)
        else:
            _psi_inds = np.arange(len(self._psi_inds)) + 1
            _phi_inds = np.arange(len(self._phi_inds)) + 1
            if self.omega:
                _omega_inds = np.arange(len(self._omega_inds)) + 1

        if self.cossin:
            sin_cos = ("COS(PSI %s)", "SIN(PSI %s)")
            labels_psi = [
                (
                    sin_cos[0] % i,
                    sin_cos[1] % i,
                )
                for i in _psi_inds
            ]
            if self.omega:
                sin_cos = ("COS(OMEGA %s)", "SIN(OMEGA %s)")
                labels_omega = [
                    (
                        sin_cos[0] % i,
                        sin_cos[1] % i,
                    )
                    for i in _omega_inds
                ]
            sin_cos = ("COS(PHI %s)", "SIN(PHI %s)")
            labels_phi = [
                (
                    sin_cos[0] % i,
                    sin_cos[1] % i,
                )
                for i in _phi_inds
            ]
            # produce the same ordering as the given indices (phi_1, psi_1, ..., phi_n, psi_n)
            # or (cos(phi_1), sin(phi_1), cos(psi_1), sin(psi_1), ..., cos(phi_n), sin(phi_n), cos(psi_n), sin(psi_n))
            if self.omega:
                zipped = zip(labels_psi, labels_omega, labels_phi)
            else:
                zipped = zip(labels_psi, labels_phi)

            res = list(
                itertools.chain.from_iterable(itertools.chain.from_iterable(zipped))
            )
        else:
            labels_psi = [f"CENTERDIH PSI    {i}" for i in _psi_inds]
            if self.omega:
                labels_omega = [f"CENTERDIH OMEGA  {i}" for i in _omega_inds]
            labels_phi = [f"CENTERDIH PHI    {i}" for i in _phi_inds]
            if self.omega:
                zipped = zip(labels_psi, labels_omega, labels_phi)
            else:
                zipped = zip(labels_psi, labels_phi)
            res = list(itertools.chain.from_iterable(zipped))

        return res

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature. The length
                is determined by the `dih_indexes` and the `cossin` argument
                in the `__init__()` method. If `cossin` is false, then
                `len(describe()) == self.angle_indexes[-1]`, else `len(describe())`
                is twice as long.

        """
        top = self.top
        getlbl = (
            lambda at: f"RESID  {at.residue.name}:{at.residue.resSeq:>4} CHAIN {at.residue.chain.index}"
        )

        if self.cossin:
            sin_cos = ("COS(PSI %s)", "SIN(PSI %s)")
            labels_psi = [
                (
                    sin_cos[0] % getlbl(top.atom(ires[1])),
                    sin_cos[1] % getlbl(top.atom(ires[1])),
                )
                for ires in self._psi_inds
            ]
            if self.omega:
                sin_cos = ("COS(OMEGA %s)", "SIN(OMEGA %s)")
                labels_omega = [
                    (
                        sin_cos[0] % getlbl(top.atom(ires[1])),
                        sin_cos[1] % getlbl(top.atom(ires[1])),
                    )
                    for ires in self._omega_inds
                ]
            sin_cos = ("COS(PHI %s)", "SIN(PHI %s)")
            labels_phi = [
                (
                    sin_cos[0] % getlbl(top.atom(ires[1])),
                    sin_cos[1] % getlbl(top.atom(ires[1])),
                )
                for ires in self._phi_inds
            ]
            # produce the same ordering as the given indices (phi_1, psi_1, ..., phi_n, psi_n)
            # or (cos(phi_1), sin(phi_1), cos(psi_1), sin(psi_1), ..., cos(phi_n), sin(phi_n), cos(psi_n), sin(psi_n))
            if self.omega:
                zipped = zip(labels_psi, labels_omega, labels_phi)
            else:
                zip(labels_psi, labels_phi)

            res = list(
                itertools.chain.from_iterable(itertools.chain.from_iterable(zipped))
            )
        else:
            labels_psi = [
                f"CENTERDIH PSI   " + getlbl(top.atom(ires[1]))
                for ires in self._psi_inds
            ]
            if self.omega:
                labels_omega = [
                    "CENTERDIH OMEGA " + getlbl(top.atom(ires[1]))
                    for ires in self._omega_inds
                ]
            labels_phi = [
                "CENTERDIH PHI   " + getlbl(top.atom(ires[1]))
                for ires in self._phi_inds
            ]
            if self.omega:
                zipped = zip(labels_psi, labels_omega, labels_phi)
            else:
                zipped = zip(labels_psi, labels_phi)
            res = list(itertools.chain.from_iterable(zipped))
        return res


class SideChainDihedrals(DihedralFeature):
    """Feature that collects all dihedrals in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        options (list[str]): A list of possible sidechain angles ['chi1' to 'chi5'].

    """

    options: list[str] = ["chi1", "chi2", "chi3", "chi4", "chi5"]

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        selstr: Optional[str] = None,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        generic_labels: bool = False,
        check_aas: bool = True,
        delayed: bool = False,
    ) -> None:
        which = self.options
        # get all dihedral index pairs
        indices_dict = {k: getattr(traj, f"indices_{k}") for k in which}
        if selstr:
            selection = traj.top.select(selstr)
            truncated_indices_dict = {}
            for k, inds in indices_dict.items():
                mask = np.in1d(inds[:, 1], selection, assume_unique=True)
                truncated_indices_dict[k] = inds[mask]
            indices_dict = truncated_indices_dict

        valid = {k: indices_dict[k] for k in indices_dict if indices_dict[k].size > 0}
        if not valid:
            # Third Party Imports
            import mdtraj

            try:
                mdtraj.compute_chi1(traj)
            except Exception as e:
                raise ValueError(
                    "Could not determine any side chain dihedrals for your topology! "
                    "This is an error inside MDTraj. It errors with this message: "
                    f"{e}. You can try to provide a custom_topology "
                    f"for this protein to supersede MDTraj's sidechain recognition "
                    f"algorithm."
                ) from e
            else:
                raise ValueError(f"No sidechain dihedrals for the trajectory {traj=}.")

        # for proteins that don't have some chi angles we filter which
        which = list(
            filter(
                lambda x: True if len(indices_dict[x]) > 0 else False,
                indices_dict.keys(),
            )
        )

        # change the sorting to be per-residue and not all chi1 and then all chi2 angles
        self.per_res_dict = {}
        for r in traj.top.residues:
            arrs = []
            bools = []
            for k in which:
                if np.any(np.in1d(valid[k], np.array([a.index for a in r.atoms]))):
                    where = np.where(
                        np.in1d(
                            valid[k].flatten(), np.array([a.index for a in r.atoms])
                        )
                    )[0]
                    arr = valid[k].flatten()[where]
                    bools.append(True)
                    arrs.append(arr)
                else:
                    bools.append(False)
            if any(bools):
                self.per_res_dict[str(r)] = np.vstack(arrs)

        self._prefix_label_lengths = np.array(
            [len(indices_dict[k]) if k in which else 0 for k in self.options]
        )
        indices = np.vstack([v for v in self.per_res_dict.values()])

        super(SideChainDihedrals, self).__init__(
            traj=traj,
            dih_indexes=indices,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
            check_aas=check_aas,
            delayed=delayed,
        )

        if generic_labels:
            self.describe = self.generic_describe

    @property
    def name(self) -> str:
        """str: The name of the class: "SideChainDihedrals"."""
        return "SideChainDihedrals"

    @property
    def indexes(self) -> np.ndarray:
        """np.ndarray: A (n_angles, 4) shaped numpy array giving the atom indices
        of the dihedral angles to be calculated."""
        return self.angle_indexes

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        if hasattr(self.traj, "clustal_w"):
            residue_mapping = {}
            i = 1
            j = 1
            for res in [*self.traj.clustal_w]:
                if res == "-":
                    j += 1
                    continue
                residue_mapping[i] = j
                i += 1
                j += 1

            def getlbl(at: md.topology.Atom):
                resSeq = at.residue.resSeq
                resSeq = residue_mapping[resSeq]
                r = f"RESID  {at.residue.name}:{resSeq:>4} CHAIN {at.residue.chain.index}"
                return r

        else:
            getlbl = (
                lambda at: f"RESID  {at.residue.name}:{at.residue.resSeq:>4} CHAIN {at.residue.chain.index}"
            )
        prefixes = []
        for lengths, label in zip(self._prefix_label_lengths, self.options):
            if self.cossin:
                lengths *= 2
            prefixes.extend([label.upper()] * lengths)
        prefixes = []
        for key, value in self.per_res_dict.items():
            if self.cossin:
                prefixes.extend(
                    [opt.upper() for opt in self.options[: value.shape[0]]] * 2
                )
            else:
                prefixes.extend([opt.upper() for opt in self.options[: value.shape[0]]])

        if self.cossin:
            cossin = ("COS({dih} {res})", "SIN({dih} {res})")
            labels = [
                s.format(
                    dih=prefixes[j + len(cossin) * i],
                    res=getlbl(self.top.atom(ires[1])),
                )
                for i, ires in enumerate(self.angle_indexes)
                for j, s in enumerate(cossin)
            ]
        else:
            labels = [
                "SIDECHDIH {dih}  {res}".format(
                    dih=prefixes[i], res=getlbl(self.top.atom(ires[1]))
                )
                for i, ires in enumerate(self.angle_indexes)
            ]
        labels = list(map(lambda x: x[:14] + x[27:31], labels))
        return labels

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature. The length
                is determined by the `dih_indexes` and the `cossin` argument
                in the `__init__()` method. If `cossin` is false, then
                `len(describe()) == self.angle_indexes[-1]`, else `len(describe())`
                is twice as long.

        """
        top = self.top
        getlbl = (
            lambda at: f"RESID  {at.residue.name}:{at.residue.resSeq:>4} CHAIN {at.residue.chain.index}"
        )
        prefixes = []
        for lengths, label in zip(self._prefix_label_lengths, self.options):
            if self.cossin:
                lengths *= 2
            prefixes.extend([label.upper()] * lengths)
        prefixes = []
        for key, value in self.per_res_dict.items():
            if self.cossin:
                prefixes.extend(
                    [opt.upper() for opt in self.options[: value.shape[0]]] * 2
                )
            else:
                prefixes.extend([opt.upper() for opt in self.options[: value.shape[0]]])

        if self.cossin:
            cossin = ("COS({dih} {res})", "SIN({dih} {res})")
            labels = [
                s.format(
                    dih=prefixes[j + len(cossin) * i],
                    res=getlbl(self.top.atom(ires[1])),
                )
                for i, ires in enumerate(self.angle_indexes)
                for j, s in enumerate(cossin)
            ]
        else:
            labels = [
                "SIDECHDIH {dih}  {res}".format(
                    dih=prefixes[i], res=getlbl(self.top.atom(ires[1]))
                )
                for i, ires in enumerate(self.angle_indexes)
            ]

        return labels


class AllCartesians(SelectionFeature):
    """Feature that collects all cartesian positions of all atoms in the trajectory.

    Note:
        The order of the cartesians is not as in standard MD coordinates.
        Rather than giving the positions of all atoms of the first residue, and
        then all positions of the second, and so on, this feature gives all
        central (backbone) cartesians first, followed by the cartesians of the
        sidechains. This allows better and faster backmapping. See
        `encodermap.misc.backmapping._full_backmapping_np` for mor info,
        why this is easier.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case, it is 'POSITION'.

    """

    prefix_label: str = "POSITION "

    def __init__(
        self,
        traj: SingleTraj,
        check_aas: bool = True,
        generic_labels: bool = False,
        delayed: bool = False,
    ) -> None:
        """Instantiate the AllCartesians class.

        Args:
            traj (em.SingleTraj): A mdtraj topology.

        """
        self.central_indices = CentralCartesians(traj).indexes
        try:
            indexes = np.concatenate(
                [self.central_indices, SideChainCartesians(traj).indexes]
            )
        except ValueError as e:
            if "Could not determine" in str(e):
                warnings.warn(
                    f"The topology of {traj} does not contain any sidechains. The "
                    f"`AllCartesians` feature will just contain backbone coordinates."
                )
                indexes = CentralCartesians(traj).indexes
            else:
                raise e
        super().__init__(traj, indexes=indexes, check_aas=check_aas, delayed=delayed)
        if generic_labels:
            self.describe = self.generic_describe

    @property
    def name(self) -> str:
        """str: The name of this class: 'AllCartesians'"""
        return "AllCartesians"

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        labels = []
        if hasattr(self.traj, "clustal_w"):
            raise NotImplementedError(
                f"SideChainCartesians can't currently handle alignments. The "
                f"implementation below won't probably work."
            )
            # clustal_w_ = [*self.traj.clustal_w]
            # clustal_w = [None] * (len(clustal_w_) * 3)
            # clustal_w[::3] = clustal_w_
            # clustal_w[1::3] = clustal_w_
            # clustal_w[2::3] = clustal_w_
            # clustal_w = np.array(clustal_w)
            # indices = (np.arange(len(clustal_w)) + 1)[clustal_w != "-"]
            # assert len(indices) == len(
            #     self.central_indexes
            # ), f"{indices.shape=} {self.indexes.shape=} {clustal_w.shape=} {clustal_w[:20]}"
        else:
            indices = self.indexes
        visited_residues = set()
        for i in indices:
            if i in self.central_indices:
                position = "c"
            else:
                position = "s"
            residx = self.traj.top.atom(i).residue.index + 1
            rescode = str(residx) + position
            if rescode not in visited_residues:
                visited_residues.add(rescode)
                atom_index = 1
            for pos in ["X", "Y", "Z"]:
                labels.append(
                    f"{self.prefix_label} {pos} {atom_index} {residx} {position}"
                )
            atom_index += 1
        return labels

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature. This list has as many entries as atoms in `self.top`.

        """
        getlbl = (
            lambda at: f"ATOM  {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4} CHAIN {at.residue.chain.index}"
        )
        labels = []
        for i in self.indexes:
            for pos in ["X", "Y", "Z"]:
                labels.append(
                    f"{self.prefix_label} {pos}     {getlbl(self.top.atom(i))}"
                )
        return labels


class CentralCartesians(SelectionFeature):
    """Feature that collects all cartesian positions of the backbone atoms.

    Examples:
        >>> import encodermap as em
        >>> from pprint import pprint
        >>> traj = em.load_project("pASP_pGLU", 0)[0]
        >>> traj  # doctest: +ELLIPSIS
        <encodermap.SingleTraj object...>
        >>> feature = em.features.CentralCartesians(traj, generic_labels=False)
        >>> pprint(feature.describe())  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ['CENTERPOS X     ATOM     N:    0 GLU:   1 CHAIN 0',
         'CENTERPOS Y     ATOM     N:    0 GLU:   1 CHAIN 0',
         'CENTERPOS Z     ATOM     N:    0 GLU:   1 CHAIN 0',
         'CENTERPOS X     ATOM    CA:    3 GLU:   1 CHAIN 0',
         'CENTERPOS Y     ATOM    CA:    3 GLU:   1 CHAIN 0',
         'CENTERPOS Z     ATOM    CA:    3 GLU:   1 CHAIN 0',
         '...
         'CENTERPOS Z     ATOM     C:   65 GLU:   6 CHAIN 0']
         >>> feature = em.features.CentralCartesians(traj, generic_labels=True)
         >>> pprint(feature.describe())  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
         ['CENTERPOS X 1',
          'CENTERPOS Y 1',
          'CENTERPOS Z 1',
          'CENTERPOS X 2',
          'CENTERPOS Y 2',
          'CENTERPOS Z 2',
          '...
          'CENTERPOS Z 18']

    """

    prefix_label: str = "CENTERPOS"

    def __init__(
        self,
        traj: SingleTraj,
        generic_labels: bool = False,
        check_aas: bool = True,
        delayed: bool = False,
    ) -> None:
        """Instantiate the CentralCartesians class.

        In contrary to PyEMMA (which has now been archived), this feature returns
        a high-dimensional array along the feature axis. In PyEMMA's and now in
        EncoderMap's `SelectionFeature`, the cartesian coordinates of the atoms are
        returned as a list of [x1, y1, z1, x2, y2, z2, x3, ..., zn]. This feature
        yields a (n_atoms, 3) array with an extra dimension (carteisan coordinate)::

            [
                [x1, y1, z1],
                [x2, y2, z2],
                ...,
                [xn, yn, zn],
            ]

        Args:
            traj (SingleTraj): An instance of `encodermap.SingleTraj`. Using
                `SingleTrajs` instead of `md.Topology` (as it was in PyEMMA),
                offers access to EncoderMap's `CustomTopology`, which can be
                used to adapt the featurization and NN training for a wide
                range of protein and non-protein MD trajectories.
            generic_labels (bool): Whether to use generic labels to describe the
                feature. Generic labels can be used to align different topologies.
                If False, the labels returned by this feature's `describe()` method
                are topology-specific ("CENTERPOS X     ATOM     N:    0 ASP:   1 CHAIN 0").
                If True, the labels are generic ("CENTERPOS X 0") and can be
                aligned with other Features, that contain topologies, of which ASP
                might not be the first amino acid.
            check_aas (bool): Whether to check if all residues in `traj` are
                known, prior to computing.

        """
        self.traj = traj
        self.indexes = self.traj.top.select("name CA or name C or name N")
        # filter out unwanted indexes
        unwanted_resnames = [
            k for k, v in self.traj._custom_top.amino_acid_codes.items() if v is None
        ]
        self.indexes = np.array(
            list(
                filter(
                    lambda x: self.traj.top.atom(x).residue.name
                    not in unwanted_resnames,
                    self.indexes,
                )
            )
        )
        super().__init__(
            self.traj, indexes=self.indexes, check_aas=check_aas, delayed=delayed
        )
        self.dimension = 3 * len(self.indexes)

        if generic_labels:
            self.describe = self.generic_describe

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        labels = []
        if hasattr(self.traj, "clustal_w"):
            clustal_w_ = [*self.traj.clustal_w]
            clustal_w = [None] * (len(clustal_w_) * 3)
            clustal_w[::3] = clustal_w_
            clustal_w[1::3] = clustal_w_
            clustal_w[2::3] = clustal_w_
            clustal_w = np.array(clustal_w)
            indices = (np.arange(len(clustal_w)) + 1)[clustal_w != "-"]
            assert len(indices) == len(
                self.indexes
            ), f"{indices.shape=} {self.indexes.shape=} {clustal_w.shape=} {clustal_w[:20]}"
        else:
            indices = np.arange(len(self.indexes)) + 1
        for i in indices:
            for pos in ["X", "Y", "Z"]:
                labels.append(f"{self.prefix_label} {pos} {i}")
        return labels

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        getlbl = (
            lambda at: f"ATOM  {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4} CHAIN {at.residue.chain.index}"
        )
        labels = []
        for i in self.indexes:
            for pos in ["X", "Y", "Z"]:
                labels.append(
                    f"{self.prefix_label} {pos}     {getlbl(self.top.atom(i))}"
                )
        return labels

    @property
    def name(self) -> str:
        """str: The name of the class: "CentralCartesians"."""
        return "CentralCartesians"


class SideChainCartesians(SelectionFeature):
    """Feature that collects all cartesian positions of all non-backbone atoms.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'SIDECHPOS'.

    """

    prefix_label: str = "SIDECHPOS"

    def __init__(
        self,
        traj: SingleTraj,
        check_aas: bool = True,
        generic_labels: bool = False,
        delayed: bool = False,
    ) -> None:
        """Instantiate the `SideChainCartesians feature.

        Uses MDTraj's 'not backbone' topology selector. Is not guaranteed to
        work with the better tested `SideChainDihedrals`.

        """
        self.traj = traj
        dihe_indices = np.unique(SideChainDihedrals(traj=traj).angle_indexes.flatten())
        backbone_indices = CentralCartesians(traj=traj).indexes
        indexes = np.setdiff1d(dihe_indices, backbone_indices)
        assert indexes[0] in dihe_indices and indexes[0] not in backbone_indices
        super().__init__(
            self.traj, indexes=indexes, check_aas=check_aas, delayed=delayed
        )
        self.dimension = 3 * len(self.indexes)
        if generic_labels:
            self.describe = self.generic_describe

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        labels = []
        if hasattr(self.traj, "clustal_w"):
            raise NotImplementedError(
                f"SideChainCartesians can't currently handle alignments. The "
                f"implementation below won't probably work."
            )
            # clustal_w_ = [*self.traj.clustal_w]
            # clustal_w = [None] * (len(clustal_w_) * 3)
            # clustal_w[::3] = clustal_w_
            # clustal_w[1::3] = clustal_w_
            # clustal_w[2::3] = clustal_w_
            # clustal_w = np.array(clustal_w)
            # indices = (np.arange(len(clustal_w)) + 1)[clustal_w != "-"]
            # assert len(indices) == len(
            #     self.central_indexes
            # ), f"{indices.shape=} {self.indexes.shape=} {clustal_w.shape=} {clustal_w[:20]}"
        else:
            indices = self.indexes
        visited_residues = set()
        for i in indices:
            residx = self.traj.top.atom(i).residue.index + 1
            if residx not in visited_residues:
                visited_residues.add(residx)
                atom_index = 1
            for pos in ["X", "Y", "Z"]:
                labels.append(f"{self.prefix_label} {pos} {atom_index} {residx}")
            atom_index += 1
        return labels

    @property
    def name(self):
        """str: The name of the class: "SideChainCartesians"."""
        return "SideChainCartesians"

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        getlbl = (
            lambda at: f"ATOM  {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4} CHAIN {at.residue.chain.index}"
        )
        labels = []
        for i in self.indexes:
            for pos in ["X", "Y", "Z"]:
                labels.append(
                    f"{self.prefix_label} {pos}     {getlbl(self.top.atom(i))}"
                )
        return labels


class AllBondDistances(DistanceFeature):
    """Feature that collects all bonds in a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'DISTANCE'.

    """

    prefix_label: str = "DISTANCE        "

    def __init__(
        self,
        traj: SingleTraj,
        distance_indexes: Optional[np.ndarray] = None,
        periodic: bool = True,
        check_aas: bool = True,
        delayed: bool = False,
    ) -> None:
        self.distance_indexes = distance_indexes
        if self.distance_indexes is None:
            self.traj = traj
            self.distance_indexes = np.vstack(
                [[b[0].index, b[1].index] for b in self.traj.top.bonds]
            )
            # print(self.distance_indexes, len(self.distance_indexes))
            super().__init__(
                self.traj,
                self.distance_indexes,
                periodic,
                check_aas=check_aas,
                delayed=delayed,
            )
        else:
            super().__init__(
                self.traj,
                self.distance_indexes,
                periodic,
                check_aas=check_aas,
                delayed=delayed,
            )
            # print(self.distance_indexes, len(self.distance_indexes))

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        if hasattr(self.traj, "clustal_w"):
            raise NotImplementedError(
                f"AllBondDistances can currently not align disjoint sequences."
            )
        else:
            indices = np.arange(len(self.distance_indexes)) + 1
        labels = []
        for i in indices:
            labels.append(f"{self.prefix_label}{i}")
        return labels

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        getlbl = (
            lambda at: f"ATOM  {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4}"
        )
        labels = []
        for i, j in self.distance_indexes:
            i, j = self.top.atom(i), self.top.atom(j)
            labels.append(
                f"{self.prefix_label}{getlbl(i)} DIST  {getlbl(j)} CHAIN {int(np.unique([a.residue.chain.index for a in [i, j]])[0])}"
            )
        return labels

    @property
    def name(self) -> str:
        """str: The name of the class: "AllBondDistances"."""
        return "AllBondDistances"

    @property
    def indexes(self) -> np.ndarray:
        """np.ndarray: A (n_angles, 2) shaped numpy array giving the atom indices
        of the distances to be calculated."""
        return self.distance_indexes


class CentralBondDistances(AllBondDistances):
    """Feature that collects all bonds in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case, it is 'CENTERDISTANCE'.

    """

    prefix_label: str = "CENTERDISTANCE  "

    def __init__(
        self,
        traj: SingleTraj,
        distance_indexes: Optional[np.ndarray] = None,
        periodic: bool = True,
        generic_labels: bool = False,
        check_aas: bool = True,
        delayed: bool = False,
    ) -> None:
        self.traj = traj
        select = traj.top.select("name CA or name C or name N")

        if distance_indexes is None:
            distance_indexes = []

        for b in traj.top.bonds:
            if np.all([np.isin(x.index, select) for x in b]):
                distance_indexes.append([x.index for x in b])
        distance_indexes = np.sort(distance_indexes, axis=0)

        if generic_labels:
            self.describe = self.generic_describe

        super().__init__(
            self.traj, distance_indexes, periodic, check_aas=check_aas, delayed=delayed
        )

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        if hasattr(self.traj, "clustal_w"):
            indices = []
            clustal_w_ = [*self.traj.clustal_w]
            clustal_w = [None] * (len(clustal_w_) * 3)
            clustal_w[::3] = clustal_w_
            clustal_w[1::3] = clustal_w_
            clustal_w[2::3] = clustal_w_
            i = 0
            for a, b in zip(clustal_w[:-1], clustal_w[1:]):
                i += 1
                if a == "-":
                    continue
                indices.append(i)
            indices = np.array(indices)
        else:
            indices = np.arange(len(self.distance_indexes)) + 1
        labels = []
        for i in indices:
            labels.append(f"{self.prefix_label}{i}")
        return labels

    @property
    def name(self) -> str:
        """str: The name of the class: "CentralBondDistances"."""
        return "CentralBondDistances"

    @property
    def indexes(self) -> np.ndarray:
        """np.ndarray: A (n_angles, 2) shaped numpy array giving the atom indices
        of the distances to be calculated."""
        return self.distance_indexes


class SideChainBondDistances(AllBondDistances):
    """Feature that collects all bonds not in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'SIDECHDISTANCE'.

    """

    prefix_label: str = "SIDECHDISTANCE  "

    def __init__(
        self,
        traj: SingleTraj,
        periodic: bool = True,
        check_aas: bool = True,
        generic_labels: bool = False,
        delayed: bool = False,
    ) -> None:
        self.traj = traj

        which = ["chi1", "chi2", "chi3", "chi4", "chi5"]
        indices_dict = {k: getattr(self.traj, f"indices_{k}") for k in which}
        # flat_list = [
        #     item
        #     for sublist in indices_dict.values()
        #     for item in sublist.flatten().tolist()
        # ]
        # atoms_in_sidechain_dihedrals = set(flat_list)

        distance_indexes = []
        for angle, indices in indices_dict.items():
            for index in indices:
                if angle == "chi1":
                    distance_indexes.append([index[1], index[2]])
                    distance_indexes.append([index[2], index[3]])
                else:
                    distance_indexes.append([index[2], index[3]])
        distance_indexes = np.sort(distance_indexes, axis=0)
        super().__init__(
            self.traj,
            distance_indexes=distance_indexes,
            periodic=periodic,
            check_aas=check_aas,
            delayed=delayed,
        )
        if generic_labels:
            self.describe = self.generic_describe

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        labels = []
        if hasattr(self.traj, "clustal_w"):
            raise NotImplementedError
            # indices = []
            # clustal_w_ = [*self.traj.clustal_w]
            # clustal_w = [None] * (len(clustal_w_) * 3)
            # clustal_w[::3] = clustal_w_
            # clustal_w[1::3] = clustal_w_
            # clustal_w[2::3] = clustal_w_
            # i = 0
            # for a, b in zip(clustal_w[:-1], clustal_w[1:]):
            #     i += 1
            #     if a == "-":
            #         continue
            #     indices.append(i)
            # indices = np.array(indices)
        else:
            indices = self.distance_indexes
        visited_residues = set()
        for a, b in indices:
            residx_a = self.traj.top.atom(a).residue.index + 1
            residx_b = self.traj.top.atom(b).residue.index + 1
            assert residx_a == residx_b, (
                f"The sidechain distance between atom {self.traj.top.atom(a)} and "
                f"{self.traj.top.atom(b)} describes a distance between two residues "
                f"({residx_a} and {residx_b})."
                f"As sidechains belong always to a single residue, something is off. "
            )
            if residx_a not in visited_residues:
                visited_residues.add(residx_a)
                distance_index = 1
            labels.append(f"{self.prefix_label} {distance_index} {residx_a}")
            distance_index += 1
        return labels

    @property
    def name(self):
        """str: The name of the class: "SideChainBondDistances"."""
        return "SideChainBondDistances"

    @property
    def indexes(self):
        """np.ndarray: A (n_angles, 2) shaped numpy array giving the atom indices
        of the distances to be calculated."""
        return self.distance_indexes


class CentralAngles(AngleFeature):
    """Feature that collects all angles in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'CENTERANGLE'.

    """

    prefix_label: str = "CENTERANGLE     "

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        generic_labels: bool = False,
        check_aas: bool = True,
        delayed: bool = False,
    ) -> None:
        self.traj = traj
        select = traj.top.select("name CA or name C or name N")
        # add 4 bonds in KAC
        # if any([r.name == "KAC" for r in top.residues]):
        #     self.top = add_KAC_backbone_bonds(self.top)
        bonds = np.vstack([[x.index for x in b] for b in traj.top.bonds])
        bond_names = np.vstack([[x for x in b] for b in traj.top.bonds])
        angle_indexes = []
        for a in select:
            where = np.where(bonds == a)
            possible_bonds = bonds[where[0], :]
            possible_bond_names = bond_names[where[0], :]
            where = np.isin(possible_bonds, select)
            hits = np.count_nonzero(np.all(where, axis=1))
            if hits <= 1:
                # atom is not part of any angles
                continue
            elif hits == 2:
                where = np.all(where, axis=1)
                these = np.unique(
                    [traj.top.atom(i).index for i in possible_bonds[where, :].flatten()]
                )
                angle_indexes.append(these)
            elif hits == 3:
                a = traj.top.atom(a)
                bonds = "\n".join(
                    [
                        f"BOND {str(i):<10}-{str(j):>10}"
                        for i, j in traj.top.bonds
                        if i == a or j == a
                    ]
                )
                raise Exception(
                    f"The atom {a} takes part in three possible angles defined "
                    f"by the C, CA, and N atoms:\n{bonds}."
                )
            elif hits == 4:
                raise Exception(
                    "Can't deal with these angles. One atom is part of four possible angles"
                )
            else:
                raise Exception(
                    "Can't deal with these angles. One atom is part of more than three angles"
                )

        angle_indexes = np.vstack(angle_indexes)
        angle_indexes = np.unique(angle_indexes, axis=0)
        if generic_labels:
            self.describe = self.generic_describe
        super().__init__(
            traj, angle_indexes, deg, cossin, periodic, check_aas, delayed=delayed
        )

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        if hasattr(self.traj, "clustal_w"):
            indices = []
            clustal_w_ = [*self.traj.clustal_w]
            clustal_w = [None] * (len(clustal_w_) * 3)
            clustal_w[::3] = clustal_w_
            clustal_w[1::3] = clustal_w_
            clustal_w[2::3] = clustal_w_
            i = 0
            for a, b, c in zip(clustal_w[:-2], clustal_w[1:-1], clustal_w[2:]):
                i += 1
                if a == "-":
                    continue
                indices.append(i)
            indices = np.array(indices)
        else:
            indices = np.arange(len(self.angle_indexes)) + 1
        labels = []
        for i in indices:
            labels.append(f"{self.prefix_label}{i}")
        return labels

    def describe(self) -> list[str]:
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        getlbl = (
            lambda at: f"ATOM  {at.name:>4}:{at.index:>5} {at.residue.name}:{at.residue.resSeq:>4}"
        )
        labels = []
        for i, j, k in self.angle_indexes:
            i, j, k = self.top.atom(i), self.top.atom(j), self.top.atom(k)
            labels.append(
                f"{self.prefix_label}{getlbl(i)} ANGLE {getlbl(j)} ANGLE "
                f"{getlbl(k)} CHAIN "
                f"{int(np.unique([a.residue.chain.index for a in [i, j, k]])[0])}"
            )
        return labels

    @property
    def name(self) -> str:
        """str: The name of the class: "CentralAngles"."""
        return "CentralAngles"

    @property
    def indexes(self) -> np.ndarray:
        """np.ndarray: A (n_angles, 3) shaped numpy array giving the atom indices
        of the angles to be calculated."""
        return self.angle_indexes


class SideChainAngles(AngleFeature):
    """Feature that collects all angles not in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'SIDECHANGLE'.

    """

    prefix_label: str = "SIDECHANGLE "

    def __init__(
        self,
        traj: SingleTraj,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        check_aas: bool = True,
        generic_labels: bool = False,
        delayed: bool = False,
    ) -> None:
        self.traj = traj
        angle_indexes = []
        for residue, ind in traj._custom_top.sidechain_indices_by_residue():
            ind = np.vstack(
                [
                    ind[:-2],
                    ind[1:-1],
                    ind[2:],
                ]
            ).T
            angle_indexes.append(ind)
        angle_indexes = np.vstack(angle_indexes)
        super().__init__(
            self.traj, angle_indexes, deg, cossin, periodic, check_aas, delayed=delayed
        )
        if generic_labels:
            self.describe = self.generic_describe

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        labels = []
        if hasattr(self.traj, "clustal_w"):
            raise NotImplementedError
            # indices = []
            # clustal_w_ = [*self.traj.clustal_w]
            # clustal_w = [None] * (len(clustal_w_) * 3)
            # clustal_w[::3] = clustal_w_
            # clustal_w[1::3] = clustal_w_
            # clustal_w[2::3] = clustal_w_
            # i = 0
            # for a, b, c in zip(clustal_w[:-2], clustal_w[1:-1], clustal_w[2:]):
            #     i += 1
            #     if a == "-":
            #         continue
            #     indices.append(i)
            # indices = np.array(indices)
        else:
            indices = self.angle_indexes
        visited_residues = set()
        for a, b, c in indices:
            residx_a = self.traj.top.atom(a).residue.index + 1
            residx_b = self.traj.top.atom(b).residue.index + 1
            residx_c = self.traj.top.atom(c).residue.index + 1
            assert residx_a == residx_b == residx_c, (
                f"The sidechain distance between atom {self.traj.top.atom(a)}, "
                f"{self.traj.top.atom(b)}, and {self.traj.top.atom(c)} describes "
                f"an angle between two or more residues ({residx_a}, {residx_b}, and {residx_c})."
                f"As sidechains belong always to a single residue, something is off. "
            )
            if residx_a not in visited_residues:
                visited_residues.add(residx_a)
                angle_index = 1
            labels.append(f"{self.prefix_label} {angle_index} {residx_a}")
            angle_index += 1
        return labels

    def describe(self):
        """Gives a list of strings describing this feature's feature-axis.

        A feature computes a collective variable (CV). A CV is aligned with an MD
        trajectory on the time/frame-axis. The feature axis is unique for every
        feature. A feature describing the backbone torsions (phi, omega, psi) would
        have a feature axis with the size 3*n-3, where n is the number of residues.
        The end-to-end distance of a linear protein in contrast would just have
        a feature axis with length 1. This `describe()` method will label these
        values unambiguously. A backbone torsion feature's `describe()` could be
        ['phi_1', 'omega_1', 'psi_1', 'phi_2', 'omega_2', ..., 'psi_n-1'].
        The end-to-end distance feature could be described by
        ['distance_between_MET1_and_LYS80'].

        Returns:
            list[str]: The labels of this feature.

        """
        getlbl = (
            lambda at: f"ATOM {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4}"
        )
        labels = []
        for i, j, k in self.angle_indexes:
            i, j, k = self.top.atom(i), self.top.atom(j), self.top.atom(k)
            labels.append(
                f"{self.prefix_label}{getlbl(i)} ANGLE {getlbl(j)} ANGLE {getlbl(k)} CHAIN {int(np.unique([a.residue.chain.index for a in [i, j, k]])[0])}"
            )
        return labels

    @property
    def name(self):
        """str: The name of the class: "SideChainAngles"."""
        return "SideChainAngles"

    @property
    def indexes(self):
        """np.ndarray: A (n_angles, 3) shaped numpy array giving the atom indices
        of the angles to be calculated."""
        return self.angle_indexes

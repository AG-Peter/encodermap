# -*- coding: utf-8 -*-
# encodermap/loading/features.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
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

"""

##############################################################################
# Imports
##############################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import inspect
import itertools

# Third Party Imports
import numpy as np
from optional_imports import _optional_import

# Encodermap imports
import encodermap

# Local Folder Imports
from ..trajinfo.info_all import TrajEnsemble
from ..trajinfo.info_single import SingleTraj
from ..trajinfo.trajinfo_utils import _AMINO_ACID_CODES


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


##############################################################################
# Typing
##############################################################################


# Standard Library Imports
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, Union


if TYPE_CHECKING:
    # Third Party Imports
    import mdtraj as md

    # Encodermap imports
    from encodermap._typing import AnyFeature


##############################################################################
# Globals
##############################################################################

__all__ = [
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
]

##############################################################################
# Functions
##############################################################################


def _check_aas(traj: Union[SingleTraj, TrajEnsemble]) -> None:
    r = set([r.name for r in traj.top.residues])
    diff = r - set(_AMINO_ACID_CODES.keys())
    if diff:
        raise Exception(
            f"I don't recognize these residues: {diff}. "
            f"Either add them the `SingleTraj` or `TrajEnsemble` via "
            f"`traj.load_custom_topology(custom_aas)` or "
            f"`trajs.load_custom_topology(custom_aas)`"
            f"Or remove them from your trajectory. See the documentation of the "
            f"`em.CustomTopology` class."
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


class FeatureMeta(type):
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
        return x


class Feature(metaclass=FeatureMeta):
    def __init__(
        self, traj: Union[SingleTraj, TrajEnsemble], check_aas: bool = True
    ) -> None:
        self.traj = traj
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

    @property
    def dimension(self):
        return self._dim

    @dimension.setter
    def dimension(self, val):
        self._dim = int(val)

    def __eq__(self, other):
        if not isinstance(other, Feature):
            return False
        return self.dimension == other.dimension and self.traj == other.traj

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if xyz is not None:
            input_atoms = xyz.shape[1]
            self_atoms = self.traj.xyz.shape[1]
            if hasattr(self, "periodic"):
                if self.periodic:
                    assert (
                        unitcell_vectors is not None and unitcell_infos is not None
                    ), (
                        f"When providing a `feature.transform` function with xyz "
                        f"data, and setting {self.periodic=} to True, please "
                        f"also provide `unitcell_vectors` and `unitcell_infos` "
                        f"to calculate distances/angles/dihedrals in periodic space."
                    )
            assert input_atoms == self_atoms, (
                f"The shape of the input xyz coordinates is off from the expected "
                f"shape. The topology {self.top} defines {self_atoms} atoms. The "
                f"provided array has {xyz.shaope[1]=} atoms."
            )
        else:
            xyz = self.traj.xyz

        if unitcell_vectors is not None:
            assert len(unitcell_vectors) == len(xyz), (
                f"The shape of the provided `unitcell_vectors` is off from the "
                f"expected shape. The xyz data contains {len(xyz)=} frames, while "
                f"the `unitcell_vectors` contains {len(unitcell_vectors)=} frames."
            )
        else:
            if self.traj._have_unitcell:
                unitcell_vectors = self.traj.unitcell_vectors
            else:
                unitcell_vectors = None
        if unitcell_infos is not None:
            assert len(unitcell_infos) == len(xyz), (
                f"The shape of the provided `unitcell_infos` is off from the "
                f"expected shape. The xyz data contains {len(xyz)=} frames, while "
                f"the `unitcell_infos` contains {len(unitcell_infos)=} frames."
            )
        else:
            if self.traj._have_unitcell:
                unitcell_infos = np.hstack(
                    [self.traj.unitcell_lengths, self.traj.unitcell_angles]
                )
            else:
                unitcell_infos = None
        return xyz, unitcell_vectors, unitcell_infos


class CustomFeature(Feature):
    def __init__(
        self,
        fun: Callable,
        dim: int,
        traj: Optional[SingleTraj] = None,
        description: Optional[str] = None,
        fun_args: tuple[Any] = tuple(),
        fun_kwargs: dict[str, Any] = None,
    ) -> None:
        self.id = None
        self.traj = traj
        if fun_kwargs is None:
            fun_kwargs = {}
        self._fun = fun
        self._args = fun_args
        self._kwargs = fun_kwargs
        self._dim = dim
        self.desc = description

    def describe(self) -> list[str]:
        if isinstance(self.desc, str):
            desc = [self.desc]
        if self.desc is None:
            arg_str = (
                f"{self._args}, {self._kwargs}" if self._kwargs else f"{self._args}"
            )
            desc = [
                f"CustomFeature[{self.id}][0] calling {self._fun} with args {arg_str}"
            ]
        elif self.desc and not (len(self.desc) == self._dim or len(self.desc) == 1):
            raise ValueError(
                f"to avoid confusion, ensure the lengths of 'description' "
                f"list matches dimension - or give a single element which will be repeated."
                f"Input was {self.desc}"
            )

        if len(desc) == 1:
            desc *= self.dimension

        return desc

    def transform(
        self,
        traj: Optional[md.traj] = None,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if xyz is not None:
            self.traj = traj
            xyz, unitcell_vectors, unitcell_infos = super().transform(
                xyz, unitcell_vectors, unitcell_infos
            )
            traj = md.Trajectory(
                xyz=xyz,
                topology=self.traj.top,
                unitcell_lengths=unitcell_infos[:, :3],
                unitcell_angles=unitcell_infos[:, 3:],
            )
        feature = self._fun(traj, *self._args, **self._kwargs)
        if not isinstance(feature, np.ndarray):
            raise ValueError("your function should return a NumPy array!")
        return feature


class SelectionFeature(Feature):
    prefix_label = "ATOM:"

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        indexes: Sequence[int],
        check_aas: bool = True,
    ) -> None:
        super().__init__(traj, check_aas)
        self.indexes = np.asarray(indexes).astype("int32")
        if len(self.indexes) == 0:
            raise ValueError(f"Empty indices in {self.__class__.__name__}.")
        self.dimension = 3 * len(self.indexes)

    def describe(self) -> list[str]:
        labels = []
        for i in self.indexes:
            labels.append(f"{self.prefix_label}{_describe_atom(self.top, i)} x")
            labels.append(f"{self.prefix_label}{_describe_atom(self.top, i)} y")
            labels.append(f"{self.prefix_label}{_describe_atom(self.top, i)} z")
        return labels

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        xyz, unitcell_vectors, unitcell_infos = super().transform(
            xyz, unitcell_vectors, unitcell_infos
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
    ) -> None:
        super().__init__(traj, check_aas)
        self.angle_indexes = np.array(angle_indexes).astype("int32")
        if len(self.angle_indexes) == 0:
            raise ValueError("empty indices")
        self.deg = deg
        self.cossin = cossin
        self.periodic = periodic
        self.dimension = len(self.angle_indexes)
        if cossin:
            self.dimension *= 2

    def describe(self) -> list[str]:
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

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if xyz is not None:
            periodic = self.periodic
        else:
            periodic = self.periodic and self.traj._have_unitcell
        xyz, unitcell_vectors, unitcell_infos = super().transform(
            xyz, unitcell_vectors, unitcell_infos
        )
        if periodic:
            assert unitcell_vectors is not None
            if unitcell_infos is None:
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
                unitcell_angles = unitcell_infos[:, 3:]
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
                recognized. Defaults to Treu.

        """
        super().__init__(
            traj=traj,
            angle_indexes=dih_indexes,
            deg=deg,
            cossin=cossin,
            periodic=periodic,
            check_aas=check_aas,
        )

    def describe(self) -> list[str]:
        """A list of strings describing the features.

        Returns:
            list[str]: A list of str describing the feature. The length
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

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Takes xyz and unitcell information to apply the topological calculations on.

        When this method is not provided with any input, it will take the
        traj_container provided as `traj` in the `__init__()` method and transforms
        this trajectory. The argument `xyz` can be the xyz coordinates in nanometer
        of a trajectory with identical topology as `self.traj`. If `periodic` was
        set to True, `unitcell_vectors` and `unitcell_infos` should also be provided.

        Args:
            xyz (Optional[np.ndarray]): A numpy array with shape (n_frames, n_atoms, 3)
                in nanometer. If None is provided the coordinates of `self.traj`
                will be used. Otherwise the topology of this set of xyz
                coordinates should match the topology of `self.atom`.
                Defaults to None.
            unitcell_vectors (Optional[np.ndarray]): When periodic is set to
                True, the `unitcell_vectors` are needed to calculate the
                minimum image convention in a periodic space. This numpy
                array should have the shape (n_frames, 3, 3). The rows of this
                array correlate to the Bravais vectors a, b, and c.
            unitcell_infos (Optional[np.ndarray]): Basically identical to
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
        xyz, unitcell_vectors, unitcell_infos = Feature.transform(
            self, xyz, unitcell_vectors, unitcell_infos
        )
        if periodic:
            assert unitcell_vectors is not None

            # convert to angles
            if unitcell_infos is None:
                unitcell_angles = []
                for fr_unitcell_vectors in unitcell_vectors:
                    _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                        fr_unitcell_vectors[0],
                        fr_unitcell_vectors[1],
                        fr_unitcell_vectors[2],
                    )
                    unitcell_angles.append(np.array([alpha, beta, gamma]))
            else:
                unitcell_angles = unitcell_infos[:, 3:]

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
    prefix_label = "DIST:"

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        distance_indexes: np.ndarray,
        periodic: bool = True,
        dim: Optional[int] = None,
        check_aas: bool = True,
    ) -> None:
        super().__init__(traj, check_aas)
        self.distance_indexes = np.array(distance_indexes)
        if len(self.distance_indexes) == 0:
            raise ValueError("empty indices")
        self.periodic = periodic
        if dim is None:
            self._dim = len(distance_indexes)
        else:
            self._dim = dim

    def describe(self) -> list[str]:
        labels = [
            (
                f"{self.prefix_label} {_describe_atom(self.top, pair[0])} "
                f"{_describe_atom(self.top, pair[1])}"
            )
            for pair in self.distance_indexes
        ]
        return labels

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if xyz is not None:
            periodic = self.periodic
        else:
            periodic = self.periodic and self.traj._have_unitcell
        xyz, unitcell_vectors, unitcell_infos = super().transform(
            xyz, unitcell_vectors, unitcell_infos
        )
        if periodic:
            assert unitcell_vectors is not None

            # convert to angles
            unitcell_angles = []
            for fr_unitcell_vectors in unitcell_vectors:
                _, _, _, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                    fr_unitcell_vectors[0],
                    fr_unitcell_vectors[1],
                    fr_unitcell_vectors[2],
                )
                unitcell_angles.append(np.array([alpha, beta, gamma]))

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
                self.distance_indexes.astype("int32"),
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
            _dist(xyz, self.distance_indexes, out)
        return out


class AlignFeature(SelectionFeature):
    prefix_label = "aligned ATOM:"

    def __init__(
        self,
        traj: SingleTraj,
        reference: md.Trajectory,
        indexes: np.ndarray,
        atom_indices: Optional[np.ndarray] = None,
        ref_atom_indices: Optional[np.ndarray] = None,
        in_place: bool = True,
    ) -> None:
        super(AlignFeature, self).__init__(traj=traj, indexes=indexes)
        self.ref = reference
        self.atom_indices = atom_indices
        self.ref_atom_indices = ref_atom_indices
        self.in_place = in_place

    def transform(
        self,
    ) -> np.ndarray:
        if not self.in_place:
            traj = self.traj.slice(slice(None), copy=True)
        aligned = traj.superpose(
            reference=self.ref,
            atom_indices=self.atom_indices,
            ref_atom_indices=self.ref_atom_indices,
        )
        # apply selection
        return super(AlignFeature, self).transform(aligned.xyz)


class InverseDistanceFeature(DistanceFeature):
    prefix_label = "INVDIST:"

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        distance_indexes: np.ndarray,
        periodic: bool = True,
    ) -> None:
        DistanceFeature.__init__(self, traj, distance_indexes, periodic=periodic)

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return 1.0 / super().transform(xyz, unitcell_vectors, unitcell_infos)


class ContactFeature(DistanceFeature):
    prefix_label = "CONTACT:"

    def __init__(
        self,
        traj: SingleTraj,
        distance_indexes: np.ndarray,
        threshold: float = 5.0,
        periodic: bool = True,
        count_contacts: bool = False,
    ) -> None:
        super(ContactFeature, self).__init__(traj, distance_indexes, periodic=periodic)
        if count_contacts:
            self.prefix_label = "counted " + self.prefix_label
        self.threshold = threshold
        self.count_contacts = count_contacts
        if count_contacts:
            self.dimension = 1
        else:
            self.dimension = len(self.distance_indexes)

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        dists = super(ContactFeature, self).transform(
            xyz, unitcell_vectors, unitcell_infos
        )
        res = np.zeros(
            (len(self.traj), self.distance_indexes.shape[0]), dtype=np.float32
        )
        I = np.argwhere(dists <= self.threshold)
        res[I[:, 0], I[:, 1]] = 1.0
        if self.count_contacts:
            return res.sum(axis=1, keepdims=True)
        else:
            return res

    def __eq__(self, other):
        raise NotImplementedError()


class BackboneTorsionFeature(DihedralFeature):
    def __init__(
        self,
        traj: SingleTraj,
        selstr: Optional[str] = None,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
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
        )

    def describe(self) -> list[str]:
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
    def __init__(
        self,
        traj: SingleTraj,
        contacts: np.ndarray,
        scheme: Literal["ca", "closest", "closest-heavy"],
        ignore_nonprotein: bool,
        threshold: float,
        periodic: bool,
        count_contacts: bool = False,
    ) -> None:
        if count_contacts and threshold is None:
            raise ValueError(
                "Cannot count contacts when no contact threshold is supplied."
            )

        self.contacts = contacts
        self.scheme = scheme
        self.threshold = threshold
        self.prefix_label = "RES_DIST (%s)" % scheme
        self.ignore_nonprotein = ignore_nonprotein

        if count_contacts:
            self.prefix_label = "counted " + self.prefix_label
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
        )

    def describe(self) -> list[str]:
        labels = []
        for a, b in self.distance_indexes:
            labels.append(
                f"{self.prefix_label} {self.traj.top.residue(a)} - {self.traj.top.residue(a)}"
            )
        return labels

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        (
            xyz,
            unitcell_vectors,
            unitcell_infos,
        ) = Feature.transform(self, xyz, unitcell_vectors, unitcell_infos)

        # create a dummy traj, with the appropriate topology
        traj = md.Trajectory(
            xyz=xyz,
            topology=self.traj.top,
            unitcell_lengths=unitcell_infos[:, :3],
            unitcell_angles=unitcell_infos[:, 3:],
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
    def __init__(
        self,
        traj: SingleTraj,
        group_definitions: Sequence[int],
        ref_geom: Optional[md.Trajectory] = None,
        image_molecules: bool = False,
        mass_weighted: bool = True,
    ) -> None:
        if not (ref_geom is None or isinstance(ref_geom, md.Trajectory)):
            raise ValueError(
                f"argument ref_geom has to be either None or and "
                f"mdtraj.Trajectory, got instead {type(ref_geom)}"
            )

        self.ref_geom = ref_geom
        self.traj = traj
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

        # Prepare and store the description
        self._describe = []
        for group in self.group_definitions:
            for coor in "xyz":
                self._describe.append(
                    f"COM-{coor} of atom group [{group[:3]}..{group[-3:]}]"
                )
        self.dimension = 3 * len(self.group_definitions)

    def describe(self) -> list[str]:
        return self._describe

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        (
            xyz,
            unitcell_vectors,
            unitcell_infos,
        ) = Feature.transform(self, xyz, unitcell_vectors, unitcell_infos)
        # create a dummy traj, with the appropriate topology
        traj = md.Trajectory(
            xyz=xyz,
            topology=self.traj.top,
            unitcell_lengths=unitcell_infos[:, :3],
            unitcell_angles=unitcell_infos[:, 3:],
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
    ) -> None:
        super(ResidueCOMFeature, self).__init__(
            traj,
            residue_atoms,
            mass_weighted=mass_weighted,
            ref_geom=ref_geom,
            image_molecules=image_molecules,
        )

        # This are the only extra attributes that residueCOMFeature should have
        self.residue_indices = residue_indices
        self.scheme = scheme

        # Overwrite the self._describe attribute, this way the method of the superclass can be used "as is"
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
        )

    def describe(self) -> list[str]:
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
    def __init__(
        self,
        traj: SingleTraj,
        ref: Union[md.Trajectory, SingleTraj],
        ref_frame: int = 0,
        atom_indices: Optional[np.ndarray] = None,
        precentered: bool = False,
    ) -> None:
        self.traj = traj

        assert isinstance(
            ref_frame, int
        ), f"ref_frame has to be of type integer, and not {type(ref_frame)}"

        if isinstance(ref, (md.Trajectory, SingleTraj)):
            self.name = ref.__repr__()[:]
        else:
            raise TypeError(
                "input reference has to be either a filename or "
                "a mdtraj.Trajectory object, and not of %s" % type(ref)
            )

        self.ref = ref
        self.ref_frame = ref_frame
        self.atom_indices = atom_indices
        self.precentered = precentered
        self.dimension = 1

    def describe(self) -> list[str]:
        label = "minrmsd to frame %u of %s" % (self.ref_frame, self.name)
        if self.precentered:
            label += ", precentered=True"
        if self.atom_indices is not None:
            label += ", subset of atoms  "
        return [label]

    def transform(
        self,
        xyz: Optional[np.ndarray] = None,
        unitcell_vectors: Optional[np.ndarray] = None,
        unitcell_infos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        (
            xyz,
            unitcell_vectors,
            unitcell_infos,
        ) = Feature.transform(self, xyz, unitcell_vectors, unitcell_infos)

        # create a dummy traj, with the appropriate topology
        traj = md.Trajectory(
            xyz=xyz,
            topology=self.traj.top,
            unitcell_lengths=unitcell_infos[:, :3],
            unitcell_angles=unitcell_infos[:, 3:],
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
            cossin (bool):  If True, each angle will be returned as a pair of
                (sin(x), cos(x)). This is useful, if you calculate the mean
                (e.g TICA/PCA, clustering) in that space. Defaults to False.
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
        if self.cossin:
            sin_cos = ("COS(PSI %s)", "SIN(PSI %s)")
            labels_psi = [
                (
                    sin_cos[0] % i,
                    sin_cos[1] % i,
                )
                for i in range(len(self._psi_inds))
            ]
            if self.omega:
                sin_cos = ("COS(OMEGA %s)", "SIN(OMEGA %s)")
                labels_omega = [
                    (
                        sin_cos[0] % i,
                        sin_cos[1] % i,
                    )
                    for i in range(len(self._omega_inds))
                ]
            sin_cos = ("COS(PHI %s)", "SIN(PHI %s)")
            labels_phi = [
                (
                    sin_cos[0] % i,
                    sin_cos[1] % i,
                )
                for i in range(len(self._phi_inds))
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
            labels_psi = [f"CENTERDIH PSI    {i}" for i in range(len(self._psi_inds))]
            if self.omega:
                labels_omega = [
                    f"CENTERDIH OMEGA  {i}" for i in range(len(self._omega_inds))
                ]
            labels_phi = [f"CENTERDIH PHI    {i}" for i in range(len(self._phi_inds))]
            if self.omega:
                zipped = zip(labels_psi, labels_omega, labels_phi)
            else:
                zipped = zip(labels_psi, labels_phi)
            res = list(itertools.chain.from_iterable(zipped))
        return res

    def describe(self) -> list[str]:
        """Returns a list of labels, that can be used to unambiguously define
        atoms in the protein topology.

        Returns:
           list[str]: A list of labels. This list has as many entries as atoms in `self.top`.

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

    @property
    def dask_transform(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "dihedral"


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
            raise ValueError(
                "Could not determine any side chain dihedrals for your topology!"
            )

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

    @property
    def dask_transform(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "dihedral"

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

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
        labels = list(map(lambda x: x[:14] + x[27:31], labels))
        return labels

    def describe(self) -> list[str]:
        """Returns a list of labels, that can be used to unambiguously define
        atoms in the protein topology.

        Returns:
           list[str]: A list of labels. This list has as many entries as atoms in `self.top`.

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

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case, it is 'POSITION'.

    """

    prefix_label = "POSITION "

    def __init__(
        self,
        traj: SingleTraj,
        check_aas: bool = True,
    ) -> None:
        """Instantiate the AllCartesians class.

        Args:
            top (mdtraj.Topology): A mdtraj topology.

        """
        self.indexes = traj.top.select("all")
        super().__init__(traj, self.indexes, check_aas=check_aas)

    @property
    def name(self) -> str:
        """str: The name of this class: 'AllCartesians'"""
        return "AllCartesians"

    def describe(self) -> list[str]:
        """Returns a list of labels, that can be used to unambiguously define
        atoms in the protein topology.

        Returns:
           list[str]: A list of labels. This list has as many entries as atoms in `self.top`.

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


class CentralCartesians(AllCartesians):
    """Feature that collects all cartesian positions of the backbone atoms.

    Examples:
        >>> import encodermap as em
        >>> traj = em.load_project("pASP_pGLU", 0)
        >>> traj  # doctest: +ELLIPSIS
        encodermap.SingleTraj object...
        >>> feature = em.features.CentralCartesians(traj, generic_labels=False)
        >>> feature.describe()  # doctest: +ELLIPSIS
        ['CENTERPOS X     ATOM     N:    0 ASP:   1 CHAIN 0',
         'CENTERPOS Y     ATOM     N:    0 ASP:   1 CHAIN 0',
         'CENTERPOS Z     ATOM     N:    0 ASP:   1 CHAIN 0',
         'CENTERPOS X     ATOM    CA:    3 ASP:   1 CHAIN 0',
         'CENTERPOS Y     ATOM    CA:    3 ASP:   1 CHAIN 0',
         'CENTERPOS Z     ATOM    CA:    3 ASP:   1 CHAIN 0',
         ...
         'CENTERPOS Z     ATOM     C:   69 ASP:   7 CHAIN 0']
         >>> feature = em.features.CentralCartesians(traj, generic_labels=True)
         >>>feature.describe()  # doctest: +ELLIPSIS
         ['CENTERPOS X 0',
          'CENTERPOS Y 0',
          'CENTERPOS Z 0',
          'CENTERPOS X 1',
          'CENTERPOS Y 1',
          'CENTERPOS Z 1',
          ...
          'CENTERPOS Z 20']

    """

    prefix_label = "CENTERPOS"

    def __init__(
        self,
        traj: SingleTraj,
        generic_labels: bool = False,
        check_aas: bool = True,
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
        super().__init__(self.traj, check_aas=check_aas)
        self.central_indexes = self.top.select("name CA or name C or name N")
        # filter out unwanted indexes
        unwanted_resnames = [k for k, v in _AMINO_ACID_CODES.items() if v is None]
        self.central_indexes = np.array(
            list(
                filter(
                    lambda x: self.top.atom(x).residue.name not in unwanted_resnames,
                    self.central_indexes,
                )
            )
        )
        assert len(self.central_indexes) < len(self.indexes)
        self.indexes = self.central_indexes
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
        for i in range(len(self.central_indexes)):
            for pos in ["X", "Y", "Z"]:
                labels.append(f"{self.prefix_label} {pos} {i}")
        return labels

    def describe(self) -> list[str]:
        """Returns a list of labels, that can be used to unambiguously define
        atoms in the protein topology.

        Returns:
           list[str]: A list of labels. This list has as many entries as atoms in `self.top`.

        """
        getlbl = (
            lambda at: f"ATOM  {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4} CHAIN {at.residue.chain.index}"
        )
        labels = []
        for i in self.central_indexes:
            for pos in ["X", "Y", "Z"]:
                labels.append(
                    f"{self.prefix_label} {pos}     {getlbl(self.top.atom(i))}"
                )
        return labels

    @property
    def name(self) -> str:
        """str: The name of the class: "CentralCartesians"."""
        return "CentralCartesians"

    @property
    def dask_transform(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "selection"


class SideChainCartesians(AllCartesians):
    """Feature that collects all cartesian positions of all non-backbone atoms.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'SIDECHPOS'.

    """

    prefix_label = "SIDECHPOS"

    def __init__(
        self,
        traj: SingleTraj,
        check_aas: bool = True,
    ) -> None:
        self.traj = traj
        super().__init__(self.traj, check_aas=check_aas)
        central_indexes = self.top.select("not backbone")
        assert len(central_indexes) < len(self.indexes)
        self.indexes = central_indexes
        self.dimension = 3 * len(self.indexes)

    @property
    def name(self):
        """str: The name of the class: "SideChainCartesians"."""
        return "SideChainCartesians"


class AllBondDistances(DistanceFeature):
    """Feature that collects all bonds in a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'DISTANCE'.

    """

    prefix_label = "DISTANCE        "

    def __init__(
        self,
        traj: SingleTraj,
        distance_indexes: Optional[np.ndarray] = None,
        periodic: bool = True,
        check_aas: bool = True,
    ) -> None:
        self.distance_indexes = distance_indexes
        if self.distance_indexes is None:
            self.traj = traj
            self.distance_indexes = np.vstack(
                [[b[0].index, b[1].index] for b in self.top.bonds]
            )
            # print(self.distance_indexes, len(self.distance_indexes))
            super().__init__(
                self.traj, self.distance_indexes, periodic, check_aas=check_aas
            )
        else:
            super().__init__(
                self.traj, self.distance_indexes, periodic, check_aas=check_aas
            )
            # print(self.distance_indexes, len(self.distance_indexes))

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        labels = []
        for i in range(len(self.distance_indexes)):
            labels.append(f"{self.prefix_label}{i}")
        return labels

    def describe(self) -> list[str]:
        """Returns a list of labels, that can be used to unambiguously define
        atoms in the protein topology.

        Returns:
           list[str]: A list of labels. This list has as many entries as atoms in `self.top`.

        """
        getlbl = (
            lambda at: f"ATOM  {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4}"
        )
        labels = []
        for i, j in self.distance_indexes:
            i, j = self.top.atom(i), self.top.atom(j)
            labels.append(
                f"{self.prefix_label}{getlbl(i)} DIST  {getlbl(j)} CHAIN {int(np.unique([a.residue.chain.index for a in [i, j]]))}"
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

    prefix_label = "CENTERDISTANCE  "

    def __init__(
        self,
        traj: SingleTraj,
        distance_indexes: Optional[np.ndarray] = None,
        periodic: bool = True,
        generic_labels: bool = False,
        check_aas: bool = True,
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

        super().__init__(self.traj, distance_indexes, periodic, check_aas=check_aas)

    @property
    def name(self) -> str:
        """str: The name of the class: "CentralBondDistances"."""
        return "CentralBondDistances"

    @property
    def indexes(self) -> np.ndarray:
        """np.ndarray: A (n_angles, 2) shaped numpy array giving the atom indices
        of the distances to be calculated."""
        return self.distance_indexes

    @property
    def dask_transform(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "distance"


class SideChainBondDistances(AllBondDistances):
    """Feature that collects all bonds not in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'SIDECHDISTANCE'.

    """

    prefix_label = "SIDECHDISTANCE  "

    def __init__(
        self,
        traj: SingleTraj,
        periodic: bool = True,
        check_aas: bool = True,
    ) -> None:
        self.traj = traj
        # Third Party Imports
        from mdtraj.geometry import dihedral

        which = ["chi1", "chi2", "chi3", "chi4", "chi5"]
        indices_dict = {k: getattr(dihedral, "indices_%s" % k)(top) for k in which}
        flat_list = [
            item
            for sublist in indices_dict.values()
            for item in sublist.flatten().tolist()
        ]
        atoms_in_sidechain_dihedrals = set(flat_list)

        distance_indexes = []
        for angle, indices in indices_dict.items():
            for index in indices:
                if angle == "chi1":
                    distance_indexes.append([index[1], index[2]])
                    distance_indexes.append([index[2], index[3]])
                else:
                    distance_indexes.append([index[2], index[3]])
        distance_indexes = np.sort(distance_indexes, axis=0)
        super().__init__(self.traj, distance_indexes, periodic, check_aas)

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

    prefix_label = "CENTERANGLE "

    def __init__(
        self,
        traj: Union[SingleTraj, TrajEnsemble],
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        generic_labels: bool = False,
        check_aas: bool = True,
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
                    f"Can't deal with these angles. One atom is part of four possible angles"
                )
            else:
                raise Exception(
                    f"Can't deal with these angles. One atom is part of more than three angles"
                )

        angle_indexes = np.vstack(angle_indexes)
        angle_indexes = np.unique(angle_indexes, axis=0)
        if generic_labels:
            self.describe = self.generic_describe
        super().__init__(traj, angle_indexes, deg, cossin, periodic, check_aas)

    def generic_describe(self) -> list[str]:
        """Returns a list of generic labels, not containing residue names.
        These can be used to stack tops of different topology.

        Returns:
            list[str]: A list of labels.

        """
        labels = []
        for i in range(len(self.angle_indexes)):
            labels.append(f"{self.prefix_label}{i}")
        return labels

    def describe(self) -> list[str]:
        """Returns a list of labels, that can be used to unambiguously define
        atoms in the protein topology.

        Returns:
           list[str]: A list of labels. This list has as many entries as atoms in `self.top`.

        """
        getlbl = (
            lambda at: f"ATOM {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4}"
        )
        labels = []
        for i, j, k in self.angle_indexes:
            i, j, k = self.top.atom(i), self.top.atom(j), self.top.atom(k)
            labels.append(
                f"{self.prefix_label}{getlbl(i)} ANGLE {getlbl(j)} ANGLE "
                f"{getlbl(k)} CHAIN "
                f"{int(np.unique([a.residue.chain.index for a in [i, j, k]]))}"
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

    @property
    def dask_transform(self) -> str:
        """str: The name of the delayed transformation to carry out with this feature."""
        return "angle"


class SideChainAngles(AngleFeature):
    """Feature that collects all angles not in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'SIDECHANGLE'.

    """

    prefix_label = "SIDECHANGLE "

    def __init__(
        self,
        traj: SingleTraj,
        deg: bool = False,
        cossin: bool = False,
        periodic: bool = True,
        check_aas: bool = True,
    ) -> None:
        self.traj = traj
        select = self.traj.select(
            "not backbone and (type C or type N or type S or type O) and not type H"
        )
        # add 4 bonds in KAC
        # if any([r.name == "KAC" for r in top.residues]):
        #     self.top = add_KAC_sidechain_bonds(self.top)
        bonds = np.vstack([[x.index for x in b] for b in self.top.bonds])
        angle_indexes = []
        for a in select:
            where = np.where(bonds == a)
            possible_bonds = bonds[where[0], :]
            where = np.isin(possible_bonds, select)
            possible_bonds = possible_bonds[
                np.where(np.all(where, axis=1))[0], :
            ]  # remove atoms not in selection (like hydrogen)
            where = where[
                np.where(np.all(where, axis=1))[0], :
            ]  # remove atoms not in selection (like hydrogen)
            hits = np.count_nonzero(np.all(where, axis=1))
            if hits <= 1:
                continue
            elif hits == 2:
                where_ax = np.all(where, axis=1)
                angle_atoms = np.unique(
                    [
                        self.top.atom(i).index
                        for i in possible_bonds[where_ax, :].flatten()
                    ]
                )
                assert len(angle_atoms) == 3, print(
                    [(i, self.top.atom(i)) for i in angle_atoms]
                )
                angle_indexes.append(angle_atoms)
            elif hits == 3:
                where_ax = np.vstack([where[:-1], [False, False]])
                where_ax = np.all(where_ax, axis=1)
                angle_atoms = np.unique(
                    [
                        self.top.atom(i).index
                        for i in possible_bonds[where_ax, :].flatten()
                    ]
                )
                assert len(angle_atoms) == 3, print(
                    [(i, self.top.atom(i)) for i in angle_atoms]
                )
                angle_indexes.append(angle_atoms)
                where_ax = np.vstack([[False, False], where[1:]])
                where_ax = np.all(where_ax, axis=1)
                angle_atoms = np.unique(
                    [
                        self.top.atom(i).index
                        for i in possible_bonds[where_ax, :].flatten()
                    ]
                )
                assert len(angle_atoms) == 3, print(
                    [(i, self.top.atom(i)) for i in angle_atoms]
                )
                angle_indexes.append(angle_atoms)
            elif hits == 4:
                raise Exception(
                    f"Can't deal with these angles. One atom is part of four possible angles"
                )
            else:
                raise Exception(
                    f"Can't deal with these angles. One atom is part of three possible angles"
                )
        angle_indexes = np.vstack(angle_indexes)
        super().__init__(self.traj, angle_indexes, deg, cossin, periodic, check_aas)

    def describe(self):
        """Returns a list of labels, that can be used to unambiguously define
        atoms in the protein topology.

        Returns:
           list[str]: A list of labels. This list has as many entries as atoms in `self.top`.

        """
        getlbl = (
            lambda at: f"ATOM {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4}"
        )
        labels = []
        for i, j, k in self.angle_indexes:
            i, j, k = self.top.atom(i), self.top.atom(j), self.top.atom(k)
            labels.append(
                f"{self.prefix_label}{getlbl(i)} ANGLE {getlbl(j)} ANGLE {getlbl(k)} CHAIN {int(np.unique([a.residue.chain.index for a in [i, j, k]]))}"
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

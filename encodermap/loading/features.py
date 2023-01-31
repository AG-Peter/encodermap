# -*- coding: utf-8 -*-
# encodermap/loading/features.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2022 University of Konstanz and the Authors
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
"""Classes to be used as custom features with pyemma add_custom_feature

ToDo:
    * Write tests
    * Put the describe_last_feats function into utils.
    * Add Nan feature.
    * Write Examples.

"""

##############################################################################
# Imports
##############################################################################


from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Callable

import numpy as np

import encodermap

from .._optional_imports import _optional_import

##############################################################################
# Typing
##############################################################################


if TYPE_CHECKING:
    from mdtraj.core.residue_names import _AMINO_ACID_CODES
    from pyemma.coordinates.data.featurization.angles import (
        AngleFeature,
        DihedralFeature,
    )
    from pyemma.coordinates.data.featurization.distances import DistanceFeature
    from pyemma.coordinates.data.featurization.misc import (
        CustomFeature,
        SelectionFeature,
    )

    from encodermap._typing import AnyFeature


##############################################################################
# Optional Imports
##############################################################################


CustomFeature: CustomFeature = _optional_import(
    "pyemma", "coordinates.data.featurization.misc.CustomFeature"
)
SelectionFeature: SelectionFeature = _optional_import(
    "pyemma", "coordinates.data.featurization.misc.SelectionFeature"
)
DihedralFeature: DihedralFeature = _optional_import(
    "pyemma", "coordinates.data.featurization.angles.DihedralFeature"
)
AngleFeature: AngleFeature = _optional_import(
    "pyemma", "coordinates.data.featurization.angles.AngleFeature"
)
DistanceFeature: DistanceFeature = _optional_import(
    "pyemma", "coordinates.data.featurization.distances.DistanceFeature"
)
_AMINO_ACID_CODES: dict = _optional_import(
    "mdtraj", "core.residue_names._AMINO_ACID_CODES"
)
indices_phi: Callable = _optional_import("mdtraj", "geometry.dihedral.indices_phi")
indices_psi: Callable = _optional_import("mdtraj", "geometry.dihedral.indices_psi")
indices_omega: Callable = _optional_import("mdtraj", "geometry.dihedral.indices_omega")
indices_chi1: Callable = _optional_import("mdtraj", "geometry.dihedral.indices_chi1")
indices_chi2: Callable = _optional_import("mdtraj", "geometry.dihedral.indices_chi2")
indices_chi3: Callable = _optional_import("mdtraj", "geometry.dihedral.indices_chi3")
indices_chi4: Callable = _optional_import("mdtraj", "geometry.dihedral.indices_chi4")
indices_chi5: Callable = _optional_import("mdtraj", "geometry.dihedral.indices_chi5")


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
]

##############################################################################
# Functions
##############################################################################


def describe_last_feats(feat: AnyFeature, n: int = 5) -> None:
    """Prints the description of the last `n` features.

    Args:
        feat (encodermap.Featurizer): An instance of a featurizer.
        n (Optional[int]): The number of last features to decribe. Defaults to 5.

    """
    for i, lbl in enumerate(feat.describe()[-n:]):
        print(lbl)


def add_KAC_backbone_bonds(top):
    """Adds acetylated Lysine specific backbone bonds to mdtraj.Topology.

    Args:
        top (mdtraj.Topology): The topology to be extended.

    Returns:
        mdtraj.Topology: The new topology with added bonds.

    Note:
        The bonds are currently not at the correct index, i.e. they are
        at the very end of top.bonds and not at the correct position.

    """
    # for index, bond in enumerate(top.bonds):
    #     if any([a.residue.name == 'KAC' for a in bond]):
    #         break
    resid_KAC = top.atom(top.select("resname KAC")[0]).residue.index
    # add C - N(KAC)
    bond = (
        top.select(f"name C and resid {resid_KAC - 1}")[0],
        top.select(f"name N and resid {resid_KAC}")[0],
    )
    bond = [top.atom(b) for b in bond]
    top.add_bond(*bond)
    # add N (KAC) - CA(KAC)
    bond = (
        top.select(f"name N and resid {resid_KAC}")[0],
        top.select(f"name CA and resid {resid_KAC}")[0],
    )
    bond = [top.atom(b) for b in bond]
    top.add_bond(*bond)
    # add CA (KAC) - C(KAC)
    bond = (
        top.select(f"name CA and resid {resid_KAC}")[0],
        top.select(f"name C and resid {resid_KAC}")[0],
    )
    bond = [top.atom(b) for b in bond]
    top.add_bond(*bond)
    return top


def add_KAC_sidechain_bonds(top):
    """Adds acetylated Lysine specific side chain bonds to mdtraj.Topology. Bonds between
    indented atoms are added:
    KAC11-N 102
    KAC11-H 103
            KAC11-CA 104
            KAC11-CB 105
            KAC11-CG 106
            KAC11-CD 107
            KAC11-CE 108
            KAC11-NZ 109
    KAC11-HZ 110
    KAC11-CH 111
    KAC11-OI2 112
    KAC11-CI1 113
    KAC11-C 114
    KAC11-O 115

    Args:
        top (mdtraj.Topology): The topology to be extended.

    Returns:
        mdtraj.Topology: The new topology with added bonds.

    Note:
        The bonds are currently not at the correct index, i.e. they are
        at the very end of top.bonds and not at the correct position.
    """
    # for r in top.residues:
    #   if r.name == 'KAC':
    #     for a in r.atoms:
    #       print(a, a.index)
    # print(a.__dir__())
    resid_KAC = top.atom(top.select("resname KAC")[0]).residue.index
    # add CA - CB(KAC)
    bond = (
        top.select(f"name CA and resid {resid_KAC - 1}")[0],
        top.select(f"name CB and resid {resid_KAC}")[0],
    )
    bond = [top.atom(b) for b in bond]
    top.add_bond(*bond)
    # add CB (KAC) - CG(KAC)
    bond = (
        top.select(f"name CB and resid {resid_KAC}")[0],
        top.select(f"name CG and resid {resid_KAC}")[0],
    )
    bond = [top.atom(b) for b in bond]
    top.add_bond(*bond)
    # add CG (KAC) - CE(KAC)
    bond = (
        top.select(f"name CG and resid {resid_KAC}")[0],
        top.select(f"name CE and resid {resid_KAC}")[0],
    )
    bond = [top.atom(b) for b in bond]
    top.add_bond(*bond)
    # add CE (KAC) - NZ(KAC)
    bond = (
        top.select(f"name CE and resid {resid_KAC}")[0],
        top.select(f"name NZ and resid {resid_KAC}")[0],
    )
    bond = [top.atom(b) for b in bond]
    top.add_bond(*bond)
    return top


##############################################################################
# Classes
##############################################################################


class CentralDihedrals(DihedralFeature):
    """Feature that collects all dihedrals in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.

    """

    __serialize_version = 0
    __serialize_fields = ("selstr", "_phi_inds", "_psi_inds", "_omega_inds")

    def __init__(
        self,
        topology,
        selstr=None,
        deg=False,
        cossin=False,
        periodic=True,
        omega=True,
        generic_labels=False,
    ):
        """Instantiate this feature class.

        Args:
            topology (mdtraj.Topology): A topology to build features from.
            selstr (Optional[str]): A string, that limits the selection of dihedral angles.
                Only dihedral angles which atoms are represented by the `selstr` argument
                are considered. This selection string follows MDTraj's atom selection
                language: https://mdtraj.org/1.9.3/atom_selection.html. Can also
                be None, in which case all backbone dihedrals (also omega) are
                considered. Defaults to None.
            deg (bool): Whether to return the result in degree (`deg=True`) or in
                radians (`deg=False`). Defaults to radions.
            cossin (bool):  If True, each angle will be returned as a pair of
                (sin(x), cos(x)). This is useful, if you calculate the mean
                (e.g TICA/PCA, clustering) in that space. Defaults to False.
            periodic (bool): Whether to recognize periodic boundary conditions and
                work under the minimum image convention. Defaults to True.

        """
        self.top = topology
        self.selstr = selstr

        indices = indices_psi(self.top)
        if not selstr:
            self._psi_inds = indices
        else:
            self._psi_inds = indices[
                np.in1d(indices[:, 1], self.top.select(selstr), assume_unique=True)
            ]

        self.omega = omega
        if self.omega:
            indices = indices_omega(self.top)
            if not selstr:
                self._omega_inds = indices
            else:
                self._omega_inds = indices[
                    np.in1d(indices[:, 1], self.top.select(selstr), assume_unique=True)
                ]

        indices = indices_phi(self.top)
        if not selstr:
            self._phi_inds = indices
        else:
            self._phi_inds = indices[
                np.in1d(indices[:, 1], self.top.select(selstr), assume_unique=True)
            ]

        if self.omega:
            zipped = zip(self._psi_inds, self._omega_inds, self._phi_inds)
        else:
            zipped = zip(self._psi_inds, self._phi_inds)

        # alternate phi, psi , omega pairs (phi_1, psi_1, omega_1..., phi_n, psi_n, omega_n)
        dih_indexes = np.array(list(psi_omega_phi for psi_omega_phi in zipped)).reshape(
            -1, 4
        )

        # set generic_labels for xarray
        if generic_labels:
            self.describe = self.generic_describe

        super(CentralDihedrals, self).__init__(
            self.top, dih_indexes, deg=deg, cossin=cossin, periodic=periodic
        )

    @property
    def name(self):
        """str: The name of the class: "CentralDihedrals"."""
        return "CentralDihedrals"

    @property
    def indexes(self):
        """np.ndarray: A (n_angles, 4) shaped numpy array giving the atom indices
        of the dihedral angles to be calculated."""
        return self.angle_indexes.astype("int32")

    def generic_describe(self):
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
            labels_psi = [
                f"CENTERDIH PSI    %s" % i for i in range(len(self._psi_inds))
            ]
            if self.omega:
                labels_omega = [
                    "CENTERDIH OMEGA  %s" % i for i in range(len(self._omega_inds))
                ]
            labels_phi = ["CENTERDIH PHI    %s" % i for i in range(len(self._phi_inds))]
            if self.omega:
                zipped = zip(labels_psi, labels_omega, labels_phi)
            else:
                zipped = zip(labels_psi, labels_phi)
            res = list(itertools.chain.from_iterable(zipped))
        return res

    def describe(self):
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
    def dask_transform(self):
        return "dihedral"


class SideChainDihedrals(DihedralFeature):
    """Feature that collects all dihedrals in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        options (list[str]): A list of possible sidechain angles ['chi1' to 'chi5'].

    """

    __serialize_version: int = 0
    __serialize_fields: tuple[str] = ("_prefix_label_lengths",)
    options: list[str] = ["chi1", "chi2", "chi3", "chi4", "chi5"]

    def __init__(
        self,
        top,
        selstr=None,
        deg=False,
        cossin=False,
        periodic=True,
        generic_labels=False,
    ):
        which = self.options
        # get all dihedral index pairs
        from mdtraj.geometry import dihedral

        indices_dict = {k: getattr(dihedral, "indices_%s" % k)(top) for k in which}
        if selstr:
            selection = top.select(selstr)
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

        # for key in indices_dict:
        #     print(key, indices_dict[key])
        # for proteins that don't have some chi angles we filter which
        which = list(
            filter(
                lambda x: True if len(indices_dict[x]) > 0 else False,
                indices_dict.keys(),
            )
        )

        # change the sorting to be per-residue and not all chi1 and then all chi2 angles
        self.per_res_dict = {}
        for r in top.residues:
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
            top=top, dih_indexes=indices, deg=deg, cossin=cossin, periodic=periodic
        )

        if generic_labels:
            self.describe = self.generic_describe

    @property
    def name(self):
        """str: The name of the class: "SideChainDihedrals"."""
        return "SideChainDihedrals"

    @property
    def indexes(self):
        """np.ndarray: A (n_angles, 4) shaped numpy array giving the atom indices
        of the dihedral angles to be calculated."""
        return self.angle_indexes

    def generic_describe(self):
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

    def describe(self):
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
    """Feature that collects all cartesian position of all atoms in the trajectory.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'POSITION'.

    """

    __serialize_version = 0
    __serialize_fields = ("indexes",)
    prefix_label = "POSITION "

    def __init__(self, top):
        """Instantiate the AllCartesians class.

        Args:
            top (mdtraj.Topology): A mdtraj topology.

        """
        self.top = top
        self.indexes = self.top.select("all")
        super().__init__(top, self.indexes)

    @property
    def name(self):
        """str: The name of this class: 'AllCartesians'"""
        return "AllCartesians"

    def describe(self):
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
    """Feature that collects all cartesian position of the backbone atoms.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'CENTERPOS'.

    """

    __serialize_version = 0
    __serialize_fields = ("indexes",)
    prefix_label = "CENTERPOS"

    def __init__(self, top, generic_labels=False):
        self.top = top
        super().__init__(self.top)
        self.central_indexes = self.top.select("name CA or name C or name N")
        assert len(self.central_indexes) < len(self.indexes)
        self.indexes = self.central_indexes
        self.dimension = 3 * len(self.indexes)

        if generic_labels:
            self.describe = self.generic_describe

    def generic_describe(self):
        labels = []
        for i in range(len(self.central_indexes)):
            for pos in ["X", "Y", "Z"]:
                labels.append(f"{self.prefix_label} {pos} {i}")
        return labels

    def describe(self):
        """Returns a list of labels, that can be used to unambiguously define
        atoms in the protein topology.

        Returns:
           list[str]: A list of labels. This list has as manyu entries as atoms in `self.top`.

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
    def name(self):
        """str: The name of the class: "CentralCartesians"."""
        return "CentralCartesians"

    # def transform(self, traj):
    #     newshape = (traj.xyz.shape[0], 3 * self.central_indexes.shape[0])
    #     return np.reshape(traj.xyz[:, self.central_indexes, :], newshape)


class SideChainCartesians(AllCartesians):
    """Feature that collects all cartesian position of all non-backbone atoms.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'SIDECHPOS'.

    """

    __serialize_version = 0
    __serialize_fields = ("indexes",)
    prefix_label = "SIDECHPOS"

    def __init__(self, top):
        self.top = top
        super().__init__(self.top)
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

    __serialize_version = 0
    __serialize_fields = ("distance_indexes", "periodic")
    prefix_label = "DISTANCE        "

    def __init__(self, top, distance_indexes=None, periodic=True, check_aas=True):
        self.distance_indexes = distance_indexes
        if any([r.name not in _AMINO_ACID_CODES for r in top.residues]) and check_aas:
            raise Exception("Unkown amino acid in top.")
        if self.distance_indexes is None:
            self.top = top
            self.distance_indexes = np.vstack(
                [[b[0].index, b[1].index] for b in self.top.bonds]
            )
            # print(self.distance_indexes, len(self.distance_indexes))
            super().__init__(self.top, self.distance_indexes, periodic)
        else:
            super().__init__(self.top, self.distance_indexes, periodic)
            # print(self.distance_indexes, len(self.distance_indexes))

    def generic_describe(self):
        labels = []
        for i in range(len(self.distance_indexes)):
            labels.append(f"{self.prefix_label}{i}")
        return labels

    def describe(self):
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
    def name(self):
        """str: The name of the class: "AllBondDistances"."""
        return "AllBondDistances"

    @property
    def indexes(self):
        """np.ndarray: A (n_angles, 2) shaped numpy array giving the atom indices
        of the distances to be calculated."""
        return self.distance_indexes


class CentralBondDistances(AllBondDistances):
    """Feature that collects all bonds in the backbone of a topology.

    Attributes:
        top (mdtraj.Topology): Topology of this feature.
        indexes (np.ndarray): The numpy array returned from `top.select('all')`.
        prefix_label (str): A prefix for the labels. In this case it is 'CENTERDISTANCE'.

    """

    __serialize_version = 0
    __serialize_fields = ("distance_indexes", "periodic")
    prefix_label = "CENTERDISTANCE  "

    def __init__(
        self,
        top,
        distance_indexes=None,
        periodic=True,
        check_aas=True,
        generic_labels=False,
    ):
        self.top = copy.deepcopy(top)
        if any([r.name == "KAC" for r in top.residues]):
            self.top = add_KAC_backbone_bonds(self.top)
            check_aas = False
        select = self.top.select("name CA or name C or name N")

        #         temp_list = []
        #         for i in range(len(select)-1):
        #             temp_list.append([select[i], select[i+1]])
        #         temp_array = np.array(temp_list)
        #         print("this is the array of the selected atoms:", temp_array)
        #         print(len(temp_array))

        if distance_indexes is None:
            distance_indexes = []

        for b in self.top.bonds:
            # print(b)
            if np.all([np.isin(x.index, select) for x in b]):
                distance_indexes.append([x.index for x in b])
        distance_indexes = np.sort(distance_indexes, axis=0)

        if generic_labels:
            self.describe = self.generic_describe

        super().__init__(
            self.top, distance_indexes, periodic, check_aas=check_aas
        )  # distance_indexes

    @property
    def name(self):
        """str: The name of the class: "CentralBondDistances"."""
        return "CentralBondDistances"

    @property
    def indexes(self):
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

    __serialize_version = 0
    __serialize_fields = ("distance_indexes", "periodic")
    prefix_label = "SIDECHDISTANCE  "

    def __init__(self, top, periodic=True):
        self.top = top
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
        super().__init__(self.top, distance_indexes, periodic)

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

    __serialize_version = 0
    __serialize_fields = ("angle_indexes", "deg", "cossin", "periodic")
    prefix_label = "CENTERANGLE "

    def __init__(
        self, top, deg=False, cossin=False, periodic=True, generic_labels=False
    ):
        self.top = copy.deepcopy(top)
        select = self.top.select("name CA or name C or name N")
        # add 4 bonds in KAC
        if any([r.name == "KAC" for r in top.residues]):
            self.top = add_KAC_backbone_bonds(self.top)
        bonds = np.vstack([[x.index for x in b] for b in self.top.bonds])
        angle_indexes = []
        for a in select:
            where = np.where(bonds == a)
            possible_bonds = bonds[where[0], :]
            where = np.isin(possible_bonds, select)
            hits = np.count_nonzero(np.all(where, axis=1))
            if hits <= 1:
                continue
            elif hits == 2:
                where = np.all(where, axis=1)
                these = np.unique(
                    [self.top.atom(i).index for i in possible_bonds[where, :].flatten()]
                )
                angle_indexes.append(these)
            elif hits == 3:
                raise Exception(
                    f"Can't deal with these angles. One atom is part of three possible angles"
                )
            elif hits == 4:
                raise Exception(
                    f"Can't deal with these angles. One atom is part of four possible angles"
                )
            else:
                raise Exception(
                    f"Can't deal with these angles. One atom is part of three possible angles"
                )

        angle_indexes = np.vstack(angle_indexes)
        angle_indexes = np.unique(angle_indexes, axis=0)
        if generic_labels:
            self.describe = self.generic_describe
        super().__init__(self.top, angle_indexes, deg, cossin, periodic)

    def generic_describe(self):
        labels = []
        for i in range(len(self.angle_indexes)):
            labels.append(f"{self.prefix_label}{i}")
        return labels

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
        """str: The name of the class: "CentralAngles"."""
        return "CentralAngles"

    @property
    def indexes(self):
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

    __serialize_version = 0
    __serialize_fields = ("angle_indexes", "deg", "cossin", "periodic")
    prefix_label = "SIDECHANGLE "

    def __init__(self, top, deg=False, cossin=False, periodic=True):
        self.top = copy.deepcopy(top)
        select = self.top.select(
            "not backbone and (type C or type N or type S or type O) and not type H"
        )
        # add 4 bonds in KAC
        if any([r.name == "KAC" for r in top.residues]):
            self.top = add_KAC_sidechain_bonds(self.top)
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
        super().__init__(self.top, angle_indexes, deg, cossin, periodic)

    def describe(self):
        """Returns a list of labels, that can be used to unambiguously define
        atoms in the protein topology.

        Retruns:
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

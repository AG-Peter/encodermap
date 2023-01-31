# -*- coding: utf-8 -*-
# encodermap/loading/featurizer.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2022 University of Konstanz and the Authors
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
"""Classes to be used as custom features with pyemma add_custom_feature

ToDo:
    * Write Docstrings.
    * Write Examples.
    * Sidechain angles, distances not working correctly.
"""


################################################################################
# Imports
################################################################################


from __future__ import annotations

import numpy as np
import pandas as pd

from .._optional_imports import _optional_import
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
xr = _optional_import("xarray")
CHI1_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI1_ATOMS")
CHI2_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI2_ATOMS")
CHI3_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI3_ATOMS")
CHI4_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI4_ATOMS")
CHI5_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI5_ATOMS")
Client = _optional_import("dask", "distributed.Client")


################################################################################
# Typing
################################################################################


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr
    from dask.distributed import Client
    from mdtraj.geometry.dihedral import (
        CHI1_ATOMS,
        CHI2_ATOMS,
        CHI3_ATOMS,
        CHI4_ATOMS,
        CHI5_ATOMS,
    )
    from pyemma.coordinates import featurizer, source
    from pyemma.coordinates.data._base.datasource import DataSource
    from pyemma.coordinates.data.featurization._base import Feature


################################################################################
# Globals
################################################################################


__all__ = ["Featurizer"]


UNDERSOCRE_MAPPING = {
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


##############################################################################
# Utils
##############################################################################

##############################################################################
# Classes
##############################################################################


class Featurizer(type):
    def __new__(cls, trajs, in_memory=True):
        if in_memory:
            cls = PyEMMAFeaturizer
        else:
            from .dask_featurizer import DaskFeaturizer

            cls = DaskFeaturizer
        return cls(trajs)


def format_output(
    inps: list[DataSource],
    feats: list[Feature],
    trajs: list[TrajEnsemble],
) -> tuple[list[np.ndarray], list[PyEMMAFeaturizer], list[TrajEnsemble]]:
    """Formats the output of multiple topologies.

    Iterates over the features in `feats` and looks for the feature
    with the greatest dimension, i.e. the longest returned describe. This
    feature yields the column names, the non-defined values are np.nan

    Args:
        inps (list[DataSource]): The list of inputs, that
            return the values of the feats, when `get_output()` is called.
        feats (list[encodermap.loading.Featurizer]: These featurizers collect the
            features and will be used to determine the highest length of feats.
        trajs (list[encodermap.trajinfo.TrajEnsemble]): List of trajs with
            identical topologies.

    Returns:
        tuple[list[np.ndarray], list[Featurizer], list[TrajEnsembe]: The
            data, that `TrajEnsemble` can work with.

    """

    class Featurizer_out:
        pass

    # append to this
    all_out = []

    feat_out = Featurizer_out()
    feat_out.features = []
    max_feat_lengths = {}
    labels = {}
    for feat in feats:
        for i, f in enumerate(feat.feat.active_features):
            name = f.__class__.__name__

            if name not in max_feat_lengths:
                max_feat_lengths[name] = 0
                feat_out.features.append(
                    EmptyFeature(name, len(f.describe()), f.describe(), f.indexes)
                )

            if name == "SideChainDihedrals":
                if name not in labels:
                    labels[name] = []
                labels[name].extend(f.describe())
            else:
                if max_feat_lengths[name] < len(f.describe()):
                    max_feat_lengths[name] = len(f.describe())
                    labels[name] = f.describe()
                    feat_out.features[i] = EmptyFeature(
                        name, len(f.describe()), f.describe(), f.indexes
                    )

    # rejig the sidechain labels
    side_key = "SideChainDihedrals"
    if side_key in labels:
        labels[side_key] = np.unique(labels[side_key])
        labels[side_key] = sorted(
            labels[side_key], key=lambda x: (int(x[-3:]), int(x[13]))
        )
        index_of_sidechain_dihedral_features = [
            f.name == side_key for f in feat_out.features
        ].index(True)
        new_empty_feat = EmptyFeature(
            side_key,
            len(labels[side_key]),
            labels[side_key],
            None,
        )
        feat_out.features[index_of_sidechain_dihedral_features] = new_empty_feat

    for (k, v), f in zip(labels.items(), feat_out.features):
        if not len(v) == len(f.describe()) == f._dim:
            raise Exception(
                f"Could not consolidate the features of the {f.name} "
                f"feature. The `labels` dict, which dictates the size "
                f"of the resulting array with np.nan's defines a shape "
                f"of {len(v)}, but the feature defines a shape of {len(f.describe())} "
                f"(or `f._dim = {f._dim}`). The labels dict gives these labels:\n\n{v}"
                f"\n\n, the feature labels gives these labels:\n\n{f.describe()}."
            )

    # flatten the labels. These will be the columns for a pandas dataframe.
    # At the start the dataframe will be full of np.nan.
    # The values of inp.get_output() will then be used in conjunction with
    # The labels of the features to fill this dataframe partially
    flat_labels = [item for sublist in labels.values() for item in sublist]
    if not len(flat_labels) == sum([f._dim for f in feat_out.features]):
        raise Exception(
            f"The length of the generic CV labels ({len(flat_labels)} "
            f"does not match the length of the labels of the generic features "
            f"({[f._dim for f in feat_out.features]})."
        )

    # iterate over the sorted trajs, inps, and feats
    for inp, feat, sub_trajs in zip(inps, feats, trajs):
        # make a flat list for this specific feature space
        describe_this_feature = []
        for f in feat.feat.active_features:
            # make sure generic labels are used
            if f.describe.__func__.__name__ != "generic_describe":
                raise Exception(
                    f"It seems like this feature: {f.__class__} does not return generic "
                    f"feature names but topology-specifc ones (generic: 'SIDECHDIH CHI1 1', "
                    f"topology specific: 'SIDECHDIH CHI1 ASP1'). Normally, encodermap's "
                    f"features can be instantiated with a `generic_labels=True` flag to "
                    f"overwrite the features `describe()` method with a `generic_describe()` "
                    f"method. This changes the `.__func__.__name__` of the `describe()` method "
                    f"to 'generic_describe'. However the func name for this feature is "
                    f"{f.describe.__func__.__name__}."
                )
            describe_this_feature.extend(f.describe())
        # use the output to fill a pandas dataframe with all labels
        out = np.vstack(inp.get_output())
        for o, traj in zip(out, sub_trajs):
            df = pd.DataFrame(np.nan, index=range(len(out)), columns=flat_labels)
            df = df.assign(**{k: v for k, v in zip(describe_this_feature, out.T)})
            all_out.append((df.to_numpy(), feat_out, traj))

    # make sure the sapes of all df matches
    shapes = [o[0].shape[1] for o in all_out]
    if not len(list(set(shapes))) == 1:
        raise Exception(
            f"Alignment was not possible. Some values exhibit different shapes: "
            f"{list(set(shapes))}. All shapes:\n\n{[o[0].shape[1] for o in all_out]}"
        )
    return all_out


class PyEMMAFeaturizer:
    def __init__(self, trajs):
        self.trajs = trajs
        self._copy_docstrings_from_pyemma()

    def _copy_docstrings_from_pyemma(self):
        if isinstance(self.feat, list):
            feat_ = self.feat[0]
        else:
            feat_ = self.feat
        self.add_all.__func__.__doc__ = feat_.add_all.__doc__
        self.add_selection.__func__.__doc__ = feat_.add_selection.__doc__
        self.add_distances.__func__.__doc__ = feat_.add_distances.__doc__
        self.add_distances_ca.__func__.__doc__ = feat_.add_distances_ca.__doc__
        self.add_inverse_distances.__func__.__doc__ = (
            feat_.add_inverse_distances.__doc__
        )
        self.add_contacts.__func__.__doc__ = feat_.add_contacts.__doc__
        self.add_residue_mindist.__func__.__doc__ = feat_.add_residue_mindist.__doc__
        self.add_group_COM.__func__.__doc__ = feat_.add_group_COM.__doc__
        self.add_residue_COM.__func__.__doc__ = feat_.add_residue_COM.__doc__
        self.add_group_mindist.__func__.__doc__ = feat_.add_group_mindist.__doc__
        self.add_angles.__func__.__doc__ = feat_.add_angles.__doc__
        self.add_dihedrals.__func__.__doc__ = feat_.add_dihedrals.__doc__
        self.add_backbone_torsions.__func__.__doc__ = (
            feat_.add_backbone_torsions.__doc__
        )
        self.add_chi1_torsions.__func__.__doc__ = feat_.add_chi1_torsions.__doc__
        self.add_sidechain_torsions.__func__.__doc__ = (
            feat_.add_sidechain_torsions.__doc__
        )
        self.add_minrmsd_to_ref.__func__.__doc__ = feat_.add_minrmsd_to_ref.__doc__

    def get_output(self) -> xr.Dataset:
        if self.mode == "single_top":
            if len(self.feat.active_features) == 0:
                print("No features loaded. No output will be returned")
                return

        if self.mode == "multiple_top":
            if len(self.feat[0].features) == 0:
                print("No features loaded. No output will be returned")
                return

        if self.mode == "single_top":
            datasets = []
            out = self.inp.get_output()
            for traj, out in zip(self.trajs, out):
                datasets.append(unpack_data_and_feature(self, traj, out))

            if len(datasets) == 1:
                assert datasets[0].coords["traj_num"] == np.array(
                    [self.trajs[0].traj_num]
                )
                return datasets[0]
            else:
                out = xr.combine_nested(datasets, concat_dim="traj_num")
                if (
                    len(out.coords["traj_num"]) != len(self.trajs)
                    and len(out.coords["traj_num"]) != self.trajs.n_trajs
                ):
                    raise Exception(
                        f"The combineNnested xarray method returned "
                        f"a bad dataset, which has {out.coords['traj_num']} "
                        f"trajectories, but the featurizer has {self.trajs} "
                        f"trajectories."
                    )
                # out = xr.concat(datasets, dim='traj_num')
        else:
            out = format_output(self.inp, self.feat, self.sorted_trajs)
            datasets = [unpack_data_and_feature(o[1], o[2], o[0]) for o in out]
            out = xr.concat(datasets, dim="traj_num")

        return out

    def add_list_of_feats(self, which="all"):
        """Adds features to the Featurizer to be loaded either in-memory or out-of-memory.
        `which` can be either 'all' or a list of the following strings. 'all' will add all of these features:
        * 'AllCartesians': Cartesian coordinates of all atoms with shape (n_frames, n_atoms, 3).
        * 'AllBondDistances': Bond distances of all bonds recognized by mdtraj. Use top = md.Topology.from_openmm()
            if mdtraj does not recognize all bonds.
        * 'CentralCartesians': Cartesians of the N, C, CA atoms in the backbone with shape (n_frames, n_residues * 3, 3).
        * 'CentralBondDistances': The bond distances of the N, C, CA bonds with shape (n_frames, n_residues * 3 - 1).
        * 'CentralAngles': The angles between the backbone bonds with shape (n_frames, n_residues * 3 - 2).
        * 'CentralDihedrals': The dihedrals between the backbone atoms (omega, phi, psi). With shape (n_frames,
            n_residues * 3 - 3).
        * 'SideChainCartesians': Cartesians of the sidechain-atoms. Starting with CB, CG, ...
        * 'SideChainBondDistances': Bond distances between the sidechain atoms. starting with the CA-CG bond.
        * 'SideChainAngles': Angles between sidechain atoms. Starting with the C-CA-CB angle.
        * 'SideChainDihedrals': Dihedrals of the sidechains (chi1, chi2, chi3).

        Args:
            which (Union[str, list], optional). Either add 'all' features or a list of features. See Above for
                possible features. Defaults to 'all'.

        """
        if isinstance(which, str):
            if which == "all":
                which = [
                    "CentralCartesians",
                    "CentralBondDistances",
                    "CentralAngles",
                    "CentralDihedrals",
                    "SideChainDihedrals",
                ]
        if not isinstance(which, list):
            which = [which]
        if self.mode == "single_top":
            for cf in which:
                if cf in UNDERSOCRE_MAPPING:
                    cf = UNDERSOCRE_MAPPING[cf]
                feature = getattr(features, cf)(self.top)
                self.feat.add_custom_feature(feature)
        else:
            for cf in which:
                if cf in UNDERSOCRE_MAPPING:
                    cf = UNDERSOCRE_MAPPING[cf]
                for top, feat in zip(self.top, self.feat):
                    feature = getattr(features, cf)(top, generic_labels=True)
                    feat.add_custom_feature(feature)

    def add_all(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_all(*args, **kwargs)

    def add_selection(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_selection(*args, **kwargs)

    def add_distances(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_distances(*args, **kwargs)

    def add_distances_ca(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_distances_ca(*args, **kwargs)

    def add_inverse_distances(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_inverse_distances(*args, **kwargs)

    def add_contacts(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_contacts(*args, **kwargs)

    def add_residue_mindist(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_residue_mindist(*args, **kwargs)

    def add_group_COM(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_group_COM(*args, **kwargs)

    def add_residue_COM(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_residue_COM(*args, **kwargs)

    def add_group_mindist(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_group_mindist(*args, **kwargs)

    def add_angles(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_angles(*args, **kwargs)

    def add_dihedrals(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_dihedrals(*args, **kwargs)

    def add_backbone_torsions(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_backbone_torsions(*args, **kwargs)

    def add_chi1_torsions(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_sidechain_torsions(which=["chi1"], *args, **kwargs)

    def add_sidechain_torsions(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_sidechain_torsions(*args, **kwargs)

    def add_minrmsd_to_ref(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_minrmsd_to_ref(*args, **kwargs)

    def add_custom_feature(self, feature):
        self.feat.add_custom_feature(feature)

    @property
    def features(self):
        if self.mode == "single_top":
            return self.feat.active_features
        else:
            return [f.features for f in self.feat]

    @property
    def sorted_info_single(self):
        if self.mode == "single_top":
            raise Exception(
                "Attribute is only accessible, when working with mutliple topologies."
            )
        out = []
        for info_all in self.sorted_trajs:
            for traj in info_all:
                out.append(traj)
        return out

    @property
    def sorted_featurizers(self):
        if self.mode == "single_top":
            raise Exception(
                "Attribute is only accessible, when working with mutliple topologies."
            )
        out = []
        for feat, info_all in zip(self.feat, self.sorted_trajs):
            out.extend([feat for i in range(info_all.n_trajs)])
        return out

    @property
    def trajs(self):
        return self._trajs

    def describe(self):
        return self.feat.describe()

    @trajs.setter
    def trajs(self, trajs):
        if isinstance(trajs, SingleTraj) or trajs.__class__.__name__ == "SingleTraj":
            self._trajs = trajs._gen_ensemble()
            self.top = trajs.top
            self.feat = featurizer(self.top)
            if _validate_uri(trajs.traj_file):
                self.inp = source([trajs.xyz], features=self.feat)
            else:
                try:
                    self.inp = source([trajs.traj_file], features=self.feat)
                except Exception:
                    print(trajs.traj_file)
                    print(trajs.top_file)
                    raise
            self.mode = "single_top"
        elif (
            isinstance(trajs, TrajEnsemble)
            or trajs.__class__.__name__ == "TrajEnsemble"
        ):
            if len(trajs.top) > 1:
                self._trajs = trajs
                # self.top = Topologies(trajs.top)
                self.top = trajs.top
                self.sorted_trajs = []
                for top in trajs.top:
                    matching_trajs = list(
                        filter(lambda x: True if x.top == top else False, trajs)
                    )
                    self.sorted_trajs.append(TrajEnsemble(matching_trajs))
                self.feat = [Featurizer(t) for t in self.sorted_trajs]
                self.inp = [
                    source([t.traj_file for t in t_subset], features=feat.feat)
                    for t_subset, feat in zip(self.sorted_trajs, self.feat)
                ]
                self.mode = "multiple_top"
            else:
                self._trajs = trajs
                self.top = trajs.top[0]
                self.feat = featurizer(self.top)
                if all([_validate_uri(traj.traj_file) for traj in trajs]):
                    self.inp = source(trajs.xtc, features=self.feat)
                else:
                    self.inp = source(
                        [traj.traj_file for traj in trajs], features=self.feat
                    )
                self.mode = "single_top"
        else:
            raise TypeError(
                f"trajs must be {SingleTraj.__class__} or {TrajEnsemble.__class__}, you provided {trajs.__class__}"
            )

    def __len__(self):
        if self.mode == "single_top":
            return len(self.feat.active_features)
        else:
            return len([f.features for f in self.feat])

    def __str__(self):
        if self.mode == "single_top":
            return self.feat.__str__()
        else:
            return ", ".join([f.__str__() for f in self.feat])

    def __repr__(self):
        if self.mode == "single_top":
            return self.feat.__repr__()
        else:
            return ", ".join([f.__repr__() for f in self.feat])


class EmptyFeature:
    """Class to fill with attributes to be read by encodermap.xarray.

    This class will be used in multiple_top mode, where the attributes
    _dim, describe and name will be overwritten with correct values to
    build features that contain NaN values.

    """

    def __init__(self, name, _dim, description, indexes):
        """Initialize the Empty feature.

        Args:
            name (str): The name of the feature.
            _dim (int): The feature length of the feature shape=(n_frames, ferature).
            description (list of str): The description for every feature.

        """
        self.name = name
        self._dim = _dim
        self.description = description
        self.indexes = indexes

    def describe(self):
        return self.description


class Topologies:
    def __init__(self, tops, alignments=None):
        self.tops = tops
        if alignments is None:
            alignments = [
                "side_dihedrals",
                "central_cartesians",
                "central_distances",
                "central_angles",
                "central_dihedrals",
            ]
        self.alignments = {k: {} for k in alignments}
        self.compare_tops()
        allowed_strings = list(
            filter(
                lambda x: True if "side" in x else False,
                (k for k in UNDERSOCRE_MAPPING.keys()),
            )
        )
        if not all([i in allowed_strings for i in alignments]):
            raise Exception(
                f"Invalid alignment string in `alignments`. Allowed strings are {allowed_strings}"
            )

    def compare_tops(self):
        if not all([t.n_residues == self.tops[0].n_residues for t in self.tops]):
            raise Exception(
                "Using Different Topologies currenlty only works if all contain the same number of residues."
            )
        generators = [t.residues for t in self.tops]
        sidechains = [t.select("sidechain") for t in self.tops]
        all_bonds = [
            list(map(lambda x: (x[0].index, x[1].index), t.bonds)) for t in self.tops
        ]

        # iterate over residues of the sequences
        n_res_max = max([t.n_residues for t in self.tops])
        for i in range(n_res_max):
            # get some info
            residues = [next(g) for g in generators]
            all_atoms = [[a.name for a in r.atoms] for r in residues]
            atoms = [
                list(
                    filter(
                        lambda x: True
                        if x.index in sel and "H" not in x.name and "OXT" not in x.name
                        else False,
                        r.atoms,
                    )
                )
                for r, sel in zip(residues, sidechains)
            ]
            atoms_indices = [[a.index for a in atoms_] for atoms_ in atoms]
            bonds = [
                list(
                    filter(
                        lambda bond: True if any([b in ai for b in bond]) else False, ab
                    )
                )
                for ai, ab in zip(atoms_indices, all_bonds)
            ]

            # reduce the integers of atoms_indices and bonds, so that N is 0. That way, we can compare them, even, when
            # two amino aicds in the chains are different
            N_indices = [
                list(filter(lambda x: True if x.name == "N" else False, r.atoms))[
                    0
                ].index
                for r in residues
            ]

            # align to respective N
            atoms_indices = [
                [x - N for x in y] for y, N in zip(atoms_indices, N_indices)
            ]
            bonds = [
                [(x[0] - N, x[1] - N) for x in y] for y, N in zip(bonds, N_indices)
            ]

            chi1 = [any(set(l).issubset(set(a)) for l in CHI1_ATOMS) for a in all_atoms]
            chi2 = [any(set(l).issubset(set(a)) for l in CHI2_ATOMS) for a in all_atoms]
            chi3 = [any(set(l).issubset(set(a)) for l in CHI3_ATOMS) for a in all_atoms]
            chi4 = [any(set(l).issubset(set(a)) for l in CHI4_ATOMS) for a in all_atoms]
            chi5 = [any(set(l).issubset(set(a)) for l in CHI5_ATOMS) for a in all_atoms]
            chi = np.array([chi1, chi2, chi3, chi4, chi5])

            self.alignments["side_dihedrals"][f"residue_{i}"] = chi

            if "side_cartesians" in self.alignments:
                raise NotImplementedError(
                    "Cartesians between different topologies can currently not be aligned."
                )

            if "side_distances" in self.alignments:
                raise NotImplementedError(
                    "Distances between different topologies can currently not be aligned."
                )

            if "side_angles" in self.alignments:
                raise NotImplementedError(
                    "Angles between different topologies can currently not be aligned."
                )

        self.drop_double_false()

    def drop_double_false(self):
        """Drops features that None of the topologies have.

        For example: Asp and Glu. Asp has a chi1 and chi2 torsion. Glu has chi1, chi2 and chi3. Both
        don't have chi4 or chi5. In self.compare_tops these dihedrals are still considered. In this
        method they will be removed.

        """
        for alignment, value in self.alignments.items():
            for residue, array in value.items():
                where = np.where(np.any(array, axis=1))[0]
                self.alignments[alignment][residue] = array[where]

    def get_max_length(self, alignment):
        """Maximum length that a feature should have given a certain axis.

        Args:
            alignment (str): The key for `self.alignments`.

        """
        alignment_dict = self.alignments[alignment]
        stacked = np.vstack([v for v in alignment_dict.values()])
        counts = np.count_nonzero(stacked, axis=0)  # Flase is 0
        return np.max(counts)

    def format_output(self, inputs, feats, sorted_trajs):
        """Formats the output of an em.Featurizer object using the alignment info.

        Args:
            inputs (list): List of pyemma.coordinates.data.feature_reader.FeatureReader objects.
            feats (list): List of encodermap.Featurizer objetcs.
            sorted_trajs (list): List of em.TrajEnsemble objects sorted in the same way as `self.tops`.

        """
        out = []
        for i, (inp, top, feat, trajs) in enumerate(
            zip(inputs, self.tops, feats, sorted_trajs)
        ):
            value_dict = {}
            for traj_ind, (data, traj) in enumerate(zip(inp.get_output(), trajs)):
                if any(
                    [isinstance(o, EmptyFeature) for o in feat.feat.active_features]
                ):
                    from ..misc.xarray import add_one_by_one

                    ffunc = lambda x: True if "NaN" not in x else False
                    indices = [0] + add_one_by_one(
                        [len(list(filter(ffunc, f.describe()))) for f in feat.features]
                    )
                else:
                    indices = get_indices_by_feature_dim(feat, traj, data.shape)

                # divide the values returned by PyEMMA
                for f, ind in zip(feat.features, indices):
                    try:
                        name = FEATURE_NAMES[f.name]
                    except KeyError:
                        name = f.__class__.__name__
                        f.name = name
                    except AttributeError:
                        name = f.__class__.__name__
                        f.name = name
                    if traj_ind == 0:
                        value_dict[name] = []
                    value_dict[name].append(data[:, ind])

            # stack along the frame axis, just like pyemma would
            value_dict = {k: np.vstack(v) for k, v in value_dict.items()}

            # put nans in all features specified by alignment
            for alignment, alignment_dict in self.alignments.items():
                if alignment not in value_dict:
                    continue
                max_length = self.get_max_length(alignment)
                new_values = np.full(
                    shape=(value_dict[alignment].shape[0], max_length),
                    fill_value=np.nan,
                )
                where = np.vstack([v for v in alignment_dict.values()])[:, i]
                new_values[:, where] = value_dict[alignment]
                value_dict[alignment] = new_values

                # find the index of the feature in feat.feat.active_features
                names = np.array(
                    [f.__class__.__name__ for f in feat.feat.active_features]
                )
                index = np.where([n in FEATURE_NAMES for n in names])[0]
                index = index[
                    np.where([FEATURE_NAMES[n] == alignment for n in names[index]])
                ]

                # get the old description and change it around
                assert len(index) == 1
                index = index[0]
                if not isinstance(feat.feat.active_features[index], EmptyFeature):
                    old_desc = np.array(
                        [i for i in feat.feat.active_features[index].describe()]
                    )
                    new_desc = np.array(
                        [
                            f"NaN due to ensemble with other topologies {i}"
                            for i in range(max_length)
                        ]
                    )
                    new_desc[where] = old_desc
                    new_desc = new_desc.tolist()

                    # get the old indexes and add the NaNs
                    old_indexes = feat.feat.active_features[index].indexes
                    new_indexes = np.full(
                        shape=(max_length, old_indexes.shape[1]), fill_value=np.nan
                    )
                    new_indexes[where] = old_indexes

                    # create empty feature
                    new_class = EmptyFeature(
                        alignment, max_length, new_desc, new_indexes
                    )
                    feat.feat.active_features[index] = new_class

            new_values = np.hstack([v for v in value_dict.values()])
            out.append([new_values, feat, trajs])
        return out

    def format_output_generator(self, inputs, feats, sorted_trajs):
        raise NotImplementedError("Will be implemented once tfRecords are implemented.")

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.tops):
            raise StopIteration
        else:
            self._index += 1
            return self.tops[self._index - 1]

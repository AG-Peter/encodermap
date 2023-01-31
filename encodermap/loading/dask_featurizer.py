# -*- coding: utf-8 -*-
# encodermap/loading/dask_featurizer.py
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

"""

################################################################################
# Imports
################################################################################


from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ..loading import delayed, features
from ..misc.xarray_save_wrong_hdf5 import save_netcdf_alongside_mdtraj
from .delayed import build_dask_xarray
from .featurizer import UNDERSOCRE_MAPPING

################################################################################
# Type Checking
################################################################################


if TYPE_CHECKING:
    import dask
    import xarray as xr
    from dask import dot_graph
    from dask.callbacks import Callback
    from dask.distributed import Client, progress
    from distributed.client import _get_global_client

    from ..trajinfo import TrajEnsemble
    from .features import AnyFeature


################################################################################
# Optional Imports
################################################################################


from .._optional_imports import _optional_import

featurizer = _optional_import("pyemma", "coordinates.featurizer")
source = _optional_import("pyemma", "coordinates.source")
CHI1_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI1_ATOMS")
CHI2_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI2_ATOMS")
CHI3_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI3_ATOMS")
CHI4_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI4_ATOMS")
CHI5_ATOMS = _optional_import("mdtraj", "geometry.dihedral.CHI5_ATOMS")
Client = _optional_import("dask", "distributed.Client")
dask = _optional_import("dask")
Callback = _optional_import("dask", "callbacks.Callback")
dot_graph = _optional_import("dask", "dot.dot_graph")
progress = _optional_import("dask", "distributed.progress")
HDF5TrajectoryFile = _optional_import("mdtraj", "formats.HDF5TrajectoryFile")
_get_global_client = _optional_import("distributed", "client._get_global_client")


################################################################################
# Globals
################################################################################


__all__ = []


################################################################################
# Utils
################################################################################


class Track(Callback):
    def __init__(
        self,
        path: str = "/tmp/dasks",
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


class DaskFeaturizer:
    def __init__(
        self,
        trajs: TrajEnsemble,
        n_workers: Union[str, int] = "cpu-2",
        client: Optional[Client] = None,
    ) -> None:
        self.in_memory = False
        self.trajs = trajs
        if not hasattr(self.trajs, "itertrajs"):
            self.trajs = self.trajs._gen_ensemble()
        self._copy_docstrings_from_pyemma()
        self.active_features = []

        if n_workers == "cpu-2":
            from multiprocessing import cpu_count

            n_workers = cpu_count() - 2
        if n_workers == "max":
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

    def _copy_docstrings_from_pyemma(self):
        from pyemma.coordinates.data.featurization.featurizer import (
            MDFeaturizer as feat_,
        )

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

    def build_graph(
        self,
        with_trajectories: bool = False,
    ) -> None:
        """Prepares the dask graph.

        Args:
            with_trajectories (Optional[bool]): Whether to also compute xyz.
                This can be useful, if you want to also save the trajectories to disk.

        """
        if with_trajectories:
            (
                self.dataset,
                self.xyz,
                self.time,
                self.unitcell_lengths,
                self.unitcell_angles,
            ) = build_dask_xarray(self, True)
        else:
            self.dataset = build_dask_xarray(self)

    def to_netcdf(
        self,
        filename: Union[str, list[str]],
        with_trajectories: bool = False,
    ) -> str:
        """Saves the dask tasks to a NetCDF4 formatted HDF5 file.

        Args:
            filename (Union[str, list[str]]): The filename to be used. If
                `with_trajectories` is True, a str sith a wildcard (*) is expected,
                in which case, the wildcard will be replaced by the basenames
                of the trajectories. Otherwise, a list of str with the same length
                as `self.trajs` can be supplied.
            with_trajectories (bool): Whether to save the trajectory data
                also. This allows you to build a library of comprehensive datafailes,
                that contain all important attributes of your trajectories. Defaults
                to False.

        Returns:
            Union[str, list[str]]: Returns the filename(s) of the created files.

        """
        if "dataset" not in self.__dict__:
            self.build_graph(with_trajectories=with_trajectories)
        if "dataset" in self.__dict__ and "xyz" not in self.__dict__:
            self.build_graph(True)

        # allows multiple writes to netcdf4 files
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

        if with_trajectories:
            e = Exception(
                "Saving trajectories with CVs is done via `xarray.mfdataset`, "
                "which is stored across multiple files. In that case, `filename` "
                "should be a list of str, with the same number as trajectories."
            )
            if not isinstance(filename, list):
                if "*" in filename:
                    filename = [
                        filename.replace("*", traj.basename) for traj in self.trajs
                    ]
                else:
                    raise e
            if len(filename) == len(self.trajs):
                raise e
            for fname, xyz, time, u_lengths, u_angles, traj, ds in zip(
                filename,
                self.xyz,
                self.time,
                self.unitcell_lengths,
                self.unitcell_angles,
                self.trajs,
                self.dataset,
            ):
                with HDF5TrajectoryFile(fname, "w", force_overwrite=True) as f:
                    f.write(
                        coordinates=xyz,
                        time=time,
                        cell_lengths=u_lengths,
                        cell_angles=u_angles,
                    )
                    f.topology = traj.top
                save_netcdf_alongside_mdtraj(fname, ds)
        else:
            self.dataset.to_netcdf(
                filename,
                format="NETCDF4",
                group="CVs",
                engine="h5netcdf",
                invalid_netcdf=True,
            )
        return filename

    def get_output(
        self,
        make_trace: bool = False,
    ) -> xr.Dataset:
        """This function passes the trajs and the features of to dask to create a
        delayed xarray out of that."""
        if "dataset" not in self.__dict__:
            self.build_graph()
        if not make_trace:
            # future = client.submit(future)
            out = self.client.compute(self.dataset)
            progress(out)
            return out.result()
        else:
            raise NotImplementedError(
                "gifsicle --delay 10 --loop=forever --colors 256 --scale=0.4 -O3 --merge dasks/part_*.png > output.gif"
            )
            # with Track():
            #     return self.dataset.compute()

    def visualize(self) -> None:
        return dask.visualize(self.dataset)

    def add_list_of_feats(
        self,
        which: Union[str, list[str]] = "all",
    ) -> None:
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
                feature = getattr(features, cf)(self.trajs.top[0])
                if hasattr(feature, "dask_transform"):
                    transform = getattr(
                        delayed, f"delayed_transfrom_{feature.dask_transform}"
                    )
                    feature.transform = transform
                self.add_custom_feature(feature)
        else:
            raise NotImplementedError
            for cf in which:
                if cf in UNDERSOCRE_MAPPING:
                    cf = UNDERSOCRE_MAPPING[cf]
                for top, feat in zip(self.trajs.top[0], self.feat):
                    feature = getattr(features, cf)(top)
                    feat.add_custom_feature(feature)

    @property
    def mode(self) -> str:
        try:
            if len(self.trajs.top) > 1:
                return "multiple_top"
            return "single_top"
        except TypeError:
            return "single_top"

    def __add_feature(
        self,
        f: AnyFeature,
    ) -> None:
        # perform sanity checks
        if f.dimension == 0:
            self.logger.error(
                "given an empty feature (eg. due to an empty/"
                "ineffective selection). Skipping it."
                " Feature desc: %s" % f.describe()
            )
            return

        if not hasattr(f.transform, "dask"):
            if f not in self.active_features:
                self.active_features.append(f)
            else:
                self.logger.warning(
                    "tried to re-add the same feature %s" % f.__class__.__name__
                )
        else:
            self.active_features.append(f)

    def _check_indices(
        self,
        pair_inds: list[list[int]],
        pair_n: int = 2,
    ) -> None:
        """ensure pairs are valid (shapes, all atom indices available?, etc.)"""

        pair_inds = np.array(pair_inds).astype(dtype=np.int, casting="safe")

        if pair_inds.ndim != 2:
            raise ValueError("pair indices has to be a matrix.")

        if pair_inds.shape[1] != pair_n:
            raise ValueError("pair indices shape has to be (x, %i)." % pair_n)

        if pair_inds.max() > max([traj.n_atoms for traj in self.trajs]):
            raise ValueError(
                "index out of bounds: %i."
                " Maximum atom index available: %i"
                % (pair_inds.max(), self.trajs.top[0].n_atoms)
            )

        return pair_inds

    def add_all(
        self,
        reference=None,
        atom_indices=None,
        ref_atom_indices=None,
    ):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.add_selection(
            list(range(self.trajs.top[0].n_atoms)),
            reference=reference,
            atom_indices=atom_indices,
            ref_atom_indices=ref_atom_indices,
        )

    def add_selection(
        self, indexes, reference=None, atom_indices=None, ref_atom_indices=None
    ):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        from pyemma.coordinates.data.featurization.misc import SelectionFeature

        from .delayed import delayed_transform_selection

        if reference is None:
            f = SelectionFeature(self.trajs.top[0], indexes)
            # monkey patch
            f.transform = delayed_transform_selection
        else:
            raise ValueError(
                "reference is not a mdtraj.Trajectory object, but {}".format(reference)
            )
            # f = AlignFeature(reference=reference, indexes=indexes,
            #                  atom_indices=atom_indices, ref_atom_indices=ref_atom_indices)
        self.__add_feature(f)

    def add_distances(self, indices, periodic=True, indices2=None):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        from pyemma.coordinates.data.featurization.distances import DistanceFeature
        from pyemma.coordinates.data.featurization.util import _parse_pairwise_input

        from .delayed import delayed_transfrom_distance

        atom_pairs = _parse_pairwise_input(
            indices, indices2, MDlogger=None, fname="add_distances()"
        )

        atom_pairs = self._check_indices(atom_pairs)
        f = DistanceFeature(self.trajs.top[0], atom_pairs, periodic=periodic)
        # monkey patch
        f.transform = delayed_transfrom_distance
        f.distance_indexes = np.ascontiguousarray(f.distance_indexes, dtype="int32")
        self.__add_feature(f)

    def add_distances_ca(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        self.feat.add_distances_ca(*args, **kwargs)

    def add_inverse_distances(self, indices, periodic=True, indices2=None):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        from pyemma.coordinates.data.featurization.distances import (
            InverseDistanceFeature,
        )
        from pyemma.coordinates.data.featurization.util import _parse_pairwise_input

        from .delayed import delayed_transfrom_inverse_distance

        atom_pairs = _parse_pairwise_input(
            indices, indices2, MDlogger=None, fname="add_distances()"
        )

        atom_pairs = self._check_indices(atom_pairs)
        f = InverseDistanceFeature(self.trajs.top[0], atom_pairs, periodic=periodic)
        # monkey patch
        f.transform = delayed_transfrom_inverse_distance
        f.distance_indexes = np.ascontiguousarray(f.distance_indexes, dtype="int32")
        self.__add_feature(f)

    def add_contacts(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        raise NotImplementedError()

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
        raise NotImplementedError()

    def add_residue_COM(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        raise NotImplementedError()

    def add_group_mindist(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        raise NotImplementedError()

    def add_angles(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        raise NotImplementedError()

    def add_dihedrals(self, indexes, deg=False, cossin=False, periodic=True):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        from pyemma.coordinates.data.featurization.angles import DihedralFeature

        from .delayed import delayed_transfrom_dihedral

        indexes = self._check_indices(indexes, pair_n=4)
        f = DihedralFeature(
            self.trajs.top[0], indexes, deg=deg, cossin=cossin, periodic=periodic
        )
        f.transform = delayed_transfrom_dihedral
        f.indexes = np.ascontiguousarray(f.angle_indexes, dtype="int32")
        self.__add_feature(f)

    def add_backbone_torsions(
        self, selstr=None, deg=False, cossin=False, periodic=True
    ):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        from pyemma.coordinates.data.featurization.angles import BackboneTorsionFeature

        from .delayed import delayed_transfrom_dihedral

        f = BackboneTorsionFeature(
            self.trajs.top[0], selstr=selstr, deg=deg, cossin=cossin, periodic=periodic
        )
        f.transform = delayed_transfrom_dihedral
        f.indexes = np.ascontiguousarray(f.angle_indexes, dtype="int32")
        self.__add_feature(f)

    def add_chi1_torsions(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        raise NotImplementedError()

    def add_sidechain_torsions(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        raise NotImplementedError()

    def add_minrmsd_to_ref(self, *args, **kwargs):
        if self.mode == "multiple_top":
            raise Exception(
                "Using PyEMMA's `add_x` functions is not possible when TrajEnsemble contains multiple topologies."
            )
        raise NotImplementedError()

    def add_custom_feature(self, feature, feature_name=None, labels=None, length=None):
        if callable(feature):
            if feature_name is None:
                raise Exception(
                    "Providing a callable as a feature also needs a `feature_name`."
                )

            # create a class and fill it with attributres
            class CustomFeature:
                pass

            f = CustomFeature()
            f.transform = feature
            f.dimension = length

            # what is labels
            if labels is not None:
                f.describe = lambda: labels
            else:
                if length == 1:
                    f.describe = lambda: [feature_name.upper()]
                else:
                    f.describe = lambda: [
                        f"{feature_name.upper()} Frame Feature {i}"
                        for i in range(length)
                    ]

            # a name for the feature
            f.name = feature_name
        else:
            f = feature
        self.__add_feature(f)

    @property
    def features(self):
        if self.mode == "single_top":
            return self.active_features
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

    def describe(self):
        all_labels = []
        for f in self.active_features:
            try:
                all_labels += f.describe()
            except TypeError:
                all_labels += [f"{f.name} features with variable/unknown length."]
        return all_labels

    def dimension(self):
        dim = sum(f.dimension for f in self.active_features)
        return dim

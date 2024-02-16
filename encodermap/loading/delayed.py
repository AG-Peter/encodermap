# -*- coding: utf-8 -*-
# encodermap/loading/delayed.py
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
"""Functions to use with the DaskFeaturizer class.

"""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Third Party Imports
import MDAnalysis as mda
import numpy as np
import xarray
import xarray as xr
from optional_imports import _optional_import

# Local Folder Imports
from ..misc.xarray import FEATURE_NAMES, make_dataarray, make_frame_CV_dataarray


################################################################################
# Optional Imports
################################################################################


dask = _optional_import("dask")
da = _optional_import("dask", "array")
source = _optional_import("pyemma", "coordinates.source")
box_vectors_to_lengths_and_angles = _optional_import(
    "mdtraj", "utils.unitcell.box_vectors_to_lengths_and_angles"
)
_dist_mic = _optional_import("mdtraj", "geometry._geometry._dist_mic")
_dist = _optional_import("mdtraj", "geometry._geometry._dist")
_dihedral_mic = _optional_import("mdtraj", "geometry._geometry._dihedral_mic")
_dihedral = _optional_import("mdtraj", "geometry._geometry._dihedral")
_angle_mic = _optional_import("mdtraj", "geometry._geometry._angle_mic")
_angle = _optional_import("mdtraj", "geometry._geometry._angle")
jit = _optional_import("numba", "jit")
prange = _optional_import("numba", "prange")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from typing import TYPE_CHECKING, Optional, Union, overload


if TYPE_CHECKING:
    # Third Party Imports
    from dask.delayed import Delayed
    from dask_featurizer import DaskFeaturizer

    # Local Folder Imports
    from .._typing import AnyFeature


################################################################################
# Numba compiled functions
################################################################################


# @njit(fastmath=True)
def calc_distances(xyz, indices):
    points_a = xyz[:, indices[:, 0]]
    points_b = xyz[:, indices[:, 1]]
    diffs = points_b - points_a
    data = diffs.reshape((-1, 3))
    a = np.sqrt((data * data).sum(axis=1))
    a = a.reshape(diffs.shape[:2])
    return a


################################################################################
# Delayed functions
################################################################################


@dask.delayed
def delayed_transform_selection(
    self: AnyFeature,
    xyz: np.ndarray,
    unitcell_vectors: None = None,
    unitcell_infos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns the cooordinates of the selected atoms.

    Args:
        self (AnyFeature): A feature. Can be PyEMMA feature, or encomderap feature.
        xyz (np.ndarray): The positions.
        unitcell_vectors (np.ndarray): Info about the unitcell in shape (n_frames, 3, 3)
        unitcell_infos (np.ndarray): Info about the unitcell in shape (n_frames, 6).

    Returns:
        np.ndarray: Positions of the atoms with ndim=2.

    """
    newshape = (xyz.shape[0], 3 * self.indexes.shape[0])
    return np.expand_dims(np.reshape(xyz[:, self.indexes, :], newshape), 0)


@dask.delayed
def delayed_transform_angle(
    self: AnyFeature,
    xyz: np.ndarray,
    unitcell_vectors: Optional[np.ndarray] = None,
    unitcell_infos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mimics MDTraj's compute_angle, but delayed and without loading the complete trajectory.

    Args:
        self (AnyFeature): A feature. Can be PyEMMA feature, or EncoderMap feature.
        xyz (np.ndarray): The positions.
        unitcell_vectors (np.ndarray): Info about the unitcell in shape (n_frames, 3, 3)
        unitcell_infos (np.ndarray): Info about the unitcell in shape (n_frames, 6).

    Returns:
        np.ndarray: The result of the dihedral calculation.

    """
    try:
        indexes = self.indexes.astype("int32")
    except AttributeError:
        indexes = self.angle_indexes.astype("int32")

    if self.periodic:
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

        out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
        _angle_mic(
            xyz, indexes, unitcell_vectors.transpose(0, 2, 1).copy(), out, orthogonal
        )
    else:
        out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
        _angle(xyz, indexes, out)

    if self.cossin:
        out = np.dstack((np.cos(out), np.sin(out)))
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])

    if self.deg:
        out = np.rad2deg(out)

    if self.cossin:
        raise NotImplementedError

    out = np.expand_dims(out, 0)
    return out


@dask.delayed
def delayed_transform_dihedral(
    self: AnyFeature,
    xyz: np.ndarray,
    unitcell_vectors: Optional[np.ndarray] = None,
    unitcell_infos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mimics MDTraj's compute_dihedral, but delayed and without loading the complete trajectory.

    Args:
        self (AnyFeature): A feature. Can be PyEMMA feature, or encomderap feature.
        xyz (np.ndarray): The positions.
        unitcell_vectors (np.ndarray): Info about the unitcell in shape (n_frames, 3, 3)
        unitcell_infos (np.ndarray): Info about the unitcell in shape (n_frames, 6).

    Returns:
        np.ndarray: The result of the dihedral calculation.

    """
    try:
        indexes = self.indexes.astype("int32")
    except AttributeError:
        indexes = self.angle_indexes.astype("int32")

    if len(indexes) == 0:
        return np.zeros((len(xyz), 0), dtype="float32")

    if self.periodic:
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

        out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
        _dihedral_mic(
            xyz, indexes, unitcell_vectors.transpose(0, 2, 1).copy(), out, orthogonal
        )
    else:
        out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
        _dihedral(xyz, indexes, out)

    if self.cossin:
        out = np.dstack((np.cos(out), np.sin(out)))
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])

    if self.deg:
        out = np.rad2deg(out)

    return np.expand_dims(out, 0)


@dask.delayed
def delayed_transform_distance(
    self: AnyFeature,
    xyz: np.ndarray,
    unitcell_vectors: np.ndarray = None,
    unitcell_infos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mimics MDTraj's compute_distance, but delayed and without loading the complete trajectory.

    Args:
        self (AnyFeature): A feature. Can be PyEMMA feature, or EncoderMap feature.
        xyz (np.ndarray): The positions.
        unitcell_vectors (np.ndarray): Info about the unitcell in shape (n_frames, 3, 3)
        unitcell_infos (np.ndarray): Info about the unitcell in shape (n_frames, 6).

    Returns:
        np.ndarray: The result of the distance calculation.

    """
    if len(self.distance_indexes) == 0:
        return np.zeros((len(xyz), 0), dtype="float32")

    if self.periodic:
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
            (xyz.shape[0], self.distance_indexes.shape[0]), dtype="float32", order="C"
        )
        _dist_mic(
            xyz,
            self.distance_indexes.astype("int32"),
            unitcell_vectors.transpose(0, 2, 1).copy(),
            out,
            orthogonal,
        )
        return np.expand_dims(out, 0)
    else:
        out = np.empty(
            (xyz.shape[0], self.distance_indexes.shape[0]), dtype="float32", order="C"
        )
        _dist(xyz, self.distance_indexes, out)
        return np.expand_dims(out, 0)


@dask.delayed
def delayed_transform_inverse_distance(
    self: AnyFeature,
    xyz: np.ndarray,
    unitcell_vectors: Optional[np.ndarray] = None,
    unitcell_infos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mimics MDTraj's compute_distance but returns inverse distances.

    Args:
        self (AnyFeature): A feature. Can be PyEMMA feature, or encomderap feature.
        xyz (np.ndarray): The positions.
        unitcell_vectors (np.ndarray): Info about the unitcell.
        unitcell_infos (np.ndarray): Info about the unitcell in shape (n_frames, 6).

    Returns:
        np.ndarray: The result of the distance calculation.

    """
    return 1 / delayed_transform_distance(self, xyz, unitcell_vectors)


@jit(parallel=True, nopython=True)
def calc_bravais_box(box_info):
    """Calculates the Bravais vectors from lengths and angles (in degrees).

    Note:
        This code is adapted from gyroid, which is licensed under the BSD
        http://pythonhosted.org/gyroid/_modules/gyroid/unitcell.html

    """
    a_length, b_length, c_length = box_info[:, :3].T
    alpha, beta, gamma = box_info[:, 3:].T

    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180

    a = np.zeros((3, len(a_length)), dtype="float32")
    a[0] = a_length
    b = np.zeros((3, len(b_length)), dtype="float32")
    b[0] = b_length * np.cos(gamma)
    b[1] = b_length * np.sin(gamma)
    cx = c_length * np.cos(beta)
    cy = c_length * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c_length * c_length - cx * cx - cy * cy)
    c = np.empty((3, len(c_length)), dtype="float32")
    c[0] = cx
    c[1] = cy
    c[2] = cz

    if not a.shape == b.shape == c.shape:
        raise TypeError("Shape is messed up.")

    # Make sure that all vector components that are _almost_ 0 are set exactly
    # to 0
    tol = 1e-6
    for i in prange(a.shape[0]):
        for j in prange(a.shape[1]):
            if a[i, j] > -tol and a[i, j] < tol:
                a[i, j] = 0.0
    for i in prange(b.shape[0]):
        for j in prange(b.shape[1]):
            if b[i, j] > -tol and b[i, j] < tol:
                b[i, j] = 0.0
    for i in prange(c.shape[0]):
        for j in prange(c.shape[1]):
            if c[i, j] > -tol and c[i, j] < tol:
                c[i, j] = 0.0

    unitcell_vectors = np.ascontiguousarray(
        np.swapaxes(np.dstack((a.T, b.T, c.T)), 1, 2)
    )

    return unitcell_vectors


@dask.delayed(nout=4, name="load_traj_data")
def _load_xyz(traj_file, u, frame_indices):
    """Distances in nm. Angles in degree."""
    ag = u.atoms
    positions = np.empty(
        shape=(len(frame_indices), len(ag), 3), dtype="float32", order="C"
    )
    time = np.empty(shape=(len(frame_indices)), dtype="float32", order="C")
    unitcell_info = np.empty(shape=(len(frame_indices), 6), dtype="float32", order="C")
    for i, ts in enumerate(u.trajectory[frame_indices]):
        positions[i] = ag.positions
        time[i] = ts.time
        unitcell_info[i] = ts._unitcell
    positions /= 10  # for some heretical reason, MDAnalysis uses angstrom
    unitcell_info[:, :3] /= 10
    unitcell_vectors = calc_bravais_box(unitcell_info)

    return positions, time, unitcell_vectors, unitcell_info


def load_xyz(traj_file, frame_indices, top_file=None):
    u = mda.Universe(top_file, traj_file)
    n_atoms = len(u.atoms)
    n_frames = len(frame_indices)
    p, t, uv, ui = _load_xyz(traj_file, u, frame_indices)
    p = da.from_delayed(p, shape=(n_frames, n_atoms, 3), dtype="float32", name="xyz")
    t = da.from_delayed(t, shape=(n_frames,), dtype="float32", name="time")
    uv = da.from_delayed(
        uv, shape=(n_frames, 3, 3), dtype="float32", name="unitcell_vectors"
    )
    ui = da.from_delayed(
        ui, shape=(n_frames, 6), dtype="float32", name="unitcell_infos"
    )
    return p, t, uv, ui


################################################################################
# Dask graph creation
################################################################################


@overload
def build_dask_xarray(
    featurizer: "DaskFeaturizer",
    return_coordinates: bool = False,
    streamable: bool = False,
) -> xarray.Dataset:
    ...


@overload
def build_dask_xarray(
    featurizer: "DaskFeaturizer",
    return_coordinates: bool = True,
    streamable: bool = False,
) -> tuple[xarray.Dataset, "Delayed", "Delayed", "Delayed", "Delayed"]:
    ...


def build_dask_xarray(
    featurizer: "DaskFeaturizer",
    return_coordinates: bool = False,
    streamable: bool = False,
) -> Union[
    xarray.Dataset, tuple[xarray.Dataset, "Delayed", "Delayed", "Delayed", "Delayed"]
]:
    """Builds a large dask xarray, which will be distributively evaluated.

    This class takes a `DaskFeaturizer` class, which contains a lis of features.
    Every feature in this list contains enough information for the delayed functions
    to calculate the requested quantities when provided the xyz coordinates of the
    atoms, the unitcell vectors, and the unitcell infos as a Bravais matrix.

    Args:
        featurizer (DaskFeaturizer): An instance of the DaskFeaturizer.
        return_coordinates (bool): Whether to add this information:
            all_xyz, all_time, all_cell_lengths, all_cell_angles
            to the returned values. Defaults to False.
        streamable (bool): Whether to divide the calculations into one-frame
            blocks, which can then only be calculated, when requested.

    Returns:
        Union[xarray.Dataset, tuple[xarray.Dataset, "Delayed", "Delayed", "Delayed", "Delayed"]]:
            When `return_coordinates` is False, only an xarray.Dataset is returned.
            Otherwise a tuple with a xarray.Dataset and a sequence of dask.Delayed
            objects is returned.


    """
    # pre-define blocks from the trajectories
    n_blocks = 10
    all_DAs = []

    if return_coordinates:
        all_xyz = []
        all_time = []
        all_cell_lengths = []
        all_cell_angles = []

    assert len(featurizer.active_features) > 0

    # collect the Datasets in this dict
    DSs = []
    # if any of the features has the self.dask_delayed == "custom"
    # attribute it is a very low level implementation and we just collect the
    # delayed objects in this dict and later make a dataset from these
    custom_delayeds = {}
    frame_indices_per_traj = {}

    for i, traj in featurizer.trajs.itertrajs():
        if not streamable:
            n_frames_per_block = len(traj.id) // n_blocks
            blocks = [
                np.arange(i * n_frames_per_block, (i + 1) * n_frames_per_block)
                for i in range(n_blocks - 1)
            ]
            blocks.append(np.arange((n_blocks - 1) * n_frames_per_block, len(traj.id)))
        else:
            n_frames_per_block = 1
            blocks = [[i] for i in range(len(traj.id))]

        # if we have a one-block we append it to the previous ones to prevent some bad shapes

        # collect multiple Dataarrays here
        DAs = {}
        ind_DAs = {}

        # these lists collect the data from distributed loading tasks
        xyzs = []
        times = []
        unitcell_vectors = []
        unitcell_infos = []

        # distribute the loading to multiple workers
        for j, frame_indices in enumerate(blocks):
            xyz, time, unitcell_vector, unitcell_info = load_xyz(
                traj.traj_file, frame_indices, traj.top_file
            )
            assert len(frame_indices) > 1
            xyzs.append(xyz)
            times.append(time)
            unitcell_vectors.append(unitcell_vector)
            unitcell_infos.append(unitcell_info)

            # iterate over the features and let them use the traj information
            for k, feat in enumerate(featurizer.active_features):
                # the name of the feature will be used for traceability
                if hasattr(feat, "name"):
                    name = feat.name
                else:
                    try:
                        name = FEATURE_NAMES[feat.name]
                    except (KeyError, AttributeError):
                        name = feat.__class__.__name__
                        feat.name = name

                # create a da.dataarray using the delayed feature
                if callable(feat.describe()):
                    feat_length = len([i for i in feat.describe()(traj.top)])
                else:
                    feat_length = len(feat.describe())
                if feat.dask_transform != "custom":
                    a = da.from_delayed(
                        feat.transform(
                            feat,
                            xyz,
                            unitcell_vector,
                            unitcell_info,
                        ),
                        shape=(1, len(frame_indices), feat_length),
                        dtype="float32",
                    )
                else:
                    custom_shape = feat.custom_shape
                    custom_shape[custom_shape.index("frame_indices")] = len(
                        frame_indices
                    )
                    custom_shape = tuple(custom_shape)
                    a = da.from_delayed(
                        feat.transform(
                            xyz,
                            unitcell_vector,
                            unitcell_info,
                            frame_indices,
                        ),
                        shape=custom_shape,
                        dtype="float32",
                    )

                # make a xarray out of that
                if traj.id.ndim == 2:
                    traj_id = traj.id[frame_indices, 1]
                else:
                    traj_id = traj.id[frame_indices]

                if feat.dask_transform != "custom":
                    if hasattr(feat, "deg"):
                        deg = feat.deg
                    else:
                        deg = None
                    if feat_length > 1:
                        dataarray, ind_dataarray = make_dataarray(
                            feat.describe(),
                            traj,
                            name,
                            a,
                            deg=deg,
                            with_time=False,
                            frame_indices=traj_id,
                            feat=feat,
                        )
                    else:
                        dataarray, ind_dataarray = make_frame_CV_dataarray(
                            feat.describe(),
                            traj,
                            name,
                            a,
                            deg=deg,
                            with_time=False,
                            frame_indices=traj_id,
                            feat=feat,
                        )

                    # append the dataarray to the DAs dictionary
                    DAs.setdefault(name, []).append(dataarray)
                    if ind_dataarray is not None:
                        ind_DAs.setdefault(name + "_feature_indices", []).append(
                            ind_dataarray
                        )

                    # concatenate the data. Xarray should then know where to write the
                    # data on disk, if the hdf5 (netcdf4) file is opened in non-blocking mode
                    # set this env variable: HDF5_USE_FILE_LOCKING=FALSE
                else:
                    assert len(featurizer.active_features) == 1, (
                        f"Using custom features, with custom transform in the "
                        f"dask featurizer is only possible, if the featurizer "
                        f"contains 1 feature. This featurizer contains "
                        f"{len(featurizer.active_features)} features."
                    )
                    custom_delayeds.setdefault(traj.traj_num, []).append(a)
                    frame_indices_per_traj.setdefault(traj.traj_num, []).append(
                        frame_indices
                    )

        # make sure all dataarrays have the same shape on the feature axis
        for key, val in DAs.items():
            assert all([val[0].shape[-1] == i.shape[-1] for i in val[1:]])

        # remove all unwanted duplicate dataarrays from the indices
        for key, val in ind_DAs.items():
            ind_DAs[key] = val[0]

        # if raw coordinates are wanted we concatenate them after the blocks
        # are finished and append them to lists, these lists have the length of
        # trajs in feat.trajs
        if return_coordinates:
            concatenated_xyzs = da.concatenate([da.concatenate(xyzs[:-1]), xyzs[-1]])
            concatenated_time = da.concatenate([da.concatenate(times[:-1]), times[-1]])
            concatenated_unitcell_vectors = da.concatenate(
                [da.concatenate(unitcell_vectors[:-1]), unitcell_vectors[-1]]
            )
            concatenated_unitcell_infos = da.concatenate(
                [da.concatenate(unitcell_infos[:-1]), unitcell_infos[-1]]
            )
            all_xyz.append(concatenated_xyzs)
            all_time.append(concatenated_time)
            all_cell_lengths.append(concatenated_unitcell_infos[:, :3])
            all_cell_angles.append(concatenated_unitcell_infos[:, 3:])

        # after the features have been iterated over for this traj, the DAs are
        # merged along the time axis and a Dataset is created from them
        if any(v for v in DAs.values()):
            for key, value in DAs.items():
                DAs[key] = xr.concat(DAs[key], "frame_num")
            ds = xr.Dataset(DAs | ind_DAs)
            DSs.append(ds)

    if DSs:
        # make a large dataset out of this
        ds = xr.concat(DSs, dim="traj_num")
        # ds = ds.assign_attrs(indexes)
    else:
        DSs = []
        data = {k: da.concatenate(v, axis=1) for k, v in custom_delayeds.items()}
        assert len(data) == featurizer.trajs.n_trajs
        for traj, (traj_num, delayeds) in zip(featurizer.trajs, data.items()):
            frame_indices = frame_indices_per_traj[traj_num]
            empty_ds = feat.empty_dataset([traj_num], frame_indices, traj.basename)
            for da_data, (feat_name, dataarray) in enumerate(
                empty_ds.data_vars.items()
            ):
                da_data = data[traj_num][:, :, :, da_data]
                if dataarray.ndim < 5:
                    da_data = da_data[..., 0]
                da_data = da_data[:, :, :, : dataarray.shape[3]]
                if dataarray.ndim == 5:
                    empty_ds[feat_name] = (
                        (
                            "traj_num",
                            "frame_num",
                            "ubq_num",
                            feat_name.upper(),
                            "COORDS",
                        ),
                        da_data,
                    )
                else:
                    empty_ds[feat_name] = (
                        ("traj_num", "frame_num", "ubq_num", feat_name.upper()),
                        da_data,
                    )
            DSs.append(empty_ds)
        # from ..trajinfo.trajinfo_utils import trajs_combine_attrs
        # ds = xr.concat(DSs, dim="traj_num")  # replaced by combine_nested_method
        # ds = xr.combine_nested(
        #     DSs,
        #     concat_dim="traj_num",
        #     compat="broadcast_equals",
        #     fill_value=np.nan,
        #     coords="all",
        #     join="outer",
        #     combine_attrs=trajs_combine_attrs,
        # )
        raise Exception("Compare concat vs combine_nested.")

    if return_coordinates:
        return ds, all_xyz, all_time, all_cell_lengths, all_cell_angles

    return ds

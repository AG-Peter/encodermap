# -*- coding: utf-8 -*-
# encodermap/loading/delayed.py
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
"""Functions to use with the DaskFeaturizer class.

"""


################################################################################
# Imports
################################################################################


from __future__ import annotations

import MDAnalysis as mda
import numpy as np
import xarray as xr

from .._optional_imports import _optional_import
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
jit = _optional_import("numba", "jit")
prange = _optional_import("numba", "prange")


################################################################################
# Typing
################################################################################


import typing

if typing.TYPE_CHECKING:
    from .._typing import AnyFeature

from typing import Optional, Union

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
    """Returns the cooordinates of the selected atoms..

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
def delayed_transfrom_dihedral(
    self: AnyFeature,
    xyz: np.ndarray,
    unitcell_vectors: np.ndarray = None,
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

        out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
        _dihedral_mic(
            xyz, indexes, unitcell_vectors.transpose(0, 2, 1).copy(), out, orthogonal
        )
        return np.expand_dims(out, 0)
    else:
        out = np.empty((xyz.shape[0], indexes.shape[0]), dtype="float32", order="C")
        _dihedral(xyz, indexes, out)

    if self.cossin:
        out = np.dstack((np.cos(out), np.sin(out)))
        out = rad.reshape(out.shape[0], out.shape[1] * out.shape[2])
        # convert to degrees
    if self.deg and not self.cossin:
        out = np.rad2deg(out)

    return np.expand_dims(out, 0)


@dask.delayed
def delayed_transfrom_distance(
    self: AnyFeature,
    xyz: np.ndarray,
    unitcell_vectors: np.ndarray = None,
    unitcell_infos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mimics MDTraj's compute_distance, but delayed and without loading the complete trajectory.

    Args:
        self (AnyFeature): A feature. Can be PyEMMA feature, or encomderap feature.
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
            self.distance_indexes,
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
def delayed_transfrom_inverse_distance(
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

    Returns:
        np.ndarray: The result of the distance calculation.

    """
    return 1 / delayed_transfrom_distance(self, xyz, unitcell_vectors)


@jit(parallel=True)
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
def load_xyz(traj_file, frame_indices, top_file=None):
    u = mda.Universe(top_file, traj_file)
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


################################################################################
# Dask graph creation
################################################################################


def build_dask_xarray(featurizer, return_coordinates=False):
    """Builds a large dask xarray, which will be distributively evaluated."""
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

    for i, traj in featurizer.trajs.itertrajs():
        n_frames_per_block = len(traj.id) // n_blocks
        blocks = [
            np.arange(i * n_frames_per_block, (i + 1) * n_frames_per_block)
            for i in range(n_blocks - 1)
        ]
        blocks.append(np.arange((n_blocks - 1) * n_frames_per_block, len(traj.id)))

        # collect multiple Dataarrays here
        DAs = {}
        indexes = {}

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
            xyzs.append(
                da.from_delayed(
                    xyz,
                    shape=(len(frame_indices), traj.top.n_atoms, 3),
                    dtype="float32",
                    name="append_xyz",
                )
            )
            times.append(
                da.from_delayed(
                    time,
                    shape=(len(frame_indices),),
                    dtype="float32",
                    name="append_time",
                )
            )
            unitcell_vectors.append(
                da.from_delayed(
                    unitcell_vector,
                    shape=(len(frame_indices), 3, 3),
                    dtype="float32",
                    name="append_bravais",
                )
            )
            unitcell_infos.append(
                da.from_delayed(
                    unitcell_info,
                    shape=(len(frame_indices), 6),
                    dtype="float32",
                    name="append_cell_length_and_angle",
                )
            )

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

                # add the indices used to create this dataarray
                if name not in indexes:
                    try:
                        indexes[name] = feat.indexes.tolist()
                    except AttributeError as e:
                        for key in feat.__dir__():
                            if "ind" in key:
                                indexes[name] = feat.__dict__[key]
                        if name not in indexes:
                            indexes[name] = []

                # create the key if not already existent
                if name not in DAs:
                    DAs[name] = []

                # create a da.dataarray using the delayed feature
                if callable(feat.describe()):
                    feat_length = len([i for i in feat.describe()(traj.top)])
                else:
                    feat_length = len(feat.describe())
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
                # a.compute_chunk_sizes()
                # make a xarray out of that
                if traj.id.ndim == 2:
                    traj_id = traj.id[frame_indices, 1]
                else:
                    traj_id = traj.id[frame_indices]

                if feat_length > 1:
                    dataarray = make_dataarray(
                        feat.describe(),
                        traj,
                        name,
                        a,
                        with_time=False,
                        frame_indices=traj_id,
                    )
                else:
                    dataarray = make_frame_CV_dataarray(
                        feat.describe(),
                        traj,
                        name,
                        a,
                        with_time=False,
                        frame_indices=traj_id,
                    )

                # append the dataarray to the DAs dictionary
                DAs[name].append(dataarray)

                # concatenate the data. Xarray should then know where to write the
                # data on disk, if the hdf5 (netcdf4) file is opened in non-blocking mode
                # set this env variable: HDF5_USE_FILE_LOCKING=FALSE

        # if raw coordinates are wanted we concatenate them after the blocks
        # are finished and append them to lists, these lists have the length of
        # trajs in feat.trajs
        if return_coordinates:
            concatenated_xyzs = da.concatenate(xyzs, axis=0)
            concatenated_time = da.concatenate(times)
            concatenated_unitcell_vectors = da.concatenate(unitcell_vectors, 0)
            concatenated_unitcell_infos = da.concatenate(unitcell_infos, 0)
            all_xyz.append(concatenated_xyzs)
            all_time.append(concatenated_time)
            all_cell_lengths.append(concatenated_unitcell_infos[:, :3])
            all_cell_angles.append(concatenated_unitcell_infos[:, 3:])

        # after the features have been iterated over for this traj, the DAs are
        # merged along the time axis and a Dataset is created from them
        for key, value in DAs.items():
            DAs[key] = xr.concat(DAs[key], "frame_num")
        ds = xr.Dataset(DAs)
        DSs.append(ds)

    # make the large dataset out of this
    ds = xr.concat(DSs, dim="traj_num")
    ds = ds.assign_attrs(indexes)

    if return_coordinates:
        return all_DAs, all_xyz, all_time, all_cell_lengths, all_cell_angles

    return ds


def analyze_block(frame_indices, universe, atomgroup, indices, unwrap=True):
    # get all positions
    positions = np.empty(shape=(len(frame_indices), len(atomgroup), 3), dtype="float32")
    # unitcell_info = np.empty(shape=(len(frame_indices), 6), dtype='float32')
    for i, ts in enumerate(universe.trajectory[frame_indices]):
        if unwrap:
            positions[i] = atomgroup.unwrap(compound="fragments")
        else:
            positions[i] = atomgroup.positions

    positions /= 10  # for some heretical reason, MDAnalysis uses angstrom
    # unitcell_info[:, :3] /= 10
    # unitcell_angles = unitcell_info[:, 3:]
    # unitcell_vectors = calc_bravais_box(unitcell_info)

    # do stuff with the positions
    if indices.ndim == 1:
        return positions[:, indices]
    elif indices.shape[1] == 2:
        func = calc_distances
    elif indices.shape[1] == 3:
        func = calc_angles
    elif indices.shape[1] == 4:
        func = calc_dihedrals
    else:
        raise Exception(
            f"Indices of shape {indices.shape} not supported. Normally "
            f"you want `indices.ndim` == 1 for cartesian coordinates. "
            f"`indices.shape[1] == 2` for distances, `indices.shape[1] == 3` "
            f"for angles, and `indices.shape[1] == 4` for dihedrals."
        )
    result = func(positions, indices)
    return result

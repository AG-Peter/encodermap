# -*- coding: utf-8 -*-
# encodermap/loading/delayed.py
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
"""Functions to use with the DaskFeaturizer class.

"""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
from pathlib import Path

# Third Party Imports
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader
from optional_imports import _optional_import


################################################################################
# Optional Imports
################################################################################


dask = _optional_import("dask")
da = _optional_import("dask", "array")
dd = _optional_import("dask", "dataframe")
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
xr = _optional_import("xarray")
md = _optional_import("mdtraj")
h5py = _optional_import("h5py")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from typing import TYPE_CHECKING, Literal, Optional, Union, overload


if TYPE_CHECKING:
    # Third Party Imports
    from dask.delayed import Delayed

    # Encodermap imports
    from encodermap.loading.featurizer import DaskFeaturizer
    from encodermap.trajinfo.info_single import SingleTraj


################################################################################
# Utils
################################################################################


# @jit(parallel=True, nopython=True)
def calc_bravais_box(box_info: np.ndarray) -> np.ndarray:
    """Calculates the Bravais vectors from lengths and angles (in degrees).

    Note:
        This code is adapted from gyroid, which is licensed under the BSD
        http://pythonhosted.org/gyroid/_modules/gyroid/unitcell.html

    Args:
        box_info (np.ndarray): The box info, where the columns are ordered as
            follows: a, b, c, alpha, beta. gamma in degree.

    Returns:
        np.ndarray: The bravais vectors as a shape (n_frames, 3, 3) array.

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
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] > -tol and a[i, j] < tol:
                a[i, j] = 0.0
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i, j] > -tol and b[i, j] < tol:
                b[i, j] = 0.0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] > -tol and c[i, j] < tol:
                c[i, j] = 0.0

    unitcell_vectors = np.ascontiguousarray(
        np.swapaxes(np.dstack((a.T, b.T, c.T)), 1, 2)
    )

    return unitcell_vectors


@dask.delayed(nout=4)
def _load_xyz(traj, frame_indices):
    """Distances in nm. Angles in degree."""
    positions = np.empty(
        shape=(len(frame_indices), traj.n_atoms, 3), dtype="float32", order="C"
    )
    time = np.empty(shape=(len(frame_indices)), dtype="float32", order="C")
    unitcell_info = np.empty(shape=(len(frame_indices), 6), dtype="float32", order="C")
    for i, ts in enumerate(traj[frame_indices]):
        positions[i] = ts.positions
        time[i] = ts.time
        unitcell_info[i] = ts._unitcell
    positions /= 10  # for some heretical reason, MDAnalysis uses angstrom
    unitcell_info[:, :3] /= 10
    unitcell_vectors = calc_bravais_box(unitcell_info)

    return positions, time, unitcell_vectors, unitcell_info


def load_xyz(
    traj_file: str,
    frame_indices: np.ndarray,
    traj_num: Optional[int] = None,
) -> tuple[da.array, da.array, da.array, da.array]:
    if Path(traj_file).suffix == ".h5":
        return load_xyz_from_h5(traj_file, frame_indices, traj_num)
    if Path(traj_file).suffix != ".xtc":
        raise Exception(
            f"Currently only .xtc and .h5 trajectory files are supported. "
            f"But adding more formats is easy. Raise an issue, if you want "
            f"to have them added."
        )
    traj = XTCReader(traj_file)
    n_atoms = traj.n_atoms
    n_frames = len(frame_indices)
    p, t, uv, ui = _load_xyz(traj, frame_indices)
    p = da.from_delayed(
        p,
        shape=(n_frames, n_atoms, 3),
        dtype="float32",
    )
    t = da.from_delayed(
        t,
        shape=(n_frames,),
        dtype="float32",
    )
    uv = da.from_delayed(
        uv,
        shape=(n_frames, 3, 3),
        dtype="float32",
    )
    ui = da.from_delayed(
        ui,
        shape=(n_frames, 6),
        dtype="float32",
    )
    return p, t, uv, ui


@dask.delayed(nout=4)
def _load_xyz_from_h5(
    traj_file: str,
    frame_indices: np.ndarray,
    traj_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distances in nm. Angles in degree.

    Args:
        traj_file (str): The file to load.
        frame_indices (np.ndarray): An int array giving the positions to load.
        traj_num (int): Which traj num the output should be put to.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray: A four-tuple of np
            arrays. The order of these arrays is:
                * positions: Shape (len(frame_indices), 3): The xyz coordinates in nm.
                * time: shape (len(frame_indices), ): The time in ps.
                * unitcell_vectors: Shape (len(frame_indices), 3, 3): The unitcell vectors.
                * unitcell_info: Shape (len(frame_indices), 6), where [:, :3] are
                    the unitcell lengths in nm and [:, 3:] are the unitcell angles
                    in degree.

    """
    keys = ["coordinates", "time", "cell_lengths", "cell_angles"]
    if traj_num is not None:
        keys_with_num = [f"{k}_{traj_num}" for k in keys]
    else:
        keys_with_num = keys
    data = {}
    with h5py.File(traj_file, "r") as f:
        for k, out in zip(keys_with_num, keys):
            if k not in f and out not in f:
                data[out] = None
            elif k in f:
                data[out] = f[k][frame_indices]
            elif out in f:
                data[out] = f[out][frame_indices]
    unitcell_info = np.empty(shape=(len(frame_indices), 6), dtype="float32", order="C")
    unitcell_info[:, :3] = data["cell_lengths"]
    unitcell_info[:, 3:] = data["cell_angles"]
    unitcell_vectors = calc_bravais_box(unitcell_info)
    return data["coordinates"], data["time"], unitcell_vectors, unitcell_info


def load_xyz_from_h5(
    traj_file: str,
    frame_indices: np.ndarray,
    traj_num: Optional[int] = None,
) -> tuple[da.array, da.array, da.array, da.array]:
    """Loads xyz coordinates and unitcell info from a block in a .h5 file.

    Standard MDTraj h5 keys are:
        ['cell_angles', 'cell_lengths', 'coordinates', 'time', 'topology']

    Args:
        traj_file (str): The file to load.
        frame_indices (np.ndarray): An int array giving the positions to load.
        traj_num (int): Which traj num the output should be put to.

    Returns:
        tuple[da.array, da.array, da.array, da.array]: A four-tuple of dask
            arrays that contain dask delayeds. The order of these arrays is:
                * positions: Shape (len(frame_indices), 3): The xyz coordinates in nm.
                * time: shape (len(frame_indices), ): The time in ps.
                * unitcell_vectors: Shape (len(frame_indices), 3, 3): The unitcell vectors.
                * unitcell_info: Shape (len(frame_indices), 6), where [:, :3] are
                    the unitcell lengths in nm and [:, 3:] are the unitcell angles
                    in degree.

    """
    # Encodermap imports
    from encodermap.trajinfo.info_all import HDF5GroupWrite

    n_frames = len(frame_indices)
    with HDF5GroupWrite(traj_file) as f:
        if "topology" not in f.keys() and traj_num is not None:
            top = f.read_topology(f"topology_{traj_num}")
        else:
            top = f.read_topology("topology")
    n_atoms = top.n_atoms
    p, t, uv, ui = _load_xyz_from_h5(traj_file, frame_indices, traj_num)
    p = da.from_delayed(p, shape=(n_frames, n_atoms, 3), dtype="float32")
    t = da.from_delayed(t, shape=(n_frames,), dtype="float32")
    uv = da.from_delayed(uv, shape=(n_frames, 3, 3), dtype="float32")
    ui = da.from_delayed(ui, shape=(n_frames, 6), dtype="float32")
    return p, t, uv, ui


################################################################################
# Dask graph creation
################################################################################


@overload
def build_dask_xarray(
    featurizer: DaskFeaturizer,
    traj: Optional[SingleTraj],
    streamable: bool,
    return_delayeds: Literal[True],
) -> tuple[xr.Dataset, dict[str, xr.Variable]]: ...


@overload
def build_dask_xarray(
    featurizer: DaskFeaturizer,
    traj: Optional[SingleTraj],
    streamable: bool,
    return_delayeds: Literal[False],
) -> tuple[xr.Dataset, None]: ...


def build_dask_xarray(
    featurizer: DaskFeaturizer,
    traj: Optional[SingleTraj] = None,
    streamable: bool = False,
    return_delayeds: bool = False,
) -> tuple[xr.Dataset, Union[None, dict[str, xr.Variable]]]:
    """Builds a large dask xarray, which will be distributively evaluated.

    This class takes a `DaskFeaturizer` class, which contains a list of features.
    Every feature in this list contains enough information for the delayed functions
    to calculate the requested quantities when provided the xyz coordinates of the
    atoms, the unitcell vectors, and the unitcell infos as a Bravais matrix.

    Args:
        featurizer (DaskFeaturizer): An instance of the DaskFeaturizer.
        return_coordinates (bool): Whether to add this information:
            all_xyz, all_time, all_cell_lengths, all_cell_angles
            to the returned values. Defaults to False.
        streamable (bool): Whether to divide the calculations into one-frame
            blocks, which can then only be calculated when requested.

    Returns:
        Union[xr.Dataset, tuple[xr.Dataset, list[dask.delayed]]:
            When `return_coordinates` is False, only a xr.Dataset is returned.
            Otherwise, a tuple with a xr.Dataset and a sequence of dask.Delayed
            objects is returned.


    """
    # Imports
    # Encodermap imports
    from encodermap.loading.features import CustomFeature
    from encodermap.misc.xarray import (
        FEATURE_NAMES,
        make_dataarray,
        make_frame_CV_dataarray,
        make_position_dataarray,
    )
    from encodermap.trajinfo.trajinfo_utils import trajs_combine_attrs

    # definitions
    coordinates = {
        "coordinates": ["md_frame", "md_atom", "md_cart"],
        "time": ["md_frame"],
        "cell_lengths": ["md_frame", "md_length"],
        "cell_angles": ["md_frame", "md_angle"],
    }

    # pre-define blocks from the trajectories
    n_blocks = 10

    # append delayeds here:
    if return_delayeds:
        delayeds = {}
    else:
        delayeds = None

    assert len(featurizer.feat.active_features) > 0

    # collect the Datasets in this list
    DSs = []

    # if the dask featurizer contains an `EnsembleFeaturizer`, we can use `itertrajs()`
    if traj is None:
        if hasattr(featurizer.feat, "trajs"):
            iterable = featurizer.feat.trajs.itertrajs()
        else:
            iterable = enumerate([featurizer.feat.traj])
    else:
        iterable = enumerate([traj])

    # iter over trajs or just the one
    for i, traj in iterable:
        n_frames = len(traj.id)
        if not streamable:
            n_frames_per_block = n_frames // n_blocks
            blocks = [
                np.arange(i * n_frames_per_block, (i + 1) * n_frames_per_block)
                for i in range(n_blocks - 1)
            ]
            blocks.append(np.arange((n_blocks - 1) * n_frames_per_block, n_frames))
            # remove empty blocks
            blocks = list(filter(lambda x: x.size > 0, blocks))
        else:
            n_frames_per_block = 1
            blocks = [[i] for i in range(n_frames)]

        # collect multiple DataArrays here
        DAs = {}
        indexes = {}

        if delayeds is not None:
            xyz_traj = []
            time_traj = []
            lengths_traj = []
            angles_traj = []

        # distribute the loading to multiple workers
        for j, block in enumerate(blocks):
            # get the actual frame indices if the traj was sliced
            if traj.id.ndim == 2:
                frame_indices = traj.id[block, 1]
            else:
                frame_indices = traj.id[block]

            assert len(frame_indices) > 0, f"{frame_indices=}"
            xyz, time, unitcell_vector, unitcell_info = load_xyz(
                traj.traj_file, frame_indices, traj.traj_num
            )

            if delayeds is not None:
                unitcell_lengths = unitcell_info[:, :3]
                unitcell_angles = unitcell_info[:, 3:]
                xyz_traj.append(xyz)
                time_traj.append(time)
                lengths_traj.append(unitcell_lengths)
                angles_traj.append(unitcell_angles)

            # iterate over the features and let them use the traj information
            if hasattr(featurizer.feat, "trajs"):
                features = featurizer.feat.active_features[traj.top]
            else:
                features = featurizer.feat.active_features

            for k, feat in enumerate(features):
                # the name of the feature will be used for traceability
                if not isinstance(feat, CustomFeature) or not issubclass(
                    feat.__class__, CustomFeature
                ):
                    assert hasattr(feat, "dask_indices") and hasattr(
                        feat, "dask_transform"
                    ), (
                        f"For `feature.transform()` to be acceptable as delayed, "
                        f"the feature needs to implement the `dask_indices` property "
                        f"and `dask_transform` staticmethod. The feature {feat} has "
                        f"this these methods and attributes "
                        f"{[a for a in feat.__dir__() if not a.startswith('_')]}"
                    )
                assert feat.delayed, (
                    f"The feature {feat} was not altered to return a delayed "
                    f"transform. Please read up in `encodermap.DaskFeaturizer` how "
                    f"to make features work with dask delayed."
                )

                # decide on the name
                try:
                    name = FEATURE_NAMES[feat.name]
                except (KeyError, AttributeError):
                    if hasattr(feat, "name"):
                        if isinstance(feat.name, str):
                            name = feat.name
                            if "mdtraj.trajectory" in feat.name.lower():
                                feat.name = feat.__class__.__name__
                                name = feat.__class__.__name__
                        else:
                            name = feat.__class__.__name__
                            feat.name = name
                    else:
                        name = feat.__class__.__name__
                        if name == "CustomFeature":
                            name = feat.describe()[0].split()[0]
                        feat.name = name

                # the feature length is given by the describe() of the feature
                if callable(feat.describe()):
                    feat_length = len([i for i in feat.describe()(traj.top)])
                else:
                    feat_length = len(feat.describe())

                # dynamically populate kwargs with feature settings
                kwargs = {"indexes": getattr(feat, feat.dask_indices)}
                if feat._use_periodic:
                    kwargs["periodic"] = feat.periodic
                if feat._use_angle:
                    kwargs["deg"] = feat.deg
                    kwargs["cossin"] = feat.cossin
                # if feat._use_omega:
                #     kwargs["omega"] = feat.omega
                if hasattr(feat, "_nonstandard_transform_args"):
                    for k in feat._nonstandard_transform_args:
                        if not hasattr(feat, k):
                            kwargs[k] = None
                        else:
                            kwargs[k] = getattr(feat, k)
                a = da.from_delayed(
                    feat.dask_transform(
                        **kwargs,
                        xyz=xyz,
                        unitcell_vectors=unitcell_vector,
                        unitcell_info=unitcell_info,
                    ),
                    shape=(len(frame_indices), feat_length),
                    dtype="float32",
                )

                if hasattr(feat, "deg"):
                    deg = feat.deg
                else:
                    deg = None

                if (
                    feat.name
                    in ["AllCartesians", "CentralCartesians", "SideChainCartesians"]
                    or feat.atom_feature
                ):
                    a = da.reshape(a, (len(frame_indices), -1, 3))
                    a = da.expand_dims(a, axis=0)
                    if hasattr(featurizer, "indices_by_top"):
                        feat.indexes = featurizer.indices_by_top[traj.top][feat.name]
                    dataarray, ind_dataarray = make_position_dataarray(
                        feat.describe(),
                        traj[block],
                        name,
                        a,
                        deg=deg,
                        feat=feat,
                    )
                else:
                    a = da.expand_dims(a, axis=0)
                    if feat._dim == 1:
                        dataarray, ind_dataarray = make_frame_CV_dataarray(
                            feat.describe(),
                            traj[block],
                            name,
                            a,
                            deg=deg,
                            feat=feat,
                        )
                    else:
                        if hasattr(featurizer, "indices_by_top"):
                            feat.indexes = featurizer.indices_by_top[traj.top][
                                feat.name
                            ]
                        dataarray, ind_dataarray = make_dataarray(
                            feat.describe(),
                            traj[block],
                            name,
                            a,
                            deg=deg,
                            feat=feat,
                        )

                assert dataarray.size > 0, (
                    f"Dataarray created for feature {feat} provided with "
                    f"traj {traj} at frame indices {block} did not contain "
                    f"any data."
                )

                # append the DataArray to the DAs dictionary
                DAs.setdefault(name, []).append(dataarray)
                if ind_dataarray is not None:
                    indexes.setdefault(name + "_feature_indices", []).append(
                        ind_dataarray
                    )
                else:
                    indexes[name + "_feature_indices"] = [None]

        # after every traj, we combine the datasets
        for key, value in DAs.items():
            DAs[key] = xr.concat(
                DAs[key],
                "frame_num",
                combine_attrs=trajs_combine_attrs,
            )
            # we only need any component from the indexes but make sure that
            # they are homogeneous. Every block of a traj should return
            # the same index array, as they don't depend on frame data
            if indexes[key + "_feature_indices"][0] is None:
                assert all(
                    [i is None for i in indexes[key + "_feature_indices"][1:]]
                ), (
                    f"Got an inhomogeneous result for indexes for feature {feat=} "
                    f"at {frame_indices=} {indexes=}"
                )
            indexes[key + "_feature_indices"] = indexes.pop(key + "_feature_indices")[0]

        # combine data per traj
        DAs_and_indexes = DAs | indexes
        DAs_and_indexes = {k: v for k, v in DAs_and_indexes.items() if v is not None}
        try:
            ds = xr.Dataset(
                DAs_and_indexes,
                attrs=trajs_combine_attrs(
                    [v.attrs if v.size > 0 else {} for v in DAs_and_indexes.values()]
                ),
            )
        except xr.core.merge.MergeError as e:
            raise Exception(f"{indexes=}") from e
        DSs.append(ds)

        # and add to the delayeds if needed
        if delayeds is not None:
            for (coord, dims), data, unit in zip(
                coordinates.items(),
                [xyz_traj, time_traj, lengths_traj, angles_traj],
                ["nanometers", "picoseconds", "nanometers", "degrees"],
            ):
                name = f"{coord}_{traj.traj_num}"
                delayeds[name] = xr.Variable(
                    dims=[f"{d}_{traj.traj_num}" for d in dims],
                    data=da.concatenate(data),
                    attrs={"units": unit.encode("utf-8")},
                )

    # make a large dataset out of this
    ds = xr.concat(
        DSs,
        data_vars="all",
        # compat="broadcast_equals",
        # coords="all",
        # join="outer",
        dim="traj_num",
        fill_value=np.nan,
        combine_attrs=trajs_combine_attrs,
    )
    assert ds, (
        f"Concatenation of chunked datasets yielded empty dataset.\n"
        f"{DSs=}\n\n{DAs_and_indexes=}"
    )

    return ds, delayeds

# -*- coding: utf-8 -*-
# encodermap/misc/xarray.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade, Tobias Lemke
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
"""EncoderMap's xarray manipulation functions.

EncoderMap uses xarray datasets to save CV data alongside with trajectory data.
These functions implement creation of such xarray datasets.

"""
################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import itertools
import warnings

# Third Party Imports
import numpy as np
from optional_imports import _optional_import

# Encodermap imports
from encodermap.misc.misc import FEATURE_NAMES


################################################################################
# Optional Imports
################################################################################


xr = _optional_import("xarray")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from numbers import Number
from typing import TYPE_CHECKING, Optional, Union, overload


if TYPE_CHECKING:
    # Third Party Imports
    import xarray as xr

    # Encodermap imports
    from encodermap.loading.features import AnyFeature
    from encodermap.loading.featurizer import Featurizer
    from encodermap.trajinfo.info_single import SingleTraj


################################################################################
# Globals
################################################################################


__all__: list[str] = [
    "construct_xarray_from_numpy",
    "unpack_data_and_feature",
]


################################################################################
# Functions
################################################################################


def _get_indexes_from_feat(f: "AnyFeature", traj: "SingleTraj") -> np.ndarray:
    """Returns the indices of this feature.

    This can be useful for later to redo some analysis or use tensorflow
    to implement the geometric operations in the compute graph.

    Note:
        Due to some PyEMMA legacy code. Sometimes the indices are called indexes.
        Furthermore, some PyEMMA feature give the indices different names, like
        'group_definitions'. Here, indices are integer values that align with
        the atomic coordinate of a trajectory or ensemble of trajectories with
        shared topology.

    Args:
        f (AnyFeature): The feature to extract the indices from. This has to
            be a subclass of :class:`encodermap.loading.features.Feature`.
        traj (SingleTraj): This argument has to be provided for the RMSD
            features. We make it mandatory for the other feature types also
            to keep the input consistent.

    Returns:
        np.ndarray: An integer array describing the atoms, this feature uses
            to compute the collevtive variables.

    """
    # Local Folder Imports
    from ..loading.features import (
        CustomFeature,
        GroupCOMFeature,
        MinRmsdFeature,
        ResidueCOMFeature,
    )

    try:
        return f.indexes
    except AttributeError as e:
        for key in f.__dir__():
            if "inde" in key:
                return np.asarray(f.__dict__[key])
        if isinstance(f, (GroupCOMFeature, ResidueCOMFeature)):
            return f.group_definitions
        if isinstance(f, MinRmsdFeature):
            if f.atom_indices is None:
                return np.arange(traj.n_atoms)
            else:
                return f.atom_indices
        if isinstance(f, CustomFeature):
            descr = f.describe()
            if any(["CustomFeature" in i for i in descr]):
                return np.asarray(
                    [f"{d.split()[0]} INDEX {i}" for i, d in enumerate(descr)]
                )
            else:
                raise Exception(
                    f"Can't decide on indexes for this custom feature:, "
                    f"{f.__class__.__name__=} {f.describe()[:2]=}"
                ) from e
        else:
            raise e


def _cast_to_int_maybe(a: np.ndarray) -> np.ndarray:
    """Casts a np.array to int, if possible.

    Args:
        a (np.ndarray): The array.

    Returns:
        np.ndarray: The output, which can be int.
    """
    if not np.any(np.mod(a, 1)):
        return a.astype(np.int32)
    return a


def construct_xarray_from_numpy(
    traj: SingleTraj,
    data: np.ndarray,
    name: str,
    deg: bool = False,
    labels: Optional[list[str]] = None,
    check_n_frames: bool = False,
) -> xr.DataArray:
    """Constructs a `xarray.DataArray` from a numpy array.

    Three cases are recognized:
        * The input array in data has ndim == 2. This kind of feature/CV is a
            per-frame feature, like the membership to clusters. Every frame of
            every trajectory is assigned a single value (most often int values).
        * The input array in data has ndim == 3: This is also a per-frame
            feature/CV, but this time every frame is characterized by a series
            of values. These values can be dihedral angles in the backbone
            starting from the protein's N-terminus to the C-terminus, or
            pairwise distance features between certain atoms. The xarray
            datarray constructed from this kind of data will have a label
            dimension that will either contain generic labels like
            'CUSTOM_FEATURE FEATURE 0' or labels defined by the featurizer,
            such as 'SIDECHAIN ANGLE CHI1 OF RESIDUE 1LYS'.
        * The input array in data has ndim == 4. Here, the same feature/CV is
            duplicated for the protein's atoms. Besides the XYZ coordinates of
            the atoms, no other CVs should fall into this case. The labels will be
            2-dimensional with 'POSITION OF ATOM H1 IN RESIDUE 1LYS' in
            dimension 0 and either 'X', 'Y' or 'Z' in dimension 1.

    Args:
        traj (em.SingleTraj): The trajectory we want to create the
            `xarray.DataArray` for.
        data (np.ndarray): The numpy array we want to create the data from.
            Note that the data passed into this function should be expanded
            by `np.expand_dim(a, axis=0)`, so to add a new axis to the complete
            data containing the trajectories of a trajectory ensemble.
        name (str): The name of the feature. This can be chosen freely. Names
            like 'central_angles', 'backbone_torsions' would make the most sense.
        deg (bool): Whether provided data is in deg or radians.
        labels (Optional[list]): If you have specific labels for your CVs in
            mind, you can overwrite the generic 'CUSTOM_FEATURE FEATURE 0'
            labels by providing a list for this argument. If None is provided,
            generic names will be given to the features. Defaults to None.
        check_n_frames (bool): Whether to check whether the number of frames in
            the trajectory matches the len of the data in at least one
            dimension. Defaults to False.

    Returns:
        xarray.DataArray: An `xarray.DataArray`.

    Examples:
        >>> import encodermap as em
        >>> import numpy as np
        >>> from encodermap.misc.xarray import construct_xarray_from_numpy
        >>> # load file from RCSB and give it traj num to represent it in a
        >>> # potential trajectory ensemble
        >>> traj = em.load('https://files.rcsb.org/view/1GHC.pdb', traj_num=1)
        >>> # single trajectory needs to be expanded into 'trajectory' axis
        >>> z_coordinate = np.expand_dims(traj.xyz[:,:,0], 0)
        >>> da = construct_xarray_from_numpy(traj, z_coordinate, 'z_coordinate')
        >>> print(da.coords['Z_COORDINATE'].values[:2])
        ['Z_COORDINATE FEATURE 0' 'Z_COORDINATE FEATURE 1']
        >>> print(da.coords['traj_num'].values)
        [1]
        >>> print(da.attrs['time_units'])
        ps

    """
    if check_n_frames:
        if not any(s == traj.n_frames for s in data.shape):
            raise Exception(
                f"Can't add CV with name '{name}' to trajectory. The trajectory has "
                f"{traj.n_frames} frames. The data has a shape of {data.shape} "
                f"No dimension of data matches the frames of the trajectory."
            )
    if traj.backend == "no_load":
        with_time = False
    else:
        with_time = True
    if data.ndim == 2:
        if labels is None:
            labels = [f"{name.upper()} FEATURE"]
        da, _ = make_frame_CV_dataarray(
            labels, traj, name, data, deg=deg, with_time=with_time
        )
    elif data.ndim == 3:
        if labels is None:
            labels = [f"{name.upper()} FEATURE {i}" for i in range(data.shape[-1])]
        da, _ = make_dataarray(labels, traj, name, data, deg=deg, with_time=with_time)
    elif data.ndim == 4:
        if labels is None:
            labels = [f"{name.upper()} FEATURE {i}" for i in range(data.shape[-2])]
        da = make_position_dataarray_from_numpy(
            labels,
            traj,
            name,
            data,
            deg=deg,
            with_time=with_time,
        )
    else:
        raise Exception(
            f"The provided data has a dimensionality of {data.ndim}, but only 2, 3 and 4 are supported."
        )
    return da


def unpack_data_and_feature(
    feat: Featurizer,
    traj: SingleTraj,
    input_data: np.ndarray,
) -> xr.Dataset:
    """Makes a `xarray.Dataset` from data and a featurizer.

    Usually, if you add multiple features to a featurizer, they are
    stacked along the feature axis. Let's say, you have a trajectory with 20 frames
    and 3 residues. If you add the Ramachandran angles, you get 6 features (3xphi, 3xpsi).
    If you then also add the end-to-end distance as a feature, the data returned by
    the featurizer will have the shape (20, 7). This function returns the correct indices,
    so that iteration of zip(Featurizer.active_features, indices) will yield the
    correct results.

    Args:
        feat (encodermap.loading.Featurizer): An instance of the currently used `encodermap.loading.Featurizer`.
        traj (encodermap.trajinfo.SingleTraj): An instance of `encodermap.SingleTraj`, that the data
            in `input_data` was computed from.
        input_data (np.ndarray): The data, as returned from PyEMMA.

    Returns:
        xarray.Dataset: An `xarray.Dataset` with all features in a nice format.

    """
    # Local Folder Imports
    from ..trajinfo.trajinfo_utils import trajs_combine_attrs

    # this needs to be done, because pyemma concatenates the data
    # along the feature axis
    indices = get_indices_by_feature_dim(feat, traj, input_data.shape)

    DAs = {}
    indexes = {}
    for f, ind in zip(feat.features, indices):
        data = input_data[:, ind]
        if data.shape[-1] == 1 and f._dim != 1:
            data = data.squeeze(-1)

        # decide on the name
        try:
            name = FEATURE_NAMES[f.name]
        except (KeyError, AttributeError):
            if hasattr(f, "name"):
                if isinstance(f.name, str):
                    name = f.name
                    if "mdtraj.trajectory" in f.name.lower():
                        f.name = f.__class__.__name__
                        name = f.__class__.__name__
                else:
                    name = f.__class__.__name__
                    f.name = name
            else:
                name = f.__class__.__name__
                if name == "CustomFeature":
                    name = f.describe()[0].split()[0]
                f.name = name
        assert data.shape[1] == f._dim

        if data.shape[0] != len(traj):
            if traj.index == (None,):
                raise Exception(
                    "Shape of provided data does not fit traj. Traj "
                    f"has {traj.n_frames=} {len(traj)=}. Data has shape {data.shape}"
                )
            else:
                data = data[traj.index].squeeze(0)
                assert data.shape[0] == len(traj), f"{data.shape=}, {len(traj)=}"
        if name in DAs:
            name = (
                name
                + f"_{len(list(filter(lambda x: True if name in x else False, list(DAs.keys()))))}"
            )
        if hasattr(f, "deg"):
            deg = f.deg
        else:
            deg = None

        if (
            f.name in ["AllCartesians", "CentralCartesians", "SideChainCartesians"]
            or f.atom_feature
        ):
            data = data.reshape(len(traj), -1, 3)
            data = np.expand_dims(data, axis=0)
            if hasattr(feat, "indices_by_top"):
                f.indexes = feat.indices_by_top[traj.top][f.name]
            DAs[name], indexes[name + "_feature_indices"] = make_position_dataarray(
                f.describe(), traj, name, data, deg=deg, feat=f
            )
        else:
            data = np.expand_dims(data, axis=0)
            if f._dim == 1:
                DAs[name], indexes[name + "_feature_indices"] = make_frame_CV_dataarray(
                    f.describe(),
                    traj,
                    name,
                    data,
                    deg=deg,
                    feat=f,
                )
            else:
                if hasattr(feat, "indices_by_top"):
                    f.indexes = feat.indices_by_top[traj.top][f.name]
                DAs[name], indexes[name + "_feature_indices"] = make_dataarray(
                    f.describe(), traj, name, data, deg=deg, feat=f
                )
        if indexes[name + "_feature_indices"] is None:
            del indexes[name + "_feature_indices"]
    DAs_and_indxes = DAs | indexes
    attrs = []
    for i in DAs_and_indxes.values():
        attrs.append(i.attrs)
    try:
        ds = xr.Dataset(DAs_and_indxes, attrs=trajs_combine_attrs(attrs))
    except ValueError as e:
        raise Exception(f"{DAs_and_indxes.keys()=}") from e
    return ds


def make_frame_CV_dataarray(
    labels: list[str],
    traj: SingleTraj,
    name: str,
    data: np.ndarray,
    deg: Union[None, bool],
    with_time: bool = True,
    frame_indices: Optional[np.ndarray] = None,
    labels2: Optional[list[str]] = None,
    feat: Optional["AnyFeature"] = None,
) -> tuple[xr.DataArray, Union[None, xr.DataArray]]:
    """Make a DataArray from a frame CV feature.

    A normal features yields multiple values per trajectory frame (e.g. the
    backbone dihedral angles give 2 * n_residues features per frame). A frame
    CV feature is just a single value for a trajectory. Examples are:
        * The distance between a binding pocket and a ligand.
        * The volume of the unitcell.
        * Whether a molecule is in state 1 or 2, could be described as a binary frame CV features.

    This method has additional logic. If `data` has a shape of (1, n_frames, 1) it
    is left as is. If it has a shape of (1, n_frames), it will be expanded on the -1 axis to
    shape (1, n_frames, 1).

    Note:
        Please make sure that the input data conforms to the nm, ps, rad coordinates.

    Args:
        labels (list[str]): The labels, that specify the `CV_num` dimension. This
            requires the expression `len(labels == data.shape[2]` to be True. If you
            build the DataArray from a feature. The `labels` argument usually will
            be `feature.describe()`.
        traj (encodermap.SingleTraj): An `encodermap.SingleTraj` trajectory.
            Why `SingleTraj` and not `TrajEnsemble`? That is, because in EncoderMap,
            an xarray.DataArray always represents one feature, of one trajectory.
            A trajectory can have multiple features, which is represented as an
            `xarray.Dataset`. For that, the function `unpack_data_and_feature` is used.
            The `TrajEnsemble` trajectory does not even have its own `xarray.Dataset`. This
            dataset is created ad hoc, by merging the datasets of the trajectories along
            the `traj_num` axis.
        name (str): The name of the feature. This name will be used to group similar
            features in the large `xarray.Dataset`s of trajectory ensembles. If you
            construct this DataArray from a feature, you can either use `feature.name`,
            or `feature.__class__.__name__`.
        data (Union[np.ndarray, dask.array]): The data to fill the DataArray with.
        deg (Union[None, bool]): Whether the provided data is in deg or radians.
            If None is provided, it will not be included in the attrs.
        with_time (Optional[Union[bool, np.ndarray]]): Whether to add the time of
            the frames to the `frame_num` axis. Can be either True, or False, or a
            `np.ndarray`, in which case the data in this array will be used. Defaults to True.
        frame_indices (Optional[np.ndarray]) The indices of the trajectory on disk.
            This can come in handy, if the trajectory is loaded out of-memory and
            distributed, in which case, MDAnalysis is used to load data.
            MDTraj does never load the trajectory coordinates. This the attribute
            `_original_frame_indices` in the trajectory will be empty. However, as
            the distributed task did already assign frames to the workers, we have
            this number, we just need to use it. If set to None `_original_frame_indices`
            will be used. Otherwise, the `np.ndarray` provided here, will be used.
        labels2 (Optional[list[str]]): Optional labels to supersede the labels in `labels`.
            This can be especially useful for trajectory ensembles, which might differ
            in a lot of places (i.e., single amino acid exchange). If the labels are
            too strict (LYS1 ATOM1 C), the data of two slightly different proteins
            cannot be concatenated. If the labels are generic (AA1 ATOM1), this can
            be done. Defaults to None.

    Returns:
        xarray.DataArray: The resulting DataArray.

    """
    if data.ndim == 3:
        pass
    elif data.ndim == 2:
        data = np.expand_dims(data, -1)
    # data = _cast_to_int_maybe(data)

    return make_dataarray(
        labels, traj, name, data, deg, with_time, frame_indices, labels2, feat
    )


def make_position_dataarray_from_numpy(
    atoms, traj, name, data, deg=None, with_time=True
):
    """Same as :func:`make_dataarray`, but with higher feature dimension."""
    attrs = {
        "length_units": "nm",
        "time_units": "ps",
        "full_path": traj.traj_file,
        "topology_file": traj.top_file,
        "feature_axis": "ATOM",
    }
    if deg is not None:
        if deg:
            attrs["angle_units"] = "deg"
        else:
            attrs["angle_units"] = "rad"
    frame_indices = traj.id
    if frame_indices.ndim > 1:
        frame_indices = frame_indices[:, 1]
    da = xr.DataArray(
        data,
        coords={
            "traj_num": ("traj_num", np.asarray([traj.traj_num])),
            "traj_name": ("traj_num", np.asarray([traj.basename])),
            "frame_num": ("frame_num", frame_indices),
            "ATOM": ("ATOM", np.asarray(atoms)),
            "COORDS": np.array(["POSITION X", "POSITION Y", "POSITION Z"]),
        },
        dims=["traj_num", "frame_num", "ATOM", "COORDS"],
        name=name,
        attrs=attrs,
    )
    if with_time:
        da = da.assign_coords(time=("frame_num", traj.time))
    return da


@overload
def make_position_dataarray(
    labels: list[str],
    traj: SingleTraj,
    name: str,
    data: np.ndarray,
    deg: Union[None, bool] = None,
    with_time: bool = True,
    frame_indices: Optional[np.ndarray] = None,
    labels2: Optional[list[str]] = None,
    feat: Optional["AnyFeature"] = None,
) -> tuple[xr.DataArray, None]: ...


@overload
def make_position_dataarray(
    labels: list[str],
    traj: SingleTraj,
    name: str,
    data: np.ndarray,
    deg: Union[None, bool] = None,
    with_time: bool = True,
    frame_indices: Optional[np.ndarray] = None,
    labels2: Optional[list[str]] = None,
    feat: Optional["AnyFeature"] = "AnyFeature",
) -> tuple[xr.DataArray, xr.DataArray]: ...


def make_position_dataarray(
    labels: list[str],
    traj: SingleTraj,
    name: str,
    data: np.ndarray,
    deg: Union[None, bool] = None,
    with_time: bool = True,
    frame_indices: Optional[np.ndarray] = None,
    labels2: Optional[list[str]] = None,
    feat: Optional["AnyFeature"] = None,
) -> tuple[xr.DataArray, Union[None, xr.DataArray]]:
    """Creates DataArray belonging to cartesian positions.

    Similar to `make_datarray`, but the shapes are even larger. As every atom
    contributes 3 coordinates (x, y, z) to the data, the shape of the returned
    DataArray is (1, no_of_frames, no_of_atoms_considered, 3).

    Note:
        Please make sure that the input data conforms to the nm, ps, rad coordinates.

    Args:
        labels (list[str]): The labels, that specify the `CV_num` dimension. This
            requires the expression `len(labels == data.shape[2]` to be True. If you
            build the DataArray from a feature. The `labels` argument usually will
            be `feature.describe()`.
        traj (encodermap.SingleTraj): An `encodermap.SingleTraj` trajectory.
            Why `SingleTraj` and not `TrajEnsemble`? That is, because in EncoderMap,
            an xarray.DataArray always represents one feature, of one trajectory.
            A trajectory can have multiple features, which is represented as an
            `xarray.Dataset`. For that, the function `unpack_data_and_feature` is used.
            The `TrajEnsemble` trajectory does not even have its own `xarray.Dataset`. This
            dataset is created ad hoc, by merging the datasets of the trajectories along
            the `traj_num` axis.
        name (str): The name of the feature. This name will be used to group similar
            features in the large `xarray.Dataset`s of trajectory ensembles. If you
            construct this DataArray from a feature, you can either use `feature.name`,
            or `feature.__class__.__name__`.
        data (Union[np.ndarray, dask.array]): The data to fill the DataArray with.
        deg (bool): Whether the provided data is in deg or radians.
            If None is provided, it will not be included in the attrs. Defaults to None.
        with_time (Union[bool, np.ndarray]): Whether to add the time of
            the frames to the `frame_num` axis. Can be either True, or False, or a
            `np.ndarray`, in which case the data in this array will be used. Defaults to True.
        frame_indices (Optional[np.ndarray]) The indices of the trajectory on disk.
            This can come in handy, if the trajectory is loaded out of-memory and
            distributed, in which case, MDAnalysis is used to load data.
            MDTraj does never load the trajectory coordinates. This the attribute
            `_original_frame_indices` in the trajectory will be empty. However, as
            the distributed task did already assign frames to the workers, we have
            this number, we just need to use it. If set to None `_original_frame_indices`
            will be used. Otherwise, the `np.ndarray` provided here, will be used.
        labels2 (Optional[list[str]]): Optional labels to supersede the labels in `labels`.
            This can be especially useful for trajectory ensembles, which might differ
            in a lot of places (i.e., single amino acid exchange). If the labels are
            too strict (LYS1 ATOM1 C), the data of two slightly different proteins
            cannot be concatenated. If the labels are generic (AA1 ATOM1), this can
            be done. Defaults to None.

    Returns:
        tuple[xr.DataArray, Union[None, xr.DataArray]]: The resulting dataarrays,
            as a tuple of (data, indices) or (data, None), if indices were not
            requested.

    """
    atom_axis: str = "ATOM"
    if feat is not None:
        if feat.__class__.__name__ == "SideChainCartesians":
            atom_axis = "SIDEATOM"
        if feat.__class__.__name__ == "AllCartesians":
            atom_axis = "ALLATOM"
    attrs = {
        "length_units": "nm",
        "time_units": "ps",
        "full_path": traj.traj_file,
        "topology_file": traj.top_file,
        "feature_axis": atom_axis,
    }
    if feat is not None:
        indices = _get_indexes_from_feat(feat, traj)
    else:
        indices = None
    if deg is not None:
        if deg:
            attrs["angle_units"] = "deg"
        else:
            attrs["angle_units"] = "rad"
    if frame_indices is None:
        frame_indices = traj.id
        if frame_indices.ndim > 1:
            frame_indices = frame_indices[:, 1]

    if labels2 is not None:
        labels = labels2
    else:
        labels = [_[11:].lstrip(" ") for _ in labels[::3]]

    coords = {
        "traj_num": ("traj_num", np.asarray([traj.traj_num])),
        "traj_name": ("traj_num", np.asarray([traj.basename])),
        "frame_num": ("frame_num", frame_indices),
        atom_axis: (atom_axis, np.asarray(labels)),
        "COORDS": np.array(["POSITION X", "POSITION Y", "POSITION Z"]),
    }
    da = xr.DataArray(
        data,
        coords=coords,
        dims=["traj_num", "frame_num", atom_axis, "COORDS"],
        name=name.upper(),
        attrs=attrs,
    )
    if indices is not None:
        if len(indices) // 3 == data.shape[-2] and name.startswith("CustomFeature"):
            warnings.warn(
                f"Can't find good labels for the feature {name}. The "
                f"data from the feature's transform has shape {data[0].shape}. "
                f"The indices have indices length {len(indices)}. I will not include the "
                f"indices of this feature in the dataset. However, the output data "
                f"will be there."
            )
            indices = None
        elif len(indices) != data.shape[-2]:
            indices_copy = indices.copy()
            indices = np.full((len(labels),), np.nan, float)
            indices[: len(indices_copy)] = indices_copy

        if indices is not None:
            coords = {
                "traj_num": ("traj_num", np.asarray([traj.traj_num])),
                "traj_name": ("traj_num", np.asarray([traj.basename])),
                atom_axis: (atom_axis, np.asarray(labels)),
            }
            ind_da = xr.DataArray(
                np.expand_dims(indices, 0),
                coords=coords,
                dims=["traj_num", atom_axis],
                name=name.upper(),
                attrs=attrs | {"feature_axis": atom_axis},
            )
        else:
            ind_da = None
    else:
        ind_da = None
    if isinstance(with_time, bool):
        if with_time:
            da = da.assign_coords(time=("frame_num", traj.time))
    else:
        da = da.assign_coords(time=("frame_num", with_time))
    return da, ind_da


@overload
def make_dataarray(
    labels: list[str],
    traj: SingleTraj,
    name: str,
    data: np.ndarray,
    deg: Union[None, bool],
    with_time: bool = True,
    frame_indices: Optional[np.ndarray] = None,
    labels2: Optional[list[str]] = None,
    feat: Optional["AnyFeature"] = None,
) -> tuple[xr.DataArray, None]: ...


@overload
def make_dataarray(
    labels: list[str],
    traj: SingleTraj,
    name: str,
    data: np.ndarray,
    deg: Union[None, bool],
    with_time: bool = True,
    frame_indices: Optional[np.ndarray] = None,
    labels2: Optional[list[str]] = None,
    feat: Optional["AnyFeature"] = "AnyFeature",
) -> tuple[xr.DataArray, xr.DataArray]: ...


def make_dataarray(
    labels: list[str],
    traj: SingleTraj,
    name: str,
    data: np.ndarray,
    deg: Union[None, bool],
    with_time: bool = True,
    frame_indices: Optional[np.ndarray] = None,
    labels2: Optional[list[str]] = None,
    feat: Optional["AnyFeature"] = None,
) -> tuple[xr.DataArray, Union[None, xr.DataArray]]:
    """Creates a DataArray belonging to a feature.

    The shapes are a bit different from what most people might be used to. As
    EncoderMap was meant to work with ensembles of trajectories, the data is usually
    shaped as (traj_num, frame_num, CV_num), or even (traj_num, frame_num, atom_num, 3).

    The `xarray.DataArray` that is returned by this function reflects this. As a
    DataArray is attributed to a single trajectory, the first shape will always be 1.
    So for a collective variable, that describes n features, the shape of the returned
    datarray will be (1, n_frames, n). Combining multiple trajectories, the first number
    can increase.

    Note:
        Please make sure that the input data conforms to the nm, ps, rad coordinates.

    Args:
        labels (list[str]): The labels, that specify the `CV_num` dimension. This
            requires the expression `len(labels == data.shape[2]` to be True. If you
            build the DataArray from a feature. The `labels` argument usually will
            be `feature.describe()`.
        traj (encodermap.SingleTraj): An `encodermap.SingleTraj` trajectory.
            Why `SingleTraj` and not `TrajEnsemble`? That is, because in EncoderMap,
            an xarray.DataArray always represents one feature, of one trajectory.
            A trajectory can have multiple features, which is represented as an
            `xarray.Dataset`. For that, the function `unpack_data_and_feature` is used.
            The `TrajEnsemble` trajectory does not even have its own `xarray.Dataset`. This
            dataset is created ad hoc, by merging the datasets of the trajectories along
            the `traj_num` axis.
        name (str): The name of the feature. This name will be used to group similar
            features in the large `xarray.Dataset`s of trajectory ensembles. If you
            construct this DataArray from a feature, you can either use `feature.name`,
            or `feature.__class__.__name__`.
        data (Union[np.ndarray, dask.array]): The data to fill the DataArray with.
        deg (Optional[bool]): Whether provided data is in deg or radians. If None
            is provided, the angle_units will not appear in the attributes.
            Defaults to None.
        with_time (Optional[Union[bool, np.ndarray]]): Whether to add the time of
            the frames to the `frame_num` axis. Can be either True, or False, or a
            `np.ndarray`, in which case the data in this array will be used. Defaults to True.
        frame_indices (Optional[np.ndarray]) The indices of the trajectory on disk.
            This can come in handy if the trajectory is loaded out of-memory and
            distributed, in which case, MDAnalysis is used to load data.
            MDTraj does never load the trajectory coordinates. This the attribute
            `_original_frame_indices` in the trajectory will be empty. However, as
            the distributed task did already assign frames to the workers, we have
            this number, we just need to use it. If set to None `_original_frame_indices`
            will be used. Otherwise, the `np.ndarray` provided here, will be used.
        labels2 (Optional[list[str]]): Optional labels to supersede the labels in `labels`.
            This can be especially useful for trajectory ensembles, which might differ
            in a lot of places (i.e., single amino acid exchange). If the labels are
            too strict (LYS1 ATOM1 C), the data of two slightly different proteins
            cannot be concatenated. If the labels are generic (AA1 ATOM1), this can
            be done. Defaults to None.

    Returns:
        xarray.DataArray: The resulting DataArray.

    """
    if feat is not None:
        indices = _get_indexes_from_feat(feat, traj)
    else:
        indices = None
    attrs = {
        "length_units": "nm",
        "time_units": "ps",
        "full_path": traj.traj_file,
        "topology_file": traj.top_file,
        "feature_axis": name.upper(),
    }
    if deg is not None:
        if deg:
            attrs["angle_units"] = "deg"
        else:
            attrs["angle_units"] = "rad"

    if frame_indices is None:
        frame_indices = traj.id
        if frame_indices.ndim > 1:
            frame_indices = frame_indices[:, 1]

    if labels2 is not None:
        labels = labels2

    if callable(labels):
        labels = [l for l in labels(traj.top)]
    else:
        if len(labels) > data.shape[-1]:
            warnings.warn(
                f"Provided labels {labels[:5]} are greater {len(labels)} than the "
                f"data {data.shape[-1]}. I will use integers to label the feature axis."
            )
            labels = np.arange(data.shape[-1])

    coords = {
        "traj_num": ("traj_num", np.asarray([traj.traj_num])),
        "traj_name": ("traj_num", np.asarray([traj.basename])),
        "frame_num": ("frame_num", frame_indices),
        name.upper(): np.asarray(labels),
    }

    da = xr.DataArray(
        data,
        coords=coords,
        dims=["traj_num", "frame_num", name.upper()],
        name=name,
        attrs=attrs,
    )

    ind_da = xr.DataArray()
    if indices is not None:
        index_labels = np.asarray(labels)
        coords = {
            "traj_num": ("traj_num", np.asarray([traj.traj_num])),
            "traj_name": ("traj_num", np.asarray([traj.basename])),
            name.upper(): index_labels,
        }
        try:
            indices = np.asarray(indices)
            inhomogeneous_shape = False
        except ValueError as e:
            if "inhomogeneous" in str(e):
                inhomogeneous_shape = True
            else:
                raise e
        # if the shape of indices is inhomogeneous, we can't put them into a DataArray
        if inhomogeneous_shape or any(
            [i in feat.__class__.__name__.lower() for i in ["groupcom", "residuecom"]]
        ):
            warnings.warn(
                f"The feature {name} will not produce a '{name}_feature_indices' "
                f"xr.DataArray. Its indices contain either inhomogeneous shapes "
                f"({indices}) or it is a center-of-mass (COM) feature. These "
                f"features are excluded from providing indices. "
                f"I will put a string representation of these "
                f"indices into the DataArray's `attrs` dictionary. Thus, it "
                f"can still be saved to disk and viewed, but it will be "
                f"a string and not a sequence of integers."
            )
            da.attrs |= {f"{name}_feature_indices": str(indices)}
        # special case: PyEMMA feature with cartesians
        elif len(indices) == data.shape[-1] // 3 and "selection" in feat.name.lower():
            indices = np.vstack([indices for i in range(3)]).flatten(order="F")
            ind_da = xr.DataArray(
                np.expand_dims(np.expand_dims(indices, 0), -1),
                coords=coords
                | {
                    "ATOM_NO": [0],
                },
                dims=["traj_num", name.upper(), "ATOM_NO"],
                name=name.upper(),
                attrs=attrs | {"feature_axis": name.upper()},
            )
        # special case: RMSD features
        # rmsd features provide their alignment atoms
        elif "rmsd" in feat.__class__.__name__.lower():
            ind_da = xr.DataArray(
                np.expand_dims(np.expand_dims(indices, 0), 0),
                coords=coords
                | {
                    "RMSD_ATOM_NO": indices,
                },
                dims=["traj_num", name.upper(), "RMSD_ATOM_NO"],
                name=name.upper(),
                attrs=attrs | {"feature_axis": name.upper()},
            )
        # special case: Align feature
        elif "align" in feat.__class__.__name__.lower():
            indices = np.vstack([indices for i in range(3)]).flatten(order="F")
            ind_da = xr.DataArray(
                np.expand_dims(np.expand_dims(indices, 0), -1),
                coords=coords
                | {
                    "ALIGN_ATOM_NO": [0],
                },
                dims=["traj_num", name.upper(), "ALIGN_ATOM_NO"],
                name=name.upper(),
                attrs=attrs | {"feature_axis": name.upper()},
            )
        elif feat.__class__.__name__ == "ResidueMinDistanceFeature":
            # indices = np.hstack(indices)
            # data = np.vstack([indices for i in range(3)]).flatten(order="F")
            ind_da = xr.DataArray(
                np.expand_dims(indices, 0),
                coords=coords | {"RES_NO": [0, 1]},
                dims=["traj_num", name.upper(), "RES_NO"],
                name=name.upper(),
                attrs=attrs | {"feature_axis": name.upper()},
            )
        elif len(indices) != data.shape[-1]:
            if all(["COS" in i for i in index_labels[::2]]) and all(
                ["SIN" in i for i in index_labels[1::2]]
            ):
                c = np.empty((indices.shape[0] * 2, 3), dtype=int)
                c[0::2] = indices
                c[1::2] = indices
                indices = np.expand_dims(c, 0)
                ind_da = xr.DataArray(
                    indices,
                    coords=coords | {"ATOM_NO": np.arange(indices.shape[-1])},
                    dims=["traj_num", name.upper(), "ATOM_NO"],
                    name=name.upper(),
                    attrs=attrs | {"feature_axis": name.upper()},
                )
            else:
                raise Exception(
                    f"{feat=}\n\n"
                    f"{feat.name=}\n\n"
                    f"{inhomogeneous_shape=}\n\n"
                    f"{len(indices)=}\n\n"
                    f"{indices.shape=}\n\n"
                    f"{indices.ndim=}\n\n"
                    f"{data.shape=}\n\n"
                    f"{indices=}\n\n"
                    f"{data=}\n\n"
                    f"{index_labels=}\n\n"
                    f"{len(indices) == data.shape[-1] // 3=}\n\n"
                    f"{'selection' in feat.name.lower()=}"
                )
        # regular case indices and data are aligned
        else:
            indices = np.expand_dims(indices, 0)
            if indices.ndim != data.ndim:
                indices = np.expand_dims(indices, -1)
            ind_da = xr.DataArray(
                indices,
                coords=coords | {"ATOM_NO": np.arange(indices.shape[-1])},
                dims=["traj_num", name.upper(), "ATOM_NO"],
                name=name.upper(),
                attrs=attrs | {"feature_axis": name.upper()},
            )
        if ind_da.size > 1:
            assert len(ind_da.coords[name.upper()]) == len(da.coords[name.upper()])

    if isinstance(with_time, bool):
        if with_time:
            da = da.assign_coords(time=("frame_num", traj.time))
    else:
        da = da.assign_coords(time=("frame_num", with_time))
    if ind_da.size == 1:
        ind_da = None
    return da, ind_da


def get_indices_by_feature_dim(feat, traj, input_data_shape):
    """Unpacks the concatenated features returned by PyEMMA.

    Usually, if you add multiple features to a PyEMMA featurizer, they are
    stacked along the feature axis. Let's say, you have a trajectory with 20 frames
    and 3 residues. If you add the Ramachandran angles, you get 6 features (3xphi, 3xpsi).
    If you then also add the end-to-end distance as a feature, the data returned by
    PyEMMA will have the shape (20, 7). This function returns the correct indices,
    so that iteration of `zip(Featurizer.active_features, indices)` will yield the
    correct results.

    Args:
        feat (em.Featurizer): Instance of `encodermap.Featurizer`. The featurizer knows
            how many entries every feature takes up in the feature space.
        traj (encodermap.SingleTraj): An instance of `SingleTraj`. This is needed
            for a single purpose: PyEMMA returns malformed data for .pdb files, that
            are loaded from the protein database
            (i.e. `traj = em.SingleTraj('https://files.rcsb.org/view/1GHC.pdb'`).
            This is prevented by providing the traj and do a small check.
        input_data_shape (tuple): The data, that should conform to the slicing.
            Also needed for some checks.

    Returns:
        list[np.ndarray]: List of `np.ndarray`s that correctly slice the data.

    """
    if len(feat.features) > 1:
        indices = [0] + add_one_by_one([f._dim for f in feat.features])
        # Since deprecating PyEMMA, we can remove this part
        # if traj.extension == ".pdb" and _validate_uri(traj.traj_file):
        #     # for internet pdb files we need to slice the data. this negates the next assert but meh.
        #     raise Exception(
        #         "For some reason `pyemma.coordinates.source` does not like working with internet pdb files."
        #     )
        # else:
        if len(input_data_shape) <= 1:
            raise Exception(f"{feat=}, {traj=}, {input_data_shape=}")
        if not indices[-1] == input_data_shape[1]:
            raise Exception(
                f"The indices in the features do not match the input data shape: "
                f"{indices[-1]=} {input_data_shape=} "
                f"{sum([len(f.describe()) for f in feat.features])=}\n\n"
                f"{indices=}\n\n{input_data_shape=}"
            )
        indices = [np.arange(i, j) for i, j in zip(indices[:-1], indices[1:])]
    else:
        indices = [np.arange(0, feat.features[0]._dim)]
    return indices


def add_one_by_one(l: list[Number]) -> list[Number]:
    """Creates a new list from l with elements added one after another.

    Args:
        l (list): The input list.

    Returns:
        list: The output list.

    Example:
        >>> l = [0, 2, 4, 5, 7]
        >>> add_one_by_one(l)
        [0, 2, 6, 11, 18]

    """
    new_l = []
    cumsum = 0
    for elt in l:
        cumsum += elt
        new_l.append(cumsum)
    return new_l

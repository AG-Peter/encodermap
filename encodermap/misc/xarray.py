# -*- coding: utf-8 -*-
# encodermap/misc/xarray.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2022 University of Konstanz and the Authors
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
################################################################################
# Imports
################################################################################


from __future__ import annotations

import numpy as np

from .._optional_imports import _optional_import
from .errors import BadError
from .misc import FEATURE_NAMES, _validate_uri

################################################################################
# Optional Imports
################################################################################


xr = _optional_import("xarray")


################################################################################
# Typing
################################################################################


from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import xarray as xr

    from ..loading.featurizer import Featurizer
    from ..trajinfo import SingleTraj


################################################################################
# Globals
################################################################################


__all__ = [
    "construct_xarray_from_numpy",
    "unpack_data_and_feature",
]


################################################################################
# Functions
################################################################################


def construct_xarray_from_numpy(
    traj: SingleTraj,
    data: np.ndarray,
    name: str,
    labels: Optional[list[str]] = None,
    check_n_frames: bool = False,
) -> xr.DataArray:
    """Constructs an xarray dataarray from a numpy array.

    Three different cases are recognized:
        * The input array in data has ndim == 2. This kind of feature/CV is a per-frame feature, like the membership
            to clusters. Every frame of every trajectory is assigned a single value (most often int values).
        * The input array in data has ndim == 3: This is also a per-frame feature/CV, but this time every frame
            is characterized by a series of values. These values can be dihedral angles in the backbone starting from
            the protein's N-terminus to the C-terminus, or pairwise distance features between certain atoms.
            The xarray datarrat constructed from this kind of data will have a label dimension that will either
            contain generic labels like 'CUSTOM_FEATURE FEATURE 0' or labels defined by the featurizer such as
            'SIDECHAIN ANGLE CHI1 OF RESIDUE 1LYS'.
        * The input array in data has ndim == 4. Here, the same feature/CV is duplicated for the protein's atoms.
            Besides the XYZ coordinates of the atoms no other CVs should fall into this case. The labels will be
            2-dimensional with 'POSITION OF ATOM H1 IN RESIDUE 1LYS' in dimension 0 and either 'X', 'Y' or 'Z' in
            dimension 1.

    Args:
        traj (em.SingleTraj): The trajectory we want to create the xarray dataarray for.
        data (np.ndarray): The numpy array we want to create the data from. Note, that the data passed into this
            function should be expanded by np.expand_dim(a, axis=0), so to add a new axis to the complete data
            containing the trajectories of a trajectory ensemble.
        name (str): The name of the feature. This can be choosen freely. Names like 'central_angles', 'backbone_torsions'
            would make the most sense.
        labels (Optional[list]): If you have specific labels for your CVs in mind, you can overwrite the
            generic 'CUSTOM_FEATURE FEATURE 0' labels with providing a list for this argument. If None is provided,
            generic names will be given to the features. Defaults to None.
        check_n_frames (bool): Whether to check whether the number of frames in the trajectory matches the len
            of the data in at least one dimension. Defaults to False.

    Returns:
        xarray.Dataarray: An `xarray.Dataarray`.

    Examples:
        >>> import encodermap as em
        >>> from encodermap.misc.xarray import construct_xarray_from_numpy
        >>> # load file from RCSB and give it traj num to represent it in a potential trajectory ensemble
        >>> traj = em.load('https://files.rcsb.org/view/1GHC.pdb', traj_num=1)
        >>> # single trajectory needs to be expaneded into 'trajectory' axis
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
        assert any(s == traj.n_frames for s in data.shape), print(
            f"Data and traj misaligned. traj_frames: {traj.n_frames}, data.shape: {data.shape}, attr_name: {name}, {data}"
        )
    if traj.backend == "no_load":
        with_time = False
    else:
        with_time = True
    if data.ndim == 2:
        if labels is None:
            labels = [f"{name.upper()} FEATURE {i}" for i in range(data.shape[-1])]
        da = make_frame_CV_dataarray(labels, traj, name, data, with_time=with_time)
    elif data.ndim == 3:
        if labels is None:
            labels = [f"{name.upper()} FEATURE {i}" for i in range(data.shape[-1])]
        da = make_dataarray(labels, traj, name, data, with_time=with_time)
    elif data.ndim == 4:
        if labels is None:
            labels = [f"{name.upper()} FEATURE {i}" for i in range(data.shape[-2])]
        da = make_position_dataarray_from_numpy(
            labels, traj, name, data, with_time=with_time
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
    put_indices_into_attrs: bool = True,
) -> xr.Dataset:
    """Makes a `xarray.Dataset` from data and a featurizer.

    Usually, if you add multiple features to a PyEMMA featurizer, they are
    stacked along the feature axis. Let's say, you have a trajectory with 20 frames
    and 3 residues. If you add the Ramachandran angles, you get 6 features (3xphi, 3xpsi).
    If you then also add the end-to-end distance as a feature, the data returned by
    PyEMMA will have the shape (20, 7). This function returns the correct indices,
    so that iteration of `zip(Featurizer.active_features, indices)` will yield the
    correct results.

    Args:
        feat (em.Featurizer): An instance of the currently used `encodermap.Featurizer`.
        traj (em.SingleTraj): An instance of `encodermap.SingleTraj`, that the data
            in `input_data` was computed from
        input_data (np.ndarray): The data, as returned from PyEMMA.
        put_indices_into_attrs (bool): Whether to put the indices into the attrs.
            This needs to be False, when Ensembles are loaded because the Features
            of the ensemble load function do not match the real indices that should be there.

    Returns:
        xarray.Dataset: An `xarray.Dataset` with all features in a nice format.

    """
    # this needs to be done, because pyemma concatenates the data
    # along the feature axis
    indices = get_indices_by_feature_dim(feat, traj, input_data.shape)

    DAs = {}
    indexes = {}
    for f, ind in zip(feat.features, indices):
        data = input_data[:, ind]
        try:
            name = FEATURE_NAMES[f.name]
        except (KeyError, AttributeError):
            name = f.__class__.__name__
            f.name = name
        assert data.shape[1] == f._dim
        if data.shape[0] != len(traj):
            if traj.index is None:
                raise Exception(
                    "Shape of provided data does not fit traj. Traj "
                    f"has {traj.n_frames=} {len(traj)=}. Data has shape {data.shape}"
                )
            else:
                data = data[traj.index]
                assert data.shape[0] == len(traj)
        if name in DAs:
            name = (
                name
                + f"_{len(list(filter(lambda x: True if name in x else False, list(DAs.keys()))))}"
            )
        if f.name in ["AllCartesians", "CentralCartesians", "SideChainCartesians"]:
            data = data.reshape(len(traj), -1, 3)
            data = np.expand_dims(data, axis=0)
            DAs[name] = make_position_dataarray(f.describe(), traj, name, data)
        else:
            data = np.expand_dims(data, axis=0)
            DAs[name] = make_dataarray(f.describe(), traj, name, data)

        # the indices/indexes used to create this dataarray.
        # This can be useful for later. To redo some analysis
        # or use tensorflow for the geometric computations of these values.
        # some PyEMMA features give the indexes different names e.g 'group_definitions'
        try:
            indexes[name] = f.indexes.tolist()
        except AttributeError as e:
            for key in f.__dir__():
                if "inde" in key:
                    indexes[name] = f.__dict__[key]
            if (
                f.__class__.__name__ == "GroupCOMFeature"
                or f.__class__.__name__ == "ResidueCOMFeature"
            ):
                indexes[name] = f.group_definitions
            if f.__class__.__name__ == "MinRmsdFeature":
                indexes[name] = f.atom_indices
            if name not in indexes:
                raise e
    ds = xr.Dataset(DAs)
    if put_indices_into_attrs:
        ds = ds.assign_attrs(indexes)
    return ds


def construct_xarray_from_feat(feat, input_data=None):
    raise Exception(
        "Marked for deprecation, because featurizer returns its own dataset."
    )


def make_frame_CV_dataarray(
    labels, traj, name, data, with_time=True, frame_indices=None, labels2=None
):
    """Make a dataarray from a frame CV feature.

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
        Please make sure, that the input data conforms to the nm, ps, rad coordinates.

    Args:
        labels (list[str]): The labels, that specify the `CV_num` dimension. This
            requires the expression `len(labels == data.shape[2]` to be True. If you
            build the dataarray from a feature. The `labels` argument usually will
            be `feature.describe()`.
        traj (encodermap.SingleTraj): An `encodermap.SingleTraj` trajectory.
            Why `SingleTraj` and not `TrajEnsemble`? That is, because in EncoderMap,
            an xarray.Dataarray always represents one feature, of one trajectory.
            A trajectory can have multiple features, which is represented as an
            `xarray.Dataset`. For that, the function `unpack_data_and_feature` is used.
            The `TrajEnsemble` trajectory does not even have its own `xarray.Dataset`. This
            dataset is created ad hoc, by merging the datasets of the trajectories along
            the `traj_num` axis.
        name (str): The name of the feature. This name will be used to group similar
            features in the large `xarray.Dataset`s of trajectory ensembles. If you
            construct this dataarray from a feature, you can either use `feature.name`,
            or `feature.__class__.__name__`.
        data (Union[np.ndarray, dask.array]): The data to fill the dataarray with.
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
            in a lot of places (i.e. single amino acid exchange). If the labels are
            too strict (LYS1 ATOM1 C), the data of two slightly different proteins
            can not be concatenated. If the labels are generic (AA1 ATOM1), this can
            be done. Defaults to None.

    Returns:
        xarray.Dataarray: The resulting dataarray.

    """
    if data.ndim == 3:
        pass
    elif data.ndim == 2:
        data = np.expand_dims(data, -1)
    return make_dataarray(
        labels,
        traj,
        name,
        data,
        with_time,
        frame_indices,
        labels2,
    )


def make_position_dataarray_from_numpy(atoms, traj, name, data, with_time=True):
    frame_indices = traj.id
    if frame_indices.ndim > 1:
        frame_indices = frame_indices[:, 1]
    da = xr.DataArray(
        data,
        coords={
            "traj_num": ("traj_num", [traj.traj_num]),
            "traj_name": ("traj_num", [traj.basename]),
            "frame_num": ("frame_num", frame_indices),
            "ATOM": ("ATOM", atoms),
            "COORDS": ["POSITION X", "POSITION Y", "POSITION Z"],
        },
        dims=["traj_num", "frame_num", "ATOM", "COORDS"],
        name=name,
        attrs={
            "length_units": "nm",
            "time_units": "ps",
            "angle_units": "rad",
            "full_path": traj.traj_file,
            "topology_file": traj.top_file,
            "feature_axis": "ATOM",
        },
    )
    if with_time:
        da = da.assign_coords(time=("frame_num", traj.time))
    return da


def make_position_dataarray(
    labels, traj, name, data, with_time=True, frame_indices=None, labels2=None
):
    """Creates dataarray belonging to cartesian positions.

    Similar to `make_datarray`, but the shapes are even larger. As every atom
    contributes 3 coordinates (x, y, z) to the data, the shape of the returned
    dataarray is (1, no_of_frames, no_of_atoms_considered, 3).

    Note:
        Please make sure, that the input data conforms to the nm, ps, rad coordinates.

    Args:
        labels (list[str]): The labels, that specify the `CV_num` dimension. This
            requires the expression `len(labels == data.shape[2]` to be True. If you
            build the dataarray from a feature. The `labels` argument usually will
            be `feature.describe()`.
        traj (encodermap.SingleTraj): An `encodermap.SingleTraj` trajectory.
            Why `SingleTraj` and not `TrajEnsemble`? That is, because in EncoderMap,
            an xarray.Dataarray always represents one feature, of one trajectory.
            A trajectory can have multiple features, which is represented as an
            `xarray.Dataset`. For that, the function `unpack_data_and_feature` is used.
            The `TrajEnsemble` trajectory does not even have its own `xarray.Dataset`. This
            dataset is created ad hoc, by merging the datasets of the trajectories along
            the `traj_num` axis.
        name (str): The name of the feature. This name will be used to group similar
            features in the large `xarray.Dataset`s of trajectory ensembles. If you
            construct this dataarray from a feature, you can either use `feature.name`,
            or `feature.__class__.__name__`.
        data (Union[np.ndarray, dask.array]): The data to fill the dataarray with.
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
            in a lot of places (i.e. single amino acid exchange). If the labels are
            too strict (LYS1 ATOM1 C), the data of two slightly different proteins
            can not be concatenated. If the labels are generic (AA1 ATOM1), this can
            be done. Defaults to None.

    Returns:
        xarray.Dataarray: The resulting dataarray.

    """
    if frame_indices is None:
        frame_indices = traj.id
        if frame_indices.ndim > 1:
            frame_indices = frame_indices[:, 1]
        # frame_indices = traj._original_frame_indices
        # if not np.any(frame_indices):
        #     frame_indices = traj.id
        # if frame_indices.ndim > 1:
        #     frame_indices = frame_indices[:, 1]

    if labels2 is not None:
        labels = labels2
    else:
        labels = [_[11:].lstrip(" ") for _ in labels[::3]]

    da = xr.DataArray(
        data,
        coords={
            "traj_num": ("traj_num", [traj.traj_num]),
            "traj_name": ("traj_num", [traj.basename]),
            "frame_num": ("frame_num", frame_indices),
            "ATOM": ("ATOM", labels),
            "COORDS": ["POSITION X", "POSITION Y", "POSITION Z"],
        },
        dims=["traj_num", "frame_num", "ATOM", "COORDS"],
        name=name.upper(),
        attrs={
            "length_units": "nm",
            "time_units": "ps",
            "angle_units": "rad",
            "full_path": traj.traj_file,
            "topology_file": traj.top_file,
            "feature_axis": "ATOM",
        },
    )
    if isinstance(with_time, bool):
        if with_time:
            da = da.assign_coords(time=("frame_num", traj.time))
    else:
        da = da.assign_coords(time=("frame_num", with_time))
    return da


def make_dataarray(
    labels, traj, name, data, with_time=True, frame_indices=None, labels2=None
):
    """Creates a dataarray belonging to a feature.

    The shapes are a bit different that what most people might be used to. As
    EncoderMap was meant to work with ensembles of trajectories, the data is usually
    shaped as (traj_num, frame_num, CV_num), or even (traj_num, frame_num, atom_num, 3).

    The `xarray.Dataarray`, that is returned by this function reflects this. As a
    dataarray is attributed to a single trajectory, the first shape will always be 1.
    So for a collective variable, that describes n features, the shape of the returned
    datarray will be (1, n_frames, n). Combining multiple trajectories, the first number
    can increase.

    Note:
        Please make sure, that the input data conforms to the nm, ps, rad coordinates.

    Args:
        labels (list[str]): The labels, that specify the `CV_num` dimension. This
            requires the expression `len(labels == data.shape[2]` to be True. If you
            build the dataarray from a feature. The `labels` argument usually will
            be `feature.describe()`.
        traj (encodermap.SingleTraj): An `encodermap.SingleTraj` trajectory.
            Why `SingleTraj` and not `TrajEnsemble`? That is, because in EncoderMap,
            an xarray.Dataarray always represents one feature, of one trajectory.
            A trajectory can have multiple features, which is represented as an
            `xarray.Dataset`. For that, the function `unpack_data_and_feature` is used.
            The `TrajEnsemble` trajectory does not even have its own `xarray.Dataset`. This
            dataset is created ad hoc, by merging the datasets of the trajectories along
            the `traj_num` axis.
        name (str): The name of the feature. This name will be used to group similar
            features in the large `xarray.Dataset`s of trajectory ensembles. If you
            construct this dataarray from a feature, you can either use `feature.name`,
            or `feature.__class__.__name__`.
        data (Union[np.ndarray, dask.array]): The data to fill the dataarray with.
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
            in a lot of places (i.e. single amino acid exchange). If the labels are
            too strict (LYS1 ATOM1 C), the data of two slightly different proteins
            can not be concatenated. If the labels are generic (AA1 ATOM1), this can
            be done. Defaults to None.

    Returns:
        xarray.Dataarray: The resulting dataarray.

    """
    if frame_indices is None:
        frame_indices = traj.id
        if frame_indices.ndim > 1:
            frame_indices = frame_indices[:, 1]
        # frame_indices = traj._original_frame_indices
        # if not np.any(frame_indices):
        #     frame_indices = traj.id
        # if frame_indices.ndim > 1:
        #     frame_indices = frame_indices[:, 1]

    if labels2 is not None:
        labels = labels2

    if callable(labels):
        labels = [l for l in labels(traj.top)]
    else:
        if len(labels) > data.shape[-1]:
            labels = np.arange(data.shape[-1])

    da = xr.DataArray(
        data,
        coords={
            "traj_num": ("traj_num", [traj.traj_num]),
            "traj_name": ("traj_num", [traj.basename]),
            "frame_num": ("frame_num", frame_indices),
            name.upper(): labels,
        },
        dims=["traj_num", "frame_num", name.upper()],
        name=name,
        attrs={
            "length_units": "nm",
            "time_units": "ps",
            "angle_units": "rad",
            "full_path": traj.traj_file,
            "topology_file": traj.top_file,
            "feature_axis": name.upper(),
        },
    )
    if isinstance(with_time, bool):
        if with_time:
            da = da.assign_coords(time=("frame_num", traj.time))
    else:
        da = da.assign_coords(time=("frame_num", with_time))
    return da


def make_ensemble_xarray(name, data):
    attrs = {"length_units": "nm", "time_units": "ps", "angle_units": "rad"}
    if data.ndim == 2:
        coords = {
            "traj_num": ("traj_num", range(data.shape[0])),
            "frame_num": ("frame_num", range(data.shape[1])),
            name.upper(): (
                "frame_num",
                [f"{name.upper()} Frame Feature {i}" for i in range(data.shape[1])],
            ),
        }
        dims = ["traj_num", "frame_num"]
    elif data.ndim == 3:
        coords = {
            "traj_num": ("traj_num", range(data.shape[0])),
            "frame_num": ("frame_num", range(data.shape[1])),
            name.upper(): [f"{name.upper()} Feature {i}" for i in range(data.shape[2])],
        }
        dims = ["traj_num", "frame_num", name.upper()]
    elif data.ndim == 4:
        coords = {
            "traj_num": ("traj_num", range(data.shape[0])),
            "frame_num": ("frame_num", range(data.shape[1])),
            "ATOM": ("ATOM", [f"ATOM {i}" for i in range(data.shape[2])]),
            "COORDS": ["POSITION X", "POSITION Y", "POSITION Z"],
        }
        dims = ["traj_num", "frame_num", "ATOM", "COORDS"]
    else:
        raise Exception("Too high dimensional data for ensemble.")
    da = xr.DataArray(data, coords=coords, dims=dims, name=name, attrs=attrs)
    return da


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
        if traj.extension == ".pdb" and _validate_uri(traj.traj_file):
            # for internet pdb files we need to slice the data. this negates the next assert but meh.
            raise BadError(
                "For some reason pyemma.coordinates.source does not like working with internet pdb files."
            )
        else:
            if not indices[-1] == input_data_shape[1]:
                for f in feat.features:
                    print(f.__class__)
                    print(f.describe())
                    print(len(f.describe()))
                    print(f._dim)
                print(traj, indices, input_data_shape)
                raise Exception
        indices = [np.arange(i, j) for i, j in zip(indices[:-1], indices[1:])]
    else:
        indices = [np.arange(0, feat.features[0]._dim)]
    return indices


def add_one_by_one(l):
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

# -*- coding: utf-8 -*-
# encodermap/trajinfo/trajinfo_utils.py
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
"""Util functions for the `TrajEnsemble` and `SingleTraj` classes.

"""


################################################################################
# Imports
################################################################################


from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .._optional_imports import _optional_import
from ..loading.featurizer import PyEMMAFeaturizer as Featurizer
from ..misc.misc import FEATURE_NAMES
from ..misc.xarray import construct_xarray_from_numpy

##############################################################################
# Optional Imports
##############################################################################


xr = _optional_import("xarray")
Feature = _optional_import("pyemma", "coordinates.data.featurization._base.Feature")
md = _optional_import("mdtraj")


################################################################################
# Typing
################################################################################


from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:
    import mdtraj as md
    import xarray as xr
    from pyemma.coordinates.data.featurization._base import Feature

    from ..loading.featurizer import Featurizer
    from .info_single import SingleTraj, TrajEnsemble

    SingleTrajFeatureType = Union[
        str, Path, np.ndarray, Feature, xr.Dataset, xr.DataArray, Featurizer
    ]
    TrajEnsembleFeatureType = Union[
        list[str], list[Path], list[np.ndarray], xr.Dataset, Featurizer, Literal["all"]
    ]
    Index = Optional[
        Union[tuple[int, list, np.ndarray, slice]], int, list, np.ndarray, slice
    ]


################################################################################
# Globals
################################################################################


CAN_BE_FEATURE_NAME = list(FEATURE_NAMES.keys()) + list(FEATURE_NAMES.values())
__all__ = ["load_CVs_singletraj", "load_CVs_ensembletraj"]


################################################################################
# Utils
################################################################################


def np_to_xr(
    data: np.ndarray,
    traj: SingleTraj,
    attr_name: Optional[str] = None,
    labels: Optional[list[str]] = None,
    filename: Optional[Union[str, Path]] = None,
) -> xr.DataArray:
    if attr_name is None:
        if filename is None:
            msg = f"Please also provide an `attr_name` under which to save the CV."
            raise Exception(msg)
        attr_name = Path(filename).stem

    if labels is not None:
        if isinstance(labels, str):
            labels = [
                f"{attr_name.upper()} {labels.upper()} {i}"
                for i in range(data.shape[1])
            ]
        elif (
            not all([isinstance(l, str) for l in labels])
            and len(labels) == data.shape[1]
        ):
            pass
        else:
            raise Exception(
                f"'labels' is either not a list of str or does not have the "
                f"same number of datapoints as self.n_frames={data.n_frames}: {labels=}"
            )
    data = np.expand_dims(data, axis=0)
    if np.any(np.isnan(data)):
        # if some nans are found along frame remove them
        if data.ndim == 2:
            data = data[:, ~np.isnan(data).any(axis=0)]
        if data.ndim == 3:
            data = data[:, ~np.isnan(data).any(axis=2)[0]]
        if data.ndim == 4:
            data = data[:, ~np.isnan(data).any(axis=2)[0].any(axis=1)]
    da = construct_xarray_from_numpy(traj, data, attr_name, labels, check_n_frames=True)
    return da


def load_CV_from_string_or_path(
    file_or_feature: str,
    traj: SingleTraj,
    attr_name: Optional[str] = None,
    cols: Optional[Union[int, list[int]]] = None,
    labels: Optional[list[str]] = None,
) -> xr.Dataset:
    """Loads CV data from a string. That string can either identify a features,
    or point to a file.

    Args:
        file_or_feature (str): The file or feature to load. If 'all' is
            provided, all "standard" features are loaded. But a feature name
            like 'sidechain_angle' can alsop be provided. If a file with
            the .txt or .npy extension is provided, the data in that file is used.
        traj (SingleTraj): The trajectory, that is used to load the features.
        attr_name (Union[None, str], optional): The name under which the CV should be found in the class.
            Is needed, if a raw numpy array is passed, otherwise the name will be generated from the filename
            (if data == str), the DataArray.name (if data == xarray.DataArray), or the feature name.
        cols (Union[list, None], optional): A list specifying the columns to use for the highD data.
            If your highD data contains (x,y,z,...)-errors or has an enumeration
            column at col=0 this can be used to remove this unwanted data.
        labels (Union[list, str, None], optional): If you want to label the data you provided pass a list of str.
            If set to None, the features in this dimension will be labelled as
            [f"{attr_name.upper()} FEATURE {i}" for i in range(self.n_frames)]. If a str is provided, the features
            will be labelled as [f"{attr_name.upper()} {label.upper()} {i}" for i in range(self.n_frames)]. If a list of str
            is provided it needs to have the same length as the traj has frames. Defaults to None.

    Returns:
        xr.Dataset: An xarray dataset.

    """
    if str(file_or_feature) == "all" or str(file_or_feature) in CAN_BE_FEATURE_NAME:
        feat = Featurizer(traj)
        if file_or_feature == "all":
            feat.add_list_of_feats("all")
        else:
            feat.add_list_of_feats([file_or_feature])
        out = feat.get_output()
        if traj.traj_num is not None:
            assert out.coords["traj_num"] == np.array([traj.traj_num]), print(
                traj.traj_num,
                out.coords["traj_num"].values,
                feat.trajs.trajs[0].traj_num,
            )
        return out
    elif (f := Path(file_or_feature)).exists():
        if f.suffix == ".txt":
            data = np.loadtxt(f, usecols=cols)
        elif f.suffix == ".npy":
            data = np.load(f)
            if cols is not None:
                data = data[:, cols]
        elif f.suffix in [".nc", ".h5"]:
            data = xr.open_dataset(f)
            if len(data.data_vars.keys()) != 1:
                if attr_name is not None:
                    raise Exception(
                        f"The dataset in {f} has "
                        f"{len(data.data_vars.keys())} dataarrays, "
                        f"but only one `attr_name`: '{attr_name}' "
                        f"was requested. The names of the dataarrays "
                        f"are: {data.data_vars.keys()}. I can't over"
                        f"ride them all with one `attr_name`. Set "
                        f"`attr_name` to None to load the data with "
                        f"their respective names"
                    )
                return data
            else:
                if attr_name is not None:
                    d = list(data.data_vars.values())[0]
                    d.name = attr_name
                return d
        else:
            raise Exception(
                f"Currently only .txt, .npy, .nc, and .h5 files can "
                f"be loaded. Your file {f} does not have the "
                f"correct extension."
            )
    else:
        raise Exception(
            f"If features are loaded via a string, the string needs "
            f"to be 'all', a features name ('central_dihedrals') or "
            f'an existing file. Your string "{file_or_feature}"'
            f"is none of those"
        )

    return np_to_xr(data, traj, attr_name, labels, file_or_feature)


def load_CVs_singletraj(
    data: SingleTrajFeatureType,
    traj: SingleTraj,
    attr_name: Optional[str] = None,
    cols: Optional[list[int]] = None,
    labels: Optional[list[str]] = None,
) -> xr.Dataset:
    if isinstance(attr_name, str):
        if not attr_name.isidentifier():
            raise Exception(
                f"Provided string for `attr_name` can not be a "
                f"python identifier. Choose another attribute name."
            )
    # load a string
    if isinstance(data, (str, Path)):
        CVs = load_CV_from_string_or_path(str(data), traj, attr_name, cols, labels)

    # load a list of strings from standard features
    elif isinstance(data, list) and all([isinstance(_, str) for _ in data]):
        feat = Featurizer(traj)
        feat.add_list_of_feats(data)
        return feat.get_output()

    # if the data is a numpy array
    elif isinstance(data, (list, np.ndarray)):
        CVs = np_to_xr(np.asarray(data), traj, attr_name, labels).to_dataset()

    # xarray objects are simply returned
    elif isinstance(data, xr.Dataset):
        return data

    elif isinstance(data, xr.DataArray):
        return data.to_dataset()

    # if this is a feature
    elif issubclass(data.__class__, Feature):
        feat = Featurizer(traj)
        feat.add_custom_feature(data)
        return feat.get_output()

    # if an instance of featurizer is provided
    elif isinstance(data, Featurizer):
        if isinstance(attr_name, str):
            if len(data) != 1:
                raise TypeError(
                    f"Provided Featurizer contains {len(data)} "
                    f"features and `attr_name` is of type `str`. "
                    f"Please provide a list of str."
                )
            attr_name = [attr_name]
        if isinstance(attr_name, list):
            if len(attr_name) != len(data):
                raise IndexError(
                    f"Provided Featurizer contains {len(data)} "
                    f"features and `attr_name` contains "
                    f"{len(attr_name)} elements. Please make sure "
                    f"they contain the same amount of items."
                )
        out = data.get_output()
        if attr_name is not None:
            if isinstance(attr_name, str):
                attr_name = [attr_name]
            _renaming = {}
            for f, v in zip(data.features, attr_name):
                _feature = False
                if hasattr(f, "name"):
                    if f.name in FEATURE_NAMES:
                        k = FEATURE_NAMES[f.name]
                        _feature = True
                if not _feature:
                    k = f.__class__.__name__
                _renaming[k] = v
            out = out.rename_vars(_renaming)
        return out
    else:
        raise TypeError(
            f"`data` must be str, np.ndarray, list, xr.DataArray, xr.Dataset, "
            f"em.Featurizer or em.features.Feature. You supplied "
            f"{type(data)}."
        )

    return CVs


def load_CVs_ensembletraj(
    trajs: TrajEnsemble,
    data: TrajEnsembleFeatureType,
    attr_name: Optional[list[str]] = None,
    cols: Optional[list[int]] = None,
    labels: Optional[list[str]] = None,
    directory: Optional[Union[Path, str]] = None,
    ensemble: bool = False,
) -> None:
    if isinstance(data, (str, Path)) and not ensemble:
        path_data = Path(data)
        npy_files = [
            (t._traj_file.parent if directory is None else Path(directory))
            / (t.basename + f"_{data}.npy")
            for t in trajs
        ]
        txt_files = [
            (t._traj_file.parent if directory is None else Path(directory))
            / (t.basename + f"_{data}.txt")
            for t in trajs
        ]
        raw_files = [
            (t._traj_file.parent if directory is None else Path(directory))
            / (t.basename + f"_{data}")
            for t in trajs
        ]
        if str(data) == "all":
            [t.load_CV("all") for t in trajs]
            return
        if path_data.is_dir():
            return load_CVs_from_dir(trajs, data, attr_name=attr_name, cols=cols)
        elif data in CAN_BE_FEATURE_NAME:
            [t.load_CV(data, attr_name, cols, labels) for t in trajs]
            return
        elif path_data.is_file() and (
            path_data.suffix == ".h5" or path_data.suffix == ".nc"
        ):
            ds = xr.open_dataset(path_data)
            if diff := set([t.traj_num for t in trajs]) - set(ds["traj_num"].values):
                raise Exception(
                    f"The dataset you try to load and the TrajEnsemble "
                    f"have different number of trajectories: {diff}."
                )
            for t, (traj_num, sub_ds) in zip(trajs, ds.groupby("traj_num")):
                assert t.traj_num == traj_num
                sub_ds = sub_ds.assign_coords(traj_num=t.traj_num)
                sub_ds = sub_ds.expand_dims("traj_num")
                assert sub_ds.coords["traj_num"] == np.array([t.traj_num])
                t.load_CV(sub_ds)
            return
        elif all([f.is_file() for f in npy_files]):
            [
                t.load_CV(f, attr_name=data, cols=cols, labels=labels)
                for t, f in zip(trajs, npy_files)
            ]
            return
        elif all([f.is_file() for f in txt_files]):
            [
                t.load_CV(f, attr_name=data, cols=cols, labels=labels)
                for t, f in zip(trajs, txt_files)
            ]
            return
        elif all([f.is_file() for f in raw_files]):
            [
                t.load_CV(f, attr_name=data, cols=cols, labels=labels)
                for t, f in zip(trajs, raw_files)
            ]
            return
        else:
            msg = (
                f"If `data` is provided a single string, the string needs to "
                f"be either a feature ({CAN_BE_FEATURE_NAME}), a .h5/.nc file "
                f"({file}), or a list of npy/txt files ({npy_files}, "
                f"{txt_files}). The provided `data` fits none of "
                f"these possibilities."
            )
            raise ValueError(msg)

    elif isinstance(data, list) and not ensemble:
        if all([i in CAN_BE_FEATURE_NAME for i in data]):
            [t.load_CV(data, attr_name, cols, labels) for t in trajs]
            return
        elif all([isinstance(i, (list, np.ndarray)) for i in data]):
            [t.load_CV(d, attr_name, cols, labels) for t, d in zip(trajs, data)]
            return
        elif all([Path(f).is_file() for f in data]):
            suffix = set([Path(f).suffix for f in data])
            if len(suffix) != 1:
                raise Exception(
                    "Please provide a list with consistent file "
                    f"extensions and not a mish-mash, like: {suffix}"
                )
            suffix = suffix.pop()
            if suffix == ".npy":
                [
                    t.load_CV(np.load(d), attr_name, cols, labels)
                    for t, d in zip(trajs, data)
                ]
            else:
                [
                    t.load_CV(np.genfromtxt(d), attr_name, cols, labels)
                    for t, d in zip(trajs, data)
                ]
            return
        else:
            msg = (
                f"If `data` is provided as a list, the list needs to contain "
                f"strings that can be features ({CAN_BE_FEATURE_NAME}), or "
                f"some combination of lists and numpy arrays."
            )
            raise ValueError(msg)

    elif isinstance(data, np.ndarray):
        if len(data) != trajs.n_trajs and len(data) != trajs.n_frames:
            raise ValueError(
                f"The provided numpy array is misshaped. It needs "
                f"to be of shape (n_trajs={trajs.n_trajs}, "
                f"n_frames={np.unique([t.n_frames for t in trajs])[0]}, "
                f"X, (Y)), but is {data.shape}."
            )
        if len(data) == trajs.n_frames:
            data = [data[t.id[:, 1]] for t in trajs]
        [t.load_CV(d, attr_name, cols, labels) for t, d in zip(trajs, data)]
        for t in trajs:
            for v in t._CVs.values():
                assert v.shape[0] == 1, print(t.basename, v)
        return

    elif isinstance(data, Featurizer):
        ds = data.get_output()
        for t, (traj_num, sub_ds) in zip(trajs, ds.groupby("traj_num")):
            assert t.traj_num == traj_num
            sub_ds = sub_ds.assign_coords(traj_num=t.traj_num)
            sub_ds = sub_ds.expand_dims("traj_num")
            t.load_CV(sub_ds)
        return

    elif isinstance(data, xr.Dataset):
        for t, (traj_num, sub_ds) in zip(trajs, data.groupby("traj_num")):
            assert t.traj_num == traj_num
            sub_ds = sub_ds.assign_coords(traj_num=t.traj_num)
            sub_ds = sub_ds.expand_dims("traj_num")
            t.load_CV(sub_ds)
        return

    if ensemble:
        return load_CVs_ensemble(trajs, data)

    else:
        raise TypeError(
            f"`data` must be str, np.ndarray, list, xr.Dataset"
            f"em.Featurizer or. You supplied {type(data)}."
        )


def load_CVs_ensemble(
    trajs: TrajEnsemble,
    data: Union[str, list[str]],
) -> None:
    if isinstance(data, str):
        if data != "all":
            data = [data]
    feat = Featurizer(trajs)
    feat.add_list_of_feats(data)
    for t, (traj_num, sub_ds) in zip(trajs, feat.get_output().groupby("traj_num")):
        assert t.traj_num == traj_num
        sub_ds = sub_ds.assign_coords(traj_num=t.traj_num)
        sub_ds = sub_ds.expand_dims("traj_num")
        if t._CVs:
            warnings.warn(
                "Using ensemble=True will drop old CV entries from "
                "trajs, because the ferature length increases."
            )
        t._CVs = sub_ds


def load_CVs_from_dir(
    trajs: TrajEnsemble,
    data: Path,
    attr_name: Optional[str] = None,
    cols: Optional[list[int]] = None,
) -> None:
    files = map(str, data.glob("*"))
    files = list(
        filter(
            lambda x: True if any([traj.basename in x for traj in trajs]) else False,
            files,
        )
    )
    key = {"npy": 1, "txt": 2}
    files = sorted(
        files,
        key=lambda x: key[x.split(".")[-1]] if x.split(".")[-1] in key else 3,
    )[: trajs.n_trajs]
    files = sorted(
        files,
        key=lambda x: [traj.basename in x for traj in trajs].index(True),
    )
    for traj, f in zip(trajs, files):
        if traj.basename not in f:
            raise Exception(f"File {f} does not contain substring of traj {traj}.")
        traj.load_CV(f, attr_name=attr_name, cols=cols)
    return

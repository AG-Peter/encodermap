# -*- coding: utf-8 -*-
# encodermap/trajinfo/load_traj.py
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

import errno
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from ..misc.misc import _validate_uri

warnings.filterwarnings(
    "ignore",
    message=(".*top= kwargs ignored since this " "file parser does not support it.*"),
)


##############################################################################
# Optional Imports
##############################################################################


from .._optional_imports import _optional_import

md = _optional_import("mdtraj")
h5 = _optional_import("h5py")


################################################################################
# Typing
################################################################################


from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from typing_extensions import TypeVarTuple, Unpack

    Ts = TypeVarTuple("Ts")
    import h5py as h5
    import mdtraj as md

    Index = Optional[
        Union[tuple[int, list, np.ndarray, slice]], int, list, np.ndarray, slice
    ]


################################################################################
# Globals
################################################################################


__all__ = []
this = sys.modules[__name__]
this.PRINTED_HDF_ANNOTATION = False


################################################################################
# Utils
################################################################################


def _load_traj_and_top(
    traj_file: Path,
    top_file: Path,
    index: Optional[Union[int, list[int], np.ndarray, slice]] = None,
) -> md.Trajectory:
    """Loads a traj and top file and raises FileNotFoundError, if they do not exist.

    Args:
        traj_file (Path): The pathlib.Path to the traj_file.
        top_file (Path): The pathlib.Path to the top_file.
        index (Optional[Union[int, list[int], np.ndarray, slice]]): The index
            to load the traj at. If ints are provided, the load_frame
            method is used.

    Returns:
        md.Trajectory: The trajectory.

    Raises:
        FileNotFoundError: If any of the files are not real.

    """
    if not traj_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), traj_file)
    if not top_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), top_file)

    if index is not None:
        if isinstance(index, (int, np.integer)):
            return md.load_frame(str(traj_file), top=str(top_file), index=index)
        elif isinstance(index, (list, np.ndarray, slice)):
            return md.load(str(traj_file), top=str(top_file))[index]
        else:
            raise TypeError(
                f"Argument `index` must be int, list, np.ndarray or "
                f"slice. You supplied: {index.__class__.__name__}"
            )
    else:
        return md.load(str(traj_file), top=str(top_file))


def _load_traj(
    *index: Unpack(Ts),
    traj_file: Union[str, Path],
    top_file: Union[str, Path],
) -> tuple[md.Trajectory, np.ndarray]:
    """Loads a trajectory from disc and applies the indices from *index.

    Args:
        *index (Unpack[Ts]): Variable length indices of which all need to be
            one of these datatypes: None, int, np.int, list[int], slice, np.ndarray.
            These indices are applied to the traj in order. So for a traj with
            100 frames, the indices (slice(None, None, 5), [0, 2, 4, 6]) would
            yield the frames 0, 10, 20, 30, 40. A None will not slice the traj at all.
        traj_file (Union[str, Path]): The pathlib.Path to the traj_file. A string
            can also be supplied. This also allows to pass a URL, like e.g:
            https://files.rcsb.org/view/1GHC.pdb.
        top_file (Union[str, Path]): The pathlib.Path to the top_file. Can also
            be str.

    Returns:
        tuple[md.Trajectory, np.ndarray]: The trajectory and a numpy array, which
            is the result of np.arange() of the unadulterated trajectory. Can
            be useful for continued slicing and indexing to keep track of
            everyhting.

    """
    # check, whether traj_file is string and can be uri.
    if isinstance(traj_file, str):
        if _validate_uri(traj_file):
            is_uri = True
        else:
            is_uri = False
            traj_file = Path(traj_file)
    else:
        is_uri = False

    top_file = Path(top_file)

    for i, ind in enumerate(index):
        if i == 0:
            if ind is None:
                if is_uri:
                    traj = md.load_pdb(str(traj_file))
                else:
                    traj = _load_traj_and_top(traj_file, top_file)
                _original_frame_indices = np.arange(traj.n_frames)
            elif isinstance(ind, (int, np.integer)):
                print("here")
                raise Exception
                if traj_file.suffix == ".h5":
                    if not this.PRINTED_HDF_ANNOTATION:
                        print(
                            "╰(◕ᗜ◕)╯ Thank you for using the HDF5 format to "
                            "accelerate loading of single frames."
                        )
                        this.PRINTED_HDF_ANNOTATION = True
                if is_uri:
                    traj = md.load_pdb(str(traj_file))
                    _original_frame_indices = np.arange(traj.n_frames)[ind]
                    traj = traj[ind]
                else:
                    traj = _load_traj_and_top(traj_file, top_file, index=ind)
                    _original_frame_indices = np.array([ind])
            elif isinstance(ind, slice):
                if Path(traj_file).suffix == ".h5":
                    with h5.File(traj_file, "r") as file:
                        n_frames = file["coordinates"].shape[0]
                    if not this.PRINTED_HDF_ANNOTATION:
                        print(
                            "╰(◕ᗜ◕)╯ Thank you for using the HDF5 format to "
                            "accelerate loading of single frames."
                        )
                        this.PRINTED_HDF_ANNOTATION = True
                    keys = np.arange(n_frames)
                    for j, ind in enumerate(keys):
                        if j == 0:
                            traj = _load_traj_and_top(traj_file, top_file, index=ind)
                        else:
                            traj = traj.join(
                                _load_traj_and_top(traj_file, top_file, index=ind)
                            )
                    _original_frame_indices = keys
                else:
                    if is_uri:
                        traj = md.load_pdb(str(traj_file))
                    else:
                        traj = _load_traj_and_top(traj_file, top_file)
                    _original_frame_indices = np.arange(traj.n_frames)[ind]
                    traj = traj[ind]
            elif isinstance(ind, (list, np.ndarray)):
                if is_uri:
                    traj = md.load_pdb(str(traj_file))
                else:
                    traj = _load_traj_and_top(traj_file, top_file)
                _original_frame_indices = np.arange(traj.n_frames)[ind]
                traj = traj[ind]
            else:
                msg = (
                    f"For indexing/slicing only int, slice, list, np.ndarray "
                    f"can be used. You supplied: {ind.__class__.__name__}"
                )
                raise TypeError(msg)
        else:
            if ind is not None:
                traj = traj[ind]
    return traj, _original_frame_indices

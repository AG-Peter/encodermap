# -*- coding: utf-8 -*-
# encodermap/trajinfo/load_traj.py
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
"""Util functions for the `TrajEnsemble` and `SingleTraj` classes.

"""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import errno
import os
import sys
import tempfile
import urllib
import warnings
from pathlib import Path

# Third Party Imports
import numpy as np
import requests
import tables

# Local Folder Imports
from .._typing import CanBeIndex
from ..misc.misc import _validate_uri


warnings.filterwarnings(
    "ignore",
    message=(".*top= kwargs ignored since this " "file parser does not support it.*"),
)


##############################################################################
# Optional Imports
##############################################################################


# Third Party Imports
from optional_imports import _optional_import


md = _optional_import("mdtraj")
h5 = _optional_import("h5py")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from typing import TYPE_CHECKING, Optional, Union


if TYPE_CHECKING:
    # Third Party Imports
    import h5py as h5
    import mdtraj as md


################################################################################
# Globals
################################################################################


__all__ = []
this = sys.modules[__name__]
this.PRINTED_HDF_ANNOTATION = False


################################################################################
# Utils
################################################################################


def _load_pdb_from_uri(
    uri: str,
) -> md.Topology:
    """Loads urls and if MDTraj misbehaves saves them in a temporary file."""
    assert _validate_uri(uri)
    try:
        return md.load_pdb(uri).top
    except urllib.error.URLError as e:
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
            text = requests.get(uri).text
            f.write(text)
            top = md.load_pdb(f.name).top
        return top


def _load_traj_and_top(
    traj_file: Path,
    top_file: Path,
    traj_num: Union[int, None],
    index: Optional[Union[int, list[int], np.ndarray, slice]] = None,
    atom_index: Optional[np.ndarray] = None,
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
    # Local Folder Imports
    from .info_all import HDF5GroupWrite

    if not traj_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), traj_file)
    if not top_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), top_file)

    if index is not None:
        if isinstance(index, (int, np.integer)):
            t = md.load_frame(str(traj_file), top=str(top_file), index=index)
        elif isinstance(index, (list, np.ndarray, slice)):
            t = md.load(str(traj_file), top=str(top_file))[index]
        else:
            raise TypeError(
                f"Argument `index` must be int, list, np.ndarray or "
                f"slice. You supplied: {index.__class__.__name__}"
            )
    else:
        try:
            t = md.load(str(traj_file), top=str(top_file))
        except tables.NoSuchNodeError as e:
            if traj_num is None:
                raise e
            with HDF5GroupWrite(traj_file) as h5file:
                t = h5file.read_traj(traj_num)
        except RuntimeError as e:
            raise Exception(f"The file {traj_file} is broken.")
        except ValueError as e:
            if "must contain" in str(e):
                raise Exception(
                    f"The files {str(traj_file)} and {str(top_file)} contain "
                    f"different number of atoms."
                ) from e
            raise e

    if atom_index is not None:
        t = t.atom_slice(atom_index)
    return t


def _load_traj(
    *index: CanBeIndex,
    traj_file: Union[str, Path],
    top_file: Union[str, Path],
    traj_num: Union[int, None],
    atom_indices: Optional[np.ndarray] = None,
) -> tuple[md.Trajectory, np.ndarray]:
    """Loads a trajectory from disc and applies the indices from *index.

    Args:
        *index (Unpack[Ts]): Variable length indices of which all need to be
            one of these datatypes: None, int, list[int], slice, np.ndarray.
            These indices are applied to the traj in order. So for a traj with
            100 frames, the indices (slice(None, None, 5), [0, 2, 4, 6]) would
            yield the frames 0, 10, 20, 30, 40. A None will not slice the traj at all.
        traj_file (Union[str, Path]): The pathlib.Path to the traj_file. A string
            can also be supplied. This also allows passing a URL, like e.g:
            https://files.rcsb.org/view/1GHC.pdb.
        top_file (Union[str, Path]): The pathlib.Path to the top_file. Can also
            be str.

    Returns:
        tuple[md.Trajectory, np.ndarray]: The trajectory and a numpy array, which
            is the result of np.arange() of the unadulterated trajectory. Can
            be useful for continued slicing and indexing to keep track of
            everything.

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
                    traj = _load_traj_and_top(traj_file, top_file, traj_num=traj_num)
                _original_frame_indices = np.arange(traj.n_frames)
            elif isinstance(ind, (int, np.integer)):
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
                    traj = _load_traj_and_top(
                        traj_file, top_file, index=ind, traj_num=traj_num
                    )
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
                            traj = _load_traj_and_top(
                                traj_file, top_file, index=ind, traj_num=traj_num
                            )
                        else:
                            traj = traj.join(
                                _load_traj_and_top(
                                    traj_file, top_file, index=ind, traj_num=traj_num
                                )
                            )
                    _original_frame_indices = keys
                else:
                    if is_uri:
                        traj = md.load_pdb(str(traj_file))
                    else:
                        traj = _load_traj_and_top(
                            traj_file, top_file, traj_num=traj_num
                        )
                    _original_frame_indices = np.arange(traj.n_frames)[ind]
                    traj = traj[ind]
            elif isinstance(ind, (list, np.ndarray)):
                if is_uri:
                    traj = md.load_pdb(str(traj_file))
                else:
                    traj = _load_traj_and_top(traj_file, top_file, traj_num=traj_num)
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

    if atom_indices is not None:
        traj = traj.atom_slice(atom_indices)

    return traj, _original_frame_indices

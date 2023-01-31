# -*- coding: utf-8 -*-
# encodermap/trajinfo/info_all.py
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
"""Classes to work with ensembles of trajectories.

The statistics of a protein can be better described by an ensemble of proteins,
rather than a single long trajectory. Treating a protein in such a way opens great
possibilities and changes the way one can treat molecular dynamics data.
Trajectory ensembles allow:
    * Faster convergence via adaptive sampling.


This subpackage contains two classes which are containers of trajecotry data.
The SingleTraj trajecotry contains information about a single trajecotry.
The TrajEnsemble class contains information about multiple trajectories. This adds
a new dimension to MD data. The time and atom dimension are already established.
Two frames can be appended along the time axis to get a trajectory with multiple
frames. If they are appended along the atom axis, the new frame contains the
atoms of these two. The trajectory works in a similar fashion. Adding two trajectories
along the trajectory axis returns a trajectory ensemble, represented as an TrajEnsemble
class in this package.

See also:
    http://statisticalbiophysicsblog.org/?p=92

"""

################################################################################
# Imports
################################################################################


from __future__ import annotations

import copy
import glob
import os
import sys
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Literal, Optional, Union

import numpy as np

from .._optional_imports import _optional_import
from ..misc.errors import BadError
from ..misc.misc import (
    _TOPOLOGY_EXTS,
    _can_be_feature,
    _datetime_windows_and_linux_compatible,
    get_full_common_str_and_ref,
)

################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    import mdtraj as md
    import pandas as pd
    import xarray as xr

    from .info_single import SingleTraj
    from .trajinfo_utils import TrajEnsembleFeatureType


################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")
pd = _optional_import("pandas")
xr = _optional_import("xarray")


################################################################################
# Globals
################################################################################


__all__ = ["TrajEnsemble"]


################################################################################
# Utilities
################################################################################


class Capturing(list):
    """Class to capture print statements from function calls.

    Examples:
        >>> # write a function
        >>> def my_func(arg='argument'):
        ...     print(arg)
        ...     return('fin')
        >>> # use capturing context manager
        >>> with Capturing() as output:
        ...     my_func('new_argument')
        >>> print(output)
        ['new_argument', "'fin'"]

    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


##############################################################################
# Functions
##############################################################################


class TrajEnsemble:
    """This class contains the info about many trajectories.
    Topologies can be mismatching.

    This class is a fancy list of `encodermap.trajinfo.SingleTraj` objects. Trajectories can have different topologies and will
    be grouped by the `common_str` argument.

    `TrajEnsemble` supports fancy indexing. You can slice to your liking trajs[::5] returns an `TrajEnsemble`
    object that only consideres every fifth frame. Besides indexing by slices and integers you can pass
    a 2 dimensional np.array. np.array([[0, 5], [1, 10], [5, 20]]) will return a `TrajEnsemble` object with
    frame 5 of trajectory 0, frame 10 of trajectory 1 and frame 20 of trajectory 5. Simply passing an integer
    as index returns the corresponding `SingleTraj` object.

    The `TrajEnsemble` class also contains an iterator to iterate over trajectores. You could do::
    >>> for traj in trajs:
    ...     for frame in traj:
    ...         print(frame)

    Attributes:
        CVs (dict): The collective variables of the `SingleTraj` classes. Only CVs with matching names in all
            `SingleTraj` classes are returned. The data is stacked along a hypothetical time axis along the trajs.
        _CVs (xarray.Dataset): The same data as in CVs but with labels. Additionally, the xarray is not stacked along the
            time axis. It contains an extra dimension for trajectories.
        n_trajs (int): Number of individual trajectories in this class.
        n_frames (int): Number of frames, sum over all trajectories.
        locations (list of str): A list with the locations of the trajectories.
        top (list of mdtraj.Topology): A list with the reference pdb for each trajecotry.
        basenames (list of str): A list with the names of the trajecotries.
            The leading path and the file extension is omitted.
        name_arr (np.ndarray of str): An array with len(name_arr) == n_frames.
            This array keeps track of each frame in this object by identifying each
            frame with a filename. This can be useful, when frames are mixed inside
            an TrajEnsemble class.
        index_arr (np.ndarray of str): index_arr.shape = (n_frames, 2). This array keeps track
            of each frame with two ints. One giving the number of the trajectory, the other the frame.

    Examples:
        >>> # Create a trajectory ensemble from a list of files
        >>> import encodermap as em
        >>> trajs = em.TrajEnsemble(['https://files.rcsb.org/view/1YUG.pdb', 'https://files.rcsb.org/view/1YUF.pdb'])
        >>> # trajs are inernally numbered
        >>> print([traj.traj_num for traj in trajs])
        [0, 1]
        >>> # Build a new traj from random frames
        >>> # Let's say frame 2 of traj 0, frame 5 of traj 1 and again frame 2 of traj 0
        >>> # Doing this every frame will now be its own trajectory for easier bookkepping
        >>> arr = np.array([[0, 2], [1, 5], [0, 2]])
        >>> new_trajs = trajs[arr]
        >>> print(new_trajs.n_trajs)
        3
        >>> # trace back a single frame
        >>> frame_num = 28
        >>> index = trajs.index_arr[frame_num]
        >>> print('Frame {}, originates from trajectory {}, frame {}.'.format(frame_num, trajs.basenames[index[0]], index[1]))
        Frame 28, originates from trajectory 1YUF, frame 13.

    """

    def __init__(
        self,
        trajs: Union[list[str], list[md.Trajectory], list[SingleTraj], list[Path]],
        tops: Optional[list[str]] = None,
        backend: Literal["mdtraj", "no_load"] = "no_load",
        common_str: Optional[list[str]] = None,
        basename_fn: Optional[Callable] = None,
    ) -> None:
        """Initialize the Info class with two lists of files.

        Args:
            trajs (Union[list[str], list[md.Trajectory], list[SingleTraj], list[Path]]):
                List of strings with paths to trajectories.
            tops (Optional[list[str]]): List of strings with paths to reference pdbs.
            backend (str, optional): Chooses the backend to load trajectories.
                * 'mdtraj' uses mdtraj which loads all trajecoties into RAM.
                * 'no_load' creates an empty trajectory object.
                Defaults to 'no_load'.
            common_str (list of str, optional): If you want to include trajectories with
                different topology. The common string is used to pair traj-files
                (.xtc, .dcd, .lammpstrj) with their topology (.pdb, .gro, ...). The common-string
                should be a substring of matching trajs and topologies.
            basename_fn (Union[None, function], optional): A function to apply to the `traj_file` string to return the
                basename of the trajectory. If None is provided, the filename without extension will be used. When
                all files are named the same and the folder they're in defines the name of the trajectory you can supply
                `lambda x: split('/')[-2]` as this argument. Defaults to None.

        Raises:
            TypeError: If some of your inputs are mismatched. If your input lists
                contain other types than str or mdtraj.Trajecotry.

        """
        # defaults
        if not trajs:
            raise BadError("You provided an empty list for `trajs`.")

        self.backend = backend

        # basename function
        if basename_fn is None:
            basename_fn = lambda x: os.path.basename(x).split(".")[0]
        self.basename_fn = basename_fn

        # common string
        if common_str is None:
            common_str = []
        if isinstance(common_str, str):
            self.common_str = [common_str]
        else:
            self.common_str = common_str

        # loading with setters
        if tops is None:
            tops = []
        self._top_files = tops
        if all([isinstance(traj, str) for traj in trajs]):
            if self._top_files == [] and all(
                ["." + top.split(".")[-1] in _TOPOLOGY_EXTS for top in trajs]
            ):
                self._top_files = trajs
        if isinstance(tops, str):
            self._top_files = [tops]
        self.traj_files = trajs

    @classmethod
    def from_textfile(
        cls,
        fname,
        basename_fn=None,
    ) -> TrajEnsemble:
        """Creates an `TrajEnsemble` object from a textfile.

        The textfile needs to be space-separated with two or three columns.
        Column 1: The trajectory file.
        Column 2: The corresponding topology file (If you are using .h5 trajs,
            column 1 and 2 will be identical).
        Column 3: The common string of the trajectory. This column can be left
            out, which will result in an `TrajEnsemble` without common_strings.

        Args:
            fname (str): File to be read.
            basename_fn (Union[None, function], optional): A function to apply to the `traj_file` string to return the
                basename of the trajectory. If None is provided, the filename without extension will be used. When
                all files are named the same and the folder they're in defines the name of the trajectory you can supply
                `lambda x: split('/')[-2]` as this argument. Defaults to None.

        Returns:
            TrajEnsemble: An instantiated TrajEnsemble class.

        """
        from ..trajinfo import info_single

        traj_files = []
        top_files = []
        common_str = []

        with open(fname, "r") as f:
            for row in f:
                traj_files.append(row.split()[0])
                top_files.append(row.split()[1])
                try:
                    common_str.append(row.split()[2])
                except IndexError:
                    common_str.append("")

        trajs = []
        for i, (traj_file, top_file, cs) in enumerate(
            zip(traj_files, top_files, common_str)
        ):
            trajs.append(info_single.SingleTraj(traj_file, top_file, cs))

        return cls(trajs, common_str=np.unique(common_str), basename_fn=basename_fn)

    @classmethod
    def from_xarray(
        cls,
        fnames,
        basename_fn=None,
    ) -> TrajEnsemble:
        from ..trajinfo import SingleTraj

        if isinstance(fnames, str):
            fnames = glob.glob(fnames)
        ds = xr.open_mfdataset(
            fnames,
            group="CVs",
            concat_dim="traj_num",
            combine="nested",
            engine="h5netcdf",
        )
        trajs = []
        for traj_num in ds.traj_num:
            sub_ds = ds.sel(traj_num=traj_num)
            fname = [
                fname for fname in fnames if str(sub_ds.traj_name.values) in fname
            ][0]
            traj = SingleTraj(fname, traj_num=traj_num)
            traj._CVs = sub_ds
            trajs.append(traj)
        return cls(trajs, basename_fn=basename_fn)

    @property
    def traj_files(self) -> list[str]:
        """list: A list of the traj_files of the individual SingleTraj classes."""
        return self._traj_files

    @traj_files.setter
    def traj_files(self, trajs):
        from ..trajinfo import info_single

        # fill this lists
        self.trajs = []

        if all([isinstance(traj, Path) for traj in trajs]):
            trajs = [str(traj) for traj in trajs]

        if all([isinstance(i, md.Trajectory) for i in trajs]):
            self.backend = "mdtraj"
            self.trajs = [
                info_single.SingleTraj(traj, traj_num=i, basename_fn=self.basename_fn)
                for i, traj in enumerate(trajs)
            ]
            self._traj_files = []
            self._top_files = []
        elif all([i.__class__.__name__ == "SingleTraj" for i in trajs]):
            self.trajs = trajs
            self._top_files = [traj.top_file for traj in self.trajs]
            self._traj_files = [traj.traj_file for traj in self.trajs]
            # check backends and common str
            if (
                not all([traj.backend == "no_load" for traj in trajs])
                or self.backend == "mdtraj"
            ):
                (traj.load_traj() for traj in trajs)
            for i, traj in enumerate(trajs):
                if traj.traj_num is None:
                    traj.traj_num = i
        elif all([isinstance(i, str) for i in trajs]) and self.top_files:
            # find common_str matches in top_files and traj_files
            (
                self._traj_files,
                self._top_files,
                self._common_str,
            ) = get_full_common_str_and_ref(trajs, self._top_files, self.common_str)
            for i, (t, top, cs) in enumerate(
                zip(self._traj_files, self._top_files, self._common_str)
            ):
                self.trajs.append(
                    info_single.SingleTraj(
                        traj=t,
                        top=top,
                        backend=self.backend,
                        common_str=cs,
                        traj_num=i,
                        basename_fn=self.basename_fn,
                    )
                )
        elif all([isinstance(i, str) for i in trajs]) and not self.top_files:
            for i, traj_file in enumerate(trajs):
                self.trajs.append(
                    info_single.SingleTraj(
                        traj=traj_file,
                        basename_fn=self.basename_fn,
                        traj_num=i,
                    )
                )
        else:
            raise TypeError(
                "The objects in the list are not of the correct type or inconsistent. "
                f"You provided {[c.__class__.__name__ for c in trajs]}. "
                "Please provide a list of `str`, list of `mdtraj.Trajectory` or list of `SingleTraj`."
            )

    @property
    def top(self) -> list[md.Topology]:
        """list: Returns a minimal set of mdtraj.Topologies.

        If all trajectories share the same topology a list
        with len 1 will be returned.

        """
        out = []
        for traj in self.trajs:
            try:
                if traj.top not in out:
                    out.append(traj.top)
            except IOError:
                print(self.trajs)
                print("A rather peculiar error")
                raise
        return out

    @property
    def id(self) -> np.ndarray:
        """np.ndarray: Duplication of self.index_arr"""
        return self.index_arr

    @property
    def top_files(self) -> list[str]:
        """list: Returns minimal set of topology files.

        If yoy want a list of top files with the same
        length as self.trajs use self._top_files and
        self._traj_files.

        """
        return list(dict.fromkeys(self._top_files))

    @property
    def n_residues(self) -> int:
        """list: List of number of residues of the SingleTraj classes"""
        return [traj.n_residues for traj in self.trajs]

    @property
    def basenames(self) -> list[str]:
        """list: List of the basenames in the Info single classes."""
        return [traj.basename for traj in self.trajs]

    @property
    def traj_nums(self) -> list[int]:
        """list: Number of info single classes in self."""
        return [traj.traj_num for traj in self.trajs]

    @property
    def n_trajs(self) -> int:
        """int: Number of trajectories in this encemble."""
        return len(self.trajs)

    @property
    def _CVs(self) -> xr.Dataset:
        """xarray.Dataset: Returns x-array Dataset of matching CVs. stacked along the trajectory-axis."""
        if len(self.top) > 1:
            print(
                "This `TrajEnsemble` object contains mulitple topologies. The "
                "output of _CVs can contain nans for some features."
            )
        return xr.combine_nested(
            [traj._CVs for traj in self.trajs],
            concat_dim="traj_num",
            compat="broadcast_equals",
            fill_value=np.nan,
            coords="all",
            join="outer",
        )
        # return xr.concat([traj._CVs for traj in self.trajs], dim='traj_num')
        # except ValueError:
        #     # when ensemble is used, concatenation is more difficult
        #     # some values cannot be combined into a smooth array (with nan)
        #     DAs = {}
        #     matching_keys = list(
        #         set.intersection(*[set(traj.CVs.keys()) for traj in self.trajs])
        #     )

        #     for key in matching_keys:
        #         data = []
        #         feat_name = key.upper()
        #         for traj in self.trajs:
        #             data.append(traj._CVs[key].values.squeeze())
        #         data = np.stack(data)
        #         da = make_ensemble_xarray(key, data)
        #         DAs[key] = da
        #     ds = xr.Dataset(DAs)

        #     return ds

    @property
    def CVs(self) -> dict[str, np.ndarray]:
        """dict: Returns dict of CVs in SingleTraj classes. Only CVs with the same names
        in all SingleTraj classes are loaded.

        """
        if (
            not all([traj.CVs for traj in self.trajs])
            or [traj.CVs for traj in self.trajs] == []
        ):
            return {}
        else:
            CVs = {}
            matching_keys = list(
                set.intersection(*[set(traj.CVs.keys()) for traj in self.trajs])
            )
            dropping_keys = set(matching_keys).difference(
                *[set(traj.CVs.keys()) for traj in self.trajs]
            )
            if dropping_keys:
                print(
                    f"The CVs {dropping_keys} will not be in the `CVs` dictionary, "
                    f"as they are only present in some, but not all of the {len(self.trajs)} "
                    f"trajectories. You can access them with "
                    f"`TrajEnsemble([t for t in trajs if any([cv in {dropping_keys} for cv in t.CVs.keys()])])`"
                )
            if matching_keys != []:
                for key in matching_keys:
                    data = []
                    for traj in self.trajs:
                        data.append(traj._CVs[key].values)
                    # check if all shapes are the same
                    shapes = [d.shape[2:] for d in data]
                    if not len(set(shapes)) == 1:
                        print(
                            f"I am not returning the CVs {key}. As, some trajectories have different "
                            f"shapes for these CVs. The shapes are {set(shapes)}."
                        )
                        continue
                    if np.all(
                        [
                            any([isinstance(ind, int) for ind in traj.index])
                            for traj in self.trajs
                        ]
                    ):
                        data = np.vstack([d for d in data]).squeeze()
                    else:
                        data = np.concatenate([d.squeeze() for d in data], axis=0)
                    CVs[key] = data
            return CVs

    @property
    def locations(self) -> list[str]:
        """list: Duplication of self.traj_files but using the trajs own traj_file attribute.
        Ensures that traj files are always returned independent from current load state."""
        return [traj.traj_file for traj in self.trajs]

    @property
    def index_arr(self) -> np.ndarray:
        """np.ndarray: Returns np.ndarray with ndim = 2. Clearly assigning every loaded frame an identifier of
        traj_num (self.index_arr[:,0]) and frame_num (self.index_arr[:,1]). Can be used to create
        a unspecified subset of frames and can be useful when used with clustering.

        """
        # can also be made to use the SingleTraj.index_arr attribute
        # but doing it this way the traj is loaded.
        # which might slow down thing significantly
        return np.vstack([traj.id for traj in self.trajs])

    @property
    def name_arr(self) -> np.ndarray:
        """np.ndarray: Trajectory names with the same length as self.n_frames."""
        name_arr = []
        if not np.all([traj.n_frames for traj in self.trajs]):
            return np.array(name_arr)
        else:
            for x, traj in enumerate(self.trajs):
                names = [traj.basename for i in range(traj.n_frames)]
                name_arr.extend(names)
            return np.array(name_arr)

    @property
    def n_frames(self) -> int:
        """int: Sum of the loaded frames."""
        return sum([traj.n_frames for traj in self.trajs])

    @property
    def frames(self) -> list[int]:
        """list: Frames of individual trajectories."""
        return [traj.n_frames for traj in self.trajs]

    @property
    def CVs_in_file(self) -> bool:
        """bool: Is true, if CVs can be loaded from file. Can be used to build a data generator from."""
        return all([traj.CVs_in_file for traj in self.trajs])

    @property
    def traj_joined(self) -> md.Trajectory:
        """mdtraj.Trajectory: Returns a mdtraj Trajectory with every frame of this class appended along the time axis.

        Can also work if different topologies (with the same number of atoms) are loaded.
        In that case, the first frame in self will be used as topology parent and the remaining frames'
        xyz coordinates are used to position the parents' atoms accordingly.


        Examples:
            >>> import encodermap as em
            >>> single_mdtraj = trajs.split_into_frames().traj_joined
            >>> print(single_mdtraj)
            <mdtraj.Trajectory with 31 frames, 720 atoms, 50 residues, without unitcells>

        """
        # use traj[0] of the trajs list as the traj from which the topology will be used
        parent_traj = self.trajs[0].traj

        # join the correct number of trajs
        # by use of the divmod method, the frames parent_traj traj will be
        # appended for a certain amount, until the remainder of the division
        # is met by that time, the parent traj will be sliced to fill the correct number of frames
        try:
            no_of_iters, rest = divmod(self.n_frames, parent_traj.n_frames)
        except Exception:
            print(parent_traj.n_frames)
            raise
        for i in range(no_of_iters + 1):
            if i == 0:
                dummy_traj = copy.deepcopy(parent_traj)
            elif i == no_of_iters:
                dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj)[:rest])
            else:
                dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj))

        # some checks
        assert self.n_frames == dummy_traj.n_frames
        assert self.n_frames == len(self.trajs)

        # change the xyz coordinates of dummy_traj according to the frames in joined trajs
        for i, traj in enumerate(self.trajs):
            dummy_traj.xyz[i] = traj.xyz

        return dummy_traj

    @property
    def xyz(self) -> np.ndarray:
        """np.ndarray: xyz coordinates of all atoms stacked along the traj-time axis. Only works if all trajs share the same topology."""
        if len(self.top) == 1:
            xyz = np.vstack([traj.xyz for traj in self.trajs])
            return xyz
        else:
            try:
                xyz = np.vstack([traj.xyz for traj in self.trajs])
                return xyz
            except Exception as e:
                msg = (
                    "Non consistent topologies don't allow to return a "
                    "common xyz. This could be achived by implementing a "
                    "high-dimensional masked numpy array with nans at "
                    "non-defined positions."
                )
                e2 = Exception(msg)
                raise e2 from e

    def split_into_frames(self, inplace: bool = False) -> None:
        """Splits self into separate frames.

        Args:
            inplace (bool, optionale): Whether to do the split inplace or not.
                Defaults to False and thus, returns a new `TrajEnsemble` class.

        """
        frames = []
        for i, frame in self.iterframes():
            frames.append(frame)
        if inplace:
            self = TrajEnsemble(frames)
        else:
            return TrajEnsemble(frames)

    def save_CVs(self, path: Union[str, Path]) -> None:
        """Saves the CVs to a NETCDF file using xarray."""
        self._CVs.to_netcdf(path, format="NETCDF4", engine="h5netcdf")

    def load_CVs(
        self,
        data: TrajEnsembleFeatureType,
        attr_name: Optional[str] = None,
        cols: Optional[list[int]] = None,
        labels: Optional[list[str]] = None,
        directory: Optional[Union[str, Path]] = None,
        ensemble: bool = False,
    ) -> None:
        """Loads CVs in various ways. Easiest way is to provide a single numpy array and a name for that array.

        Besides np.ndarrays, files (.txt and .npy) can be loaded. Features or Featurizers can be
        provided. An xarray.Dataset can be provided. A str can be provided that either
        is the name of one of encodermap's features (encodermap.loading.features) or the
        string can be 'all', which loads all features required for encodermap's
        `AngleDihedralCarteisanEncoderMap` class.

        Args:
            data (Union[str, list, np.ndarray, 'all', xr.Dataset]): The CV to load. When a numpy array is provided,
                it needs to have a shape matching n_frames. The data is distributed to the trajs.
                When a list of files is provided, len(data) needs to match n_trajs. The first file
                will be loaded by the first traj and so on. If a list of np.arrays is provided,
                the first array will be assigned to the first traj. If a None is provided,
                the arg directory will be used to construct
                fname = directory + traj.basename + '_' + attr_name. The filenames will be used.
                These files will then be loaded and put into the trajs. Defaults to None.
            attr_name (Optional[str]): The name under which the CV should be found in the class.
                Choose whatever you like. `highd`, `lowd`, `dists`, etc...
            cols (Optional[list[int]]): A list of integers indexing the columns of the data to be loaded.
                This is useful, if a file contains feature1, feature1, ..., feature1_err, feature2_err
                formatted data. This option will only be used, when loading multiple .txt files. If None is
                provided all columns will be loaded. Defaults to None.
            labels (list): A list containing the labels for the dimensions of the data.
                Defaults to None.
            directory (Optional[str]): The directory to save the data at, if data is an instance of `em.Featurizer`
                and this featurizer has in_memory set to Fase. Defaults to ''.
            ensemble (bool): Whether the trajs in this class belong to an ensemble. This implies that
                they contain either the same topology or are very similar (think wt, and mutant). Setting this
                option True will try to match the CVs of the trajs onto a same dataset. If a VAL residue has been replaced
                by LYS in the mutant, the number of sidechain dihedrals will increase. The CVs of the trajs with
                VAL will thus contain some NaN values. Defaults to False.

        Raises:
            TypeError: When wrong Type has been provided for data.

        """
        from .trajinfo_utils import load_CVs_ensembletraj

        load_CVs_ensembletraj(self, data, attr_name, cols, labels, directory, ensemble)

    def save(self):
        raise NotImplementedError()

    def _return_trajs_by_index(self, index: list[int]) -> TrajEnsemble:
        """Creates a TrajEnsemble object with the trajs specified by index."""
        # new_common_str = []
        # new_trajs = []
        # new_refs = []
        # new_lowd = []
        # new_highd = []
        # for i, traj in enumerate(self.trajs):
        #     if i not in index:
        #         continue
        #     new_common_str.append(traj.common_str)
        #     new_trajs.append(traj.traj_file)
        #     new_refs.append(traj.top_file)
        #     new_lowd.append(traj.lowd)
        #     new_highd.append(traj.highd)
        # new_common_str = list(set(new_common_str))
        # trajs_subset = TrajEnsemble(new_trajs, tops=new_refs, backend=self.backend, common_str=new_common_str)
        # return trajs_subset

        # is this faster?
        new_common_str = []
        for i, traj in enumerate(self.trajs):
            if i not in index:
                continue
            new_common_str.append(traj.common_str)
        new_common_str = list(set(new_common_str))
        for i, ind in enumerate(index):
            if i == 0:
                trajs_subset = self.trajs[ind]._gen_ensemble()
            else:
                new_traj = self.trajs[ind]._gen_ensemble()
                trajs_subset = trajs_subset + new_traj
        trajs_subset.common_str = new_common_str
        trajs_subset.basename_fn = self.basename_fn
        return trajs_subset

    def _pyemma_indexing(self, key: np.ndarray) -> TrajEnsemble:
        """Returns a new TrajEnsemble by giving the indices of traj and frame"""
        if key.ndim == 1:
            key = key.reshape(len(key), 1).T
        trajs = []
        for i, (num, frame) in enumerate(key):
            trajs.append(self.trajs[num][frame])
        return TrajEnsemble(
            trajs, basename_fn=self.basename_fn, common_str=self.common_str
        )

    def subsample(
        self,
        stride: int,
        inplace: bool = False,
    ) -> Union[None, TrajEnsemble]:
        """Returns a subset of this TrajEnsemble class given the provided stride.

        This is a faster alternative than using the trajs[trajs.index_arr[::1000]]
        when HDF5 trajs are used, because the slicing information is saved in the
        respective SingleTraj classes and loading of single frames is faster in
        HDF5 formatted trajs.

        Note:
            The result from `subsample()` is different from `trajs[trajs.index_arr[::1000]]`.
            With subsample every trajectory is subsampled independently. Cosnider
            a TrajEnsemble with two SingleTraj trajectories with 18 frames each.
            `subsampled = trajs.subsample(5)` would return an TrajEnsemble with two
            trajs with 3 frames each (`subsampled.n_frames` is 6). Whereas
            `subsampled = trajs[trajs.index_arr[::5]]` would return an TrajEnsemble
            with 7 SingleTrajs with 1 frame each (`subsampled.n_frames` is 7).
            Because the times and frame numbers are saved all the time this should not
            be too much of a problem.


        """
        trajs = []
        for i, traj in enumerate(self.trajs):
            _ = traj[slice(None, None, stride)]
            trajs.append(_)
        if inplace:
            self = TrajEnsemble(
                trajs, common_str=self.common_str, basename_fn=self.basename_fn
            )
        else:
            return TrajEnsemble(
                trajs, common_str=self.common_str, basename_fn=self.basename_fn
            )

    def get_single_frame(self, key: int) -> SingleTraj:
        """Returns a single frame from all loaded trajectories.

        Consider a TrajEnsemble class with two SingleTraj classes. One has 10 frames,
        the other 5 (`trajs.n_frames` is 15). Calling `trajs.get_single_frame(12)`
        is equal to calling `trajs[1][1]`.

        Args:
            key (int): The frame to return.

        Returns:
            encodermap.SingleTraj: The frame.

        """
        # some input checks
        if self.n_frames == 0:
            raise BadError(
                "Indexing a no_load backend does not work. I need some information about the frames in each trajectory. Please load either highd or lowd."
            )
        if key >= self.n_frames:
            raise IndexError(
                "index {} is out of bounds for trajectory with {} frames".format(
                    key, self.n_frames
                )
            )
        if not isinstance(key, (int, np.int32, np.int64)):
            raise IndexError(
                "if you want a single frame, please provide an integer. If you want multiple frames use ep.TrajEnsemble[]"
            )

        if len(self.trajs) == 1:
            return self.trajs[0][key]
        else:
            num, frame = np.hstack(
                [
                    np.array([np.full(t.n_frames, t.traj_num), np.arange(t.n_frames)])
                    for t in self.trajs
                ]
            ).T[key]
            traj_out = self.trajs[num][frame]
            return traj_out

    def unload(self) -> None:
        """Unloads all trajs in self."""
        [traj.unload() for traj in self]
        self.backend = "no_load"

    def load_trajs(self) -> None:
        """Loads all trajs in self."""
        [traj.load_traj() for traj in self]
        self.backend = "mdtraj"

    def itertrajs(self) -> Iterator[tuple[int, SingleTraj]]:
        """Generator over the SingleTraj classes.

        Yields:
            tuple: A tuple containing the following:
                int: A loop-counter integer. Is identical with traj.traj_num.
                encodermap.SingleTraj: An SingleTraj object.

        Examples:
            >>> import encodermap as em
            >>> trajs = em.TrajEnsemble(['https://files.rcsb.org/view/1YUG.pdb', 'https://files.rcsb.org/view/1YUF.pdb'])
            >>> for i, traj in trajs.itertrajs():
            ...     print(traj.basename)
            1YUG
            1YUF

        """
        for i, traj in enumerate(self.trajs):
            yield i, traj

    def iterframes(self) -> Iterator[tuple[int, SingleTraj]]:
        """Generator over the frames in this class.

        Yields:
            tuple: A tuple containing the following:
                int: A loop-counter integer.
                encodermap.SingleTraj: An SingleTraj object.

        Examples:
            >>> import encodermap as em
            >>> trajs = em.TrajEnsemble(['https://files.rcsb.org/view/1YUG.pdb', 'https://files.rcsb.org/view/1YUF.pdb'])
            >>> for i, frame in trajs.iterframes():
            ...     print(frame.basename)
            ...     print(frame.n_frames)
            ...     break
            1YUG
            1

        """
        iter_ = 0
        for traj in self.trajs:
            for frame in traj:
                yield iter_, frame
                iter_ += 1

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __getitem__(self, key):
        if isinstance(key, (int, np.int32, np.int64)):
            return self.trajs[key]
        elif isinstance(key, list):
            new_class = self._return_trajs_by_index(key)
            return new_class
        elif isinstance(key, np.ndarray):
            if key.ndim == 1:
                new_class = self._return_trajs_by_index(key)
                return new_class
            elif key.ndim == 2:
                new_class = self._pyemma_indexing(key)
                return new_class
            else:
                raise IndexError(
                    "Passing a key with more than 2 dims makes no sense. One dim for trajs, one for frames. Your key has {} dims.".format(
                        key.ndims
                    )
                )
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.n_trajs)
            list_ = list(range(start, stop, step))
            new_class = self[list_]
            return new_class
        else:
            raise IndexError("Invalid argument for slicing.")

    def __reversed__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        # check if traj_files and ids are the same
        if len(self) != len(other):
            return False
        else:
            import functools

            same_strings = functools.reduce(
                lambda x, y: x and y,
                map(
                    lambda a, b: a == b,
                    [traj.traj_file for traj in self.trajs],
                    [traj2.traj_file for traj2 in other.trajs],
                ),
                True,
            )
            same_ids = all(
                [
                    np.array_equal(traj1.id, traj2.id)
                    for traj1, traj2 in zip(self.trajs, other.trajs)
                ]
            )
            same_CVs = self._CVs.equals(other._CVs)
            return same_strings and same_ids and same_CVs

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.n_trajs:
            raise StopIteration
        else:
            self._index += 1
            return self.trajs[self._index - 1]

    def __add__(self, y):
        """Addition of two TrajEnsemble objects returns new TrajEnsemble with
        trajectories joined along the traj axis.

        """
        # decide on the new backend
        if self.backend != y.backend:
            print("Mismatch between the backends. Using 'mdtraj'.")
            y.load_trajs()
            self.load_trajs()
        # build a common_str_ array with the correct number of entries
        # use this to create a new class
        # if there are no references in self or y. One of them was created from mdtraj.Trajectories
        if not any([self._top_files + y._top_files]):
            new_class = self.__class__(self.trajs + y.trajs, backend=self.backend)
        else:
            common_str_ = (
                get_full_common_str_and_ref(
                    self.traj_files, self._top_files, self.common_str
                )[2]
                + get_full_common_str_and_ref(y.traj_files, y._top_files, y.common_str)[
                    2
                ]
            )
            common_str_ = list(dict.fromkeys(common_str_))
            new_class = self.__class__(
                self.traj_files + y.traj_files,
                self._top_files + y._top_files,
                backend=self.backend,
                common_str=common_str_,
            )
        # put the trajs directly in the new class. This way the frames of the SingleTraj classes are preserved
        new_class.trajs = self.trajs + y.trajs

        return new_class

    def __getattr__(self, attr: str):
        if attr in self.CVs:
            return self.CVs[attr]
        else:
            return self.__getattribute__(attr)

    def _string_summary(self) -> str:
        if all([i.trajectory for i in self.trajs]):
            types = "frames"
            amount = self.n_frames
        else:
            types = "trajs"
            amount = self.n_trajs
        s = f"encodermap.TrajEnsemble object. Current backend is {self.backend}. Containing {amount} {types}."
        if "n_frames" in self.__dict__.keys():
            s += f" In {self.n_frames} frames total."
        if self.common_str:
            s += f" Common str is {self.common_str}."
        if self.CVs:
            for key, value in self.CVs.items():
                s += f" CV {key} with shape {value.shape} loaded."
        else:
            s += " Not containing any CVs."
        return s

    def __len__(self) -> int:
        return self.n_frames

    def __str__(self) -> str:
        return self._string_summary()

    def __repr__(self) -> str:
        return f"<{self._string_summary()} Object at 0x{id(self):02x}>"

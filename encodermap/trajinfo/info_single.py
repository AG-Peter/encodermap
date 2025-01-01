# -*- coding: utf-8 -*-
# encodermap/trajinfo/info_single.py
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
"""Classes to work with ensembles of trajectories.

The statistics of a protein can be better described by an ensemble of proteins,
rather than a single long trajectory. Treating a protein in such a way opens great
possibilities and changes the way one can treat molecular dynamics data.
Trajectory ensembles allow:
    * Faster convergence via adaptive sampling.
    * Better anomaly detection of unique structural states.

This subpackage contains two classes which are containers of trajectory data.
The SingleTraj trajectory contains information about a single trajectory.
The TrajEnsemble class contains information about multiple trajectories. This adds
a new dimension to MD data. The time and atom dimension are already established.
Two frames can be appended along the time axis to get a trajectory with multiple
frames. If they are appended along the atom axis, the new frame contains the
atoms of these two. The trajectory works in a similar fashion. Adding two trajectories
along the trajectory axis returns a trajectory ensemble, represented as a TrajEnsemble
class in this package.

See also:
    http://statisticalbiophysicsblog.org/?p=92

"""

################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import errno
import os
import re
import sys
import warnings
from copy import deepcopy
from io import StringIO
from pathlib import Path

# Third Party Imports
import numpy as np
import tables
from optional_imports import _optional_import

# Encodermap imports
from encodermap._typing import CanBeIndex, CustomAAsDict
from encodermap.misc.misc import _TOPOLOGY_EXTS
from encodermap.misc.xarray_save_wrong_hdf5 import save_netcdf_alongside_mdtraj
from encodermap.trajinfo.load_traj import _load_pdb_from_uri, _load_traj


################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")
mda = _optional_import("MDAnalysis")
h5 = _optional_import("h5py")
xr = _optional_import("xarray")
h5py = _optional_import("h5py")
pd = _optional_import("pandas")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, overload


if TYPE_CHECKING:  # pragma: no cover
    # Third Party Imports
    import h5py as h5
    import MDAnalysis as mda
    import mdtraj as md
    import xarray as xr

    # Encodermap imports
    from encodermap.trajinfo.info_all import TrajEnsemble
    from encodermap.trajinfo.trajinfo_utils import CustomTopology, SingleTrajFeatureType


################################################################################
# Globals
################################################################################


__all__: list[str] = ["SingleTraj"]


################################################################################
# Utilities
################################################################################


class MixedUpInputs(Exception):
    """For when the user provides trajectories as topologies and vice versa."""

    pass


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


def _hash_numpy_array(x: np.ndarray) -> int:
    hash_value = hash(x.shape)
    hash_value ^= hash(x.strides)
    hash_value ^= hash(x.data.tobytes())
    return hash_value


################################################################################
# Classes
################################################################################


class SingleTrajFsel:
    def __init__(self, other):
        self.other = other

    def __getitem__(self, item: CanBeIndex) -> SingleTraj:
        if self.other.traj_num is None:
            if isinstance(item, (int, np.int64)):
                idx = np.where(self.other.id == item)[0]
            elif isinstance(item, (list, np.ndarray)):
                idx = np.where(np.in1d(self.other.id, np.asarray(item)))[0]
            elif isinstance(item, slice):
                raise NotImplementedError("Currently can't index frames with slice.")
            else:
                raise ValueError(
                    f"The `fsel[]` method of `SingleTraj` takes {CanBeIndex} types, "
                    f"but {type(item)} was provided."
                )
        else:
            if isinstance(item, (int, np.int64)):
                idx = np.where(self.other.id[:, 1] == item)[0]
            elif isinstance(item, (list, np.ndarray)):
                idx = np.where(np.in1d(self.other.id[:, 1], np.asarray(item)))[0]
            elif isinstance(item, slice):
                raise NotImplementedError("Currently can't index frames with slice.")
            else:
                raise ValueError(
                    f"The `fsel[]` method of `SingleTraj` takes {CanBeIndex} types, "
                    f"but {type(item)} was provided."
                )
        if len(idx) == 0:
            raise ValueError(
                f"No frames with frame index {item} in trajectory {self.other} "
                f"with frames: {self.other._frames}"
            )
        return self.other[idx]


class SingleTraj:
    """This class contains the info about a single trajectory.

    This class contains many of the attributes and methods of
    `mdtraj.Trajectory <https://www.mdtraj.org/1.9.8.dev0/api/generated/mdtraj.Trajectory.html>`_.
    It is meant to be used as a standalone single trajetcory or in an ensemble
    defined in the :obj:`encodermap.trajinfo.info_all.TrajEnsemble` class.
    Other than the standard
    `mdtraj.Trajectory <https://www.mdtraj.org/1.9.8.dev0/api/generated/mdtraj.Trajectory.html>`_
    this class loads the MD data only when needed. The location of the file(s)
    and other attributes like indices (int, list[int], numpy.ndarray, slice)
    are stored until the traj is accessed via the
    :attr:`SingleTraj.traj` attribute. The returned traj is a
    `mdtraj.Trajectory <https://www.mdtraj.org/1.9.8.dev0/api/generated/mdtraj.Trajectory.html>`_
    with the correct number of frames in the correct sequence.

    Besides MD data,  this class keeps track of your collective variables. Oftentimes
    the raw xyz data of a trajectory is not needed for understanding the conformation
    and suitable CVs are selected to represent a protein via internal coordinates
    (torsions, pairwise distances, etc.). This class keeps tack of your CVs.
    Whether you call them ``'highd'`` or ``'torsions'``, this class keeps
    track of everything and returns the values when you need them.

    :obj:`SingleTraj` supports fancy indexing, so you can extract one or more frames
    from a Trajectory as a separate trajectory. For example, to form a
    trajectory with every other frame, you can slice with ``traj[::2]``.

    Note:
        SingleTraj uses the nanometer, degree & picosecond unit system.

    Attributes:
        backend (str): Current state of loading. If ``backend == 'no_load'`` xyz
            data will be loaded from disk, if accessed. If ``backend == 'mdtraj'``,
            the data is already in RAM.
        common_str (str): Substring of :attr:`traj_file` and :attr:`top_file`.
            Used to group multiple trajectory and topology files.
            If ``traj files=['protein1_traj1.xtc', 'protein1_traj2.xtc']``
            have the same topolgy stored in a file called ``'protein1.pdb'``, you
            can load them with ``common_str='protein1'`` together with more ``.xtc``
            and ``.pdb`` files and these two ``.xtc`` files will use the correct
            ``.pdb`` file.
        index (Sequence[Union[None, int, list, numpy.ndarray, slice]]): A sequence of
            fancy slices of the trajectory. When file is loaded from disk, the
            fancy indexes will be applied one after the other.
        traj_num (int): Integer to identify a SingleTraj class in a TrajEnsemble class.
        traj_file (str): Trajectory file used to create this class.
        top_file (str): Topology file used to create this class. If a .h5 trajectory
            was used traj_file and top_file are identical. If a
            ``mdtraj.Trajectory`` was used to create SingleTraj, these strings are empty.

    Examples:
        Load a pdb file with 14 frames from rcsb.org

        >>> import encodermap as em
        >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
        >>> traj  # doctest: +ELLIPSIS
        <encodermap.SingleTraj object...
        >>> traj.n_frames
        14

        Providing common_str sets this attribute.

        >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb", common_str="1GHC")
        >>> traj.common_str
        '1GHC'

        Indexing using integers returns a SingleTraj with only one frame.

        >>> frame = traj[5]
        >>> frame.n_frames
        1

        Indexing can also use lists of integers.

        >>> subset = traj[[0, 1, 5]]
        >>> subset.n_frames
        3

        Further indexing this ``subset`` uses the current trajectory 'as is'. Although
        frame 0, 1, and 5 have been extracted from ``traj``, we get frame 5 from
        ``subset`` by indexing with 2.

        >>> frame = subset[2]
        >>> frame.id
        array([5])

        Indexing using the original frame indices from the file is done using the
        ``fsel[]`` accessor.

        >>> frame = subset.fsel[5]
        >>> frame.id
        array([5])

        Advanced slicing

        >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")[-1:7:-2]
        >>> [frame.id[0] for frame in traj]
        [13, 11, 9]

        The ``traj_num`` argument is mainly used in :obj:`encodermap.TrajEnsemble`,
        but can be provided manually.

        >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb", traj_num=2)
        >>> traj.traj_num
        2

        The argument ``basename_fn`` should be a callable, that takes a string
        and returns a string.

        >>> from pathlib import Path
        >>> def my_basename_fn(filename):
        ...     stem = str(Path(filename).stem)
        ...     return "custom_" + stem
        >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb", basename_fn=my_basename_fn)
        >>> traj.basename
        'custom_1GHC'

        Build a trajectory ensemble from multiple :obj:`SingleTraj` objects.

        >>> traj1 = em.SingleTraj("https://files.rcsb.org/view/1YUG.pdb")
        >>> traj2 = em.SingleTraj("https://files.rcsb.org/view/1YUF.pdb")
        >>> trajs = traj1 + traj2
        >>> print(trajs.n_trajs, trajs.n_frames, [traj.n_frames for traj in trajs])
        2 31 [15, 16]

    """

    _mdtraj_attr = [
        "n_frames",
        "n_atoms",
        "n_chains",
        "n_residues",
        "openmm_boxes",
        "openmm_positions",
        "time",
        "timestep",
        "xyz",
        "unitcell_vectors",
        "unitcell_lengths",
        "unitcell_angles",
        "_check_valid_unitcell",
        "_distance_unit",
        "_have_unitcell",
        "_rmsd_traces",
        "_savers",
        "_string_summary_basic",
        "_time",
        "_time_default_to_arange",
        "_topology",
        "_unitcell_angles",
        "_unitcell_lengths",
        "_xyz",
    ]

    def __init__(
        self,
        traj: Union[str, Path, md.Trajectory],
        top: Optional[Union[str, Path]] = None,
        common_str: str = "",
        backend: Literal["no_load", "mdtraj"] = "no_load",
        index: Optional[Union[int, list[int], np.ndarray, slice]] = None,
        traj_num: Optional[int] = None,
        basename_fn: Optional[Callable[[str], str]] = None,
        custom_top: Optional[CustomAAsDict] = None,
    ) -> None:
        """Initialize the SingleTraj object with location and reference pdb file.

        Args:
            traj (Union[str, mdtraj.Trajectory]): The trajectory. This argument
                can either be the filename of a trajectory file
                (``.xtc, .dcd, .h5, .trr``) or an instance of
                 ``mdtraj.Trajectory``.
            top (Union[str, mdtraj.Topology], optional): The path to the
                topology file. Defaults to :obj:`None`. If a ``mdtraj.Trajectory``
                or a ``.h5`` file is provided in :obj:`traj`, this argument will
                not be used and the topology from the corresponding ``traj``
                argument will be used.
            common_str (str, optional): A string to group traj of similar
                topology. If multiple :obj:`SingleTraj` are grouped in one
                :obj:`encodermap.trajinfo.info_all.TrajEnsemble`, the
                ``common_str`` is used to group them together. Defaults to ''
                which means this instance of :obj:`SingleTraj` won't have a common
                string.
            backend (Literal['no_load', 'mdtraj'], optional): Choose the
                backend to load trajectories.
                    - 'mdtraj' uses mdtraj, which loads all trajectories into RAM.
                    - 'no_load' creates an empty trajectory object.
                Defaults to 'no_load'
            index (Optional[Union[int, list[int], numpy.ndarray, slice]]): An
                integer or a Sequence of ``int``. If an integer is provided,
                only the frame at this position will be loaded once the internal
                ``mdtraj.Trajectory`` is accessed. If an array or list is provided,
                the corresponding frames will be used. Indices always slice the
                trajectory as is, meaning they don't index the original frames
                of the trajectory on disk (see Example section). These indices
                can have duplicates: ``[0, 1, 1, 2, 0, 1]``. A slice object
                can also be provided. Supports fancy slicing (``traj[1:50:3]``).
                If :obj:`None` is provided, the traj is loaded fully.
                Defaults to :obj:`None`.
            traj_num (Union[int, None], optional): If working with multiple
                trajs, this is the easiest unique identifier. If multiple
                :obj:`SingleTraj` are instantiated by
                :obj:`encodermap.trajinfo.info_all.TrajEnsemble` the ``traj_num``
                is used as a unique identifier per traj. Defaults to :obj:`None`.
            basename_fn (Optional[Callable[[str], str]]): A function to apply to
                ``traj`` to give it another identifier. If all your trajs are called
                ``'traj.xtc'`` and only the directory they're in gives them a
                unique identifier, you can provide a function into this
                argument to split the path. The function has to take a :obj:`str` and
                return :obj:`str`. If None is provided, the basename
                is extracted like so: ``lambda x: x.split('/')[0].split('.')[-1]``.
                Defaults to None, in which case the filename without
                extension will be used.
            custom_top: Optional[:obj:`encodermap._typing.CustomAAsDict`]: An
                instance of the :obj:`encodermap.trajinfo.trajinfo_utils.CustomTopology`
                class or a dictionary that can be made into such.

        """
        # Imports
        # Encodermap imports
        from encodermap.misc.xarray import construct_xarray_from_numpy
        from encodermap.trajinfo.trajinfo_utils import CustomTopology

        # defaults
        self.__traj = traj
        self.backend = backend
        self.common_str = common_str
        self.index = index if isinstance(index, tuple) else (index,)
        self.traj_num = traj_num
        self._loaded_once = False if backend == "no_load" else True
        self._orig_frames = np.array([])
        self._CVs = xr.Dataset()

        # custom topology to load dihedral angles
        self._custom_top = custom_top
        if self._custom_top is not None:
            if isinstance(self._custom_top, dict):
                self._custom_top = CustomTopology.from_dict(self._custom_top, traj=self)
        else:
            self._custom_top = CustomTopology(traj=self)

        # _atom indices are for delayed atom-slicing
        self._atom_indices = None

        # decide the basename
        if basename_fn is None:
            basename_fn = lambda x: os.path.basename(x).split(".")[0]
        self.basename_fn = basename_fn

        # save the filename
        if isinstance(traj, str):
            if self._validate_uri(traj):
                self._traj_file = traj
            else:
                self._traj_file = Path(traj)
        elif isinstance(traj, Path):
            self._traj_file = traj
        elif isinstance(traj, md.Trajectory):
            self._traj_file = Path("")
            self._top_file = Path("")
            self.backend = "mdtraj"
            self.trajectory = traj
            self._loaded_once = True
            self._orig_frames = np.arange(traj.n_frames)
        else:
            raise ValueError(
                f"Argument `traj` takes either str, Path, or "
                f"mdtraj.Trajectory. You supplied: {type(traj)}."
            )

        if top is not None:
            if isinstance(top, md.Topology):
                if custom_top is not None:
                    raise Exception(
                        f"Providing an MDTraj Topology as the `top` argument interferes "
                        f"with the argument `custom_topology`. Use one or the other. "
                    )
                self._top_file = Path("")
            else:
                if self._validate_uri(top):
                    self._top_file = top
                else:
                    self._top_file = Path(top)
                if isinstance(self._traj_file, Path):
                    if (
                        self._traj_file.suffix in _TOPOLOGY_EXTS
                        and self._traj_file != self._top_file
                    ):
                        raise MixedUpInputs(
                            f"You probably mixed up the input. Normally you "
                            f"want to instantiate with `SingleTraj(traj, top)`."
                            f"Based on the files and the  extensions you provided "
                            f"(traj={self._traj_file.name} and top="
                            f"{self._top_file.name}), you want to change the "
                            f"order of the arguments, or use keyword arguments."
                        )
        else:
            if isinstance(self._traj_file, Path):
                if self._traj_file.suffix in _TOPOLOGY_EXTS:
                    if self._validate_uri(traj):
                        self._top_file = traj
                    else:
                        self._top_file = self._traj_file
            else:
                self._top_file = self._traj_file

        if self.backend == "no_load":
            self.trajectory = False
            self.topology = False
        else:
            if isinstance(self._traj_file, str) and self._validate_uri(self._traj_file):
                traj = md.load_pdb(str(self.traj_file))
            elif self._traj_file != Path(""):
                try:
                    traj = md.load(str(self._traj_file), top=str(self._top_file))
                except tables.NoSuchNodeError as e:
                    if self.traj_num is None:
                        raise
                    # Local Folder Imports
                    from .info_all import HDF5GroupWrite

                    with HDF5GroupWrite(self.top_file) as h5file:
                        traj = h5file.read_traj(self.traj_num)
            self.trajectory = traj
            self.topology = False
            self._loaded_once = True
            self.topology = self._custom_top.top
            self._orig_frames = np.arange(traj.n_frames)

        # check the h5file. Maybe it contains CVs and topology
        CVs_in_file = False
        custom_top_in_file = False
        common_str_in_file = False
        if isinstance(self._traj_file, Path):
            if self._traj_file.suffix == ".h5":
                with h5.File(self.traj_file, "r") as file:
                    h5_attrs = list(file.attrs.keys())
                    if "CVs" in file.keys():
                        CVs_in_file = True
                    if any(["custom_top" in s for s in h5_attrs]):
                        custom_top_in_file = True
                    if any(["common_str" in s for s in h5_attrs]):
                        common_str_in_file = True

        # if group 'CVs' is in h5store, but no data
        if CVs_in_file:
            test = xr.open_dataset(
                self.traj_file,
                group="CVs",
                engine="h5netcdf",
                backend_kwargs={"phony_dims": "access"},
            )
            if len(test.data_vars) == 0:
                CVs_in_file = False

        # inform the use about a grandfathered traj_num
        if custom_top_in_file:
            h5_top_attrs = list(filter(lambda x: x.startswith("_custom_top"), h5_attrs))
            finds = [re.findall(r"\d+", s) for s in h5_top_attrs]
            if not all([f == [] for f in finds]):
                h5_top_traj_nums = [int(f[-1]) for f in finds]
                if self.traj_num not in h5_top_traj_nums:
                    if len(h5_top_attrs) == 1:
                        h5_top_traj_num = h5_top_traj_nums[0]
                        with h5py.File(self.traj_file, "a") as file:
                            file.attrs["_custom_top"] = file.attrs[h5_top_attrs[0]]
                            del file.attrs[h5_top_attrs[0]]
                        print(
                            f"The file {self.traj_file} was previously saved with the `traj_num`="
                            f"{h5_top_traj_num}, but this instance uses `traj_num`={self.traj_num}. "
                            f"I have overwritten the `traj_num` for the custom_topology in this file."
                        )
                    elif len(h5_top_attrs) > 1:
                        raise Exception(
                            f"The h5File contains multiple '_custom_top' attrs: {h5_top_attrs} "
                            f"of which none matches this Traj's `traj_num`: {self.traj_num}."
                        )

        # inform the use about a grandfathered traj_num
        if common_str_in_file:
            common_str_key = "common_str"
            h5_common_str_traj_num = None
            h5_common_str_attrs = list(
                filter(lambda x: x.startswith("common_str"), h5_attrs)
            )
            finds = [re.findall(r"\d+", s) for s in h5_common_str_attrs]
            if not all([f == [] for f in finds]):
                h5_common_str_traj_nums = [int(f[-1]) for f in finds]
                if self.traj_num not in h5_common_str_traj_nums:
                    if len(h5_common_str_attrs) == 1:
                        h5_common_str_traj_num = h5_common_str_traj_nums[0]
                        with h5py.File(self.traj_file, "a") as file:
                            file.attrs["common_str"] = file.attrs[
                                h5_common_str_attrs[0]
                            ]
                            del file.attrs[h5_common_str_attrs[0]]
                        print(
                            f"The file {self.traj_file} was previously saved with the `traj_num`="
                            f"{h5_common_str_traj_num}, but this instance uses `traj_num`={self.traj_num}. "
                            f"I have overwritten the `traj_num` for the custom_topology in this file."
                        )
                    elif len(h5_common_str_attrs) > 1:
                        raise Exception(
                            f"The h5File contains multiple '_custom_top' attrs: {h5_common_str_attrs} "
                            f"of which none matches this Traj's `traj_num`: {self.traj_num}."
                        )
                else:
                    h5_common_str_traj_num = self.traj_num
            if h5_common_str_traj_num is not None:
                common_str_key = f"common_str_{h5_common_str_traj_num}"

        # maybe load CVs from h5 file
        if CVs_in_file:
            try:
                ds = xr.open_dataset(
                    self.traj_file,
                    group="CVs",
                    engine="h5netcdf",
                    backend_kwargs={"phony_dims": "access"},
                )
                if ds.sizes["traj_num"] > 1:
                    assert self.traj_num in ds.coords["traj_num"], (
                        f"This trajectory with {self.traj_num=} is not in "
                        f"the dataset with traj_nums: {ds.coords['traj_num']}."
                    )
                    ds = ds.sel(traj_num=self.traj_num)
                    ds = ds.expand_dims("traj_num")
                if str(ds.coords["traj_name"].values) != self.basename:
                    ds.coords["traj_name"] = [self.basename]
                ds = ds.dropna("frame_num", how="all")
                self._CVs = ds
            # bad formatted h5 file
            except OSError:
                DAs = {
                    k: construct_xarray_from_numpy(self, i[()], k)
                    for k, i in file["CVs"].items()
                }
                DS = xr.Dataset(DAs)
                self._CVs.update(DS)
            # other exceptions probably due to formatting
            except Exception as e:
                raise Exception(
                    f"The formatting of the data in the file "
                    f"{self.traj_file} is off. Xarray could "
                    f"not load the group 'CVs' and failed with {e}. "
                    f"Some debug: {CVs_in_file=} and {file.keys()=}."
                ) from e

            # get the original frame indices from the dataset
            # this is the only case where we want to overwrite
            # this variable
            if not self._loaded_once:
                self._loaded_once = True
            self._orig_frames = self._CVs["frame_num"].values

            # iteratively apply index
            index = self._orig_frames
            for ind in self.index:
                if ind is not None:
                    index = index[ind]

            # set the _CVs accordingly
            self._CVs = self._CVs.loc[{"frame_num": index}]

        # maybe load _custom_top_from file
        if custom_top_in_file:
            ctop = CustomTopology.from_hdf5_file(
                self.traj_file,
                traj=self,
            )
            if custom_top is not None:
                if self._custom_top != ctop:
                    print(
                        f"The file {self.traj_file} contained a custom_topology: "
                        f"{ctop}. The custom topology you provided {self._custom_top} "
                        f"is different from that in the file. The provided one will "
                        f"take precedent over the one in the file. "
                    )
            else:
                self._custom_top = ctop

        # maybe load common_str form file
        if common_str_in_file:
            with h5py.File(self.traj_file, "r") as f:
                common_str_file = f.attrs[common_str_key]

            if self.common_str == "":
                self.common_str = common_str_file
            else:
                if common_str_file != self.common_str:
                    warnings.warn(
                        f"The file {self.traj_file} contains a "
                        f"`common_str`={common_str_file}, which differs from the "
                        f"`common_str` provided in __init__ ({self.common_str}). "
                        f"I will use the one from "
                        f"__init__ and overwrite the one in the file."
                    )
                    try:
                        with h5py.File(self.traj_file, "a") as f:
                            f.attrs[common_str_key] = self.common_str
                    except OSError as e:
                        warnings.warn(
                            f"I could not write the common str in the file, because "
                            f"it is already opened by another python object: {str(e)}"
                        )

    @classmethod
    def from_pdb_id(cls, pdb_id: str, traj_num: Optional[int] = None) -> SingleTraj:
        """Alternate constructor for the TrajEnsemble class.

        Builds an SingleTraj class from a pdb-id.

        Args:
            pdb_id (str): The 4-letter pdb id.

        Returns:
            SingleTraj: An SingleTraj class.

        """
        url = f"https://files.rcsb.org/view/{pdb_id.upper()}.pdb"
        return cls(url, common_str=pdb_id.upper(), traj_num=traj_num)

    @property
    def featurizer(self):
        # Local Folder Imports
        from ..loading.featurizer import SingleTrajFeaturizer

        if not hasattr(self, "_featurizer"):
            self._featurizer = SingleTrajFeaturizer(self)
        return self._featurizer

    @property
    def indices_chi1(self) -> np.ndarray:
        """numpy.ndarray: A numpy array with shape (n_dihedrals, 4) indexing the
        atoms that take part in this dihedral angle. This index is 0-based."""
        return self._custom_top.indices_chi1()

    @property
    def indices_chi2(self) -> np.ndarray:
        """numpy.ndarray: A numpy array with shape (n_dihedrals, 4) indexing the
        atoms that take part in this dihedral angle. This index is 0-based."""
        return self._custom_top.indices_chi2()

    @property
    def indices_chi3(self) -> np.ndarray:
        """numpy.ndarray: A numpy array with shape (n_dihedrals, 4) indexing the
        atoms that take part in this dihedral angle. This index is 0-based."""
        return self._custom_top.indices_chi3()

    @property
    def indices_chi4(self) -> np.ndarray:
        """numpy.ndarray: A numpy array with shape (n_dihedrals, 4) indexing the
        atoms that take part in this dihedral angle. This index is 0-based."""
        return self._custom_top.indices_chi4()

    @property
    def indices_chi5(self) -> np.ndarray:
        """numpy.ndarray: A numpy array with shape (n_dihedrals, 4) indexing the
        atoms that take part in this dihedral angle. This index is 0-based."""
        return self._custom_top.indices_chi5()

    @property
    def indices_phi(self) -> np.ndarray:
        """numpy.ndarray: A numpy array with shape (n_dihedrals, 4) indexing the
        atoms that take part in this dihedral angle. This index is 0-based."""
        return self._custom_top.indices_phi()

    @property
    def indices_omega(self) -> np.ndarray:
        """numpy.ndarray: A numpy array with shape (n_dihedrals, 4) indexing the
        atoms that take part in this dihedral angle. This index is 0-based."""
        return self._custom_top.indices_omega()

    @property
    def indices_psi(self) -> np.ndarray:
        """numpy.ndarray: A numpy array with shape (n_dihedrals, 4) indexing the
        atoms that take part in this dihedral angle. This index is 0-based."""
        return self._custom_top.indices_psi()

    @property
    def _original_frame_indices(self) -> np.ndarray:
        """numpy.ndarray: If trajectory has not been loaded, it is loaded and the
        frames of the trajectory file on disk are put into a `np.arange()`. If
        the trajectory is sliced in weird ways, this array tracks the original frames.
        """
        if self._loaded_once:
            return self._orig_frames
        else:
            self.load_traj()
            return self._orig_frames

    @property
    def _frames(self) -> np.ndarray:
        """numpy.ndarray: Applies self.index over self._orig_frames."""
        frames = self._orig_frames.copy()
        for ind in self.index:
            if ind is not None:
                frames = frames[ind]
        return np.asarray(frames)

    def _trace(self, CV: Sequence[str]) -> np.ndarray:
        """Creates a low-dimensional representation of the loaded CV data by
        stacking all arguments in `CV` along a single axis.

        If this `SingleTraj` has 100 frames and a CV with shape (100, 50, 3) with
        the name 'cartesians', then `traj._trace` will return a numpy.ndarray of shape
        (100, 150).

        Args:
            CV (Sequence[str]): The CVs to combine in the trace.

        Returns:
            numpy.ndarray: The trace.

        """
        out = []
        for i in CV:
            v = self.CVs[i]
            out.append(v.reshape(v.shape[0], -1))
        return np.concatenate(out)

    @property
    def traj_file(self) -> str:
        """str: The traj file as a string (rather than a `pathlib.Path`)."""
        return str(self._traj_file)

    @property
    def top_file(self) -> str:
        """str: The topology file as a string (rather than a `pathlib.Path`)."""
        return str(self._top_file)

    @property
    def traj(self) -> md.Trajectory:
        """mdtraj.Trajectory: This attribute always returns an mdtraj.Trajectory.
        if `backend` is 'no_load', the trajectory will be loaded into memory and returned.

        """
        if self.backend == "no_load":
            self.load_traj()
            out = self.trajectory
            self.unload()
            return out
        else:
            return self.trajectory

    @property
    def _traj(self):
        """Needs to be here to complete setter.
        Not returning anything, because setter is also not returning anything."""
        pass

    @_traj.setter
    def _traj(self, traj_file):
        """Sets the traj and trajectory attributes. Can be provided str or
        mdtraj.Trajectory and sets the attributes based on the chosen backend."""
        if self.topology:
            reinject_top = deepcopy(self.topology)
        else:
            reinject_top = False
        self.trajectory, _ = _load_traj(
            *self.index,
            traj_file=traj_file,
            top_file=self._top_file,
            traj_num=self.traj_num,
            atom_indices=self._atom_indices,
        )
        if not self._loaded_once:
            self._loaded_once = True
            self._orig_frames = _
        if reinject_top:
            self.trajectory.top = reinject_top
            self.topology = reinject_top

    @property
    def basename(self) -> str:
        """str: Basename is the filename without path and without extension. If `basename_fn` is not None, it will be
        applied to `traj_file`."""
        if self.traj_file:
            if str(self.traj_file) == ".":
                return None
            return self.basename_fn(self.traj_file)

    @property
    def extension(self) -> str:
        """str: Extension is the file extension of the trajectory file (self.traj_file)."""
        if isinstance(self._traj_file, Path):
            return self._traj_file.suffix
        else:
            return "." + self._traj_file.split(".")[-1]

    @property
    def id(self) -> np.ndarray:
        """numpy.ndarray: id is an array of unique identifiers which identify the frames in
        this SingleTraj object when multiple Trajectories are considered.

        If the traj was initialized from an TrajEnsemble class, the traj gets a unique
        identifier (traj_num) which will also be put into the id array, so that id
        can have two shapes ((n_frames, ), (n_frames, 2)) This corresponds to
        self.id.ndim = 1 and self.id.ndim = 2. In the latter case self.id[:,1] are the
        frames and self.id[:,0] is an array full of traj_num.

        """
        values = self._original_frame_indices
        if isinstance(values, (int, np.integer)):
            if self.traj_num is None:
                return np.array([values])
            else:
                return np.array([[self.traj_num, values]])
        else:
            for i, ind in enumerate(self.index):
                if ind is not None:
                    values = values[ind]

            # if reduced all the way to single frame
            if isinstance(values, (int, np.integer)):
                if self.traj_num is None:
                    return np.array([values])
                else:
                    return np.array([[self.traj_num, values]])

            # else
            if self.traj_num is None:
                return values
            else:
                return np.array([np.full(len(values), self.traj_num), values]).T

    @property
    def n_frames(self) -> int:
        """int: Number of frames in traj.

        Loads the traj into memory if not in HDF5 file format. Be aware.

        """
        if any([isinstance(ind, (int, np.integer)) for ind in self.index]):
            self._loaded_once = True
            ind = [i for i in self.index if isinstance(i, (int, np.integer))][0]
            self._orig_frames = ind
            return 1
        elif self._traj_file.suffix == ".h5":
            with h5.File(self.traj_file, "r") as file:
                if self.index == (None,):
                    if (
                        "coordinates" not in list(file.keys())
                        and self.traj_num is not None
                    ):
                        n_frames = np.arange(
                            file[f"coordinates_{self.traj_num}"].shape[0]
                        )
                    else:
                        n_frames = np.arange(file["coordinates"].shape[0])
                else:
                    for i, ind in enumerate(self.index):
                        if i == 0:
                            if (
                                "coordinates" not in list(file.keys())
                                and self.traj_num is not None
                            ):
                                n_frames = np.arange(
                                    file[f"coordinates_{self.traj_num}"].shape[0]
                                )
                            else:
                                n_frames = np.arange(file["coordinates"].shape[0])
                            if ind is not None:
                                n_frames = n_frames[ind]
                        else:
                            if ind is not None:
                                n_frames = n_frames[ind]
                if not self._loaded_once:
                    if (
                        "coordinates" not in list(file.keys())
                        and self.traj_num is not None
                    ):
                        self._orig_frames = np.arange(
                            file[f"coordinates_{self.traj_num}"].shape[0]
                        )
                    else:
                        self._orig_frames = np.arange(file["coordinates"].shape[0])

            # return single int or length of array
            if isinstance(n_frames, (int, np.integer)):
                return n_frames
            else:
                return len(n_frames)
        elif self._traj_file.suffix == ".xtc":
            with mda.coordinates.XTC.XTCReader(self.traj_file) as reader:
                if self.index == (None,):
                    n_frames = np.arange(reader.n_frames)
                else:
                    for i, ind in enumerate(self.index):
                        if i == 0:
                            n_frames = np.arange(reader.n_frames)[ind]
                        else:
                            n_frames = n_frames[ind]
                        if ind is None:
                            n_frames = n_frames[0]
                if not self._loaded_once:
                    self._loaded_once = True
                    self._orig_frames = np.arange(reader.n_frames)
                if isinstance(n_frames, (int, np.integer)):
                    return n_frames
                else:
                    return len(n_frames)
        else:
            self.load_traj()
            return self.traj.n_frames

    @property
    def _n_frames_base_h5_file(self) -> int:
        """int: Can be used to get n_frames without loading an HDF5 into memory."""
        if self.extension == ".h5":
            with h5.File(self.traj_file, "r") as file:
                return file["coordinates"].shape[0]
        else:
            return -1

    @property
    def CVs_in_file(self) -> bool:
        """bool: Is True, if `traj_file` has exyension .h5 and contains CVs."""
        if self.extension == ".h5":
            with h5.File(self.traj_file, "r") as file:
                if "CVs" in file.keys():
                    return True
        return False

    @property
    def n_atoms(self) -> int:
        """int: Number of atoms in traj.

        Loads the traj into memory if not in HDF5 file format. Be aware.

        """
        if self._atom_indices is not None:
            return len(self._atom_indices)
        if self.extension == ".h5":
            with h5.File(self.traj_file, "r") as file:
                try:
                    h5_coordinates_shape = file["coordinates"].shape[1]
                except KeyError as e:
                    if self.traj_num is not None:
                        h5_coordinates_shape = file[
                            f"coordinates_{self.traj_num}"
                        ].shape[1]
                    else:
                        raise Exception(
                            f"Neither key 'coordinates', nor 'coordinates_{self.traj_num}' "
                            f"exists in h5file {self.traj_file} with keys {file.keys()}."
                        ) from e
            return h5_coordinates_shape
        else:
            return self.top.n_atoms

    @property
    def n_residues(self) -> int:
        """int: Number of residues in traj."""
        return self.top.n_residues

    @property
    def n_chains(self) -> int:
        """int: Number of chains in traj."""
        return self.top.n_chains

    @property
    def top(self) -> md.Topology:
        """mdtraj.Topology: The structure of a Topology object is similar to that of a PDB file.

        It consists. of a set of Chains (often but not always corresponding to
        polymer chains).  Each Chain contains a set of Residues, and each Residue
        contains a set of Atoms.  In addition, the Topology stores a list of which
        atom pairs are bonded to each other.
        Atom and residue names should follow the PDB 3.0 nomenclature for all
        molecules for which one exists

        Attributes:
            chains (generator): Iterate over chains.
            residues (generator): Iterate over residues.
            atoms (generator): Iterate over atoms.
            bonds (generator): Iterate over bonds.

        """
        return self._get_top()

    def _get_raw_top(self) -> md.Topology:
        """Reads different files and loads md.Topology from them.

        This topology will *NOT* be corrected with `CustomTopology`.

        Returns:
            mdtraj.Topology: The raw topology.

        """
        # Third Party Imports
        import tables

        # Local Folder Imports
        from .info_all import HDF5GroupWrite

        if self.top_file:
            if self.top_file != ".":
                if self._validate_uri(self.top_file):
                    assert self.top_file.endswith(".pdb")
                    top = _load_pdb_from_uri(self.top_file)
                elif not os.path.isfile(self.top_file):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), self.top_file
                    )
        if self.backend == "no_load" and not self.extension == ".h5" and self.traj_file:
            if self._validate_uri(self.top_file):
                top = _load_pdb_from_uri(self.top_file)
            else:
                top = md.load_topology(self.top_file)
        if self.extension == ".h5":
            try:
                top = md.load_topology(self.top_file)
            except tables.NoSuchNodeError as e:
                if self.traj_num is None:
                    raise e
                with HDF5GroupWrite(self.top_file) as h5file:
                    top = h5file.read_traj(self.traj_num).top
        if self.backend == "no_load" and "top" not in locals():
            try:
                top = md.load_topology(self.top_file)
            except tables.NoSuchNodeError as e:
                if self.traj_num is None:
                    raise e
                with HDF5GroupWrite(self.top_file) as h5file:
                    top = h5file.read_traj(self.traj_num).top
        if self.backend == "mdtraj":
            top = self.traj.top
        else:
            if self._validate_uri(self.top_file):
                top = _load_pdb_from_uri(self.top_file)
            else:
                try:
                    top = md.load_topology(self.top_file)
                except tables.NoSuchNodeError as e:
                    if self.traj_num is None:
                        raise e
                    with HDF5GroupWrite(self.top_file) as h5file:
                        top = h5file.read_traj(self.traj_num).top
        return top

    def _get_top(self) -> md.Topology:
        """Reads different files and loads md.Topology from them.

        Returns:
            mdtraj.Topology: The structure of a Topology object is similar to that of a PDB file.

        """
        if self.topology:
            top = self.topology
        else:
            top = self._custom_top.top
        return top

    def copy(self) -> SingleTraj:
        """Returns a copy of ``self``."""
        return deepcopy(self)

    def del_CVs(self) -> None:
        """Resets the ``_CVs`` attribute to an empty :obj:`xarray.Dataset`."""
        del self._CVs
        self._CVs = xr.Dataset()

    def _calc_CV(self) -> dict[str, np.ndarray]:
        """Returns the current CVs as a dictionary."""
        if self._CVs:
            out = {}
            for key, val in self._CVs.data_vars.items():
                if "feature_indices" in key:
                    if key.startswith("MinRmsdFeature"):
                        continue
                    if "cartesian" in key:
                        assert val.shape[0] == 1, (
                            f"The substring 'feature_indices' is special and can "
                            f"only contain a (1, n_frames) or (1, n_frames, 4) arrays. "
                            f"Your value of {key=} has the shape: {val.shape=} "
                            f"If you have manually "
                            f"loaded a feature with this substring, use a different "
                            f"one. These CVs contain integer indices and not values."
                        )
                    else:
                        assert val.shape[-1] <= 4, (
                            f"The substring 'feature_indices' is special and can "
                            f"only contain a (1, n_frames) or (1, n_frames, <=4) arrays. "
                            f"Your value of {key=} has the shape: {val.shape=} "
                            f"If you have manually "
                            f"loaded a feature with this substring, use a different "
                            f"one. These CVs contain integer indices and not values."
                        )
                    continue
                axis_name = (
                    "feature_axis"
                    if "feature_axis" in val.attrs
                    else "feature_axes" if "feature_axes" in val.attrs else None
                )
                if key == "central_angles_indices":
                    raise Exception(
                        f"{val.shape=}. {axis_name=} {val.attrs[axis_name]=}"
                    )
                if np.any(np.isnan(val)):
                    if axis_name is not None:
                        val = val.dropna(val.attrs[axis_name])
                    else:
                        val = val.dropna(key.upper())
                try:
                    out[key] = val.values.squeeze(0)
                except ValueError as e:
                    raise Exception(f"{key=} {val=}") from e
            return out
        else:
            return {}

    @property
    def CVs(self) -> dict[str, np.ndarray]:
        """dict[str, numpy.ndarray]: Returns a simple dict from the more complicated self._CVs xarray Dataset.

        If self._CVs is empty and self.traj_file is a HDF5 (.h5) file, the contents
        of the HDF5 will be checked, whether CVs have been stored there.
        If not and empty dict will be returned.

        """
        return self._calc_CV()

    def _validate_uri(self, uri: Union[Path, str]) -> bool:
        """Checks whether `uri` is a valid uri."""
        # Encodermap imports
        from encodermap.misc.misc import _validate_uri

        return _validate_uri(str(uri))

    def load_traj(
        self,
        new_backend: Literal["no_load", "mdtraj"] = "mdtraj",
    ) -> None:
        """Loads the trajectory, with a new specified backend.

        After this is called the instance variable self.trajectory
        will contain a mdtraj Trajectory object.

        Args:
            new_backend (str, optional): Can either be:
                    - ``'mdtraj'`` to load the trajectory using mdtraj.
                    - ``'no_load'`` to not load the traj (unload).
                Defaults to ``'mdtraj'``.

        """
        if self.backend == new_backend:
            return
        if self.backend == "mdtraj" and new_backend == "no_load":
            self.unload()
            self.topology = False
        if self.backend == "no_load" and new_backend == "mdtraj":
            self.backend = new_backend
            # call the setter again
            try:
                self._traj = self.traj_file
            except Exception:
                self.backend = "no_load"
                raise
            self.topology = self.top

    def select(
        self,
        sel_str: str = "all",
    ) -> np.ndarray:
        """Execute a selection against the topology.

        Args:
            sel_str (str, optional): What to select. Defaults to 'all'.

        See also:
            https://mdtraj.org/1.9.4/atom_selection.html

        Examples:
            >>> import encodermap as em
            >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
            >>> select = traj.top.select("name CA and resSeq 1")
            >>> select
            array([1])

            >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
            >>> select = traj.top.select("name CA and resSeq 1")
            >>> traj.top.atom(select[0])
            MET1-CA

        """
        return self.top.select(sel_str)

    def unload(
        self,
        CVs: bool = False,
    ) -> None:
        """Clears up RAM by deleting the trajectory info and the CV data.

        If ``CVs`` is set to True the loaded CVs will also be deleted.

        Args:
            CVs (bool, optional): Whether to also delete CVs, defaults to False.

        """
        if self.backend == "no_load":
            return
        self.backend = "no_load"
        for key in self._mdtraj_attr:
            try:
                del self.__dict__[key]
            except KeyError:
                pass
        if CVs:
            self._CVs = xr.Dataset()
        self.trajectory, self.topology = False, False

    def _gen_ensemble(self) -> TrajEnsemble:
        """Creates a :obj:`encodermap.trajinfo.info_all.TrajEnsemble` instance
        with this :obj:`SingleTraj` in it.

        This method is needed to add two SingleTraj objects
        along the 'trajectory' axis with the method add_new_traj.
        This method is also called by the __getitem__ method of the TrajEnsemble class.

        """
        # Encodermap imports
        from encodermap.trajinfo.info_all import TrajEnsemble

        info_all = TrajEnsemble(
            trajs=[self],
            backend=self.backend,
            common_str=[],
            basename_fn=self.basename_fn,
        )
        info_all.trajs[0]._CVs = self._CVs
        info_all.trajs[0].common_str = self.common_str
        info_all.common_str = [self.common_str]
        info_all.trajs[0].traj_num = self.traj_num
        info_all.trajs[0].index = self.index
        info_all.trajs[0]._custom_top = self._custom_top
        info_all.trajs[0].topology = self._custom_top.top
        return info_all

    def _add_along_traj(self, y: SingleTraj) -> TrajEnsemble:
        """Puts self and y into a :obj:`encodermap.trajinfo.info_all.TrajEnsemble`.

        This way the trajectories are not appended along the time
        axis but rather along the 'trajectory' axis.

        Args:
            y (SingleTraj): The other :obj:`SingleTraj` trajectory.

        """
        if self.traj_num == y.traj_num:
            if y.traj_num is not None:
                y.traj_num += 1
            else:
                y.traj_num = 1
        class_1 = self._gen_ensemble()
        class_2 = y._gen_ensemble()
        new_class = class_1 + class_2
        return new_class

    def get_single_frame(self, key: int) -> SingleTraj:
        """Returns a single frame from the trajectory.

        Args:
            key (Union[int, np.int]): Index of the frame.

        Examples:

            Import EncoderMap and load :obj:`SingleTraj`.

            >>> import encodermap as em
            >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
            >>> traj.n_frames
            14

            Load the same traj and give it a ``traj_num`` for recognition in a
            set of multiple trajectories.

            >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb", traj_num=5)
            >>> frame = traj.get_single_frame(2)
            >>> frame.id
            array([[5, 2]])

        """
        return self.__getitem__(key)

    def show_traj(self, gui: bool = True) -> Any:
        """Returns an nglview view object.

        See Also:
            https://nglviewer.org/nglview/latest/

        Returns:
            view (nglview.widget): The nglview widget object.

        """
        # Third Party Imports
        import nglview

        view = nglview.show_mdtraj(self.traj, gui=gui)
        return view

    def dash_summary(self) -> pd.DataFrame:
        """Returns a :obj:`pandas.DataFrame` with useful information about this instance.

        Returns:
            pd.DataFrame: The dataframe.

        """
        dt = self.traj.time
        dt = np.unique(dt[1:] - dt[:-1])
        if len(dt) == 1:
            dt = dt[0]
        elif len(dt) == 0:
            dt = "single frame"
        if self.index == (None,):
            index = "[::]"
        else:
            index = self.index[1:]
        if len(index) == 1:
            index = index[0]
        df = pd.DataFrame(
            {
                "field": [
                    "n_frames",
                    "n_atoms",
                    "dt (ps)",
                    "traj_file",
                    "top_file",
                    "index",
                    "common_str",
                ],
                "value": [
                    self.n_frames,
                    self.n_atoms,
                    dt,
                    self.traj_file,
                    self.top_file,
                    index,
                    self.common_str,
                ],
            }
        )
        return df.astype(str)

    def load_custom_topology(
        self,
        custom_top: Optional[Union[CustomTopology, CustomAAsDict]] = None,
    ) -> None:
        """Loads a custom_topology from a `CustomTopology` class or a dict.

        See Also:
            :obj:`encodermap.trajinfo.trajinfo_utils.CustomTopology`

        Args:
            custom_top: Optional[Union[CustomTopology, CustomAAsDict]]: An
                instance of :obj:`encodermap.trajinfo.trajinfo_utils.CustomTopology`
                or a dictionary that can be made into such.

        """
        # Encodermap imports
        from encodermap.trajinfo.trajinfo_utils import CustomTopology

        if isinstance(custom_top, CustomTopology):
            self._custom_top = custom_top
        else:
            self._custom_top = CustomTopology.from_dict(custom_top, traj=self)
        # overwrite the old topology
        self.topology = self._custom_top.top

    def load_CV(
        self,
        data: SingleTrajFeatureType,
        attr_name: Optional[str] = None,
        cols: Optional[list[int]] = None,
        deg: Optional[bool] = None,
        periodic: bool = True,
        labels: Optional[list[str]] = None,
        override: bool = False,
    ) -> None:
        """Load CVs into traj. Many options are possible. Provide xarray,
        numpy array, em.loading.feature, em.featurizer, and even string!

        This method loads CVs into the SingleTraj instance. Many ways of doing so
        are available:
            - :obj:`numpy.ndarray`: The easiest way. Provide a np array and a name for
                the array, and the data will be saved as an instance variable,
                accesible via `SingleTraj.name`.
            - :obj:`xarray.DataArray`: You can load a multidimensional xarray as
                data into the class. Please refer to xarrays own documentation
                if you want to create one yourself.
            - :obj:`xarray.Dataset`: You can add another dataset to the existing _CVs.
            - :obj:`encodermap.loading.features.Feature`: If you provide one of the
                features from ``encodermap.loading.features`` the resulting
                features will be loaded and also be placed under the set name.
            - :obj:`encodermap.loading.featurizer.Featurizer`: If you provide a
                full featurizer, the data will be generated and be accessible
                as an attribute.
            - str: If a string is provided, the data will be loaded from a
                ``.txt``, ``.npy``, or NetCDF / HDF5 ``.nc`` file.

        Args:
            data (Union[str, numpy.ndarray, xr.DataArray,
                em.loading.features.Feature, em.loading.featurizer.Featurizer]):
                The CV to load. Either as :obj:`numpy.ndarray`,
                :obj:`xarray.DataArray`, EncoderMap feature,
                or EncoderMap Featurizer.
            attr_name (Optional[str]): The name under which the CV
                should be found in the class. Is needed, if a raw numpy array
                is passed, otherwise the name will be generated from the filename
                (if ``data == str``), the DataArray.name (if ``data == xarray.DataArray``),
                or the feature name.
            cols (Optional[list]): A list specifying the columns
                to use it for the high-dimensional data. If your highD data contains
                (x,y,z,...)-errors or has an enumeration column at ``col=0``
                this can be used to remove this unwanted data.
            deg (Optional[bool]): Whether the provided data is in radians (False)
                or degree (True). It can also be None for non-angular data.
            labels (Optional[Union[list, str]]): If you want to label
                the data you provided, pass a list of str. If set to None,
                the features in this dimension will be labeled as
                ``[f"{attr_name.upper()} FEATURE {i}" for i in range(self.n_frames)]``.
                If a str is provided, the features will be labeled as
                ``[f"{attr_name.upper()} {label.upper()} {i}" for i in range(self.n_frames)]``.
                If a list of str is provided, it needs to have the same length
                as the traj has frames. Defaults to None.
            override (bool): Whether to overwrite existing CVs. The method will also
                print a message which CVs have been overwritten.

        Examples:

            Import EncoderMap and load an example Trajectory.

            >>> import encodermap as em
            >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")

            Load the central dihedrals using ``data='central_dihedrals'` as shortcut.

            >>> traj.load_CV("central_dihedrals")
            >>> traj.central_dihedrals.shape
            (14, 222)


            >>> traj._CVs['central_dihedrals']['CENTRAL_DIHEDRALS'].values[:2]
            ['CENTERDIH PSI   RESID  MET:   1 CHAIN 0'
             'CENTERDIH OMEGA RESID  MET:   1 CHAIN 0']

            Slicing the :obj:`SingleTraj` keeps all CVs in order.

            >>> import numpy as np
            >>> from pathlib import Path
            >>> traj1 = em.SingleTraj(
            ...     Path(em.__file__).parent.parent / "tests/data/1am7_corrected.xtc",
            ...     Path(em.__file__).parent.parent / "tests/data/1am7_protein.pdb",
            ... )
            >>> traj1.load_CV(traj1.xyz[..., -1], 'z_coordinate')
            ...
            >>> for i, frame in enumerate(traj1):
            ...     print(np.array_equal(frame.z_coordinate[0], frame.xyz[0, :, -1]))
            ...     if i == 3:
            ...         break
            True
            True
            True
            True

        Raises:
            FileNotFoundError: When the file given by `data` does not exist.
            IOError: When the provided filename does not have .txt, .npy or .nc extension.
            TypeError: When `data` does not match the specified input types.
            Exception: When a numpy array has been passed as `data` and no `attr_name` has been provided.
            Exception: When the provided `attr_name` is str, but cannot be a python identifier.

        """
        # Local Folder Imports
        from .trajinfo_utils import load_CVs_singletraj, trajs_combine_attrs

        if isinstance(attr_name, str):
            if "feature_indices" in attr_name:
                raise Exception(
                    f"The substring 'feature_indices' is a protected attribute. "
                    f"Your attribute can't contain this substring."
                )

        new_CVs = load_CVs_singletraj(
            data,
            self,
            attr_name,
            cols,
            deg=deg,
            periodic=periodic,
            labels=labels,
        )
        if self._CVs:
            assert (
                len(new_CVs.coords["traj_num"]) == 1
            ), f"something bad happened: {self._CVs=}"
        if len(new_CVs.coords["traj_num"]) > 1:
            raise Exception(
                f"The provided feature resulted in a dataset with "
                f"{new_CVs.sizes['traj_num']} trajectories. A `SingleTraj` "
                f"class can't accept such a feature."
            )
        if self.traj_num is not None:
            assert new_CVs.coords["traj_num"] == np.array([self.traj_num]), (
                data,
                self.traj_num,
                new_CVs.coords["traj_num"],
            )

        # check the sizes
        len_CVs = new_CVs.coords["frame_num"].shape[0]
        if self._CVs:
            n_frames = self._CVs.coords["frame_num"].shape[0]
        else:
            n_frames = self.n_frames
        if n_frames != len_CVs:
            raise Exception(
                f"Loading the requested {data} CVs is not possible, as "
                f"they are not aligned with the number of frames in the "
                f"trajectory. The CVs have {len_CVs} frames, the trajectory "
                f"{self._traj_file} has {n_frames} frames."
            )

        # check the keys and whether they get overwritten
        if hasattr(new_CVs, "keys"):
            new_keys = new_CVs.keys()
        else:
            new_keys = set([new_CVs.name])
        if override:
            if overwritten_keys := self._CVs.keys() & new_keys:
                warnings.warn(
                    f"Overwriting the following CVs with new values: "
                    f"{overwritten_keys}."
                )
            self._CVs = xr.merge(
                [new_CVs, self._CVs],
                combine_attrs=trajs_combine_attrs,
                compat="override",
                join="left",
            )
        else:
            try:
                CVs = xr.merge([self._CVs, new_CVs], combine_attrs=trajs_combine_attrs)
                assert len(CVs.coords["traj_num"]) == 1, (
                    f"Can't merge\n\n{self._CVs=}\n\nand\n\n{new_CVs=}\n\n, "
                    f"because they would stack along the traj axis."
                )
                self._CVs = CVs
            except xr.core.merge.MergeError as e:
                msg = (
                    f"Could not add the CV `{attr_name}` to the CVs of the traj "
                    f"likely due to it being already in the CVs "
                    f"({list(self.CVs.keys())}). Set `override` to True to "
                    f"overwrite these CVs. In case you are faced with "
                    f"conflicting values on 'traj_name', here they are:\n\n"
                    f"{self._CVs.coords['traj_name']=}\n\n{new_CVs.coords['traj_name']=}"
                )
                raise Exception(msg) from e

    def _common_str_to_hdf_file(self, fname: Union[Path, str]) -> None:
        """Saves this traj's common_str as an attribute to a hdf5 file."""
        common_str = self.common_str
        if common_str is None:
            common_str = ""
        with h5py.File(fname, "a") as f:
            f.attrs[f"common_str_{self.traj_num}"] = common_str

    def save_CV_as_numpy(
        self,
        attr_name: str,
        fname: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Saves a specified collective variable of this traj as a ``.npy`` file.

        This got its own method for parallelization purposes.

        Args:
            attr_name (str): Name of the CV to save.
            fname (str, optional): Can be either
            overwrite (bool, opt): Whether to overwrite the file. Defaults to False.

        Raises:
            IOError: When the file already exists and overwrite is set to False.

        """
        if fname is None:  # pragma: no cover
            fname = f"{self.basename}_{attr_name}.npy"
        if os.path.isdir(fname):
            fname = os.path.join(fname, f"{self.basename}_{attr_name}.npy")
        if os.path.isfile(fname) and not overwrite:
            raise IOError(f"{fname} already exists. Set overwrite=True to overwrite.")
        np.save(fname, self.CVs[attr_name])

    def sidechain_info(self) -> dict[int, dict[int, Sequence[int]]]:
        """Indices used for the AngleDihedralCartesianEncoderMap class to
        allow training with multiple different sidechains.

        Returns:
            dict[str, Sequence[int]]: The indices. The key '-1' is used for
            the hypothetical convex hull of all feature spaces (the output of
            the tensorflow model). The other keys match the common_str of the
            trajs.

        Raises:
            Exception: When the common_strings and topologies are not
                aligned. An exception is raised. Aligned means that all trajs
                with the same common_str should possess the same topology.

        """
        # make sure no clustal w has not been loaded
        if hasattr(self, "clustal_w"):
            raise NotImplementedError(
                f"This is currently not supported for TrajEsnembles with "
                f"clustal_w alignments."
            )
        if not "side_dihedrals" in self._CVs:
            self.load_CV("side_dihedrals")

        max_residues = self.top.n_residues
        out = {}

        for residx in range(1, max_residues + 1):
            labels = [
                l
                for l in self._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"].values
                if int(re.findall(r"\d+", l)[-2]) == residx
            ]
            first_frame = self._CVs.coords["frame_num"].min()
            length = (
                self._CVs.side_dihedrals.sel(
                    frame_num=first_frame, SIDE_DIHEDRALS=labels
                )
                .dropna(dim="SIDE_DIHEDRALS")
                .sizes["SIDE_DIHEDRALS"]
            )
            out[residx] = length
        return out

    def atom_slice(
        self,
        atom_indices: np.ndarray,
        invert: bool = False,
    ) -> None:
        """Deletes atoms from this :obj:`SingleTraj` instance.

        Args:
            atom_indices (Union[list, numpy.ndarray]): The indices of the
                atoms to keep.
            invert (bool): If False, it is assumed, that the atoms in ``atom_indices``
                are the ones to be kept. If True, the atoms in ``atom_indices``
                are the ones to be removed.

        """
        atom_indices = np.asarray(atom_indices)
        if invert:
            atom_indices = np.array(
                [a.index for a in self.top.atoms if a.index not in atom_indices]
            )
        self._atom_indices = atom_indices
        if self._CVs:
            warnings.warn(
                "Dropping CVs from trajectory. Slicing CVs with this method is "
                "currently not possible. Raise an issue if you want to have this "
                "feature added."
            )
            self._CVs = xr.Dataset()
        self._custom_top._parsed = False
        self.topology = self._custom_top.top.subset(atom_indices)
        if self.traj_file != ".":
            self._traj = self.traj_file

    def join(self, other: Union[SingleTraj, md.Trajectory]) -> md.Trajectory:
        """Join two trajectories together along the time/frame axis.

        Note:
            Returns a ``mdtraj.Trajectory`` and thus loses CVs, filenames, etc.

        """
        if isinstance(other, md.Trajectory):
            return self.traj.join(other)
        return self.traj.join(other.traj)

    def stack(self, other: SingleTraj) -> md.Trajectory:
        """Stack two trajectories along the atom axis

        Note:
            Returns a m``dtraj.Trajectory`` and thus loses CVs, filenames, etc.

        """
        if isinstance(other, md.core.trajectory.Trajectory):
            return self.traj.stack(other)
        return self.traj.stack(other.traj)

    def superpose(
        self,
        reference: Union[md.Trajectory, SingleTraj],
        frame: int = 0,
        atom_indices: Optional[np.ndarray] = None,
        ref_atom_indices: Optional[np.ndarray] = None,
        parallel: bool = True,
        inherit_CVs: bool = False,
    ) -> SingleTraj:
        """Superpose each conformation in this trajectory upon a reference

        Args:
            reference (Union[mdtraj.Trajectory, SingleTraj]): The reference
                frame to align to. If the reference has multiple frames and
                you want to use a specific frame as reference, use the
                ``frame`` argument also.
            frame (int, optional): Align to this frame in reference.
                Default is 0.
            atom_indices (Union[np.array, None], optional): Indices in
                self, used to calculate RMS values. Defaults to None which
                means all atoms will be used.
            ref_atom_indices (Union[np.array, None], optional): Indices
                in reference, used to calculate RMS values. Defaults to None
                which means all atoms will be used.
            parallel (bool, optional): Use OpenMP to run the superposition
                in parallel over multiple cores.
            inherit_CVs (bool, optional): Whether to also inherit the CVs.
                This feature is currently not implemented. It would require
                additional code in all Feature classes discerning intrinsic
                (distance, angle, cluster_membership, etc.) or an extrinsic
                feature (absolute coordinate, COG position, etc.). Then this
                extrinsic/intrinsic boolean flag also needs to accompany the
                xarray Datasets, so that the intrinsic CVs can be inherited,
                and the extrinsic can be dropped with a corresponding message.

        Returns:
            SingleTraj: A new trajectory with atoms aligned.

        """
        if isinstance(reference, md.core.trajectory.Trajectory):
            new = SingleTraj(
                self.traj.superpose(
                    reference, frame, atom_indices, ref_atom_indices, parallel
                )
            )
        new = SingleTraj(
            self.traj.superpose(
                reference.traj, frame, atom_indices, ref_atom_indices, parallel
            )
        )
        if self.traj_file:
            new._traj_file = self._traj_file
        if self.top_file:
            new._top_file = self._top_file
        if self._CVs and inherit_CVs:
            raise NotImplementedError("CV inheritance not implemented yet.")
        return new

    def save(
        self,
        fname: str,
        CVs: Union[Literal["all"], list[str]] = "all",
        overwrite: bool = False,
    ) -> None:
        """Save the trajectory as HDF5 file format to disk.

        Args:
            fname (str): The filename.
            CVs (Union[List, 'all'], optional): Either provide a list of strings
                of the CVs you would like to save to disk, or set to 'all' to save
                all CVs. Defaults to [].
            overwrite (bool, optional): Whether force overwrite an existing file.
                Defaults to False.

        Raises:
            IOError: When the file already exists and overwrite is False.

        """
        # check and drop inhomogeneous attributes
        offending_keys = []
        if self._CVs:
            for da in self._CVs.data_vars.values():
                for key, val in da.attrs.items():
                    if isinstance(val, list):
                        offending_keys.append(key)
        for key in offending_keys:
            for da in self._CVs.data_vars.values():
                if key in da.attrs:
                    del da.attrs[key]
            if key in self._CVs.attrs:
                del self._CVs.attrs[key]
        # raise exception if file already exists
        if os.path.isfile(fname) and not overwrite:
            raise IOError(f"{fname} already exists. Set overwrite=True to overwrite.")
        else:
            self.traj.save_hdf5(fname, force_overwrite=overwrite)

        # save the topology as a nested dict
        if self._custom_top:
            self._custom_top.traj = self
            self._custom_top.to_hdf_file(fname=fname)

        if self.common_str:
            with h5.File(fname, "a") as file:
                if self.traj_num is None:
                    file.attrs["common_str"] = self.common_str
                else:
                    file.attrs[f"common_str_{self.traj_num}"] = self.common_str

        # save the CVs with xarray
        if self._CVs and CVs == "all":
            save_netcdf_alongside_mdtraj(fname, self._CVs)
            return
        if self._CVs and isinstance(CVs, list):
            with h5.File(fname, "a") as file:
                if "CVs" in list(file.keys()):
                    grp = file["CVs"]
                else:
                    grp = file.create_group("CVs")
                for key in CVs:
                    value = self._CVs[key]
                    assert self.n_frames == value.shape[1]
                    grp.create_dataset(name=key, data=value)

    @overload
    def iterframes(
        self,
        with_traj_num: bool = False,
    ) -> Iterable[tuple[int, SingleTraj]]: ...

    @overload
    def iterframes(
        self,
        with_traj_num: bool = True,
    ) -> Iterable[tuple[int, int, SingleTraj]]: ...

    def iterframes(
        self,
        with_traj_num: bool = False,
    ) -> Iterable[tuple[Any, ...]]:
        """Iterator over the frames in this class.

        Args:
            with_traj_num (bool): Whether to return a three-tuple of traj_num,
                frame_num, frame (True) or just traj_num, frame (False).

        Yields:
            tuple: A tuple containing the following:
                - int: The traj_num.
                - int: The frame_num.
                - encodermap.SingleTraj: An SingleTraj object.

        Examples:

            Import EncoderMap and create :obj:`SingleTraj` instance.

            >>> import encodermap as em
            >>> traj = em.SingleTraj('https://files.rcsb.org/view/1YUG.pdb')
            >>> traj.n_frames
            15

            Slicing the trajectory every 5th frame

            >>> traj = traj[::5]
            >>> traj.n_frames
            3

            Using the ``iterframes()`` iterator.

            >>> for frame_num, frame in traj.iterframes():
            ...     print(frame_num, frame.n_frames)
            0 1
            5 1
            10 1

        """
        if self.id.ndim == 2:
            a = self.id[:, 1]
        else:
            a = self.id
        for i, frame in zip(a, self):
            if with_traj_num:
                yield self.traj_num, i, frame
            else:
                yield i, frame

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        # Standard Library Imports
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __hash__(self) -> int:
        hash_value = hash(self.top)
        # combine with hashes of arrays
        hash_value ^= _hash_numpy_array(self._xyz)
        hash_value ^= _hash_numpy_array(self.time)
        if self._unitcell_lengths is not None:
            hash_value ^= _hash_numpy_array(self._unitcell_lengths)
        if self._unitcell_angles is not None:
            hash_value ^= _hash_numpy_array(self._unitcell_angles)
        return hash_value

    def __eq__(self, other: SingleTraj) -> bool:
        """Two SingleTraj objects are the same, when the trajectories are the same,
        the files are the same, and the loaded CVs are the same."""
        if self._traj_file != other._traj_file:
            return False
        if self.__hash__() != other.__hash__():
            return False
        if not self._CVs.equals(other._CVs):
            return False
        return True

    def __reversed__(self) -> SingleTraj:
        """Reverses the frame order of the traj. Same as traj[::-1]"""
        return self[::-1]

    def __enter__(self):
        """Enters context manager. Inside the context manager, the traj stays loaded."""
        self.load_traj()
        return self

    def __exit__(self, type, value, traceback):
        """Exits the context manager and deletes unwanted variables."""
        self.unload()

    def __iter__(self):
        """Iterate over frames in this class. Returns the correct
        CVs along with the frame of the trajectory."""
        self._index = 0
        if len(self) == 0 and self.index is None:
            self.load_traj()
        return self

    def __next__(self):
        if len(self.id) == 1:
            return self
        if self._index >= self.n_frames:
            raise StopIteration
        else:
            self._index += 1
            return self[self._index - 1]

    def __getitem__(self, key: CanBeIndex) -> SingleTraj:
        """This method returns another trajectory as a SingleTraj class.

        Args:
            key (Union[int, list[int], numpy.ndarray, slice]): Indexing the trajectory
                can be done by int (returns a traj with 1 frame), lists of int or
                numpy.ndarray (returns a new traj with len(traj) == len(key)), or
                slice ([::3]), which returns a new traj with the correct number of
                frames.

        Returns:
            Info_Single: A SingleTraj object with this frame in it.

        """
        if not isinstance(key, (int, np.int_, list, np.ndarray, slice)):
            raise TypeError(
                f"Indexing of `SingleTraj` requires the index to "
                f"be one of the following types: (int, "
                f"list, np.ndarray, slice), you provided {type(key)}."
            )

        if any([isinstance(i, (int, np.integer)) for i in self.index]) and key != 0:
            raise IndexError(
                f"SingleTraj index {key} out of range for traj with only 1 frame."
            )

        if isinstance(key, (int, np.integer)):
            if key > self.n_frames:
                raise IndexError(
                    f"Index {key} out of range for traj with "
                    f"{self.n_frames} frames."
                )
        if isinstance(key, (list, np.ndarray)):
            if any([k > self.n_frames for k in key]):
                raise IndexError(
                    f"At least one index in {key} out of range for "
                    f"traj with {self.n_frames} frames. Normally frames are "
                    f"selected by current integer index. If you are trying to "
                    f"access frames by their number as it is in the file {self.traj_file}, "
                    f"you can use the `fsel[]` locator of this class:\n\n"
                    f"traj = em.load('traj_file.xtc', 'top_file.xtc')\n"
                    f"traj.fsel[{key}]."
                )

        # append the index to the list of "transformations"
        new_index = (*self.index, key)

        # build a new traj from that
        if self.backend == "no_load":
            traj_out = SingleTraj(
                self.traj_file,
                self.top_file,
                backend=self.backend,
                common_str=self.common_str,
                index=new_index,
                traj_num=self.traj_num,
                basename_fn=self.basename_fn,
            )
        else:
            traj_out = SingleTraj(
                self.trajectory[key],
                self.top_file,
                backend=self.backend,
                common_str=self.common_str,
                index=new_index,
                traj_num=self.traj_num,
                basename_fn=self.basename_fn,
            )
            traj_out._traj_file = self._traj_file
            traj_out._top_file = self._top_file
        assert traj_out._traj_file == self._traj_file

        # the original_frames
        if self._atom_indices is not None:
            traj_out.atom_slice(self._atom_indices)
        assert self.n_atoms == traj_out.n_atoms
        assert self.n_residues == traj_out.n_residues
        traj_out._orig_frames = self._orig_frames
        traj_out._loaded_once = self._loaded_once

        # last the CVs
        if self._CVs:
            traj_out._CVs = self._CVs.isel(frame_num=key)
            if "frame_num" not in traj_out._CVs.dims:
                traj_out._CVs = traj_out._CVs.expand_dims(
                    {
                        "frame_num": [key],
                    },
                )
                traj_out._CVs.assign_coords(time=("frame_num", traj_out.time))

        return traj_out

    @property
    def fsel(self):
        return SingleTrajFsel(self)

    def __add__(self, y: SingleTraj) -> TrajEnsemble:
        """Addition of two SingleTraj classes yields TrajEnsemble class.

        Args:
            y (encodermap.SingleTraj): The other traj, that will be added.

        Returns:
            encodermap.TrajEnsemble: The new trajs.

        """
        return self._add_along_traj(y)

    def __getattr__(self, attr):
        """What to do when attributes cannot be obtained in a normal way?.

        This method allows access to the `self.CVs` dictionary's values as
        instance variables. Furthermore, if a mdtraj variable is called,
        the traj is loaded, and the correct variable is returned.

        """
        if attr in self._mdtraj_attr:
            self.load_traj()
            return getattr(self.traj, attr)
        elif attr in self._CVs:
            val = self._CVs[attr]  # [index]
            axis_name = (
                "feature_axis"
                if "feature_axis" in val.attrs
                else "feature_axes" if "feature_axes" in val.attrs else None
            )
            if np.any(np.isnan(val)):
                if axis_name is not None:
                    if "indices" in val.name:
                        assert val.dropna("ATOM_NO").size > 0, (
                            f"Can't `dropna` from DataArray {val=}. This would "
                            f"result in an empty array. Here are the nan indices: "
                            f"{np.where(np.isnan(val.values))[0]=}"
                        )
                        val = val.dropna("ATOM_NO")
                    else:
                        val = val.dropna(val.attrs[axis_name])
                else:
                    val = val.dropna(attr.upper())
            val_squeezed = val.values.squeeze(0)
            assert val_squeezed.size > 0, (
                f"The value of {attr} is unexpectedly empty {val.size=} after dropping nans "
                f"and dropping nans from he feature axis '{val.attrs[axis_name]}' and "
                f"squeezing out the trajectory axis. This is the original "
                f"DataArray:\n{self._CVs[attr]}\n\nAfter the dropna, the value was:\n{val}"
            )
            return val_squeezed
        elif attr == "traj":  # pragma: no cover
            self.__getattribute__(attr)
        elif attr == "id":  # pragma: no cover
            self.__getattribute__(attr)
        elif attr == "top":
            return self._get_top()
        else:
            raise AttributeError(f"'SingleTraj' object has no attribute '{attr}'")

    def _string_summary(self) -> str:  # pragma: no cover
        """Returns a summary about the current instance.

        Number of frames, index, loaded CVs.

        """
        s = f"encodermap.SingleTraj object."
        if self.backend == "no_load":
            s += " Currently not in memory."
        else:
            s += " Data currently in memory."
        if self.basename:
            s += f" Basename is '{self.basename}'."
        if self.index != (None,):
            with np.printoptions(threshold=1, edgeitems=1):
                s += f" At indices {self.index[1:]}."
        if self._CVs:
            for key, value in self._CVs.items():
                if "feature_indices" in key:
                    continue
                shape = value.shape
                if not shape:
                    shape = 1
                s += f" CV {key} with shape {shape} loaded."
        else:
            s += " Not containing any CVs."
        if "n_atoms" in self.__dict__.keys():
            s += f" Containing {self.n_atoms} atoms."
        if "n_frames" in self.__dict__.keys():
            s += f" Containing {self.n_frames} frames."
        if self.common_str:
            s += f" Common string is '{self.common_str}'."
        if self._custom_top:
            s += (
                f" Custom topology with residues: "
                f"{[r.name for r in self._custom_top.residues]}."
            )
        return s

    def __len__(self):
        return self.n_frames

    def __str__(self):
        return self._string_summary()

    def __repr__(self):
        return f"<{self._string_summary()} Object at 0x{id(self):02x}>"

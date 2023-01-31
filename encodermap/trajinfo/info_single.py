# -*- coding: utf-8 -*-
# encodermap/trajinfo/info_single.py
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
    * Better anomaly detection of unique structural states.


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

import errno
import os
import sys
import warnings
from io import StringIO
from pathlib import Path

import numpy as np

from .._optional_imports import _optional_import
from ..misc.errors import BadError
from ..misc.misc import _TOPOLOGY_EXTS
from ..misc.xarray import construct_xarray_from_numpy
from ..misc.xarray_save_wrong_hdf5 import save_netcdf_alongside_mdtraj
from .info_all import TrajEnsemble
from .load_traj import _load_traj

################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")
mda = _optional_import("MDAnalysis")
h5 = _optional_import("h5py")
xr = _optional_import("xarray")


################################################################################
# Typing
################################################################################


from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

if TYPE_CHECKING:
    import h5py as h5
    import MDAnalysis as mda
    import mdtraj as md
    import xarray as xr

    from .trajinfo_utils import SingleTrajFeatureType


################################################################################
# Globals
################################################################################


__all__ = ["SingleTraj"]


################################################################################
# Utilities
################################################################################


def combine_attrs(attrs, context):
    out = {"length_units": "nm", "time_units": "ps", "angle_units": "rad"}
    for attr in attrs:
        for key, val in attr.items():
            if key in ["full_path", "topology_file", "feature_axis"]:
                if key in out:
                    if isinstance(out[key], list):
                        out[key].append(val)
                    else:
                        out[key] = [out[key], val]
                else:
                    out[key] = val
            else:
                out[key] = val
    return out


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


##############################################################################
# Classes
##############################################################################


class SingleTraj:
    """This class contains the info about a single trajectory.

    This class contains many of the attributes and methods of mdtraj's Trajectory.
    It is meant to be used as a single trajectory in a ensemble defined in the
    TrajEnsemble class. Other than the standard mdtraj Trajectory this class loads the
    MD data only when needed. The location of the file and other attributes like
    a single integer index (single frame of trajectory) or a list of integers
    (multiple frames of the same traj) are stored until the traj is accessed via the
    `SingleTraj.traj` attribute. The returned traj is a mdtraj Trajectory
    with the correct number of frames in the correct sequence.

    Furthermore this class keeps track of your collective variables. Oftentimes
    the raw xyz data of a trajectory is not needed and suitable CVs are selected
    to represent a protein via internal coordinates (torsions, pairwise distances, etc.).
    This class keeps tack of your CVs. Whether you call them `highd` or
    `torsions`, this class keeps track of everything and returns the values when
    you need them.

    SingleTraj supports fancy indexing, so you can extract one or more frames
    from a Trajectory as a separate trajectory. For example, to form a
    trajectory with every other frame, you can slice with `traj[::2]`.

    SingleTraj uses the nanometer, degree & picosecond unit system.

    Attributes:
        backend (str): Current state of loading. If backend == 'no_load' xyz data
            will be loaded from disk, if accessed. If backend == 'mdtraj', the
            data is already in RAM.
        common_str (str): Substring of traj_file. Used to group multiple
            trajectories together based on common topology files. If traj files
            protein1_traj1.xtc and protein1_traj2.xtc share the sameprotein1.pdb
            common_str can be set to group them together.
        index (Union[int, list, np.array, slice]): Fancy slices of the
            trajectory. When file is loaded from disk, the fancy indexes will
            be applied.
        traj_num (int): Integer to identify a SingleTraj class in a TrajEnsemble class.
        traj_file (str): Trajectory file used to create this class.
        top_file (str): Topology file used to create this class. If a .h5 trajectory
            was used traj_file and top_file are identical. If a mdtraj.Trajectory was
            used to create SingleTraj, these strings are empty.

    Examples:
        >>> # load a pdb file with 14 frames from rcsb.org
        >>> import encodermap as em
        >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
        >>> print(traj)
        encodermap.SingleTraj object. Current backend is no_load. Basename is 1GHC. Not containing any CVs.
        >>> traj.n_frames
        14

        >>> # advanced slicing
        >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")[-1:7:-2]
        >>> print([frame.id for frame in traj])
        [13, 11, 9]

        >>> # Build a trajectory ensemble from multiple trajs
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
        top: Optional[str, Path] = None,
        common_str: str = "",
        backend: Literal["no_load", "mdtraj"] = "no_load",
        index: Optional[Union[int, list[int], np.ndarray, slice]] = None,
        traj_num: Optional[int] = None,
        basename_fn: Optional[Callable] = None,
    ) -> None:
        """Initilaize the SingleTraj object with location and reference pdb file.

        Args:
            traj (Union[str, mdtraj.Trajectory]): The trajectory. Can either be teh filename
                of a trajectory file (.xtc, .dcd, .h5, .trr) or a mdtraj.Trajectory.
            top (Union[str, mdtraj.Topology], optional): The path to the reference pdb file.
                Defaults to ''. If an mdtraj.Trajectory or a .h5 traj filename is provided
                this option is not needed.
            common_str (str, optional): A string to group traj of similar
                topology. If multiple trajs are loaded (TrajEnsemble) this common_str is
                used to group them together. Defaults to '' and won't be matched to other trajs.
                If traj files protein1_traj1.xtc and protein1_traj2.xtc share the sameprotein1.pdb
                and protein2_traj.xtc uses protein2.pdb as its topology this argument
                can be ['protein1', 'protein2'].
            backend (Literal['no_load', 'mdtraj'], optional): Chooses the backend to load trajectories.
                * 'mdtraj' uses mdtraj which loads all trajecoties into RAM.
                * 'no_load' creates an empty trajectory object.
                Defaults to 'no_load'
            index (): An integer or an array giving the indices.
                If an integer is provided only the frame at this position will be loaded once the internal
                mdtraj.Trajectory is accessed. If an array or list is provided the corresponding frames will be used.
                These indices can have duplicates: [0, 1, 1, 2, 0, 1]. A slice object can also be provided.
                Supports fancy slicing like traj[1:50:3]. If None is provided the trajectory is simply loaded as is.
                Defaults to None
            traj_num (Union[int, None], optional): If working with multiple trajs this is the easiest unique identifier.
                If multiple SingleTrajs are instantiated by TrajEnsemble the traj_num is used as unique identifier per traj.
                Defaults to None.
            basename_fn (Optional[Callable]): A function to apply to `traj_file` to give it
                a unique identifier. If all your trajs are called traj.xtc and only the directory they're in
                gives them a unique identifier you can provide a function into this argument to split the path.
                If None is provided the basename is extracted liek so: `lambda x: x.split('/')[0].split('.')[-1].
                Defaults to None.

        """
        # defaults
        self.backend = backend
        self.common_str = common_str
        self.index = index if isinstance(index, tuple) else (index,)
        self.traj_num = traj_num
        self._loaded_once = False if backend == "no_load" else True
        self._orig_frames = np.array([])
        self._CVs = xr.Dataset()
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
                self.topology = top
                self._top_file = Path("")
            else:
                self._top_file = Path(top)
                if isinstance(self._traj_file, Path):
                    if (
                        self._traj_file.suffix in _TOPOLOGY_EXTS
                        and self._traj_file != self._top_file
                    ):
                        raise BadError(
                            f"You probably mixed up the input. Normally you "
                            f"want to instantiate with `SingleTraj(traj, top)`. Based on "
                            f"the files and the  extensions you provided "
                            f"(traj={self._traj_file.name} and top="
                            f"{self._top_file.name}), you want to change the "
                            f"order of the arguments, or use keyword arguments."
                        )
        else:
            if isinstance(self._traj_file, Path):
                if self._traj_file.suffix in _TOPOLOGY_EXTS:
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
                traj = md.load(str(self._traj_file), top=str(self._top_file))
            self.trajectory = traj
            self.topology = self.trajectory.top
            self._loaded_once = True
            self._orig_frames = np.arange(traj.n_frames)

    @classmethod
    def from_pdb_id(cls, pdb_id: str) -> SingleTraj:
        """Alternate constructor for the TrajEnsemble class.

        Builds an SingleTraj class from a pdb-id.

        Args:
            pdb_id (str): The 4-letter pdb id.

        Returns:
            SingleTraj: An SingleTraj class.

        """
        url = f"https://files.rcsb.org/view/{pdb_id.upper()}.pdb"
        return cls(url)

    @property
    def _original_frame_indices(self):
        if self._loaded_once:
            return self._orig_frames
        else:
            self.load_traj()
            return self._orig_frames

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
        self.trajectory, _ = _load_traj(
            *self.index, traj_file=traj_file, top_file=self._top_file
        )
        if not self._loaded_once:
            self._loaded_once = True
            self._orig_frames = _

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
        """np.ndarray: id is an array of unique identifiers which identify the frames in
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
                    n_frames = np.arange(file["coordinates"].shape[0])
                else:
                    for i, ind in enumerate(self.index):
                        if i == 0:
                            n_frames = np.arange(file["coordinates"].shape[0])
                            if ind is not None:
                                n_frames = n_frames[ind]
                        else:
                            if ind is not None:
                                n_frames = n_frames[ind]
                if not self._loaded_once:
                    self._orig_frames = np.arange(file["coordinates"].shape[0])

            # return single int or length of array
            if isinstance(n_frames, (int, np.integer)):
                return n_frames
            else:
                return len(n_frames)
        elif self._traj_file.suffix == ".xtc":
            reader = mda.coordinates.XTC.XTCReader(self.traj_file)
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
        with h5.File(self.traj_file, "r") as file:
            return file["coordinates"].shape[0]

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
        if self.extension == ".h5":
            with h5.File(self.traj_file, "r") as file:
                return file["coordinates"].shape[1]
        else:
            self.load_traj()
            return self.traj.n_atoms

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
        if self.top_file:
            if self.top_file != ".":
                if self._validate_uri(self.top_file):
                    return md.load_pdb(self.top_file).top
                if not os.path.isfile(self.top_file):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), self.top_file
                    )
        if self.backend == "no_load" and not self.extension == ".h5" and self.traj_file:
            return md.load_topology(self.top_file)
        if self.extension == ".h5":
            return md.load_topology(self.top_file)
        if self.backend == "no_load":
            return md.load_topology(self.top_file)
        if self.backend == "mdtraj":
            return self.traj.top
        else:
            return self.topology

    @property
    def CVs(self) -> dict[str, np.ndarray]:
        """dict: Returns a simple dict from the more complicated self._CVs xarray Dataset.

        If self._CVs is empty and self.traj_file is a HDF5 (.h5) file, the contents
        of the HDF5 will be checked, whether CVs have been stored there.
        If not and empty dict will be returned.

        """
        if self._CVs:
            return {
                key: val.dropna(val.attrs["feature_axis"]).values.squeeze(0)
                if np.any(np.isnan(val))
                else val.values.squeeze(0)
                for key, val in self._CVs.data_vars.items()
            }
        elif self.extension == ".h5":
            with h5.File(self.traj_file, "r") as file:
                if "CVs" in file.keys():
                    try:
                        self._CVs = xr.open_dataset(
                            self.traj_file,
                            group="CVs",
                            engine="h5netcdf",
                            backend_kwargs={"phony_dims": "access"},
                        )
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
                        raise BadError(
                            f"The formatting of the data in the file "
                            f"{self.traj_file} is off. Xarray could "
                            f"not load the group 'CVs' and failed with {e}"
                        )

                    # get the original frame indices from the dataset
                    # this is the only case
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

                    # return again. Yay recursion
                    return self.CVs
                else:
                    return {}
        else:
            return {}

    def _validate_uri(self, uri: str) -> bool:
        """Checks whether `uri` is a valid uri."""
        from encodermap.misc.misc import _validate_uri

        return _validate_uri(str(uri))

    def load_traj(
        self,
        new_backend: Literal["no_load", "mdtraj"] = "mdtraj",
    ) -> None:
        """Loads the trajectory, with a new specified backend.

        After this is called the instance variable self.trajectory
        will contain an mdtraj Trajectory object.

        Args:
            new_backend (str, optional): Can either be:
                * `mdtraj` to load the trajectory using mdtraj.
                * `no_load` to not load the traj (unload).
                Defaults to `mdtraj`.

        """
        if self.backend == new_backend:
            return
        if self.backend == "mdtraj" and new_backend == "no_load":
            self.unload()
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
        """Execute a selection against the topology

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
        """Clears up RAM by deleting the trajectory Info and the CV data.

        If CVs is set to True the loaded CVs will also be deleted.

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
        """Creates an TrajEnsemble class with this traj in it.

        This method is needed to add two SingleTraj objects
        along the `trajectory` axis with the method add_new_traj.
        This method is also called by the __getitem__ method of the TrajEnsemble class.

        """
        if self.traj_file != ".":
            self.info_all = TrajEnsemble(
                [self._traj_file],
                [self._top_file],
                backend=self.backend,
                common_str=[self.common_str],
            )
        else:
            self.info_all = TrajEnsemble(
                [self.traj],
                [self.top],
                backend=self.backend,
                common_str=[self.common_str],
            )
        self.info_all.trajs[0]._CVs = self._CVs
        self.info_all.trajs[0].traj_num = self.traj_num
        return self.info_all

    def _add_along_traj(self, y: SingleTraj) -> TrajEnsemble:
        """Puts self and y into a TrajEnsemble object.

        This way the trajectories are not appended along the timed
        axis but rather along the `trajectory` axis.

        Args:
            y (SingleTraj): The other ep.SingleTraj trajectory.

        """
        class_1 = self._gen_ensemble()
        class_2 = y._gen_ensemble()
        new_class = class_1 + class_2
        return new_class

    def get_single_frame(self, key: int) -> SingleTraj:
        """Returns a single frame from the trajectory.

        Args:
            key (Union[int, np.int]): Index of the frame.

        Examples:
            >>> # Load traj from pdb
            >>> import encodermap as em
            >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
            >>> traj.n_frames
            14

            >>> # Load the same traj and give it a number for recognition in a set of multiple trajs
            >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb", traj_num=5)
            >>> frame = traj.get_single_frame(2)
            >>> frame.id
            array([[5, 2]])

        """
        return self.__getitem__(key)

    def show_traj(self, gui: bool = True) -> nglview.view:
        """Returns an nglview view object.

        Returns:
            view (nglview.widget): The nglview widget object.

        """
        import nglview

        view = nglview.show_mdtraj(self.traj, gui=gui)
        return view

    def load_CV(
        self,
        data: SingleTrajFeatureType,
        attr_name: Optional[str] = None,
        cols: Optional[list[int]] = None,
        labels: Optional[list[str]] = None,
        override: bool = False,
    ) -> None:
        """Load CVs into traj. Many options are possible. Provide xarray,
        numpy array, em.loading.feature, em.featurizer, and even string!

        This method loads CVs into the SingleTraj class. Many ways of doing so are available:
            * np.ndarray: The easiest way. Provide a np array and a name for the array and the data
                will be saved as a instance variable, accesible via instance.name.
            * xarray.DataArray: You can load a multidimensional xarray as data into the class. Please
                refer to xarrays own documentation if you want to create one yourself.
            * xarray.Dataset: You can add another dataset to the existing _CVs.
            * em.loading.feature: If you provide one of the features from em.loading.features the resulting
                features will be loaded and also placed under the provided name.
            * em.Featurizer: If you provide a full featurizer, the data will be generated and put as an
                instance variable as the provided name.
            * str: If a string is provided, the data will be loaded from a .txt, .npy, or NetCDF / HDF5 .nc file.

        Args:
            data (Union[str, np.ndarray, xr.DataArray, em.loading.feature, em.Featurizer]):
                The CV to load. Either as numpy array, xarray DataArray, encodermap or pyemma feature, or full
                encodermap Featurzier.
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
            override (bool): Whether to overwrite existing CVs. The method will also
                print a message which CVs have been overwritten.

        Examples:
            >>> # Load the backbone torsions from a time-resolved NMR ensemble from the pdb
            >>> import encodermap as em
            >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
            >>> central_dihedrals = em.loading.features.CentralDihedrals(traj.top)
            >>> traj.load_CV(central_dihedrals)
            >>> traj.central_dihedrals.shape
            (1, 14, 222)
            >>> # The values are stored in an xarray Dataset to track every possible datafield
            >>> traj = em.SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
            >>> traj.load_CV(em.loading.features.CentralDihedrals(traj.top))
            >>> print(traj._CVs['central_dihedrals']['CENTRALDIHEDRALS'].values[:2])
            ['CENTERDIH PSI   RESID  MET:   1 CHAIN 0'
             'CENTERDIH OMEGA RESID  MET:   1 CHAIN 0']

        Raises:
            FileNotFoundError: When the file given by `data` does not exist.
            IOError: When the provided filename does not have .txt, .npy or .nc extension.
            TypeError: When `data` does not match the specified input types.
            Exception: When a numpy array has been passed as `data` and no `attr_name` has been provided.
            BadError: When the provided `attr_name` is str, but can not be a python identifier.

        """
        from .trajinfo_utils import load_CVs_singletraj

        new_CVs = load_CVs_singletraj(data, self, attr_name, cols, labels)
        if self.traj_num is not None:
            assert new_CVs.coords["traj_num"] == np.array([self.traj_num]), print(
                data, self.traj_num, new_CVs.coords["traj_num"]
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
                print(
                    f"Overwriting the following CVs with new values: "
                    f"{overwritten_keys}."
                )
            self._CVs = xr.merge(
                [new_CVs, self._CVs],
                combine_attrs="override",
                compat="override",
                join="left",
            )
        else:
            try:
                self._CVs = xr.merge([self._CVs, new_CVs], combine_attrs=combine_attrs)
            except xr.core.merge.MergeError as e:
                msg = (
                    f"Could not add the CV `{attr_name}` to the CVs of the traj "
                    f"likely due to it being already in the CVs "
                    f"({list(self.CVs.keys())}). Set `override` to True to "
                    f"overwrite these CVs."
                )
                raise Exception(msg) from e

    def save_CV_as_numpy(
        self,
        attr_name: str,
        fname: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Saves the highD data of this traj.

        This got its own method for parallelization purposes.

        Args:
            attr_name (str): Name of the CV to save.
            fname (str, optional): Can be either
            overwrite (bool, opt): Whether to overwrite the file. Defaults to False.

        Raises:
            IOError: When the file already exists and overwrite is set to False.

        """
        if fname is None:
            fname = f"{self.basename}_{attr_name}.npy"
        if os.path.isdir(fname):
            fname = os.path.join(fname, f"{self.basename}_{attr_name}.npy")
        if os.path.isfile(fname) and not overwrite:
            raise IOError(f"{fname} already exists. Set overwrite=True to overwrite.")
        np.save(fname, self.CVs[attr_name])

    def atom_slice(
        self,
        atom_indices: np.ndarray,
        inplace: bool = False,
    ) -> Union[None, SingleTraj]:
        """Create a new trajectory from a subset of atoms.

        Args:
            atom_indices (Union[list, np.array]): The indices of the
            atoms to keep.
            inplace(bool, optional): Whether to overwrite the current instance,
            or return a new instance. Defaults to False.

        """
        if self._CVs:
            warnings.warn(
                "Dropping CVs from trajectory. Defining CVs for atom slice is currently not possible."
            )
        new = SingleTraj(
            self.traj.atom_slice(atom_indices, inplace=inplace),
            common_str=self.common_str,
            backend="mdtraj",
            index=self.index,
            traj_num=self.traj_num,
        )
        if inplace:
            self = new
        else:
            return new

    def join(self, other: SingleTraj) -> md.Trajectory:
        """Join two trajectories together along the time/frame axis.

        Returns a mdtraj.Trajectory and thus loses CVs, filenames, etc.

        """
        if isinstance(other, md.core.trajectory.Trajectory):
            return self.traj.join(other)
        return self.traj.join(other.traj)

    def stack(self, other: SingleTraj) -> md.Trajectory:
        """Stack two trajectories along the atom axis

        Returns a mdtraj.Trajectory and thus loses CVs, filenames, etc.

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
    ) -> SingleTraj:
        """Superpose each conformation in this trajectory upon a reference

        Args:
            reference (Union[mdtraj.Trajectory, SingleTraj]): The reference frame to align to.
            reame (int, optional): Align to this frame in reference. Defaults to 1.
            atom_indices (Union[np.array, None], optional): Indices in self, used to calculate
                RMS values. Defaults to None, whcih means all atoms will be used.
            ref_atom_indices (Union[np.array, None], optional): Indices in reference, used to calculate
                RMS values. Defaults to None, whcih means all atoms will be used.
            parallel (bool, optional): Use OpenMP to run the superposition in parallel over multiple cores.

        Returns:
            SingleTraj: A new aligned trajectory.

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
            new.traj_file = self.traj_file
        if self.top_file:
            new.top_file = self.top_file
        if self._CVs:
            raise NotImplementedError("CV inheritance not implemented yet.")
        return new

    def save(
        self,
        fname: str,
        CVs: Union[str, list[str]] = "all",
        overwrite: bool = False,
    ) -> None:
        """Save the trajectory as HDF5 fileformat to disk,

        Args:
            fname (str): The filename.
            CVs (Union[List, 'all'], optional): Either provide a list of strings
                of the CVs you would like to save to disk, or set to 'all' to save
                all CVs. Defaults to [].
            overwrite (bool, optional): Whether to force overwrite an existing file.
                Defaults to False.

        Raises:
            IOError: When the file already exists and overwrite is False.

        """
        if os.path.isfile(fname) and not overwrite:
            raise IOError(f"{fname} already exists. Set overwrite=True to overwrite.")
        else:
            self.traj.save_hdf5(fname, force_overwrite=overwrite)
        if CVs == "all":
            save_netcdf_alongside_mdtraj(fname, self._CVs)
            return
        if self._CVs and CVs:
            with h5.File(fname, "a") as file:
                if "CVs" in list(file.keys()):
                    grp = file["CVs"]
                else:
                    grp = file.create_group("CVs")
                for key in CVs:
                    value = self._CVs[key]
                    assert self.n_frames == value.shape[1]
                    grp.create_dataset(name=key, data=value)

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
        """Two SingleTraj objetcs are the same, when the trajectories are the same,
        the files are the same and the loaded CVs are the same."""
        trajs = self.__hash__() == other.__hash__()
        data = self._CVs.equals(other._CVs)
        files = self._traj_file == other._traj_file
        return all([trajs, data, files])

    def __reversed__(self) -> SingleTraj:
        """Reverses the frame order of the traj. Same as traj[::-1]"""
        return self[::-1]

    def __enter__(self):
        """Enters context manager. Inside context manager, the traj stays loaded."""
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

    def __getitem__(self, key):
        """This method returns another trajectory as an SingleTraj class.

        Args:
            key (Union[int, list[int], np.ndarray, slice]): Indexing the trajectory
                can be done by int (returns a traj with 1 frame), lists of int or
                np.ndarray (returns a new traj with len(traj) == len(key)), or
                slice ([::3]), which returns a new traj with the correct number of
                frames.

        Returns:
            Info_Single: An SingleTraj object with this frame in it.

        """
        if not isinstance(key, (int, np.int_, list, np.ndarray, slice)):
            raise TypeError(
                f"Indexing of `SingleTraj` requires the index to "
                f"be one of the following types: (int, np.int, "
                f"list, np.ndarray, slice), you provided {type(key)}."
            )

        if any([isinstance(i, (int, np.integer)) for i in self.index]) and key != 0:
            raise IndexError("SingleTraj index out of range for traj with 1 frame.")

        if isinstance(key, (int, np.integer)):
            if key > self.n_frames:
                raise IndexError(
                    f"Index {key} out of range for traj with "
                    f"{self.n_frames} frames."
                )
        if isinstance(key, (list, np.ndarray)):
            if any([k > self.n_frames for k in key]):
                raise IndexError(
                    f"At least one index in {key} out of range for"
                    f"traj with {self.n_frames} frames."
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
                self.topology,
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
        traj_out._orig_frames = self._orig_frames
        traj_out._loaded_once = self._loaded_once

        # last the CVs
        if self._CVs:
            traj_out._CVs = self._CVs.loc[{"frame_num": key}]

        return traj_out

    def __add__(self, y: SingleTraj) -> TrajEnsemble:
        """Addition of two SingleTraj classes yields TrajEnsemble class. A `trajectory ensemble`.

        Args:
            y (encodermap.SingleTraj): The other traj, that will be added.

        Returns:
            encodermap.TrajEnsemble: The new trajs.

        """
        return self._add_along_traj(y)

    def __getattr__(self, attr):
        """What to do when attributes can not be obtained in a normal way?.

        This method allows access of the self.CVs dictionary's values as
        instance variables. Furthermore, of a mdtraj variable is called,
        the traj is loaded and the correct variable is returned.

        """
        if attr in self._mdtraj_attr:
            self.load_traj()
            return getattr(self.traj, attr)
            # if self.index == (None, ):
            #     return getattr(self.traj, attr)
            # else:
            #     if len(self.id) == len(self.traj):
            #         return getattr(self.traj, attr)
            #     else:
            #         return getattr(self.traj[self.index], attr)
        elif attr in self._CVs:
            # if self._already_indexed:
            #     return self._CVs[attr].values.squeeze()
            # if self._original_frame_indices.size != 0:
            #     index = self._original_frame_indices
            # elif self._original_frame_indices.size == 0 and self.index is not None:
            #     index = self.index
            # else:
            #     return self._CVs[attr].values.squeeze()
            return self._CVs[attr].values.squeeze()  # [index]
        elif attr == "traj":
            self.__getattribute__(attr)
        elif attr == "id":
            self.__getattribute__(attr)
        else:
            raise AttributeError(f"'SingleTraj' object has no attribute '{attr}'")

    def _string_summary(self) -> str:
        """Returns a summary about the current instance.

        Number of frames, index, loaded CVs.

        """
        s = f"encodermap.SingleTraj object. Current backend is {self.backend}."
        if self.basename:
            s += f" Basename is {self.basename}."
        if self.index is not None:
            with np.printoptions(threshold=1, edgeitems=1):
                s += f" At indices {self.index}."
        if self._CVs:
            for key, value in self._CVs.items():
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
            s += f" Common string is {self.common_str}."
        return s

    def __len__(self):
        return self.n_frames

    def __str__(self):
        return self._string_summary()

    def __repr__(self):
        return f"<{self._string_summary()} Object at 0x{id(self):02x}>"

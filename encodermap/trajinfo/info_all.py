# -*- coding: utf-8 -*-
# encodermap/trajinfo/info_all.py
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
"""Classes to work with ensembles of trajectories.

The statistics of a protein can be better described by an ensemble of proteins,
rather than a single long trajectory. Treating a protein in such a way opens great
possibilities and changes the way one can treat molecular dynamics data.
Trajectory ensembles allow:
    * Faster convergence via adaptive sampling.


This subpackage contains two classes which are containers of trajectory data.
The SingleTraj trajectory contains information about a single trajectory.
The TrajEnsemble class contains information about multiple trajectories. This adds
a new dimension to MD data. The time and atom dimension are already established.
Two frames can be appended along the time axis to get a trajectory with multiple
frames. If they are appended along the atom axis, the new frame contains the
atoms of these two. The trajectory works in a similar fashion. Adding two trajectories
along the trajectory axis returns a trajectory ensemble, represented as a `TrajEnsemble`
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
import contextlib
import copy
import glob
import json
import operator
import os
import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, overload

# Third Party Imports
import h5py
import numpy as np
import tables
from optional_imports import _optional_import

# Local Folder Imports
from .._typing import CanBeIndex, CustomAAsDict
from ..misc.misc import _TOPOLOGY_EXTS, get_full_common_str_and_ref
from ..misc.xarray_save_wrong_hdf5 import save_netcdf_alongside_mdtraj


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:  # pragma: no cover
    # Third Party Imports
    import mdtraj as md
    import pandas as pd
    import xarray as xr

    # Local Folder Imports
    from .info_single import SingleTraj
    from .trajinfo_utils import CustomTopology, TrajEnsembleFeatureType
string_types = (str,)


################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")
pd = _optional_import("pandas")
xr = _optional_import("xarray")
HDF5TrajectoryFile = _optional_import("mdtraj", "formats.hdf5")

################################################################################
# Globals
################################################################################


__all__ = ["TrajEnsemble"]


################################################################################
# Utils
################################################################################


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _check_mode(m, modes):
    if m not in modes:
        raise ValueError(
            "This operation is only available when a file " 'is open in mode="%s".' % m
        )


################################################################################
# Classes
################################################################################


class HDF5GroupWrite(md.formats.HDF5TrajectoryFile):
    def _initialize_headers(
        self,
        group_id: str,
        n_atoms: int,
        set_coordinates: bool,
        set_time: bool,
        set_cell: bool,
        set_velocities: bool,
        set_kineticEnergy: bool,
        set_potentialEnergy: bool,
        set_temperature: bool,
        set_alchemicalLambda: bool,
    ) -> None:
        # Local Folder Imports
        from .._version import get_versions

        version = get_versions()["version"]
        self._n_atoms = n_atoms

        self._handle.root._v_attrs.conventions = "Pande"
        self._handle.root._v_attrs.conventionVersion = "1.1"
        self._handle.root._v_attrs.program = "MDTraj"
        self._handle.root._v_attrs.programVersion = version
        self._handle.root._v_attrs.title = "title"

        # if the client has not the title attribute themselves, we'll
        # set it to MDTraj as a default option.
        if not hasattr(self._handle.root._v_attrs, "application"):
            self._handle.root._v_attrs.application = "MDTraj"

        # create arrays that store frame level informat
        if set_coordinates:
            self._create_earray(
                where="/",
                name=f"coordinates_{group_id}",
                atom=self.tables.Float32Atom(),
                shape=(0, self._n_atoms, 3),
            )
            getattr(self._handle.root, f"coordinates_{group_id}").attrs[
                "units"
            ] = "nanometers"

        if set_time:
            self._create_earray(
                where="/",
                name=f"time_{group_id}",
                atom=self.tables.Float32Atom(),
                shape=(0,),
            )
            getattr(self._handle.root, f"time_{group_id}").attrs[
                "units"
            ] = "picoseconds"

        if set_cell:
            self._create_earray(
                where="/",
                name=f"cell_lengths_{group_id}",
                atom=self.tables.Float32Atom(),
                shape=(0, 3),
            )
            self._create_earray(
                where="/",
                name=f"cell_angles_{group_id}",
                atom=self.tables.Float32Atom(),
                shape=(0, 3),
            )
            getattr(self._handle.root, f"cell_lengths_{group_id}").attrs[
                "units"
            ] = "nanometers"
            getattr(self._handle.root, f"cell_angles_{group_id}").attrs[
                "units"
            ] = "degrees"

        if set_velocities:
            self._create_earray(
                where="/",
                name=f"velocities_{group_id}",
                atom=self.tables.Float32Atom(),
                shape=(0, self._n_atoms, 3),
            )
            getattr(self._handle.root, f"velocities_{group_id}").attrs[
                "units"
            ] = "nanometers/picosecond"

        if set_kineticEnergy:
            self._create_earray(
                where="/",
                name=f"kineticEnergy_{group_id}",
                atom=self.tables.Float32Atom(),
                shape=(0,),
            )
            getattr(self._handle.root, f"kineticEnergy_{group_id}").attrs[
                "units"
            ] = "kilojoules_per_mole"

        if set_potentialEnergy:
            self._create_earray(
                where="/",
                name=f"potentialEnergy_{group_id}",
                atom=self.tables.Float32Atom(),
                shape=(0,),
            )
            getattr(self._handle.root, f"potentialEnergy_{group_id}").attrs[
                "units"
            ] = "kilojoules_per_mole"

        if set_temperature:
            self._create_earray(
                where="/",
                name=f"temperature_{group_id}",
                atom=self.tables.Float32Atom(),
                shape=(0,),
            )
            getattr(self._handle.root, f"temperature_{group_id}").attrs[
                "units"
            ] = "kelvin"

        if set_alchemicalLambda:
            self._create_earray(
                where="/",
                name=f"lambda_{group_id}",
                atom=self.tables.Float32Atom(),
                shape=(0,),
            )
            self._get_node("/", name=f"lambda_{group_id}").attrs[
                "units"
            ] = "dimensionless"

    def write_into_group(
        self,
        group_id: str,
        coordinates: np.ndarray,
        time: np.ndarray,
        cell_lengths: np.ndarray,
        cell_angles: np.ndarray,
        topology: md.Topology,
    ) -> None:
        # Third Party Imports
        from mdtraj.utils import ensure_type, in_units_of

        _check_mode(self.mode, ("w", "a"))

        if self.mode == "a":
            try:
                self._frame_index = len(
                    getattr(self._handle.root, f"coordinates_{group_id}")
                )
                self._needs_initialization = False
            except self.tables.NoSuchNodeError:
                self._frame_index = 0
                self._needs_initialization = True

        # these must be either both present or both absent. since
        # we're going to throw an error if one is present w/o the other,
        # lets do it now.
        if cell_lengths is None and cell_angles is not None:
            raise ValueError("cell_lengths were given, but no cell_angles")
        if cell_lengths is not None and cell_angles is None:
            raise ValueError("cell_angles were given, but no cell_lengths")

        # if the input arrays are openmm.unit.Quantities, convert them
        # into md units. Note that this acts as a no-op if the user doesn't
        # have openmm.unit installed (e.g. they didn't install OpenMM)
        coordinates = in_units_of(coordinates, None, "nanometers")
        time = in_units_of(time, None, "picoseconds")
        cell_lengths = in_units_of(cell_lengths, None, "nanometers")
        cell_angles = in_units_of(cell_angles, None, "degrees")

        # do typechecking and shapechecking on the arrays
        # this ensure_type method has a lot of options, but basically it lets
        # us validate most aspects of the array. Also, we can upconvert
        # on defficent ndim, which means that if the user sends in a single
        # frame of data (i.e. coordinates is shape=(n_atoms, 3)), we can
        # realize that. obviously the default mode is that they want to
        # write multiple frames at a time, so the coordinate shape is
        # (n_frames, n_atoms, 3)
        coordinates = ensure_type(
            coordinates,
            dtype=np.float32,
            ndim=3,
            name="coordinates",
            shape=(None, None, 3),
            can_be_none=False,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True,
        )
        (
            n_frames,
            n_atoms,
        ) = coordinates.shape[0:2]
        time = ensure_type(
            time,
            dtype=np.float32,
            ndim=1,
            name="time",
            shape=(n_frames,),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True,
        )
        cell_lengths = ensure_type(
            cell_lengths,
            dtype=np.float32,
            ndim=2,
            name="cell_lengths",
            shape=(n_frames, 3),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True,
        )
        cell_angles = ensure_type(
            cell_angles,
            dtype=np.float32,
            ndim=2,
            name="cell_angles",
            shape=(n_frames, 3),
            can_be_none=True,
            warn_on_cast=False,
            add_newaxis_on_deficient_ndim=True,
        )

        # if this is our first call to write(), we need to create the headers
        # and the arrays in the underlying HDF5 file
        if self._needs_initialization:
            self._initialize_headers(
                group_id=group_id,
                n_atoms=n_atoms,
                set_coordinates=True,
                set_time=(time is not None),
                set_cell=(cell_lengths is not None or cell_angles is not None),
                set_velocities=False,
                set_kineticEnergy=False,
                set_potentialEnergy=False,
                set_temperature=False,
                set_alchemicalLambda=False,
            )
            self._needs_initialization = False

            # we need to check that that the entries that the user is trying
            # to save are actually fields in OUR file

        try:
            # try to get the nodes for all the fields that we have
            # which are not None
            names = [
                f"coordinates_{group_id}",
                f"time_{group_id}",
                f"cell_angles_{group_id}",
                f"cell_lengths_{group_id}",
            ]
            for name in names:
                contents = locals()[name.replace(f"_{group_id}", "")]
                if contents is not None:
                    self._get_node(where="/", name=name).append(contents)
                if contents is None:
                    # for each attribute that they're not saving, we want
                    # to make sure the file doesn't explect it
                    try:
                        self._get_node(where="/", name=name)
                        raise AssertionError()
                    except self.tables.NoSuchNodeError:
                        pass
        except self.tables.NoSuchNodeError:
            raise ValueError(
                "The file that you're trying to save to doesn't "
                "contain the field %s. You can always save a new trajectory "
                "and have it contain this information, but I don't allow 'ragged' "
                "arrays. If one frame is going to have %s information, then I expect "
                "all of them to. So I can't save it for just these frames. Sorry "
                "about that :)" % (name, name)
            )
        except AssertionError:
            raise ValueError(
                "The file that you're saving to expects each frame "
                "to contain %s information, but you did not supply it."
                "I don't allow 'ragged' arrays. If one frame is going "
                "to have %s information, then I expect all of them to. " % (name, name)
            )

        self._frame_index += n_frames
        self.flush()
        self.write_topology(group_id, topology)

    def write_topology(
        self,
        group_id: str,
        topology_object: md.Topology,
    ) -> None:
        _check_mode(self.mode, ("w", "a"))

        try:
            node = self._handle.get_node("/", name=f"topology_{group_id}")
        except tables.NoSuchNodeError:
            pass
        else:
            if self.mode != "a":
                raise Exception(
                    f"File already exists and has trajectory information. "
                    f"Set `overwrite` to True to overwrite."
                )
            self._handle.remove_node("/", name=f"topology_{group_id}")

        # we want to be able to handle the openmm Topology object
        # here too, so if it's not an mdtraj topology we'll just guess
        # that it's probably an openmm topology and convert
        if not isinstance(topology_object, md.Topology):
            topology_object = md.Topology.from_openmm(topology_object)

        try:
            topology_dict = {"chains": [], "bonds": []}

            for chain in topology_object.chains:
                chain_dict = {"residues": [], "index": int(chain.index)}
                for residue in chain.residues:
                    residue_dict = {
                        "index": int(residue.index),
                        "name": str(residue.name),
                        "atoms": [],
                        "resSeq": int(residue.resSeq),
                        "segmentID": str(residue.segment_id),
                    }

                    for atom in residue.atoms:
                        try:
                            element_symbol_string = str(atom.element.symbol)
                        except AttributeError:
                            element_symbol_string = ""

                        residue_dict["atoms"].append(
                            {
                                "index": int(atom.index),
                                "name": str(atom.name),
                                "element": element_symbol_string,
                            }
                        )
                    chain_dict["residues"].append(residue_dict)
                topology_dict["chains"].append(chain_dict)

            for atom1, atom2 in topology_object.bonds:
                topology_dict["bonds"].append([int(atom1.index), int(atom2.index)])

        except AttributeError as e:
            raise AttributeError(
                "topology_object fails to implement the"
                "chains() -> residue() -> atoms() and bond() protocol. "
                "Specifically, we encountered the following %s" % e
            )

        # actually set the tables
        try:
            self._remove_node(where="/", name="topology")
        except self.tables.NoSuchNodeError:
            pass

        data = json.dumps(topology_dict)
        if not isinstance(data, bytes):
            data = data.encode("ascii")

        if self.tables.__version__ >= "3.0.0":
            self._handle.create_array(
                where="/", name=f"topology_{group_id}", obj=[data]
            )
        else:
            self._handle.createArray(
                where="/", name=f"topology_{group_id}", object=[data]
            )

    def read(self, traj_num: int):
        # Third Party Imports
        from mdtraj.utils import in_units_of

        def get_field(name, slice, out_units, can_be_none=True):
            try:
                node = self._get_node(where="/", name=name + f"_{traj_num}")
                data = node.__getitem__(slice)
                in_units = node.attrs.units
                if not isinstance(in_units, string_types):
                    in_units = in_units.decode()
                data = in_units_of(data, in_units, out_units)
                return data
            except self.tables.NoSuchNodeError:
                if can_be_none:
                    return None
                raise

        out = {
            "coordinates": get_field(
                "coordinates",
                (slice(None), slice(None), slice(None)),
                out_units="nanometers",
                can_be_none=False,
            ),
            "time": get_field(
                "time", slice(None), out_units="picoseconds", can_be_none=False
            ),
            "cell_lengths": get_field(
                "cell_lengths",
                (slice(None), slice(None)),
                out_units="nanometers",
                can_be_none=False,
            ),
            "cell_angles": get_field(
                "cell_angles",
                (slice(None), slice(None)),
                out_units="degrees",
                can_be_none=False,
            ),
        }
        return out

    def read_topology(
        self,
        group_id: str,
    ) -> md.Topology:
        # Third Party Imports
        import mdtraj.core.element as elem

        try:
            raw = self._get_node("/", name=group_id)[0]
            if not isinstance(raw, string_types):
                raw = raw.decode()
            topology_dict = json.loads(raw)
        except self.tables.NoSuchNodeError:
            return None

        topology = md.Topology()

        for chain_dict in sorted(
            topology_dict["chains"], key=operator.itemgetter("index")
        ):
            chain = topology.add_chain()
            for residue_dict in sorted(
                chain_dict["residues"], key=operator.itemgetter("index")
            ):
                try:
                    resSeq = residue_dict["resSeq"]
                except KeyError:
                    resSeq = None
                    warnings.warn(
                        "No resSeq information found in HDF file, defaulting to zero-based indices"
                    )
                try:
                    segment_id = residue_dict["segmentID"]
                except KeyError:
                    segment_id = ""
                residue = topology.add_residue(
                    residue_dict["name"], chain, resSeq=resSeq, segment_id=segment_id
                )
                for atom_dict in sorted(
                    residue_dict["atoms"], key=operator.itemgetter("index")
                ):
                    try:
                        element = elem.get_by_symbol(atom_dict["element"])
                    except KeyError:
                        element = elem.virtual
                    topology.add_atom(atom_dict["name"], element, residue)

        atoms = list(topology.atoms)
        for index1, index2 in topology_dict["bonds"]:
            topology.add_bond(atoms[index1], atoms[index2])

        return topology

    def read_trajs(self) -> Sequence[md.Trajectory]:
        # Third Party Imports
        from mdtraj.core.trajectory import Trajectory

        nodes = [n.name for n in self._handle.list_nodes("/") if n.name != "CVs"]
        traj_nums = []
        trajs = {}
        for node in nodes:
            traj_nums.extend(re.findall("\d+", node))
        traj_nums = list(sorted(map(int, set(traj_nums))))
        for traj_num in traj_nums:
            topology = self.read_topology(f"topology_{traj_num}")
            data = self.read(traj_num)
            trajs[traj_num] = Trajectory(
                xyz=data["coordinates"],
                topology=topology,
                time=data["time"],
                unitcell_lengths=data["cell_lengths"],
                unitcell_angles=data["cell_angles"],
            )
        return trajs

    def read_traj(self, traj_num: int) -> md.Trajectory:
        # Third Party Imports
        from mdtraj.core.trajectory import Trajectory

        topology = self.read_topology(f"topology_{traj_num}")
        data = self.read(traj_num)
        traj = Trajectory(
            xyz=data["coordinates"],
            topology=topology,
            time=data["time"],
            unitcell_lengths=data["cell_lengths"],
            unitcell_angles=data["cell_angles"],
        )
        return traj


class TrajEnsembleTsel:
    def __init__(self, other):
        self.other = other

    def __getitem__(self, item: CanBeIndex) -> Union[TrajEnsemble, SingleTraj]:
        items = np.array(list(self.other.trajs_by_traj_num.keys()))
        if isinstance(item, (int, np.int64)):
            if item not in items:
                raise ValueError(
                    f"No trajectories with traj_num {item} in TrajEnsemble {self.other} "
                    f"with trajectories: {items}"
                )
            return self.other.trajs_by_traj_num[item]
        elif isinstance(item, (list, np.ndarray)):
            idx = np.where(np.in1d(items, np.asarray(item)))[0]
        elif isinstance(item, slice):
            raise NotImplementedError("Currently can't index trajs with slice.")
        else:
            raise ValueError(
                f"The `tsel[]` method of `TrajEnsmeble` takes {CanBeIndex} types, "
                f"but {type(item)} was provided."
            )
        if len(idx) == 0:
            raise ValueError(
                f"No trajs with traj_nums {item} in TrajEnsmble {self.other} "
                f"with trajectories: {items}"
            )
        return self.other[idx]


class TrajEnsemble:
    """This class contains the info about many trajectories.
    Topologies can be mismatched.

    This class is a fancy list of `encodermap.trajinfo.SingleTraj` objects.
    Trajectories can have different topologies and will be grouped by
    the `common_str` argument.

    `TrajEnsemble` supports fancy indexing. You can slice to your liking trajs[::5]
    returns an `TrajEnsemble` object that only consideres every fifth frame.
    Besides indexing by slices and integers, you can pass a 2-dimensional
    np.array. np.array([[0, 5], [1, 10], [5, 20]]) will return a `TrajEnsemble`
    object with frame 5 of trajectory 0, frame 10 of trajectory 1 and frame 20
    of trajectory 5. Simply passing an integer as index returns the
    corresponding `SingleTraj` object.

    The `TrajEnsemble` class also contains an iterator to iterate over trajectores.
    You could do::
    >>> for traj in trajs:  # doctest: +SKIP
    ...     for frame in traj:
    ...         print(frame)

    Attributes:
        CVs (dict): The collective variables of the `SingleTraj` classes. Only
            CVs with matching names in all `SingleTraj` classes are returned.
            The data is stacked along a hypothetical time axis along the trajs.
        _CVs (xarray.Dataset): The same data as in CVs but with labels.
            Additionally, the xarray is not stacked along the time axis.
            It contains an extra dimension for trajectories.
        n_trajs (int): Number of individual trajectories in this class.
        n_frames (int): Number of frames, sum over all trajectories.
        locations (list of str): A list with the locations of the trajectories.
        top (list of mdtraj.Topology): A list with the reference pdb for each trajecotry.
        basenames (list of str): A list with the names of the trajecotries.
            The leading path and the file extension is omitted.
        name_arr (np.ndarray of str): An array with len(name_arr) == n_frames.
            This array keeps track of each frame in this object by identifying each
            frame with a filename. This can be useful, when frames are mixed inside
            a `TrajEnsemble` class.

    Examples:
        >>> # Create a trajectory ensemble from a list of files
        >>> import encodermap as em
        >>> trajs = em.TrajEnsemble(
        ...     [
        ...         'https://files.rcsb.org/view/1YUG.pdb',
        ...         'https://files.rcsb.org/view/1YUF.pdb',
        ...     ],
        ... )
        >>> # trajs are inernally numbered
        >>> print([traj.traj_num for traj in trajs])
        [0, 1]
        >>> # Build a new traj from random frames
        >>> # Let's say frame 2 of traj 0, frame 5 of traj 1 and again frame 2 of traj 0
        >>> # Doing this every frame will now be its own trajectory for easier bookkeeping
        >>> arr = np.array([[0, 2], [1, 5], [0, 2]])
        >>> new_trajs = trajs[arr]
        >>> print(new_trajs.n_trajs)
        2
        >>> # trace back a single frame
        >>> frame_num = 28
        >>> index = trajs.index_arr[frame_num]
        >>> print(
        ...     f"Frame {frame_num}, originates from trajectory "
        ...     f"{trajs.basenames[index[0]]}, frame {index[1]}."
        ... )
        Frame 28, originates from trajectory 1YUF, frame 13.

    """

    def __init__(
        self,
        trajs: Union[
            Sequence[str],
            Sequence[Path],
            Sequence[md.Trajectory],
            Sequence["SingleTraj"],
        ],
        tops: Union[None, Sequence[str], Sequence[Path]] = None,
        backend: Literal["mdtraj", "no_load"] = "no_load",
        common_str: Optional[Sequence[str]] = None,
        basename_fn: Optional[Callable] = None,
        traj_nums: Optional[Sequence[int]] = None,
        custom_top: Optional[CustomAAsDict] = None,
    ) -> None:
        """Initialize the Info class with two lists of files.

        Args:
            trajs (Union[list[str], list[md.Trajectory], list[SingleTraj], list[Path]]):
                List of strings with paths to trajectories.
            tops (Optional[list[str]]): List of strings with paths to reference pdbs.
            backend (str, optional): Choose the backend to load trajectories.
                * 'mdtraj' uses mdtraj, which loads all trajectories into RAM.
                * 'no_load' creates an empty trajectory object.
                Defaults to 'no_load'.
            common_str (list of str, optional): If you want to include trajectories with
                different topology. The common string is used to pair traj-files
                (.xtc, .dcd, .lammpstrj) with their topology (.pdb, .gro, ...).
                The common-string should be a substring of matching trajs
                and topologies.
            basename_fn (Union[None, function], optional): A function to apply
                to the `traj_file` string to return the basename of the
                trajectory. If None is provided, the filename without extension
                will be used. When all files are named the same and the folder
                they're in defines the name of the trajectory, you can supply
                `lambda x: split('/')[-2]` as this argument. Defaults to None.
            custom_top: Optional[CustomAAsDict]: An instance of the
                `CustomTopology` class or a dictionary that can be made into such.

        """
        # defaults
        # Local Folder Imports
        from .info_single import SingleTraj

        self.backend = backend

        # custom topology to load dihedral angles
        self._custom_top = custom_top

        # set the trajnums
        if traj_nums is not None:
            # Standard Library Imports
            from copy import deepcopy

            trajs_ = []
            if not len(traj_nums) == len(trajs):
                raise Exception(
                    f"Uneven length of `traj_nums` ({len(traj_nums)} "
                    f"and `trajs` ({len(trajs)}) provided."
                )
            for n, t in zip(traj_nums, trajs):
                t = deepcopy(t)
                t.traj_num = n
                trajs_.append(t)
            trajs = trajs_

        # make sure, that traj_nums are not duplicated
        elif all([isinstance(t, SingleTraj) for t in trajs]) and isinstance(
            trajs, list
        ):
            if any([t.traj_num is None for t in trajs]) and any(
                [isinstance(t.traj_num, int) for t in trajs]
            ):
                raise Exception(
                    f"The `SingleTraj`s you provided have bad `traj_num`s "
                    f"one has `None`, the others have int: {[t.traj_num for t in trajs]}"
                )
            if not all([(i.traj_num is None) for i in trajs]):
                uniques, counts = np.unique(
                    np.asarray([t.traj_num for t in trajs]), return_counts=True
                )
                if np.any(counts > 1):
                    ex_num = uniques[np.argmax(counts)]
                    raise Exception(
                        f"The `traj_num` attributes of the provided {len(trajs)} `SingleTraj`s is "
                        f"not unique, the `traj_num` {ex_num} occurs {np.max(counts)} times. "
                        f"This can happen, if you use `SingleTraj`s, that are already part of "
                        f"a `TrajEnsemble`. To create copies of the `SingleTraj`s and over"
                        f"write their `traj_num`s, use the `overwrite_trajnums()` constructor."
                    )
                trajs = list(sorted(trajs, key=lambda x: x.traj_num))

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
        if all([isinstance(traj, Path) for traj in trajs]) and not tops:
            self._top_files = [str(t) for t in trajs]
        if isinstance(tops, str):
            self._top_files = [tops]
        self.traj_files = trajs

    @classmethod
    def overwrite_trajnums(
        cls,
        trajs: Sequence["SingleTraj"],
    ) -> TrajEnsemble:
        """Creates an `TrajEnsemble` by copying the provided `SingleTrajs` and
        changing their `traj_num`.

        Args:
            trajs (Sequence[SingleTraj]): The sequence of trajs.

        Returns:
            TrajEnsemble: A `TrajEnsemble` class.

        """
        # Standard Library Imports
        from copy import deepcopy

        new_trajs = []
        for i, traj in enumerate(trajs):
            traj = deepcopy(traj)
            traj.traj_num = i
            new_trajs.append(traj)
        return cls(new_trajs)

    @classmethod
    def from_textfile(
        cls,
        fname: Union[str, Path],
        basename_fn: Optional[Callable] = None,
    ) -> TrajEnsemble:
        """Creates an `TrajEnsemble` object from a textfile.

        The textfile needs to be space-separated with two or three columns.
        Column 1: The trajectory file.
        Column 2: The corresponding topology file (If you are using .h5 trajs,
            column 1 and 2 will be identical).
        Column 3: The common string of the trajectory. This column can be left
            out, which will result in an `TrajEnsemble` without common_strings.

        Args:
            fname (Union[str, Path]): File to be read.
            basename_fn (Union[None, function], optional): A function to apply
                to the `traj_file` string to return the basename of the trajectory.
                If None is provided, the filename without extension will be used.
                When all files are named the same and the folder they're in
                defines the name of the trajectory, you can supply
                `lambda x: split('/')[-2]` as this argument. Defaults to None.

        Returns:
            TrajEnsemble: An instantiated TrajEnsemble class.

        """
        # Local Folder Imports
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

        return cls(
            trajs, common_str=np.unique(common_str).tolist(), basename_fn=basename_fn
        )

    @classmethod
    def from_xarray(
        cls,
        fnames: Union[Sequence[str], Sequence[Path]],
        basename_fn: Optional[Callable] = None,
    ) -> TrajEnsemble:
        """Loads multiple .h5 files and combines them into an Ensemble.

        Args:
            fnames (Union[Sequence[str], Sequence[Path]]): The files, which will
                be opened as a `xr.mfdataset`. The dataset will be iterated over
                along the trajectory axis and `SingleTrajs` will be created.
            basename_fn (Optional[Callable]): A callable to define the `basename`
                attribute of the `SingleTraj`s.

        Returns:
            TrajEnsemble: An instance of the `TrajEnsemble` class.

        """
        # Local Folder Imports
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

    @classmethod
    def from_dataset(
        cls,
        fname: Union[str, Path],
        basename_fn: Optional[Callable] = None,
    ) -> TrajEnsemble:
        # Local Folder Imports
        from .info_single import SingleTraj

        traj_nums = []
        with h5py.File(fname) as h5file:
            for key in h5file.keys():
                if key == "CVs":
                    continue
                traj_nums.extend(re.findall("\d+", key))
        traj_nums = list(sorted(map(int, set(traj_nums))))

        trajs = []
        for traj_num in traj_nums:
            trajs.append(
                SingleTraj(traj=fname, traj_num=traj_num, basename_fn=basename_fn)
            )
        newclass = cls(trajs=trajs)
        return newclass

    def load_custom_topology(
        self,
        custom_top: Optional["CustomTopology", CustomAAsDict] = None,
    ) -> None:
        """Loads a custom_topology from a `CustomTopology` class or a dict.

        See Also:
            `CustomTopology`

        Args:
            custom_top: Optional[Union[CustomTopology, CustomAAsDict]]: An instance of the
                `CustomTopology` class or a dictionary that can be made into such.

        """
        for traj in self.trajs:
            traj.load_custom_topology(custom_top)

    @property
    def tsel(self):
        return TrajEnsembleTsel(self)

    @property
    def featurizer(self):
        # Local Folder Imports
        from ..loading.featurizer import Featurizer

        if not hasattr(self, "_featurizer"):
            self._featurizer = Featurizer(self)
        return self._featurizer

    @property
    def traj_files(self) -> list[str]:
        """list: A list of the traj_files of the individual SingleTraj classes."""
        return self._traj_files

    @property
    def top_files(self) -> list[str]:
        """list: Returns minimal set of topology files.

        If yoy want a list of top files with the same
        length as self.trajs use self._top_files and
        self._traj_files.

        """
        return list(dict.fromkeys(self._top_files))

    @traj_files.setter
    def traj_files(self, trajs):
        # Local Folder Imports
        from ..trajinfo import info_single

        traj_nums = np.arange(len(trajs))
        # fill this lists
        self.trajs = []

        if all([isinstance(traj, Path) for traj in trajs]):
            trajs = [str(traj) for traj in trajs]

        if all([isinstance(i, md.Trajectory) for i in trajs]):
            self.backend = "mdtraj"
            self.trajs = [
                info_single.SingleTraj(
                    traj,
                    traj_num=i,
                    basename_fn=self.basename_fn,
                    custom_top=self._custom_top,
                )
                for i, traj in zip(traj_nums, trajs)
            ]
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
            for i, traj in zip(traj_nums, trajs):
                if traj.traj_num is None:
                    traj.traj_num = i
                    if traj._CVs:
                        traj._CVs = traj._CVs.assign_coords(traj_num=[i])
        elif all([isinstance(i, str) for i in trajs]) and self.top_files:
            # find common_str matches in top_files and traj_files
            (
                self._traj_files,
                self._top_files,
                self._common_str,
            ) = get_full_common_str_and_ref(trajs, self._top_files, self.common_str)
            for i, t, top, cs in zip(
                traj_nums, self._traj_files, self._top_files, self._common_str
            ):
                t = info_single.SingleTraj(
                    traj=t,
                    top=top,
                    backend=self.backend,
                    common_str=cs,
                    traj_num=i,
                    basename_fn=self.basename_fn,
                    custom_top=self._custom_top,
                )
                self.trajs.append(t)
        elif all([isinstance(i, str) for i in trajs]) and not self.top_files:
            for i, traj_file in zip(traj_nums, trajs):
                self.trajs.append(
                    info_single.SingleTraj(
                        traj=traj_file,
                        basename_fn=self.basename_fn,
                        traj_num=i,
                        custom_top=self._custom_top,
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
            except IOError as e:
                if "no such file" in str(e).lower():
                    raise e
                raise Exception(
                    f"I have no idea how this IOError can occur. {self.trajs=}."
                ) from e
        return out

    @property
    def trajs_by_top(self) -> dict[md.Topology, TrajEnsemble]:
        """dict[md.Topology, TrajEnsemble]: Returns the trajs in `self` ordered by top.

        If all trajectories share the same topology, a dict with
        one key will be returned.

        """
        out = {}
        for traj in self.trajs:
            out.setdefault(traj.top, []).append(traj)
        out = {k: TrajEnsemble(v) for k, v in out.items()}
        return out

    @property
    def trajs_by_common_str(self) -> dict[Union[None, str], TrajEnsemble]:
        """dict[str, TrajEnsemble]: Returns the trajs in `self` ordered by top.

        If all trajectories share the same common_str, a dict with
        one key will be returned. As the common_str can be None, None can also
        occur as a key in this dict.

        """
        out = {}
        for traj in self.trajs:
            out.setdefault(traj.common_str, []).append(traj)
        out = {k: TrajEnsemble(v) for k, v in out.items()}
        return out

    @property
    def trajs_by_traj_num(self) -> dict[int, SingleTraj]:
        out = {}
        for traj in self:
            out[traj.traj_num] = traj
        return out

    @property
    def id(self) -> np.ndarray:
        """np.ndarray: Duplication of self.index_arr"""
        return self.index_arr

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
        """int: Number of trajectories in this ensemble."""
        return len(self.trajs)

    def del_CVs(self) -> None:
        """Deletes all currently loaded from memory."""
        for traj in self.trajs:
            traj.del_CVs()

    @property
    def _CVs(self) -> xr.Dataset:
        """xarray.Dataset: Returns x-array Dataset of matching CVs. stacked
        along the trajectory-axis."""
        # Local Folder Imports
        from .trajinfo_utils import trajs_combine_attrs

        return xr.combine_nested(
            [traj._CVs for traj in self.trajs],
            concat_dim="traj_num",
            compat="broadcast_equals",
            fill_value=np.nan,
            coords="all",
            join="outer",
            combine_attrs=trajs_combine_attrs,
        )

    def _calc_CV(self) -> dict[str, np.ndarray]:
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
                        try:
                            data = np.concatenate([d.squeeze() for d in data], axis=0)
                        except ValueError as e:
                            if "zero-dimensional" in str(e):
                                data = np.hstack([d.squeeze() for d in data])
                            else:
                                raise e
                    CVs[key] = data
            return CVs

    @property
    def CVs(self) -> dict[str, np.ndarray]:
        """dict: Returns dict of CVs in SingleTraj classes. Only CVs with the same names
        in all SingleTraj classes are loaded.

        """
        return self._calc_CV()

    @property
    def locations(self) -> list[str]:
        """list: Duplication of self.traj_files but using the trajs own traj_file attribute.
        Ensures that traj files are always returned independent of the current load state.
        """
        return [traj.traj_file for traj in self.trajs]

    @property
    def index_arr(self) -> np.ndarray:
        """np.ndarray: Returns np.ndarray with ndim = 2. Clearly assigning every
        loaded frame an identifier of traj_num (self.index_arr[:,0]) and
        frame_num (self.index_arr[:,1]). Can be used to create an unspecified
        subset of frames and can be useful when used with clustering.

        """
        # can also be made to use the SingleTraj.index_arr attribute,
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
        """bool: Is true, if CVs can be loaded from file. Can be used to build a
        data generator from."""
        return all([traj.CVs_in_file for traj in self.trajs])

    @property
    def traj_joined(self) -> md.Trajectory:
        """mdtraj.Trajectory: Returns a mdtraj Trajectory with every frame of
        this class appended along the time axis.

        Can also work if different topologies (with the same number of atoms) are loaded.
        In that case, the first frame in self will be used as topology parent and the remaining frames'
        xyz coordinates are used to position the parents' atoms accordingly.


        Examples:
            >>> import encodermap as em
            >>> trajs = em.load_project("pASP_pGLU")
            >>> subsample = trajs[0][:20] + trajs[1][:20]
            >>> subsample.split_into_frames().traj_joined  # doctest: +ELLIPSIS
            <mdtraj.Trajectory with 40 frames, 73 atoms, 7 residues, and unitcells at ...>

        """
        # use traj[0] of the trajs list as the traj from which the topology will be used
        parent_traj = self.trajs[0].traj

        # join the correct number of trajs
        # by use of the `divmod` method, the frames parent_traj traj will be
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
        # assert self.n_frames == len(self.trajs), f"{self.n_frames=}, {len(self.trajs)=}"

        # change the xyz coordinates of dummy_traj according to the frames in joined trajs'
        for i, (_, traj) in enumerate(self.iterframes()):
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
                e2 = NotImplementedError(msg)
                raise e2 from e

    def copy(self):
        return deepcopy(self)

    def split_into_frames(self, inplace: bool = False) -> None:
        """Splits self into separate frames.

        Args:
            inplace (bool): Whether to do the split inplace or not.
                Defaults to False and thus, returns a new `TrajEnsemble` class.

        """
        frames = []
        for frame_num, frame in self.iterframes():
            frames.append(frame)
        if inplace:
            self = TrajEnsemble.overwrite_trajnums(frames)
        else:
            return TrajEnsemble.overwrite_trajnums(frames)

    def save_CVs(self, path: Union[str, Path]) -> None:
        """Saves the CVs to a NETCDF file using xarray."""
        self._CVs.to_netcdf(path, format="NETCDF4", engine="h5netcdf")

    def cluster(
        self,
        cluster_id: int,
        col: str,
        memberships: Optional[np.ndarray] = None,
        n_points: int = -1,
    ) -> TrajEnsemble:
        """Clusters this `TrajEnsemble` based on the provided cluster_id and col.

        The `col` parameter takes any CV name, that is per-frame and integer.

        Examples:
            >>> import encodermap as em

        """
        # Standard Library Imports
        from copy import deepcopy

        if memberships is not None:
            self.load_CVs(memberships, "col")

        assert (
            col in self._CVs
        ), f"To use the CV {col} for clustering. Add it to the CVs with `load_CVs`."

        # find the index
        index = self.index_arr[self.CVs[col] == cluster_id]
        frame_index = np.arange(self.n_frames)[self.CVs[col] == cluster_id]
        if n_points > 0:
            ind = np.unique(
                np.round(np.linspace(0, len(index) - 1, n_points)).astype(int)
            )
            index = index[ind]
            frame_index = frame_index[ind]

        try:
            return self[index]
        except IndexError as e:
            return self._return_frames_by_index(frame_index)

    def join(
        self,
        align_string: str = "name CA",
        superpose: bool = True,
        ref_align_string: str = "name CA",
        base_traj: Optional[md.Trajectory] = None,
    ) -> dict[md.Topology, md.Trajectory]:
        all_trajs = []
        out_by_top = {}
        for top, traj in self.trajs_by_top.items():
            traj = traj.traj_joined
            if superpose:
                if base_traj is not None:
                    traj = traj.superpose(
                        base_traj,
                        atom_indices=traj.top.select(align_string),
                        ref_atom_indices=base_traj.top.select(ref_align_string),
                    )
                else:
                    traj = traj.superpose(
                        traj,
                        atom_indices=traj.top.select(align_string),
                    )
            all_trajs.append(traj)
            out_by_top[top] = traj

        # if frames have the same xyz, we can join them
        if all([t.n_atoms == all_trajs[0].n_atoms for t in all_trajs]):
            # superpose all
            for i, traj in enumerate(all_trajs):
                all_trajs[i] = traj.superpose(
                    all_trajs[0],
                    frame=0,
                    atom_indices=traj.top.select(align_string),
                    ref_atom_indices=all_trajs[0].top.select(align_string),
                )

            parent_traj = base_traj
            if parent_traj is None:
                parent_traj = all_trajs[0]

            # divmod
            try:
                no_of_iters, rest = divmod(
                    sum([t.n_frames for t in all_trajs]), parent_traj.n_frames
                )
            except Exception as e:
                raise Exception(
                    f"Can not build a dummy trajectory. Maybe you selected the "
                    f"wrong cluster num. Here's the original Error: {e}"
                )
            for i in range(no_of_iters + 1):
                if i == 0:
                    dummy_traj = copy.deepcopy(parent_traj)
                elif i == no_of_iters:
                    dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj)[:rest])
                else:
                    dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj))

            # set the xyz
            i = 0
            for traj in all_trajs:
                for frame in traj:
                    dummy_traj[0].xyz = frame.xyz

        # return
        return out_by_top

    def stack(
        self,
        align_string: str = "name CA",
        superpose: bool = True,
        ref_align_string: str = "name CA",
        base_traj: Optional[md.Trajectory] = None,
    ) -> md.Trajectory:
        all_trajs = self.join(align_string, superpose, ref_align_string, base_traj)
        atoms = 0
        # stack
        for i, traj in enumerate(all_trajs.values()):
            for j, frame in enumerate(traj):
                atoms += frame.n_atoms
                if i == 0 and j == 0:
                    stacked = deepcopy(frame)
                else:
                    stacked = stacked.stack(frame)
        assert stacked.n_atoms == atoms
        return stacked

    def _trace(self, CV: Sequence[str]) -> np.ndarray:
        """Creates a low-dimensional represnetation of the loaded CV data by
        stacking all arguments in `CV` along a single axis.

        If this `TrajEnsemble` has 10 trajectories with 100 frames each
        and a CV with shape (100, 50, 3) in each of them with the name 'cartesians'
        then `trajs._trace` will return a np.ndarray of shape
        (1000, 150).

        Args:
            CV (Sequence[str]): The CVs to combine in the trace.

        Returns:
            np.ndarray: The trace.

        """
        out = []
        for i in CV:
            v = self.CVs[i]
            out.append(v.reshape(v.shape[0], -1))
        return np.concatenate(out)

    def dash_summary(self) -> pd.DataFrame:
        if self.n_trajs == 1:
            return self.trajs[0].dash_summary()
        else:
            # atoms
            n_atoms = np.unique([t.n_atoms for t in self])
            if len(n_atoms) == 1:
                n_atoms = n_atoms[0]

            # dt
            dt = []
            for t in self:
                dt.extend(np.unique(t.traj.time[1:] - t.traj.time[:-1]))
            dt = np.unique(np.asarray(dt))
            if len(dt) == 1:
                dt = dt[0]

            # traj_files
            traj_files = [t.basename for t in self]

            # topologies
            multiple_tops = len(self.top) != 1
            df = pd.DataFrame(
                {
                    "field": [
                        "n_trajs",
                        "n_frames",
                        "n_atoms",
                        "dt (ps)",
                        "trajs",
                        "multiple tops",
                    ],
                    "value": [
                        self.n_trajs,
                        self.n_frames,
                        n_atoms,
                        dt,
                        traj_files,
                        multiple_tops,
                    ],
                }
            )
            return df.astype(str)

    def load_CVs(
        self,
        data: TrajEnsembleFeatureType,
        attr_name: Optional[str] = None,
        cols: Optional[list[int]] = None,
        deg: Optional[bool] = None,
        labels: Optional[list[str]] = None,
        directory: Optional[Union[str, Path]] = None,
        ensemble: bool = False,
        override: bool = False,
    ) -> None:
        """Loads CVs in various ways. The easiest way is to provide a single
        numpy array and a name for that array.

        Besides np.ndarray, files (.txt and .npy) can be loaded. Features
        or Featurizers can be provided. A xarray.Dataset can be provided.
        A str can be provided which either is the name of one of EncoderMap's
        features (`encodermap.loading.features`) or the string can be 'all',
        which loads all features required for EncoderMap's
        `AngleDihedralCartesianEncoderMap` class.

        Args:
            data (TrajEnsembleFeatureType): The CV to
                load. When a numpy array is provided, it needs to have a shape
                matching `n_frames`. The data is distributed to the trajs.
                When a list of files is provided, `len(data)` needs to match
                `n_trajs`. The first file will be loaded by the first traj
                (based on the traj's `traj_num`) and so on. If a list of
                `np.ndarray` is provided, the first array will be assigned to
                the first traj (based on the traj's `traj_num`). If None is provided,
                the argument `directory` will be used to construct a str like:
                fname = directory + traj.basename + '_' + attr_name. If there are
                .txt or .npy files matching that string in the `directory`,
                the CVs will be loaded from these files to the corresponding
                trajs. Defaults to None.
            attr_name (Optional[str]): The name under which the CV should
                be found in the class. Choose whatever you like. `highd`, `lowd`,
                `dists`, etc. The CV can then be accessed via dot-notation:
                `trajs.attr_name`. Defaults to None, in which case, the argument
                `data` should point to existing files and the `attr_name` will
                be extracted from these files.
            cols (Optional[list[int]]): A list of integers indexing the columns
                of the data to be loaded. This is useful if a file contains
                columns which are not features (i.e. an indexer or the error of
                the features. eg::

                    id   f1    f2    f1_err    f2_err
                    0    1.0   2.0   0.1       0.1
                    1    2.5   1.2   0.11      0.52

                In that case, you would want to supply `cols=[1, 2]` to the `cols`
                argument. If None all columns are loaded. Defaults to None.
            deg (Optional[bool]): Whether to return angular CVs using degrees.
                If None or False, CVs will be in radian. Defaults to None.
            labels (list): A list containing the labels for the dimensions of
                the data. If you provide a `np.ndarra` with shape (n_trajs,
                n_frames, n_feat), this list needs to be of len(n_feat)
                Defaults to None.
            directory (Optional[str]): The directory to save the data at if data
                is an instance of `em.Featurizer` and this featurizer has
                `in_memory` set to Fase. Defaults to ''.
            ensemble (bool): Whether the trajs in this class belong to an ensemble.
                This implies that they contain either the same topology or are
                very similar (think wt, and mutant). Setting this option True will
                try to match the CVs of the trajs onto the same dataset.
                If a VAL residue has been replaced by LYS in the mutant,
                the number of sidechain dihedrals will increase. The CVs of the
                trajs with VAL will thus contain some NaN values. Defaults to False.
            override (bool): Whether to override CVs with the same name as `attr_name`.

        Raises:
            TypeError: When wrong Type has been provided for data.

        """
        # Local Folder Imports
        from .trajinfo_utils import load_CVs_ensembletraj

        # if some trajs are missing time
        b, c = np.unique(
            np.asarray([t.backend for t in self.trajs]), return_counts=True
        )
        if len(b) > 1:
            for traj in self.trajs:
                traj.load_traj()

        load_CVs_ensembletraj(
            self,
            data,
            attr_name,
            cols,
            deg,
            labels,
            directory,
            ensemble,
            override,
        )

    def save(
        self,
        fname: Union[str, Path],
        CVs: Union[Literal["all"], list[str]] = "all",
        overwrite: bool = False,
    ) -> None:
        # Third Party Imports
        from mdtraj.utils import in_units_of

        fname = Path(fname)
        assert (
            fname.suffix == ".h5"
        ), "We recommend the .h5 file extension for these files."
        if fname.is_file() and not overwrite:
            raise IOError(
                f"File {fname} already exists. Set `overwrite` to True to overwrite."
            )
        if fname.is_file() and overwrite:
            fname.unlink()

        for i, traj in self.itertrajs():
            with HDF5GroupWrite(fname, "a", force_overwrite=overwrite) as f:
                f.write_into_group(
                    group_id=str(i),
                    coordinates=in_units_of(
                        traj.xyz, md.Trajectory._distance_unit, f.distance_unit
                    ),
                    time=traj.time,
                    cell_lengths=in_units_of(
                        traj.unitcell_lengths,
                        md.Trajectory._distance_unit,
                        f.distance_unit,
                    ),
                    cell_angles=traj.unitcell_angles,
                    topology=traj.top,
                )
        if CVs == "all":
            save_netcdf_alongside_mdtraj(fname, self._CVs)
            return
        if self._CVs and CVs:
            with h5py.File(fname, "a") as file:
                if "CVs" in list(file.keys()):
                    grp = file["CVs"]
                else:
                    grp = file.create_group("CVs")
                for key in CVs:
                    value = self._CVs[key]
                    assert self.n_frames == value.shape[1]
                    grp.create_dataset(name=key, data=value)

    def _return_trajs_by_index(self, index: Sequence[int]) -> TrajEnsemble:
        """Creates a TrajEnsemble object with the trajs specified by index."""
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

    def _return_frames_by_index(self, index: Sequence[int]) -> TrajEnsemble:
        """Creates a TrajEnsemble object with the frames specified by `index`."""
        new_common_str = []
        frames = []
        for frame_num, frame in self.iterframes():
            if frame_num not in index:
                continue
            frames.append(frame)
            new_common_str.append(frame.common_str)
        new_common_str = list(set(new_common_str))
        for i, frame in enumerate(frames):
            if i == 0:
                trajs_subset = frame._gen_ensemble()
            else:
                new_traj = frame._gen_ensemble()
                trajs_subset = trajs_subset + new_traj
        trajs_subset.common_str = new_common_str
        trajs_subset.basename_fn = self.basename_fn
        return trajs_subset

    def _pyemma_indexing(self, key: np.ndarray) -> TrajEnsemble:
        """Returns a new TrajEnsemble by giving the indices of traj and frame"""
        if key.ndim == 1:
            key = key.reshape(len(key), 1).T
        trajs = []
        for i, num in enumerate(np.unique(key[:, 0])):
            num_ = np.where(np.asarray(self.traj_nums) == num)[0]
            assert (
                len(num_) == 1
            ), f"Can't identify trajectory {num}. These trajs are available: {self.traj_nums}."
            frames = key[key[:, 0] == num, 1]
            trajs.append(self.trajs[num_[0]][frames])
        return TrajEnsemble(
            trajs, basename_fn=self.basename_fn, common_str=self.common_str
        )

    def subsample(
        self,
        stride: int,
        inplace: bool = False,
    ) -> Optional[TrajEnsemble]:
        """Returns a subset of this TrajEnsemble class given the provided stride.

        This is a faster alternative than using the trajs[trajs.index_arr[::1000]]
        when HDF5 trajs are used, because the slicing information is saved in the
        respective SingleTraj classes and loading of single frames is faster in
        HDF5 formatted trajs.

        Note:
            The result from `subsample(1000)` is different from `trajs[trajs.index_arr[::1000]]`.
            With subsample every trajectory is subsampled independently. Consider
            a TrajEnsemble with two `SingleTraj` trajectories with 18 frames each.
            `subsampled = trajs.subsample(5)` would return a `TrajEnsemble` with two
            trajs with 3 frames each (`subsampled.n_frames` is 6). Whereas
            `subsampled = trajs[trajs.index_arr[::5]]` would return a TrajEnsemble
            with 7 SingleTrajs with 1 frame each (`subsampled.n_frames` is 7).
            Because the time and frame numbers are saved all the time, this should not
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

    def get_single_frame(self, key: int) -> "SingleTraj":
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
            raise Exception(
                "Indexing a no_load backend does not work. I need some "
                "information about the frames in each trajectory. Please "
                "load either highd or lowd."
            )
        if key >= self.n_frames:
            raise IndexError(
                "index {} is out of bounds for trajectory with {} frames".format(
                    key, self.n_frames
                )
            )
        if not isinstance(key, (int, np.int32, np.int64)):
            raise IndexError(
                "if you want a single frame, please provide an integer. "
                "If you want multiple frames use ep.TrajEnsemble[]"
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

    @overload
    def batch_iterator(
        self,
        batch_size: int,
        replace: bool = False,
        order: Optional[list[str]] = None,
        deterministic: bool = True,
        yield_index: bool = True,
        start: int = 1,
    ) -> Iterator[
        tuple[
            np.ndarray,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ]
    ]:
        ...

    @overload
    def batch_iterator(
        self,
        batch_size: int,
        replace: bool = False,
        order: Optional[list[str]] = None,
        deterministic: bool = True,
        yield_index: bool = False,
        start: int = 1,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        ...

    def batch_iterator(
        self,
        batch_size: int,
        replace: bool = False,
        order: Optional[list[str]] = None,
        deterministic: bool = False,
        yield_index: bool = False,
        start: int = 1,
    ) -> Iterator[Any]:
        # added .transpose("frame", ...) and removed .T
        stacked_ds = (
            self._CVs.stack({"frame": ("traj_num", "frame_num")})
            .transpose("frame", ...)
            .dropna("frame", how="all")
        )
        full_index = np.arange(len(stacked_ds.coords["frame"]))

        if order is None:
            order = [
                "central_angles",
                "central_dihedrals",
                "central_cartesians",
                "central_distances",
                "side_dihedrals",
            ]

        i = start
        while True:
            # if deterministic:
            #     with temp_seed(i):
            #         idx = np.random.choice(full_index, batch_size, replace=replace)
            # else:
            #     idx = np.random.choice(full_index, batch_size, replace=replace)
            if deterministic:
                np.random.seed(i)
            idx = np.random.choice(full_index, batch_size, replace=replace)
            sub_ds = stacked_ds.isel(frame=idx)
            out = []
            for o in order:
                v = sub_ds[o].values.astype("float32")
                if np.any(np.isnan(v)):
                    raise NotImplementedError(
                        f"Sparse training with `trajs.tf_dataset` currently not possible."
                    )
                out.append(v)
            i += 1
            if yield_index:
                yield idx, tuple(out)
            else:
                yield tuple(out)

    def tf_dataset(
        self,
        batch_size: int,
        replace: bool = False,
        order: Optional[list[str]] = None,
        deterministic: bool = False,
        prefetch: bool = True,
        start: int = 1,
    ) -> "tf.data.Dataset":
        # Third Party Imports
        import tensorflow as tf

        gen = lambda: self.batch_iterator(
            batch_size, replace, order, deterministic, start=start
        )
        if order is None:
            order = [
                "central_angles",
                "central_dihedrals",
                "central_cartesians",
                "central_distances",
                "side_dihedrals",
            ]
        input_shapes = [
            getattr(self._CVs, o).shape[-1:]
            if o != "central_cartesians"
            else getattr(self._CVs, o).shape[-2:]
            for o in order
        ]
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=tuple(
                [
                    tf.TensorSpec(shape=(batch_size, *i), dtype="float32")
                    for i in input_shapes
                ]
            ),
        )
        if prefetch:
            dataset = dataset.prefetch(batch_size * 4)
        if deterministic:
            options = tf.data.Options()
            options.deterministic = True
            dataset = dataset.with_options(options)
            assert dataset.options().deterministic
        return dataset

    def itertrajs(self) -> Iterator[tuple[int, "SingleTraj"]]:
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
        for traj in self:
            yield traj.traj_num, traj

    def iterframes(self) -> Iterator[tuple[int, int, "SingleTraj"]]:
        """Generator over the frames in this class.

        Yields:
            tuple: A tuple containing the following:
                int: The traj_num
                int: The frame_num
                encodermap.SingleTraj: An SingleTraj object.

        Examples:
            >>> import encodermap as em
            >>> trajs = em.TrajEnsemble(
            ...     [
            ...         'https://files.rcsb.org/view/1YUG.pdb',
            ...         'https://files.rcsb.org/view/1YUF.pdb',
            ...     ],
            ... )
            >>> print(trajs.n_frames)
            31
            >>> trajs = trajs.subsample(10)
            >>> trajs.n_frames
            4
            >>> for frame_num, frame in trajs.iterframes():
            ...     print(frame_num, frame.n_frames)
            0 1
            10 1
            0 1
            10 1

        """
        for traj in self:
            yield from traj.iterframes()

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

    def __getitem__(self, key: CanBeIndex) -> TrajEnsemble:
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
            raise IndexError(f"Invalid argument for slicing: {key=}")

    def __reversed__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        # check if traj_files and ids are the same
        if len(self) != len(other):
            return False
        else:
            # Standard Library Imports
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

    def __radd__(self, y):
        """Reverse addition to make sum() work."""
        if isinstance(y, int):
            return self
        return self.__add__(y)

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

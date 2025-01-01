# -*- coding: utf-8 -*-
# encodermap/trajinfo/info_all.py
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
    * Better grasp of equilibrium and off-equilibrium dynamics.


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
import json
import operator
import os
import re
import warnings
from collections.abc import Callable, Iterator, KeysView, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, overload

# Third Party Imports
import numpy as np
import tables
from optional_imports import _optional_import
from tqdm import tqdm as normal_tqdm_
from tqdm.notebook import tqdm as notebook_tqdm_

# Encodermap imports
from encodermap._typing import CanBeIndex, CustomAAsDict
from encodermap.misc.misc import (
    _TOPOLOGY_EXTS,
    _is_notebook,
    get_full_common_str_and_ref,
)
from encodermap.misc.xarray_save_wrong_hdf5 import save_netcdf_alongside_mdtraj
from encodermap.trajinfo.trajinfo_utils import CustomTopology, TrajEnsembleFeatureType


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:  # pragma: no cover
    # Third Party Imports
    import mdtraj as md
    import pandas as pd
    import tensorflow as tf
    import xarray as xr

    # Encodermap imports
    from encodermap.trajinfo.info_single import SingleTraj


string_types = (str,)


################################################################################
# Optional Imports
################################################################################


md = _optional_import("mdtraj")
pd = _optional_import("pandas")
xr = _optional_import("xarray")
HDF5TrajectoryFile = _optional_import("mdtraj", "formats.hdf5")
h5py = _optional_import("h5py")

################################################################################
# Globals
################################################################################


__all__: list[str] = ["TrajEnsemble"]


################################################################################
# Utils
################################################################################


class notebook_tqdm(notebook_tqdm_):
    def __init__(self, *args, **kwargs):
        kwargs.pop("function", None)
        super().__init__(*args, **kwargs)

    def reset(self, total=None, **kwargs):
        self.total = total
        self.refresh()

    def update(self, n=1, **kwargs):
        kwargs.pop("function", None)
        super().update(n=n)


class normal_tqdm(normal_tqdm_):
    def __init__(self, *args, **kwargs):
        self._calls = {}
        function = kwargs.pop("function")
        super().__init__(*args, **kwargs)
        if function not in self._calls:
            self._calls[function] = {
                "update_calls": 0,
                "total": self.total,
            }
        self.print = os.getenv("ENCODERMAP_PRINT_PROG_UPDATES", "False") == "True"
        if self.print:
            print("INSTANTIATION")
            self.debug_print()

    def debug_print(self):
        print(f"Progbar {id(self)}")
        for function, data in self._calls.items():
            print(
                f"{function:<15} total: {data['total']:>3} n: {data['update_calls']:>3}"
            )
        print("\n")

    def update(self, n=1, **kwargs):
        function = kwargs.pop("function", None)
        if function is not None:
            if function not in self._calls:
                self._calls[function] = {
                    "update_calls": 0,
                    "total": 0,
                }
            if self.print:
                print(f"BEFORE UPDATE ({function})")
                self.debug_print()
        super().update(n)
        if function is not None:
            self._calls[function]["update_calls"] += 1
        if self.print and function is not None:
            print(f"AFTER  UPDATE ({function})")
            self.debug_print()

    def reset(self, total=None, **kwargs):
        assert total > self.total
        function = kwargs.pop("function", None)
        if function is not None:
            if function not in self._calls:
                self._calls[function] = {
                    "update_calls": 0,
                    "total": total - self.total,
                }
            else:
                self._calls[function]["total"] += total - self.total
            if self.print:
                print(f"BEFORE RESET ({function})")
                self.debug_print()
        self.total = total
        self.refresh()
        if self.print and function is not None:
            print(f"AFTER  RESET ({function})")
            self.debug_print()


@contextlib.contextmanager
def temp_seed(seed: int) -> Iterator[None]:
    """Temporarily set a numpy seed in a context manager.

    Args:
        seed (int): The seed.

    Examples:
        >>> from encodermap.trajinfo.info_all import temp_seed
        >>> import numpy as np
        >>> with temp_seed(123456789):
        ...     print(np.random.randint(low=0, high=10, size=(5, )))
        [8 2 9 7 4]

    """
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
    def keys(self) -> KeysView:
        if not self._open:
            raise Exception(f"Can't view keys of closed HDF5 file.")
        nodes = [n.name for n in self._handle.list_nodes("/")]
        return KeysView(nodes)

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
            self._get_node("/", name=f"lambda_{group_id}").attrs["units"] = (
                "dimensionless"
            )

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
        """Writes the topology into the group_id.

        Args:
            group_id (str): The name of the group. Normally 'topology' is
                used for single traj HDF5 files. Can also be 'topology_<traj_num>',
                where <traj_num> is the traj_num of a trajectory.
            topology_object (md.Topology): The topology to put into the group.

        """
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
            traj_nums.extend(re.findall(r"\d+", node))
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
            if np.asarray(item).ndim == 1:
                idx = np.where(np.in1d(items, np.asarray(item)))[0]
            else:
                return self.other._pyemma_indexing_tsel(item)
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
    """A fancy list of single trajectories. Topologies can be different across trajs.

    Check out http://statisticalbiophysicsblog.org/?p=92 for why trajectory ensembles are awesome.

    This class is a fancy list of :obj:`encodermap.trajinfo.info_single.SingleTraj``.
    Trajectories can have different topologies and will be grouped by
    the ``common_str`` argument. Each trajectory has its own unique ``traj_num``,
    which identifies it in the ensemble - even when the ensemble is sliced or
    subsampled.

    Examples:
        >>> import encodermap as em
        >>> traj1 = em.SingleTraj.from_pdb_id("1YUG")
        >>> traj2 = em.SingleTraj.from_pdb_id("1YUF")

        Addition of two :obj:`encodermap.trajinfo.info_single.SingleTraj` also creates an ensemble.

        >>> trajs = traj1 + traj2
        >>> trajs  # doctest: +ELLIPSIS
        <encodermap.TrajEnsemble object. Current backend is no_load. Containing 2 trajectories. Common str is ['1YUG', '1YUF']. Not containing any CVs...>

    Indexing a :obj:`TrajEnsemble` returns a :obj:`encodermap.trajinfo.info_single.SingleTraj`
    based on its 0-based index. Think of the :obj:`TrajEnsmeble` as a list of
    :obj:`encodermap.trajinfo.info_single.SingleTraj`. But trajectories can also
    have ``traj_nums``, which do not have to adhere to ``[0, 1, 2, ...]``. This
    is similar to how a :obj:`pandas.DataFrame` offers indexing via ``.loc[]``
    and ``.iloc[]`` (https://pandas.pydata.org/docs/user_guide/indexing.html#different-choices-for-indexing).
    For indexing trajs based on their ``traj_num``, you can use the ``.tsel[]``
    accessor of the :obj:`TrajEnsmeble`

    Examples:
        >>> import encodermap as em
        >>> traj1 = em.SingleTraj.from_pdb_id("1YUG")
        >>> traj2 = em.SingleTraj.from_pdb_id("1YUF")

        Addition of two `SingleTraj` also creates an ensemble.

        >>> trajs = traj1 + traj2
        >>> trajs.traj_nums
        [0, 1]

        Change the ``traj_num`` of ``traj2``

        >>> trajs[1].traj_num = 4
        >>> trajs.traj_nums
        [0, 4]
        >>> trajs[1]  # doctest: +ELLIPSIS
        <encodermap.SingleTraj object. Currently not in memory. Basename is '1YUF'. Not containing any CVs. Common string is '1YUF'. Object at ...>
        >>> trajs.tsel[4]  # doctest: +ELLIPSIS
        <encodermap.SingleTraj object. Currently not in memory. Basename is '1YUF'. Not containing any CVs. Common string is '1YUF'. Object at ...>

    :obj:`TrajEnsemble` supports fancy indexing. You can slice to your liking
    (``trajs[::5]`` returns a :obj:`TrajEnsemble` object that only consideres
    every fifth frame). Besides indexing by slices and integers, you can pass a
    2-dimensional :obj:`numpy.ndarray`. ``np.array([[0, 5], [1, 10], [5, 20]])``
    will return a :obj:`TrajEnsemble` object with frame 5 of trajectory 0, frame
    10 of trajectory 1 and frame 20 of trajectory 5.

    Examples:
        >>> import encodermap as em
        >>> traj1 = em.SingleTraj.from_pdb_id("1YUG")
        >>> traj2 = em.SingleTraj.from_pdb_id("1YUF")
        >>> trajs = traj1 + traj2
        >>> sel = trajs[[[0, 0], [0, 1], [0, 2], [1, 10]]]
        >>> sel  # doctest: +ELLIPSIS
        <encodermap.TrajEnsemble object. Current backend is no_load. Containing 4 frames and 2 trajectories. Common str is...>


    The :obj:`TrajEnsemble` class also is an iterator to iterate over trajectores.
    Besides plain iteration, the :obj:`TrajEnsmeble` also offers alternate iterators.
    The ``itertrajs()`` iterator returns a two-tuple of ``traj_num`` and ``traj``.
    The ``iterframes()`` iterator returns a three-tuple of ``traj_num``,
    ``frame_num``, and ``traj``.

    Examples:
        >>> import encodermap as em
        >>> traj1 = em.SingleTraj.from_pdb_id("1YUG")
        >>> traj2 = em.SingleTraj.from_pdb_id("1YUF")
        >>> trajs = traj1 + traj2
        >>> trajs[1].traj_num = 4
        >>> for traj_num, traj in trajs.itertrajs():
        ...     print(traj_num, traj.n_frames)
        0 15
        4 16
        >>> for traj_num, frame_num ,traj in trajs.subsample(10).iterframes():
        ...     print(traj_num, frame_num, traj.n_frames)
        0 0 1
        0 10 1
        4 0 1
        4 10 1

    The :obj:`TrajEnsemble` has multiple alternative constructors. The
    :meth:`with_overwrite_trajnums` constructor fixes inhomogeneous sequences of
    :obj:`encodermap.trajinfo.info_single.SingleTraj` and :obj:`TrajEnsemble`.

    Examples:
        >>> import encodermap as em
        >>> traj1 = em.SingleTraj.from_pdb_id("1YUG", traj_num=0)
        >>> traj2 = em.SingleTraj.from_pdb_id("1YUF", traj_num=0)
        >>> trajs = em.TrajEnsemble([traj1, traj2])  # doctest: +IGNORE_EXCEPTION_DETAIL, +ELLIPSIS, +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        Exception: The `traj_num` attributes of the provided 2 `SingleTraj`s is not unique, the `traj_num` 0 occurs 2 times. This can happen, if you use `SingleTraj`s, that are already part of a `TrajEnsemble`. To create copies of the `SingleTraj`s and overwrite their `traj_num`s, use the `with_overwrite_trajnums()` constructor.
        >>> trajs = em.TrajEnsemble.with_overwrite_trajnums(traj1, traj2)
        >>> trajs  # doctest: +ELLIPSIS
        <encodermap.TrajEnsemble...>

    The :meth:`from_dataset` constructor can be used to load an ensemble from
    an ``.h5`` file

    Examples:
        >>> import encodermap as em
        >>> from tempfile import TemporaryDirectory
        >>> traj1 = em.SingleTraj.from_pdb_id("1YUG")
        >>> traj2 = em.SingleTraj.from_pdb_id("1YUF")
        >>> trajs = em.TrajEnsemble([traj1, traj2])
        >>> with TemporaryDirectory() as td:
        ...     trajs.save(td + "/trajs.h5")
        ...     new = em.TrajEnsemble.from_dataset(td + "/trajs.h5")
        ...     print(new)  # doctest: +ELLIPSIS
        encodermap.TrajEnsemble object. Current backend is no_load. Containing 2 trajectories. Common str is...Not containing any CVs.

    Attributes:
        CVs (dict[str, np.ndarray]): The collective variables of the ``SingleTraj``
            classes. Only CVs with matching names in all ``SingleTraj`` classes
            are returned. The data is stacked along a hypothetical time axis
            along the trajs.
        _CVs (xarray.Dataset): The same data as in CVs but with labels.
            Additionally, the xarray is not stacked along the time axis.
            It contains an extra dimension for trajectories.
        n_trajs (int): Number of individual trajectories in this class.
        n_frames (int): Number of frames, sum over all trajectories.
        locations (list[str]): A list with the locations of the trajectories.
        top (list[mdtraj.Topology]): A list with the reference pdb for each trajecotry.
        basenames (list[str]): A list with the names of the trajecotries.
            The leading path and the file extension is omitted.
        name_arr (np.ndarray): An array with ``len(name_arr) == n_frames``.
            This array keeps track of each frame in this object by identifying each
            frame with a filename. This can be useful, when frames are mixed inside
            a :obj:`TrajEnsemble` class.

    """

    def __init__(
        self,
        trajs: Union[
            Sequence[str],
            Sequence[Path],
            Sequence[md.Trajectory],
            Sequence[SingleTraj],
        ],
        tops: Union[None, Sequence[str], Sequence[Path]] = None,
        backend: Literal["mdtraj", "no_load"] = "no_load",
        common_str: Optional[Sequence[str]] = None,
        basename_fn: Optional[Callable[[str], str]] = None,
        traj_nums: Optional[Sequence[int]] = None,
        custom_top: Optional[CustomAAsDict] = None,
    ) -> None:
        """Instantiate the :obj:`TrajEnsmeble` class with two lists of files.

        Args:
            trajs (Union[Sequence[str], Sequence[md.Trajectory],
                Sequence[SingleTraj], Sequence[Path]]): List of strings with
                paths to trajectories. Can also be a list of md.Trajectory or
                em.SingleTraj.
            tops (Optional[list[str]]): List of strings with paths to reference pdbs.
            backend (str, optional): Choose the backend to load trajectories:
                    - 'mdtraj' uses mdtraj, which loads all trajectories into RAM.
                    - 'no_load' creates an empty trajectory object.
                Defaults to 'no_load', which makes the instantiation of large
                ensembles fast and RAM efficient.
            common_str (list[str], optional): If you want to include trajectories with
                different topology. The common string is used to pair traj-files
                (``.xtc, .dcd, .lammpstrj, ...``) with their topology
                (``.pdb, .gro, ...``). The common-string should be a substring
                of matching traj and topology files.
            basename_fn (Union[None, Callable[[str], str], optional): A function
                to apply to the trajectory file path string to return the basename
                of the trajectory. If None is provided, the filename without
                extension will be used. When all files are named the same and
                the folder they're in defines the name of the trajectory, you
                can supply ``lambda x: split('/')[-2]`` as this argument.
                Defaults to None.
            custom_top: Optional[CustomAAsDict]: An instance of the
                :obj:`encodermap.trajinfo.trajinfo_utils.CustomTopology` or a
                dictionary that can be made into such.

        """
        # defaults
        # Local Folder Imports
        from .info_single import SingleTraj

        # check if h5file might be a complete dataset
        if isinstance(trajs, (str, Path)):
            if Path(trajs).suffix == ".h5":
                return TrajEnsemble.from_dataset(trajs, basename_fn=basename_fn)

        if tops == []:
            raise Exception(
                f"Your list of topology files is empty: {tops=}. Pass None, if "
                f"your trajectories are all .pdb/.gro files."
            )

        self.backend = backend

        # custom topology to load dihedral angles
        self._custom_top = custom_top

        # set the trajnums
        if traj_nums is not None:
            # Standard Library Imports
            from copy import deepcopy

            if not len(traj_nums) == len(trajs):
                raise Exception(
                    f"Uneven length of `traj_nums` ({len(traj_nums)} "
                    f"and `trajs` ({len(trajs)}) provided."
                )
            if all([isinstance(t, SingleTraj) for t in trajs]):
                trajs_ = []
                for n, t in zip(traj_nums, trajs):
                    t = deepcopy(t)
                    t.traj_num = n
                    trajs_.append(t)
                trajs = trajs_

        # make sure, that traj_nums are not duplicated
        elif all([isinstance(t, SingleTraj) for t in trajs]) and isinstance(
            trajs, (list, tuple)
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
                        f"The `traj_num` attributes of the provided {len(trajs)} `SingleTraj`s are "
                        f"not unique, the `traj_num` {ex_num} occurs {np.max(counts)} times. "
                        f"This can happen, if you use `SingleTraj`s, that are already part of "
                        f"a `TrajEnsemble`. To create copies of the `SingleTraj`s and over"
                        f"write their `traj_num`s, use the `with_overwrite_trajnums()` constructor."
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

        # set the traj nums afterwards
        if traj_nums is not None:
            for i, traj in zip(traj_nums, self.trajs):
                traj.traj_num = i

    @classmethod
    def with_overwrite_trajnums(
        cls,
        *trajs: Union[TrajEnsemble, "SingleTraj"],
    ) -> TrajEnsemble:
        """Creates a :obj:`TrajEnsemble` by copying the provided
        :obj:`encodermap.trajinfo.info_single.SingleTraj` instances and
        changing their ``traj_num`` attribute to adhere to ``[0, 1, 2, ...]``.

        Args:
            trajs (Sequence[SingleTraj]): The sequence of trajs.

        Returns:
            TrajEnsemble: A :obj:`TrajEnsemble` instance.

        """
        # Standard Library Imports
        from copy import deepcopy

        # Local Folder Imports
        from .info_single import SingleTraj

        new_trajs = []
        i = 0
        for t in trajs:
            if isinstance(t, SingleTraj):
                t = deepcopy(t)
                t.traj_num = i
                new_trajs.append(t)
                i += 1
            elif isinstance(t, TrajEnsemble):
                for traj in t:
                    traj = deepcopy(traj)
                    traj.traj_num = i
                    new_trajs.append(traj)
                    i += 1
            else:
                raise TypeError(
                    f"Classmethod `with_overwrite_trajnums` can only accept `SingleTraj` "
                    f"and `TrajEnsemble`, but {t} is not an instance of either."
                )
        return cls(new_trajs)

    @classmethod
    def from_textfile(
        cls,
        fname: Union[str, Path],
        basename_fn: Optional[Callable[[str], str]] = None,
    ) -> TrajEnsemble:
        """Creates a :obj:`TrajEnsemble` object from a textfile.

        The textfile needs to be space-separated with two or three columns:
            - Column 1:
                The trajectory file.
            - Column 2:
                The corresponding topology file (If you are using ``.h5`` trajs,
                column 1 and 2 will be identical, but column 2 needs to be there
                nonetheless).
            - Column 3:
                The common string of the trajectory. This column can be left
                out, which will result in an :obj:`TrajEnsemble` without common
                strings.

        Args:
            fname (Union[str, Path]): File to be read.
            basename_fn (Union[None, Callable[[str], str]], optional): A function
                to apply to the ``traj_file`` string to return the basename of
                the trajectory. If None is provided, the filename without
                extension will be used.  When all files are named the same and
                the folder they're in defines the name of the trajectory, you
                can supply ``lambda x: split('/')[-2]`` as this argument.
                Defaults to None.

        Returns:
            TrajEnsemble: A :obj:`TrajEnsemble` instance.

        """
        # Local Folder Imports
        from ..trajinfo import info_single

        traj_files = []
        top_files = []
        common_str = []
        traj_nums = []

        with open(fname, "r") as f:
            for row in f:
                traj_files.append(row.split()[0])
                top_files.append(row.split()[1])
                try:
                    traj_nums.append(int(row.split()[2]))
                except ValueError:
                    traj_nums.append(None)
                try:
                    common_str.append(row.split()[3])
                except IndexError:
                    common_str.append("")

        trajs = []
        for i, (traj_file, top_file, cs, traj_num) in enumerate(
            zip(traj_files, top_files, common_str, traj_nums)
        ):
            trajs.append(info_single.SingleTraj(traj_file, top_file, cs, traj_num))

        return cls(
            trajs, common_str=np.unique(common_str).tolist(), basename_fn=basename_fn
        )

    @classmethod
    def from_dataset(
        cls,
        fname: Union[str, Path],
        basename_fn: Optional[Callable[[str], str]] = None,
    ) -> TrajEnsemble:
        # Local Folder Imports
        from .info_single import SingleTraj

        traj_nums = []
        with h5py.File(fname) as h5file:
            for key in h5file.keys():
                if key == "CVs":
                    continue
                traj_nums.extend(re.findall(r"\d+", key))
        traj_nums = list(sorted(map(int, set(traj_nums))))

        trajs = []
        for traj_num in traj_nums:
            trajs.append(
                SingleTraj(
                    traj=fname,
                    top=fname,
                    traj_num=traj_num,
                    basename_fn=basename_fn,
                )
            )
        common_str = list(set([t.common_str for t in trajs]))
        newclass = cls(trajs=trajs, common_str=common_str, basename_fn=basename_fn)
        return newclass

    def load_custom_topology(
        self,
        custom_top: Optional[Union[CustomTopology, CustomAAsDict]] = None,
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

    def del_featurizer(self) -> None:
        """Deletes the current instance of ``self.featurizer``."""
        if hasattr(self, "_featurizer"):
            del self._featurizer

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
        for v in out.values():
            v.common_str = list(set([t.common_str for t in v]))
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
        if any([hasattr(t, "clustal_w") for t in self.trajs]):
            raise NotImplementedError(
                f"This is currently not supported for TrajEsnembles with "
                f"clustal_w alignments."
            )
        else:
            max_residues = max([t.n_residues for t in self])

        # make sure CVs are loaded and contain the appropriate values
        should_be = {
            "central_cartesians",
            "central_dihedrals",
            "central_distances",
            "central_angles",
            "side_dihedrals",
            "side_cartesians",
            "side_distances",
            "side_angles",
        }
        diff = should_be - set(self._CVs.data_vars.keys())
        if len(diff) > 0:
            raise Exception(
                f"The TrajEnsemble misses these CVs to calculate the sidechain_info: "
                f"{list(diff)}. Please load them with `trajs.load_CVs({list(diff)})`."
            )

        # make sure we are using an ensemble with generic indices
        forbidden_names = set(
            [residue.name for traj in self for residue in traj.top.residues]
        )
        if (
            len(
                (
                    offending := [
                        label
                        for label in self._CVs.central_distances.coords[
                            "CENTRAL_DISTANCES"
                        ].values
                        if any(f in label for f in forbidden_names)
                    ]
                )
            )
            > 0
        ):
            raise Exception(
                f"The CVs in this TrajEnsemble were not loaded with the `ensemble=True` "
                f"keyword in `trajs.load_CVs()`. Finding the sidechain_info in "
                f"such a set of CVs is not possible. The offending labels {offending} "
                f"contain residue names, which should not occur if CVs were loaded "
                f"with `ensemble=True`."
            )

        # the key -1 is the feature hull, telling tensorflow
        # how to create the branched chain of backbone and sidechains
        # i.e. how many sidechains there are per residue max
        # the other keys correspond to which sidechain atoms are non nan and
        # can be used when calculating the distance matrices per different atom
        out = {-1: {}}

        # the feature hull
        max_sidechains = self._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"].values
        for residx in range(1, max_residues + 1):
            labels = [l for l in max_sidechains if l.endswith(" " + str(residx))]
            out[-1][residx] = len(labels)

        # for every traj
        for traj in self.trajs:
            assert (
                traj.traj_num not in out
            ), f"This traj has the same `traj_num` as another traj."
            out[traj.traj_num] = {}
            for residx in range(1, max_residues + 1):
                labels = [
                    l
                    for l in traj._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"].values
                    if l.endswith(" " + str(residx))
                ]
                first_frame = traj._CVs.coords["frame_num"].min()
                length = (
                    traj._CVs.side_dihedrals.sel(
                        frame_num=first_frame, SIDE_DIHEDRALS=labels
                    )
                    .dropna(dim="SIDE_DIHEDRALS")
                    .sizes["SIDE_DIHEDRALS"]
                )
                out[traj.traj_num][residx] = length
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

    def to_alignment_query(self) -> str:
        """A string, that cen be put into sequence alignment software."""
        for cs, trajs in self.trajs_by_common_str.items():
            assert len(trajs.top) == 1, (
                f"Can't provide a query for a `TrajEnsemble`, where a common string "
                f"has multiple topologies. In this case, the common string '{cs}' "
                f"has these topologies: {trajs.top}. When you are unhappy with how "
                f"EncoderMap automatically applies common strings to trajectory "
                f"filenames, keep in mind that you can always generate a `TrajEnsemble` "
                f"from multiple `TrajEnsembles`. You can do: \n"
                f"trajs1 = em.load(\n"
                f"  traj_files1,\n"
                f"  top_files1,\n"
                f"  common_str=['trajs1'],\n"
                f")\n"
                f"trajs2 = em.load(\n"
                f"  traj_files2,\n"
                f"  top_files2,\n"
                f"  common_str=['trajs2'],\n"
                f")\n"
                f"trajs = trajs1 + trajs2\n"
                f"to force a `TrajEnsemble` to adhere to your common strings."
            )
        out = ""
        for cs, trajs in self.trajs_by_common_str.items():
            seq = trajs.top[0].to_fasta()
            for j, s in enumerate(seq):
                add = f">{cs}n{s}\n"
        return out

    def parse_clustal_w_alignment(self, aln: str) -> None:
        """Parse an alignment in ClustalW format and add the info to the trajectories.

        Args:
            aln (str): The alignment in ClustalW format.

        """
        self.clustal_w = aln
        # remove empty lines
        aln = "\n".join(
            list(filter(lambda x: not re.match(r"^\s*$", x), aln.splitlines()))
        )

        # every three lines represent one trajectory
        lines = aln.splitlines()
        n_lines = len(lines)
        assert (
            n_lines % (self.n_trajs + 1) == 0
            or n_lines % (len(self.common_str) + 1) == 0
        ), (
            f"The CLUSTAL W aln string, that you provided has the wrong number of "
            f"lines. I've expected to receive a multiple of {self.n_trajs + 1} ("
            f"which is the number of trajs ({self.n_trajs}) plus one for the score-"
            f"characters ' ', '.', ':', '*'), but the number of provided lines was "
            f"{n_lines}."
        )

        if "|" in aln:
            for i, (_, sub_trajs) in enumerate(self.trajs_by_top.items()):
                for cs, trajs in sub_trajs.trajs_by_common_str.items():
                    for traj in trajs:
                        for j, chain in enumerate(traj.top.chains):
                            search = (
                                f"{cs}|TrajNum_{traj.traj_num}Topology_{i}Chain_{j}"
                            )
                            data = ""
                            for line in lines:
                                if line.startswith(search):
                                    data += line.split()[-1]
                            assert (
                                test := len(data.replace("-", ""))
                            ) == traj.n_residues, (
                                f"The CLUSTAL W sequence {data} with {test} one-letter "
                                f"residues has not the same number of residues as trajectory "
                                f"{traj}, which has {traj.n_residues}"
                            )
                            traj.clustal_w = data
        else:
            for cs, trajs in self.trajs_by_common_str.items():
                search = cs
                data = ""
                for line in lines:
                    if line.startswith(search):
                        data += line.split()[-1]
                assert (test := len(data.replace("-", ""))) == trajs[0].n_residues, (
                    f"The CLUSTAL W sequence {data} with {test} one-letter "
                    f"residues has not the same number of residues as trajectory "
                    f"{trajs[0]}, which has {trajs[0].n_residues}"
                )
                for traj in trajs:
                    traj.clustal_w = data

    def del_CVs(self, CVs: Optional[Sequence[str]] = None) -> None:
        """Deletes all CVs in all trajs. Does not affect the files."""
        if CVs is None:
            for traj in self.trajs:
                traj.del_CVs()
        else:
            if not isinstance(CVs, (list, tuple)):
                CVs = [CVs]
            remove = deepcopy(CVs)
            for CV in CVs:
                remove.append(f"{CV}_feature_indices")
            for traj in self.trajs:
                traj._CVs = traj._CVs.drop_vars(remove, errors="ignore")

    @property
    def _CVs(self) -> xr.Dataset:
        """xarray.Dataset: Returns x-array Dataset of matching CVs. stacked
        along the trajectory-axis."""
        # Local Folder Imports
        from .trajinfo_utils import trajs_combine_attrs

        ds = xr.combine_nested(
            [traj._CVs for traj in self.trajs],
            concat_dim="traj_num",
            compat="broadcast_equals",
            fill_value=np.nan,
            coords="all",
            join="outer",
            combine_attrs=trajs_combine_attrs,
        )

        # if ensemble we don't need to reorder labels. That was already done
        # by the Featurizer. We know if we have an ensemble if the trajs
        # have features with nans (except the "feature_indices" dataarrays
        if any(
            [
                np.any(np.isnan(v.values))
                for traj in self
                for n, v in traj._CVs.data_vars.items()
                if "feature_indices" not in n
            ]
        ):
            return ds

        # sort the combined arrays
        new_label_order = {}
        non_indices_data_vars = [k for k in ds.keys() if "feature_indices" not in k]
        for k in non_indices_data_vars:
            if (ind_k := f"{k}_feature_indices") not in ds:
                continue
            argsort = []
            da = ds[ind_k]

            # ResidueMinDistanceFeature
            if "RES_NO" in da.coords:
                continue

            try:
                feature_axis_name = da.attrs["feature_axis"]
            except KeyError as e:
                raise Exception(f"There is no feature_axis attribute in {da=}") from e
            labels = da.coords[feature_axis_name].values
            if "ATOM_NO" not in da.coords and "frame_num" not in da.coords:
                iterable = da.values.T
            elif "ATOM_NO" not in da.coords and "frame_num" in da.coords:
                iterable = (
                    da.stack({"frame": ("traj_num", "frame_num")})
                    .transpose("frame", ...)
                    .dropna("frame", how="all")
                )
                iterable = iterable.transpose(feature_axis_name, "frame").values
            elif "ATOM_NO" in da.coords and "frame_num" not in da.coords:
                iterable = da.transpose(feature_axis_name, "traj_num", "ATOM_NO").values
            else:
                iterable = (
                    da.stack({"frame": ("traj_num", "frame_num")})
                    .transpose("frame", ...)
                    .dropna("frame", how="all")
                )
                iterable = iterable.transpose(feature_axis_name, "frame", "ATOM_NO")
            for i, (row, label) in enumerate(zip(iterable, labels)):
                if (
                    "ATOM_NO" not in da.coords and "RES_NO" not in da.coords
                ):  # selection feature
                    row = row[~np.isnan(row)]
                    uniques, counts = np.unique(row, return_counts=True)
                    ind = uniques[np.argmax(counts)]
                    argsort.append(ind)
                else:
                    row = np.nan_to_num(row + 1, 0).sum(1)
                    row = row[np.nonzero(row)]
                    uniques, counts = np.unique(row, return_counts=True)
                    if "side" in feature_axis_name:
                        raise Exception(f"{new_label_order=}")
                    try:
                        ind = uniques[np.argmax(counts)]
                    except ValueError as e:
                        if "attempt to get argmax of an empty" in str(e):
                            raise Exception(
                                f"Can't order the data_var {k}, as the {row=} "
                                f"has become empty ({row=})."
                            )
                        raise e
                    argsort.append(ind)
            labels = labels[np.argsort(argsort)]
            new_label_order[feature_axis_name] = labels

        ds = ds.reindex(new_label_order, fill_value=np.nan)
        return ds

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
                            f"I am not returning the CVs for the feature '{key}' "
                            f"because some trajectories have different "
                            f"shapes for these CVs. The shapes are {set(shapes)}. "
                            f"If you want to access these CVs, use the `_CVs` "
                            f"xarray dataset of `TrajEnsemble` instead."
                        )
                        continue
                    if np.all(
                        [
                            any([isinstance(ind, int) for ind in traj.index])
                            for traj in self.trajs
                        ]
                    ):
                        data = np.vstack([d for d in data])
                        if data.ndim <= 3:
                            data = data.reshape(-1, data.shape[-1])
                        else:
                            data = data.reshape(-1, *data.shape[-2:])
                    else:
                        try:
                            data = np.concatenate(
                                [d.squeeze(axis=0) for d in data], axis=0
                            )
                        except ValueError as e:
                            if "zero-dimensional" in str(e):
                                data = np.hstack([d.squeeze(axis=0) for d in data])
                            if "all the input arrays must have the same" in str(e):
                                err_shapes = "\n".join(
                                    [
                                        f"Traj: {self.trajs[i].traj_num:<3} CV '{key}' shape: {d.shape}"
                                        for i, d in enumerate(data)
                                    ]
                                )
                                raise Exception(
                                    f"Can't concatenate the data of the CV '{key}'."
                                    f"The shapes of this CV for the individual "
                                    f"trajectories are:\n"
                                    f"{err_shapes}"
                                )
                            else:
                                raise e
                    if data.shape[-1] != 1:
                        CVs[key] = data
                    else:
                        CVs[key] = data.squeeze(-1)
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

    def _traj_joined(self, progbar: Optional[Any] = None) -> md.Trajectory:
        # use traj[0] of the trajs list as the traj from which the topology will be used
        parent_traj = self.trajs[0].traj

        # join the correct number of trajs
        # by use of the `divmod` method, the frames parent_traj traj will be
        # appended for a certain amount, until the remainder of the division
        # is met by that time, the parent traj will be sliced to fill the correct number of frames
        no_of_iters, rest = divmod(self.n_frames, parent_traj.n_frames)

        total = self.n_frames + no_of_iters + 1
        if progbar is None:
            if _is_notebook():
                progbar = notebook_tqdm(
                    total=total,
                    leave=False,
                    position=0,
                    desc="Clustering...",
                )
            else:
                progbar = normal_tqdm(
                    total=total,
                    leave=False,
                    position=0,
                    desc="Clustering...",
                    function="_traj_joined",
                )
        else:
            if not isinstance(progbar, bool):
                progbar.reset(progbar.total + total, function="_traj_joined")
            else:
                progbar = None

        for i in range(no_of_iters + 1):
            if i == 0:
                dummy_traj = copy.deepcopy(parent_traj)
            elif i == no_of_iters:
                if rest != 0:
                    dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj)[:rest])
            else:
                dummy_traj = dummy_traj.join(copy.deepcopy(parent_traj))
            if progbar is not None:
                progbar.update(function="_traj_joined")

        # some checks
        assert self.n_frames == dummy_traj.n_frames
        # assert self.n_frames == len(self.trajs), f"{self.n_frames=}, {len(self.trajs)=}"

        # change the xyz coordinates of dummy_traj according to the frames in joined trajs
        for i, (_, __, traj) in enumerate(self.iterframes()):
            try:
                dummy_traj.xyz[i] = traj.xyz
            except ValueError as e:
                if "broadcast" not in str(e):
                    raise e
                warnings.warn(
                    f"This`TrajEnsemble` has {len(self.top)} unique topologies. "
                    f"I will use the topology with {self.top[0].n_atoms} for joining "
                    f"and discard atoms in the other trajectories."
                )
                dummy_traj.xyz[i] = traj.xyz[0, : dummy_traj.n_atoms]
            if progbar is not None:
                progbar.update(function="_traj_joined")

        return dummy_traj

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
            <mdtraj.Trajectory with 40 frames, 69 atoms, 6 residues, and unitcells at ...>

        """
        return self._traj_joined()

    @property
    def xyz(self) -> np.ndarray:
        """np.ndarray: xyz coordinates of all atoms stacked along the traj-time axis.

        Only works if all trajs share the same topology.

        """
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
        for traj_num, frame_num, frame in self.iterframes():
            frames.append(frame)
        out = TrajEnsemble.with_overwrite_trajnums(*frames)
        assert out.trajs != []
        if inplace:
            self = out
        else:
            return out

    def save_CVs(self, path: Union[str, Path]) -> None:
        """Saves the CVs to a NETCDF file using xarray."""
        self._CVs.to_netcdf(path, format="NETCDF4", engine="h5netcdf")

    def cluster(
        self,
        cluster_id: int,
        col: str = "cluster_membership",
        memberships: Optional[np.ndarray] = None,
        n_points: int = -1,
        overwrite: bool = True,
    ) -> TrajEnsemble:
        """Clusters this :obj:`TrajEnsemble` based on the provided
        ``cluster_id`` and ``col``.

        With 'clustering' we mean to extract a subset given a certain membership.
        Take two trajectories with 3 frames each as an ensemble. Let's say we
        calculate the end-to-end distance of the trajectories and use it as
        a collective variable of the system. The values are
        ``[0.8, 1.3, 1.2, 1.9, 0.2, 1.3]``. Based on these values, we define a
        boolean CV (using 0 as False and 1 as True) which says whether the
        end-to-end distance is smaller or grather than 1.0. We give this CV the
        name ``'end_to_end_binary'`` and the values are ``[0, 1, 1, 1, 0, 1]``.
        We can use this CV to 'cluster' the :obj:`TrajEnsemble` via:
            - ``cluster = trajs.cluster(cluster_id=0, col='end_to_end_binary')``:
                This gives a :obj:`TrajEnsemble` with 2 frames.
            - ``cluster = trajs.cluster(cluster_id=0, col='end_to_end_binary')``:
                This gives a :obj:`TrajEnsemble` with 4 frames.
        Sometimes, you want to save this a cluster in a format that can be rendered
        by graphical programs (``.xtc, .pdb``), you can use either the :meth:`join` or
        :meth:`stack` method of the resulting :obj:``TrajEnsemble` to get a
        `mdtraj.Trajectory`, which is either stacked along the atom axis or
        joined along the time axis.

        Note:
            If the resulting :obj:`TrajEnsemble` has inhomogeneous topologies, the
            :meth:`join` method will return a dict[md.Topology, md.Trajectory]
            instead. This dict can be used to save multiple (``.xtc, .pdb``) files
            and visualize your cluster in external programs.

        The ``col`` parameter takes any CV name, that is per-frame and integer.

        Args:
            cluster_id (int): The cluster id to use. Needs to be an integer,
                that is present in the ``col`` parameter.
            col (str): Which 'column' of the collective variables to use.
                Needs to be a key, that can be found in ``trajs.CVs.keys()``.
            memberships (Optional[np.ndarray]): If a :obj:`numpy.ndarray` is
                provided here, the memberships from this array will be used.
                In this case, the ``col`` argument will be unused.
            n_points (int): How many points the resulting cluster should contain.
                Subsamples the points in ``col == cluster_id`` evenly and without
                repeat. If set to -1, all points will be used.
            overwrite (bool): When the ``memberships`` argument is used, but the
                :obj:`TrajEnsemble` already has a CV under the name specified by
                ``col``, you can set this to True to overwrite this column. Can
                be helpful, when you iteratively conduct multiple clusterings.

        Examples:

            Import EncoderMap and NumPy.

            >>> import encodermap as em
            >>> import numpy as np

            Load an example project.

            >>> trajs = em.load_project("pASP_pGLU", load_autoencoder=False)

            Create an array full of ``-1``'s. These are the 'outliers'.

            >>> cluster_membership = np.ones(shape=(trajs.n_frames, )) * -1

            Select the first 5 frames of every traj to be in cluster 0.

            >>> cluster_membership[trajs.id[:, 1] < 5] = 0

            Select all frames between 50 and 55 to be cluster 1.

            >>> cluster_membership[(50 <= trajs.id[:, 1]) & (trajs.id[:, 1] <= 55)] = 1
            >>> np.unique(cluster_membership)
            array([-1.,  0.,  1.])

            Load this array as a CV called ``'clu_mem'``.

            >>> trajs.load_CVs(cluster_membership, attr_name='clu_mem')

            Extract all of cluster 0 with ``n_points=-1``.

            >>> clu0 = trajs.cluster(0, "clu_mem")
            >>> clu0.n_frames
            35

            Extract an evenly spaced subset of cluster 1 with 10 total points.

            >>> clu1 = trajs.cluster(1, "clu_mem", n_points=10)
            >>> clu1.n_frames
            10

            Cclusters with inhomogeneous topologies can be stacked along the atom axis.

            >>> [t.n_atoms for t in trajs]
            [69, 83, 103, 91, 80, 63, 73]
            >>> stacked = clu1.stack()
            >>> stacked.n_atoms
            795

            But joining the trajectories returns a ``dict[top, traj]`` if the
            topologies are inhomogeneous.

            >>> joined = clu1.join()
            >>> type(joined)
            <class 'dict'>

        """
        if memberships is not None:
            if not overwrite:
                assert col not in self._CVs, (
                    f"Can't load {memberships} as new CVs. "
                    f"The CV {col} containing cluster memberships already exists. "
                    f"Choose a different name for the argument `col`."
                )
            self.load_CVs(memberships, col, override=overwrite)

        assert (
            col in self._CVs
        ), f"To use the CV '{col}' for clustering, add it to the CVs with `load_CVs`."

        # find the index
        index_ = (self.CVs[col] == cluster_id).squeeze()
        index = self.index_arr[index_]
        frame_index = np.arange(self.n_frames)[index_]
        assert index.size > 0, (
            f"The `cluster_id` {cluster_id} is not present in the `col` {col}: "
            f"{np.unique(self.CVs[col])=}"
        )
        if n_points > 0:
            ind = np.unique(
                np.round(np.linspace(0, len(index) - 1, n_points)).astype(int)
            )
            index = index[ind]
            frame_index = frame_index[ind]
        try:
            out = self[index]
        except IndexError as e:
            out = self._return_frames_by_index(frame_index)
        if hasattr(self, "clustal_w"):
            out.parse_clustal_w_alignment(self.clustal_w)
        return out

    def join(
        self,
        align_string: str = "name CA",
        superpose: bool = True,
        ref_align_string: str = "name CA",
        base_traj: Optional[md.Trajectory] = None,
        progbar: Optional[Any] = None,
    ) -> dict[md.Topology, md.Trajectory]:
        if len(self.top) > 1 and superpose:
            assert align_string == ref_align_string == "name CA", (
                f"Aligning different topologies only possible, when the `align"
                f"_string` and `ref_align_string` both are 'name CA'."
            )
        if progbar is None:
            if _is_notebook():
                progbar = notebook_tqdm(
                    total=len(self.top),
                    leave=False,
                    position=0,
                    desc="Joining...",
                )
            else:
                progbar = normal_tqdm(
                    total=len(self.top),
                    leave=False,
                    position=0,
                    desc="Joining...",
                    function="join",
                )
        else:
            if not isinstance(progbar, bool):
                progbar.reset(progbar.total + len(self.top), function="join")
            else:
                progbar = None

        all_trajs = []
        out_by_top = {}
        for i, (top, traj) in enumerate(self.trajs_by_top.items()):
            traj = traj._traj_joined(progbar=progbar)
            if superpose:
                if base_traj is not None:
                    CAs_traj = traj.top.select(align_string)
                    CAs_ref = base_traj.top.select(ref_align_string)
                    if hasattr(self, "clustal_w"):
                        new_CAs_traj = []
                        new_CAs_ref = []
                        i_t = 0
                        i_r = 0
                        for aln_t, aln_r in zip(
                            self.trajs_by_top[traj.top][0].clustal_w,
                            self.trajs_by_top[base_traj.top][0].clustal_w,
                        ):
                            if aln_t == aln_r == "-":
                                pass
                            elif aln_t != "-" and aln_r == "-":
                                i_t += 1
                            elif aln_t == "-" and aln_r != "-":
                                i_r += 1
                            else:
                                new_CAs_traj.append(CAs_traj[i_t])
                                new_CAs_ref.append(CAs_ref[i_r])
                                i_t += 1
                                i_r += 1
                        new_CAs_traj = np.array(new_CAs_traj)
                        new_CAs_ref = np.array(new_CAs_ref)
                        CAs_traj = new_CAs_traj.copy()
                        CAs_ref = new_CAs_ref.copy()
                    else:
                        length = min(len(CAs_traj), len(CAs_ref))
                        CAs_traj = CAs_traj[:length]
                        CAs_ref = CAs_ref[:length]
                    traj = traj.superpose(
                        base_traj,
                        atom_indices=CAs_traj,
                        ref_atom_indices=CAs_ref,
                    )
                else:
                    traj = traj.superpose(
                        traj,
                        atom_indices=traj.top.select(align_string),
                    )
                    if i == 0:
                        base_traj = traj[0]
            all_trajs.append(traj)
            out_by_top[top] = traj
            if progbar is not None:
                progbar.update(function="join")

        # return
        return out_by_top

    def stack(
        self,
        align_string: str = "name CA",
        superpose: bool = True,
        ref_align_string: str = "name CA",
        base_traj: Optional[md.Trajectory] = None,
        progbar: Optional[Any] = None,
    ) -> md.Trajectory:
        if progbar is None:
            if _is_notebook():
                progbar = notebook_tqdm(
                    total=self.n_frames,
                    leave=False,
                    position=0,
                    desc="Stacking...",
                )
            else:
                progbar = normal_tqdm(
                    total=self.n_frames,
                    leave=False,
                    position=0,
                    desc="Stacking...",
                    function="stack",
                )
        else:
            if not isinstance(progbar, bool):
                progbar.reset(progbar.total + self.n_frames, function="stack")
            else:
                progbar = None

        all_trajs = self.join(
            align_string,
            superpose,
            ref_align_string,
            base_traj,
            progbar=progbar,
        )
        atoms = 0
        # stack
        for i, traj in enumerate(all_trajs.values()):
            for j, frame in enumerate(traj):
                atoms += frame.n_atoms
                if i == 0 and j == 0:
                    stacked = deepcopy(frame)
                else:
                    stacked = stacked.stack(frame)
                if progbar is not None:
                    progbar.update(function="stack")
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

    def to_dataframe(self, CV: Union[str, Sequence[str]]) -> pd.DataFrame:
        # frame nums can be inhomogeneous
        frame_num = []
        for traj in self:
            if (_ := traj.id).ndim == 2:
                frame_num.extend(_[:, 1])
            else:
                frame_num.extend(_)
        time = []
        for traj in self:
            time.extend(traj.time)

        # the CV data can be directly extracted from xarray
        if isinstance(CV, str):
            data = [
                self._CVs[CV]
                .stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .to_pandas()
            ]
        else:
            data = []
            for cv in CV:
                df = (
                    self._CVs[cv]
                    .stack({"frame": ("traj_num", "frame_num")})
                    .transpose("frame", ...)
                    .dropna("frame", how="all")
                    .to_pandas()
                )
                if len(df.columns) == 1:
                    df = df.rename(columns={0: cv.upper()})
                data.append(df)

        df = pd.DataFrame(
            {
                "traj_file": [
                    traj.traj_file for traj in self for i in range(traj.n_frames)
                ],
                "top_file": [
                    traj.top_file for traj in self for i in range(traj.n_frames)
                ],
                "traj_num": [
                    traj.traj_num for traj in self for i in range(traj.n_frames)
                ],
                "frame_num": frame_num,
                "time": time,
            },
        )
        df = df.set_index(["traj_num", "frame_num"])
        return pd.concat([df, *data], axis=1)

    def dash_summary(self) -> pd.DataFrame:
        """A :obj:`pandas.DataFrame` that summarizes this ensemble.

        Returns:
            pd.DataFrame: The DataFrame.

        """
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
                dt.extend(np.unique(t.time[1:] - t.time[:-1]))
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
                        "common_str",
                    ],
                    "value": [
                        self.n_trajs,
                        self.n_frames,
                        n_atoms,
                        dt,
                        traj_files,
                        multiple_tops,
                        list(set(self.common_str)),
                    ],
                }
            )
            return df.astype(str)

    def load_CVs(
        self,
        data: TrajEnsembleFeatureType = None,
        attr_name: Optional[str] = None,
        cols: Optional[list[int]] = None,
        deg: Optional[bool] = None,
        periodic: bool = True,
        labels: Optional[list[str]] = None,
        directory: Optional[Union[str, Path]] = None,
        ensemble: bool = False,
        override: bool = False,
        custom_aas: Optional[CustomAAsDict] = None,
        alignment: Optional[str] = None,
    ) -> None:
        """Loads CVs in various ways. The easiest way is to provide a single
        :obj:`numpy.ndarray` and a name for that array.

        Besides np.ndarray, files (``.txt and .npy``) can be loaded. Features
        or Featurizers can be provided. A :obj:`xarray.Dataset` can be provided.
        A str can be provided which either is the name of one of EncoderMap's
        features (`encodermap.features`) or the string can be 'all',
        which loads all features required for EncoderMap's
        :obj:`encodermap.autoencoder.autoencoder`AngleDihedralCartesianEncoderMap`.

        Args:
            data (Optional[TrajEnsembleFeatureType]): The CV to
                load. When a :obj:`numpy.ndarray` is provided, it needs to have
                a shape matching ``n_frames`` and the data will be distributed
                to the trajs, When a list of files is provided, ``len(data)``
                (the files) needs to match ``n_trajs``. The first file will be
                loaded by the first traj (based on the traj's ``traj_num``) and
                so on. If a list of :obj:`numpy.ndarray` is provided, the first
                array will be assigned to the first traj (based on the traj's
                ``traj_num``). If None is provided, the argument ``directory``
                will be used to construct a str using this expression
                ``fname = directory + traj.basename + '_' + attr_name``. If
                there are ``.txt`` or ``.npy`` files matching that string in
                the ``directory``, the CVs will be loaded from these files to
                the corresponding trajs. Defaults to None.
            attr_name (Optional[str]): The name under which the CV should
                be found in the class. Choose whatever you like. ``'highd'``,
                ``'lowd'``, ``'dists'``, etc. The CV can then be accessed via
                dot-notation: ``trajs.attr_name``. Defaults to None, in which
                case, the argument ``data`` should point to existing files.
                The ``attr_name`` will be extracted from these files.
            cols (Optional[list[int]]): A list of integers indexing the columns
                of the data to be loaded. This is useful if a file contains
                columns which are not features (i.e. an indexer or the error of
                the features. eg::

                    id   f1    f2    f1_err    f2_err
                    0    1.0   2.0   0.1       0.1
                    1    2.5   1.2   0.11      0.52

                In that case, you would want to supply ``cols=[1, 2]`` to the
                ``cols`` argument. If None is provided all columns are loaded.
                Defaults to None.
            deg (Optional[bool]): Whether to return angular CVs using degrees.
                If None or False, CVs will be in radian. Defaults to None.
            periodic (bool): Whether to use the minimum image convention to
                calculate distances/angles/dihedrals. This is generally recommended,
                when you don't clean up your trajectories and the proteins break
                over the periodic boundary conditions. However, when the protein is
                large, the distance between one site and another might be shorter
                through the periodic boundary. This can lead to wrong results
                in your distance calculations.
            labels (list[str]): A list containing the labels for the dimensions of
                the data. If you provide a :obj:`numpy.ndarray` with shape
                ``(n_trajs, n_frames, n_feat)``, this list needs to be of
                ``len(n_feat)``. An exception will be raised otherwise. If None is
                privided, the labels will be automatically generated. Defaults to None.
            directory (Optional[str]): If this argument is provided, the
                directory will be searched for ``.txt`` or ``.npy`` files which
                have the same names as the trajectories have basenames. The
                CVs will then be loaded from these files.
            ensemble (bool): Whether the trajs in this class belong to an ensemble.
                This implies that they contain either the same topology or are
                very similar (think wt, and mutant). Setting this option True will
                try to match the CVs of the trajs onto the same dataset.
                If a VAL residue has been replaced by LYS in the mutant,
                the number of sidechain dihedrals will increase. The CVs of the
                trajs with VAL will thus contain some NaN values. Defaults to False.
            override (bool): Whether to override CVs with the same name as ``attr_name``.
            custom_aas (Optional[CustomAAsDict]): You can provide non-standard
                residue definitions in this argument. See
                :obj:`encodermap.trajinfo.trajinfo_utils.CustomTopology` for
                information how to use the custom_aas argument. If set to None
                (default), only standard residue names are assumed.
            alignment (Optional[str]): If your proteins have similar but different
                sequences, you can provide a CLUSTAL W alignment as this argument
                and the featurization will align the features accordingly.

        Raises:
            TypeError: When wrong Type has been provided for data.

        """
        # Local Folder Imports
        from .trajinfo_utils import load_CVs_ensembletraj

        if data is None:
            data = self.featurizer

        if custom_aas is not None:
            self.load_custom_topology(custom_aas)

        if alignment is not None:
            if ensemble:
                self.parse_clustal_w_alignment(alignment)
            else:
                print(
                    "Providing a CLUSTAL W alignment for featurization of ensembles "
                    "of protein families, makes only sense, when `ensemble` is also "
                    "set to True. This makes EncoderMap align the features based "
                    "on their sequence alignment."
                )

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
            periodic,
            labels,
            directory,
            ensemble,
            override,
        )

    def save(
        self,
        fname: Union[str, Path],
        CVs: Union[Literal["all"], list[str], Literal[False]] = "all",
        overwrite: bool = False,
        only_top: bool = False,
    ) -> None:
        """Saves this TrajEnsemble into a single ``.h5`` file.

        Args:
            fname (Union[str, Path]): Where to save the file.
            CVs (Union[Literal["all"], list[str], Literal[False]]): Which CVs
                to alos store in the file. If set to ``'all'``, all CVs will
                be saved. Otherwise, a list[str] can be provided to only save
                specific CVs. Can also be set to False, no CVs are stored in the
                file.
            overwrite (bool): If the file exists, it is overwritten.
            only_top (bool): Only writes the trajectorie's topologies into the file.

        Raises:
            IOError: If file already exists and overwrite is not True.

        """
        # Third Party Imports
        from mdtraj.utils import in_units_of

        if any([hasattr(traj, "clustal_w") for traj in self]):
            warnings.warn(
                "Can't currently save a `TrajEnsemble` with a clustal w alignment"
            )
            return

        fname = Path(fname)
        assert (
            fname.suffix == ".h5"
        ), "We recommend the .h5 file extension for these files."
        if fname.is_file() and not overwrite and not only_top:
            raise IOError(
                f"File {fname} already exists. Set `overwrite` to True to overwrite."
            )
        if fname.is_file() and overwrite:
            fname.unlink()

        for i, traj in self.itertrajs():
            with HDF5GroupWrite(fname, "a", force_overwrite=overwrite) as f:
                if not only_top:
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
                else:
                    f.write_topology(traj.traj_num, traj.top)
        for i, traj in self.itertrajs():
            traj._custom_top.traj = traj
            traj._custom_top.to_hdf_file(fname)
            traj._common_str_to_hdf_file(fname)
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
        """Creates a :obj:`TrajEnsemble` object with the trajs specified by ``index``.

        This is a sub-method of the ``trajs[]`` indexer.

        """
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
                trajs_subset += new_traj
        trajs_subset.common_str = new_common_str
        trajs_subset.basename_fn = self.basename_fn
        return trajs_subset

    def _return_frames_by_index(self, index: Sequence[int]) -> TrajEnsemble:
        """Creates a :obj:`TrajEnsemble` object with the frames specified by ``index``."""
        new_common_str = []
        frames = []
        for traj_num, frame_num, frame in self.iterframes():
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

    def _pyemma_indexing_tsel(self, key: np.ndarray) -> TrajEnsemble:
        """Returns a new :obj:`TrajEnsemble` by giving the indices of traj and frame."""
        if key.ndim == 1:
            key = key.reshape(len(key), 1).T
        trajs = []
        for i, num in enumerate(np.unique(key[:, 0])):
            frames = key[key[:, 0] == num, 1]
            trajs.append(self.tsel[num].fsel[frames])
        return TrajEnsemble(
            trajs, basename_fn=self.basename_fn, common_str=self.common_str
        )

    def _pyemma_indexing_no_tsel(self, key: np.ndarray) -> TrajEnsemble:
        """Returns a new :obj:`TrajEnsemble` by giving the indices of traj and frame."""
        if key.ndim == 1:
            key = key.reshape(len(key), 1).T
        trajs = []
        for i, num in enumerate(np.unique(key[:, 0])):
            assert num < self.n_trajs, (
                f"Can't identify trajectory with number {num} in an ensemble "
                f"with {self.n_trajs} trajectories."
            )
            frames = key[key[:, 0] == num, 1]
            trajs.append(self.trajs[num][frames])
        return TrajEnsemble(
            trajs, basename_fn=self.basename_fn, common_str=self.common_str
        )

    def subsample(
        self,
        stride: Optional[int] = None,
        total: Optional[int] = None,
    ) -> Optional[TrajEnsemble]:
        """Returns a subset of this :obj:`TrajEnsemble` given the provided stride
        or total.

        This is a faster alternative than using the ``trajs[trajs.index_arr[::1000]]``
        when HDF5 trajs are used, because the slicing information is saved in the
        respective :obj:`encodermap.trajinfo.info_single.SingleTraj`
         and loading of single frames is faster in HDF5 formatted trajs.

        Args:
            stride (Optional[int]): Return a frame ever stride frames.
            total (Optional[int]): Return a total of evenly sampled frames.

        Returns:
            TrajEnsemble: A trajectory ensemble.

        Note:
            The result from ``subsample(1000)` `is different from
            ``trajs[trajs.index_arr[::1000]]``. With subsample every trajectory
            is sub-sampled independently. Consider a :obj:`TrajEnsemble` with two
            :obj:`encodermap.trajinfo.info_single.SingleTraj` trajectories with
            18 frames each. ``subsampled = trajs.subsample(5)`` would return a
            :obj:`TrajEnsemble` with two trajs with 3 frames each
            (``subsampled.n_frames == 6``). Whereas,
            ``subsampled = trajs[trajs.index_arr[::5]]`` would return a
            :obj:`TrajEnsemble` with 7 SingleTrajs with 1 frame each
            (``subsampled.n_frames == 7``). Because the time and frame numbers
            are saved all the time, this should not be too much of a problem.

        """
        if stride is None and total is not None:
            idx = self.id[
                np.unique(
                    np.round(np.linspace(0, self.n_frames - 1, total)).astype(int)
                )
            ]
            return self[idx]
        elif total is None and stride is not None:
            trajs = []
            for i, traj in enumerate(self.trajs):
                _ = traj[slice(None, None, stride)]
                trajs.append(_)
            return TrajEnsemble(
                trajs, common_str=self.common_str, basename_fn=self.basename_fn
            )
        else:
            print("Provide either stride or total.")

    def get_single_frame(self, key: int) -> "SingleTraj":
        """Returns a single frame from all loaded trajectories.

        Consider a :obj:`TrajEnsemble` class with two trajectories. One has 10
        frames, the other 5 (``trajs.n_frames`` is 15). Calling
        ``trajs.get_single_frame(12)`` is equal to calling ``trajs[1][1]``.
        Calling ``trajs.get_single_frame(16)`` will error, and
        ``trajs.get_single_frame(1)`` is the same as ``trajs[0][1]``.

        Args:
            key (int): The frame to return.

        Returns:
            encodermap.trajinfo.info_single.SingleTraj: The frame.

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
            traj_nums = np.unique(self.id[:, 0])
            if not np.array_equal(traj_nums, np.arange(len(traj_nums) + 1)):
                traj_out = self.tsel[num][frame]
            else:
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
        CV_names: tuple[str] = ("",),
        deterministic: bool = True,
        yield_index: bool = True,
        start: int = 1,
    ) -> Iterator[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ]: ...

    @overload
    def batch_iterator(
        self,
        batch_size: int,
        replace: bool = False,
        CV_names: tuple[str] = ("",),
        deterministic: bool = True,
        yield_index: bool = False,
        start: int = 1,
    ) -> Iterator[np.ndarray]: ...

    @overload
    def batch_iterator(
        self,
        batch_size: int,
        replace: bool = False,
        CV_names: Optional[Sequence[str]] = None,
        deterministic: bool = True,
        yield_index: bool = True,
        start: int = 1,
    ) -> Iterator[
        tuple[
            np.ndarray,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ]
    ]: ...

    @overload
    def batch_iterator(
        self,
        batch_size: int,
        replace: bool = False,
        CV_names: Optional[Sequence[str]] = None,
        deterministic: bool = True,
        yield_index: bool = False,
        start: int = 1,
    ) -> Iterator[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]: ...

    def batch_iterator(
        self,
        batch_size: int,
        replace: bool = False,
        CV_names: Optional[Sequence[str]] = None,
        deterministic: bool = False,
        yield_index: bool = False,
        start: int = 1,
    ) -> Iterator[Any]:
        """Lazy batched iterator of CV data.

        This iterator extracts batches of CV data from the ensemble. If the
        ensemble is a large HDF5 datset, this provides the ability to use all
        data without loading it all into memory.

        Examples:

            Import EncoderMap and load some example trajectories.

            >>> import encodermap as em
            >>> trajs = em.TrajEnsemble(
            ...     [
            ...         'https://files.rcsb.org/view/1YUG.pdb',
            ...         'https://files.rcsb.org/view/1YUF.pdb'
            ...     ]
            ... )

            This iterator will yield new samples forever. The batch is a tuple
            of :obj:`numpy.ndarray`.

            >>> for batch in trajs.batch_iterator(batch_size=2):
            ...     print([b.shape for b in batch])
            ...     break
            [(2, 148), (2, 147), (2, 150, 3), (2, 149), (2, 82)]

            Use it with Python's builtin ``next()`` function. The ``deterministic``
            flag returns deterministic batches. The ``yield_index`` flag also
            provides the index of the extracted batch. In this example, both batches
            are extracted from the 1YUG trajectory (``traj_num==0``).

            >>> iterator = trajs.batch_iterator(deterministic=True, batch_size=2, yield_index=True)
            >>> index, batch = next(iterator)
            >>> index
            [[0 5]
             [0 8]]
            >>> index, batch = next(iterator)
            >>> index
            [[ 0  3]
             [ 0 10]]

             If a single string is requested for ``CV_names``, the batch, will
             be a sinlge :obj:`numpy.ndarray`, rather than a tuple thereof.

             >>> iterator = trajs.batch_iterator(batch_size=2, CV_names=["central_dihedrals"])
            >>> batch = next(iterator)
            >>> batch.shape
            (2, 147)

        Args:
            batch_size (int): The size of the batch.
            replace (bool): Whether inside a single batch a sample can occur
                more than once. Set to False (default) to only allow unique
                samples in a batch.
            CV_names (Sequence[str]): The names of the CVs to be used in the
                iterator. If a list/tuple with a single string is provided, the
                batch will be a :obj:`numpy.ndarray`, rather than a tuple
                thereof.
            deterministic (bbol): Whether the samples should be deterministic.
            yield_index (bool): Whether to also yield the index of the extracted
                samples.
            start (int): A start ineteger, which can be used together with
                ``deterministic=True`` to get different deterministic datasets.

        Returns:
            Iterator[Any]: Different iterators based on chosen arguments.

        """
        # Encodermap imports
        from encodermap.autoencoder.autoencoder import np_to_sparse_tensor

        # the standard CV_names
        if CV_names is None:
            CV_names = [
                "central_angles",
                "central_dihedrals",
                "central_cartesians",
                "central_distances",
                "side_dihedrals",
            ]

        if self.CVs_in_file and all([t.index == (None,) for t in self.trajs]):
            ds = xr.open_dataset(
                self.trajs[0].traj_file, engine="h5netcdf", group="CVs"
            )
            ds = ds[CV_names]
            total_len = (
                ds.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .sizes["frame"]
            )
        else:
            ds = self._CVs[CV_names]
            total_len = (
                ds.stack({"frame": ("traj_num", "frame_num")})
                .transpose("frame", ...)
                .dropna("frame", how="all")
                .sizes["frame"]
            )
        traj_nums_and_frames = self.id.copy()
        if self.CVs_in_file:
            assert len(traj_nums_and_frames) == total_len, (
                f"The CVs of the trajs are not aligned with the frames. The "
                f"CVs stacked along the traj/frame axis have a shape of {total_len}, "
                f"while the id array of the trajs has a shape of {self.id.shape}. "
                f"The frames of the trajs are reported as {self.n_frames}. The "
                f"CV data was extracted from the .h5 file {self.trajs[0].traj_file}, "
                f"by stacking the traj/frame axis into a combined axis and dropping "
                f"the frames full of NaNs for the CVs {CV_names}."
            )
        else:
            assert len(traj_nums_and_frames) == total_len, (
                f"The CVs of the trajs are not aligned with the frames. The "
                f"CVs stacked along the traj/frame axis have a shape of {total_len}, "
                f"while the id array of the trajs has a shape of {self.id.shape}. "
                f"The frames of the trajs are reported as {self.n_frames}. The CV "
                f"data was obtained from combining the CVs {CV_names} of the trajectories "
                f"in this ensemble along a traj axis."
            )

        # detect sparse or sidechain only sparse
        sparse = set()
        sub_ds = ds.sel(frame_num=0)
        for o in CV_names:
            datum = sub_ds[o].values
            if np.isnan(datum).any(1).any(None):
                sparse.add(o)

        # start the loop
        # i is the counter for the sample
        # j is the counter for how many tries were needed to select either
        # unique indices (if replace is False) or select indices where not
        i = start
        while True:
            index = []
            out = [[] for o in CV_names]
            j = 0
            while len(index) < batch_size:
                if j > 100 * batch_size:
                    raise Exception(
                        f"Can't find unique indices after 100 iterations. "
                        f"Current index is {index=}."
                    )
                if deterministic:
                    np.random.seed(i + j)
                idx = tuple(traj_nums_and_frames[np.random.randint(0, total_len, 1)[0]])
                if idx in index and not replace:
                    j += 1
                    continue
                data = ds.sel(traj_num=idx[0], frame_num=idx[1])
                # check if any values in CV_names are all nans
                # this can happen for ensembles with different length trajectories
                # we append to `out_`, because if not all nans for this frame
                # we can append `out_` to `out`
                out_ = []
                for k, o in enumerate(CV_names):
                    v = data[o].values
                    if "dist" in o:
                        assert np.all(np.nan_to_num(v, copy=True, nan=1.0) > 0.0), (
                            f"Distances for the selection traj_num={idx[0]} frame={idx[1]} "
                            f"contained a 0. This will result in problems with the cartesian "
                            f"cost."
                        )
                    if np.all(np.isnan(v)):
                        # if all nans break
                        out_ = []
                        break
                    else:
                        out_.append(v)
                # and continue
                if out_ == []:
                    j += 1
                    continue
                # if not, we can append
                index.append(idx)
                for k, o in enumerate(out_):
                    out[k].append(o)
                j += 1
            # stack
            out = [np.stack(o, 0) for o in out]
            index = np.array(index)

            # make sparse tensors
            for i, o in enumerate(CV_names):
                if o in sparse:
                    if out[i].ndim > 2:
                        out[i] = out[i].reshape(batch_size, -1)
                    out[i] = np_to_sparse_tensor(out[i])

            i += 1
            # and yield
            if len(CV_names) > 1:
                out = tuple(out)
            else:
                out = out[0]
            if yield_index:
                yield index, out
            else:
                yield out

    def tf_dataset(
        self,
        batch_size: int,
        replace: bool = False,
        sidechains: bool = False,
        reconstruct_sidechains: bool = False,
        CV_names: Optional[list[str]] = None,
        deterministic: bool = False,
        prefetch: bool = True,
        start: int = 1,
    ) -> tf.data.Dataset:
        # Third Party Imports
        import tensorflow as tf

        gen = lambda: self.batch_iterator(
            batch_size, replace, CV_names, deterministic, start=start
        )
        if CV_names is None and not sidechains and not reconstruct_sidechains:
            CV_names = [
                "central_angles",
                "central_dihedrals",
                "central_cartesians",
                "central_distances",
            ]
        elif CV_names is None and sidechains and not reconstruct_sidechains:
            CV_names = [
                "central_angles",
                "central_dihedrals",
                "central_cartesians",
                "central_distances",
                "side_dihedrals",
            ]
        elif CV_names is None and reconstruct_sidechains:
            CV_names = [
                "central_angles",
                "central_dihedrals",
                "all_cartesians",
                "central_distances",
                "side_angles",
                "side_dihedrals",
                "side_distances",
            ]
        for o in CV_names:
            assert o in self._CVs, f"The CV '{o}' is not loaded in this ensemble."

        # define the TensorSpecs
        sample = next(
            self.batch_iterator(
                batch_size=batch_size,
                replace=replace,
                CV_names=CV_names,
                deterministic=deterministic,
            )
        )
        if isinstance(sample, tuple):
            tensor_specs = []
            for o, s in enumerate(sample):
                if isinstance(s, tf.sparse.SparseTensor):
                    tensor_specs.append(
                        tf.SparseTensorSpec(shape=s.dense_shape, dtype="float32")
                    )
                else:
                    tensor_specs.append(tf.TensorSpec(shape=s.shape, dtype="float32"))
            tensor_specs = tuple(tensor_specs)
        else:
            tensor_specs = tf.TensorSpec(shape=sample.shape, dtype="float32")
        dataset = tf.data.Dataset.from_generator(gen, output_signature=tensor_specs)
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
                - int: A loop-counter integer. Is identical with traj.traj_num.
                - encodermap.SingleTraj: An SingleTraj object.

        Examples:
            >>> import encodermap as em
            >>> trajs = em.TrajEnsemble(
            ...     [
            ...         'https://files.rcsb.org/view/1YUG.pdb',
            ...         'https://files.rcsb.org/view/1YUF.pdb'
            ...     ]
            ... )
            >>> for i, traj in trajs.itertrajs():
            ...     print(traj.basename)
            1YUG
            1YUF

        """
        for traj in self:
            yield traj.traj_num, traj

    def iterframes(self) -> Iterator[tuple[int, int, "SingleTraj"]]:
        """Generator over the frames in this instance.

        Yields:
            tuple: A tuple containing the following:
                - int: The traj_num
                - int: The frame_num
                - encodermap.SingleTraj: An SingleTraj object.

        Examples:

            Import EncoderMap and load an example :obj:`TrajEnsemble`.


            >>> import encodermap as em
            >>> trajs = em.TrajEnsemble(
            ...     [
            ...         'https://files.rcsb.org/view/1YUG.pdb',
            ...         'https://files.rcsb.org/view/1YUF.pdb',
            ...     ],
            ... )
            >>> print(trajs.n_frames)
            31

            Subsample every tenth frame.

            >>> trajs = trajs.subsample(10)
            >>> trajs.n_frames
            4

            Call the :meth:`iterframes` method.

            >>> for traj_num, frame_num, frame in trajs.iterframes():
            ...     print(traj_num, frame_num, frame.n_frames)
            0 0 1
            0 10 1
            1 0 1
            1 10 1

        """
        for traj in self:
            yield from traj.iterframes(with_traj_num=True)

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
        elif isinstance(key, list) and not isinstance(key[0], list):
            new_class = self._return_trajs_by_index(key)
            return new_class
        elif isinstance(key, np.ndarray):
            if key.ndim == 1:
                new_class = self._return_trajs_by_index(key)
                return new_class
            elif key.ndim == 2:
                new_class = self._pyemma_indexing_no_tsel(key)
                return new_class
            else:
                raise IndexError(
                    f"Passing a key with more than 2 dimensions makes no sense. "
                    f"One dim for trajs, one for frames. Your key has "
                    f"{key.ndim} dimensions."
                )
        elif isinstance(key, slice):
            start, stop, step = key.indices(self.n_trajs)
            list_ = list(range(start, stop, step))
            new_class = self[list_]
            return new_class
        elif isinstance(key, list) and all(isinstance(k, list) for k in key):
            return self[np.asarray(key)]
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
        # assert the other contains trajs
        if len(y.trajs) == 0:  # pragma: nocover
            raise Exception(
                f"The `TrajEnsemble` {y} does not contain any trajs and can't "
                f"be used in addition."
            )
        # decide on the new backend
        if self.backend != y.backend:
            print("Mismatch between the backends. Using 'mdtraj'.")
            y.load_trajs()
            self.load_trajs()

        if not set(self.traj_nums).isdisjoint(set(y.traj_nums)):
            raise Exception(
                f"Can't add two `TrajEnsemble` with overlapping traj_nums: "
                f"left side: {self.traj_nums}\n"
                f"right side: {y.traj_nums}"
            )

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
                traj_nums=self.traj_nums + y.traj_nums,
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
            s = (
                f"encodermap.TrajEnsemble object. Current backend is "
                f"{self.backend}. Containing {self.n_frames} frames and "
                f"{self.n_trajs} trajectories."
            )
        else:
            s = (
                f"encodermap.TrajEnsemble object. Current backend is "
                f"{self.backend}. Containing {self.n_trajs} trajectories."
            )
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

# -*- coding: utf-8 -*-
# encodermap/moldata/moldata.py
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

"""New MolData class. Uses PyEMMA to calculate many trajectories in Parallel.

Even when the set of trajectories or even collective variables is too large to keep in memory.

Allows creation of tfrecord files to pass large datasets to tensorflow that normally won't fit into memory.

Is Backwards-compatible to the old MolData class.

ToDo:
    * Add tfrecord capabilities


"""

##############################################################################
# Imports
##############################################################################

import numpy as np

from .._optional_imports import _optional_import
from ..encodermap_tf1.moldata import MolData
from ..loading import Featurizer
from ..trajinfo.info_all import TrajEnsemble
from ..trajinfo.info_single import SingleTraj

##############################################################################
# Optional Imports
##############################################################################


mda = _optional_import("MDAnalysis")
md = _optional_import("mdtraj")

##############################################################################
# Globals
##############################################################################

__all__ = ["NewMolData"]

##############################################################################
# Public Classes
##############################################################################


class NewMolData:
    """MolData version 2. Extracts and holds conformational information of trajectories.

    In version 2. You can either use MDAnalysis or the out-of memory option using
    encodermap's new TrajEnsemble and SingleTraj classes.

    Collective Variables is a term used for data of some dimension matching the dimension of your trajectory.
        Collective variables of dimensionality 1 assign a single (float) value to every frame of a simulation or
        simulation ensemble. This could the the membership to a cluster, the distance between the termini of a
        protein or the distance between two spin labels. Collective variables of dimensionality 2
        assign a list of floats to every simulation frame. The backbone torsions are such a collective variable.
        A flattened array of pairwise distances between CA atoms would also fall into this category. CVs of
        dimensionality 3 ascribe a value to every atom in every frame. This could be the xyz-coordinates of the atom
        or the beta-factor or the charge.

    Encodermap in its Angle-Dihedral-Cartesioan mode uses the following collective variables:
        * cartesians: The xyz-coordinates of every atom in every frame in every trajectory.
        * central_cartesians: The xyz-coordinates of the backbone C, CA, N atoms.
        * dihedrals: The omega-phi-psi angles of the backbone.
        * angles: The angles between the central_cartesian atoms.
        * lengths: The distances between the central_cartesian atoms.
        * sidedihedrals: The dihedrals of the sidechains in order residue1-chi1-chi5 residue2-ch1-chi5.

    """

    def __init__(
        self,
        trajs,
        cache_path="",
        top=None,
        write_traj=False,
        fmt=".nc",
        start=None,
        stop=None,
        step=None,
    ):
        """Instantiate the MolData Class.

        The trajs parameter can take a number of possible inputs:
            * MDAnalysis.AtomGroup: Ensuing backwards-compatibility to the old MolData class.
            * em.TrajEnsemble: EncoderMap's TrajEnsemble class which keeps track of frames and collective
                variables.
            * list of str: If you don't want to bother yourself with the TrajEnsemble class you can pass a
                list of str giving the filenames of many trajetcory files (.xtc, .dcd, .h5). Make sure
                to also provide a topology in case of non-topology trajectories.

        Args:
            trajs (Union[MDAnalysis.AtomGroup, encodermap.TrajEnsemble, list]): The trajectories to load.
                Can be either one of the following:
                * MDAnalysis.AtomGroup. For Backwards-compatibility.
                * encodermap.TrajEnsemble. New TrajEnsemble class which manages frames and collective variables.
                * list: Simply provide a list of trajectory files and don't forget to provide a topology.
            cache_path (str, optional): Where to save generated Data to. Saves either numpy arrays (when AtomGroup
                is provided as trajs, or fmt is '.npy') or NetCDF-HDF5 files with xarray (fmt is '.nc'). When an
                empty string is provided nothing is written to disk. Defaults to '' (empty string).
            top (Union[str, mdtraj.Topology, None], optional): The topology of trajs in case trajs is a list of str.
                Can take filename of a topology file or already loaded mdtraj.Topology. Defaults to None.
            write_traj (bool, optional): Whether to include the trajectory (+topology) into the NetCDF-HDF5 file.
                This option only works in conjunction with fmt='.nc' and if set to True will use mdtraj to write the
                trajectory, topology and the collective variables to one comprehensive file.
            fmt (str, optional): The format to save the CVs as. Can be either '.npy' or '.nc'. Defaults to '.nc'.
                The default is NetCDF-HDF5, because these files can be read iteratively and such can be larger
                than memory allows. This helps in the construction of tfrecord files that can also be used to train
                a network with large datasets.
            start (Union[int, None], optional): First frame to analyze. Is there for backwards-compatibility. This
                feature is dropped in the newer TrajEnsemble pipeline.
            stop (Union[int, None], optional): Last frame to analyze. Is there for backwards-compatibility. This
                feature is dropped in the newer TrajEnsemble pipeline.
            step (Union[int, None], optional): Step provided to old MolData class. Is there for backwards-compatibility.
                This feature is dropped in the newer TrajEnsemble pipeline.

        Examples:
            >>> import encodermap as em
            >>> traj =

        """
        if isinstance(trajs, mda.AtomGroup):
            self = MolDatav1(trajs, cache_path, start, stop, step)
            return
        if isinstance(trajs, str):
            trajs = [trajs]
        if all([isinstance(i, str) for i in trajs]):
            self.trajs = TrajEnsemble(trajs, tops)
        elif isinstance(trajs, TrajEnsemble):
            self.trajs = trajs
        elif isinstance(trajs, SingleTraj):
            self.trajs = trajs._gen_ensemble()
        else:
            raise TypeError(
                f"trajs musst be str, list, TrajEnsemble, SingleTraj, or mda.AtomGroup. You supplied {type(trajs)}"
            )

        if cache_path:
            feat = Featurizer(self.trajs, in_memory=False)
            feat.add_list_of_feats("all")
            self.trajs.load_CVs(feat, directory=cache_path)
        else:
            feat = Featurizer(self.trajs)
            feat.add_list_of_feats("all")
            self.trajs.load_CVs(feat)

        # Use the data from self.trajs
        self.cartesians = self.trajs.all_cartesians
        self.central_cartesians = self.trajs.central_cartesians
        self.dihedrals = self.trajs.central_dihedrals
        self.sidedihedrals = self.trajs.side_dihedrals
        self.angles = self.trajs.central_angles
        self.lengths = self.trajs.central_distances

    def __iadd__(self, other):
        self.cartesians = np.concatenate([self.cartesians, other.cartesians], axis=0)
        self.central_cartesians = np.concatenate(
            [self.central_cartesians, other.central_cartesians], axis=0
        )
        self.dihedrals = np.concatenate([self.dihedrals, other.dihedrals], axis=0)
        self.sidedihedrals = np.concatenate(
            [self.sidedihedrals, other.sidedihedrals], axis=0
        )
        self.angles = np.concatenate([self.angles, other.angles], axis=0)
        self.lengths = np.concatenate([self.lengths, other.lengths], axis=0)
        return self

    def write_tfrecords(self, path=None):
        """Todo"""
        pass

# -*- coding: utf-8 -*-
# encodermap/__init__.py
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
# Typing
################################################################################


from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Sequence, Union

if TYPE_CHECKING:
    import mdtraj as md
    import numpy as np

    from .trajinfo.info_all import TrajEnsemble
    from .trajinfo.info_single import SingleTraj


################################################################################
# Globals
################################################################################


__all__ = [
    "features",
    "__version__",
    "Autoencoder",
    "EncoderMap",
    "AngleDihedralCartesianEncoderMap",
    "EncoderMapBaseCallback",
    "Featurizer",
    "function",
    "MolData",
    "ADCParameters",
    "Parameters",
    "InteractivePlotting",
    "Repository",
    "load",
]

__doc__ = """EncoderMap: Dimensionality reduction for molecular dynamics.

**EncoderMap** provides a framework for using molecular dynamics data with
with the tensorflow library. It started as the implementation of a neural
network autoencoder to do dimensionality reduction and also create new
high-dimensional data from the low-dimensional embedding. The user was still
required to create their own dataset and provide the numpy arrays. In the second
iteration of EncoderMap, the possibility to provide molecular dynamics data with
the `MolData` class was added. A new neural network architecture was implemented
to try and rebuild cartesian coordinates from the low-dimensional embedding.

This iteration of EncoderMap continues this endeavour by porting the old
code to the newer tensorflow version (2.x). However, more has been added which
should aid computational chemists and also structural biologists:

* New trajectory classes with lazy loading of coordinates to save disk space.
* Featurization which can be parallelized using the distributed computing
    library dask.
* Interactive matplotlib plots for clustering and structure creation.
* Neural network building blocks that allows users to easily build new
    nural networks.
* Sparse networks allow comparison of proteins with different topologies.

"""


################################################################################
# Imports
################################################################################


from encodermap._version import __version__
from encodermap.autoencoder.autoencoder import (
    AngleDihedralCartesianEncoderMap,
    Autoencoder,
    EncoderMap,
)
from encodermap.callbacks.callbacks import EncoderMapBaseCallback
from encodermap.loading import features
from encodermap.loading.featurizer import Featurizer
from encodermap.misc.function_def import function
from encodermap.moldata.moldata import MolData
from encodermap.parameters.parameters import ADCParameters, Parameters
from encodermap.plot.interactive_plotting import InteractivePlotting
from encodermap.trajinfo.info_all import TrajEnsemble
from encodermap.trajinfo.info_single import SingleTraj
from encodermap.trajinfo.repository import Repository


def load(
    trajs: Union[str, md.Trajectory, Sequence[str], Sequence[md.Trajectory]],
    tops: Optional[
        Union[str, md.Topology, Sequence[str], Sequence[md.Topology]]
    ] = None,
    common_str: Optional[str, list[str]] = None,
    backend: Literal["no_load", "mdtraj"] = "no_load",
    index: Optional[Union[int, np.ndarray, list[int], slice]] = None,
    traj_num: Optional[int] = None,
    basename_fn: Optional[Callable] = None,
) -> Union[SingleTraj, TrajEnsemble]:
    """Encodermap's forward facing function to work with MD data of single or more trajectories.

    Based what's provided for `trajs`, you either get a `SingleTraj` object, that
    collects information about a single traj, or a `TrajEnsemble` object, that
    contains information of multiple trajectories (even with different topologies).

    Args:
        trajs (Union[str, md.Trajectory, Sequence[str], Sequence[md.Trajectory], Sequence[SingleTraj]]):
            Here, you can provide a single string pointing to a trajectory on your
            computer (`/path/to/traj_file.xtc`) or (`/path/to/protein.pdb`) or
            a list of such strings. In the former case, you will get a
            `SingleTraj` object which is encodermap's way of storing data
            (positions, CVs, times) of a single trajectory. In
            the latter case, you will get a `TrajEnsemble` object, which is
            Encodermap's way of working with mutlipel `SingleTrajs`.
        tops (Optional[Union[str, md.Topology, Sequence[str], Sequence[md.Topology]]]):
            For this argument, you can provide the topology(ies) of the corresponding traj(s).
            Trajectory file formats like `.xtc` and `.dcd` only store atomic positions
            and not weights, elements, or bonds. That's what the `tops` argument is
            for. There are some trajectory file formats out there (MDTraj HDF5, AMBER netCDF4)
            that store both trajectory and topology in a single file. Also `.pdb`
            file can also be used as If you provide
            such files for `trajs`, you can leave tops as None. If you provide multiple
            files for `trajs`, you can still provide a single `tops` file, if the trajs
            in `trajs` share the same topology. If that is not the case, you can either
            provide a list of topologies, matched to the trajs in `trajs`, or use the
            `common_str` argument to match them. Defaults to None.
        common_str (Optional[str, list[str]]): If you provided a different number
            of `trajs` and `tops`, this argument is used to match them. Let's say,
            you have 5 trajectories of a wild type protein and 5 trajectories of
            a mutant. If the path to these files is somewhat consistent (e.g:
                * /path/to/wt/traj1.xtc
                * /different/path/to/wt/traj_no_water.xtc
                * ...
                * /data/path/to/mutant/traj0.xtc
                * /data/path/to/mutant/traj0.xtc
            ), you can provide `['wt', 'mutant']` for the `common_str` argument
            and the files are grouped based on the occurence of 'wt' and 'mutant'
            in ther filepaths. Defaults to None.
        backend (Literal["no_load", "mdtraj"]): Normally, encodermap postpones the
            actual loading of the atomic positions until you really need them.
            This accelerates the handling of large trajectory ensembles. Choosing
            'mdtraj' as the `backend`, all atomic positions are always loaded,
            taking up space on your system memory, but accessing positions in
            a non-sequential fashion is faster. Defaults to 'no_load'.
        index (Optional[Union[int, np.ndarray, list[int], slice]]): Only used, if
            argument `trajs` is a single trajectory. This argument can be used
            to index the trajectory data. If you want to exclude the first 100 frames
            of your trajectory, because the protein relaxes from its crystal
            structure, you can load it like so:
                `em.load(traj_file, top_file, index=slice(100))`
            As encodermap lazily evaluates positional data, the `slice(100)` argument
            is stored until the data is accessed in which case the first 100 frames are
            not accessible. Just like, if you would have deleted them. Besides
            a slice, you can also provide int (which returns a single frame at the
            requested index) and lists of int (which returns frames at the locations
            indexed by the ints in the list). If None is provided the trajectory
            data is not sliced/subsampled. Defaults to None.
        traj_num (Optional[int]): Only used, if argument `trajs` is a single trajectory.
            This argument is meant to organize the `SingleTraj` trajectories in a
            `TrajEnsemble` class. Of course you can build your own `TrajEnsemble` from
             a list of `SingleTraj`s and provide this list as the `trajs` argument to
            `em.load()`. In this case you need to set the `traj_num`s of the `SingleTraj`s
            yourself. Defaults to None.
        basename_fn (Optional[Callable]): A function to apply to the `traj_file` string to return the
            basename of the trajectory. If None is provided, the filename without extension will be used. When
            all files are named the same and the folder they're in defines the name of the trajectory you can supply
            `lambda x: split('/')[-2]` as this argument. Defaults to None.

    Examples:
        >>> # load a pdb file with 14 frames from rcsb.org
        >>> import encodermap as em
        >>> traj = em.load("https://files.rcsb.org/view/1GHC.pdb")
        >>> print(traj)
        encodermap.SingleTraj object. Current backend is no_load. Basename is 1GHC. Not containing any CVs.
        >>> traj.n_frames
        14
        >>> # load multiple trajs
        >>> trajs = em.load(['https://files.rcsb.org/view/1YUG.pdb', 'https://files.rcsb.org/view/1YUF.pdb'])
        >>> # trajs are inernally numbered
        >>> print([traj.traj_num for traj in trajs])

    """
    import numpy as _np

    if isinstance(trajs, (list, tuple, _np.ndarray)):
        from encodermap.trajinfo.info_all import TrajEnsemble

        if index is not None:
            print(
                "The `index` argument is not used when building a trajectory ensemble "
                "Use `trajs.subsample()` to reduce the number of frames staged for analysis."
            )
        return TrajEnsemble(trajs, tops, backend, common_str, basename_fn)
    else:
        from encodermap.trajinfo.info_single import SingleTraj

        return SingleTraj(
            trajs, tops, common_str, backend, index, traj_num, basename_fn
        )


# delte unwanted stuff
del annotations, TYPE_CHECKING, Callable, List, Literal, Optional, Sequence, Union

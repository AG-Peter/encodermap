# -*- coding: utf-8 -*-
# encodermap/__init__.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
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
"""EncoderMap: Dimensionality reduction for molecular dynamics.

**EncoderMap** provides a framework for using molecular dynamics data
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

Todo:
    * [ ] Fix the docker-compose.yaml and add two args: -type [normal, dask] The dask type should add two containers inheriting from base with different ssh signature.
    * [ ] Rework all notebooks.
    * [ ] Record videos.
    * [x] Add a convenience script to run all doctests.
    * [ ] Add a python/bash script to add to crontab to build tests and host.
    * [ ] Run all tests.
        * [x] test_angles.py
        * [x] test_autoencoder.py
            * [x] Fix the `test_encodermap_with_dataset` test
            * [x] Fix the splits with an appropriate test.
            * [x] Put a variation of the two-state system test here, without the deterministic stuff.
            * [x] `test_normal_autoencoder_has_correct_activations`
            * [x] `test_encodermap_with_dataset`
            * [x] `test_save_train_load`
            * [x] `load_legacy_model`
        * [x] test_backmapping_em1_em2.py
            * [x] Rework the features and Featurizers.
            * [x] Fix `test_mdtraj_with_given_inputs`
            * [x] Fix `test_custom_AAs_with_KAC`
            * [x] Fix `test_custom_aas_with_OTU11` <- moved into `test_backmapping_cases`
            * [x] Fix `test_backmapping_cases`
            * [x] Add a performance metric to the test_backmapping_em1_em2.py and try to beat MDAnalysis
                * [x] Make MDTRaj rotation faster with joblib parallel and cython.
            * The problem was my parallel implementation which was baaaad
        * [x] test_dihedral_to_cartesian.py
        * [ ] test_featurizer.py
            * [ ] Run all tests for the dask featurizer.
        * [ ] test_interactive_plotting.py
        * [x] test_losses.py
        * [x] test_moldata.py
        * [x] test_non_backbone_atoms.py
        * [x] test_optional_imports.py
        * [x] test_pairwise_distances.py
        * [ ] test_project_structure.py
        * [ ] test_trajinfo.py
            * [x] `test_CV_slicing_SingleTraj`
            * [x] `test_clustering_different_atom_counts`
            * [x] `test_adding_mixed_pyemma_features_with_custom_names`
            * [x] `test_atom_slice`
            * [x] `test_clustering`
            * [x] `test_load_CVs_from_other_sources`
            * [x] `test_load_all_with_deg_and_rad`
            * [x] `test_load_single_traj_with_traj_and_top`
            * [x] `test_n_frames_in_h5_file`
            * [x] `test_reversed`
            * [x] `test_save_and_load_custom_amino_acids`
            * [x] `test_save_and_load_traj_ensemble_to_h5_and_slice`
            * [x] `test_save_hdf5_ensemble_with_different_top
            * [x] `test_traj_CVs_retain_attrs`
        * [ ] test_version.py
        * [x] test_xarray.py
        * [x] test_tf1_tf2_deterministically.py
            * I've put all tests here as @expensive tests. These won't be part of the official unittest suite.
    * [X] Why does the get_output() not display a progress bar?
        * Removed by new Featurizer.
    * [ ] Add extensive docstring to CustomTopology.
    * [x] Would be nice to display progress bars. Also in dashboard.
    * [ ] Add GAN
    * [ ] Add Unet
    * [ ] Add Multimer training.
    * [ ] Run vulture
    * [ ] Delete commented stuff (i.e. all occurences of more than 3 # signs in lines)
    * [ ] in `xarray.py` delete the occurences of '_INDICES'
    * [ ] Write examples in features.py
    * [ ] Write docstrings in featurizer.py
    * [ ] Write Examples in featurizer.py
    * [ ] Write a runner for my local machine at Uni.

"""
# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import os as _os
import re as _re
import sys as _sys
import warnings as _warnings
from io import StringIO as _StringIO


################################################################################
# Warnings
################################################################################


class _suppress_stderr:
    def __init__(self, filters: Sequence[str]) -> None:
        self.filter = filters

    def __enter__(self):
        self._stderr = _sys.stderr
        _sys.stderr = self._stringio = _StringIO()
        return self

    def __exit__(self, *args, **kwargs):
        _sys.stderr = self._stderr
        self.combine_warnings()
        for warning in self.warnings:
            if warning == "  warnings.warn(\n":
                continue
            if not any([_re.findall(p, warning) for p in self.filter]):
                try:
                    print(warning, file=_sys.stderr)
                except ValueError as e:
                    if "I/O operation on closed file" in str(e):
                        print(warning)
                    else:
                        raise e
        del self.warnings

    def combine_warnings(self):
        self.warnings = []
        index = -1
        previous_is_warn = False
        for line in self._stringio.getvalue().splitlines():
            if "warning" not in line.lower():
                self.warnings[index] += f"{line}\n"
            elif previous_is_warn:
                previous_is_warn = False
                self.warnings[index] += f"{line}\n"
            elif (
                "warnings.warn" in line or "warn(warning_message" in line
            ) and not previous_is_warn:
                index += 1
                previous_is_warn = True
                self.warnings.append(f"{line}\n")
            else:
                index += 1
                self.warnings.append(f"{line}\n")
        del self._stringio


_warnings.filterwarnings(
    "ignore",
    message="'XTCReader' object has no attribute '_xdr'",
)
_warnings.filterwarnings(
    "ignore",
    message=".*unit cell vectors detected in PDB.*",
)
_IGNORE_WARNINGS_REGEX = [
    r".*going maintenance burden of keeping command line.*",
    r".*not pure-Python.*",
    r".*deprecated by PEP 585.*",
]


################################################################################
# Global type checking with beartype
################################################################################


# Standard Library Imports
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Optional, Union


# from beartype.claw import beartype_this_package
# beartype_this_package()


if TYPE_CHECKING:
    # Third Party Imports
    import mdtraj as md
    import numpy as np

    # Local Folder Imports
    from .trajinfo.info_all import TrajEnsemble
    from .trajinfo.info_single import SingleTraj
    from .trajinfo.trajinfo_utils import CustomAAsDict


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
    "load",
    "plot",
]


################################################################################
# Disable tf logging
################################################################################


_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


################################################################################
# Inform User about new tf version
################################################################################


# Third Party Imports
import tensorflow as _tf
from packaging import version as _pkg_version


if _pkg_version.parse(_tf.__version__) < _pkg_version.parse("2.13.0"):
    raise Exception(
        f"Please install the newest tensorflow version (>=2.13.0) to use EncoderMap. "
        f"Your version: {_tf.__version__}."
    )


################################################################################
# Imports
################################################################################


# Encodermap imports
# There are some nasty non pure-python functions in EncoderMap, that beartype
# can't check. The warnings filter does also not work. If beartype likes to
# play the hard way, I will just catch stderr and filter it myself.
with _suppress_stderr(_IGNORE_WARNINGS_REGEX):
    # Encodermap imports
    import encodermap.misc as misc
    import encodermap.plot as plot

    # Local Folder Imports
    from .autoencoder.autoencoder import (
        AngleDihedralCartesianEncoderMap,
        Autoencoder,
        DihedralEncoderMap,
        EncoderMap,
    )
    from .callbacks.callbacks import EncoderMapBaseCallback
    from .kondata import get_from_kondata
    from .loading import features
    from .loading.featurizer import Featurizer
    from .misc.function_def import function
    from .moldata.moldata import NewMolData as MolData
    from .parameters.parameters import ADCParameters, Parameters
    from .plot.interactive_plotting import InteractivePlotting
    from .trajinfo.info_all import TrajEnsemble
    from .trajinfo.info_single import SingleTraj
    from .trajinfo.trajinfo_utils import CustomTopology


################################################################################
# Trajectory API
################################################################################


def load(
    trajs: Union[str, md.Trajectory, Sequence[str], Sequence[md.Trajectory]],
    tops: Optional[
        Union[str, md.Topology, Sequence[str], Sequence[md.Topology]]
    ] = None,
    common_str: Optional[str, list[str]] = None,
    backend: Literal["no_load", "mdtraj"] = "no_load",
    index: Optional[Union[int, np.ndarray, list[int], slice]] = None,
    traj_num: Optional[Union[int], Sequence[int]] = None,
    basename_fn: Optional[Callable] = None,
    custom_top: Optional["CustomAAsDict"] = None,
) -> Union[SingleTraj, TrajEnsemble]:
    """Load MD data.

    Based what's provided for `trajs`, you either get a `SingleTraj` object that
    collects information about a single traj, or a `TrajEnsemble` object, that
    contains information of multiple trajectories (even with different topologies).

    Args:
        trajs (Union[str, md.Trajectory, Sequence[str], Sequence[md.Trajectory], Sequence[SingleTraj]]):
            Here, you can provide a single string pointing to a trajectory on your
            computer (`/path/to/traj_file.xtc`) or (`/path/to/protein.pdb`) or
            a list of such strings. In the former case, you will get a
            `SingleTraj` object which is EncoderMap's way of storing data
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
        encodermap.SingleTraj object. Current backend is no_load. Basename is 1GHC. At indices (None,). Not containing any CVs.
        >>> traj.n_frames
        14
        >>> # load multiple trajs
        >>> trajs = em.load([
        ...     'https://files.rcsb.org/view/1YUG.pdb',
        ...     'https://files.rcsb.org/view/1YUF.pdb'
        ... ])
        >>> # trajs are internally numbered
        >>> print([traj.traj_num for traj in trajs])
        [0, 1]

    """
    # Third Party Imports
    import numpy as _np

    if isinstance(trajs, (list, tuple, _np.ndarray)):
        # Encodermap imports
        from encodermap.trajinfo.info_all import TrajEnsemble

        if index is not None:
            print(
                "The `index` argument is not used when building a trajectory ensemble "
                "Use `trajs.subsample()` to reduce the number of frames staged for analysis."
            )
        return TrajEnsemble(
            trajs,
            tops,
            backend,
            common_str,
            basename_fn,
            traj_nums=traj_num,
            custom_top=custom_top,
        )
    else:
        # Standard Library Imports
        from pathlib import Path

        if Path(trajs).suffix in [".h5", ".nc"]:
            # Encodermap imports
            from encodermap.trajinfo.info_all import TrajEnsemble

            return TrajEnsemble.from_dataset(trajs)

        # Encodermap imports
        from encodermap.trajinfo.info_single import SingleTraj

        return SingleTraj(
            trajs, tops, common_str, backend, index, traj_num, basename_fn, custom_top
        )


def load_project(
    project_name: Literal["linear_dimers", "pASP_pGLU"],
    traj: int = -1,
) -> Union[SingleTraj, TrajEnsemble]:
    """Loads an encodermap project directly into a SingleTraj or TrajEnsemble.

    Args:
        project_name (Literal["linear_dimers"]): The name of the project.
        traj (int): If you want only one traj from the ensemble, set this
            to the appropriate index. If set to -1 the ensemble will be returned.
            Defaults to -1.

    Returns:
        Union[SingleTraj, TrajEnsemble]: The trajectory class.

    """
    if project_name not in ["linear_dimers", "pASP_pGLU"]:
        raise Exception(
            f"The project {project_name} is not part of the EnocoderMap projects."
        )

    # Standard Library Imports
    import os

    # Local Folder Imports
    from .kondata import get_from_kondata

    output_dir = get_from_kondata(
        project_name, mk_parentdir=True, silence_overwrite_message=True
    )
    traj_file = os.path.join(output_dir, "trajs.h5")
    trajs = load(traj_file)
    if isinstance(trajs, TrajEnsemble):
        if traj > -1:
            return trajs[traj]
    return trajs


################################################################################
# Versioning
################################################################################


# Local Folder Imports
from . import _version


__version__ = _version.get_versions()["version"]


################################################################################
# Delete unwanted stuff to prevent clutter
################################################################################


del (
    annotations,
    TYPE_CHECKING,
    Callable,
    Literal,
    Optional,
    Sequence,
    Union,
    _version,
    _tf,
    _pkg_version,
    _warnings,
    _IGNORE_WARNINGS_REGEX,
    _re,
    _os,
    _StringIO,
    _sys,
)

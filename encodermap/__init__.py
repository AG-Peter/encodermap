# -*- coding: utf-8 -*-
# encodermap/__init__.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
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

* New trajectory classes with lazy loading of coordinates to accelerate analysis.
* Featurization which can be parallelized using the distributed computing
    library dask.
* Interactive plotly plots for clustering and structure creation.
* Neural network building blocks that allows users to easily build new
    neural networks.
* Sparse networks allow comparison of proteins with different topologies.

Todo:
    * [ ] Rework all notebooks.
        * [x] 01 Basic cube
        * [x] 02 asp7
        * [x] 03 your data
        * [ ] customization
        * [ ] Ensembles and ensemble classes
        * [ ] Ub mutants
        * [ ] sidechain reconstruction (if possible)
        * [ ] FAT10 (if possible)
    * [ ] Rewrite the install encodermap script in a github gist and add that to the notebooks.
    * [ ] Record videos.
    * [~] Fix FAT 10 Nans
        * [  ] NaNs are fixed, but training still bad.
        * [x] Check whether sigmoid values are good for FAT10
            * [x] Test [40, 10, 5, 1, 2, 5] (from linear dimers) and compare.
        * [ ] Test (20, 10, 5, 1, 2, 5)
    * [~] Fix sidechain reconstruction NaNs
        * [ ] Try out LSTM layers
        * [ ] Try out gradient clipping
        * [~] Try out a higher regularization cost (increase l2 reg constant from 0.001 to 0.1)
    * [ ] Remove OTU11 from tests
    * [ ] Image for FAT10 decoding, if NaN error is fixed.
    * [ ] Delete commented stuff (i.e. all occurrences of more than 3 # signs in lines)
    * [ ] Fix the deterministic training for M1diUb
    * [ ] Add FAT10 to the deterministic training.

"""
# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import os as _os
import re as _re
import sys as _sys
import warnings
import warnings as _warnings
from io import StringIO as _StringIO


################################################################################
# Warnings
################################################################################


class _suppress_stderr:
    """Some modules (looking at you BioPython) are nasty with their warnings.
    They won't accept the builtin catch_warnings() decorator. So we capture
    standard error and prevent them from using it anyway. Delightfully
    devilish, if I say so myself.

    """

    def __init__(
        self,
        filters: Sequence[str],
        issue_warnings: bool = False,
    ) -> None:
        """Instantiate the _supress_stderr class.

        Args:
            filters (Sequence[str]): Sequence of regex patterns to ignore.
            issue_warnings (bool): When set to True, stderr will not be
                suppressed. Can be used in development.

        """
        self.filter = filters
        self.issue_warnings = issue_warnings

    def __enter__(self):
        if self.issue_warnings:
            return self
        # Standard Library Imports
        import copy

        # We need to ignore warnings here, otherwise BioPython will issue
        # a deprecation warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Local Folder Imports
            from .misc.misc import _is_notebook

        if _is_notebook():
            self._stderr = copy.copy(_sys.stderr)
        else:
            self._stderr = _sys.stderr
        _sys.stderr = self._stringio = _StringIO()
        return self

    def __exit__(self, *args, **kwargs):
        if self.issue_warnings:
            return
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

    def combine_warnings(self) -> None:
        """Once concluded, all warnings that have not been ignored will be
        combined here and issued to stderr. Just like Guido intended.

        """
        self.warnings: list[str] = []
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


################################################################################
# GPU stuff
################################################################################


class GPUsAreDisabledWarning(UserWarning):
    """Warning to inform users, that EncoderMap runs with higher compatibility,
    if GPUs are disabled."""

    pass


_enable_GPU: bool = _os.getenv("ENCODERMAP_ENABLE_GPU", "False") == "True"


if not _enable_GPU:
    _warnings.warn(
        message=(
            "EncoderMap disables the GPU per default because most tensorflow code "
            "runs with a higher compatibility when the GPU is disabled. If you "
            "want to enable GPUs manually, set the environment variable "
            "'ENCODERMAP_ENABLE_GPU' to 'True' before importing EncoderMap. "
            "To do this in python you can run:\n\n"
            "import os; os.environ['ENCODERMAP_ENABLE_GPU'] = 'True'\n\n"
            "before importing encodermap."
        ),
        category=GPUsAreDisabledWarning,
    )
    _os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union, overload


if _os.getenv("ENCODERMAP_BEARTYPE", "False") == "True":
    _beartyping = True
    # Third Party Imports
    from beartype.claw import beartype_this_package

    beartype_this_package()
else:
    _beartyping = False


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


__all__: list[str] = [
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


ALL_PROJECT_NAMES = Union[
    Literal["linear_dimers"],
    Literal["pASP_pGLU"],
    Literal["Ub_K11_mutants"],
    Literal["cube"],
    Literal["1am7"],
    Literal["H1Ub"],
]


################################################################################
# Disable tf logging
################################################################################


_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


################################################################################
# Allow distributed writes into H5 files
################################################################################


_os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


################################################################################
# Inform User about new tf version
################################################################################


# Third Party Imports
import tensorflow as _tf
from packaging import version as _pkg_version


if _pkg_version.parse(_tf.__version__) < _pkg_version.parse("2.13.0"):
    raise Exception(
        f"Please install the newest tensorflow version (>=2.15.0) to use EncoderMap. "
        f"Your version: {_tf.__version__}."
    )


################################################################################
# Imports
################################################################################


# Encodermap imports
# There are some nasty non-pure python functions in EncoderMap, that beartype
# can't check. The `warnings` filter does also not work. If beartype likes to
# play the hard way, I will just catch stderr and filter it myself.
with _suppress_stderr(_IGNORE_WARNINGS_REGEX, _beartyping):
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
    from .loading.featurizer import DaskFeaturizer, Featurizer
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
    trajs: Union[
        str,
        md.Trajectory,
        Sequence[str],
        Sequence[md.Trajectory],
        Sequence[Path],
        Sequence[SingleTraj],
    ],
    tops: Optional[
        Union[str, md.Topology, Sequence[str], Sequence[md.Topology], Sequence[Path]]
    ] = None,
    common_str: Optional[Union[str, list[str]]] = None,
    backend: Literal["no_load", "mdtraj"] = "no_load",
    index: Optional[Union[int, np.ndarray, list[int], slice]] = None,
    traj_num: Optional[Union[int, Sequence[int]]] = None,
    basename_fn: Optional[Callable[[str], str]] = None,
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
        basename_fn (Optional[Callable[[str], str]]): A function to apply to the `traj_file` string to return the
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
    # Standard Library Imports
    from pathlib import Path
    from typing import Sequence

    if (
        (isinstance(trajs, Sequence) and not isinstance(trajs, (Path, str)))
        and (isinstance(tops, Sequence) or tops is None)
        and (isinstance(traj_num, Sequence) or traj_num is None)
        and (isinstance(common_str, Sequence) or common_str is None)
    ):
        # Encodermap imports
        from encodermap.trajinfo.info_all import TrajEnsemble

        if index is not None:
            print(
                "The `index` argument to `em.load()` is not used when building "
                "a trajectory ensemble. It is only passed to `SingleTraj.__init__()`. "
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
    elif (
        isinstance(trajs, (str, Path))
        and (isinstance(tops, (str, Path)) or tops is None)
        and (isinstance(traj_num, int) or traj_num is None)
        and (isinstance(common_str, str) or common_str is None)
    ):
        if Path(trajs).suffix in [".h5", ".nc"]:
            # Encodermap imports
            from encodermap.trajinfo.info_all import TrajEnsemble

            return TrajEnsemble.from_dataset(trajs)

        if common_str is None:
            common_str = ""

        # Encodermap imports
        from encodermap.trajinfo.info_single import SingleTraj

        return SingleTraj(
            trajs,
            tops,
            common_str,
            backend,
            index,
            traj_num,
            basename_fn,
            custom_top,
        )
    else:
        raise TypeError(
            f"Incompatible types of 'trajs' and 'tops'. Either both have to "
            f"be a sequence or both have to be not a sequence. The provided types "
            f"are 'trajs'={type(trajs)} and 'tops'={type(tops)}."
        )


@overload
def load_project(
    project_name: Literal["linear_dimers"],
    traj: int,
    load_autoencoder: Literal[True],
) -> tuple[TrajEnsemble, AngleDihedralCartesianEncoderMap]: ...


@overload
def load_project(
    project_name: Literal["linear_dimers"],
    traj: int,
    load_autoencoder: Literal[False],
) -> TrajEnsemble: ...


@overload
def load_project(
    project_name: Literal["pASP_pGLU"],
    traj: int,
    load_autoencoder: Literal[True],
) -> tuple[TrajEnsemble, AngleDihedralCartesianEncoderMap]: ...


@overload
def load_project(
    project_name: Literal["pASP_pGLU"],
    traj: int,
    load_autoencoder: Literal[False],
) -> TrajEnsemble: ...


@overload
def load_project(
    project_name: Literal["Ub_K11_mutants"],
    traj: int,
    load_autoencoder: Literal[True],
) -> tuple[TrajEnsemble, AngleDihedralCartesianEncoderMap]: ...


@overload
def load_project(
    project_name: Literal["Ub_K11_mutants"],
    traj: int,
    load_autoencoder: Literal[False],
) -> TrajEnsemble: ...


@overload
def load_project(
    project_name: Literal["1am7"],
    traj: int,
    load_autoencoder: Literal[True],
) -> tuple[TrajEnsemble, EncoderMap]: ...


@overload
def load_project(
    project_name: Literal["1am7"],
    traj: int,
    load_autoencoder: Literal[False],
) -> TrajEnsemble: ...


@overload
def load_project(
    project_name: Literal["cube"],
    traj: int,
    load_autoencoder: Literal[True],
) -> tuple[np.ndarray, EncoderMap]: ...


@overload
def load_project(
    project_name: Literal["cube"],
    traj: int,
    load_autoencoder: Literal[False],
) -> np.ndarray: ...


@overload
def load_project(
    project_name: Literal["H1Ub"],
    traj: int,
    load_autoencoder: Literal[True],
) -> tuple[np.ndarray, EncoderMap]: ...


@overload
def load_project(
    project_name: Literal["H1Ub"],
    traj: int,
    load_autoencoder: Literal[False],
) -> np.ndarray: ...


def load_project(
    project_name: ALL_PROJECT_NAMES,
    traj: int = -1,
    load_autoencoder: bool = False,
) -> Union[
    Union[SingleTraj, TrajEnsemble, np.ndarray],
    tuple[
        Union[SingleTraj, TrajEnsemble, np.ndarray],
        Union[EncoderMap, AngleDihedralCartesianEncoderMap],
    ],
]:
    """Loads an encodermap project directly into a SingleTraj or TrajEnsemble.
    Also loads an instance of an AutoEncoder, when requested.

    Args:
        project_name (Literal["linear_dimers"]): The name of the project.
        traj (int): If you want only one traj from the ensemble, set this
            to the appropriate index. If set to -1, the ensemble will be returned.
            Defaults to -1.
        load_autoencoder (bool): Whether to also reload a trained autoencoder model.

    Returns:
        Union[
            Union[SingleTraj, TrajEnsemble],
            tuple[
                Union[SingleTraj, TrajEnsemble],
                Union[EncoderMap, AngleDihedralCartesianEncoderMap],
            ],
        ]: either the trajectory class or a tuple of Trajectory and Autoencoder.

    """
    autoencoder_mapping = {
        "linear_dimers": AngleDihedralCartesianEncoderMap,
        "pASP_pGLU": AngleDihedralCartesianEncoderMap,
        "Ub_K11_mutants": AngleDihedralCartesianEncoderMap,
        "cube": EncoderMap,
        "1am7": EncoderMap,
        "H1Ub": EncoderMap,
    }
    if project_name not in autoencoder_mapping.keys():
        raise Exception(
            f"The project {project_name} is not part of the EnocoderMap projects."
        )

    # Standard Library Imports
    import os
    from pathlib import Path

    # Third Party Imports
    import tensorflow
    from packaging import version

    # Local Folder Imports
    from .kondata import get_from_kondata

    output_dir = Path(
        get_from_kondata(
            project_name,
            mk_parentdir=True,
            silence_overwrite_message=True,
            download_checkpoints=True,
            download_h5=True,
        )
    )

    if project_name != "cube":
        traj_file = os.path.join(output_dir, "trajs.h5")
        trajs = load(traj_file)
        if isinstance(trajs, TrajEnsemble):
            if traj > -1:
                trajs = trajs[traj]
        if not load_autoencoder:
            return trajs
    else:
        # Local Folder Imports
        from .misc.misc import create_n_cube

        positions, _ = misc.create_n_cube()
        autoencoder = EncoderMap(
            train_data=positions,
        )
        return positions, autoencoder

    if version.parse(tensorflow.__version__) >= version.parse("2.15"):
        keras_files = list(
            (output_dir / "checkpoints/finished_training/tf2_15").glob("*.keras")
        )
        if len(keras_files) == 0:
            raise Exception(
                f"Found no keras files in {output_dir / 'checkpoints/finished_training'}: "
                f"{keras_files}. Not all EncoderMap projects are accompanied by "
                f"trained tensorflow models."
            )
    else:
        keras_files = list(
            (output_dir / "checkpoints/finished_training").glob("*.keras")
        )
        if len(keras_files) == 0:
            raise Exception(
                f"Found no keras files in {output_dir / 'checkpoints/finished_training'}: "
                f"{keras_files}. Most EncoderMap models are trained with tensorflow "
                f"version >= 2.15. Not all models are available for older versions. "
                f"You can update your tensorflow version and try again."
            )
    if len(keras_files) > 1:
        raise Exception(f"Found multiple keras files in {output_dir}: {keras_files}. ")
    keras_file = keras_files[0]
    if autoencoder_mapping[project_name].__name__ == "AngleDihedralCartesianEncoderMap":
        autoencoder = autoencoder_mapping[project_name].from_checkpoint(
            trajs, checkpoint_path=keras_file
        )
    else:
        autoencoder = autoencoder_mapping[project_name].from_checkpoint(
            checkpoint_path=keras_file,
            train_data=trajs.central_dihedrals,
        )
    return trajs, autoencoder


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
    _beartyping,
    GPUsAreDisabledWarning,
    _enable_GPU,
    warnings,
    overload,
    Path,
)

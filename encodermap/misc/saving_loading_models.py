# -*- coding: utf-8 -*-
# encodermap/misc/saving_loading_models.py
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
"""
Implementation of saving and loading models.


"""
################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import copy
import os
import re
import shutil
import warnings
from collections.abc import Callable
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, overload

# Third Party Imports
import numpy as np
import tensorflow as tf

# Encodermap imports
from encodermap.misc.misc import _datetime_windows_and_linux_compatible, run_path
from encodermap.parameters.parameters import ADCParameters, Parameters


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    # Third Party Imports
    import numpy as np

    # Encodermap imports
    from encodermap.autoencoder.autoencoder import AutoencoderClass
    from encodermap.trajinfo.info_all import TrajEnsemble


################################################################################
# Globals
################################################################################


__all__: list[str] = ["save_model", "load_model"]


################################################################################
# Utils
################################################################################


def _change_setting_inform_user(
    p: Union[Parameters, ADCParameters],
    setting: str,
    value: Any,
    parameters_file: Path,
    compat: bool = False,
) -> None:
    """Changes a setting in a parameter file and informs the user with a print message.

    Args:
        p (Union[Parameters, ADCParameters]): An instance of the Parameters class.
            Either `Parameters`, or `ADCParameters`.
        setting (str): The setting to be changed.
        value (Any): The new value of the setting.
        parameters_file (Path): The file in which to change the setting.
        compat (bool): When loading old .model files and some parameters are
            inferred from this file.

    """
    curr = getattr(p, setting)
    if curr != value:
        setattr(p, setting, value)
        msg = (
            f"In the parameters file {parameters_file}, the parameter '"
            f"{setting}' is set to '{curr}', but the architecture of the model "
            f"being loaded requires this parameter to be {value}. This parameter "
            f"file might not belong to the model you're trying to load."
        )
        print(msg)
        if not parameters_file.is_file():
            return

        if not compat:
            msg += (
                f"This tensorflow model was saved in the now deprecated .model "
                f"format. Since moving to the new .keras files, some changes have "
                f"been made to how parameters are saved. Some parameters can "
                f"be inferred from the old .model files by setting the `compat` "
                f"argument to True. This will create a backup of this parameters "
                f"file ({parameters_file}) and try to create a new one."
            )
            raise Exception(msg)
        else:
            msg += (
                f"I will backup the {parameters_file} and set appropriate values to"
                f"a new parameters.json."
            )
    return


def _model_sort_key(model_name: str) -> int:
    """Returns numerical values based on whether `model_name` contains substrings.

    Args:
        model_name (str): The filepath to the saved model.

    Returns:
        int: Returns 0 for 'encoder', 1 for 'decoder', 2 for everything else.

    """
    x = Path(model_name).name
    return 0 if "encoder" in x else (1 if "decoder" in x else 2)


def _load_list_of_models(
    models: list[str],
    custom_objects: Optional[dict[str, Callable]] = None,
) -> list[tf.keras.Model]:
    """Load the models supplied in `models` using keras.

    Args:
        models (list[str]): The paths of the models to be loaded

    """
    return [
        tf.keras.models.load_model(x, custom_objects=custom_objects) for x in models
    ]


def filter_greater_than(step: int) -> Callable:
    """Returns a function that can used for filtering.

    Examples:
        >>> from encodermap.misc.saving_loading_models import filter_greater_than
        >>> test = ["one_1.keras", "two_2.keras", "three_3.keras", "four_4.keras"]
        >>> list(filter(filter_greater_than(3), test))
        ['three_3.keras', 'four_4.keras']

    Args:
        step (int): All files containing this step number or more will not
            be removed from the Sequence by the builtin filter function.

    """

    def closure(path: Path) -> bool:
        r"""The closue of the `filter_greater_than` function.

        Takes a `pathlib.Path` and extracts the last number using regexp (\d+).
        Returns True, if this number is equal or greater than `step`.

        Args:
            path (Path): The path to use.

        Returns:
            bool: Whether the last number in `path` is equal or greater than step.

        """
        current_step = int(re.findall(r"\d+", str(path))[-1])
        return current_step >= step

    return closure


################################################################################
# Functions
################################################################################


def save_model(
    model: tf.keras.Model,
    main_path: Union[str, Path],
    inp_class_name: Optional[str] = None,
    step: Optional[int] = None,
    print_message: bool = False,
) -> Path:
    """Saves a model in the portable .keras format.

    Args:
        model (tf.keras.models.Model): The keras model to save. If the
            keras model has the attribute 'encoder_model' the encoder_model
            will be saved separately. The same with the attribute 'decoder_model'.
        main_path (Union[str, Path]): Which directory to save the model to.
            If step is None, the nae will be saved_model_{time}.keras, where time
            is a current ISO-8601 formatted string.
        step (Optional[int]): Can be None, in which case the model will bve saved
            using the current time. Otherwise, the step argument will be used like
            so: saved_model_{step}.keras Defaults to None.
        print_message (bool): Whether to print a message after saving the model
            Defaults to False.

    Returns:
        Path: The path, where the model was saved.

    """
    main_path = Path(main_path)
    assert main_path.is_dir(), f"Please provide a directory as `main_path`."
    if step is None:
        time = _datetime_windows_and_linux_compatible()
        fname = main_path / f"saved_model_{time}.keras"
        encoder_name = main_path / f"saved_model_{time}_encoder.keras"
        decoder_name = main_path / f"saved_model_{time}_decoder.keras"
    else:
        fname = main_path / f"saved_model_{step}.keras"
        encoder_name = main_path / f"saved_model_{step}_encoder.keras"
        decoder_name = main_path / f"saved_model_{step}_decoder.keras"

    if print_message:
        if inp_class_name is not None:
            print(
                f"Saving the model to {fname}. Use `em.{inp_class_name}.from_checkpoint('{main_path}')` "
                f"to load the most recent model, or `em.{inp_class_name}.from_checkpoint('{fname}')` "
                f"to load the model with specific weights.."
            )
        else:
            print(f"Saving the model to {fname}.")
    model.save(fname)

    if hasattr(model, "encoder_model"):
        if print_message:
            print(
                f"This model has a subclassed encoder, which can be loaded inde"
                f"pendently. Use `tf.keras.load_model('{encoder_name}')` to load "
                f"only this model."
            )
        model.encoder_model.save(encoder_name)

    if hasattr(model, "decoder_model"):
        if print_message:
            print(
                f"This model has a subclassed decoder, which can be loaded inde"
                f"pendently. Use `tf.keras.load_model('{decoder_name}')` to load "
                f"only this model."
            )
        model.decoder_model.save(decoder_name)

    return fname


def sort_model_files_with_timestr(path: Path) -> int:
    """Returns -1 for all files that have an ISO time in their filename and
    other numbers for files with numbers in their names. When a file has
    multiple number in its name '/path/to20/directory5/file_200.txt', the last
    number (200 in this case) will be used.

    Returns:
        int: The sorting value.

    """
    m = re.match(
        r".*\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]" r"\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z).*",
        str(path),
    )
    if m is not None:
        return -1
    number = int(re.findall(r"\d+", str(path))[-1])
    return number


def _find_and_sort_files(
    path: Path,
    globstr: str,
    sort_criterion: Literal["creation_time", "number"] = "number",
) -> list[Path]:
    """Helper function to list possible files in `path`, using a `globstr` and
    a `sort_criterion`.

    Args:
        path (Path): The directory to start with.
        globstr (str): The globstring to use. Example "*saved_model*".
        sort_criterion (Literal["creation_time", "number"]): Files can
            either be sorted by the creation time (`Path.stat().st_ctime`) or
            by "number" in which case the file 'saved_model_500.keras' will
            appear before the file 'saved_model_1000.keras'.

    Returns:
        list[Path]: A list of paths with the applied sorting.

    """
    if sort_criterion == "creation_time":
        sortkey = lambda x: x.stat().st_ctime
    elif sort_criterion == "number":
        sortkey = sort_model_files_with_timestr
    else:
        raise ValueError(
            f"The argument `sort_criterion` has to be 'creation_time', or "
            f"'number', you supplied {sort_criterion=}."
        )
    l = list(
        sorted(
            filter(
                lambda x: "encoder" not in x.stem and "decoder" not in x.stem,
                path.glob(globstr),
            ),
            key=sortkey,
        )
    )
    return l


@overload
def load_model(
    autoencoder: Union[None, "AutoencoderClass"],
    checkpoint_path: Union[str, Path],
    trajs: Optional[TrajEnsemble],
    sparse: bool,
    dataset: Optional[Union[tf.data.Dataset, np.ndarray]],
    print_message: bool,
    submodel: Literal[None],
    use_previous_model: bool,
    compat: bool,
) -> "AutoencoderClass": ...  # pragma: no doccheck


@overload
def load_model(
    autoencoder: Union[None, "AutoencoderClass"],
    checkpoint_path: Union[str, Path],
    trajs: Optional[TrajEnsemble],
    sparse: bool,
    dataset: Optional[Union[tf.data.Dataset, np.ndarray]],
    print_message: bool,
    submodel: Literal["encoder", "decoder"],
    use_previous_model: bool,
    compat: bool,
) -> tf.keras.Model: ...  # pragma: no doccheck


def load_model(
    autoencoder: Union[None, "AutoencoderClass"],
    checkpoint_path: Union[str, Path],
    trajs: Optional[TrajEnsemble] = None,
    sparse: bool = False,
    dataset: Optional[Union[tf.data.Dataset, np.ndarray]] = None,
    print_message: bool = False,
    submodel: Optional[Literal["encoder", "decoder"]] = None,
    use_previous_model: bool = False,
    compat: bool = False,
) -> Union["AutoencoderClass", tf.keras.Model]:
    """Reloads a model from a checkpoint path.

    An implementation of saving the .keras files procuded by EncoderMap.
    The old legacy .model files can still be loaded by this function. Or use
    the `load_model_legacy` function directly.

    Args:
        autoencoder (Union[None, "AutoencoderClass"]): Kept for
            legacy reasons. The old .model files had a list of "custom_objects"
            that was created by the autoencoder classes (`AutoEncoder`,
            `EncoderMap`. `AngleDihedralCartesianEncoderMap`) and needed to
            be supplied when reloading the models from disk. The new implementations
            use the `from_config` and `get_config` implementations of serializable
            keras objects and thus, the layers and cost functions can save their
            own state. Is only needed to load legacy models and can be None if a
            new .keras model is loaded.
        checkpoint_path (Union[str, Path]): Can be either the path to a .keras
            file or to a directory with multiple .keras files in which case, the
            most recent .keras file will be loaded.
        trajs (Optional[TrajEnsemble]): A `TrajEnsemble` class for when
            a `AngleDihedralCartesianEncoderMap` is reloaded.
        sparse (bool): This argument is also only needed to load legacy .model
            files. Defaults to False.
        dataset (Optional[Union[tf.data.Dataset, np.ndarray]]): A pass-through to
            the `dataset` argument of the autoencoder classes (`AutoEncoder`,
            `EncoderMap`. `AngleDihedralCartesianEncoderMap`) which all can take
            a tf.data.Dataset. Can be None, in which case, the data will be
            sourced differently (The `EncoderMap` class uses example data from
            a 4D hypercube, the `AngleDihedralCartesianEncoderMap` uses the
            data from the provided `trajs`.)
        print_message (bool): Whether to print some debug information. Defaults to False.
        submodel (Optional[Literal["encoder", "decoder"]]): Whether to only load
            a specific submodel. In order to use this argument, a file with
            the name *encoder.keras or *decoder.keras has to be in the
            in `checkpoint_path` specified directory.
        use_previous_model (bool): Whether to load a model from an intermediate
            checkpoint step.
        compat (bool): Whether to fix a parameters.json file that has been saved
            with the legacy .model file.

    Returns:
        Union[tf.keras.models.Model, "AutoencoderClass"]: A tf.keras.models.Model
            when you specified submodel. And an appropriate "AutoencoderClass"
            otherwise.


    """
    if "decoder.keras" in str(checkpoint_path) and submodel is None:
        raise Exception(
            f"The file you provided is just the decoder submodel of the complete "
            f"{autoencoder.__name__} class. Loading submodels, requires "
            f"you to explicitly set the argument `submodel='decoder'`. Note, "
            f"that loading submodels will return a `tf.keras.models.Model` instead "
            f"of an instance of {autoencoder.__name__}."
        )
    if "encoder.keras" in str(checkpoint_path) and submodel is None:
        raise Exception(
            f"The file you provided is just the emcoder submodel of the complete "
            f"{autoencoder.__name__} class. Loading submodels, requires "
            f"you to explicitly set the argument `submodel='emcoder'`. Note, "
            f"that loading submodels will return a `tf.keras.models.Model` instead "
            f"of an instance of {autoencoder.__name__}."
        )
    checkpoint_path = Path(checkpoint_path)
    if ".model" in str(checkpoint_path):
        print("Will use the legacy loader for old '*.model' file.")
        return load_model_legacy(
            autoencoder_class=autoencoder,
            checkpoint_path=str(checkpoint_path),
            trajs=trajs,
            sparse=sparse,
            dataset=dataset,
            compat=compat,
        )
    if checkpoint_path.is_dir():
        possible_models = _find_and_sort_files(checkpoint_path, "*saved_model*")
        try:
            newest_model = possible_models[-1]
        except IndexError as e:
            raise Exception(
                f"{checkpoint_path=} has no .keras files: {possible_models=}"
            ) from e
        if ".model" not in str(newest_model):
            if print_message:
                print(
                    f"Found {len(possible_models)} in {checkpoint_path}. I will reload "
                    f"{newest_model}, because this is the newest file."
                )
            model = tf.keras.models.load_model(newest_model)
            checkpoint_path = newest_model
        else:
            possible_old_models = possible_models = _find_and_sort_files(
                checkpoint_path, "*.model"
            )
            print("Will use the legacy loader for old '*.model' file.")
            return load_model_legacy(
                autoencoder_class=autoencoder,
                checkpoint_path=str(possible_old_models[-1]),
                trajs=trajs,
                sparse=sparse,
                dataset=dataset,
                compat=compat,
            )
    else:
        if ".model" in str(checkpoint_path):
            return load_model_legacy(
                autoencoder_class=autoencoder,
                checkpoint_path=str(checkpoint_path),
                trajs=trajs,
                sparse=sparse,
                dataset=dataset,
                compat=compat,
            )
        else:
            model = tf.keras.models.load_model(checkpoint_path)

    # maybe load just encoder or decoder, if requested
    if submodel is not None:
        if submodel == "encoder":
            encoder_file = checkpoint_path.parent / checkpoint_path.name.replace(
                ".keras", "_encoder.keras"
            )
            return tf.keras.models.load_model(encoder_file)
        elif submodel == "decoder":
            decoder_file = checkpoint_path.parent / checkpoint_path.name.replace(
                ".keras", "_decoder.keras"
            )
            return tf.keras.models.load_model(decoder_file)
        else:
            raise ValueError(
                f"Argument `submodel` can only be either 'enocer' or 'decoder'. "
                f"You supplied: {submodel=}."
            )

    # load the params in the directory
    parameter_file = checkpoint_path.parent / "parameters.json"
    if not parameter_file.is_file() and autoencoder is not None:
        warnings.warn(
            f"There was no parameters.json file in the directory. {parameter_file.parent}. "
            f"I will load the model from the keras file, but I can't build a "
            f"{autoencoder} instance without the parameters."
        )
    if parameter_file.is_file():
        assert (
            autoencoder is not None
        ), f"Please provide a class inheriting from `Autoencoder`."
        if "cartesian" in parameter_file.read_text():
            p = ADCParameters.from_file(parameter_file)

            # make sure parameters and current training step are the same
            current_step = re.findall(r"\d+", str(checkpoint_path))
            backup_parameters = (
                parameter_file.parent
                / f"parameters_at_{p.current_training_step}_{_datetime_windows_and_linux_compatible()}_{parameter_file.suffix}"
            )
            if len(current_step) < 3 and len(current_step) >= 1:
                current_step = int(current_step[-1])
                files_to_backup = list(
                    filter(
                        filter_greater_than(current_step),
                        checkpoint_path.parent.glob("*.keras"),
                    )
                )
                backup_files = [
                    f.parent
                    / (
                        f.stem
                        + f"_backup_from_{current_step}_{_datetime_windows_and_linux_compatible()}.keras"
                    )
                    for f in files_to_backup
                ]
                if current_step != p.current_training_step:
                    if not use_previous_model:
                        raise Exception(
                            f"The model was saved at step {current_step}, but the parameters "
                            f"file has its current step at {p.current_training_step}. "
                            f"It seems like you are reloading a model at an intermediate "
                            f"step. If you set the `use_previous_model` flag to True, "
                            f"I will backup the parameters file to {backup_parameters} and "
                            f"set the new training step so, that you can use/retrain this "
                            f"model."
                        )
                    else:
                        shutil.move(parameter_file, backup_parameters)
                        for f1, f2 in zip(files_to_backup, backup_files):
                            shutil.copyfile(f1, f2)
                        p.current_training_step = current_step + 1
                        p.n_steps = current_step + 1
                        p.save()

            # then load and return the autoencoder
            _using_hypercube = deepcopy(p.using_hypercube)
            out = autoencoder(
                trajs,
                parameters=p,
                read_only=False,
                dataset=dataset,
                model=model,
            )
            out.p.using_hypercube = _using_hypercube
            return out
        else:
            p = Parameters.from_file(parameter_file)

            # make sure parameters and current training step are the same
            current_step = re.findall(r"\d+", str(checkpoint_path))
            backup_parameters = (
                parameter_file.parent
                / f"parameters_at_{p.current_training_step}_{_datetime_windows_and_linux_compatible()}_{parameter_file.suffix}"
            )
            if len(current_step) < 3 and len(current_step) >= 1:
                current_step = int(current_step[-1])
                files_to_backup = list(
                    filter(
                        filter_greater_than(current_step),
                        checkpoint_path.parent.glob("*.keras"),
                    )
                )
                backup_files = [
                    f.parent
                    / (
                        f.stem
                        + f"_backup_from_{current_step}_{_datetime_windows_and_linux_compatible()}.keras"
                    )
                    for f in files_to_backup
                ]
                if current_step != p.current_training_step:
                    if not use_previous_model:
                        raise Exception(
                            f"The model was saved at step {current_step}, but the parameters "
                            f"file has its current step at {p.current_training_step}. "
                            f"It seems like you are reloading a model at an intermediate "
                            f"step. If you set the `use_previous_model` flag to True, "
                            f"I will backup the parameters file to {backup_parameters} and "
                            f"set the new training step so, that you can use/retrain this "
                            f"model."
                        )
                    else:
                        shutil.move(parameter_file, backup_parameters)
                        for f1, f2 in zip(files_to_backup, backup_files):
                            shutil.copyfile(f1, f2)
                        p.current_training_step = current_step
                        p.n_steps = current_step
                        p.save()

            # then load and return the autoencoder
            _using_hypercube = deepcopy(p.using_hypercube)
            out = autoencoder(
                parameters=p,
                train_data=dataset,
                read_only=False,
                model=model,
            )
            out.p.using_hypercube = _using_hypercube
            return out
    return model


def load_model_legacy(
    autoencoder_class: Union[None, "AutoencoderClass"],
    checkpoint_path: Union[str, Path],
    trajs: Optional[TrajEnsemble] = None,
    sparse: bool = False,
    dataset: Optional[Union[tf.data.Dataset, np.ndarray]] = None,
    compat: bool = False,
) -> "AutoencoderClass":
    """Loads legacy .model files.

    Note:
        The .model format has been deprecated. Please update your saved models
        to the .keras format. You can yse this function to rebuild a new
        model from the legacy .model files.

    Args:
        autoencoder_class (Union[None, AutoencoderClass]): A class of the in
            EncoderMap implemented autoencoder classes.
        checkpoint_path (Union[str, Path]): The path to the file to load.
        trajs (Optional[TrajEnsemble]): When loading an AngleDihedralCartesianEncoderMap,
            the trajectories need to be supplied to verify the input/output shapes
            of the model.
        sparse (bool): Whether the model contains sparse inputs.
        dataset (Optional[Union[tf.data.Dataset, np.ndarray]): Either a tf.data.Dataset
            or a np.ndarray to infer the input shapre from.
        compat (bool): Whether


    """
    # Local Folder Imports
    from ..autoencoder import AngleDihedralCartesianEncoderMap
    from ..models import gen_functional_model, gen_sequential_model

    if "*" in str(checkpoint_path):
        checkpoint_path = list(
            sorted(map(Path, glob(str(checkpoint_path))), key=_model_sort_key)
        )
        parameters_file = checkpoint_path[0].parent / "parameters.json"
        found = re.findall(r"\d+", str(checkpoint_path[0].name))
    else:
        checkpoint_path = Path(checkpoint_path)
        parameters_file = checkpoint_path.parent / "parameters.json"
        found = re.findall(r"\d+", str(checkpoint_path.name))
    read_only = False

    if dataset is not None:
        d = dataset.take(1)
        if any([isinstance(i, tf.sparse.SparseTensor) for i in d]):
            sparse = True

    try:
        step = int(found[-1])
    except IndexError:
        step = None
    except ValueError as e:
        raise Exception(f"{checkpoint_path=} {found=}") from e

    if autoencoder_class is AngleDihedralCartesianEncoderMap:
        p_class = ADCParameters
    else:
        p_class = Parameters

    if not parameters_file.is_file():
        parameters = p_class()
        print(
            f"Couldn't find the parameter's file at {parameters_file}. "
            f"Will use default {parameters.__class__.__name__} and will "
            f"infer architecture parameters from the model on disk. Weights "
            f"from the old model will be transferred to the new model. "
            f"From now on, you can save the model as a new .keras file."
        )
        read_only = True
    else:
        parameters = p_class.from_file(parameters_file)
        print(
            "Weights from the old model will be transferred to the new "
            "model. From now on, you can save the model as a new .keras file."
        )

    # set the current training step
    if parameters.current_training_step == 0 and step is not None:
        parameters.current_training_step = step

    # make assumptions on data based on input shape
    if autoencoder_class is AngleDihedralCartesianEncoderMap:
        old_model = tf.keras.models.load_model(checkpoint_path, compile=False)
        input_shape = old_model.input_shape
        encoder_input_shape = old_model.encoder_model.input_shape
    else:

        class OldModel:
            pass

        old_model = OldModel()
        assert len(checkpoint_path) == 2
        old_model.encoder = tf.keras.models.load_model(
            checkpoint_path[0], compile=False
        )
        old_model.decoder = tf.keras.models.load_model(
            checkpoint_path[1], compile=False
        )
        old_model.encoder_model = old_model.encoder
        old_model.decoder_model = old_model.decoder
        input_shape = old_model.encoder.input_shape[1]
        if dataset is not None:
            d = dataset.take(1)
            for a in d:
                break
            d = a
            if d[0].shape[1] != input_shape:
                if d[0].shape[1] * 2 == input_shape:
                    _change_setting_inform_user(
                        parameters,
                        "periodicity",
                        np.pi,
                        parameters_file=parameters_file,
                        compat=compat,
                    )
                else:
                    raise Exception(
                        f"The shape of the provided data {d[0].shape[1]} does not "
                        f"match the input shape {input_shape} of the network. Are you sure, you "
                        f"are loading the correct checkpoint?"
                    )
            else:
                _change_setting_inform_user(
                    parameters,
                    "periodicity",
                    float("inf"),
                    parameters_file=parameters_file,
                    compat=compat,
                )
        if not parameters_file.is_file():
            print(
                f"I will create a model with an input shape of {input_shape}. For "
                f"periodic data (e.g., angles), this network will not work. If you "
                f"are reloading a network for periodic data please manually "
                f"create a parameters file at {parameters_file} with the appropriate "
                f"periodicity."
            )

    if autoencoder_class is AngleDihedralCartesianEncoderMap:
        if len(encoder_input_shape) == 3:
            _change_setting_inform_user(
                parameters,
                "use_sidechains",
                True,
                parameters_file=parameters_file,
                compat=compat,
            )
            _change_setting_inform_user(
                parameters,
                "use_backbone_angles",
                True,
                parameters_file=parameters_file,
                compat=compat,
            )
        elif len(encoder_input_shape) == 2:
            _change_setting_inform_user(
                parameters,
                "use_sidechains",
                False,
                parameters_file=parameters_file,
                compat=compat,
            )
            _change_setting_inform_user(
                parameters,
                "use_backbone_angles",
                True,
                parameters_file=parameters_file,
                compat=compat,
            )
        else:
            _change_setting_inform_user(
                parameters,
                "use_sidechains",
                False,
                parameters_file=parameters_file,
                compat=compat,
            )
            _change_setting_inform_user(
                parameters,
                "use_backbone_angles",
                False,
                parameters_file=parameters_file,
                compat=compat,
            )

    # make some assumptions about the n_neurons and activation_function parameters
    n_neurons = []
    activation_functions = []
    for layer in old_model.encoder_model.layers:
        if hasattr(layer, "units"):
            n_neurons.append(layer.units)
            act = layer.activation.__name__
            if act == "linear":
                activation_functions.append("")
            else:
                activation_functions.append(act)
    activation_functions = [activation_functions[-1]] + activation_functions

    _change_setting_inform_user(
        parameters,
        "n_neurons",
        n_neurons,
        parameters_file=parameters_file,
        compat=compat,
    )
    _change_setting_inform_user(
        parameters,
        "activation_functions",
        activation_functions,
        parameters_file=parameters_file,
        compat=compat,
    )

    if autoencoder_class is AngleDihedralCartesianEncoderMap:
        new_model = gen_functional_model(
            input_shapes=tuple([v[1:] for v in old_model.input_shape]),
            parameters=parameters,
            sparse=sparse,
            write_summary=not read_only,
        )
    else:
        new_model = gen_sequential_model(
            input_shape=input_shape,
            parameters=parameters,
            sparse=sparse,
        )
    try:
        new_model.encoder_model.set_weights(old_model.encoder.get_weights())
    except AttributeError as e:
        new_model.encoder_model.set_weights(old_model.encoder_model.get_weights())
    except Exception as e:
        raise Exception(
            f"{[i.shape for i in new_model.encoder_model.get_weights()]=}\n\n"
            f"{[i.shape for i in old_model.encoder_model.get_weights()]=}"
        ) from e

    try:
        new_model.decoder_model.set_weights(old_model.decoder.get_weights())
    except AttributeError as e:
        new_model.decoder_model.set_weights(old_model.decoder_model.get_weights())
    except Exception as e:
        raise Exception(
            f"{[i.shape for i in new_model.decoder_model.get_weights()]=}\n\n"
            f"{[i.shape for i in old_model.decoder_model.get_weights()]=}"
        ) from e

    if autoencoder_class is AngleDihedralCartesianEncoderMap:
        new_class = autoencoder_class(
            trajs=trajs,
            parameters=parameters,
            model=new_model,
            read_only=read_only,
            dataset=dataset,
        )
        if not read_only:
            new_class.save()
        return new_class
    else:
        new_class = autoencoder_class(
            parameters=parameters,
            train_data=dataset,
            model=new_model,
            read_only=read_only,
            sparse=sparse,
        )
        if not read_only:
            new_class.save()
        return new_class


def load_model_legacy_dep(
    autoencoder_class: AutoencoderClass,
    checkpoint_path: Union[str, Path],
    read_only: bool = True,
    overwrite_tensorboard_bool: bool = False,
    trajs: Optional[TrajEnsemble] = None,
    sparse: bool = False,
    dataset: Optional[tf.data.Dataset] = None,
) -> AutoencoderClass:  # pragma: no doccheck
    """Reloads a tf.keras.Model from a checkpoint path.


    For this, an AutoencoderClass is necessary to provide the corresponding
    custom objects, such as loss functions.


    """
    basedir = os.path.split(checkpoint_path)[0]

    # remove wildcard
    if "*" in checkpoint_path:
        cp_path = checkpoint_path.replace("*", "")
    else:
        cp_path = checkpoint_path

    if trajs is None and dataset is None:
        params = Parameters.from_file(basedir + "/parameters.json")
        _params = copy.deepcopy(params)
        if overwrite_tensorboard_bool:
            params.tensorboard = False
        directory = run_path("/".join(checkpoint_path.split("/")[:-1]))
        if directory != params.main_path:
            print(
                f"The saved model files have been moved from {params.main_path} "
                f"to {directory}. I will overwrite the 'main_path' attribute of "
                f"these parameters."
            )
            params = deepcopy(params)
            params.main_path = run_path(directory)
            if not hasattr(params, "write_summary"):
                params.write_summary = params.tensorboard
        out = autoencoder_class(parameters=params, read_only=read_only)
    else:
        params = ADCParameters.from_file(basedir + "/parameters.json")
        _params = copy.deepcopy(params)
        if overwrite_tensorboard_bool:
            params.tensorboard = False
        if os.path.isfile(cp_path + "_current_step.txt"):
            with open(cp_path + "_current_step.txt", "r") as f:
                step = int(f.read())
        elif read_only:
            step = 0
        else:
            print(
                "Cannot find cartesian loss step. Retraining of this model might "
                "lead to unexpected results."
            )
        directory = run_path("/".join(checkpoint_path.split("/")[:-1]))
        if directory != params.main_path:
            print(
                f"The saved model files have been moved from {params.main_path} "
                f"to {directory}. I will overwrite the 'main_path' attribute of "
                f"these parameters."
            )
            params = deepcopy(params)
            params.main_path = directory
            if not hasattr(params, "write_summary"):
                params.write_summary = params.tensorboard
        out = autoencoder_class(
            trajs,
            parameters=params,
            read_only=read_only,
            dataset=dataset,
        )
    out.p = _params

    # see if there are multiple models
    if "*" not in checkpoint_path:
        models = glob.glob(checkpoint_path + "*/")
    else:
        models = glob.glob(checkpoint_path + "/")

    # three different ways of loading models
    if len(models) == 2:
        models.sort(key=_model_sort_key)
        custom_objects = {fn.__name__: fn for fn in out.loss}
        models = _load_list_of_models(models, custom_objects=custom_objects)
        n_inputs = models[0].inputs[0].shape[-1]
        if _params.periodicity < float("inf"):
            n_inputs = int(n_inputs / 2)
        model = SequentialModel(n_inputs, out.p, models)
    elif len(models) == 3:
        print("Loading a functional model can take up to a minute.")
        models.sort(key=_model_sort_key)
        encoder_model_name = models[0]
        custom_objects = {fn.__name__: fn for fn in out.loss}
        models = _load_list_of_models(models, custom_objects=custom_objects)
        model = models[2]
        model.encoder_model = models[0]
        model.decoder_model = models[1]

        msg = None
        if not _params.use_backbone_angles and not _params.use_sidechains:
            if len(models[0].input_shape) != 2:
                msg = (
                    f"Reloading the models seemed to have failed. I expected the "
                    f"Encoder model to take an input of shape (None, Any), but the "
                    f"file at {encoder_model_name} takes an input shape of "
                    f"{models[0].input_shape}. This error can also be caused by bad "
                    f"filenames."
                )
        elif _params.use_backbone_angles and not _params.use_sidechains:
            if len(models[0].input_shape) != 2:
                msg = (
                    f"Reloading the models seemed to have failed. I expected the "
                    f"Encoder model to take an input of shape [(None, Any), (None, Any)] but the "
                    f"file at {encoder_model_name} takes an input shape of "
                    f"{models[0].input_shape}. This error can also be caused by bad "
                    f"filenames."
                )
        else:
            if len(models[0].input_shape) != 3:
                msg = (
                    f"Reloading the models seemed to have failed. I expected the "
                    f"Encoder model to take an input of shape [(None, Any), (None, Any), (None, Any)] but the "
                    f"file at {encoder_model_name} takes an input shape of "
                    f"{models[0].input_shape}. This error can also be caused by bad "
                    f"filenames."
                )
        if msg is not None:
            raise Exception(msg)
    else:
        print("Model is neither Sequential, nor functional. I try to reload it.")
        custom_objects = {fn.__name__: fn for fn in out.loss}
        model = tf.keras.models.load_model(
            checkpoint_path, custom_objects=custom_objects
        )
        if hasattr(model, "encoder_model") and not hasattr(model, "encode"):
            print(
                "The loaded model lost its `encode` function. I will try to rebuild it."
            )

            models = [model.encoder_model, model.decoder_model]
            n_inputs = models[0].inputs[0].shape[-1]
            if _params.periodicity < float("inf"):
                n_inputs = int(n_inputs / 2)

            if sparse:
                # Third Party Imports
                from tensorflow.keras.layers import Dense, Input

                shape = n_inputs
                _input_layer = Input(
                    shape=(int(shape),),
                    sparse=True,
                )
                x = Dense(shape)(_input_layer)
                get_dense_model = tf.keras.Model(
                    inputs=_input_layer,
                    outputs=x,
                )
                model.get_dense_model = get_dense_model
            else:
                get_dense_model = None
            model = SequentialModel(
                n_inputs, out.p, models, sparse=sparse, get_dense_model=get_dense_model
            )

    out._model = model
    if os.path.isfile(cp_path + "_step.txt"):
        out.cartesian_loss_step = step + 1
    return out

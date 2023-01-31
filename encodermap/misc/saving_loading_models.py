# -*- coding: utf-8 -*-
# encodermap/misc/saving_loading_models.py
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

"""
ToDo:
    * This is in a desperate need of rework.

"""
from __future__ import annotations

import typing
from typing import Callable, Optional, Union

if typing.TYPE_CHECKING:
    from .._typing import AutoencoderClass

import copy
import glob
import os
from pathlib import Path

import tensorflow as tf

from ..misc.misc import _datetime_windows_and_linux_compatible
from ..models.models import ADCSequentialModel, FunctionalModel, SequentialModel
from ..parameters import ADCParameters, Parameters
from .errors import BadError


def save_model(model, main_path, inp_class_name, step=None, current_step=None):
    if step is None:
        time = _datetime_windows_and_linux_compatible()
        fname = f"{main_path}/saved_model_{time}.model"
    else:
        fname = f"{main_path}/saved_model_{step}.model"
    if len(model.layers) == 2:
        names = ["_" + l.name.lower() for l in model.layers]
        for i, submodel in enumerate(model.layers):
            tf.keras.models.save_model(submodel, fname + names[i])
        if step is None:
            print(f"Saved current state of model.")
            print(
                f"Use em.{inp_class_name}.from_checkpoint('{fname}*') to reload the current state of the two submodels."
            )
    elif issubclass(model.__class__, FunctionalModel):
        if step is None:
            print("Saving the ACD Functional Model can take up to a minute.")
        tf.keras.models.save_model(model, fname)
        tf.keras.models.save_model(model.encoder_model, fname + "_encoder")
        tf.keras.models.save_model(model.decoder_model, fname + "_decoder")
        if step is None:
            print(
                f"Saved current state of functional model at the end of step "
                f"{current_step}. Use em.{inp_class_name}.from_checkpoint('{fname}*') "
                f"to reload the current state."
            )
    else:
        print(
            f"Current model is not a subclass of Union[SequentialModel, "
            f"ACDSequentialModel, FunctionalModel]. I will try to save it at "
            f"{fname}, but can't guarantee that you can reload it."
        )
        tf.keras.models.save_model(model, fname)

    if current_step is not None:
        with open(fname + "_current_step.txt", "w") as f:
            f.write(str(current_step))


def model_sort_key(model_name: str) -> int:
    """Returns numerical values baed on whether `model_name` contains substrings.

    Args:
        model_name (str): The filepath to the saved model.

    Returns:
        int: Returns 0 for 'encoder', 1 for 'decoder', 2 for everything else.

    """
    x = Path(model_name).name
    return 0 if "encoder" in x else (1 if "decoder" in x else 2)


def load_list_of_models(
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


def load_model(
    autoencoder_class: AutoencoderClass,
    checkpoint_path: str,
    read_only: bool = True,
    overwrite_tensorboard_bool: bool = False,
    trajs: Optional[TrajEnsemble] = None,
    sparse: bool = False,
) -> AutoencoderClass:
    """Reloads a tf.keras.Model from a checkpoint path.


    For this, an AutoencoderClass is necessary, to provide the corresponding
    custom objects, such as loss functions.


    """
    basedir = os.path.split(checkpoint_path)[0]

    # remove wildcard
    if "*" in checkpoint_path:
        cp_path = checkpoint_path.replace("*", "")
    else:
        cp_path = checkpoint_path

    if trajs is None:
        params = Parameters.from_file(basedir + "/parameters.json")
        _params = copy.deepcopy(params)
        if overwrite_tensorboard_bool:
            params.tensorboard = False
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
            raise BadError(
                "Cannot find cartesian loss step. Model will not be trainable without knowing the step the model was saved at."
            )
        out = autoencoder_class(
            trajs, parameters=params, read_only=read_only, cartesian_loss_step=step
        )
    out.p = _params

    # see if there are multiple models
    if "*" not in checkpoint_path:
        models = glob.glob(checkpoint_path + "*/")
    else:
        models = glob.glob(checkpoint_path + "/")

    # three different ways of loading models
    if len(models) == 2:
        models.sort(key=model_sort_key)
        custom_objects = {fn.__name__: fn for fn in out.loss}
        models = load_list_of_models(models, custom_objects=custom_objects)
        n_inputs = models[0].inputs[0].shape[-1]
        if _params.periodicity < float("inf"):
            n_inputs = int(n_inputs / 2)
        model = SequentialModel(n_inputs, out.p, models)
    elif len(models) == 3:
        print("Loading a functional model can take up to a minute.")
        models.sort(key=model_sort_key)
        encoder_model_name = models[0]
        custom_objects = {fn.__name__: fn for fn in out.loss}
        models = load_list_of_models(models, custom_objects=custom_objects)
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

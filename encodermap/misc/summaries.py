# -*- coding: utf-8 -*-
# encodermap/misc/summaries.py
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

"""
Functions that write stuff to tensorboard. Mainly used for the image callbacks.
"""

################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import io
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

# Third Party Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


matplotlib.use("Agg")


################################################################################
# Globals
################################################################################


__all__ = ["add_layer_summaries", "image_summary"]


################################################################################
# Summary Functions
################################################################################


def add_layer_summaries(
    layer: tf.keras.layers.Layer,
    step: Optional[int] = None,
) -> None:
    """Adds summaries for a layer to Tensorboard.

    Args:
        layer (tf.keras.layers.Layer): The layer.
        step (Union[tf.Tensor, int, None], optional): The current step.
            Can be either a Tensor or None. Defaults to None.

    """
    weights = layer.variables[0]
    biases = layer.variables[1]
    if "encoder" in layer.name.lower():
        namescope = "Encoder"
    elif "decoder" in layer.name.lower():
        namescope = "Decoder"
    elif "latent" in layer.name.lower():
        namescope = "Latent"
    else:
        namescope = "InputOutputLayers"
    variable_summaries(namescope, layer.name + "/weights", weights, step)
    variable_summaries(namescope, layer.name + "/biases", biases, step)


def variable_summaries(
    namescope: str,
    name: str,
    variables: tf.Tensor,
    step: Optional[int] = None,
) -> None:
    """
    Attach several summaries to a Tensor for TensorBoard visualization.

    Args:
        namescope (str): The string to prepend to the layer names.
            Makes it easier to group the layers.
        name (str): The name of the layer.
        variables (tf.Tensor): The variables (weighhts, biases) of the layer.
        step (Union[tf.Tensor, int, None], optional): The current step.
            Can be either a Tensor or None. Defaults to None.

    """
    if not isinstance(variables, list):
        variables = [variables]

    for i, var in enumerate(variables):
        try:
            add_index = len(variables) > 1
        except TypeError:
            add_index = True
        if add_index:
            name = name + str(i)
        with tf.name_scope(namescope + "/" + name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean, step=step)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev, step=step)
            tf.summary.scalar("max", tf.reduce_max(var), step=step)
            tf.summary.scalar("min", tf.reduce_min(var), step=step)
            tf.summary.histogram("histogram", var, step=step)
            tf.compat.v1.summary.tensor_summary("values", var)


def image_summary(
    lowd: np.ndarray,
    step: Optional[int] = None,
    scatter_kws: Optional[dict[str, Any]] = None,
    hist_kws: Optional[dict[str, Any]] = None,
    additional_fns: Optional[Sequence[Callable]] = None,
) -> None:
    """Writes an image to Tensorboard.

    Args:
        lowd (np.ndarray): The data to plot. Usually that
            will be the output of the latent space of the Autoencoder.
            This array has to be of dimensionality 2 (rows and columns).
            The first two points of the rows will be used as xy coordinates
            in a scatter plot.
        step (Optional[int]): The training step under which you can find the
            image in tensorboard. Defaults to None.
        scatter_kws (Optional[dict[str, Any]]): A dictionary with keyword
            arguments to be passed to matplotlib.pyplot.scatter(). If None is
            provided, this dict: {'s': 20} will be used. Defaults to None.
        hist_kws (Optional[dict[str, Any]]): A dictionary with keyword arguments
            to be passed to matplotlib.pyplot.hist2d(). If None is provided, this
             dict: {'bins': 50} will be used. Defaults to None.
        additional_fns (Optional[Sequence[Callable]]): A sequence of functions that
            take the data of the latent space and return a tf.Tensor that can
            be logged to tensorboard with tf.summary.image().

    Raises:
        AssertionError: When lowd.ndim is not 2 and when len(lowd) != len(ids)

    """
    if scatter_kws is None:
        scatter_kws = {"s": 20}
    if hist_kws is None:
        hist_kws = {"bins": 50}
    if np.any(np.isnan(lowd)):
        image = _gen_nan_image()
        with tf.name_scope("Latent Scatter"):
            tf.summary.image(f"Latent at step {step}", image, step=step)
        return
    scatter_image = _gen_scatter(lowd[:, :2], scatter_kws)
    hist_image = _gen_hist(lowd[:, :2], hist_kws)
    with tf.name_scope("Latent Scatter"):
        tf.summary.image(f"Latent at step {step}", scatter_image, step=step)
    with tf.name_scope("Latent Density"):
        tf.summary.image(f"Latent at step {step}", hist_image, step=step)

    if additional_fns is not None:
        with tf.name_scope("User Provided Plotting Functions"):
            for i, fn in enumerate(additional_fns):
                tf.summary.image(
                    f"User Plotting {i} at step {step}", fn(lowd), step=step
                )


def _gen_hist(
    data: np.ndarray,
    hist_kws: dict[str, Any],
) -> tf.Tensor:
    """Creates matplotlib histogram and returns tensorflow Tensor that
    represents an image.

    Args:
        data (Union[np.ndarray, tf.Tensor]): The xy data to be used.
            `data.ndim` should be 2. 1st dimension the datapoints, 2nd dimension x, y.
        hist_kws (dict): Additional keywords to be passed to matplotlib.pyplot.hist2d().

    Returns:
        tf.Tensor: A tensorflow tensor that can be written to Tensorboard with tf.summary.image().

    """
    plt.close("all")
    matplotlib.use("Agg")  # overwrites current backend of notebook
    plt.figure()
    plt.hist2d(*data.T, **hist_kws)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), 4)
    image = tf.expand_dims(image, 0)
    return image


def _gen_nan_image() -> tf.Tensor:
    """Creates matplotlib image, with debug info.

    Returns:
        tf.Tensor: A tensorflow tensor that can be written to Tensorboard with tf.summary.image().

    """
    plt.close("all")
    matplotlib.use("Agg")  # overwrites current backend of notebook
    fig, ax = plt.subplots()
    ax.text(
        0.5,
        0.5,
        "Some data of lowd is nan",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), 4)
    image = tf.expand_dims(image, 0)
    return image


def _gen_scatter(
    data: np.ndarray,
    scatter_kws: dict[str, Any],
) -> tf.Tensor:
    """Creates matplotlib scatter plot and returns tensorflow Tensor that represents an image.

    Args:
        data (Union[np.ndarray, tf.Tensor]): The xy data to be used. data.ndim should be 2.
            1st dimension the datapoints, 2nd dimension x, y.
        scatter_kws (dict): Additional keywords to be passed to matplotlib.pyplot.scatter().

    Returns:
        tf.Tensor: A tensorflow tensor that can be written to Tensorboard with tf.summary.image().

    """
    plt.close("all")
    matplotlib.use("Agg")  # overwrites current backend of notebook
    plt.figure()
    # plt.plot([1, 2])
    if isinstance(data, np.ndarray):
        plt.scatter(*data.T, **scatter_kws)
    else:
        plt.scatter(*data.numpy().T, **scatter_kws)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), 4)
    image = tf.expand_dims(image, 0)
    return image

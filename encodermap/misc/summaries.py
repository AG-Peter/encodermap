# -*- coding: utf-8 -*-
# encodermap/misc/summaries.py
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
from typing import Any, Literal, Optional, Union

# Third Party Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from optional_imports import _optional_import

# Encodermap imports
from encodermap.plot.plotting import _plot_free_energy


################################################################################
# Optional Imports
################################################################################


px = _optional_import("plotly", "express")
go = _optional_import("plotly", "graph_objects")


################################################################################
# Globals
################################################################################


__all__: list[str] = ["add_layer_summaries", "image_summary"]


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


# def intermediate_summary(
#     highd_data: Union[np.ndarray, Sequence[np.ndarray]],
#     step: int,
#     model: tf.keras.Model,
#     parameters: "AnyParameters",
#     backend: Literal["matplotlib", "plotly"] = "matplotlib",
# ) -> dict[str, io.BytesIO]:
#     """Produces images of intermediate layers during training.
#
#     Args:
#         highd_data (Union[np.ndarray, Sequence[np.ndarray]]): The high dimensional
#             data to be used with the autoencoder model. If the autoencoder is
#             of type `Autoencoder`, `EncoderMap`, or `DihedralEncoderMap`, the
#             `highd_data` should be a single np.ndarray. For the `AngleDihedralCartesianEncderMap`,
#             the `highd_data` should be a Sequence of np.ndarray in the sequence
#             angles, dihedrals, cartesians, distance, (and maybe side_dihedrals,
#             depending on whether side_dihedrals are used for the training).
#         step (int): The current training ste.
#         model (tf.keras.Model): The autoencoder model.
#         parameters (AnyParameters): Either an instance of `encodermap.parameters.Parameters`, or
#             `encodermap.parameters.ADCParameters`, depending on which autoencoder model is used.
#             The parameters are used to fix periodic input data.
#         backend (Literal["matplotlib", "plotly"]: Which backend to use for
#             plotting. Defaults to 'matplotlib'.
#
#     """
#     if model.encoder_model.layers[0].name == "Encoder_0":
#         return _encodermap_intermediate_summary(
#             highd_data,
#             step,
#             model,
#             parameters,
#             backend,
#         )
#     else:
#         return _angledihedralcartesianencodermap_intermediate_summary(
#             highd_data,
#             step,
#             model,
#             parameters,
#             backend,
#         )
#
#
# def _layer_output_and_input_summary(
#     data: np.ndarray,
#     name: str,
#     step: int,
##     type: Literal["input", "layer"] = "layer",
#     backend: Literal["matplotlib", "plotly"] = "matplotlib",
# ) -> io.BytesIO:
#     buf = io.BytesIO()
#     raise Exception(f"Here")
#     if isinstance(data, tf.SparseTensor):
#         data = tf.sparse.to_dense(data, default_value=np.nan)
#     if isinstance(data, tf.Tensor):
#         data = data.numpy()
#     mean = np.mean(data, 0)
#     if backend == "plotly":
#         fig = px.bar(
#             x=mean,
#             y=np.arange(data.shape[1]),
#             orientation="h",
#             color=mean,
#             color_continuous_scale="Viridis",
#             labels={
#                 "x": f"mean value of {data.shape[0]} samples",
#                 "y": "feature" if type == "input" else "neuron",
#             },
#             width=500,
#             height=500,
#         )
#         fig.update_layout(
#             margin={"l": 0, "r": 0, "t": 0, "b": 0},
#             coloraxis_showscale=False,
#         )
#         fig.write_image(buf)
#     elif backend == "matplotlib":
#         raise NotImplementedError(f"{mean.shape=}")
#     else:
#         raise Exception(
#             f"Argument `backend` must be either 'plotly' or 'matplotlib'."
#         )
#     image = tf.image.decode_png(buf.getvalue(), 4)
#     image = tf.expand_dims(image, 0)
#     with tf.name_scope("Layer Outputs"):
#         tf.summary.image(name, image, step=step)
#     buf.seek(0)
#     return buf
#
#
# def _encodermap_intermediate_summary(
#     highd_data: Union[np.ndarray, Sequence[np.ndarray]],
#     step: int,
#     model: tf.keras.Model,
#     parameters: "AnyParameters",
#     backend: Literal["matplotlib", "plotly"] = "matplotlib",
# ) -> dict[str, io.BytesIO]:
#     # Local Folder Imports
#
#     out = {}
#     layers = []
#     for layer in model.encoder_model.layers:
#         layers.append(layer)
#     for layer in model.decoder_model.layers:
#         layers.append(layer)
#
#     if model.sparse:
#         input = model.get_dense_model(highd_data)
#     else:
#         input = highd_data
#
#     out["input"] = _layer_output_and_input_summary(
#         input, name="Input", step=step, type="input", backend=backend
#     )
#
#     if parameters.periodicity != float("inf"):
#         if parameters.periodicity != 2 * np.pi:
#             input = input / parameters.periodicity * 2 * np.pi
#         input = tf.concat([tf.sin(input), tf.cos(input)], 1)
#
#     for layer in layers[:-1]:
#         input = layer(input)
#         out[layer.name] = _layer_output_and_input_summary(input, layer.name, step=step, backend=backend)
#
#     # last layer needs to be treated individually because of maybe being periodic
#     input = layers[-1](input)
#     if parameters.periodicity != float("inf"):
#         input = tf.atan2(*tf.split(input, 2, 1))
#         if parameters.periodicity != 2 * np.pi:
#             input = input / (2 * np.pi) * p.periodicity
#         out[layer.name] = _layer_output_and_input_summary(
#             input, layers[-1].name, step=step, backend=backend
#         )
#     return out
#
#
# def _angledihedralcartesianencodermap_intermediate_summary(
#     highd_data: Union[np.ndarray, Sequence[np.ndarray]],
#     step: int,
#     model: tf.keras.Model,
#     parameters: "AnyParameters",
#     backend: Literal["matplotlib", "plotly"] = "matplotlib",
# ) -> dict[str, io.BytesIO]:
#     out = {}
#
#     if isinstance(highd_data, (list, tuple)):
#         if len(highd_data) == 1:
#             dihedrals = highd_data[0]
#             angles = None
#             side_dihedrals = None
#         elif len(highd_data) == 2:
#             angles, dihedrals = highd_data
#             side_dihedrals = None
#         elif len(highd_data) == 3:
#             angles, dihedrals, side_dihedrals = highd_data
#         else:
#             raise Exception(
#                 f"Can't construct intermediate layer outputs for ADCEMap when "
#                 f"provided high-dimensional contains more than 3 elements."
#             )
#     else:
#         dihedrals = highd_data
#         angles = None
#         side_dihedrals = None
#
#     layers = {}
#     for layer in model.encoder_model.layers:
#         layers[layer.name] = layer
#     for layer in model.decoder_model.layers:
#         layers[layer.name] = layer
#
#     # angles
#     if angles is not None:
#         layer = layers["input_angles_to_unit_circle"]
#         raise Exception(f"Here {layer=} {model.__class__.__name__=} {angles.shape=} {model.get_dense_model_central_angles.input_shape=}")
#         if model.__class__.__name__ == "ADCSparseFunctionalModel":
#             if model.get_dense_model_central_angles is not None:
#                 angles = model.get_dense_model_central_angles(
#                     angles
#                 )
#         raise Exception(f"Here")
#         angles = layer(angles)
#         out[layer.name] = _layer_output_and_input_summary(angles, layer.name, step=step, backend=backend)
#
#     raise Exception("Here")
#
#     # sidechain dihedrals
#     if side_dihedrals is not None:
#         layer = layers["input_side_dihedrals_to_unit_circle"]
#         out["input_side_dihedrals"] = _layer_output_and_input_summary(
#             side_dihedrals, name="Input Side Dihedrals", step=step, type="input"
#         )
#         if model.__class__.__name__ == "ADCSparseFunctionalModel":
#             if model.get_dense_model_side_dihedrals is not None:
#                 side_dihedrals = model.get_dense_model_side_dihedrals(
#                     side_dihedrals
#                 )
#         side_dihedrals = layer(side_dihedrals)
#         out[layer.name] = _layer_output_and_input_summary(
#             side_dihedrals, layer.name, step=step, backend=backend
#         )
#
#     layer = layers["input_central_dihedrals_to_unit_circle"]
#     out["input_dihedrals"] = _layer_output_and_input_summary(
#         dihedrals, name="Input Dihedrals", step=step, type="input"
#     )
#     if model.__class__.__name__ == "ADCSparseFunctionalModel":
#         if model.get_dense_model_central_dihedrals is not None:
#             dihedrals = model.get_dense_model_side_dihedrals(
#                 dihedrals
#             )
#     dihedrals = layer(dihedrals)
#     out[layer.name] = _layer_output_and_input_summary(dihedrals, layer.name, step=step, backend=backend)
#
#     # concatenate
#     if angles is not None and side_dihedrals is not None:
#         layer = layers["concatenate_angular_inputs"]
#         input = layer((angles, dihedrals, side_dihedrals))
#         splits = [angles.shape[1], dihedrals.shape[1], side_dihedrals.shape[1]]
#     elif angles is not None and side_dihedrals is None:
#         layer = layers["concatenate_angular_inputs"]
#         input = layer((angles, dihedrals))
#         splits = [angles.shape[1], dihedrals.shape[1]]
#     else:
#         input = dihedrals
#         splits = [dihedrals.shape[1]]
#     out[layer.name] = _layer_output_and_input_summary(input, layer.name, step=step, backend=backend)
#
#     # Encoder
#     i = 0
#     while True:
#         try:
#             layer = layers[f"Encoder_{i}"]
#         except KeyError:
#             break
#         input = layer(input)
#         out[layer.name] = _layer_output_and_input_summary(input, layer.name, step=step, backend=backend)
#         i += 1
#
#     # Decoder
#     i = 0
#     while True:
#         try:
#             layer = layers[f"Decoder_{i}"]
#         except KeyError:
#             break
#         input = layer(input)
#         out[layer.name] = _layer_output_and_input_summary(input, layer.name, step=step, backend=backend)
#         i += 1
#
#     # split
#     if angles is not None and side_dihedrals is not None:
#         input = tf.split(input, splits, 1)
#         angles, dihedrals, side_dihedrals = input
#     elif angles is not None and side_dihedrals is None:
#         input = tf.split(input, splits, 1)
#         angles, dihedrals = input
#     else:
#         dihedrals = input
#
#     # rejig
#     if angles is not None:
#         layer = layers["angles_from_unit_circle"]
#         angles = layer(angles)
#         out["output_angles"] = _layer_output_and_input_summary(
#             angles, name="Output Angles", step=step, backend=backend
#         )
#
#     # sidechain dihedrals
#     if side_dihedrals is not None:
#         layer = layers["side_dihedrals_from_unit_circle"]
#         side_dihedrals = layer(side_dihedrals)
#         out["output_side_dihedrals"] = _layer_output_and_input_summary(
#             side_dihedrals, name="Output Side Dihedrals", step=step, backend=backend
#         )
#
#     layer = layers["dihedrals_from_unit_circle"]
#     dihedrals = layer(dihedrals)
#     out["output_dihedrals"] = _layer_output_and_input_summary(
#         dihedrals, name="Output Dihedrals", step=step, backend=backend
#     )
#     return out


def image_summary(
    lowd: np.ndarray,
    step: Optional[int] = None,
    scatter_kws: Optional[dict[str, Any]] = None,
    hist_kws: Optional[dict[str, Any]] = None,
    additional_fns: Optional[Sequence[Callable]] = None,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
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
        scatter_kws (Optional[dict[str, Any]]): A dict with items that
            `plotly.express.scatter()` will accept. If None is provided,
            a dict with size 20 will be passed to
            `px.scatter(**{'size_max': 10, 'opacity': 0.2})`,
            which sets an appropriate size of scatter points for the size of
            datasets encodermap is usually used for.
        hist_kws (Optional[dict[str, Any]]): A dict with items that
            `encodermap.plot.plotting._plot_free_energy()` will accept. If None is provided a
            dict with bins 50 will be passed to
            `encodermap.plot.plotting._plot_free_energy(**{'bins': 50})`.
            You can choose a colormap here by providing `{'bins': 50, 'cmap':
            'plasma'}` for this argument.
        additional_fns (Optional[Sequence[Callable]]): A sequence of functions that
            take the data of the latent space and return a tf.Tensor that can
            be logged to tensorboard with tf.summary.image().
        backend (Literal["matplotlib", "plotly"]: Which backend to use for
            plotting. Defaults to 'matplotlib'.

    Raises:
        AssertionError: When lowd.ndim is not 2 and when len(lowd) != len(ids)

    """
    if backend == "plotly":
        if scatter_kws is None:
            scatter_kws = {"size_max": 1, "opacity": 0.2}
        if hist_kws is None:
            hist_kws = {"bins": 50}
    elif backend == "matplotlib":
        if scatter_kws is None:
            scatter_kws = {"s": 20}
        if hist_kws is None:
            hist_kws = {"bins": 50}
    else:
        raise Exception(f"Argument `backend` must be either 'plotly' or 'matplotlib'.")
    if np.any(np.isnan(lowd)):
        if backend == "plotly":
            image = _gen_nan_image_plotly()
        else:
            image = _gen_nan_image_matplotlib()
        with tf.name_scope("Latent Scatter"):
            tf.summary.image(f"Latent at step {step}", image, step=step)
        return
    if backend == "plotly":
        scatter_image = _gen_scatter_plotly(lowd[:, :2], scatter_kws)
        hist_image = _gen_hist_plotly(lowd[:, :2], hist_kws)
    else:
        scatter_image = _gen_scatter_matplotlib(lowd[:, :2], scatter_kws)
        hist_image = _gen_hist_matplotlib(lowd[:, :2], hist_kws)
    with tf.name_scope("Latent Output"):
        tf.summary.image(f"Latent Scatter", scatter_image, step=step)
        tf.summary.image(f"Latent Density", hist_image, step=step)

    if additional_fns is not None:
        with tf.name_scope("User Provided Plotting Functions"):
            for i, fn in enumerate(additional_fns):
                tf.summary.image(f"User Plotting {i}", fn(lowd), step=step)


def _gen_hist_matplotlib(
    data: np.ndarray,
    hist_kws: dict[str, Any],
) -> tf.Tensor:
    """Creates matplotlib histogram and returns tensorflow Tensor that represents an image.

    Args:
        data (Union[np.ndarray, tf.Tensor]): The xy data to be used. data.ndim should be 2.
            1st dimension the datapoints, 2nd dimension x, y.
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


def _gen_hist_plotly(
    data: np.ndarray,
    hist_kws: dict[str, Any],
) -> tf.Tensor:
    """Creates matplotlib histogram and returns tensorflow Tensor that
    represents an image.

    Args:
        data (Union[np.ndarray, tf.Tensor]): The xy data to be used.
            `data.ndim` should be 2. 1st dimension the datapoints, 2nd dimension x, y.
        hist_kws (Optional[dict[str, Any]]): A dict with items that
            `encodermap.plot.plotting._plot_free_energy()` will accept. If None is provided a
            dict with bins 50 will be passed to
            `encodermap.plot.plotting._plot_free_energy(**{'bins': 50})`.
            You can choose a colormap here by providing `{'bins': 50, 'cmap':
            'plasma'}` for this argument.

    Returns:
        tf.Tensor: A tensorflow tensor that can be written to Tensorboard with tf.summary.image().

    """
    trace = _plot_free_energy(
        x=data[:, 0],
        y=data[:, 1],
        **hist_kws,
    )
    fig = go.Figure(
        data=[
            trace,
        ],
        layout={
            "width": 500,
            "height": 500,
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        },
    )
    buf = io.BytesIO()
    fig.write_image(buf)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), 4)
    image = tf.expand_dims(image, 0)
    return image


def _gen_nan_image_plotly() -> tf.Tensor:
    """Creates matplotlib image, with debug info.

    Returns:
        tf.Tensor: A tensorflow tensor that can be written to Tensorboard with tf.summary.image().

    """
    fig = go.Figure(
        layout={
            "height": 500,
            "width": 500,
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        }
    )
    fig.add_annotation(
        x=2.5,
        y=1.5,
        text="Some data of lowd is nan",
        showarrow=False,
        font={"size": 36},
    )
    buf = io.BytesIO()
    fig.write_image(buf)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), 4)
    image = tf.expand_dims(image, 0)
    return image


def _gen_nan_image_matplotlib() -> tf.Tensor:
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


def _gen_scatter_plotly(
    data: np.ndarray,
    scatter_kws: dict[str, Any],
) -> tf.Tensor:
    """Creates matplotlib scatter plot and returns tensorflow Tensor that represents an image.

    Args:
        data (Union[np.ndarray, tf.Tensor]): The xy data to be used. data.ndim should be 2.
            1st dimension the datapoints, 2nd dimension x, y.
        scatter_kws (Optional[dict[str, Any]]): A dict with items that
            `plotly.express.scatter()` will accept. If None is provided,
            a dict with size 20 will be passed to
            `px.scatter(**{'size_max': 10, 'opacity': 0.2})`,
            which sets an appropriate size of scatter points for the size of
            datasets encodermap is usually used for.

    Returns:
        tf.Tensor: A tensorflow tensor that can be written to Tensorboard with tf.summary.image().

    """
    if not isinstance(data, np.ndarray):
        data = data.numpy()
    fig = px.scatter(
        x=data[:, 0],
        y=data[:, 1],
        height=500,
        width=500,
        labels={"x": "", "y": ""},
        **scatter_kws,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_showscale=False,
        showlegend=False,
    )
    buf = io.BytesIO()
    fig.write_image(buf)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), 4)
    image = tf.expand_dims(image, 0)
    return image


def _gen_scatter_matplotlib(
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

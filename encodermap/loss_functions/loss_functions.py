# -*- coding: utf-8 -*-
# encodermap/loss_functions/loss_functions.py
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
"""Loss functions for encodermap

ToDo:
    * Debug Autograph for distance cost
    * WARNING: AutoGraph could not transform <function sigmoid_loss at 0x00000264AB761040> and will run it as-is.
    * Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    * Cause: module 'gast' has no attribute 'Index'
    * To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert

"""
##############################################################################
# Imports
##############################################################################

import tensorflow as tf
import tensorflow.keras.backend as K

from ..encodermap_tf1.misc import distance_cost
from ..misc.distances import (
    pairwise_dist,
    pairwise_dist_periodic,
    periodic_distance,
    sigmoid,
)
from ..parameters.parameters import ADCParameters, Parameters, ParametersFramework

##############################################################################
# Globals
##############################################################################

__all__ = [
    "reconstruction_loss",
    "auto_loss",
    "center_loss",
    "regularization_loss",
    "loss_combinator",
    "distance_loss",
    "cartesian_loss",
    "cartesian_distance_loss",
    "angle_loss",
    "dihedral_loss",
]

##############################################################################
# Functions for tf.cond
# Don't know if this is really faster than logging every step to tensorboard
##############################################################################


def _do_nothing(*args):
    """This function does nothing. One of the functions provided to tf.cond."""
    pass


def _summary_cost(name, cost):
    """This functions logs a scalar to a name. One of the functions provided to tf.cond."""
    tf.summary.scalar(name, cost)


##############################################################################
# Legacy Code to make some tests
##############################################################################


def old_distance_loss(model, parameters=None):
    # choose parameters
    if parameters is None:
        p = ParametersFramework(Parameters.defaults)
    else:
        p = parameters
    # check Layers
    if len(model.layers) == 2:
        # sequential API
        latent = model.encoder
    else:
        # functional API
        latent = model.encoder

    # closure
    def loss(y_true, y_pred=None, step=None):
        loss.name = "distance_loss"
        y_pred = latent(y_true, training=True)
        # print(f'For testing: Loss latent of model to test em.encodermap_tf1.misc: {y_pred}')
        if p.distance_cost_scale is not None:
            dist_cost = distance_cost(
                y_true, y_pred, *p.dist_sig_parameters, p.periodicity
            )
            if p.distance_cost_scale != 0:
                dist_cost *= p.distance_cost_scale
        else:
            dist_cost = 0
        tf.summary.scalar("Distance Cost", dist_cost)
        return dist_cost

    return loss


##############################################################################
# Public Functions
##############################################################################


def basic_loss_combinator(*losses):
    """Calculates the sum of a list of losses and returns a combined loss.

    The basic loss combinator does not write to summary. Can be used for debugging.

    """

    def loss(y_true, y_pred=None):
        return sum([loss(y_true, y_pred) for loss in losses])

    return loss


def loss_combinator(*losses):
    """Calculates the sum of a list of losses and returns a combined loss.

    Args:
        *losses: Variable length argument list of loss functions.

    Returns:
        function: A combined loss function that can be used in custom training or with model.fit()

    Example:
        >>> import encodermap as em
        >>> from encodermap import loss_functions
        >>> import tensorflow as tf
        >>> import numpy as np
        >>> tf.random.set_seed(1) # fix random state to pass doctest :)

        >>> model = tf.keras.Sequential([
        ...     tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(), activation='relu'),
        ...     tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.l2(), activation='relu'),
        ...     tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(), activation='relu')
        ... ])

        >>> # Set up losses and bundle them using the loss combinator
        >>> auto_loss = loss_functions.auto_loss(model)
        >>> reg_loss = loss_functions.regularization_loss(model)
        >>> loss = loss_functions.loss_combinator(auto_loss, reg_loss)

        >>> # Compile model, model.fit() usually takes a tuple of (data, classes) but in
        >>> # regression learning the data needs to be provided twice. That's why we use fit(data, data)
        >>> model.compile(tf.keras.optimizers.Adam(), loss=loss)
        >>> data = np.random.random((100, 100))
        >>> history = model.fit(data, data, verbose=0)
        >>> tf.random.set_seed(None) # reset seed

        >>> # This weird contraption is also there to make the output predictable and pass tests
        >>> # Somehow the tf.random.seed(1) does not work here. :(
        >>> loss = history.history['loss'][0]
        >>> print(loss) # doctest: +SKIP
        {'loss': array([2.6])}
        >>> print(type(loss))
        <class 'float'>

    """

    def combined_loss_func(y_true, y_pred=None):
        cost = sum([loss(y_true, y_pred) for loss in losses])
        tf.summary.scalar("Combined Cost", cost)
        return cost

    return combined_loss_func


def distance_loss(model, parameters=None, callback=None):
    """Encodermap distance_loss

    Transforms space using sigmoid function first proposed by sketch-map.

    Args:
        model (tf.keras.Model): A model you want to use the loss function on.
        parameters (Union[encodermap.Parameters, None], optional): The parameters. If None is
                provided default values (check them with print(em.Parameters.defaults_description()))
                are used. Defaults to None.

    Note:
        If the model contains two layers. The first layer will be assumed to be the decoder.
        If the model contains more layers, one layer needs to be named 'latent' (case insensitive).

    Raises:
        Exception: When no bottleneck/latent layer can be found in the model.

    Returns:
        function: A loss function.

    References::

        @article{ceriotti2011simplifying,
          title={Simplifying the representation of complex free-energy landscapes using sketch-map},
          author={Ceriotti, Michele and Tribello, Gareth A and Parrinello, Michele},
          journal={Proceedings of the National Academy of Sciences},
          volume={108},
          number={32},
          pages={13023--13028},
          year={2011},
          publisher={National Acad Sciences}
        }

    """
    # choose parameters
    if parameters is None:
        p = ParametersFramework(Parameters.defaults)
    else:
        p = parameters

    # check Layers
    if len(model.layers) == 2:
        # sequential API
        latent = model.encoder
    else:
        # functional API
        latent = model.encoder
        # functional API without multiple models to be removed
        # layer_index = [layer.name.lower() for layer in model.layers].index('latent')
        # latent = tf.keras.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)

    if callback is None:
        write_bool = K.constant(False, "bool", name="log_bool")
    else:
        write_bool = callback.log_bool

    # define dist loss
    dist_loss = sigmoid_loss(p)

    # closure
    def distance_loss_func(y_true, y_pred=None):
        """y_true can be whatever input you like, dihedrals, angles, pairwise dist, contact maps. That will be
        transformed with Sketchmap's sigmoid function, as will the output of the latent layer of the autoencoder.
        the difference of these two will result in a loss function."""
        distance_loss_func.name = "distance_loss"
        y_pred = latent(y_true, training=True)
        # functional model gives a tuple
        if isinstance(y_true, tuple):
            y_true = tf.concat(y_true, axis=1)
        if p.distance_cost_scale is not None:
            dist_cost = dist_loss(y_true, y_pred)
            if p.distance_cost_scale != 0:
                dist_cost *= p.distance_cost_scale
        else:
            dist_cost = 0
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost("Distance Cost", dist_cost),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return dist_cost

    return distance_loss_func


def sigmoid_loss(
    parameters=None, periodicity_overwrite=None, dist_dig_parameters_overwrite=None
):
    """Sigmoid loss closure for use in distance cost and cartesian distance cost.

    Outer function prepares callable sigmoid. Sigmoid can then be called with just y_true and y_pred.

    Args:
        parameters (Union[encodermap.Parameters, None], optional): The parameters. If None is
                provided default values (check them with print(em.Parameters.defaults_description()))
                are used. Defaults to None.
        periodicity overwrite(Union[float, None]), optional): Cartesian distance cost is always non-periodic.
            To make sure no periodicity is applied to the data, set periodicity_overwrite to float('inf'). If
            None is provided the periodicity of the parameters class (default 2*pi) will be used.
            Defaults to None.

    Returns:
        function: A function that takes y_true and y_pred. Both need to be of the same shape.

    """
    if parameters is None:
        p = ParametersFramework(Parameters.defaults)
    else:
        p = parameters

    if periodicity_overwrite is not None:
        periodicity = periodicity_overwrite
    else:
        periodicity = p.periodicity

    if dist_dig_parameters_overwrite is not None:
        dist_sig_parameters = dist_dig_parameters_overwrite
    else:
        dist_sig_parameters = p.dist_sig_parameters

    # @tf.autograph.experimental.do_not_convert
    def sigmoid_loss_func(y_true, y_pred):
        r_h = y_true
        r_l = y_pred
        if periodicity == float("inf"):
            dist_h = pairwise_dist(r_h)
        else:
            dist_h = pairwise_dist_periodic(r_h, periodicity)
        dist_l = pairwise_dist(r_l)

        sig_h = sigmoid(*dist_sig_parameters[:3])(dist_h)
        sig_l = sigmoid(*dist_sig_parameters[3:])(dist_l)

        cost = tf.reduce_mean(tf.square(sig_h - sig_l))
        return cost

    return sigmoid_loss_func


def center_loss(model, parameters=None, callback=None):
    """Encodermap center_loss

    Use in custom training loops or in model.fit() training.

    Args:
        model (tf.keras.Model): A model you want to use the loss function on.
        parameters (Union[encodermap.Parameters, None], optional): The parameters. If None is
                provided default values (check them with print(em.Parameters.defaults_description()))
                are used. Defaults to None.

    Note:
        If the model contains two layers. The first layer will be assumed to be the decoder.
        If the model contains more layers, one layer needs to be named 'latent' (case insensitive).

    Raises:
        Exception: When no bottleneck/latent layer can be found in the model.

    Returns:
        function: A loss function.

    """
    # choose parameters
    if parameters is None:
        p = ParametersFramework(Parameters.defaults)
    else:
        p = parameters
    # check Layers
    if len(model.layers) == 2:
        # sequential API
        latent = model.encoder
    else:
        # functional API
        latent = model.encoder

    if callback is None:
        write_bool = K.constant(False, "bool", name="log_bool")
    else:
        write_bool = callback.log_bool

    # closure
    def center_loss_func(y_true, y_pred=None):
        """y_true will not be used in this loss function. y_pred can be supplied, but if None will be taken from the
        latent layer. This loss function tries to center the points in the latent layer."""
        center_loss_func.name = "center_loss"
        y_pred = latent(y_true, training=True)
        # functional model gives a tuple
        if isinstance(y_true, tuple):
            y_true = tf.concat(y_true, axis=1)
        # center cost
        # this is still a bit finicky
        # needs to have tf.GradentTape() context manager to watch a single layer
        if p.center_cost_scale is not None:
            # custom model dep code
            # center_cost = tf.reduce_mean(tf.square(model.get_layer('Encoder')(y_true)))
            # sequential model
            # get the output according to https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
            # this can be omitted by implementing a custom metric. Todo.
            center_cost = tf.reduce_mean(tf.square(y_pred))
            if p.center_cost_scale != 0:
                center_cost *= p.center_cost_scale
        else:
            center_cost = 0
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost("Center Cost", center_cost),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return center_cost

    return center_loss_func


def regularization_loss(model, parameters=None, callback=None):
    """Regularization loss of arbitrary tf.keras.Model

    Use in custom training loops or in model.fit() training.
    Loss is obtained as tf.math.add_n(model.losses)

    Args:
        model (tf.keras.Model): A model you want to use the loss function on.

    Returns:
        function: A loss function.

    """
    if parameters is None:
        p = ParametersFramework(Parameters.defaults)
    else:
        p = parameters

    if callback is None:
        write_bool = K.constant(False, "bool", name="log_bool")
    else:
        write_bool = callback.log_bool

    def regularization_loss_func(y_true=None, y_pred=None):
        """y_true and y_pred will not be considered here, because the regularization loss is accessed via model.losses."""
        regularization_loss.name = "regularization_loss"
        reg_loss = tf.math.add_n(model.losses)
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost("Regularization Cost", reg_loss),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return reg_loss

    return regularization_loss_func


def reconstruction_loss(model):
    """Simple Autoencoder recosntruction loss.

    Use in custom training loops or in model.fit training.

    Args:
        model (tf.keras.Model): A model you want to use the loss function on.

    Returns:
        function: A loss function to be used in custom training or model.fit.
            Function takes the following arguments:
            y_true (tf.Tensor): The true tensor.
            y_pred (tf.Tensor, optional): The output tensor. If not supplied
                the model will be called to get this tensor. Defaults to None.
            step (int): A step for tensorboard callbacks. Defaults to None.

    Examples:
        >>> import tensorflow as tf
        >>> import encodermap as em
        >>> from encodermap import loss_functions
        >>> model = tf.keras.Model()
        >>> loss = loss_functions.reconstruction_loss(model)
        >>> x = tf.random.normal(shape=(10, 10))
        >>> loss(x, x).numpy()
        0.0

    """

    def reconstruction_loss_func(y_true, y_pred=None):
        # if y_pred is None, this function is used in custom training
        # and should use model to get the output
        if y_pred is None:
            y_pred = model(y_true)
        # calculate error
        reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true)))
        return reconstruction_error

    return reconstruction_loss_func


def auto_loss(model, parameters=None, callback=None):
    """Encodermap auto_loss.

    Use in custom training loops or in model.fit() training.

    Args:
        model (tf.keras.Model): A model you want to use the loss function on.
        parameters (Union[encodermap.Parameters, None], optional): The parameters. If None is
                provided default values (check them with print(em.Parameters.defaults_description()))
                are used. Defaults to None.

    Returns:
        function: A loss function.

    """
    if parameters is None:
        p = ParametersFramework(Parameters.defaults)
    else:
        p = parameters

    if callback is None:
        write_bool = K.constant(False, "bool", name="log_bool")
    else:
        write_bool = callback.log_bool

    def auto_loss_func(y_true, y_pred=None):
        """y_true is complete model input, y_pred is complete model output. Because here it is not intended to unpack
        the output into dihedrals and angles, y_pred can be None and will be directly taken from the model."""
        auto_loss_func.name = "auto_loss"

        if y_pred is None:
            y_pred = model(y_true)
        if p.auto_cost_scale is not None:
            if p.auto_cost_variant == "mean_square":
                auto_cost = tf.reduce_mean(
                    tf.square(periodic_distance(y_true, y_pred, p.periodicity))
                )
            elif p.auto_cost_variant == "mean_abs":
                auto_cost = tf.reduce_mean(
                    tf.abs(periodic_distance(y_true, y_pred, p.periodicity))
                )
            elif p.auto_cost_variant == "mean_norm":
                auto_cost = tf.reduce_mean(
                    tf.norm(periodic_distance(y_true, y_pred, p.periodicity), axis=1)
                )
            else:
                raise ValueError(
                    "auto_cost_variant {} not available".format(p.auto_cost_variant)
                )
            if p.auto_cost_scale != 0:
                auto_cost *= p.auto_cost_scale
        else:
            auto_cost = 0
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost("Auto Cost", auto_cost),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return auto_cost

    return auto_loss_func


def dihedral_loss(model, parameters=None, callback=None):
    """Encodermap dihedral loss. Calculates distances between true and predicted dihedral angles.

    Respects periodicity in a [-a, a] interval if the provided parameters have a periodicity of 2 * a.

    Note:
        The interval should be (-a, a], but due to floating point precision we can't make this
        distinction here.

    Args:
        model (tf.keras.Model): The model to use the loss function on.
        parameters (Union[encodermap.ADCParameters, None], optional): The parameters. If None is
            provided default values (check them with print(em.ADCParameters.defaults_description()))
            are used. Defaults to None.

    Returns:
        function: A loss function. Can be used in either custom training or model.fit().

    """
    if parameters is None:
        p = ParametersFramework(ADCParameters.defaults)
    else:
        p = parameters

    if callback is None:
        write_bool = K.constant(False, "bool", name="log_bool")
    else:
        write_bool = callback.log_bool

    # closure
    def dihedral_loss_func(y_pred, y_true=None):
        """y_pred should be model input dihedrals, y_true should be model output dihedrals."""
        dihedral_loss_func.name = "dihedral_loss"
        if p.dihedral_cost_scale is not None:
            if p.dihedral_cost_variant == "mean_square":
                dihedral_cost = tf.reduce_mean(
                    tf.square(periodic_distance(y_true, y_pred, p.periodicity))
                )
            elif p.dihedral_cost_variant == "mean_abs":
                dihedral_cost = tf.reduce_mean(
                    tf.abs(periodic_distance(y_true, y_pred, p.periodicity))
                )
            elif p.dihedral_cost_variant == "mean_norm":
                dihedral_cost = tf.reduce_mean(
                    tf.norm(periodic_distance(y_true, y_pred, p.periodicity), axis=1)
                )
            else:
                raise ValueError(
                    "dihedral_cost_variant {} not available".format(p.auto_cost_variant)
                )
            dihedral_cost /= p.dihedral_cost_reference
            if p.dihedral_cost_scale != 0:
                dihedral_cost *= p.dihedral_cost_scale
        else:
            dihedral_cost = 0
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost("Dihedral Cost", dihedral_cost),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return dihedral_cost

    return dihedral_loss_func


def side_dihedral_loss(model, parameters=None, callback=None):
    """Encodermap sidechain dihedral loss. Calculates distances between true and predicted sidechain dihedral angles.

    Respects periodicity in a [-a, a] interval if the provided parameters have a periodicity of 2 * a.

    Note:
        The interval should be (-a, a], but due to floating point precision we can't make this
        distinction here.

    Args:
        model (tf.keras.Model): The model to use the loss function on.
        parameters (Union[encodermap.ADCParameters, None], optional): The parameters. If None is
            provided default values (check them with print(em.ADCParameters.defaults_description()))
            are used. Defaults to None.

    Returns:
        function: A loss function. Can be used in either custom training or model.fit().

    """
    if parameters is None:
        p = ParametersFramework(ADCParameters.defaults)
    else:
        p = parameters

    if callback is None:
        write_bool = K.constant(False, "bool", name="log_bool")
    else:
        write_bool = callback.log_bool

    # closure
    def side_dihedral_loss_func(y_pred, y_true=None):
        """y_pred should be model input side dihedrals, y_true should be model output side dihedrals."""
        side_dihedral_loss_func.name = "side_dihedral_loss"
        if p.side_dihedral_cost_scale is not None:
            if p.side_dihedral_cost_variant == "mean_square":
                side_dihedral_cost = tf.reduce_mean(
                    tf.square(periodic_distance(y_true, y_pred, p.periodicity))
                )
            elif p.side_dihedral_cost_variant == "mean_abs":
                side_dihedral_cost = tf.reduce_mean(
                    tf.abs(periodic_distance(y_true, y_pred, p.periodicity))
                )
            elif p.side_dihedral_cost_variant == "mean_norm":
                side_dihedral_cost = tf.reduce_mean(
                    tf.norm(periodic_distance(y_true, y_pred, p.periodicity), axis=1)
                )
            else:
                raise ValueError(
                    "dihedral_cost_variant {} not available".format(p.auto_cost_variant)
                )
            side_dihedral_cost /= p.side_dihedral_cost_reference
            if p.side_dihedral_cost_scale != 0:
                side_dihedral_cost *= p.side_dihedral_cost_scale
        else:
            side_dihedral_cost = 0
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost(
                "Sidechain Dihedral Cost", side_dihedral_cost
            ),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return side_dihedral_cost

    return side_dihedral_loss_func


def angle_loss(model, parameters=None, callback=None):
    """Encodermap angle loss. Calculates distances between true and predicted angles.

    Respects periodicity in a [-a, a] interval if the provided parameters have a periodicity of 2 * a.

    Note:
        The interval should be (-a, a], but due to floating point precision we can't make this
        distinction here.

    Args:
        model (tf.keras.Model): The model to use the loss function on.
        parameters (Union[encodermap.ADCParameters, None], optional): The parameters. If None is
            provided default values (check them with print(em.ADCParameters.defaults_description()))
            are used. Defaults to None.

    Returns:
        function: A loss function. Can be used in either custom training or model.fit().

    """
    if parameters is None:
        p = ParametersFramework(ADCParameters.defaults)
    else:
        p = parameters

    if callback is None:
        write_bool = K.constant(False, "bool", name="log_bool")
    else:
        write_bool = callback.log_bool

    # closure
    def angle_loss_func(y_pred, y_true=None):
        """y_true should be input angles. y_pred should be output angles (either from mean input angles or, when
        ADCParameters.use_backbone_angles == True, directly from model output)."""
        angle_loss_func.name = "angle_loss"
        if p.angle_cost_scale is not None:
            if p.angle_cost_variant == "mean_square":
                angle_cost = tf.reduce_mean(
                    tf.square(periodic_distance(y_true, y_pred, p.periodicity))
                )
            elif p.angle_cost_variant == "mean_abs":
                angle_cost = tf.reduce_mean(
                    tf.abs(periodic_distance(y_true, y_pred, p.periodicity))
                )
            elif p.angle_cost_variant == "mean_norm":
                angle_cost = tf.reduce_mean(
                    tf.norm(periodic_distance(y_true, y_pred, p.periodicity), axis=1)
                )
            else:
                raise ValueError(
                    "angle_cost_variant {} not available".format(p.auto_cost_variant)
                )
            angle_cost /= p.angle_cost_reference
            if p.angle_cost_scale != 0:
                angle_cost *= p.angle_cost_scale
        else:
            angle_cost = 0
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost("Angle Cost", angle_cost),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return angle_cost

    return angle_loss_func


def cartesian_distance_loss(model, parameters=None, callback=None):
    """Encodermap cartesian distance loss. Calculates sigmoid-weighted distances between pairwise cartesians and latent.

    Uses sketch-map's sigmoid function to transform the high-dimensional space of the input and the
    low-dimensional space of latent.

    Make sure to provide the pairwise cartesian distances. The output of the latent will be compared to the input.

    Note:
        If the model contains two layers. The first layer will be assumed to be the decoder.
        If the model contains more layers, one layer needs to be named 'latent' (case insensitive).

    Args:
        model (tf.keras.Model): The model to use the loss function on.
        parameters (Union[encodermap.ADCParameters, None], optional): The parameters. If None is
            provided default values (check them with print(em.ADCParameters.defaults_description()))
            are used. Defaults to None.

    Returns:
        function: A loss function. Can be used in either custom training or model.fit().

    """
    if parameters is None:
        p = ParametersFramework(ADCParameters.defaults)
    else:
        p = parameters

    if callback is None:
        write_bool = K.constant(False, "bool", name="log_bool")
    else:
        write_bool = callback.log_bool

    dist_loss = sigmoid_loss(
        p,
        periodicity_overwrite=float("inf"),
        dist_dig_parameters_overwrite=p.cartesian_dist_sig_parameters,
    )

    def cartesian_distance_loss_func(y_true, y_pred):
        """y_true can be whatever input you like, dihedrals, angles, pairwise dist, contact maps. That will be
        transformed with Sketchmap's sigmoid function, as will the output of the latent layer of the autoencoder.
        the difference of these two will result in a loss function."""
        cartesian_distance_loss_func.name = "cartesian_distance_loss"
        if p.cartesian_distance_cost_scale is not None:
            dist_cost = dist_loss(y_true, y_pred)
            if p.distance_cost_scale != 0:
                dist_cost *= p.cartesian_distance_cost_scale
        else:
            dist_cost = 0
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost("Cartesian Distance Cost", dist_cost),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return dist_cost

    return cartesian_distance_loss_func


def cartesian_loss(
    model,
    scale_callback=None,
    parameters=None,
    log_callback=None,
    print_current_scale=False,
):
    """Encodermap cartesian distance loss. Calculates sigmoid-weighted distances between pairwise cartesians and latent.

    Uses sketch-map's sigmoid function to transform the high-dimensional space of the input and the
    high-dimensional space of the output.

    Adjustments to this cost_function via the soft_start parameter need to be made via a callback that re-compiles the
    model during training. For this, the soft_start parameters of the outer function will be used.
    It must be either 0 or 1, indexing the 1st or 2nd element of the cartesian_cost_scale_soft_start
    tuple. The callback should also be provided when model.fit is executed.

    Three cases are possible:
        * Case 1: step < cartesian_cost_scale_soft_start[0]: cost_scale = 0
        * Case 2: cartesian_cost_scale_soft_start[0] <= step <= cartesian_cost_scale_soft_start[1]:
            cost_scale = p.cartesian_cost_scale / (cartesian_cost_scale_soft_start[1] - cartesian_cost_scale_soft_start[0]) * step
        * Case 3: cartesian_cost_scale_soft_start[1] < step: cost_scale = p.cartesian_cost_scale

    Make sure to provide the pairwise cartesian distances. This function will be adjusted as training increases via a
    callback. See encodermap.callbacks.callbacks.IncreaseCartesianCost for more info.

    Args:
        model (tf.keras.Model): The model to use the loss function on.
        parameters (Union[encodermap.ADCParameters, None], optional): The parameters. If None is
            provided default values (check them with print(em.ADCParameters.defaults_description()))
            are used. Defaults to None.
        soft_start (Union[int, None], optional): How to scale the cartesian loss. The ADCParameters class contains a
            two-tuple of integers. These integers can be used to scale this loss function. If soft_start is 0, the
            first value of ADCParameters.cartesian_cost_scale_soft_start will be used, if it is 1, the second.
            if it is None, or both values of ADCParameters.cartesian_cost_scale_soft_start are None, the cost will
            not be scaled. Defaults to None.
        print_current_scale (bool, optional): Whether to print the current scale. Is used in unittesting. Defaults to False.

    Raises:
        Exception: When no bottleneck/latent layer can be found in the model.
        Exception: When soft_start is greater than 1 and can't index the two-tuple.

    Returns:
        function: A loss function. Can be used in either custom training or model.fit().

    """
    if parameters is None:
        p = ParametersFramework(ADCParameters.defaults)
    else:
        p = parameters
    if scale_callback is not None:
        current_scale_callback = scale_callback.current_cartesian_cost_scale
    else:
        current_scale_callback = K.constant(
            p.cartesian_cost_scale, dtype="float32", name="current_cartesian_cost_scale"
        )

    if print_current_scale:
        print(current_scale_callback)

    if log_callback is None:
        write_bool = K.constant(False, "bool", name="log_bool")
    else:
        write_bool = log_callback.log_bool

    def cartesian_loss_func(y_true, y_pred=None):
        """y_true should be pairwise distances of input cartesians,
        y_pred should be pairwise distances of back-mapped output cartesians."""
        scale = current_scale_callback
        if p.cartesian_cost_variant == "mean_square":
            cartesian_cost = tf.reduce_mean(tf.square(y_true - y_pred))
        elif p.cartesian_cost_variant == "mean_abs":
            cartesian_cost = tf.reduce_mean(tf.abs(y_true - y_pred))
        elif p.cartesian_cost_variant == "mean_norm":
            cartesian_cost = tf.reduce_mean(tf.norm(y_true - y_pred, axis=1))
        else:
            raise ValueError(
                "cartesian_cost_variant {} not available".format(
                    p.dihedral_to_cartesian_cost_variant
                )
            )
        cartesian_cost /= p.cartesian_cost_reference
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost(
                "Cartesian Cost before scaling", cartesian_cost
            ),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost("Cartesian Cost current scaling", scale),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        cartesian_cost *= scale
        tf.cond(
            write_bool,
            true_fn=lambda: _summary_cost(
                "Cartesian Cost after scaling", cartesian_cost
            ),
            false_fn=lambda: _do_nothing(),
            name="Cost",
        )
        return cartesian_cost

    return cartesian_loss_func

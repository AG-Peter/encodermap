# -*- coding: utf-8 -*-
# encodermap/models/models.py
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
"""ToDo:
    * Add some nice images to the plot_model of the functional model.

"""

##############################################################################
# Imports
##############################################################################


import warnings
from math import pi

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, Dense, Input, Lambda

from ..encodermap_tf1.backmapping import chain_in_plane, dihedrals_to_cartesian_tf
from ..misc import pairwise_dist
from ..misc.summaries import add_layer_summaries
from ..parameters.parameters import ADCParameters, Parameters
from ..trajinfo.info_all import Capturing
from .layers import (
    BackMapLayer,
    MeanAngles,
    PairwiseDistances,
    PeriodicInput,
    PeriodicOutput,
    Sparse,
    SparseReshape,
)

##############################################################################
# Globals
##############################################################################


__all__ = ["gen_sequential_model", "gen_functional_model"]


##############################################################################
# Public Functions
##############################################################################


def gen_sequential_model(input_shape, parameters=None, sparse=False):
    """Returns a tf.keras Model build with the specified input shape and the parameters in the Parameters class.

    Args:
        input_shape (int): The input shape of the returned model. In most cases that is data.shape[1] of your data.
        parameters (Union[encodermap.Parameters, encodermap.ADCParameters, None], optional): The parameters to
            use on the returned model. If None is provided the default parameters in encodermap.Parameters.defaults
            is used. You can look at the defaults with print(em.Parameters.defaults_description()). Defaults to None.

    Returns:
        em.SequentialModel: A subclass of tf.keras.Model build with specified parameters.

    """
    if parameters is None:
        parameters = Parameters()
    else:
        if isinstance(parameters, Parameters):
            return SequentialModel(input_shape, parameters, sparse=sparse)
        elif isinstance(parameters, ADCParameters):
            return ADCSequentialModel(input_shape, parameters)
        else:
            p = parameters
            raise TypeError(
                f"parameters need to be ecodermap.Parameters or encodermap.ACDParameters. You supplied {type(p)}"
            )


class Sparse(tf.keras.layers.Dense):
    def call(self, inputs):
        outputs = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs


class SparseModel(tf.keras.Model):
    def __init__(self, name, input_shape):
        super(SparseModel, self).__init__(name)
        self._sparse_layer = Sparse(input_shape)
        inputs = tf.keras.layers.Input(
            shape=(input_shape,), sparse=True, name="sparse_tensor"
        )
        self._set_inputs(inputs)

    def call(self, sparse_tensor):
        return self._sparse_layer(sparse_tensor)


def gen_functional_model(
    input_dataset, parameters=None, reload_layers=None, sparse=False
):
    """Builds a model to specification of parameters using the functional API.

    The functional API is much more flexible than the sequential API, in that models with multiple inputs and outputs
    can be defined. Custom-layers and sub-models can be intermixed. In EncoderMap's case the functional API is used to
    build the AngleDihedralCartesianAutoencoder, which takes input data in form of a tf.data.Dataset with:
        * backbone_angles (angles between C, CA, N - atoms in the backbone).
        * backbone_torsions (dihedral angles in the backbone, commonly known as omega, phi, psi).
        * cartesian_coordinates (coordinates of the C, CA, N backbone atoms. This data has ndim 3, the other have ndim 2).
        * backbone_distances (distances between the C, CA, N backbone atoms).
        * sidechain_torsions (dihedral angles in the sidechain, commonly known as chi1, chi2, chi3, chi4, chi5).
    Packing and unpacking that data in the correct manner is important. Make sure to double check whether you are using
    angles or dihedrals. A simple print of the shape can be enough.

    In the functional model all operations are tf.keras.layers, meaning that the projection onto a unit_circle that
    the `SequentialModel` does in its `call()` method needs to be a layer. The FunctionalModel consist of 5 main parts:
        * Angle Inputs: The provided dataset is unpacked and the periodic data of the angles is projected onto
            a unit-circle. If the angles are in gradians, they will also be normalized into a [-pi, pi) interval.
        * Autoencoder: The trainable part of the network consists of the Autoencoder part build to the specifications
            in the provided parameters. Here, Dense layers are stacked. Only the angles and torsions are fed into the
            Autoencoder. The Distances and Cartesians are used later.
        * Angle Outputs: The angles are recalculated from their unit-circle inputs.
        * Back-Mapping. The backmapping layer takes backbone_angles and backbone_dihedrals, backbone_distances to
            calculate new cartesian coordinates.
        * Pairwise Distances: The pairwise distances of the input cartesians and the back-mapped cartesians are calculated.

    Args:
        input_dataset (tf.data.Dataset): The dataset with the data in the order given in the explanation.
        parameters (Union[em.ADCParameters, None], optional): The parameters to be used to build the network.
            If None is provided the default parameters in encodermap.ADCParameters.defaults
            is used. You can look at the defaults with print(em.ADCParameters.defaults_description()). Defaults to None.
        reload_layers (Union[None, list], optional): List of layers that will be reloaded when reloading the model from
            disk. Defaults to None, when a new model should be built.

    Raises:
        AssertionError: AssertionErrors will be raised when the input data is not formatted correctly.
            This means, if len(cartesians) != len(distances) - 1, or len(cartesians) != len(angles) - 2.
            This can also mean, the input dataset is not packed correctly. Please keep the order specified above.
            This can also mean, that the provided protein is not linear (branched, circular, ...).

    Returns:
        em.FunctionalModel: A subclass of tf.keras.Model build with specified parameters.

    """
    if parameters is None:
        p = Parameters()
    else:
        p = parameters

    if isinstance(p, Parameters):
        raise Exception(
            "Functional Model is currently reserved for the ADCAutoencoder,"
            "because of the way the data is packed and unpacked."
        )

    # unpack the shapes of the input
    for i, d in enumerate(input_dataset):
        angles, dihedrals, cartesians, distances, side_dihedrals = d
        break

    # These assertions need to be changed for all proteins that are not-linear.
    if isinstance(cartesians, tf.sparse.SparseTensor):
        assert distances.shape[1] == cartesians.shape[1] // 3 - 1
    else:
        assert distances.shape[1] == cartesians.shape[1] - 1, print(
            distances.shape, cartesians.shape
        )
    assert angles.shape[1] == distances.shape[1] - 1, print(
        angles.shape, cartesians.shape
    )
    assert dihedrals.shape[1] == distances.shape[1] - 2, print(
        dihedrals.shape, cartesians.shape
    )

    if reload_layers is not None:
        raise Exception("currently not reloadable.")

    # define regularizer
    regularizer = tf.keras.regularizers.l2(p.l2_reg_constant)

    # central cartesians
    if not sparse or not isinstance(dihedrals, tf.sparse.SparseTensor):
        inp_dihedrals = Input(
            shape=(dihedrals.shape[1],),
            name="input_dihedrals",
        )
        x = PeriodicInput(p, "dihedrals")(inp_dihedrals)
        get_dense_model_central_dihedrals = None
    else:
        inp_dihedrals = Input(
            shape=(dihedrals.shape[1],),
            name="input_dihedrals",
            sparse=True,
        )
        x = Dense(dihedrals.shape[1])(inp_dihedrals)
        get_dense_model_central_dihedrals = tf.keras.Model(
            inputs=inp_dihedrals,
            outputs=x,
        )
        x = PeriodicInput(p, "dihedrals")(x)

    # backbone angles
    if p.use_backbone_angles:
        if not sparse or not isinstance(angles, tf.sparse.SparseTensor):
            inp_angles = Input(
                shape=(angles.shape[1],),
                name="input_angles",
            )
            y = PeriodicInput(p, "angles")(inp_angles)
            get_dense_model_central_angles = None
        else:
            inp_angles = Input(
                shape=(angles.shape[1],),
                name="input_angles",
                sparse=True,
            )
            y = Dense(angles.shape[1])(inp_angles)
            get_dense_model_central_angles = tf.keras.Model(
                inputs=inp_angles,
                outputs=y,
            )
            y = PeriodicInput(p, "angles")(y)

    # sidechains
    get_dense_model_side_dihedrals = None
    if p.use_sidechains:
        if not sparse or not isinstance(side_dihedrals, tf.sparse.SparseTensor):
            inp_side_dihedrals = Input(
                shape=(side_dihedrals.shape[1],), name="input_side_dihedrals"
            )
            z = PeriodicInput(p, "side_dihedrals")(inp_side_dihedrals)
        else:
            inp_side_dihedrals = Input(
                shape=(side_dihedrals.shape[1],),
                name="input_side_dihedrals",
                sparse=True,
            )
            z = Dense(side_dihedrals.shape[1])(inp_side_dihedrals)
            # z = SparseModel(name="Sparse_Model", input_shape=side_dihedrals.shape[1])(inp_side_dihedrals)
            get_dense_model_side_dihedrals = tf.keras.Model(
                inputs=inp_side_dihedrals,
                outputs=z,
            )
            z = PeriodicInput(p, "side_dihedrals")(z)

    # these inputs will be passed through and will be used for backmapping
    # and RMSD metrics
    if not sparse or not isinstance(cartesians, tf.sparse.SparseTensor):
        inp_cartesians = Input(
            shape=(
                cartesians.shape[1],
                3,
            ),
            name="input_cartesians",
        )
        inp_distances = Input(
            shape=(distances.shape[1],),
            name="input_distances",
        )
        get_dense_model_cartesians = None
        get_dense_model_distances = None
    else:
        inp_cartesians = Input(
            shape=(cartesians.shape[1],),
            name="input_cartesians",
            sparse=True,
        )
        dc = Dense(cartesians.shape[1])(inp_cartesians)
        get_dense_model_cartesians = tf.keras.Model(
            inputs=inp_cartesians,
            outputs=dc,
        )
        dc = tf.keras.layers.Reshape(
            target_shape=(
                cartesians.shape[1] // 3,
                3,
            ),
            input_shape=(cartesians.shape[1],),
        )(dc)
        inp_distances = Input(
            shape=(distances.shape[1],),
            name="input_distances",
            sparse=True,
        )
        dd = Dense(distances.shape[1])(inp_distances)
        get_dense_model_distances = tf.keras.Model(
            inputs=inp_distances,
            outputs=dd,
        )

    # stack the three datasources going through the network
    if not p.use_backbone_angles and not p.use_sidechains:
        splits = None
    elif p.use_backbone_angles and not p.use_sidechains:
        splits = [x.shape[1], y.shape[1]]
        x = Concatenate(axis=1, name="Main_Inputs")([x, y])
    elif p.use_backbone_angles and p.use_sidechains:
        # ToDo: make sure the splits work here. There seems to be different outputs from tf and np.
        splits = [x.shape[1], y.shape[1], z.shape[1]]
        x = Concatenate(axis=1, name="Main_Inputs")([x, y, z])

    # save the out_shape now and use it for an output layer
    out_shape = x.shape[1]

    # rename empty string in parameters to None
    activation_functions = list(
        map(lambda x: x if x != "" else None, p.activation_functions)
    )

    # define how layers are stacked
    layer_data = list(
        zip(
            p.n_neurons + p.n_neurons[-2::-1],
            activation_functions[1:] + activation_functions[-1::-1],
        )
    )
    # add a layer that reshapes the output
    layer_data.append([out_shape, "tanh"])

    # decide layer names
    names = []
    for i, (n_neurons, act_fun) in enumerate(layer_data):
        if i < len(p.n_neurons) - 1:
            name = f"Encoder_{i}"
        elif i > len(p.n_neurons) - 1:
            ind = i - len(p.n_neurons)
            name = f"Decoder_{ind}"
        else:
            name = "Latent"
        names.append(name)
    layer_data = list((*i, j) for i, j in zip(layer_data, names))

    # unpack layer data into encoder and decoder
    neurons = [i[0] for i in layer_data]
    bottleneck_index = neurons.index(min(neurons)) + 1
    encoder_layers = layer_data[:bottleneck_index]
    decoder_layers = layer_data[bottleneck_index:]

    # enocder layers
    for n_neurons, act_fun, name in encoder_layers:
        layer = Dense(
            units=n_neurons,
            activation=act_fun,
            name=name,
            kernel_initializer=tf.initializers.VarianceScaling(),
            kernel_regularizer=regularizer,
            bias_initializer=tf.initializers.RandomNormal(0.1, 0.5),
        )
        x = layer(x)

    # encoder model
    if p.use_backbone_angles and p.use_sidechains:
        encoder = tf.keras.Model(
            inputs=[inp_angles, inp_dihedrals, inp_side_dihedrals],
            outputs=[x],
            name="Encoder",
        )
    elif p.use_backbone_angles and not p.use_sidechains:
        encoder = tf.keras.Model(
            inputs=[inp_angles, inp_dihedrals],
            outputs=[x],
            name="Encoder",
        )
    else:
        encoder = tf.keras.Model(
            inputs=[inp_dihedrals, inp_side_dihedrals],
            outputs=[x],
            name="Encoder",
        )
    if p.tensorboard:
        with Capturing() as output:
            encoder.summary()
        with open(p.main_path + "/encoder_summary.txt", "w") as f:
            f.write("\n".join(output))

    # decoder input
    decoder_input = Input(shape=(encoder_layers[-1][0],), name="Decoder_Input")
    x = decoder_input

    # decoder layers
    for i, (n_neurons, act_fun, name) in enumerate(decoder_layers):
        layer = Dense(
            units=n_neurons,
            activation=act_fun,
            name=name,
            kernel_initializer=tf.initializers.VarianceScaling(),
            kernel_regularizer=regularizer,
            bias_initializer=tf.initializers.RandomNormal(0.1, 0.5),
        )
        x = layer(x)

    # split output accordingly
    if splits is None:
        out_dihedrals = x
        out_angles = MeanAngles(p, "Mean_Angles", out_dihedrals.shape[0])(inp_angles)
        decoder_input = [decoder_input, inp_angles]
        decoder_output = [out_angles, out_dihedrals]
    elif len(splits) == 2:
        out_angles, out_dihedrals = Lambda(
            lambda x: tf.split(x, splits, 1), name="Split_Output"
        )(x)
        out_angles = PeriodicOutput(p, "Angles")(out_angles)
        out_dihedrals = PeriodicOutput(p, "Dihedrals")(out_dihedrals)
        decoder_input = [decoder_input]
        decoder_output = [out_angles, out_dihedrals]
    elif len(splits) == 3:
        out_dihedrals, out_angles, out_side_dihedrals = Lambda(
            lambda x: tf.split(x, splits, 1), name="Split_Output"
        )(x)
        out_angles = PeriodicOutput(p, "Angles")(out_angles)
        out_dihedrals = PeriodicOutput(p, "Dihedrals")(out_dihedrals)
        out_side_dihedrals = PeriodicOutput(p, "Side_Dihedrals")(out_side_dihedrals)
        decoder_input = [decoder_input]
        decoder_output = [out_angles, out_dihedrals, out_side_dihedrals]

    # decoder model before backmapping
    decoder = tf.keras.Model(
        inputs=decoder_input, name="Decoder", outputs=decoder_output
    )
    if p.tensorboard:
        with Capturing() as output:
            decoder.summary()
        with open(p.main_path + "/decoder_summary.txt", "w") as f:
            f.write("\n".join(output))

    # backmap input
    back_inp_angles = Input(shape=(out_angles.shape[1],), name="Back_Angles_Input")
    back_inp_dihedrals = Input(
        shape=(out_dihedrals.shape[1],), name="Back_Dihedrals_Input"
    )
    if p.use_sidechains:
        pass_sidedihedrals = Input(
            shape=(out_side_dihedrals.shape[1],), name="Side_Dihedrals_Pass_Through"
        )

    # backmapping. The hardest part
    if not sparse or not isinstance(distances, tf.sparse.SparseTensor):
        back_cartesians = BackMapLayer()(
            (inp_distances, back_inp_angles, back_inp_dihedrals)
        )
    else:
        back_cartesians = BackMapLayer()((dd, back_inp_angles, back_inp_dihedrals))

    # pairwise distances is the last part
    if not sparse or not isinstance(cartesians, tf.sparse.SparseTensor):
        inp_pair = PairwiseDistances(p, "Input")(inp_cartesians)
    else:
        inp_pair = PairwiseDistances(p, "Input")(dc)
    out_pair = PairwiseDistances(p, "Backmapped")(back_cartesians)

    # backmap_model
    if p.use_sidechains:
        backmap_model = tf.keras.Model(
            name="Backmapping",
            inputs=[
                back_inp_angles,
                back_inp_dihedrals,
                pass_sidedihedrals,
                inp_distances,
                inp_cartesians,
            ],
            outputs=[
                back_inp_angles,
                back_inp_dihedrals,
                back_cartesians,
                inp_pair,
                out_pair,
                pass_sidedihedrals,
            ],
        )
    else:
        backmap_model = tf.keras.Model(
            name="Backmapping",
            inputs=[
                back_inp_angles,
                back_inp_dihedrals,
                inp_distances,
                inp_cartesians,
            ],
            outputs=[
                back_inp_angles,
                back_inp_dihedrals,
                back_cartesians,
                inp_pair,
                out_pair,
            ],
        )
    if p.tensorboard:
        with Capturing() as output:
            backmap_model.summary()
        with open(p.main_path + "/backmap_summary.txt", "w") as f:
            f.write("\n".join(output))

    # call all the models hierarchically to rebuild a complete model
    if p.use_sidechains:
        main_inputs = [
            inp_angles,
            inp_dihedrals,
            inp_cartesians,
            inp_distances,
            inp_side_dihedrals,
        ]
        main_outputs = backmap_model(
            (
                *decoder(encoder((inp_angles, inp_dihedrals, inp_side_dihedrals))),
                inp_distances,
                inp_cartesians,
            )
        )
    else:
        main_inputs = [
            inp_angles,
            inp_dihedrals,
            inp_cartesians,
            inp_distances,
        ]
        main_outputs = backmap_model(
            (
                *decoder(encoder((inp_angles, inp_dihedrals))),
                inp_distances,
                inp_cartesians,
            )
        )

    # full_model = tf.keras.Model(inputs=[inp_angles, inp_dihedrals, inp_cartesians, inp_distances, inp_side_dihedrals],
    #                             outputs=main_outputs, name="Full_Model")

    # pass input and outputs to FunctionalModel
    # In FunctionalModel train_step is overwritten. Train_step should unpack the data and assign the inputs/outputs to the
    # differtent loss functions.
    if not sparse:
        model = FunctionalModel(
            parameters=p,
            inputs=main_inputs,
            outputs=main_outputs,
            encoder=encoder,
            decoder=decoder,
        )
    else:
        model = SparseFunctionalModel(
            parameters=p,
            inputs=main_inputs,
            outputs=main_outputs,
            encoder=encoder,
            decoder=decoder,
            get_dense_model_central_dihedrals=get_dense_model_central_dihedrals,
            get_dense_model_central_angles=get_dense_model_central_angles,
            get_dense_model_side_dihedrals=get_dense_model_side_dihedrals,
            get_dense_model_cartesians=get_dense_model_cartesians,
            get_dense_model_distances=get_dense_model_distances,
        )

    return model


##############################################################################
# Public Classes
##############################################################################


class FunctionalModel(tf.keras.Model):
    def __init__(self, parameters, inputs, outputs, encoder, decoder):
        super().__init__(inputs=inputs, outputs=outputs)
        self.p = parameters
        self.encoder_model = encoder
        self.decoder_model = decoder

        # train counter
        self._train_counter = K.variable(0, "int64", name="train_counter")

    def encoder(self, x, training=False):
        return self.encoder_model(x, training=training)

    def decoder(self, x, training=False):
        return self.decoder_model(x, training=training)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.unpacked_loss_fns = {fn.__name__: fn for fn in self.compiled_loss._losses}

    def get_loss(self, inp):
        # unpack the inputs
        if self.p.use_sidechains:
            (
                inp_angles,
                inp_dihedrals,
                inp_cartesians,
                inp_distances,
                inp_side_dihedrals,
            ) = inp
        else:
            (
                inp_angles,
                inp_dihedrals,
                inp_cartesians,
                inp_distances,
            ) = inp
        # unpack the outputs
        out = self(inp, training=True)
        if self.p.use_sidechains:
            (
                out_angles,
                out_dihedrals,
                back_cartesians,
                inp_pair,
                out_pair,
                out_side_dihedrals,
            ) = out
        else:
            (
                out_angles,
                out_dihedrals,
                back_cartesians,
                inp_pair,
                out_pair,
            ) = out

        # define latent for cartesian_distance_loss
        if self.p.use_sidechains:
            latent = self.encoder_model(
                (inp_angles, inp_dihedrals, inp_side_dihedrals),
                training=True,
            )
        else:
            latent = self.encoder_model(
                (inp_angles, inp_dihedrals),
                training=True,
            )

        with tf.name_scope("Cost"):
            loss = 0.0
            # dihedral loss
            loss += self.unpacked_loss_fns["dihedral_loss_func"](
                inp_dihedrals, out_dihedrals
            )
            # angle loss
            loss += self.unpacked_loss_fns["angle_loss_func"](inp_angles, out_angles)
            # cartesian loss
            loss += self.unpacked_loss_fns["cartesian_loss_func"](inp_pair, out_pair)
            # distance loss
            loss += self.unpacked_loss_fns["distance_loss_func"](
                (inp_angles, inp_dihedrals, inp_side_dihedrals)
            )
            # cartesian distance cost
            loss += self.unpacked_loss_fns["cartesian_distance_loss_func"](
                inp_pair, latent
            )
            # center loss
            loss += self.unpacked_loss_fns["center_loss_func"](
                (inp_angles, inp_dihedrals, inp_side_dihedrals)
            )
            # reg loss
            loss += self.unpacked_loss_fns["regularization_loss_func"]()
            # side dihedral loss
            if self.p.use_sidechains:
                loss += self.unpacked_loss_fns["side_dihedral_loss_func"](
                    inp_side_dihedrals, out_side_dihedrals
                )
            tf.summary.scalar("Combined Cost", loss)

            # autoloss
            from encodermap.loss_functions.loss_functions import periodic_distance

            angle_auto_loss = 2 * tf.reduce_mean(
                tf.square(periodic_distance(inp_angles, out_angles, 2 * np.pi))
            )
            dihedral_auto_loss = 2 * tf.reduce_mean(
                tf.square(periodic_distance(inp_dihedrals, out_dihedrals, 2 * np.pi))
            )
            side_dihedral_auto_loss = 2 * tf.reduce_mean(
                tf.square(
                    periodic_distance(inp_side_dihedrals, out_side_dihedrals, 2 * np.pi)
                )
            )
        return loss

    def train_step(self, data):
        # Data will always contain all inputs, depending on p, the model will return different tuples
        # but the input will be the same, only when the encoder is called with teh train data, and the
        # decoder is called does packing and unpacking matter.
        # gradient tape to calculate loss for backmapping

        with tf.GradientTape() as tape:
            tf.summary.experimental.set_step(self._train_counter)
            loss = self.get_loss(data)
            loggable_encoder_layers = [
                l for l in self.encoder_model.layers if l.__class__.__name__ == "Dense"
            ]
            loggable_decoder_layers = [
                l for l in self.decoder_model.layers if l.__class__.__name__ == "Dense"
            ]
            for l in loggable_encoder_layers + loggable_decoder_layers:
                add_layer_summaries(l, step=self._train_counter)

        # optimization happens here
        # Compute Gradients
        # trainable_vars = self.trainable_variables
        trainable_vars = (
            self.encoder_model.trainable_variables
            + self.decoder_model.trainable_variables
        )
        # maybe self.encoder_model.trainable_vars + self.decoder_model.trainable_vars
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        # self.compiled_metrics.update_state(data, self(data))
        # Return a dict mapping metric names to current value
        # Add loss to the dict so the ProgressBar callback can pick it up
        # return {**{m.name: m.result() for m in self.metrics}, **{'loss': loss}}
        # udpate train counter because tensorflow seems to have deprecated it
        self._train_counter.assign_add(1)
        return {"loss": loss}


class SparseFunctionalModel(FunctionalModel):
    def __init__(
        self,
        parameters,
        inputs,
        outputs,
        encoder,
        decoder,
        get_dense_model_central_dihedrals,
        get_dense_model_central_angles,
        get_dense_model_side_dihedrals,
        get_dense_model_cartesians,
        get_dense_model_distances,
    ):
        super().__init__(parameters, inputs, outputs, encoder, decoder)
        self.get_dense_model_central_dihedrals = get_dense_model_central_dihedrals
        self.get_dense_model_central_angles = get_dense_model_central_angles
        self.get_dense_model_side_dihedrals = get_dense_model_side_dihedrals
        self.get_dense_model_cartesians = get_dense_model_cartesians
        self.get_dense_model_distances = get_dense_model_distances

    def get_loss(self, inp):
        # unpack the inputs
        (
            sparse_inp_angles,
            sparse_inp_dihedrals,
            sparse_inp_cartesians,
            sparse_inp_distances,
            sparse_side_dihedrals,
        ) = inp

        # make the side dihedrals dense
        if isinstance(sparse_inp_angles, tf.sparse.SparseTensor):
            inp_angles = self.get_dense_model_central_angles(
                sparse_inp_angles, training=True
            )
        else:
            inp_angles = sparse_inp_angles
        if isinstance(sparse_inp_dihedrals, tf.sparse.SparseTensor):
            inp_dihedrals = self.get_dense_model_central_dihedrals(
                sparse_inp_dihedrals, training=True
            )
        else:
            inp_dihedrals = sparse_inp_dihedrals
        if isinstance(sparse_inp_cartesians, tf.sparse.SparseTensor):
            inp_cartesians = self.get_dense_model_cartesians(
                sparse_inp_cartesians, training=True
            )
        else:
            inp_cartesians = sparse_inp_cartesians
        if isinstance(sparse_inp_distances, tf.sparse.SparseTensor):
            inp_distances = self.get_dense_model_distances(
                sparse_inp_distances, training=True
            )
        else:
            inp_distances = sparse_inp_distances

        if self.p.use_sidechains:
            if isinstance(sparse_side_dihedrals, tf.sparse.SparseTensor):
                inp_side_dihedrals = self.get_dense_model_side_dihedrals(
                    sparse_side_dihedrals, training=True
                )
            else:
                inp_side_dihedrals = sparse_side_dihedrals

        # make them into an Iterable again
        if self.p.use_sidechains:
            data = (
                inp_angles,
                inp_dihedrals,
                inp_cartesians,
                inp_distances,
                inp_side_dihedrals,
            )
        else:
            data = (
                inp_angles,
                inp_dihedrals,
                inp_cartesians,
                inp_distances,
            )

        # call the loss
        return super().get_loss(data)


class SequentialModel(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        parameters=None,
        reload_layers=None,
        sparse=False,
        get_dense_model=None,
    ):
        if parameters is None:
            self.p = Parameters()
        else:
            self.p = parameters
        super().__init__()
        self.sparse = sparse
        self.input_dim = input_dim
        self.get_dense_model = get_dense_model

        # tensors for using tf.cond inside self.train_step()
        self.update_step = tf.constant(self.p.summary_step, dtype="int64")
        self.debug_tensor = tf.constant(self.p.tensorboard, dtype="bool")

        # periodicity doubles the inputs and outputs
        if self.p.periodicity < float("inf"):
            self.input_dim *= 2

        # if layers is provided we are rebuilding a trained model
        if reload_layers is not None:
            if self.p.periodicity < float("inf"):
                print(
                    f"rebuilding Model with input_dim = {int(self.input_dim/2)} and periodicity = {self.p.periodicity}"
                )
            else:
                print(f"rebuilding Model with input_dim = {self.input_dim}")
            if len(reload_layers) != 2:
                raise Exception(
                    "currently only works with 2 layers. Encoder and Decoder."
                )
            assert all([isinstance(i, tf.keras.Sequential) for i in reload_layers])
            # input_layer = tf.keras.layers.InputLayer(input_shape=(self.input_dim,), dtype='float32')
            self.encoder_model = reload_layers[0]
            self.decoder_model = reload_layers[1]
            self.build(input_shape=(1, self.input_dim))
            return

        # define regularizer
        regularizer = tf.keras.regularizers.l2(self.p.l2_reg_constant)

        # rename empty string in parameters to None
        activation_functions = list(
            map(lambda x: x if x != "" else None, self.p.activation_functions)
        )

        # define how layers are stacked
        layer_data = list(
            zip(
                self.p.n_neurons + self.p.n_neurons[-2::-1],
                activation_functions[1:] + activation_functions[-1::-1],
            )
        )
        # add a layer that reshapes the output
        layer_data.append([self.input_dim, "tanh"])

        # decide layer names
        names = []
        for i, (n_neurons, act_fun) in enumerate(layer_data):
            if i < len(self.p.n_neurons) - 1:
                name = f"Encoder_{i}"
            elif i > len(self.p.n_neurons) - 1:
                ind = i - len(self.p.n_neurons)
                name = f"Decoder_{ind}"
            else:
                name = "Latent"
            names.append(name)
        layer_data = list((*i, j) for i, j in zip(layer_data, names))

        # define encoder and decoder layers
        neurons = [i[0] for i in layer_data]
        bottleneck_index = neurons.index(min(neurons)) + 1
        self.encoder_layers = layer_data[:bottleneck_index]
        self.decoder_layers = layer_data[bottleneck_index:]

        # input
        # Instead of using InputLayer use Dense with kwarg input_shape
        # allows model to be reloaded better <- weird english... reloaded better
        if self.sparse:
            shape = self.input_dim
            if self.p.periodicity < float("inf"):
                shape /= 2
            _input_layer = Input(
                shape=(int(shape),),
                sparse=True,
            )
            x = Dense(shape)(_input_layer)
            self.get_dense_model = tf.keras.Model(
                inputs=_input_layer,
                outputs=x,
            )

        input_layer = tf.keras.layers.Dense(
            input_shape=(self.input_dim,),
            units=self.encoder_layers[0][0],
            activation=self.encoder_layers[0][1],
            name=self.encoder_layers[0][2],
            kernel_initializer=tf.initializers.VarianceScaling(),
            kernel_regularizer=regularizer,
            bias_initializer=tf.initializers.RandomNormal(0.1, 0.5),
        )

        # what model to use for the encoder

        # output
        # output_layer = tf.keras.layers.Dense(
        #     self.input_dim,
        #     name="Output",
        #     activation=None,
        #     kernel_initializer=tf.initializers.VarianceScaling(),
        #     kernel_regularizer=regularizer,
        #     bias_initializer=tf.initializers.RandomNormal(0.1, 0.05),
        #     trainable=True)

        # encoder
        self.encoder_model = tf.keras.Sequential(
            [input_layer]
            + [
                tf.keras.layers.Dense(
                    n_neurons,
                    activation=act_fun,
                    name=name,
                    kernel_initializer=tf.initializers.VarianceScaling(),
                    kernel_regularizer=regularizer,
                    bias_initializer=tf.initializers.RandomNormal(0.1, 0.5),
                )
                for n_neurons, act_fun, name in self.encoder_layers[1:]
            ],
            name="Encoder",
        )

        # decoder
        self.decoder_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    n_neurons,
                    activation=act_fun,
                    name=name,
                    kernel_initializer=tf.initializers.VarianceScaling(),
                    kernel_regularizer=regularizer,
                    bias_initializer=tf.initializers.RandomNormal(0.1, 0.5),
                )
                for n_neurons, act_fun, name in self.decoder_layers
            ],
            name="Decoder",
        )

        # build
        self.build(input_shape=(1, self.input_dim))

        # train counter
        self._train_counter = K.variable(0, "int64", name="train_counter")

    def build(self, *args, **kwargs):
        input_shape = kwargs["input_shape"]
        # Because build calls self.call and self.call calls self.encode
        # the input dim needs to be halved here
        if self.p.periodicity < float("inf"):
            input_shape = (*input_shape[:-1], int(input_shape[-1] / 2))
        try:
            super().build(*args, **{**kwargs, **dict(input_shape=input_shape)})
        except Exception:
            if self.p.periodicity < float("inf"):
                print(
                    "Exception is raised because of periodicity. In general "
                    "you don't need to call the build method from outside of "
                    "this class, because it is called at the end of __init__."
                )
            raise

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        try:
            self.unpacked_loss_fns = {
                fn.__name__: fn for fn in self.compiled_loss._losses
            }
        except AttributeError:
            for i in dir(self):
                if "loss" in i:
                    print(i)
            print(self._callable_losses)
            raise

    def encoder(self, x, training=False):
        if self.sparse:
            x = self.get_dense_model(x)
        if self.p.periodicity < float("inf"):
            if self.p.periodicity != 2 * pi:
                x = x / self.p.periodicity * 2 * pi
            x = tf.concat([tf.sin(x), tf.cos(x)], 1)
        return self.encoder_model(x, training=training)

    def decoder(self, x, training=False):
        x = self.decoder_model(x, training=training)
        if self.p.periodicity < float("inf"):
            x = tf.atan2(*tf.split(x, 2, 1))
            if self.p.periodicity != 2 * pi:
                x = x / (2 * pi) * self.p.periodicity
        return x

    def call(self, x, training=False):
        # encode and decode
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def train_step(self, data):
        """Overwrites the normal train_step. What is different?

        Not much. Even the provided data is expected to be a tuple of (data, classes) (x, y) in classification tasks.
        The data is unpacked and y is discarded, because the Autoencoder Model is a regression task.

        Args:
            data (tuple): The (x, y) data of this train step.

        """
        x, _ = data
        if self.sparse:
            x = self.get_dense_model(x)

        with tf.GradientTape() as tape:
            tf.summary.experimental.set_step(self._train_counter)
            with tf.name_scope("Cost"):
                loss = 0.0
                for l in self.compiled_loss._losses:
                    loss += l(x, self(x, training=True))
                tf.summary.scalar("Combined Cost", loss)
            for l in self.encoder_model.layers + self.decoder_model.layers:
                add_layer_summaries(l, step=self._train_counter)

        # Compute Gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(x, self(x))
        # udpate train counter because tensorflow seems to have deprecated it
        self._train_counter.assign_add(1)
        # Return a dict mapping metric names to current value
        return {**{m.name: m.result() for m in self.metrics}, **{"loss": loss}}


class ADCSequentialModel(SequentialModel):
    def __init__(self, input_dim, parameters=None, reload_layers=None, sparse=False):
        warnings.warn("check split")
        if parameters is None:
            self.p = ADCParameters()
        else:
            self.p = parameters
        self.multiples = tf.TensorShape((self.p.batch_size, 1))
        super(ADCSequentialModel, self).__init__(input_dim, self.p, reload_layers)

    def call(self, x, training=False):
        # encode and decode
        if isinstance(x, tf.Tensor):
            pass
        elif isinstance(x, tuple):
            (
                inp_angles,
                inp_dihedrals,
                inp_cartesians,
                inp_distances,
                inp_side_dihedrals,
            ) = x
            if not self.p.use_backbone_angles and not self.p.use_sidechains:
                x = inp_dihedrals
            elif self.p.use_backbone_angles and not self.p.use_sidechains:
                x = tf.concat([inp_angles, inp_dihedrals], 1)
            elif self.p.use_backbone_angles and self.p.use_sidechains:
                x = tf.concat([inp_angles, inp_dihedrals, inp_side_dihedrals], 1)
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def call_and_map_back(
        self, x, distances, angles, dihedrals, cartesians, splits, side_dihedrals=None
    ):
        # latent = self.encoder(x, training=False)
        out = self(x, training=True)
        latent = self.encoder(x, training=True)

        # unpack out
        if splits is None:
            out_dihedrals = out
            out_angles = tf.tile(
                tf.expand_dims(tf.reduce_mean(angles, 0), 0), multiples=self.multiples
            )
        elif len(splits) == 2:
            out_angles, out_dihedrals = tf.split(out, splits, 1)
        elif len(splits) == 3:
            out_angles, out_dihedrals, out_side_dihedrals = tf.split(out, splits, 1)

        # do back-mapping
        back_mean_lengths = tf.expand_dims(tf.reduce_mean(distances, 0), 0)
        back_chain_in_plane = chain_in_plane(back_mean_lengths, out_angles)
        back_cartesians = dihedrals_to_cartesian_tf(
            out_dihedrals + pi, back_chain_in_plane
        )

        # get pairwise distances of CA atoms
        inp_pair = pairwise_dist(
            cartesians[
                :,
                self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
            ],
            flat=True,
        )
        out_pair = pairwise_dist(
            back_cartesians[
                :,
                self.p.cartesian_pwd_start : self.p.cartesian_pwd_stop : self.p.cartesian_pwd_step,
            ],
            flat=True,
        )

        with tf.name_scope("Cost"):
            loss = 0.0
            # dihedral loss
            loss += self.unpacked_loss_fns["dihedral_loss_func"](
                dihedrals, out_dihedrals
            )
            # angle loss
            loss += self.unpacked_loss_fns["angle_loss_func"](angles, out_angles)
            # cartesian loss
            loss += self.unpacked_loss_fns["cartesian_loss_func"](inp_pair, out_pair)
            # distance loss
            loss += self.unpacked_loss_fns["distance_loss_func"](x)
            # cartesian distance cost
            loss += self.unpacked_loss_fns["cartesian_distance_loss_func"](
                inp_pair, latent
            )
            # center loss
            loss += self.unpacked_loss_fns["center_loss_func"](x)
            # reg loss
            loss += self.unpacked_loss_fns["regularization_loss_func"]()
            if self.p.use_sidechains:
                loss += self.unpacked_loss_fns["side_dihedral_loss_func"](
                    side_dihedrals, out_side_dihedrals
                )
            tf.summary.scalar("Combined Cost", loss)
        return loss

    def train_step(self, data):
        # unpack the data
        (
            inp_angles,
            inp_dihedrals,
            inp_cartesians,
            inp_distances,
            inp_side_dihedrals,
        ) = data
        if not self.p.use_backbone_angles and not self.p.use_sidechains:
            main_inputs = inp_dihedrals
            splits = None
        elif self.p.use_backbone_angles and not self.p.use_sidechains:
            main_inputs = tf.concat([inp_angles, inp_dihedrals], 1)
            splits = [inp_angles.shape[1], inp_dihedrals.shape[1]]
        elif self.p.use_backbone_angles and self.p.use_sidechains:
            # ToDo: make sure the splits work here. There seems to be different outputs from tf and np.
            main_inputs = tf.concat([inp_angles, inp_dihedrals, inp_side_dihedrals], 1)
            splits = [
                inp_angles.shape[1],
                inp_dihedrals.shape[1],
                inp_side_dihedrals.shape[1],
            ]

        # gradient tape to calculate loss for backmapping
        with tf.GradientTape() as tape:
            tf.summary.experimental.set_step(self._train_counter)
            if self.p.use_sidechains:
                loss = self.call_and_map_back(
                    main_inputs,
                    inp_distances,
                    inp_angles,
                    inp_dihedrals,
                    inp_cartesians,
                    splits,
                    inp_side_dihedrals,
                )
            else:
                loss = self.call_and_map_back(
                    main_inputs,
                    inp_distances,
                    inp_angles,
                    inp_dihedrals,
                    inp_cartesians,
                    splits,
                )
            for l in self.encoder_model.layers + self.decoder_model.layers:
                add_layer_summaries(l, step=self._train_counter)

        # optimization happens here
        # Compute Gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(data, self(data))
        # Return a dict mapping metric names to current value
        # Add loss to the dict so the ProgressBar callback can pick it up
        return {**{m.name: m.result() for m in self.metrics}, **{"loss": loss}}

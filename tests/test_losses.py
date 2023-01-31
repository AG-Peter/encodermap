# -*- coding: utf-8 -*-
# tests/test_losses.py
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
# from encodermap.loss_functions import loss_functions
import os
import unittest

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist

from encodermap.callbacks import callbacks
from encodermap.encodermap_tf1.backmapping import (
    chain_in_plane,
    dihedrals_to_cartesian_tf,
)
from encodermap.loss_functions.loss_functions import old_distance_loss
from encodermap.misc import pairwise_dist
from encodermap.models.models import SequentialModel
from encodermap.parameters.parameters import ADCParameters, Parameters

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# If scipy was compiled against an older version of numpy these warnings are raised
# warnings in a testing environment are somewhat worrying
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

##############################################################################
# Classes for Testing fixing output to predefined values
##############################################################################


class LayerThatOutputsConstant(tf.keras.layers.Layer):
    def __init__(self, units, output_len, output_constant=0, name="Latent"):
        super(LayerThatOutputsConstant, self).__init__()
        self.output_constant = tf.constant(
            output_constant, shape=(output_len, units), dtype="float32"
        )
        self._name = name

    def call(self, inputs):
        return self.output_constant


class LayerThatOutputsRandom(tf.keras.layers.Layer):
    def __init__(self, units, output_len, name="Latent"):
        super(LayerThatOutputsRandom, self).__init__()
        output = np.random.random((output_len, units))
        self.output_constant = tf.constant(
            output, shape=(output_len, units), dtype="float32"
        )
        self._name = name + "_random"

    def call(self, inputs):
        return self.output_constant


class ConstantOutputAutoencoder(SequentialModel):
    def __init__(self, input_dim, len_data, parameters=None, latent_constant=0):
        super(ConstantOutputAutoencoder, self).__init__(input_dim, parameters)
        regularizer = tf.keras.regularizers.l2(self.p.l2_reg_constant)
        # input
        # Instead of using InputLayer use Dense with kwarg input_shape
        # allows model to be reloaded better
        input_layer = tf.keras.layers.Dense(
            input_shape=(input_dim,),
            units=self.encoder_layers[0][0],
            activation=self.encoder_layers[0][1],
            name=self.encoder_layers[0][2],
            kernel_initializer=tf.initializers.VarianceScaling(),
            kernel_regularizer=regularizer,
            bias_initializer=tf.initializers.RandomNormal(0.1, 0.5),
        )

        # constant output
        if latent_constant != "random":
            self.constant_layer = LayerThatOutputsConstant(
                self.p.n_neurons[-1], len_data, latent_constant, name="Latent"
            )
        else:
            self.constant_layer = LayerThatOutputsRandom(
                self.p.n_neurons[-1], len_data, name="Latent"
            )

        # overwrite encdoer
        self.encoder = tf.keras.Sequential(
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
                for n_neurons, act_fun, name in self.encoder_layers[1:-1]
            ]
            + [self.constant_layer],
            name="Encoder",
        )
        self.build(input_shape=(1, input_dim))

    def call(self, x):
        return x


##############################################################################
# Metrics to use in scipy cdist to easily compute the correct pairwise distances
##############################################################################


def sigmoid_closure(sig, a, b):
    def func(i, j):
        x = 1 - (1 + (2 ** (a / b) - 1) * ((np.linalg.norm(j - i)) / sig) ** a) ** (
            -b / a
        )
        return x

    return func


def periodicity_closure(periodicity):
    def func(i, j):
        dx = np.linalg.norm(j - i)
        if dx > periodicity * 0.5:
            dx = dx - periodicity
        if dx <= -periodicity * 0.5:
            dx = dx + periodicity
        dx = np.abs(dx)
        return dx

    return func


def sigmoid(r, sig, a, b):
    return 1 - (1 + (2 ** (a / b) - 1) * (r / sig) ** a) ** (-b / a)


def periodicity_closure_w_sigmoid(sig, a, b, periodicity):
    def func(i, j):
        dx = np.linalg.norm(j - i)
        if dx > periodicity * 0.5:
            dx = dx - periodicity
        if dx <= -periodicity * 0.5:
            dx = dx + periodicity
        dx = np.abs(dx)
        dx = 1 - (1 + (2 ** (a / b) - 1) * (dx / sig) ** a) ** (-b / a)
        return dx

    return func


##############################################################################
# Tests
##############################################################################


class TestDistanceLossScipy(tf.test.TestCase):
    def test_non_periodic(self):
        highd = (
            np.random.random((256, 51)).astype("float32") * 100
        )  # high values in feature space
        lowd = (
            np.random.random((256, 2)).astype("float32") * 10
        )  # lower values in latent space
        sig_params = Parameters.defaults["dist_sig_parameters"]

        # encodermap1
        from encodermap.encodermap_tf1.misc import distance_cost as distance_cost_em1

        cost_em1 = distance_cost_em1(highd, lowd, *sig_params, float("inf"))

        # encodermap2
        from encodermap.loss_functions.loss_functions import (
            sigmoid_loss as distance_loss_em2,
        )

        sig_loss_func = distance_loss_em2(periodicity_overwrite=float("inf"))
        cost_em2 = sig_loss_func(highd, lowd)

        # scipy
        highd_sigmoid_func = sigmoid_closure(*sig_params[:3])
        lowd_sigmoid_func = sigmoid_closure(*sig_params[3:])
        sig_h = cdist(highd, highd, highd_sigmoid_func)
        sig_l = cdist(lowd, lowd, lowd_sigmoid_func)
        cost_scipy = np.mean(np.square(sig_h - sig_l))

        self.assertEqual(cost_em1, cost_em2)
        self.assertAllClose(cost_em1.numpy(), cost_scipy)
        self.assertAllClose(cost_em2.numpy(), cost_scipy)

    def test_periodic(self):
        highd = (
            np.random.random((256, 51)).astype("float32") * 2 * np.pi
        ) - np.pi  # high values in feature space
        lowd = (
            np.random.random((256, 2)).astype("float32") * 10
        )  # lower values in latent space
        sig_params = Parameters.defaults["dist_sig_parameters"]

        # encodermap1
        from encodermap.encodermap_tf1.misc import distance_cost as distance_cost_em1

        cost_em1 = distance_cost_em1(highd, lowd, *sig_params, 2 * np.pi)

        # encodermap2
        from encodermap.loss_functions.loss_functions import (
            sigmoid_loss as distance_loss_em2,
        )

        sig_loss_func = distance_loss_em2()
        cost_em2 = sig_loss_func(highd, lowd)

        # scipy
        highd_sigmoid_func = periodicity_closure_w_sigmoid(*sig_params[:3], 2 * np.pi)
        lowd_sigmoid_func = sigmoid_closure(*sig_params[3:])
        sig_h = cdist(highd, highd, highd_sigmoid_func)
        sig_l = cdist(lowd, lowd, lowd_sigmoid_func)
        cost_scipy = np.mean(np.square(sig_h - sig_l))

        # numpy
        a = np.expand_dims(highd, axis=1)
        b = np.expand_dims(highd, axis=0)
        d = np.abs(b - a)
        d = np.minimum(d, 2 * np.pi - d)
        mask = np.equal(d, 0.0)
        d = d + mask * 1e-16
        d = np.linalg.norm(d, axis=2)

        sig_h = sigmoid(d, *sig_params[:3])

        lowd_sigmoid_func = sigmoid_closure(*sig_params[3:])
        sig_l = cdist(lowd, lowd, lowd_sigmoid_func)
        cost_numpy = np.mean(np.square(sig_h - sig_l))

        self.assertEqual(cost_em1, cost_em2)
        self.assertAllClose(cost_em1.numpy(), cost_numpy)
        self.assertAllClose(cost_em2.numpy(), cost_numpy)
        self.assertAllClose(cost_numpy, cost_scipy, atol=1e-3)


class TestLossesNonPeriodic(tf.test.TestCase):
    def test_losses_not_periodic(self):
        p = Parameters(batch_size=5, l2_reg_constant=0)
        input_dim = 10
        len_data = 100
        model = ConstantOutputAutoencoder(input_dim, len_data, p)
        inp = np.random.random((len_data, input_dim)).astype("float32")

        # center loss
        center_loss = loss_functions.center_loss(model)
        self.assertEqual(tf.reduce_sum(tf.abs(model.encoder(inp))), 0)
        self.assertEqual(center_loss(inp), 0)

        # Distance loss
        # Four cases
        # 1) input uniform output uniform == 0
        # 2) input uniform output random != 0 (without abs it would be < 0)
        # 3) input random output uniform != 0 (without abs it would be > 0)
        # 4) input random output random != 0
        # Case 1
        inp = np.ones((len_data, input_dim)).astype("float32")
        model_ = ConstantOutputAutoencoder(input_dim, len_data, p, latent_constant=1)
        distance_loss_tf1_case1 = old_distance_loss(model_)(inp)
        self.assertEqual(distance_loss_tf1_case1, 0)
        distance_loss_tf2_case1 = loss_functions.distance_loss(model_)(inp)
        self.assertEqual(distance_loss_tf2_case1, 0)
        self.assertEqual(distance_loss_tf1_case1, distance_loss_tf2_case1)
        # Case 2 # I don't know, why this Test fails. I can perfectly reproduce it in a normal python environment
        # This test also succeeds, when called with python -m unittest discover -s tests -p *losses*
        # But pytest tests makes this tets fail. pytest test
        # Excluding everything besides test_losses.py
        # pytest tests --html=docs/source/_static/pytest_report.html --self-contained-html --ignore-glob='!*losses!'
        inp = np.zeros((len_data, input_dim)).astype("float32")
        model_ = ConstantOutputAutoencoder(
            input_dim, len_data, p, latent_constant="random"
        )
        latent = model_.encoder(inp)
        self.assertEqual(model_.constant_layer._name, "Latent_random")
        self.assertFalse(
            np.array_equal(latent, np.zeros((len_data, 2))),
            msg=f"Encoder output is {model_.encoder(inp)}, which is not expected. It should be random.",
        )
        self.assertFalse(
            np.array_equal(latent, np.ones((len_data, 2))),
            msg=f"Encoder output is {model_.encoder(inp)}, which is not expected. It should be random.",
        )
        distance_loss_tf1_case2 = old_distance_loss(model_)(inp)
        if not isinstance(distance_loss_tf1_case2, (int, np.integer)):
            distance_loss_tf1_case2 = distance_loss_tf1_case2.numpy()
        self.assertNotEqual(
            distance_loss_tf1_case2,
            0,
            msg=f"Uniform input (np.zeros(100, 10) {inp[:5,:5]} and random latent (100, 2) {latent[:5,:]} did not NOT equal 0 in distance_loss_tf1 (non-periodic), but {distance_loss_tf1_case2}",
        )
        distance_loss_tf2_case2 = loss_functions.distance_loss(model_)(inp)
        self.assertNotEqual(
            distance_loss_tf2_case2.numpy(),
            0,
            msg=f"Uniform input (np.zeros(100, 10) {inp[:5,:5]} and random latent (100, 2) {latent[:5,:]} did not NOT equal 0 in distance_loss_tf2 (non-periodic), but {distance_loss_tf2_case2}",
        )
        self.assertEqual(distance_loss_tf1_case2, distance_loss_tf2_case2)
        # Case 3

        inp = np.random.random((len_data, input_dim)).astype("float32")
        model_ = ConstantOutputAutoencoder(input_dim, len_data, p, latent_constant=0)
        distance_loss_tf1_case3 = old_distance_loss(model_)(inp)
        self.assertAllClose(distance_loss_tf1_case3.numpy(), 0)
        distance_loss_tf2_case3 = loss_functions.distance_loss(model_)(inp)
        self.assertNotEqual(distance_loss_tf2_case3, 0)
        self.assertEqual(distance_loss_tf1_case3, distance_loss_tf2_case3)
        # Case 4
        inp = np.random.random((len_data, input_dim)).astype("float32")
        model_ = ConstantOutputAutoencoder(
            input_dim, len_data, p, latent_constant="random"
        )
        distance_loss_tf1_case4 = old_distance_loss(model_)(inp)
        self.assertNotEqual(distance_loss_tf1_case4, 0)
        distance_loss_tf2_case4 = loss_functions.distance_loss(model_)(inp)
        self.assertNotEqual(distance_loss_tf2_case4, 0)
        self.assertEqual(distance_loss_tf1_case4, distance_loss_tf2_case4)

        # reg loss
        inp = np.random.random((len_data, input_dim)).astype("float32")
        reg_loss = loss_functions.regularization_loss(model)
        self.assertEqual(reg_loss(inp), 0)

        # auto loss
        inp = np.random.random((len_data, input_dim)).astype("float32")
        inp2 = np.random.random((len_data, input_dim)).astype("float32")
        auto_loss = loss_functions.auto_loss(model, p)
        self.assertEqual(auto_loss(inp), 0)
        self.assertNotEqual(auto_loss(inp, inp2), 0)


class TestLossesPeriodic(tf.test.TestCase):
    def test_losses_periodic(self):
        p = Parameters(batch_size=5, l2_reg_constant=0, periodicity=2 * np.pi)
        input_dim = 10
        len_data = 100
        model = ConstantOutputAutoencoder(input_dim, len_data, p)
        inp = (
            np.random.random((len_data, input_dim)).astype("float32") * 2 * np.pi
        ) - np.pi

        # center loss
        center_loss = loss_functions.center_loss(model)
        self.assertEqual(tf.reduce_sum(tf.abs(model.encoder(inp))), 0)
        self.assertEqual(center_loss(inp), 0)

        # Distance loss
        # Four cases
        # 1) input uniform output uniform == 0
        # 2) input uniform output random != 0 (without abs it would be < 0)
        # 3) input random output uniform != 0 (without abs it would be > 0)
        # 4) input random output random != 0
        # Case 1
        inp = np.ones((len_data, input_dim)).astype("float32")
        model_ = ConstantOutputAutoencoder(input_dim, len_data, p, latent_constant=1)
        distance_loss_tf1_case1 = old_distance_loss(model_)(inp)
        self.assertEqual(distance_loss_tf1_case1, 0)
        distance_loss_tf2_case1 = loss_functions.distance_loss(model_)(inp)
        self.assertEqual(distance_loss_tf2_case1, 0)
        self.assertEqual(distance_loss_tf1_case1, distance_loss_tf2_case1)
        # Case 2
        inp = np.zeros((len_data, input_dim)).astype("float32")
        model_ = ConstantOutputAutoencoder(
            input_dim, len_data, p, latent_constant="random"
        )
        latent = model_.encoder(inp)
        self.assertEqual(model_.constant_layer._name, "Latent_random")
        self.assertFalse(
            np.array_equal(latent, np.zeros((len_data, 2))),
            msg=f"Encoder output is {model_.encoder(inp)}, which is not expected. It should be random.",
        )
        self.assertFalse(
            np.array_equal(latent, np.ones((len_data, 2))),
            msg=f"Encoder output is {model_.encoder(inp)}, which is not expected. It should be random.",
        )
        distance_loss_tf1_case2 = old_distance_loss(model_)(inp)
        if not isinstance(distance_loss_tf1_case2, (int, np.integer)):
            distance_loss_tf1_case2 = distance_loss_tf1_case2.numpy()
        self.assertNotEqual(
            distance_loss_tf1_case2,
            0,
            msg=f"Uniform input (np.zeros(100, 10) {inp[:5, :5]} and random latent (100, 2) {latent[:5, :]} did not NOT equal 0 in distance_loss_tf1 (periodic), but {distance_loss_tf1_case2}",
        )
        distance_loss_tf2_case2 = loss_functions.distance_loss(model_)(inp)
        self.assertNotEqual(
            distance_loss_tf2_case2.numpy(),
            0,
            msg=f"Uniform input (np.zeros(100, 10) {inp[:5, :5]} and random latent (100, 2) {latent[:5, :]} did not NOT equal 0 in distance_loss_tf2 (periodic), but {distance_loss_tf2_case2}",
        )
        self.assertEqual(distance_loss_tf1_case2, distance_loss_tf2_case2)
        # Case 3
        inp = np.random.random((len_data, input_dim)).astype("float32")
        model_ = ConstantOutputAutoencoder(input_dim, len_data, p, latent_constant=0)
        distance_loss_tf1_case3 = old_distance_loss(model_)(inp)
        self.assertNotEqual(distance_loss_tf1_case3, 0)
        distance_loss_tf2_case3 = loss_functions.distance_loss(model_)(inp)
        self.assertNotEqual(distance_loss_tf2_case3, 0)
        self.assertEqual(distance_loss_tf1_case3, distance_loss_tf2_case3)
        # Case 4
        inp = np.random.random((len_data, input_dim)).astype("float32")
        model_ = ConstantOutputAutoencoder(
            input_dim, len_data, p, latent_constant="random"
        )
        distance_loss_tf1_case4 = old_distance_loss(model_)(inp)
        self.assertNotEqual(distance_loss_tf1_case4, 0)
        distance_loss_tf2_case4 = loss_functions.distance_loss(model_)(inp)
        self.assertNotEqual(distance_loss_tf2_case4, 0)
        self.assertEqual(distance_loss_tf1_case4, distance_loss_tf2_case4)

        # reg loss
        inp = (
            np.random.random((len_data, input_dim)).astype("float32") * 2 * np.pi
        ) - np.pi
        reg_loss = loss_functions.regularization_loss(model)
        self.assertEqual(reg_loss(inp), 0)

        # auto loss
        inp = (
            np.random.random((len_data, input_dim)).astype("float32") * 2 * np.pi
        ) - np.pi
        inp2 = (
            np.random.random((len_data, input_dim)).astype("float32") * 2 * np.pi
        ) - np.pi
        auto_loss = loss_functions.auto_loss(model, p)
        self.assertEqual(auto_loss(inp), 0)
        self.assertNotEqual(auto_loss(inp, inp2), 0)


class TestLossesADCAutoencoder(tf.test.TestCase):
    def test_losses_ADC(self):
        # create parameters and data
        p = ADCParameters(
            l2_reg_constant=0,
            periodicity=2 * np.pi,
            use_backbone_angles=True,
            distance_cost_scale=1,
            cartesian_dist_sig_parameters=(1, 1, 1, 1, 1, 1),
            cartesian_cost_scale_soft_start=(6, 12),
        )
        # print(p)
        no_central_cartesians = 474  # same as 1am7 protein dihedral length
        if p.use_backbone_angles:
            input_dim = no_central_cartesians - 3 + no_central_cartesians - 2
        else:
            input_dim = no_central_cartesians - 3
        len_data = 100
        model_0 = ConstantOutputAutoencoder(input_dim, p.batch_size, p)
        # model_1 for different latent checks
        model_1 = ConstantOutputAutoencoder(input_dim, p.batch_size, p, 1)

        dihedral_data = (
            np.random.random((len_data, no_central_cartesians - 3)).astype("float32")
            * np.pi
            * 2
        ) - np.pi
        angle_data = (
            np.random.random((len_data, no_central_cartesians - 2)).astype("float32")
            * np.pi
            * 2
        ) - np.pi
        distance_data = np.random.random((len_data, no_central_cartesians - 1)).astype(
            "float32"
        )
        central_cartesian_data = np.random.random(
            (len_data, no_central_cartesians, 3)
        ).astype("float32")
        cartesian_data = np.random.random(
            (len_data, no_central_cartesians * 5, 3)
        ).astype("float32")

        dataset = tf.data.Dataset.from_tensor_slices(
            (angle_data, dihedral_data, central_cartesian_data)
        )
        dataset = dataset.shuffle(buffer_size=len_data, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(p.batch_size)

        for i, d in enumerate(dataset):
            if i == 0:
                inp_angles, inp_dihedrals, inp_cartesians = d
            elif i == 1:
                inp_angles_2, inp_dihedrals_2, inp_cartesians_2 = d
            else:
                break

        # what to pass through network
        if p.use_backbone_angles:
            print("using backbone angles")
            main_inputs = tf.concat([inp_angles, inp_dihedrals], 1)
        else:
            print("not using backbone angles")
            main_inputs = inp_dihedrals

        # run inputs through network
        encoded = model_0.encoder(main_inputs)
        decoded = model_0(main_inputs)
        self.assertAllEqual(main_inputs, decoded)

        # unpack the output
        if p.use_backbone_angles:
            assert decoded.shape == main_inputs.shape
            out_angles = decoded[:, : angle_data.shape[1]]
            out_dihedrals = decoded[:, angle_data.shape[1] :]
        else:
            out_dihedrals = model_0(main_inputs)
            # If angles are not trained, use the mean from all provided angles
            out_angles = tf.tile(
                np.expand_dims(np.mean(angle_data, 0), 0),
                multiples=(out_dihedrals.shape[0], 1),
            )

        self.assertAllEqual(inp_dihedrals, out_dihedrals)
        self.assertAllEqual(inp_angles, out_angles)

        # losses
        # distance_cost
        dist_cost = loss_functions.distance_loss(model_0, p)
        self.assertNotEqual(dist_cost(main_inputs), 0)
        self.assertEqual(dist_cost(tf.zeros(main_inputs.shape)), 0)

        # center loss
        center_loss = loss_functions.center_loss(model_0, p)
        self.assertEqual(center_loss(main_inputs), 0)
        center_loss = loss_functions.center_loss(model_1, p)
        self.assertEqual(center_loss(tf.ones(main_inputs.shape)), p.center_cost_scale)

        # regularization loss
        # is zero because l2_reg_constant was set to zero
        reg_loss = loss_functions.regularization_loss(model_0)
        self.assertEqual(reg_loss(), 0)

        # dihedral loss
        dihedral_loss = loss_functions.dihedral_loss(model_0, p)
        self.assertEqual(dihedral_loss(inp_dihedrals, model_0(inp_dihedrals)), 0)
        self.assertNotEqual(dihedral_loss(inp_dihedrals, model_1(inp_dihedrals_2)), 0)
        self.assertNotEqual(dihedral_loss(inp_dihedrals, model_0(inp_dihedrals_2)), 0)

        # angle loss
        angle_loss = loss_functions.angle_loss(model_0, p)
        self.assertEqual(angle_loss(inp_angles, model_0(inp_angles)), 0)
        self.assertNotEqual(angle_loss(inp_angles, model_1(inp_angles_2)), 0)
        self.assertNotEqual(angle_loss(inp_angles, model_0(inp_angles_2)), 0)

        # cartesian distance loss is pairwise of inp cartesians vs pairwise of latent
        # get output cartesians from backmapping
        mean_lengths = np.expand_dims(np.mean(distance_data, 0), 0)
        out_chain_in_plane = chain_in_plane(mean_lengths, out_angles)
        inp_chain_in_plane = chain_in_plane(mean_lengths, inp_angles)
        self.assertAllEqual(inp_chain_in_plane, out_chain_in_plane)
        out_cartesians = dihedrals_to_cartesian_tf(
            out_dihedrals + np.pi, out_chain_in_plane
        )
        inp_cartesians_backmapped = dihedrals_to_cartesian_tf(
            inp_dihedrals + np.pi, inp_chain_in_plane
        )
        self.assertAllEqual(inp_cartesians_backmapped, out_cartesians)

        # check diff. If diff is 0.0 backmapping is either perfect or wrong
        diff = np.mean(inp_cartesians_backmapped - inp_cartesians)
        if diff == 0:
            warnings.warn(
                "\033[1;37;91m Difference between backmapped and input cartesian coordinates is zero. Either backmapping is perfect, or something is wrong.\033[0m"
            )
        # print('diff:', diff)

        # define the loss
        cartesian_distance_loss = loss_functions.cartesian_distance_loss(model_0, p)

        # calculate the pairwise stuff use only the first 10 entries, to save memory
        inp_pair = pairwise_dist(
            inp_cartesians[
                :, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )[:10]
        inp_pair_2 = pairwise_dist(
            inp_cartesians_2[
                :, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )[:10]
        inp_pair_backmapped = pairwise_dist(
            inp_cartesians_backmapped[
                :, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )[:10]
        out_pair = pairwise_dist(
            out_cartesians[
                :, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )[:10]

        self.assertAllEqual(inp_pair_backmapped, out_pair)
        self.assertNotAllEqual(inp_pair, inp_pair_2)

        self.assertNotEqual(cartesian_distance_loss(inp_pair, encoded[:10]), 0)
        self.assertEqual(
            cartesian_distance_loss(tf.zeros(inp_pair.shape), encoded[:10]), 0
        )
        self.assertEqual(cartesian_distance_loss(inp_pair, inp_pair), 0)
        self.assertEqual(
            cartesian_distance_loss(tf.ones(inp_pair.shape), tf.ones(inp_pair.shape)), 0
        )
        self.assertNotEqual(
            cartesian_distance_loss(inp_pair_backmapped, encoded[:10]), 0
        )

        # cartesian cost is pairwise of inp_cartesians vs pairwise of out_cartesians
        # three cases:
        # step < a => cost_scale = 0
        # a <= step <= b => cost_scale = p.cartesian_cost_scale / (b - a) * step
        # b < step: cost_scale = p.cartesian_cost_scale

        from encodermap.trajinfo.info_all import Capturing

        callback = callbacks.IncreaseCartesianCost(p, start_step=0)

        # step 0 no scale
        with Capturing() as output:
            cartesian_loss = loss_functions.cartesian_loss(
                model_0, callback, p, print_current_scale=True
            )
        self.assertEqual(
            output,
            [
                "<tf.Variable 'current_cartesian_cost_scale:0' shape=() dtype=float32, numpy=0.0>"
            ],
        )
        self.assertEqual(cartesian_loss(inp_pair, out_pair), 0)
        self.assertEqual(cartesian_loss(inp_pair, inp_pair_backmapped), 0)

        # step 9 scale is 0.5
        callback = callbacks.IncreaseCartesianCost(p, start_step=9)
        with Capturing() as output:
            cartesian_loss = loss_functions.cartesian_loss(
                model_0, callback, p, print_current_scale=True
            )
        self.assertEqual(
            output,
            [
                "<tf.Variable 'current_cartesian_cost_scale:0' shape=() dtype=float32, numpy=0.5>"
            ],
        )
        self.assertNotEqual(cartesian_loss(inp_pair, out_pair), 0)
        self.assertEqual(cartesian_loss(inp_pair, inp_pair), 0)

        # step 12 scale is 1.0
        callback = callbacks.IncreaseCartesianCost(p, start_step=12)
        with Capturing() as output:
            cartesian_loss = loss_functions.cartesian_loss(
                model_0, callback, p, print_current_scale=True
            )
        self.assertEqual(
            output,
            [
                "<tf.Variable 'current_cartesian_cost_scale:0' shape=() dtype=float32, numpy=1.0>"
            ],
        )
        self.assertNotEqual(cartesian_loss(inp_pair, out_pair), 0)
        self.assertEqual(cartesian_loss(inp_pair, inp_pair), 0)


# Remove Phantom Tests from tensorflow skipped test_session
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
test_cases = (
    TestLossesPeriodic,
    TestLossesNonPeriodic,
    TestLossesADCAutoencoder,
    TestDistanceLossScipy,
)

# doctests
import doctest

import encodermap.loss_functions.loss_functions as loss_functions

doc_tests = (doctest.DocTestSuite(loss_functions),)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    suite.addTests(doc_tests)
    return suite


# unittest.TextTestRunner(verbosity = 2).run(testSuite)

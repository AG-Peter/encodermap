# -*- coding: utf-8 -*-
# tests/test_backmapping_em1_em2.py
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
import unittest

import numpy as np
import tensorflow as tf

import encodermap as em
from encodermap.encodermap_tf1.backmapping import (
    chain_in_plane,
    dihedrals_to_cartesian_tf,
)
from encodermap.encodermap_tf1.backmapping import guess_amide_H as guess_amide_H_tf1
from encodermap.encodermap_tf1.backmapping import guess_amide_O as guess_amide_O_tf1
from encodermap.encodermap_tf1.backmapping import (
    merge_cartesians as merge_cartesians_tf1,
)
from encodermap.misc import pairwise_dist
from encodermap.misc.backmapping import (
    guess_amide_H,
    guess_amide_O,
    merge_cartesians,
    split_and_reverse_cartesians,
    split_and_reverse_dihedrals,
)
from encodermap.models.models import SequentialModel


class LayerThatOutputsConstant(tf.keras.layers.Layer):
    def __init__(self, units, output_len, output_constant=0, name="Latent"):
        super(LayerThatOutputsConstant, self).__init__()
        self.output_constant = tf.constant(
            output_constant, shape=(output_len, units), dtype="float32"
        )
        self._name = name

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
        constant_layer = LayerThatOutputsConstant(
            self.p.n_neurons[-1], parameters.batch_size, latent_constant, name="Latent"
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
            + [constant_layer],
            name="Encoder",
        )
        self.build(input_shape=(1, input_dim))

    def call(self, x):
        return x


class TestBackmappingEm1Em2(tf.test.TestCase):
    def test_backmapping_wo_angles(self):
        p = em.ADCParameters(l2_reg_constant=0, periodicity=2 * np.pi)
        print(p)
        no_central_cartesians = 474  # same as 1am7 protein dihedral length
        if p.use_backbone_angles:
            input_dim = no_central_cartesians - 3 + no_central_cartesians - 2
        else:
            input_dim = no_central_cartesians - 3
        len_data = 100
        model_0 = ConstantOutputAutoencoder(input_dim, len_data, p)
        # model_1 for different latent checks
        model_1 = ConstantOutputAutoencoder(input_dim, len_data, p, 1)
        # model.compile(tf.keras.optimizers.Adam())

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
                angles, dihedrals, cartesians = d
            elif i == 1:
                angles_2, dihedrals_2, cartesians_2 = d
            else:
                break

        # what to pass through network
        if p.use_backbone_angles:
            print("using backbone angles")
            main_inputs = tf.concat([angles, dihedrals], 1)
        else:
            print("not using backbone angles")
            main_inputs = dihedrals

        # run inputs through network
        print("main_inputs:", main_inputs.shape)
        encoded = model_0.encoder(main_inputs)
        print("encoded:", encoded.shape)
        decoded = model_0(main_inputs)
        print(tf.math.reduce_all(tf.equal(main_inputs, decoded)))
        print("decoded:", decoded.shape)

        # unpack the output
        if p.use_backbone_angles:
            assert decoded.shape == main_inputs.shape
            generated_angles = decoded[:, : angle_data.shape[1]]
            generated_dihedrals = decoded[:, angle_data.shape[1] :]
        else:
            generated_dihedrals = model_0(main_inputs)
            # If angles are not trained, use the mean from all provided angles
            generated_angles = tf.tile(
                np.expand_dims(np.mean(angle_data, 0), 0),
                multiples=(out_dihedrals.shape[0], 1),
            )

        # mean lengths over trajectory used for backmapping
        mean_lengths = np.expand_dims(np.mean(distance_data, 0), 0)

        # build s chain in plane with lengths and angles
        _chain_in_plane = chain_in_plane(mean_lengths, generated_angles)

        # add a third dimension by adding torsion angles to that chain
        cartesians = dihedrals_to_cartesian_tf(
            generated_dihedrals + np.pi, _chain_in_plane
        )

        # for a standard protein these names are always the same, that's why I removed them
        # and exchanged them with indices
        atom_names = ["N", "CA", "C"] * int(no_central_cartesians / 3)

        # compare H cartesians
        amide_H_cartesians_tf1 = guess_amide_H_tf1(cartesians, atom_names)
        amide_H_cartesians = guess_amide_H(
            cartesians, np.arange(cartesians.shape[1])[::3]
        )
        self.assertAllEqual(amide_H_cartesians_tf1, amide_H_cartesians)

        # compare O cartesians
        amide_O_cartesians_tf1 = guess_amide_O_tf1(cartesians, atom_names)
        amide_O_cartesians = guess_amide_O(
            cartesians, np.arange(cartesians.shape[1])[2::3]
        )
        self.assertAllEqual(amide_O_cartesians_tf1, amide_O_cartesians)

        # merge the cartesians from chain_in_plane backmapping with the amide atoms
        merged_cartesians_tf1 = merge_cartesians_tf1(
            cartesians, atom_names, amide_H_cartesians_tf1, amide_O_cartesians_tf1
        )
        merged_cartesians = merge_cartesians(
            cartesians,
            np.arange(cartesians.shape[1])[::3],
            np.arange(cartesians.shape[1])[2::3],
            amide_H_cartesians,
            amide_O_cartesians,
        )
        self.assertAllEqual(merged_cartesians_tf1, merged_cartesians)

        # These are here to test whether they run or not
        # They will be tested with the always zero model in the test_losses unittest
        inp_pairwise = pairwise_dist(
            cartesian_data[
                :10, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )
        encoded_pairwise = pairwise_dist(
            cartesians[
                :10, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )
        clashes = tf.math.count_nonzero(
            pairwise_dist(cartesians, flat=True) < 1, axis=1, dtype=tf.float32
        )

    def test_backmapping_wo_angles(self):
        p = em.ADCParameters(
            l2_reg_constant=0, periodicity=2 * np.pi, use_backbone_angles=True
        )
        print(p)
        no_central_cartesians = 474  # same as 1am7 protein dihedral length
        if p.use_backbone_angles:
            input_dim = no_central_cartesians - 3 + no_central_cartesians - 2
        else:
            input_dim = no_central_cartesians - 3
        len_data = 100
        model_0 = ConstantOutputAutoencoder(input_dim, len_data, p)
        # model_1 for different latent checks
        model_1 = ConstantOutputAutoencoder(input_dim, len_data, p, 1)
        # model.compile(tf.keras.optimizers.Adam())

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
                angles, dihedrals, cartesians = d
            elif i == 1:
                angles_2, dihedrals_2, cartesians_2 = d
            else:
                break

        # what to pass through network
        if p.use_backbone_angles:
            print("using backbone angles")
            main_inputs = tf.concat([angles, dihedrals], 1)
        else:
            print("not using backbone angles")
            main_inputs = dihedrals

        # run inputs through network
        print("main_inputs:", main_inputs.shape)
        encoded = model_0.encoder(main_inputs)
        print("encoded:", encoded.shape)
        decoded = model_0(main_inputs)
        print(tf.math.reduce_all(tf.equal(main_inputs, decoded)))
        print("decoded:", decoded.shape)

        # unpack the output
        if p.use_backbone_angles:
            assert decoded.shape == main_inputs.shape
            generated_angles = decoded[:, : angle_data.shape[1]]
            generated_dihedrals = decoded[:, angle_data.shape[1] :]
        else:
            generated_dihedrals = model_0(main_inputs)
            # If angles are not trained, use the mean from all provided angles
            generated_angles = tf.tile(
                np.expand_dims(np.mean(angle_data, 0), 0),
                multiples=(out_dihedrals.shape[0], 1),
            )

        # mean lengths over trajectory used for backmapping
        mean_lengths = np.expand_dims(np.mean(distance_data, 0), 0)

        # build s chain in plane with lengths and angles
        _chain_in_plane = chain_in_plane(mean_lengths, generated_angles)

        # add a third dimension by adding torsion angles to that chain
        cartesians = dihedrals_to_cartesian_tf(
            generated_dihedrals + np.pi, _chain_in_plane
        )

        # for a standard protein these names are always the same, that's why I removed them
        # and exchanged them with indices
        atom_names = ["N", "CA", "C"] * int(no_central_cartesians / 3)

        # compare H cartesians
        amide_H_cartesians_tf1 = guess_amide_H_tf1(cartesians, atom_names)
        amide_H_cartesians = guess_amide_H(
            cartesians, np.arange(cartesians.shape[1])[::3]
        )
        self.assertAllEqual(amide_H_cartesians_tf1, amide_H_cartesians)

        # compare O cartesians
        amide_O_cartesians_tf1 = guess_amide_O_tf1(cartesians, atom_names)
        amide_O_cartesians = guess_amide_O(
            cartesians, np.arange(cartesians.shape[1])[2::3]
        )
        self.assertAllEqual(amide_O_cartesians_tf1, amide_O_cartesians)

        # merge the cartesians from chain_in_plane backmapping with the amide atoms
        merged_cartesians_tf1 = merge_cartesians_tf1(
            cartesians, atom_names, amide_H_cartesians_tf1, amide_O_cartesians_tf1
        )
        merged_cartesians = merge_cartesians(
            cartesians,
            np.arange(cartesians.shape[1])[::3],
            np.arange(cartesians.shape[1])[2::3],
            amide_H_cartesians,
            amide_O_cartesians,
        )
        self.assertAllEqual(merged_cartesians_tf1, merged_cartesians)

        # These are here to test whether they run or not
        # They will be tested with the always zero model in the test_losses unittest
        inp_pairwise = pairwise_dist(
            cartesian_data[
                :10, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )
        encoded_pairwise = pairwise_dist(
            cartesians[
                :10, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )
        clashes = tf.math.count_nonzero(
            pairwise_dist(cartesians, flat=True) < 1, axis=1, dtype=tf.float32
        )


class TestCompareSplits(tf.test.TestCase):
    def test_random_shapes(self):
        for i in np.random.randint(0, 1000, size=10):
            # create example tensors
            example_dihedrals = (
                tf.convert_to_tensor(np.random.random((256, i - 3))) + np.pi
            )
            example_cartesians = tf.convert_to_tensor(np.random.random((256, i, 3)))

            # print('example_dihedrals.shape:', example_dihedrals.shape)
            # print('example_cartesians.shape:', example_cartesians.shape)

            if len(example_cartesians.get_shape()) == 2:
                expanded = tf.expand_dims(example_cartesians, axis=0)
                example_cartesians = tf.tile(expanded, [256, 1, 1])

            split = int(int(example_cartesians.shape[1]) / 2)

            # old way:
            # dihedrals needs to be an even split
            # the middle element of cartesians needs to be repeated, because it belongs to two dihedrals
            cartesian_left = example_cartesians[:, split + 1 :: -1]
            dihedrals_left = example_dihedrals[:, split - 2 :: -1]
            cartesian_right = example_cartesians[:, split - 1 :]
            dihedrals_right = example_dihedrals[:, split - 1 :]

            # print(cartesian_left.shape, dihedrals_left.shape)
            # print(cartesian_right.shape, dihedrals_right.shape)

            # new way
            dihedrals_left_test, dihedrals_right_test = split_and_reverse_dihedrals(
                example_dihedrals
            )
            cartesians_left_test, cartesians_right_test = split_and_reverse_cartesians(
                example_cartesians
            )
            # print(cartesians_left_test.shape, dihedrals_left_test.shape)
            # print(cartesians_right_test.shape, dihedrals_right_test.shape)

            self.assertAllEqual(dihedrals_left_test, dihedrals_left)
            self.assertAllEqual(dihedrals_right_test, dihedrals_right)
            self.assertAllEqual(cartesians_left_test, cartesian_left)
            self.assertAllEqual(cartesians_right_test, cartesian_right)


# print('generated_angles:', generated_angles.shape)
# print('generated_dihedrals:', generated_dihedrals.shape)
# mean_lengths = np.expand_dims(np.mean(distance_data, 0), 0)
# print('mean_lengths:', mean_lengths.shape)
# # build s chain in plane with lengths and angles
# _chain_in_plane = chain_in_plane(mean_lengths, generated_angles)
# print('chain_in_plane:', _chain_in_plane.shape)
# # add a third dimension by adding torsion angles to that chain
# cartesians = dihedrals_to_cartesian_tf(generated_dihedrals + np.pi, _chain_in_plane)
# print('cartesians:', cartesians.shape)
# atom_names = ['N', 'CA', 'C'] * int(no_central_cartesians / 3)
# amide_H_cartesians_tf1 = guess_amide_H_tf1(cartesians, atom_names)
# print('amide_H_cartesians_tf1:', amide_H_cartesians_tf1.shape, amide_H_cartesians_tf1[0, :5, 0])
# amide_H_cartesians = guess_amide_H(cartesians, np.arange(cartesians.shape[1])[::3])
# print('amide_H_cartesians:', amide_H_cartesians.shape, amide_H_cartesians[0, :5, 0])
# amide_O_cartesians_tf1 = guess_amide_O_tf1(cartesians, atom_names)
# print('amide_O_cartesians_tf1:', amide_O_cartesians_tf1.shape, amide_O_cartesians_tf1[0, :5, 0])
# amide_O_cartesians = guess_amide_O(cartesians, np.arange(cartesians.shape[1])[2::3])
# print('amide_O_cartesians:', amide_O_cartesians.shape, amide_O_cartesians[0, :5, 0])
# merged_cartesians_tf1 = merge_cartesians_tf1(cartesians, atom_names, amide_H_cartesians_tf1, amide_O_cartesians_tf1)
# print('merged_cartesians_tf1:', merged_cartesians_tf1.shape, merged_cartesians_tf1[0, :5, 0])
# merged_cartesians = merge_cartesians(cartesians, np.arange(cartesians.shape[1])[::3], np.arange(cartesians.shape[1])[2::3],
#                                      amide_H_cartesians, amide_O_cartesians)
# print('merged_cartesians:', merged_cartesians.shape, merged_cartesians[0, :5, 0])
# inp_pairwise = pairwise_dist(cartesian_data[:10, p.cartesian_pwd_start:
#                              p.cartesian_pwd_stop:
#                              p.cartesian_pwd_step], flat=True)
# print('inp_pairwise:', inp_pairwise.shape)
# encoded_pairwise = pairwise_dist(cartesians[:10,  p.cartesian_pwd_start:
#                                  p.cartesian_pwd_stop:
#                                  p.cartesian_pwd_step], flat=True)
# print('encoded_pairwise:', encoded_pairwise.shape)
# clashes = tf.math.count_nonzero(pairwise_dist(cartesians, flat=True) < 1, axis=1, dtype=tf.float32)
# print('clashes:', clashes.shape)

# Remove Phantom Tests from tensorflow skipped test_session
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
test_cases = (
    TestBackmappingEm1Em2,
    TestCompareSplits,
)

# doctests
import doctest

import encodermap.misc.backmapping as backmapping

doc_tests = (doctest.DocTestSuite(backmapping),)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    suite.addTests(doc_tests)
    return suite

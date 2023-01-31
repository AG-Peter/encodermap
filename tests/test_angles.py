# -*- coding: utf-8 -*-
# tests/test_angles.py
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
import os
import unittest
from math import pi
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import MDAnalysis as md
import numpy as np
import tensorflow as tf
from matplotlib.testing.compare import compare_images

import encodermap as em
import encodermap.encodermap_tf1 as em_tf1

matplotlib.use("Agg")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# If scipy was compiled against an older version of numpy these warnings are raised
# warnings in a testing environment are somewhat worrying
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class TestAngles(unittest.TestCase):
    def test_ala10_angles(self):
        uni = md.Universe(str(Path(__file__).resolve().parent / "data/Ala10_helix.pdb"))
        print(os.getcwd())
        selected_atoms = uni.select_atoms("backbone or name O1 or name H or name CB")
        moldata = em_tf1.MolData(selected_atoms)
        self.assertTrue(
            np.allclose(
                moldata.angles,
                [
                    [
                        1.9216446,
                        2.0355537,
                        2.128159,
                        1.9212531,
                        2.0357149,
                        2.1278918,
                        1.9220486,
                        2.0346954,
                        2.1269655,
                        1.9218233,
                        2.0352163,
                        2.1275373,
                        1.9212493,
                        2.035614,
                        2.128058,
                        1.9211367,
                        2.0354483,
                        2.128482,
                        1.9212018,
                        2.034529,
                        2.1266387,
                        1.9220015,
                        2.034642,
                        2.1270595,
                        1.9208968,
                        2.0354831,
                        2.127831,
                        1.9212908,
                    ]
                ],
            )
        )

    def test_chain_in_plane(self):
        bond_lengths = tf.constant(
            [[1, 0.5, 1, 0.5, 1, 0.5], [1, 1, 1, 1, 1, 1]], tf.float32
        )
        angles = tf.constant(
            [
                [pi / 1.5, pi / 8, pi / 2, pi / 8, pi / 2],
                [pi / 4, pi / 4, pi / 4, pi / 4, pi / 4],
            ],
            tf.float32,
        )

        cartesians = em_tf1.chain_in_plane(bond_lengths, angles)
        fig, axe = plt.subplots(figsize=(5, 5))
        axe.plot(cartesians.numpy()[0, :, 0], cartesians.numpy()[0, :, 1])
        axe.axis("equal")
        img_name = "test_chain_in_plane"
        plt.savefig(
            Path(__file__).resolve().parent / "data/{}_actual.png".format(img_name),
            dpi=100,
        )
        self.assertIsNone(
            compare_images(
                expected=Path(__file__).resolve().parent
                / "data/{}_expected.png".format(img_name),
                actual=Path(__file__).resolve().parent
                / "data/{}_actual.png".format(img_name),
                tol=10.0,
            )
        )


# Remove Phantom Tests from tensorflow skipped test_session
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
test_cases = (TestAngles,)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite

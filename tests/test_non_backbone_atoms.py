# -*- coding: utf-8 -*-
# tests/test_non_backbone_atoms.py
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


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import os
import unittest
import warnings
from pathlib import Path

# Third Party Imports
import matplotlib
import matplotlib.pyplot as plt
import MDAnalysis as md
import numpy as np
import tensorflow as tf
from matplotlib.testing.compare import compare_images

# Encodermap imports
import encodermap.encodermap_tf1 as em_tf1
from conftest import skip_all_tests_except_env_var_specified


import encodermap as em  # isort: skip


matplotlib.use("Agg")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# If scipy was compiled against an older version of numpy these warnings are raised
# warnings in a testing environment are somewhat worrying
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@skip_all_tests_except_env_var_specified(unittest.skip)
class TestNonBackboneAtoms(tf.test.TestCase):
    def test_guess_amide_H(self):
        cartesians_non_tf = np.array(
            [
                [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0], [5, 1, 0]],
                [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0], [5, 1, 0]],
            ]
        )
        cartesians = tf.constant(cartesians_non_tf, dtype=tf.float32)
        atom_names = ["N", "CA", "C", "N", "CA", "C"]

        H_cartesians = em_tf1.guess_amide_H(cartesians, atom_names).numpy()

        fig, axe = plt.subplots(figsize=(5, 5))
        axe.plot(
            cartesians_non_tf[0, :, 0],
            cartesians_non_tf[0, :, 1],
            linestyle="",
            marker="o",
        )
        axe.plot(H_cartesians[0, :, 0], H_cartesians[0, :, 1], linestyle="", marker="o")
        axe.axis("equal")
        img_name = "test_guess_amide_H"
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

    def test_guess_amide_O(self):
        cartesians_non_tf = np.array(
            [
                [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0], [5, 1, 0]],
                [[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0], [5, 1, 0]],
            ]
        )
        cartesians = tf.constant(cartesians_non_tf, dtype=tf.float32)
        atom_names = ["CA", "N", "C", "CA", "N", "C"]

        O_cartesians = em_tf1.guess_amide_O(cartesians, atom_names).numpy()

        fig, axe = plt.subplots(figsize=(5, 5))
        axe.plot(
            cartesians_non_tf[0, :, 0],
            cartesians_non_tf[0, :, 1],
            linestyle="",
            marker="o",
        )
        axe.plot(O_cartesians[0, :, 0], O_cartesians[0, :, 1], linestyle="", marker="o")
        axe.axis("equal")
        img_name = "test_guess_amide_O"
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
test_cases = (TestNonBackboneAtoms,)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite


if __name__ == "__main__":
    unittest.main()

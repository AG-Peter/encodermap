# -*- coding: utf-8 -*-
# tests/test_dihedral_to_cartesian.py
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
import MDAnalysis as md
import numpy as np
import tensorflow as tf
from matplotlib.testing.compare import compare_images

import encodermap.encodermap_tf1 as em_tf1

matplotlib.use("Agg")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# If scipy was compiled against an older version of numpy these warnings are raised
# warnings in a testing environment are somewhat worrying
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class TestDihedralToCartesianTf(tf.test.TestCase):
    def test_straight_to_helix_array(self):
        phi = (+57.8 / 180) * pi + pi
        psi = (+47.0 / 180) * pi + pi
        omega = 0
        dihedrals = tf.convert_to_tensor(
            np.array([[phi, psi, omega] * 10] * 10, dtype=np.float32)
        )
        result = np.array(
            [
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [1.0, 0.0, 0.0],
                    [1.33166722, 0.94339645, 0.0],
                    [0.96741215, 1.42302374, 0.79829563],
                    [1.08880021, 0.85355615, 1.61129723],
                    [0.72454514, 1.33318344, 2.40959286],
                    [-0.23997373, 1.54789329, 2.25596008],
                    [-0.70886876, 0.73651815, 1.90695029],
                    [-1.67338763, 0.951228, 1.75331751],
                    [-1.74499922, 1.71664339, 1.11377778],
                    [-1.14157263, 1.55412119, 0.33309675],
                    [-1.21318422, 2.31953658, -0.30644299],
                    [-0.97755207, 3.16914765, 0.16540288],
                    [-0.13573978, 3.03560281, 0.68839222],
                    [0.09989237, 3.88521388, 1.16023808],
                    [-0.65820911, 4.15648941, 1.75327411],
                    [-0.96694806, 3.3645823, 2.28011695],
                    [-1.72504954, 3.63585784, 2.87315298],
                    [-2.46428712, 4.01272662, 2.31503316],
                    [-2.63384221, 3.40789298, 1.53694105],
                    [-3.37307979, 3.78476176, 0.97882123],
                    [-3.12477419, 4.70531417, 0.6772792],
                    [-2.18945319, 4.69745201, 0.32356631],
                    [-1.94114759, 5.61800442, 0.02202428],
                    [-2.03578205, 6.25454616, 0.7874385],
                    [-1.60222551, 5.86370596, 1.59939456],
                    [-1.69685998, 6.50024771, 2.36480878],
                    [-2.66649702, 6.68630791, 2.52350853],
                    [-3.17315552, 5.82417585, 2.52855474],
                    [-4.14279257, 6.01023606, 2.68725449],
                    [-4.48234517, 6.62813877, 1.97809986],
                    [-4.17869445, 6.30364579, 1.08227588],
                    [-4.51824706, 6.92154852, 0.37312124],
                ]
            ]
            * 10,
            dtype=np.float32,
        )

        # with self.test_session():
        cartesians = em_tf1.dihedrals_to_cartesian_tf(
            dihedrals, tf.constant(em_tf1.straight_tetrahedral_chain(33))
        ).numpy()
        # self.assertAllClose(result, cartesians, atol=1e-4)

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*np.array(result[0]).T)
        ax.plot(*np.array(cartesians[0]).T)
        set_axes_equal(ax)
        img_name = "test_straight_to_helix_array"
        plt.savefig(
            str(
                Path(__file__).resolve().parent / "data/{}_actual.png".format(img_name)
            ),
            dpi=100,
        )
        self.assertIsNone(
            compare_images(
                expected=str(
                    Path(__file__).resolve().parent
                    / "data/{}_expected.png".format(img_name)
                ),
                actual=str(
                    Path(__file__).resolve().parent
                    / "data/{}_actual.png".format(img_name)
                ),
                tol=10.0,
            )
        )

    def test_straight_tetrahedral_chain_with_bond_lenght(self):
        result = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.6633345, 1.8867929, 0.0],
            [4.6633344, 1.8867929, 0.0],
            [4.995002, 2.8301892, 0.0],
            [6.995002, 2.8301892, 0.0],
            [7.990003, 5.6603785, 0.0],
        ]
        cartesian = em_tf1.straight_tetrahedral_chain(bond_lengths=[1, 2, 3, 1, 2, 3])
        self.assertAllClose(result, cartesian)

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(*cartesian.T)
        # set_axes_equal(ax)
        # plt.show()

    def test_straight_to_helix_v2(self):
        phi = (57.8 / 180) * pi + pi
        psi = (47.0 / 180) * pi + pi

        omega = 0
        dihedrals = tf.constant([[phi, psi, omega] * 10] * 2)

        lengths = tf.constant(np.ones((2, 32), dtype=np.float32))
        angles = tf.constant(np.ones((2, 31), dtype=np.float32) * 120 / 180 * pi)

        # with self.test_session() as sess:
        chain_in_plane = em_tf1.chain_in_plane(lengths, angles)
        cartesians = em_tf1.dihedral_to_cartesian_tf_one_way(
            dihedrals, chain_in_plane
        ).numpy()
        cartesians2 = em_tf1.dihedrals_to_cartesian_tf(
            dihedrals, chain_in_plane
        ).numpy()

        cartesians_old = em_tf1.dihedrals_to_cartesian_tf_old(
            dihedrals, chain_in_plane
        ).numpy()

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # ax.plot(*np.array(result).T)
        ax.plot(*np.array(cartesians[0]).T)
        ax.plot(*np.array(cartesians[0, 0:1]).T, marker="o")
        ax.plot(*np.array(cartesians2[0]).T)
        ax.plot(*np.array(cartesians2[0, 0:1]).T, marker="o")
        ax.plot(*np.array(cartesians_old[0]).T)
        ax.plot(*np.array(cartesians_old[0, 0:1]).T, marker="o")
        set_axes_equal(ax)
        img_name = "test_straight_to_helix_v2"
        plt.savefig(
            str(
                Path(__file__).resolve().parent / "data/{}_actual.png".format(img_name)
            ),
            dpi=100,
        )
        self.assertIsNone(
            compare_images(
                expected=str(
                    Path(__file__).resolve().parent
                    / "data/{}_expected.png".format(img_name)
                ),
                actual=str(
                    Path(__file__).resolve().parent
                    / "data/{}_actual.png".format(img_name)
                ),
                tol=10.0,
            )
        )

        # self.assertAllClose(result, cartesians, atol=1e-4)


# Remove Phantom Tests from tensorflow skipped test_session
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
test_cases = (TestDihedralToCartesianTf,)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite

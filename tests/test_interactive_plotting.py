# -*- coding: utf-8 -*-
# tests/test_interactive_plotting.py
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


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import glob
import os
import shutil
import unittest
from pathlib import Path

# Third Party Imports
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd

# Encodermap imports
from encodermap import ADCParameters, AngleDihedralCartesianEncoderMap
from encodermap.plot.utils import (
    PolygonSelector,
    _unpack_cluster_info,
    get_cluster_frames,
)
from encodermap.trajinfo import SingleTraj, TrajEnsemble


import encodermap as em  # isort: skip


class TestInteractivePlotting(unittest.TestCase):
    def test_interactive_plotting(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part2.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        trajs = TrajEnsemble([traj1, traj2])
        trajs.load_CVs("all")
        os.makedirs("tmp", exist_ok=True)
        p = ADCParameters(
            use_sidechains=True, use_backbone_angles=True, main_path="tmp"
        )
        e_map = AngleDihedralCartesianEncoderMap(trajs, p, read_only=False)

        lowd = e_map.encode()
        trajs.load_CVs(lowd, attr_name="lowd")
        self.assertEqual(trajs.lowd.shape[-1], 2)

        # invent a clustering
        cluster_membership = np.full(trajs.n_frames, -1)
        cluster_membership[::2] = 0
        trajs.load_CVs(cluster_membership, "user_selected_points")
        # get a dummy traj
        _, dummy_traj = get_cluster_frames(
            trajs,
            0,
            nglview=True,
            shorten=True,
            stack_atoms=True,
            col="user_selected_points",
            align_string="name CA",
        )
        self.assertIsInstance(dummy_traj, list)
        self.assertEqual(len(dummy_traj), 10)

        class Selector:
            fig, ax = plt.subplots()
            lasso = PolygonSelector(ax=ax, onselect=None)
            lasso._xs = np.random.random(5)
            lasso._ys = np.random.random(5)
            # repeat first and last point to better replicate PolygonSelector behavior
            lasso._xs = np.stack([*lasso._xs, lasso._xs[0]])
            lasso._ys = np.stack([*lasso._ys, lasso._ys[0]])

        selector = Selector()
        self.assertEqual(selector.lasso._xs[0], selector.lasso._xs[-1])
        self.assertIsInstance(selector.lasso, PolygonSelector)
        cluster_num, main_path = _unpack_cluster_info(
            trajs, e_map.p.main_path, selector, dummy_traj, "name CA"
        )

        new_trajs = TrajEnsemble.from_textfile(
            os.path.join(
                main_path, "cluster_id_0_all_plotted_trajs_in_correct_order.txt"
            )
        )
        self.assertEqual(new_trajs.n_trajs, trajs.n_trajs)
        df = pd.read_csv(os.path.join(main_path, "cluster_id_0_selected_points.csv"))
        self.assertEqual(df.shape[0], np.ceil(len(cluster_membership) / 2))
        combined_n_frames = 0
        for pdb_file, xtc_file in zip(
            glob.glob(main_path + "/*traj*pdb"), glob.glob(main_path + "/*traj*xtc")
        ):
            combined_n_frames += md.load(xtc_file, top=pdb_file).n_frames
        self.assertEqual(combined_n_frames, df.shape[0])

        shutil.rmtree("tmp/")


# Remove Phantom Tests from tensorflow skipped test_session
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
test_cases = (TestInteractivePlotting,)

# Standard Library Imports
# doctests
import doctest


doc_tests = ()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    suite.addTests(doc_tests)
    return suite


# unittest.TextTestRunner(verbosity = 2).run(testSuite)

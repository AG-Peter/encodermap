# -*- coding: utf-8 -*-
# tests/test_interactive_plotting.py
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
import glob
import os
import shutil
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Third Party Imports
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd

# Encodermap imports
from conftest import skip_all_tests_except_env_var_specified
from encodermap import ADCParameters, AngleDihedralCartesianEncoderMap
from encodermap.plot.dashboard import Dashboard
from encodermap.trajinfo import SingleTraj, TrajEnsemble


import encodermap as em  # isort: skip


class MockDashContext:
    pass


mock_dash_context = MockDashContext()


@skip_all_tests_except_env_var_specified(unittest.skip)
class TestDashboard(unittest.TestCase):
    def test_plot_ball_and_stick_can_use_mdtraj(self):
        self.assertTrue(False)

    def assert_can_load_1YUF_from_mstraj(self):
        traj = md.load_pdb("https://files.rcsb.org/view/1YUF.pdb")
        traj = em.SingleTraj(traj[0])

    @patch("encodermap.plot.dashboard.ctx", mock_dash_context)
    def test_dashboard_linear_dimers(self):
        """This is just a regular run through EncoderMap's dashboard with the
        linear-dimers dataset.

        """
        # instantiate the dashboard
        dashboard = Dashboard()

        # load the linear dimers dataset via teh quick select
        mock_dash_context.triggered_id = "linear-dimers-button"
        _, __, main_store = dashboard.upload_traj_page.load_trajs(
            upload_n_clicks=None,
            reset_n_clicks=None,
            linear_dimers_n_clicks=1,
            main_store=None,
            list_of_contents=None,
            list_of_names=None,
            list_of_dates=None,
            textarea_value=None,
            project_value=None,
        )

        # as a next step, we want to look at the topology
        dashboard.top_page.display_plot(
            top_value=0,
            radio_value=0,
            dummy=None,
            rangeslider_value=[0, 1521],  # M1-Ubi dimer has 1521 atoms
        )

        # try a different rangeslider value
        dashboard.top_page.display_plot(
            top_value=0,
            radio_value=0,
            dummy=None,
            rangeslider_value=[20, 1521],  # M1-Ubi dimer has 1521 atoms
        )

        dashboard.top_page.display_plot(
            top_value=0,
            radio_value=0,
            dummy=None,
            rangeslider_value=[430, 450],  # focuses on PHE-45
        )

        self.assertEqual(main_store, {"traj": "linear_dimers", "traj_type": "project"})

        self.assertIsNotNone(
            dashboard.trajs[0].indices_chi1,
        )

        # add some custom topology as a json string
        json_string = '{"PHE45": ["F", {"optional_delete_bonds": [["CE1", "CZ"], ["CD2", "CE2"]], "not_CHI1": null}]}'
        mock_dash_context.triggered_id = "top-json-load"
        dashboard.top_page.display_top(
            json_values=json_string,
            top_value=0,
            n_clicks=None,
            main_store=main_store,
            dummy=None,
        )

        self.assertIsNotNone(
            dashboard.trajs[0].indices_chi1,
        )

        self.fail("Under development")


@skip_all_tests_except_env_var_specified(unittest.skip)
class TestInteractivePlotting(unittest.TestCase):
    def test_instantiation_generate_cluster(self):
        """The Interactive Plotting class can be instantiated with a multitude of parameters:
        From each of these lists, pick one:

        Autoencoder:
            * EncoderMap
            * AngleDihedralCartesianEncoderMap

        Trajs:
            * Ensemble
            * SingleTraj
            * None (skip, when EncoderMap, overwrite and check when AngleDihedralCartesianEncoderMap)

        Lowd:
            * None (use encode)
            * Numpy (check shape)
            * In trajs (check shape)

        Highd:
            * None (use decode)
            * Numpy (check shape)
            * in Autoencoder (use TrainData)
            * In trajs (check shape)

        """
        trajs = em.load_project("pASP_pGLU")
        traj = trajs[0]

        path = np.linspace([-1, -1], [1, 1], 100)

        # some assertion cases
        assertion_cases = []

        # iterate over the Emap classes
        for autoencoder_type in [
            "None",
            "EncoderMap",
            "AngleDihedralCartesianEncoderMap",
        ]:
            if autoencoder_type == "EncoderMap":
                train_data = traj.central_dihedrals
                autoencoder = em.EncoderMap(
                    train_data=train_data,
                    read_only=True,
                )
            elif autoencoder_type == "AngleDihedralCartesianEncoderMap":
                autoencoder = em.AngleDihedralCartesianEncoderMap(
                    trajs=trajs,
                    read_only=True,
                )
            else:
                autoencoder = None

            # iterate over the type of trajs
            for inp_trajs_type in ["ensemble", "single", "None"]:
                if inp_trajs_type == "ensemble":
                    inp_trajs = deepcopy(trajs[:2])
                elif inp_trajs_type == "single":
                    inp_trajs = deepcopy(trajs[0])
                else:
                    inp_trajs = None

                # iterate over lowd
                for lowd_type in ["None", "np", "in_trajs"]:
                    if lowd_type == "np":
                        lowd = np.random.random((inp_trajs.n_frames, 2))
                    elif lowd_type == "in_trajs":
                        inp_trajs = deepcopy(inp_trajs)
                        if isinstance(inp_trajs, em.TrajEnsemble):
                            inp_trajs.load_CVs(
                                np.random.random((inp_trajs.n_frames, 2)),
                                attr_name="lowd",
                            )
                        else:
                            inp_trajs.load_CV(
                                np.random.random((inp_trajs.n_frames, 2)),
                                attr_name="lowd",
                            )
                    else:
                        lowd = None

                    # iterate over highd
                    for highd_type in ["None", "np", "in_trajs", "in_autoencder"]:
                        if highd_type == "np":
                            highd = np.random.random(
                                (500, traj.central_dihedrals.shape[-1])
                            )
                        elif highd_type == "in_trajs":
                            if isinstance(inp_trajs, em.TrajEnsemble):
                                inp_trajs.load_CVs(
                                    np.random.random(
                                        (
                                            inp_trajs.n_frames,
                                            traj.central_dihedrals.shape[-1],
                                        )
                                    ),
                                    attr_name="highd",
                                )
                            else:
                                inp_trajs.load_CV(
                                    np.random.random(
                                        (
                                            inp_trajs.n_frames,
                                            traj.central_dihedrals.shape[-1],
                                        )
                                    ),
                                    attr_name="highd",
                                )

                        elif highd_type == "in_autoencder":
                            highd = None
                        else:
                            highd = None

                        case = (autoencoder_type, inp_trajs_type, lowd_type, highd_type)
                        print(f"Testing case: {case=}")

                        if case in assertion_cases or (
                            autoencoder_type == "None"
                            and (lowd_type == "None" or highd_type == "None")
                        ):
                            with self.assertRaises(AssertionError):
                                # instantiate the sess
                                sess = em.InteractivePlotting(
                                    trajs=trajs,
                                    autoencoder=autoencoder,
                                    lowd_data=lowd,
                                    highd_data=highd,
                                )

                        else:
                            # instantiate the sess
                            sess = em.InteractivePlotting(
                                trajs=trajs,
                                autoencoder=autoencoder,
                                lowd_data=lowd,
                                highd_data=highd,
                            )

                            # a random cluster_selection
                            # click a point
                            points = SimpleNamespace(**{"point_inds": 0})
                            sess.scatter_on_click(None, points, None)

                            # select a cluster
                            points = SimpleNamespace(
                                **{
                                    "point_inds": np.arange(20),
                                }
                            )
                            selector = SimpleNamespace(
                                **{
                                    "xs": np.arange(10),
                                    "ys": np.arange(10),
                                },
                            )
                            sess.on_select(None, points, selector)
                            sess.cluster(None)
                            sess.write_cluster()
                            sess.generate(path)
                            sess.write_path()

        self.assertTrue(False, "Check scatter_kws.")

    # @unittest.skip("Devel")
    # def test_interactive_plotting(self):
    #     traj1 = SingleTraj(
    #         Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc",
    #         top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
    #     )
    #     traj2 = SingleTraj(
    #         Path(__file__).resolve().parent / "data/1am7_corrected_part2.xtc",
    #         top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
    #     )
    #     trajs = TrajEnsemble([traj1, traj2])
    #     trajs.load_CVs("all")
    #     os.makedirs("tmp.json", exist_ok=True)
    #     p = ADCParameters(
    #         use_sidechains=True, use_backbone_angles=True, main_path="tmp.json"
    #     )
    #     e_map = AngleDihedralCartesianEncoderMap(trajs, p, read_only=False)
    #
    #     lowd = e_map.encode()
    #     trajs.load_CVs(lowd, attr_name="lowd")
    #     self.assertEqual(trajs.lowd.shape[-1], 2)
    #
    #     # invent a clustering
    #     cluster_membership = np.full(trajs.n_frames, -1)
    #     cluster_membership[::2] = 0
    #     trajs.load_CVs(cluster_membership, "user_selected_points")
    #     # get a dummy traj
    #     _, dummy_traj = get_cluster_frames(
    #         trajs,
    #         0,
    #         nglview=True,
    #         shorten=True,
    #         stack_atoms=True,
    #         col="user_selected_points",
    #         align_string="name CA",
    #     )
    #     self.assertIsInstance(dummy_traj, list)
    #     self.assertEqual(len(dummy_traj), 10)
    #
    #     class Selector:
    #         fig, ax = plt.subplots()
    #         lasso = PolygonSelector(ax=ax, onselect=None)
    #         lasso._xs = np.random.random(5)
    #         lasso._ys = np.random.random(5)
    #         # repeat first and last point to better replicate PolygonSelector behavior
    #         lasso._xs = np.stack([*lasso._xs, lasso._xs[0]])
    #         lasso._ys = np.stack([*lasso._ys, lasso._ys[0]])
    #
    #     selector = Selector()
    #     self.assertEqual(selector.lasso._xs[0], selector.lasso._xs[-1])
    #     self.assertIsInstance(selector.lasso, PolygonSelector)
    #     cluster_num, main_path = _unpack_cluster_info(
    #         trajs, e_map.p.main_path, selector, dummy_traj, "name CA"
    #     )
    #
    #     new_trajs = TrajEnsemble.from_textfile(
    #         os.path.join(
    #             main_path, "cluster_id_0_all_plotted_trajs_in_correct_order.txt"
    #         )
    #     )
    #     self.assertEqual(new_trajs.n_trajs, trajs.n_trajs)
    #     df = pd.read_csv(os.path.join(main_path, "cluster_id_0_selected_points.csv"))
    #     self.assertEqual(df.shape[0], np.ceil(len(cluster_membership) / 2))
    #     combined_n_frames = 0
    #     for pdb_file, xtc_file in zip(
    #         glob.glob(main_path + "/*traj*pdb"), glob.glob(main_path + "/*traj*xtc")
    #     ):
    #         combined_n_frames += md.load(xtc_file, top=pdb_file).n_frames
    #     self.assertEqual(combined_n_frames, df.shape[0])
    #
    #     shutil.rmtree("tmp.json/")


################################################################################
# Collect Test Cases and Filter
################################################################################


def load_tests(loader, tests, pattern):
    # Remove Phantom Tests from tensorflow skipped test_session
    # https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
    test_cases = (TestInteractivePlotting,)
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite


################################################################################
# Main
################################################################################


if __name__ == "__main__":
    unittest.main()

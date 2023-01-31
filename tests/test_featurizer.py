# -*- coding: utf-8 -*-
# tests/test_featurizer.py
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
# traj1 = SingleTraj("data/1am7_corrected.xtc", "data/1am7_protein.pdb")
# traj1.load_CV(traj1.xyz[:,:,0], 'z_coordinate')
#
# for i, frame in enumerate(traj1):
#     print(frame)
#     print(frame.z_coordinate)
#     if i == 3:
#         break
#
# This porduces wrong output in jupyter
"""Available TestSuites:
    * TestFeatures: Uses mock to mock atomic positions. The calculation of the
        features can thus be tested.
    * TestDaskFeatures: Test the performance of the distributed featurization.
    * TestDaskFeatureAndFeaturizerReturnDataSets: Tests that both features return
        similar shaped data.

"""

################################################################################
# Imports
################################################################################


import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import MDAnalysis as mda
import mdtraj as md
import numpy as np
from pyemma.coordinates import load, source

import encodermap as em
from encodermap import SingleTraj

warnings.filterwarnings("ignore", category=DeprecationWarning)


################################################################################
# Globals
################################################################################


################################################################################
# Mocks
################################################################################


def add_B1_and_B2_as_ca(*args, **kwargs):
    self = args[0]
    sel = self.topology.select("name B1 or name B2")
    pairs = self.pairs(sel, 0)
    self.add_distances(pairs, periodic=True)


def add_B1_and_B4_as_ca(*args, **kwargs):
    self = args[0]
    sel = self.topology.select("name B1 or name B4")
    pairs = self.pairs(sel, 0)
    self.add_distances(pairs, periodic=True)


################################################################################
# Utils
################################################################################


def format_msg(out1, out2):
    msg = (
        f"The two arrays `out1` and `out2` are created using "
        f"the PyEMMAFeaturizer class. For `out1`, a single trajectory "
        f"is provided, for `out2` a `TrajEnsemble` class with multiple "
        f"trajs is provided. These multiple trajs were obtained by "
        f"splitting the traj for `out1` into two. Thus, they should "
        f"have the same results. Here are the results:"
        f"Shapes: out1: {out1.shape}, out2: {out2.shape}."
        f"First five entries: out1: {out1[:5]}, out2: {out2[:5]}"
    )
    return msg


################################################################################
# Test suites
################################################################################


class TestDaskFeatureAndFeaturizerReturnDataSets(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.PFFP_xtc_file = (
            Path(__file__) / "../data/PFFP_MD_fin_protonly_dt_100.xtc"
        ).resolve()
        cls.PFFP_tpr_file = (
            Path(__file__) / "../data/PFFP_MD_fin_protonly.tpr"
        ).resolve()
        cls.PFFP_gro_file = (
            Path(__file__) / "../data/PFFP_MD_fin_protonly.gro"
        ).resolve()

    def test_in_memory_and_dask_featurizer_return_similar_data(self):
        import xarray as xr

        traj1 = em.load(self.PFFP_xtc_file, self.PFFP_gro_file)
        traj2 = em.load(self.PFFP_xtc_file, self.PFFP_gro_file)

        feat1 = em.Featurizer(traj1, in_memory=True)
        feat2 = em.Featurizer(traj2, in_memory=False)

        feat1.add_all()
        feat1.add_distances_ca()
        feat2.add_all()

        out1 = feat1.get_output()
        out2 = feat2.get_output()

        self.assertIsInstance(out1, xr.Dataset)
        self.assertIsInstance(out2, xr.Dataset)


class TestDaskFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.PFFP_xtc_file = (
            Path(__file__) / "../data/PFFP_MD_fin_protonly_dt_100.xtc"
        ).resolve()
        cls.PFFP_tpr_file = (
            Path(__file__) / "../data/PFFP_MD_fin_protonly.tpr"
        ).resolve()
        cls.PFFP_gro_file = (
            Path(__file__) / "../data/PFFP_MD_fin_protonly.gro"
        ).resolve()
        top = md.load(str(cls.PFFP_gro_file)).top

        # get all bonds of the 125 tetrapeptides
        cls.bonds = []
        backbone = top.select("backbone")
        for i in range(top.n_residues // 4):
            selection = top.select(f"resid {i} to {i + 4}")
            this_peptide_backbone = np.intersect1d(backbone, selection)
            this_peptide_bonds = np.vstack(
                [this_peptide_backbone[:-1], this_peptide_backbone[1:]]
            )
            cls.bonds.append(this_peptide_bonds)
        cls.bonds = np.hstack(cls.bonds).T

    @unittest.skip
    def test_and_compare_mdtraj_pbc(self):
        traj = md.load(self.PFFP_xtc_file, top=self.PFFP_gro_file)
        self.assertGreater(traj.n_frames, 1)

        # compute the distances with and without pbc
        dists_no_pbc = md.compute_distances(traj, self.bonds, periodic=False)
        dists_with_pbc = md.compute_distances(traj, self.bonds, periodic=True)

        # compare
        self.assertGreater(dists_no_pbc.max(), dists_with_pbc.max())

        print(dists_no_pbc.max(), dists_with_pbc.max())
        print(dists_no_pbc.mean(), dists_with_pbc.mean())
        print(dists_no_pbc.min(), dists_with_pbc.min())

    @unittest.skip
    def test_mdanalysis_fixes_pbc(self):
        from MDAnalysis.transformations import unwrap

        from encodermap.loading.delayed import analyze_block

        u1 = mda.Universe(str(self.PFFP_tpr_file), str(self.PFFP_xtc_file))
        frame_indices = np.arange(len(u1.trajectory))
        ag1 = u1.atoms

        u2 = mda.Universe(str(self.PFFP_tpr_file), str(self.PFFP_xtc_file))
        ag2 = u2.atoms
        transform = unwrap(ag2)
        u2.trajectory.add_transformations(transform)

        # use mdtraj for comparison
        traj = md.load(self.PFFP_xtc_file, top=self.PFFP_gro_file)
        mdtraj_dists_with_pbc = md.compute_distances(traj, self.bonds, periodic=True)

        dists_no_pbc = analyze_block(frame_indices, u1, ag1, self.bonds, unwrap=False)
        dists_with_pbc = analyze_block(frame_indices, u2, ag2, self.bonds, unwrap=False)

        print(dists_no_pbc.max(), dists_with_pbc.max(), mdtraj_dists_with_pbc.max())

    def test_performance_full_featurization(self):
        pass


class TestFeatures(unittest.TestCase):
    def setUp(self) -> None:
        traj_path = (Path(__file__) / "../../tests/data/known_angles.h5").resolve()
        self.traj = SingleTraj(traj_path)

        traj_paths = (Path(__file__) / "../../tests/data").resolve()
        traj_paths = list(traj_paths.glob("known_angles_*.h5"))[::-1]
        self.trajs = em.TrajEnsemble(traj_paths)

        md_traj_xtc = (
            Path(__file__) / "../../tests/data"
        ).resolve() / "alanine_dipeptide.xtc"
        md_traj_pdb = (
            Path(__file__) / "../../tests/data"
        ).resolve() / "alanine_dipeptide.pdb"
        self.md_traj = SingleTraj(md_traj_xtc, md_traj_pdb)

        traj_file = (Path(__file__) / "../../tests/data/1am7_corrected.xtc").resolve()
        top_file = (Path(__file__) / "../../tests/data/1am7_protein.pdb").resolve()
        self.protein_1am7 = SingleTraj(traj_file, top_file)

        traj_file = (Path(__file__) / "../../tests/data/Ala10_helix.xtc").resolve()
        top_file = (Path(__file__) / "../../tests/data/Ala10_helix.pdb").resolve()
        self.ala10_helix = SingleTraj(traj_file, top_file)

    def test_ala_dipeptide_correct_number_of_residues(self):
        self.assertEqual(2, self.md_traj.n_residues)

    def test_add_all(self):
        # create featurizers for the two Info classes
        from encodermap.loading.featurizer import PyEMMAFeaturizer

        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_all()
        feat2.add_all()
        self.assertIsInstance(feat1, PyEMMAFeaturizer)

        out1 = feat1.get_output()
        out2 = feat2.get_output()
        self.assertEqual(len(out2.coords["traj_num"]), 2)

        # check the positions
        self.assertTrue(np.alltrue(out1.SelectionFeature.values[0][:, :3] == 0))
        self.assertTrue(np.alltrue(out2.SelectionFeature.values[0][:, :3] == 0))
        self.assertTrue(np.alltrue(out2.SelectionFeature.values[1][:, :3] == 0))
        self.assertEqual(out1.SelectionFeature.values[0][5, 10], 1.5)

        # use the load_CV feature of TrajEnsemble and SingleTraj
        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        # test the CVs
        msg = (
            "Usually, the `add_all` method of the Featurizer class"
            "adds a `SelectionFeature` to the active features of the Featurizer. "
            "Thus, the key in `traj.CVs` should also be named 'SelectionFeature'."
        )
        self.assertIn("SelectionFeature", self.traj.CVs, msg=msg)
        # check for same contents as feat.get_output()
        self.assertTrue(np.alltrue(self.traj.CVs["SelectionFeature"][:, :3] == 0))

        msg = (
            "This error can be very serious. Normally, the `traj.CVs` attribute "
            "was meant to be built from the `traj._CVs` dataarray. If the values "
            "of these two arrays are not the same, something with the `CVs` property "
            "of `SingleTraj` is broken."
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.SelectionFeature.values[0],
                self.traj.CVs["SelectionFeature"],
            ),
            msg=msg,
        )

        # check the coordinates of this dataarray
        self.assertIsNone(self.traj._CVs.SelectionFeature.coords["traj_num"].values[0])
        self.assertEqual(
            self.traj._CVs.SelectionFeature.coords["traj_name"].values[0],
            "known_angles",
        )
        self.assertIn("SELECTIONFEATURE", self.traj._CVs.SelectionFeature.coords)
        self.assertIn(
            "ATOM", self.traj._CVs.SelectionFeature.coords["SELECTIONFEATURE"].values[0]
        )

        # check for some additional info in the attrs
        self.assertEqual(
            self.traj._CVs.SelectionFeature.attrs["full_path"], self.traj.traj_file
        )
        self.assertEqual(
            self.traj._CVs.SelectionFeature.attrs["topology_file"], self.traj.top_file
        )
        self.assertEqual(
            self.traj._CVs.SelectionFeature.attrs["feature_axis"], "SELECTIONFEATURE"
        )
        self.assertTrue(
            np.array_equal(self.traj._CVs.attrs["SelectionFeature"], np.arange(4))
        )

        # do the same things with the trajs
        self.assertTrue(
            np.array_equal(
                self.trajs.CVs["SelectionFeature"], self.traj.CVs["SelectionFeature"]
            )
        )
        self.assertEqual(self.trajs._CVs.dims["traj_num"], 2)
        self.assertEqual(self.trajs._CVs.dims["frame_num"], 3)
        self.assertEqual(self.trajs._CVs.dims["SELECTIONFEATURE"], 12)
        self.assertEqual(self.trajs._CVs.dims["traj_num"], 2)

    def test_add_selection(self):
        # create featurizers for the two Info classes
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_selection(self.traj.top.select("name B1"))
        feat2.add_selection(self.traj.top.select("name B1"))

        out1 = feat1.get_output()
        self.assertTrue(np.alltrue(out1.SelectionFeature.values[0] == 0))
        out2 = feat2.get_output()
        self.assertTrue(np.alltrue(out2.SelectionFeature.values[0] == 0))
        self.assertTrue(np.alltrue(out2.SelectionFeature.values[0] == 0))

    @patch(
        "pyemma.coordinates.data.featurization.featurizer.MDFeaturizer.add_distances_ca",
        add_B1_and_B2_as_ca,
    )
    def test_add_distances_ca(self):
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_distances_ca()
        feat2.add_distances_ca()

        out1 = feat1.get_output().DistanceFeature.values[
            0
        ]  # index with [0] to get 1st traj
        out2 = np.vstack(feat2.get_output().DistanceFeature.values)
        should_be = np.array([[1], [1], [2], [1], [1], [1]])
        self.assertTrue(np.array_equal(out1, should_be))
        self.assertEqual(out1.shape, should_be.shape)
        self.assertTrue(np.array_equal(out1, out2), msg=format_msg(out1, out2))

        # also load the CV
        self.traj.load_CV(feat1)

        # and check
        self.assertIn("DISTANCEFEATURE", self.traj._CVs.DistanceFeature.coords)
        self.assertEqual(self.traj._CVs.DistanceFeature.shape, (1, 6, 1))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.attrs["DistanceFeature"],
                feat1.feat.active_features[0].distance_indexes,
            )
        )

        # use the load_CV feature of TrajEnsemble and SingleTraj
        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertEqual(self.traj._CVs.DistanceFeature.shape, (1, 6, 1))
        self.assertEqual(self.trajs._CVs.DistanceFeature.shape, (2, 3, 1))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.DistanceFeature.values[0, :3],
                self.trajs._CVs.DistanceFeature.values[0],
            )
        )
        self.assertTrue(
            np.array_equal(self.traj._CVs.attrs["DistanceFeature"], np.array([[0, 1]]))
        )

    @patch(
        "pyemma.coordinates.data.featurization.featurizer.MDFeaturizer.add_distances_ca",
        add_B1_and_B4_as_ca,
    )
    def test_add_inverse_distances(self):
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_distances_ca()
        feat1.add_inverse_distances([0, 3])
        feat2.add_distances_ca()
        feat2.add_inverse_distances(([0, 3]))
        msg = f"{feat1.inp.featurizer.active_features}"
        self.assertEqual(len(feat1.inp.featurizer.active_features), 2, msg=msg)
        out1_distances = feat1.get_output().DistanceFeature.values[
            0
        ]  # use [0] to get 1st and only traj
        out1_inverse_distances = feat1.get_output().InverseDistanceFeature.values[0]
        out2_distances = np.vstack(feat2.get_output().DistanceFeature.values)
        out2_inverse_distances = np.vstack(
            feat2.get_output().InverseDistanceFeature.values
        )
        self.assertTrue(
            np.array_equal(out1_inverse_distances, out2_inverse_distances),
            msg=format_msg(out1_inverse_distances, out2_inverse_distances),
        )
        self.assertEqual(len(out1_inverse_distances), len(self.traj))
        self.assertTrue(np.array_equal(1 / out1_distances, out1_inverse_distances))

        # use the load_CV feature of TrajEnsemble and SingleTraj
        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn("InverseDistanceFeature", self.traj._CVs.attrs)
        self.assertEqual(self.traj._CVs.InverseDistanceFeature.shape, (1, 6, 1))
        self.assertEqual(self.trajs._CVs.InverseDistanceFeature.shape, (2, 3, 1))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.InverseDistanceFeature.values[0, :3],
                self.trajs._CVs.InverseDistanceFeature.values[0],
            )
        )
        self.assertTrue(
            np.array_equal(self.traj._CVs.attrs["DistanceFeature"], np.array([[0, 3]]))
        )

    def test_add_contacts(self):
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_distances([0, 1, 3])
        feat1.add_contacts([0, 1, 3], threshold=1.2)
        feat2.add_distances([0, 1, 3])
        feat2.add_contacts([0, 1, 3], threshold=1.2)

        print(feat1.get_output().data_vars)

        out1_distances = feat1.get_output().DistanceFeature.values[0]
        out1_contacts = feat1.get_output().ContactFeature.values[0]
        out2_distances = np.vstack(feat2.get_output().DistanceFeature.values)
        out2_contacts = np.vstack(feat2.get_output().ContactFeature.values)
        self.assertTrue(
            np.array_equal(
                (out1_distances < 1.2).astype(int), out1_contacts.astype(int)
            )
        )
        self.assertTrue(
            np.array_equal(out1_contacts, out2_contacts),
            msg=format_msg(out1_contacts, out2_contacts),
        )

        # use the load_CV feature of TrajEnsemble and SingleTraj
        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)
        self.assertIn("ContactFeature", self.traj._CVs.attrs)
        self.assertEqual(self.traj._CVs.ContactFeature.shape, (1, 6, 3))
        self.assertEqual(self.trajs._CVs.ContactFeature.shape, (2, 3, 3))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.ContactFeature.values[0, :3],
                self.trajs._CVs.ContactFeature.values[0],
            )
        )

    def test_add_residue_mindist(self):
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        pairs = np.array([[0, 1], [0, 2], [0, 3]])
        feat1.add_residue_mindist(pairs)
        feat2.add_residue_mindist(pairs)
        feat1.add_distances([0, 1])
        feat2.add_distances([0, 1])

        out1 = feat1.get_output().ResidueMinDistanceFeature.values[0]
        out2 = np.vstack(feat2.get_output().ResidueMinDistanceFeature.values)

        # self.assertTrue(np.array_equal(out1[:, 0], out1[:, -1]))
        self.assertTrue(np.array_equal(out1, out2), msg=format_msg(out1, out2))

        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn("ResidueMinDistanceFeature", self.traj._CVs.attrs)
        self.assertEqual(self.traj._CVs.ResidueMinDistanceFeature.shape, (1, 6, 3))
        self.assertEqual(self.trajs._CVs.ResidueMinDistanceFeature.shape, (2, 3, 3))
        self.assertTrue(
            np.array_equal(self.traj._CVs.attrs["ResidueMinDistanceFeature"], pairs)
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.ResidueMinDistanceFeature.values[0, :3],
                self.trajs._CVs.ResidueMinDistanceFeature.values[0],
            )
        )

    def test_add_group_COM(self):
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)
        feat1.add_group_COM(
            [[0, 1], [0, 1, 2, 3]], image_molecules=False, mass_weighted=False
        )
        feat2.add_group_COM(
            [[0, 1], [0, 1, 2, 3]], image_molecules=False, mass_weighted=False
        )

        out1 = feat1.get_output().GroupCOMFeature.values[0]
        out2 = np.vstack(feat2.get_output().GroupCOMFeature.values)

        # atom 0 is always 0, 0, 0
        for i, row in enumerate(out1[:, :3]):
            if i == 2:
                should_be = np.array([1, 0, 0])
            else:
                should_be = np.array([0.5, 0, 0])
                # only for frame 2, atom 1 is 2, 0, 0
                # otherwise it is alsways 1, 0, 0
                continue
            msg = (
                f"Because, the positions of atoms 0 and 1 are [0, 0, 0] and "
                f"[1, 0, 0] for frames [0, 1, 3, 4, 5], the COM should be the"
                f"[0.5, 0, 0] for these frames. And [1, 0, 0] for frame 2. "
                f"In tis case, frame {i} gives a wrong COM: {row}"
            )
            self.assertTrue(np.array_equal(row, should_be), msg=msg)

        self.assertTrue(np.array_equal(out1[0, 3:], np.array([1, 0.5, 0])))
        self.assertTrue(np.array_equal(out1, out2), msg=format_msg(out1, out2))

        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn("GroupCOMFeature", self.traj._CVs.attrs)
        self.assertEqual(self.traj._CVs.GroupCOMFeature.shape, (1, 6, 6))
        self.assertEqual(self.trajs._CVs.GroupCOMFeature.shape, (2, 3, 6))
        self.assertTrue(
            np.array_equal(self.traj._CVs.attrs["GroupCOMFeature"][0], np.array([0, 1]))
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.attrs["GroupCOMFeature"][1], np.array([0, 1, 2, 3])
            )
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.GroupCOMFeature.values[0, :3],
                self.trajs._CVs.GroupCOMFeature.values[0],
            )
        )

    def test_add_residue_COM(self):
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_residue_COM([0, 1, 2, 3], image_molecules=False, mass_weighted=False)
        feat1.add_all()
        feat2.add_residue_COM([0, 1, 2, 3], image_molecules=False, mass_weighted=False)
        feat2.add_all()

        out1_com = feat1.get_output().ResidueCOMFeature.values[0]
        out1_sel = feat1.get_output().SelectionFeature.values[0]
        out2_com = np.vstack(feat2.get_output().ResidueCOMFeature.values)
        out2_sel = np.vstack(feat2.get_output().SelectionFeature.values)

        self.assertTrue(np.array_equal(out1_com, out1_sel))
        self.assertTrue(
            np.array_equal(out1_com, out2_com), msg=format_msg(out1_com, out2_com)
        )

        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn("ResidueCOMFeature", self.traj._CVs.attrs)
        self.assertEqual(self.traj._CVs.ResidueCOMFeature.shape, (1, 6, 12))
        self.assertEqual(self.trajs._CVs.ResidueCOMFeature.shape, (2, 3, 12))
        self.assertTrue(
            np.array_equal(self.traj._CVs.attrs["ResidueCOMFeature"][0], np.array([0]))
        )
        self.assertTrue(
            np.array_equal(self.traj._CVs.attrs["ResidueCOMFeature"][1], np.array([1]))
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.ResidueCOMFeature.values[0, :3],
                self.trajs._CVs.ResidueCOMFeature.values[0],
            )
        )

    def test_add_angles(self):
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_angles([[0, 1, 2], [1, 2, 3]], deg=True)
        feat2.add_angles([[0, 1, 2], [1, 2, 3]], deg=True)

        out1 = feat1.get_output().AngleFeature.values[0]
        out2 = np.vstack(feat2.get_output().AngleFeature.values)
        ninety_deg_angles = np.array(
            [
                [True, True],
                [True, True],
                [True, False],
                [True, True],
                [True, False],
                [True, False],
            ]
        ).astype(bool)
        self.assertTrue(np.all(out1[:, 0] == 90))
        self.assertTrue(np.array_equal(ninety_deg_angles, out1 == 90))
        self.assertTrue(np.array_equal(out1, out2), msg=format_msg(out1, out2))

        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn("AngleFeature", self.traj._CVs.attrs)
        self.assertEqual(self.traj._CVs.AngleFeature.shape, (1, 6, 2))
        self.assertEqual(self.trajs._CVs.AngleFeature.shape, (2, 3, 2))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.AngleFeature.values[0, :3],
                self.trajs._CVs.AngleFeature.values[0],
            )
        )

        self.assertTrue(
            np.array_equal(
                self.traj._CVs.attrs["AngleFeature"], np.array([[0, 1, 2], [1, 2, 3]])
            )
        )

    def test_add_dihedrals(self):
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_dihedrals([[0, 1, 2, 3]], deg=True)
        feat2.add_dihedrals([[0, 1, 2, 3]], deg=True)

        out1 = feat1.get_output().DihedralFeature.values[0]
        out2 = np.vstack(feat2.get_output().DihedralFeature.values)

        self.assertEqual(out1[0], 180)
        self.assertEqual(out1[1], 0)
        self.assertEqual(np.round(out1[2], 0), 153)
        self.assertEqual(out1[3], 180)
        self.assertEqual(np.round(out1[4], 0), 117)
        self.assertEqual(out1[5], 90)

        self.assertTrue(np.array_equal(out1, out2), msg=format_msg(out1, out2))

        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn("DihedralFeature", self.traj._CVs.attrs)
        self.assertEqual(self.traj._CVs.DihedralFeature.shape, (1, 6, 1))
        self.assertEqual(self.trajs._CVs.DihedralFeature.shape, (2, 3, 1))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.DihedralFeature.values[0, :3],
                self.trajs._CVs.DihedralFeature.values[0],
            )
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.attrs["DihedralFeature"], np.array([[0, 1, 2, 3]])
            )
        )

    def test_add_backbone_torsions(self):
        feat1 = em.Featurizer(self.md_traj)
        feat1.add_backbone_torsions(deg=True)

        out1 = feat1.get_output().BackboneTorsionFeature.values[0]
        # alanine dipeptide should have 1 psi, 1 phi (and 1 omega) torsions
        self.assertEqual(len(feat1.describe()), 2)
        self.assertEqual(out1.shape[1], 2)

        feat2 = em.Featurizer(self.protein_1am7)
        feat2.add_backbone_torsions()
        print(self.protein_1am7.top.n_residues)
        self.assertEqual(
            len(feat2.describe()), 2 * self.protein_1am7.top.n_residues - 2
        )

        feat3 = em.Featurizer(self.ala10_helix)
        feat3.add_backbone_torsions()
        self.assertEqual(len(feat3.describe()), 18)

    def test_add_chi1_torsions(self):
        feat1 = em.Featurizer(self.protein_1am7)
        feat1.add_chi1_torsions(deg=True)

        out1 = feat1.get_output().SideChainTorsions.values[0]

        self.assertEqual(len(out1), len(self.protein_1am7))
        self.assertEqual(len(feat1.describe()), out1.shape[1])

        feat2 = em.Featurizer(self.md_traj)
        with self.assertRaises(ValueError):
            feat2.add_chi1_torsions()

    @unittest.skip
    def test_add_sidechain_torsions(self):
        self.assertTrue(False)

    def test_add_minrmsd_to_ref(self):
        feat1 = em.Featurizer(self.traj)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_minrmsd_to_ref(self.traj.traj, 0)
        feat2.add_minrmsd_to_ref(self.traj.traj, 0)

        out1 = feat1.get_output().MinRmsdFeature.values[0]
        out2 = np.vstack(feat2.get_output().MinRmsdFeature.values)

        self.assertEqual(out1[0, 0], 0)
        self.assertGreater(out1[1, 0], out1[0, 0])
        self.assertEqual(len(out1), len(self.traj))
        self.assertLess(out1[2, 0], out1[1, 0])

        self.assertTrue(np.array_equal(out1, out2), msg=format_msg(out1, out2))

        # use the load_CV feature of TrajEnsemble and SingleTraj
        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn("MinRmsdFeature", self.traj._CVs.attrs)
        self.assertEqual(self.traj._CVs.MinRmsdFeature.shape, (1, 6, 1))
        self.assertEqual(self.trajs._CVs.MinRmsdFeature.shape, (2, 3, 1))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.MinRmsdFeature.values[0, :3],
                self.trajs._CVs.MinRmsdFeature.values[0],
            )
        )
        self.assertIsNone(self.traj._CVs.attrs["MinRmsdFeature"])

    @unittest.skip
    def test_add_custom_feature(self):
        self.assertFalse(True)

    def test_encodermap_features_cartesians(self):
        feat1 = em.Featurizer(self.trajs)
        feat2 = em.Featurizer(self.trajs)

        feat1.add_list_of_feats(["all_cartesians"])
        feat2.add_all()

        out1 = np.vstack(feat1.get_output().all_cartesians.values)
        out2 = np.vstack(feat2.get_output().SelectionFeature.values).reshape((6, 4, 3))

        self.assertTrue(len(feat1.describe()), self.traj.n_atoms * 3)
        self.assertTrue(np.array_equal(out1, out2))

        self.traj.load_CV(["all_cartesians"])
        self.trajs.load_CVs(["all_cartesians"])

        # feat1 uses the wrong trajs and can't be used on traj.
        with self.assertRaises(Exception):
            self.traj.load_CV(feat1)

        # build a correct feat1
        feat1 = em.Featurizer(self.traj)
        feat1.add_list_of_feats(["all_cartesians"])

        # at this point both should only have one active feature
        self.assertEqual(len(feat1), 1)
        self.assertEqual(len(feat2), 1)

        # adding the same feature to feat2 should trigger a warning
        with self.assertLogs(feat2.feat.logger, "WARNING"):
            feat2.add_all()
        with self.assertLogs(feat1.feat.logger, "WARNING"):
            feat1.add_all()

        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertEqual(len(self.traj.CVs), 1)

        # some checks for the 3D atomic coordinates
        self.assertIn("all_cartesians", self.traj.CVs)
        self.assertIn("all_cartesians", self.trajs._CVs)
        # check for same contents as feat.get_output()
        print(self.traj.CVs["all_cartesians"].shape)
        print(self.traj.CVs["all_cartesians"][:, :3])
        self.assertTrue(np.alltrue(self.traj.CVs["all_cartesians"][:, 0] == 0))
        self.assertTrue(np.alltrue(self.trajs._CVs.all_cartesians.values[:, :, 0] == 0))

        msg = (
            "This error can be very serious. Normally, the `traj.CVs` attribute "
            "was meant to be built from the `traj._CVs` dataarray. If the values "
            "of these two arrays are not the same, something with the `CVs` property "
            "of `SingleTraj` is broken."
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.all_cartesians.values[0], self.traj.CVs["all_cartesians"]
            ),
            msg=msg,
        )

        # check the coordinates of this dataarray
        self.assertIsNone(self.traj._CVs.all_cartesians.coords["traj_num"].values[0])
        self.assertEqual(
            self.traj._CVs.all_cartesians.coords["traj_name"].values[0], "known_angles"
        )
        self.assertIn("ATOM", self.traj._CVs.all_cartesians.coords)
        self.assertIn("COORDS", self.traj._CVs.all_cartesians.coords)

        # check for some additional info in the attrs
        self.assertEqual(
            str(self.traj._CVs.all_cartesians.attrs["full_path"][0]),
            str(self.traj.traj_file),
            msg=f"Files {self.traj._CVs.all_cartesians.attrs['full_path']} and {self.traj.traj_file} "
            f"do not match.",
        )
        self.assertEqual(
            str(self.traj._CVs.all_cartesians.attrs["topology_file"][0]),
            str(self.traj.top_file),
            msg=f"Files {self.traj._CVs.all_cartesians.attrs['topology_file']} and {self.traj.top_file} "
            f"do not match.",
        )
        self.assertEqual(self.traj._CVs.all_cartesians.attrs["feature_axis"][0], "ATOM")
        self.assertTrue(
            np.array_equal(self.traj._CVs.attrs["all_cartesians"], np.arange(4))
        )

        # do the same things with the trajs
        self.assertTrue(
            np.array_equal(
                self.trajs.CVs["all_cartesians"], self.traj.CVs["all_cartesians"]
            )
        )
        self.assertEqual(self.trajs._CVs.dims["traj_num"], 2)
        self.assertEqual(self.trajs._CVs.dims["frame_num"], 3)
        self.assertEqual(self.trajs._CVs.dims["ATOM"], 4)
        self.assertEqual(self.trajs._CVs.dims["COORDS"], 3)
        self.assertEqual(self.trajs._CVs.dims["traj_num"], 2)

    def test_encodermap_features_ala10(self):
        self.ala10_helix.load_CV("all_cartesians")
        self.assertTrue(
            np.array_equal(self.ala10_helix.xyz, self.ala10_helix.CVs["all_cartesians"])
        )
        self.assertTrue(
            np.array_equal(
                self.ala10_helix.xyz, self.ala10_helix._CVs.all_cartesians.values[0]
            )
        )
        self.assertIn("all_cartesians", self.ala10_helix.CVs)

        # alanine should have 5 bond lengths for united atoms
        self.ala10_helix.load_CV("all_distances")
        self.assertEqual(self.ala10_helix.CVs["all_distances"].shape, (1, 50))
        self.assertTrue(np.all(self.ala10_helix.CVs["all_distances"] < 1))
        self.assertIn("all_distances", self.ala10_helix.CVs)

        # 10 alanines should have 10 * 3 - 1 = 29 central distances
        self.ala10_helix.load_CV("central_distances")
        self.assertEqual(self.ala10_helix.CVs["central_distances"].shape, (1, 29))
        self.assertIn("central_distances", self.ala10_helix.CVs)

    def test_encodermap_features_1am7(self):
        """Test these features:

        * AllCartesians
        * AllBondDistances
        * CentralCartesians
        * CentralBondDistances
        * CentralAngles
        * CentralDihedrals
        * SideChainCartesians
        * SideChainBondDistances
        * SideChainAngles
        * SideChainDihedrals

        """
        self.protein_1am7.load_CV("all_cartesians")
        self.assertTrue(
            np.array_equal(
                self.protein_1am7.xyz, self.protein_1am7.CVs["all_cartesians"]
            )
        )
        self.assertTrue(
            np.array_equal(
                self.protein_1am7.xyz, self.protein_1am7._CVs.all_cartesians.values[0]
            )
        )
        self.assertIn("all_cartesians", self.protein_1am7.CVs)

        # alanine should have 5 bond lengths for united atoms
        self.protein_1am7.load_CV("all_distances")
        n_bonds = len(list(b for b in self.protein_1am7.top.bonds))
        self.assertEqual((51, n_bonds), self.protein_1am7.CVs["all_distances"].shape)
        self.assertTrue(np.all(self.protein_1am7.CVs["all_distances"] < 1))
        self.assertIn("all_distances", self.protein_1am7.CVs)

        # 154 alanines should have 154 * 3 - 1 = 473 central distances
        self.protein_1am7.load_CV("central_distances")
        self.assertEqual(self.protein_1am7.CVs["central_distances"].shape, (51, 473))
        self.assertIn("central_distances", self.protein_1am7.CVs)

        # thus the side distances should be
        no_of_h_bonds = 0
        for b in self.protein_1am7.top.bonds:
            if any([a.element.symbol == "H" for a in b]):
                no_of_h_bonds += 1
        self.protein_1am7.load_CV("side_distances")

        no_of_bonds = 0
        which = ["chi1", "chi2", "chi3", "chi4", "chi5"]
        from mdtraj.geometry import dihedral

        indices_dict = {
            k: getattr(dihedral, "indices_%s" % k)(self.protein_1am7.top) for k in which
        }

        no_of_sidechain_bonds = (
            2 * len(indices_dict["chi1"])
            + len(indices_dict["chi2"])
            + len(indices_dict["chi3"])
            + len(indices_dict["chi4"])
            + len(indices_dict["chi5"])
        )
        print(no_of_sidechain_bonds)
        print(self.protein_1am7.CVs["side_distances"].shape)

        self.assertEqual(
            self.protein_1am7.CVs["side_distances"].shape, (51, no_of_sidechain_bonds)
        )
        self.assertIn("side_distances", self.protein_1am7.CVs)

        # check out the dihedrals
        self.protein_1am7.load_CV(["central_dihedrals", "side_dihedrals"])

        with self.assertRaises(ValueError):
            self.md_traj.load_CV(["central_dihedrals", "side_dihedrals"])


test_cases = (
    TestFeatures,
    TestDaskFeatures,
    TestDaskFeatureAndFeaturizerReturnDataSets,
)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite


# class TestFeatures(unittest.TestCase):
#     """Mocks a """
#     def test_central_dihedrals(self):
#         from encodermap.loading.features import CentralDihedrals
#         from encodermap import Featurizer
#         feat = CentralDihedrals(traj_with_mocks.top, periodic=True)
#         self.assertEqual(feat.name, 'CentralDihedrals')
#         self.assertEqual(feat.describe(), ['CENTERDIH PSI   RESID  ALA:   1 CHAIN 0',
#                                            'CENTERDIH OMEGA RESID  ALA:   1 CHAIN 0',
#                                            'CENTERDIH PHI   RESID  ALA:   2 CHAIN 0'])
#         feat.topologyfile = ALANINE_DIPEPTIDE_PDB_FILE
#         feat.topology = feat.top
#         print(load(ALANINE_DIPEPTIDE_XTC_FILE, feat, top=ALANINE_DIPEPTIDE_PDB_FILE))

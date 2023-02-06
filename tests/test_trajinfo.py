# -*- coding: utf-8 -*-
# tests/test_trajinfo.py
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
"""Main tests for the `TrajEnsemble` and `SingleTraj` classes. Following suites
are available:
    * TestTraj: Tests all aspects of the classes

"""

################################################################################
# Print installed packages for debugging
################################################################################


import pkg_resources

installed_packages = pkg_resources.working_set
installed_packages_list = sorted(
    ["%s==%s" % (i.key, i.version) for i in installed_packages]
)
if not any(["encodermap" in pkg for pkg in installed_packages_list]):
    raise Exception(
        "Encodermap is not installed. The unittests are meant to "
        "run on an environment, that has encodermap installed. "
        "That way, we can also verify the installation."
    )


################################################################################
# Imports
################################################################################


import unittest
from pathlib import Path

import mdtraj as md
import numpy as np
import xarray as xr

from encodermap.loading.features import CentralDihedrals
from encodermap.loading.featurizer import Featurizer
from encodermap.trajinfo import SingleTraj, TrajEnsemble
from encodermap.trajinfo.trajinfo_utils import np_to_xr

################################################################################
# Classes
################################################################################


class TestTraj(unittest.TestCase):
    def assertHasAttr(self, obj, intendedAttr):
        testBool = hasattr(obj, intendedAttr)
        self.assertTrue(
            testBool, msg=f"obj lacking an attribute. {obj=}, {intendedAttr=}"
        )

    def test_load_url(self):
        traj = SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
        # here it is not loaded
        self.assertFalse(traj.trajectory)
        self.assertFalse(traj.topology)
        self.assertEqual(traj.backend, "no_load")
        self.assertEqual(traj.n_frames, 14)
        self.assertEqual(traj.n_atoms, 1132)
        self.assertEqual(traj.n_residues, 75)
        self.assertEqual(traj.n_chains, 1)
        self.assertEqual(traj.backend, "mdtraj")
        self.assertEqual(traj.basename, "1GHC")
        self.assertEqual(traj.extension, ".pdb")
        self.assertEqual(traj.index, (None,))
        self.assertIsNone(traj.traj_num)
        self.assertEqual("https://files.rcsb.org/view/1GHC.pdb", traj.traj_file)
        self.assertEqual("https://files.rcsb.org/view/1GHC.pdb", traj.top_file)

    def test_SingleTraj_equality(self):
        """Test whether two instances with the same data are equal."""
        traj1 = SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
        traj2 = SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
        self.assertEqual(traj1, traj2)

    def test_SingleTraj_raises_error_on_wrong_dtype_for_traj(self):
        with self.assertRaises(ValueError):
            SingleTraj(traj=1)

    def test_singletraj_raises_error_on_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            t = SingleTraj("/tmp/imaginary_pdb_file.pdb")
            t.load_traj()

        with self.assertRaises(FileNotFoundError):
            t = SingleTraj("/tmp/imaginary_xtc_file.xtc", "/tmp/imaginary_pdb_file.pdb")
            t.load_traj()

        with self.assertRaises(FileNotFoundError):
            t = SingleTraj(
                Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
                "/tmp/imaginary_pdb_file.pdb",
            )
            t.load_traj()

    def test_load_h5_with_integer_index(self):
        t = SingleTraj(Path(__file__).resolve().parent / "data/traj.h5", index=2)
        self.assertEqual(t.n_frames, 1)

    def test_load_pdb_with_mdtraj_backend(self):
        traj = SingleTraj("https://files.rcsb.org/view/1GHC.pdb", backend="mdtraj")
        self.assertEqual(traj.basename, "1GHC")

    def test_n_frames_in_h5_file(self):
        traj = SingleTraj(Path(__file__).resolve().parent / "data/asp7.h5")
        self.assertEqual(traj._n_frames_base_h5_file, 5)

    def test_CVs_in_file(self):
        traj = SingleTraj(Path(__file__).resolve().parent / "data/asp7.h5")
        self.assertTrue(traj.CVs_in_file)

    def test_gen_ensemble_no_files(self):
        traj = md.load_pdb("https://files.rcsb.org/view/1GHC.pdb")
        traj = SingleTraj(traj)
        self.assertEqual(traj.traj_file, ".")
        trajs = traj._gen_ensemble()
        self.assertEqual(trajs.n_frames, traj.n_frames)

    def test_traj_ensemble_equality(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part2.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        trajs1 = TrajEnsemble([traj1, traj2])
        trajs2 = TrajEnsemble([traj1, traj2])
        self.assertEqual(trajs1, trajs2)

    def test_load_uri_with_index(self):
        traj = SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
        self.assertEqual(traj.index, (None,))
        self.assertEqual(traj.n_frames, 14)
        traj = SingleTraj(
            "https://files.rcsb.org/view/1GHC.pdb", index=slice(None, None, 2)
        )
        self.assertEqual(traj.index, (slice(None, None, 2),))
        self.assertEqual(traj.n_frames, 7)
        traj = traj[::2]
        self.assertEqual(traj.index, (slice(None, None, 2), slice(None, None, 2)))
        self.assertEqual(traj.n_frames, 4)

    def test_load_singletraj_with_traj_and_top(self):
        traj = md.load_pdb("https://files.rcsb.org/view/1GHC.pdb")
        traj = SingleTraj(traj, traj.top)
        self.assertEqual(traj.n_frames, 14)

    def test_load_info_all_with_trajs_and_one_top_does_not_raise_error(self):
        trajs = ["tests/data/1YUF.pdb", "tests/data/1YUG.pdb"]
        trajs = TrajEnsemble(trajs=trajs, tops=trajs)

        trajs = [
            "tests/data/1am7_corrected_part1.xtc",
            "tests/data/1am7_corrected_part2.xtc",
        ]
        top = "tests/data/1am7_protein.pdb"
        trajs = TrajEnsemble(trajs=trajs, tops=top)

    def test_single_traj_double_index_with_int(self):
        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        with self.assertRaises(IndexError):
            traj = traj[5][10]

    def test_load_xtc(self):
        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        xyz = np.load(
            Path(__file__).resolve().parent / "data/1am7_first_frame_first_atom_xyz.npy"
        )
        # should be false, because .xyz has not been calles
        self.assertFalse(traj.trajectory)
        self.assertFalse(traj.topology)
        self.assertEqual(traj.backend, "no_load")
        # data loaded
        self.assertTrue(np.array_equal(xyz, traj.xyz[0, 0]))
        self.assertEqual(traj.n_frames, 51)
        self.assertEqual(traj.n_atoms, 2504)
        self.assertEqual(traj.n_residues, 158)
        self.assertEqual(traj.n_chains, 1)
        self.assertEqual(
            traj.traj_file,
            str(Path(__file__).resolve().parent / "data/1am7_corrected.xtc"),
        )
        self.assertEqual(
            traj.top_file,
            str(Path(__file__).resolve().parent / "data/1am7_protein.pdb"),
        )
        self.assertEqual(
            traj._traj_file, Path(__file__).resolve().parent / "data/1am7_corrected.xtc"
        )
        self.assertEqual(
            traj._top_file, Path(__file__).resolve().parent / "data/1am7_protein.pdb"
        )
        self.assertEqual(traj.extension, ".xtc")
        self.assertEqual(traj.basename, "1am7_corrected")
        self.assertEqual(traj.index, (None,))
        self.assertIsNone(traj.traj_num)
        self.assertEqual(traj.backend, "mdtraj")
        # mdtraj equality checks
        self.assertEqual(
            traj.trajectory,
            md.load(
                str(Path(__file__).resolve().parent / "data/1am7_corrected.xtc"),
                top=str(Path(__file__).resolve().parent / "data/1am7_protein.pdb"),
            ),
        )
        self.assertEqual(
            traj.topology,
            md.load_topology(
                str(Path(__file__).resolve().parent / "data/1am7_protein.pdb")
            ),
        )

        with self.assertRaises(TypeError):
            frame = traj["wow"]

    def test_load_CVs_for_single_traj_as_string(self):
        """The `SingleTraj` class offers the possibility to load CVs from a str
        that can be either 'all' to load the standard CVs ("CentralCartesians",
        "CentralBondDistances", "CentralAngles", "CentralDihedrals",
        "SideChainDihedrals") or any of the CVs defined by the `misc.misc.FEATURE_NAMES`
        global variable. The str can also point to a file (.txt, .npy, .nc, .h5)
        which contains the CV data to be loaded (a `pathlib.Path` object can
        also be provided).

        """
        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        traj.load_CV("all")
        self.assertIn("central_dihedrals", traj._CVs)

        # check a bad string is not working
        with self.assertRaises(Exception):
            traj.load_CV("some CV")

        # check the override message
        from encodermap.trajinfo.info_single import Capturing

        with Capturing() as output:
            traj.load_CV("central_dihedrals", override=True)
        self.assertIn("Overwriting", output[0])

        # load unaligned data from npy file
        with self.assertRaises(Exception):
            traj.load_CV(
                Path(__file__).resolve().parent / "data/1am7_center_of_mass.npy"
            )

        # check loading with a .npy file
        traj.load_CV(
            Path(__file__).resolve().parent / "data/1am7_center_of_mass_x.npy",
            attr_name="center_of_mass_x",
            override=True,
        )
        self.assertIn("center_of_mass_x", traj._CVs)

        # check loading with npy file without attr_name
        traj.load_CV(Path(__file__).resolve().parent / "data/1am7_center_of_mass_x.npy")
        self.assertIn("1am7_center_of_mass_x", traj._CVs)

        # check loading with a txt file and usecols
        traj.load_CV(
            Path(__file__).resolve().parent / "data/1am7_center_of_mass.txt",
            attr_name="center_of_mass",
            cols=[1, 2],
            override=True,
        )
        self.assertIn("center_of_mass", traj._CVs)
        self.assertEqual(traj.CVs["center_of_mass"].shape[1], 2)

        # check loading with nc file
        traj.load_CV(
            Path(__file__).resolve().parent / "data/1am7_center_of_mass_x.nc",
            attr_name="test",
        )
        self.assertIn("test", traj._CVs)

        # check loading nc dataset with attr_name raises error
        with self.assertRaises(Exception):
            traj.load_CV(
                Path(__file__).resolve().parent / "data/larger_dataset.nc",
                attr_name="test",
            )

        # check loading without override
        with self.assertRaises(Exception):
            traj.load_CV(
                Path(__file__).resolve().parent / "data/1am7_center_of_mass.txt",
                attr_name="center_of_mass_x",
                cols=[2],
                override=False,
            )
        self.assertTrue(np.isclose(traj.CVs["center_of_mass_x"][0][0], 3.81903247))

        # check loading with override
        traj.load_CV(
            Path(__file__).resolve().parent / "data/1am7_center_of_mass.txt",
            attr_name="center_of_mass_x",
            cols=[2],
            override=True,
        )
        self.assertFalse(np.isclose(traj.CVs["center_of_mass_x"][0][0], 3.81903247))

        # check loading a npy array from memory
        data = np.load(
            Path(__file__).resolve().parent / "data/1am7_center_of_mass_x.npy"
        )
        traj.load_CV(data, attr_name="npy_data")
        self.assertIn("npy_data", traj._CVs)

        # check loading npy array from memory without attr_name fails
        with self.assertRaises(Exception):
            traj.load_CV(data)

        # check loading npy array with same name overrides
        traj.load_CV(data, attr_name="npy_data", override=True)

    def test_load_CVs_from_other_sources(self):
        xtc_file = Path(__file__).resolve().parent / "data/1am7_corrected.xtc"
        pdb_file = Path(__file__).resolve().parent / "data/1am7_protein.pdb"
        traj = SingleTraj(xtc_file, pdb_file)

        # list of str
        traj.load_CV(["central_distances", "central_angles"])
        self.assertEqual(
            [i for i in traj._CVs.data_vars], ["central_distances", "central_angles"]
        )

        # np array
        traj.load_CV(np.ones((traj.n_frames, 5)), attr_name="ones")
        traj.load_CV(np.ones((traj.n_frames, 5, 3)), attr_name="pos_arr")
        self.assertTrue(np.all(traj.ones == 1))
        self.assertTrue(np.all(traj.pos_arr == 1))
        self.assertEqual(
            list(traj._CVs.data_vars),
            ["central_distances", "central_angles", "ones", "pos_arr"],
        )

        # da and ds
        da = np_to_xr(np.ones((traj.n_frames, 5)), traj, attr_name="ones2")
        traj.load_CV(da)
        ds = xr.Dataset(
            {"ones3": np_to_xr(np.ones((traj.n_frames, 5)), traj, attr_name="ones3")}
        )
        traj.load_CV(ds)
        self.assertTrue(np.all(traj.ones2 == 1))
        self.assertTrue(np.all(traj.ones3 == 1))
        self.assertEqual(
            list(traj._CVs.data_vars),
            [
                "central_distances",
                "central_angles",
                "ones",
                "pos_arr",
                "ones2",
                "ones3",
            ],
        )

        # duplicate the stuff and check override
        with self.assertRaises(Exception):
            traj.load_CV(np.zeros((traj.n_frames, 5)), attr_name="ones")
        traj.load_CV(np.zeros((traj.n_frames, 5)), attr_name="ones", override=True)
        traj.load_CV(
            np.zeros((traj.n_frames, 5, 3)), attr_name="pos_arr", override=True
        )
        self.assertTrue(np.all(traj.ones == 0))
        self.assertTrue(np.all(traj.pos_arr == 0))

        da = np_to_xr(np.zeros((traj.n_frames, 5)), traj, attr_name="ones2")
        traj.load_CV(da, override=True)
        ds = xr.Dataset(
            {"ones3": np_to_xr(np.zeros((traj.n_frames, 5)), traj, attr_name="ones3")}
        )
        traj.load_CV(ds, override=True)
        self.assertTrue(np.all(traj.ones2 == 0))
        self.assertTrue(np.all(traj.ones3 == 0))

        # test feature and featurizer
        feat = Featurizer(traj)
        feat.add_backbone_torsions()
        traj.load_CV(feat)
        self.assertIn("BackboneTorsionFeature", traj._CVs.data_vars)

        feature = CentralDihedrals(traj.top)
        traj.load_CV(feature)
        self.assertIn("central_dihedrals", traj._CVs.data_vars)

    def test_load_CVs_TrajEnsemble(self):
        traj1 = SingleTraj(Path(__file__).resolve().parent / "data/1YUG.pdb")[:15]
        traj2 = SingleTraj(Path(__file__).resolve().parent / "data/1YUF.pdb")[:15]
        trajs = TrajEnsemble([traj1, traj2])
        print(trajs)

        # load single string
        # assert raises ValueError trajs.load_CVs("numpy_test")
        trajs.load_CVs("numpy")
        trajs.load_CVs("text")
        self.assertIn("numpy", trajs._CVs.data_vars)
        self.assertIn("text", trajs._CVs.data_vars)

        # load nc dataset this overrides everything
        trajs = TrajEnsemble([traj1, traj2])
        trajs.load_CVs(
            Path(__file__).resolve().parent / "data/1YUG_and_1YUF_dataset.nc"
        )
        self.assertIn("numpy", trajs._CVs.data_vars)
        self.assertIn("text", trajs._CVs.data_vars)

        # load a list of feats
        # currently not possible with pyemma
        # make a self assert raises
        # trajs.load_CVs(["central_distances", "central_angles"])

        feats = [
            np.load(Path(__file__).resolve().parent / "data/1YUG_numpy.npy"),
            np.load(Path(__file__).resolve().parent / "data/1YUG_numpy.npy").tolist(),
        ]
        trajs.load_CVs(feats, attr_name="numpy2")
        self.assertIn("numpy2", trajs._CVs.data_vars)

        # assert raises ValueError
        # trajs.load_CVs(np.ones((3, 15, 300)), attr_name="ones")
        trajs.load_CVs(np.ones((2, 15, 300)), attr_name="ones")
        self.assertIn("ones", trajs._CVs.data_vars)
        self.assertTrue(np.all(trajs.ones == 1))
        trajs.load_CVs(np.zeros((2, 15, 300)), attr_name="ones2")
        self.assertIn("ones2", trajs._CVs.data_vars)
        self.assertTrue(np.all(trajs.ones2 == 0))

        # load ensemble features
        trajs = TrajEnsemble(
            [
                Path(__file__).resolve().parent / "data/asp7.xtc",
                Path(__file__).resolve().parent / "data/glu7.xtc",
            ],
            [
                Path(__file__).resolve().parent / "data/asp7.pdb",
                Path(__file__).resolve().parent / "data/glu7.pdb",
            ],
            common_str=["asp7", "glu7"],
        )

        trajs.load_CVs("all", ensemble=True)

        self.assertIn("central_distances", trajs._CVs)
        self.assertTrue(np.any(np.isnan(trajs._CVs.side_dihedrals.values)))
        self.assertFalse(np.any(np.isnan(trajs._CVs.central_distances.values)))

    def test_SingleTraj_throws_error_on_wrong_way_round(self):
        from encodermap.misc.errors import BadError

        with self.assertRaises(BadError):
            test = SingleTraj("test.pdb", "test.xtc")

    def test_CVs_stay_after_subsample(self):
        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        traj.load_CV("central_angles")
        self.assertGreater(traj._CVs.central_angles.size, 0)
        traj_sliced = traj[::10]
        self.assertGreater(
            traj._CVs.central_angles.size, traj_sliced._CVs.central_angles.size
        )

        # the same with ensemble
        trajs = traj._gen_ensemble()
        trajs_subsampled = trajs.subsample(10)
        print(trajs_subsampled._CVs)
        self.assertNotEqual(trajs_subsampled._CVs.central_angles.size, 0)
        self.assertGreater(
            trajs._CVs.central_angles.size, trajs_subsampled._CVs.central_angles.size
        )
        self.assertEqual(
            trajs_subsampled._CVs.central_angles.size,
            traj_sliced._CVs.central_angles.size,
        )

    def test_from_pdbid(self):
        traj = SingleTraj.from_pdb_id("1UBQ")

    def test_load_h5(self):
        traj = SingleTraj(Path(__file__).resolve().parent / "data/traj.h5")
        # should be false, because .xyz has not been calles
        self.assertFalse(traj.trajectory)
        self.assertFalse(traj.topology)
        self.assertEqual(traj.backend, "no_load")
        # data loaded not loaded because h5
        self.assertEqual(traj.n_frames, 100)
        self.assertEqual(traj.backend, "no_load")
        self.assertEqual(traj.n_atoms, 22)
        self.assertEqual(traj.n_residues, 3)
        self.assertEqual(traj.n_chains, 1)
        self.assertFalse(traj.trajectory)

        self.assertEqual(
            traj.traj_file, str(Path(__file__).resolve().parent / "data/traj.h5")
        )
        self.assertEqual(
            traj._traj_file, Path(__file__).resolve().parent / "data/traj.h5"
        )
        self.assertEqual(
            traj._top_file, Path(__file__).resolve().parent / "data/traj.h5"
        )
        self.assertEqual(traj.extension, ".h5")
        self.assertEqual(traj.basename, "traj")
        self.assertEqual(traj.index, (None,))
        self.assertIsNone(traj.traj_num)
        traj.load_traj()
        self.assertEqual(traj.backend, "mdtraj")
        # mdtraj equality checks
        self.assertEqual(
            traj.trajectory,
            md.load(str(Path(__file__).resolve().parent / "data/traj.h5")),
        )
        self.assertEqual(
            traj.topology,
            md.load_topology(str(Path(__file__).resolve().parent / "data/traj.h5")),
        )

        # check indexing
        self.assertEqual(traj[::2].n_frames, 50)
        self.assertEqual(traj[:10].n_frames, 10)
        self.assertEqual(traj[[0, 1, 5, 10, 50]].n_frames, 5)

    def test_traj_id(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        traj2 = SingleTraj(Path(__file__).resolve().parent / "data/1GHC.pdb")
        traj3 = SingleTraj(Path(__file__).resolve().parent / "data/traj.h5")

        for t, length in zip([traj1, traj2, traj3], [51, 14, 100]):
            self.assertTrue(np.array_equal(t.id, np.arange(length)))

        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1GHC.pdb", traj_num=2
        )
        traj3 = SingleTraj(Path(__file__).resolve().parent / "data/traj.h5", traj_num=3)

        for t, length, traj_num in zip([traj1, traj2, traj3], [51, 14, 100], [1, 2, 3]):
            self.assertTrue(
                np.array_equal(
                    t.id, np.vstack([np.full(length, traj_num), np.arange(length)]).T
                )
            )

    def test_slicing_and_CVs_xtc_and_h5(self):
        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/asp7.xtc",
            top=Path(__file__).resolve().parent / "data/asp7.pdb",
        )

        traj.load_CV(np.ones((len(traj), 5)), attr_name="ones")
        traj.load_CV(np.zeros((len(traj), 5, 3)), attr_name="zeros")
        self.assertEqual(traj.CVs["ones"].shape, (100, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (100, 5, 3))
        traj = traj[::2]
        self.assertEqual(traj.CVs["ones"].shape, (50, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (50, 5, 3))
        traj = traj[[0, 2, 4, 6, 8]]
        self.assertEqual(traj.CVs["ones"].shape, (5, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (5, 5, 3))

        traj = SingleTraj(Path(__file__).resolve().parent / "data/asp7.h5")
        self.assertIn("ones", traj.CVs.keys())
        self.assertIn("zeros", traj.CVs.keys())
        test_arr = np.array([0, 2, 4, 6, 8])
        msg = f"Checking {traj=}, with {traj._orig_frames=}, and {test_arr=}"
        self.assertTrue(np.array_equal(traj._orig_frames, test_arr), msg=msg)
        self.assertEqual(traj.index, (None,))
        traj = traj[::2]
        self.assertEqual(traj.CVs["ones"].shape, (3, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (3, 5, 3))

        traj = SingleTraj(Path(__file__).resolve().parent / "data/asp7.h5")
        self.assertEqual(traj.xyz.shape, (5, 73, 3))
        self.assertEqual(traj.CVs["ones"].shape, (5, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (5, 5, 3))

        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/asp7.h5", index=([0, 1],)
        )
        self.assertEqual(traj.xyz.shape, (2, 73, 3))
        self.assertEqual(traj.CVs["ones"].shape, (2, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (2, 5, 3))

        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/asp7.h5", index=slice(None, None, 2)
        )
        self.assertEqual(traj.index, (slice(None, None, 2),))
        self.assertEqual(traj.CVs["ones"].shape, (3, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (3, 5, 3))

        traj = SingleTraj(Path(__file__).resolve().parent / "data/traj.h5")
        self.assertEqual(traj.CVs, {})

    def test_double_slicing(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1GHC.pdb", traj_num=2
        )
        traj3 = SingleTraj(Path(__file__).resolve().parent / "data/traj.h5", traj_num=3)

        test1 = np.array([[1, 0], [1, 5]])
        test2 = np.array([[2, 0], [2, 5]])
        test3 = np.array([[3, 0], [3, 5]])

        for t, len1, len2, len3, arr1 in zip(
            [traj1, traj2, traj3],
            [51, 14, 100],
            [11, 3, 20],
            [6, 2, 10],
            [test1, test2, test3],
        ):
            self.assertEqual(t.index, (None,))
            self.assertEqual(t.n_frames, len1)
            frames = np.arange(len1)
            self.assertEqual(t[::5].index, (None, slice(None, None, 5)))
            self.assertEqual(t[::5].n_frames, len2, print(t.n_frames, t[::5].n_frames))
            self.assertEqual(t[::5][::2].n_frames, len3)
            self.assertHasAttr(t, "_orig_frames")
            self.assertHasAttr(t[::5], "_orig_frames")
            self.assertHasAttr(t[::5][::2], "_orig_frames")
            self.assertEqual(t[::5][::2].n_frames, len(frames[::5][::2]))
            self.assertEqual(len(t[::5][::2].id), len(frames[::5][::2]))
            self.assertTrue(np.array_equal(t[::5].id[:2], arr1))

    def test_slicing_and_indexing_mixed(self):
        traj1_md = md.load(
            str(Path(__file__).resolve().parent / "data/1am7_corrected.xtc"),
            top=str(Path(__file__).resolve().parent / "data/1am7_protein.pdb"),
        )
        traj2_md = md.load_pdb(str(Path(__file__).resolve().parent / "data/1GHC.pdb"))
        traj3_md = md.load(str(Path(__file__).resolve().parent / "data/traj.h5"))
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1GHC.pdb", traj_num=2
        )
        traj3 = SingleTraj(Path(__file__).resolve().parent / "data/traj.h5", traj_num=3)

        for t, t_test in zip([traj1, traj2, traj3], [traj1_md, traj2_md, traj3_md]):
            self.assertEqual(t[1:14:3].n_frames, t_test[1:14:3].n_frames)
            self.assertEqual(t[1:14:3][::2].n_frames, t_test[1:14:3][::2].n_frames)
            self.assertEqual(t[1:14:3][1:4:2].n_frames, t_test[1:14:3][1:4:2].n_frames)
            self.assertTrue(
                np.array_equal(
                    t[1:14:3][1:4:2].id, np.array([[t.traj_num, 4], [t.traj_num, 10]])
                )
            )
            self.assertEqual(
                t[np.array([1, 4, 6, 8, 12])].n_frames,
                t_test[np.array([1, 4, 6, 8, 12])].n_frames,
            )
            if t.basename == "1am7_corrected":
                print(t[5].id)
                print(t[::2].id)
                print(t[[1, 4, 5]].id)
                print(t[[1, 4, 5]][0].id)
            self.assertTrue(
                np.array_equal(
                    t[np.array([1, 4, 6, 8, 12])][0].id, np.array([[t.traj_num, 1]])
                ),
                msg=(
                    f"The slicing of the trajectory with basename {t.basename} and "
                    f"traj_file {t.traj_file} did not work. It was expected, that the "
                    f"`.id` attribute of this traj results in an array with shape (1, 2) and "
                    f"the values ({t.traj_num}, 1), but returned was {t[np.array([1, 4, 6, 8, 12])][0].id}."
                ),
            )
            self.assertTrue(
                np.array_equal(
                    t[np.array([1, 4, 6, 8, 12])][::2].id,
                    np.array([[t.traj_num, 1], [t.traj_num, 6], [t.traj_num, 12]]),
                )
            )

    def test_CVs_for_TrajEnsemble_containing_only_single_frames(self):
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
        trajs.load_CVs(np.random.randint(0, 10, trajs.n_frames), "cluster_membership")
        shapes = {
            "central_distances": (51, 473),
            "cluster_membership": (51,),
            "central_angles": (51, 472),
            "side_dihedrals": (51, 316),
            "central_cartesians": (51, 474, 3),
            "central_dihedrals": (51, 471),
        }
        self.assertEqual({k: v.shape for k, v in trajs.CVs.items()}, shapes)
        new_trajs = trajs.split_into_frames()
        self.assertEqual({k: v.shape for k, v in new_trajs.CVs.items()}, shapes)

    def test_CV_slicing_SingleTraj(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1GHC.pdb", traj_num=2
        )
        traj3 = SingleTraj(Path(__file__).resolve().parent / "data/traj.h5", traj_num=3)

        time1 = np.array(
            [45100.0, 45300.0, 45500.0, 45700.0, 45900.0, 46100.0, 46300.0]
        )
        time2 = np.array([1, 3, 5, 7, 9, 11, 13])
        time3 = np.array([0.004, 0.008, 0.012, 0.016, 0.02, 0.024, 0.028])

        for i, (t, time) in enumerate(
            zip([traj1, traj2, traj3], [time1, time2, time3])
        ):
            t.load_CV(t.xyz[:, :, 1], "y_coordinate")
            frame = t[0]
            self.assertEqual(frame.CVs["y_coordinate"].shape, (frame.n_atoms,))
            frames = t[np.array([0, 2, 4])]
            self.assertEqual(frames.CVs["y_coordinate"].shape, (3, frame.n_atoms))
            frame = frames[0]
            self.assertTrue(
                np.array_equal(
                    frame.CVs["y_coordinate"][0], frames.CVs["y_coordinate"][0, 0]
                )
            )
            frames = t[1:14:2]
            self.assertEqual(frames.CVs["y_coordinate"].shape, (7, frame.n_atoms))
            if i < 2:
                self.assertTrue(np.array_equal(frames._CVs.coords["time"].values, time))
            else:
                # The values are different within floating point precision
                self.assertTrue(np.allclose(frames._CVs.coords["time"].values, time))

    def test_SingleTraj_mdtraj_duplication(self):
        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        try:
            self.assertEqual(
                traj.select("name CA")[:5].tolist(), [4, 21, 37, 52, 71]
            )  # await for fix from mdtraj developers
        except AttributeError as e:
            if e.__str__() == "'Constant' object has no attribute 'kind'":
                # some weird bug in mdtraj happended
                self.assertEqual(
                    traj.select("backbone")[:5].tolist(), [0, 4, 17, 18, 19]
                )
            else:
                raise e
        self.assertEqual(
            md.compute_dssp(traj.traj)[0, :5].tolist(), ["C", "C", "C", "C", "C"]
        )
        com = np.load(Path(__file__).resolve().parent / "data/1am7_center_of_mass.npy")
        self.assertTrue(np.allclose(md.compute_center_of_mass(traj.traj)[0], com))

    def test_SingleTraj_subsample(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        traj1.load_traj()
        self.assertEqual(traj1.backend, "mdtraj")
        self.assertTrue(traj1.trajectory)
        subsample = traj1[::2]
        self.assertEqual(subsample.backend, "mdtraj")
        self.assertTrue(subsample.trajectory)
        subsample = subsample.traj
        self.assertEqual(subsample.n_frames, 26)
        subsample = traj1[[0, 1, 5, 6]].traj
        self.assertEqual(subsample.n_frames, 4)
        subsample = traj1[5:46:3].traj
        self.assertEqual(subsample.n_frames, 14)

    def test_SingleTraj_subsample_without_loading(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        self.assertEqual(traj1.backend, "no_load")
        self.assertFalse(traj1.trajectory)
        subsample = traj1[::2]
        self.assertEqual(subsample.backend, "no_load")
        subsample = subsample.traj
        self.assertEqual(subsample.n_frames, 26)
        subsample = traj1[[0, 1, 5, 6]].traj
        self.assertEqual(subsample.n_frames, 4)
        subsample = traj1[5:46:3].traj
        self.assertEqual(subsample.n_frames, 14)

    def test_SingleTraj_subsample_h5(self):
        traj1 = SingleTraj(Path(__file__).resolve().parent / "data/traj.h5")
        traj1.load_traj()
        subsample = traj1[::3].traj
        self.assertEqual(subsample.n_frames, 34)
        subsample = traj1[[0, 1, 5, 6]].traj
        self.assertEqual(subsample.n_frames, 4)
        subsample = traj1[5:46:3].traj
        self.assertEqual(subsample.n_frames, 14)

    def test_addition_along_TrajEnsemble(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part2.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=2,
        )
        traj1.load_CV(traj1.xyz[:, :, 1], "y_coordinate")
        traj2.load_CV(traj2.xyz[:, :, 1], "y_coordinate")
        traj2.load_CV(traj2.xyz[:, :, 2], "z_coordinate")

        trajs = traj1 + traj2
        self.assertIsInstance(trajs, TrajEnsemble)
        self.assertEqual(trajs.n_frames, 51)
        self.assertEqual(trajs.CVs["y_coordinate"].shape, (51, 2504))
        self.assertEqual(trajs.n_trajs, 2)
        self.assertEqual(list(trajs.CVs.keys()), ["y_coordinate"])
        self.assertEqual(
            trajs.traj_files,
            [
                str(Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc"),
                str(Path(__file__).resolve().parent / "data/1am7_corrected_part2.xtc"),
            ],
        )
        trajs.load_trajs()
        self.assertEqual(
            trajs.top_files, [Path(__file__).resolve().parent / "data/1am7_protein.pdb"]
        )
        self.assertEqual(trajs.y_coordinate.shape, (51, 2504))

    def test_gen_ensemble(self):
        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        trajs = traj._gen_ensemble()
        self.assertIsInstance(trajs, TrajEnsemble)
        self.assertEqual(trajs.n_trajs, 1)
        self.assertEqual(trajs.n_frames, 51)
        self.assertEqual(trajs.xyz.shape, (51, 2504, 3))

    def test_atom_slice(self):
        traj = SingleTraj(Path(__file__).resolve().parent / "data/1UBQ.pdb", traj_num=2)
        self.assertEqual(traj.n_chains, 2)
        # printing chains does not require load
        self.assertEqual(traj.backend, "no_load")
        self.assertEqual(traj.basename, "1UBQ")
        self.assertEqual(traj.extension, ".pdb")
        self.assertEqual(traj.n_atoms, 660)
        traj.load_CV(traj.xyz[:, :, 1], "y_coordinate")
        with self.assertWarns(UserWarning):
            traj.atom_slice(traj.top.select("name CA"))
        traj.atom_slice(traj.top.select("name CA"), inplace=True)
        self.assertEqual(traj.n_atoms, 76)

    def test_stack(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part2.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=2,
        )
        with self.assertRaises(ValueError):
            traj1.stack(traj2)
        new = traj1.stack(traj2[:25])
        self.assertEqual(new.n_atoms, 5008)
        self.assertEqual(new.n_residues, 316)
        self.assertEqual(new.n_frames, 25)

    def test_join(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part2.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
            traj_num=2,
        )
        new = traj1.join(traj2)
        self.assertEqual(new.n_atoms, 2504)
        self.assertEqual(new.n_residues, 158)
        self.assertEqual(new.n_frames, 51)

    def test_wrong_formatted_CVs(self):
        traj = SingleTraj(Path(__file__).resolve().parent / "data/1YUF.pdb")
        test = np.append(traj.xyz[:, 0], [5])
        with self.assertRaises(Exception):
            traj.load_CV(test, "test")

    def test_info_all_loading(self):
        with self.assertRaises(Exception):
            trajs = TrajEnsemble(
                [
                    Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc",
                    Path(__file__).resolve().parent / "data/1am7_corrected_part2.xtc",
                ],
                tops=[
                    Path(__file__).resolve().parent / "data/1am7_protein.pdb",
                    Path(__file__).resolve().parent / "data/1am7_protein1.pdb",
                    Path(__file__).resolve().parent / "data/1am7_protein2.pdb",
                ],
            )
        trajs = TrajEnsemble(
            [
                Path(__file__).resolve().parent / "data/1YUG.pdb",
                Path(__file__).resolve().parent / "data/1YUF.pdb",
            ]
        )
        self.assertEqual(trajs.n_frames, 31)

    def test_pyemma_indexing_and_get_single_frame(self):
        trajs = TrajEnsemble(
            [
                Path(__file__).resolve().parent / "data/1YUG.pdb",
                Path(__file__).resolve().parent / "data/1YUF.pdb",
            ],
            common_str=["YUG", "YUF"],
        )
        self.assertEqual(trajs.n_frames, 31)
        test_frames = []
        for traj in trajs:
            self.assertEqual(traj.__class__.__name__, "SingleTraj")
            for i, frame in enumerate(traj):
                self.assertEqual(frame.__class__.__name__, "SingleTraj")
                if i > 5 and i <= 7:
                    test_frames.append(frame)

        test_arrays = [
            np.array([[0, 6]]),
            np.array([[0, 7]]),
            np.array([[1, 6]]),
            np.array([[1, 7]]),
        ]
        for tf, ta in zip(test_frames, test_arrays):
            self.assertTrue(np.array_equal(tf.id, ta))

        single_frame = trajs.get_single_frame(30)
        self.assertTrue(np.array_equal(single_frame.id, np.array([[1, 15]])))

        index = np.array([[0, 1], [0, 2], [1, 0], [1, 15]])
        frames = trajs[index]
        self.assertIsInstance(frames, TrajEnsemble)
        self.assertTrue(np.array_equal(frames.id, index))

    def test_traj_joined(self):
        traj1 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        traj2 = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part2.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        trajs = TrajEnsemble([traj1.traj, traj2.traj])
        self.assertIsInstance(trajs, TrajEnsemble)
        self.assertEqual(
            trajs.top[0],
            md.load_topology(
                str(Path(__file__).resolve().parent / "data/1am7_protein.pdb")
            ),
            msg=f"The tops of traj does seem to be an empty list {trajs.top}.",
        )
        self.assertEqual(trajs.n_residues, [158, 158])
        self.assertEqual(trajs.basenames, [None, None])
        split_into_frames = trajs.split_into_frames()
        self.assertEqual(split_into_frames.n_frames, 51)
        test = split_into_frames.traj_joined
        self.assertIsInstance(test, md.Trajectory)

    def test_adding_mixed_pyemma_features_with_custom_names(self):
        traj = SingleTraj(
            Path(__file__).resolve().parent / "data/1am7_corrected_part1.xtc",
            top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        from encodermap import Featurizer

        featurizer = Featurizer(traj)
        featurizer.add_distances_ca(excluded_neighbors=0)
        traj.load_CV(featurizer, attr_name="Custom_Feature_1")
        self.assertHasAttr(traj, "Custom_Feature_1")

        featurizer.add_list_of_feats("central_dihedrals")
        with self.assertRaises(TypeError):
            traj.load_CV(featurizer, attr_name="Custom_Feature")
        traj.load_CV(
            featurizer,
            attr_name=["Custom_Feature_2", "Custom_Feature_3"],
            override=True,
        )

        self.assertEqual(traj.Custom_Feature_1.shape, (25, 12403))
        self.assertEqual(traj.Custom_Feature_2.shape, (25, 12403))
        self.assertEqual(traj.Custom_Feature_3.shape, (25, 471))

    def test_info_all_load_CVs_from_file(self):
        traj1 = SingleTraj(Path(__file__).resolve().parent / "data/1YUG.pdb")[:15]
        traj2 = SingleTraj(Path(__file__).resolve().parent / "data/1YUF.pdb")[:15]
        trajs = TrajEnsemble([traj1, traj2])
        with self.assertRaises(Exception):
            trajs.load_CVs(
                [
                    Path(__file__).resolve().parent / "data//1NOT_text.txt",
                    Path(__file__).resolve().parent / "data//1YUG_text.txt",
                ]
            )
        trajs.load_CVs(
            [
                Path(__file__).resolve().parent / "data/1YUF_numpy.npy",
                Path(__file__).resolve().parent / "data/1YUG_numpy.npy",
            ],
            "y_coordinate_1",
        )
        self.assertEqual(trajs.y_coordinate_1.shape, (30, 720))
        trajs.load_CVs(
            [
                Path(__file__).resolve().parent / "data/1YUF_text.txt",
                Path(__file__).resolve().parent / "data/1YUG_text.txt",
            ],
            "y_coordinate_2",
        )
        self.assertEqual(trajs.y_coordinate_2.shape, (30, 720))
        trajs.load_CVs(
            [
                np.load(Path(__file__).resolve().parent / "data/1YUG_numpy.npy"),
                np.load(Path(__file__).resolve().parent / "data/1YUF_numpy.npy"),
            ],
            "y_coordinate_3",
        )
        self.assertEqual(trajs.y_coordinate_3.shape, (30, 720))
        trajs.load_CVs(Path(__file__).resolve().parent / "data/", "y_coordinate_4")
        self.assertEqual(trajs.y_coordinate_4.shape, (30, 720))

        traj1 = SingleTraj(Path(__file__).resolve().parent / "data/1YUG.pdb")[:15]
        traj2 = SingleTraj(Path(__file__).resolve().parent / "data/1YUF.pdb")[:15]
        trajs = TrajEnsemble([traj1, traj2])
        with self.assertRaises(Exception):
            trajs.load_CVs(
                [
                    Path(__file__).resolve().parent / "data//1NOT_text.txt",
                    Path(__file__).resolve().parent / "data//1YUG_text.txt",
                ]
            )
        trajs.load_CVs(
            [
                Path(__file__).resolve().parent / "data/1YUF_numpy.npy",
                Path(__file__).resolve().parent / "data/1YUG_numpy.npy",
            ],
            "y_coordinate_1",
        )
        self.assertEqual(trajs.y_coordinate_1.shape, (30, 720))
        trajs.load_CVs(
            [
                Path(__file__).resolve().parent / "data/1YUF_text.txt",
                Path(__file__).resolve().parent / "data/1YUG_text.txt",
            ],
            "y_coordinate_2",
        )
        self.assertEqual(trajs.y_coordinate_2.shape, (30, 720))
        trajs.load_CVs(
            [
                np.load(Path(__file__).resolve().parent / "data/1YUG_numpy.npy"),
                np.load(Path(__file__).resolve().parent / "data/1YUF_numpy.npy"),
            ],
            "y_coordinate_3",
        )
        self.assertEqual(trajs.y_coordinate_3.shape, (30, 720))
        trajs.load_CVs(Path(__file__).resolve().parent / "data/", "y_coordinate_4")
        self.assertEqual(trajs.y_coordinate_4.shape, (30, 720))

    def test_info_all_load_CVs_from_numpy(self):
        traj1 = SingleTraj(Path(__file__).resolve().parent / "data/1YUG.pdb")
        traj2 = SingleTraj(Path(__file__).resolve().parent / "data/1YUF.pdb")
        trajs = TrajEnsemble([traj1, traj2])
        # load some random CVs
        # Seven cases:
        # Normal with shape (n_trajs, max([traj.n_frames for traj in trajs]), n) with nans
        # Normal with shape (n_frames, n)
        # Per Frame with shape (n_trajs, max([traj.n_frames for traj in trajs])) with nans
        # Per Frame with shape (n_frames, )
        # Coordinates with shape (n_trajs, max([traj.n_frames for traj in trajs]), n_atoms, 3) with nans
        # Coordinates with shape (n_frames, n_atoms, 3)
        # Wrong aligned array
        wrong_array = np.random.random((20, 12))
        with self.assertRaises(ValueError):
            trajs.load_CVs(wrong_array, "wrong")

        n_trajs = 2
        tot_frames = trajs.n_frames
        frames = [t.n_frames for t in trajs]
        n_atoms = list(set([t.n_atoms for t in trajs]))[0]

        normal_CVs = np.random.random((tot_frames, 12))
        trajs.load_CVs(normal_CVs, "normal_CVs")
        normal_CVs_per_traj = np.random.random((n_trajs, max(frames), 12))
        normal_CVs_per_traj[0, 15, :] = np.nan
        trajs.load_CVs(normal_CVs_per_traj, "normal_CVs_per_traj")
        self.assertEqual(trajs[0].normal_CVs_per_traj.shape[0], 15)
        self.assertEqual(trajs[1].normal_CVs_per_traj.shape[0], 16)

        per_frame_CVs = np.random.random((tot_frames))
        trajs.load_CVs(per_frame_CVs, "per_frame_CVs")
        per_frame_CVs_per_traj = np.random.random((n_trajs, max(frames)))
        per_frame_CVs_per_traj[0, 15] = np.nan
        trajs.load_CVs(per_frame_CVs_per_traj, "per_frame_CVs_per_traj")
        self.assertEqual(trajs[0].per_frame_CVs_per_traj.shape[0], 15)
        self.assertEqual(trajs[1].per_frame_CVs_per_traj.shape[0], 16)

        coordinates_CVs = np.random.random((tot_frames, n_atoms, 3))
        trajs.load_CVs(coordinates_CVs, "coordinates_CVs")
        coordinates_CVs_per_traj = np.random.random((n_trajs, max(frames), n_atoms, 3))
        coordinates_CVs_per_traj[0, 15, :, :] = np.nan
        trajs.load_CVs(coordinates_CVs_per_traj, "coordinates_CVs_per_traj")
        self.assertEqual(trajs[0].per_frame_CVs_per_traj.shape[0], 15)
        self.assertEqual(trajs[1].per_frame_CVs_per_traj.shape[0], 16)


testSuite = unittest.TestSuite()
testSuite.addTests(unittest.makeSuite(TestTraj))

import doctest

import encodermap.trajinfo as trajinfo

testSuite.addTest(doctest.DocTestSuite(trajinfo))
unittest.TextTestRunner(verbosity=2).run(testSuite)

if __name__ == "__main__":
    unittest.main()

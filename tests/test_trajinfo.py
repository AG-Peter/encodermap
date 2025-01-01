# -*- coding: utf-8 -*-
# tests/test_trajinfo.py
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
"""Main tests for the `TrajEnsemble` and `SingleTraj` classes. The following suites
are available:
    * TestTraj: Tests all aspects of the classes `TrajEnsemble` and `SingleTraj`.

"""
# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import importlib.metadata
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from io import StringIO
from pathlib import Path
from typing import Optional

# Third Party Imports
import mdtraj as md
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from numpy.testing import assert_array_equal

# Encodermap imports
from conftest import skip_all_tests_except_env_var_specified


################################################################################
# Print installed packages for debugging
################################################################################


installed_packages = importlib.metadata.distributions()
installed_packages_list = sorted(
    ["%s==%s" % (i.metadata.get("name"), i.version) for i in installed_packages]
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


# Encodermap imports
from encodermap.kondata import get_from_kondata, get_from_url
from encodermap.loading.features import CentralDihedrals
from encodermap.loading.featurizer import Featurizer
from encodermap.trajinfo import SingleTraj, TrajEnsemble
from encodermap.trajinfo.info_single import Capturing
from encodermap.trajinfo.trajinfo_utils import np_to_xr


try:
    # Local Folder Imports
    from .conftest import expensive_test
except ImportError:
    # Encodermap imports
    from conftest import expensive_test


################################################################################
# Classes
################################################################################


class CapturingStderr(list):
    """Class to capture print statements from function calls"""

    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stderr = self._stderr


@skip_all_tests_except_env_var_specified(unittest.skip)
class TestTraj(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).resolve().parent / "data"
        return cls

    def assertAllEqual(
        self,
        x: np.ndarray,
        y: np.ndarray,
        msg: Optional[str] = None,
    ):
        """Helper that implements numpys assert_allclose"""
        if msg is None:
            msg = ""
        try:
            assert_array_equal(x, y, err_msg=msg)
        except AssertionError as e:
            self.fail(str(e))
        except TypeError as e:
            raise Exception(
                f"assertAllEqual got bad types: {x=} {y=} {type(x)=} {type(y)=}"
            )

    def assertHasAttr(self, obj, intendedAttr):
        """Helper to check whether an attr is present."""
        testBool = hasattr(obj, intendedAttr)
        self.assertTrue(
            testBool, msg=f"obj lacking an attribute. {obj=}, {intendedAttr=}"
        )

    def index_a_single_traj(
        self,
        traj,
        basename: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        """Helper function that takes a traj and tortures it with difficult_slicing"""
        if traj.n_frames == 1:
            return self.index_a_single_frame(traj)

        # test loading and unloading
        self.load_and_unload(traj)

        # test mdtraj and SingleTraj linking
        self.assertEqual(traj.n_frames, traj.traj.n_frames)
        self.assertEqual(traj[:2].n_frames, traj.traj[:2].n_frames)
        self.assertEqual(traj.n_atoms, traj.traj.n_atoms)
        self.assertEqual(traj.n_atoms, traj.traj.n_atoms)
        self.assertEqual(traj.n_residues, traj.traj.n_residues)
        self.assertEqual(traj.n_chains, traj.traj.n_chains)

        # some additional attrs
        if basename is not None:
            self.assertEqual(traj.basename, basename)
        if extension is not None:
            self.assertEqual(traj.extension, extension)

        # test the dash summary
        df = traj.dash_summary()
        self.assertIsInstance(df, pd.DataFrame)

        # test some indexing
        # first take a single frame and run it through the usual shenanigans
        self.index_a_single_frame(traj[0])
        self.assertEqual(traj[0].n_frames, 1)

        # load CVs from numpy
        zeros = np.zeros(len(traj))
        if traj.n_frames > 1:
            self.assertEqual(traj[:2].n_frames, 2)
        if not traj.CVs:
            traj.load_CV(zeros, attr_name="zeros")
        with tempfile.NamedTemporaryFile(suffix=".npy") as f:
            np.save(f.name, zeros)
            traj.load_CV(f.name, "zeros1")
            traj.load_CV(Path(f.name), "zeros2")
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            ar = np.vstack([zeros, np.ones(len(traj))]).T
            np.savetxt(f.name, ar)
            traj.load_CV(Path(f.name), cols=[1])
            self.assertEqual(traj.CVs[Path(f.name).stem][0], 1)
        key = list(traj.CVs.keys())[0]
        self.assertAllEqual(
            traj.CVs[key][0],
            traj[0].CVs[key][0],
        )

        # slice with list and slice
        self.assertAllEqual(traj[:2].xyz, traj[[0, 1]].xyz)
        if traj.traj_num is not None:
            self.assertAllEqual(
                traj[
                    np.array(
                        [
                            [traj.traj_num, 0],
                            [traj.traj_num, 1],
                        ]
                    )
                ].xyz,
                traj[[0, 1]].xyz,
            )

    def index_a_single_frame(self, frame):
        """Helper function"""
        self.assertEqual(frame.n_frames, 1)
        self.load_and_unload(frame)

        # test some mdtraj attributes
        self.assertEqual(len(frame.time), 1)
        self.assertEqual(frame.traj.n_frames, 1)

        if frame.traj_num is not None:
            self.assertEqual(frame.id.shape, (1, 2))
            frame_num = frame.id[0, 1]
        else:
            frame_num = frame.id[0]
        self.assertAllEqual(frame.xyz, frame.fsel[frame_num].xyz)
        if not frame.CVs:
            frame.load_CV(np.zeros((1,)), "zeros")
            self.assertEqual(frame._CVs.zeros.values[0], 0)

    def load_and_unload(self, traj):
        traj.unload()
        self.assertFalse(traj.trajectory)
        self.assertFalse(traj.topology)
        self.assertEqual(traj.backend, "no_load")
        print(traj.xyz[0].shape)
        self.assertEqual(traj.backend, "mdtraj")

    def test_1am7(self):
        # Encodermap imports
        from encodermap import EncoderMap, Parameters, load
        from encodermap.models import gen_sequential_model

        trajs = load(
            trajs=[
                self.data_dir / "1am7_corrected_part1.xtc",
                self.data_dir / "1am7_corrected_part2.xtc",
            ],
            tops=[
                self.data_dir / "1am7_protein.pdb",
            ],
        )
        trajs.load_CVs("all")
        self.assertIn(
            "traj_name",
            trajs._CVs.coords.keys(),
        )
        self.assertTrue(not np.any(np.isnan(trajs.central_dihedrals)))
        parameters = Parameters(periodicity=2 * np.pi)
        self.assertEqual(parameters.model_api, "sequential")
        test = gen_sequential_model(451, parameters)
        self.assertEqual(test.__class__.__name__, "SequentialModel")
        emap = EncoderMap(
            train_data=trajs.central_dihedrals,
            parameters=parameters,
            read_only=True,
        )
        lowd = emap.encode()
        self.assertTrue(not np.any(np.isnan(lowd)))
        trajs.load_CVs(lowd, "lowd")
        self.assertTrue(
            not np.any(np.isnan(trajs.lowd)),
            msg=(f"There's a NaN in the lowds from emap."),
        )
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "trajs.h5"
            trajs.save(file)
            test = load(file)
            self.assertIn(
                "traj_name",
                test._CVs.coords.keys(),
            )
            self.assertTrue(
                not np.any(np.isnan(test.lowd)),
                msg=(f"There's a NaN in the loaded lowds."),
            )

    def test_load_url(self):
        """Test whether `SingleTraj` can be loaded from a URL."""
        traj = SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
        self.index_a_single_traj(
            traj,
            basename="1GHC",
            extension=".pdb",
        )
        self.assertEqual(traj.index, (None,))
        self.assertIsNone(traj.traj_num)
        self.assertEqual("https://files.rcsb.org/view/1GHC.pdb", traj.traj_file)
        self.assertEqual("https://files.rcsb.org/view/1GHC.pdb", traj.top_file)

    def test_single_traj_equality(self):
        """Test, whether the equality operator (==) works with `SingleTraj`."""
        traj1 = SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
        traj2 = SingleTraj("https://files.rcsb.org/view/1GHC.pdb")
        self.assertEqual(traj1, traj2)

    def test_reversed(self):
        """Test, whether the builtin `reversed()` works on `SingleTraj`."""
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            self.data_dir / "1am7_protein.pdb",
        )
        t = np.array([1, 2, 3, 4, 5])
        CV = np.vstack([np.ones((len(traj) - 1, 5)), np.expand_dims(t, 0)])
        traj.load_CV(
            CV, attr_name="ones", labels=["one", "two", "three", "four", "five"]
        )
        self.index_a_single_traj(traj)
        assert np.array_equal(traj.ones[-1], t)

        r = reversed(traj)
        assert np.array_equal(r.ones[0], t)
        assert np.array_equal(r.xyz[0], traj.xyz[-1])

    def test_context_manager(self):
        """Test, whether the context manager of `SingleTraj` works."""
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            self.data_dir / "1am7_protein.pdb",
        )
        with traj as t:
            self.assertIsInstance(t.trajectory, md.Trajectory)
            self.assertIsInstance(t.topology, md.Topology)

        self.assertFalse(t.trajectory)
        self.assertFalse(t.topology)

    def test_load_bad_xr_dataset(self):
        """Test whether an Exception is thrown when a `xr.Dataset` has too many frames."""
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            self.data_dir / "1am7_protein.pdb",
        )
        ds = xr.DataArray(
            np.ones((1, 100, 5)),
            coords={
                "traj_num": ("traj_num", np.asarray([1])),
                "traj_name": ("traj_num", np.asarray([traj.basename])),
                "frame_num": ("frame_num", np.arange(100)),
                "ONES": np.asarray(["one", "two", "three", "four", "five"]),
            },
            dims=["traj_num", "frame_num", "ONES"],
            name="ones",
        ).to_dataset()
        with self.assertRaises(Exception):
            traj.load_CV(ds)

    def load_CVs_from_h5_file(self):
        """Test, whether `SingleTraj.load_CVs` can take an `xr.Dataset` and check the times."""
        # Standard Library Imports
        from copy import deepcopy

        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            self.data_dir / "1am7_protein.pdb",
        )
        traj.load_CV(
            np.ones((len(traj), 5)),
            attr_name="ones",
            labels=["one", "two", "three", "four", "five"],
        )
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            tmp = Path(f.name)
            tmp.touch()

            # raises IOerror
            with self.assertRaises(IOError):
                traj.save("/tmp.json/tmp_file.h5")
            tmp.unlink()
            traj.save("/tmp.json/tmp_file.h5")

            traj2 = SingleTraj("/tmp/tmp_file.h5")
            self.asserTrue(np.array_equal(traj.ones, traj2.ones))

            traj3 = deepcopy(traj2)
            self.asserTrue(np.array_equal(traj3.ones, traj2.ones))
            self.assertEqual(traj3.traj_file, traj2.traj_file)
            self.assertEqual(traj3, traj2)

    def test_save_CV_as_numpy(self):
        """Test whether CVs can be saved as a numpy array."""
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            self.data_dir / "1am7_protein.pdb",
        )
        ds = xr.DataArray(
            np.ones((1, 51, 5)),
            coords={
                "traj_num": ("traj_num", np.asarray([1])),
                "traj_name": ("traj_num", np.asarray([traj.basename])),
                "frame_num": ("frame_num", np.arange(51)),
                "ONES": np.asarray(["one", "two", "three", "four", "five"]),
            },
            dims=["traj_num", "frame_num", "ONES"],
            name="ones",
        ).to_dataset()
        traj.load_CV(ds)
        with tempfile.NamedTemporaryFile(suffix=".npy") as f:
            tmp = Path(f.name)
            tmp.touch()
            with self.assertRaises(OSError):
                traj.save_CV_as_numpy("ones", f.name)
            traj.save_CV_as_numpy("ones", f.name, overwrite=True)
            test = np.load(tmp)
            self.assertTrue(np.array_equal(traj.ones, test))

    def test_too_large_key_raises_index_error(self):
        """Test whetyher too large ineteger keys can raise errors."""
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            self.data_dir / "1am7_protein.pdb",
        )
        with self.assertRaises(IndexError):
            f = traj[100]
        with self.assertRaises(IndexError):
            f = traj[np.arange(100)]

    def test_single_traj_raises_error_on_wrong_dtype_for_traj(self):
        """Test whether the SingleTraj class handles bad dtypes."""
        with self.assertRaises(ValueError):
            SingleTraj(traj=1)

    def test_single_traj_raises_error_on_file_not_found(self):
        """Tests the FileNotFoundError capabilities of `SingleTraj`."""
        with self.assertRaises(FileNotFoundError):
            t = SingleTraj("/tmp/imaginary_pdb_file.pdb")
            t.load_traj()

        with self.assertRaises(FileNotFoundError):
            t = SingleTraj("/tmp/imaginary_xtc_file.xtc", "/tmp/imaginary_pdb_file.pdb")
            t.load_traj()

        with self.assertRaises(FileNotFoundError):
            t = SingleTraj(
                self.data_dir / "1am7_corrected.xtc",
                "/tmp/imaginary_pdb_file.pdb",
            )
            t.load_traj()

    def test_load_h5_with_integer_index(self):
        t = SingleTraj(self.data_dir / "traj.h5", index=2)
        self.index_a_single_traj(t)
        self.assertEqual(t.n_frames, 1)

    def test_load_pdb_with_mdtraj_backend(self):
        traj = SingleTraj("https://files.rcsb.org/view/1GHC.pdb", backend="mdtraj")
        self.assertEqual(traj.basename, "1GHC")

    def test_n_frames_in_h5_file(self):
        traj = SingleTraj(self.data_dir / "asp7.h5")
        self.assertEqual(traj._n_frames_base_h5_file, 5)
        self.index_a_single_traj(traj)

    def test_CVs_in_file(self):
        traj = SingleTraj(self.data_dir / "asp7.h5")
        self.assertTrue(traj.CVs_in_file)

    def test_gen_ensemble_no_files(self):
        traj = md.load_pdb("https://files.rcsb.org/view/1GHC.pdb")
        traj = SingleTraj(traj)
        self.assertEqual(traj.traj_file, ".")
        trajs = traj._gen_ensemble()
        self.assertEqual(trajs.n_frames, traj.n_frames)

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

    def test_load_single_traj_with_traj_and_top(self):
        traj = md.load_pdb("https://files.rcsb.org/view/1GHC.pdb")
        traj = SingleTraj(traj, traj.top)
        self.assertEqual(traj.n_frames, 14)
        self.assertEqual(traj.top.n_chains, 1)

    def test_load_info_all_with_trajs_and_one_top_does_not_raise_error(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".xtc") as f1,
            tempfile.NamedTemporaryFile(suffix=".xtc") as f2,
        ):
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            shutil.copyfile(self.data_dir / "1am7_corrected_part2.xtc", f2.name)
            trajs = ["tests/data/1YUF.pdb", "tests/data/1YUG.pdb"]
            trajs = TrajEnsemble(trajs=trajs, tops=trajs)

            trajs = [
                f1.name,
                f2.name,
            ]
            top = "tests/data/1am7_protein.pdb"
            trajs = TrajEnsemble(trajs=trajs, tops=top)

    def test_single_traj_double_index_with_int(self):
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
        )
        with self.assertRaises(IndexError):
            traj = traj[5][10]

    def test_load_xtc(self):
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
        )
        xyz = np.load(self.data_dir / "1am7_first_frame_first_atom_xyz.npy")
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
            str(self.data_dir / "1am7_corrected.xtc"),
        )
        self.assertEqual(
            traj.top_file,
            str(self.data_dir / "1am7_protein.pdb"),
        )
        self.assertEqual(traj._traj_file, self.data_dir / "1am7_corrected.xtc")
        self.assertEqual(traj._top_file, self.data_dir / "1am7_protein.pdb")
        self.assertEqual(traj.extension, ".xtc")
        self.assertEqual(traj.basename, "1am7_corrected")
        self.assertEqual(traj.index, (None,))
        self.assertIsNone(traj.traj_num)
        self.assertEqual(traj.backend, "mdtraj")
        # mdtraj equality checks
        self.assertEqual(
            traj.trajectory,
            md.load(
                str(self.data_dir / "1am7_corrected.xtc"),
                top=str(self.data_dir / "1am7_protein.pdb"),
            ),
        )
        self.assertEqual(
            traj.topology,
            md.load_topology(str(self.data_dir / "1am7_protein.pdb")),
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
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
        )
        traj.load_CV("all")
        self.assertIn("central_dihedrals", traj._CVs)

        # check a bad string is not working
        with self.assertRaises(Exception):
            traj.load_CV("some CV")

        # check the override message
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex=r".*the following CVs.*",
            msg=(
                "Overwriting existing CVs should issue a warning to the user, but "
                "it does not."
            ),
        ):
            traj.load_CV("central_dihedrals", override=True)

        # load unaligned data from npy file
        with self.assertRaises(Exception):
            traj.load_CV(self.data_dir / "1am7_center_of_mass.npy")

        # check loading with a .npy file
        traj.load_CV(
            self.data_dir / "1am7_center_of_mass_x.npy",
            attr_name="center_of_mass_x",
            override=True,
        )
        self.assertIn("center_of_mass_x", traj._CVs)

        # check loading with npy file without attr_name
        traj.load_CV(self.data_dir / "1am7_center_of_mass_x.npy")
        self.assertIn("1am7_center_of_mass_x", traj._CVs)

        # check loading with a txt file and usecols
        traj.load_CV(
            self.data_dir / "1am7_center_of_mass.txt",
            attr_name="center_of_mass",
            cols=[1, 2],
            override=True,
        )
        self.assertIn("center_of_mass", traj._CVs)
        self.assertEqual(traj.CVs["center_of_mass"].shape[1], 2)

        # check loading with nc file
        traj.load_CV(
            self.data_dir / "1am7_center_of_mass_x.nc",
            attr_name="test",
        )
        self.assertIn("test", traj._CVs)

        # check loading nc dataset with attr_name raises error
        with self.assertRaises(Exception, msg=f"{traj._CVs=}"):
            traj.load_CV(
                self.data_dir / "larger_dataset.nc",
                attr_name="test",
            )

        # check loading without override
        with self.assertRaises(Exception, msg=f"{traj._CVs=}"):
            traj.load_CV(
                self.data_dir / "1am7_center_of_mass.txt",
                attr_name="center_of_mass_x",
                cols=[2],
                override=False,
            )
        self.assertTrue(np.isclose(traj.CVs["center_of_mass_x"][0][0], 3.81903247))

        # check loading with override
        traj.load_CV(
            self.data_dir / "1am7_center_of_mass.txt",
            attr_name="center_of_mass_x",
            cols=[2],
            override=True,
        )
        self.assertFalse(np.isclose(traj.CVs["center_of_mass_x"][0][0], 3.81903247))

        # check loading a npy array from memory
        data = np.load(self.data_dir / "1am7_center_of_mass_x.npy")
        traj.load_CV(data, attr_name="npy_data")
        self.assertIn("npy_data", traj._CVs)

        # check loading npy array from memory without attr_name fails
        with self.assertRaises(Exception):
            traj.load_CV(data)

        # check loading npy array with same name overrides
        traj.load_CV(data, attr_name="npy_data", override=True)

    def test_load_CVs_from_other_sources(self):
        xtc_file = self.data_dir / "1am7_corrected.xtc"
        pdb_file = self.data_dir / "1am7_protein.pdb"
        traj = SingleTraj(xtc_file, pdb_file)

        # list of str
        traj.load_CV(["central_distances", "central_angles"])
        self.assertEqual(
            [i for i in traj._CVs.data_vars],
            [
                "central_distances",
                "central_angles",
                "central_distances_feature_indices",
                "central_angles_feature_indices",
            ],
        )

        # np array
        traj.load_CV(np.ones((traj.n_frames, 5)), attr_name="ones")
        traj.load_CV(np.ones((traj.n_frames, 5, 3)), attr_name="pos_arr")
        self.assertTrue(np.all(traj.ones == 1))
        self.assertTrue(np.all(traj.pos_arr == 1))
        self.assertEqual(
            list(traj._CVs.data_vars),
            [
                "central_distances",
                "central_angles",
                "central_distances_feature_indices",
                "central_angles_feature_indices",
                "ones",
                "pos_arr",
            ],
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
                "central_distances_feature_indices",
                "central_angles_feature_indices",
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

        feature = CentralDihedrals(traj)
        traj.load_CV(feature)
        self.assertIn("central_dihedrals", traj._CVs.data_vars)

    def test_load_CVs_traj_ensemble(self):
        traj1 = SingleTraj(self.data_dir / "1YUG.pdb")[:15]
        traj2 = SingleTraj(self.data_dir / "1YUF.pdb")[:15]
        trajs = TrajEnsemble([traj1, traj2])

        # load single string
        # assert raises ValueError trajs.load_CVs("numpy_test")
        trajs.load_CVs("numpy")
        trajs.load_CVs("text")
        self.assertIn("numpy", trajs._CVs.data_vars)
        self.assertIn("text", trajs._CVs.data_vars)

        # load nc dataset this overrides everything
        trajs = TrajEnsemble([traj1, traj2])
        trajs.load_CVs(self.data_dir / "1YUG_and_1YUF_dataset.nc")
        self.assertIn("numpy", trajs._CVs.data_vars)
        self.assertIn("text", trajs._CVs.data_vars)

        # load a list of feats
        # currently not possible with pyemma
        # make a self assert raises
        # trajs.load_CVs(["central_distances", "central_angles"])

        feats = [
            np.load(self.data_dir / "1YUG_numpy.npy"),
            np.load(self.data_dir / "1YUG_numpy.npy").tolist(),
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
                self.data_dir / "asp7.xtc",
                self.data_dir / "glu7.xtc",
            ],
            [
                self.data_dir / "asp7.pdb",
                self.data_dir / "glu7.pdb",
            ],
            common_str=["asp7", "glu7"],
        )

        trajs.load_CVs("all", ensemble=True)

        self.assertIn("central_distances", trajs._CVs)
        self.assertTrue(np.any(np.isnan(trajs._CVs.side_dihedrals.values)))
        self.assertFalse(np.any(np.isnan(trajs._CVs.central_distances.values)))

    def test_traj_ensemble_equality(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".xtc") as f1,
            tempfile.NamedTemporaryFile(suffix=".xtc") as f2,
        ):
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            shutil.copyfile(self.data_dir / "1am7_corrected_part2.xtc", f2.name)
            traj1 = SingleTraj(
                f1.name,
                top=self.data_dir / "1am7_protein.pdb",
            )
            traj2 = SingleTraj(
                f2.name,
                top=self.data_dir / "1am7_protein.pdb",
            )
            trajs1 = TrajEnsemble([traj1, traj2])
            trajs2 = TrajEnsemble([traj1, traj2])
            self.assertEqual(trajs1, trajs2)

    def test_save_and_load_custom_amino_acids(self):
        # Encodermap imports
        from encodermap.trajinfo.trajinfo_utils import CustomTopology

        yaml_file = self.data_dir / "test.yaml"

        # fmt: off
        custom_aas = CustomTopology.from_dict({
            "CLA": None,
            "SOD": None,
            "POPC": None,
            "POPE": None,
            "SAPI": None,
            "THR": (
                "T",
                {
                    "optional_bonds": [
                        ("-C", "N"),  # the peptide bond to the previous aa
                        ("N", "CA"),
                        ("N", "H"),
                        ("CA", "HA"),
                        ("CB", "HB"),
                        ("CB", "OG1"),
                        ("OG1", "P"),
                        ("P", "O1P"),
                        ("P", "O2P"),
                        ("P", "OXT"),
                        ("OXT", "HT"),
                        ("CB", "CG2"),
                        ("CG2", "HG21"),
                        ("CG2", "HG22"),
                        ("CG2", "HG23"),
                        ("CA", "C"),
                        ("C", "O"),
                        ("C", "+N"),  # the peptide bond to the next aa
                    ],
                    "optional_delete_bonds": [
                        ("OXT", "C"),
                    ],
                    "CHI2": ["CA", "CB", "OG1", "P"],
                    "CHI3": ["CB", "OG1", "P", "OXT"],
                },
            ),
            "SER": (
                "S",
                {
                    "optional_bonds": [
                        ("-C", "N"),  # the peptide bond to the previous aa
                        ("N", "CA"),
                        ("N", "H"),
                        ("CA", "HA"),
                        ("CB", "HB1"),
                        ("CB", "HB2"),
                        ("CB", "OG"),
                        ("OG", "P"),
                        ("P", "O1P"),
                        ("P", "O2P"),
                        ("P", "OXT"),
                        ("OXT", "HT"),
                        ("CA", "C"),
                        ("C", "O"),
                        ("C", "+N"),  # the peptide bond to the next aa
                    ],
                    "optional_delete_bonds": [
                        ("OXT", "C"),
                    ],
                    "CHI2": ["CA", "CB", "OG", "P"],
                    "CHI3": ["CB", "OG", "P", "OXT"],
                },
            ),
        },
        )
        with open(yaml_file, "w") as f:
            f.write(custom_aas.to_yaml())
        new_custom_aas = CustomTopology.from_yaml(yaml_file)
        # fmt: on
        yaml_file.unlink()
        self.assertEqual(custom_aas, new_custom_aas)

    def test_save_hdf5_ensemble_with_different_top(self):
        # Encodermap imports
        from encodermap.trajinfo import TrajEnsemble

        output_dir = self.data_dir / "pASP_pGLU"
        get_from_url(
            "https://sawade.io/encodermap_data/pASP_pGLU",
            output_dir,
            mk_parentdir=True,
            silence_overwrite_message=True,
        )
        trajs = list(output_dir.glob("*.xtc"))
        tops = list(output_dir.glob("*.pdb"))
        self.assertEqual(len(trajs), len(tops))
        trajs = TrajEnsemble(
            trajs,
            tops,
            common_str=["glu6", "asp6", "asp7", "asp8", "glu7", "glu8", "asp10"],
        )
        trajs.load_CVs("all", ensemble=True)
        with tempfile.NamedTemporaryFile(suffix=".h5") as full_file:
            trajs.save(full_file.name, overwrite=True)
            loaded_trajs = TrajEnsemble.from_dataset(full_file.name)

            # test some MDTraj attributes
            self.assertEqual(trajs.top, loaded_trajs.top)
            self.assertAllEqual(trajs[0][0].xyz, loaded_trajs[0][0].xyz)
            self.assertAllEqual(trajs[0][0].time, loaded_trajs[0][0].time)
            self.assertAllEqual(
                trajs[0][0].unitcell_lengths, loaded_trajs[0][0].unitcell_lengths
            )
            self.assertAllEqual(trajs[1][0].xyz, loaded_trajs[1][0].xyz)
            self.assertAllEqual(trajs[1][0].time, loaded_trajs[1][0].time)
            self.assertAllEqual(
                trajs[1][0].unitcell_lengths, loaded_trajs[1][0].unitcell_lengths
            )

            # test Ensemble slicing
            self.assertEqual(
                trajs[1:14:3].n_frames,
                loaded_trajs[1:14:3].n_frames,
            )
            self.assertEqual(
                trajs[1:14:3][::2].n_frames, loaded_trajs[1:14:3][::2].n_frames
            )
            self.assertEqual(
                trajs[1:14:3][1:4:2].n_frames, loaded_trajs[1:14:3][1:4:2].n_frames
            )
            self.assertAllEqual(
                trajs[1:14:3].CVs["central_dihedrals"],
                loaded_trajs[1:14:3].CVs["central_dihedrals"],
            )
            self.assertAllEqual(
                trajs[1:14:3][::2].CVs["central_cartesians"],
                loaded_trajs[1:14:3][::2].CVs["central_cartesians"],
            )
            self.assertAllEqual(
                trajs[1:14:3][1:4:2].central_distances,
                loaded_trajs[1:14:3][1:4:2].central_distances,
            )

    @expensive_test
    def test_save_and_load_traj_ensemble_to_h5_and_slice(self):
        # Encodermap imports
        from encodermap import load

        output_dir = Path(
            get_from_kondata(
                "linear_dimers",
                mk_parentdir=True,
                silence_overwrite_message=True,
            ),
        )

        trajs = load(
            [output_dir / f"{i:02}.xtc" for i in range(1, 13)],
            [output_dir / f"{i:02}.pdb" for i in range(1, 13)],
        )
        trajs.load_CVs("all")
        trajs_subsample = trajs.subsample(1000)

        self.assertAllEqual(
            trajs[0][1000].xyz,
            trajs_subsample[0].fsel[1000].xyz,
        )

        with tempfile.NamedTemporaryFile(suffix=".h5") as full_file:
            trajs.save(full_file.name, overwrite=True)
            loaded_trajs = TrajEnsemble.from_dataset(full_file.name)
            self.assertAllEqual(
                loaded_trajs[11]._CVs.central_dihedrals.values,
                trajs[11]._CVs.central_dihedrals.values,
            )
            self.assertIsNotNone(loaded_trajs.tf_dataset(256))

            # test some MDTraj attributes
            self.assertEqual(trajs.top, loaded_trajs.top)
            self.assertAllEqual(trajs[0][0].xyz, loaded_trajs[0][0].xyz)
            self.assertAllEqual(trajs[0][0].time, loaded_trajs[0][0].time)
            self.assertAllEqual(
                trajs[0][0].unitcell_lengths, loaded_trajs[0][0].unitcell_lengths
            )
            self.assertAllEqual(trajs[1][0].xyz, loaded_trajs[1][0].xyz)
            self.assertAllEqual(trajs[1][0].time, loaded_trajs[1][0].time)
            self.assertAllEqual(
                trajs[1][0].unitcell_lengths, loaded_trajs[1][0].unitcell_lengths
            )

            # test Ensemble slicing
            self.assertEqual(trajs[1:14:3].n_frames, loaded_trajs[1:14:3].n_frames)
            self.assertEqual(
                trajs[1:14:3][::2].n_frames, loaded_trajs[1:14:3][::2].n_frames
            )
            self.assertEqual(
                trajs[1:14:3][1:4:2].n_frames, loaded_trajs[1:14:3][1:4:2].n_frames
            )
            self.assertAllEqual(
                trajs[1:14:3].CVs["central_dihedrals"],
                loaded_trajs[1:14:3].CVs["central_dihedrals"],
            )
            self.assertAllEqual(
                trajs[1:14:3][::2].CVs["central_cartesians"],
                loaded_trajs[1:14:3][::2].CVs["central_cartesians"],
            )
            self.assertAllEqual(
                trajs[1:14:3][1:4:2].central_distances,
                loaded_trajs[1:14:3][1:4:2].central_distances,
            )

            # Test Single Traj slicing
            self.assertEqual(
                trajs[1][1:14:3].n_frames, loaded_trajs[1][1:14:3].n_frames
            )
            self.assertEqual(
                trajs[1][1:14:3][::2].n_frames, loaded_trajs[1][1:14:3][::2].n_frames
            )
            self.assertEqual(
                trajs[1][1:14:3][1:4:2].n_frames,
                loaded_trajs[1][1:14:3][1:4:2].n_frames,
            )
            self.assertAllEqual(
                trajs[1][1:14:3].CVs["central_dihedrals"],
                loaded_trajs[1][1:14:3].CVs["central_dihedrals"],
            )
            self.assertAllEqual(
                trajs[1][1:14:3][::2].CVs["central_cartesians"],
                loaded_trajs[1][1:14:3][::2].CVs["central_cartesians"],
            )
            self.assertAllEqual(
                trajs[1][1:14:3][1:4:2].central_distances,
                loaded_trajs[1][1:14:3][1:4:2].central_distances,
            )

            # Test subsample slicing
            trajs_loaded_subsample = loaded_trajs.subsample(1000)
            self.assertEqual(trajs_subsample.top, loaded_trajs.top)
            self.assertAllEqual(
                trajs_subsample[0][0].xyz, trajs_loaded_subsample[0][0].xyz
            )
            self.assertAllEqual(
                trajs_subsample[0][0].time, trajs_loaded_subsample[0][0].time
            )
            self.assertAllEqual(
                trajs_subsample[0][0].unitcell_lengths,
                trajs_loaded_subsample[0][0].unitcell_lengths,
            )
            self.assertAllEqual(
                trajs_subsample[1][0].xyz, trajs_loaded_subsample[1][0].xyz
            )
            self.assertAllEqual(
                trajs_subsample[1][0].time, trajs_loaded_subsample[1][0].time
            )
            self.assertAllEqual(
                trajs_subsample[1][0].unitcell_lengths,
                trajs_loaded_subsample[1][0].unitcell_lengths,
            )
            self.assertEqual(
                trajs_subsample[1:14:3].n_frames,
                trajs_loaded_subsample[1:14:3].n_frames,
            )
            self.assertEqual(
                trajs_subsample[1:14:3][::2].n_frames,
                trajs_loaded_subsample[1:14:3][::2].n_frames,
            )
            self.assertEqual(
                trajs_subsample[1:14:3][1:4:2].n_frames,
                trajs_loaded_subsample[1:14:3][1:4:2].n_frames,
            )
            self.assertAllEqual(
                trajs_subsample[1:14:3].CVs["central_dihedrals"],
                trajs_loaded_subsample[1:14:3].CVs["central_dihedrals"],
            )
            self.assertAllEqual(
                trajs_subsample[1:14:3][::2].CVs["central_cartesians"],
                trajs_loaded_subsample[1:14:3][::2].CVs["central_cartesians"],
            )
            self.assertAllEqual(
                trajs_subsample[1:14:3][1:4:2].central_distances,
                trajs_loaded_subsample[1:14:3][1:4:2].central_distances,
            )

            cluster_membership = np.full(
                (trajs_loaded_subsample.n_frames), fill_value=-1
            )
            cluster_membership[:100] = 0
            trajs_loaded_subsample.load_CVs(cluster_membership, "cluster_membership")
            clu = trajs_loaded_subsample.cluster(0, "cluster_membership")
            self.assertIsInstance(clu, TrajEnsemble)

            cluster_membership = np.full((trajs_subsample.n_frames), fill_value=-1)
            cluster_membership[:10] = 0
            trajs_subsample.load_CVs(cluster_membership, "cluster_membership")
            clu = trajs_subsample.cluster(0, "cluster_membership")
            self.assertIsInstance(clu, TrajEnsemble)

            # some more slicing
            self.assertAllEqual(
                np.unique(trajs[1:14:3].id[:, 0]), np.array([1, 4, 7, 10])
            )
            self.assertAllEqual(
                np.unique(loaded_trajs[1:14:3].id[:, 0]), np.array([1, 4, 7, 10])
            )
            self.assertAllEqual(
                trajs[1][1:14:3].id,
                np.array(
                    [
                        [1, 1],
                        [1, 4],
                        [1, 7],
                        [1, 10],
                        [1, 13],
                    ]
                ),
            )
            self.assertAllEqual(
                loaded_trajs[1][1:14:3].id,
                np.array(
                    [
                        [1, 1],
                        [1, 4],
                        [1, 7],
                        [1, 10],
                        [1, 13],
                    ]
                ),
            )
            self.assertAllEqual(
                trajs_subsample[1][1:14:3].id,
                np.array(
                    [
                        [1, 1000],
                        [1, 4000],
                    ]
                ),
            )
            self.assertAllEqual(
                trajs_loaded_subsample[1][1:14:3].id,
                np.array(
                    [
                        [1, 1000],
                        [1, 4000],
                    ]
                ),
            )
            self.assertTrue(
                np.array_equal(
                    trajs[0][1:14:3][1:4:2].id,
                    np.array(
                        [[loaded_trajs[0].traj_num, 4], [loaded_trajs[0].traj_num, 10]]
                    ),
                ),
                msg=f"{trajs.id=}",
            )
            self.assertEqual(
                trajs[1][np.array([1, 4, 6, 8, 12])].n_frames,
                loaded_trajs[1][np.array([1, 4, 6, 8, 12])].n_frames,
            )
            self.assertTrue(
                np.array_equal(
                    trajs[2][np.array([1, 4, 6, 8, 12])][0].id,
                    np.array([[loaded_trajs[2].traj_num, 1]]),
                ),
                msg=(
                    f"The slicing of the trajectory with basename {loaded_trajs[2].basename} and "
                    f"traj_file {loaded_trajs[2].traj_file} did not work. It was expected, that the "
                    f"`.id` attribute of this traj results in an array with shape (1, 2) and "
                    f"the values ({loaded_trajs[2].traj_num}, 1), but returned was {loaded_trajs[1][np.array([1, 4, 6, 8, 12])][0].id}."
                ),
            )
            self.assertTrue(
                np.array_equal(
                    trajs[3][np.array([1, 4, 6, 8, 12])][::2].id,
                    np.array(
                        [
                            [loaded_trajs[3].traj_num, 1],
                            [loaded_trajs[3].traj_num, 6],
                            [loaded_trajs[3].traj_num, 12],
                        ]
                    ),
                )
            )

    def test_traj_ensemble_from_trajs_with_bad_traj_nums_and_load_features(self):
        """If you provide `SingleTraj` objects to the `TrajEnsemble` class with
        bad (duplicated) traj_nums, the `TrajEnsemble` should fix these traj nums
        at instantiation."""

        traj1 = md.load(
            str(self.data_dir / "asp7.xtc"),
            top=str(self.data_dir / "asp7.pdb"),
        )
        traj2 = md.load(
            str(self.data_dir / "glu7.xtc"),
            top=str(self.data_dir / "glu7.pdb"),
        )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            traj1[0].save_pdb(str(td / "asp7.pdb"))
            traj1[:5].save_xtc(str(td / "asp7_1.xtc"))
            traj1[5:20].save_xtc(str(td / "asp7_2.xtc"))
            traj2[0].save_pdb(str(td / "glu7.pdb"))
            traj2[:7].save_xtc(str(td / "glu7_1.xtc"))
            traj2[7:18].save_xtc(str(td / "glu7_2.xtc"))

            straj1 = SingleTraj(td / "asp7_1.xtc", td / "asp7.pdb", traj_num=1)
            straj2 = SingleTraj(td / "asp7_2.xtc", td / "asp7.pdb", traj_num=5)
            straj3 = SingleTraj(td / "glu7_1.xtc", td / "glu7.pdb", traj_num=1)
            straj4 = SingleTraj(td / "glu7_2.xtc", td / "glu7.pdb", traj_num=2)

            with self.assertRaises(Exception):
                trajs = TrajEnsemble(
                    [
                        straj1,
                        straj2,
                        straj3,
                        straj4,
                    ]
                )

            trajs = TrajEnsemble.with_overwrite_trajnums(
                *[
                    straj1,
                    straj2,
                    straj3,
                    straj4,
                ]
            )

            assert np.array_equal(trajs.traj_nums, np.arange(4))

            straj1 = SingleTraj(td / "asp7_1.xtc", td / "asp7.pdb", traj_num=3)
            straj2 = SingleTraj(td / "asp7_2.xtc", td / "asp7.pdb", traj_num=0)
            straj3 = SingleTraj(td / "glu7_1.xtc", td / "glu7.pdb", traj_num=10)
            straj4 = SingleTraj(td / "glu7_2.xtc", td / "glu7.pdb", traj_num=None)

            with self.assertRaises(Exception):
                trajs = TrajEnsemble(
                    [
                        straj1,
                        straj2,
                        straj3,
                        straj4,
                    ]
                )

            trajs = TrajEnsemble.with_overwrite_trajnums(
                *[
                    straj1,
                    straj2,
                    straj3,
                    straj4,
                ]
            )

            trajs.load_CVs("all", ensemble=True)
            print(trajs._CVs)

    def test_SingleTraj_throws_error_on_wrong_way_round(self):
        with self.assertRaises(Exception):
            test = SingleTraj("test.pdb", "test.xtc")

    def test_CVs_stay_after_subsample(self):
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
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
        self.assertEqual(traj.n_frames, 1)
        traj = SingleTraj.from_pdb_id("1GHC")
        self.assertEqual(traj.n_frames, 14)

    def test_traj_CVs_retain_attrs(self):
        # Encodermap imports
        from encodermap import SingleTraj

        traj1 = SingleTraj(self.data_dir / "1YUG.pdb")[:15]
        traj2 = SingleTraj(self.data_dir / "1YUF.pdb")[:15]
        traj1.load_CV(np.ones((15, 3)), attr_name="ones")
        traj2.load_CV(np.ones((15, 3)), attr_name="ones")
        trajs = TrajEnsemble([traj1, traj2])
        self.assertEqual(
            set(trajs._CVs.attrs["full_paths"]),
            set([traj1.traj_file, traj2.traj_file]),
        )
        self.assertEqual(
            set(trajs._CVs.attrs["full_paths"]),
            set([traj1.top_file, traj2.top_file]),
        )
        self.assertNotIn("angle_unit", trajs._CVs.attrs)
        traj1.load_CV(np.random.random((15, 3)), attr_name="angles", deg=True)
        traj2.load_CV(np.random.random((15, 3)), attr_name="angles", deg=True)
        self.assertIn("angle_units", trajs._CVs.attrs)
        self.assertEqual(trajs._CVs.attrs["angle_units"], "deg")

    def test_traj_ensemble_subsample_CVs_stay_consistent(self):
        traj1 = SingleTraj(self.data_dir / "1YUG.pdb")[:10]
        traj2 = SingleTraj(self.data_dir / "1YUF.pdb")[:15]
        trajs = TrajEnsemble([traj1, traj2])
        trajs.load_CVs(np.ones((25, 3)), attr_name="ones")
        subsample = trajs.subsample(2)
        self.assertEqual(
            dict(subsample._CVs.dims), {"traj_num": 2, "frame_num": 8, "ONES": 3}
        )
        self.assertEqual(subsample.CVs["ones"].shape, (13, 3))
        subsample2 = trajs[subsample.index_arr]
        self.assertEqual(
            dict(subsample2._CVs.dims), {"traj_num": 2, "frame_num": 8, "ONES": 3}
        )
        self.assertEqual(subsample2.CVs["ones"].shape, (13, 3))

    def test_traj_ensemble_labels(self):
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            self.data_dir / "1am7_protein.pdb",
        )

        # random phi/psi angles in a [0, 2pi] interval
        random_raman_angles = (
            np.random.random((traj.n_frames, 2 * traj.n_residues)) * 2 * np.pi
        )

        # define labels:
        phi_angles = [f"phi {i}" for i in range(traj.n_residues)]
        psi_angles = [f"psi {i}" for i in range(traj.n_residues)]
        raman_labels = [None] * (len(phi_angles) + len(psi_angles))
        raman_labels[::2] = phi_angles
        raman_labels[1::2] = psi_angles

        # load the CV
        traj.load_CV(random_raman_angles, "raman", labels=raman_labels)
        self.assertTrue(
            np.array_equal(
                traj._CVs.raman.coords["RAMAN"].values, np.array(raman_labels)
            )
        )

    @unittest.mock.patch.dict(os.environ, {"ENCODERMAP_PRINT_PROG_UPDATES": "True"})
    def test_clustering(self):
        # download the M1-Ubq dataset and the K48 dataset
        m1_diUbi_dir = Path(
            get_from_kondata(
                "linear_dimers",
                silence_overwrite_message=True,
                mk_parentdir=True,
            )
        )
        diUbi_dir = Path(
            get_from_kondata(
                "Ub_dimers",
                silence_overwrite_message=True,
                mk_parentdir=True,
            )
        )
        traj1 = SingleTraj(
            self.data_dir / "asp7.xtc",
            top=self.data_dir / "asp7.pdb",
        )
        traj2 = SingleTraj(
            self.data_dir / "glu7.xtc",
            top=self.data_dir / "glu7.pdb",
        )

        # trajs are an ensemble of asp7 and glu7
        trajs = TrajEnsemble([traj1, traj2])

        # cluster_points with only one frame in asp7
        cluster_points = np.full((200,), -1, int)
        cluster_points[np.array([0, 150, 151, 152])] = 0
        trajs.load_CVs(cluster_points, "clu")
        trajs.load_CVs(np.ones((200, 15), int), "ones")
        cluster = trajs.cluster(0, "clu")
        self.assertTrue(np.all(cluster.ones == 1))

        # some integer points as clusters
        trajs.del_CVs()
        cluster_points = np.random.randint(-1, 2, trajs.n_frames)

        # this should not raise a warning so catch it as error
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            trajs.load_CVs(cluster_points, "cluster_membership")

        # the key user selected points is not present, so an exception is raised
        with self.assertRaises(Exception):
            cluster = trajs.cluster(0, "_user_selected_points")

        # create a cluster
        cluster1 = trajs.cluster(0, "cluster_membership", n_points=10)

        # some manual computations to get the cluster frames from trajs
        index = trajs.id[trajs.cluster_membership == 0]
        index = index[
            np.unique(np.round(np.linspace(0, len(index) - 1, 10)).astype(int))
        ]
        frames = trajs[index]
        n_atoms_cluster1 = sum([f[2].n_atoms for f in frames.iterframes()])

        # the n_frames should be equal to n_points, no matter what
        self.assertEqual(
            cluster1.n_frames,
            10,
        )

        # some more trajs
        traj1 = SingleTraj(
            m1_diUbi_dir / "01.xtc",
            top=m1_diUbi_dir / "01.pdb",
        )
        traj2 = SingleTraj(
            diUbi_dir / "GROMOS/K48/traj.xtc",
            top=diUbi_dir / "GROMOS/K48/start.gro",
        )
        traj3 = SingleTraj(
            diUbi_dir / "GROMOS/K63/traj.xtc",
            top=diUbi_dir / "GROMOS/K63/start.gro",
        )

        # create an inhomogeneous ensemble
        trajs = TrajEnsemble([traj1, traj2, traj3])

        # randomly choose integer cluster memberships
        cluster_points = np.random.randint(-1, 2, trajs.n_frames)

        # this should not raise a warning so catch it as error
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            trajs.load_CVs(cluster_points, "cluster_membership")

        # cluster with different n_points
        cluster2 = trajs.cluster(0, "cluster_membership", n_points=12)

        # manual calculations to assert stuff about cluster2
        index = trajs.id[trajs.cluster_membership == 0]
        index = index[
            np.unique(np.round(np.linspace(0, len(index) - 1, 12)).astype(int))
        ]
        frames = trajs[index]
        n_atoms_cluster2 = sum([f[2].n_atoms for f in frames.iterframes()])

        with CapturingStderr() as output:
            cluster1_stacked = cluster1.stack()

        self.assertEqual(
            cluster1_stacked.n_atoms,
            n_atoms_cluster1,
        )
        self.assertTrue(any(["/26" in i for i in output]))
        with CapturingStderr() as output:
            cluster2_stacked = cluster2.stack()
        self.assertTrue(any(["/33" in i for i in output]))
        self.assertEqual(
            cluster2_stacked.n_atoms,
            n_atoms_cluster2,
        )

        # import the interactive plotting
        # Encodermap imports
        from encodermap.plot.interactive_plotting import InteractivePlotting

        trajs.load_CVs(np.random.random((trajs.n_frames, 2)), "lowd")
        trajs.load_CVs(np.random.random((trajs.n_frames, 20)), "highd")
        sess = InteractivePlotting(
            autoencoder=None,
            trajs=trajs,
        )
        sess.selected_point_ids = np.where(cluster_points == 0)[0]
        with Capturing() as output:
            sess.cluster(None)
        self.assertEqual(
            output[1],
            "{'join': {'update_calls': 3, 'total': 3}, '_traj_joined': {'update_calls': 16, 'total': 16}}",
        )

        cluster3 = trajs.cluster(
            0,
            memberships=np.random.randint(low=-1, high=500, size=(trajs.n_frames,)),
            overwrite=True,
            n_points=10,
        )
        self.assertEqual(cluster3.n_frames, 10)

        class Progbar:
            def __init__(self):
                self.total = 0
                self.n = 0

            def update(self, value=1, **kwargs):
                self.n += value

            def reset(self, new_total, **kwargs):
                self.total = new_total

        progbar_joined = Progbar()
        progbar_stacked = Progbar()
        cluster3_joined = cluster3.join(progbar=progbar_joined)
        cluster3_stacked = cluster3.stack(progbar=progbar_stacked)

        self.assertEqual(
            progbar_joined.n,
            progbar_joined.total,
            msg=(
                f"The progress bar provided to `join` did not end on the same "
                f"value as total: {progbar_joined.total=} {progbar_joined.n=}"
            ),
        )

        self.assertEqual(
            progbar_stacked.n,
            progbar_stacked.total,
            msg=(
                f"The progress bar provided to `join` did not end on the same "
                f"value as total: {progbar_joined.total=} {progbar_joined.n=}"
            ),
        )

    def test_clustering_different_atom_counts(self):
        # Encodermap imports
        from encodermap import load
        from encodermap.misc.clustering import cluster_to_dict

        traj1 = SingleTraj(
            self.data_dir / "asp7.xtc",
            self.data_dir / "asp7.pdb",
        )
        traj2 = SingleTraj(
            self.data_dir / "glu7.xtc",
            self.data_dir / "glu7.pdb",
        )
        trajs = TrajEnsemble([traj1, traj2])
        cluster = np.random.random(trajs.n_frames)
        cluster[cluster <= 0.5] = 0
        cluster[cluster > 0.5] = 1
        cluster = cluster.astype(int)
        trajs.load_CVs(cluster, "cluster")
        cluster = trajs.cluster(1, "cluster")
        cluster = cluster_to_dict(cluster)
        self.assertNotIn("joined", cluster)
        self.assertIsInstance(cluster["series"], pd.Series)
        nums, counts = np.unique(
            cluster["series"].index.get_level_values(0), return_counts=True
        )
        self.assertEqual(
            cluster["stacked"].n_atoms,
            sum([trajs[n].n_atoms * c for n, c in zip(nums, counts)]),
        )

        # Encodermap imports
        from encodermap.kondata import get_from_url

        output_dir = self.data_dir / "OTU11"
        get_from_url(
            "https://sawade.io/encodermap_data/OTU11",
            output_dir,
            mk_parentdir=True,
            silence_overwrite_message=True,
        )

        trajs = load(
            [
                output_dir / "OTU11_phospho_threonine.xtc",
                output_dir / "OTU11_phospho_dead.xtc",
            ],
            [
                output_dir / "OTU11_phospho_threonine.pdb",
                output_dir / "OTU11_phospho_dead.pdb",
            ],
        )
        self.assertNotEqual(trajs[0].n_atoms, trajs[1].n_atoms)
        cluster = np.random.random(trajs.n_frames)
        cluster[cluster <= 0.5] = 0
        cluster[cluster > 0.5] = 1
        cluster = cluster.astype(int)
        trajs.load_CVs(cluster, "cluster_membership")
        cluster = trajs.cluster(0, "cluster_membership", n_points=12)
        cluster = cluster_to_dict(cluster)
        self.assertEqual(cluster["ensemble"].n_frames, 12)
        self.assertNotIn("joined", cluster)
        self.assertIsInstance(cluster["series"], pd.Series)
        nums, counts = np.unique(
            cluster["series"].index.get_level_values(0), return_counts=True
        )
        self.assertEqual(
            cluster["stacked"].n_atoms,
            sum([trajs[n].n_atoms * c for n, c in zip(nums, counts)]),
        )

    def test_clustering_centroids_join_and_stack(self):
        """The with_centroids argument to the `TrajEnsemble.cluster()` should
        be implemented and return the centroid with a specified precision (also
        use the more broad centroid finding mechanism with backbones."""
        self.assertHasAttr(TrajEnsemble, "join")
        self.assertHasAttr(TrajEnsemble, "stack")

    def test_load_all_with_deg_and_rad(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".xtc") as f1,
            tempfile.NamedTemporaryFile(suffix=".xtc") as f2,
        ):
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            shutil.copyfile(self.data_dir / "1am7_corrected_part2.xtc", f2.name)
            traj1 = SingleTraj(
                f1.name,
                top=self.data_dir / "1am7_protein.pdb",
            )
            traj2 = SingleTraj(
                f2.name,
                top=self.data_dir / "1am7_protein.pdb",
            )

            # Encodermap imports
            from encodermap.loading.features import CentralDihedrals

            self.assertTrue(hasattr(CentralDihedrals(traj1), "deg"))

            trajs = TrajEnsemble([traj1, traj2])
            trajs.load_CVs("all", deg=True)
            self.assertEqual(trajs._CVs.central_dihedrals.attrs["angle_units"], "deg")
            self.assertTrue(np.any(trajs._CVs.central_dihedrals.values > 10))

            traj1.load_CV(np.ones((traj1.n_frames, 3)), attr_name="ones", deg=True)
            with self.assertRaisesRegex(
                AssertionError, r".*inhomogeneous angle types.*"
            ):
                traj2.load_CV(np.ones((traj2.n_frames, 3)), attr_name="ones", deg=False)

    def test_load_h5(self):
        traj = SingleTraj(self.data_dir / "traj.h5")
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

        self.assertEqual(traj.traj_file, str(self.data_dir / "traj.h5"))
        self.assertEqual(traj._traj_file, self.data_dir / "traj.h5")
        self.assertEqual(traj._top_file, self.data_dir / "traj.h5")
        self.assertEqual(traj.extension, ".h5")
        self.assertEqual(traj.basename, "traj")
        self.assertEqual(traj.index, (None,))
        self.assertIsNone(traj.traj_num)
        traj.load_traj()
        self.assertEqual(traj.backend, "mdtraj")
        # mdtraj equality checks
        self.assertEqual(
            traj.trajectory,
            md.load(str(self.data_dir / "traj.h5")),
        )
        self.assertEqual(
            traj.topology,
            md.load_topology(str(self.data_dir / "traj.h5")),
        )

        # check indexing
        self.assertEqual(traj[::2].n_frames, 50)
        self.assertEqual(traj[:10].n_frames, 10)
        self.assertEqual(traj[[0, 1, 5, 10, 50]].n_frames, 5)

    def test_traj_id(self):
        traj1 = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
        )
        traj2 = SingleTraj(self.data_dir / "1GHC.pdb")
        traj3 = SingleTraj(self.data_dir / "traj.h5")

        for t, length in zip([traj1, traj2, traj3], [51, 14, 100]):
            self.assertTrue(np.array_equal(t.id, np.arange(length)))

        traj1 = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(self.data_dir / "1GHC.pdb", traj_num=2)
        traj3 = SingleTraj(self.data_dir / "traj.h5", traj_num=3)

        for t, length, traj_num in zip([traj1, traj2, traj3], [51, 14, 100], [1, 2, 3]):
            self.assertTrue(
                np.array_equal(
                    t.id, np.vstack([np.full(length, traj_num), np.arange(length)]).T
                )
            )

    def test_slicing_and_CVs_xtc_and_h5(self):
        traj = SingleTraj(
            self.data_dir / "asp7.xtc",
            top=self.data_dir / "asp7.pdb",
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

        traj = SingleTraj(self.data_dir / "asp7.h5")
        self.assertIn("ones", traj.CVs.keys())
        self.assertIn("zeros", traj.CVs.keys())
        test_arr = np.array([0, 2, 4, 6, 8])
        msg = f"Checking {traj=}, with {traj._orig_frames=}, and {test_arr=}"
        self.assertTrue(np.array_equal(traj._orig_frames, test_arr), msg=msg)
        self.assertEqual(traj.index, (None,))
        traj = traj[::2]
        self.assertEqual(traj.CVs["ones"].shape, (3, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (3, 5, 3))

        traj = SingleTraj(self.data_dir / "asp7.h5")
        self.assertEqual(traj.xyz.shape, (5, 73, 3))
        self.assertEqual(traj.CVs["ones"].shape, (5, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (5, 5, 3))

        traj = SingleTraj(self.data_dir / "asp7.h5", index=([0, 1],))
        self.assertEqual(traj.xyz.shape, (2, 73, 3))
        self.assertEqual(traj.CVs["ones"].shape, (2, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (2, 5, 3))

        traj = SingleTraj(self.data_dir / "asp7.h5", index=slice(None, None, 2))
        self.assertEqual(traj.index, (slice(None, None, 2),))
        self.assertEqual(traj.CVs["ones"].shape, (3, 5))
        self.assertEqual(traj.CVs["zeros"].shape, (3, 5, 3))

        traj = SingleTraj(self.data_dir / "traj.h5")
        self.assertEqual(traj.CVs, {})

    def test_double_slicing(self):
        traj1 = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(self.data_dir / "1GHC.pdb", traj_num=2)
        traj3 = SingleTraj(self.data_dir / "traj.h5", traj_num=3)

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
            str(self.data_dir / "1am7_corrected.xtc"),
            top=str(self.data_dir / "1am7_protein.pdb"),
        )
        traj2_md = md.load_pdb(str(self.data_dir / "1GHC.pdb"))
        traj3_md = md.load(str(self.data_dir / "traj.h5"))
        traj1 = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(self.data_dir / "1GHC.pdb", traj_num=2)
        traj3 = SingleTraj(self.data_dir / "traj.h5", traj_num=3)

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
        with (
            tempfile.NamedTemporaryFile(suffix=".xtc") as f1,
            tempfile.NamedTemporaryFile(suffix=".xtc") as f2,
        ):
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            shutil.copyfile(self.data_dir / "1am7_corrected_part2.xtc", f2.name)
            traj1 = SingleTraj(
                f1.name,
                top=self.data_dir / "1am7_protein.pdb",
            )
            traj2 = SingleTraj(
                f2.name,
                top=self.data_dir / "1am7_protein.pdb",
            )
            trajs = TrajEnsemble([traj1, traj2])
            trajs.load_CVs("all")
            trajs.load_CVs(
                np.random.randint(0, 10, trajs.n_frames), "cluster_membership"
            )
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
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
            traj_num=1,
        )
        traj2 = SingleTraj(self.data_dir / "1GHC.pdb", traj_num=2)
        traj3 = SingleTraj(self.data_dir / "traj.h5", traj_num=3)

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
            self.assertEqual(
                (
                    1,
                    frame.n_atoms,
                ),
                frame.CVs["y_coordinate"].shape,
            )
            frames = t[np.array([0, 2, 4])]
            self.assertEqual(frames.CVs["y_coordinate"].shape, (3, frame.n_atoms))
            frame = frames[0]
            self.assertTrue(
                np.array_equal(
                    frame.CVs["y_coordinate"][0, 0], frames.CVs["y_coordinate"][0, 0]
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
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
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
        com = np.load(self.data_dir / "1am7_center_of_mass.npy")
        self.assertTrue(np.allclose(md.compute_center_of_mass(traj.traj)[0], com))

    def test_single_raj_subsample(self):
        traj1 = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
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

    def test_single_traj_subsample_without_loading(self):
        traj1 = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
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
        traj1 = SingleTraj(self.data_dir / "traj.h5")
        traj1.load_traj()
        subsample = traj1[::3].traj
        self.assertEqual(subsample.n_frames, 34)
        subsample = traj1[[0, 1, 5, 6]].traj
        self.assertEqual(subsample.n_frames, 4)
        subsample = traj1[5:46:3].traj
        self.assertEqual(subsample.n_frames, 14)

    def test_addition_along_TrajEnsemble(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".xtc") as f1,
            tempfile.NamedTemporaryFile(suffix=".xtc") as f2,
        ):
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            shutil.copyfile(self.data_dir / "1am7_corrected_part2.xtc", f2.name)
            traj1 = SingleTraj(
                f1.name,
                top=self.data_dir / "1am7_protein.pdb",
                traj_num=1,
            )
            traj2 = SingleTraj(
                f2.name,
                top=self.data_dir / "1am7_protein.pdb",
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
                    f1.name,
                    f2.name,
                ],
            )
            trajs.load_trajs()
            self.assertEqual(trajs.top_files, [str(self.data_dir / "1am7_protein.pdb")])
            self.assertEqual(trajs.y_coordinate.shape, (51, 2504))

    def test_gen_ensemble(self):
        traj = SingleTraj(
            self.data_dir / "1am7_corrected.xtc",
            top=self.data_dir / "1am7_protein.pdb",
        )
        trajs = traj._gen_ensemble()
        self.assertIsInstance(trajs, TrajEnsemble)
        self.assertEqual(trajs.n_trajs, 1)
        self.assertEqual(trajs.n_frames, 51)
        self.assertEqual(trajs.xyz.shape, (51, 2504, 3))

    def test_atom_slice(self):
        traj = SingleTraj(self.data_dir / "1UBQ.pdb", traj_num=2)
        self.assertGreater(traj.n_atoms, 76)
        old_atoms = [a.name for a in traj.top.atoms]
        self.assertEqual(traj.n_chains, 2)
        # printing chains does not require load
        self.assertEqual(traj.backend, "no_load")
        self.assertEqual(traj.basename, "1UBQ")
        self.assertEqual(traj.extension, ".pdb")
        self.assertEqual(traj.n_atoms, 660)
        traj.load_CV(traj.xyz[:, :, 1], "y_coordinate")
        self.assertEqual(
            76,
            len(traj.top.select("name CA")),
            msg=(
                f"Did not select the correct number of atoms: {traj.top.select('name CA')=}"
            ),
        )
        with self.assertWarns(UserWarning):
            traj.atom_slice(traj.top.select("name CA"))
        self.assertEqual(
            76,
            len(traj.top.select("name CA")),
            msg=(
                f"Did not select the correct number of atoms: "
                f"{traj.top.select('name CA')=} {[a for a in traj.top.atoms]=} {traj.n_atoms=}"
            ),
        )
        traj.atom_slice(traj.top.select("name CA"))
        self.assertEqual(
            traj.n_atoms,
            76,
            msg=(
                f"{traj.traj=} {traj.top=} {traj._atom_indices=} "
                f"{old_atoms=} {old_atoms.count('CA')=}"
            ),
        )

    def test_stack(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".xtc") as f1,
            tempfile.NamedTemporaryFile(suffix=".xtc") as f2,
        ):
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            shutil.copyfile(self.data_dir / "1am7_corrected_part2.xtc", f2.name)
            traj1 = SingleTraj(
                f1.name,
                top=self.data_dir / "1am7_protein.pdb",
                traj_num=1,
            )
            traj2 = SingleTraj(
                f2.name,
                top=self.data_dir / "1am7_protein.pdb",
                traj_num=2,
            )
            with self.assertRaises(ValueError):
                traj1.stack(traj2)
            new = traj1.stack(traj2[:25])
            self.assertEqual(new.n_atoms, 5008)
            self.assertEqual(new.n_residues, 316)
            self.assertEqual(new.n_frames, 25)

    def test_join(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".xtc") as f1,
            tempfile.NamedTemporaryFile(suffix=".xtc") as f2,
        ):
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            shutil.copyfile(self.data_dir / "1am7_corrected_part2.xtc", f2.name)
            traj1 = SingleTraj(
                f1.name,
                top=self.data_dir / "1am7_protein.pdb",
                traj_num=1,
            )
            traj2 = SingleTraj(
                f2.name,
                top=self.data_dir / "1am7_protein.pdb",
                traj_num=2,
            )
            new = traj1.join(traj2)
            self.assertEqual(new.n_atoms, 2504)
            self.assertEqual(new.n_residues, 158)
            self.assertEqual(new.n_frames, 51)

    def test_wrong_formatted_CVs(self):
        traj = SingleTraj(self.data_dir / "1YUF.pdb")
        test = np.append(traj.xyz[:, 0], [5])
        with self.assertRaises(Exception):
            traj.load_CV(test, "test")

    def test_batch_iterator_correctly_stacks_sparse_cartesians(self):
        """If cartesians contain NaNs they get stacked to a rank 2 array/tensor,
        before they are made to a sparse Tensor. It is important to make the stacking
        and unstacking of the cartesians reversible.

        During programming of the `TrajEnsemble.batch_iterator()` method, I got this
        result:

        Exception: [k.shape for k in out]=[(10, 493), (10, 492), (10, 495, 3), (10, 494), (10, 408)]
            out[2][:2, :5, -1]=array([[2.9060001, 3.045    , 3.1290002, 3.246    , 3.3400002],
                   [8.176001 , 8.211    , 8.229    , 8.185    , 8.192    ]],
                  dtype=float32)
            test[:, :5, -1]=array([[2.9060001, 3.045    , 3.1290002, 3.246    , 3.3400002],
                   [8.176001 , 8.211    , 8.229    , 8.185    , 8.192    ]],
                  dtype=float32)

        """
        with (
            tempfile.NamedTemporaryFile(suffix=".xtc") as f1,
            tempfile.NamedTemporaryFile(suffix=".xtc") as f2,
        ):
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            shutil.copyfile(self.data_dir / "1am7_corrected_part2.xtc", f2.name)
            trajs = TrajEnsemble(
                [
                    f1.name,
                    f2.name,
                ],
                tops=[
                    self.data_dir / "1am7_protein.pdb",
                    self.data_dir / "1am7_protein.pdb",
                ],
            )
            trajs.load_CVs("all")
            self.assertEqual(
                trajs.traj_files,
                [
                    f1.name,
                    f2.name,
                ],
            )
            for index, batch in trajs.batch_iterator(10, yield_index=True):
                break

            self.assertIsInstance(index, np.ndarray)
            for i, type_ in enumerate(
                [
                    "central_angles",
                    "central_dihedrals",
                    "central_cartesians",
                    "central_distances",
                    "side_dihedrals",
                ]
            ):
                test = np.stack(
                    [
                        trajs._CVs[type_]
                        .sel(
                            traj_num=index[0, 0],
                            frame_num=index[0, 1],
                        )
                        .values,
                        trajs._CVs[type_]
                        .sel(
                            traj_num=index[1, 0],
                            frame_num=index[1, 1],
                        )
                        .values,
                    ],
                )
                self.assertEqual(
                    test.shape,
                    batch[i][:2].shape,
                    msg=(
                        f"Comparing shapes of arrays for {type_}. "
                        f"Shapes are not identical: {test.shape=} {batch[2].shape=}"
                    ),
                )
                self.assertAllEqual(test, batch[i][:2])

            # Encodermap imports
            from encodermap.autoencoder.autoencoder import np_to_sparse_tensor
            from encodermap.models.models import (
                _create_inputs_non_periodic_maybe_sparse,
            )
            from encodermap.parameters.parameters import ADCParameters

            test = np.stack(
                [
                    trajs._CVs.central_cartesians.sel(
                        traj_num=index[0, 0],
                        frame_num=index[0, 1],
                    ).values,
                    trajs._CVs.central_cartesians.sel(
                        traj_num=index[1, 0],
                        frame_num=index[1, 1],
                    ).values,
                ],
            )
            test_copy = test.copy()
            test_copy[1, [0, 1, 2, 470, 471, 472, 473]] = np.nan
            test_tensor = np_to_sparse_tensor(test_copy.reshape((2, -1)))

            input, output, _ = _create_inputs_non_periodic_maybe_sparse(
                shape=(474 * 3,),
                p=ADCParameters(),
                name="testing_reshaping_central_cartesians",
                sparse=True,
                reshape=3,
            )
            model = tf.keras.models.Model(inputs=input, outputs=output)
            test_tensor = model(test_tensor).numpy()
            self.assertTrue(not np.any(np.isnan(test_tensor)))
            self.assertAllEqual(
                test_copy,
                tf.keras.layers.Reshape(
                    target_shape=(474, 3),
                    input_shape=(474 * 3,),
                    name="reshape_sparse_to_dense_test",
                )(test_copy.reshape((2, -1))).numpy(),
            )

    def test_info_all_loading(self):
        with self.assertRaises(Exception):
            trajs = TrajEnsemble(
                [
                    self.data_dir / "1am7_corrected_part1.xtc",
                    self.data_dir / "1am7_corrected_part2.xtc",
                ],
                tops=[
                    self.data_dir / "1am7_protein.pdb",
                    self.data_dir / "1am7_protein1.pdb",
                    self.data_dir / "1am7_protein2.pdb",
                ],
            )
        trajs = TrajEnsemble(
            [
                self.data_dir / "1YUG.pdb",
                self.data_dir / "1YUF.pdb",
            ]
        )
        self.assertEqual(trajs.n_frames, 31)

    def test_pyemma_indexing_and_get_single_frame(self):
        trajs = TrajEnsemble(
            [
                self.data_dir / "1YUG.pdb",
                self.data_dir / "1YUF.pdb",
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
        with (
            tempfile.NamedTemporaryFile(suffix=".xtc") as f1,
            tempfile.NamedTemporaryFile(suffix=".xtc") as f2,
        ):
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            shutil.copyfile(self.data_dir / "1am7_corrected_part2.xtc", f2.name)
            traj1 = SingleTraj(
                f1.name,
                top=self.data_dir / "1am7_protein.pdb",
            )
            traj2 = SingleTraj(
                f2.name,
                top=self.data_dir / "1am7_protein.pdb",
            )
            trajs = TrajEnsemble([traj1.traj, traj2.traj])
            self.assertIsInstance(trajs, TrajEnsemble)
            self.assertEqual(
                trajs.top[0],
                md.load_topology(str(self.data_dir / "1am7_protein.pdb")),
                msg=f"The tops of traj does seem to be an empty list {trajs.top}.",
            )
            self.assertEqual(trajs.n_residues, [158, 158])
            self.assertEqual(trajs.basenames, [None, None])
            split_into_frames = trajs.split_into_frames()
            self.assertEqual(split_into_frames.n_frames, 51)
            test = split_into_frames.traj_joined
            self.assertIsInstance(test, md.Trajectory)

    def test_adding_mixed_pyemma_features_with_custom_names(self):
        with tempfile.NamedTemporaryFile(suffix=".xtc") as f1:
            shutil.copyfile(self.data_dir / "1am7_corrected_part1.xtc", f1.name)
            traj = SingleTraj(
                f1.name,
                top=self.data_dir / "1am7_protein.pdb",
            )
            # Encodermap imports
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

            self.assertEqual(traj.Custom_Feature_1.shape, (25, 12_403))
            self.assertEqual(traj.Custom_Feature_2.shape, (25, 12_403))
            self.assertEqual(
                traj.Custom_Feature_3.shape,
                (25, 471),
                msg=f"{traj._CVs.Custom_Feature_3.coords['CENTRAL_DIHEDRALS'].values.tolist()}",
            )

    def test_info_all_load_CVs_from_file(self):
        traj1 = SingleTraj(self.data_dir / "1YUG.pdb")[:15]
        traj2 = SingleTraj(self.data_dir / "1YUF.pdb")[:15]
        trajs = TrajEnsemble([traj1, traj2])
        with self.assertRaises(Exception):
            trajs.load_CVs(
                [
                    self.data_dir / "/1NOT_text.txt",
                    self.data_dir / "/1YUG_text.txt",
                ]
            )
        trajs.load_CVs(
            [
                self.data_dir / "1YUF_numpy.npy",
                self.data_dir / "1YUG_numpy.npy",
            ],
            "y_coordinate_1",
        )
        self.assertEqual(trajs.y_coordinate_1.shape, (30, 720))
        trajs.load_CVs(
            [
                self.data_dir / "1YUF_text.txt",
                self.data_dir / "1YUG_text.txt",
            ],
            "y_coordinate_2",
        )
        self.assertEqual(trajs.y_coordinate_2.shape, (30, 720))
        trajs.load_CVs(
            [
                np.load(self.data_dir / "1YUG_numpy.npy"),
                np.load(self.data_dir / "1YUF_numpy.npy"),
            ],
            "y_coordinate_3",
        )
        self.assertEqual(trajs.y_coordinate_3.shape, (30, 720))
        trajs.load_CVs(self.data_dir / "", "y_coordinate_4")
        self.assertEqual(trajs.y_coordinate_4.shape, (30, 720))

        traj1 = SingleTraj(self.data_dir / "1YUG.pdb")[:15]
        traj2 = SingleTraj(self.data_dir / "1YUF.pdb")[:15]
        trajs = TrajEnsemble([traj1, traj2])
        with self.assertRaises(Exception):
            trajs.load_CVs(
                [
                    self.data_dir / "/1NOT_text.txt",
                    self.data_dir / "/1YUG_text.txt",
                ]
            )
        trajs.load_CVs(
            [
                self.data_dir / "1YUF_numpy.npy",
                self.data_dir / "1YUG_numpy.npy",
            ],
            "y_coordinate_1",
        )
        self.assertEqual(trajs.y_coordinate_1.shape, (30, 720))
        trajs.load_CVs(
            [
                self.data_dir / "1YUF_text.txt",
                self.data_dir / "1YUG_text.txt",
            ],
            "y_coordinate_2",
        )
        self.assertEqual(trajs.y_coordinate_2.shape, (30, 720))
        trajs.load_CVs(
            [
                np.load(self.data_dir / "1YUG_numpy.npy"),
                np.load(self.data_dir / "1YUF_numpy.npy"),
            ],
            "y_coordinate_3",
        )
        self.assertEqual(trajs.y_coordinate_3.shape, (30, 720))
        trajs.load_CVs(self.data_dir / "", "y_coordinate_4")
        self.assertEqual(trajs.y_coordinate_4.shape, (30, 720))

    def test_info_all_load_CVs_from_numpy(self):
        traj1 = SingleTraj(self.data_dir / "1YUG.pdb")
        traj2 = SingleTraj(self.data_dir / "1YUF.pdb")
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

    def test_traj_ensemble_from_multiple_pdbs(self):
        trajs = TrajEnsemble(
            [
                self.data_dir / "1YUG.pdb",
                self.data_dir / "1YUF.pdb",
            ]
        )
        self.assertEqual(
            trajs._top_files,
            [
                str(self.data_dir / "1YUG.pdb"),
                str(self.data_dir / "1YUF.pdb"),
            ],
        )

    def test_traj_ensemble_from_textfile(self):
        test_str = """\
        {{ traj1 }} {{ top1 }} 0 asp7
        {{ traj2 }} {{ top2 }} 1 glu7

        """
        # Standard Library Imports
        import textwrap

        # Third Party Imports
        import jinja2

        template = jinja2.Template(textwrap.dedent(test_str))
        test_str = template.render(
            {
                "traj1": self.data_dir / "asp7.xtc",
                "top1": self.data_dir / "asp7.pdb",
                "traj2": self.data_dir / "glu7.xtc",
                "top2": self.data_dir / "glu7.pdb",
            }
        )

        tmp_file = Path("/tmp/tmp_textfile.txt")
        tmp_file.write_text(test_str)

        trajs = TrajEnsemble.from_textfile(tmp_file)

        tmp_file.unlink()

        self.assertEqual(trajs.common_str, ["asp7", "glu7"])
        self.assertEqual(trajs.n_trajs, 2)
        self.assertEqual(trajs.n_frames, 200)
        self.assertEqual(trajs.basenames, trajs.common_str)


################################################################################
# Create and filter suite
################################################################################


testSuite = unittest.TestSuite()
testSuite.addTests((unittest.makeSuite(TestTraj),))


################################################################################
# Main
################################################################################


if __name__ == "__main__":
    unittest.main()

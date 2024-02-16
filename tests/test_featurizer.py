# -*- coding: utf-8 -*-
# tests/test_featurizer.py
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
    * TestFeatures: Uses mock to make MDTraj and thus EncoderMap load atomic
        positions from an artificial trajectory. This artificial trajectory
        has angles that are easy to understand. E.g. The atoms are at positions:
            (0, 0, 0)
            (0, 0, 1)
            (0, 0, 2)
            (0, 1, 2)
        which results in bond-lengths of 1, angles of 90 deg and a dihedral
        angle of 180 deg.
    * TestDaskFeatures: Copy of the `TestFeatures` class, but this time using
        the dask featurizer as base class. Doing so, we can copy all the tests
        from `TestFeatures` to `testDaskFeatures`
    * TestSpecialDaskFeatures: Test the performance of the distributed featurization.

"""
################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import tempfile
import unittest
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union
from unittest.mock import patch

# Third Party Imports
import MDAnalysis as mda
import mdtraj as md
import numpy as np
import tensorflow as tf
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.distances import dist as mda_dist
from MDAnalysis.transformations import unwrap
from numpy.testing import assert_array_equal

# Encodermap imports
from encodermap.loading.dask_featurizer import DaskFeaturizer
from encodermap.loading.featurizer import Featurizer, pairs


import encodermap as em  # isort: skip


try:
    # Third Party Imports
    from numpy.core.numeric import _no_nep50_warning
except ImportError:

    class _no_nep50_warning:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


# Third Party Imports
from numpy.testing import assert_allclose, assert_array_compare

# Encodermap imports
from encodermap import SingleTraj


warnings.filterwarnings("ignore", category=DeprecationWarning)


################################################################################
# Mocks
################################################################################


def add_B1_and_B2_as_ca(*args, **kwargs):
    self = args[0]
    sel = self.traj.top.select("name B1 or name B2")
    pairs = self.pairs(sel, 0)
    self.add_distances(pairs, periodic=True)


def add_B1_and_B4_as_ca(*args, **kwargs):
    self = args[0]
    if hasattr(self, "traj"):
        sel = self.traj.top.select("name B1 or name B4")
    else:
        sel = self.trajs[0].top.select("name B1 or name B4")
    pairs_ = pairs(sel, 0)
    self.add_distances(pairs_, periodic=True)


################################################################################
# Utils
################################################################################


def format_msg(out1, out2):
    msg = (
        f"The two arrays `out1` and `out2` are created using "
        f"the Featurizer class. For `out1`, a single trajectory "
        f"is provided, for `out2` a `TrajEnsemble` class with multiple "
        f"trajs is provided. These multiple trajs were obtained by "
        f"splitting the traj for `out1` into two. Thus, they should "
        f"have the same results. Here are the results:"
        f"Shapes: out1: {out1.shape}, out2: {out2.shape}."
        f"First five entries: out1: {out1[:5]}, out2: {out2[:5]}"
    )
    return msg


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def assert_allclose_periodic(
    actual: Union[np.ndarray, Iterable, int, float],
    desired: Union[np.ndarray, Iterable, int, float],
    rtol: Optional[float] = 1e-7,
    atol: Optional[float] = 0.0,
    equal_nan: Optional[bool] = True,
    err_msg: Optional[str] = "",
    verbose: Optional[bool] = True,
    periodicity: float = 2 * np.pi,
    max_percentage_mismatched: float = 0.0,
) -> None:
    actual, desired = np.asanyarray(actual), np.asanyarray(desired)
    header = f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}"
    max_number_mismatched = int(max_percentage_mismatched * actual.size)

    def compare(
        x,
        y,
    ):
        with np.errstate(invalid="ignore"), _no_nep50_warning():
            d = np.abs(x - y)
            d = np.minimum(d, periodicity - d)
            tol = atol + rtol * periodicity * np.abs(y)
            le = np.less_equal(d, tol) & (d != 0)
            mismatched = np.where((~np.less_equal(d, tol) & (d != 0)))[0]
            if mismatched.size > max_number_mismatched:
                with np.printoptions(suppress=True):
                    print(
                        f"There are {len(mismatched)} elements in a periodic space of {periodicity:.4f} "
                        f"that exhibit differences, that are out of tolerance. "
                        f"In the flattened `actual` and desired` arrays, these "
                        f"5 example elements exhibit a"
                        f"difference over the tolerance: {mismatched[:5]}\n"
                        f"The values of 5 example elements of these arrays are:\n\n"
                        f"Actual:\n{x[mismatched][:5]}\n\nDesired:\n{y[mismatched][:5]}\n\n"
                        f"Their difference in periodic space is:\n"
                        f"{d[mismatched][:5]}\n\nwhich is above the tolerance at "
                        f"these respective positions:\n{tol[mismatched][:5]}\n\n"
                        f"You can either adjust the parameters `atol`, `rtol`, or increase "
                        f"the `max_percentage_mismatched` which can be useful for periodic distances."
                    )
            elif mismatched.size == 0:
                return True
            else:
                print(
                    f"There are {len(mismatched)} outlier elements in a periodic "
                    f"space of {periodicity:.4f}. Because of the chosen "
                    f"`max_percentage_mismatched`={max_percentage_mismatched}, "
                    f"which for this set of {x.size} samples equates to "
                    f"{max_number_mismatched} I will accept this test."
                )
                return True
            return le

    diff = np.abs(desired - actual)
    diff = np.minimum(diff, periodicity - diff)

    err_msg += (
        f"Max absolute difference (periodic): {np.max(diff):.6f}\n"
        f"Max relative difference (periodic): {np.max(diff) / np.max(desired):.6f}\n"
    )

    assert_array_compare(
        compare,
        actual,
        desired,
        err_msg=str(err_msg),
        verbose=verbose,
        header=header,
        equal_nan=equal_nan,
    )


################################################################################
# Test suites
################################################################################


class TestSpecialDaskFeatures(unittest.TestCase):
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

    def assertAllClose(
        self,
        actual: Union[np.ndarray, Iterable, int, float],
        desired: Union[np.ndarray, Iterable, int, float],
        rtol: Optional[float] = 1e-7,
        atol: Optional[float] = 0,
        equal_nan: Optional[bool] = True,
        err_msg: Optional[str] = "",
        verbose: Optional[bool] = True,
    ) -> None:
        try:
            assert_allclose(
                actual=actual,
                desired=desired,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                err_msg=err_msg,
                verbose=verbose,
            )
        except AssertionError as e:
            self.fail(str(e))

    def assertAllClosePeriodic(
        self,
        actual: np.ndarray,
        desired: np.ndarray,
        rtol: float = 1e-7,
        atol: float = 0.0,
        equal_nan: bool = True,
        err_msg: str = "",
        verbose: bool = True,
        periodicity: float = 2 * np.pi,
    ) -> None:
        try:
            assert_allclose_periodic(
                actual=actual,
                desired=desired,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                err_msg=err_msg,
                verbose=verbose,
                periodicity=periodicity,
            )
        except AssertionError as e:
            self.fail(str(e))

    def test_dask_visualization(self):
        traj = em.load(self.PFFP_xtc_file, self.PFFP_gro_file)
        feat = em.Featurizer(traj, in_memory=False)
        feat.add_all()
        feat.get_output(make_trace=True)
        self.assertTrue(False)

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

    def test_in_memory_and_dask_featurizer_return_similar_data(self):
        # Third Party Imports
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

    def test_periodic_distances_angles_dihedrals(self):
        """This test uses a trajectory with many small peptides in it. That way,
        we can be sure, that some pbc breakings occur. MDTRaj and MDAnalysis should
        be able to resolve them.
        """
        print("TESTPERIODIC start")
        PFFP_gro = Path(__file__).resolve().parent / "data/PFFP_MD_fin_protonly.gro"
        PFFP_tpr = Path(__file__).resolve().parent / "data/PFFP_MD_fin_protonly.tpr"
        PFFP_xtc = (
            Path(__file__).resolve().parent / "data/PFFP_MD_fin_protonly_dt_100.xtc"
        )
        PFFP_fixed_xtc = (
            Path(__file__).resolve().parent
            / "data/PFFP_MD_fin_protonly_dt_100_fixed_pbc.xtc"
        )

        print("TESTPERIODIC creating universe")
        u1 = mda.Universe(PFFP_tpr, PFFP_xtc)
        l_atoms = []
        r_atoms = []
        print("TESTPERIODIC created universe")
        for chunk in chunker(u1.residues, 4):
            for l_a, r_a in zip(
                chunk.atoms.select_atoms("backbone")[:-1],
                chunk.atoms.select_atoms("backbone")[1:],
            ):
                l_atoms.append(l_a.index)
                r_atoms.append(r_a.index)

        # assert 0-based indexing in mda atoms:
        self.assertEqual(u1.atoms[0].index, 0)

        l_ag = mda.AtomGroup(l_atoms, u1)
        r_ag = mda.AtomGroup(r_atoms, u1)
        print("TESTPERIODIC Distance coordinates extracted.")

        # mdanalysis without unwrap
        mda_dists_no_box = (
            AnalysisFromFunction(mda_dist, l_ag, r_ag)
            .run()
            .results["timeseries"][:, -1]
        )
        mda_dists_box = []
        for ts in u1.trajectory:
            _, __, d = mda_dist(l_ag, r_ag, box=ts.dimensions)
            mda_dists_box.append(d)
        mda_dists_box = np.vstack(mda_dists_box)

        print("TESTPERIODIC MDAnalysis without unwrap finished.")
        try:
            assert_allclose(mda_dists_no_box, mda_dists_box)
        except AssertionError as e:
            lines = str(e).splitlines()
            max_diff = float(lines[4].split()[-1])
            if max_diff <= 50:
                self.fail(
                    f"The difference between box/no box is too small: {max_diff}\n\n{e}"
                )

        # mdanalysis with unwrap
        u_fixed = mda.Universe(PFFP_tpr, PFFP_fixed_xtc)
        u2 = mda.Universe(PFFP_tpr, PFFP_xtc)
        ag2 = u2.atoms
        ag_fixed = u_fixed.atoms
        transform = unwrap(ag2)
        u2.trajectory.add_transformations(transform)
        l_ag2 = mda.AtomGroup(l_atoms, u2)
        r_ag2 = mda.AtomGroup(r_atoms, u2)
        l_ag_fixed = mda.AtomGroup(l_atoms, u_fixed)
        r_ag_fixed = mda.AtomGroup(r_atoms, u_fixed)
        mda_dists_transform = []
        dists_fixed = []
        for ts2, ts_fixed in zip(u2.trajectory, u_fixed.trajectory):
            _, __, d = mda_dist(l_ag2, r_ag2)
            mda_dists_transform.append(d)
            _, __, d = mda_dist(l_ag_fixed, r_ag_fixed)
            dists_fixed.append(d)
        mda_dists_transform = np.vstack(mda_dists_transform)
        dists_fixed = np.vstack(dists_fixed)

        print("TESTPERIODIC MDAnalysis with unwrap finished.")
        # for such short distances dist_periodic and dist_unwrap should be the same.
        self.assertAllClosePeriodic(mda_dists_transform, mda_dists_box, atol=0.1)
        self.assertAllClosePeriodic(mda_dists_transform, dists_fixed, atol=0.01)

        # compare with mdtrajs
        traj_w_pbc = md.load(str(PFFP_xtc), top=str(PFFP_gro))
        traj_no_pbc = md.load(str(PFFP_fixed_xtc), top=str(PFFP_gro))

        atom_pairs = np.vstack([l_atoms, r_atoms]).T
        md_dists_not_periodic = md.compute_distances(
            traj_w_pbc, atom_pairs, periodic=False
        )
        md_dists_periodic = md.compute_distances(traj_w_pbc, atom_pairs, periodic=True)
        md_dists_fixed = md.compute_distances(traj_no_pbc, atom_pairs, periodic=False)
        print("TESTPERIODIC MDTraj distances finished.")

        self.assertAllClosePeriodic(mda_dists_box / 10, md_dists_periodic, atol=0.01)
        self.assertAllClosePeriodic(
            mda_dists_no_box / 10, md_dists_not_periodic, atol=0.01
        )
        self.assertAllClosePeriodic(mda_dists_transform / 10, md_dists_fixed, atol=0.01)

        # now encodermap
        em_traj_w_pbc = em.SingleTraj(PFFP_xtc, PFFP_gro)
        em_traj_no_pbc = em.SingleTraj(PFFP_fixed_xtc, PFFP_gro)

        feat1 = em.Featurizer(em_traj_w_pbc)
        feat1.add_distances(atom_pairs, periodic=False)
        feat2 = em.Featurizer(em_traj_w_pbc)
        feat2.add_distances(atom_pairs, periodic=True)
        em_dists_not_periodic = feat1.get_output().DistanceFeature.values.squeeze()
        em_dists_periodic = feat2.get_output().DistanceFeature.values.squeeze()

        self.assertAllClosePeriodic(mda_dists_box / 10, em_dists_periodic, atol=0.01)
        self.assertAllClosePeriodic(
            mda_dists_no_box / 10, em_dists_not_periodic, atol=0.01
        )

        # with dask
        dask_feat1 = em.Featurizer(em_traj_w_pbc, in_memory=False)
        dask_feat1.add_distances(atom_pairs, periodic=False)
        dask_feat2 = em.Featurizer(em_traj_w_pbc, in_memory=False)
        dask_feat2.add_distances(atom_pairs, periodic=True)
        dask_em_dists_not_periodic = feat1.get_output().DistanceFeature.values.squeeze()
        dask_em_dists_periodic = feat2.get_output().DistanceFeature.values.squeeze()

        self.assertAllClosePeriodic(
            mda_dists_box / 10, dask_em_dists_periodic, atol=0.01
        )
        self.assertAllClose(
            mda_dists_no_box / 10, dask_em_dists_not_periodic, atol=0.01
        )

        # do the same with dihedrals
        dihedral_atoms = []
        for residues in chunker(list(em_traj_w_pbc.top.residues), 4):
            atoms = [a.index for r in residues for a in r.atoms if a.name == "CA"]
            assert len(atoms) == 4
            dihedral_atoms.append(atoms)
        dihedral_atoms = np.vstack(dihedral_atoms)
        self.assertEqual(dihedral_atoms.shape[1], 4)

        md_dih_not_periodic = md.compute_dihedrals(
            traj_w_pbc, dihedral_atoms, periodic=False
        )
        md_dih_periodic = md.compute_dihedrals(
            traj_w_pbc, dihedral_atoms, periodic=True
        )

        try:
            assert_allclose_periodic(md_dih_not_periodic, md_dih_periodic)
        except AssertionError as e:
            pass
        else:
            self.fail(
                "Periodic and non-periodc calculations are too close to each " "other"
            )

        feat1 = em.Featurizer(em_traj_w_pbc)
        feat1.add_dihedrals(dihedral_atoms, periodic=False)
        feat2 = em.Featurizer(em_traj_w_pbc)
        feat2.add_dihedrals(dihedral_atoms, periodic=True)
        em_dih_not_periodic = feat1.get_output().DihedralFeature.values.squeeze()
        em_dih_periodic = feat2.get_output().DihedralFeature.values.squeeze()

        self.assertAllClosePeriodic(md_dih_periodic, em_dih_periodic, atol=0.01)
        self.assertAllClosePeriodic(md_dih_not_periodic, em_dih_not_periodic, atol=0.01)

        dask_feat1 = em.Featurizer(em_traj_w_pbc, in_memory=False)
        dask_feat1.add_dihedrals(dihedral_atoms, periodic=False)
        dask_feat2 = em.Featurizer(em_traj_w_pbc, in_memory=False)
        dask_feat2.add_dihedrals(dihedral_atoms, periodic=True)

        dask_em_dih_not_periodic = feat1.get_output().DihedralFeature.values.squeeze()
        dask_em_dih_periodic = feat2.get_output().DihedralFeature.values.squeeze()

        self.assertAllClosePeriodic(md_dih_periodic, dask_em_dih_periodic, atol=0.01)
        self.assertAllClosePeriodic(
            md_dih_not_periodic, dask_em_dih_not_periodic, atol=0.01
        )

    def test_performance_full_featurization(self):
        """Less of a test, more of a comparison:

        Test three datasets:
            * Single traj.
            * Two trajs.
            * Large diUbq dataset.

        Test these approaches to characterize all backbone features:
            * Numpy/Scipy
            * MDTraj
            * MDAnalysis
            * PyEMMA
            * dask/delayed with numba etc. (i.e. EncoderMap)

        """
        self.assertTrue(False)


class TestFeatures(tf.test.TestCase):
    featurizer_class = Featurizer

    def assertHasAttr(self, obj, intendedAttr):
        """Helper to check whether an attr is present."""
        testBool = hasattr(obj, intendedAttr)
        self.assertTrue(
            testBool, msg=f"obj lacking an attribute. {obj=}, {intendedAttr=}"
        )

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

    def setUp(self) -> None:
        traj_path = (Path(__file__) / "../../tests/data/known_angles.h5").resolve()
        self.traj = SingleTraj(traj_path)
        self.traj.load_custom_topology({"RES": ("ARG", {})})

        traj_paths = (Path(__file__) / "../../tests/data").resolve()
        traj_paths = list(traj_paths.glob("known_angles_*.h5"))
        traj_paths = list(sorted(traj_paths, key=lambda x: int(str(x)[-4])))
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

    def test_ensemble_with_diff_length(self):
        # Encodermap imports
        from encodermap.trajinfo import TrajEnsemble

        trajs = TrajEnsemble(
            [
                Path(__file__).resolve().parent.parent
                / "tutorials/notebooks_starter/asp7.xtc",
                Path(__file__).resolve().parent / "data/glu7.xtc",
            ],
            [
                Path(__file__).resolve().parent.parent
                / "tutorials/notebooks_starter/asp7.pdb",
                Path(__file__).resolve().parent / "data/glu7.pdb",
            ],
            common_str=["asp7", "glu7"],
        )
        self.assertNotEqual(trajs[0].n_frames, trajs[1].n_frames)
        trajs.load_CVs("all", ensemble=True)

    def test_ala_dipeptide_correct_number_of_residues(self):
        self.assertEqual(2, self.md_traj.n_residues)

    def test_featurization_of_random_subset(self):
        """Create three cases in temporary directories

        * Ensemble of two trajs with different files and different topologies.
        * Same, but with complicated indices.
        * Ensemble of three trajs, two with the same, one different topology.
        * Ensemble of six trajs, with two topologies and three files.
        Each one of those should also employ complex slicing in a second iteration.
        """
        # get asp7, asp10 etc data
        # Encodermap imports
        from encodermap.kondata import get_from_url

        output_dir = Path(__file__).resolve().parent / "data/pASP_pGLU"
        get_from_url(
            "https://sawade.io/encodermap_data/pASP_pGLU",
            output_dir,
            mk_parentdir=True,
            silence_overwrite_message=True,
        )

        self.asp7_md_traj = md.load(
            str(output_dir / "asp7.xtc"), top=str(output_dir / "asp7.gro")
        )

        self.glu7_md_traj = md.load(
            str(output_dir / "glu7.xtc"), top=str(output_dir / "glu7.gro")
        )

        self.glu8_md_traj = md.load(
            str(output_dir / "glu8.xtc"), top=str(output_dir / "glu8.gro")
        )

        self.asp10_md_traj = md.load(
            str(output_dir / "asp10.xtc"), top=str(output_dir / "asp10.gro")
        )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            self.asp7_md_traj.save_xtc(str(td / "asp7_1.xtc"))
            self.asp7_md_traj[0].save_pdb(str(td / "asp7.pdb"))

            self.glu7_md_traj.save_xtc(str(td / "glu7_1.xtc"))
            self.glu7_md_traj[0].save_pdb(str(td / "glu7.pdb"))

            trajs = em.TrajEnsemble(
                [td / "asp7_1.xtc", td / "glu7_1.xtc"],
                [td / "asp7.pdb", td / "glu7.pdb"],
            )
            trajs.load_CVs("all", ensemble=True)
            for traj in trajs:
                self.assertEqual(len(traj._CVs.coords["frame_num"]), traj.n_frames)
                if "asp7" in traj.basename:
                    self.assertEqual(traj.side_dihedrals.shape[1], 14)
                elif "glu7" in traj.basename:
                    self.assertEqual(traj.side_dihedrals.shape[1], 21)
                else:
                    self.assertTrue(False, msg="Unknown basename in trajfiles")

            traj1 = em.SingleTraj(td / "asp7_1.xtc", td / "asp7.pdb", traj_num=4)[::10][
                [0, 2, 4, 6, 8]
            ]
            self.assertEqual(traj1.n_frames, 5)
            self.assertEqual(traj1.__class__.__name__, "SingleTraj")

            trajs = em.TrajEnsemble(
                [
                    traj1,
                    em.SingleTraj(td / "glu7_1.xtc", td / "glu7.pdb", traj_num=2)[::12][
                        [1, 1, 1, 5, 10]
                    ],
                ]
            )
            with self.assertRaises(Exception):
                trajs.load_CVs("all", ensemble=True)

            trajs = em.TrajEnsemble(
                [
                    traj1,
                    em.SingleTraj(td / "glu7_1.xtc", td / "glu7.pdb", traj_num=2)[::12][
                        [
                            0,
                            5,
                            10,
                            7,
                            2,
                        ]
                    ],
                ]
            )

            with self.assertRaises(AttributeError):
                trajs[0].side_dihedrals

            trajs.load_CVs("all", ensemble=True)
            for traj in trajs:
                self.assertEqual(len(traj._CVs.coords["frame_num"]), traj.n_frames)
                if "asp7" in traj.basename:
                    self.assertEqual(traj.side_dihedrals.shape[1], 14)
                elif "glu7" in traj.basename:
                    self.assertEqual(traj.side_dihedrals.shape[1], 21)
                else:
                    self.assertTrue(False, msg="Unknown basename in trajfiles")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            self.asp7_md_traj[:100].save_xtc(str(td / "asp7_1.xtc"))
            self.asp7_md_traj[100:220].save_xtc(str(td / "asp7_2.xtc"))
            self.asp7_md_traj[0].save_pdb(str(td / "asp7.pdb"))

            self.glu7_md_traj.save_xtc(str(td / "glu7_1.xtc"))
            self.glu7_md_traj[0].save_pdb(str(td / "glu7.pdb"))

            trajs = em.TrajEnsemble(
                [td / "asp7_1.xtc", td / "asp7_2.xtc", td / "glu7_1.xtc"],
                [td / "asp7.pdb", td / "glu7.pdb"],
                common_str=["asp7", "glu7"],
            )
            trajs.load_CVs("all", ensemble=True)
            for traj in trajs:
                self.assertEqual(len(traj._CVs.coords["frame_num"]), traj.n_frames)
                if "asp7" in traj.basename:
                    self.assertEqual(traj.side_dihedrals.shape[1], 14)
                elif "glu7" in traj.basename:
                    self.assertEqual(traj.side_dihedrals.shape[1], 21)
                else:
                    self.assertTrue(False, msg="Unknown basename in trajfiles")

            traj1 = em.SingleTraj(td / "asp7_1.xtc", td / "asp7.pdb", traj_num=1)[:25]
            traj2 = em.SingleTraj(td / "asp7_2.xtc", td / "asp7.pdb", traj_num=2)[:75]
            traj3 = em.SingleTraj(td / "asp7_1.xtc", td / "asp7.pdb", traj_num=3)[25:50]
            traj4 = em.SingleTraj(td / "asp7_2.xtc", td / "asp7.pdb", traj_num=4)[75:]
            traj5 = em.SingleTraj(td / "asp7_1.xtc", td / "asp7.pdb", traj_num=5)[50:]
            traj6 = em.SingleTraj(td / "glu7_1.xtc", td / "glu7.pdb", traj_num=6)

            trajs = em.TrajEnsemble([traj1, traj2, traj3, traj4, traj5, traj6])
            trajs.load_CVs("all", ensemble=True)
            for traj in trajs:
                self.assertEqual(len(traj._CVs.coords["frame_num"]), traj.n_frames)
                if "asp7" in traj.basename:
                    self.assertEqual(traj.side_dihedrals.shape[1], 14)
                elif "glu7" in traj.basename:
                    self.assertEqual(traj.side_dihedrals.shape[1], 21)
                else:
                    self.assertTrue(False, msg="Unknown basename in trajfiles")

    def test_featurizer_provided_with_mdtraj_single_traj_and_traj_ensemble(self):
        """Test whether the featurizer also works if it is provided a list
        of mdtraj features.
        """
        top = Path(__file__).resolve().parent / "data/1am7_protein.pdb"
        traj = em.SingleTraj(
            md.load(
                Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
                top=top,
            )
        )
        traj_paths = Path(__file__).resolve().parent / "data"
        traj_paths = list(traj_paths.glob("1am7*part*.xtc"))[::-1]
        trajs = em.TrajEnsemble([md.load(t, top=top) for t in traj_paths])

        feat = self.featurizer_class(traj)
        self.assertFalse(feat._can_load)

        traj.load_CV("all", deg=True)
        trajs.load_CVs("all", deg=True)

        self.assertTrue(np.any(trajs.central_dihedrals > np.pi))
        self.assertTrue(np.any(trajs._CVs.central_dihedrals.values > np.pi))
        for key, data in traj._CVs.data_vars.items():
            o1 = data[0].values
            o2 = np.vstack(trajs._CVs[key].values)
            if o2.ndim == 3:
                ind = ~np.all(np.isnan(o2), (1, 2))
            else:
                ind = ~np.all(np.isnan(o2), 1)
            tmp = o2[ind]
            if "indices" not in key:
                self.assertEqual(o1.shape, tmp.shape)
                self.assertTrue(np.array_equal(o1, tmp))

    def test_add_all(self):
        # create featurizers for the two Info classes
        # Encodermap imports
        from encodermap.loading.featurizer import SingleTrajFeaturizer

        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

        feat1.add_all()
        feat2.add_all()
        self.assertIsInstance(feat1, SingleTrajFeaturizer)

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
        # Encodermap imports
        from encodermap.loading.featurizer import EnsembleFeaturizer

        self.assertIsInstance(feat2, EnsembleFeaturizer)
        out = list(feat2.feature_containers.values())[0]
        # Encodermap imports
        from encodermap.misc.xarray import unpack_data_and_feature

        out = unpack_data_and_feature(
            out,
            self.trajs[0],
            feat2.transform(self.trajs[0]),
        )
        self.assertNotIn(
            "Selection",
            out.coords["SELECTIONFEATURE"],
        )
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
            f"of `SingleTraj` is broken: {self.traj._CVs=} {self.traj.CVs=}"
        )
        self.assertEqual(
            self.traj._CVs.SelectionFeature.values[0].shape,
            self.traj.CVs["SelectionFeature"].shape,
            msg=msg,
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
        self.assertIn(
            "full_path",
            self.traj._CVs.SelectionFeature.attrs,
            msg=f"{self.traj._CVs.SelectionFeature.attrs.keys()=}",
        )
        self.assertNotIn(
            "full_paths",
            self.traj._CVs.SelectionFeature.attrs,
            msg=f"{self.traj._CVs.SelectionFeature.attrs.keys()=}",
        )
        self.assertEqual(
            self.traj._CVs.SelectionFeature.attrs["full_path"],
            self.traj.traj_file,
            msg=f"{self.traj._CVs.SelectionFeature.attrs=}",
        )
        self.assertEqual(
            self.traj._CVs.SelectionFeature.attrs["topology_file"], self.traj.top_file
        )
        self.assertEqual(
            self.traj._CVs.SelectionFeature.attrs["feature_axis"], "SELECTIONFEATURE"
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs["SelectionFeature_feature_indices"].values[0, ::3, 0],
                np.arange(4),
            ),
            msg=(
                f"{self.traj._CVs.SelectionFeature.shape=}\n"
                f"{self.traj._CVs.SelectionFeature_feature_indices.values[0, ::3, 0]=}\n"
            ),
        )

        # do the same things with the trajs
        self.assertTrue(
            np.array_equal(
                self.trajs.CVs["SelectionFeature"], self.traj.CVs["SelectionFeature"]
            )
        )
        self.assertEqual(self.trajs._CVs.sizes["traj_num"], 2)
        self.assertEqual(self.trajs._CVs.sizes["frame_num"], 3)
        self.assertEqual(self.trajs._CVs.sizes["SELECTIONFEATURE"], 12)
        self.assertEqual(self.trajs._CVs.sizes["traj_num"], 2)

    def test_add_selection(self):
        # create featurizers for the two Info classes
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

        feat1.add_selection(self.traj.top.select("name B1"))
        feat2.add_selection(self.traj.top.select("name B1"))

        out1 = feat1.get_output()
        self.assertTrue(np.alltrue(out1.SelectionFeature.values[0] == 0))
        out2 = feat2.get_output()
        self.assertTrue(np.alltrue(out2.SelectionFeature.values[0] == 0))
        self.assertTrue(np.alltrue(out2.SelectionFeature.values[0] == 0))

    @patch(
        "encodermap.loading.featurizer.SingleTrajFeaturizer.add_distances_ca",
        add_B1_and_B2_as_ca,
    )
    def test_add_distances_ca(self):
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

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
                self.traj._CVs.DistanceFeature_feature_indices.values[0],
                feat1.active_features[0].distance_indexes,
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
            np.array_equal(
                self.traj._CVs.DistanceFeature_feature_indices.values[0],
                np.array([[0, 1]]),
            )
        )

    @patch(
        "encodermap.loading.featurizer.SingleTrajFeaturizer.add_distances_ca",
        add_B1_and_B4_as_ca,
    )
    def test_add_inverse_distances(self):
        # Encodermap imports
        from encodermap.loading.featurizer import EnsembleFeaturizer

        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

        self.assertIsInstance(feat2, EnsembleFeaturizer)
        self.assertHasAttr(feat2, "add_distances_ca")

        feat1.add_distances_ca()
        feat1.add_inverse_distances([0, 3])

        feat2.add_distances_ca()
        feat2.add_inverse_distances([0, 3])
        self.assertIn(
            "Adds the distances between all Ca",
            feat2.add_distances_ca.__doc__,
        )
        self.assertIn(
            "DistanceFeature",
            [f.__class__.__name__ for f in feat2.active_features[self.trajs[0].top]],
        )
        msg = f"{feat1.active_features}"
        self.assertEqual(len(feat1.active_features), 2, msg=msg)
        out1_distances = feat1.get_output().DistanceFeature.values[
            0
        ]  # use [0] to get 1st and only traj
        out1_inverse_distances = feat1.get_output().InverseDistanceFeature.values[0]
        self.assertIn(
            "DistanceFeature",
            list(feat2.get_output().data_vars.keys()),
            msg=(f"{feat2.get_output()=}"),
        )
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

        self.assertIn("InverseDistanceFeature", self.traj._CVs.data_vars)
        self.assertIn(
            "InverseDistanceFeature_feature_indices", self.traj._CVs.data_vars
        )
        self.assertEqual(self.traj._CVs.InverseDistanceFeature.shape, (1, 6, 1))
        self.assertEqual(self.trajs._CVs.InverseDistanceFeature.shape, (2, 3, 1))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.InverseDistanceFeature.values[0, :3],
                self.trajs._CVs.InverseDistanceFeature.values[0],
            )
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.DistanceFeature_feature_indices.values[0],
                np.array([[0, 3]]),
            )
        )

    def test_add_contacts(self):
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

        feat1.add_distances([0, 1, 3])
        feat1.add_contacts([0, 1, 3], threshold=1.2)
        feat2.add_distances([0, 1, 3])
        feat2.add_contacts([0, 1, 3], threshold=1.2)

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
        self.assertIn("ContactFeature_feature_indices", self.traj._CVs.data_vars)
        self.assertEqual(self.traj._CVs.ContactFeature.shape, (1, 6, 3))
        self.assertEqual(self.trajs._CVs.ContactFeature.shape, (2, 3, 3))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.ContactFeature.values[0, :3],
                self.trajs._CVs.ContactFeature.values[0],
            )
        )

    def test_add_residue_mindist(self):
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

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

        self.assertIn("ResidueMinDistanceFeature", self.traj._CVs.data_vars)
        self.assertIn(
            "ResidueMinDistanceFeature_feature_indices", self.traj._CVs.data_vars
        )
        self.assertEqual(self.traj._CVs.ResidueMinDistanceFeature.shape, (1, 6, 3))
        self.assertEqual(self.trajs._CVs.ResidueMinDistanceFeature.shape, (2, 3, 3))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs["ResidueMinDistanceFeature_feature_indices"].values[0],
                pairs,
            )
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.ResidueMinDistanceFeature.values[0, :3],
                self.trajs._CVs.ResidueMinDistanceFeature.values[0],
            )
        )

    def test_add_group_COM(self):
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)
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
                # otherwise it is always 1, 0, 0
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
        self.assertIn("GroupCOMFeature", self.traj._CVs)
        self.assertEqual(self.traj._CVs.GroupCOMFeature.shape, (1, 6, 6))
        self.assertEqual(self.trajs._CVs.GroupCOMFeature.shape, (2, 3, 6))
        self.assertIn("GroupCOMFeature", self.traj._CVs.GroupCOMFeature.attrs)
        self.assertTrue(
            np.array_equal(self.traj._CVs.attrs["GroupCOMFeature"][0], np.array([0, 1]))
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.attrs["GroupCOMFeature"][1], np.array([0, 1, 2, 3])
            )
        )
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self.traj.save(
                td / "traj.h5",
            )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.GroupCOMFeature.values[0, :3],
                self.trajs._CVs.GroupCOMFeature.values[0],
            )
        )

    def test_add_residue_COM(self):
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

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

        self.assertIn("ResidueCOMFeature", self.traj._CVs.data_vars)
        self.assertIn("ResidueCOMFeature_feature_indices", self.traj._CVs.data_vars)
        self.assertEqual(self.traj._CVs.ResidueCOMFeature.shape, (1, 6, 12))
        self.assertEqual(self.trajs._CVs.ResidueCOMFeature.shape, (2, 3, 12))
        self.assertEqual(
            self.traj._CVs["ResidueCOMFeature_feature_indices"].values[0][0][0],
            0.0,
            msg=f"{self.traj._CVs['ResidueCOMFeature_feature_indices'].values[0][0]=}",
        )
        self.assertEqual(
            self.traj._CVs["ResidueCOMFeature_feature_indices"].values[0][3][0],
            1.0,
            msg=(f"{self.traj._CVs['ResidueCOMFeature_feature_indices'].values=}"),
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.ResidueCOMFeature.values[0, :3],
                self.trajs._CVs.ResidueCOMFeature.values[0],
            )
        )

    def test_add_angles(self):
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

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

        self.assertEqual(self.traj._CVs.AngleFeature.attrs["angle_units"], "deg")
        self.assertTrue(np.any(self.traj._CVs.AngleFeature.values > 5))
        self.assertIn(
            "AngleFeature_feature_indices",
            self.traj._CVs.data_vars,
            msg=f"{self.traj._CVs=}",
        )
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
                self.traj._CVs.AngleFeature_feature_indices.values[0],
                np.array([[0, 1, 2], [1, 2, 3]]),
            )
        )

    def test_add_dihedrals(self):
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

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

        self.assertIn("DihedralFeature", self.traj._CVs.data_vars)
        self.assertIn("DihedralFeature_feature_indices", self.traj._CVs.data_vars)
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
                self.traj._CVs.DihedralFeature_feature_indices[0],
                np.array([[0, 1, 2, 3]]),
            )
        )

    def test_add_backbone_torsions(self):
        feat1 = self.featurizer_class(self.md_traj)
        feat1.add_backbone_torsions(deg=True)

        out1 = feat1.get_output().BackboneTorsionFeature.values[0]
        # alanine dipeptide should have 1 psi, 1 phi (and 1 omega) torsions
        self.assertEqual(len(feat1.describe()), 2)
        self.assertEqual(out1.shape[1], 2)

        feat2 = self.featurizer_class(self.protein_1am7)
        feat2.add_backbone_torsions()
        print(self.protein_1am7.top.n_residues)
        self.assertEqual(
            len(feat2.describe()), 2 * self.protein_1am7.top.n_residues - 2
        )

        feat3 = self.featurizer_class(self.ala10_helix)
        feat3.add_backbone_torsions()
        self.assertEqual(len(feat3.describe()), 18)

    def test_add_chi1_torsions(self):
        feat1 = self.featurizer_class(self.protein_1am7)
        feat1.add_sidechain_torsions(which=["chi1"], deg=True)

        ds = feat1.get_output()
        out1 = ds.SideChainTorsions.values[0]

        self.assertEqual(len(out1), len(self.protein_1am7))
        self.assertEqual(len(feat1.describe()), out1.shape[1])
        self.assertTrue(np.any(out1 > np.pi))
        self.assertTrue(all(["CHI2" not in i for i in ds.coords["SIDECHAINTORSIONS"]]))

        feat2 = self.featurizer_class(self.md_traj)
        with self.assertRaises(ValueError):
            feat2.add_sidechain_torsions(which=["chi1"])

    def test_add_sidechain_torsions(self):
        feat1 = self.featurizer_class(self.protein_1am7)
        feat1.add_sidechain_torsions(deg=False)

        ds = feat1.get_output()
        out1 = ds.SideChainTorsions.values[0]

        self.assertEqual(len(out1), len(self.protein_1am7))
        self.assertEqual(len(feat1.describe()), out1.shape[1])
        self.assertTrue(np.all(out1 <= np.pi))
        contains_chi1 = any(
            ["CHI1" in i for i in ds.coords["SIDECHAINTORSIONS"].values]
        )
        self.assertTrue(
            contains_chi1, msg=f"{contains_chi1=}, {ds.coords['SIDECHAINTORSIONS']=}"
        )

        feat2 = self.featurizer_class(self.md_traj)
        with self.assertRaises(ValueError):
            feat2.add_sidechain_torsions()

    def test_add_minrmsd_to_ref(self):
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

        feat1.add_minrmsd_to_ref(self.traj.traj, 0)
        feat2.add_minrmsd_to_ref(self.traj.traj, 0)

        out1 = feat1.get_output()
        out1 = out1.MinRmsdFeature.values[0]
        out2 = feat2.get_output()
        out2 = np.vstack(out2.MinRmsdFeature.values)

        self.assertEqual(out1[0, 0], 0)
        self.assertGreater(out1[1, 0], out1[0, 0])
        self.assertEqual(len(out1), len(self.traj))
        self.assertLess(out1[2, 0], out1[1, 0])

        self.assertTrue(np.array_equal(out1, out2), msg=format_msg(out1, out2))

        # use the load_CV feature of TrajEnsemble and SingleTraj
        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn(
            "MinRmsdFeature", self.traj._CVs.data_vars, msg=f"{self.traj._CVs}"
        )
        self.assertIn(
            "MinRmsdFeature_feature_indices",
            self.traj._CVs.data_vars,
            msg=f"{self.traj._CVs}",
        )
        self.assertEqual(self.traj._CVs.MinRmsdFeature.shape, (1, 6, 1))
        self.assertEqual(self.trajs._CVs.MinRmsdFeature.shape, (2, 3, 1))
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.MinRmsdFeature.values[0, :3],
                self.trajs._CVs.MinRmsdFeature.values[0],
            )
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.MinRmsdFeature_feature_indices.values[0][0],
                np.array([0, 1, 2, 3]),
            )
        )

    def test_add_custom_feature(self):
        """Two types of custom features can be added.
        pyemma.CustomFeature or subclassing of encodermap.Feature
        """
        # providing a function and dim
        dim = self.md_traj.n_atoms * 3
        feat1 = em.features.CustomFeature(
            fun=lambda x: (x.xyz**2).reshape(-1, dim),
            dim=dim,
        )

        class RandomIntForAtomFeature(em.features.CustomFeature):
            def __init__(self, traj, selstr="all"):
                self.traj = traj
                self.indexes = self.traj.top.select(selstr)
                self.dimension = len(self.indexes)

            def describe(self):
                getlbl = (
                    lambda at: f"atom {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4}"
                )
                labels = []
                for i in self.indexes:
                    i = self.traj.top.atom(i)
                    labels.append(f"Random int for {getlbl(i)}")
                return labels

            def transform(self, traj):
                values = traj.xyz[:, :, 0]
                for i in self.indexes:
                    values[:, i] = float(str(hash(str(self.traj.top.atom(i))))[-5:])
                return values

            @property
            def name(self):
                return "MyAwesomeFeature"

        # instantiate the featurizer
        feat = self.featurizer_class(self.md_traj)

        # add the features
        feat.add_custom_feature(feat1)
        out_squared = feat.transform().copy()
        self.assertAllEqual(
            (self.md_traj.xyz**2).reshape(-1, dim),
            out_squared,
        )
        feat.add_custom_feature(RandomIntForAtomFeature(self.md_traj))

        # get output
        data = feat.get_output()

        # checks
        self.assertEqual(
            (self.md_traj.xyz**2).reshape(-1, dim).shape,
            data["CustomFeature[0][0]"].values.reshape(-1, dim).shape,
        )
        self.assertAllEqual(
            (self.md_traj.xyz**2).reshape(-1, dim),
            data["CustomFeature[0][0]"].values.reshape(-1, dim),
        )
        self.assertIn("CUSTOMFEATURE[0][0]", data.coords)
        self.assertIn("CustomFeature[0][0]", data.attrs)
        self.assertIn("MYAWESOMEFEATURE", data.coords)
        self.assertIn("MyAwesomeFeature", data.attrs)
        self.assertIs(data["MyAwesomeFeature"].values.dtype, np.dtype(int))

    def test_custom_features_with_phosphothreonine(self):
        """Test, whether a theoretical chi2 and chi3 angle in a phospho-threonine
        protein could be detected."""
        # Encodermap imports
        from encodermap.kondata import get_from_url

        output_dir = Path(__file__).resolve().parent / "data/OTU11"
        get_from_url(
            "https://sawade.io/encodermap_data/OTU11",
            output_dir,
            mk_parentdir=True,
            silence_overwrite_message=True,
        )

        custom_aas = {
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
        }

        trajs = em.load(
            [
                output_dir / "OTU11_phospho_threonine.xtc",
                output_dir / "OTU11_phospho_dead.xtc",
            ],
            [
                output_dir / "OTU11_phospho_threonine.pdb",
                output_dir / "OTU11_phospho_dead.pdb",
            ],
        )

        trajs.load_CVs("all", ensemble=True, custom_aas=custom_aas)
        ds = trajs._CVs.copy()
        for r in trajs[0].top.residues:
            if (r.name == "THR" or r.name == "SER") and any(
                [a.name == "P" for a in r.atoms]
            ):
                test = f"SIDECHDIH CHI3 {r.resSeq:>3}"
                labels = ds.side_dihedrals.SIDE_DIHEDRALS.values.tolist()
                self.assertTrue(
                    any([label == test for label in labels]),
                    msg=(
                        f"The phospho-{r.name} at index {r.index}, resSeq {r.resSeq} does not contain the requested line {test}: {labels=}"
                    ),
                )

        # test saving and loading for undefined sidechain_dihedrals attr
        self.assertNotEqual(
            ds["side_dihedrals_feature_indices"].size,
            0,
            msg=f"{ds['side_dihedrals_feature_indices']=}\n{ds['side_dihedrals_feature_indices'].size=}",
        )
        h5_file = Path(__file__).resolve().parent / "data/test.h5"
        trajs.save_CVs(h5_file)
        self.assertTrue(h5_file.is_file())

        # Encodermap imports
        from encodermap.misc.backmapping import mdtraj_backmapping

        fake_central_dih_rad = np.random.uniform(
            low=-np.pi, high=np.pi, size=(5, trajs[0].central_dihedrals.shape[1])
        )
        fake_side_dih_rad = np.random.uniform(
            low=-np.pi, high=np.pi, size=(5, trajs[0].side_dihedrals.shape[1])
        )

        test, back_indices = mdtraj_backmapping(
            top=0,
            dihedrals=fake_central_dih_rad,
            sidechain_dihedrals=fake_side_dih_rad,
            trajs=trajs,
            verify_every_rotation=True,
            custom_aas=custom_aas,
            return_indices=True,
        )
        test = em.SingleTraj(test)
        test.load_CV("all", deg=False)

        self.assertEqual(test.central_dihedrals.shape, fake_central_dih_rad.shape)
        self.assertEqual(test.side_dihedrals.shape, fake_side_dih_rad.shape)

        # exclude prolines in this assessment
        for data_var_name, data_var in test._CVs.data_vars.items():
            self.assertGreater(data_var.size, 0)
            self.assertGreater(getattr(test, data_var_name).size, 0)
        self.assertFalse(
            any(["PRO" in i and "PHI" in i for i in back_indices["dihedrals_labels"]])
        )
        self.assertGreater(
            test._CVs.central_dihedrals.values[0].shape[-1],
            test._CVs.central_dihedrals.sel(
                CENTRAL_DIHEDRALS=back_indices["dihedrals_labels"]
            )
            .values[0]
            .shape[-1],
        )

        tol = 0.12
        ind = np.in1d(
            test._CVs.central_dihedrals.coords["CENTRAL_DIHEDRALS"].values,
            back_indices["dihedrals_labels"],
        )
        for col1, col2, name in zip(
            test.central_dihedrals[:, ind].T,
            fake_central_dih_rad[:, ind].T,
            test._CVs.central_dihedrals.CENTRAL_DIHEDRALS[ind].values,
        ):
            for v1, v2 in zip(col1, col2):
                d = np.abs(v1 - v2)
                d = np.minimum(d, 2 * np.pi - d)
                self.assertLess(
                    d,
                    tol,
                    msg=(
                        f"The dihedral angle {name} was not back-mapped correctly. It was requested to "
                        f"be set to {v2} rad, but the actual value is {v1} rad. The periodic distance "
                        f"is {d} rad, which is greater than the tolerance of {tol} rad."
                    ),
                )

        ind = np.in1d(
            test._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"].values,
            back_indices["side_dihedrals_labels"],
        )
        for col1, col2, name in zip(
            test.side_dihedrals[:, ind].T,
            fake_side_dih_rad[:, ind].T,
            test._CVs.side_dihedrals.SIDE_DIHEDRALS[ind].values,
        ):
            for v1, v2 in zip(col1, col2):
                d = np.abs(v1 - v2)
                d = np.minimum(d, 2 * np.pi - d)
                self.assertLess(
                    d,
                    tol,
                    msg=(
                        f"The dihedral angle {name} was not back-mapped correctly. It was requested to "
                        f"be set to {v2} rad, but the actual value is {v1} rad. The periodic distance "
                        f"is {d} rad, which is greater than the tolerance of {tol} rad."
                    ),
                )

    def test_encodermap_features_cartesians(self):
        feat1 = self.featurizer_class(self.trajs)
        feat2 = self.featurizer_class(self.trajs)

        feat1.add_list_of_feats(["all_cartesians"], check_aas=False)
        feat2.add_all()

        out1 = np.vstack(feat1.get_output().all_cartesians.values)
        out2 = np.vstack(feat2.get_output().SelectionFeature.values).reshape((6, 4, 3))

        self.assertTrue(len(feat1.describe()), self.traj.n_atoms * 3)
        self.assertTrue(np.array_equal(out1, out2))

        custom_aas = {
            "RES": None,
        }
        self.traj.load_CV(["all_cartesians"], custom_aas=custom_aas)
        self.assertIsNone(self.traj._CVs.coords["traj_num"].values[0])
        self.trajs.load_CVs(["all_cartesians"], custom_aas=custom_aas)

        # feat1 uses the wrong trajs and can't be used on traj.
        with self.assertRaises(Exception):
            self.traj.load_CV(feat1)

        # build a correct feat1
        feat1 = self.featurizer_class(self.traj)
        feat1.add_list_of_feats(["all_cartesians"])

        # at this point, both should only have one active feature
        self.assertEqual(len(feat1), 1)
        self.assertEqual(len(feat2), 1)

        # adding the same feature to feat2 should trigger a warning
        with self.assertLogs(feat2.feat.logger, "WARNING"):
            feat2.add_all()
        with self.assertLogs(feat1.feat.logger, "WARNING"):
            feat1.add_all()
        self.assertEqual(self.traj._CVs.sizes["traj_num"], 1)
        self.assertEqual(self.traj.basename, "known_angles")
        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertEqual(len(self.traj.CVs), 1)

        # some checks for the 3D atomic coordinates
        self.assertIn("all_cartesians", self.traj.CVs)
        self.assertIn("all_cartesians", self.trajs._CVs)
        # check for the same contents as feat.get_output()
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
        print(self.traj._CVs.all_cartesians)
        self.assertEqual(
            str(self.traj._CVs.all_cartesians.attrs["full_path"]),
            str(self.traj.traj_file),
            msg=f"Files {self.traj._CVs.all_cartesians.attrs['full_path']} and {self.traj.traj_file} "
            f"do not match.",
        )
        self.assertEqual(
            str(self.traj._CVs.all_cartesians.attrs["topology_file"]),
            str(self.traj.top_file),
            msg=f"Files {self.traj._CVs.all_cartesians.attrs['topology_file']} and {self.traj.top_file} "
            f"do not match.",
        )
        self.assertEqual(self.traj._CVs.all_cartesians.attrs["feature_axis"], "ATOM")
        self.assertTrue(
            np.array_equal(
                self.traj._CVs["all_cartesians_feature_indices"].values[0], np.arange(4)
            )
        )

        # do the same things with the trajs
        self.assertTrue(
            np.array_equal(
                self.trajs.CVs["all_cartesians"], self.traj.CVs["all_cartesians"]
            )
        )
        self.assertEqual(self.trajs._CVs.sizes["traj_num"], 2)
        self.assertEqual(self.trajs._CVs.sizes["frame_num"], 3)
        self.assertEqual(self.trajs._CVs.sizes["ATOM"], 4)
        self.assertEqual(self.trajs._CVs.sizes["COORDS"], 3)
        self.assertEqual(self.trajs._CVs.sizes["traj_num"], 2)

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
        # Third Party Imports
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


class TestDaskFeatures(TestFeatures):
    featurizer_class = DaskFeaturizer


################################################################################
# Add Doctests here if needed
################################################################################


test_cases = (
    TestFeatures,
    # TestDaskFeatures
    # TestSpecialDaskFeatures,
)


################################################################################
# Filter Tests (because tensorflow is sometimes weird)
################################################################################


def load_tests(loader, tests, pattern):
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

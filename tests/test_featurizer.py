# -*- coding: utf-8 -*-
# tests/test_featurizer.py
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
import os.path
import tempfile
import unittest
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union
from unittest.mock import patch

# Third Party Imports
import dask
import MDAnalysis as mda
import mdtraj as md
import numpy as np
import tensorflow as tf
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis.distances import dist as mda_dist
from MDAnalysis.transformations import unwrap
from numpy.testing import assert_array_equal

# Encodermap imports
from conftest import skip_all_tests_except_env_var_specified
from encodermap.loading.featurizer import (
    DaskFeaturizer,
    EnsembleFeaturizer,
    Featurizer,
    SingleTrajFeaturizer,
)


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
from conftest import expensive_test
from encodermap import SingleTraj
from encodermap.loading.featurizer import pairs


warnings.filterwarnings("ignore", category=DeprecationWarning)


################################################################################
# Globals
################################################################################


ALIGNMENT = """\
linear_dimer  -------MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRL
FAT10         MAPNASCLCVHVRSEEWDLMTFDANPYDSVKKIKEHVRSKTKVPVQDQVLLLGSKILKPRRSLSSYGIDKEKTIHLTLKV
              : :.*::   . :*::.:* *:::::* :::.*  :* ::* *::..* *:  *:**.*.*:**.*:**.*::

linear_dimer  ----RGGMQIFV--KTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVL
FAT10         VKPSDEELPLFLVESGDEAKRHLLQVRRSSSVAQVKAMIETKTGIIPETQIVTCNGKRLEDGKMMADYGIRKGNLLFLAS
              : :*:  .   .*   *:*. *.:: :*** *: * ** *: * :   **:****: ::**.*:* . *.*.

linear_dimer  RLRGG
FAT10         YCIGG
              **

"""


################################################################################
# Mocks
################################################################################


def add_B1_and_B2_as_ca(*args, **kwargs):
    self = args[0]
    sel = self.traj.top.select("name B1 or name B2")
    pairs_ = pairs(sel, 0)
    if "delayed" not in kwargs:
        delayed = False
    else:
        delayed = kwargs.pop("delayed")
    self.add_distances(pairs_, periodic=True, delayed=delayed)


def add_B1_and_B4_as_ca(*args, **kwargs):
    self = args[0]
    if hasattr(self, "traj"):
        sel = self.traj.top.select("name B1 or name B4")
    else:
        sel = self.trajs[0].top.select("name B1 or name B4")
    pairs_ = pairs(sel, 0)
    if "delayed" not in kwargs:
        delayed = False
    else:
        delayed = kwargs.pop("delayed")
    self.add_distances(pairs_, periodic=True, delayed=delayed)


def mocked_all_cartesians_init(
    self,
    traj,
    check_aas: bool = False,
    generic_labels: bool = False,
    delayed: bool = False,
) -> None:
    super(em.features.AllCartesians, self).__init__(
        traj=traj,
        indexes=traj.top.select("name B1 or name B2 or name B3 or name B4"),
        check_aas=check_aas,
        delayed=delayed,
    )


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
    indices: Optional[np.ndarray] = None,
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
            total = x.size
            tol = atol + rtol * periodicity * np.abs(y)
            le = np.less_equal(d, tol) & (d != 0)
            mismatched = np.where((~np.less_equal(d, tol) & (d != 0)))[0]
            percent = len(mismatched) / total * 100.0
            if mismatched.size > max_number_mismatched:
                with np.printoptions(suppress=True):
                    msg = (
                        f"There are {len(mismatched)} out of {total} ({percent:.1f}%) "
                        f"elements in a periodic space of {periodicity:.4f} "
                        f"that exhibit differences, that are out of tolerance. "
                        f"In the flattened `actual` and `desired` arrays, these "
                        f"5 example elements exhibit a "
                        f"difference over the tolerance: {mismatched[:5]}\n"
                        f"The values of 5 example elements of these arrays are:\n\n"
                    )
                    if indices is not None:
                        msg += f"Indices:\n{indices.flatten()[mismatched][:5]}\n\n"
                    msg += (
                        f"Actual:\n{x[mismatched][:5]}\n\nDesired:\n{y[mismatched][:5]}\n\n"
                        f"Their difference in periodic space is:\n"
                        f"{d[mismatched][:5]}\n\nwhich is above the tolerance at "
                        f"these respective positions:\n{tol[mismatched][:5]}\n\n"
                        f"You can either adjust the parameters `atol`, `rtol`, or increase "
                        f"the `max_percentage_mismatched` which can be useful for periodic distances."
                    )
                    print(msg)
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


@skip_all_tests_except_env_var_specified(unittest.skip)
@expensive_test
class TestSpecialDaskFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).resolve().parent / "data"
        cls.PFFP_xtc_file = cls.data_dir / "PFFP_MD_fin_protonly_dt_100.xtc"
        cls.PFFP_tpr_file = cls.data_dir / "PFFP_MD_fin_protonly.tpr"
        cls.PFFP_gro_file = cls.data_dir / "PFFP_MD_fin_protonly.gro"
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
        return cls

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

    @unittest.skip("This test is under development.")
    def test_save_ensemble_with_coords(self):
        # self.fail("This breaks my computer. Don't know why.")
        output_dir = Path(
            em.get_from_kondata(
                "Ub_K11_mutants",
                mk_parentdir=True,
                silence_overwrite_message=True,
            )
        )

        trajs = em.load(
            [
                output_dir / "Ub_K11Ac_I/traj.xtc",
                output_dir / "Ub_K11C_I/traj.xtc",
                output_dir / "Ub_K11Q_I/traj.xtc",
                output_dir / "Ub_K11R_I/traj.xtc",
                output_dir / "Ub_wt_I/traj.xtc",
            ],
            [
                output_dir / "Ub_K11Ac_I/start.pdb",
                output_dir / "Ub_K11C_I/start.pdb",
                output_dir / "Ub_K11Q_I/start.pdb",
                output_dir / "Ub_K11R_I/start.pdb",
                output_dir / "Ub_wt_I/start.pdb",
            ],
            common_str=[
                "Ac",
                "C",
                "Q",
                "R",
                "wt",
            ],
        )

        custom_aas = {
            "KAC": (
                "K",
                {
                    "optional_bonds": [
                        ("-C", "N"),  # the peptide bond to the previous aa
                        ("N", "CA"),
                        ("N", "H"),
                        ("CA", "C"),
                        ("C", "O"),
                        ("CA", "CB"),
                        ("CB", "CG"),
                        ("CG", "CD"),
                        ("CD", "CE"),
                        ("CE", "NZ"),
                        ("NZ", "HZ"),
                        ("NZ", "CH"),
                        ("CH", "OI2"),
                        ("CH", "CI1"),
                        ("C", "+N"),  # the peptide bond to the next aa
                        ("C", "+N"),  # the peptide bond to the next aa
                    ],
                    "CHI1": ["N", "CA", "CB", "CG"],
                    "CHI2": ["CA", "CB", "CG", "CD"],
                    "CHI3": ["CB", "CG", "CD", "CE"],
                    "CHI4": ["CG", "CD", "CE", "NZ"],
                    "CHI5": ["CD", "CE", "NZ", "CH"],
                },
            )
        }
        trajs.load_custom_topology(custom_aas)

        feat1 = em.DaskFeaturizer(trajs=trajs)
        feat1.add_list_of_feats("all")
        td = Path.home()
        file = feat1.to_netcdf(
            filename=td / "test.h5",
            overwrite=True,
            with_trajectories=True,
        )
        self.assertTrue(os.path.isfile(file))

        new_trajs = em.load(Path(file))
        self.assertEqual(
            new_trajs.n_trajs,
            5,
        )
        self.assertEqual(new_trajs._CVs.sizes["traj_num"], 5)
        self.assertTrue(
            new_trajs[0]._custom_top.residues != set(),
        )
        self.assertEqual(set(new_trajs.common_str), set(trajs.common_str))

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

        feat1 = em.Featurizer(traj1)
        feat2 = em.DaskFeaturizer(traj2)

        feat1.add_all()
        feat1.add_distances_ca()
        feat2.add_all()

        out1 = feat1.get_output()
        out2 = feat2.get_output()

        self.assertIsInstance(out1, xr.Dataset)
        self.assertIsInstance(out2, xr.Dataset)

    def test_periodic_distances_angles_dihedrals(self):
        """This test uses a trajectory with many small peptides in it. That way,
        we can be sure that some pbc breaking occurs. MDTRaj and MDAnalysis should
        be able to resolve them.
        """
        print("TESTPERIODIC start")
        PFFP_gro = self.data_dir / "PFFP_MD_fin_protonly.gro"
        PFFP_tpr = self.data_dir / "PFFP_MD_fin_protonly.tpr"
        PFFP_xtc = self.data_dir / "PFFP_MD_fin_protonly_dt_100.xtc"
        PFFP_fixed_xtc = self.data_dir / "PFFP_MD_fin_protonly_dt_100_fixed_pbc.xtc"

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
        dask_feat1 = em.DaskFeaturizer(em_traj_w_pbc)
        dask_feat1.add_distances(atom_pairs, periodic=False)
        dask_feat2 = em.DaskFeaturizer(em_traj_w_pbc)
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

        dask_feat1 = em.DaskFeaturizer(em_traj_w_pbc)
        dask_feat1.add_dihedrals(dihedral_atoms, periodic=False)
        dask_feat2 = em.DaskFeaturizer(em_traj_w_pbc)
        dask_feat2.add_dihedrals(dihedral_atoms, periodic=True)

        dask_em_dih_not_periodic = feat1.get_output().DihedralFeature.values.squeeze()
        dask_em_dih_periodic = feat2.get_output().DihedralFeature.values.squeeze()

        self.assertAllClosePeriodic(md_dih_periodic, dask_em_dih_periodic, atol=0.01)
        self.assertAllClosePeriodic(
            md_dih_not_periodic, dask_em_dih_not_periodic, atol=0.01
        )


@skip_all_tests_except_env_var_specified(unittest.skip)
class TestFeatures(tf.test.TestCase):
    featurizer_class = Featurizer

    def assertHasAttr(self, obj, intendedAttr):
        """Helper to check whether an attr is present."""
        testBool = hasattr(obj, intendedAttr)
        self.assertTrue(
            testBool, msg=f"obj lacking an attribute. {obj=}, {intendedAttr=}"
        )

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
            self.fail(msg)
        except TypeError as e:
            raise Exception(
                f"assertAllEqual got bad types: {x=} {y=} {type(x)=} {type(y)=}"
            )

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).resolve().parent / "data"
        return cls

    def setUp(self) -> None:
        traj_path = self.data_dir / "known_angles.h5"
        self.traj = SingleTraj(traj_path)
        self.traj.load_custom_topology({"RES": ("ARG", {})})

        traj_paths = list(self.data_dir.glob("known_angles_*.h5"))
        traj_paths = list(sorted(traj_paths, key=lambda x: int(str(x)[-4])))
        self.trajs = em.TrajEnsemble(traj_paths)

        md_traj_xtc = self.data_dir / "alanine_dipeptide.xtc"
        md_traj_pdb = self.data_dir / "alanine_dipeptide.pdb"
        self.md_traj = SingleTraj(md_traj_xtc, md_traj_pdb)

        self.traj_file_1am7 = self.data_dir / "1am7_corrected.xtc"
        self.top_file_1am7 = self.data_dir / "1am7_protein.pdb"
        self.protein_1am7 = SingleTraj(self.traj_file_1am7, self.top_file_1am7)

        traj_file = self.data_dir / "Ala10_helix.xtc"
        top_file = self.data_dir / "Ala10_helix.pdb"
        self.ala10_helix = SingleTraj(traj_file, top_file)

    def test_OTU_11_ensemble(self):
        # Encodermap imports
        from encodermap.kondata import get_from_url

        output_dir = Path(
            get_from_url(
                "https://sawade.io/encodermap_data/OTU11_preequilibrated",
                mk_parentdir=True,
                silence_overwrite_message=True,
            )
        )
        pdb_files = list(output_dir.glob("*.pdb"))
        xtc_files = list(output_dir.glob("*.xtc"))
        self.assertGreater(
            len(pdb_files),
            0,
            msg=f"No pdb files from OTU11_preequilibrated dataset downloaded to {output_dir}",
        )
        self.assertGreater(
            len(xtc_files),
            0,
            msg=f"No xtc files from OTU11_preequilibrated dataset downloaded to {output_dir}",
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
                        ("OT", "C"),
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
                        ("OT", "C"),
                    ],
                    "CHI2": ["CA", "CB", "OG", "P"],
                    "CHI3": ["CB", "OG", "P", "OXT"],
                },
            ),
        }

        trajs = em.TrajEnsemble(pdb_files)
        trajs.load_custom_topology(custom_aas)
        trajs.load_CVs("all", ensemble=True)
        self.assertIn("side_dihedrals", trajs.CVs)
        self.assertIn("side_dihedrals", trajs._CVs)

        trajs = em.TrajEnsemble(
            trajs=xtc_files,
            tops=[f.with_suffix(".pdb") for f in xtc_files],
        )
        trajs.load_custom_topology(custom_aas)
        trajs.load_CVs("all", ensemble=True)
        self.assertIn("side_dihedrals", trajs.CVs)
        self.assertIn("side_dihedrals", trajs._CVs)
        print(trajs._CVs)

    def test_ensemble_with_diff_length(self):
        # Encodermap imports
        from encodermap.trajinfo import TrajEnsemble

        trajs = TrajEnsemble(
            [
                Path(__file__).resolve().parent.parent
                / "tutorials/notebooks_starter/asp7.xtc",
                self.data_dir / "glu7.xtc",
            ],
            [
                Path(__file__).resolve().parent.parent
                / "tutorials/notebooks_starter/asp7.pdb",
                self.data_dir / "glu7.pdb",
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

        output_dir = self.data_dir / "pASP_pGLU"
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
        top = self.data_dir / "1am7_protein.pdb"
        traj = em.SingleTraj(
            md.load(
                self.data_dir / "1am7_corrected.xtc",
                top=top,
            )
        )
        traj_paths = self.data_dir / ""
        traj_paths = list(traj_paths.glob("1am7*part*.xtc"))[::-1]
        trajs = em.TrajEnsemble([md.load(t, top=top) for t in traj_paths])

        feat = self.featurizer_class(traj)
        # self.assertFalse(feat._can_load)

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

        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

        feat1.add_all()
        feat2.add_all()
        self.assertIsInstance(feat1, SingleTrajFeaturizer)

        out1 = feat1.get_output()
        out2 = feat2.get_output()
        self.assertEqual(len(out2.coords["traj_num"]), 2)

        # check the positions
        self.assertTrue(np.all(out1.SelectionFeature.values[0][:, :3] == 0))
        self.assertTrue(np.all(out2.SelectionFeature.values[0][:, :3] == 0))
        self.assertTrue(np.all(out2.SelectionFeature.values[1][:, :3] == 0))
        self.assertEqual(out1.SelectionFeature.values[0][5, 10], 1.5)

        # use the load_CV feature of TrajEnsemble and SingleTraj
        self.traj.load_CV(feat1)
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

        # get the output of the featurizers
        # as datasets to check whether some coordinate mangling occurs
        if self.featurizer_class is DaskFeaturizer:
            self.assertIsNotNone(feat1.dataset.SelectionFeature.chunks)

        # test the CVs
        msg = (
            "Usually, the `add_all` method of the Featurizer class"
            "adds a `SelectionFeature` to the active features of the Featurizer. "
            "Thus, the key in `traj.CVs` should also be named 'SelectionFeature'."
        )
        self.assertIn("SelectionFeature", self.traj.CVs, msg=msg)
        # check for the same contents as feat.get_output()
        self.assertTrue(np.all(self.traj.CVs["SelectionFeature"][:, :3] == 0))

        msg = (
            "This error can be very serious. Normally, the `traj.CVs` attribute "
            "was meant to be built from the `traj._CVs` DataArray. If the values "
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

        # check the coordinates of this DataArray
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
        self.assertIsInstance(
            self.traj,
            SingleTraj,
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
        self.assertIn(
            "SelectionFeature",
            self.trajs.CVs,
            msg=(
                f"The ensemble '{self.trajs}' has no CV with name 'SelectionFeature' "
                f"loaded. Here is the dataset:\n\n{self.trajs._CVs}"
            ),
        )
        self.assertEqual(
            self.trajs.CVs["SelectionFeature"].shape,
            self.traj.CVs["SelectionFeature"].shape,
            msg=(
                f"Shapes of SingleTraj and TrajEnsemble do not match:\n"
                f"Single Traj: {self.traj.CVs['SelectionFeature'].shape}\n"
                f"Traj Ensemble: {self.trajs.CVs['SelectionFeature'].shape}"
            ),
        )
        self.assertAllEqual(
            self.trajs._CVs["SelectionFeature"].sel(traj_num=0, frame_num=2).values,
            self.traj._CVs["SelectionFeature"].sel(traj_num=None, frame_num=2).values,
            msg=(f"The coordinate of frame 0 "),
        )
        self.assertAllEqual(
            self.trajs.CVs["SelectionFeature"],
            self.traj.CVs["SelectionFeature"],
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
        self.assertTrue(np.all(out1.SelectionFeature.values[0] == 0))
        out2 = feat2.get_output()
        self.assertTrue(np.all(out2.SelectionFeature.values[0] == 0))
        self.assertTrue(np.all(out2.SelectionFeature.values[0] == 0))

    @patch(
        "encodermap.loading.featurizer.SingleTrajFeaturizer.add_distances_ca",
        add_B1_and_B2_as_ca,
    )
    def test_add_distances_ca(self):
        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

        feat1.add_distances_ca()
        feat2.add_distances_ca()

        if self.featurizer_class is DaskFeaturizer:
            self.assertTrue(feat1.active_features[0].delayed)

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
        self.assertEqual(
            self.trajs._CVs.ResidueMinDistanceFeature.shape,
            (2, 3, 3),
            msg=(
                f"The shape of `TrajEnsemble` `ResidueMinDistanceFeature` is unexpected."
                f"The shape is expected to be (trajs, frames, n_features), which for "
                f"the pairs {pairs=} equates to (2, 3, 3), but {self.trajs._CVs.ResidueMinDistanceFeature.shape} "
                f"was received. The indices are {self.trajs._CVs.ResidueMinDistanceFeature.coords['RESIDUEMINDISTANCEFEATURE']=}"
            ),
        )
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
        self.assertIn(
            "GroupCOMFeature_feature_indices", self.traj._CVs.GroupCOMFeature.attrs
        )
        self.assertEqual(
            self.traj._CVs.attrs["GroupCOMFeature_feature_indices"],
            "[array([0, 1]), array([0, 1, 2, 3])]",
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
        with warnings.catch_warnings():
            warnings.filterwarnings("error", r".*re-add.*", UserWarning)
            feat1.add_all()
        feat2.add_residue_COM([0, 1, 2, 3], image_molecules=False, mass_weighted=False)
        feat2.add_all()

        out1_com = feat1.get_output().ResidueCOMFeature.values[0]
        try:
            out1_sel = feat1.get_output().SelectionFeature.values[0]
        except AttributeError as e:
            raise Exception(f"{feat1.get_output()=}\n\n{feat1.active_features=}") from e
        out2_com = np.vstack(feat2.get_output().ResidueCOMFeature.values)
        out2_sel = np.vstack(feat2.get_output().SelectionFeature.values)

        self.assertTrue(np.array_equal(out1_com, out1_sel))
        self.assertTrue(
            np.array_equal(out1_com, out2_com), msg=format_msg(out1_com, out2_com)
        )

        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn("ResidueCOMFeature", self.traj._CVs.data_vars)
        self.assertIn("ResidueCOMFeature_feature_indices", self.traj._CVs.attrs)
        self.assertEqual(self.traj._CVs.ResidueCOMFeature.shape, (1, 6, 12))
        self.assertEqual(self.trajs._CVs.ResidueCOMFeature.shape, (2, 3, 12))
        self.assertEqual(
            self.traj._CVs.attrs["ResidueCOMFeature_feature_indices"],
            "[[0]\n [1]\n [2]\n [3]]",
            msg=f"{self.traj._CVs.attrs['ResidueCOMFeature_feature_indices']=}",
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

        if self.featurizer_class == DaskFeaturizer:
            self.assertTrue(feat1.feat.delayed)
            self.assertTrue(feat2.feat.delayed)
        else:
            self.assertFalse(feat1.delayed)
            self.assertFalse(feat2.delayed)

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

        feat1 = self.featurizer_class(self.traj)
        feat2 = self.featurizer_class(self.trajs)

        feat1.add_angles([[0, 1, 2], [1, 2, 3]], deg=True, cossin=True)
        feat2.add_angles([[0, 1, 2], [1, 2, 3]], deg=True, cossin=True)

        out1 = feat1.get_output().AngleFeature.values[0]
        out2 = np.vstack(feat2.get_output().AngleFeature.values)
        self.assertAllEqual(
            out1[:, 1],
            1,
        )
        self.assertAllEqual(
            out2[:, 1],
            1,
        )
        self.assertAllEqual(
            out1[[0, 1, 3], -1],
            1,
        )
        self.assertAllEqual(
            out2[[0, 1, 3], -1],
            1,
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
        out1 = out1.MinRmsdFeature_with_4_atoms_in_reference.values[0]
        out2 = feat2.get_output()
        out2 = np.vstack(out2.MinRmsdFeature_with_4_atoms_in_reference.values)

        self.assertEqual(out1[0, 0], 0)
        self.assertGreater(out1[1, 0], out1[0, 0])
        self.assertEqual(len(out1), len(self.traj))
        self.assertLess(out1[2, 0], out1[1, 0])

        self.assertAllClose(out1, out2)

        # use the load_CV feature of TrajEnsemble and SingleTraj
        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertIn(
            "MinRmsdFeature_with_4_atoms_in_reference",
            self.traj._CVs.data_vars,
            msg=f"{self.traj._CVs}",
        )
        self.assertIn(
            "MinRmsdFeature_with_4_atoms_in_reference_feature_indices",
            self.traj._CVs.data_vars,
            msg=f"{self.traj._CVs}",
        )
        self.assertEqual(
            self.traj._CVs.MinRmsdFeature_with_4_atoms_in_reference.shape, (1, 6, 1)
        )
        self.assertEqual(
            self.trajs._CVs.MinRmsdFeature_with_4_atoms_in_reference.shape, (2, 3, 1)
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.MinRmsdFeature_with_4_atoms_in_reference.values[0, :3],
                self.trajs._CVs.MinRmsdFeature_with_4_atoms_in_reference.values[0],
            )
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.MinRmsdFeature_with_4_atoms_in_reference_feature_indices.values[
                    0
                ][
                    0
                ],
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
            fun=lambda traj: (traj.xyz**2).reshape(-1, dim),
            dim=dim,
            delayed=self.featurizer_class is DaskFeaturizer,
        )

        class RandomFloatForAtomFeature(em.features.CustomFeature):
            def __init__(self, traj, selstr="all"):
                """The init method can implement whatever code you like."""
                self.traj = traj
                self.top = traj.top
                self.indexes = self.traj.top.select(selstr)
                self.dimension = len(self.indexes)

            def describe(self):
                getlbl = (
                    lambda at: f"atom {at.name:>4}:{at.index:5} {at.residue.name}:{at.residue.resSeq:>4}"
                )
                labels = []
                for i in self.indexes:
                    i = self.traj.top.atom(i)
                    labels.append(f"Random float for {getlbl(i)}")
                return labels

            @staticmethod
            @dask.delayed
            def delayed_call(
                traj,
                indexes,
                **kwargs,
            ):
                """Delayed implementation of `self.call()` needs to be a staticmethod

                It also needs to be decorated with the dask.delayed decorator.

                Per default, the instance arguments `traj` and `indexes` are passed
                to this method. If you need more than these two arguments, you can define the
                instance attribute `self._kwargs` in the `__init__()` method.
                This dict will then be passed as further keyword arguments.

                """
                values = traj.xyz[..., 0]
                for i in indexes:
                    values[:, i] = float(str(hash(str(traj.top.atom(i)))))
                return values

            def call(self, traj):
                values = traj.xyz[..., 0]
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
        self.assertAllClose(
            (self.md_traj.xyz**2).reshape(-1, dim),
            out_squared,
        )
        feat.add_custom_feature(RandomFloatForAtomFeature(self.md_traj))

        if self.featurizer_class is DaskFeaturizer:
            self.assertHasAttr(feat.feat.active_features[1], "name")
            self.assertEqual(feat.feat.active_features[1].name, "MyAwesomeFeature")

        # get output
        data = feat.get_output()

        # checks
        self.assertEqual(
            (self.md_traj.xyz**2).reshape(-1, dim).shape,
            data["CustomFeature_0"].values.reshape(-1, dim).shape,
        )
        self.assertAllClose(
            (self.md_traj.xyz**2).reshape(-1, dim),
            data["CustomFeature_0"].values.reshape(-1, dim),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertIn("CUSTOMFEATURE_0", data.coords)
        self.assertIn("MYAWESOMEFEATURE", data.coords)
        self.assertIn("CUSTOMFEATURE_0", data.attrs["feature_axes"])
        self.assertIn("MYAWESOMEFEATURE", data.attrs["feature_axes"])
        self.assertIs(data["MyAwesomeFeature"].values.dtype, np.dtype("float32"))

        # instantiate an ensemble featurizer
        feat = self.featurizer_class(self.trajs)
        feat.add_custom_feature(RandomFloatForAtomFeature(self.trajs[0]))
        dim = self.trajs[0].n_atoms * 3
        feature = em.features.CustomFeature(
            fun=lambda traj: (traj.xyz**2).reshape(-1, dim),
            dim=dim,
        )
        feat.add_custom_feature(feature)
        out = feat.get_output()
        test = []
        for t in self.trajs:
            test.append((t.xyz.copy() ** 2).reshape(-1, dim))
        test = np.array(test)
        self.assertAllEqual(test, out["CustomFeature_0"].values)

        # try loading from featurizer
        self.trajs.load_CVs(feat)

        # add some 1-d, 2-d, and 2-d features
        feat = self.featurizer_class(self.trajs)
        # a per-frame-feature
        feat1 = em.features.CustomFeature(
            fun=lambda traj: np.array([0, 1, 2]).astype("float32"),
            dim=1,
        )
        # a normal feature
        feat2 = em.features.CustomFeature(
            fun=lambda traj: np.array(
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                ],
            ).astype("float32"),
            dim=4,
        )
        # a cartesian feature
        feat3 = em.features.CustomFeature(
            fun=lambda traj: np.random.random(traj.xyz.shape).astype("float32"),
            dim=self.trajs[0].n_atoms * 3,
        )
        feat.add_custom_feature(feat1)
        feat.add_custom_feature(feat2)
        feat.add_custom_feature(feat3)
        self.assertEqual(
            [f.id for l in feat.active_features.values() for f in l], [0, 1, 2]
        )
        out = feat.get_output()
        self.trajs.del_CVs()
        self.trajs.load_CVs(out)
        self.trajs.del_CVs()
        self.trajs.load_CVs(feat)
        self.assertEqual(
            set(self.trajs.CVs.keys()),
            {"CustomFeature_2", "CustomFeature_0", "CustomFeature_1"},
        )

    def test_feature_MAE(self):
        output_dir = Path(
            em.get_from_kondata(
                "topological_examples",
                silence_overwrite_message=True,
                mk_parentdir=True,
            )
        )
        mae = em.load(output_dir / "MAE.pdb")
        f = em.features.SideChainAngles(mae)
        self.assertIn(
            "SIDECHANGLE ATOM   CG:   19 GLU:   3 ANGLE ATOM   CD:   20 GLU:   3 ANGLE ATOM  OE1:   21 GLU:   3 CHAIN 0",
            f.describe(),
        )
        self.assertAllEqual(
            f.angle_indexes[-1],
            np.array([19, 20, 21]),
        )

    def test_feature_equality(self):
        """To ensure features are not added twice, every feature comes with an `__eq__()`
        method. Check them here.
        """
        asp7_traj = em.load(
            self.data_dir / "asp7.xtc",
            self.data_dir / "asp7.pdb",
        )
        # some checks for protein_1am7
        self.assertIsNotNone(self.protein_1am7.indices_chi1)
        # Encodermap imports
        from encodermap.loading import features
        from encodermap.loading.features import __all__

        trajs = self.protein_1am7.copy()._gen_ensemble()
        count = 0
        for a in __all__:
            f = getattr(features, a)
            if a == "CustomFeature":
                continue
            elif a == "SelectionFeature":
                f1 = f(traj=self.protein_1am7, indexes=[0, 1, 2, 3])
                f2 = f(traj=self.protein_1am7, indexes=[0, 1, 2, 3])
                self.assertEqual(f1, f2)
                f3 = f(traj=asp7_traj, indexes=[0, 1, 2, 3])
                self.assertNotEqual(f1, f3)
            elif a in ["DistanceFeature", "InverseDistanceFeature", "ContactFeature"]:
                f1 = f(
                    traj=self.protein_1am7, distance_indexes=np.array([[0, 1], [1, 2]])
                )
                f2 = f(
                    traj=self.protein_1am7, distance_indexes=np.array([[0, 1], [1, 2]])
                )
                self.assertEqual(f1, f2)
                f3 = f(traj=asp7_traj, distance_indexes=np.array([[0, 1], [1, 2]]))
                self.assertNotEqual(f1, f3)
            elif a == "ResidueMinDistanceFeature":
                f1 = f(
                    traj=self.protein_1am7,
                    contacts=np.array([[0, 1], [1, 2]]),
                    scheme="ca",
                    ignore_nonprotein=True,
                    threshold=2,
                    periodic=True,
                )
                f2 = f(
                    traj=self.protein_1am7,
                    contacts=np.array([[0, 1], [1, 2]]),
                    scheme="ca",
                    ignore_nonprotein=True,
                    threshold=2,
                    periodic=True,
                )
                self.assertEqual(f1, f2)
                f3 = f(
                    traj=asp7_traj,
                    contacts=np.array([[0, 1], [1, 2]]),
                    scheme="ca",
                    ignore_nonprotein=True,
                    threshold=2,
                    periodic=True,
                )
                self.assertNotEqual(f1, f3)
            elif a == "GroupCOMFeature":
                idx = np.array([[0, 1, 2, 3, 4, 5]])
                f1 = f(traj=self.protein_1am7, group_definitions=idx)
                f2 = f(traj=self.protein_1am7, group_definitions=idx)
                self.assertEqual(f1, f2)
                f3 = f(traj=asp7_traj, group_definitions=idx)
                self.assertNotEqual(f1, f3)
            elif a == "ResidueCOMFeature":
                scheme = "backbone"
                idx = [0, 1, 2, 3, 4]
                # Encodermap imports
                from encodermap.loading.featurizer import _atoms_in_residues

                residue_atoms = _atoms_in_residues(
                    self.protein_1am7,
                    idx,
                    subset_of_atom_idxs=self.protein_1am7.select(scheme),
                )
                f1 = f(
                    traj=self.protein_1am7,
                    residue_indices=idx,
                    residue_atoms=np.asarray(residue_atoms),
                )
                f2 = f(
                    traj=self.protein_1am7,
                    residue_indices=idx,
                    residue_atoms=np.asarray(residue_atoms),
                )
                self.assertEqual(f1, f2)
                residue_atoms = _atoms_in_residues(
                    asp7_traj, idx, subset_of_atom_idxs=asp7_traj.select(scheme)
                )
                f3 = f(
                    traj=asp7_traj,
                    residue_indices=idx,
                    residue_atoms=np.asarray(residue_atoms),
                )
                self.assertNotEqual(f1, f3)
                with self.assertRaises(Exception):
                    f(
                        traj=asp7_traj,
                        residue_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        residue_atoms=residue_atoms,
                    )
            elif a == "AngleFeature":
                f1 = f(
                    traj=self.protein_1am7,
                    angle_indexes=np.array([[0, 1, 2], [1, 2, 3]]),
                )
                f2 = f(
                    traj=self.protein_1am7,
                    angle_indexes=np.array([[0, 1, 2], [1, 2, 3]]),
                )
                self.assertEqual(f1, f2)
                f3 = f(traj=asp7_traj, angle_indexes=np.array([[0, 1, 2], [1, 2, 3]]))
                self.assertNotEqual(f1, f3)
            elif a == "DihedralFeature":
                f1 = f(
                    traj=self.protein_1am7,
                    dih_indexes=np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
                )
                f2 = f(
                    traj=self.protein_1am7,
                    dih_indexes=np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
                )
                self.assertEqual(f1, f2)
                f3 = f(
                    traj=asp7_traj, dih_indexes=np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
                )
                self.assertNotEqual(f1, f3)
            elif a == "MinRmsdFeature":
                f1 = f(traj=self.protein_1am7, ref=self.protein_1am7.traj)
                f2 = f(traj=self.protein_1am7, ref=self.protein_1am7.traj)
                self.assertEqual(f1, f2)
                f3 = f(traj=asp7_traj, ref=asp7_traj.traj)
                self.assertNotEqual(f1, f3)
            elif a == "AlignFeature":
                f1 = f(
                    traj=self.protein_1am7,
                    indexes=np.array([0, 1, 2, 3, 4]),
                    reference=self.protein_1am7.traj,
                )
                f2 = f(
                    traj=self.protein_1am7,
                    indexes=np.array([0, 1, 2, 3, 4]),
                    reference=self.protein_1am7.traj,
                )
                self.assertEqual(f1, f2)
                f3 = f(
                    traj=asp7_traj,
                    indexes=np.array([0, 1, 2, 3, 4]),
                    reference=asp7_traj.traj,
                )
                self.assertNotEqual(f1, f3)
                f4 = f(
                    traj=self.protein_1am7,
                    indexes=np.array([0, 1, 2, 3, 4, 5]),
                    reference=self.protein_1am7.traj,
                )
                self.assertNotEqual(f1, f4)
            else:
                f1 = f(traj=self.protein_1am7)
                f2 = f(traj=self.protein_1am7)
                self.assertEqual(f1, f2)
                f3 = f(traj=asp7_traj)
                self.assertNotEqual(f1, f3)
            count += 1
            trajs.featurizer._add_feature(f1, trajs.top[0], trajs)

        self.assertLen(trajs.featurizer, 23)
        self.assertEqual(len(trajs.featurizer), count)
        test = trajs.featurizer.transform(trajs[0])
        self.assertEqual(test.shape, (51, 11361))
        with self.assertWarnsRegex(
            UserWarning,
            r".*re-add.*",
            msg=("No warning issued after adding features a second time."),
        ):
            trajs.featurizer.add_list_of_feats("all")
        test2 = trajs.featurizer.transform(trajs[0])
        self.assertLen(trajs.featurizer, 23)
        self.assertEqual(len(trajs.featurizer), count)
        self.assertEqual(test2.shape, (51, 11361))

        test = trajs.featurizer.get_output()
        self.assertIn("GroupCOMFeature_feature_indices", test.GroupCOMFeature.attrs)

        # some tests before saving
        trajs.load_CVs(trajs.featurizer)
        self.assertIn(
            "GroupCOMFeature_feature_indices",
            trajs._CVs.GroupCOMFeature.attrs,
            f"{trajs._CVs.data_vars.keys()=}",
        )
        self.assertIn("feature_axes", trajs._CVs.attrs)

        # save and test
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            trajs.save(td / "trajs.h5")

            test = em.TrajEnsemble.from_dataset(td / "trajs.h5")
            print(test._CVs)
            print(test._CVs.GroupCOMFeature.attrs)
            print(test._CVs.MinRmsdFeature_with_2504_atoms_in_reference_feature_indices)

    def test_generic_features_diUbi_FAT10(self):
        """Check the feature alignment of FAT10 and M1-diUbi. This requires an
        alignment with the ClustalW library."""
        ubi_output_dir = Path(
            em.get_from_kondata(
                dataset_name="linear_dimers",
                mk_parentdir=True,
                silence_overwrite_message=True,
            )
        )
        fat10_output_dir = Path(
            em.get_from_kondata(
                dataset_name="FAT10", mk_parentdir=True, silence_overwrite_message=True
            )
        )

        ubi_trajs = [ubi_output_dir / "01.xtc", ubi_output_dir / "02.xtc"]
        ubi_tops = [f.with_suffix(".pdb") for f in ubi_trajs]
        ubi_trajs = em.TrajEnsemble(
            trajs=ubi_trajs,
            tops=ubi_tops,
            traj_nums=[0, 1],
            common_str=["linear_dimer", "linear_dimer"],
        )
        fat10_trajs = [fat10_output_dir / "01.xtc", fat10_output_dir / "02.xtc"]
        fat10_tops = [f.with_suffix(".pdb") for f in fat10_trajs]
        fat10_trajs = em.TrajEnsemble(
            trajs=fat10_trajs,
            tops=fat10_tops,
            traj_nums=[2, 3],
            common_str=["FAT10", "FAT10"],
        )

        self.assertEqual(ubi_trajs[0].traj_num, 0)
        self.assertEqual(ubi_trajs[1].traj_num, 1)
        self.assertEqual(fat10_trajs[0].traj_num, 2)
        self.assertEqual(fat10_trajs[1].traj_num, 3)

        tmp = fat10_trajs.copy()
        tmp[0].traj_num = 0
        self.assertEqual(tmp[0].traj_num, 0)

        with self.assertRaisesRegex(
            Exception, expected_regex=r".*overlapping traj_nums.*"
        ):
            _ = ubi_trajs + tmp

        trajs = ubi_trajs + fat10_trajs
        for i, traj in enumerate(trajs):
            self.assertEqual(traj.traj_num, i)
        alignment_str = trajs.to_alignment_query()
        trajs.parse_clustal_w_alignment(ALIGNMENT)
        fat10_traj = trajs[2].copy()
        ubi_traj = trajs[0].copy()

        # central dihedral alignment, because some residues can't have all angles
        central_dihedrals = em.features.CentralDihedrals(ubi_traj)
        self.assertLen(central_dihedrals.describe(), 152 * 3 - 3)

        # load a distance feature to ensure alignment
        self.assertHasAttr(fat10_traj, "clustal_w")
        self.assertHasAttr(ubi_traj, "clustal_w")
        fat10_distances = em.features.CentralBondDistances(
            fat10_traj
        ).generic_describe()
        ubi_distances = em.features.CentralBondDistances(ubi_traj).generic_describe()
        self.assertNotEqual(len(fat10_distances), len(ubi_distances))
        self.assertEqual(ubi_distances[0], "CENTERDISTANCE  22")
        self.assertEqual(fat10_distances[0], "CENTERDISTANCE  1")

        # load some side_dihedral features to ensure presence
        fat10_traj = trajs[2].copy()
        side_dihedrals = em.features.SideChainDihedrals(fat10_traj).generic_describe()
        self.assertEqual(
            side_dihedrals[:4],
            [
                "SIDECHDIH CHI1   1",
                "SIDECHDIH CHI2   1",
                "SIDECHDIH CHI3   1",
                "SIDECHDIH CHI1   3",
            ],
        )

        trajs.load_CVs("all", ensemble=True, alignment=ALIGNMENT)
        ds = trajs._CVs.copy()

        # the first N sidechain dihedrals should be nan in the diUbi
        sidechain_dihedrals_per_residue = {
            "ALA": 0,
            "ARG": 5,
            "ASN": 2,
            "ASP": 2,
            "CYS": 1,
            "GLN": 3,
            "GLU": 3,
            "GLY": 0,
            "HIS": 2,
            "ILE": 2,
            "LEU": 2,
            "LYS": 4,
            "MET": 3,
            "PHE": 2,
            "PRO": 2,
            "SER": 1,
            "THR": 1,
            "TRP": 2,
            "TYR": 2,
            "VAL": 1,
            "A": 0,
            "R": 5,
            "N": 2,
            "D": 2,
            "C": 1,
            "Q": 3,
            "E": 3,
            "G": 0,
            "H": 2,
            "I": 2,
            "L": 2,
            "K": 4,
            "M": 3,
            "F": 2,
            "P": 2,
            "S": 1,
            "T": 1,
            "W": 2,
            "Y": 2,
            "V": 1,
        }

        # count the number of sidechain angles in the N-terminal "tail" of FAT10
        seq = "MAPNASC"
        count = 0
        for l in seq:
            count += sidechain_dihedrals_per_residue[l]
        test = ds.sel(traj_num=[0, 1]).side_dihedrals[..., :count]
        self.assertTrue(
            np.all(np.isnan(test.values)),
            msg=(
                f"This error happens, because the sidechain dihedral labels are "
                f"not correctly ordered: "
                f"{test.coords['SIDE_DIHEDRALS']=}\n"
                f"Ubi   sidechain dihedrals generic features {trajs.featurizer.active_features[trajs[0].top][-1].describe()=}\n"
                f"FAT10 sidechain dihedrals generic features {trajs.featurizer.active_features[trajs[2].top][-1].describe()=}\n"
            ),
        )
        self.assertTrue(
            np.all(~np.isnan(ds.sel(traj_num=[2, 3]).side_dihedrals[..., :count])),
            msg=f"{test=} {count=} {test['SIDE_DIHEDRALS']=}",
        )

        # check residue 85 which should contain nan in FAT10 but not in diUbi
        ind1 = ds.side_dihedrals["SIDE_DIHEDRALS"].str.endswith("85")
        ind2 = ds.side_dihedrals["SIDE_DIHEDRALS"].str.contains("CHI 3|CHI 4|CHI5")
        ind = ds.side_dihedrals["SIDE_DIHEDRALS"][ind1 & ind2]
        test = ds.side_dihedrals.sel(SIDE_DIHEDRALS=ind)
        self.assertTrue(np.all(~np.isnan(test[[0, 1]])))
        self.assertTrue(np.all(np.isnan(test[[2, 3]])))

        # test whether it trains and saves
        total_steps = 5
        main_path = em.misc.run_path(Path(fat10_output_dir) / "runs")
        parameters = dict(
            n_steps=total_steps,
            main_path=main_path,
            cartesian_cost_scale=1,
            cartesian_cost_variant="mean_abs",
            cartesian_cost_scale_soft_start=(
                int(total_steps / 10 * 9),
                int(total_steps / 10 * 9) + total_steps // 50,
            ),
            cartesian_pwd_start=1,
            cartesian_pwd_step=3,
            dihedral_cost_scale=1,
            dihedral_cost_variant="mean_abs",
            distance_cost_scale=0,
            cartesian_distance_cost_scale=100,
            cartesian_dist_sig_parameters=[40, 10, 5, 1, 2, 5],
            checkpoint_step=max(1, int(total_steps / 10)),
            l2_reg_constant=0.001,
            center_cost_scale=0,
            tensorboard=True,
            use_sidechains=True,
            use_backbone_angles=True,
        )

        parameters = em.ADCParameters(**parameters)
        emap = em.AngleDihedralCartesianEncoderMap(trajs=trajs, parameters=parameters)
        lowd = emap.encode()
        self.assertEqual(
            lowd.shape[1],
            2,
        )

        # debug tensorflow
        # first call the decoder and see, whether that works
        highd = emap.model.decoder(np.random.random((100, 2)))
        for h, s in zip(highd, [493, 492, 408]):
            self.assertEqual(
                h.shape,
                (100, s),
            )
        shapes = emap.model.encoder.input_shape
        lowd = emap.model.encoder([np.random.random((100, s[1])) for s in shapes])
        self.assertEqual(
            lowd.shape,
            (100, 2),
        )
        _, data, __ = emap.get_train_data_from_trajs(
            trajs[0], emap.p, attr="_CVs", max_size=100
        )
        lowd = emap.encode(data)
        self.assertEqual(
            lowd.shape,
            (100, 2),
        )
        # Encodermap imports
        from encodermap.autoencoder.autoencoder import np_to_sparse_tensor

        ds_dict = {
            key: trajs._CVs[key][0, :100].values
            for key in [
                "central_angles",
                "central_dihedrals",
                "central_cartesians",
                "central_distances",
                "side_dihedrals",
            ]
        }
        ds = []
        for k, v in ds_dict.items():
            if k != "central_cartesians":
                ds.append(np_to_sparse_tensor(v))
            else:
                ds.append(np_to_sparse_tensor(v.reshape(100, -1)))
        models = [
            "get_dense_model_central_angles",
            "get_dense_model_central_dihedrals",
            "get_dense_model_cartesians",
            "get_dense_model_distances",
            "get_dense_model_side_dihedrals",
        ]
        for d, model in zip(ds, models):
            model = getattr(emap.model, model)
            self.assertEqual(
                d.dense_shape[1],
                model.input_shape[1],
            )
            out = model(d)
            self.assertEqual(d.dense_shape[1], out.shape[1])
        out = emap.model(ds)

    def test_sidechain_label_order(self):
        """The sidechain dihedrals should follow a residue-first order.

        So for MET-ALA-GLU, we would have
        MET1-chi1
        MET1-chi2
        MET1-chi3
        GLU3-chi1
        GLU3-chi2
        GLU3-chi3

        This needs to be tested for:
            * SingleTraj
            * TrajEnsemble
            * SingleTrajFeaturizer
            * EnsembleFeaturizer
            * SideChainDihedralFeature

        """
        traj = em.load(self.data_dir / "glu7.pdb")
        traj.load_CV("all")
        labels = traj._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"].values.tolist()
        self.assertIn("CHI1", labels[0])
        self.assertIn("GLU", labels[0])
        self.assertIn("CHI2", labels[1])
        self.assertIn("CHI3", labels[2])
        output_dir = Path(
            em.get_from_kondata(
                "topological_examples",
                mk_parentdir=True,
                silence_overwrite_message=True,
            )
        )
        trajs = em.load(list(output_dir.glob("*.pdb")))
        trajs.load_CVs("all", ensemble=False)
        self.assertTrue(trajs[0]._CVs)
        self.assertLen(trajs.featurizer, 0, msg=f"{trajs.featurizer=}")
        one, two, three, four, five, six = trajs._CVs.side_dihedrals.coords[
            "SIDE_DIHEDRALS"
        ].values[:6]
        self.assertIn(
            "MET",
            one,
            msg=(
                f"There's no MET in the first label of the sidechain dihedrals:\n"
                f"{one=}\n{trajs._CVs.side_dihedrals.coords['SIDE_DIHEDRALS']=}\n"
                f"{trajs[0]._CVs.side_dihedrals}\n{trajs[0].basename=}"
            ),
        )
        self.assertIn("CHI1", one)
        self.assertIn("MET", two)
        self.assertIn("CHI2", two)
        self.assertIn("MET", three)
        self.assertIn("CHI3", three)
        self.assertIn("GLU", four)
        self.assertIn("CHI1", four)
        self.assertIn(
            "ASP",
            five,
            msg=(
                f"There's no ASP in the fifth label of the sidechain dihedrals:\n"
                f"{five=}\n{trajs._CVs.side_dihedrals.coords['SIDE_DIHEDRALS']=}\n"
                f"{trajs[0]._CVs.side_dihedrals}\n{trajs[0].basename=}"
            ),
        )
        self.assertIn("CHI1", five)
        self.assertIn("GLU", six)
        self.assertIn("CHI2", six)

        one, two, three = trajs._CVs.central_dihedrals.coords[
            "CENTRAL_DIHEDRALS"
        ].values[:3]
        self.assertIn("PSI", one)
        self.assertIn("OMEGA", two)
        self.assertIn("PHI", three)

    def test_unnatural_amino_acids(self):
        traj_file = (
            Path(em.__file__).resolve().parent.parent
            / "tests/data/unnatural_aminoacids.pdb"
        )
        traj = em.load(traj_file)

        custom_aas = {
            "ALL": ("A", None),  # makes EncoderMap treat 2-allyl-glycine as alanine
            "OAS": (
                "S",  # OAS is 2-acetylserine
                {
                    "CHI2": [
                        "CA",
                        "CB",
                        "OG",
                        "CD",
                    ],  # this is a non-standard chi2 angle
                    "CHI3": [
                        "CB",
                        "OG",
                        "CD",
                        "CE",
                    ],  # this is a non-standard chi3 angle
                },
            ),
            "CSR": (  # CSR is selenocysteine
                "S",
                {
                    "bonds": [  # we can manually define bonds for selenocysteine like so:
                        ("-C", "N"),  # bond between previous carbon and nitrogen CSR
                        ("N", "CA"),
                        ("N", "H1"),
                        ("CA", "C"),
                        ("CA", "HA"),  # this topology includes hydrogens
                        ("C", "O"),
                        (
                            "C",
                            "OXT",
                        ),  # As the C-terminal residue, we don't need to put ("C", "+N") here
                        ("CA", "CB"),
                        ("CB", "HB1"),
                        ("CB", "HB2"),
                        ("CB", "SE"),
                        ("SE", "HE"),
                    ],
                    "CHI1": [
                        "N",
                        "CA",
                        "CB",
                        "SE",
                    ],  # this is a non-standard chi1 angle
                },
            ),
            "TPO": (  # TPO is phosphothreonine
                "T",
                {
                    "CHI2": ["CA", "CB", "OG1", "P"],  # a non-standard chi2 angle
                    "CHI3": ["CB", "OG1", "P", "OXT"],  # a non-standard chi3 angle
                },
            ),
        }

        with self.assertRaisesRegex(Exception, r".*already exists.*"):
            traj.load_custom_topology(custom_aas)

        # rename bonds to optional bonds
        custom_aas["CSR"][1]["optional_bonds"] = custom_aas["CSR"][1].pop("bonds")
        traj.load_custom_topology(custom_aas)

        # make sure the chi2 and chi3 of OAS is present
        self.assertIn(
            [65, 66, 67, 71],
            traj.indices_chi2.tolist(),
            msg=(f"The CA-CB-OG-CD dihedral does not exist: {traj.indices_chi2=}"),
        )
        self.assertIn(
            [66, 67, 71, 70],
            traj.indices_chi3.tolist(),
            msg=(f"The CA-CB-OG-CD dihedral does not exist: {traj.indices_chi3=}"),
        )

        # find the chi2 and chi3 in OAS in the sidechaindihedral feature
        f = em.features.SideChainDihedrals(traj=traj)
        self.assertIn(
            "SIDECHDIH CHI2  RESID  OAS:   4 CHAIN 0",
            f.describe(),
        )
        self.assertIn(
            "SIDECHDIH CHI3  RESID  OAS:   4 CHAIN 0",
            f.describe(),
        )

        # find chi1 in selenocysteine
        self.assertIn(
            [94, 95, 98, 99],
            traj.indices_chi1.tolist(),
        )
        self.assertIn(
            "SIDECHDIH CHI1  RESID  CSR:   6 CHAIN 0",
            f.describe(),
        )

        # find chi2 and chi3 in phosphothreonine
        self.assertIn(
            [1, 3, 5, 6],
            traj.indices_chi2.tolist(),
        )
        self.assertIn(
            "SIDECHDIH CHI2  RESID  TPO:   1 CHAIN 0",
            f.describe(),
        )
        self.assertIn(
            [3, 5, 6, 10],
            traj.indices_chi3.tolist(),
        )
        self.assertIn(
            "SIDECHDIH CHI3  RESID  TPO:   1 CHAIN 0",
            f.describe(),
        )

    def test_custom_features_with_phosphothreonine(self):
        """Test, whether a theoretical chi2 and chi3 angle in a phospho-threonine
        protein could be detected."""
        # Encodermap imports
        from encodermap.kondata import get_from_url

        output_dir = self.data_dir / "OTU11"
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

        trajs.load_custom_topology(custom_aas)
        trajs.load_CVs("all", ensemble=True)
        ds = trajs._CVs.copy()
        self.assertIn("central_dihedrals", ds)
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
        h5_file = self.data_dir / "test.h5"
        trajs.save_CVs(h5_file)
        self.assertTrue(h5_file.is_file())

        # Encodermap imports
        from encodermap.misc.backmapping import mdtraj_backmapping

        self.assertHasAttr(trajs, "central_dihedrals")
        self.assertHasAttr(trajs, "_CVs")
        self.assertHasAttr(trajs[0], "central_dihedrals")
        self.assertHasAttr(trajs[0], "_CVs")

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
            return_indices=True,
        )
        self.assertEqual(test.n_frames, 5)
        self.assertEqual(test.n_atoms, trajs[0].n_atoms)
        test = em.SingleTraj(test)

        # special tests for central_distances_feature_indices
        test2 = test.copy()
        feat = em.features.CentralBondDistances(traj=test2)
        self.assertGreater(len(feat.describe()), 0)
        self.assertGreater(feat.transform().size, 0)
        test2.featurizer.add_list_of_feats(["central_distances"])
        ds = em.misc.xarray.unpack_data_and_feature(
            test2.featurizer, test2, test2.featurizer.transform()
        )
        self.assertGreater(ds.central_distances_feature_indices.size, 0)

        # back to regular tests
        test.load_CV("all", deg=False)
        self.assertGreater(test._CVs.central_distances_feature_indices.size, 0)
        self.assertGreater(getattr(test, "central_distances_feature_indices").size, 0)
        self.assertEqual(test.central_dihedrals.shape, fake_central_dih_rad.shape)
        self.assertEqual(test.side_dihedrals.shape, fake_side_dih_rad.shape)

        # exclude prolines in this assessment
        for data_var_name, data_var in test._CVs.data_vars.items():
            # if data_var_name.endswith("feature_indices"):
            #     continue
            self.assertGreater(data_var.size, 0)
            self.assertGreater(
                getattr(test, data_var_name).size,
                0,
                msg=f"{data_var_name}\n{getattr(test, data_var_name)}\n{data_var}",
            )
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

    @patch(
        "encodermap.loading.features.AllCartesians.__init__", mocked_all_cartesians_init
    )
    def test_encodermap_features_cartesians(self):
        feat1 = self.featurizer_class(self.trajs)
        feat2 = self.featurizer_class(self.trajs)

        feat1.add_list_of_feats(["all_cartesians"], check_aas=False)
        feat2.add_all()
        self.assertEqual(len(feat2), 1)
        if self.featurizer_class is DaskFeaturizer:
            out1 = feat1.feat.get_output()
            self.assertIn(
                "all_cartesians",
                out1.data_vars.keys(),
            )
            self.assertIsInstance(
                list(feat1.feat.active_features.values())[0][0],
                em.features.AllCartesians,
            )
            out2 = feat1.get_output()
            self.assertIn(
                "all_cartesians",
                out2.data_vars.keys(),
                msg=(
                    f"The data_var 'all_cartesians' is not in the datase {out2}, "
                    f"but without the DaskFeaturzier, it is present {out1}."
                ),
            )
        out1 = np.vstack(feat1.get_output().all_cartesians.values)
        out2 = np.vstack(feat2.get_output().SelectionFeature.values).reshape((6, 4, 3))

        self.assertTrue(len(feat1.describe()), self.traj.n_atoms * 3)
        self.assertTrue(np.array_equal(out1, out2))

        custom_aas = {
            "RES": None,
        }
        self.traj.load_custom_topology(custom_aas)
        self.traj.load_CV(["all_cartesians"])
        self.assertIsNone(self.traj._CVs.coords["traj_num"].values[0])
        self.trajs.load_CVs(["all_cartesians"])

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
        with self.assertWarnsRegex(
            UserWarning,
            r".*re-add.*",
            msg=(
                f"Calling `add_all` a second time did not issue a UserWarning. "
                f"The active features are: {feat2.active_features=}"
            ),
        ):
            feat2.add_all()
        feat1.add_all()
        with self.assertWarnsRegex(UserWarning, r".*re-add.*"):
            feat1.add_all()
        self.assertEqual(self.traj._CVs.sizes["traj_num"], 1)
        self.assertEqual(self.traj.basename, "known_angles")
        self.traj.load_CV(feat1)
        self.trajs.load_CVs(feat2)

        self.assertEqual(len(self.traj.CVs), 2)

        # some checks for the 3D atomic coordinates
        self.assertIn("all_cartesians", self.traj.CVs)
        self.assertIn("all_cartesians", self.trajs._CVs)
        # check for the same contents as feat.get_output()
        print(self.traj.CVs["all_cartesians"].shape)
        print(self.traj.CVs["all_cartesians"][:, :3])
        self.assertTrue(np.all(self.traj.CVs["all_cartesians"][:, 0] == 0))
        self.assertTrue(np.all(self.trajs._CVs.all_cartesians.values[:, :, 0] == 0))

        msg = (
            "This error can be very serious. Normally, the `traj.CVs` attribute "
            "was meant to be built from the `traj._CVs` DataArray. If the values "
            "of these two arrays are not the same, something with the `CVs` property "
            "of `SingleTraj` is broken."
        )
        self.assertTrue(
            np.array_equal(
                self.traj._CVs.all_cartesians.values[0], self.traj.CVs["all_cartesians"]
            ),
            msg=msg,
        )

        # check the coordinates of this DataArray
        self.assertIsNone(self.traj._CVs.all_cartesians.coords["traj_num"].values[0])
        self.assertEqual(
            self.traj._CVs.all_cartesians.coords["traj_name"].values[0], "known_angles"
        )
        self.assertIn("ALLATOM", self.traj._CVs.all_cartesians.coords)
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
        self.assertEqual(self.traj._CVs.all_cartesians.attrs["feature_axis"], "ALLATOM")
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
        self.assertEqual(self.trajs._CVs.sizes["ALLATOM"], 4)
        self.assertEqual(self.trajs._CVs.sizes["COORDS"], 3)
        self.assertEqual(self.trajs._CVs.sizes["traj_num"], 2)

    def test_encodermap_features_ala10(self):
        self.ala10_helix.load_CV("all_cartesians")
        indices = self.ala10_helix._CVs.all_cartesians_feature_indices.values[0]
        self.assertAllEqual(
            self.ala10_helix.xyz[:, indices],
            self.ala10_helix.CVs["all_cartesians"],
        )
        self.assertAllEqual(
            self.ala10_helix.xyz[:, indices],
            self.ala10_helix._CVs.all_cartesians.values[0],
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
        index = self.protein_1am7._CVs.all_cartesians_feature_indices.values[0]
        self.assertAllEqual(
            self.protein_1am7.xyz[:, index],
            self.protein_1am7.CVs["all_cartesians"],
            msg=(f"{self.protein_1am7._CVs.all_cartesians_feature_indices.values[0]=}"),
        )
        self.assertAllEqual(
            self.protein_1am7.xyz[:, index],
            self.protein_1am7._CVs.all_cartesians.values[0],
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

        self.assertEqual(
            self.protein_1am7.CVs["side_distances"].shape, (51, no_of_sidechain_bonds)
        )
        self.assertIn("side_distances", self.protein_1am7.CVs)

        # check out the dihedrals
        self.protein_1am7.load_CV(
            ["central_angles", "central_dihedrals", "side_dihedrals"]
        )

        with self.assertRaises(ValueError):
            self.md_traj.load_CV(["central_dihedrals", "side_dihedrals"])

        # make a trace of the central and side dihedrals for the first frame
        # and then calculate the chi1 and chi2 torsions using mdtraj
        traj = md.load(
            self.traj_file_1am7,
            top=self.top_file_1am7,
        )
        for i in range(1, 5):
            indices_md, dihedrals_md = getattr(md, f"compute_chi{i}")(traj)
            where = np.where(
                self.protein_1am7._CVs.side_dihedrals.coords[
                    "SIDE_DIHEDRALS"
                ].str.contains(f"CHI{i}")
            )
            dihedrals_em = self.protein_1am7.side_dihedrals[:, where][:, 0]
            indices_em = self.protein_1am7._CVs.side_dihedrals_feature_indices.values[
                0, where
            ][0]
            self.assertAllEqual(
                indices_md,
                indices_em,
            )
            self.assertAllEqual(
                dihedrals_md,
                dihedrals_em,
            )

        frame = 0
        trace = np.tile(
            np.hstack(
                [
                    self.protein_1am7.central_angles,
                    self.protein_1am7.central_dihedrals,
                    self.protein_1am7.side_dihedrals,
                ]
            )[frame],
            (100, 1),
        )

        # import matplotlib.pyplot as plt
        # from matplotlib.testing.compare import compare_images
        # img_name = "trace_image"
        # ax = plt.imshow(trace[:, ::-1].T, cmap="viridis")
        # ax.figure.savefig(
        #     Path(__file__).resolve().parent / "data/{}_actual.png".format(img_name)
        # )
        # self.assertIsNone(
        #     compare_images(
        #         expected=str(
        #             Path(__file__).resolve().parent
        #             / "data/{}_expected.png".format(img_name)
        #         ),
        #         actual=str(
        #             Path(__file__).resolve().parent
        #             / "data/{}_actual.png".format(img_name)
        #         ),
        #         tol=10.0,
        #     )
        # )


@skip_all_tests_except_env_var_specified(unittest.skip)
class TestDaskFeatures(TestFeatures):
    def assertIsInstance(self, a, b):
        """Makes this class work with instance checks in `TestFeatures`."""
        if isinstance(a, DaskFeaturizer) and (
            b is SingleTrajFeaturizer or b is EnsembleFeaturizer
        ):
            return
        super().assertIsInstance(a, b)

    featurizer_class = DaskFeaturizer


# class TestSLURMFeatures(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls) -> None:
#         from dask_jobqueue import SLURMCluster
#         from dask.distributed import Client
#         cls.cluster = SLURMCluster(
#             queue="regular",
#             cores=6,
#             memory="8GB",
#         )
#         cls.client = Client(cls.cluster)
#         return cls
#
#     def assertIsInstance(self, a, b):
#         """Makes this class work with instance checks in `TestFeatures`."""
#         if isinstance(a, DaskFeaturizer) and (
#             b is SingleTrajFeaturizer or b is EnsembleFeaturizer
#         ):
#             return
#         super().assertIsInstance(a, b)
#
#     featurizer_class = DaskFeaturizer
#
#
# class TestDaskFeaturesOnSLURM(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         raise NotImplementedError


################################################################################
# Collect Test Cases and Filter
################################################################################


def load_tests(loader, tests, pattern):
    test_cases = (
        # TestFeatures,
        TestDaskFeatures,
        # TestSpecialDaskFeatures,
    )
    suite = unittest.TestSuite()
    for test_class in test_cases:
        try:
            tests = loader.loadTestsFromTestCase(test_class)
        except TypeError:
            raise Exception(f"{test_class=} {type(test_class)=}")
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite


################################################################################
# Main
################################################################################


if __name__ == "__main__":
    unittest.main()

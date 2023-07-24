# -*- coding: utf-8 -*-
# tests/test_moldata.py
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
import unittest
from pathlib import Path

import MDAnalysis as mda
import mdtraj as md
import numpy as np
from mdtraj.geometry import dihedral as md_dihedral

import encodermap.encodermap_tf1 as em_tf1


class TestTrajinfo(unittest.TestCase):
    def test_moldata_tf1(self):
        u = mda.Universe(
            str(Path(__file__).resolve().parent / "data/1am7_protein.pdb"),
            str(Path(__file__).resolve().parent / "data/1am7_corrected.xtc"),
        )
        top = md.load(
            str(Path(__file__).resolve().parent / "data/1am7_protein.pdb")
        ).top
        group = u.select_atoms("all")
        moldata = em_tf1.MolData(group)

        datafields = [
            "cartesians",
            "central_cartesians",
            "dihedrals",
            "sidedihedrals",
            "angles",
            "lengths",
        ]
        values = [(2504, 3), (474, 3), (471,), (316,), (472,), (473,)]

        total_data = 0
        for df, v in zip(datafields, values):
            if len(getattr(moldata, df).shape[1:]) == 2:
                total_data += (
                    getattr(moldata, df).shape[1] * getattr(moldata, df).shape[2]
                )
            else:
                total_data += getattr(moldata, df).shape[1]
            self.assertEqual(getattr(moldata, df).shape[1:], v)
        self.assertEqual(total_data, 10666)

    def test_moldata_tf2(self):
        pass

    def test_compare_moldata_tf1_tf2(self):
        pass


# Remove Phantom Tests from tensorflow skipped test_session
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
test_cases = (TestTrajinfo,)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite

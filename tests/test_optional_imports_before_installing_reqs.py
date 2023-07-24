# -*- coding: utf-8 -*-
# tests/test_optional_imports_before_installing_reqs.py
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
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestMissingImport(unittest.TestCase):
    def test_missing_import(self):
        try:
            from pyemma.coordinates.data.featurization.angles import DihedralFeature

            pyemma_installed = True
        except ModuleNotFoundError:
            pyemma_installed = False
        if not pyemma_installed:
            with self.assertRaises((ValueError, ModuleNotFoundError)):
                from encodermap.loading.features import CentralDihedrals

                central_dihedrals = CentralDihedrals()
        else:
            self.assertTrue(True)


# Remove Phantom Tests from tensorflow skipped test_session
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
test_cases = (TestMissingImport,)
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

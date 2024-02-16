# -*- coding: utf-8 -*-
# tests/test_optional_imports.py
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
# Standard Library Imports
import unittest


class TestOptionalImport(unittest.TestCase):
    def test_working_import(self):
        # Third Party Imports
        from optional_imports import _optional_import

        np = _optional_import("numpy")
        try:
            a = np.array([[1, 2], [3, 4]])
        except ValueError:
            self.fail("np = _optional_import('numpy') raised an unexpected Error")
        rando = _optional_import("numpy", "random.random")
        try:
            a = rando((2, 2))
        except ValueError:
            self.fail(
                "rando = _optional_import('numpy', 'random.random') raised an unexpected Error"
            )

    def test_non_working_import(self):
        # Third Party Imports
        from optional_imports import _optional_import

        non_existent_package = _optional_import("non_existent_package")
        with self.assertRaises(ValueError):
            return_vale = non_existent_package()
        with self.assertRaises(ValueError):
            return_var = non_existent_package.non_existent_var


test_cases = (TestOptionalImport,)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tests/run_unittests_and_coverage.py
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
import sys
import unittest
from datetime import datetime
from pathlib import Path

# Third Party Imports
import coverage
import HtmlTestRunner


def sort_tests(tests):
    for test in tests:
        try:
            if test._testMethodName == "test_losses_not_periodic":
                return 1
        except AttributeError:
            for t in test._tests:
                try:
                    if t._testMethodName == "test_losses_not_periodic":
                        return 1
                except AttributeError:
                    for _ in t._tests:
                        if _._testMethodName == "test_losses_not_periodic":
                            return 1
    return 2


def unpack_tests(tests):
    for test in tests:
        try:
            print(test._testMethodName)
        except AttributeError:
            for t in test._tests:
                try:
                    print(t._testMethodName)
                except AttributeError:
                    for _ in t._tests:
                        print(_._testMethodName)


def filter_key_test_suites(suite):
    """Returns filters True/False depending on whether a
    test needs to be executed before or after the optional
    requirements are installed"""
    if "before_installing_reqs" in suite.__str__():
        return False
    else:
        return True


class SortableSuite(unittest.TestSuite):
    def sort(self):
        # print("old")
        # unpack_tests(self._tests)
        self._tests = list(sorted(self._tests, key=sort_tests))
        # print("\n new")
        # unpack_tests(self._tests)

    def filter(self):
        self._tests = list(filter(filter_key_test_suites, self._tests))


if __name__ == "__main__":
    omit = ["*Test*", "*test*", "*/usr/local/lib*", "*Users*", "*__init__*"]
    cov = coverage.Coverage(
        config_file=str(Path(__file__).resolve().parent.parent / "pyproject.toml")
    )
    cov.start()
    loader = unittest.TestLoader()
    loader.suiteClass = SortableSuite
    test_suite = loader.discover(
        start_dir=str(Path(__file__).resolve().parent),
        top_level_dir=str(Path(__file__).resolve().parent.parent),
    )

    if any(
        [
            isinstance(test, unittest.loader._FailedTest)
            for suite in test_suite
            for test in suite._tests
        ]
    ):
        add = Path(__file__).resolve().parent
        print(f"Adding {add} to path.")
        sys.path.insert(0, str(add))
        loader = unittest.TestLoader()
        loader.suiteClass = SortableSuite
        test_suite = loader.discover(
            start_dir=str(Path(__file__).resolve().parent),
        )

    test_suite.sort()
    test_suite.filter()
    # runner = unittest.TextTestRunner()
    # result = runner.run(test_suite)
    # print("Unittest Result:", result.wasSuccessful())

    # output to a file
    out_dir = Path(__file__).resolve().parent.parent / "docs/source/_static/coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().replace(microsecond=0).isoformat()
    runner = HtmlTestRunner.HTMLTestRunner(
        output=str(out_dir.parent),
        report_title=f"EncoderMap Unittest Report from {now}",
        report_name="html_test_runner_report",
        combine_reports=True,
        add_timestamp=False,
        buffer=True,
    )
    print("saved html report")

    # run the test
    result = runner.run(test_suite)
    cov.stop()
    print(f"Saving coverage report to {out_dir}")
    cov_percentage = cov.html_report(
        directory=str(out_dir),
        title="coverage_report",
    )

    print("Unittest Result:", result.wasSuccessful())
    print("Coverage Percentage:", cov_percentage)

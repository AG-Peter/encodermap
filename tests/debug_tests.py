#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tests/debug_tests.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade
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
import os
import unittest
from pathlib import Path

# Third Party Imports
import click


################################################################################
# Globals
################################################################################


BAD_STR = """\
        if exctype is test.failureException:
            # Skip assert*() traceback levels
            length = self._count_relevant_tb_levels(tb)
            msg_lines = traceback.format_exception(exctype, value, tb, length)"""


GOOD_STR = """\
        if exctype is test.failureException:
            # Skip assert*() traceback levels
            msg_lines = traceback.format_exception(exctype, value, tb)"""


################################################################################
# Prevent tf from printing unnecessary stuff to our logs
################################################################################


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


################################################################################
# Functions and Classes
################################################################################


def yield_tests(test_suite):
    if not hasattr(test_suite, "__iter__"):
        pass
    else:
        for suite in test_suite:
            if "TestSuite" in str(suite):
                yield from yield_tests(suite)
            elif str(suite).startswith("test_"):
                yield suite
            else:
                yield from yield_tests(suite)


def main_unittests(
    debug_test: str,
) -> None:
    loader = unittest.TestLoader()
    start_dir = str(Path(__file__).resolve().parent)
    test_suite = loader.discover(
        start_dir=start_dir,
        top_level_dir=str(Path(__file__).resolve().parent.parent),
    )
    debug_test_dir = Path(__file__).resolve().parent / "debug_tests"
    debug_test_dir.mkdir(parents=True, exist_ok=True)
    all_tests = []
    for test in yield_tests(test_suite):
        if str(test).startswith(f"test_{debug_test}"):
            debug_test = test
        else:
            all_tests.append(test)

    for t in all_tests:
        debug_test_file = debug_test_dir / str(t)
        if not debug_test_file.is_file():
            new_test_suite = unittest.TestSuite()
            new_test_suite.addTest(debug_test)
            new_test_suite.addTest(t)
            with open(debug_test_file, "w") as f:
                runner = unittest.TextTestRunner(stream=f)
                runner.run(new_test_suite)

    return 0


@click.command(
    help=(
        "If some tests only fail, when the complete test suite is run "
        "this script discovers all tests inside the directory it is placed in "
        "and runs combinations of two test against each other."
    )
)
@click.argument(
    "debug-test",
    type=str,
    required=True,
)
def main(
    debug_test: str,
) -> int:
    main_unittests(debug_test=debug_test)
    return 0


if __name__ == "__main__":
    main()

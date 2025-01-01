#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tests/run_two_tests.py
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
import importlib
import inspect
import os
import unittest

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


def main_unittests(
    test1: str,
    test2: str,
) -> None:
    full_test1 = getattr(
        importlib.import_module(test1.split(".")[0]), test1.split(".")[1]
    )
    for method_name, _ in inspect.getmembers(full_test1, predicate=inspect.isfunction):
        if method_name.startswith("test_"):
            if method_name != test1.split(".")[-1]:
                delattr(full_test1, method_name)

    full_test2 = getattr(
        importlib.import_module(test2.split(".")[0]), test2.split(".")[1]
    )
    for method_name, _ in inspect.getmembers(full_test2, predicate=inspect.isfunction):
        if method_name.startswith("test_"):
            if method_name != test2.split(".")[-1] and method_name != "test_session":
                delattr(full_test2, method_name)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test1 = loader.loadTestsFromTestCase(full_test1)
    test2 = loader.loadTestsFromTestCase(full_test2)
    suite.addTests([test1, test2])
    runner = unittest.TextTestRunner()
    result = runner.run(suite)


@click.command(
    help=(
        "Use this script to run two tests together as a test suite. Tests "
        "are named as test_filename.TestClass.test_method"
    )
)
@click.argument(
    "test1",
    type=str,
    required=True,
)
@click.argument(
    "test2",
    type=str,
    required=True,
)
def main(test1: str, test2: str) -> int:
    main_unittests(test1, test2)
    return 0


if __name__ == "__main__":
    main()

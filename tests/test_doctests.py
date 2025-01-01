# -*- coding: utf-8 -*-
# tests/test_doctests.py
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
"""Available TestSuites:
    * TestsDocs:
        A test-suite, that searches for python code in *.py, *.rst, and *.md files
        and runs them via the doctest module.

"""
################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import doctest
import os
import unittest
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

# Third Party Imports
import toml


################################################################################
# Doctests
################################################################################


def doctest_files(
    start: Path,
    suffixes: Iterable[str] = (".py",),
) -> list[Path]:
    pyproject_file = start / "pyproject.toml"
    with open(pyproject_file, "r") as f:
        pyproject_data = toml.loads(f.read())
    omit = pyproject_data["tool"]["coverage"]["report"]["omit"]

    # collect files
    files = set()
    for suffix in suffixes:
        files.update(set(start.rglob(f"*{suffix}")))
    for o in omit:
        if "*" in o:
            if o.startswith("*") and o.endswith("*"):
                files = set(filter(lambda x: o.replace("*", "") not in str(x), files))
            else:
                to_remove = files.intersection(set((start / o.rstrip("/*")).rglob("*")))
                if len(to_remove) == 0:
                    continue
                files = files.difference(to_remove)
        files.discard(str(start / o))
    files.discard(start / "setup.py")
    return files


################################################################################
# Collect Test Cases and Filter
################################################################################


def load_tests(loader, tests, ignore):
    found_tests = []
    start = Path(__file__).resolve().parent.parent
    if os.getenv("ENCODERMAP_SKIP_DOCTESTS", "False").lower() == "true":
        return tests
    for file in doctest_files(start):
        file = str(file.relative_to(start)).replace(".py", "").replace("/", ".")
        found_tests.append(doctest.DocTestSuite(file))
    tests.addTests(found_tests)
    return tests


################################################################################
# Main
################################################################################


if __name__ == "__main__":
    unittest.main()

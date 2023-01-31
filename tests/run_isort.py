#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tests/run_isort.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2022 University of Konstanz and the Authors
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
"""Small script that runs isort on all .py files starting from project root (..)."""


from pathlib import Path

import isort

from test_documentation import TestDocumentation


def run_black():
    # collect files in encodermap
    project_root = Path("..")
    files_to_check = project_root.glob("**/*.py")
    filter_func = lambda x: not any(
        [str(x).startswith(excl) for excl in TestDocumentation.EXCLUDE_DIRS]
    )
    files_to_check = list(filter(filter_func, files_to_check))

    # run isort
    for file in files_to_check:
        res = isort.code(file.read_text())
        file.write_text(res)


if __name__ == "__main__":
    run_black()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encodermap/tests/find_long_comments.py
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
import random
from pathlib import Path
from typing import Optional

# Third Party Imports
import click

# Encodermap imports
import encodermap as em
from test_project_structure import find_long_comments


@click.command()
@click.option("--start-dir", default=None)
@click.option(
    "--n-print",
    default=5,
    help=(
        f"How many missing docstrings should be printed. Set to -1 to print all "
        f"missing docstrings. Defaults to 5."
    ),
)
def main(
    start_dir: Optional[str] = None,
    n_print: int = 5,
) -> int:
    if start_dir is None:
        start_dir = em.__file__
    start_dir = Path(start_dir).resolve()
    py_files = start_dir.parent.rglob("*.py")
    py_files = list(filter(lambda x: "tf1" not in str(x), py_files))
    exclude_files = ["utils.py"]
    long_comments = find_long_comments(py_files, exclude_files=exclude_files)
    print(f"Found {len(long_comments)} missing documentations.")
    if n_print == -1:
        for m in range(n_print):
            print(long_comments[m])
    else:
        if len(long_comments) < n_print:
            for m in long_comments:
                print(m)
        else:
            for m in random.sample(long_comments, n_print):
                print(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

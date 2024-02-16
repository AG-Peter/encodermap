#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encodermap/tests/delete_run_directories.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
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
import shutil
from pathlib import Path

# Third Party Imports
from git import Repo


def sort_fn(path: Path) -> bool:
    return path.is_dir() and path.stem == "runs"


if __name__ == "__main__":
    # Encodermap imports
    import encodermap as em

    project_dir = Path(em.__file__).resolve().parent.parent
    repo = Repo(project_dir / ".git")
    tracked_dirs = []
    for entry in repo.commit().tree.traverse():
        tracked_dirs.append(Path(entry.abspath).parent)
    tracked_dirs = set(tracked_dirs)
    run_dirs = list(filter(sort_fn, project_dir.glob("**")))
    if input(f"Deleting these directories: {run_dirs}? [y/N]").lower() in [
        "y",
        "yes",
        "ye",
    ]:
        for rd in run_dirs:
            if rd in tracked_dirs:
                print(f"Won't delete {rd}, because it contains tracked files.")
            else:
                print(f"Deleting {rd}...")
                shutil.rmtree(rd)
    else:
        print("Aborting.")

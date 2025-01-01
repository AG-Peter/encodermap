#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encodermap/tests/delete_run_directories.py
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
    tracked_files = []
    for entry in repo.commit().tree.traverse():
        tracked_dirs.append(Path(entry.abspath).parent)
        tracked_files.append(Path(entry.abspath))
    tracked_dirs = set(tracked_dirs)
    run_dirs = list(filter(sort_fn, project_dir.glob("**")))
    q = "\n".join(map(str, run_dirs))
    if input(f"Deleting these directories:\n\n{q}\n\n[y/N]?").lower() in [
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
        print("Aborting. Deleting runs.")

    # delete summaries
    model_summaries_txt = list(
        filter(
            lambda x: "finished_training" not in str(x),
            project_dir.rglob("*_summary.txt"),
        )
    )
    q = "\n".join(map(str, model_summaries_txt))
    if input(f"Deleting these model_summary files:\n\n{q}\n\n[y/N]?").lower() in [
        "y",
        "yes",
        "ye",
    ]:
        for file in model_summaries_txt:
            if file in tracked_files:
                print(f"Won't delete {file}, because it is tracked in git.")
            else:
                print(f"Deleting {file}...")
                file.unlink()

    # delete parameters.json
    parameters_files = list(project_dir.rglob("parameters*json"))
    parameters_files = list(
        filter(
            lambda x: not (
                "finished_training" in str(x) and x.name == "parameters.json"
            ),
            parameters_files,
        )
    )
    q = "\n".join(map(str, parameters_files))
    if input(f"Deleting these parameters.json files:\n\n{q}n\n\[y/N]?").lower() in [
        "y",
        "yes",
        "ye",
    ]:
        for file in parameters_files:
            if file in tracked_files:
                print(f"Won't delete {file}, because it is tracked in git.")
            else:
                print(f"Deleting {file}...")
                file.unlink()

    # delete model.keras
    keras_files = list(project_dir.rglob("saved_model*keras"))
    keras_files = list(
        filter(
            lambda x: not ("finished_training" in str(x)),
            keras_files,
        )
    )
    q = "\n".join(map(str, keras_files))
    if input(f"Deleting these saved_model*keras files:\n\n{q}\n\n[y/N]?").lower() in [
        "y",
        "yes",
        "ye",
    ]:
        for file in keras_files:
            if file in tracked_files:
                print(f"Won't delete {file}, because it is tracked in git.")
            else:
                print(f"Deleting {file}...")
                file.unlink()

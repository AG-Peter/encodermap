# -*- coding: utf-8 -*-
# tests/test_notebooks.py
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
    * TestNotebooks:
        A test-suite, that searches for .ipynb files in the tutorials direcotry
        and checks whether they successfully run.
        Set the environment variable EMAP_ONLY_NOTEBOOK to a filename (with .ipynb)
        extension to only run this notebook.

"""

# Standard Library Imports
import json
import os
import unittest
from pathlib import Path
from typing import Literal

# Third Party Imports
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

# Encodermap imports
import encodermap as em
from conftest import skip_all_tests_except_env_var_specified
from run_docbuild_test_and_cover import SkipExecutionPreprocessor


################################################################################
# Imports
################################################################################


################################################################################
# TestCases
################################################################################


class CollectNotebooks(type):
    def __new__(cls, name, bases, attrs):
        notebook_files = (Path(em.__file__).parent.parent / "tutorials").rglob(
            "*.ipynb"
        )
        notebook_files = list(
            filter(
                lambda f: not any(
                    "notebooks_tensorflow1" in p or ".ipynb_checkpoints" in p
                    for p in f.parts
                ),
                notebook_files,
            ),
        )
        if (nbf := os.getenv("EMAP_ONLY_NOTEBOOK", None)) is not None:
            notebook_files = list(filter(lambda x: x.name == nbf, notebook_files))

        cwd = os.getcwd()
        for i, notebook_file in enumerate(notebook_files):

            # check whether notebook contains "emap": "run" metadata
            # Otherwise, this notebook is not part of the test suite, and we
            # will continue
            with open(notebook_file) as f:
                json_data = json.load(f)
            if json_data["metadata"].get("emap", "skip") != "run":
                print(
                    f"Notebook {notebook_file} is not part of doc/test suite. "
                    f"To add this notebook to the test suite, add the "
                    "{'emap': 'run'} noteboook metadata."
                )
                continue

            # Dynamically create a function
            def test_notebook(self):
                # decide on exporter
                if self.exporter == "SkipExecution":
                    exporter = HTMLExporter()
                    exporter.register_preprocessor(SkipExecutionPreprocessor(), True)
                elif self.exporter == "none":
                    exporter = HTMLExporter()
                    exporter.register_preprocessor(ExecutePreprocessor(), True)
                else:
                    raise Exception(
                        f"`self.exporter` needs to be one of 'SkipExecution' "
                        f"or 'none', but the class-attribute is set to "
                        f"`{self.exporter}`."
                    )
                try:
                    os.chdir(notebook_file.parent)
                    nb_html, _ = exporter.from_filename(notebook_file)
                    with open("/tmp/tmp.html", "w") as f:
                        f.write(nb_html)
                except Exception as e:
                    self.fail(
                        msg=(
                            f"Could not execute {notebook_file}, because an "
                            f"exception was raised:\n{e}\n. To mark cells for "
                            f"allowing errors look into EncoderMap's "
                            f"CONTRIBUTING.md document to see how to mark "
                            f"notebook cells as expected to fail.\nThe "
                            f"notebooks that were successfully run until "
                            f"now are:\n{self.successful_notebooks}"
                        )
                    )
                except KeyboardInterrupt as e:
                    self.fail(
                        msg=(
                            f"Could not execute {notebook_file}, because an "
                            f"exception was raised:\n{e}\n. To mark cells for "
                            f"allowing errors look into EncoderMap's "
                            f"CONTRIBUTING.md document to see how to mark "
                            f"notebook cells as expected to fail.\nThe "
                            f"notebooks that were successfully run until "
                            f"now are:\n{self.successful_notebooks}"
                        )
                    )
                else:
                    print(f"Notebook ran through. Saved at /tmp/tmp.html")
                    self.successful_notebooks.append(notebook_file)
                finally:
                    os.chdir(cwd)

            # set the function as an attribute of the class
            attrs[f"test_notebook_{notebook_file.stem}"] = test_notebook

        # return the class
        return super().__new__(cls, name, bases, attrs)


class TestNotebooks(unittest.TestCase, metaclass=CollectNotebooks):
    """The SkipExecution preprocessor is defined in `run_docbuild_test_and_cover`."""

    exporter: Literal["SkipExecution", "none"] = "SkipExecution"

    @classmethod
    def setUpClass(cls):
        cls.successful_notebooks = []


################################################################################
# Collect Test Cases and Filter
################################################################################


def load_tests(loader, tests, pattern):
    test_cases = (TestNotebooks,)
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite


################################################################################
# Main
################################################################################


if __name__ == "__main__":
    unittest.main()

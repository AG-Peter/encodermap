# -*- coding: utf-8 -*-
# tests/test_documentation.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2022 University of Konstanz and the Authors
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
"""encodermap documentation tests

Only one test suite available here: TestDocumentation:
sifts through all files in the encodermap package and alerts to the following:
* Missing headers for files
* Missing documentation for modules/classes/methods/functions

"""


################################################################################
# Imports
################################################################################


import ast
import os
import unittest
from pathlib import Path

################################################################################
# Globals
################################################################################


LICENSE_HEADER1 = """\
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2022 University of Konstanz and the Authors
#\
"""
LICENSE_HEADER2 = """\
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
################################################################################\
"""


CODING_HEADER = "# -*- coding: utf-8 -*-"


TYPE_INFERRED_ASSIGNMENTS = (ast.Call, ast.JoinedStr, ast.BinOp)


################################################################################
# Test suites
################################################################################


class TestDocumentation(unittest.TestCase):
    EXCLUDE_DIRS = [
        Path(f"{__file__}/../../development").resolve(),
        Path(f"{__file__}/../../docs").resolve(),
        Path(f"{__file__}/../../encodermap/encodermap_tf1").resolve(),
        Path(f"{__file__}/../../encodermap/examples").resolve(),
    ]
    TEST_DIRS = [Path(f"{__file__}/../../tests").resolve()]
    EXCLUDE_FILES = [Path(f"{__file__}/../../setup.py").resolve()]
    EXCLUDE_CONTENT_HEADERS = [
        Path(f"{__file__}/../../encodermap/_optional_imports.py").resolve(),
        Path(f"{__file__}/../../encodermap/misc/transformations.py").resolve(),
    ]

    @classmethod
    def setUpClass(cls) -> None:
        """Collect all possible files in encodermap, tests and encodermap/examples"""
        cls.project_root = Path(f"{__file__}/../..").resolve()
        cls.files_to_check = list(cls.project_root.glob("**/*.py"))
        cls.files_to_check = [file.resolve() for file in cls.files_to_check]

    @unittest.skip
    def test_type_declarations(self):
        missing_type_info = []
        for file in self.files_to_check:
            if any([excl in file.parents for excl in self.EXCLUDE_DIRS]):
                continue
            elif file.absolute() in self.EXCLUDE_FILES:
                continue
            file = Path(
                "/mnt/data/kevin/encoder_map_private/encodermap/_optional_imports.py"
            )
            content = file.read_text().splitlines()
            module = ast.parse(file.read_text())

            for node in ast.walk(module):
                # functions
                if isinstance(node, ast.FunctionDef):
                    for arg in node.args.args:
                        if arg.annotation is None:
                            if arg.arg == "self":
                                continue
                            msg = (
                                f"The function `{node.name}` in File "
                                f'"{file.resolve()}:{node.lineno}" is missing '
                                f"type hints for the `{arg.arg}` argument. "
                                "Please add it."
                            )
                            missing_type_info.append(msg)

                    if node.returns is None:
                        msg = (
                            f"The function `{node.name}` in File "
                            f'"{file.resolve()}:{node.lineno}" is missing '
                            f"type hints for the returned value. "
                            "Please add it."
                        )
                        missing_type_info.append(msg)

                # assignments
                # the ast.AnnAssign node type already is typed
                # only need to filter out the ones having function calls
                if isinstance(node, ast.Assign):
                    targets = tuple(t.id for t in node.targets)
                    # if not isinstance(assignment.value, ast.Call):
                    if not isinstance(node.value, TYPE_INFERRED_ASSIGNMENTS):
                        msg = (
                            f"The assigment `{content[node.lineno - 1].strip()}` in File "
                            f'"{file.resolve()}:{node.lineno}" is missing '
                            f"type hints for the returned value. The ast type of "
                            f"value is {node.value}."
                        )
                        missing_type_info.append(msg)

                # print(type(node))

            # for i, node in enumerate(module.body):
            #     print(node)
            #     print(dir(node))
            #     if i == 10:
            #         break
            # print(module)
            # print(dir(module))
            break
        self.assertFalse(bool(missing_type_info), msg="\n\n".join(missing_type_info))

    def test_coding_headers(self):
        files = []
        for file in self.files_to_check:
            if any([excl in file.parents for excl in self.EXCLUDE_DIRS]):
                continue
            elif file.absolute() in self.EXCLUDE_FILES:
                continue
            else:
                content = file.read_text().splitlines()
                try:
                    msg = (
                        f"The file {file} does not follow the styleguide for "
                        f"encodermap. The first lines reads {content[:2]}. "
                        f"Make sure, the file starts with a line "
                        f"reading: '{CODING_HEADER}'. For scripts, this line "
                        f"can be the second line."
                    )
                    if content[0].startswith("#!/usr/bin/"):
                        if content[1] != "# -*- coding: utf-8 -*-":
                            files.append(msg)
                    else:
                        if content[0] != CODING_HEADER:
                            files.append(msg)
                except IndexError:
                    msg = (
                        f"The file {file} does not follow the styleguide for "
                        f"encodermap. The file is empty. "
                        f"Make sure, the file starts with a line "
                        f"reading: '{CODING_HEADER}'. For scripts, this line "
                        f"can be the second line."
                    )
                    files.append(msg)
        self.assertFalse(bool(files), msg="\n\n".join(files))

    def test_filenames(self):
        files = []
        for file in self.files_to_check:
            if any([excl in file.parents for excl in self.EXCLUDE_DIRS]):
                continue
            elif file.absolute() in self.EXCLUDE_FILES:
                continue
            else:
                content = file.read_text().splitlines()
                file_id = "# " + str(file.relative_to(self.project_root.resolve()))
                if os.name == "nt":
                    file_id = file_id.replace("\\", "/")
                try:
                    msg = (
                        f"The file {file} does not follow the styleguide for "
                        f"encodermap. The first lines reads {content[:3]}. "
                        f"Make sure, the 2nd line in a file contains the name "
                        f"of the file: '{file_id}'. For scripts, this line "
                        f"can be the third line."
                    )
                    if content[0].startswith("#!/usr/bin/"):
                        if content[2] != file_id:
                            files.append(msg)
                    else:
                        if content[1] != file_id:
                            files.append(msg)
                except IndexError:
                    msg = (
                        f"The file {file} does not follow the styleguide for "
                        f"encodermap. The file is empty. "
                        f"Make sure, the file starts with a line "
                        f"reading: '{file_id}'. For scripts, this line "
                        f"can be the third line."
                    )
                    files.append(msg)
        self.assertFalse(bool(files), msg="\n\n".join(files))

    def test_license_info(self):
        files = []
        for file in self.files_to_check:
            if any([excl in file.parents for excl in self.EXCLUDE_DIRS]):
                continue
            elif file.absolute() in self.EXCLUDE_FILES + self.EXCLUDE_CONTENT_HEADERS:
                continue
            else:
                content = file.read_text()
                if LICENSE_HEADER1 not in content or LICENSE_HEADER2 not in content:
                    msg = (
                        f"The file {file} does not contain the standard "
                        f"encodermap content header. Please add it to the file."
                    )
                    files.append(msg)
        self.assertFalse(bool(files), msg="\n\n".join(files))

    def test_complete_docstrings(self):
        missing_docstrings = []
        for file in self.files_to_check:
            if any(
                [excl in file.parents for excl in self.EXCLUDE_DIRS + self.TEST_DIRS]
            ):
                continue
            elif file.absolute() in self.EXCLUDE_FILES + self.EXCLUDE_CONTENT_HEADERS:
                continue
            else:
                module = ast.parse(file.read_text())
                module_docstring = ast.get_docstring(module)
                if module_docstring is None:
                    msg = f"The module {file} is missing its docstring. Please add it."
                    missing_docstrings.append(msg)

                # functions
                function_defs = [
                    node for node in module.body if isinstance(node, ast.FunctionDef)
                ]
                func_missing_docs = [
                    node for node in function_defs if ast.get_docstring(node) is None
                ]
                for func in func_missing_docs:
                    msg = (
                        f"The function `{func.name}` in File "
                        f'"{file.resolve()}:{func.lineno}" is missing its '
                        f"docstring. Please add it."
                    )
                    missing_docstrings.append(msg)

                # classes
                class_defs = [
                    node for node in module.body if isinstance(node, ast.ClassDef)
                ]
                class_missing_docs = [
                    node for node in class_defs if ast.get_docstring(node) is None
                ]
                for clas in class_missing_docs:
                    msg = (
                        f"The class `{clas.name}` in File "
                        f'"{file.resolve()}:{clas.lineno}" is missing its '
                        f"docstring. Please add it."
                    )
                    missing_docstrings.append(msg)

                # methods
                method_defs = [
                    node
                    for clas in class_defs
                    for node in clas.body
                    if isinstance(node, ast.FunctionDef)
                ]
                method_missing_docs = [
                    node for node in method_defs if ast.get_docstring(node) is None
                ]
                for method in method_missing_docs:
                    if method.decorator_list:
                        if any(
                            [
                                decorator.attr == "setter"
                                for decorator in method.decorator_list
                                if isinstance(decorator, ast.Attribute)
                            ]
                        ):
                            continue
                    if method.name.startswith("__") and method.name != "__init__":
                        continue
                    msg = (
                        f"The method `{method.name}` in File "
                        f'"{file.resolve()}:{method.lineno}" is missing its '
                        f"docstring. Please add it."
                    )
                    missing_docstrings.append(msg)

        self.assertFalse(bool(missing_docstrings), msg="\n\n".join(missing_docstrings))


test_cases = (TestDocumentation,)
test_cases = tuple()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite

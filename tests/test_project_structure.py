# -*- coding: utf-8 -*-
# tests/test_project_structure.py
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
"""encodermap documentation tests

Only one test suite available here: TestDocumentation:
sifts through all files in the encodermap package and alerts to the following:
* Missing headers for files
* Missing documentation for modules/classes/methods/functions

"""


################################################################################
# Imports
################################################################################


# Standard Library Imports
import ast
import os
import re
import unittest
from pathlib import Path

# Third Party Imports
from docstring_parser import parse as docparse
from pydoctest.main import PyDoctestService, get_configuration, get_reporter


################################################################################
# Globals
################################################################################


LICENSE_HEADER1 = """\
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
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


_directive_regex = re.compile(r"\.\. \S+::")


################################################################################
# Test suites
################################################################################


def check_missing_docstrings(
    files_to_check,
    exclude_files=None,
    exclude_dirs=None,
    exclude_class_decorators=None,
) -> list[str]:
    if exclude_files is None:
        exclude_files = ["_version.py", "transformations.py"]
    else:
        exclude_files.extend(["_version.py", "transformations.py"])
    if exclude_dirs is None:
        exclude_dirs = []
    if exclude_class_decorators is None:
        exclude_class_decorators = ["testing"]
    else:
        exclude_class_decorators = [*exclude_class_decorators, "testing"]
    missing_docstrings = []
    missing_args_and_returns = []
    for file in files_to_check:
        if any([excl in file.parents for excl in exclude_dirs]):
            continue
        elif file.absolute() in exclude_files or file.name in exclude_files:
            continue
        else:
            module = ast.parse(file.read_text())
            module_docstring = ast.get_docstring(module)
            if module_docstring is None:
                msg = f'The module "{file}" is missing its docstring. Please add it.'
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

            # find functions which docstrings are incomplete
            for func in function_defs:
                all_args = func.args.posonlyargs + func.args.args + func.args.kwonlyargs
                arg_names = [a.arg for a in all_args]
                if len(all_args) == 0:
                    continue
                docstring = ast.get_docstring(func)
                if docstring is None:
                    continue
                docstring = docparse(docstring)
                if undocumented_args := (
                    set(arg_names) - set([a.arg_name for a in docstring.params])
                ):
                    for i in list(undocumented_args):
                        missing_args_and_returns.append(
                            f"The argument {i} in function `{func.name}` in file "
                            f'"{file.resolve()}:{func.lineno}" is not present  '
                            f"in the function's docstring. Please add it."
                        )
                if not isinstance(func.returns, ast.Constant):
                    if docstring.returns is None:
                        missing_args_and_returns.append(
                            f"The 'Return' definition in function `{func.name}` "
                            f'in file "{file.resolve()}:{func.lineno}" is not present  '
                            f"in the function's docstring. Please add it."
                        )

            # classes
            class_defs = [
                node for node in module.body if isinstance(node, ast.ClassDef)
            ]
            class_missing_docs = [
                node for node in class_defs if ast.get_docstring(node) is None
            ]
            for clas in class_missing_docs:
                decorator_names = [d.id for d in clas.decorator_list]
                if any([i in decorator_names for i in exclude_class_decorators]):
                    class_missing_docs.remove(clas)
                else:
                    msg = (
                        f"The class `{clas.name}` in file "
                        f'"{file.resolve()}:{clas.lineno}" is missing its '
                        f"docstring. Please add it."
                    )
                    missing_docstrings.append(msg)

            # methods
            method_defs = [
                (node, clas)
                for clas in class_defs
                for node in clas.body
                if isinstance(node, ast.FunctionDef)
            ]
            method_missing_docs = [
                node for node in method_defs if ast.get_docstring(node[0]) is None
            ]
            for method, clas in method_missing_docs:
                decorator_names = [
                    d.id if isinstance(d, ast.Name) else d.func.attr
                    for d in clas.decorator_list
                ]
                if method.decorator_list:
                    if any(
                        [
                            decorator.attr == "setter"
                            for decorator in method.decorator_list
                            if isinstance(decorator, ast.Attribute)
                        ]
                    ) or any([i in decorator_names for i in exclude_class_decorators]):
                        continue
                if method.name.startswith("__") and method.name != "__init__":
                    continue
                # filter overloaded functions
                if len(method.body) == 1 and hasattr(method.body[0], "value"):
                    if hasattr(method.body[0].value, "value"):
                        if method.body[0].value.value == ...:
                            continue
                msg = (
                    f"The method `{method.name}` in file "
                    f'"{file.resolve()}:{method.lineno}" is missing its '
                    f"docstring. Please add it."
                )
                missing_docstrings.append(msg)
    return missing_docstrings + missing_args_and_returns


class TestDocumentation(unittest.TestCase):
    EXCLUDE_DIRS = [
        Path(f"{__file__}/../../development").resolve(),
        Path(f"{__file__}/../../docs").resolve(),
        Path(f"{__file__}/../../encodermap/encodermap_tf1").resolve(),
        Path(f"{__file__}/../../encodermap/examples").resolve(),
    ]
    TEST_DIRS = [Path(f"{__file__}/../../tests").resolve()]
    EXCLUDE_FILES = [
        Path(f"{__file__}/../../setup.py").resolve(),
        Path(f"{__file__}/../../versioneer.py").resolve(),
    ]
    EXCLUDE_CONTENT_HEADERS = [
        Path(f"{__file__}/../../encodermap/_optional_imports.py").resolve(),
        Path(f"{__file__}/../../encodermap/misc/transformations.py").resolve(),
    ]

    @classmethod
    def setUpClass(cls) -> None:
        """Collect all possible files in encodermap, tests and encodermap/examples"""
        cls.project_root = Path(f"{__file__}/../..").resolve()
        cls.files_to_check = list(cls.project_root.rglob("*.py"))
        cls.files_to_check = [file.resolve() for file in cls.files_to_check]
        cls.test_files = list(Path(__file__).resolve().parent.glob("*.py"))

    @unittest.skip
    def test_type_declarations(self):
        """Test whether all functions have type declarations."""
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
        """Test whether all files have -*- coding: utf-8 -*- in the 1st or 2nd line."""
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
        """Test whether all files have their filename in the 2nd or 3rd line.
        1st line is either encoding info or shebang, 2nd line is either file name
        or encoding."""
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
        """Test whether all files have a license header in them."""
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
        """Test, whether we have a good coverage of docstrings"""
        missing_docstrings = check_missing_docstrings(
            self.files_to_check,
            self.EXCLUDE_FILES + self.EXCLUDE_CONTENT_HEADERS,
            self.EXCLUDE_DIRS + self.TEST_DIRS,
        )
        self.assertFalse(bool(missing_docstrings), msg="\n\n".join(missing_docstrings))

    def test_correct_docstrings(self):
        """Test whether all args are included in docstrings."""
        # instantiate the config
        config = get_configuration(
            str(Path(f"{__file__}/../../").resolve()),
        )
        config.exclude_paths.extend(
            [
                "development/*",
                "tests/*",
            ]
        )
        config.fail_on_missing_docstring = True
        config.fail_on_missing_summary = True

        # instantiate the report
        reporter = get_reporter(config, "text")
        ds = PyDoctestService(config)
        result = ds.validate()
        counts = result.get_counts()
        output = reporter.get_output(result)
        print(output)
        self.assertEqual(counts.functions_failed, 0)

    def test_test_files(self):
        """Make sure all tests have some docstrings outlining what will be tested."""
        missing_docstrings = check_missing_docstrings(
            self.test_files,
        )
        self.assertFalse(bool(missing_docstrings), msg="\n\n".join(missing_docstrings))


test_cases = (TestDocumentation,)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    return suite

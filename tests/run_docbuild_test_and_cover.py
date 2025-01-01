#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tests/run_docbuild_tests_and_cover.py
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
import datetime
import json
import logging
import os
import pkgutil
import shutil
import sys
import typing as t
import unittest
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Third Party Imports
import click
import coverage
import git
import HtmlTestRunner
import nbformat
from dateutil.parser import parse as dateparse
from git import Repo
from jupyter_client import KernelManager
from nbconvert.exporters import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import NotebookNode


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


COVERAGE_BADGE_JSON = {
    "schemaVersion": 1,
    "label": "Coverage",
    "message": "76%",
    "color": "orange",
    "namedLogo": "Codecov",
}


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


@contextmanager
def delete_files_contextmanager(logger: logging.Logger) -> None:
    files = []
    try:
        yield files
    finally:
        for file in files:
            logger.warning(f"Removing {file}")
            Path(file).unlink()


class SkipExecutionPreprocessor(ExecutePreprocessor):
    def preprocess(
        self,
        nb: NotebookNode,
        resources: t.Any = None,
        km: KernelManager | None = None,
    ) -> tuple[NotebookNode, dict[str, t.Any]]:
        self.start = datetime.now()
        nb, resources = super().preprocess(nb=nb, resources=resources, km=km)
        out_cells = []
        for i, c in enumerate(nb.cells):
            if c.cell_type == "code":
                if "emap" in c.metadata:
                    if c.metadata["emap"] == "hidden":
                        del c
                        continue
            out_cells.append(c)
        nb.cells = out_cells
        return nb, resources

    def preprocess_cell(self, cell, resources, index):
        """Four cases for this processor:

        * ["metadata"]["emap"] = "hidden" -> execute cell, but don't show
        * ["metddata"]["emap"] = "skip" -> don't execute cell, but show
        * ["metadata"]["emap"] = "clear_output" clears the output of a certain cell.
        * ["metadata"]["emap"] = "allow_error"

        """
        if not cell.cell_type == "code":
            return cell, resources
        line1 = cell["source"].split("\n")[0]
        # print(
        #     f"Executing code cell, with metadata keys: {cell.metadata.keys()}, "
        #     f"which starts with:\n>>> {line1}"
        # )
        if "emap" not in cell.metadata:
            out = super().preprocess_cell(cell, resources, index)
        else:
            print(f"Emap metadata: {cell.metadata['emap']=}")
            if cell.metadata["emap"] == "skip":
                return cell, resources
            elif cell.metadata["emap"] == "allow_error":
                cell.metadata.setdefault("tags", []).append("raises-exception")
                cell, resources = super().preprocess_cell(cell, resources, index)
                out = (cell, resources)
            elif cell.metadata["emap"] == "clear_output":
                cell, resources = super().preprocess_cell(cell, resources, index)
                cell["outputs"] = []
                out = (cell, resources)
            elif cell.metadata["emap"] == "hidden":
                out = super().preprocess_cell(cell, resources, index)
            else:
                out = super().preprocess_cell(cell, resources, index)
        delta = (datetime.now() - self.start).total_seconds()
        # print(f"Execution ran for {delta} seconds.\n")
        return out


def sort_tests(tests):
    for test in tests:
        try:
            if test._testMethodName == "test_losses_not_periodic":
                return 1
        except AttributeError:
            for t in test._tests:
                try:
                    if t._testMethodName == "test_losses_not_periodic":
                        return 1
                except AttributeError:
                    for _ in t._tests:
                        if _._testMethodName == "test_losses_not_periodic":
                            return 1
    return 2


def unpack_tests(tests):
    for test in tests:
        try:
            print(test._testMethodName)
        except AttributeError:
            for t in test._tests:
                try:
                    print(t._testMethodName)
                except AttributeError:
                    for _ in t._tests:
                        print(_._testMethodName)


def filter_key_test_suites(suite):
    """Returns filters True/False depending on whether a
    test needs to be executed before or after the optional
    requirements are installed"""
    if "before_installing_reqs" in suite.__str__():
        return False
    else:
        return True


class SortableSuite(unittest.TestSuite):
    def sort(self):
        self._tests = list(sorted(self._tests, key=sort_tests))

    def all_tests(self, logger):
        out = []

        # Standard Library Imports
        import inspect

        for suite in self._tests:
            if "1am7" in str(suite):
                for testcase in suite._tests:
                    break
                raise Exception(f"Gotcha {testcase=}")
            for testcase in suite:
                for test in filter(
                    lambda x: x[0].startswith("test_"),
                    inspect.getmembers(testcase, predicate=inspect.ismethod),
                ):
                    out.append(test[0])
        return out

    def get_test_combinations(self, test_name, logger):
        tests = self.all_tests(logger)
        raise Exception(tests)
        test_id = tests.index(f"test_{test_name}")
        raise Exception(f"{test_id=}")

    def filter(self):
        self._tests = list(filter(filter_key_test_suites, self._tests))


def fix_html_test_runner(logger: logging.Logger) -> None:
    html_test_runner_dir = pkgutil.get_loader("HtmlTestRunner").get_filename()
    file = Path(html_test_runner_dir).parent.resolve() / "result.py"
    assert file.is_file(), (
        f"The file {file} is not present. Can't run Unittests and report results "
        f"without this package."
    )

    if BAD_STR not in file.read_text():
        logger.info("HtmlTestRunner package has already been fixed. Continuing.")
    else:
        new_text = file.read_text().replace(BAD_STR, GOOD_STR)
        file.write_text(new_text)
        logger.info("Fixed HtmlTestRunner.")


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def get_logger() -> logging.Logger:
    name = "EncoderMap Tests/Docs"
    logger = logging.getLogger(name)
    logger.propagate = True
    _add_logger_id_factory(logger)
    return logger


def _get_default_formatter() -> logging.Formatter:
    # format = ('[time:[%(asctime)s] file:[%(filename)s %(lineno)s '
    #           '%(funcName)s] %(name)s %(levelname)s]: %(message)s')
    format = (
        '%(name)s %(levelname)8s [%(asctime)s] ["%(pathname)s:'
        '%(lineno)s", in %(funcName)s]: %(message)s'
    )
    # "{file.resolve()}:{func.lineno}"
    logFormatter = logging.Formatter(format, datefmt="%Y-%m-%dT%H:%M:%S%z")
    return logFormatter


def _get_console_formatter() -> logging.Formatter:
    format = "\033[92m%(name)s %(logger_id)s %(levelname)8s [%(asctime)s]: %(message)s\033[0m"
    logFormatter = logging.Formatter(format, datefmt="%Y-%m-%dT%H:%M:%S%z")
    return logFormatter


def _logger_add_file_handler(
    logger: logging.Logger, logfile: str, loglevel: int = logging.DEBUG
) -> None:
    logFormatter = _get_default_formatter()
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(loglevel)
    logger.addHandler(fileHandler)


def _logger_add_console_handler(
    logger: logging.Logger, loglevel: int = logging.WARNING
) -> None:
    logFormatter = _get_console_formatter()
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(loglevel)
    logger.addHandler(consoleHandler)


def _add_logger_id_factory(logger: logging.Logger) -> None:
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.logger_id = f"<0x{id(logger):02x}>"
        return record

    logging.setLogRecordFactory(record_factory)


def set_up_logger(
    logfile: Union[str, Path],
    log_to_console: bool = False,
) -> logging.Logger:
    file_loglevel = logging.DEBUG
    console_loglevel = logging.INFO
    backed_up = False
    name = "EncoderMap Tests/Docs"
    # get the file from config and back up if needed
    if logfile.is_file() and os.stat(logfile).st_size > 0:
        with open(logfile) as f:
            content = list(
                filter(
                    lambda x: True if name in x else False,
                    f.read().splitlines(),
                )
            )
        last_line = content[-1]
        time_last_line = last_line[last_line.find("[") + 1 : last_line.find("]")]
        try:
            time_last_line = dateparse(time_last_line).date().replace(day=1)
        except Exception:
            print(content)
            raise
        this_month = datetime.date(datetime.today().replace(day=1))
        if time_last_line < this_month:
            backed_up = True
            backup_to = (
                os.path.split(logfile)[0]
                + "/"
                + time_last_line.strftime("%Y-%m")
                + "_"
                + os.path.split(logfile)[1]
            )
            shutil.move(logfile, backup_to)

    # set up the logger and its levels
    logger = logging.getLogger(name=name)
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # write to file
    if logfile is not None:
        _logger_add_file_handler(logger, logfile, loglevel=file_loglevel)

    # write to console
    if log_to_console:
        _logger_add_console_handler(logger, loglevel=console_loglevel)

    # set level
    logger.setLevel(min([file_loglevel, console_loglevel]))

    # and propagate
    logger.propagate = True

    # add new factory
    _add_logger_id_factory(logger)
    # logger = logging.LoggerAdapter(logger, extra={"logger_id": f"<0x{id(logger):02x}>"})
    # logger.addFilter(IdFilter(logger_id=f"<0x{id(logger):02x}>"))

    if backed_up:
        logger.info(
            f"Last write to old log file was {time_last_line}. "
            f"Backup was placed at {backup_to}"
        )

    return logger


def main_unittests(
    logger: logging.Logger,
    commit_hash: str,
    debug_test: Optional[str] = None,
) -> tuple[bool, float]:
    cov = coverage.Coverage(
        config_file=str(Path(__file__).resolve().parent.parent / "pyproject.toml")
    )
    cov.start()
    logger.info(f"Starting coverage: {cov}...")
    loader = unittest.TestLoader()
    logger.info(f"Starting Loader: {loader}...")
    # loader.suiteClass = SortableSuite
    start_dir = str(Path(__file__).resolve().parent)
    logger.info(f"Discovering tests in {start_dir=}")
    test_suite = loader.discover(
        start_dir=start_dir,
        top_level_dir=str(Path(__file__).resolve().parent.parent),
    )
    logger.info(f"Loader discovered {test_suite.countTestCases()} tests.")
    test_suite._tests = test_suite._tests[:2]
    # if any(
    #     [
    #         isinstance(test, unittest.loader._FailedTest)
    #         for suite in test_suite
    #         for test in suite._tests
    #     ]
    # ):
    #     add = Path(__file__).resolve().parent
    #     logger.info(f"Adding {add} to path.")
    #     sys.path.insert(0, str(add))
    #     loader = unittest.TestLoader()
    #     loader.suiteClass = MySuite
    #     test_suite = loader.discover(
    #         start_dir=str(Path(__file__).resolve().parent),
    #     )

    if debug_test is not None:
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
                logger.info(f"DEBUG TEST with {t=}")
                new_test_suite = unittest.TestSuite()
                new_test_suite.addTest(debug_test)
                new_test_suite.addTest(t)
                with open(debug_test_file, "w") as f:
                    runner = unittest.TextTestRunner(stream=f)
                    runner.run(new_test_suite)

        return 0

    # test_suite.filter()
    # runner = unittest.TextTestRunner()
    # result = runner.run(test_suite)
    # print("Unittest Result:", result.wasSuccessful())

    # output to a file
    # Encodermap imports
    from encodermap import __version__

    out_dir = Path(__file__).resolve().parent.parent / "docs/source/_static/coverage"
    html_report_name = "html_test_runner_report"
    logger.info(f"The coverage report will be at {out_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().replace(microsecond=0).isoformat()
    runner = HtmlTestRunner.HTMLTestRunner(
        output=str(out_dir.parent),
        report_title=f"EncoderMap Unittest Report from {now} with EncoderMap version {__version__}.",
        report_name=html_report_name,
        combine_reports=True,
        add_timestamp=False,
        buffer=True,
    )
    logger.info(
        f"The test results will be saved in {out_dir.parent / html_report_name}."
    )

    # run the test
    logger.info("Running the unittests now...")
    # with console.capture():
    result = runner.run(test_suite)
    cov.stop()
    logger.info(f"Saving coverage report to {out_dir}")
    cov_percentage = cov.html_report(
        directory=str(out_dir),
        title="coverage_report",
    )

    logger.info("Unittest Result:", result.wasSuccessful())
    logger.info("Coverage Percentage:", cov_percentage)

    return result.wasSuccessful(), cov_percentage


def main_sphinx(
    logger: logging.Logger,
    repo: git.Repo,
    force_notebook_rerun: bool = False,
) -> None:
    # Third Party Imports
    from sphinx.cmd.build import build_main as _sphinx_main

    logger.info("Starting Sphinx.")
    docs_dir = Path(__file__).parent.parent / "docs"
    doctrees = str(docs_dir / "build/doctrees")
    sourcedir = str(docs_dir / "source")
    htmldir = str(docs_dir / "build/html")
    argv = ["-b", "html", "-d", doctrees, sourcedir, htmldir]
    nblink_files = list(Path(sourcedir).rglob("*.nblink"))
    logger.info(f"Found {len(nblink_files)} .nblink files.")
    if force_notebook_rerun:
        logger.info(f"Will rerun all notebooks, because {force_notebook_rerun=}")
    else:
        logger.info(
            f"Will not rerun all notebooks. I will check each nblink file separately."
        )

    for nblink_file in nblink_files:
        if "nb_intermediate" not in str(nblink_file):
            continue

        with delete_files_contextmanager(logger) as files_to_delete:
            try:
                nblinbk_content = json.loads(nblink_file.read_text())
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Could not decode the nblink file: {nblink_file} due to wrong json."
                ) from e
            logger.info(f"{nblinbk_content=}")
            nb_file_copy_from = (
                nblink_file.parent / Path(nblinbk_content["path"])
            ).resolve()

            if not nb_file_copy_from.is_file():
                raise Exception(
                    f"The nblink file {nblink_file} requested the notebook file "
                    f"{nb_file_copy_from}, but this file does not exist."
                )

            # check whether this file should be part of the documentation
            try:
                with open(nb_file_copy_from, "r") as f:
                    json_data = json.load(f)
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Could not decode the nb file: {nb_file_copy_from} due to wrong json."
                ) from e
            if json_data["metadata"].get("emap", "skip") != "run":
                raise Exception(
                    f"The nblink file {nblink_file} requested the notebook file "
                    f"{nb_file_copy_from}, but this file does not have the "
                    "{'emap': 'run'} metadata."
                )

            nb_file = nblink_file.with_suffix(".ipynb")
            if not nb_file.is_file():
                shutil.copy(nb_file_copy_from, nb_file)
            else:
                # check if nb file has only empty cells
                nb_cells = nbformat.read(nb_file, as_version=4)
                if not all(
                    [
                        c["outputs"] == []
                        for c in nb_cells.cells
                        if c["cell_type"] == "code"
                    ]
                ):
                    diffs = repo.index.diff("HEAD~1")
                    for diff in diffs:
                        if (
                            Path(repo.git_dir).parent / diff.a_path
                        ) == nb_file_copy_from:
                            break
                    else:
                        if not force_notebook_rerun:
                            logger.info(
                                f"The notebook file {nb_file_copy_from} was not changed "
                                f"in the last commit, so I won't execute it again. Set "
                                f"the --force-notebook flag, to force overrun notebooks."
                            )
                            continue
                else:
                    logger.info(
                        f"Notebook {nb_file} is completely empty. I will run it now."
                    )

            # copy the files and add them to be deleted
            if "extra-media" in nblinbk_content:
                for file in nblinbk_content["extra-media"]:
                    copy_from = (nblink_file.parent / Path(file)).resolve()
                    copy_to = nblink_file.parent / copy_from.name
                    # copy_to = Path.cwd() / copy_from.name
                    shutil.copy(copy_from, copy_to)
                    files_to_delete.append(copy_to)

            nb_file.unlink()
            shutil.copy(nb_file_copy_from, nb_file)
            logger.info(f"Executing notebook {nb_file_copy_from}, copied to {nb_file}")

            # set up exporter
            exporter = NotebookExporter()
            exporter.register_preprocessor(SkipExecutionPreprocessor(), True)

            # export
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                output, _ = exporter.from_filename(nb_file)
            logger.info(f"Notebook was generated. Saving.")
            nb_file.write_text(output)
            logger.info(f"Converted {nblink_file} successfully. Will not delete files.")
            files_to_delete.clear()
    _sphinx_main(argv)


def commit_new(
    repo: Repo,
    remote: str,
    logger: logging.Logger,
    branch: Optional[str] = None,
) -> bool:
    repo_cache_file = Path(repo.working_tree_dir) / "tests/.em_ci_cache"
    logger.info(f"Checking {repo_cache_file} for old commits.")
    local_sha = repo.head.object.hexsha
    if branch is None:
        branch = repo.active_branch.name
    _remote = getattr(repo.remotes, remote)
    _remote.fetch()
    remote_sha = repo.rev_parse(f"{remote}/{branch}")
    logger.info(f"Remote sha is {remote_sha=} {remote_sha.message=}")
    # check if remote is ahead of local
    ahead_commits = list(repo.iter_commits(f"{local_sha}..{remote_sha}"))
    if len(ahead_commits) > 0:
        logger.info(
            f"Remote {remote} is ahead {len(ahead_commits)} of local. Pulling..."
        )
        remote_ = getattr(repo.remotes, remote)
        remote_.pull()
        current_sha = remote_sha
    else:
        logger.info(f"Local is on-level or ahead of remote {remote}.")
        current_sha = local_sha
    if repo_cache_file.is_file():
        file_sha = repo_cache_file.read_text()
        if file_sha == current_sha:
            return False
    repo_cache_file.write_text(local_sha)
    logger.info(f"This commit is new. Will continue.")
    return True


@click.command(
    help=(
        "EncoderMap's main CI script.\n\n"
        "Execution:\n"
        "    0) Check whether a new commit is on the current branch. Force with -f.\n"
        "    1) Run tests with coverage. Skip with -d\n"
        "    2) Run new notebooks and put the executed notebooks in docs\n"
        "    3) Build json files for dynamic shield.io badges.\n"
        "    4) Run and create a mypy report.\n"
        "Also logs these executions into ``docbuild.log`` into the same directory, "
        "as this file is also in. Logfiles older than a month "
        "are backed-up into YYYY-MM_docbuild.log files. This script is meant to be "
        "added as a cronjob. To make it run every 3 hours, add this line to crontab:\n\n"
        "``0 */3 * * * $python /path/to/encodermap/tests/run_docbuild_test_and_cover.py``\n\n"
        "And set the correct python executable and path to this script.."
    )
)
@click.option(
    "-f",
    "--force",
    default=False,
    is_flag=True,
    help="Forces the script to rerun tests, rebuild documentation.",
)
@click.option(
    "-fb",
    "--force-notebook",
    default=False,
    is_flag=True,
    help="Forces the script to rerun notebooks, even when not changed since last commit. This can potentially take a long time.",
)
@click.option(
    "-d",
    "--doc-only",
    is_flag=True,
    help="Skips tests and just build docs. Can be combined with the f-flag to force only docs.",
)
@click.option(
    "-t",
    "--test-only",
    is_flag=True,
    help="Skips docs and just runs tests.",
)
@click.option(
    "--run-expensive-tests",
    is_flag=True,
    help="Sets the env-variable 11ENCODERMAP_RUN_EXPENSIVE_TESTS='True'`` prior to running tests.",
)
@click.option(
    "--remote",
    default="gitlab",
    help="To check and pull the latest remotes from a specific remote. Set it here. Can be used with passwordless ssh remote configurations.",
)
@click.option(
    "--log", is_flag=True, default=False, help="Whether to log to console or just file."
)
@click.option(
    "--del-robots",
    is_flag=True,
    default=False,
    help="Set this flag, so that the `robots.txt` of the site will be deletd and your site will be visible by search engines.",
)
def main(
    force: bool = False,
    doc_only: bool = False,
    test_only: bool = False,
    run_expensive_tests: bool = True,
    remote: str = "gitlab",
    log: bool = False,
    force_notebook: bool = False,
    debug_test: Optional[str] = None,
    del_robots: bool = False,
) -> int:
    if debug_test is not None and not test_only:
        print(
            "Can't debug a test, when also building docs. Please run "
            "tests only with the -t flag."
        )
        return 0

    # define logfiles
    logfile = Path(__file__).parent.resolve() / "docbuild.log"

    # instantiate the logger
    logger = set_up_logger(logfile, log_to_console=log)
    logger.info("Starting logging for EncoderMap tests/docs.")

    # redirect stdout and stderr to logger
    sys.stdout = StreamToLogger(logger, logging.DEBUG)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    # test
    print(
        "This print statement is normally put to stdout, but due to the setup "
        "of the logger all stdout and stderr is redirected to logging."
    )

    # docbuild
    print("Will now set the docbuild badge to false.")
    docbuild_badge_file = (
        Path(__file__).resolve().parent.parent
        / "docs/source/_static/docbuild_badge.json"
    )
    docbuild_badge_json_data = {
        "schemaVersion": 1,
        "label": "Docbuild",
        "message": f"failing",
        "color": "red",
        "namedLogo": "Google Docs",
    }
    with open(docbuild_badge_file, "w") as f:
        json.dump(docbuild_badge_json_data, f)

    # instantiate the repo
    repo = Repo(Path(__file__).parent.parent, search_parent_directories=True)

    fix_html_test_runner(logger)
    # the progress message
    if not doc_only:
        msg = f"Running tests and building docs for EncoderMap."
    else:
        msg = f"Building Docs for EncoderMap."
    if run_expensive_tests:
        msg += " With expensive tests."
    msg += f" Current remote: {remote}, current commit: {repo.head.reference.commit.message}."

    if run_expensive_tests:
        os.env["RUN_EXPENSIVE_TESTS"] = "True"

    if test_only:
        unittest_result, cov_percentrage = main_unittests(
            logger=logger, debug_test=debug_test, commit_hash=repo.head.commit.hexsha
        )
        return 0

    if commit_new(repo, remote=remote, logger=logger) or force or force_notebook:
        if not doc_only:
            unittest_result, cov_percentrage = main_unittests(
                logger=logger, commit_hash=repo.head.commit.hexsha
            )
        main_sphinx(logger=logger, repo=repo, force_notebook_rerun=force_notebook)
    else:
        logger.info(
            f"Tests and Docs for this commit already finished. Set --force "
            f"to force a new Docbuild."
        )

    if del_robots:
        robots_file = (
            Path(__file__).resolve().parent.parent / "docs/source/_static/robots.txt"
        )
        robots_file.unlink()

    if unittest_result:
        test_badge_json_data = {
            "schemaVersion": 1,
            "label": "Unittests",
            "message": "passing",
            "color": "green",
            "namedLogo": "pytest",
        }
    else:
        test_badge_json_data = {
            "schemaVersion": 1,
            "label": "Unittests",
            "message": "failing",
            "color": "red",
            "namedLogo": "pytest",
        }
    with open(
        Path(__file__).resolve().parent.parent / "docs/source/_static/test_badge.json",
        "w",
    ) as f:
        json.dump(test_badge_json_data, f)

    if cov_percentrage <= 50:
        cov_badge_json_data = {
            "schemaVersion": 1,
            "label": "Coverage",
            "message": f"{cov_percentrage:.0f}",
            "color": "red",
            "namedLogo": "Codecov",
        }
    elif cov_percentrage <= 80:
        cov_badge_json_data = {
            "schemaVersion": 1,
            "label": "Coverage",
            "message": f"{cov_percentrage:.0f}",
            "color": "orange",
            "namedLogo": "Codecov",
        }
    else:
        cov_badge_json_data = {
            "schemaVersion": 1,
            "label": "Coverage",
            "message": f"{cov_percentrage:.0f}",
            "color": "green",
            "namedLogo": "Codecov",
        }
    with open(
        Path(__file__).resolve().parent.parent
        / "docs/source/_static/coverage_badge.json",
        "w",
    ) as f:
        json.dump(cov_badge_json_data, f)

    with open(docbuild_badge_file, "w") as f:
        docbuild_badge_json_data["color"] = "green"
        docbuild_badge_json_data["message"] = "passing"
        json.dump(docbuild_badge_json_data, f)

    raise Exception(f"Create a MyPy report.")

    return 0


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# encodermap/trajinfo/repository.py
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
"""Python endpoint to download files from a webserver on the fly.

Idea from Christoph Wehmeyer: https://github.com/markovmodel/mdshare
I liked his idea of the possibility to distribute MD data via a simple python
backend, but wanted to make it smaller. A simple `fetch()` should suffice. Also
I liked the yaml syntax and wanted to use it.

References:
    @article{wehmeyer2018introduction,
      title={Introduction to Markov state modeling with the PyEMMA software [Article v1. 0]},
      author={Wehmeyer, Christoph and Scherer, Martin K and Hempel, Tim and Husic, Brooke E and Olsson, Simon and No{\'e}, Frank},
      journal={Living Journal of Computational Molecular Science},
      volume={1},
      number={1},
      pages={5965},
      year={2018}
    }

"""


##############################################################################
# Imports
##############################################################################


import errno
import fnmatch
import glob
import inspect
import os
import re
import sys
from itertools import chain
from operator import methodcaller

import requests

from .._optional_imports import _optional_import
from .hash_files import hash_files
from .info_all import TrajEnsemble
from .info_single import SingleTraj

##############################################################################
# Optional Imports
##############################################################################


yaml_load = _optional_import("yaml", "load")
yaml_dump = _optional_import("yaml", "dump")
Loader = _optional_import("yaml", "CLoader")
Dumper = _optional_import("yaml", "CDumper")
download_wrapper = _optional_import("mdshare", "utils.download_wrapper")
tarfile = _optional_import("tarfile")


##############################################################################
# Globals
##############################################################################


__all__ = ["Repository"]


##############################################################################
# Functions
##############################################################################


def gen_dict_extract(key, var):
    """Copied from hexerei software's solution for nested dicts:

    Finds the value of a key anywhere in a nested dict.

    """
    if hasattr(var, "iteritems"):
        for k, v in var.iteritems():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result


def find_mime_type(d, mime_type):
    """Thanks to KobeJohn

    https://stackoverflow.com/questions/22162321/search-for-a-value-in-a-nested-dictionary-python

    """
    reverse_linked_q = list()
    reverse_linked_q.append((list(), d))
    while reverse_linked_q:
        this_key_chain, this_v = reverse_linked_q.pop()
        # finish search if found the mime type
        if this_v == mime_type:
            return this_key_chain
        # not found. keep searching
        # queue dicts for checking / ignore anything that's not a dict
        try:
            items = this_v.items()
        except AttributeError:
            continue  # this was not a nested dict. ignore it
        for k, v in items:
            reverse_linked_q.append((this_key_chain + [k], v))
    # if we haven't returned by this point, we've exhausted all the contents
    raise KeyError


def sizeof_fmt(num, suffix="B"):
    """Thanks to Fred Cirera and Sridhar Ratnakumar

    https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    https://web.archive.org/web/20111010015624/http://blogmag.net/blog/read/38/Print_human_readable_file_size

    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


##############################################################################
# Classes
##############################################################################


class Repository:
    """Main Class to work with Repositories of MD data and download the data.

    This class handles the download of files from a repository source. All
    data are obtained from a .yaml file (default at data/repository.yaml), which
    contains trajectory files and topology files organized in a readable manner.
    With this class the repository.yaml file can be queried using unix-like file
    patterns. Files can be downloaded on-the-fly (if they already exist, they won't
    be downloaded again). Besides files full projects can be downloaded and rebuilt.

    Attributes:
        current_path (str): Path of the .py file containing this class.
            If no working directory is given (None), all files will be
            downloaded to a directory named 'data' (will be created) which will
            be placed in the directory of this .py file.
        url (str): The url to the current repo source.
        maintainer (str): The maintainer of the current repo source.
        files_dict (dict): A dictionary summarizing the files in this repo.
            dict keys are built from `'project_name' + 'filetype'`. So for a
            project called 'protein_sim', possible keys are 'protein_sim_trajectory',
            'protein_sim_topology', 'protein_sim_log'. The values of these keys
            are all str and they give the actual filename of the files. If 'protein_sim'
            was conducted with GROMACS, these files would be 'traj_comp.xtc', 'confout.gro'
            and 'md.log'.
        files (list): Just a list of str of all downloadable files.
        data (dict): The main organization of the repository. This is the complete
            .yaml file as it was read and returned by pyyaml.

    Examples:
        >>> import encodermap as em
        >>> repo = em.Repository()
        >>> print(repo.search('*PFFP_sing*')) # doctest: +SKIP
        {'PFFP_single_trajectory': 'PFFP_single.xtc', 'PFFP_single_topology': 'PFFP_single.gro', 'PFFP_single_input': 'PFFP.mdp', 'PFFP_single_log': 'PFFP.log'}
        >>> print(repo.url)
        http://134.34.112.158

    """

    def __init__(
        self,
        repo_source="data/repository.yaml",
        checksum_file="data/repository.md5",
        ignore_checksums=False,
        debug=True,
    ):
        """Initialize the repository,

        Args:
            repo_source (str): The source .yaml file to build the repository from.
                Defaults to 'data/repository.yaml'.
            checksum_file (str): A file containing the md5 hash of the repository file.
                This ensures no one tampers with the repository.yaml file and injects
                malicious code. Defaults to 'data/repository.md5'.
            ignore_checksums (bool): If you want to ignore the checksum check of
                the repo_source file set this top True. Can be useful for
                developing, when the repository.yaml file undergoes a lot of changes.
                Defaults to False.
            debug (bool, optional): Whether to print debug info. Defaults to False.

        """
        # this will point to this file, no matter where it is (venv, etc.)
        self.current_path = os.path.split(inspect.getfile(inspect.currentframe()))[0]
        self.debug = debug

        # with that the source files can be defined
        repo_source = os.path.join(self.current_path, repo_source)
        checksum_file = os.path.join(self.current_path, checksum_file)

        # check the hash sum of the repo.yml file
        if checksum_file is not None and not ignore_checksums:
            with open(checksum_file, "r") as fh:
                if hash_files(repo_source)["repository.yaml"]["md5"] != fh.read():
                    raise RuntimeError(
                        "Checksums do not match, check your catalogue files!"
                    )

        # read the repo.yml file
        with open(repo_source, "r") as f:
            self.data = yaml_load(f, Loader=Loader)

        # define variables based on that
        self.url = self.data["url"]
        self.maintainer = (
            self.data["maintainer"]["name"] + ", " + self.data["maintainer"]["email"]
        )
        self.projects = self.data["projects"]
        self._connection = None
        self.files_dict = {}
        for dataset in self.datasets:
            for filetype in self.data[dataset]:
                if filetype == "credit":
                    continue
                self.files_dict[f"{dataset}_{filetype}"] = self.data[dataset][filetype][
                    "file"
                ]
        self.files = list(self.files_dict.values())

    @property
    def catalogue(self):
        """dict: Returns the underlying catalogue data."""
        return self.data

    def print_catalogue(self):
        """Prints the catalogue nicely formatted."""
        print(self.__str__())

    @property
    def projects(self):
        """dict: A dictionary containing project names and their associated files.
        Projects are a larger collection of individual sims, that belong together.
        The project names are the dictionary's keys, the files are given as lists
        in the dict's values.

        """
        return self._projects

    @projects.setter
    def projects(self, projects):
        self._projects = {}
        for item in self.data["projects"]:
            if isinstance(self.data["projects"][item], list):
                self._projects[item] = self.data["projects"][item]
            elif isinstance(self.data["projects"][item], dict):
                _ = []
                for key, value in self.data["projects"][item].items():
                    if key == "type":
                        continue
                    _.extend(value)
                self._projects[item] = _
            else:
                raise ValueError(f"Wrong Type in projects: {item}: {type(item)}")

    @property
    def datasets(self):
        """set: A set of datasets in this repository. A dataset can either be
        characterized by a set of trajectory-, topology-, log- and input-file
        or a dataset is a .tar.gz container, which contains all necessary files.

        """
        return set(self.data.keys()).difference(
            set(["name", "url", "maintainer", "projects"])
        )

    def search(self, pattern):
        out = {}
        if isinstance(pattern, list):
            _ = [self.search(i) for i in pattern]
            return dict(chain.from_iterable(map(methodcaller("items"), _)))
        for key, item in self.files_dict.items():
            if fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(item, pattern):
                out[key] = item
        return out

    def load_project(
        self,
        project,
        working_directory=None,
        overwrite=False,
        max_attempts=3,
        makdedir=False,
        progress_bar=True,
    ):
        """This will return `TrajEnsemble` / `SingleTraj` objects that are correctly formatted.

        This method allows one to directly rebuild projects from the repo source,
        using encodermap's own `SingleTraj` and `TrajEnsemble` classes.

        Args:
            project (str): The name of the project to be loaded. See
                Repository.projects.keys() for a list of projects.
            working_directory (Union[str, None], optional): Can be a string to a directory to save the
                files at. Can also be None. In that case `self.current_path` + `'/data'` will be used
                to save the file at. Which is retrieved by `inspect.getfile(inspect.currentframe))`. If
                the files are already there and overwrite is false, the file path is simply returned.
                Defaults to None.
            overwrite (bool, optional): Whether to overwrite local files.   Defaults to False.
            max_attempts (int, optional): Number of download attempts. Defaults to 3.
            makdedir (bool, optional): Whether to create `working_directory`, if it is not already existing.
                Defaults to False.
            progress_bar (bool, optional): Uses the package progress-reporter to display a progress bar.

        Returns:
            Union[encodermap.SingleTraj, encodermap.TrajEnsemble]: The project already loaded into encodermap's
                `SingleTraj` or `TrajEnsemble` classes.

        Examples:
            >>> import encodermap as em
            >>> repo = em.Repository()
            >>> trajs = repo.load_project('Tetrapeptides_Single')
            >>> print(trajs)
            encodermap.TrajEnsemble object. Current backend is no_load. Containing 2 trajs. Common str is ['PFFP', 'FPPF']. Not containing any CVs.
            >>> print(trajs.n_trajs)
            2

        """
        if isinstance(self.data["projects"][project], dict):
            common_strings = list(
                filter(
                    lambda x: False if x == "type" else True,
                    list(self.data["projects"][project].keys()),
                )
            )
            traj_files = [
                file
                for cs in common_strings
                for file in self.data["projects"][project][cs][:-1]
            ]
            top_files = [
                self.data["projects"][project][cs][-1] for cs in common_strings
            ]
            if self.data["projects"][project]["type"] == "files":
                files, directory = self.fetch(
                    traj_files + top_files,
                    working_directory=working_directory,
                    overwrite=overwrite,
                    max_attempts=max_attempts,
                    makdedir=makdedir,
                    progress_bar=progress_bar,
                )
            elif self.data["projects"][project]["type"] == "container":
                pattern = project + ".tar.gz"
                files, directory = self.fetch(
                    pattern,
                    working_directory=working_directory,
                    overwrite=overwrite,
                    max_attempts=max_attempts,
                    makdedir=makdedir,
                    progress_bar=progress_bar,
                )
            else:
                raise Exception(
                    f"Unknown type of project: {self.data['projects'][project]['type']}. `type` needs to be either 'files' or 'container'."
                )
            traj_files = [os.path.join(directory, i) for i in traj_files]
            top_files = [os.path.join(directory, i) for i in top_files]
            return TrajEnsemble(traj_files, top_files, common_str=common_strings)
        else:
            files, directory = self.fetch(
                self.projects[project],
                working_directory=working_directory,
                overwrite=overwrite,
                max_attempts=max_attempts,
                makdedir=makdedir,
                progress_bar=progress_bar,
            )
            return SingleTraj(files[0], files[1])

    def lookup(self, file):
        """Piece of code to allow some compatibility to mdshare.

        The complete `self.data` dictionary will be traversed to find
        `file` and its location in the `self.data` dictionary. This will be
        used to get the filesize and its md5 hash. The returned tuple also tells
        whether the file is a .tar.gz container or not. In the case of a container,
        the container needs to be extracted using tarfile.

        Args:
            file (str): The file to search for.

        Returns:
            tuple: A tuple containing the follwing:
                str: A string that is either 'container' or 'index' (for normal files).
                dict: A dict with dict(file=filename, hash=filehas, size=filesize)

        """
        simulation, filetype, _ = find_mime_type(self.data, file)
        out = dict(
            file=self.data[simulation][filetype]["file"],
            hash=self.data[simulation][filetype]["md5"],
            size=self.data[simulation][filetype]["size"],
        )
        if filetype == "container":
            return "containers", out
        else:
            return "index", out

    def _get_connection(self):
        """Also compatibility with mdshare"""
        if self._connection is None:
            self._connection = requests.session()
        return self._connection

    @staticmethod
    def _split_proj_filetype(proj_filetype):
        """Splits the strings that index the self.datasets dictionary."""
        if proj_filetype.count("_") == 1:
            return proj_filetype.split("_")
        else:
            substrings = proj_filetype.split("_")
            return "_".join(substrings[:-1]), substrings[-1]

    def get_sizes(self, pattern):
        """Returns a list of file-sizes of a given pattern.

        Args:
            pattern (Union[str, list]): A unix-like pattern ('traj*.xtc') or a
                list of files (['traj_1.xtc', 'traj_2.xtc']).

        Returns:
            list: A list of filesizes in bytes.

        """
        sizes = []
        for proj_filetype, file in self.search(pattern).items():
            project, filetype = Repository._split_proj_filetype(proj_filetype)
            size = self.data[project][filetype]["size"]
            sizes.append(size)
        return sizes

    def stack(self, pattern):
        """Creates a stack to prepare for downloads.

        Args:
            pattern (Union[str, list]): A unix-like pattern ('traj*.xtc') or a
                list of files (['traj_1.xtc', 'traj_2.xtc']).

        Returns:
            list: A list of dicts. Each dict contains filename, size and a boolean
                value telling whether the downloaded file needs to be extracted
                after downloading.

        """
        stack = []
        sizes = self.get_sizes(pattern)
        for (proj_filetype, file), size in zip(self.search(pattern).items(), sizes):
            project, filetype = Repository._split_proj_filetype(proj_filetype)
            unpack = filetype == "container"
            stack.append(dict(file=file, size=size, unpack=unpack))
        return stack

    def fetch(
        self,
        remote_filenames,
        working_directory=None,
        overwrite=False,
        max_attempts=3,
        makdedir=False,
        progress_bar=True,
    ):
        """This fetches a singular file from self.files.

        Displays also progress bar with the name of the file. Uses requests.

        Args:
            remote_filename (str): The name of the remote file. Check `self.files` for more info.
            working_directory (Union[str, None], optional): Can be a string to a directory to save the
                files at. Can also be None. In that case `self.current_path` + `'/data'` will be used
                to save the file at. Which is retrieved by `inspect.getfile(inspect.currentframe))`. If
                the files are already there and overwrite is false, the file path is simply returned.
                Defaults to None.
            overwrite (bool, optional): Whether to overwrite local files.   Defaults to False.
            max_attempts (int, optional): Number of download attempts. Defaults to 3.
            makdedir (bool, optional): Whether to create `working_directory`, if it is not already existing.
                Defaults to False.
            progress_bar (bool, optional): Uses the package progress-reporter to display a progress bar.

        Returns:
            tuple: A tuple containing the following:
                list: A list of files that have just been downloaded.
                str: A string leading to the directory the files have been downloaded to.

        """
        # import progress-reporter
        try:
            import progress_reporter

            have_progress_reporter = True
        except ImportError:
            if self.debug:
                print(
                    "Downloading files without progress bar. Run `pip install progress-reporter` to use this feature."
                )
            have_progress_reporter = False

        # find files to import
        stack = self.stack(remote_filenames)

        # define the filename
        if working_directory is None:
            working_directory = os.path.join(self.current_path, "data")
        if isinstance(working_directory, str):
            local_filenames = [
                os.path.join(working_directory, s["file"]) for s in stack
            ]
            if not os.path.isdir(working_directory):
                if makdedir:
                    os.makedirs(working_directory)
                else:
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), working_directory
                    )
        else:
            raise ValueError(
                f"Type of argument `working_directory` needs to be either `None` or `str`, you provided {type(working_directory)}"
            )

        # split .tar.gz and regular files
        # and check what already exists
        local_containers = [f for f in local_filenames if f.endswith(".tar.gz")]
        local_containers = list(map(lambda x: x.split(".")[0], local_containers))
        local_files = [f for f in local_filenames if not f.endswith(".tar.gz")]
        result = []
        if local_files:
            if all([os.path.isfile(lf) for lf in local_files]) and not overwrite:
                if self.debug:
                    print(
                        f"Files '{local_files}' already exists. Set `overwrite` to `True` to download them again."
                    )
                result.extend(local_files)
            elif any([os.path.isfile(lf) for lf in local_files]) and not overwrite:
                existing_files = glob.glob(os.path.join(working_directory, "*"))
                existing_files = [os.path.split(i)[-1] for i in existing_files]
                missing_files = list(
                    set([i["file"] for i in stack]).difference(set(existing_files))
                )
                result.extend(
                    [
                        os.path.join(working_directory, i)
                        for i in existing_files
                        if fnmatch.fnmatch(i, remote_filenames)
                    ]
                )
                if self.debug:
                    print(
                        f"{len(stack) - len(missing_files)} Files already exist. I will only download '{missing_files}'. Set `overwrite` to `True` to download all files again."
                    )
                stack = list(
                    filter(
                        lambda x: True if x["file"] in missing_files else False, stack
                    )
                )
        if local_containers:
            if all([os.path.isdir(lc) for lc in local_containers]) and not overwrite:
                if self.debug:
                    print(
                        f"Directories '{local_containers}' already exists. Set `overwrite` to `True` to download them again."
                    )
                result.extend(local_containers)
            elif any([os.path.isdir(lc) for lc in local_containers]) and not overwrite:
                existing_directories = glob.glob(os.path.join(working_directory, "*/"))
                existing_directories = [
                    os.path.split(os.path.split(i)[0])[-1] for i in existing_directories
                ]
                missing_directories = list(
                    set([i["file"].split(".")[0] for i in stack]).difference(
                        set(existing_directories)
                    )
                )
                result.extend(
                    [
                        os.path.join(working_directory, i)
                        for i in existing_directories
                        if fnmatch.fnmatch(i + ".tar.gz", remote_filenames)
                    ]
                )
                if self.debug:
                    print(
                        f"{len(stack) - len(missing_directories)} Directories already exist. I will only download '{missing_directories}'. Set `overwrite` to `True` to download all files again."
                    )
                stack = list(
                    filter(
                        lambda x: True
                        if x["file"].split(".")[0] in missing_directories
                        else False,
                        stack,
                    )
                )
        if len(result) == len(local_filenames):
            return result, working_directory

        # instantiate ProgressBars
        if have_progress_reporter and progress_bar:
            callbacks = []
            pg = progress_reporter.ProgressReporter_()
            total = sum(item["size"] for item in stack)

            def update(n, blk, stage):
                downloaded = n * blk
                inc = max(0, downloaded - pg._prog_rep_progressbars[stage].n)
                pg.update(inc, stage=stage)
                # total progress
                try:
                    pg.update(inc, stage=-1)
                except RuntimeError:
                    pass

            from functools import partial

            tqdm_args = dict(unit="B", file=sys.stdout, unit_scale=True, position=0)

            n_progress_bars = 0
            for stage, item in enumerate(stack):
                if working_directory is not None:
                    path = os.path.join(working_directory, item["file"])
                    if os.path.exists(path) and not overwrite:
                        callbacks.append(None)
                    else:
                        pg.register(
                            item["size"],
                            description=f'downloading {item["file"]}',
                            tqdm_args=tqdm_args,
                            stage=stage,
                        )
                        callbacks.append(partial(update, stage=stage))
                        n_progress_bars += 1
            if n_progress_bars > 1:
                pg.register(total, description="total", tqdm_args=tqdm_args, stage=-1)
        else:
            from unittest.mock import MagicMock

            pg = MagicMock()
            callbacks = [None] * len(stack)

        # download and unpack
        result = []
        with pg.context():
            for item, progress in zip(stack, callbacks):
                file = download_wrapper(
                    self,
                    item["file"],
                    working_directory=working_directory,
                    max_attempts=max_attempts,
                    force=overwrite,
                    callback=progress,
                )
                if item["unpack"]:

                    def inspect(members):
                        for member in members:
                            path, filename = os.path.split(member.name)
                            if path == "":
                                yield member, filename

                    with tarfile.open(file, "r:gz") as fh:
                        members = []
                        for i, (member, filename) in enumerate(inspect(fh)):
                            members.append(member)
                            result.append(os.path.join(working_directory, filename))
                        fh.extractall(path=working_directory, members=members)
                    os.remove(file)
                    result.append(file.split(".")[0])
                    # print(result)
                    # raise Exception("STOP")
                    # def inspect(members):
                    #     for member in members:
                    #         path, filename = os.path.split(member.name)
                    #         if path == '':
                    #             yield member, filename
                    #
                    # with tarfile.open(file, 'r:gz') as fh:
                    #     members = []
                    #     for i, (member, filename) in enumerate(inspect(fh)):
                    #         members.append(member)
                    #         result.append(
                    #             os.path.join(working_directory, filename))
                    #     fh.extractall(
                    #         path=working_directory, members=members)
                    # os.remove(file)
                else:
                    result.append(file)

        return result, working_directory

    def __str__(self):
        string = f"Repository: {self.url}\n"
        string += f"Maintainer: {self.maintainer}\n"
        for dataset in self.datasets:
            string += f"  Dataset: {dataset}\n"
            for filetype in self.data[dataset]:
                if filetype == "credit":
                    try:
                        string += f"    Author: {self.data[dataset][filetype]['author']}, {self.data[dataset][filetype]['email']}\n"
                    except KeyError:
                        string += (
                            f"    Author: {self.data[dataset][filetype]['author']}\n"
                        )
                    continue
                try:
                    substr = f"    {filetype.capitalize()} File: {self.data[dataset][filetype]['file']}"
                    string += f"{substr:<50}{sizeof_fmt(self.data[dataset][filetype]['size'])}\n"
                except KeyError:
                    print("Could not build summary string")
                    print(filetype)
                    print(type(filetype))
                    raise
        return string

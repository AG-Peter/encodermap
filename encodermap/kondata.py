# -*- coding: utf-8 -*-
# encodermap/kondata.py
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
"""Functions for interfacing with the University of Konstanz's repository service KonDATA.

"""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import getpass
import shutil
import tarfile
import time
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional, Union

# Third Party Imports
import requests
from optional_imports import _optional_import
from requests.auth import HTTPBasicAuth
from rich.status import Status


################################################################################
# Optional Imports
################################################################################


BeautifulSoup = _optional_import("bs4", "BeautifulSoup")


################################################################################
# Globals
################################################################################


__all__: list[str] = ["get_from_kondata"]
DATASET_URL_MAPPING = {
    "test": "https://dx.doi.org/10.48606/108",
    "H1Ub": "https://dx.doi.org/10.48606/99",
}


################################################################################
# Functions
################################################################################


def untar(
    tar_file: Path,
    doi_tar: str,
    output: Path,
    force_overwrite: bool = False,
    silence_overwrite: bool = False,
) -> list[str]:
    """Untars files in a tar archive downloaded from KonDATA.

    As the files are found under {doi_tar}/data/dataset/, only this directory is considered.
    The other directories in the tar file contain RADAR metadata, that is not important.

    Args:
        tar_file (Path): The pathlib.Path to the tar file.
        doi_tar (str): The doi_tar of the tar file. This arg is needed to find the
            datafiles in {doi_tar}/data/dataset. The doi_tar can be obtained from
            the doi of the dataset by removing 'https://dx.doi.org/' and replacing
            '/' in the doi with '-'.
        output (Path): The directory, where to put the files of the archive.
        force_overwrite (bool): Whether to overwrite files that are already there.
            Will print 'file already exists' if file exists and `force_overwrite`
            is set to False. Defaults to False.
        silence_overwrite (bool): Will not print information about already
            existing files. Useful for when this function is used in a script.

    Returns:
        list[str]: A list of the new files.

    """
    untarred_files: list[str] = []
    if not silence_overwrite:
        print(f"{tar_file} already exists. Set `force_overwrite` to True to overwrite.")
    with tarfile.open(tar_file) as tar:
        for member in tar.getmembers():
            if (
                member.path.startswith(f"{doi_tar}/data/dataset")
                and member.path != f"{doi_tar}/data/dataset"
            ):
                filename = member.path.lstrip(f"{doi_tar}/data/dataset/")
                extract_to = output / filename
                if extract_to.exists() and not force_overwrite:
                    if not silence_overwrite:
                        print(
                            f"{extract_to} already exists. Set "
                            f"`force_overwrite` to True to overwrite."
                        )
                else:
                    if member.isdir():
                        extract_to.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        untarred_files.append(str(extract_to))
                        tar.makefile(member, extract_to)
                    else:
                        print(f"Unknown TarInfo type: {member}")
    return untarred_files


def get_from_kondata(
    dataset_name: str,
    output: Optional[Union[str, Path]] = None,
    force_overwrite: bool = False,
    mk_parentdir: bool = False,
    silence_overwrite_message: bool = False,
    tqdm_class: Optional[Any] = None,
    download_extra_data: bool = False,
    download_checkpoints: bool = False,
    download_h5: bool = True,
) -> str:
    """Get dataset from the University of Konstanz's data repository KONData.

    Args:
        dataset_name (str): The name of the dataset. Refer to `DATASET_URL_MAPPING`
            to get a list of the available datasets.
        output (Union[str, Path]): The output directory.
        force_overwrite (bool): Whether to overwrite existing files. Defaults to False.
        mk_parentdir (bool): Whether to create the `output` directory if it does
            not already exist. Defaults to False.
        silence_overwrite_message (bool): Whether to silence the 'file already exists'
            warning. Can be useful in scripts. Defaults to False.
        tqdm_class (Optional[Any]): A class that is similar to tqdm.tqdm. This
            is mainly useful if this function is used inside a `rich.status.Status`
            context manager, as the normal tqdm does not work inside this context.
            If None is provided, the default tqdm will be used.
        download_extra_data (bool): Whether to download extra data. It Is only used
            if the dataset is not available on KonDATA. Defaults to False.
        download_checkpoints (bool): Whether to download pretrained checkpoints.
            It is only used if the dataset is not available on KonDATA.
            Defaults to False.
        download_h5 (bool): Whether to also download an h5 file of the
            ensemble. Defaults to True.

    Returns:
        str: The output directory.

    """
    # Local Folder Imports
    from .misc.misc import _is_notebook

    if dataset_name not in DATASET_URL_MAPPING:
        return get_from_url(
            f"https://sawade.io/encodermap_data/{dataset_name}",
            output=output,
            force_overwrite=force_overwrite,
            mk_parentdir=mk_parentdir,
            silence_overwrite_message=silence_overwrite_message,
            tqdm_class=tqdm_class,
            download_extra_data=download_extra_data,
            download_checkpoints=download_checkpoints,
            download_h5=download_h5,
        )
    if output is None:
        # Standard Library Imports
        import pkgutil

        package = pkgutil.get_loader("encodermap")
        if package is None:
            output = Path("~").resolve() / f".encodermap_data/{dataset_name}"
        else:
            emfile = package.get_filename()
            output = Path(emfile).parent.parent / "tests"
            if not output.is_dir():
                output = Path("~").resolve() / f".encodermap_data/{dataset_name}"
            else:
                output /= f"data/{dataset_name}"
                output.parent.mkdir(exist_ok=True)
        if not output.parent.is_dir():  # pragma: nocover
            question = input(
                f"I will create the directory {output.parent} and download "
                f"the dataset {dataset_name} to it."
            )
            if question.lower() not in ["y", "ye", "yes"]:
                raise Exception(f"User has answered to not overwrite {output.parent}.")
        output.mkdir(parents=True, exist_ok=True)

    # in all other cases make sure its path
    output = Path(output)

    if tqdm_class is None:
        if _is_notebook():
            # Third Party Imports
            from tqdm.notebook import tqdm as tqdm_class  # type: ignore[no-redef]
        else:
            # Third Party Imports
            from tqdm import tqdm as tqdm_class  # type: ignore[no-redef]
    assert tqdm_class is not None

    if not (output := Path(output)).is_dir():
        if not mk_parentdir:
            raise Exception(
                f"Output directory {output} does not exists. Create it "
                f"or set `mk_parentdir` to True."
            )
        else:
            output.mkdir()

    # clear partial downloads
    partial_downloads = output.glob("*.crdownload")
    for partial_download in partial_downloads:
        partial_download.unlink()

    # define stuff for the tarfile and maybe unpack the files if not already done so
    doi_tar = (
        DATASET_URL_MAPPING[dataset_name]
        .lstrip("https://dx.doi.org/")
        .replace("/", "-")
    )
    tar_file = output / f"{doi_tar}.tar"
    if tar_file.is_file() and not force_overwrite:
        untarred_files = untar(
            tar_file, doi_tar, output, force_overwrite, silence_overwrite_message
        )
        if not force_overwrite:
            return str(output)

    # instantiate the status
    s = Status(f"Downloading {dataset_name}", spinner="material")
    s.start()

    # use selenium to download the archive
    url = DATASET_URL_MAPPING[dataset_name]
    try:
        # Third Party Imports
        from selenium import webdriver
        from selenium.webdriver.common.by import By
    except ImportError as e:
        raise Exception(
            f"Programmatically downloading from KonDATA requires selenium. Beware "
            f"it uses Google code to interact with web pages. Either "
            f"install it with `pip install selenium`, or manually download the "
            f"files from {url} and un-tar them to {output}"
        ) from e
    prefs = {"download.default_directory": str(output)}
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    download_element = driver.find_element(
        By.XPATH, '//a[starts-with(@class, "dataset-download")]'
    )
    download_element.click()
    accept_element = driver.find_element(By.XPATH, '//button[text()="Accept"]')
    accept_element.click()

    # play an animation while downloading
    i = 0
    while True:
        time.sleep(0.2)
        i += 1
        if tar_file.is_file():
            s.stop()
            break
    copied_tar = output / f"{doi_tar} (1).tar"
    if copied_tar.is_file():
        shutil.move(copied_tar, tar_file)
    untarred_files = untar(
        tar_file, doi_tar, output, force_overwrite, silence_overwrite=True
    )

    if dataset_name == "H1Ub" and download_h5:
        trajs_file = output / "trajs.h5"
        if trajs_file.is_file() and not force_overwrite:
            print(f"trajs.h5 file for H1Ub already present.")
        else:
            print(
                f"The H1Ub dataset does not contain a trajs.h5 file. "
                f"I will now create it."
            )

    return str(output)


def get_assign_from_file(file: Path, assign: str) -> str:
    """Reads a file and extracts lines with assignments. Can be
    used for reading simple secret files which look like::
        PASSWORD=my_sekret_password
        USERNAME=oll_korrect_username

    Args:
        file (Path): The file.
        assign (str): The string to look for (e.g. PASSWORD).

    Returns:
        str: The assignment after the equal (=) sign.

    """
    content = file.read_text().splitlines()
    content_line = content[[c.startswith(assign) for c in content].index(True)]
    return content_line.split("=")[-1]


def is_directory(url: str) -> bool:
    """Returns, whether a string ends with '/' which makes that an url of a dir.

    Args:
        url (str): The url.

    Returns:
        bool: Whether that string ends with '/'.

    """
    if url.endswith("/"):
        return True
    return False


def find_links(url: str, auth: Optional[HTTPBasicAuth] = None) -> Generator:
    """Recourses through an html content file with beautifulsoup and extracts links.

    Can be used to mimic `wget -R` with python.

    Args:
        url (str): The url to recourse.
        auth (Optional[HTTPBasicAuth]): The authentication to use. Can be None
            for unprotected urls. Can be an instance of `requests.auth.HTTPBasicAuth`.

    Yields:
        tuple: A tuple of the following:
            str: The complete link to the file.
            str: The truncated link (without the `url` substring), which can
                be used to set the filename on disk, the link will be downloaded
                to.

    """
    organization_names = ["?C=N;O=D", "?C=M;O=A", "?C=S;O=A", "?C=D;O=A"]
    content = requests.get(url, auth=auth).content
    soup = BeautifulSoup(content, "html.parser")
    maybe_directories = soup.findAll("a", href=True)
    for link in maybe_directories:
        if is_directory(link["href"]) and "Parent Directory" not in link.text:
            if not url.endswith("/"):
                new_url = url + "/" + link["href"]
            else:
                new_url = url + link["href"]
            yield from find_links(new_url, auth)
        else:
            if link["href"] not in organization_names:
                filename = url + "/" + link["href"]
                if not filename.endswith("/"):
                    yield filename


def get_from_url(
    url: str,
    output: Optional[Union[str, Path]] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    force_overwrite: bool = False,
    mk_parentdir: bool = False,
    silence_overwrite_message: bool = False,
    tqdm_class: Optional[Any] = None,
    download_extra_data: bool = False,
    download_checkpoints: bool = False,
    download_h5: bool = True,
    combine_progbars: bool = False,
) -> str:
    """Recourses through `url` and downloads all strings into `output`.

    Args:
        url (str): The url to visit.
        output (Optional[Union[str, Path]]): Where to put the files.
        username (Optional[str]): The username for protected sites. If the site
            is protected and this arg is None, the `input()` builtin will be used
            to get the username.
        password (Optional[str]): The password for protected sites. If the site
            is protected and this arg is None, the `getpass()` method will be
            used to get the password.
        force_overwrite (bool): Whether to overwrite existing files.
        mk_parentdir (bool): Whether to create the `output` directory in case it
            is missing.
        tqdm_class (Optional[Any]): A class implementing a tqdm feature.
        download_extra_data (bool): Whether to download (potentially) large
            extra data. Mainly useful for unittests. Defaults to False.
        download_checkpoints (bool): Whether to download checkpoints.
            Good for skipping long training.
        combine_progbars (bool): Whether to make the download print one long
            progression bar.

    Returns:
        str: The output directory.

    """
    # Local Folder Imports
    from .misc.misc import _is_notebook

    if "sawade.io" in url:
        dataset_name = url.replace("https://sawade.io/encodermap_data/", "")

    if not url.startswith("https"):
        raise Exception(f"Not downloading from {url=}. Missing https.")

    if output is None:
        # Standard Library Imports
        import pkgutil

        package = pkgutil.get_loader("encodermap")
        if package is None:
            output = Path("~").resolve() / f".encodermap_data/{dataset_name}"
        else:
            emfile = package.get_filename()
            output = Path(emfile).parent.parent / "tests"
            if not output.is_dir():
                output = Path("~").resolve() / f".encodermap_data/{dataset_name}"
            else:
                output /= f"data/{dataset_name}"
                output.parent.mkdir(exist_ok=True)
        if not output.parent.is_dir():  # pragma: nocover
            question = input(
                f"I will create the directory {output.parent} and download "
                f"the dataset {dataset_name} to it."
            )
            if question.lower() not in ["y", "ye", "yes"]:
                raise Exception(f"User has answered to not overwrite {output.parent}.")
        output.mkdir(parents=True, exist_ok=True)

    if tqdm_class is None:
        if _is_notebook():
            # Third Party Imports
            from tqdm.notebook import tqdm as tqdm_class  # type: ignore[no-redef]
        else:
            # Third Party Imports
            from tqdm import tqdm as tqdm_class  # type: ignore[no-redef]
    assert tqdm_class is not None

    downloaded_files: list[str] = []
    if not (output := Path(output)).is_dir():
        if not mk_parentdir:
            raise Exception(
                f"Output directory {output} does not exists. Create it "
                f"or set `mk_parentdir` to True."
            )
        else:
            output.mkdir()
    # check for the act.vault file
    try:
        status_code = requests.get(url, timeout=3).status_code
    except requests.exceptions.Timeout:
        warnings.warn(f"EncoderMap's repository at {url} timed out.")
        return str(output)
    except requests.exceptions.ConnectionError:
        warnings.warn(f"EncoderMap's repository at {url} is unreachable.")
        return str(output)
    if status_code == 401:
        vault_file = Path(__file__).resolve().parent.parent / "act.vault"
        if vault_file.is_file() and username is None:
            username = get_assign_from_file(vault_file, "ENCODERMAP_DATA_USER")
        elif username is None:
            username = input("Please enter the username: ")
        if vault_file.is_file() and password is None:
            password = get_assign_from_file(vault_file, "ENCODERMAP_DATA_PASSWORD")
        elif password is None:
            password = getpass.getpass("Please enter the password: ")
        auth = HTTPBasicAuth(username, password)
    elif status_code == 200:
        auth = None
    else:
        raise Exception(f"Url {url} returned error: {status_code}")

    # try whether the password works
    status_code_with_auth = requests.get(url, auth=auth).status_code
    if status_code_with_auth == 401:
        raise Exception(f"Wrong username/password.")

    files = list(find_links(url, auth))
    in_files = []
    out_files = []
    for f1 in files:
        in_files.append(f1)
        o = f1.replace(url, "").replace("//", "/").lstrip("/")
        f2 = output / o
        out_files.append(f2)

    if combine_progbars:
        raise NotImplementedError

    for in_file, out_file in zip(in_files, out_files):
        if out_file.is_file() and not force_overwrite:
            if not silence_overwrite_message:
                print(
                    f"{out_file} already exists. Set `force_overwrite` to True to overwrite."
                )
            continue
        if "extra_data" in str(in_file) and not download_extra_data:
            continue
        if "checkpoints" in str(in_file) and not download_checkpoints:
            continue
        if str(in_file).split(".")[-1] == "h5" and not download_h5:
            print(f"Skipping {in_file}")
            continue
        out_file.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(in_file, auth=auth, stream=True)
        total_length = int(response.headers.get("content-length", 0))
        with (
            open(out_file, "wb") as file,
            tqdm_class(
                desc=str(out_file),
                total=total_length,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        downloaded_files.append(str(out_file))
    return str(output)

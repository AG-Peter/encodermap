#!/usr/bin/python3
# -*- coding: utf-8 -*-
# encodermap/trajinfo/hash_files.py
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

import argparse
import glob
import hashlib
import os
import pprint
import sys


def hash_files(files):
    """Returns a dict of file hashes

    Args:
        files (Union[str, list]) A file or a list of files.

    Returns:
        dict: A nested dict, indexed by filenames and sha1 and md5 hashes.

    """
    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    if isinstance(files, str):
        files = [files]

    out = {}
    for file in files:
        md5 = hashlib.md5()
        sha1 = hashlib.sha1()

        with open(file, "rb") as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
                sha1.update(data)
        out[os.path.basename(file)] = {"md5": md5.hexdigest(), "sha1": sha1.hexdigest()}

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get sha1 and md5 hashes of single file or list of files."
    )
    parser.add_argument(
        "files", nargs="*", help="Files to get hashes from. Can be used with wildcard."
    )
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    out = hash_files(args.files)
    pprint.pprint(out)

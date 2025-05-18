# -*- coding: utf-8 -*-
# tests/test_xarray.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
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
# Standard Library Imports
import unittest

# Third Party Imports
import numpy as np

# Encodermap imports
from conftest import skip_all_tests_except_env_var_specified


################################################################################
# Test Suites
################################################################################


@skip_all_tests_except_env_var_specified(unittest.skip)
class TestXarray(unittest.TestCase):
    def test_xr_to_np(self):
        # Standard Library Imports
        from types import SimpleNamespace

        # Encodermap imports
        from encodermap.trajinfo.trajinfo_utils import np_to_xr

        for shape, nan_dim in zip([(103,), (103, 103), (103, 103, 3)], [1, 2, 2]):
            # 1 = frame feature
            # 2 = normal feature
            # 3 = position feature
            test = np.random.random(shape)

            # add some nans along the last or second_dim
            size = (3,) if nan_dim == 1 else (3, 10)
            nan_indices = np.random.randint(0, 100, size=size)

            if nan_dim == 1:
                test[nan_indices] = np.nan
            else:
                test[-3:] = np.nan

            assert np.any(~np.isnan(test))

            for id_type in ["small", "full"]:
                if id_type == "small":
                    traj_id = np.arange(100)
                    traj_num = None
                else:
                    traj_id = np.vstack([np.full((100,), 0), np.arange(100)]).T
                    traj_num = 0
                traj = SimpleNamespace(
                    **{
                        "n_frames": 100,
                        "backend": "mdtraj",
                        "traj_file": "tmp.json.xtc",
                        "top_file": "tmp.json.pdb",
                        "id": traj_id,
                        "traj_num": traj_num,
                        "basename": "tmp.json",
                        "time": np.arange(100) * 10,
                    },
                )
                da = np_to_xr(test, traj=traj, attr_name="test")


################################################################################
# Collect Test Cases and Filter
################################################################################


def load_tests(loader, tests, pattern):
    test_cases = (TestXarray,)
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

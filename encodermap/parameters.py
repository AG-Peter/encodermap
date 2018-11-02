"""
EncoderMap
Copyright (C) 2018  Tobias Lemke

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
from math import pi
import json
from .misc import search_and_replace


class ParametersFramework:
    """
    See example in examples/parameters.py for usage
    """
    def __init__(self):
        self.main_path = ""

    def save(self, path=None):
        if not path:
            path = os.path.join(self.main_path, "parameters.json")
        with open(path, 'w') as file:
            json.dump(self.__dict__, file, indent=4, sort_keys=True)
        return path

    @classmethod
    def load(cls, path):
        with open(path, 'r') as file:
            params = json.load(file)
        if params["main_path"] != os.path.dirname(path):
            print("seams like the directory was moved. Parameter file is updated ...")
            search_and_replace(path, params["main_path"], os.path.dirname(path))
            with open(path, 'r') as file:
                params = json.load(file)

        param = cls()
        param._setattrs(params)
        return param

    def _setattrs(self, dictionary):
        for key, value in dictionary.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError("The parameter " + key + " is not supported")

    def __getitem__(self, item):
        return getattr(self, item)


class Parameters(ParametersFramework):
    def __init__(self):
        self.n_neurons = [128, 128, 2]
        self.activation_functions = ["", "tanh", "tanh", ""]
        self.learning_rate = 0.001
        self.n_steps = 100000
        self.batch_size = 256
        self.summary_step = max(1, int(self.n_steps / 1000))
        self.checkpoint_step = max(1, int(self.n_steps / 10))
        self.main_path = ""
        self.analysis_path = ""
        self.sketch_parameters = (4.5, 12, 6, 1, 2, 6)
        self.sketch_cost_scale = 500
        self.auto_cost_scale = 1
        self.mean_cost_scale = 0.0001
        self.id = ""
        self.gpu_memory_fraction = 1
        self.l2_reg_constant = 0.001
        self.periodicity = 2*pi  # use float("inf") for non periodic inputs

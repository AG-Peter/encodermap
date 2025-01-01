# -*- coding: utf-8 -*-
# tests/test_autoencoder.py
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


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import functools
import inspect
import json
import platform
import shutil
import struct
import tempfile
import unittest
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, Union
from unittest.mock import patch

# Third Party Imports
import h5py
import MDAnalysis as mda
import mdtraj
import numpy as np
import scipy
import tensorflow as tf
import transformations
import xarray as xr
from rich.console import Console

# Encodermap imports
import encodermap.models.models
from conftest import skip_all_tests_except_env_var_specified
from encodermap.misc.saving_loading_models import load_model, save_model
from encodermap.models.models import gen_functional_model
from encodermap.trajinfo.info_single import Capturing
from encodermap.trajinfo.trajinfo_utils import load_CVs_ensembletraj


import encodermap as em  # isort: skip

try:
    # Local Folder Imports
    from .conftest import expensive_test
    from .test_featurizer import assert_allclose_periodic
except ImportError:
    # Encodermap imports
    from conftest import expensive_test
    from test_featurizer import assert_allclose_periodic


################################################################################
# Helper
################################################################################


class MockSingleTraj:
    index = (None,)
    _calc_CV = em.SingleTraj.__dict__["_calc_CV"]
    load_CV = em.SingleTraj.__dict__["load_CV"]

    def __init__(self, traj_num):
        self._CVs = xr.Dataset()
        self.traj_num = traj_num

    @property
    def CVs(self):
        return self._calc_CV()


class MockTrajEnsemble:
    _calc_CV = em.TrajEnsemble.__dict__["_calc_CV"]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        self._index += 1
        try:
            return self.trajs[self._index - 1]
        except IndexError:
            raise StopIteration

    def __getitem__(self, key):
        return self.trajs[key]

    @property
    def CVs(self):
        return self._calc_CV()


@functools.cache
def is_wsl() -> bool:
    """
    detects if Python is running in WSL
    """
    return "microsoft-standard-WSL" in platform.uname().release


def test_trajs_with_dataset(
    dataset: Union[xr.Dataset, str],
    path: Optional[str],
) -> bool:
    if isinstance(dataset, str):
        em.kondata.get_from_kondata(
            dataset,
            output=path,
            mk_parentdir=True,
            silence_overwrite_message=True,
        )
        nc_files = list(Path(path).glob("*.nc"))
        assert len(nc_files) == 1, f"{nc_files=} {path=} {dataset=}"
        dataset = xr.load_dataset(nc_files[0], engine="h5netcdf")

    MockTrajEnsemble.CVs = em.TrajEnsemble.CVs

    trajs = MockTrajEnsemble()
    trajs.trajs = [MockSingleTraj(i) for i in dataset.coords["traj_num"]]
    for traj in trajs:
        test = dataset.sel(traj_num=traj.traj_num).dropna("frame_num", how="all")
        print(test)
        traj.n_frames = len(test.coords["frame_num"])
        traj._traj_file = Path(f"/tmp/traj_{traj.traj_num:02d}.xtc")
        traj.traj_file = f"/tmp/traj_{traj.traj_num:02d}.xtc"
    load_CVs_ensembletraj(trajs, dataset)
    side_dihedrals = trajs.CVs["side_dihedrals"]

    class P:
        use_backbone_angles = True
        use_sidechains = True
        model_api = "functional"
        sparse = True
        l2_reg_constant = 0.001
        periodicity = 2 * np.pi
        n_neurons = [126, 126, 2]
        activation_functions = ["", "relu", "tanh", ""]
        tensorboard = False
        cartesian_pwd_start = 1
        cartesian_pwd_stop = None
        cartesian_pwd_step = 3
        write_summary = False
        trainable_dense_to_sparse = False
        multimer_training = None
        multimer_topology_classes = None
        multimer_connection_bridges = None
        multimer_lengths = None
        reconstruct_sidechains = False

    full_dataset = em.AngleDihedralCartesianEncoderMap.get_train_data_from_trajs(
        trajs, P
    )
    assert full_dataset[0]
    full_dataset = full_dataset[2]

    input_dataset = tf.data.Dataset.from_tensor_slices(
        (
            full_dataset["central_angles"],
            full_dataset["central_dihedrals"],
            full_dataset["central_cartesians"],
            full_dataset["central_distances"],
            full_dataset["side_dihedrals"],
        )
    )
    input_dataset = input_dataset.shuffle(
        buffer_size=side_dihedrals.shape[0],
        reshuffle_each_iteration=True,
    )
    input_dataset = input_dataset.batch(256)

    full_model = em.models.gen_functional_model(input_dataset, P)
    central_angles_sparse_tensor = full_dataset["central_angles"]
    central_dihedrals_sparse_tensor = full_dataset["central_dihedrals"]
    side_dihedrals_sparse_tensor = full_dataset["side_dihedrals"]
    central_cartesians_sparse_tensor = full_dataset["central_cartesians"]
    central_distances_sparse_tensor = full_dataset["central_distances"]

    inp_side_dihedrals = tf.keras.layers.Input(
        shape=(side_dihedrals.shape[1],),
        name="input_side_dihedrals",
        sparse=True,
    )
    z = tf.keras.layers.Dense(side_dihedrals.shape[1])(inp_side_dihedrals)
    model = tf.keras.models.Model(
        inputs=inp_side_dihedrals,
        outputs=z,
    )

    test = model(side_dihedrals_sparse_tensor).numpy()
    one = np.any(np.isnan(side_dihedrals))
    two = np.any(np.isnan(test))
    latent = full_model.encoder_model(
        (
            central_angles_sparse_tensor,
            central_dihedrals_sparse_tensor,
            side_dihedrals_sparse_tensor,
        )
    )
    three = np.any(np.isnan(latent))
    output = full_model(
        (
            central_angles_sparse_tensor,
            central_dihedrals_sparse_tensor,
            central_cartesians_sparse_tensor,
            central_distances_sparse_tensor,
            side_dihedrals_sparse_tensor,
        )
    )
    four = np.any([np.array([np.any(np.isnan(o)) for o in output])])

    return one and not two and not three and not four


def custom_get_loss_dense(self, inp: Any) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    tf.ensure_shape(
        inp[2],
        (None, 30, 3),
    )
    return (
        tf.constant(0.0, dtype=tf.float32),
        inp[2],
        inp[2],
    )


def custom_get_loss_sparse(self, inp: Any) -> tf.Tensor:
    assert inp[2].shape.as_list() == [None, 90]
    return (
        tf.constant(0.0, dtype=tf.float32),
        inp[2],
        inp[2],
    )


################################################################################
# TestSuites
################################################################################


def log_successful_test(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if "success_msg" in kwargs:
            msg = kwargs.pop("success_msg")
        else:
            msg = None
            # msg = f"Test {func.__name__} in class {self.__class__.__name__} was successful."
        out = func(self, *args, **kwargs)
        if msg is not None:
            self.console.log(msg)
        return out

    return wrapper


def for_all_test_methods(decorator):
    def decorate(cls):
        for attr in dir(cls):
            if attr.startswith("assert") and callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def nan_after_100_iter_generator(
    shapes: list[Union[tuple[int, int], tuple[int, int, int]]],
    sparse_indices: Any,
) -> Generator:
    # we will do a theoretical protein with 10 residues.
    # that means, we need 30 coordinates, 29 distances, 28 angles and 27 dihedrals
    # we will do 35 sidechain dihedrals
    # the dataset will be sparse, so that the first topology provides a full, dense
    # dataset, the second topology will be from a protein with 9 residues, so that
    # 3 coordinates, 3 distances, 3 angles, 3 dihedrals are missing
    # furthermore, the second topology will be sparse in the sidechains at indices
    # [[0, ..., 20, 27, ..., 32]]
    # the dense and not dense inputs will always present 5 samples, but along
    # the sample dimension they will be mixed

    # Encodermap imports
    from encodermap.autoencoder.autoencoder import np_to_sparse_tensor

    i = 0
    while True:
        data = []
        for k, (s, ss) in enumerate(zip(shapes, sparse_indices)):
            datum = np.random.random(s).astype("float32")
            sparse_rows = np.random.choice(np.arange(10), 5)
            if isinstance(ss, np.ndarray):
                rows = sparse_rows.astype(np.intp)
                columns = ss.astype(np.intp)
                datum[rows[:, np.newaxis], columns] = np.nan
            else:
                datum[sparse_rows, ss] = np.nan

            # reshape cartesians to rank 2
            if datum.ndim == 3:
                datum = datum.reshape(10, -1)

            # add a cheeky nan to iteration 8
            if i == 200:
                indices = np.stack(np.where(~np.isnan(datum))).T.astype("int64")
                dense_shape = datum.shape
                datum[0, 0] = np.nan
                inde = ~np.isnan(datum)
                inde[0, 0] = True
                datum = datum[inde].flatten()
                datum = tf.sparse.SparseTensor(indices, datum, dense_shape)
            else:
                datum = np_to_sparse_tensor(datum)
            data.append(datum)
        if i <= 300:
            yield tuple(data)
        else:
            break
        i += 1


class TestParameters(tf.test.TestCase):
    def test_parameter_can_load_arbitrary_files(self):
        # Standard Library Imports
        import json

        with tempfile.TemporaryDirectory() as td:
            tempdir = Path(td)
            params = em.Parameters()
            as_dict = params.to_dict()
            as_dict["bad_param"] = 2
            self.assertIn("main_path", as_dict)
            with open(tempdir / "tmp.json", "w+") as f:
                json.dump(as_dict, f)
            with Capturing() as output:
                new_params = em.Parameters.from_file(tempdir / "tmp.json")
            self.assertTrue(any("Dropping" in o and "bad_param" in o for o in output))


@for_all_test_methods(log_successful_test)
@skip_all_tests_except_env_var_specified(unittest.skip)
class TestAutoencoder(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.console = Console(width=150)
        return cls

    def assertHasAttr(self, obj, intendedAttr):
        """Helper to check whether an attr is present."""
        testBool = hasattr(obj, intendedAttr)
        self.assertTrue(
            testBool, msg=f"obj lacking an attribute. {obj=}, {intendedAttr=}"
        )

    def assertAllClosePeriodic(
        self,
        a: np.ndarray,
        b: np.ndarray,
        rtol: float = 1e-7,
        atol: float = 0.0,
        equal_nan: bool = True,
        err_msg: str = "",
        verbose: bool = True,
        periodicity: float = 2 * np.pi,
        max_percentage_mismatched: float = 0.0,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        (a, b) = self.evaluate_if_both_tensors(a, b)
        a = self._GetNdArray(a)
        b = self._GetNdArray(b)
        try:
            assert_allclose_periodic(
                actual=a,
                desired=b,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                err_msg=err_msg,
                verbose=verbose,
                periodicity=periodicity,
                max_percentage_mismatched=max_percentage_mismatched,
                indices=indices,
            )
        except AssertionError as e:
            self.fail(str(e))

    def test_metrics(self):
        # Encodermap imports
        from encodermap.callbacks.metrics import (
            ADCClashMetric,
            ADCRMSDMetric,
            rmsd_numpy,
        )

        parameters = em.Parameters(batch_size=128)
        metric = em.callbacks.ADCClashMetric(
            distance_unit="nm",
            parameters=parameters,
        )

        cartesians = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                [
                    [3.0, 0.0, 0.0],
                    [3.01, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
            ]
        ).astype("float32")
        cartesians2 = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                    [0.02, 0.0, 0.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.99, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
            ]
        ).astype("float32")
        # metric.update_state(tf.constant(0.0), cartesians)
        # self.assertAllEqual(
        #     metric.result(),
        #     1.0,
        # )
        # metric.update_state(tf.constant(0.0), cartesians2)
        # self.assertAllEqual(
        #     metric.result(),
        #     2.0,
        # )

        rmsd = rmsd_numpy(cartesians, cartesians)
        self.assertAllClose(
            rmsd,
            np.array([0.0, 0.0]),
            atol=1e-3,
        )
        rmsd = rmsd_numpy(cartesians, cartesians2)
        self.assertNotAllClose(
            rmsd,
            np.array([0.0, 0.0]),
            atol=1e-3,
        )
        rmsd_metric = ADCRMSDMetric(parameters=em.Parameters(batch_size=2))
        rmsd_metric.update_state(cartesians, cartesians)
        self.assertAllClose(
            rmsd_metric.result(),
            np.array([0.0, 0.0]),
            atol=1e-2,
        )
        rmsd_metric.update_state(cartesians, cartesians2)
        self.assertNotAllClose(
            rmsd_metric.result(),
            np.array([0.0, 0.0]),
            atol=1e-3,
        )
        cartesians = np.random.random((parameters.batch_size, 100, 3)).astype("float32")
        dataset = tf.data.Dataset.from_tensor_slices((cartesians, cartesians))
        dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.repeat()
        dataset = dataset.batch(parameters.batch_size)

        for d in dataset:
            break
        self.assertEqual(d[0].shape[0], 128, msg=f"{parameters.batch_size=}")

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Reshape(
                    target_shape=(100 * 3,),
                    input_shape=(100, 3),
                ),
                tf.keras.layers.Reshape(
                    target_shape=(100, 3),
                    input_shape=(100 * 3,),
                ),
            ]
        )

        # ensure input equals output
        test = model.predict(cartesians)
        self.assertAllClose(test, cartesians)

        @tf.function
        def reconstruction_loss(y_true, y_pred):
            return tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true)))

        model.compile(
            optimizer="Adam",
            loss=[reconstruction_loss],
            metrics=[metric],
        )

        history = model.fit(dataset, epochs=3, steps_per_epoch=10, verbose=0)
        self.assertIsNotNone(history)
        self.assertIn(
            "ADCClashMetric",
            history.history,
        )

        output_dir = Path(
            em.get_from_kondata(
                "linear_dimers",
                mk_parentdir=True,
                silence_overwrite_message=True,
            )
        )
        trajs = em.load(output_dir / "trajs.h5")
        total_steps = 150
        main_path = em.misc.run_path(output_dir / "runs")
        parameters = dict(
            n_steps=total_steps,
            main_path=main_path,
            cartesian_cost_scale=1,
            cartesian_cost_variant="mean_abs",
            cartesian_cost_scale_soft_start=(
                int(total_steps / 10 * 9),
                int(total_steps / 10 * 9) + total_steps // 50,
            ),
            cartesian_pwd_start=1,
            cartesian_pwd_step=3,
            dihedral_cost_scale=1,
            dihedral_cost_variant="mean_abs",
            distance_cost_scale=0,
            cartesian_distance_cost_scale=100,
            cartesian_dist_sig_parameters=[40, 10, 5, 1, 2, 5],
            checkpoint_step=max(1, int(total_steps / 10)),
            l2_reg_constant=0.001,
            center_cost_scale=0,
            tensorboard=True,
            use_sidechains=False,
            use_backbone_angles=True,
            batch_size=10,
            summary_step=10,
            learning_rate=0.0001,
            track_clashes=True,
            track_RMSD=True,
            angle_cost_reference=0.046751507,
            cartesian_cost_reference=0.62146354,
            dihedral_cost_reference=0.89925313,
        )
        parameters = em.ADCParameters(**parameters)
        emap = em.AngleDihedralCartesianEncoderMap(
            parameters=parameters,
            trajs=trajs,
            read_only=False,
        )
        self.assertTrue(
            any(
                [
                    isinstance(i, em.callbacks.metrics.ADCClashMetric)
                    for i in emap.metrics
                ]
            ),
            msg=f"Metric not in {emap.metrics=}",
        )
        self.assertGreater(
            len(emap.loss),
            0,
        )
        history = emap.train()
        self.assertIn(
            "ADCClashMetric", history.history.keys(), msg=(f"{history.history.keys()=}")
        )

        # Third Party Imports
        from tensorflow.python.summary.summary_iterator import summary_iterator

        # Encodermap imports
        from test_tf1_tf2_deterministic import data_loss_iterator

        tfevents_files = list(Path(parameters.main_path).rglob("*tfevents*"))
        all_tags = set()
        for records_file in tfevents_files:
            records = summary_iterator(str(records_file))
            for i, summary in enumerate(data_loss_iterator(records)):
                for j, v in enumerate(summary.summary.value):
                    if "clash" in v.tag.lower():
                        break
                    else:
                        all_tags.add(v.tag)
                else:
                    continue
                break
            else:
                continue
            break
        else:
            self.fail(
                f"Did not encounter the clash metric in the tfevents files. Here "
                f"are all the tags, I've encountered:\n{all_tags}"
            )

    @unittest.skip("This test needs too much memory.")
    def test_sidechain_reconstruction(self):
        # Standard Library Imports
        from types import SimpleNamespace

        # Third Party Imports
        from mdtraj.geometry.angle import _angle
        from mdtraj.geometry.dihedral import _dihedral
        from mdtraj.geometry.distance import _distance

        # Encodermap imports
        from encodermap.misc.backmapping import _full_backmapping_np

        # test the values
        output_dir = Path(
            em.get_from_kondata(
                "Ub_K11_mutants",
                mk_parentdir=True,
                silence_overwrite_message=True,
            )
        )
        h5_file = output_dir / "subset_full_features.h5"
        overwrite = False

        custom_aas = {
            "KAC": (
                "K",
                {
                    "optional_bonds": [
                        ("-C", "N"),  # the peptide bond to the previous aa
                        ("N", "CA"),
                        ("N", "H"),
                        ("CA", "C"),
                        ("C", "O"),
                        ("CA", "CB"),
                        ("CB", "CG"),
                        ("CG", "CD"),
                        ("CD", "CE"),
                        ("CE", "NZ"),
                        ("NZ", "HZ"),
                        ("NZ", "CH"),
                        ("CH", "OI2"),
                        ("CH", "CI1"),
                        ("C", "+N"),  # the peptide bond to the next aa
                    ],
                    "CHI1": ["N", "CA", "CB", "CG"],
                    "CHI2": ["CA", "CB", "CG", "CD"],
                    "CHI3": ["CB", "CG", "CD", "CE"],
                    "CHI4": ["CG", "CD", "CE", "NZ"],
                    "CHI5": ["CD", "CE", "NZ", "CH"],
                },
            )
        }

        traj = em.load(
            trajs=output_dir / "Ub_K11Ac_I/traj.xtc",
            tops=output_dir / "Ub_K11Ac_I/start.pdb",
        )
        traj.load_custom_topology(custom_aas)
        traj.load_CV(["all_cartesians"])

        labels = traj._CVs.all_cartesians.coords["ALLATOM"]
        self.assertIn("ATOM    CH:  111 KAC:  11 CHAIN 0", labels)
        for l in labels.values[1::3].tolist()[:10]:
            self.assertIn("CA", l)
            print(l)
        sidechain_info = traj.sidechain_info()
        n_residues = max(list(sidechain_info.keys()))
        atom = n_residues * 3
        indices = []
        for residue, n_sidechains_in_residue in sidechain_info.items():
            if n_sidechains_in_residue == 0:
                continue
            if residue == 1:
                atom += n_sidechains_in_residue
            else:
                atom += n_sidechains_in_residue + 1
            indices.append(atom)
        labels = labels.values[indices]
        self.assertIn("CE:    8 MET", labels[0])
        self.assertEqual("ATOM    CH:  111 KAC:  11 CHAIN 0", labels[9])

        if not h5_file.is_file() or overwrite:
            h5_file.unlink(missing_ok=True)
            trajs = em.load(
                [
                    output_dir / "Ub_K11Ac_I/traj.xtc",
                    output_dir / "Ub_K11C_I/traj.xtc",
                    output_dir / "Ub_K11Q_I/traj.xtc",
                    output_dir / "Ub_K11R_I/traj.xtc",
                    output_dir / "Ub_wt_I/traj.xtc",
                ],
                [
                    output_dir / "Ub_K11Ac_I/start.pdb",
                    output_dir / "Ub_K11C_I/start.pdb",
                    output_dir / "Ub_K11Q_I/start.pdb",
                    output_dir / "Ub_K11R_I/start.pdb",
                    output_dir / "Ub_wt_I/start.pdb",
                ],
                common_str=[
                    "Ac",
                    "C",
                    "Q",
                    "R",
                    "wt",
                ],
            )
            trajs.load_custom_topology(custom_aas)
            trajs.load_CVs(
                data=[
                    "CentralCartesians",
                    "CentralBondDistances",
                    "CentralAngles",
                    "CentralDihedrals",
                    "SideChainDihedrals",
                    "SideChainCartesians",
                    "AllCartesians",
                    "SideChainBondDistances",
                    "SideChainAngles",
                ],
                ensemble=True,
            )
            trajs.save(h5_file)
        else:
            trajs = em.load(h5_file)

        trajs_subset = em.TrajEnsemble([t[:5] for t in trajs])
        self.assertEqual(
            trajs_subset.n_frames,
            trajs_subset.n_trajs * 5,
        )

        data = trajs._CVs.side_distances.sel(traj_num=1)
        index = (
            data.coords["SIDE_DISTANCES"]
            .str.split(dim="splitted")
            .values[:, 2]
            .astype(int)
            == 11
        )
        data = np.nan_to_num(data.sel(SIDE_DISTANCES=index).values, False, 1)
        self.assertAllGreater(data, 0.0)

        input = (
            trajs_subset._CVs.stack({"frame": ("traj_num", "frame_num")})
            .transpose("frame", ...)
            .dropna("frame", how="all")
        )
        input = [
            np.nan_to_num(input.central_distances.values, copy=False, nan=1.0),
            np.nan_to_num(input.central_angles.values, copy=False, nan=np.pi / 2),
            np.nan_to_num(input.central_dihedrals.values, copy=False, nan=np.pi / 2),
            np.nan_to_num(input.side_distances.values, copy=False, nan=1.0),
            np.nan_to_num(input.side_angles.values, copy=False, nan=np.pi / 2),
            np.nan_to_num(input.side_dihedrals.values, copy=False, nan=np.pi / 2),
        ]
        self.assertAllGreater(input[3], 0.0)

        for i in input:
            self.assertTrue(not np.any(np.isnan(i)))

        out, buf, indices = _full_backmapping_np(
            trajs.sidechain_info(),
            *input,
            return_indices=True,
        )
        with open(
            Path(__file__).resolve().parent
            / "data/sidechain_reconstruction_UbAc_expected.png",
            "bw",
        ) as f:
            f.write(buf.read())
        self.assertEqual(
            out.shape[-1],
            3,
        )

        # a dummy traj for MDTRaj
        # Here's to duck typing
        dummy_traj = SimpleNamespace(
            xyz=out,
            n_atoms=out.shape[1],
            n_frames=out.shape[0],
        )

        # distances
        self.assertEqual(indices["central_distance_indices"].shape, (227, 2))
        out_central_distances = _distance(
            xyz=out, pairs=indices["central_distance_indices"]
        )
        self.assertEqual(out_central_distances.shape, (25, 227))
        self.assertAllClose(out_central_distances, input[0], rtol=1e-3)
        out_side_distances = _distance(xyz=out, pairs=indices["side_distance_indices"])
        self.assertAllClose(out_side_distances, input[3], rtol=1e-3)

        # angles
        # Encodermap imports
        from encodermap.loading.features import CentralAngles

        out_central_angles = np.empty(
            shape=(25, indices["central_angles_indices"].shape[0]), dtype="float32"
        )
        index_arr = []
        labels = CentralAngles(trajs_subset[0]).describe()
        for i in range(5):
            index_arr.append([f"{i} {l}" for l in labels])
        index_arr = np.array(index_arr)
        _angle(
            traj=dummy_traj,
            angle_indices=indices["central_angles_indices"],
            out=out_central_angles,
            periodic=False,
        )
        self.assertAllClosePeriodic(
            out_central_angles,
            input[1],
            atol=0.03,
            indices=index_arr,
        )

        # Encodermap imports
        from encodermap.loading.features import SideChainAngles

        out_side_angles = np.empty(
            shape=(25, indices["side_angles_indices"].shape[0]), dtype="float32"
        )
        index_arr = []
        labels = SideChainAngles(trajs_subset[0]).describe()
        for i in range(5):
            index_arr.append([f"{i} {l}" for l in labels])
        index_arr = np.array(index_arr)
        _angle(
            traj=dummy_traj,
            angle_indices=indices["side_angles_indices"],
            out=out_side_angles,
            periodic=False,
        )
        self.assertTrue(
            not np.any(np.isnan(out_side_angles)),
            msg=(f"Detected NaN in the out_side_angles"),
        )
        self.assertAllClosePeriodic(
            out_side_angles, input[4], atol=0.03, indices=index_arr
        )

        # dihedrals
        out_central_dihedrals = np.empty(
            shape=(25, indices["central_dihedrals_indices"].shape[0]), dtype="float32"
        )
        _dihedral(
            traj=dummy_traj,
            indices=indices["central_dihedrals_indices"],
            out=out_central_dihedrals,
            periodic=False,
        )
        self.assertAllClosePeriodic(
            out_central_dihedrals,
            input[2],
            atol=0.03,
        )
        out_side_dihedrals = np.empty(
            shape=(25, indices["side_dihedrals_indices"].shape[0]), dtype="float32"
        )
        _dihedral(
            traj=dummy_traj,
            indices=indices["side_dihedrals_indices"],
            out=out_side_dihedrals,
            periodic=False,
        )
        self.assertAllClosePeriodic(
            out_side_dihedrals,
            input[5],
            atol=0.03,
        )

        # continue with some other tests
        self.assertEqual(
            trajs._CVs.side_cartesians.coords["SIDEATOM"].shape,
            (230,),
            msg=(f"{trajs._CVs.side_cartesians.coords['SIDEATOM']=}"),
        )
        self.assertTrue(
            trajs._CVs.side_cartesians.coords["SIDEATOM"][-1].str.endswith("74")
        )
        self.assertTrue(
            trajs._CVs.all_cartesians.coords["ALLATOM"][-1].str.endswith("74 s")
        )
        index1 = trajs._CVs.side_cartesians_feature_indices.values[
            0, trajs._CVs.side_cartesians.coords["SIDEATOM"].str.endswith("11")
        ]
        index2 = trajs._CVs.side_distances_feature_indices.values[
            0, trajs._CVs.side_distances.coords["SIDE_DISTANCES"].str.endswith("11"), :2
        ]
        index3 = trajs._CVs.side_angles_feature_indices.values[
            0, trajs._CVs.side_angles.coords["SIDE_ANGLES"].str.endswith("11"), :3
        ]
        index4 = trajs._CVs.side_dihedrals_feature_indices.values[
            0, trajs._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"].str.endswith("11")
        ]
        for a in index1:
            print("ATOM", trajs.tsel[0].top.atom(int(a)))
        print(index2)
        for a, b in index2:
            print(
                "DIST", trajs.tsel[0].top.atom(int(a)), trajs.tsel[0].top.atom(int(b))
            )
        for a, b, c in index3:
            print(
                "ANGL",
                trajs.tsel[0].top.atom(int(a)),
                trajs.tsel[0].top.atom(int(b)),
                trajs.tsel[0].top.atom(int(c)),
            )
        for a, b, c, d in index4:
            print(
                "DIHE",
                trajs.tsel[0].top.atom(int(a)),
                trajs.tsel[0].top.atom(int(b)),
                trajs.tsel[0].top.atom(int(c)),
                trajs.tsel[0].top.atom(int(d)),
            )
        self.assertEqual(len(index1), len(index2))
        self.assertEqual(len(index1), len(index3))
        self.assertEqual(len(index1), len(index4) + 1)

        # this dict is presented by TrajEnsemble to make the new layer work
        sidechain_info = trajs.sidechain_info()

        # this checks the residue K11, which should have 5 sidechains for KAC
        self.assertEqual(
            sidechain_info[-1][11],
            5,
        )
        # check that for C, the number of sidechain dihedrals is 1
        self.assertEqual(
            sidechain_info[1][11],
            1,
        )

        # instantiate the layer
        # Encodermap imports
        from encodermap.models.layers import BackMapLayerWithSidechains

        input = tuple(tf.convert_to_tensor(i, dtype="float32") for i in input)
        layer = BackMapLayerWithSidechains(
            feature_description=trajs_subset.sidechain_info()
        )

        # Test saving the layer
        model = tf.keras.models.Sequential([layer])
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model.save(td / "tmp.keras")
            new_model = tf.keras.models.load_model(td / "tmp.keras")
            self.assertIsNotNone(new_model)

        # call the layer
        out = layer(input).numpy()
        self.assertNotIn(np.nan, out)
        self.assertNotIn(float("inf"), out)
        dummy_traj = SimpleNamespace(
            xyz=out,
            n_atoms=out.shape[1],
            n_frames=out.shape[0],
        )
        self.assertEqual(
            out.shape[-1],
            3,
        )
        self.assertFalse(
            np.any(np.isnan(out)),
            msg=(f"There are nan's in the layer output:\n\n{out}"),
        )

        # distances
        out_central_distances = _distance(
            xyz=out, pairs=indices["central_distance_indices"]
        )
        self.assertEqual(out_central_distances.shape, (25, 227))
        self.assertAllClose(out_central_distances, input[0], rtol=1e-3)
        out_side_distances = _distance(xyz=out, pairs=indices["side_distance_indices"])
        self.assertAllClose(out_side_distances, input[3], rtol=1e-3)

        # angles
        # Encodermap imports
        from encodermap.loading.features import CentralAngles

        out_central_angles = np.empty(
            shape=(25, indices["central_angles_indices"].shape[0]), dtype="float32"
        )
        index_arr = []
        labels = CentralAngles(trajs_subset[0]).describe()
        for i in range(5):
            index_arr.append([f"{i} {l}" for l in labels])
        index_arr = np.array(index_arr)
        _angle(
            traj=dummy_traj,
            angle_indices=indices["central_angles_indices"],
            out=out_central_angles,
            periodic=False,
        )
        self.assertAllClosePeriodic(
            out_central_angles,
            input[1],
            atol=0.03,
            indices=index_arr,
        )

        # Encodermap imports
        from encodermap.loading.features import SideChainAngles

        out_side_angles = np.empty(
            shape=(25, indices["side_angles_indices"].shape[0]), dtype="float32"
        )
        index_arr = []
        labels = SideChainAngles(trajs_subset[0]).describe()
        for i in range(5):
            index_arr.append([f"{i} {l}" for l in labels])
        index_arr = np.array(index_arr)
        _angle(
            traj=dummy_traj,
            angle_indices=indices["side_angles_indices"],
            out=out_side_angles,
            periodic=False,
        )
        self.assertAllClosePeriodic(
            out_side_angles, input[4], atol=0.03, indices=index_arr
        )

        # dihedrals
        out_central_dihedrals = np.empty(
            shape=(25, indices["central_dihedrals_indices"].shape[0]), dtype="float32"
        )
        _dihedral(
            traj=dummy_traj,
            indices=indices["central_dihedrals_indices"],
            out=out_central_dihedrals,
            periodic=False,
        )
        self.assertAllClosePeriodic(
            out_central_dihedrals,
            input[2],
            atol=0.03,
        )
        out_side_dihedrals = np.empty(
            shape=(25, indices["side_dihedrals_indices"].shape[0]), dtype="float32"
        )
        _dihedral(
            traj=dummy_traj,
            indices=indices["side_dihedrals_indices"],
            out=out_side_dihedrals,
            periodic=False,
        )
        self.assertAllClosePeriodic(
            out_side_dihedrals,
            input[5],
            atol=0.03,
        )

        main_path = em.misc.run_path(output_dir / "runs")
        total_steps = 10
        parameters = em.ADCParameters(
            n_steps=total_steps,
            main_path=main_path,
            cartesian_cost_scale=1,
            cartesian_cost_variant="mean_abs",
            cartesian_cost_scale_soft_start=(
                int(total_steps / 10 * 9),
                int(total_steps / 10 * 9) + total_steps // 50,
            ),
            cartesian_pwd_start=1,
            cartesian_pwd_step=3,
            dihedral_cost_scale=1,
            dihedral_cost_variant="mean_abs",
            distance_cost_scale=0,
            cartesian_distance_cost_scale=100,
            cartesian_dist_sig_parameters=[40, 10, 5, 1, 2, 5],
            checkpoint_step=max(1, int(total_steps / 10)),
            l2_reg_constant=0.001,
            center_cost_scale=0,
            tensorboard=True,
            use_sidechains=True,
            use_backbone_angles=True,
            batch_size=10,
            summary_step=10,
            learning_rate=0.0001,
            track_clashes=True,
            angle_cost_reference=0.046751507,
            cartesian_cost_reference=0.62146354,
            dihedral_cost_reference=0.89925313,
            reconstruct_sidechains=True,
        )
        self.assertTrue(parameters.reconstruct_sidechains)
        e_map = em.AngleDihedralCartesianEncoderMap(
            trajs=trajs,
            parameters=parameters,
        )

        # test the pairwise distance layer
        # Encodermap imports
        from encodermap.models.layers import PairwiseDistances

        pairwise_distances = PairwiseDistances(
            parameters=parameters,
            print_name="pairwise_distances_of_reconstructed_sidechains",
        )
        indices = trajs_subset._CVs.all_cartesians.coords["ALLATOM"][
            pairwise_distances.indices
        ].values.tolist()
        central_coords = [i for i in indices if i.endswith("c")]
        side_coords = [i for i in indices if i.endswith("s")]
        self.assertTrue(all([i.startswith("2 ") for i in central_coords]))
        residx = np.array([int(i.split()[1]) for i in central_coords])
        self.assertAllEqual(residx, np.arange(1, 77))
        test = pairwise_distances(out)
        self.assertEqual(
            test.shape, (25, 10296)  # 10296 is nCr(144, 2) and 144 is len(indices)
        )

        # get dense values and then pass them to the instantiated layer
        for batch in e_map.dataset:
            break
        (
            sparse_inp_central_angles,
            sparse_inp_central_dihedrals,
            sparse_inp_all_cartesians,
            sparse_inp_central_distances,
            sparse_inp_side_angles,
            sparse_inp_side_dihedrals,
            sparse_inp_side_distances,
        ) = batch
        out = e_map.model(batch)

        inp_central_angles = e_map.model.get_dense_model_central_angles(
            sparse_inp_central_angles, training=True
        )
        self.assertTrue(not tf.math.reduce_any(tf.math.is_nan(inp_central_angles)))
        inp_central_dihedrals = e_map.model.get_dense_model_central_dihedrals(
            sparse_inp_central_dihedrals, training=True
        )
        self.assertTrue(not tf.math.reduce_any(tf.math.is_nan(inp_central_dihedrals)))
        inp_central_distances = e_map.model.get_dense_model_distances(
            sparse_inp_central_distances, training=True
        )
        self.assertTrue(not tf.math.reduce_any(tf.math.is_nan(inp_central_distances)))
        inp_side_angles = e_map.model.get_dense_model_side_angles(
            sparse_inp_side_angles, training=True
        )
        self.assertTrue(not tf.math.reduce_any(tf.math.is_nan(inp_side_angles)))
        inp_side_dihedrals = e_map.model.get_dense_model_side_dihedrals(
            sparse_inp_side_dihedrals, training=True
        )
        self.assertTrue(not tf.math.reduce_any(tf.math.is_nan(inp_side_dihedrals)))
        inp_side_distances = e_map.model.get_dense_model_side_distances(
            sparse_inp_side_distances, training=True
        )
        self.assertTrue(not tf.math.reduce_any(tf.math.is_nan(inp_side_distances)))

        out = layer(
            (
                inp_central_distances,
                inp_central_angles,
                inp_central_dihedrals,
                inp_side_distances,
                inp_side_angles,
                inp_side_dihedrals,
            )
        )
        self.assertTrue(not tf.math.reduce_any(tf.math.is_nan(out)))
        out = e_map.encode(trajs_subset[0])
        self.assertTrue(not tf.math.reduce_any(tf.math.is_nan(out)))
        e_map.train()
        out = e_map.encode(trajs[0], top="KAc")
        self.assertIsInstance(
            out,
            mdtraj.Trajectory,
        )

    def test_none_interrupt_callback(self):
        """Test a callback, that stops training, when a part of the weights
        becomes np.nan value is encountered."""

        shapes = [
            (10, 28),
            (10, 27),
            (10, 30, 3),
            (10, 29),
            (10, 35),
        ]
        sparse_indices = [
            slice(25, None),
            slice(24, None),
            slice(27, None),
            slice(26, None),
            np.array([20, 21, 22, 23, 24, 25, 26, 32, 33, 34]),
        ]
        tensor_specs = []
        for s in shapes:
            if len(s) == 2:
                tensor_specs.append(tf.SparseTensorSpec(shape=s, dtype="float32"))
            else:
                tensor_specs.append(
                    tf.SparseTensorSpec(shape=(s[0], s[1] * s[2]), dtype="float32")
                )

        gen = lambda: nan_after_100_iter_generator(
            shapes=shapes, sparse_indices=sparse_indices
        )
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=tuple(tensor_specs),
        )
        p = em.ADCParameters(
            use_backbone_angles=True,
            use_sidechains=True,
        )
        e_map = em.AngleDihedralCartesianEncoderMap(
            parameters=p,
            read_only=True,
            dataset=ds,
        )
        e_map.callbacks.append(em.callbacks.NoneInterruptCallback())
        with self.assertRaises(Exception):
            e_map.train()
        print("Test was a success.")

    @expensive_test
    def test_fat10_training(self):
        total_steps = 10_000
        main_path = Path(__file__).resolve().parent / "data/FAT10/long_training_test"
        if main_path.is_dir():
            shutil.rmtree(main_path)
        main_path.mkdir(parents=True)
        fat10_output_dir = Path(
            em.get_from_kondata(
                "FAT10",
                mk_parentdir=True,
                silence_overwrite_message=True,
            )
        )
        trajs_file = fat10_output_dir / "linear_dimers_and_fat10.h5"
        self.assertTrue(trajs_file.is_file())
        trajs = em.load(trajs_file)

        # alignment and distance histogram
        # Third Party Imports
        import matplotlib.pyplot as plt

        # Encodermap imports
        from encodermap.plot.plotting import distance_histogram
        from test_featurizer import ALIGNMENT

        trajs.parse_clustal_w_alignment(ALIGNMENT)

        CA_pos = trajs.CVs["central_cartesians"][::1000, 1::3]
        fig, axes = plt.subplots(ncols=1, nrows=3)
        distance_histogram(
            CA_pos,
            periodicity=float("inf"),
            sigmoid_parameters=[40, 10, 5, 1, 2, 5],
            axes=axes,
        )
        plt.savefig("/home/kevin/tmp_FAT10.png")

        _ = em.load(
            "/home/kevin/git/encoder_map_private/tests/data/linear_dimers/trajs.h5"
        )
        CA_pos = _.CVs["central_cartesians"][::1000, 1::3]
        plt.close("all")
        fig, axes = plt.subplots(ncols=1, nrows=3)
        distance_histogram(
            CA_pos,
            periodicity=float("inf"),
            sigmoid_parameters=[40, 10, 5, 1, 2, 5],
            axes=axes,
        )
        plt.savefig("/home/kevin/tmp_M1diUb.png")

        raise Exception

        # test that at least some random choices select linear_dimers
        visited_common_strs = set()
        ids = trajs.id
        total = len(ids)
        for i in range(10_000):
            ind = ids[np.random.randint(0, total, 1)[0]]
            frame = trajs[ind[0]][ind[1]]
            visited_common_strs.add(frame.common_str)
            if len(visited_common_strs) == 2:
                break
        else:
            self.fail(
                "Was not able to get a sample from the linear_dimers part of "
                "the dataset after 10,000 steps."
            )

        parameters = dict(
            n_steps=total_steps,
            main_path=main_path,
            cartesian_cost_scale=1,
            cartesian_cost_variant="mean_abs",
            cartesian_cost_scale_soft_start=(
                int(total_steps / 10 * 9),
                int(total_steps / 10 * 9) + total_steps // 50,
            ),
            cartesian_pwd_start=1,
            cartesian_pwd_step=3,
            dihedral_cost_scale=1,
            dihedral_cost_variant="mean_abs",
            distance_cost_scale=0,
            cartesian_distance_cost_scale=100,
            cartesian_dist_sig_parameters=[40, 10, 5, 1, 2, 5],
            checkpoint_step=max(1, int(total_steps / 10)),
            l2_reg_constant=0.001,
            center_cost_scale=0,
            tensorboard=False,
            use_sidechains=True,
            use_backbone_angles=True,
            batch_size=10,
            learning_rate=0.0001,
            angle_cost_reference=18.746202,
            cartesian_cost_reference=25.0,
            dihedral_cost_reference=25.323647,
        )
        parameters = em.ADCParameters(**parameters)

        # test the gen separately
        ds = trajs.batch_iterator(
            batch_size=parameters.batch_size,
            deterministic=True,
            yield_index=True,
        )
        for i, (index, batch) in enumerate(
            ds,
        ):
            self.assertTrue(all([isinstance(i, tf.SparseTensor) for i in batch]))
            # fill sparse tensor with zeros and check whether nans exist
            # nans cannot exist in sparse tensor, otherwise they would
            # propagate
            # make sure all distances are greater than 0
            self.assertAllGreater(batch[3].values, 0.0)
            for t in batch:
                t = tf.sparse.to_dense(t, default_value=0).numpy()
                self.assertTrue(np.all(~np.isnan(t)))
            if i in [1, 5, 10, 20]:
                cartesians = tf.sparse.to_dense(batch[2], np.nan).numpy()
                self.assertEqual(cartesians.shape[1], 495 * 3)
                cartesians = tf.keras.layers.Reshape(
                    target_shape=(cartesians.shape[1] // 3, 3),
                    input_shape=(cartesians.shape[1],),
                    name="reshape_cartesians_test",
                )(cartesians).numpy()
                for ind, cart in zip(index, cartesians):
                    frame = trajs[ind[0]][ind[1]]
                    xyz = frame.central_cartesians[0]
                    cart = cart[~np.isnan(cart).all(-1)]
                    self.assertAllClose(
                        xyz, cart, msg=(f"Cartesians differ at {frame=}")
                    )
            if i == 100:
                break
        ds = trajs.tf_dataset(
            batch_size=parameters.batch_size,
            deterministic=True,
            prefetch=False,
        )

        for i, datum in enumerate(ds):
            for d in datum:
                self.assertTrue(
                    np.all(~np.isnan(tf.sparse.to_dense(d, default_value=0).numpy()))
                )
            if i == 100:
                break

    def test_image_callback_saving(self):
        """The image callback should also be able to save images to a directory."""
        ubi_output_dir = Path(
            em.get_from_kondata(
                "linear_dimers", mk_parentdir=True, silence_overwrite_message=True
            )
        )
        fat10_output_dir = Path(
            em.get_from_kondata(
                "FAT10", mk_parentdir=True, silence_overwrite_message=True
            )
        )

        ubi_trajs = [ubi_output_dir / "01.xtc", ubi_output_dir / "02.xtc"]
        ubi_tops = [f.with_suffix(".pdb") for f in ubi_trajs]
        ubi_trajs = em.TrajEnsemble(
            trajs=ubi_trajs,
            tops=ubi_tops,
            traj_nums=[0, 1],
            common_str=["linear_dimer", "linear_dimer"],
        )
        fat10_trajs = [fat10_output_dir / "01.xtc", fat10_output_dir / "02.xtc"]
        fat10_tops = [f.with_suffix(".pdb") for f in fat10_trajs]
        fat10_trajs = em.TrajEnsemble(
            trajs=fat10_trajs,
            tops=fat10_tops,
            traj_nums=[12, 13],
            common_str=["FAT10", "FAT10"],
        )

        self.assertEqual(ubi_trajs[0].traj_num, 0)
        self.assertEqual(ubi_trajs[1].traj_num, 1)
        self.assertEqual(fat10_trajs[0].traj_num, 12)
        self.assertEqual(fat10_trajs[1].traj_num, 13)

        tmp = fat10_trajs.copy()
        tmp[0].traj_num = 0
        self.assertEqual(tmp[0].traj_num, 0)

        with self.assertRaisesRegex(Exception, r".*overlapping traj_nums.*"):
            _ = ubi_trajs + tmp

        # xr.open_dataset does not tax the system memory at all
        file = fat10_output_dir / "linear_dimers_and_fat10.h5"
        with h5py.File(file, "r") as f:
            self.assertIn("CVs", f.keys())
        self.assertTrue(file.is_file())
        ds = xr.open_dataset(file, group="CVs")
        trajs = ubi_trajs + fat10_trajs
        sub_trajs = trajs[[0, 2]]

        # load the dataset
        sub_trajs.load_CVs(
            ds.sel(traj_num=[0, 12]),
        )
        self.assertIsInstance(sub_trajs, em.TrajEnsemble)
        parameters = em.ADCParameters.load(
            fat10_output_dir / "checkpoints/finished_training/tf2_15/parameters.json"
        )
        parameters.main_path = em.misc.run_path(fat10_output_dir / "runs")
        parameters.summary_step = 1

        emap = em.AngleDihedralCartesianEncoderMap(
            trajs=sub_trajs,
            parameters=parameters,
        )
        emap.add_images_to_tensorboard(save_to_disk=True)
        self.assertIsNotNone(emap.callbacks[-1].save_dir)
        emap.callbacks[-1].on_epoch_end(1)

        file = Path(parameters.main_path) / "train_images/epoch_1.png"
        self.assertTrue(
            file.is_file(),
            msg=(
                f"The add_images_to_tensorboard with save_to_disk=True did not "
                f"produce an image at {file}"
            ),
        )

    def test_sparse_training(self):
        """Piece by piece, build a sparse network and try to find weak points."""
        names = [
            "glu7_asp7_sparse",
            "messy_dataset_large_feature_space",
        ]
        datasets = [
            "glu7_asp7_sparse",
            "messy_dataset_large_feature_space",
        ]
        outdirs = [
            str(Path(__file__).resolve().parent / "data/sparse"),
            str(
                Path(__file__).resolve().parent
                / "data/messy_dataset_large_feature_space"
            ),
        ]
        for name, ds, dr in zip(names, datasets, outdirs):
            self.assertTrue(
                test_trajs_with_dataset(ds, dr),
                msg=f"The dataset {name} produced unexpected nans in the ADC sparse network.",
            )

    def test_shuffled_multi_tensor_dataset_returns_correct_order(self):
        """Hopefully, dataset.shuffle.repeat.batch() returns a dataset, where the
        CVs are also aligned"""
        t1 = np.mgrid[0:512, 0:5][0]
        t2 = np.mgrid[0:512, 0:5][1] + t1
        t3 = t2.copy()
        t3[:, 1:] *= 2

        ds = tf.data.Dataset.from_tensor_slices(
            (
                t1,
                t2,
                t3,
            )
        )
        ds = ds.shuffle(buffer_size=512, reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.batch(256)

        for i, (i1, i2, i3) in enumerate(ds):
            i1 = i1.numpy()
            i2 = i2.numpy()
            i3 = i3.numpy()
            self.assertAllEqual(i1[:, 0], i2[:, 0])
            self.assertAllEqual(i2[:, 0], i3[:, 0])

            test = i1.copy()
            test[:, 1] += 1
            test[:, 2] += 2
            test[:, 3] += 3
            test[:, 4] += 4
            test2 = test.copy()
            test2[:, 1:] *= 2

            self.assertAllEqual(test, i2)
            self.assertAllEqual(test2, i3)

            if i >= 20:
                break

    def test_omega_angles_are_trained_correctly(self):
        """Omega angles should stay within their natural ranges for EncoderMap
        and AngleDihedralCartesianEncoderMap"""
        traj_file = Path(__file__).resolve().parent / "data/asp7.xtc"
        self.assertTrue(traj_file.is_file(), msg=f"{traj_file=} is missing.")
        traj = em.load(
            traj_file,
            Path(__file__).resolve().parent / "data/asp7.pdb",
        )
        self.traj_omega_angles(traj)

    def traj_omega_angles(self, traj):
        traj.load_CV("all")
        p = em.Parameters(
            n_steps=100,
            learning_rate=0.001,
            periodicity=2 * np.pi,
            tensorboard=False,
        )
        highd = traj.central_dihedrals
        emap = em.EncoderMap(p, highd)
        emap.train()
        lowd = emap.encode(highd)
        self.assertEqual(lowd.shape[1], 2)
        self.assertEqual(emap.train_data.shape, highd.shape)
        x_max, y_max = np.amax(lowd, axis=0)
        x_min, y_min = np.amin(lowd, axis=0)
        x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        grid = np.stack([x.ravel(), y.ravel()]).T
        new_highd = emap.decode(grid)
        decoded = emap.decode(lowd)
        generated = emap.generate(lowd)
        self.assertEqual(decoded.shape, highd.shape)
        self.assertEqual(decoded.shape, generated.shape)
        self.assertEqual(new_highd.shape[1], highd.shape[1])
        phi_inds = (
            traj._CVs.central_dihedrals.coords["CENTRAL_DIHEDRALS"]
            .str.contains("PHI")
            .values
        )
        omega_inds = (
            traj._CVs.central_dihedrals.coords["CENTRAL_DIHEDRALS"]
            .str.contains("OMEGA")
            .values
        )
        psi_inds = (
            traj._CVs.central_dihedrals.coords["CENTRAL_DIHEDRALS"]
            .str.contains("PSI")
            .values
        )
        phi_angles = new_highd[:, phi_inds]
        omega_angles = new_highd[:, omega_inds]
        psi_angles = new_highd[:, psi_inds]
        # check [-pi, pi)
        self.assertTrue(np.all(omega_angles > -np.pi))
        self.assertTrue(np.all(omega_angles <= np.pi))

        omegas = omega_angles.flatten()
        omegas += np.pi
        omegas[omegas > np.pi] -= 2 * np.pi

        mu = np.mean(omegas)
        sigma = np.std(omegas)

        self.assertAlmostEqual(mu, 0, places=1)
        self.assertAlmostEqual(
            sigma, 0.06, places=1, msg=f"{sigma=} is not close enough to 0.6"
        )

    def test_normal_autoencoder_has_correct_activations(self):
        """Test, whether the bog-standard EncoderMap Autoencoder has the
        activation functions as the tf1 implementation.

        """
        emap = em.EncoderMap()
        activations_should_be = []
        act_funs = ["", "tanh", "tanh", ""]
        n_neurons = [126, 126, 2]

        # encoder
        for i, (n, act_fun) in enumerate(zip(n_neurons, act_funs[1:])):
            if act_fun == "":
                activations_should_be.append("linear")
            else:
                activations_should_be.append(act_fun)

        # decoder
        n_neurons_with_inputs = [emap.train_data.shape[1]] + n_neurons
        for n, act_fun in zip(n_neurons_with_inputs[-2::-1], act_funs[-2::-1]):
            if act_fun == "":
                activations_should_be.append("linear")
            else:
                activations_should_be.append(act_fun)

        # check
        activations_are = []
        for layer in emap.model.encoder_model.layers + emap.model.decoder_model.layers:
            if hasattr(layer, "activation"):
                activations_are.append(layer.activation.__name__)

        self.assertEqual(
            activations_should_be, activations_are, msg=f"{activations_are=}"
        )

    def test_flatten_and_reshape_for_sparse_cartesians_works(self):
        """As sparse tensors can only be of rank 2 (and not rank 3, as it is
        necessary to describe cartesians with shape (batch_size, n_atoms, 3), the
        sparse training flattening the input cartesians, before constructing a sparse
        tensor. This makes it necessary to reshape the dense output of the model
        to make sure, flattening and reshaping produces the same arrays.

        Note:
            Newer tensorflow versions allow for higher rank sparse
                tensors. However, adjusting EncoderMap's sparse code
                is not a priority.

        """
        cartesians = np.random.random((256, 30, 3)).astype("float32")

        pairwise_dists = np.empty(
            (
                cartesians.shape[0],
                cartesians.shape[1],
                cartesians.shape[1],
            ),
            dtype="float32",
        )
        for i, row in enumerate(cartesians):
            dists = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(
                    row[:, 1::3],
                ),
            )
            pairwise_dists[i] = dists

        cartesians_flattened = tf.convert_to_tensor(
            cartesians.copy().reshape(len(cartesians), -1, order="C")
        )
        cartesians_reshaped = tf.keras.layers.Reshape(
            target_shape=(cartesians_flattened.shape[1] // 3, 3),
            input_shape=(cartesians_flattened.shape[1],),
        )(cartesians_flattened)
        self.assertTrue(np.array_equal(cartesians, cartesians_reshaped.numpy()))

    def test_encodermap_with_artificial_two_state_system(self):
        """Use M1-diUbq as an artificial two-state system. Run a short tf2
        EncoderMap and then cluster the latent space. It is expected to
        show two clusters.

        """
        # Third Party Imports
        from sklearn.cluster import HDBSCAN

        output_dir = Path(em.get_from_kondata("two_state"))
        trajs = em.TrajEnsemble.from_dataset(output_dir / "trajs.h5")
        p = em.ADCParameters(
            main_path=em.misc.run_path(Path(__file__).parent / "data/runs")
        )
        e_map = em.AngleDihedralCartesianEncoderMap(trajs=trajs, parameters=p)
        e_map.p.n_steps = 50
        e_map.train()
        lowd = e_map.encode(trajs.central_dihedrals)
        clusterer = HDBSCAN(min_cluster_size=250).fit(lowd)
        self.assertTrue(
            0 in clusterer.labels_ and 1 in clusterer.labels_,
            msg=(
                f"The artificial two-state system is expected to produce three "
                f"cluster labels: (-1, 0, and 1), "
                f"but {np.unique(clusterer.labels_)=} were produced."
            ),
        )

    @expensive_test
    def test_m1_diubi_state_decoding(self):
        """Uses the decoder to make sure, that a path from one minimum of the
        M1-diUbq projection (which correspond to the start of the non-diverging
        simulations) correctly rebuilds the states. We can use the already trained
        M1-diUbi EncoderMap. See also `run_long_training` in the `tests` directroy."""
        trajs, emap = em.load_project("linear_dimers", load_autoencoder=True)
        # Third Party Imports
        import plotly.express as px

        # quick check, whether the lowd and the highd still fit somewhat
        test_frame = trajs.get_single_frame(0)
        test_lowd = emap.encode(test_frame)
        self.assertAllClose(test_lowd, test_frame.lowd)

        # decode this point for fun
        generated = emap.generate(test_lowd)
        traj = em.SingleTraj(generated)
        traj.load_CV("all")
        rmsd = mdtraj.rmsd(test_frame, generated)
        raise Exception(
            f"{traj.central_dihedrals[0, :5]=}\n\n"
            f"{test_frame.central_dihedrals[0, :5]=}\n\n"
            f"{rmsd=}"
        )

        # now we will decode the lowd point and check
        start = np.array([[-1.718504, -0.1113482]])
        end = np.array([[1.380024, 2.347491]])
        path = np.linspace(start, end, num=100, endpoint=True)

        # Now we will decode a path connecting two regions of high density
        # fig = px.scatter(x=trajs.lowd[::100, 0], y=trajs.lowd[::100, 1])
        # fig.show(renderer="browser")
        raise Exception(f"{trajs._CVs.coords=}")

    @expensive_test
    def test_encodermap_with_dataset(self):
        """Use a fake dataset, that does not shuffle the data (each forward-pass
        gets the same data) and compare the weights/biases/costs of EncoderMap's tf1
        and EncoderMap's tf2 version.

        """
        self.assertIn("em", globals())
        output_dir = em.get_from_kondata(
            "linear_dimers",
            mk_parentdir=True,
            silence_overwrite_message=True,
        )

        # Encodermap imports
        from encodermap.encodermap_tf1.angle_dihedral_cartesian_encodermap import (
            AngleDihedralCartesianEncoderMap,
        )
        from encodermap.encodermap_tf1.misc import (
            add_layer_summaries,
            variable_summaries,
        )
        from encodermap.encodermap_tf1.moldata import MolData

        class CustomMolData(MolData):
            pass

        output_dir = Path(output_dir)
        structure_path = output_dir / "01.tpr"
        trajectory_paths = list(
            sorted(output_dir.glob("*.xtc"), key=lambda x: int(x.stem))
        )
        uni = mda.Universe(structure_path, trajectory_paths)
        selected_atoms = uni.select_atoms(
            "backbone or name H or name O1 or (name CD and resname PRO)"
        )

        custom_moldata = CustomMolData(
            selected_atoms, cache_path=str(output_dir / "moldata_cache")
        )

        for key in [
            "cartesians",
            "central_cartesians",
            "dihedrals",
            "sidedihedrals",
            "angles",
            "lengths",
        ]:
            d = getattr(custom_moldata, key)[:256]
            setattr(custom_moldata, key, d)
            assert len(getattr(custom_moldata, key)) == 256

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                getattr(custom_moldata, "angles"),
                getattr(custom_moldata, "dihedrals"),
                getattr(custom_moldata, "central_cartesians"),
            )
        )
        dataset = dataset.repeat()
        dataset = dataset.batch(256)
        for i in range(5):
            data = dataset.take(1)
            values = []
            for d in data:
                for j in d:
                    values.append(j.numpy())
            a, d, c = values
            self.assertAllClose(a, getattr(custom_moldata, "angles"))
            self.assertAllClose(d, getattr(custom_moldata, "dihedrals"))
            self.assertAllClose(c, getattr(custom_moldata, "central_cartesians"))

        class CustomADCEncodermap(AngleDihedralCartesianEncoderMap):
            def __init__(self, *args, **kwargs):
                self._encoder_layers = {}
                self._called_encoder_layers = {}
                self._decoder_layers = {}
                self._called_decoder_layers = {}
                super().__init__(*args, **kwargs)

            def _setup_data_iterator(self):
                """This function replaces the shuffle from the dataset, creating a dataset,
                that will forever return the same batch of self.p.batch_size samples."""
                self.data_placeholders = tuple(
                    tf.compat.v1.placeholder(dat.dtype, dat.shape)
                    for dat in self.train_data
                )
                self.data_set = tf.compat.v1.data.Dataset.from_tensor_slices(
                    self.data_placeholders
                )
                self.data_set = self.data_set.repeat()
                self.data_set = self.data_set.batch(self.p.batch_size)
                if not tf.executing_eagerly():
                    self.data_iterator = self.data_set.make_initializable_iterator()
                else:
                    self.data_iterator = self.data_set

            def _encode(self, inputs):
                with tf.name_scope("encoder"):
                    if self.p.periodicity < float("inf"):
                        if self.p.periodicity != 2 * np.pi:
                            inputs = inputs / self.p.periodicity * 2 * np.pi
                        self.unit_circle_inputs = tf.concat(
                            [tf.sin(inputs), tf.cos(inputs)], 1
                        )
                        current_layer = self.unit_circle_inputs
                    else:
                        current_layer = inputs

                    assert (
                        len(self.p.n_neurons) == len(self.p.activation_functions) - 1
                    ), "you need one activation function more then layers given in n_neurons"
                    for i, (n_neurons, act_fun) in enumerate(
                        zip(self.p.n_neurons, self.p.activation_functions[1:])
                    ):
                        if act_fun:
                            act_fun = getattr(tf.nn, act_fun)
                        else:
                            act_fun = None
                        variable_summaries(
                            "activation{}".format(i), current_layer, debug=True
                        )
                        dense = tf.compat.v1.layers.Dense(
                            n_neurons,
                            activation=act_fun,
                            kernel_initializer=tf.keras.initializers.Constant(1),
                            kernel_regularizer=self.regularizer,
                            bias_initializer=tf.keras.initializers.Constant(1),
                        )
                        self._encoder_layers[f"encoder_{i}"] = dense
                        current_layer = dense(current_layer)
                        self._called_encoder_layers[f"encoder_{i}"] = current_layer
                        add_layer_summaries(dense, debug=self.debug)
                    return current_layer

            def _generate(self, inputs):
                with tf.name_scope("generator"):
                    current_layer = inputs
                    if self.p.periodicity < float("inf"):
                        n_neurons_with_inputs = [
                            self.main_inputs.shape[1] * 2
                        ] + self.p.n_neurons
                    else:
                        n_neurons_with_inputs = [
                            self.main_inputs.shape[1]
                        ] + self.p.n_neurons
                    for i, (n_neurons, act_fun) in enumerate(
                        zip(
                            n_neurons_with_inputs[-2::-1],
                            self.p.activation_functions[-2::-1],
                        )
                    ):
                        if act_fun:
                            act_fun = getattr(tf.nn, act_fun)
                        else:
                            act_fun = None
                        dense = tf.compat.v1.layers.Dense(
                            n_neurons,
                            activation=act_fun,
                            kernel_initializer=tf.keras.initializers.Constant(1),
                            kernel_regularizer=self.regularizer,
                            bias_initializer=tf.keras.initializers.Constant(1),
                        )
                        self._decoder_layers[f"decoder_{i}"] = dense
                        current_layer = dense(current_layer)
                        self._called_decoder_layers[f"decoder_{i}"] = current_layer
                    if self.p.periodicity < float("inf"):
                        split = self.main_inputs.shape[1]
                        current_layer = tf.atan2(
                            current_layer[:, :split], current_layer[:, split:]
                        )
                        if self.p.periodicity != 2 * np.pi:
                            current_layer = (
                                current_layer / (2 * np.pi) * self.p.periodicity
                            )
                    self._called_decoder_layers[f"decoder_{i}"] = current_layer
                    return current_layer

        # define the parameters
        # Encodermap imports
        from encodermap.encodermap_tf1.parameters import ADCParameters

        parameters = ADCParameters()

        # set all cost scales to 1
        # this follows the same order as the `_setup_cost()` method in `AngleDihedralCartesianEncoderMap`.
        parameters.dihedral_cost_scale = 1
        parameters.angle_cost_scale = 1
        parameters.cartesian_cost_scale = 1
        parameters.distance_cost_scale = 1
        parameters.cartesian_distance_cost_scale = 1
        parameters.center_cost_scale = 1
        parameters.l2_reg_constant = 0.001

        # Wondering, why ADCParameters implements an auto_cost_scale
        parameters.auto_cost_scale = 1

        # remaining parameters
        parameters.main_path = em.misc.run_path(str(output_dir / "runs"))
        total_steps = 500
        parameters.cartesian_cost_variant = "mean_abs"
        parameters.cartesian_cost_scale_soft_start = (
            int(total_steps / 10 * 9),
            int(total_steps / 10 * 9) + 250,
        )
        parameters.cartesian_pwd_start = 1
        parameters.cartesian_pwd_step = 3
        parameters.dihedral_cost_variant = "mean_abs"
        parameters.cartesian_dist_sig_parameters = [400, 10, 5, 1, 2, 5]
        parameters.checkpoint_step = max(1, int(total_steps / 10))
        parameters.id = "diUbq"

        # initialize and overwrite internal attributes
        e_map = CustomADCEncodermap(parameters, custom_moldata)
        # and check the inputs again
        tf1_inputs = e_map.sess.run(e_map.main_inputs)
        self.assertAllClosePeriodic(custom_moldata.dihedrals, tf1_inputs)
        print("Inputs of MolData and trajs are identical")

        # This dict will track the tf1 and per-hand output and
        # can be used to compare it to the tf2 code.
        track = {}

        # Combine my own and the tf1 code. This is gonna be yanky...
        d = custom_moldata.dihedrals
        track["my_input"] = d.copy()
        d = np.concatenate([np.sin(d), np.cos(d)], 1)
        track["my_unitcircle_input"] = d.copy()

        # iterate over the encoder
        for i, ((layer_name, layer), (n_neurons, act_fun)) in enumerate(
            zip(
                e_map._encoder_layers.items(),
                zip(parameters.n_neurons, parameters.activation_functions[1:]),
            )
        ):
            # initialize my weights
            my_w = np.ones((d.shape[1], n_neurons))
            my_b = np.ones((1, n_neurons))

            # get EncoderMap's weights
            tf1_w = e_map.sess.run(layer.weights[0])
            tf1_b = e_map.sess.run(layer.bias)

            # compare
            self.assertAllEqual(
                my_w,
                tf1_w,
                msg=(
                    f"At {i}: Although the `CustomADCEncodermap` class initializes the "
                    f"weights with the Constant(1) it did not match the provided array of ones."
                    f"Here are the shapes: {my_w.shape=}, {tf1_w.shape=}, {n_neurons=}"
                ),
            )
            self.assertAllEqual(
                my_b[0],
                tf1_b,
                msg=(
                    f"At {i}: Although the `CustomADCEncodermap` class initializes the "
                    f"bias with the Constant(1) it did not match the array of ones."
                ),
            )

            # call my layer
            if act_fun == "tanh":
                d = np.tanh(d @ my_w + my_b)
            elif act_fun == "":
                d = d @ my_w + my_b
            else:
                raise Exception(f"Unknown act fun")

            # call EncoderMap's layer
            tf1_out = e_map.sess.run(
                e_map._called_encoder_layers[layer_name],
                feed_dict={e_map.main_inputs: custom_moldata.dihedrals},
            )

            # compare
            self.assertAllEqual(
                d, tf1_out, msg=f"{i=}, {d[:2]=} {tf1_out[:2]=} {act_fun=}"
            )

            track[f"my_w_{i}"] = my_w.copy()
            track[f"my_b_{i}"] = my_b.copy()
            track[f"tf1_w_{i}"] = tf1_w.copy()
            track[f"tf1_b_{i}"] = tf1_b.copy()
            track[f"my_out_{i}"] = d.copy()
            track[f"tf1_out_{i}"] = tf1_out.copy()
            track[f"act_fun_{i}"] = act_fun
            print(
                f"Weights, biases and output values of the encoder layer {i} "
                f"are identical within rtol=1e-6 between manual and tf1."
            )

        # compare the output of the latent space
        self.assertAllEqual(d, e_map.encode(custom_moldata.dihedrals))

        n_neurons_with_inputs = [e_map.main_inputs.shape[1] * 2] + parameters.n_neurons
        for i, ((layer_name, layer), (n_neurons, act_fun)) in enumerate(
            zip(
                e_map._decoder_layers.items(),
                zip(
                    n_neurons_with_inputs[-2::-1],
                    parameters.activation_functions[-2::-1],
                ),
            )
        ):
            i += len(e_map._encoder_layers)
            # initialize my weights
            my_w = np.ones((d.shape[1], n_neurons))
            my_b = np.ones((1, n_neurons))

            # get EncoderMap's weights
            tf1_w = e_map.sess.run(layer.weights[0])
            tf1_b = e_map.sess.run(layer.bias)

            # compare
            self.assertAllEqual(
                my_w,
                tf1_w,
                msg=(
                    f"At {i}: Although the `CustomADCEncodermap` class initializes the "
                    f"weights with the Constant(1) it did not match the provided array of ones: "
                    f"{tf1_w.shape=} {my_w.shape=} {layer.units=} {n_neurons=}"
                ),
            )
            self.assertAllEqual(
                my_b[0],
                tf1_b,
                msg=(
                    f"At {i}: Although the `CustomADCEncodermap` class initializes the "
                    f"bias with the Constant(1) it did not match the array of ones: "
                    f"{tf1_b.shape=} {my_b.shape=} {layer.units=} {n_neurons=}"
                ),
            )

            # call my layer
            if act_fun == "tanh":
                d = np.tanh(d @ my_w + my_b)
            elif act_fun == "":
                d = d @ my_w + my_b
            else:
                raise Exception(f"Unknown act fun")

            if act_fun == "":
                d = np.arctan2(
                    d[:, : e_map.main_inputs.shape[1]],
                    d[:, e_map.main_inputs.shape[1] :],
                )

                # call EncoderMap's layer
            tf1_out = e_map.sess.run(
                e_map._called_decoder_layers[layer_name],
                feed_dict={e_map.main_inputs: custom_moldata.dihedrals},
            )

            # compare
            self.assertAllClose(
                d, tf1_out, msg=f"{i=}, {d[:2, :2]=} {tf1_out[:2, :2]=} {act_fun=}"
            )

            track[f"my_w_{i}"] = my_w.copy()
            track[f"my_b_{i}"] = my_b.copy()
            track[f"tf1_w_{i}"] = tf1_w.copy()
            track[f"tf1_b_{i}"] = tf1_b.copy()
            track[f"my_out_{i}"] = d.copy()
            track[f"tf1_out_{i}"] = tf1_out.copy()
            track[f"act_fun_{i}"] = act_fun
            print(
                f"Weights, biases and output values of the decoder layer {i} "
                f"are identical within rtol=1e-6 between manual and tf1."
            )

        costs_contributions = {}
        encoded_summaries = e_map.sess.run(
            e_map.merged_summaries,
            feed_dict={e_map.main_inputs: custom_moldata.dihedrals},
        )
        summary_proto = tf.compat.v1.Summary()
        summary_proto.ParseFromString(encoded_summaries)
        for val in summary_proto.value:
            if "cost" not in val.tag:
                continue
            costs_contributions[val.tag.replace("cost/", "")] = val.simple_value

        # test the center cost
        # it's just the mean of the square of the latent
        track["tf1_center_cost"] = costs_contributions["center_cost"]
        track["my_center_cost"] = np.mean(track["tf1_out_2"] ** 2)
        self.assertAllClose(track["tf1_center_cost"], track["my_center_cost"])
        print(f"The center_cost is identical within rtol=1e-6 between manual and tf1.")

        # test the reg_cost
        # the reg cost is not part of the costs_contributions,
        # as it was defined without a summary_scalar name of ""
        # and so, it does not appear in the summary protobuf
        # the l2 reg for every layer is calculated as np.sum(weights ** 2) / 2
        # the l2 reg for the model is calculated via lambda / 2 * np.sum(l2_reg_of_layers)
        # where lambda is the regularization parameter
        my_reg_cost_total = 0
        tf1_reg_cost_total = 0
        with e_map.graph.as_default():
            track["tf1_reg_cost_computed"] = e_map.sess.run(
                tf.compat.v1.losses.get_regularization_loss()
            )
            for v in tf.compat.v1.trainable_variables():
                if "kernel" in v.name:
                    tf1_reg_cost = e_map.sess.run(tf.nn.l2_loss(v))
                    my_reg_cost = np.sum(e_map.sess.run(v) ** 2) / 2
                    self.assertAllClose(my_reg_cost, tf1_reg_cost)
                    my_reg_cost_total += my_reg_cost
                    tf1_reg_cost_total += tf1_reg_cost
        track["my_reg_cost"] = my_reg_cost_total * parameters.l2_reg_constant * 2
        self.assertAllClose(track["tf1_reg_cost_computed"], track["my_reg_cost"])
        print(
            f"The regularization_cost is identical within rtol=1e-6 between manual and tf1."
        )

        # test the dihedral cost
        # the dihedral cost is calculated via the distance of inp vs output
        # if only dihedrals are trained. If angles are also trained, the input
        # and output needs to be splitted
        # the dihedrals lie in a periodic space of 2pi and such, we need to subtract
        # or add pi for the elements falling outside of this periodic space
        tf1_dihedral_cost = costs_contributions["dihedral_cost"]

        my_out = track["my_out_5"]
        with e_map.graph.as_default():
            tf1_in = e_map.sess.run(e_map.main_inputs)
            tf1_out = e_map.sess.run(e_map.generated_dihedrals)
        self.assertAllClose(my_out, tf1_out)
        self.assertAllClosePeriodic(custom_moldata.dihedrals, tf1_in)

        # it is not input vs output, but the generated...
        # is generated the raw output or does it get passed through the chain in plane?
        # it is just the output
        # Encodermap imports
        from encodermap.encodermap_tf1.misc import periodic_distance_np

        my_dihedral_cost = np.mean(
            np.abs(
                periodic_distance_np(
                    custom_moldata.dihedrals,
                    track[f"my_out_5"],
                    2 * np.pi,
                ),
            ),
        )
        self.assertAllClose(tf1_dihedral_cost, my_dihedral_cost)
        track["my_dihedral_cost"] = my_dihedral_cost
        print(
            f"The dihedral_cost is identical within rtol=1e-6 between manual and tf1."
        )

        # angle cost
        # the angle cost is just the same as the dihedral cost
        # if backbone angles are not used, the generated angles are the mean input angles
        tf1_angle_cost = costs_contributions["angle_cost"]
        tf1_generated_angles = e_map.sess.run(e_map.generated_angles)
        my_generated_angles = np.tile(
            np.expand_dims(
                np.mean(
                    custom_moldata.angles,
                    axis=0,
                ),
                axis=0,
            ),
            reps=(custom_moldata.dihedrals.shape[0], 1),
        )
        self.assertAllClose(tf1_generated_angles, my_generated_angles)
        my_angle_cost = np.mean(
            np.abs(
                periodic_distance_np(
                    custom_moldata.angles,
                    my_generated_angles,
                    2 * np.pi,
                )
            )
        )
        self.assertAllClose(tf1_angle_cost, my_angle_cost)
        track["my_angle_cost"] = my_angle_cost
        print(f"The angle_cost is identical within rtol=1e-6 between manual and tf1.")

        # cartesian cost
        # the cartesian cost uses the pairwise CA distances of the input cartesians
        # vs the generated cartesians
        # generated cartesians are created by using the chain_in_plane and apply_torsion method
        tf1_cartesian_cost = costs_contributions["cartesian_cost"]
        tf1_input_cartesians = e_map.sess.run(e_map.inputs[2])
        tf1_input_selected_cartesians = tf1_input_cartesians[
            :,
            parameters.cartesian_pwd_start : parameters.cartesian_pwd_stop : parameters.cartesian_pwd_step,
        ]
        arr = custom_moldata.central_cartesians[
            :,
            parameters.cartesian_pwd_start : parameters.cartesian_pwd_stop : parameters.cartesian_pwd_step,
        ]

        # test whether the input cartesian is still the same as moldata
        self.assertAllClose(tf1_input_cartesians, custom_moldata.central_cartesians)
        self.assertEqual(
            arr.shape,
            tf1_input_selected_cartesians.shape,
            msg=f"{arr.shape=}, {tf1_input_selected_cartesians.shape=}",
        )

        # compare the input pairwise dists
        tf1_input_cartesian_pairwise_dist = e_map.sess.run(
            e_map.input_cartesian_pairwise_dist
        )
        my_input_cartesians_pairwise_dist = np.empty(
            (parameters.batch_size, int(scipy.special.binom(arr.shape[1], 2)))
        )
        self.assertEqual(
            tf1_input_cartesian_pairwise_dist.shape,
            my_input_cartesians_pairwise_dist.shape,
        )
        for i, a in enumerate(arr):
            my_input_cartesians_pairwise_dist[i] = scipy.spatial.distance.pdist(a)

        # mean lengths can't be compared, as it is not an instance variable in `ADCEncoderMap`.
        # i.e. it's missing the `self.` and thus can't be access from outside.
        my_mean_lengths = np.expand_dims(
            np.mean(
                custom_moldata.lengths,
                0,
            ),
            0,
        )

        # define some helper functions
        def coordinates_from_dists_and_angles(
            dists: np.ndarray, angles: np.ndarray
        ) -> np.ndarray:
            """Similar to `chain_in_plane`, but in numpy.

            Args:
                dists (np.ndarray): The distances with the shape (n_batches, n_dists).
                    If n_batches is 1, the dists will be stacked to reach the same
                    n_batches as `angles`.
                angles (np.ndarray): The angles with the shape (n_batches, n_angles).
                    The n_angles must be one less, than n_dists.

            Returns:
                np.ndarray: The new 2D coordinates with shape (n_batches, n_dists + 1, 2).

            """
            if dists.shape[0] == 1:
                dists = np.repeat(dists, repeats=angles.shape[0], axis=0)
            self.assertEqual(dists.shape[1], angles.shape[1] + 1)

            # fill this array
            new_points = np.zeros((dists.shape[0], dists.shape[1] + 1, 3))

            prev_angle = np.zeros(angles.shape[0])
            for i, (d, a) in enumerate(zip(dists.T, angles.T)):
                idx = i + 1
                new_points[:, idx, 0] = new_points[:, idx - 1, 0] + d * np.cos(
                    prev_angle
                )
                new_points[:, idx, 1] = new_points[:, idx - 1, 1] + d * np.sin(
                    prev_angle
                ) * (1 if i % 2 == 0 else -1)
                prev_angle = np.pi - a - prev_angle
            else:
                new_points[:, idx + 1, 0] = new_points[:, idx, 0] + dists[
                    :, i + 1
                ] * np.cos(prev_angle)
                new_points[:, idx + 1, 1] = new_points[:, idx, 1] + dists[
                    :, i + 1
                ] * np.sin(prev_angle) * (-1 if i % 2 == 0 else 1)

            return new_points

        def apply_rotations(
            indices: np.ndarray,
            torsion_indices: np.ndarray,
            torsions: np.ndarray,
            coords: np.ndarray,
            dihes: np.ndarray,
            location: Literal["left", "right"],
            comparison_coords: np.ndarray,
        ) -> None:
            """Applies rotations starting from the middle of a protein backbone.
            Similar to `dihedral_to_cartesian_tf_one_way`.

            indices (np.ndarray): An array of shape (n_torsions // 2, ). This array can
                be used to get the correct torsions from the torsions_set array.
            dih_indices (np.ndarray): An array of shape (n_torsions // 2
            torsion_indices (np.ndarray): An array of shape (n_torsions // 2, 2). This
                array indexes the atoms in coords.
            torsions (np.ndarray): An array of shape (n_batches, n_points - 3).
            coords (np.ndarray): An array of shape (n_batches, n_points, 3).
            dihes (np.ndarray): An array of the shape (n_batches, n_points - 3). This
                array will hold the new dihedrals
            location (Literal["left", "right"]): Whether we are walking from the center to
                the end of the chain ("right"), or from the center to the left of the chain
                ("left"). Based on that the indexing of the atoms who rotate needs to change.
            comparison_coords (np.ndarray): known coordinates to compare against.

            """
            self.assertEqual(len(indices), len(torsion_indices))
            for iter_, (idx, (i, j)) in enumerate(zip(indices, torsion_indices)):
                if location == "right":
                    side_index = np.arange(j, coords.shape[1])
                else:
                    side_index = np.arange(0, j)[::-1]
                for frame in range(coords.shape[0]):
                    if location == "right":
                        dih_index = np.array([i - 1, i, j, j + 1])
                        current_dihedral = deepcopy(
                            dihedral_test(coords[frame][dih_index])
                        )
                    else:
                        dih_index = np.array([j + 1, j, i, i - 1])
                        current_dihedral = deepcopy(
                            dihedral_test(coords[frame][dih_index])
                        )
                    angle = deepcopy(current_dihedral - torsions[frame, idx])
                    if location == "right":
                        pivot_point = coords[frame, i]
                        direction = coords[frame, j] - coords[frame, i]
                    else:
                        pivot_point = coords[frame, j]
                        direction = coords[frame, i] - coords[frame, j]
                    rotation_matrix = transformations.rotation_matrix(
                        angle, direction, pivot_point
                    )
                    far_side = coords[frame, side_index]
                    padded = np.pad(
                        far_side,
                        ((0, 0), (0, 1)),
                        mode="constant",
                        constant_values=1,
                    )
                    coords[frame, side_index] = rotation_matrix.dot(padded.T).T[:, :3]
                    test1 = dihedral_test(coords[frame][dih_index])
                    test2 = dihedral_test(comparison_coords[frame][dih_index])
                    self.assertAllClose(
                        comparison_coords[frame, dih_index],
                        coords[frame, dih_index],
                        atol=0.3,
                        msg=(
                            f"The {location=}, {frame=}, {iter_=}, with index {dih_index=} "
                            f"was not right: {coords[frame, dih_index]=} and "
                            f"{comparison_coords[frame, dih_index]=}. Trying to rotate {angle} to get "
                            f"the current angle of {current_dihedral=} to {torsions[frame, idx]=}. The "
                            f"dihedral for the coords is {test1=} and for the comparison coords {test2=}"
                        ),
                    )
                    test3 = torsions[frame, idx]
                    dihes[frame, idx] = test1
                    self.assertAllClose(
                        test1,
                        test2,
                        atol=0.3,
                        msg=(
                            f"At {frame=}, {iter_=}, {i=}, {j=}. Discrepancy between requested: "
                            f"{test3=} and "
                            f"set: {test1=} dihedral. Current dihedral is {current_dihedral=}. "
                            f"Tried to apply a rotation of {angle=}. {side_index[0]=}, {side_index[-1]=}"
                        ),
                    )

        def coordinates_from_torsions(
            coords: np.ndarray,
            torsions: np.ndarray,
            comparison_coords: np.ndarray,
        ) -> np.ndarray:
            """Similar to `dihedrals_to_cartesian_tf`, but in numpy.

            Args:
                coords (np.ndarray): The coords with shape (n_batches, n_points, 3). Here,
                    the z coordinate should be zero for all points over all batches.
                torsions (np.ndarray): The torsions to apply in the shape (n_batches, n_points - 3).
                comparison_coords (np.ndarray): known coordinates to compare against.

            Returns:
                np.ndarray: The new coords.

            """
            # negate the torsions to match `dihedral_to_cartesian_tf_one_way`
            torsions = -torsions.copy()
            self.assertEqual(len(torsions), len(coords))
            self.assertEqual(torsions.shape[1], coords.shape[1] - 3)
            self.assertTrue(np.all(coords[:, :, 2] == 0))
            n_atoms = coords.shape[1]
            dihes_out = np.empty(torsions.shape)
            coords_out = coords.copy()

            # split the torsion indices
            indices = np.dstack(
                [
                    np.arange(1, n_atoms - 2),
                    np.arange(2, n_atoms - 1),
                ]
            )[0]
            dih_indices = np.dstack(
                [
                    np.arange(0, n_atoms - 3),
                    np.arange(1, n_atoms - 2),
                    np.arange(2, n_atoms - 1),
                    np.arange(3, n_atoms - 0),
                ]
            )[0]

            split = coords.shape[1] // 2
            idx_right = np.arange(split - 1, indices.shape[0])
            idx_left = np.arange(0, split - 1)[::-1]
            self.assertEqual(len(idx_left) + len(idx_right), len(indices))
            indices_right = indices[split - 1 :]
            test = indices[idx_right]
            self.assertAllEqual(indices_right, test)
            indices_left = indices[split - 2 :: -1]
            test = indices[idx_left]
            self.assertAllEqual(indices_left, test)
            self.assertEqual(
                indices_left.shape[0] + indices_right.shape[0],
                indices.shape[0],
                msg=f"{indices.shape=}, {indices_left.shape=}, {indices_right.shape=}",
            )
            self.assertAllEqual(indices[split - 1], indices_right[0])
            self.assertAllEqual(indices[split - 2], indices_left[0])
            test = np.vstack([indices_left[::-1], indices_right])
            self.assertAllEqual(test, indices)

            # start with the right side
            apply_rotations(
                idx_right,
                indices_right,
                torsions,
                coords_out,
                dihes_out,
                "right",
                comparison_coords=comparison_coords,
            )
            apply_rotations(
                idx_left,
                indices_left,
                torsions,
                coords_out,
                dihes_out,
                "left",
                comparison_coords=comparison_coords,
            )
            # negate the dihedrals to match `dihedral_to_cartesian_tf_one_way`
            return coords_out, -dihes_out

        def dihedral_test(points: np.ndarray) -> float:
            """Computes the dihedral between 4 points.

            Args:
                points (np.ndarray): NumPy array of shape (3, 3).

            Returns:
                float: The current dihedral.

            """
            self.assertEqual(points.shape, (4, 3))
            b1 = points[0] - points[1]
            b2 = points[1] - points[2]
            b3 = points[2] - points[3]
            c1 = np.cross(b2, b3)
            c2 = np.cross(b1, b2)
            p1 = (b1 * c1).sum(-1)
            p1 *= (b2 * b2).sum(-1) ** 0.5
            p2 = (c1 * c2).sum(-1)
            dih = np.arctan2(p1, p2, None)
            return dih

        # test the chain in plane
        my_chain_in_plane = coordinates_from_dists_and_angles(
            my_mean_lengths, my_generated_angles
        )
        tf1_chain_in_plane = e_map.sess.run(e_map.chain_in_plane)
        self.assertTrue(np.all(tf1_chain_in_plane[:, :, 2] == 0))
        self.assertAllClose(tf1_chain_in_plane, my_chain_in_plane, atol=1e-3)

        # test the chain in plane by providing my_mean_lengths and my_generated_angles
        # Encodermap imports
        from encodermap.encodermap_tf1.backmapping import chain_in_plane

        tf1_chain_in_plane_with_my_data = chain_in_plane(
            my_mean_lengths, my_generated_angles
        ).numpy()
        self.assertAllEqual(tf1_chain_in_plane, tf1_chain_in_plane_with_my_data)

        # test the dihedrals to cartesians
        tf1_cartesians = e_map.sess.run(e_map.cartesian)
        my_cartesian, dihedrals_from_backmapping = coordinates_from_torsions(
            my_chain_in_plane,
            track["my_out_5"],
            comparison_coords=tf1_cartesians,
        )

        # firs test the shapes
        self.assertEqual(tf1_cartesians.shape, my_chain_in_plane.shape)
        self.assertAllClosePeriodic(track["my_out_5"], tf1_out)
        self.assertEqual(my_cartesian.shape, tf1_cartesians.shape)

        # this test is important. The output of the backmapping should match the output of the
        # encoder, because these dihedrals were applied to get the cartesians
        self.assertAllClosePeriodic(tf1_out, dihedrals_from_backmapping)

        # iterate over my_cartesian and dihedrals_from_backmapping to find possible mistakes
        n_atoms = custom_moldata.central_cartesians.shape[1]
        indices = np.dstack(
            [
                np.arange(0, n_atoms - 3),
                np.arange(1, n_atoms - 2),
                np.arange(2, n_atoms - 1),
                np.arange(3, n_atoms - 0),
            ]
        )[0]
        self.assertEqual(indices.shape, (n_atoms - 3, 4))
        self.assertEqual(
            dihedrals_from_backmapping.shape, (parameters.batch_size, n_atoms - 3)
        )
        self.assertEqual(my_cartesian.shape, (parameters.batch_size, n_atoms, 3))
        # because the weights are constant ones the cartesians for all
        # samples should be the same
        for i in range(1, parameters.batch_size):
            self.assertAllEqual(my_cartesian[0], my_cartesian[i])

        for i, (ind, d1) in enumerate(zip(indices, dihedrals_from_backmapping[0])):
            d2 = -dihedral_test(my_cartesian[0, ind])
            self.assertAllClosePeriodic(
                d1, d2, err_msg=f"At {i=}, {ind=}, {d1=}, {d2=}"
            )

        # compare the center of the chain, where the z-coordinate is still 0.
        bond_indices = np.dstack(
            [
                np.arange(1, n_atoms - 2),
                np.arange(2, n_atoms - 1),
            ]
        )[0]

        split = my_chain_in_plane.shape[1] // 2
        idx_right = np.arange(split - 1, indices.shape[0])
        idx_left = np.arange(0, split - 1)[::-1]

        # the central positions are left in the z-axis
        my_central_left = my_cartesian[0, idx_left[0]]
        tf1_central_left = tf1_cartesians[0, idx_left[0]]
        my_central_right = my_cartesian[0, idx_right[0]]
        tf1_central_right = tf1_cartesians[0, idx_right[0]]
        self.assertAllClose(
            tf1_central_left,
            my_central_left,
            rtol=1e-5,
            msg=f"{tf1_central_left=}, {my_central_left=}",
        )
        self.assertAllClose(
            tf1_central_right,
            my_central_right,
            rtol=1e-5,
            msg=f"{tf1_central_right=}, {my_central_right=}",
        )

        # finally compare all cartesians
        self.assertAllClose(my_cartesian, tf1_cartesians, rtol=1e-1)
        print(
            f"The cartesian output is identical within rtol=1e-1 between manual and tf1."
        )

        # calculate the dihedrals of tf1_cartesian using proper displacements
        # firs the indices
        ix10 = indices[:, [0, 1]]
        self.assertEqual(ix10.shape, (indices.shape[0], 2))
        ix21 = indices[:, [1, 2]]
        self.assertEqual(ix21.shape, (indices.shape[0], 2))
        ix32 = indices[:, [2, 3]]
        self.assertEqual(ix32.shape, (indices.shape[0], 2))

        # then the displacements
        tf1_b1_displacements = np.diff(tf1_cartesians[:, ix10], axis=2)[:, :, 0]
        self.assertEqual(
            tf1_b1_displacements.shape, (len(tf1_cartesians), indices.shape[0], 3)
        )
        tf1_b2_displacements = np.diff(tf1_cartesians[:, ix21], axis=2)[:, :, 0]
        self.assertEqual(
            tf1_b2_displacements.shape, (len(tf1_cartesians), indices.shape[0], 3)
        )
        tf1_b3_displacements = np.diff(tf1_cartesians[:, ix32], axis=2)[:, :, 0]
        self.assertEqual(
            tf1_b3_displacements.shape, (len(tf1_cartesians), indices.shape[0], 3)
        )
        tf1_c1 = np.cross(tf1_b2_displacements, tf1_b3_displacements)
        tf1_c2 = np.cross(tf1_b1_displacements, tf1_b2_displacements)
        tf1_p1 = (tf1_b1_displacements * tf1_c1).sum(-1)
        tf1_p1 *= (tf1_b2_displacements * tf1_b2_displacements).sum(-1) ** 0.5
        tf1_p2 = (tf1_c1 * tf1_c2).sum(-1)
        tf1_computed_dih = np.arctan2(tf1_p1, tf1_p2, None)
        self.assertAllClosePeriodic(tf1_out, tf1_computed_dih, atol=1e-3)

        # calculate my dihedrals using proper displacements
        my_b1_displacements = np.diff(my_cartesian[:, ix10], axis=2)[:, :, 0]
        self.assertAllClose(tf1_b1_displacements, my_b1_displacements, atol=1e-2)
        my_b2_displacements = np.diff(my_cartesian[:, ix21], axis=2)[:, :, 0]
        self.assertAllClose(tf1_b2_displacements, my_b2_displacements, atol=1e-2)
        my_b3_displacements = np.diff(my_cartesian[:, ix32], axis=2)[:, :, 0]
        self.assertAllClose(tf1_b3_displacements, my_b3_displacements, atol=1e-2)
        my_c1 = np.cross(my_b2_displacements, my_b3_displacements)
        my_c2 = np.cross(my_b1_displacements, my_b2_displacements)
        my_p1 = (my_b1_displacements * my_c1).sum(-1)
        my_p1 *= (my_b2_displacements * my_b2_displacements).sum(-1) ** 0.5
        my_p2 = (my_c1 * my_c2).sum(-1)
        my_computed_dih = np.arctan2(my_p1, my_p2, None)
        self.assertAllClosePeriodic(
            tf1_computed_dih, dihedrals_from_backmapping, atol=1e-3
        )

        # why are my_computed_dih and the dihedrals_from_backmapping different?
        self.assertAllClosePeriodic(
            my_computed_dih, dihedrals_from_backmapping, atol=1e-3
        )

        tf1_output_cartesians_pairwise_dist = e_map.sess.run(
            e_map.gen_cartesian_pairwise_dist
        )
        my_output_cartesians_pairwise_dist = np.empty(
            (parameters.batch_size, int(scipy.special.binom(arr.shape[1], 2)))
        )
        arr = my_cartesian[
            :,
            parameters.cartesian_pwd_start : parameters.cartesian_pwd_stop : parameters.cartesian_pwd_step,
        ]
        self.assertEqual(
            my_output_cartesians_pairwise_dist.shape,
            tf1_output_cartesians_pairwise_dist.shape,
            msg=(
                f"{arr.shape=}, {my_output_cartesians_pairwise_dist.shape=}, "
                f"{tf1_output_cartesians_pairwise_dist.shape=}"
            ),
        )
        for i, a in enumerate(arr):
            my_output_cartesians_pairwise_dist[i] = scipy.spatial.distance.pdist(a)
        self.assertAllClose(
            tf1_output_cartesians_pairwise_dist,
            my_output_cartesians_pairwise_dist,
            rtol=1e-2,
        )

        # compute the cost
        my_cartesian_cost = np.mean(
            np.abs(
                my_input_cartesians_pairwise_dist - my_output_cartesians_pairwise_dist
            )
        )
        self.assertAllClose(my_cartesian_cost, tf1_cartesian_cost, rtol=1e-2)
        track["my_cartesian_cost"] = my_cartesian_cost
        print(
            f"The cartesian_cost is identical within rtol=1e-2 between manual and tf1."
        )

        # Test the cartesian_distance_cost
        def my_sigmoid(x, sig, a, b):
            return 1 - (1 + (2 ** (a / b) - 1) * (x / sig) ** a) ** (-b / a)

        tf1_cartesian_distance_cost = costs_contributions["cartesian_distance_cost"]
        tf1_latent = e_map.sess.run(e_map.latent)
        my_latent = track["my_out_2"]
        self.assertAllEqual(tf1_latent, my_latent)
        my_lowd = my_sigmoid(
            scipy.spatial.distance.cdist(
                my_latent,
                my_latent,
            ),
            *parameters.cartesian_dist_sig_parameters[3:],
        )
        my_highd = my_sigmoid(
            scipy.spatial.distance.cdist(
                my_input_cartesians_pairwise_dist,
                my_input_cartesians_pairwise_dist,
            ),
            *parameters.cartesian_dist_sig_parameters[:3],
        )
        my_cartesian_distance_cost = np.mean((my_highd - my_lowd) ** 2)
        self.assertAllClose(
            my_cartesian_distance_cost, tf1_cartesian_distance_cost, rtol=1e-3
        )
        track["my_cartesian_distance_cost"] = my_cartesian_distance_cost
        print(
            f"The cartesian_distance_cost is identical within rtol=1e-6 between manual and tf1."
        )

        # Test the distance cost
        tf1_distance_cost = costs_contributions["distance_cost"]
        my_lowd = my_sigmoid(
            scipy.spatial.distance.cdist(
                my_latent,
                my_latent,
            ),
            *parameters.dist_sig_parameters[3:],
        )
        my_highd = my_sigmoid(
            scipy.spatial.distance.cdist(
                track["my_input"],
                track["my_input"],
            ),
            *parameters.dist_sig_parameters[:3],
        )
        my_distance_cost = np.mean((my_highd - my_lowd) ** 2)
        track["my_distance_cost"] = my_distance_cost
        self.assertAllClose(
            my_distance_cost,
            tf1_distance_cost,
            atol=0.1,
            msg=f"{my_distance_cost=}, {tf1_distance_cost=}",
        )
        print(
            f"The distance_cost is identical within atol=1e-1 between manual and tf1."
        )

        # get the gradients
        tf1_gradients = {}
        with e_map.graph.as_default():
            for k, v in zip(
                tf.compat.v1.trainable_variables(), e_map.sess.run(e_map.gradients)
            ):
                tf1_gradients[k.name + "gradient"] = v

        # do all of this with tf2
        # Encodermap imports
        from encodermap.models.models import gen_functional_model

        trajs = em.load(
            output_dir / "01.xtc",
            output_dir / "01.pdb",
        )[: parameters.batch_size]._gen_ensemble()
        trajs.load_CVs("all")

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                trajs.central_angles,
                trajs.central_dihedrals,
                trajs.central_cartesians,
                trajs.central_distances,
                trajs.side_dihedrals,
            )
        )
        dataset = dataset.repeat()
        dataset = dataset.batch(parameters.batch_size)

        for ca, cdih, cc, cdist, sd in dataset:
            break

        self.assertAllEqual(ca.numpy(), trajs.central_angles)
        self.assertAllEqual(cdih.numpy(), trajs.central_dihedrals)
        self.assertAllEqual(cc.numpy(), trajs.central_cartesians)
        self.assertAllEqual(cdist.numpy(), trajs.central_distances)
        self.assertAllEqual(sd.numpy(), trajs.side_dihedrals)

        parameters.use_sidechains = False
        parameters.tensorboard = False
        parameters.write_summary = False
        parameters.multimer_training = None
        parameters.multimer_topology_classes = None
        parameters.multimer_connection_bridges = None
        parameters.multimer_lengths = None
        parameters.reconstruct_sidechains = False

        input_shapes = (
            trajs.central_angles.shape[1:],
            trajs.central_dihedrals.shape[1:],
            trajs.central_cartesians.shape[1:],
            trajs.central_distances.shape[1:],
            trajs.side_dihedrals.shape[1:],
        )

        model = gen_functional_model(
            input_shapes=input_shapes,
            parameters=parameters,
            bias_initializer="ones",
            kernel_initializer="ones",
        )

        self.assertLen(
            model.input_shape, 4, msg=f"The model {model=} must take three inputs"
        )

        # Accumulate the neurons and activation function data
        neurons = []
        activations_functions = []
        for i, (n_neurons, act_fun) in enumerate(
            zip(parameters.n_neurons, parameters.activation_functions[1:])
        ):
            neurons.append(n_neurons)
            activations_functions.append(act_fun)

        if parameters.periodicity < float("inf"):
            n_neurons_with_inputs = [
                custom_moldata.dihedrals.shape[1] * 2
            ] + parameters.n_neurons
        else:
            n_neurons_with_inputs = [
                custom_moldata.dihedrals.shape[1]
            ] + parameters.n_neurons
        for n_neurons, act_fun in zip(
            n_neurons_with_inputs[-2::-1], parameters.activation_functions[-2::-1]
        ):
            neurons.append(n_neurons)
            activations_functions.append(act_fun)

        layers = []
        i = 0
        for layer in model.encoder_model.layers + model.decoder_model.layers:
            if len(layer.weights) >= 1:
                layers.append(layer)
                self.assertEqual(
                    layer.units, neurons[i], msg=f"{layer.units=}, {neurons[i]=}"
                )
                ac = layer.activation.__name__
                if ac == "linear":
                    self.assertEqual(activations_functions[i], "")
                else:
                    self.assertEqual(ac, activations_functions[i])
                i += 1

        self.assertEqual(len(neurons), len(activations_functions))
        self.assertEqual(len(neurons), len(layers))

        d = custom_moldata.dihedrals.copy()
        self.assertEqual(model.encoder_model.input_shape[1], d.shape[1])
        self.assertEqual(d.shape[0], parameters.batch_size)
        track["tf2_input"] = d.copy()
        d = np.concatenate([np.sin(d), np.cos(d)], 1)
        track["tf2_unitcircle_input"] = d.copy()

        for i, (layer, n_neurons, act_fun) in enumerate(
            zip(layers, neurons, activations_functions)
        ):
            w, b = layer.weights
            w = w.numpy()
            b = b.numpy()

            tf2_test_w = np.ones((d.shape[1], n_neurons))
            tf2_test_b = np.ones((1, n_neurons))

            self.assertAllEqual(tf2_test_w, w)
            self.assertAllEqual(track[f"my_w_{i}"], w)
            self.assertAllEqual(tf2_test_b[0], b)
            self.assertAllEqual(track[f"my_b_{i}"][0], b)

            # call the layer
            tf2_out = layer(d).numpy()

            # call my layer
            if act_fun == "tanh":
                d = np.tanh(d @ tf2_test_w + tf2_test_b)
            elif act_fun == "":
                d = d @ tf2_test_w + tf2_test_b
            else:
                raise Exception(f"Unknown act fun")

            if i == 5:
                tf2_out = np.arctan2(
                    tf2_out[:, : e_map.main_inputs.shape[1]],
                    tf2_out[:, e_map.main_inputs.shape[1] :],
                )
                d = np.arctan2(
                    d[:, : e_map.main_inputs.shape[1]],
                    d[:, e_map.main_inputs.shape[1] :],
                )

            self.assertAllClose(
                tf2_out,
                d,
                msg=(
                    f"Test1 fail: {i=}, {tf2_out.shape=}, {d.shape=}, "
                    f"{act_fun=}, {layer.activation=}"
                ),
            )
            self.assertAllClose(
                track[f"my_out_{i}"],
                tf2_out,
                msg=(
                    f"Test2 fail: {i=}, {track[f'act_fun_{i}']=} "
                    f"{track[f'my_out_{i}'].shape=}, {tf2_out.shape=}"
                ),
            )

            track[f"tf2_out_{i}"] = tf2_out.copy
            track[f"tf2_w_{i}"] = w.copy()
            track[f"tf2_b_{i}"] = b.copy()
            print(
                f"Weights, biases and outputs of layer {i} are identical "
                f"within rtol=1e-6 between tf2, manual, and tf1."
            )

        # call the model to calculate cost functions
        (
            tf2_output_angles,
            tf2_output_dihedrals,
            tf2_output_cartesians,
            tf2_input_cartesians_pairwise,
            tf2_output_cartesians_pairwise,
        ) = model(
            (
                custom_moldata.angles.copy(),
                custom_moldata.dihedrals.copy(),
                custom_moldata.central_cartesians.copy(),
                custom_moldata.lengths.copy(),
            )
        )
        tf2_latent = model.encoder_model(custom_moldata.dihedrals.copy())

        # compare the cost functions
        # center
        # Encodermap imports
        from encodermap.loss_functions import center_loss

        tf2_center_cost = center_loss(model, parameters)(
            custom_moldata.dihedrals.copy()
        ).numpy()
        self.assertAllClose(tf2_center_cost, track["my_center_cost"])
        print(
            f"The center_cost is identical within rtol=1e-6 between tf2, manual, and tf1."
        )

        # reg
        # Encodermap imports
        from encodermap.loss_functions import regularization_loss

        tf2_reg_cost = regularization_loss(model, parameters)().numpy()
        self.assertAllClose(tf2_reg_cost, track["my_reg_cost"])
        print(
            f"The regularization_cost is identical within rtol=1e-6 between tf2, manual, and tf1."
        )

        # dihedral
        # Encodermap imports
        from encodermap.loss_functions import dihedral_loss

        tf2_dihedral_loss = dihedral_loss(model, parameters)(
            custom_moldata.dihedrals.copy(), tf2_output_dihedrals
        )
        self.assertAllClose(tf2_dihedral_loss, track["my_dihedral_cost"])
        print(
            f"The dihedral_cost is identical within rtol=1e-6 between tf2, manual, and tf1."
        )

        # angle
        # Encodermap imports
        from encodermap.loss_functions import angle_loss

        tf2_angle_loss = angle_loss(model, parameters)(
            custom_moldata.angles.copy(), tf2_output_angles
        )
        self.assertAllClose(tf2_angle_loss, track["my_angle_cost"])
        print(
            f"The angle_cost is identical within rtol=1e-6 between tf2, manual, and tf1."
        )

        # cartesian
        # Encodermap imports
        from encodermap.loss_functions import cartesian_loss

        tf2_cartesian_loss = cartesian_loss(model, None, parameters)(
            tf2_input_cartesians_pairwise, tf2_output_cartesians_pairwise
        )
        self.assertAllClose(tf2_cartesian_loss, track["my_cartesian_cost"], rtol=5e-4)
        print(
            f"The cartesian_cost is identical within rtol=5e-4 between tf2, manual, and tf1."
        )

        # cartesian distance
        # Encodermap imports
        from encodermap.loss_functions import cartesian_distance_loss

        tf2_cartesian_distance_loss = cartesian_distance_loss(model, parameters)(
            tf2_input_cartesians_pairwise, tf2_latent
        )
        self.assertAllClose(
            tf2_cartesian_distance_loss, track["my_cartesian_distance_cost"], rtol=1e-3
        )
        print(
            f"The cartesian_distance_cost is identical within rtol=1e-6 between tf2, manual, and tf1."
        )

        # distance
        # Encodermap imports
        from encodermap.loss_functions import distance_loss

        tf2_distance_loss = distance_loss(model, parameters)(
            custom_moldata.dihedrals.copy(), tf2_latent
        )
        self.assertAllClose(tf2_distance_loss, tf1_distance_cost)
        print(
            f"The cartesian_cost is identical within rtol=1e-6 between tf2, manual, and tf1."
        )

    @patch(
        "encodermap.models.models.ADCFunctionalModel.get_loss", custom_get_loss_dense
    )
    @patch(
        "encodermap.models.models.ADCSparseFunctionalModel.get_loss",
        custom_get_loss_sparse,
    )
    def examine_dense_and_sparse_cartesians(self):
        with self.assertRaisesRegex(
            ValueError,
            expected_regex=r".*gradients provided.*",
            msg=(
                f"This call is expected to fail with a ValueError, because the "
                f"method get_loss in ADCFunctionalModel was patched, giving a loss "
                f"of 0 and thus no "
                f"gradients can be computed. However, this method is meant to "
                f"check the difference between sparse training with sparse central_"
                f"cartesians and dense central_cartesians."
            ),
        ):
            self.ADC_autoencoder_with_dataset(
                sparse=False,
                use_sidechains=True,
                use_backbone_angles=True,
                trainable_sparse_to_dense=False,
                sidechain_only_sparse=False,
            )
        with self.assertRaisesRegex(
            ValueError,
            expected_regex=r".*gradients provided.*",
            msg=(
                f"This call is expected to fail with a ValueError, because the "
                f"method get_loss in ADCFunctionalModel was patched, giving a loss "
                f"of 0 and thus no "
                f"gradients can be computed. However, this method is meant to "
                f"check the difference between sparse training with sparse central_"
                f"cartesians and dense central_cartesians."
            ),
        ):
            self.ADC_autoencoder_with_dataset(
                sparse=True,
                use_sidechains=True,
                use_backbone_angles=True,
                trainable_sparse_to_dense=False,
                sidechain_only_sparse=False,
            )

    @patch(
        "encodermap.trajinfo.info_single.SingleTraj", MockSingleTraj, spec=em.SingleTraj
    )
    @patch(
        "encodermap.trajinfo.info_all.TrajEnsemble",
        MockTrajEnsemble,
        spec=em.TrajEnsemble,
    )
    def test_sparse_to_dense_produces_not_zeros_and_trains(self):
        emap, dataset, parameters = self.ADC_autoencoder_with_dataset(
            sparse=True,
            use_sidechains=True,
            use_backbone_angles=True,
            trainable_sparse_to_dense=False,
            sidechain_only_sparse=False,
            return_dataset_and_model=True,
        )

        for batch in dataset:
            self.assertLen(batch, 5)
            dihedrals = batch[1]
            cartesians = batch[2]
            side_dihedrals = batch[4]
            break

        dense_dihedrals_before = (
            emap.model.get_dense_model_central_dihedrals(dihedrals).numpy().copy()
        )
        dense_cartesians_before = (
            emap.model.get_dense_model_cartesians(cartesians).numpy().copy()
        )
        dense_side_dihedrals_before = (
            emap.model.get_dense_model_side_dihedrals(side_dihedrals).numpy().copy()
        )

        self.assertNotIn(
            0.0,
            dense_cartesians_before,
        )
        self.assertNotIn(
            0.0,
            dense_cartesians_before,
        )
        self.assertNotIn(
            0.0,
            dense_side_dihedrals_before,
        )

        emap.train()

        dense_dihedrals_after = (
            emap.model.get_dense_model_central_dihedrals(dihedrals).numpy().copy()
        )
        dense_cartesians_after = (
            emap.model.get_dense_model_cartesians(cartesians).numpy().copy()
        )
        dense_side_dihedrals_after = (
            emap.model.get_dense_model_side_dihedrals(side_dihedrals).numpy().copy()
        )

        self.assertAllClose(
            dense_dihedrals_before,
            dense_dihedrals_after,
        )
        self.assertAllClose(
            dense_cartesians_before,
            dense_cartesians_after,
        )
        self.assertAllClose(
            dense_side_dihedrals_before,
            dense_side_dihedrals_after,
        )

        emap, dataset, parameters = self.ADC_autoencoder_with_dataset(
            sparse=True,
            use_sidechains=True,
            use_backbone_angles=True,
            trainable_sparse_to_dense=True,
            sidechain_only_sparse=False,
            return_dataset_and_model=True,
        )

        for batch in dataset:
            self.assertLen(batch, 5)
            dihedrals = batch[1]
            cartesians = batch[2]
            side_dihedrals = batch[4]
            break

        dense_dihedrals_before = (
            emap.model.get_dense_model_central_dihedrals(dihedrals).numpy().copy()
        )
        dense_cartesians_before = (
            emap.model.get_dense_model_cartesians(cartesians).numpy().copy()
        )
        dense_side_dihedrals_before = (
            emap.model.get_dense_model_side_dihedrals(side_dihedrals).numpy().copy()
        )

        self.assertNotIn(
            0.0,
            dense_cartesians_before,
        )
        self.assertNotIn(
            0.0,
            dense_cartesians_before,
        )
        self.assertNotIn(
            0.0,
            dense_side_dihedrals_before,
        )

        emap.train()

        dense_dihedrals_after = (
            emap.model.get_dense_model_central_dihedrals(dihedrals).numpy().copy()
        )
        dense_cartesians_after = (
            emap.model.get_dense_model_cartesians(cartesians).numpy().copy()
        )
        dense_side_dihedrals_after = (
            emap.model.get_dense_model_side_dihedrals(side_dihedrals).numpy().copy()
        )

        self.assertNotAllClose(
            dense_dihedrals_before,
            dense_dihedrals_after,
        )
        self.assertNotAllClose(
            dense_cartesians_before,
            dense_cartesians_after,
        )
        self.assertNotAllClose(
            dense_side_dihedrals_before,
            dense_side_dihedrals_after,
        )

    def test_reload_model_without_data_raises_descriptive_error(self):
        """Reloading a model without the correct data should be possible.
        However, when the input_data is set and we're not using the hypercube,
        we should get a good warning.

        """
        train_data1 = np.random.random((1000, 20)).astype("float32")
        train_data2 = np.random.random((1000, 4)).astype("float32")
        for desc, train_data in zip(
            ["20-shaped", "4-shaped (same as hypercube)"], [train_data1, train_data2]
        ):
            with tempfile.TemporaryDirectory() as td:
                p = em.Parameters(n_steps=10, periodicity=float("inf"), main_path=td)
                emap = em.EncoderMap(train_data=train_data, parameters=p)
                self.assertFalse(emap.p.using_hypercube)
                emap.train()
                file = emap.save()
                parameter_file = Path(td) / "parameters.json"
                self.assertTrue(parameter_file.is_file())
                with open(parameter_file, "r") as f:
                    json_data = json.load(f)
                self.assertEqual(json_data["using_hypercube"], False)
                self.assertIsNotNone(file)
                loaded_emap = em.EncoderMap.from_checkpoint(file)
                self.assertFalse(
                    loaded_emap.p.using_hypercube,
                    msg=f"{loaded_emap.p.main_path=} {file=}",
                )
                loaded_emap.p.n_steps += 10
                with Capturing() as output:
                    out = loaded_emap.train()
                self.assertTrue(
                    any("was reloaded from disk" in l for l in output),
                    msg="The model does train with the hypercube data, which it shouldn't.",
                )
                self.assertIsNone(out)
                loaded_emap.set_train_data(train_data)
                self.assertEqual(loaded_emap.train_data.shape[1], train_data.shape[1])
                out = loaded_emap.train()
                self.assertIsNotNone(
                    out,
                    msg=(
                        f"The reloaded emap using {desc} did not train, even when "
                        f"the correct data was provided."
                    ),
                )

    @unittest.skip
    @patch(
        "encodermap.trajinfo.info_single.SingleTraj", MockSingleTraj, spec=em.SingleTraj
    )
    @patch(
        "encodermap.trajinfo.info_all.TrajEnsemble",
        MockTrajEnsemble,
        spec=em.TrajEnsemble,
    )
    def test_save_train_load(self):
        """Permutes all different parameters and checks whether the networks train,
        save, load, and retrain.

        Parameters: n
            * Encodermap class: `EncoderMap` or `AngleDihedralCartesianEncoderMap`
            * sparse: True or False
            * For `AngleDihedralCartesianEncoderMap`: Use sidechains True or False
            * For `AngleDihedralCartesianEncoderMap`: Use backbone angles True or False
            * For `AngleDihedralCartesianEncoderMap`: only sidechains are sparse True or False
            * For `EncoderMap` use periodicity float("inf") or 2 * np.pi

        """
        ########################################################################
        # Test how sparse cartesians are routed through the model
        ########################################################################
        self.examine_dense_and_sparse_cartesians()

        ########################################################################
        # Autoencoder, DihedralEncoderMap, EncoderMap
        ########################################################################
        self.autoencoder_with_dataset("Autoencoder", False, False, False)
        self.autoencoder_with_dataset("Autoencoder", True, False, False)
        self.autoencoder_with_dataset("DihedralEncoderMap", False, False, False)
        self.autoencoder_with_dataset("DihedralEncoderMap", True, False, False)
        self.autoencoder_with_dataset("EncoderMap", False, False, False, float("inf"))
        self.autoencoder_with_dataset("EncoderMap", True, False, False, float("inf"))
        self.autoencoder_with_dataset("EncoderMap", False, False, False, 2 * np.pi)
        self.autoencoder_with_dataset("EncoderMap", True, False, False, 2 * np.pi)
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap", False, True, True
        )

        # Wrong combination of use_sidechains and use_backbone_angles
        with self.assertRaisesRegex(
            Exception,
            expected_regex=r".*allowed combinations.*",
        ) as exc:
            self.autoencoder_with_dataset(
                "AngleDihedralCartesianEncoderMap",
                False,
                True,
                False,
            )

        ########################################################################
        # AngleDihedralCartesianEncoderMap
        ########################################################################
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            False,
            False,
            True,
        )
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            False,
            False,
            False,
        )
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            False,
            True,
            True,
            trainable_sparse_to_dense=True,
            sidechain_only_sparse=True,
        )
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            False,
            True,
            True,
            trainable_sparse_to_dense=False,
            sidechain_only_sparse=True,
        )
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            True,
            True,
            True,
            trainable_sparse_to_dense=True,
        )
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            True,
            False,
            True,
            trainable_sparse_to_dense=True,
        )
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            True,
            False,
            False,
            trainable_sparse_to_dense=True,
        )
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            True,
            True,
            True,
            trainable_sparse_to_dense=False,
        )
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            True,
            False,
            True,
            trainable_sparse_to_dense=False,
        )
        self.autoencoder_with_dataset(
            "AngleDihedralCartesianEncoderMap",
            True,
            False,
            False,
            trainable_sparse_to_dense=False,
        )

    def autoencoder_with_dataset(
        self,
        use_class: str,
        sparse: bool,
        use_sidechains: bool,
        use_backbone_angles: bool,
        periodicity: float = float("inf"),
        trainable_sparse_to_dense: bool = False,
        sidechain_only_sparse: bool = False,
    ) -> Any:
        print(
            f"Testing {use_class} with dataset. {sparse=} {use_sidechains=} "
            f"{use_backbone_angles=} {periodicity=} {trainable_sparse_to_dense=} "
            f"{sidechain_only_sparse=}"
        )
        if use_class == "AngleDihedralCartesianEncoderMap":
            self.ADC_autoencoder_with_dataset(
                sparse=sparse,
                use_sidechains=use_sidechains,
                use_backbone_angles=use_backbone_angles,
                trainable_sparse_to_dense=trainable_sparse_to_dense,
                sidechain_only_sparse=sidechain_only_sparse,
            )
        elif use_class in ["EncoderMap", "Autoencoder"]:
            self.encodermap_with_dataset(sparse=sparse, periodicity=periodicity)

    def ADC_autoencoder_with_dataset(
        self,
        sparse,
        use_sidechains,
        use_backbone_angles,
        trainable_sparse_to_dense,
        sidechain_only_sparse,
        return_dataset_and_model: bool = False,
    ):
        # Encodermap imports
        from encodermap.trajinfo.info_all import TrajEnsemble
        from encodermap.trajinfo.info_single import SingleTraj

        # create a dense dataset
        dataset_length = 512
        shapes = {
            "central_cartesians": (30, 3),
            "central_distances": (29,),
            "central_angles": (28,),
            "central_dihedrals": (27,),
            "side_dihedrals": (22,),
        }

        # create mock trajs
        trajs = TrajEnsemble()
        trajs.CVs_in_file = False
        trajs.traj_files = ["/tmp.json/traj_0.xtc", "/tmp.json/traj_1.xtc"]
        trajs.trajs = [SingleTraj(traj_num=0), SingleTraj(traj_num=1)]
        for i, traj in enumerate(trajs):
            traj._traj_file = Path(f"/tmp/traj_{i}.xtc")
            traj.traj_file = f"/tmp/traj_{i}.xtc"
            traj.top_file = f"/tmp/traj_{i}.pdb"
            traj.n_frames = dataset_length
            traj.backend = "mdtraj"
            traj.id = np.vstack(
                [np.full((dataset_length,), i), np.arange(dataset_length)]
            ).T
            traj.basename = f"{i:02d}"
            traj.time = np.arange(dataset_length) * 10
            for k, s in shapes.items():
                traj.load_CV(np.ones((dataset_length, *s), dtype="float32"), k)

        if use_sidechains:
            dense_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    np.ones(
                        (dataset_length, *shapes["central_angles"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["central_dihedrals"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["central_cartesians"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["central_distances"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["side_dihedrals"]),
                        dtype="float32",
                    ),
                )
            )
        else:
            dense_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    np.ones(
                        (dataset_length, *shapes["central_angles"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["central_dihedrals"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["central_cartesians"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["central_distances"]), dtype="float32"
                    ),
                )
            )
        dense_dataset = dense_dataset.shuffle(
            buffer_size=shapes["central_cartesians"][0],
            reshuffle_each_iteration=True,
        )
        dense_dataset = dense_dataset.repeat()
        dense_dataset = dense_dataset.batch(256)

        # create the parameters
        p = em.ADCParameters(
            dihedral_cost_scale=1,
            angle_cost_scale=1,
            cartesian_cost_scale=1,
            distance_cost_scale=1,
            cartesian_distance_cost_scale=1,
            center_cost_scale=1,
            l2_reg_constant=0.001,
            auto_cost_scale=1,
            main_path=em.misc.run_path(
                str(Path(__file__).resolve().parent / "data/runs")
            ),
            n_steps=50,
            cartesian_cost_variant="mean_abs",
            cartesian_cost_scale_soft_start=(5, 75),
            cartesian_pwd_start=1,
            cartesian_pwd_step=3,
            dihedral_cost_variant="mean_abs",
            cartesian_dist_sig_parameters=[400, 10, 5, 1, 2, 5],
            checkpoint_step=25,
            use_backbone_angles=use_backbone_angles,
            use_sidechains=use_sidechains,
            tensorboard=True,
            summary_step=1,
        )

        if trainable_sparse_to_dense:
            with Capturing() as output:
                p.trainable_dense_to_sparse = True
            self.assertLen(output, 0)

        # create a sparse dataset
        # here; we also compute the pairwise distances of the dense dataset to make sure
        # that reshape(len(data), -1, order="C") can be undone by stack
        sparse_shapes = {
            "central_cartesians": (27, 3),
            "central_distances": (26,),
            "central_angles": (25,),
            "central_dihedrals": (24,),
            "side_dihedrals": (20,),
        }

        sparse_dataset_tensors = {}
        for name, dense_shape in shapes.items():
            sparse_shape = sparse_shapes[name]
            values = np.ones((dataset_length, *dense_shape)).astype("float32")
            if "cartesians" in name:
                values = values.reshape(len(values), -1)
                values[:] = np.nan
                values[:, : sparse_shape[0]] = 1
            else:
                values[:] = np.nan
                values[:, : sparse_shape[0]] = 1
            dense_shape = values.shape
            indices = np.stack(np.where(~np.isnan(values))).T.astype("int64")
            values = values[~np.isnan(values)].flatten()
            t = tf.sparse.SparseTensor(indices, values, dense_shape)
            sparse_dataset_tensors[name] = t

        if use_sidechains:
            sparse_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    sparse_dataset_tensors["central_angles"],
                    sparse_dataset_tensors["central_dihedrals"],
                    sparse_dataset_tensors["central_cartesians"],
                    sparse_dataset_tensors["central_distances"],
                    sparse_dataset_tensors["side_dihedrals"],
                ),
            )
        else:
            sparse_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    sparse_dataset_tensors["central_angles"],
                    sparse_dataset_tensors["central_dihedrals"],
                    sparse_dataset_tensors["central_cartesians"],
                    sparse_dataset_tensors["central_distances"],
                ),
            )
        sparse_dataset = sparse_dataset.shuffle(
            buffer_size=shapes["central_cartesians"][0],
            reshuffle_each_iteration=True,
        )
        sparse_dataset = sparse_dataset.repeat()
        sparse_dataset = sparse_dataset.batch(256)

        # define the dataset
        if sparse:
            assert (
                not sidechain_only_sparse
            ), "Sidechain_only sparse and sparse are mutually exclusive."
        if sidechain_only_sparse:
            assert (
                use_sidechains and use_backbone_angles
            ), "Sidechain only sparse only works when use_sidechains and use-Backbone_angles are true"
        if sparse:
            sig = inspect.signature(trajs[0].load_CV)
            self.assertIn(
                "override",
                sig.parameters,
            )
            dataset = sparse_dataset
            for traj in trajs:
                for k, s in sparse_shapes.items():
                    traj._CVs[k][:, :, s[0] :] = np.nan
            self.assertTrue(np.any(np.isnan(trajs.CVs["central_cartesians"])))
            self.assertTrue(np.any(np.isnan(trajs.CVs["central_angles"])))
            self.assertTrue(np.any(np.isnan(trajs.CVs["central_dihedrals"])))
            self.assertTrue(np.any(np.isnan(trajs.CVs["central_distances"])))
        elif sidechain_only_sparse:
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    np.ones(
                        (dataset_length, *shapes["central_angles"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["central_dihedrals"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["central_cartesians"]), dtype="float32"
                    ),
                    np.ones(
                        (dataset_length, *shapes["central_distances"]), dtype="float32"
                    ),
                    sparse_dataset_tensors["side_dihedrals"],
                ),
            )
            dataset = dataset.repeat()
            dataset = dataset.batch(256)
            for datum in dataset:
                break
            for d, test in zip(datum, [True, True, True, True, False]):
                if test:
                    self.assertNotIsInstance(d, tf.SparseTensor)
                else:
                    self.assertIsInstance(d, tf.SparseTensor)
                    d = tf.sparse.to_dense(d, default_value=np.nan).numpy()
                    self.assertTrue(np.any(np.isnan(d)))
        else:
            dataset = dense_dataset

        if sidechain_only_sparse:
            for i, batch in enumerate(dataset):
                self.assertTrue(
                    not tf.reduce_any(
                        tf.math.is_nan(
                            tf.sparse.to_dense(
                                batch[-1],
                                default_value=0.0,
                            )
                        )
                    ).numpy()
                )
                if i > 100:
                    break

        model = gen_functional_model(
            dataset,
            p,
            write_summary=False,
            sparse=sparse,
            sidechain_only_sparse=sidechain_only_sparse,
        )

        if sidechain_only_sparse:
            self.assertIsNone(
                model.get_dense_model_distances,
            )
            self.assertIsNone(
                model.get_dense_model_cartesians,
            )
            self.assertIsNone(
                model.get_dense_model_central_dihedrals,
            )
            self.assertIsNone(
                model.get_dense_model_central_angles,
            )
            self.assertIsNotNone(
                model.get_dense_model_side_dihedrals,
            )

        # make a forward pass
        for d in dataset:
            break
        self.assertIsNotNone(model(d))

        # save and load it and compare trainable weights
        save_model(model, "/tmp")
        loaded_model = load_model(None, "/tmp")

        for w1, w2 in zip(model.trainable_variables, loaded_model.trainable_variables):
            self.assertAllEqual(w1, w1)
        for w1, w2 in zip(
            model.encoder_model.trainable_variables,
            loaded_model.encoder_model.trainable_variables,
        ):
            self.assertAllEqual(w1, w1)
        for w1, w2 in zip(
            model.decoder_model.trainable_variables,
            loaded_model.decoder_model.trainable_variables,
        ):
            self.assertAllEqual(w1, w1)

        # test the encoder with dataset and with trajs
        emap = em.AngleDihedralCartesianEncoderMap(
            trajs=None,
            parameters=p,
            dataset=dataset,
            read_only=False,
        )

        if return_dataset_and_model:
            return emap, dataset, p

        emap.callbacks.append(em.callbacks.NoneInterruptCallback())

        if sidechain_only_sparse:
            self.assertIsNone(emap.model.get_dense_model_central_angles)
            self.assertIsNone(emap.model.get_dense_model_central_dihedrals)
            self.assertIsNone(emap.model.get_dense_model_distances)
            self.assertIsNone(emap.model.get_dense_model_cartesians)
            self.assertIsNotNone(emap.model.get_dense_model_side_dihedrals)

        lowd = emap.encode()
        self.assertEqual(
            lowd.shape[1],
            2,
        )
        lowd = emap.encode(trajs[0])
        self.assertIsNotNone(lowd)
        self.assertEqual(
            lowd.shape[1],
            2,
        )
        emap2 = em.AngleDihedralCartesianEncoderMap(
            trajs=trajs,
            parameters=p,
            dataset=dataset,
            read_only=False,
        )
        self.assertHasAttr(
            emap2,
            "trajs",
        )
        self.assertEqual(
            emap2.trajs.__class__.__name__,
            "MockTrajEnsemble",
        )
        self.assertEqual(
            trajs[0].CVs["central_dihedrals"].shape[0],
            dataset_length,
            msg=f"{trajs[0].CVs['central_dihedrals'].shape=}",
        )
        lowd = emap2.encode()
        self.assertEqual(
            lowd.shape[1],
            2,
        )
        lowd = emap2.encode(trajs[0])
        self.assertEqual(
            lowd.shape[1],
            2,
        )

        # call the model
        if use_sidechains:
            assert_shapes = [
                (28,),
                (27,),
                (22,),
                (30, 3),
                (45,),
                (45,),
            ]
        else:
            assert_shapes = [
                (28,),
                (27,),
                (30, 3),
                (45,),
                (45,),
            ]

        for o, sh in zip(emap.model(d), assert_shapes):
            self.assertEqual(
                o.shape, (d[0].shape[0], *sh), msg=f"{o.shape=}, {d[0].shape=}, {sh=}"
            )

        # make sure images are not none
        with Capturing() as output:
            emap.add_images_to_tensorboard(max_size=100)
        if use_backbone_angles and use_sidechains:
            self.assertEqual(
                output[0].count("("),
                3,
                msg=(
                    f"Add images to tensorboard will log with wrong data: {output=} "
                    f"{emap.p.use_backbone_angles=} {emap.p.use_sidechains=}"
                ),
            )
        if not use_backbone_angles and not use_sidechains:
            image_callback_data = (
                np.ones((200, *shapes["central_dihedrals"]), dtype="float32"),
            )
        elif use_backbone_angles and not use_sidechains:
            image_callback_data = [
                np.ones((200, *shapes["central_angles"]), dtype="float32"),
                np.ones((200, *shapes["central_dihedrals"]), dtype="float32"),
            ]
        else:
            image_callback_data = [
                np.ones((200, *shapes["central_angles"]), dtype="float32"),
                np.ones((200, *shapes["central_dihedrals"]), dtype="float32"),
                np.ones(
                    (200, *shapes["side_dihedrals"]),
                    dtype="float32",
                ),
            ]
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex=r".*allowed to have multiple ImageCallbacks.*",
            msg=(
                "Overwriting existing CVs should issue a warning to the user, but "
                "it does not."
            ),
        ):
            emap.add_images_to_tensorboard(image_callback_data, max_size=100)
        with self.assertRaises(AssertionError):
            emap.add_images_to_tensorboard(np.random.random((100, 5)))

        # train and assert prints
        with Capturing() as output:
            history = emap.train()
        self.assertIsNotNone(
            history,
            msg=(f"Training did not succeed. Here's the output: {output}"),
        )
        self.assertEqual(emap.p.current_training_step, 50)
        self.assertEqual(
            emap.callbacks[-4].current_step,
            50,
            msg=(f"{output=}\n\n" f"{history=}"),
        )
        self.assertTrue(any(["Saving the model" in line for line in output]))
        self.assertTrue((Path(p.main_path) / "saved_model_25.keras").is_file())

        # check that images are not nan
        auto_image_callback = emap.callbacks[-1]
        self.assertIsNotNone(auto_image_callback.model)
        self.assertIsInstance(
            auto_image_callback, em.callbacks.ImageCallback, msg=f"{emap.callbacks=}"
        )
        lowd = auto_image_callback.model.encoder(auto_image_callback.highd_data).numpy()
        self.assertTrue(
            not np.any(np.isnan(lowd)),
            msg=f"The image callbacks (added with data=None) lowd projection contains np.nans: {lowd=}",
        )
        manual_image_callback = emap.callbacks[-2]
        self.assertIsNotNone(manual_image_callback.model)
        lowd = manual_image_callback.model.encoder(
            manual_image_callback.highd_data
        ).numpy()
        self.assertTrue(
            not np.any(np.isnan(lowd)),
            msg=f"The image callbacks (added with data=np.ndarray) lowd projection contains np.nans: {lowd=}",
        )

        # Third Party Imports
        from tensorflow.python.summary.summary_iterator import summary_iterator

        scaling = []
        after_scaling = []
        train_dir = Path(p.main_path) / "train"
        records_files = train_dir.glob("*")
        for records_file in records_files:
            records = summary_iterator(str(records_file))
            for rec in records:
                for v in rec.summary.value:
                    if "Cartesian Cost current scaling" in v.tag:
                        scaling.append(struct.unpack("f", v.tensor.tensor_content)[0])
                    if "Cartesian Cost after scaling" in v.tag:
                        after_scaling.append(
                            struct.unpack("f", v.tensor.tensor_content)[0]
                        )
        scaling = np.array(scaling)
        after_scaling = np.array(after_scaling)
        assert len(scaling) == 49
        scaling_should_be = []
        for i in range(50):
            if i < 5:
                scaling_should_be.append(0)
            elif 5 <= i <= 75:
                scaling_should_be.append(1 / (75 - 5) * (i - 5))
            else:
                scaling_should_be.append(1)
        scaling_should_be = np.array(scaling_should_be)[1:]
        self.assertLen(
            scaling,
            49,
            msg=(
                f"The AngleDihedralCartesianEncoderMap did not log the correct number of"
                f"'Cartesian Cost current scaling' tf.summaries to tensorboard."
            ),
        )
        self.assertLess(
            scaling[0],
            scaling[-1],
            msg=(
                f"The 'Cartesian Cost current scaling' did not increase during training. "
                f"The parameters asked to increase the cartesian cost scaling starting "
                f"from 0, ramping up at step {p.cartesian_cost_scale_soft_start[0]=} to step "
                f"{p.cartesian_cost_scale_soft_start[1]=} where it should stay at 1, however,"
                f"these are the scales every 5 steps:\n{scaling[::5]}\n\nAnd here are the "
                f"callbacks:\n{emap.callbacks[-4].current_cartesian_cost_scale=}"
                f"\n\nand the loss functions:\n{emap.loss[2]=}\n\n{output}"
            ),
        )
        self.assertAllClose(
            scaling, scaling_should_be, msg=f"{scaling=}, {scaling_should_be=}"
        )

        # test encode
        highd_data = (
            np.random.random((100, shapes["central_dihedrals"][0])) * 2 * np.pi - np.pi
        )
        if use_backbone_angles:
            highd_data = [
                np.random.random((100, shapes["central_angles"][0])) * 2 * np.pi
                - np.pi,
                highd_data,
            ]
        if use_sidechains:
            highd_data.append(
                np.random.random((100, shapes["side_dihedrals"][0])) * 2 * np.pi - np.pi
            )

        lowd = emap.encode(highd_data)
        self.assertEqual(lowd.shape[1], p.n_neurons[-1])

        # test decoder and generate
        decoded = emap.decode(lowd)
        for inp, out in zip(highd_data, decoded):
            self.assertEqual(inp.shape, out.shape)

        # load the parameter file ourselves
        # Standard Library Imports
        import json

        with open(Path(p.main_path) / "parameters.json") as f:
            json_data = json.load(f)
        self.assertEqual(
            json_data["current_training_step"],
            50,
        )

        loaded_emap = em.AngleDihedralCartesianEncoderMap.from_checkpoint(
            trajs=None,
            checkpoint_path=p.main_path,
            dataset=dataset,
        )

        # assert weights
        for w1, w2, w3 in zip(
            emap.model.trainable_variables,
            loaded_emap.model.trainable_variables,
            model.trainable_variables,
        ):
            self.assertAllEqual(w1, w2)
            self.assertNotAllEqual(w1, w3)
        for w1, w2, w3 in zip(
            emap.model.encoder_model.trainable_variables,
            loaded_emap.model.encoder_model.trainable_variables,
            model.encoder_model.trainable_variables,
        ):
            self.assertAllEqual(w1, w2)
            self.assertNotAllEqual(w1, w3)
        for w1, w2, w3 in zip(
            emap.model.decoder_model.trainable_variables,
            loaded_emap.model.decoder_model.trainable_variables,
            model.decoder_model.trainable_variables,
        ):
            self.assertAllEqual(w1, w2)
            self.assertNotAllEqual(w1, w3)

        # retrain
        try:
            cartesian_cost_scale = loaded_emap.callbacks[
                -1
            ].current_cartesian_cost_scale.numpy()
        except AttributeError as e:
            raise Exception(f"{loaded_emap.callbacks=}") from e
        self.assertAllClose(
            cartesian_cost_scale, 0.64285713434, msg=f"{cartesian_cost_scale=}"
        )
        with Capturing() as output:
            loaded_emap.train()
        self.assertTrue(any(["has already been trained" in line for line in output]))
        loaded_emap.p.n_steps += 50
        with Capturing() as output:
            loaded_emap.train()
        self.assertAllClose(loaded_emap.callbacks[-1].current_cartesian_cost_scale, 1.0)

    def encodermap_with_dataset(self, sparse, periodicity):
        if periodicity < float("inf"):
            if not sparse:
                data = np.random.random((2000, 24)) * 2 * np.pi - np.pi
            else:
                data = np.random.random((2000, 24)) * 2 * np.pi - np.pi
                data[1000:, -4:] = np.nan
        else:
            if not sparse:
                data = np.random.random((2000, 24)) * 10
            else:
                data = np.random.random((2000, 24)) * 10
                data[1000:, -4:] = np.nan

        p = em.Parameters(
            periodicity=periodicity,
            distance_cost_scale=1,
            center_cost_scale=1,
            l2_reg_constant=0.001,
            auto_cost_scale=1,
            main_path=em.misc.run_path(
                str(Path(__file__).resolve().parent / "data/runs")
            ),
            n_steps=50,
            checkpoint_step=25,
            tensorboard=True,
        )

        # current_step = 0
        emap = em.EncoderMap(p, data)
        self.assertIsNotNone(emap.model)

        # make sure images are not none
        emap.add_images_to_tensorboard(max_size=100)
        emap.add_images_to_tensorboard(data[::2], max_size=100)
        with self.assertRaises(AssertionError):
            emap.add_images_to_tensorboard(np.random.random((100, 5)))

        # train
        with Capturing() as output:
            emap.train()
        output = "\n".join(output)
        self.assertIn("Saving the model", output)
        self.assertTrue((Path(p.main_path) / "saved_model_25.keras").is_file())
        self.assertEqual(emap.p.current_training_step, 50)

        # check that images are not nan
        auto_image_callback = emap.callbacks[-2]
        self.assertIsNotNone(auto_image_callback.model)
        lowd = auto_image_callback.model.encoder(auto_image_callback.highd_data).numpy()
        self.assertTrue(
            not np.any(np.isnan(lowd)),
            msg=f"The image callbacks (added with data=None) lowd projection contains np.nans: {lowd=}",
        )
        manual_image_callback = emap.callbacks[-1]
        self.assertIsNotNone(manual_image_callback.model)
        lowd = manual_image_callback.model.encoder(
            manual_image_callback.highd_data
        ).numpy()
        self.assertTrue(
            not np.any(np.isnan(lowd)),
            msg=f"The image callbacks (added with data=np.ndarray) lowd projection contains np.nans: {lowd=}",
        )

        # provide manual data if sparse it should be transformed
        with self.assertRaises(Exception) as exc:
            loaded_emap = em.EncoderMap.from_checkpoint(
                checkpoint_path=Path(p.main_path) / "saved_model_25.keras",
                sparse=sparse,
            )
        self.assertIn("The model was saved at step", str(exc.exception))

        loaded_emap = em.EncoderMap.from_checkpoint(
            checkpoint_path=p.main_path, sparse=sparse
        )
        self.assertTrue(loaded_emap._using_hypercube)
        self.assertEqual(
            loaded_emap.p.current_training_step,
            50,
            msg=f"{loaded_emap.p.current_training_step=}",
        )
        with Capturing() as output:
            loaded_emap.train()
        self.assertTrue(any(["has already been trained" in line for line in output]))

        # assert the weights
        for w1, w2 in zip(
            emap.model.trainable_variables, loaded_emap.model.trainable_variables
        ):
            self.assertAllEqual(w1, w2)

        loaded_emap.p.n_steps += 50
        self.assertEqual(loaded_emap.p.n_steps, 100)
        self.assertTrue(loaded_emap._using_hypercube)
        with Capturing() as output:
            loaded_emap.train()
        self.assertTrue(
            any(["yet provided with train data." in line for line in output]),
            f"{output=}",
        )

        loaded_emap = em.EncoderMap.from_checkpoint(
            checkpoint_path=p.main_path, sparse=sparse, train_data=data
        )
        loaded_emap.p.n_steps += 50
        h = loaded_emap.train()
        self.assertIsNotNone(h)

        # assert not raises
        loaded_emap = em.EncoderMap.from_checkpoint(
            checkpoint_path=Path(p.main_path) / "saved_model_25.keras",
            train_data=data,
            sparse=sparse,
            use_previous_model=True,
        )
        self.assertEqual(
            loaded_emap.p.n_steps,
            25,
        )
        self.assertEqual(
            loaded_emap.p.current_training_step,
            25,
        )
        loaded_emap.p.n_steps += 50
        self.assertEqual(
            loaded_emap.p.n_steps,
            75,
        )
        loaded_emap.train()
        self.assertEqual(
            loaded_emap.p.current_training_step,
            75,
            msg=f"{loaded_emap.p.current_training_step=}",
        )

    def test_load_legacy_model(self):
        dataset_name = "encodermap_legacy_data"
        output_dir = Path(__file__).resolve().parent / "data/legacy"
        em.kondata.get_from_kondata(
            dataset_name=dataset_name,
            output=output_dir,
            silence_overwrite_message=True,
            mk_parentdir=True,
        )

        dataset_length = 512
        shapes = {
            "cartesians": (369, 3),
            "distances": (368,),
            "angles": (367,),
            "dihedrals": (366,),
            "side_dihedrals": (272,),
        }
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                np.ones((dataset_length, *shapes["angles"]), dtype="float32"),
                np.ones((dataset_length, *shapes["dihedrals"]), dtype="float32"),
                np.ones((dataset_length, *shapes["cartesians"]), dtype="float32"),
                np.ones((dataset_length, *shapes["distances"]), dtype="float32"),
                np.ones((dataset_length, *shapes["side_dihedrals"]), dtype="float32"),
            )
        )
        dataset = dataset.shuffle(
            buffer_size=dataset_length,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.repeat()
        dataset = dataset.batch(256)

        emap = em.AngleDihedralCartesianEncoderMap.from_checkpoint(
            trajs=None,
            checkpoint_path=output_dir / "acd_saved_model.model",
            dataset=dataset,
        )

        # fmt: off
        self.assertIsInstance(emap, em.AngleDihedralCartesianEncoderMap)
        self.assertIsNotNone(
            emap.encode(
                [
                    np.ones((dataset_length, *shapes["angles"]), dtype="float32"),
                    np.ones((dataset_length, *shapes["dihedrals"]), dtype="float32"),
                    np.ones((dataset_length, *shapes["side_dihedrals"]), dtype="float32"),
                ]
            )
        )
        # fmt: on

        bad_dataset = tf.data.Dataset.from_tensor_slices(
            (
                np.ones((dataset_length, 50), dtype="float32"),
                np.ones((dataset_length, 50), dtype="float32"),
            )
        )
        bad_dataset = bad_dataset.shuffle(
            buffer_size=dataset_length,
            reshuffle_each_iteration=True,
        )
        bad_dataset = bad_dataset.repeat()
        bad_dataset = bad_dataset.batch(256)

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                np.ones((dataset_length, 165), dtype="float32"),
                np.ones((dataset_length, 165), dtype="float32"),
            )
        )
        dataset = dataset.shuffle(
            buffer_size=dataset_length,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.repeat()
        dataset = dataset.batch(256)

        with self.assertRaises(Exception) as exc:
            emap = em.EncoderMap.from_checkpoint(
                checkpoint_path=output_dir / "saved_model_50000.model*",
                train_data=bad_dataset,
            )
        self.assertIn(
            "Are you sure",
            str(exc.exception),
            msg=(f"Are you sure not found in {exc.exception}"),
        )

        emap = em.EncoderMap.from_checkpoint(
            checkpoint_path=output_dir / "saved_model_50000.model*",
            train_data=dataset,
        )
        self.assertIsNotNone(emap.encode(np.ones((200, 165), dtype="float32")))


################################################################################
# Collect Test Cases and Filter
################################################################################


def load_tests(loader, tests, pattern):
    test_cases = (TestAutoencoder, TestParameters)
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

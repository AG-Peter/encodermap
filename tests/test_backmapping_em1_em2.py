# -*- coding: utf-8 -*-
# tests/test_backmapping_em1_em2.py
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
"""Tests the backmapping of EncoderMap. Make sure to differentiate between:

* Backmapping while training:
    While training, EncoderMap uses the distances, angles and dihedrals of the
    backbone to create a new backbone chain. The coordinates of this chain will
    be used in two different loss functions to steer the training. However, the
    coordinates of the sidechains are dropped during this phase. The sidechains
    are only trained by matching input and output sidechain angles (i.e., chi1-chi5).
* Backmapping after training:
    After the training has finished, 2D coordinates can be provided to the decoder
    part of the network. The decoder will then output dihedrals, angles, and
    sidechain dihedrals (based on chosen parameters). The backbone and sidechain
    dihedrals can be used to adjust a molecular conformation to these decoded
    dihedral angles. Here, all provided atoms are also put out.

"""


################################################################################
# Imports
################################################################################

# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import inspect
import unittest
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from itertools import cycle, islice
from pathlib import Path
from time import perf_counter
from typing import Optional, Union

# Third Party Imports
import mdtraj as md
import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

# Encodermap imports
from encodermap.encodermap_tf1.backmapping import (
    chain_in_plane,
    dihedrals_to_cartesian_tf,
)
from encodermap.encodermap_tf1.backmapping import guess_amide_H as guess_amide_H_tf1
from encodermap.encodermap_tf1.backmapping import guess_amide_O as guess_amide_O_tf1
from encodermap.encodermap_tf1.backmapping import (
    merge_cartesians as merge_cartesians_tf1,
)
from encodermap.misc import pairwise_dist
from encodermap.misc.backmapping import (
    guess_amide_H,
    guess_amide_O,
    merge_cartesians,
    split_and_reverse_cartesians,
    split_and_reverse_dihedrals,
)
from encodermap.models.models import SequentialModel


try:
    # Encodermap imports
    from conftest import expensive_test
except ImportError:
    from .conftest import expensive_test

import encodermap as em  # isort: skip


################################################################################
# Utils
################################################################################


class LayerThatOutputsConstant(tf.keras.layers.Layer):
    def __init__(self, units, output_len, output_constant=0, name="Latent"):
        super(LayerThatOutputsConstant, self).__init__()
        self.output_constant = tf.constant(
            output_constant, shape=(output_len, units), dtype="float32"
        )
        self._name = name

    def call(self, inputs):
        return self.output_constant


class ConstantOutputAutoencoder(SequentialModel):
    def __init__(self, input_dim, len_data, parameters=None, latent_constant=0):
        super(ConstantOutputAutoencoder, self).__init__(input_dim, parameters)
        regularizer = tf.keras.regularizers.l2(self.p.l2_reg_constant)
        input_layer = tf.keras.layers.Dense(
            input_shape=(input_dim,),
            units=self.encoder_layers[0][0],
            activation=self.encoder_layers[0][1],
            name=self.encoder_layers[0][2],
            kernel_initializer=tf.initializers.VarianceScaling(),
            kernel_regularizer=regularizer,
            bias_initializer=tf.initializers.RandomNormal(0.1, 0.5),
        )

        # constant output
        constant_layer = LayerThatOutputsConstant(
            self.p.n_neurons[-1], parameters.batch_size, latent_constant, name="Latent"
        )

        # overwrite encoder
        self.encoder = tf.keras.Sequential(
            [input_layer]
            + [
                tf.keras.layers.Dense(
                    n_neurons,
                    activation=act_fun,
                    name=name,
                    kernel_initializer=tf.initializers.VarianceScaling(),
                    kernel_regularizer=regularizer,
                    bias_initializer=tf.initializers.RandomNormal(0.1, 0.5),
                )
                for n_neurons, act_fun, name in self.encoder_layers[1:-1]
            ]
            + [constant_layer],
            name="Encoder",
        )
        self.build(input_shape=(1, input_dim))

    def call(self, x):
        return x


@contextmanager
def catchtime() -> float:
    """Catches execution time in a contextmanager.

    Examples:
        >>> import sleep
        >>> with catchtime() as c:
        ...     sleep(1)
        >>> c()
        1  # doctest: +SKIP

    """
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()


def roundrobin(*iterables):
    """Creates a roundrobin iteration of *iterables

    Examples:
        >>> a = roundrobin('ABC', 'D', 'EF')
        >>> a
        A D E B F C

    """
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


################################################################################
# Test Suites
################################################################################


class TestBackmappingEm1Em2(tf.test.TestCase):
    def test_backmapping_wo_angles(self):
        p = em.ADCParameters(l2_reg_constant=0, periodicity=2 * np.pi)
        print(p)
        no_central_cartesians = 474  # same as 1am7 protein dihedral length
        if p.use_backbone_angles:
            input_dim = no_central_cartesians - 3 + no_central_cartesians - 2
        else:
            input_dim = no_central_cartesians - 3
        len_data = 100
        model_0 = ConstantOutputAutoencoder(input_dim, len_data, p)
        # model_1 for different latent checks
        model_1 = ConstantOutputAutoencoder(input_dim, len_data, p, 1)
        # model.compile(tf.keras.optimizers.Adam())

        dihedral_data = (
            np.random.random((len_data, no_central_cartesians - 3)).astype("float32")
            * np.pi
            * 2
        ) - np.pi
        angle_data = (
            np.random.random((len_data, no_central_cartesians - 2)).astype("float32")
            * np.pi
            * 2
        ) - np.pi
        distance_data = np.random.random((len_data, no_central_cartesians - 1)).astype(
            "float32"
        )
        central_cartesian_data = np.random.random(
            (len_data, no_central_cartesians, 3)
        ).astype("float32")
        cartesian_data = np.random.random(
            (len_data, no_central_cartesians * 5, 3)
        ).astype("float32")

        dataset = tf.data.Dataset.from_tensor_slices(
            (angle_data, dihedral_data, central_cartesian_data)
        )
        dataset = dataset.shuffle(buffer_size=len_data, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(p.batch_size)

        for i in range(2):
            d = dataset.take(1)
            if i == 0:
                angles, dihedrals, cartesians = d
            elif i == 1:
                angles_2, dihedrals_2, cartesians_2 = d
            else:
                break

        # what to pass through network
        if p.use_backbone_angles:
            main_inputs = tf.concat([angles, dihedrals], 1)
        else:
            main_inputs = dihedrals

        # run inputs through network
        print("main_inputs:", main_inputs.shape)
        encoded = model_0.encoder(main_inputs)
        print("encoded:", encoded.shape)
        decoded = model_0(main_inputs)
        print(tf.math.reduce_all(tf.equal(main_inputs, decoded)))
        print("decoded:", decoded.shape)

        # unpack the output
        if p.use_backbone_angles:
            assert decoded.shape == main_inputs.shape
            generated_angles = decoded[:, : angle_data.shape[1]]
            generated_dihedrals = decoded[:, angle_data.shape[1] :]
        else:
            generated_dihedrals = model_0(main_inputs)
            # If angles are not trained, use the mean from all provided angles
            generated_angles = tf.tile(
                np.expand_dims(np.mean(angle_data, 0), 0),
                multiples=(out_dihedrals.shape[0], 1),
            )

        # mean lengths over trajectory used for backmapping
        mean_lengths = np.expand_dims(np.mean(distance_data, 0), 0)

        # build s chain in plane with lengths and angles
        _chain_in_plane = chain_in_plane(mean_lengths, generated_angles)

        # add a third dimension by adding torsion angles to that chain
        cartesians = dihedrals_to_cartesian_tf(
            generated_dihedrals + np.pi, _chain_in_plane
        )

        # for a standard protein these names are always the same, that's why I removed them
        # and exchanged them with indices
        atom_names = ["N", "CA", "C"] * int(no_central_cartesians / 3)

        # compare H cartesians
        amide_H_cartesians_tf1 = guess_amide_H_tf1(cartesians, atom_names)
        amide_H_cartesians = guess_amide_H(
            cartesians, np.arange(cartesians.shape[1])[::3]
        )
        self.assertAllEqual(amide_H_cartesians_tf1, amide_H_cartesians)

        # compare O cartesians
        amide_O_cartesians_tf1 = guess_amide_O_tf1(cartesians, atom_names)
        amide_O_cartesians = guess_amide_O(
            cartesians, np.arange(cartesians.shape[1])[2::3]
        )
        self.assertAllEqual(amide_O_cartesians_tf1, amide_O_cartesians)

        # merge the cartesians from chain_in_plane backmapping with the amide atoms
        merged_cartesians_tf1 = merge_cartesians_tf1(
            cartesians, atom_names, amide_H_cartesians_tf1, amide_O_cartesians_tf1
        )
        merged_cartesians = merge_cartesians(
            cartesians,
            np.arange(cartesians.shape[1])[::3],
            np.arange(cartesians.shape[1])[2::3],
            amide_H_cartesians,
            amide_O_cartesians,
        )
        self.assertAllEqual(merged_cartesians_tf1, merged_cartesians)

        # These are here to test whether they run or not
        # They will be tested with the always zero model in the test_losses unittest
        inp_pairwise = pairwise_dist(
            cartesian_data[
                :10, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )
        encoded_pairwise = pairwise_dist(
            cartesians[
                :10, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )
        clashes = tf.math.count_nonzero(
            pairwise_dist(cartesians, flat=True) < 1, axis=1, dtype=tf.float32
        )

    def test_backmapping_wo_angles(self):
        p = em.ADCParameters(
            l2_reg_constant=0, periodicity=2 * np.pi, use_backbone_angles=True
        )
        print(p)
        no_central_cartesians = 474  # same as 1am7 protein dihedral length
        if p.use_backbone_angles:
            input_dim = no_central_cartesians - 3 + no_central_cartesians - 2
        else:
            input_dim = no_central_cartesians - 3
        len_data = 100
        model_0 = ConstantOutputAutoencoder(input_dim, len_data, p)
        # model_1 for different latent checks
        model_1 = ConstantOutputAutoencoder(input_dim, len_data, p, 1)
        # model.compile(tf.keras.optimizers.Adam())

        dihedral_data = (
            np.random.random((len_data, no_central_cartesians - 3)).astype("float32")
            * np.pi
            * 2
        ) - np.pi
        angle_data = (
            np.random.random((len_data, no_central_cartesians - 2)).astype("float32")
            * np.pi
            * 2
        ) - np.pi
        distance_data = np.random.random((len_data, no_central_cartesians - 1)).astype(
            "float32"
        )
        central_cartesian_data = np.random.random(
            (len_data, no_central_cartesians, 3)
        ).astype("float32")
        cartesian_data = np.random.random(
            (len_data, no_central_cartesians * 5, 3)
        ).astype("float32")

        dataset = tf.data.Dataset.from_tensor_slices(
            (angle_data, dihedral_data, central_cartesian_data)
        )
        dataset = dataset.shuffle(buffer_size=len_data, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(p.batch_size)

        for i, d in enumerate(dataset):
            if i == 0:
                angles, dihedrals, cartesians = d
            elif i == 1:
                angles_2, dihedrals_2, cartesians_2 = d
            else:
                break

        # what to pass through network
        if p.use_backbone_angles:
            main_inputs = tf.concat([angles, dihedrals], 1)
        else:
            main_inputs = dihedrals

        # run inputs through network
        print("main_inputs:", main_inputs.shape)
        encoded = model_0.encoder(main_inputs)
        print("encoded:", encoded.shape)
        decoded = model_0(main_inputs)
        print(tf.math.reduce_all(tf.equal(main_inputs, decoded)))
        print("decoded:", decoded.shape)

        # unpack the output
        if p.use_backbone_angles:
            assert decoded.shape == main_inputs.shape
            generated_angles = decoded[:, : angle_data.shape[1]]
            generated_dihedrals = decoded[:, angle_data.shape[1] :]
        else:
            generated_dihedrals = model_0(main_inputs)
            # If angles are not trained, use the mean from all provided angles
            generated_angles = tf.tile(
                np.expand_dims(np.mean(angle_data, 0), 0),
                multiples=(out_dihedrals.shape[0], 1),
            )

        # mean lengths over trajectory used for backmapping
        mean_lengths = np.expand_dims(np.mean(distance_data, 0), 0)

        # build s chain in plane with lengths and angles
        _chain_in_plane = chain_in_plane(mean_lengths, generated_angles)

        # add a third dimension by adding torsion angles to that chain
        cartesians = dihedrals_to_cartesian_tf(
            generated_dihedrals + np.pi, _chain_in_plane
        )

        # for a standard protein these names are always the same, that's why I removed them
        # and exchanged them with indices
        atom_names = ["N", "CA", "C"] * int(no_central_cartesians / 3)

        # compare H cartesians
        amide_H_cartesians_tf1 = guess_amide_H_tf1(cartesians, atom_names)
        amide_H_cartesians = guess_amide_H(
            cartesians, np.arange(cartesians.shape[1])[::3]
        )
        self.assertAllEqual(amide_H_cartesians_tf1, amide_H_cartesians)

        # compare O cartesians
        amide_O_cartesians_tf1 = guess_amide_O_tf1(cartesians, atom_names)
        amide_O_cartesians = guess_amide_O(
            cartesians, np.arange(cartesians.shape[1])[2::3]
        )
        self.assertAllEqual(amide_O_cartesians_tf1, amide_O_cartesians)

        # merge the cartesians from chain_in_plane backmapping with the amide atoms
        merged_cartesians_tf1 = merge_cartesians_tf1(
            cartesians, atom_names, amide_H_cartesians_tf1, amide_O_cartesians_tf1
        )
        merged_cartesians = merge_cartesians(
            cartesians,
            np.arange(cartesians.shape[1])[::3],
            np.arange(cartesians.shape[1])[2::3],
            amide_H_cartesians,
            amide_O_cartesians,
        )
        self.assertAllEqual(merged_cartesians_tf1, merged_cartesians)

        # These are here to test whether they run or not
        # They will be tested with the always zero model in the test_losses unittest
        inp_pairwise = pairwise_dist(
            cartesian_data[
                :10, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )
        encoded_pairwise = pairwise_dist(
            cartesians[
                :10, p.cartesian_pwd_start : p.cartesian_pwd_stop : p.cartesian_pwd_step
            ],
            flat=True,
        )
        clashes = tf.math.count_nonzero(
            pairwise_dist(cartesians, flat=True) < 1, axis=1, dtype=tf.float32
        )


class TestBackmappingMdtrajMdanalysis(unittest.TestCase):
    custom_aas_K48_diUbi = {
        "LYQ-123": (
            "K",
            {  # LYQ-123 is basically just lysine
                "bonds": [  # the bonds are defined as a list of tuples
                    ("-C", "N"),  # the peptide bond to the previous aa
                    ("N", "CA"),
                    ("N", "H"),
                    ("CA", "C"),
                    ("C", "O"),
                    ("CA", "CB"),
                    ("CB", "CG"),
                    ("CG", "CD"),
                    ("CD", "CQ"),
                    ("CQ", "NQ"),
                    ("NQ", "HQ"),
                    ("NQ", 759),  # the isopeptide bond to atom index 758 (GLQ-75 C)
                    ("C", "+N"),  # the peptide bond to the next aa
                ],
                "CHI1": ["N", "CA", "CB", "CG"],  # LYQ-123 has special atom names
                "CHI2": ["CA", "CB", "CG", "CD"],  # for its chi angles, so we define
                "CHI3": ["CB", "CG", "CD", "CQ"],  # them here, so they can be picked
                "CHI4": ["CG", "CD", "CQ", "NQ"],  # up by the featurizer
            },
        ),
        "GLQ-75": (
            "G",
            {  # GLQ-75 is basically just glycine
                "bonds": [
                    ("-C", "N"),  # the peptide bond to the previous aa
                    ("N", "CA"),
                    ("N", "H"),
                    ("CA", "C"),
                    ("C", "O"),
                ],
                "delete_bonds": [
                    (
                        "C",
                        761,
                    )  # remove the automatically generated bond to MET1 (MET-76) of the 2nd Ubi unit
                ],
                "PHI": [
                    "-C",
                    "N",
                    "CA",
                    "C",
                ],  # GLQ-75 only takes part in a phi angle, not in psi or omega.
                "PSI": "delete",
                "OMEGA": "delete",
            },
        ),
        "MET-76": (
            "M",
            {  # MET-67 has no dihedral angles to GLQ-75
                "PHI": "delete",
                "PSI": ["N", "CA", "C", "+N"],
                "OMEGA": ["CA", "C", "+N", "+CA"],
            },
        ),
    }

    def assertHasAttr(self, obj, intendedAttr):
        """Helper to check whether an attr is present."""
        testBool = hasattr(obj, intendedAttr)
        self.assertTrue(
            testBool, msg=f"obj lacking an attribute. {obj=}, {intendedAttr=}"
        )

    def assertHasMember(self, obj, intendedMember, msg=None):
        base_msg = f"Object {obj} lacking member {intendedMember=}."
        if msg is not None:
            msg = f"{base_msg} {msg}"
        else:
            msg = base_msg
        for member in inspect.getmembers(obj):
            if member[0] == intendedMember:
                break
        else:
            self.fail(msg)

    def assertAllClose(
        self,
        actual: Union[np.ndarray, Iterable, int, float],
        desired: Union[np.ndarray, Iterable, int, float],
        rtol: Optional[float] = 1e-7,
        atol: Optional[float] = 0,
        equal_nan: Optional[bool] = True,
        err_msg: Optional[str] = "",
        verbose: Optional[bool] = True,
    ) -> None:
        try:
            assert_allclose(
                actual=actual,
                desired=desired,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                err_msg=err_msg,
                verbose=verbose,
            )
        except AssertionError as e:
            self.fail(str(e))

    @expensive_test
    def test_backmapping_mdtraj_vs_mdanalysis_performance(self):
        output_dir = Path(
            em.get_from_kondata(
                "pASP_pGLU",
                mk_parentdir=True,
                silence_overwrite_message=True,
            )
        )
        traj = em.load(output_dir / "asp7.xtc", output_dir / "asp7.pdb")
        traj.load_CV(["central_dihedrals", "side_dihedrals"])
        dihedrals = np.random.uniform(
            low=-np.pi, high=np.pi, size=(100, traj.central_dihedrals.shape[-1])
        )
        # Encodermap imports
        from encodermap.encodermap_tf1.backmapping import dihedral_backmapping
        from encodermap.misc.backmapping import mdtraj_backmapping

        with catchtime() as ctime1:
            out1 = dihedral_backmapping(
                pdb_path=str(output_dir / "asp7.pdb"),
                dihedral_trajectory=dihedrals,
            )
        with catchtime() as ctime2:
            out2 = mdtraj_backmapping(
                top=str(output_dir / "asp7.pdb"),
                dihedrals=dihedrals,
                remove_component_size=10,
                parallel=False,
            )

        print(
            f"For the ASP7 test:\n"
            f"Execution time for MDAnalysis: {ctime1()}\n"
            f"Execution time for MDTraj: {ctime2()}"
        )

        traj = em.load(
            Path(__file__).resolve().parent / "data/1am7_corrected.xtc",
            Path(__file__).resolve().parent / "data/1am7_protein.pdb",
        )
        traj.load_CV(["central_dihedrals"])

        dihedrals = np.random.uniform(
            low=-np.pi, high=np.pi, size=(100, traj.central_dihedrals.shape[-1])
        )

        pro_indices = traj._CVs.central_dihedrals_feature_indices.coords[
            "CENTRAL_DIHEDRALS"
        ].str.contains("CENTERDIH PHI   RESID  PRO")
        self.assertTrue(np.any(pro_indices))
        ind = np.where(pro_indices)[0]
        # sets the proline angles inbetween +90 and +180 degrees
        # which is far off from what they should be.
        dihedrals[:, ind] = np.random.uniform(
            low=np.pi / 2, high=np.pi, size=(100, len(ind))
        )

        with catchtime() as ctime1:
            out1 = dihedral_backmapping(
                pdb_path=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
                dihedral_trajectory=np.rad2deg(dihedrals),
            )
        with catchtime() as ctime2:
            out2 = mdtraj_backmapping(
                top=Path(__file__).resolve().parent / "data/1am7_protein.pdb",
                dihedrals=dihedrals,
                remove_component_size=10,
            )
        out2 = em.SingleTraj(out2)
        out2.load_CV("central_dihedrals")
        angles = np.rad2deg(out2.central_dihedrals[:, pro_indices])
        ind = (-80 <= angles) & (angles <= -46)
        print(f"{np.all(ind)=}")
        self.assertTrue(
            np.all(ind),
            msg=f"Some proline angles of the MDTraj backmapping function "
            f"`mdtraj_backmapping` are outside of their natural phi angle range "
            f"of -63 +/- 17 degrees:\n{angles[~ind]}",
        )

        print(
            f"For the 1am7_protein test:\n"
            f"Execution time for MDAnalysis: {ctime1()}\n"
            f"Execution time for MDTraj: {ctime2()}"
        )

        # psi = [res.psi_selection() for res in out1.residues]
        # omega = [res.omega_selection() for res in out1.residues]
        # phi = [res.phi_selection() for res in out1.residues]
        # ags = roundrobin(psi, omega, phi)
        ags = [res.omega_selection() for res in out1.residues if res.resname == "PRO"]
        ags = list(filter(lambda x: x is not None, ags))

        # Third Party Imports
        from MDAnalysis.analysis.dihedrals import Dihedral

        angles = Dihedral(ags).run().results["angles"]
        ind = (-80 <= angles) & (angles <= -46)
        if not np.all(ind):
            print(
                f"Some proline angles of the MDAnalysis backmapping function "
                f"`dihedral_backmapping` are outside of their natural phi angle range "
                f"of -63 +/- 17 degrees:\n{angles[~ind]}"
            )

    @expensive_test
    def test_custom_AAs_with_KAC(self):
        # define output dir and load trajs
        output_dir = Path(
            em.get_from_kondata(
                "mono_Ub_Ac",
                mk_parentdir=True,
                silence_overwrite_message=True,
            ),
        )
        trajs = list(output_dir.rglob("*_I/*.xtc"))
        self.assertGreater(len(trajs), 0)

        custom_aas = {
            "KAC": (
                "K",
                {
                    "bonds": [
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
        for t in trajs:
            self.assertTrue(Path(t).is_file())
        tops = [t.parent / "start.pdb" for t in trajs]
        for t in tops:
            self.assertTrue(Path(t).is_file())
        trajs = em.load(
            trajs=trajs,
            tops=tops,
            basename_fn=lambda x: str(x).split("/")[-2],
            common_str=[
                "K6",
                "K11",
                "K27",
                "K29",
                "K33",
                "K48",
            ],
            custom_top=custom_aas,
        )

        # test some common strings
        for cs, sub_trajs in trajs.trajs_by_common_str.items():
            for t in sub_trajs:
                self.assertIn(cs, t.traj_file)
                self.assertIn(cs, t.top_file)

        # assert some bonds and dihedrals
        self.assertIn(
            [106, 107, 108, 109],
            trajs.trajs_by_common_str["K11"][0].indices_chi4.tolist(),
        )
        self.assertIn(
            [107, 108, 109, 111],
            trajs.trajs_by_common_str["K11"][0].indices_chi5.tolist(),
        )
        self.assertIn(
            [109, 111],
            [
                [a.index, b.index]
                for a, b in trajs.trajs_by_common_str["K11"][0].top.bonds
            ],
        )
        self.assertIn(
            [316, 317, 318, 319],
            trajs.trajs_by_common_str["K33"][0].indices_chi4.tolist(),
        )
        self.assertIn(
            [317, 318, 319, 321],
            trajs.trajs_by_common_str["K33"][0].indices_chi5.tolist(),
        )
        self.assertIn(
            [319, 321],
            [
                [a.index, b.index]
                for a, b in trajs.trajs_by_common_str["K33"][0].top.bonds
            ],
        )

        # assert that for K11Ac, the sidechain dihedrals follow the mandated order
        # indices and labels
        # Encodermap imports
        from encodermap.loading.features import SideChainDihedrals

        feat = SideChainDihedrals(trajs[0])
        self.assertTrue(all(["MET" in i and "1" in i for i in feat.describe()[:3]]))
        indices = feat.angle_indexes.copy()
        diffs = np.diff(indices, axis=0)
        self.assertTrue(
            np.all(diffs > 0), msg=f"Indices are not monotonically increasing."
        )
        feat = SideChainDihedrals(trajs[0], generic_labels=True)
        self.assertIn(
            "CHI1",
            feat.describe()[0],
        )
        self.assertIn(
            "CHI2",
            feat.describe()[1],
        )
        self.assertIn(
            "CHI3",
            feat.describe()[2],
        )
        self.assertTrue(np.array_equal(feat.angle_indexes, indices))
        feat = SideChainDihedrals(trajs[0])
        traj = trajs[0].copy()
        traj.load_CV(feat)
        self.assertTrue(
            all(
                [
                    "MET" in i and "1" in i
                    for i in traj._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"].values[
                        :3
                    ]
                ]
            )
        )
        trajs.load_CVs("all", ensemble=True)
        chi5_angle_of_resid_11 = trajs._CVs.side_dihedrals.sel(
            SIDE_DIHEDRALS="SIDECHDIH CHI5  11"
        )
        self.assertTrue(
            np.any(np.isnan(chi5_angle_of_resid_11)),
            msg=f"There is no nan in {chi5_angle_of_resid_11.values=}",
        )
        self.assertEqual(
            [
                "SIDECHDIH CHI1   1",
                "SIDECHDIH CHI2   1",
                "SIDECHDIH CHI3   1",
                "SIDECHDIH CHI1   2",
                "SIDECHDIH CHI2   2",
                "SIDECHDIH CHI3   2",
            ],
            trajs._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"].values[:6].tolist(),
        )

        self.assertFalse(
            (trajs._CVs.side_dihedrals.values >= 4).any(),
            msg=(
                f"There seems to be side dihedral angles in deg in the features "
                f"and not in rad, as expected: "
                f"{(trajs._CVs.side_dihedrals.values < 4).all()=}"
                f"{np.where(trajs._CVs.side_dihedrals.values >= 4)=}"
                f"{trajs._CVs.side_dihedrals.values=}"
            ),
        )
        self.assertFalse(
            (trajs._CVs.central_angles.values >= 4).any(),
            msg=(
                f"There seems to be central angles in deg in the features and "
                f"not in rad, as expected: "
                f"{trajs._CVs.central_angles.values[trajs._CVs.central_angles.values >= 4]=}"
            ),
        )
        self.assertFalse(
            (trajs._CVs.central_dihedrals.values >= 4).any(),
            msg=(
                f"There seems to be central dihedral angles in deg in the "
                f"features and not in rad, as expected: "
                f"{trajs._CVs.central_dihedrals.values[trajs._CVs.central_dihedrals.values >= 4]=}"
            ),
        )
        for i in trajs.traj_nums:
            fake_central_dih_rad = np.random.uniform(
                low=-np.pi, high=np.pi, size=(5, trajs._CVs.central_dihedrals.shape[-1])
            )
            fake_side_dih_rad = np.random.uniform(
                low=-np.pi, high=np.pi, size=(5, trajs._CVs.side_dihedrals.shape[-1])
            )

            rad_traj, index = em.misc.backmapping.mdtraj_backmapping(
                top=i,
                dihedrals=fake_central_dih_rad,
                sidechain_dihedrals=fake_side_dih_rad,
                trajs=trajs,
                verify_every_rotation=True,
                angle_type="radian",
                return_indices=True,
            )
            rad_traj = em.SingleTraj(rad_traj, custom_top=custom_aas)
            rad_traj.load_CV(["central_dihedrals", "side_dihedrals"])
            central_index = np.in1d(
                trajs._CVs.central_dihedrals.coords["CENTRAL_DIHEDRALS"],
                index["generic_dihedrals_labels"],
            )
            self.assertAllClose(
                rad_traj._CVs.central_dihedrals.sel(
                    CENTRAL_DIHEDRALS=index["dihedrals_labels"]
                ).values[0],
                fake_central_dih_rad[:, central_index],
                atol=1e-3,
            )
            side_index = np.in1d(
                trajs._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"],
                index["generic_side_dihedrals_labels"],
            )
            self.assertAllClose(
                rad_traj._CVs.side_dihedrals.sel(
                    SIDE_DIHEDRALS=index["side_dihedrals_labels"]
                ).values[0],
                fake_side_dih_rad[:, side_index],
                atol=1e-3,
            )

    def test_mdtraj_with_given_inputs(self):
        """Take a trajectory, extract angles and apply the angles to the first frame of the traj.
        Check whether mdtraj backmapping works"""
        # Standard Library Imports
        from pathlib import Path

        # Encodermap imports
        from encodermap.misc.backmapping import mdtraj_backmapping

        traj = em.SingleTraj(
            Path(__file__).resolve().parent / "data/K48_diUbi.xtc",
            Path(__file__).resolve().parent / "data/K48_diUbi.gro",
        )
        self.assertHasMember(traj, "indices_chi1")
        self.assertHasAttr(traj, "indices_chi2")
        self.assertHasAttr(traj, "indices_chi3")
        self.assertHasAttr(traj, "indices_chi4")
        self.assertHasAttr(traj, "indices_chi5")
        self.assertHasAttr(traj, "indices_psi")
        self.assertHasAttr(traj, "indices_phi")
        self.assertHasAttr(traj, "indices_omega")

        # check bonds
        self.assertNotIn(
            (759, 1227),
            [(a.index, b.index) for a, b in traj.top.bonds],
            msg=(
                "The bond between GLQ-C (0-based index 758) and LYQ-NQ (0-based "
                "index 1227) should not be present in a traj without a custom topology. "
                "In this test, it is present."
            ),
        )
        self.assertIn(
            (759, 761),
            [(a.index, b.index) for a, b in traj.top.bonds],
            msg=(
                "The bond between GLQ-N (0-based index 759) and MET-C (0-based "
                "index 761) should be present in a traj with *NO* custom topology. "
                "In this test, it is not present."
            ),
        )

        # chi1
        # LYS chi1 angle N-CA-CB-CG
        self.assertIn(
            [1378, 1380, 1381, 1382],
            traj.indices_chi1.tolist(),
            msg=f"{traj.indices_chi1=}",
        )
        # LYQ chi1 angle is also in the top, because it contains standard names
        self.assertIn(
            [1220, 1222, 1223, 1224],
            traj.indices_chi1.tolist(),
            msg=f"{traj.indices_chi1=}",
        )

        # chi2
        # LYS chi2 angle CA-CB-CG-CD
        self.assertIn(
            [1380, 1381, 1382, 1383],
            traj.indices_chi2.tolist(),
            msg=f"{traj.indices_chi2=}",
        )
        # LYQ chi2 angle is also in the top, because it contains standard names
        self.assertIn(
            [1222, 1223, 1224, 1225],
            traj.indices_chi2.tolist(),
            msg=f"{traj.indices_chi2=}",
        )

        # chi3
        # LYS chi3 angle CB-CG-CD-CE
        self.assertIn(
            [1381, 1382, 1383, 1384],
            traj.indices_chi3.tolist(),
            msg=f"{traj.indices_chi3=}",
        )
        # LYQ chi3 angle is not in the top, because it contains CQ
        self.assertNotIn(
            [1223, 1224, 1225, 1226],
            traj.indices_chi3.tolist(),
            msg=f"{traj.indices_chi3=}",
        )

        # chi4
        # LYS chi4 angle CG-CD-CE-NZ
        self.assertIn(
            [1382, 1383, 1384, 1385],
            traj.indices_chi4.tolist(),
            msg=f"{traj.indices_chi4=}",
        )
        # LYQ chi4 angle is not in the top, because it contains CQ and NQ
        self.assertNotIn(
            [1224, 1225, 1226, 1227],
            traj.indices_chi4.tolist(),
            msg=f"{traj.indices_chi4=}",
        )

        # each ubiquitin contains 8 ARG residues
        self.assertEqual((8, 4), traj.indices_chi5.shape)

        # there is a wrong psi and a wrong omega bond in GLQ, that should be present here,
        # but not later when the custom topology has been loaded
        self.assertIn(
            [756, 758, 759, 761], traj.indices_psi.tolist(), msg=f"{traj.indices_psi=}"
        )
        self.assertIn(
            [758, 759, 761, 765],
            traj.indices_omega.tolist(),
            msg=f"{traj.indices_omega=}",
        )
        self.assertIn(
            [759, 761, 765, 770], traj.indices_phi.tolist(), msg=f"{traj.indices_phi=}"
        )

        # fmt: off
        # fmt: on
        self.assertHasAttr(traj, "_custom_top")
        self.assertHasAttr(traj, "load_custom_topology")

        # debug the CustomTopology because AttributeErrors are propagated weirdly
        # Encodermap imports
        from encodermap.trajinfo.trajinfo_utils import CustomTopology

        _top = CustomTopology.from_dict(self.custom_aas_K48_diUbi, deepcopy(traj))
        self.assertIn(
            (759, 1227),
            [(a.index, b.index) for a, b in _top.top.bonds],
            msg=(
                "The bond between GLQ-C (0-based index 758) and LYQ-NQ (0-based "
                "index 1227) should be present in this custom topology. "
                "In this test, it is not present."
            ),
        )

        traj.load_custom_topology(self.custom_aas_K48_diUbi)
        self.assertTrue(hasattr(traj._custom_top, "to_dict"))
        self.assertIs(
            traj.top,
            traj._custom_top.top,
            msg=(
                f"The traj.top should always point to the CustomTopology.top. This "
                f"seems to not be the case."
            ),
        )
        # Third Party Imports
        from networkx import connected_components

        self.assertEqual(
            len([*connected_components(traj.top.to_bondgraph())]),
            1,
            msg=(
                f"The topolgy {traj.top} is disconnected. There should be more bonds."
            ),
        )
        self.assertIn(
            (759, 1227),
            [(a.index, b.index) for a, b in traj.top.bonds],
            msg=(
                "The bond between GLQ-C (0-based index 758) and LYQ-NQ (0-based "
                "index 1227) should be present in a traj with a custom topology. "
                "In this test, it is not present. It was present when the CustomTopology "
                "object was used by itself a few lines above. This fail must be "
                "caused by caching."
            ),
        )
        self.assertNotIn(
            (759, 761),
            [(a.index, b.index) for a, b in traj.top.bonds],
            msg=(
                "The bond between GLQ-N (0-based index 759) and MET-C (0-based "
                "index 761) should be NOT present in a traj with a custom topology. "
                "In this test, it is present."
            ),
        )
        # chi1
        # LYS chi1 angle N-CA-CB-CG
        self.assertIn(
            [1378, 1380, 1381, 1382],
            traj.indices_chi1.tolist(),
            msg=f"{traj.indices_chi1=}",
        )
        # LYQ chi1 angle is also in the top, because it contains standard names
        self.assertIn(
            [1220, 1222, 1223, 1224],
            traj.indices_chi1.tolist(),
            msg=f"{traj.indices_chi1=}",
        )

        # chi2
        # LYS chi2 angle CA-CB-CG-CD
        self.assertIn(
            [1380, 1381, 1382, 1383],
            traj.indices_chi2.tolist(),
            msg=f"{traj.indices_chi2=}",
        )
        # LYQ chi2 angle is also in the top, because it contains standard names
        self.assertIn(
            [1222, 1223, 1224, 1225],
            traj.indices_chi2.tolist(),
            msg=f"{traj.indices_chi2=}",
        )

        # chi3
        # LYS chi3 angle CB-CG-CD-CE
        self.assertIn(
            [1381, 1382, 1383, 1384],
            traj.indices_chi3.tolist(),
            msg=f"{traj.indices_chi3=}",
        )
        # LYQ chi3 angle is not in the top, because it contains CQ
        self.assertIn(
            [1223, 1224, 1225, 1226],
            traj.indices_chi3.tolist(),
            msg=f"{traj.indices_chi3=}",
        )

        # chi4
        # LYS chi4 angle CG-CD-CE-NZ
        self.assertIn(
            [1382, 1383, 1384, 1385],
            traj.indices_chi4.tolist(),
            msg=f"{traj.indices_chi4=}",
        )
        # LYQ chi4 angle is not in the top, because it contains CQ and NQ
        self.assertIn(
            [1224, 1225, 1226, 1227],
            traj.indices_chi4.tolist(),
            msg=f"{traj.indices_chi4=}",
        )

        # each ubiquitin contains 8 ARG residues
        self.assertEqual((8, 4), traj.indices_chi5.shape)

        # there is a wrong psi and a wrong omega bond in GLQ, that should be present here,
        # but not later when the custom topology has been loaded
        self.assertHasAttr(traj, "indices_chi1")
        self.assertHasAttr(traj, "indices_chi2")
        self.assertHasAttr(traj, "indices_chi3")
        self.assertHasAttr(traj, "indices_chi4")
        self.assertHasAttr(traj, "indices_chi5")
        self.assertHasAttr(traj._custom_top, "indices_phi")
        self.assertHasAttr(traj._custom_top, "indices_psi")
        self.assertHasAttr(traj._custom_top, "indices_omega")
        self.assertHasMember(traj, "indices_phi")
        self.assertHasAttr(traj, "indices_phi")
        self.assertHasAttr(traj, "indices_psi")
        self.assertHasAttr(traj, "indices_omega")
        self.assertNotIn(
            [756, 758, 759, 761], traj.indices_psi.tolist(), msg=f"{traj.indices_psi=}"
        )
        self.assertNotIn(
            [758, 759, 761, 765],
            traj.indices_omega.tolist(),
            msg=f"{traj.indices_omega=}",
        )
        self.assertNotIn(
            [759, 761, 765, 770], traj.indices_phi.tolist(), msg=f"{traj.indices_phi=}"
        )

        # make sure the GLQ bond is present, otherwise it will be caught by
        # the backmapping
        self.assertIn([756, 757], [[a.index, b.index] for a, b in traj.top.bonds])

        traj.load_CV("all")
        dih_indices = traj._CVs.central_dihedrals_feature_indices.values[0]
        self.assertNotIn([759, 761], dih_indices[:, 1:3].tolist())
        central_dih_shape = traj.central_dihedrals.shape[1]
        side_dih_shape = traj.side_dihedrals.shape[1]
        fake_central_dih_rad = np.random.uniform(
            low=-np.pi, high=np.pi, size=(10, central_dih_shape)
        )
        fake_side_dih_rad = np.random.uniform(
            low=-np.pi, high=np.pi, size=(10, side_dih_shape)
        )
        fake_central_dih_deg = np.random.uniform(
            low=-180, high=180, size=(10, central_dih_shape)
        )
        fake_side_dih_deg = np.random.uniform(
            low=-180, high=180, size=(10, side_dih_shape)
        )

        with self.assertRaises(Exception):
            deg_traj = mdtraj_backmapping(
                None, fake_central_dih_deg, fake_side_dih_deg, traj, angle_type="radian"
            )
        with self.assertRaises(Exception):
            rad_traj = mdtraj_backmapping(
                None, fake_central_dih_rad, fake_side_dih_rad, traj, angle_type="degree"
            )
        with self.assertRaises(Exception):
            deg_traj = mdtraj_backmapping(
                None,
                fake_central_dih_deg,
                np.random.uniform(low=-180, high=180, size=(50, side_dih_shape)),
                traj,
                angle_type="degree",
            )

        # assert that traj is not disconnected
        # Third Party Imports
        from networkx import connected_components

        self.assertEqual(
            len([*connected_components(traj.top.to_bondgraph())]),
            1,
            msg=(f"The topolgy became disconnected."),
        )

        deg_traj, index = mdtraj_backmapping(
            top=None,
            dihedrals=fake_central_dih_deg,
            sidechain_dihedrals=fake_side_dih_deg,
            trajs=traj,
            verify_every_rotation=True,
            angle_type="degree",
            return_indices=True,
        )
        deg_traj = em.SingleTraj(deg_traj, custom_top=self.custom_aas_K48_diUbi)
        deg_traj.load_CV("all", deg=True)
        all_dihedrals = deg_traj._CVs.central_dihedrals.coords["CENTRAL_DIHEDRALS"]
        central_index = np.in1d(all_dihedrals, index["dihedrals_labels"])
        all_side_dihedrals = deg_traj._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"]
        side_index = np.in1d(all_side_dihedrals, index["side_dihedrals_labels"])
        self.assertAllClose(
            deg_traj.central_dihedrals[:, central_index],
            fake_central_dih_deg[:, central_index],
            atol=5e-2,
        )
        self.assertAllClose(
            deg_traj.side_dihedrals[:, side_index],
            fake_side_dih_deg[:, side_index],
            atol=5e-2,
        )

        rad_traj, index = mdtraj_backmapping(
            top=None,
            dihedrals=fake_central_dih_rad,
            sidechain_dihedrals=fake_side_dih_rad,
            trajs=traj,
            verify_every_rotation=True,
            angle_type="radian",
            return_indices=True,
        )
        rad_traj = em.SingleTraj(rad_traj, custom_top=self.custom_aas_K48_diUbi)
        rad_traj.load_CV("all")
        all_dihedrals = rad_traj._CVs.central_dihedrals.coords["CENTRAL_DIHEDRALS"]
        central_index = np.in1d(all_dihedrals, index["dihedrals_labels"])
        all_side_dihedrals = rad_traj._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"]
        side_index = np.in1d(all_side_dihedrals, index["side_dihedrals_labels"])
        self.assertAllClose(
            rad_traj.central_dihedrals[:, central_index],
            fake_central_dih_rad[:, central_index],
            atol=1e-3,
        )
        self.assertAllClose(
            rad_traj.side_dihedrals[:, side_index],
            fake_side_dih_rad[:, side_index],
            atol=1e-3,
        )

    @expensive_test
    def test_backmapping_cases(self):
        """Test multiple different cases of backmapping.

        Cases:
            * sidechain: yes/no
            * omega: yes/no
            * radians/degrees
            * provide top as: int, str, md.Topology
            * TrajEnsemble with 1 traj, with multiple trajs
            * differences only in sidechain dihedrals, or in sidechains and centrals

        """
        # Encodermap imports
        from encodermap.kondata import get_from_kondata
        from encodermap.misc.backmapping import mdtraj_backmapping

        output_dir_OTU11 = Path("/home/kevin/git/encoder_map_private/tests/data/OTU11")
        get_from_kondata(
            "OTU11",
            output_dir_OTU11,
            mk_parentdir=True,
            silence_overwrite_message=True,
        )
        output_dir_pASP_pGLU = Path(
            "/home/kevin/git/encoder_map_private/tests/data/pASP_pGLU"
        )
        get_from_kondata(
            "pASP_pGLU",
            output_dir_pASP_pGLU,
            mk_parentdir=True,
            silence_overwrite_message=True,
        )

        custom_aas = {
            "CLA": None,
            "SOD": None,
            "POPC": None,
            "POPE": None,
            "SAPI": None,
            "THR": (
                "T",
                {
                    "optional_bonds": [
                        ("-C", "N"),  # the peptide bond to the previous aa
                        ("N", "CA"),
                        ("N", "H"),
                        ("CA", "HA"),
                        ("CB", "HB"),
                        ("CB", "OG1"),
                        ("OG1", "P"),
                        ("P", "O1P"),
                        ("P", "O2P"),
                        ("P", "OXT"),
                        ("OXT", "HT"),
                        ("CB", "CG2"),
                        ("CG2", "HG21"),
                        ("CG2", "HG22"),
                        ("CG2", "HG23"),
                        ("CA", "C"),
                        ("C", "O"),
                        ("C", "+N"),  # the peptide bond to the next aa
                    ],
                    "optional_delete_bonds": [
                        ("OXT", "C"),
                    ],
                    "CHI2": ["CA", "CB", "OG1", "P"],
                    "CHI3": ["CB", "OG1", "P", "OXT"],
                },
            ),
            "SER": (
                "S",
                {
                    "optional_bonds": [
                        ("-C", "N"),  # the peptide bond to the previous aa
                        ("N", "CA"),
                        ("N", "H"),
                        ("CA", "HA"),
                        ("CB", "HB1"),
                        ("CB", "HB2"),
                        ("CB", "OG"),
                        ("OG", "P"),
                        ("P", "O1P"),
                        ("P", "O2P"),
                        ("P", "OXT"),
                        ("OXT", "HT"),
                        ("CA", "C"),
                        ("C", "O"),
                        ("C", "+N"),  # the peptide bond to the next aa
                    ],
                    "optional_delete_bonds": [
                        ("OXT", "C"),
                    ],
                    "CHI2": ["CA", "CB", "OG", "P"],
                    "CHI3": ["CB", "OG", "P", "OXT"],
                },
            ),
        }

        asp_glu_custom_aas = {
            "ASP": (
                "A",
                {
                    "optional_bonds": [
                        ("N", "H1"),
                        ("N", "H2"),
                        ("N", "H"),
                        ("N", "CA"),
                        ("CA", "CB"),
                        ("CB", "CG"),
                        ("CG", "OD1"),
                        ("CG", "OD2"),
                        ("OD2", "HD2"),
                        ("CA", "C"),
                        ("C", "O"),
                        ("C", "OT"),
                        ("O", "HO"),
                        ("C", "+N"),
                    ],
                },
            ),
            "GLU": (
                "E",
                {
                    "optional_bonds": [
                        ("N", "H1"),
                        ("N", "H2"),
                        ("N", "H"),
                        ("N", "CA"),
                        ("CA", "CB"),
                        ("CB", "CG"),
                        ("CG", "CD"),
                        ("CD", "OE1"),
                        ("CD", "OE2"),
                        ("OE2", "HE2"),
                        ("CA", "C"),
                        ("C", "O"),
                        ("C", "OT"),
                        ("O", "HO"),
                        ("C", "+N"),
                    ],
                },
            ),
        }

        single_traj_OTU11 = em.load(
            output_dir_OTU11 / "OTU11_dead_only_prot.xtc",
            output_dir_OTU11 / "OTU11_dead_only_prot.pdb",
        )

        all_trajs_OTU11 = em.load(
            [
                output_dir_OTU11 / "OTU11_dead_only_prot.xtc",
                output_dir_OTU11 / "OTU11_mock_only_prot.xtc",
                output_dir_OTU11 / "OTU11_phospho_only_prot.xtc",
                output_dir_OTU11 / "OTU11_wt_only_prot.xtc",
            ],
            [
                output_dir_OTU11 / "OTU11_dead_only_prot.pdb",
                output_dir_OTU11 / "OTU11_mock_only_prot.pdb",
                output_dir_OTU11 / "OTU11_phospho_only_prot.pdb",
                output_dir_OTU11 / "OTU11_wt_only_prot.pdb",
            ],
        )

        single_traj_pASP = em.load(
            output_dir_pASP_pGLU / "asp10.xtc",
            output_dir_pASP_pGLU / "asp10.pdb",
        )

        all_trajs_pASP = em.load(
            [
                output_dir_pASP_pGLU / "asp7.xtc",
                output_dir_pASP_pGLU / "glu7.xtc",
                output_dir_pASP_pGLU / "glu8.xtc",
                output_dir_pASP_pGLU / "asp10.xtc",
            ],
            [
                output_dir_pASP_pGLU / "asp7.pdb",
                output_dir_pASP_pGLU / "glu7.pdb",
                output_dir_pASP_pGLU / "glu8.pdb",
                output_dir_pASP_pGLU / "asp10.pdb",
            ],
        )

        # make sure OTU11 has 4 topologies

        self.assertEqual(len(all_trajs_OTU11.trajs_by_top), 4)
        single_traj_OTU11.load_custom_topology(custom_aas)
        all_trajs_OTU11.load_custom_topology(custom_aas)
        single_traj_pASP.load_custom_topology(asp_glu_custom_aas)
        all_trajs_pASP.load_custom_topology(asp_glu_custom_aas)
        single_traj_OTU11.load_CV("all")
        all_trajs_OTU11.load_CVs("all", ensemble=True)
        single_traj_pASP.load_CV("all")
        all_trajs_pASP.load_CVs("all", ensemble=True)
        self.assertTrue(
            np.any(np.isnan(all_trajs_pASP[0]._CVs.central_dihedrals_feature_indices)),
            msg=(f"{all_trajs_pASP[0]._CVs.central_dihedrals_feature_indices=}"),
        )

        # first, make sure that we have some Nan features in the ds
        ds = all_trajs_pASP._CVs
        stacked_ds = (
            ds.stack({"frame": ("traj_num", "frame_num")})
            .transpose("frame", ...)
            .dropna("frame", "all")
        )

        # some side dihedrals have to be nan
        self.assertFalse(
            np.all(np.all(~np.isnan(stacked_ds.central_dihedrals.values), axis=1)),
            msg="Might be because of the addition of xr.Dataset.transpose('frame', ...).",
        )
        # some side dihedrals have to be nan
        self.assertFalse(
            np.all(np.all(~np.isnan(stacked_ds.side_dihedrals.values), axis=1)),
            msg="Might be because of the addition of xr.Dataset.transpose('frame', ...).",
        )
        self.assertEqual(
            single_traj_pASP._CVs.coords["CENTRAL_DIHEDRALS"].shape,
            all_trajs_pASP._CVs.coords["CENTRAL_DIHEDRALS"].shape,
        )

        ds = all_trajs_OTU11._CVs
        stacked_ds = (
            ds.stack({"frame": ("traj_num", "frame_num")})
            .transpose("frame", ...)
            .dropna("frame", "all")
        )
        # all central dihedrals in OTU11 are defined
        self.assertTrue(
            np.all(np.all(~np.isnan(stacked_ds.central_dihedrals.values), axis=1)),
            msg="Might be because of the addition of xr.Dataset.transpose('frame', ...).",
        )
        # some side dihedrals have to be nan
        self.assertFalse(
            np.all(np.all(~np.isnan(stacked_ds.side_dihedrals.values), axis=1)),
            msg="Might be because of the addition of xr.Dataset.transpose('frame', ...).",
        )

        for trajs, c_aas in zip(
            [all_trajs_pASP, all_trajs_OTU11], [asp_glu_custom_aas, custom_aas]
        ):
            for angle_type in ["radian", "degree"]:
                if angle_type == "radian":
                    inp_dihedrals = (
                        np.random.random((3, trajs._CVs.central_dihedrals.shape[-1]))
                        * 2
                        * np.pi
                        - np.pi
                    )
                    inp_side_dihedrals = (
                        np.random.random((3, trajs._CVs.side_dihedrals.shape[-1]))
                        * 2
                        * np.pi
                        - np.pi
                    )
                else:
                    inp_dihedrals = (
                        np.random.random((3, trajs._CVs.central_dihedrals.shape[-1]))
                        * 360
                        - 180
                    )
                    inp_side_dihedrals = (
                        np.random.random((3, trajs._CVs.side_dihedrals.shape[-1])) * 360
                        - 180
                    )
                for omega in [False, True]:
                    # vary ints
                    for i in range(trajs.n_trajs):
                        print(
                            f"Doing backmapping for {trajs.basenames=}, "
                            f"{angle_type=}, {omega=}, {i=} {trajs[i].basename=}"
                        )
                        # mdtraj
                        traj, indices = mdtraj_backmapping(
                            top=i,
                            dihedrals=inp_dihedrals,
                            sidechain_dihedrals=inp_side_dihedrals,
                            trajs=trajs,
                            verify_every_rotation=True,
                            angle_type=angle_type,
                            omega=omega,
                            return_indices=True,
                            parallel=False,
                        )

                        # assert some stuff about the output
                        self.assertIsInstance(traj, md.Trajectory)
                        self.assertEqual(traj.n_frames, 3)
                        traj = em.SingleTraj(traj)
                        traj.load_CV(["central_dihedrals", "side_dihedrals"])
                        if not omega:
                            self.assertFalse(
                                any(
                                    trajs._CVs.central_dihedrals.sel(
                                        CENTRAL_DIHEDRALS=indices[
                                            "generic_dihedrals_labels"
                                        ]
                                    )
                                    .coords["CENTRAL_DIHEDRALS"]
                                    .str.contains("OMEGA")
                                ),
                            )
                        else:
                            self.assertTrue(
                                any(
                                    trajs._CVs.central_dihedrals.sel(
                                        CENTRAL_DIHEDRALS=indices[
                                            "generic_dihedrals_labels"
                                        ]
                                    )
                                    .coords["CENTRAL_DIHEDRALS"]
                                    .str.contains("OMEGA")
                                ),
                            )
                        central_index = np.in1d(
                            trajs._CVs.central_dihedrals.coords["CENTRAL_DIHEDRALS"],
                            indices["generic_dihedrals_labels"],
                        )
                        angles = traj._CVs.central_dihedrals.sel(
                            CENTRAL_DIHEDRALS=indices["dihedrals_labels"]
                        ).values[0]
                        if angle_type == "degree":
                            angles = np.rad2deg(angles)
                        self.assertAllClose(
                            angles,
                            inp_dihedrals[:, central_index],
                            atol=1e-3,
                        )
                        side_index = np.in1d(
                            trajs._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"],
                            indices["generic_side_dihedrals_labels"],
                        )
                        angles = traj._CVs.side_dihedrals.sel(
                            SIDE_DIHEDRALS=indices["side_dihedrals_labels"]
                        ).values[0]
                        if angle_type == "degree":
                            angles = np.rad2deg(angles)
                        self.assertAllClose(
                            angles,
                            inp_side_dihedrals[:, side_index],
                            atol=1e-3,
                        )

        for traj, c_aas in zip(
            [single_traj_pASP, single_traj_OTU11], [asp_glu_custom_aas, custom_aas]
        ):
            for angle_type in ["radian", "degree"]:
                if angle_type == "radian":
                    inp_dihedrals = (
                        np.random.random((3, traj._CVs.central_dihedrals.shape[-1]))
                        * 2
                        * np.pi
                        - np.pi
                    )
                    inp_side_dihedrals = (
                        np.random.random((3, traj._CVs.side_dihedrals.shape[-1]))
                        * 2
                        * np.pi
                        - np.pi
                    )
                else:
                    inp_dihedrals = (
                        np.random.random((3, traj._CVs.central_dihedrals.shape[-1]))
                        * 360
                        - 180
                    )
                    inp_side_dihedrals = (
                        np.random.random((3, traj._CVs.side_dihedrals.shape[-1])) * 360
                        - 180
                    )
                for omega in [False, True]:
                    print(
                        f"Doing backmapping for {traj.basename=}, {angle_type=}, {omega=}"
                    )
                    # mdtraj
                    traj_out, indices = mdtraj_backmapping(
                        top=i,
                        dihedrals=inp_dihedrals,
                        sidechain_dihedrals=inp_side_dihedrals,
                        trajs=traj,
                        verify_every_rotation=True,
                        angle_type=angle_type,
                        omega=omega,
                        return_indices=True,
                        parallel=False,
                    )

                    # assert some stuff about the output
                    self.assertIsInstance(traj_out, md.Trajectory)
                    self.assertEqual(traj_out.n_frames, 3)
                    traj_out = em.SingleTraj(traj_out)
                    traj_out.load_CV(["central_dihedrals", "side_dihedrals"])
                    if not omega:
                        self.assertFalse(
                            any(
                                trajs._CVs.central_dihedrals.sel(
                                    CENTRAL_DIHEDRALS=indices[
                                        "generic_dihedrals_labels"
                                    ]
                                )
                                .coords["CENTRAL_DIHEDRALS"]
                                .str.contains("OMEGA")
                            ),
                        )
                    else:
                        self.assertTrue(
                            any(
                                trajs._CVs.central_dihedrals.sel(
                                    CENTRAL_DIHEDRALS=indices[
                                        "generic_dihedrals_labels"
                                    ]
                                )
                                .coords["CENTRAL_DIHEDRALS"]
                                .str.contains("OMEGA")
                            ),
                        )
                    central_index = np.in1d(
                        trajs._CVs.central_dihedrals.coords["CENTRAL_DIHEDRALS"],
                        indices["generic_dihedrals_labels"],
                    )
                    angles = traj_out._CVs.central_dihedrals.sel(
                        CENTRAL_DIHEDRALS=indices["dihedrals_labels"]
                    ).values[0]
                    if angle_type == "degree":
                        angles = np.rad2deg(angles)
                    self.assertAllClose(
                        angles,
                        inp_dihedrals[:, central_index],
                        atol=1e-3,
                    )
                    side_index = np.in1d(
                        trajs._CVs.side_dihedrals.coords["SIDE_DIHEDRALS"],
                        indices["generic_side_dihedrals_labels"],
                    )
                    angles = traj_out._CVs.side_dihedrals.sel(
                        SIDE_DIHEDRALS=indices["side_dihedrals_labels"]
                    ).values[0]
                    if angle_type == "degree":
                        angles = np.rad2deg(angles)
                    self.assertAllClose(
                        angles,
                        inp_side_dihedrals[:, side_index],
                        atol=1e-3,
                    )


def test_backmapping_dihedral(self):
    # Standard Library Imports
    from pathlib import Path

    # Third Party Imports
    import mdtraj as md

    traj = md.load(str(Path(__file__).resolve().parent / "data/known_angles.h5"))
    # Encodermap imports
    from encodermap.misc.backmapping import _dihedral

    for frame in traj:
        dih = _dihedral(frame.xyz[0], np.array([0, 1, 2, 3]))
        self.assertTrue(-np.pi < dih <= np.pi, msg=f"Dihedral not in radian: {dih}.")


class TestCompareSplits(tf.test.TestCase):
    def test_random_shapes(self):
        for i in np.random.randint(0, 1000, size=10):
            # create example tensors
            example_dihedrals = (
                tf.convert_to_tensor(np.random.random((256, i - 3))) + np.pi
            )
            example_cartesians = tf.convert_to_tensor(np.random.random((256, i, 3)))

            # print('example_dihedrals.shape:', example_dihedrals.shape)
            # print('example_cartesians.shape:', example_cartesians.shape)

            if len(example_cartesians.get_shape()) == 2:
                expanded = tf.expand_dims(example_cartesians, axis=0)
                example_cartesians = tf.tile(expanded, [256, 1, 1])

            split = int(int(example_cartesians.shape[1]) / 2)

            # old way:
            # dihedrals need to be an even split
            # the middle element of cartesians needs to be repeated, because it belongs to two dihedrals
            cartesian_left = example_cartesians[:, split + 1 :: -1]
            dihedrals_left = example_dihedrals[:, split - 2 :: -1]
            cartesian_right = example_cartesians[:, split - 1 :]
            dihedrals_right = example_dihedrals[:, split - 1 :]

            # print(cartesian_left.shape, dihedrals_left.shape)
            # print(cartesian_right.shape, dihedrals_right.shape)

            # new way
            dihedrals_left_test, dihedrals_right_test = split_and_reverse_dihedrals(
                example_dihedrals
            )
            cartesians_left_test, cartesians_right_test = split_and_reverse_cartesians(
                example_cartesians
            )
            # print(cartesians_left_test.shape, dihedrals_left_test.shape)
            # print(cartesians_right_test.shape, dihedrals_right_test.shape)

            self.assertAllEqual(dihedrals_left_test, dihedrals_left)
            self.assertAllEqual(dihedrals_right_test, dihedrals_right)
            self.assertAllEqual(cartesians_left_test, cartesian_left)
            self.assertAllEqual(cartesians_right_test, cartesian_right)


# print('generated_angles:', generated_angles.shape)
# print('generated_dihedrals:', generated_dihedrals.shape)
# mean_lengths = np.expand_dims(np.mean(distance_data, 0), 0)
# print('mean_lengths:', mean_lengths.shape)
# # build s chain in plane with lengths and angles
# _chain_in_plane = chain_in_plane(mean_lengths, generated_angles)
# print('chain_in_plane:', _chain_in_plane.shape)
# # add a third dimension by adding torsion angles to that chain
# cartesians = dihedrals_to_cartesian_tf(generated_dihedrals + np.pi, _chain_in_plane)
# print('cartesians:', cartesians.shape)
# atom_names = ['N', 'CA', 'C'] * int(no_central_cartesians / 3)
# amide_H_cartesians_tf1 = guess_amide_H_tf1(cartesians, atom_names)
# print('amide_H_cartesians_tf1:', amide_H_cartesians_tf1.shape, amide_H_cartesians_tf1[0, :5, 0])
# amide_H_cartesians = guess_amide_H(cartesians, np.arange(cartesians.shape[1])[::3])
# print('amide_H_cartesians:', amide_H_cartesians.shape, amide_H_cartesians[0, :5, 0])
# amide_O_cartesians_tf1 = guess_amide_O_tf1(cartesians, atom_names)
# print('amide_O_cartesians_tf1:', amide_O_cartesians_tf1.shape, amide_O_cartesians_tf1[0, :5, 0])
# amide_O_cartesians = guess_amide_O(cartesians, np.arange(cartesians.shape[1])[2::3])
# print('amide_O_cartesians:', amide_O_cartesians.shape, amide_O_cartesians[0, :5, 0])
# merged_cartesians_tf1 = merge_cartesians_tf1(cartesians, atom_names, amide_H_cartesians_tf1, amide_O_cartesians_tf1)
# print('merged_cartesians_tf1:', merged_cartesians_tf1.shape, merged_cartesians_tf1[0, :5, 0])
# merged_cartesians = merge_cartesians(cartesians, np.arange(cartesians.shape[1])[::3], np.arange(cartesians.shape[1])[2::3],
#                                      amide_H_cartesians, amide_O_cartesians)
# print('merged_cartesians:', merged_cartesians.shape, merged_cartesians[0, :5, 0])
# inp_pairwise = pairwise_dist(cartesian_data[:10, p.cartesian_pwd_start:
#                              p.cartesian_pwd_stop:
#                              p.cartesian_pwd_step], flat=True)
# print('inp_pairwise:', inp_pairwise.shape)
# encoded_pairwise = pairwise_dist(cartesians[:10,  p.cartesian_pwd_start:
#                                  p.cartesian_pwd_stop:
#                                  p.cartesian_pwd_step], flat=True)
# print('encoded_pairwise:', encoded_pairwise.shape)
# clashes = tf.math.count_nonzero(pairwise_dist(cartesians, flat=True) < 1, axis=1, dtype=tf.float32)
# print('clashes:', clashes.shape)

# Remove Phantom Tests from tensorflow skipped test_session
# https://stackoverflow.com/questions/55417214/phantom-tests-after-switching-from-unittest-testcase-to-tf-test-testcase
test_cases = (
    TestBackmappingEm1Em2,
    TestBackmappingMdtrajMdanalysis,
    TestCompareSplits,
)

################################################################################
# Doctests
################################################################################


# Standard Library Imports
import doctest

# Encodermap imports
import encodermap.misc.backmapping as backmapping


doc_tests = (doctest.DocTestSuite(backmapping),)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_class in test_cases:
        tests = loader.loadTestsFromTestCase(test_class)
        filtered_tests = [t for t in tests if not t.id().endswith(".test_session")]
        suite.addTests(filtered_tests)
    suite.addTests(doc_tests)
    return suite


################################################################################
# Main
################################################################################


if __name__ == "__main__":
    unittest.main()

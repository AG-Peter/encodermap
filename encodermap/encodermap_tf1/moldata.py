import os
import warnings
from collections import OrderedDict
from math import pi

import MDAnalysis as md
import numpy as np
from MDAnalysis.analysis.align import AlignTraj
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.lib.distances import calc_angles
from tqdm import tqdm

from .misc import create_dir


class Positions(AnalysisBase):
    def __init__(self, atomgroup, **kwargs):
        super(Positions, self).__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup

    def _prepare(self):
        self.result = []

    def _single_frame(self):
        self.result.append(self._ag.positions)

    def _conclude(self):
        self.result = np.asarray(self.result)


class Angles(AnalysisBase):
    def __init__(self, atomgroups, **kwargs):
        super(Angles, self).__init__(atomgroups[0].universe.trajectory, **kwargs)
        self.atomgroups = atomgroups

        if any([len(ag) != 3 for ag in atomgroups]):
            raise ValueError("All AtomGroups must contain 3 atoms")

        self.ag1 = md.AtomGroup([ag[0] for ag in atomgroups])
        self.ag2 = md.AtomGroup([ag[1] for ag in atomgroups])
        self.ag3 = md.AtomGroup([ag[2] for ag in atomgroups])

    def _prepare(self):
        self.result = []

    def _single_frame(self):
        angle = calc_angles(
            self.ag1.positions,
            self.ag2.positions,
            self.ag3.positions,
            box=self.ag1.dimensions,
        )
        self.result.append(angle)

    def _conclude(self):
        self.result = np.asarray(self.result)


class MolData:
    """
    MolData is designed to extract and hold conformational information from trajectories.

    :ivar cartesians: numpy array of the trajectory atom coordinates
    :ivar central_cartesians: cartesian coordinates of the central backbone atoms (N-CA-C-N-CA-C...)
    :ivar dihedrals: all backbone dihederals (phi, psi, omega)
    :ivar angles: all bond angles of the central backbone atoms
    :ivar lengths: all bond lengths between neighbouring central atoms
    :ivar sidedihedrals: all sidechain dihedrals
    :ivar aminoaciddict: number of sidechain diheadrals
    """

    def __init__(
        self,
        atom_group,
        cache_path="",
        start=None,
        stop=None,
        step=None,
    ):
        """
        :param atom_group: MDAnalysis atom group
        :param cache_path: Allows to define a path where the calculated variables can be cached.
        :param start: first frame to analyze
        :param stop: last frame to analyze
        :param step: step of the analyzes
        """
        self.universe = atom_group.universe

        self.sorted_atoms = self.universe.atoms[
            [atom.ix for atom in sorted(atom_group.atoms, key=self.sort_key)]
        ]

        self.central_atom_indices = [
            i
            for i, atom in enumerate(self.sorted_atoms)
            if atom.name in ["N", "CA", "C"]
        ]
        self.central_atoms = self.sorted_atoms[self.central_atom_indices]

        ######## Problems with ILE and PRO TRP

        self.aminoaciddict = {
            "ALA": 0,
            "ARG": 5,
            "ASN": 2,
            "ASP": 2,
            "CYS": 1,
            "GLU": 3,
            "GLN": 3,
            "GLY": 0,
            "HIS": 2,
            "HID": 2,
            "ILE": 1,
            "LEU": 2,
            "LYS": 4,
            "MET": 3,
            "PHE": 2,
            "PRO": 0,
            "SER": 1,
            "THR": 1,
            "TRP": 2,
            "TYR": 2,
            "VAL": 1,
            "KAC": 4,
        }

        # Cartesians:
        try:
            self.cartesians = np.load(os.path.join(cache_path, "cartesians.npy"))
            print("Loaded cartesians from {}".format(cache_path))

        except FileNotFoundError:
            print("Loading Cartesians...")
            positions = Positions(self.sorted_atoms, verbose=True).run(
                start=start, stop=stop, step=step
            )
            self.cartesians = positions.result.astype(np.float32)

            if cache_path:
                np.save(
                    os.path.join(create_dir(cache_path), "cartesians.npy"),
                    self.cartesians,
                )

        self.central_cartesians = self.cartesians[:, self.central_atom_indices]

        # Dihedrals:
        try:
            self.dihedrals = np.load(os.path.join(cache_path, "dihedrals.npy"))
            print("Loaded dihedrals from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating dihedrals...")
            dihedral_atoms = []
            for i in OrderedDict.fromkeys(self.sorted_atoms.resnums):
                phi_atoms = self.sorted_atoms.select_atoms(
                    "resnum {} and name C".format(i - 1)
                ) + self.sorted_atoms.select_atoms(
                    "resnum {} and (name N or name CA or name C)".format(i)
                )
                if len(phi_atoms) == 4:
                    dihedral_atoms.append(phi_atoms.dihedral)

                psi_atoms = self.sorted_atoms.select_atoms(
                    "resnum {} and (name N or name CA or name C)".format(i)
                ) + self.sorted_atoms.select_atoms("resnum {} and name N".format(i + 1))
                if len(psi_atoms) == 4:
                    dihedral_atoms.append(psi_atoms.dihedral)

                omega_atoms = self.sorted_atoms.select_atoms(
                    "resnum {} and (name CA or name C)".format(i)
                ) + self.sorted_atoms.select_atoms(
                    "resnum {} and (name N or name CA)".format(i + 1)
                )
                if len(psi_atoms) == 4:
                    dihedral_atoms.append(omega_atoms.dihedral)

            dihedrals = Dihedral(dihedral_atoms, verbose=True).run(
                start=start, stop=stop, step=step
            )
            self.dihedrals = dihedrals.angles.astype(np.float32)
            self.dihedrals *= pi / 180

            if cache_path:
                np.save(os.path.join(cache_path, "dihedrals.npy"), self.dihedrals)

        # SideDihedrals

        try:
            self.sidedihedrals = np.load(os.path.join(cache_path, "sidedihedrals.npy"))
            print("Loaded dihedrals from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating sidedihedrals...")
            sidedihedral_atoms = []

            for i in OrderedDict.fromkeys(self.sorted_atoms.resnums):
                residue_atoms = self.sorted_atoms.select_atoms("resnum {}".format(i))
                for n in range(
                    self.aminoaciddict[
                        self.universe.select_atoms(
                            "resnum {} and name CA".format(i)
                        ).resnames[0]
                    ]
                ):
                    side_atoms = residue_atoms[n : int(n + 4)]
                    sidedihedral_atoms.append(side_atoms)
            if sidedihedral_atoms == []:
                self.sidedihedrals = np.nan
            else:
                warnings.showwarning(
                    "\033[1;37;40m This version of the MolData Class does not produce expected results for side-dihedrals.",
                    category=UserWarning,
                    filename="",
                    lineno=-1,
                )
                warnings.showwarning(
                    "\033[1;37;40m To make this class work the 'residue_atoms[n:int(n+4)]' needs to be reworked. It does not index the sidechains.",
                    category=UserWarning,
                    filename="",
                    lineno=-1,
                )
                sidedihedrals = Dihedral(sidedihedral_atoms, verbose=True).run(
                    start=start, stop=stop, step=step
                )
                self.sidedihedrals = sidedihedrals.angles.astype(np.float32)
                self.sidedihedrals *= pi / 180

            if cache_path:
                np.save(
                    os.path.join(cache_path, "sidedihedrals.npy"), self.sidedihedrals
                )

        # Angles:
        try:
            self.angles = np.load(os.path.join(cache_path, "angles.npy"))
            print("Loaded angles from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating angles...")
            angle_atoms = []
            for i in range(len(self.central_atom_indices) - 2):
                angle_atoms.append(
                    self.sorted_atoms[self.central_atom_indices[i : i + 3]]
                )

            angles = Angles(angle_atoms, verbose=True).run(
                start=start, stop=stop, step=step
            )
            self.angles = angles.result.astype(np.float32)

            if cache_path:
                np.save(os.path.join(create_dir(cache_path), "angles.npy"), self.angles)

        # Lengths:
        try:
            self.lengths = np.load(os.path.join(cache_path, "lengths.npy"))
            print("Loaded lengths from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating lengths...")
            vecs = self.central_cartesians[:, :-1] - self.central_cartesians[:, 1:]
            self.lengths = np.linalg.norm(vecs, axis=2)
            if cache_path:
                np.save(
                    os.path.join(create_dir(cache_path), "lengths.npy"), self.lengths
                )

        assert self.lengths.shape[1] == self.central_cartesians.shape[1] - 1
        assert self.angles.shape[1] == self.central_cartesians.shape[1] - 2
        assert self.dihedrals.shape[1] == self.central_cartesians.shape[1] - 3

    def __iadd__(self, other):
        assert np.all(self.sorted_atoms.names == other.sorted_atoms.names)
        self.cartesians = np.concatenate([self.cartesians, other.cartesians], axis=0)
        self.central_cartesians = np.concatenate(
            [self.central_cartesians, other.central_cartesians], axis=0
        )
        self.dihedrals = np.concatenate([self.dihedrals, other.dihedrals], axis=0)
        self.sidedihedrals = np.concatenate(
            [self.sidedihedrals, other.sidedihedrals], axis=0
        )
        self.angles = np.concatenate([self.angles, other.angles], axis=0)
        self.lengths = np.concatenate([self.lengths, other.lengths], axis=0)
        return self

    @staticmethod
    def sort_key(atom):
        positions = {"N": 1, "CA": 2, "C": 5, "O": 6, "OXT": 7, "O1": 8, "O2": 9}
        try:
            result = positions[atom.name]
        except KeyError:
            result = 4
        return atom.resnum, result

    def write(
        self,
        path,
        coordinates,
        name="generated",
        formats=("pdb", "xtc"),
        only_central=False,
        align_reference=None,
        align_select="all",
    ):
        """
        Writes a trajectory for the given coordinates.

        :param path: directory where to save the trajectory
        :param coordinates: numpy array of xyz coordinates (frames, atoms, xyz)
        :param name: filename (without extension)
        :param formats: specify which formats schould be used to write structure and trajectory. default: ("pdb", "xtc")
        :param only_central: if True only central atom coordinates are expected (N-Ca-C...)
        :param align_reference: Allows to allign the generated conformations according to some reference.
            The reference should be given as MDAnalysis atomgroup
        :param align_select: Allows to select which atoms should be used for the alignment. e.g. "resid 5:60"
            default is "all". Have a look at the MDAnalysis selection syntax for more details.
        :return:
        """
        coordinates = np.array(coordinates)
        if coordinates.ndim == 2:
            coordinates = np.expand_dims(coordinates, 0)
        if only_central:
            output_universe = md.Merge(self.central_atoms)
            self.sorted_atoms[self.central_atom_indices].write(
                os.path.join(path, "{}.{}".format(name, formats[0]))
            )
        else:
            output_universe = md.Merge(self.sorted_atoms)
            self.sorted_atoms.write(
                os.path.join(path, "{}.{}".format(name, formats[0]))
            )
        output_universe.load_new(coordinates, format=MemoryReader)

        if align_reference is not None:
            align_traj = AlignTraj(
                output_universe, align_reference, align_select, in_memory=True
            )
            align_traj.run()

        with md.Writer(os.path.join(path, "{}.{}".format(name, formats[1]))) as w:
            for step in output_universe.trajectory:
                w.write(output_universe.atoms)

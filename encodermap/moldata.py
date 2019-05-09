import MDAnalysis as md
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from math import pi
from MDAnalysis.analysis.base import AnalysisBase
import os
from tqdm import tqdm
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.lib.distances import calc_angles
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
        angle = calc_angles(self.ag1.positions, self.ag2.positions,
                            self.ag3.positions,
                            box=self.ag1.dimensions)
        self.result.append(angle)

    def _conclude(self):
        self.result = np.asarray(self.result)


class MolData:
    def __init__(self, atom_group, cache_path="", start=None, stop=None, step=None,):
        self.universe = atom_group.universe

        self.sorted_atoms = self.universe.atoms[[atom.ix for atom in sorted(atom_group.atoms, key=self.sort_key)]]

        self.central_atom_indices = [i for i, atom in enumerate(self.sorted_atoms) if atom.name in ["N", "CA", "C"]]

        # Cartesians:
        try:
            self.cartesians = np.load(os.path.join(cache_path, "cartesians.npy"))
            print("Loaded cartesians from {}".format(cache_path))

        except FileNotFoundError:
            print("Loading Cartesians...")
            positions = Positions(self.sorted_atoms, verbose=True).run(start=start, stop=stop, step=step)
            self.cartesians = positions.result.astype(np.float32)

            if cache_path:
                np.save(os.path.join(create_dir(cache_path), "cartesians.npy"), self.cartesians)

        # Dihedrals:
        try:
            self.dihedrals = np.load(os.path.join(cache_path, "dihedrals.npy"))
            print("Loaded dihedrals from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating dihedrals...")
            dihedral_atoms = []
            for i in set(self.sorted_atoms.resnums):
                phi_atoms = (self.universe.select_atoms("resnum {} and name C".format(i - 1)) +
                             self.universe.select_atoms("resnum {} and (name N or name CA or name C)".format(i)))
                if len(phi_atoms) == 4:
                    dihedral_atoms.append(phi_atoms.dihedral)
                psi_atoms = (self.universe.select_atoms("resnum {} and (name N or name CA or name C)".format(i)) +
                             self.universe.select_atoms("resnum {} and name N".format(i + 1)))
                if len(psi_atoms) == 4:
                    dihedral_atoms.append(psi_atoms.dihedral)
            dihedrals = Dihedral(dihedral_atoms, verbose=True).run(start=start, stop=stop, step=step)
            self.dihedrals = dihedrals.angles.astype(np.float32)
            self.dihedrals *= pi / 180

            if cache_path:
                np.save(os.path.join(cache_path, "dihedrals.npy"), self.dihedrals)

        # Angles:
        try:
            self.angles = np.load(os.path.join(cache_path, "angles.npy"))
            print("Loaded angles from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating angles...")
            angle_atoms = []
            for i in range(len(self.central_atom_indices)-2):
                angle_atoms.append(self.sorted_atoms[self.central_atom_indices[i:i+3]])

            angles = Angles(angle_atoms, verbose=True).run(start=start, stop=stop, step=step)
            self.angles = angles.result.astype(np.float32)

            if cache_path:
                np.save(os.path.join(create_dir(cache_path), "angles.npy"), self.angles)

        # Lengths:
        try:
            self.lengths = np.load(os.path.join(cache_path, "lengths.npy"))
            print("Loaded lengths from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating lengths...")
            central_cartesians = self.cartesians[:, self.central_atom_indices]
            vecs = central_cartesians[:, :-1] - central_cartesians[:, 1:]
            self.lengths = np.linalg.norm(vecs, axis=2)
            if cache_path:
                np.save(os.path.join(create_dir(cache_path), "lengths.npy"), self.lengths)

    def __iadd__(self, other):
        assert np.all(self.sorted_atoms.names == other.sorted_atoms.names)
        self.cartesians = np.concatenate([self.cartesians, other.cartesians], axis=0)
        self.dihedrals = np.concatenate([self.dihedrals, other.dihedrals], axis=0)
        return self

    @staticmethod
    def sort_key(atom):
        positions = {"N": 1,
                     "H": 2,
                     "CA": 3,
                     "C": 5,
                     "O": 6,
                     "OXT": 7
                     }
        try:
            result = positions[atom.name]
        except KeyError:
            result = 4
        return atom.resnum, result

    def write(self, path, coordinates, name="generated"):
        coordinates = np.array(coordinates)
        if coordinates.ndim == 2:
            coordinates = np.expand_dims(coordinates, 0)
        output_universe = md.Merge(self.sorted_atoms)
        output_universe.load_new(coordinates, format=MemoryReader)
        self.sorted_atoms.write(os.path.join(path, "{}.pdb".format(name)))
        with md.Writer(os.path.join(path, "{}.xtc".format(name))) as w:
            for step in output_universe.trajectory:
                w.write(output_universe.atoms)

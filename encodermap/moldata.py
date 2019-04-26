import MDAnalysis as md
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from math import pi
from MDAnalysis.analysis.base import AnalysisBase
import os
from tqdm import tqdm
from MDAnalysis.analysis.dihedrals import Dihedral
from .misc import create_dir


class MolData:
    def __init__(self, atom_group, cache_path=""):
        self.universe = atom_group.universe
        self.atom_group = atom_group

        self.sorted_atoms = self.universe.atoms[[atom.ix for atom in sorted(atom_group.atoms, key=self.sort_key)]]

        self.central_atom_indices = [i for i, atom in enumerate(self.sorted_atoms) if atom.name in ["N", "CA", "C"]]

        try:
            self.cartesians = np.load(os.path.join(cache_path, "cartesians.npy"))
            print("Loaded cartesians from {}".format(cache_path))

        except FileNotFoundError:
            print("Loading Cartesians...")
            self.cartesians = np.zeros((len(self.universe.trajectory), self.atom_group.n_atoms, 3), dtype=np.float32)
            for i, frame in tqdm(enumerate(self.universe.trajectory), total=len(self.universe.trajectory)):
                self.cartesians[i, ...] = self.sorted_atoms.positions
            if cache_path:
                np.save(os.path.join(create_dir(cache_path), "cartesians.npy"), self.cartesians)

        try:
            self.dihedrals = np.load(os.path.join(cache_path, "dihedrals.npy"))
            print("Loaded dihedrals from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating dihedrals...")
            self.dihedral_atoms = []
            for residue in self.atom_group.residues:
                phi = residue.phi_selection()
                if phi:
                    self.dihedral_atoms.append(phi.dihedral)
                psi = residue.psi_selection()
                if psi:
                    self.dihedral_atoms.append(psi.dihedral)

            dihedrals = Dihedral(self.dihedral_atoms, verbose=True).run()
            self.dihedrals = dihedrals.angles.astype(np.float32)
            self.dihedrals *= pi/180
            if cache_path:
                np.save(os.path.join(cache_path, "dihedrals.npy"), self.dihedrals)

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
        return (atom.resindex, result)

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

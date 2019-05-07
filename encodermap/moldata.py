import MDAnalysis as md
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from math import pi
from MDAnalysis.analysis.base import AnalysisBase
import os
from tqdm import tqdm
from MDAnalysis.analysis.dihedrals import Dihedral
from .misc import create_dir


class Positions(AnalysisBase):
    def __init__(self, atomgroup, **kwargs):
        super(Positions, self).__init__(atomgroup.universe.trajectory,
                                          **kwargs)
        self._ag = atomgroup

    def _prepare(self):
        # OPTIONAL
        # Called before iteration on the trajectory has begun.
        # Data structures can be set up at this time
        self.result = []

    def _single_frame(self):
        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store result of `some_function` for a single frame
        self.result.append(self._ag.positions)

    def _conclude(self):
        # OPTIONAL
        # Called once iteration on the trajectory is finished.
        # Apply normalisation and averaging to results here.
        self.result = np.asarray(self.result)


class MolData:
    def __init__(self, atom_group, cache_path="", start=None, stop=None, step=None,):
        self.universe = atom_group.universe

        self.sorted_atoms = self.universe.atoms[[atom.ix for atom in sorted(atom_group.atoms, key=self.sort_key)]]

        self.central_atom_indices = [i for i, atom in enumerate(self.sorted_atoms) if atom.name in ["N", "CA", "C"]]

        try:
            self.cartesians = np.load(os.path.join(cache_path, "cartesians.npy"))
            print("Loaded cartesians from {}".format(cache_path))

        except FileNotFoundError:
            print("Loading Cartesians...")
            # self.cartesians = np.zeros((len(self.universe.trajectory), self.atom_group.n_atoms, 3), dtype=np.float32)
            # for i, frame in tqdm(enumerate(self.universe.trajectory), total=len(self.universe.trajectory)):
            #     self.cartesians[i, ...] = self.sorted_atoms.positions
            positions = Positions(self.sorted_atoms, verbose=True).run(start=start, stop=stop, step=step)
            self.cartesians = positions.result

            if cache_path:
                np.save(os.path.join(create_dir(cache_path), "cartesians.npy"), self.cartesians)

        try:
            self.dihedrals = np.load(os.path.join(cache_path, "dihedrals.npy"))
            print("Loaded dihedrals from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating dihedrals...")
            dihedral_atoms = []
            for i in set(self.sorted_atoms.resnums):
                phi_atoms = (self.universe.select_atoms("resnum {} and name C".format(i-1)) +
                       self.universe.select_atoms("resnum {} and (name N or name CA or name C)".format(i)))
                if len(phi_atoms) == 4:
                    dihedral_atoms.append(phi_atoms.dihedral)
                psi_atoms = (self.universe.select_atoms("resnum {} and (name N or name CA or name C)".format(i)) +
                       self.universe.select_atoms("resnum {} and name N".format(i+1)))
                if len(psi_atoms) == 4:
                    dihedral_atoms.append(psi_atoms.dihedral)
            dihedrals = Dihedral(dihedral_atoms, verbose=True).run(start=start, stop=stop, step=step)

            if cache_path:
                np.save(os.path.join(cache_path, "dihedrals.npy"), self.dihedrals)

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
        return (atom.resnum, result)

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

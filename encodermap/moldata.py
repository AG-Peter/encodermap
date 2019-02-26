import MDAnalysis as md
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader
from math import pi
from MDAnalysis.analysis.base import AnalysisBase
import os
from tqdm import tqdm


class AllProteinPhiPsi(AnalysisBase):
    """
    Calculates all phi and spi dihedral angles for a protein in the trajectory of the given universe
    """

    def __init__(self, atomgroup, **kwargs):
        super(AllProteinPhiPsi, self).__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup

    def _prepare(self):
        protein = self._ag.select_atoms("protein")

        self.dihedrals = []

        for residue in protein.residues:
            phi = residue.phi_selection()
            if phi:
                self.dihedrals.append(phi.dihedral)
            psi = residue.psi_selection()
            if psi:
                self.dihedrals.append(psi.dihedral)

        self.dihedral_values = []
        self.times = []

    def _single_frame(self):
        # Called after the trajectory is moved onto each new frame.
        # store result of `some_function` for a single frame

        self.dihedral_values.append([dihedral.value() / 180 * pi for dihedral in self.dihedrals])
        self.times.append(self._ts.time / 1000)


class MolData:
    def __init__(self, universe, cache_path=""):
        self.universe = universe

        self.sorted_atoms = [atom
                             for residue in self.universe.residues
                             for atom in sorted(residue.atoms, key=self.sort_key)]

        self.central_atom_indices = [i for i in range(len(self.sorted_atoms))
                                     if self.sorted_atoms[i].name in ["N", "CA", "C"]]

        try:
            self.cartesians = np.load(os.path.join(cache_path, "cartesians.npy"))
            print("Loaded cartesians from {}".format(cache_path))

        except FileNotFoundError:
            print("Loading Cartesians...")
            self.cartesians = np.zeros((len(self.universe.trajectory), self.universe.atoms.n_atoms, 3), dtype=np.float32)
            for i, frame in tqdm(enumerate(self.universe.trajectory), total=len(self.universe.trajectory)):
                self.cartesians[i, ...] = [atom.position for atom in self.sorted_atoms]
            if cache_path:
                np.save(os.path.join(cache_path, "cartesians.npy"), self.cartesians)

        try:
            self.dihedrals = np.load(os.path.join(cache_path, "dihedrals.npy"))
            print("Loaded dihedrals from {}".format(cache_path))

        except FileNotFoundError:
            print("Calculating dihedrals...")
            allproteinphipsi = AllProteinPhiPsi(self.universe, verbose=True)
            allproteinphipsi.run()
            self.dihedrals = np.array(allproteinphipsi.dihedral_values, dtype=np.float32)
            if cache_path:
                np.save(os.path.join(cache_path, "dihedrals.npy"), self.dihedrals)

    @staticmethod
    def sort_key(atom):
        positions = {"N": 1,
                     "CA": 2,
                     "C": 4,
                     "O": 5,
                     "OXT": 6
                     }
        try:
            result = positions[atom.name]
        except KeyError:
            result = 3
        return result

    def write(self, path, coordinates, name="generated"):
        coordinates = np.array(coordinates)
        if coordinates.ndim == 2:
            coordinates = np.expand_dims(coordinates, 0)
        output_universe = md.Merge(self.universe.atoms)
        sorted_atom_ids = [atom.index for atom in self.sorted_atoms]
        atom_places = np.zeros(len(sorted_atom_ids), dtype=np.int)
        for i, aid in enumerate(sorted_atom_ids):
            atom_places[aid] = i
        sorted_coordinates = coordinates[:, atom_places, :]
        output_universe.load_new(sorted_coordinates, format=MemoryReader)
        self.universe.atoms.write(os.path.join(path, "{}.pdb".format(name)), bonds="all")
        with md.Writer(os.path.join(path, "{}.xtc".format(name))) as w:
            for step in output_universe.trajectory:
                w.write(output_universe.atoms)

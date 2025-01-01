# Standard Library Imports
import os

# Third Party Imports
import matplotlib.pyplot as plt
import MDAnalysis as md
import numpy as np

# Encodermap imports
import encodermap as em


molname = "diubi"
run_id = 3
step = 50000
selection_for_alignment = "resid 0:77"

main_path = "runs/{}/run{}".format(molname, run_id)


# ######################### Load data #########################

structure_path = "data/{}/01.pdb".format(molname)
trajectory_paths = ["data/{}/{:02d}.xtc".format(molname, i + 1) for i in range(12)]

uni = md.Universe(structure_path, trajectory_paths)
selected_atoms = uni.select_atoms(
    "backbone or name H or name O1 or (name CD and resname PRO)"
)
moldata = em.MolData(selected_atoms, cache_path="data/{}/cache".format(molname))


# ######################### Load parameters and checkpoint #########################

parameters = em.ADCParameters.load(os.path.join(main_path, "parameters.json"))
e_map = em.AngleDihedralCartesianEncoderMap(
    parameters,
    moldata,
    checkpoint_path=os.path.join(main_path, "checkpoints", "step{}.ckpt".format(step)),
    read_only=True,
)


# ######################### Project Data to map #########################

projected = e_map.encode(moldata.dihedrals)


# ######################### Plot histogram with path generator and lasso Select #########################

hist, xedges, yedges = np.histogram2d(projected[:, 0], projected[:, 1], bins=500)

fig1, axe1 = plt.subplots()

caxe = axe1.imshow(
    -np.log(hist.T),
    origin="low",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    aspect="auto",
)
cbar = fig1.colorbar(caxe)
cbar.set_label("-ln(p)", labelpad=0)
axe1.set_title("Path Generator")
generator = em.plot.PathGenerateCartesians(
    axe1,
    e_map,
    moldata,
    vmd_path="/home/soft/bin/vmd",
    align_reference=moldata.sorted_atoms,
    align_select=selection_for_alignment,
)

fig2, axe2 = plt.subplots()
caxe = axe2.imshow(
    -np.log(hist.T),
    origin="low",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    aspect="auto",
)
cbar = fig2.colorbar(caxe)
cbar.set_label("-ln(p)", labelpad=0)
axe2.set_title("Selector")
selector = em.plot.PathSelect(
    axe2,
    projected,
    moldata,
    e_map.p.main_path,
    vmd_path="/home/soft/bin/vmd",
    align_reference=moldata.sorted_atoms,
    align_select=selection_for_alignment,
)

plt.show()

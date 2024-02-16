# Standard Library Imports
import os

# Third Party Imports
import matplotlib.pyplot as plt
import numpy as np

# Encodermap imports
import encodermap as em


# setting parameters
data_path = "./data"
run_path = em.misc.run_path("./runs")
csv_path = os.path.join(data_path, "trp_cage.csv")  # can be downloaded from:
# https://www.kaggle.com/tobiasle/trp-cage-dihedrals
parameters = em.Parameters()
parameters.main_path = run_path
parameters.n_steps = 50000


print("loading data ...")
data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
dihedrals = data[:, 3:41]


print("training autoencoder ...")
e_map = em.EncoderMap(parameters, dihedrals)
e_map.train()


print("projecting data ...")
low_d_projection = e_map.encode(dihedrals)


print("plotting result ...")
fig, axe = plt.subplots()
caxe = axe.scatter(
    low_d_projection[:, 0],
    low_d_projection[:, 1],
    c=data[:, -1],
    s=0.1,
    cmap="nipy_spectral",
    marker="o",
    linewidths=0,
)
cbar = fig.colorbar(caxe)
cbar.set_label("helix rmsd")


# generate structures along path
pdb_path = os.path.join(data_path, "trp_cage_extended.pdb")
generator = em.plot.PathGenerateDihedrals(axe, e_map, pdb_path)

plt.show()

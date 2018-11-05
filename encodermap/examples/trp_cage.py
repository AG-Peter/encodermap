import encodermap as em
import numpy as np
import os
import matplotlib.pyplot as plt


# setting parameters
parent_path = "/home/tobias/Desktop/mini_tut"
run_path = em.misc.run_path(parent_path)
csv_path = os.path.join(parent_path, "trp_cage.csv")
parameters = em.Parameters()
parameters.main_path = run_path
parameters.n_steps = 50000


print("loading data ...")
data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
dihedrals = data[:, 3:41]


print("training autoencoder ...")
autoencoder = em.Autoencoder(parameters, dihedrals)
autoencoder.train()


print("projecting data ...")
low_d_projection = autoencoder.encode(dihedrals)


print("plotting result ...")
fig, axe = plt.subplots()
caxe = axe.scatter(low_d_projection[:, 0], low_d_projection[:, 1], c=data[:, -1], s=0.1, cmap="nipy_spectral",
                   marker="o", linewidths=0)
cbar = fig.colorbar(caxe)
cbar.set_label("helix rmsd")


# generate structures along path
pdb_path = os.path.join(parent_path, "trp_cage_extended.pdb")
generator = em.plot.PathGenerate(axe, autoencoder, pdb_path)

plt.show()

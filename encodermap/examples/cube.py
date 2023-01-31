import encodermap as em

# generating data:
high_d_data, ids = em.misc.random_on_cube_edges(10000, sigma=0.05)

# setting parameters:
parameters = em.Parameters()
parameters.main_path = em.misc.run_path("runs/cube/")
parameters.n_steps = 10000
parameters.dist_sig_parameters = (0.2, 3, 6, 1, 2, 6)
parameters.periodicity = float("inf")

# training:
e_map = em.EncoderMap(parameters, high_d_data)
e_map.train()

# projecting:
low_d_projection = e_map.encode(high_d_data)
generated = e_map.generate(low_d_projection)


#########################################################################
# Plotting:

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import (  # somehow this conflicts with tensorflow if imported earlier
    Axes3D,
)

fig = plt.figure()
axe = fig.add_subplot(111, projection="3d")
axe.scatter(
    high_d_data[:, 0],
    high_d_data[:, 1],
    high_d_data[:, 2],
    c=ids,
    marker="o",
    linewidths=0,
    cmap="tab10",
)

fig, axe = plt.subplots()
axe.scatter(
    low_d_projection[:, 0],
    low_d_projection[:, 1],
    c=ids,
    s=5,
    marker="o",
    linewidths=0,
    cmap="tab10",
)

fig = plt.figure()
axe = fig.add_subplot(111, projection="3d")
axe.scatter(
    generated[:, 0],
    generated[:, 1],
    generated[:, 2],
    c=ids,
    marker="o",
    linewidths=0,
    cmap="tab10",
)

plt.show()

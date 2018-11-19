import encodermap as em
import numpy as np
import matplotlib.pyplot as plt


data, ids = em.misc.random_on_cube_edges(1000, sigma=0.05)

high_d_dist_sig_parameters = (0.2, 3, 6)
periodicity = float("inf")

axe = em.plot.distance_histogram(data, periodicity, high_d_dist_sig_parameters)
plt.show()

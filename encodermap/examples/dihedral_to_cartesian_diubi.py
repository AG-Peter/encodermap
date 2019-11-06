import MDAnalysis as md
import os
import numpy as np
import encodermap as em
import tensorflow as tf
import matplotlib.pyplot as plt
import locale
import copy


molname = "diubi"


# ######################### Load data #########################

structure_path = "data/{}/start.gro".format(molname)
trajectory_path = "data/{}/traj.xtc".format(molname)

uni = md.Universe(structure_path, trajectory_path)

# select only atoms in the backbone or directly bound to the backbone. Atoms in the side chains are not supported (yet)
selected_atoms = uni.select_atoms("backbone or name H or name O1 or (name CD and resname PRO)")

# The MolData object takes care of calculating all the necessary information like dihedrals, angles, bondlengths ...
# You can provide a cache_path to store this information for the next time.
moldata = em.MolData(selected_atoms, cache_path="data/{}/cache".format(molname))


# ######################### Define parameters #########################

total_steps = 50000

# First we want to train without C_alpha cost.
# Finally, we want to activate the C_alpha cost to improve the long-range order of the generated conformations
parameters = em.ADCParameters()
parameters.main_path = em.misc.run_path("runs/{}".format(molname))

parameters.cartesian_cost_scale = 0
parameters.cartesian_cost_variant = "mean_abs"
parameters.cartesian_cost_scale_soft_start = (int(total_steps/10*9), int(total_steps/10*9)+1000)
parameters.cartesian_pwd_start = 1  # Calculate pairwise distances starting form the second backbone atom ...
parameters.cartesian_pwd_step = 3  # for every third atom. These are the C_alpha atoms

parameters.dihedral_cost_scale = 1
parameters.dihedral_cost_variant = "mean_abs"

parameters.distance_cost_scale = 0  # no distance cost in dihedral space
parameters.cartesian_distance_cost_scale = 100  # instead we use distance cost in C_alpha distance space
parameters.cartesian_dist_sig_parameters = [400, 10, 5, 1, 2, 5]

parameters.checkpoint_step = max(1, int(total_steps/10))
parameters.l2_reg_constant = 0.001
parameters.center_cost_scale = 0
parameters.id = molname


# ########### Check how distances are mapped with your choice of Sigmoid parameters ##################

pwd = em.misc.pairwise_dist(
    moldata.central_cartesians[::1000, parameters.cartesian_pwd_start::parameters.cartesian_pwd_step], flat=True)
with tf.Session() as sess:
    pwd = sess.run(pwd)
axes = em.plot.distance_histogram(pwd, float("inf"), parameters.cartesian_dist_sig_parameters)
plt.show()
# somehow matplotlib messes with this setting and causes problems in tensorflow
locale.setlocale(locale.LC_NUMERIC, "en_US.UTF-8")


# ######################### Get references from dummy model #########################
dummy_parameters = copy.deepcopy(parameters)
dummy_parameters.main_path = em.misc.create_dir(os.path.join(parameters.main_path, "dummy"))
dummy_parameters.n_steps = int(len(moldata.dihedrals) / parameters.batch_size)
dummy_parameters.summary_step = 1

e_map = em.AngleDihedralCartesianEncoderDummy(dummy_parameters, moldata)
e_map.train()
e_map.close()
e_map = None

costs = em.misc.read_from_log(os.path.join(dummy_parameters.main_path, "train"),
                              ["cost/angle_cost", "cost/dihedral_cost", "cost/cartesian_cost"])
means = [np.mean(cost[:, 2]) for cost in costs]
parameters.angle_cost_reference = means[0]
parameters.dihedral_cost_reference = means[1]
parameters.cartesian_cost_reference = means[2]

np.savetxt(os.path.join(dummy_parameters.main_path, "adc_cost_means.txt"), np.array(means))


# ######################### Run Training #########################

# First run without C_alpha cost
parameters.n_steps = parameters.cartesian_cost_scale_soft_start[0]
e_map = em.AngleDihedralCartesianEncoder(parameters, moldata)
e_map.train()
e_map.close()
e_map = None

# Now we turn on C_alpha cost and continue the training run
parameters.n_steps = total_steps - parameters.cartesian_cost_scale_soft_start[0]
parameters.cartesian_cost_scale = 1
ckpt_path = os.path.join(parameters.main_path, "checkpoints", "step{}.ckpt"
                         .format(parameters.cartesian_cost_scale_soft_start[0]))

e_map = em.AngleDihedralCartesianEncoder(parameters, moldata, checkpoint_path=ckpt_path)
e_map.train()
e_map.close()
e_map = None

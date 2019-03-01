import encodermap as em
import MDAnalysis as md
import os


structure_path = "/home/andrejb/Research/SIMS/2017_10_20_monoUb_nat/start.pdb"
traj_path = "/home/andrejb/Research/SIMS/2017_10_20_monoUb_nat/traj.xtc"

uni = md.Universe(structure_path, traj_path)
selected_atoms = uni.select_atoms("resid 0:72 and (backbone or name O1 or name H or name CB)")

moldata = em.MolData(selected_atoms, cache_path=em.misc.create_dir("data/ubi_without_tail_bb"))

parameters = em.Parameters()
parameters.main_path = em.misc.run_path(em.misc.create_dir("runs"))
parameters.dihedral_to_cartesian_cost_scale = 1
parameters.auto_cost_scale = 1
parameters.distance_cost_scale = 0
parameters.dist_sig_parameters = (6, 12, 6, 1, 2, 6)
parameters.center_cost_scale = 0.0001
parameters.l2_reg_constant = 0.
parameters.batch_size = 256
parameters.n_steps = 10000
parameters.summary_step = 100
parameters.gpu_memory_fraction = 0.4
parameters.checkpoint_step = 1000

e_map = em.DihedralCartesianEncoder(parameters, moldata)
e_map.train()
e_map.close()

# e_map.p.dihedral_to_cartesian_cost_scale = 1
ckpt_path = os.path.join(parameters.main_path, "checkpoints", "step{}.ckpt".format(parameters.n_steps))
e_map = em.DihedralCartesianEncoder(parameters, moldata, checkpoint_path=ckpt_path)
e_map.train()


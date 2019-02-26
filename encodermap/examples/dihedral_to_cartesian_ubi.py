import encodermap as em
import MDAnalysis as md
import os


structure_path = "/home/andrejb/Research/SIMS/2017_10_20_monoUb_nat/start.pdb"
traj_path = "/home/andrejb/Research/SIMS/2017_10_20_monoUb_nat/traj.xtc"

uni = md.Universe(structure_path, traj_path, guess_bonds=True)
selection = uni.select_atoms("backbone or name O1 or name H")
# uni = md.Merge(selection)

moldata = em.MolData(uni, cache_path=em.misc.create_dir("data/ubi"))

parameters = em.Parameters()
parameters.main_path = em.misc.run_path(em.misc.create_dir("runs"))
parameters.dihedral_to_cartesian_cost_scale = 0
parameters.auto_cost_scale = 1
parameters.distance_cost_scale = 0
parameters.l2_reg_constant = 0.
parameters.batch_size = 256
parameters.n_steps = 10000
parameters.summary_step = 100
parameters.gpu_memory_fraction = 0.4

e_map = em.DihedralCartesianEncoder(parameters, moldata)
e_map.train()

e_map.p.dihedral_to_cartesian_cost_scale = 1
ckpt_path = os.path.join(parameters.main_path, "checkpoints", "step{}.ckpt".format(parameters.n_steps))
e_map = em.DihedralCartesianEncoder(parameters, moldata, checkpoint_path=ckpt_path)
e_map.train()


import encodermap as em
import MDAnalysis as md
import os


path = "/home/andrejb/Research/SIMS/2017_10_20_monoUb_nat/start.pdb"

uni = md.Universe(path, guess_bonds=True)
selection = uni.select_atoms("backbone or name O1 or name H")
# uni = md.Merge(selection)

moldata = em.MolData(uni)

parameters = em.Parameters()
parameters.main_path = em.misc.run_path(em.misc.create_dir("runs"))
parameters.dihedral_to_cartesian_cost_scale = 1
parameters.auto_cost_scale = 0
parameters.distance_cost_scale = 0
parameters.l2_reg_constant = 0.
parameters.batch_size = 1
parameters.n_steps = 5000
parameters.summary_step = 1
parameters.gpu_memory_fraction = 0.4

e_map = em.DihedralCartesianEncoder(parameters, moldata)
e_map.train()

moldata.write(parameters.main_path, e_map.cartesians)



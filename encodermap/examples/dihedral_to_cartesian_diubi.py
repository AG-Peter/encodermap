import encodermap as em
import MDAnalysis as md
import os

run_settings = [[1, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 0]]  # dihedral, Cartesian1, Cartesian2

structure_path = "/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/start.gro"
traj_path = "/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/traj.xtc"

uni = md.Universe(structure_path, traj_path)
selected_atoms = uni.select_atoms("backbone or name O1 or name H or name CB")

moldata = em.MolData(selected_atoms)

for i in range(5):
    for setting in run_settings:
        parameters = em.Parameters()
        parameters.main_path = em.misc.run_path(em.misc.create_dir("runs/diubi"))
        parameters.dihedral_to_cartesian_cost_scale = setting[1]
        parameters.auto_cost_scale = setting[0]
        parameters.distance_cost_scale = 0
        parameters.dist_sig_parameters = (6, 12, 6, 1, 2, 6)
        parameters.center_cost_scale = 0.0001
        parameters.l2_reg_constant = 0.
        parameters.batch_size = 256
        parameters.n_steps = 10000
        parameters.summary_step = 100
        parameters.gpu_memory_fraction = 1.0
        parameters.checkpoint_step = 1000

        e_map = em.DihedralCartesianEncoder(parameters, moldata)
        e_map.train()
        e_map.close()

        e_map.p.dihedral_to_cartesian_cost_scale = setting[2]
        ckpt_path = os.path.join(parameters.main_path, "checkpoints", "step{}.ckpt".format(parameters.n_steps))
        e_map = em.DihedralCartesianEncoder(parameters, moldata, checkpoint_path=ckpt_path)
        e_map.train()
        e_map.close()



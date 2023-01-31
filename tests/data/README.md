Example data taken from mdtraj. https://github.com/mdtraj/mdtraj/tree/master/tests/data

1am7_protein.pdb, 1am7_corrected.xtc, and 1am7_uncorrected.xtc are Gromacs trajectories of lambda phage lysozyme (PDB: 1AM7). The uncorrected xtc file has the protein drifting across the periodic box. The corrected xtc file has been corrected with trjconv -pbc mol. This are useful as a sanity check for PBC issues.

The trajectory PFFP_MD_fin_protonly_dt_100.xtc and the topologies (PFFP_MD_fin_protonly.grp, PFFP_MD_fin_protonly.tpr, and PFFP_vac.top) was created by Kevin Sawade during his Bachelor thesis in 2015. The creation employed GROMACS4.6.1. The protein was solvated in a 50/50 MeOH/Water mixture. Here is the mdp-file:

```
title                    = Full MD
cpp                      = /lib/cpp
;
; Run control:
;
constraints              = H-bonds
integrator               = md
dt                       = 0.002 ; ps
nsteps                   = 2500 ; total 5 ps
;
; Neighbor searching:
;
nstlist                  = 5
ns_type                  = grid
rlist                    = 1.4 ; nm
;
; Output control:
;
nstcomm                  = 1
nstxout                  = 250000
nstvout                  = 500000
nstfout                  = 0
nstlog                   = 1000
nstenergy                = 500
nstxtcout                = 500
xtc-precision            = 1000
xtc_grps                 = Protein Non-Protein
;
; Electrostatics
;
coulombtype              = PME
pme_order                = 4
fourierspacing           = 0.14 ; nm
fourier_nx               = 0
fourier_ny               = 0
fourier_nz               = 0
rcoulomb                 = 1.4 ; nm
ewald_rtol               = 1e-5
optimize_fft             = yes
;
; VdW
;
vdwtype                  = cut-off
rvdw                     = 1.4 ; nm
;
; Temperature & Pressure coupling:
;
tcoupl                   = v-rescale
tc-grps                  = Protein Non-Protein
tau_t                    = 0.1 0.1 ; ps
ref_t                    = 300 300 ; K

pcoupl                   = Berendsen
pcoupltype               = isotropic
tau_p                    = 1
compressibility          = 4.5e-5
ref_p                    = 1
;
; Velocity generation:
;
gen_vel                  = no
;gen_temp                 = 300 ; K
;gen_seed                 = -1


```

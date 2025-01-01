#!/bin/bash

. sh_libs/liblog.sh

info "Killing gmx commands"
pkill gmx
info "Deleting test_sims/ and sims.h5 and .sims*"
rm -rf test_sims/
rm -f sims.h5
rm -f .sims*
info "Creating dirs"
mkdir -p test_sims/sim1
mkdir -p test_sims/sim2
mkdir -p test_sims/sim3
info "Running gmx"
gmx grompp -f water_simulation/production.mdp -c water_simulation/spc216_stacked.gro -p water_simulation/topol_stacked.top -o test_sims/sim1/production.tpr -po test_sims/sim1/mdout.mdp -maxwarn 1 &> /dev/null
gmx grompp -f water_simulation/production.mdp -c water_simulation/spc216_fails.gro -p water_simulation/topol_stacked.top -o test_sims/sim2/producton_fails.tpr -po test_sims/sim2/mdout.mdp -maxwarn 1 &> /dev/null
gmx grompp -f water_simulation/production_short.mdp -c water_simulation/spc216_stacked.gro -p water_simulation/topol_stacked.top -o test_sims/sim3/production_short.tpr -po test_sims/sim3/mdout.mdp -maxwarn 1 &> /dev/null
cp ../simulation_attender/simulation_attender.py .
exit
info "Running sim_attender commands"
python simulation_attender.py --help
python simulation_attender.py collect test_sims/
python simulation_attender.py template --command "gmx mdrun -deffnm {{ stem }}"
python simulation_attender.py submit -cm local
python simulation_attender.py run -cm local
info "Checking again for completion or crash"
python simulation_attender.py run -cm local
info "Sleeping for 15 seconds"
sleep 15
info "Checking again"
python simulation_attender.py run -cm local
sleep 6
info "Checking short simulation."
python simulation_attender.py run -cm local
sleep 6
info "Checking failed simulation."
python simulation_attender.py run -cm local
info "Clearing workspace"
rm -r test_sims/
rm sims.h5
rm .sims*

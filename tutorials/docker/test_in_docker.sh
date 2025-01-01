#!/bin/bash

# turn on bash's job control
set -m

# load libs
. /sh_libs/liblog.sh

# loading gromacs
info "Starting tests for simulation_attender.py"
info "Staring general tests to ensure the environment is working."
info "Sourcing environment modules"
source /usr/share/Modules/init/profile.sh
info "Loading gromacs module"
module load gromacs/2023.1
info "Checking whether a gromacs command exists"
if ! command -v gmx &> /dev/null ; then
    error "The command gmx did not succeed."
    gmx
    exit
else
  info "Gromacs installation present."
fi

# anaconda
info "Sourcing anaconda"
source /usr/local/anaconda3/bin/activate &> /dev/null
info "Testing the python installation."
if ! command -v pytest &> /dev/null ; then
    error "The command pytest did not succeed."
    gmx
    exit
else
  info "Pytest is working. Starting the tests."
fi

# testing
pytest test_simulation_attender.py

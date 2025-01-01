#!/bin/bash

set -m
. /sh_libs/liblog.sh

info "Testing the base image."
info "Testing current user."
current_user=$(whoami)
if [[ $current_user == "encodermap" ]] ; then
  info "Correct user selected."
else
  error "Tests not running with the encodermap user. Current user is ${current_user}. Make sure to call docker with -u=1000."
  error "Here's a cat of /etc/passwd. Check, whether the encodermap user is even listed here."
  cat /etc/passwd
fi

info "Testing bash_profile"
if [ -f ~/.profile ] ; then
  info "User $current_user has a ~/.profile file."
else
  error "The user doesn't have a ~/.profile file. Environment modules can't load without it."
  exit
fi

info "Testing user's login-shell"
login_shell=$(awk -F: -v user=$current_user '$1 == user {print $NF}' /etc/passwd)
if [[ $login_shell == "/usr/bin/bash" ]] ; then
  info "User has correct login shell ${login_shell}."
else
  error "User has wrong login shell ${login_shell}."
  exit
fi

info "Testing whether environment modules is available"
if command -v module &> /dev/null ; then
  info "Environment modules is available. Sourcing gromacs now."
else
  error "Environment modules is either not available or not sourced."
  exit
fi

module load gromacs/$GMX_VERSION
if command -v gmx &> /dev/null ; then
  info "Gromacs sourced via environment modules."
else
  error "Gromacs was not sourced."
  exit
fi
info "All tests passed."

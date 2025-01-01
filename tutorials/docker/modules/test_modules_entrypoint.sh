#!/bin/bash

# turn on bash's job control
set -m

# load libs
. /sh_libs/liblog.sh

# print uid
info "Running a test docker container with environment-modules installed."
info "Sourcing /usr/share/Modules/init/profile.sh"
source /usr/share/Modules/init/profile.sh
info "Finished. Happy Testing."

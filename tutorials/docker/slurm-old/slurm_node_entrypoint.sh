#!/bin/bash

# turn on bash's job control
set -m

# bring up sshd
/usr/sbin/sshd

# print uid
id

# start the munge daeomon
service munge start
su -u munge /sbin/munged
munge -n
munge -n | unmunge
remunge

# replace nproc
sed -i "s/REPLACE_IT/CPUs=$(nproc)/g" /etc/slurm-llnl/slurm.conf

# start the slurm daemon
service slurmd start

# now we bring the primary process back into the foreground
# and leave it there
# use the forgrounding of a process, if the process docker is running doesn't automatically terminate (i.e. webserver)
# fg %1

# use this, if the docker container automatically terminates, but you want to keep it running
tail -f /dev/null

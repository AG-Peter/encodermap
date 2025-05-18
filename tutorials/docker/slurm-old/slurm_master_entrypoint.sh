#!/bin/bash

# turn on bash's job control
set -m

# bring up sshd
/usr/sbin/sshd

# print uid
id

# wait for the nodes to spin up and create passwordless ssh
/wait-for-it.sh slurm-node1.local.dev:22 --strict -- echo "slurm-node1.local.dev ssh(22) is up" ; sshpass -p adminpassword ssh-copy-id -i /etc/ssh/ssh_host_rsa_key.pub slurm-node1.local.dev
/wait-for-it.sh slurm-node2.local.dev:22 --strict -- echo "slurm-node2.local.dev ssh(22) is up" ; sshpass -p adminpassword ssh-copy-id -i /etc/ssh/ssh_host_rsa_key.pub slurm-node2.local.dev
/wait-for-it.sh slurm-login-node.local.dev:22 --strict -- echo "slurm-login-node.local.dev ssh(22) is up" ; sshpass -p adminpassword ssh-copy-id -i /etc/ssh/ssh_host_rsa_key.pub slurm-login-node.local.dev

# wait for the slurmdbd to spin up
/wait-for-it.sh slurm-database.local.dev:6819 --strict -- echo "slurm-database.local.dev db(6819) is up"

# start the munge daeomon
service munge start
su -u munge /sbin/munged
munge -n
munge -n | unmunge
remunge

# fix stuff in the slurm configuration
sed -i "s/REPLACE_IT/CPUs=$(nproc)/g" /etc/slurm-llnl/slurm.conf


# start the slurm control daemons
sacctmgr -i add_cluster "cluster"
sleep 2s
/wait-for-it.sh slurm-node1.local.dev:6818 --strict -- service slurmctld start

# now we bring the primary process back into the foreground
# and leave it there
# use the forgrounding of a process, if the process docker is running doesn't automatically terminate (i.e. webserver)
# fg %1

# use this, if the docker container automatically terminates, but you want to keep it running
tail -f /dev/null

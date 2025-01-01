#!/bin/bash

# turn on bash's job control
set -m

# bring up sshd
/usr/sbin/sshd

# print uid
id

# wait for the sql-server to be available and add slurm to the database
MYSQL_ROOT_PASSWORD=sql_root_passw0rd
/wait-for-it.sh slurm-db.local.dev:3306 --strict -- echo "slurm-db.local.dev db(3306) is up" ; mysql -h slurm-db.local.dev -u root -p$MYSQL_ROOT_PASSWORD < /slurm_acct_db.sql

# start the munge daeomon
sudo -u munge service munge start
# su -u munge /sbin/munged
munge -n
munge -n | unmunge
remunge

# start the slurmdb daemon
sudo -u slurm service slurmdbd start

# now we bring the primary process back into the foreground
# and leave it there
# use the forgrounding of a process, if the process docker is running doesn't automatically terminate (i.e. webserver)
# fg %1

# use this, if the docker container automatically terminates, but you want to keep it running
tail -f /dev/null

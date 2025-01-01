#!/bin/bash

# load libs
. /sh_libs/liblog.sh

# turn on bash's job control
set -m

# write the password
info "Running tests in EncoderMap's SLURM node with PROMETHEUS."
echo $LDAP_ADMIN_PASSWORD > /etc/ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/pam_ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/libnss-ldap.secret
unset LDAP_ADMIN_PASSWORD

# wait for the nodes to spin up and create passwordless ssh
info "Waiting for OPENLDAP to spool up."
/wait-for-it.sh openldap:636 --strict -- echo "openldap.example.org 636 is up" && /etc/init.d/ssh start && /etc/init.d/nscd restart && /etc/init.d/nslcd restart

# bring up sshd
info "Bringing up sshd."
/usr/sbin/sshd

# start the munge daemon
info "Starting munge daemon."
gosu root service munge start

# fix stuff in the slurm configuration
info "Fixing the SLURM configuration by replacing REPLACE_IT with $(nproc)."
sed -i "s/REPLACE_IT/CPUs=$(nproc)/g" /usr/etc/slurm.conf

# wait for the slurm master to become active
info "Waiting for slurm-master 6817 to open."
/wait-for-it.sh slurm-master.example.org:6817 --timeout=100 --strict -- echo "slurm-master.example.org 6817 is up."
exec gosu root /usr/sbin/slurmd &

# start the prometheus exporter
info "Starting the slurm-exporter."
exec gosu localadmin /usr/bin/slurm-exporter &

# Starting jupyter notebook
info "Starting jupyter notebook."
exec gosu encodermap /usr/venv/emap_env/bin/python -m jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password='' --port="8888"

# use this, if the docker container automatically terminates, but you want to keep it running
tail -f /dev/null

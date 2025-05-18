#!/bin/bash

# turn on bash's job control
set -e

# load libs
. /sh_libs/liblog.sh
info "Starting dask tests on EncoderMap SLURM."

echo $LDAP_ADMIN_PASSWORD > /etc/ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/pam_ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/libnss-ldap.secret
unset LDAP_ADMIN_PASSWORD

# wait for the nodes to spin up and create passwordless ssh
/wait-for-it.sh openldap:636 --strict -- echo "openldap.example.org 636 is up" && /etc/init.d/ssh start && /etc/init.d/nscd restart && /etc/init.d/nslcd restart

# bring up sshd
/usr/sbin/sshd

# start the munge daemon
gosu root service munge start

# fix stuff in the slurm configuration
sed -i "s/REPLACE_IT/CPUs=$(nproc)/g" /usr/etc/slurm.conf

# wait for the slurm master to become active
/wait-for-it.sh slurm-master.example.org:6817 --timeout=100 --strict -- echo "slurm-master.example.org 6817 is up"
exec gosu root /usr/sbin/slurmd &

info "Started LDAP, Munge, and SLURMD."

cd /app
/usr/venv/emap_env/bin/python -m pip freeze | grep tensor
/usr/venv/emap_env/bin/python tests/test_featurizer.py TestSLURMFeatures

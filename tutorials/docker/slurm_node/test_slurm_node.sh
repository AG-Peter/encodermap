#!/bin/bash

# turn on bash's job control
set -e

# load libs
. /sh_libs/liblog.sh

# write the password
info "Running tests in EncoderMap's SLURM node."
echo $LDAP_ADMIN_PASSWORD > /etc/ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/pam_ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/libnss-ldap.secret
unset LDAP_ADMIN_PASSWORD

# wait for the nodes to spin up and create passwordless ssh
info "Using wait-for-it to wait for openldap to spool up."
/wait-for-it.sh openldap:636 --strict -- echo "openldap.example.org 636 is up" && /etc/init.d/ssh start && /etc/init.d/nscd restart && /etc/init.d/nslcd restart

# bring up sshd
info "Bringing up sshd."
/usr/sbin/sshd

info "Testing ssh to master. First with localadmin."
if sshpass -p password ssh -oStrictHostKeyChecking=no -q localadmin@slurm-master.example.org exit ; then
  info "Can ssh into slurm-master using localadmin."
else
  error "Can't ssh into slurm-master using localadmin."
fi

info "Testing ssh to master with user01."
if sshpass -p password1 ssh -oStrictHostKeyChecking=no -q user01@slurm-master.example.org exit ; then
  info "Can ssh into slurm-master using user01."
else
  error "Can't ssh into slurm-master using user01."
fi

# start the munge daemon
info "Starting munge daemon."
gosu root service munge start

# fix stuff in the slurm configuration
info "Fixing the SLURM configuration by replacing REPLACE_IT with $(nproc)}"
sed -i "s/REPLACE_IT/CPUs=$(nproc)/g" /usr/etc/slurm.conf

# wait for the slurm master to become active
info "Waiting for slurm-master 6817 to open."
/wait-for-it.sh slurm-master.example.org:6817 --timeout=100 --strict -- echo "slurm-master.example.org 6817 is up"
info "Starting slurmd."
exec gosu root /usr/sbin/slurmd &
sleep 3

info "Testing sinfo command."
mapfile -t arr < <(sinfo)
if [[ $? -eq 0 ]] && [[ ${arr[@]} =~ "normal*      up       1:00      1   idle c1" ]]; then
  info "The sinfo command succeeded. Shutting down."
  exit 0
else
  sinfo
  error "The sinfo command errored with code $? .Shutting down."
  exit
fi

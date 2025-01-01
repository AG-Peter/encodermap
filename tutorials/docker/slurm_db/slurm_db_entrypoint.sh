#!/bin/bash

# turn on bash's job control
set -m

. /opt/bitnami/scripts/liblog.sh

# some debug for startup
info "EncoderMap's SLURM DB spooling up."

# write the password
info "Writing passwords to .secret files."
echo $LDAP_ADMIN_PASSWORD > /etc/ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/pam_ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/libnss-ldap.secret
unset LDAP_ADMIN_PASSWORD

# wait for the nodes to spin up and create passwordless ssh
info "Waiting for OPENLDAP server to be up."
/wait-for-it.sh openldap:636 --strict -- echo "openldap.example.org 636 is up" && /etc/init.d/ssh start && /etc/init.d/nscd restart && /etc/init.d/nslcd restart

# bring up sshd
/usr/sbin/sshd

# print uid
id

# fix the slurm config
sed -i "s/REPLACE_IT/CPUs=$(nproc)/g" /etc/slurm/slurm.conf

# start the munge daemon
gosu root service munge start


# wait for the sql-server to be available and add slurm to the database
MYSQL_ROOT_PASSWORD=sql_root_passw0rd
/wait-for-it.sh db.example.org:3306 --timeout=30 --strict -- echo "db.example.org db(3306) is up"
echo "SELECT 1" | mysql -h db.example.org -u root -p$MYSQL_ROOT_PASSWORD
mysql -h db.example.org -u root -p$MYSQL_ROOT_PASSWORD < /slurm_acct_db.sql
unset MYSQL_ROOT_PASSWORD

# start the slurm database
exec gosu root /usr/sbin/slurmdbd -Dvvv
# tail -f /dev/null

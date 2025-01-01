#!/bin/bash

# turn on bash's job control
set -m

. /opt/bitnami/scripts/liblog.sh

# some debug for startup
info "EncoderMap's LDAP client spooling up."
info "Checking whether port 636 is open."
if nc -z openldap 636 ; then
  info "Port is open."
else
  error "Port is not open."
  exit
fi

# write the password
info "Writing passwords to .secret files."
echo $LDAP_ADMIN_PASSWORD > /etc/ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/pam_ldap.secret
echo $LDAP_ADMIN_PASSWORD > /etc/libnss-ldap.secret
unset LDAP_ADMIN_PASSWORD

# wait for the nodes to spin up and create passwordless ssh
/wait-for-it.sh openldap:636 --strict -- echo "openldap 636 is up" && /etc/init.d/ssh start && /etc/init.d/nscd restart && /etc/init.d/nslcd restart

# use this, if the docker container automatically terminates, but you want to keep it running
tail -f /dev/null

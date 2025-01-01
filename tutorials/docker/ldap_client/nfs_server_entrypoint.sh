#!/bin/bash

# turn on bash's job control
set -m

# print uid
id


# wait for the nodes to spin up and create passwordless ssh
# /wait-for-it.sh openldap:636 --strict -- echo "openldap.example.org 636 is up"

# use this, if the docker container automatically terminates, but you want to keep it running
tail -f /dev/null

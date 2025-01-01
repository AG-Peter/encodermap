# Simulation Attender Tests

These tests build a complete HPC network with SLURM/LDAP/SSH/ENVIRONMENT MODULES

## Important stuff at the start

The nfs docker also needs the nfs kernel module. Do

```bash
sudo modprobe {nfs,nfsd,rpcsec_gss_krb5}
```

to check, whether nfs is enabled in your kernel. If no error is raised, nfs is enabled in your kernel.

## Quickstart

Start the container network with

```bash
$ bash start_slurm.sh
```

ssh into the client machine

```bash
$ ssh -p 222 localadmin@localhost
```

with the password `password`

from there, you can log in into an ssh-machine which has gromacs installed:

```bash
$ ssh gromacs@gromacs
```

with password `gromacs`

or you can ssh into the SLURM cluster with

```bash
$ ssh user01@cluster
```

with password `password1`.

## LDAP

The users on the docker-composed SLURM cluster are centrally managed by an LDAP server. This server running openldap is provided by the `bitnami/openldap` docker image.

### Graphical interface for LDAP

A graphical interface for the LDAP server is available under https://127.0.0.1:10443, use

 `````
 user: cn=admin,dc=example,dc=org
 password: adminpassword
 `````

for login credentials. The graphical interface is provided by the `osixia/phpldapadmin` image.

### Log into LDAP server

The LDAP server itself can be accessed via `docker exec -it openldap /bin/bash`. There, you can probe the LDAP configuration with some of the these commands:

List everything:

```bash
slapcat
ldapsearch -H ldapi:/// -Y EXTERNAL -b "cn=config" -LLL -Q
```

List organizational units:

```bash
ldapsearch -H ldapi:/// -Y EXTERNAL -b "dc=example,dc=org" -LLL -Q
```

List users:

```bash
ldapsearch -H ldapi:/// -Y EXTERNAL -b "ou=users,dc=example,dc=org" -LLL -Q
```

## MariaDB

SLURM needs access to a sql database. Inside the docker compose environment, the database is provided by the `mariadb` image.

### Graphical interface for MariaDB

You can inspect the database with a graphical user interface using these credentials on http://localhost:8080

```
Username: mysql_user
Password: sql_passw0rd
```

## Grafana

## Permissions

Sometimes permissions of the persistent directories can make docker containers fail. I fixed it with

```bash
chmod -R ugo+rwx openldap_data
```

But I have to admit, that this is not the most elegant solution. There needs to be a better way in the future.

FROM kevinsawade/slurm-base
MAINTAINER kevinsawade

# fix permissions on slurm stuff
RUN mkdir -p /etc/slurm-llnl
RUN chmod 777 /etc/slurm-llnl

# ENV DBD_ADDR=database \
#   DBD_HOST=database \
#   DBD_PORT=6819 \
#   STORAGE_HOST=database.local.dev \
#   STORAGE_PORT=3306 \
#   STORAGE_PASS=password \
#   STORAGE_USER=slurm

# set up slurmdbd
RUN mkdir -p /var/spool/slurm/d
RUN mkdir -p /var/log/slurm
RUN chown slurm: /var/spool/slurm/d
RUN chown slurm: /var/log/slurm
RUN mkdir -p /var/spool/slurmd

# set up slurmctld
RUN mkdir /var/spool/slurmctld
RUN chown slurm:slurm /var/spool/slurmctld
RUN chmod 755 /var/spool/slurmctld
RUN mkdir -p /var/log/slurm
RUN touch /var/log/slurm/slurmctld.log
RUN touch /var/log/slurm/slurm_jobacct.log /var/log/slurm/slurm_jobcomp.log
RUN chown -R slurm: /var/log/slurm/
RUN mkdir -p /var/run/slurm
RUN chown slurm: /var/run/slurm
RUN mkdir -p /usr//etc/slurm
RUN chown slurm:slurm /etc/slurm
COPY slurm_base/slurm.conf /usr/etc/slurm.conf
COPY slurm_base/cgroup.conf /usr/etc/cgroup.conf
COPY slurm_base/cgroup_allowed_devices_file.conf /usr/etc/cgroup_allowed_devices_file.conf


# ENV USE_SLURMDBD=true \
#   CLUSTER_NAME=snowflake \
#   CONTROL_MACHINE=controller \
#   SLURMCTLD_PORT=6817 \
#   SLURMD_PORT=6818 \
#   ACCOUNTING_STORAGE_HOST=database \
#   ACCOUNTING_STORAGE_PORT=6819 \
#   PARTITION_NAME=docker

# copy the database conf
COPY slurm_db/slurmdbd.conf /usr/etc/slurmdbd.conf
RUN chmod 600 /usr/etc/slurmdbd.conf
RUN chown -R slurm: /usr/etc/

# run the entrypoint
COPY slurm_db/slurm_db_entrypoint.sh slurm_db_entrypoint.sh
RUN chmod a+wrx slurm_db_entrypoint.sh
COPY slurm_db/slurm_acct_db.sql slurm_acct_db.sql
RUN chmod a+wrx slurm_acct_db.sql
EXPOSE 6817 6818 6819
CMD ["/slurm_db_entrypoint.sh"]

FROM kevinsawade/slurm_base
MAINTAINER kevinsawade

# install slurm-required packages
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y \
    openssl \
    libssl-dev \
    rpm \
    numactl \
    hwloc \
    lua5.3 \
    lua5.3-dev \
    lua5.3-filesystem \
    lua5.3-posix \
    man2html \
    libibmad-dev \
    libibumad-dev \
    slurmdbd \
    make \
    mysql-client

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
COPY slurmdbd.conf /etc/slurm-llnl/slurmdbd.conf

# set up slurmctld
RUN mkdir /var/spool/slurmctld
RUN chown slurm:slurm /var/spool/slurmctld
RUN chmod 755 /var/spool/slurmctld
RUN mkdir -p /var/log/slurm
RUN touch /var/log/slurm/slurmctld.log
RUN touch /var/log/slurm/slurm_jobacct.log /var/log/slurm/slurm_jobcomp.log
RUN chown -R slurm: /var/log/slurm/
COPY slurm.conf /etc/slurm-llnl/slurm.conf
COPY cgroup.conf /etc/slurm-llnl/cgroup.conf


# ENV USE_SLURMDBD=true \
#   CLUSTER_NAME=snowflake \
#   CONTROL_MACHINE=controller \
#   SLURMCTLD_PORT=6817 \
#   SLURMD_PORT=6818 \
#   ACCOUNTING_STORAGE_HOST=database \
#   ACCOUNTING_STORAGE_PORT=6819 \
#   PARTITION_NAME=docker

# run the entrypoint
COPY slurm_db_entrypoint.sh slurm_db_entrypoint.sh
RUN chmod a+wrx slurm_db_entrypoint.sh
COPY slurm_acct_db.sql slurm_acct_db.sql
RUN chmod a+wrx slurm_acct_db.sql
CMD ["/slurm_db_entrypoint.sh"]

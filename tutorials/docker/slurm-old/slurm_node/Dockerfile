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
    slurmd \
    slurmctld \
    slurmdbd \
    make \
    mysql-client

# fix permissions on slurm stuff
RUN mkdir -p /etc/slurm-llnl
RUN chmod 777 /etc/slurm-llnl

# set up slurmdbd
RUN mkdir -p /var/spool/slurm/d
RUN mkdir -p /var/log/slurm
RUN chown slurm: /var/spool/slurm/d
RUN chown slurm: /var/log/slurm
RUN mkdir -p /var/spool/slurmd
COPY slurmdbd.conf /etc/slurm-llnl/slurmdbd.conf

# # set up slurmctld
RUN mkdir /var/spool/slurmctld
RUN chown slurm:slurm /var/spool/slurmctld
RUN chmod 755 /var/spool/slurmctld
RUN mkdir -p /var/log/slurm
RUN touch /var/log/slurm/slurmctld.log
RUN touch /var/log/slurm/slurm_jobacct.log /var/log/slurm/slurm_jobcomp.log
RUN chown -R slurm: /var/log/slurm/
COPY slurm.conf /etc/slurm-llnl/slurm.conf
COPY cgroup.conf /etc/slurm-llnl/cgroup.conf
COPY cgroup_allowed_devices_file.conf /etc/slurm/cgroup_allowed_devices_file.conf

# set up slurmd
RUN mkdir -p /var/spool/slurmd
RUN chown slurm:slurm /var/spool/slurmd
RUN chmod 755 /var/spool/slurmd
RUN touch /var/log/slurmd.log
RUN chown -R slurm: /var/log/slurmd.log

# run the entrypoint
COPY slurm_node_entrypoint.sh slurm_node_entrypoint.sh
RUN chmod a+wrx slurm_node_w_prometheus_entrypoint.sh
CMD ["/slurm_node_entrypoint.sh"]

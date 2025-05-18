FROM kevinsawade/slurm_base
MAINTAINER kevinsawade

# install slurm-required packages
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install -y \
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
    mysql-client \
    python3 \
    python3-dev \
    python3-pip

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
RUN touch /var/log/slurmctld.log
RUN chown slurm: /var/log/slurmctld.log
RUN touch /var/log/slurm_jobacct.log /var/log/slurm_jobcomp.log
RUN chown slurm: /var/log/slurm*
RUN chown slurm: /var/log/slurm/
COPY slurm.conf /etc/slurm-llnl/slurm.conf
COPY cgroup.conf /etc/slurm-llnl/cgroup.conf
COPY cgroup_allowed_devices_file.conf /etc/slurm/cgroup_allowed_devices_file.conf

# install slurm-web
RUN pip install flask
RUN pip install cython
RUN git clone https://github.com/PySlurm/pyslurm.git
RUN ln -sf /usr/include/slurm-wlm/ /usr/include/slurm
RUN mkdir /usr/lib/slurm
RUN ln -sf /usr/lib/x86_64-linux-gnu /usr/lib/slurm/lib
# RUN python3 pyslurm/setup.py build
# RUN python3 pyslurm/setup.py build  --slurm-inc=/usr/include --slurm-lib=/usr/lib/slurm
# RUN python3 pyslurm/setup.py build --slurm-inc=/usr/include/slurm-wlm --slurm-lib=/usr/lib/x86_64-linux-gnu/
# RUN python3 pyslurm/setup.py install
# RUN git clone https://github.com/edf-hpc/slurm-web.git


# ENV USE_SLURMDBD=true \
#   CLUSTER_NAME=snowflake \
#   CONTROL_MACHINE=controller \
#   SLURMCTLD_PORT=6817 \
#   SLURMD_PORT=6818 \
#   ACCOUNTING_STORAGE_HOST=database \
#   ACCOUNTING_STORAGE_PORT=6819 \
#   PARTITION_NAME=docker

COPY slurm_master_entrypoint.sh slurm_master_entrypoint.sh
RUN chmod a+wrx slurm_master_entrypoint.sh
CMD ["/slurm_master_entrypoint.sh"]

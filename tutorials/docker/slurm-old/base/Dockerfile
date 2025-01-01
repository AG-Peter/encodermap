FROM ubuntu:20.04
MAINTAINER kevinsawade

# install munge packages
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    sudo

# set up munge for slurm following this tutorial:
# https://southgreenplatform.github.io/trainings/hpc/slurminstallation/
ENV MUNGEUSER=1001
RUN groupadd -g $MUNGEUSER munge
RUN useradd -m -c "Munge Uid 'N' Gid Emporium" -d /var/lib/munge -u $MUNGEUSER -g munge -s /sbin/nologin munge
ENV SLURMUSER=1002
RUN groupadd -g $SLURMUSER slurm
RUN useradd -m -c "SLURM workload manager" -d /var/lib/slurm -u $SLURMUSER -g slurm -s /bin/bash slurm

# create admin user which has sudo priviliges
RUN useradd -ms /bin/bash admin
RUN usermod -aG sudo admin
RUN echo "admin:adminpassword" | chpasswd
RUN echo 'admin  ALL=(ALL:ALL) ALL' >> /etc/sudoers
RUN chown -R admin /home/admin

# install munge packages
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    munge \
    git \
    wget \
    mysql-client \
    libmunge-dev \
    libpam0g-dev \
    perl \
    python3 \
    python3-dev \
    libreadline-dev \
    openssh-server \
    sshpass \
    netcat \
    libslurm-dev

# make changes to openssh
RUN mkdir -p /var/run/sshd
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin no" >> /etc/ssh/sshd_config
RUN if [ ! -d /var/run/sshd ] ; then mkdir /var/run/sshd ; ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -N '' ; fi
RUN chown admin /etc/ssh/ssh_host_rsa_key
RUN chown admin /etc/ssh/ssh_host_rsa_key.pub
COPY ssh_config /home/admin/.ssh/config
RUN mkdir -p /home/admin/.ssh \
    && chmod 0700 /home/admin/.ssh \
    && ssh-keygen -b 2048 -t rsa -f /home/admin/.ssh/id_rsa -q -N "" -C "admin@$(hostname)-$(date -I)" \
    && touch /home/admin/.ssh/authorized_keys \
    && cat /home/admin/.ssh/id_rsa.pub > /home/admin/.ssh/authorized_keys \
    && chmod 0640 home/admin/.ssh/authorized_keys \
    && chown -R admin /home/admin/.ssh \
    && chmod 0644 home/admin/.ssh/config


# copy the munge-key
COPY munge.key /etc/munge/munge.key

# fix permissions on munge stuff
RUN mkdir -p /run/munge
RUN chown -R munge: /etc/munge/ /var/log/munge/ /var/lib/munge/ /run/munge/
RUN chmod 0700 /etc/munge/ /var/log/munge/
RUN chmod 0711 /var/lib/munge/
RUN chmod 0755 /var/run/munge/
RUN chmod a+x /run/munge

# cgroup stuff
RUN mkdir -p /etc/slurm/cgroup
RUN chown -R slurm: /etc/slurm/cgroup

# expose some ports
EXPOSE 22 6817 6818 6819

# set up tini and wait-for-it.sh
ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /wait-for-it.sh
RUN chmod a+rx /wait-for-it.sh

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod a+rx /tini
ENTRYPOINT ["/tini", "--"]

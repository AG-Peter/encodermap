FROM kevinsawade/ldap-client

#####################
# NEEDED PACKAGES
#####################
# install munge packages
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    perl \
    gcc \
    build-essential\
    git \
    gnupg \
    make \
    munge \
    libmunge-dev \
    libpython3-dev \
    python3-pip \
    python3 \
    psmisc \
    mysql-client \
    libmysqlclient-dev \
    cgroup-tools \
    bash-completion
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y libgtk-3-dev gtk3.0 gtk2.0

#####################
# FIXING CGROUPv2 stuff
#####################
# See https://serverfault.com/questions/1053187/systemd-fails-to-run-in-a-docker-container-when-using-cgroupv2-cgroupns-priva

STOPSIGNAL SIGRTMIN+3
VOLUME [ "/tmp", "/run", "/run/lock" ]
WORKDIR /
# Remove unnecessary units
RUN rm -f /lib/systemd/system/multi-user.target.wants/* \
  /etc/systemd/system/*.wants/* \
  /lib/systemd/system/local-fs.target.wants/* \
  /lib/systemd/system/sockets.target.wants/*udev* \
  /lib/systemd/system/sockets.target.wants/*initctl* \
  /lib/systemd/system/sysinit.target.wants/systemd-tmpfiles-setup* \
  /lib/systemd/system/systemd-update-utmp*

#####################
# SLURM
#####################
# Install slurm
#  ./configure --enable-debug --prefix=/usr --sysconfdir=/etc/slurm --with-mysql_config=/usr/bin  --libdir=/usr/lib64 --with-cgroup=yes \
ARG SLURM_VERSION=slurm-22-05-8-1
RUN mkdir -p /slurm_install
RUN set -x \
    && git clone -b ${SLURM_VERSION} --single-branch --depth=1 https://github.com/SchedMD/slurm.git /slurm_install/slurm \
    && cd /slurm_install/slurm \
    && ./configure --enable-debug --prefix=/usr --with-munge=/usr --with-mysql_config=/usr/bin --with-cgroup=yes \
    && make \
    && make install \
    && ldconfig \
    && install -D -m644 etc/cgroup.conf.example /etc/slurm/cgroup.conf.example \
    && install -D -m644 etc/slurm.conf.example /etc/slurm/slurm.conf.example \
    && install -D -m644 etc/slurmdbd.conf.example /etc/slurm/slurmdbd.conf.example \
    && install -D -m644 contribs/slurm_completion_help/slurm_completion.sh /etc/profile.d/slurm_completion.sh \
    && cd .. \
    && groupadd -r --gid=990 slurm \
    && useradd -r -g slurm --uid=990 slurm \
    && mkdir -p /etc/sysconfig/slurm \
        /var/spool/slurmd \
        /var/run/slurmd \
        /var/run/slurmdbd \
        /var/lib/slurmd \
        /var/log/slurm \
        /data \
    && touch /var/lib/slurmd/node_state \
        /var/lib/slurmd/front_end_state \
        /var/lib/slurmd/job_state \
        /var/lib/slurmd/resv_state \
        /var/lib/slurmd/trigger_state \
        /var/lib/slurmd/assoc_mgr_state \
        /var/lib/slurmd/assoc_usage \
        /var/lib/slurmd/qos_usage \
        /var/lib/slurmd/fed_mgr_state \
    && chown -R slurm:slurm /var/*/slurm*

# copy the munge-key
COPY munge/munge.key /etc/munge/munge.key

# another try at fixing munge permissions
RUN mkdir -p /etc/munge/ /var/log/munge/ /var/lib/munge/ /run/munge/
RUN chown -R munge: /etc/munge/
RUN chown -R munge: /var/log/munge/
RUN chown -R munge: /var/lib/munge/
RUN chown -R munge: /run/munge/
RUN chmod 0700 /etc/munge/
RUN chmod 0700 /var/log/munge/
RUN chmod 0700 /var/lib/munge/
RUN chmod -R 0711 /run
RUN chown 105:103 /etc/munge/munge.key
RUN chmod 600 /etc/munge/munge.key

# expose some ports
EXPOSE 22

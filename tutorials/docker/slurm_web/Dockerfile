FROM kevinsawade/ldap-client

#####################
# NEEDED PACKAGES
#####################
# install munge packages
RUN #apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
#    wget \
#    bzip2 \
#    perl \
#    gcc \
#    build-essential\
#    git \
#    gnupg \
#    make \
#    munge \
#    libmunge-dev \
#    libpython3-dev \
#    python3-pip \
#    python3 \
#    psmisc \
#    mysql-client \
#    libmysqlclient-dev \
#    cgroup-tools

#####################
# INSTALL SLURM WEB
#####################

ARG SLURM_WEB_VERSION=2.4.0
RUN mkdir -p /slurm_web_install
RUN set -x \
    && wget https://github.com/rackslab/slurm-web/archive/refs/tags/$SLURM_WEB_VERSION.zip -O /slurm_web_install \
    && cd /slurm_web_install \

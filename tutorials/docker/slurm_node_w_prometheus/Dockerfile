FROM kevinsawade/slurm-node

#RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
#    wget \
#    bzip2 \
#    perl \

#####################
# GO
#####################
ARG GO_VERSION=1.15
RUN mkdir -p /golang
RUN set -x \
    && wget https://dl.google.com/go/go$GO_VERSION.linux-amd64.tar.gz -O /golang/go$GO_VERSION.linux-amd64.tar.gz \
    && cd /golang \
    && tar -xzvf go$GO_VERSION.linux-amd64.tar.gz
ENV PATH="${PATH}:/golang/go/bin"

#####################
# HASKELL_STACK
#####################
RUN curl -sSL https://get.haskellstack.org/ | sh

#####################
# Prometheus
#####################
RUN set -x \
    && mkdir -p /slurm_prometheus_exporter \
    && cd  /slurm_prometheus_exporter \
    && git clone https://github.com/flatironinstitute/slurm-prometheus-exporter.git . \
    && stack install --flag=slurm-prometheus-exporter:-pkgconfig --extra-lib-dirs=/usr/lib/slurm
RUN cp /root/.local/bin/slurm-exporter /usr/bin/slurm-exporter
RUN chmod a+wrx /usr/bin/slurm-exporter
RUN chown localadmin /usr/bin/slurm-exporter

#####################
# Encodermap
#####################
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 \
    python3.10-venv

RUN mkdir -p /usr/venv
RUN python3.10 -m venv /usr/venv/emap_env
RUN mkdir /app
COPY requirements.txt /app/requirements.txt
COPY md_requirements.txt /app/md_requirements.txt
COPY tests/test_requirements.txt /app/test_requirements.txt
WORKDIR /app
RUN /usr/venv/emap_env/bin/python -m pip install Cython==3.0.9
RUN /usr/venv/emap_env/bin/python -m pip install -r requirements.txt
RUN /usr/venv/emap_env/bin/python -m pip install -r md_requirements.txt
RUN /usr/venv/emap_env/bin/python -m pip install -r test_requirements.txt
COPY encodermap /app/encodermap
COPY description.md /app
COPY setup.py /app
COPY pyproject.toml /app
COPY versioneer.py /app
RUN /usr/venv/emap_env/bin/python -m pip install .

####################
# Jupyter
####################

RUN mkdir -p /home/encodermap
RUN chown -R 1000:1000 /home/encodermap
COPY tutorials/docker/slurm_node_w_prometheus/slurm_node_w_prometheus_entrypoint.sh /slurm_node_w_prometheus_entrypoint.sh
RUN chmod a+wrx /slurm_node_w_prometheus_entrypoint.sh
COPY tutorials/notebooks_starter /home/encodermap
COPY tutorials/notebooks_intermediate /home/encodermap
COPY tutorials/notebooks_MD /home/encodermap
COPY tutorials/notebooks_customization /home/encodermap
WORKDIR /home/encodermap

# run the entrypoint
EXPOSE 8090 8888
CMD ["/slurm_node_w_prometheus_entrypoint.sh"]

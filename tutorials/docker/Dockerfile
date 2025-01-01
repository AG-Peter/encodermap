FROM kevinsawade/slurm-node

#####################
# PIPENV
#####################
RUN mkdir -p /app
WORKDIR /app
COPY requirements.txt /app/base_requirements.txt
COPY md_requirements.txt /app/md_requirements.txt
COPY sphinx_requirements.txt /app/docs_requirements.txt
COPY test_requirements.txt /app/test_requirements.txt
RUN cat /app/*requirements.txt > /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt
RUN #set -ex \
#    && mkdir -p /usr/local/anaconda \
#    && wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh -O /usr/local/anaconda/Anaconda3-2023.03-1-Linux-x86_64.sh \
#    && bash /usr/local/anaconda/Anaconda3-2023.03-1-Linux-x86_64.sh -b -p /usr/local/anaconda3 \
#    && /usr/local/anaconda3/bin/python -m pip install -r requirements.txt

#####################
# TESTS
#####################
RUN mkdir -p /sh_libs
COPY sh_libs/liblog.sh /sh_libs/liblog.sh
COPY water_simulation/production.tpr /app/production.tpr
COPY water_simulation/production_fails.tpr /app/production_fails.tpr
COPY water_simulation/production_short.tpr /app/production_short.tpr

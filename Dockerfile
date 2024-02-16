# then jupyter datascience
FROM jupyter/tensorflow-notebook:python-3.9.10
WORKDIR /app

# install requirements
ADD encodermap /home/jovyan/encodermap/encodermap
ADD tutorials /home/jovyan/tutorials
COPY setup.py /home/jovyan/encodermap/setup.py
COPY description.md /home/jovyan/encodermap/description.md
COPY requirements.txt /home/jovyan/encodermap/requirements.txt
COPY md_requirements.txt /home/jovyan/encodermap/md_requirements.txt
RUN if [ -f /home/jovyan/tutorials/notebooks_intermeidate/glu7.xtc ] ; then rm /home/jovyan/tutorials/notebooks_intermeidate/glu7.xtc ; fi
COPY tests/data/glu7.xtc /home/jovyan/tutorials/notebooks_intermediate/glu7.xtc
USER root
RUN chown -R ${NB_UID} /home/jovyan

USER jovyan
WORKDIR /home/jovyan/encodermap

RUN pip install -r requirements.txt
RUN pip install -r md_requirements.txt
RUN pip install .
RUN pip install --upgrade numpy==1.23.4

WORKDIR /home/jovyan/tutorials

# expose tensorboard port
EXPOSE 6006

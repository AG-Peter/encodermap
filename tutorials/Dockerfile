FROM tensorflow/tensorflow:1.13.1-py3-jupyter

RUN rm -rf tensorflow-tutorials &&\
    pip install encodermap

RUN pip install jupyter_contrib_nbextensions &&\
    jupyter contrib nbextension install &&\
    jupyter nbextension enable spellchecker/main --system

RUN pip install nglview &&\
    jupyter-nbextension enable nglview --py --system

EXPOSE 8888
EXPOSE 6006


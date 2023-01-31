FROM tensorflow/tensorflow:1.15.0-py3-jupyter

RUN rm -rf tensorflow-tutorials &&\
    pip install encodermap==2.0.1

RUN pip install jupyter_contrib_nbextensions &&\
    jupyter contrib nbextension install &&\
    jupyter nbextension enable spellchecker/main --system

RUN pip install nglview &&\
    jupyter-nbextension enable nglview --py --system

EXPOSE 8888
EXPOSE 6006

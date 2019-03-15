FROM tensorflow/tensorflow:1.13.1-py3-jupyter

RUN rm -rf tensorflow-tutorials &&\
    pip install encodermap==0.0.10

RUN pip install jupyter_contrib_nbextensions &&\
    jupyter contrib nbextension install &&\
    jupyter nbextension enable spellchecker/main --system

EXPOSE 8888
EXPOSE 6006


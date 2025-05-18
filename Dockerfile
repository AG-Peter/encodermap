# inherit from python's base image
FROM python:3.12
LABEL maintainer="kevinsawade" version="0.0.1"

# update pip
RUN apt-get update
RUN pip install --upgrade pip

# create non-root user
RUN adduser encodermap_user

# according to the
WORKDIR /app

# install encodermap in /app
COPY requirements.txt /app/requirements.txt
COPY md_requirements.txt /app/md_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r md_requirements.txt
RUN pip install --no-cache-dir notebook ipython

# copy encodermap
ADD encodermap /app/encodermap
COPY setup.py /app/setup.py
COPY pyproject.toml /app/pyproject.toml
COPY versioneer.py /app/versioneer.py
COPY description.md /app/description.md
COPY requirements.txt /app/requirements.txt
RUN pip install .

# change into homne and copy tutorials
ADD tutorials /home/encodermap_user/tutorials
WORKDIR /home/encodermap_user/tutorials

# expose tensorboard and jupyter notebook port port
EXPOSE 6006
EXPOSE 8888
EXPOSE 8787

# run
# CMD ["which", "python"]
CMD ["/usr/local/bin/python", "-m", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]

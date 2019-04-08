# Tutorials

There are two options how to do the tutorials. 
You can either install all requirements directly on your system or you can run the tutorials in an isolated Docker environment where all requirements for the tutorials are set up for you.

## Inside Docker

If Docker is not installed on your system already, first [install Docker](https://docs.docker.com/install/).
If you are on a linux system you should also [add yor user to the docker group](https://docs.docker.com/install/linux/linux-postinstall/) to get permissions to start docker containers.

Clone this repository to some location of your choice:
```bash
git clone https://github.com/AG-Peter/encodermap.git
```

Go into the tutorial directory:
```bash
cd encodermap/tutorials
```

Build the EncoderMap Docker container:
```bash
docker build --tag encodermap .
```

and run the container:
```bash
docker run --rm -u $(id -u):$(id -g) -it -p 6006:6006 -p 8888:8888 -v $(pwd)/notebooks:/tf/notebooks --name emap encodermap
```
You might need to adapt the above run command if you are not on a Linux system. Try to leave away the ```-u $(id -u):$(id -g)``` and specify the notebooks path in the notation of your operating system with: ```-v path-to-notebooks:/tf/notebooks```

This should start a jupyter-notebook server and you should be able to open the tutorial notebooks in your browser.

## Outside Docker
To run the tutorials outside the provided Docker container the following software needs to be installed on your system:
* [TensorFlow](https://www.tensorflow.org/install)
* TensorBoard (usually comes along with TensorFlow)
* [EncoderMap](https://github.com/AG-Peter/encodermap)
* [Jupyter-Notebook](https://jupyter.org/) (with python3)

Once everything is installed, clone this repository to some location of your choice:
```bash
git clone https://github.com/AG-Peter/encodermap.git
```

Go into the tutorial directory:
```bash
cd encodermap/tutorials
```

Start jupyter-notebook:
```bash
jupyter-notbook
```
This should start a jupyter-notebook server and you should be able to open the tutorial notebooks in your browser.
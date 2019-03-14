docker run --rm -u $(id -u):$(id -g) -it -p 6006:6006 -p 8888:8888 -v $(pwd)/notebooks:/tf/notebooks --name emap encodermap
# docker exec -it emap bash
# docker run -u $(id -u):$(id -g) -it -p 6006:6006  encodermap


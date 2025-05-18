#!/bin/bash

set -m

docker build -t modules-test --build-arg ENVIRONMENT_MODULES_VERSION="5.2.0" .
docker run --rm modules-test

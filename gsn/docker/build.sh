#!/bin/bash

IMAGE="gsn:sn8"

DOCKER_DIR=$(dirname "$(readlink -f "$0")")
echo "Docker directory: $DOCKER_DIR"

docker build -t ${IMAGE} "${DOCKER_DIR}"
#!/bin/bash

IMAGE="gsn:sn8"
CONTAINER="gsn-sn8"

DOCKER_DIR=$(dirname "$(readlink -f "$0")")
echo "Docker directory: $DOCKER_DIR"
PROJ_DIR=$(dirname "$DOCKER_DIR")

DATA_DIR="$PROJ_DIR/inputs/dataset"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist. Run 'python prepare_dataset.py' scripts first!!!"
    exit 1
fi

docker run --runtime nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
	-v "${PROJ_DIR}":/gsn-sn8 \
	--name ${CONTAINER} \
	${IMAGE} /bin/bash
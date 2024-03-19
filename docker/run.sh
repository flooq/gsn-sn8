#!/bin/bash

IMAGE="gsn:sn8"
CONTAINER="gsn-sn8"

DATA_DIR="$HOME/.spacenet8/dataset"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' does not exist. Run 'scripts/prepare_dataset.sh' scripts first!!!"
    exit 1
fi

DOCKER_DIR=$(dirname "$(readlink -f "$0")")
echo "Docker directory: $DOCKER_DIR"
PROJ_DIR=$(dirname "$DOCKER_DIR")

docker run --runtime nvidia --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
	-v "${PROJ_DIR}":/work \
	-v "${DATA_DIR}":/data \
	--name ${CONTAINER} \
	${IMAGE} /bin/bash
#!/bin/bash

# Run this script to prepare fresh dataset from tarballs.

spacenet8_directory="$HOME/.spacenet8"

tarballs_directory="$spacenet8_directory/tarballs"

if [ ! -d "$tarballs_directory" ]; then
    echo "Directory '$tarballs_directory' does not exist."

    current_dir=$(dirname "$(readlink -f "$0")")
    "${current_dir}"/aws_download_tarballs.sh
fi

dataset_directory="$spacenet8_directory/dataset"

if [ -d "$dataset_directory" ]; then
    echo "Removing directory content: $dataset_directory"
    # shellcheck disable=SC2115
    rm -r "$dataset_directory"/*
else
    echo "Creating directory: $dataset_directory"
    mkdir -p "$dataset_directory"
fi


# shellcheck disable=SC2164
cd "$dataset_directory"

mkdir Germany_Training_Public
tar -xvzf "$tarballs_directory/Germany_Training_Public.tar.gz" --exclude=Germany_Training_Public.tar.gz -C Germany_Training_Public

mkdir Louisiana-East_Training_Public
tar -xvzf "$tarballs_directory/Louisiana-East_Training_Public.tar.gz" -C Louisiana-East_Training_Public

mkdir Louisiana-West_Test_Public
tar -xvzf "$tarballs_directory/Louisiana-West_Test_Public.tar.gz" -C Louisiana-West_Test_Public


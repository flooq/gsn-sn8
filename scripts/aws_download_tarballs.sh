#!/bin/bash

# Run this script once to download zipped dataset from AWS.
# All files will be stored in $HOME/.spacenet8/tarballs directory.
# Generally, you do not have to run it directly, because
# this script is executed from 'prepare_dataset.sh' script.

spacenet8_directory="$HOME/.spacenet8"

if [ ! -d "$spacenet8_directory" ]; then
    echo "Creating directory: $spacenet8_directory"
    mkdir -p "$spacenet8_directory"
fi

tarballs_directory="$spacenet8_directory/tarballs"

if [ ! -d "$tarballs_directory" ]; then
    echo "Creating directory: $tarballs_directory"
    mkdir -p "$tarballs_directory"
fi

# shellcheck disable=SC2164
cd "$tarballs_directory"
echo "Downloading tarballs to directory: $tarballs_directory"

aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet/SN8_floods/tarballs/Germany_Training_Public.tar.gz .
aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet/SN8_floods/tarballs/Louisiana-East_Training_Public.tar.gz .
aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet/SN8_floods/tarballs/Louisiana-West_Test_Public.tar.gz .


#!/bin/bash

# Uncomment if conda command is not available in the PATH
# source ~/anaconda3/etc/profile.d/conda.sh

current_dir=$(dirname "$(readlink -f "$0")")
conda env create -f "$current_dir/environment.yaml"

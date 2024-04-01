# SpaceNet-8 GSN
Each year, natural disasters such as hurricanes, tornadoes, earthquakes and floods significantly damage infrastructure and result in loss of life, property and billions of dollars. As these events become more frequent and severe, there is an increasing need to rapidly develop maps and analyze the scale of destruction to better direct resources and first responders. To help address this need, the SpaceNet 8 Flood Detection Challenge will focus on infrastructure and flood mapping related to hurricanes and heavy rains that cause route obstructions and significant damage. The goal of SpaceNet 8 is to leverage the existing repository of datasets and algorithms from SpaceNet Challenges 1-7 (https://spacenet.ai/datasets/) and apply them to a real-world disaster response scenario, expanding to multiclass feature extraction and characterization for flooded roads and buildings and predicting road speed.

Detecting flooded roads and buildings.

## Dataset

The data is hosted on AWS. Download and unzip dataset to inputs directory.

`python prepare_dataset.py`

In case, you need a fresh dataset later, run this script again. 
It should be refreshed fast as data will not be downloaded from AWS again. 

In case, you need to download tarballs from AWS only then run  

`python aws_tarballs.py`

## Setup
There are 2 preferred ways to build and run this project:  
- docker
- conda
  
### Docker

1. Install nvidia-container-toolkit and configure docker as described here
   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

2. Build docker image  
`./docker/build.sh`

3. Create and run the container  
`./docker/run.sh`

    Mounted volumes:
   - /gsn-sn8 - this project repository with dataset inside

### Conda

1. There is a command in provided baseline Dockerfile (install if needed)
`apt-get install libpq-dev gdal-bin libgdal-dev -y`

2. Create environment gsn-sn8

   `./conda/create.sh`

3. Activate environment

   `conda activate gsn-sn8`

4. (optional) Install kernel for jupyter notebook

   `python -m ipykernel install --user --name gsn-sn8 --display-name "Deep neural networks: postgraduate studies"`

### Baseline runner

Hydra framework has been used to run experiments.
Hydra configuration is located in code/baseline_runner/conf 

#### Preprocess dataset
Run once for all experiments. 
It creates directories 'prepped_cleaned' and 'masks' in each 'aoi_dir' directory.

 `python code/baseline_runner/preprocess.py`

#### Environment variables

'GSN_SN8_DIR' environment variable is set to project repository.

It might be necessary to set PROJ_LIB env variable. See code/gsn_sn8.py

#### Train and inference network

Examples:
1. Train foundation network

    `python code/baseline_runner/train_foundation.py`

2. Train flood network

   `python code/baseline_runner/train_flood.py`

3. Train foundation & flood network

   `python code/baseline_runner/train_all.py`

4. Train with overridden values
 
   `python code/baseline_runner/train_foundation.py foundation=unet`

   `python code/baseline_runner/train_all.py foundation=unet flood=unet_siamese`

5. Foundation eval with latest train execution

   `python code/baseline_runner/eval_foundation.py`

6. Flood eval with latest train execution

   `python code/baseline_runner/eval_flood.py`

7. Run all (preprocess, train and eval)

   `python code/baseline_runner/run_all.py`


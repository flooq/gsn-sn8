# SpaceNet-8 GSN
Each year, natural disasters such as hurricanes, tornadoes, earthquakes and floods significantly damage infrastructure and result in loss of life, property and billions of dollars. As these events become more frequent and severe, there is an increasing need to rapidly develop maps and analyze the scale of destruction to better direct resources and first responders. To help address this need, the SpaceNet 8 Flood Detection Challenge will focus on infrastructure and flood mapping related to hurricanes and heavy rains that cause route obstructions and significant damage. The goal of SpaceNet 8 is to leverage the existing repository of datasets and algorithms from SpaceNet Challenges 1-7 (https://spacenet.ai/datasets/) and apply them to a real-world disaster response scenario, expanding to multiclass feature extraction and characterization for flooded roads and buildings and predicting road speed.

Detecting flooded roads and buildings.

## Dataset

The data is hosted on AWS. Download and unzip dataset to $HOME/.spacenet8 directory.

`./scripts/prepare_dataset.sh`

In case, you need a fresh dataset later, run this script again. 
It should be refreshed fast as data will not be downloaded from AWS again. 

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
   - /work - this project repository
   - /data - local $HOME/.spacenet8/dataset

### Conda

1. There is a command in provided baseline Dockerfile 
`apt-get install postgresql-client libpq-dev gdal-bin libgdal-dev curl -y`
Install if needed

2. Create environment gsn-sn8
`./conda/create.sh`

3. Activate environment
`conda activate gsn-sn8`

4. (optional) Install kernel for jupyter notebook
`python -m ipykernel install --user --name gsn-sn8 --display-name "Deep neural networks: postgraduate studies"`

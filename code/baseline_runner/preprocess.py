import os
import subprocess
import sys
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils


# It creates directories 'prepped_cleaned' and 'masks' in each 'aoi_dir' directory.
# We do not use hydra here as it creates a new directory for each execution.
# We want to avoid it.
def preprocess():

    baseline_directory = utils.baseline_directory()

    with open(os.path.join(utils.baseline_runner_directory(), "conf", "config.yaml"), "r") as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # required for create_masks when run from Intellij Idea with conda environment
    os.environ['GDAL_DATA'] = cfg['env']['GDAL_DATA']
    os.environ['PROJ_LIB'] = cfg['env']['PROJ_LIB']

    dataset_dir = ["--root_dir", utils.dataset_directory()]
    aoi_dirs = ["--aoi_dirs"] + cfg['aoi_dirs']

    # geojson_prep
    geojson_prep = ["python", os.path.join(baseline_directory, "data_prep", "geojson_prep.py")]
    geojson_prep_args = dataset_dir + aoi_dirs
    subprocess.run(geojson_prep + geojson_prep_args)

    # create_masks
    create_masks = ["python", os.path.join(baseline_directory, "data_prep", "create_masks.py")]
    create_masks_args = dataset_dir + aoi_dirs
    subprocess.run(create_masks + create_masks_args)


if __name__ == "__main__":
    preprocess()

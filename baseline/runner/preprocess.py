import os
import subprocess
import yaml
import env


# It creates directories 'prepped_cleaned' and 'masks' in each 'aoi_dir' directory.
# We do not use hydra here as it creates a new directory for each execution.
# We want to avoid it.
def preprocess():
    with open(os.path.join(env.baseline_runner_directory(), "conf", "config.yaml"), "r") as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    preprocess_cfg(cfg)


def preprocess_cfg(cfg):
    baseline_directory = env.baseline_directory()
    dataset_dir = ["--root_dir", env.dataset_directory()]
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

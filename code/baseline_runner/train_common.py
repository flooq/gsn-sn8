import sys

from omegaconf import DictConfig
import subprocess
import os
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils


def split_train_val_dataset(cfg: DictConfig):
    print(f"Splitting dataset train/val into {cfg.output_dir}")
    print(f"Val percent {cfg.val_percent}")
    print(f"Area of interests (AOI):")
    for aoi in cfg.aoi_dirs:
        print(f"\t{aoi}")
    baseline_directory = utils.baseline_directory()
    dataset_dir = ["--root_dir", utils.dataset_directory()]
    aoi_dirs = ["--aoi_dirs"] + cfg.aoi_dirs
    out_csv_basename = ["--out_csv_basename", cfg.output_csv_basename]
    val_percent = ["--val_percent", cfg.val_percent]
    output_dir = ["--out_dir", cfg.output_dir]

    # generate_train_val_test_csvs
    generate_train_val_test_csvs = \
        ["python", os.path.join(baseline_directory, "data_prep", "generate_train_val_test_csvs.py")]
    generate_train_val_test_csvs_args = (
            dataset_dir + aoi_dirs + out_csv_basename + val_percent + output_dir)
    subprocess.run(generate_train_val_test_csvs + generate_train_val_test_csvs_args)


def train(cfg: DictConfig, network, script):
    if cfg.enable_env:
        print(f"Setting GDAL_DATA & PROJ_LIB environment variables")
        # required for flood network when run from Intellij Idea with conda environment
        os.environ['GDAL_DATA'] = cfg['env']['GDAL_DATA']
        os.environ['PROJ_LIB'] = cfg['env']['PROJ_LIB']

    network_capitalize = network.capitalize()
    print("==================================================================================")
    print(f"{network_capitalize} network with model {cfg[network].model} training started...")
    start_time = time.time()

    print(f"Parameters used for training:")
    for key, value in cfg[network].items():
        print(f"\t{key}: {value}")
    network_dir = os.path.join(cfg.output_dir, network)
    os.makedirs(network_dir, exist_ok=True)
    output_cvs_prefix = os.path.join(cfg.output_dir, cfg.output_csv_basename)
    train_csv = output_cvs_prefix + "_train.csv"
    val_csv = output_cvs_prefix + "_val.csv"
    train_csv = ["--train_csv", train_csv]
    val_csv = ["--val_csv", val_csv]
    save_dir = ["--save_dir", network_dir]
    model_name = ["--model_name", cfg[network].model]
    learning_rate = ["--lr", cfg[network].learning_rate]
    epochs = ["--n_epochs", cfg[network].epochs]
    batch_size = ["--batch_size", cfg[network].batch_size]
    gpu = ["--gpu", cfg[network].gpu]
    # train_network
    baseline_directory = utils.baseline_directory()
    train_network = ["python", os.path.join(baseline_directory, script)]
    train_network_args = (train_csv + val_csv + save_dir +
                                      model_name + learning_rate + epochs + batch_size + gpu)
    subprocess.run(train_network + train_network_args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"{network_capitalize} network with model {cfg[network].model} training finished. ")
    if hours > 0:
        print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds.")
    else:
        print(f"Elapsed time: {minutes} minutes, {seconds} seconds.")
    print("==================================================================================")

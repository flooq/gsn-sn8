import sys

from omegaconf import DictConfig
import subprocess
import os
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
import gsn_sn8


def split_train_val_dataset(cfg: DictConfig):
    print(f"Splitting dataset train/val into {cfg.output_dir}")
    print(f"Val percent {cfg.val_percent}")
    print(f"Area of interests (AOI):")
    for aoi in cfg.aoi_dirs:
        print(f"\t{aoi}")
    baseline_directory = gsn_sn8.baseline_directory()
    dataset_dir = ["--root_dir", gsn_sn8.dataset_directory()]
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
    start_time = time.time()
    print_start_message(cfg, network, "training")

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
    baseline_directory = gsn_sn8.baseline_directory()
    train_network = ["python", os.path.join(baseline_directory, script)]
    train_network_args = (train_csv + val_csv + save_dir +
                          model_name + learning_rate + epochs + batch_size + gpu)
    subprocess.run(train_network + train_network_args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print_end_message(cfg, network, "training", elapsed_time)


# network = foundation or flood
# script = foundation_eval.py or flood_eval.py
# output_train_dir = /some/path/to/outputs/train_[all|foundation|flood]/%Y-%m-%d_%H-%M-%S
def eval(cfg: DictConfig, network: str, script: str, output_train_dir: str):
    start_time = time.time()
    print_start_message(cfg, network, "inference")

    network_dir = os.path.join(output_train_dir, network)
    best_model_path = os.path.join(network_dir, get_subdirectory_name(network_dir), "best_model.pth")
    print(f"Inference on model {best_model_path}")

    model_path = ["--model_path", best_model_path]
    output_cvs_prefix = os.path.join(output_train_dir, cfg.output_csv_basename)
    val_csv = output_cvs_prefix + "_val.csv"
    val_csv = ["--in_csv", val_csv]
    # TODO
    #   read from .hydra train_* subdirectory to be able to inference from any directory
    #   and to be resistant to any changes in hydra config.yaml
    #   now we assume that eval follows train
    model_name = ["--model_name", cfg[network].model]
    gpu = ["--gpu", cfg[network].gpu]

    baseline_directory = gsn_sn8.baseline_directory()
    eval_network = ["python", os.path.join(baseline_directory, script)]
    eval_network_common_args = model_path + val_csv + model_name + gpu

    network_eval_preds_dir = os.path.join(cfg.output_dir, network + "_eval_preds")
    os.makedirs(network_eval_preds_dir, exist_ok=True)
    print('save_preds_dir', network_eval_preds_dir)
    save_preds_dir = ["--save_preds_dir", network_eval_preds_dir]
    subprocess.run(eval_network + eval_network_common_args + save_preds_dir)

    network_eval_fig_dir = os.path.join(cfg.output_dir, network + "_eval_fig")
    os.makedirs(network_eval_fig_dir, exist_ok=True)
    print('save_fig_dir', network_eval_fig_dir)
    save_fig_dir = ["--save_fig_dir", network_eval_fig_dir]
    subprocess.run(eval_network + eval_network_common_args + save_fig_dir)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print_end_message(cfg, network, "inference", elapsed_time)


def print_start_message(cfg, network, action):
    network_capitalize = network.capitalize()
    print("==================================================================================")
    print(f"{network_capitalize} network with model {cfg[network].model} {action} started...")
    print(f"Parameters used for eval:")
    for key, value in cfg[network].items():
        print(f"\t{key}: {value}")


def print_end_message(cfg, network, action, elapsed_time):
    network_capitalize = network.capitalize()
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"{network_capitalize} network with model {cfg[network].model} {action} finished. ")
    if hours > 0:
        print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds.")
    else:
        print(f"Elapsed time: {minutes} minutes, {seconds} seconds.")
    print("==================================================================================")


def get_subdirectory_name(directory_path):
    items = os.listdir(directory_path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    if len(subdirectories) == 1:
        return subdirectories[0]
    elif len(subdirectories) == 0:
        raise Exception("No subdirectories found in the specified directory.")
    else:
        raise Exception("Multiple subdirectories found in the specified directory.")


def get_latest_hydra_job_execution(job_names):
    all_directories = []

    outputs_dir = gsn_sn8.outputs_directory()
    for job_name in job_names:
        directory_path = os.path.join(outputs_dir, job_name)
        if os.path.isdir(directory_path):
            directories = [os.path.join(directory_path, d) for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
            all_directories.extend(directories)

    if not all_directories:
        raise Exception(f"No directories found for given job ${job_names}.")

    latest_directory = max(all_directories, key=lambda x: datetime.strptime(os.path.basename(x), '%Y-%m-%d_%H-%M-%S'))

    return latest_directory

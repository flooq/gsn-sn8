import os
from pathlib import Path

project_directory = str(Path(__file__).resolve().parent.parent)
os.environ['GSN_SN8_DIR'] = project_directory
print(f"Setting environment variable GSN_SN8_DIR={os.environ['GSN_SN8_DIR']}")


def parent_baseline_directory():
    return os.path.join(project_directory, "baseline")


def baseline_directory():
    return os.path.join(parent_baseline_directory(), "baseline")


def baseline_runner_directory():
    return os.path.join(parent_baseline_directory(), "runner")


def dataset_directory():
    return os.path.join(project_directory, "inputs", "dataset")


def outputs_directory():
    return os.path.join(project_directory, "outputs", "baseline")

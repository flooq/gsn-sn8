import os
from pathlib import Path


def code_directory():
    return os.path.join(project_directory(), "code")


def baseline_directory():
    return os.path.join(code_directory(), "baseline")


def baseline_runner_directory():
    return os.path.join(code_directory(), "baseline_runner")


def dataset_directory():
    return os.path.join(project_directory(), "inputs", "dataset")


def project_directory():
    return str(Path(__file__).resolve().parent.parent)

import os
from pathlib import Path

project_directory = str(Path(__file__).resolve().parent.parent)
os.environ['GSN_SN8_DIR'] = project_directory
print(f"Setting environment variable GSN_SN8_DIR={os.environ['GSN_SN8_DIR']}")

# GDAL_DATA & PROJ_LIB are required when run from Intellij Idea & conda env
# otherwise can be ignored
# see issue https://github.com/PDAL/PDAL/issues/2544
# e.g. in my case it helped to set:
#   GDAL_DATA: /home/pawel/anaconda3/envs/gsn-sn8/share/gdal
#   PROJ_LIB: /home/pawel/anaconda3/envs/gsn-sn8/share/proj
if 'GDAL_DATA' in os.environ:
    print(f"Environment variable GDAL_DATA={os.environ['GDAL_DATA']}")
else:
    print(f"No environment variable GDAL_DATA, which is required in some environments!")
if 'PROJ_LIB' in os.environ:
    print(f"Environment variable PROJ_LIB={os.environ['PROJ_LIB']}")
else:
    print(f"No environment variable PROJ_LIB, which is required in some environments!")


def parent_baseline_directory():
    return os.path.join(project_directory, "baseline")


def baseline_directory():
    return os.path.join(parent_baseline_directory(), "baseline")


def baseline_runner_directory():
    return os.path.join(parent_baseline_directory(), "runner")


def dataset_directory():
    return os.path.join(project_directory, "inputs", "dataset")


def outputs_directory():
    return os.path.join(project_directory, "outputs")

import os
import subprocess


# Run this script once to download zipped dataset from AWS.
# All files will be stored in input/tarballs directory.
# Generally, you do not have to run it directly, because
# this script is executed from 'prepare_dataset.py' script.

tarballs = ["Germany_Training_Public.tar.gz",
            "Louisiana-East_Training_Public.tar.gz",
            "Louisiana-West_Test_Public.tar.gz"]


def download():
    current_script = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script)

    inputs_directory = os.path.join(current_dir, "inputs")
    tarballs_directory = os.path.join(inputs_directory, "tarballs")
    if not os.path.isdir(tarballs_directory):
        print(f"Creating directory: {tarballs_directory}")
        os.makedirs(tarballs_directory)

    os.chdir(tarballs_directory)
    print(f"Downloading tarballs to directory: {tarballs_directory}")

    # Download tarballs using AWS CLI
    for tarball in tarballs:
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", "s3://spacenet-dataset/spacenet/SN8_floods/tarballs/" + tarball, "."])


if __name__ == "__main__":
    download()

import os
import shutil
import aws_tarballs
from aws_tarballs import tarballs


# Run this script to prepare fresh dataset from tarballs.
def main():
    current_script = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script)

    inputs_directory = os.path.join(current_dir, "inputs")
    tarballs_directory = os.path.join(inputs_directory, "tarballs")
    dataset_directory = os.path.join(inputs_directory, "dataset")

    if not os.path.isdir(tarballs_directory):
        aws_tarballs.download()

    if os.path.isdir(dataset_directory):
        print(f"Removing directory content: {dataset_directory}")
        shutil.rmtree(dataset_directory)

    os.makedirs(dataset_directory, exist_ok=True)
    os.chdir(dataset_directory)
    for tarball in tarballs:
        dirname = os.path.splitext(os.path.splitext(tarball)[0])[0]
        os.makedirs(dirname, exist_ok=True)
        os.system(f"tar -xvzf {os.path.join(tarballs_directory, tarball)} -C {dirname}")


if __name__ == "__main__":
    main()

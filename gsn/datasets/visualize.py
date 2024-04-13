import click
import random

from datasets import SN8Dataset


@click.command()
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--output_dir", type=click.Path(), required=True)
def main(data_dir, output_dir, n_images=5, randomize=True):
    dataset = SN8Dataset(data_dir)
    idx = random.choices(range(len(dataset)), k=n_images) if randomize else range(n_images)
    for i in idx:
        preimg, postimg, building, road, roadspeed, flood = dataset[i]


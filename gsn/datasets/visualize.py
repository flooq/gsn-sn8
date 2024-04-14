import click
import random
import os
import torch
import numpy as np
import cv2

from datasets import SN8Dataset


COLORS = {
    'building': np.array((0, 0, 255)),
    'road': np.array((0, 255, 255)),
    'flood': np.array((255, 0, 0))
}
def draw_mask(img, mask, mask_type):
    color = np.array(COLORS[mask_type])
    img[mask] = (img[mask] * 0.5 + color * 0.5).astype(np.uint8)

def draw_preimage(img, road, building):
    draw_mask(img, road, "road")
    draw_mask(img, building, "building")
    return img

def draw_postimage(img, flood, road, building):
    draw_mask(img, flood, "flood")
    draw_mask(img, road, "road")
    draw_mask(img, building, "building")
    return img

@click.command()
@click.option("--data_csv", type=click.Path(exists=True), required=True)
@click.option("--output_dir", type=click.Path(), required=True)
@click.option("--n_images", type=int, default=5)  # set n_images to a large number to visualize all images
@click.option("--randomize/--first", type=bool, default=True)
def main(data_csv, output_dir, n_images, randomize):
    dataset = SN8Dataset(data_csv, data_to_load=["preimg","postimg","building","road","flood"])
    os.makedirs(output_dir, exist_ok=True)
    n_images = max(n_images, len(dataset))
    idx = random.choices(range(len(dataset)), k=n_images) if randomize else range(n_images)
    for i in idx:
        try:
            preimg, postimg, building, road, roadspeed, flood = dataset[i] # roadspeed=0
        except Exception:
            print('File not found')
            continue
        road = torch.squeeze(road).numpy().astype(bool)
        building = torch.squeeze(building).numpy().astype(bool)
        # flood = flood.numpy()[:, :, flood_idx].astype(bool)
        flood = flood.numpy().max(axis=2).astype(bool)
        pre_vis = draw_preimage(preimg.numpy(), road, building)
        post_vis = draw_postimage(postimg.numpy(), flood, road, building)
        preimg_filename = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_PRE.png")
        postimg_filename = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST.png")
        cv2.imwrite(os.path.join(output_dir, preimg_filename), pre_vis)
        cv2.imwrite(os.path.join(output_dir, postimg_filename), post_vis)


if __name__ == '__main__':
    main()
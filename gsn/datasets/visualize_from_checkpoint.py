import click
import random
import os
import numpy as np
import cv2

from datasets import SN8Dataset

import torch

from models.baseline_unet import UNetSiamese
from models.flood_model import FloodModel
from models.flood_simple_model import FloodSimpleModel

COLORS = {
    'building': np.array((0, 0, 255)),
    'road': np.array((0, 255, 255)),
    'non-flood-building': np.array((0, 0, 255)),
    'flood-building': np.array((255, 0, 255)),
    'non-flood-road': np.array((0, 255, 255)),
    'flood-road': np.array((255, 255, 0))
}

def draw_mask(img, mask, mask_type):
    color = np.array(COLORS[mask_type])
    img[mask] = (img[mask] * 0.5 + color * 0.5).astype(np.uint8)

def draw_preimage(img, road, building):
    draw_mask(img, road, "road")
    draw_mask(img, building, "building")
    return img

def draw_postimage(img, flood):
    draw_mask(img, flood_mask(flood, 0), "non-flood-building")
    draw_mask(img, flood_mask(flood, 1), "flood-building")
    draw_mask(img, flood_mask(flood, 2), "non-flood-road")
    draw_mask(img, flood_mask(flood, 3), "flood-road")
    return img

def flood_mask(flood, idx):
    return flood.numpy()[idx, :, :].astype(bool)

def get_model(model_name):
    model_classes = {
        'baseline': UNetSiamese(in_channels=3, n_classes=5, bilinear=True),
        'simple': FloodSimpleModel(encoder_name="resnet50", device="cpu"),
        'complex': FloodModel(encoder_name="resnet50", device="cpu")
    }

    if model_name not in model_classes:
        raise ValueError(f"Invalid model name: {model_name}. Choose from {list(model_classes.keys())}")

    model = model_classes[model_name]
    return model


def load_model(checkpoint_path, model_name):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            new_key = key[len('model.'):]  # Remove the 'model.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value

    model = get_model(model_name)
    model.load_state_dict(new_state_dict)

    return model


@click.command()
@click.option("--data_csv", type=click.Path(exists=True), required=True)
@click.option("--output_dir", type=click.Path(), required=True)
@click.option("--n_images", type=int, default=5)  # set n_images to a large number to visualize all images
@click.option("--randomize/--first", type=bool, default=False)
@click.option("--checkpoint_path", type=str, required=True)
@click.option("--model_name", type=str, required=True)
def main(data_csv, output_dir, n_images, randomize, checkpoint_path, model_name):
    model = load_model(checkpoint_path, model_name)
    model.eval()

    dataset = SN8Dataset(data_csv,
                         data_to_load=["preimg","postimg","building","road","flood"])
    os.makedirs(output_dir, exist_ok=True)
    n_images = min(n_images, len(dataset))
    idx = random.choices(range(len(dataset)), k=n_images) if randomize else range(n_images)
    for i in idx:
        try:
            preimg, postimg, building, road, roadspeed, flood = dataset[i] # roadspeed=0
        except Exception:
            print('File not found')
            continue

        with torch.no_grad():
            output = model(preimg.unsqueeze(0), postimg.unsqueeze(0))
            max_indices = torch.argmax(output, dim=1, keepdim=True)
            one_hot_tensor = torch.zeros_like(output).scatter_(1, max_indices, 1)
            flood_pred = one_hot_tensor[:, 1:, :, :].squeeze(0)

        if len(preimg.shape)==3:
            preimg = torch.permute(preimg, (1, 2, 0))
            postimg = torch.permute(postimg, (1, 2, 0))

        road = torch.squeeze(road).numpy().astype(bool)
        building = torch.squeeze(building).numpy().astype(bool)
        pre_vis = draw_preimage(preimg.numpy(), road, building)
        post_vis = draw_postimage(postimg.clone().numpy(), flood)
        post_vis_pred = draw_postimage(postimg.clone().numpy(), flood_pred)
        preimg_filename = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_PRE.png")
        postimg_filename = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST.png")
        postimg_pred_filename = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST_pred.png")
        cv2.imwrite(os.path.join(output_dir, preimg_filename), pre_vis)
        cv2.imwrite(os.path.join(output_dir, postimg_filename), post_vis)
        cv2.imwrite(os.path.join(output_dir, postimg_pred_filename), post_vis_pred)

if __name__ == '__main__':
    main()
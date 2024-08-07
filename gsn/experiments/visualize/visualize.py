import numpy as np
import torch
import os
import cv2

from omegaconf import DictConfig

from experiments.datasets.datasets import SN8Dataset
from neptune.types import File

COLORS = {
    'building': np.array((169, 169, 169)),           # Dark Grey
    'road': np.array((211, 211, 211)),               # Light Grey
    'non-flood-building': np.array((169, 169, 169)), # Dark Grey
    'flood-building': np.array((0, 0, 139)),         # Darker Blue
    'non-flood-road': np.array((211, 211, 211)),     # Light Grey
    'flood-road': np.array((173, 216, 230))          # Light Blue
}

def save_eval_fig_in_neptune(cfg: DictConfig, model_from_checkpoint, logger):
    dataset = SN8Dataset(cfg.val_csv, data_to_load=["preimg","postimg","building","road","flood"])
    save_images_count = cfg.logger.neptune.save_images_count
    n_images = min(save_images_count, len(dataset))
    for i in range(n_images):
        try:
            preimg, postimg, building, road, roadspeed, flood = dataset[i]
        except Exception:
            print('File not found')
            continue

        with torch.no_grad():
            output = model_from_checkpoint(preimg.unsqueeze(0), postimg.unsqueeze(0))
            max_indices = torch.argmax(output, dim=1, keepdim=True)
            one_hot_tensor = torch.zeros_like(output).scatter_(1, max_indices, 1)
            flood_pred = one_hot_tensor[:, 1:, :, :].squeeze(0)

        if len(preimg.shape)==3:
            preimg = torch.permute(preimg, (1, 2, 0))
            postimg = torch.permute(postimg, (1, 2, 0))

        road = torch.squeeze(road).numpy().astype(bool)
        building = torch.squeeze(building).numpy().astype(bool)
        pre_vis = draw_preimage(preimg.clone().numpy(), road, building)
        post_vis = draw_postimage(postimg.clone().numpy(), flood)
        post_vis_pred = draw_postimage(postimg.clone().numpy(), flood_pred)
        autoscale_images = cfg.logger.neptune.autoscale_images
        logger.experiment["val/preimg"].append(File.as_image(preimg.clone().numpy(), autoscale=autoscale_images))
        logger.experiment["val/preimg_with_masks"].append(File.as_image(pre_vis, autoscale=autoscale_images))
        logger.experiment["val/post_img"].append(File.as_image(postimg.clone().numpy(), autoscale=autoscale_images))
        logger.experiment["val/post_img_with_masks"].append(File.as_image(post_vis, autoscale=autoscale_images))
        logger.experiment["val/prediction_with_masks"].append(File.as_image(post_vis_pred, autoscale=autoscale_images))


def save_eval_fig_on_disk(cfg: DictConfig, model_from_checkpoint, dir_name):
    dataset = SN8Dataset(cfg.val_csv, data_to_load=["preimg","postimg","building","road","flood"])
    fig_dir = os.path.join(cfg.output_dir, dir_name)
    os.makedirs(fig_dir, exist_ok=True)
    n_images = min(cfg.save_images_on_disk_count, len(dataset))
    for i in range(n_images):
        try:
            preimg, postimg, building, road, roadspeed, flood = dataset[i]
        except Exception:
            print('File not found')
            continue

        with torch.no_grad():
            output = model_from_checkpoint(preimg.unsqueeze(0), postimg.unsqueeze(0))
            max_indices = torch.argmax(output, dim=1, keepdim=True)
            one_hot_tensor = torch.zeros_like(output).scatter_(1, max_indices, 1)
            flood_pred = one_hot_tensor[:, 1:, :, :].squeeze(0)

        if len(preimg.shape)==3:
            preimg = torch.permute(preimg, (1, 2, 0))
            postimg = torch.permute(postimg, (1, 2, 0))

        road = torch.squeeze(road).numpy().astype(bool)
        building = torch.squeeze(building).numpy().astype(bool)
        pre_vis = draw_preimage(preimg.clone().numpy(), road, building)
        post_vis = draw_postimage(postimg.clone().numpy(), flood)
        post_vis_pred = draw_postimage(postimg.clone().numpy(), flood_pred)
        preimg_filename = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_PRE.png")
        preimg_filename_with_masks = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_PRE_with_masks.png")
        postimg_filename = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST.png")
        postimg_filename_with_masks = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST_with_masks.png")
        postimg_pred_filename_with_masks = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST_PRED_with_masks.png")
        cv2.imwrite(os.path.join(fig_dir, preimg_filename), preimg.clone().numpy())
        cv2.imwrite(os.path.join(fig_dir, preimg_filename_with_masks), pre_vis)
        cv2.imwrite(os.path.join(fig_dir, postimg_filename), postimg.clone().numpy())
        cv2.imwrite(os.path.join(fig_dir, postimg_filename_with_masks), post_vis)
        cv2.imwrite(os.path.join(fig_dir, postimg_pred_filename_with_masks), post_vis_pred)

def draw_mask(img, mask, mask_type):
    # first implementation
    color = np.array(COLORS[mask_type])
    #img[mask] = (img[mask] * 0.5 + color * 0.5).astype(np.uint8)
    img[mask] = color.astype(np.uint8)



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
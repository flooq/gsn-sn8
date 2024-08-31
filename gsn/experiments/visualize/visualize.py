import numpy as np
import torch
import os
import cv2

from omegaconf import DictConfig
from torch.utils.data import Dataset

from experiments.datasets.datasets import SN8Dataset
from neptune.types import File

COLORS_BLENDING = {
     'building': np.array((47 * 2, 79 * 2, 79 * 2)).clip(0, 255),            # Dark Slate Grey
     'road': np.array((105 * 2, 105 * 2, 105 * 2)).clip(0, 255),             # Dim Grey
     'non-flood-building': np.array((47 * 2, 79 * 2, 79 * 2)).clip(0, 255),  # Dark Slate Grey
     'flood-building': np.array((65 * 2, 105 * 2, 225 * 2)).clip(0, 255),    # Royal Blue
     'non-flood-road': np.array((105 * 2, 105 * 2, 105 * 2)).clip(0, 255),   # Dim Grey
     'flood-road': np.array((70 * 2, 130 * 2, 180 * 2)).clip(0, 255)         # Steel Blue
}

COLORS = {
    'building': np.array((47, 79, 79)),            # Dark Slate Grey
    'road': np.array((105, 105, 105)),             # Dim Grey
    'non-flood-building': np.array((47, 79, 79)),  # Dark Slate Grey
    'flood-building': np.array((65, 105, 225)),    # Royal Blue
    'non-flood-road': np.array((105, 105, 105)),   # Dim Grey
    'flood-road': np.array((70, 130, 180))         # Steel Blue
}


def save_eval_fig_in_neptune(cfg: DictConfig, model_from_checkpoint, logger):
    dataset = SN8Dataset(cfg.val_csv, data_to_load=["preimg","postimg","building","road","flood"])
    save_images_count = cfg.logger.save_images_count
    n_images = min(save_images_count, len(dataset))
    blending_color = cfg.visualize_blending_color
    for i in range(n_images):
        try:
            preimg, postimg, building, road, roadspeed, flood = dataset[i]
        except Exception:
            print('File not found')
            continue

        with torch.no_grad():
            output, class_pred = model_from_checkpoint(preimg.unsqueeze(0), postimg.unsqueeze(0))
            max_indices = torch.argmax(output, dim=1, keepdim=True)
            one_hot_tensor = torch.zeros_like(output).scatter_(1, max_indices, 1)
            flood_pred = one_hot_tensor[:, 1:, :, :].squeeze(0)
            flood_pred_smooth = morphology(flood_pred)

        if len(preimg.shape)==3:
            preimg = torch.permute(preimg, (1, 2, 0))
            postimg = torch.permute(postimg, (1, 2, 0))

        road = torch.squeeze(road).numpy().astype(bool)
        building = torch.squeeze(building).numpy().astype(bool)
        pre_vis = draw_preimage(preimg.clone().numpy(), blending_color, road, building)
        post_vis = draw_postimage(postimg.clone().numpy(), blending_color, flood)
        pre_vis_pred = draw_postimage(preimg.clone().numpy(), blending_color, flood_pred)
        post_vis_pred = draw_postimage(postimg.clone().numpy(), blending_color, flood_pred)
        pre_vis_pred_smooth = draw_postimage(preimg.clone().numpy(), blending_color, flood_pred_smooth)
        post_vis_pred_smooth = draw_postimage(postimg.clone().numpy(), blending_color, flood_pred_smooth)
        autoscale_images = cfg.logger.autoscale_images
        logger.experiment["val/preimg"].append(File.as_image(preimg.clone().numpy(), autoscale=autoscale_images))
        logger.experiment["val/preimg_with_masks"].append(File.as_image(pre_vis, autoscale=autoscale_images))
        logger.experiment["val/post_img"].append(File.as_image(postimg.clone().numpy(), autoscale=autoscale_images))
        logger.experiment["val/post_img_with_masks"].append(File.as_image(post_vis, autoscale=autoscale_images))
        logger.experiment["val/prediction_with_masks_on_pre"].append(File.as_image(pre_vis_pred, autoscale=autoscale_images))
        logger.experiment["val/prediction_with_masks_on_post"].append(File.as_image(post_vis_pred, autoscale=autoscale_images))
        logger.experiment["val/prediction_with_smooth_masks_on_pre"].append(File.as_image(pre_vis_pred_smooth, autoscale=autoscale_images))
        logger.experiment["val/prediction_with_smooth_masks_on_post"].append(File.as_image(post_vis_pred_smooth, autoscale=autoscale_images))


def save_eval_fig_on_disk(cfg: DictConfig, model_from_checkpoint, dir_name, dataset: Dataset = None):
    if not dataset:
        dataset = SN8Dataset(cfg.val_csv, data_to_load=["preimg","postimg","building","road","flood"])
    fig_dir = os.path.join(cfg.output_dir, dir_name)
    os.makedirs(fig_dir, exist_ok=True)
    n_images = min(cfg.save_images_on_disk_count, len(dataset))
    blending_color = cfg.visualize_blending_color
    for i in range(n_images):
        try:
            preimg, postimg, building, road, roadspeed, flood = dataset[i]
        except Exception:
            print('File not found')
            continue

        with torch.no_grad():
            output, class_pred = model_from_checkpoint(preimg.unsqueeze(0), postimg.unsqueeze(0))
            if class_pred is not None:
                print(f'Flood prediction for image {dataset.files[i]["preimg"]} is {torch.sigmoid(class_pred).item()} and ground truth is {_get_class_mask(flood).item()}')

            max_indices = torch.argmax(output, dim=1, keepdim=True)
            one_hot_tensor = torch.zeros_like(output).scatter_(1, max_indices, 1)
            flood_pred = one_hot_tensor[:, 1:, :, :].squeeze(0)
            flood_pred_smooth = morphology(flood_pred)

        if len(preimg.shape)==3:
            preimg = torch.permute(preimg, (1, 2, 0))
            postimg = torch.permute(postimg, (1, 2, 0))

        road = torch.squeeze(road).numpy().astype(bool)
        building = torch.squeeze(building).numpy().astype(bool)
        pre_vis = draw_preimage(preimg.clone().numpy(), blending_color, road, building)
        post_vis = draw_postimage(postimg.clone().numpy(), blending_color, flood)
        pre_vis_pred = draw_postimage(preimg.clone().numpy(), blending_color, flood_pred)
        post_vis_pred = draw_postimage(postimg.clone().numpy(), blending_color, flood_pred)
        pre_vis_pred_smooth = draw_postimage(preimg.clone().numpy(), blending_color, flood_pred_smooth)
        post_vis_pred_smooth = draw_postimage(postimg.clone().numpy(), blending_color, flood_pred_smooth)
        preimg_filename = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_PRE.png")
        preimg_filename_with_masks = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_PRE_with_masks.png")
        postimg_filename = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST.png")
        postimg_filename_with_masks = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST_with_masks.png")
        preimg_pred_filename_with_masks = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_PRE_PRED_with_masks.png")
        postimg_pred_filename_with_masks = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST_PRED_with_masks.png")
        preimg_pred_filename_with_smooth_masks = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_PRE_PRED_with_smooth_masks.png")
        postimg_pred_filename_with_smooth_masks = dataset.files[i]["preimg"].split("/")[-1].replace(".tif", "_POST_PRED_with_smooth_masks.png")
        cv2.imwrite(os.path.join(fig_dir, preimg_filename), preimg.clone().numpy())
        cv2.imwrite(os.path.join(fig_dir, preimg_filename_with_masks), pre_vis)
        cv2.imwrite(os.path.join(fig_dir, postimg_filename), postimg.clone().numpy())
        cv2.imwrite(os.path.join(fig_dir, postimg_filename_with_masks), post_vis)
        cv2.imwrite(os.path.join(fig_dir, preimg_pred_filename_with_masks), pre_vis_pred)
        cv2.imwrite(os.path.join(fig_dir, postimg_pred_filename_with_masks), post_vis_pred)
        cv2.imwrite(os.path.join(fig_dir, preimg_pred_filename_with_smooth_masks), pre_vis_pred_smooth)
        cv2.imwrite(os.path.join(fig_dir, postimg_pred_filename_with_smooth_masks), post_vis_pred_smooth)

def morphology(flood_pred):
    flood_pred_np = flood_pred.cpu().numpy().astype(np.uint8)
    for c in range(4):
        if c == 0 or c == 1: # buildings
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            flood_pred_np[c,:,:] = cv2.morphologyEx(flood_pred_np[c,:,:], cv2.MORPH_OPEN, kernel)
            flood_pred_np[c,:,:] = cv2.morphologyEx(flood_pred_np[c,:,:], cv2.MORPH_CLOSE, kernel)
        else: # roads
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            flood_pred_np[c, :, :] = cv2.erode(flood_pred_np[c, :, :], kernel, iterations=1)
            flood_pred_np[c,:,:] = cv2.morphologyEx(flood_pred_np[c,:,:], cv2.MORPH_OPEN, kernel)
            flood_pred_np[c,:,:] = cv2.morphologyEx(flood_pred_np[c,:,:], cv2.MORPH_CLOSE, kernel)
            flood_pred_np[c, :, :] = cv2.medianBlur(flood_pred_np[c, :, :], 3)
            flood_pred_np[c,:,:] = cv2.GaussianBlur(flood_pred_np[c,:,:], (3, 3), 0)

        contours, _ = cv2.findContours(flood_pred_np[c,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(flood_pred_np[c,:,:])
        for contour in contours:
            if c == 0 or c == 1: # buildings
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(contour_mask, [box], 0, 255, -1)
            else: # roads
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(contour_mask, [approx], 0, 255, -1)
        # if c == 0 or c == 1:
        flood_pred_np[c,:,:] = contour_mask
    return torch.from_numpy(flood_pred_np)


def draw_mask(img, blending_color, mask, mask_type):
    if blending_color:
        color = np.array(COLORS_BLENDING[mask_type])
        img[mask] = (img[mask] * 0.5 + color * 0.5).astype(np.uint8)
    else:
        color = np.array(COLORS[mask_type])
        img[mask] = color.astype(np.uint8)

def draw_preimage(img, blending_color, road, building):
    draw_mask(img, blending_color, road, "road")
    draw_mask(img, blending_color, building, "building")
    return img

def draw_postimage(img, blending_color, flood):
    draw_mask(img, blending_color, flood_mask(flood, 0), "non-flood-building")
    draw_mask(img, blending_color, flood_mask(flood, 1), "flood-building")
    draw_mask(img, blending_color, flood_mask(flood, 2), "non-flood-road")
    draw_mask(img, blending_color, flood_mask(flood, 3), "flood-road")
    return img

def flood_mask(flood, idx):
    return flood.numpy()[idx, :, :].astype(bool)

def _get_class_mask(flood_batch):
    flooded_channels = flood_batch[[1, 3], :, :] # only flooded channels
    summed_spatial_tensor = torch.sum(flooded_channels, dim=[1, 2])
    summed_channel_tensor = torch.sum(summed_spatial_tensor, dim=0)
    class_mask = (summed_channel_tensor > 0).float()
    return class_mask
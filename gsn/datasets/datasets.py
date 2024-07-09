import csv
import copy
from typing import List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations.augmentations.geometric.resize import Resize

from .augmentations import spatial_augmentations, color_augmentations


class SN8Dataset(Dataset):
    def __init__(self,
                 csv_filename: str,
                 data_to_load: List[str] = ("preimg", "postimg" ,"building" ,"road", "roadspeed" ,"flood"),
                 img_size: Tuple[int, int] = (1300, 1300),
                 out_img_size: Tuple[int, int] = (1024, 1024),
                 augment: bool = False):
        """ pytorch dataset for spacenet-8 data. loads images from a csv that contains filepaths to the images

        Parameters:
        ------------
        csv_filename (str): absolute filepath to the csv to load images from. the csv should have columns: preimg, postimg, building, road, roadspeed, flood.
            preimg column contains filepaths to the pre-event image tiles (.tif)
            postimg column contains filepaths to the post-event image tiles (.tif)
            building column contains the filepaths to the binary building labels (.tif)
            road column contains the filepaths to the binary road labels (.tif)
            roadspeed column contains the filepaths to the road speed labels (.tif)
            flood column contains the filepaths to the flood labels (.tif)
        data_to_load (list): a list that defines which of the images and labels to load from the .csv.
        img_size (tuple): the size of the input pre-event image in number of pixels before resizing.
        out_img_size (tuple): the size of the input pre- and post-event images in number of pixels after resizing.
        augment (bool): whether to add augmented images to the dataset.
        """
        self.all_data_types = ("preimg", "postimg", "building", "road", "roadspeed", "flood")
        self.mask_data_types = ("building", "road", "roadspeed", "flood")
        self.img_size = img_size
        self.out_img_size = out_img_size
        self.data_to_load = data_to_load
        self.files = []

        self.img_resize = Resize(*out_img_size)  # default interpolation method is linear
        self.mask_resize = Resize(*out_img_size, interpolation=cv2.INTER_NEAREST)

        self.spatial_augmentations = spatial_augmentations if augment else None
        self.color_augmentations = color_augmentations if augment else None
        self.n_augmentations = len(spatial_augmentations) * len(color_augmentations) if augment else 1

        dict_template = {}
        for i in self.all_data_types:
            dict_template[i] = None

        with open(csv_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                in_data = copy.copy(dict_template)
                for j in self.data_to_load:
                    in_data[j] = row[j]
                self.files.append(in_data)

        print("loaded", len(self.files), "image filepaths")

    def __len__(self):
        return len(self.files) * self.n_augmentations

    def _resize(self, image, data_type):
        if data_type in self.mask_data_types:
            return self.mask_resize.apply(image, interpolation=cv2.INTER_NEAREST)
        return self.img_resize.apply(image, interpolation=cv2.INTER_LINEAR)

    def _conform_axes(self, image):
        if len(image.shape) == 2:  # add a channel axis if read image is only shape (H,W).
            return torch.unsqueeze(torch.from_numpy(image), dim=0).float()
        else:
            image = np.moveaxis(image, -1, 0)
            return torch.from_numpy(image).float()

    def _augment(self, image, index, data_type):
        if self.n_augmentations == 1:
            return image
        aug_index = index % self.n_augmentations
        spatial_aug = self.spatial_augmentations[aug_index // len(self.color_augmentations)]
        aug_image = spatial_aug(image)
        if data_type in self.mask_data_types:
            return aug_image
        color_aug = self.color_augmentations[aug_index % len(self.color_augmentations)]
        return color_aug(aug_image)

    def __getitem__(self, index):
        file_index = index // self.n_augmentations
        data_dict = self.files[file_index]
        
        returned_data = []
        for i in self.all_data_types:
            filepath = data_dict[i]
            if filepath is not None:
                image = cv2.imread(filepath)
                image = self._resize(image, i)
                image = self._augment(image, index, i)
                image = self._conform_axes(image)
                returned_data.append(image)
            else:
                returned_data.append(0)

        return returned_data


def get_flood_mask(flood_batch):
    assert flood_batch.shape[1] == 3, f"invalid flood shape: {flood_batch.shape}"
    nonzero_mask = torch.sum(flood_batch, dim=1) > 0
    class_mask = torch.argmax(flood_batch, dim=1) + 1
    return class_mask.long() * nonzero_mask.long()

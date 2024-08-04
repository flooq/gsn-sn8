import csv
import copy
import random
from itertools import combinations_with_replacement
from typing import List, Tuple
import cv2
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations.augmentations.geometric.resize import Resize

from datasets.augmentations import GeometricTransform, ColorTransform


class SN8Dataset(Dataset):
    def __init__(self,
                 csv_filename: str,
                 data_to_load: List[str] = ("preimg", "postimg", "building", "road", "roadspeed", "flood"),
                 img_size: Tuple[int, int] = (1300, 1300),
                 out_img_size: Tuple[int, int] = (1280, 1280),
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

        self.augment = augment
        if augment:
            # All combinations of the three flips generate the D_8 group
            self.spatial_augmentations = [GeometricTransform(*c) for c in combinations_with_replacement((False, True), 3)]
            # Color augmentations are randomized
            n_color_transforms = 4
            self.color_augmentations = [lambda x: x] + [ColorTransform() for _ in range(n_color_transforms)]

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
        return len(self.files)

    def _resize(self, image, data_type):
        if data_type in self.mask_data_types:
            return self.mask_resize.apply(image, interpolation=cv2.INTER_NEAREST)
        return self.img_resize.apply(image, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _conform_axes(image):
        if len(image.shape) == 2:  # add a channel axis if read image is only shape (H,W).
            return torch.unsqueeze(torch.from_numpy(image), dim=0).float()
        else:
            image = np.moveaxis(image, -1, 0)
            return torch.from_numpy(image).float()

    def _augment(self, image, data_type, spatial_aug, color_aug):
        if not self.augment:
            return image
        aug_image = spatial_aug(image)
        if data_type in self.mask_data_types:
            return aug_image
        return color_aug(aug_image)

    def __getitem__(self, index):
        data_dict = self.files[index]

        returned_data = []
        spatial_aug = random.choice(self.spatial_augmentations) if self.augment else None
        color_aug = random.choice(self.color_augmentations) if self.augment else None
        for data_type in self.all_data_types:
            filepath = data_dict[data_type]
            if filepath is not None:
                image = io.imread(filepath)
                image = self._resize(image, data_type)
                image = self._augment(image, data_type, spatial_aug, color_aug)
                image = self._conform_axes(image)
                returned_data.append(image)
            else:
                returned_data.append(0)

        return returned_data

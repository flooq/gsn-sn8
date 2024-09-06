import pytorch_lightning as pl
import torch
import torch.nn
from omegaconf import DictConfig

from experiments.datasets.datasets import SN8Dataset


class SN8DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(SN8DataModule, self).__init__()

        self.train_csv = cfg.train_csv
        self.val_csv = cfg.val_csv
        self.batch_size = int(cfg.batch_size)
        self.augment = cfg.augment.enabled
        self.augment_color = cfg.augment.color.enabled
        self.augment_spatial = cfg.augment.spatial.enabled
        # color
        self.n_color_transforms = cfg.augment.color.n_transforms
        self.brightness = cfg.augment.color.brightness
        self.contrast = cfg.augment.color.contrast
        self.saturation = cfg.augment.color.saturation
        self.hue = cfg.augment.color.hue
        # spatial
        self.rotate = cfg.augment.spatial.rotate
        self.vertical_flip = cfg.augment.spatial.vertical_flip
        self.horizontal_flip = cfg.augment.spatial.horizontal_flip
        self.transpose = cfg.augment.spatial.transpose
        self.data_to_load = ["preimg", "postimg", "flood"]
        self.exclude_files = set(cfg.exclude_files)
        self.random_crop = cfg.image_random_crop.enabled

    def setup(self, stage):
        self.train_dataset = SN8Dataset(csv_filename=self.train_csv,
                                        data_to_load=self.data_to_load,
                                        augment=self.augment,
                                        augment_color=self.augment_color,
                                        augment_spatial=self.augment_spatial,
                                        n_color_transforms=self.n_color_transforms,
                                        brightness=self.brightness,
                                        contrast=self.contrast,
                                        saturation=self.saturation,
                                        hue=self.hue,
                                        rotate=self.rotate,
                                        vertical_flip=self.vertical_flip,
                                        horizontal_flip=self.horizontal_flip,
                                        transpose=self.transpose,
                                        exclude_files=self.exclude_files,
                                        random_crop=self.random_crop)

        self.val_dataset = SN8Dataset(self.val_csv,
                                      data_to_load=self.data_to_load,
                                      exclude_files=self.exclude_files) # no crop during validation


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, num_workers=4, batch_size=1) # always 1


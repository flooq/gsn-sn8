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
        self.n_color_transforms = cfg.augment.color.n_transforms
        self.brightness = cfg.augment.color.brightness
        self.contrast = cfg.augment.color.contrast
        self.saturation = cfg.augment.color.saturation
        self.hue = cfg.augment.color.hue
        self.data_to_load = ["preimg", "postimg", "flood"]
        self.exclude_files = set(cfg.exclude_files)

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
                                        exclude_files=self.exclude_files)
        self.val_dataset = SN8Dataset(self.val_csv, data_to_load=self.data_to_load, exclude_files=self.exclude_files)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, num_workers=4, batch_size=self.batch_size)


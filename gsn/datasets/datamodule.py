from typing import List

import pytorch_lightning as pl
import torch
import torch.nn

from datasets.datasets import SN8Dataset


class SN8DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_csv: str,
                 val_csv: str,
                 data_to_load: List[str] = None,
                 batch_size: int = 1):
        super(SN8DataModule, self).__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.data_to_load = data_to_load
        self.batch_size = batch_size

    def setup(self, stage):
        if self.data_to_load is None:
            self.train_dataset = SN8Dataset(self.train_csv)
            self.val_dataset = SN8Dataset(self.val_csv)
        else:
            self.train_dataset = SN8Dataset(self.train_csv, data_to_load=self.data_to_load)
            self.val_dataset = SN8Dataset(self.val_csv, data_to_load=self.data_to_load)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, num_workers=4, batch_size=self.batch_size)

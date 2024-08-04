import pytorch_lightning as pl
import torch
import torch.nn

from datasets.datasets import SN8Dataset


class SN8DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_csv: str,
                 val_csv: str,
                 batch_size: int = 1,
                 augment: bool = False):
        super(SN8DataModule, self).__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = int(batch_size)
        self.augment = augment
        self.data_to_load = ["preimg", "postimg", "flood"]

    def setup(self, stage):
        self.train_dataset = SN8Dataset(self.train_csv, data_to_load=self.data_to_load, augment=self.augment)
        self.val_dataset = SN8Dataset(self.val_csv, data_to_load=self.data_to_load)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, num_workers=4, batch_size=self.batch_size)

import numpy as np
import pytorch_lightning as pl
import torch

from metrics.metrics import get_val_metrics
from metrics.metrics import get_train_metrics
from scipy.ndimage import distance_transform_edt
from torch import nn

from experiments.schedulers.get_scheduler import get_scheduler


class FloodTrainer(pl.LightningModule):
    def __init__(self, loss, model, cfg):
        super(FloodTrainer, self).__init__()
        self.loss = loss
        self.class_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1000.0]))
        self.model = model
        self.cfg = cfg
        self.train_loss_sum = 0.0
        self.train_sample_count = 0
        if hasattr(cfg.model, 'distance_transform'):
            self.distance_transform_enabled = cfg.model.distance_transform.enabled
        else:
            self.distance_transform_enabled = False
        self.distance_transform_weight = cfg.model.distance_transform.weight if self.distance_transform_enabled else 0
        print(f"Distance transform weight: {self.distance_transform_weight}")
        if hasattr(cfg.model, 'flood_classification'):
            self.flood_classification_enabled = cfg.model.flood_classification.enabled
        else:
            self.flood_classification_enabled = False
        self.flood_classification_weight = cfg.model.flood_classification.initial_weight if self.flood_classification_enabled else 0
        self.flood_classification_increase_every_n_epochs = cfg.model.flood_classification.increase_every_n_epochs if self.flood_classification_enabled else None
        self.flood_classification_increase_factor = cfg.model.flood_classification.increase_factor if self.flood_classification_enabled else None
        self.flood_classification_max_weight = cfg.model.flood_classification.max_weight if self.flood_classification_enabled else None
        print(f"Initial flood classification weight: {self.flood_classification_weight}")
        self.main_weight = 1 - (self.distance_transform_weight + self.flood_classification_weight)
        self.epoch = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _loss, _, _ = self._do_step(batch)
        self.train_loss_sum += _loss.item()
        self.train_sample_count += 1
        return _loss

    def validation_step(self, batch, batch_idx):
        _loss, flood_pred, flood = self._do_step(batch)
        metrics_by_class = self.cfg.logger.metrics_by_class
        metrics = get_val_metrics(_loss, flood_pred, flood, metrics_by_class)
        self.log_dict(metrics)
        return _loss

    def on_train_epoch_start(self):
        # Reset the sum of the losses at the beginning of each epoch
        self.train_loss_sum = 0.0
        self.train_sample_count = 0

    def on_train_epoch_end(self):
        self.epoch += 1
        self._increase_flood_classification_weight()
        # Log the summed loss at the end of the epoch
        train_loss = self.train_loss_sum/self.train_sample_count
        metrics = get_train_metrics(train_loss)
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        scheduler = get_scheduler(optimizer, self.cfg)
        return [optimizer], [scheduler]

    def _do_step(self, batch):
        preimg, postimg, building, road, roadspeed, flood = batch
        flood_pred, class_pred = self.model(preimg, postimg)

        flood_with_background = self._get_flood_with_background(flood)
        _loss = self.main_weight * self.loss(flood_pred, flood_with_background)
        # print(f"Main Loss: {_loss.item()}")
        if self.distance_transform_enabled:
            distance_transform_flood = self._get_distance_transform_flood_mask(flood)
            flood_dt_with_background = self._get_flood_with_background(distance_transform_flood)
            distance_transform_loss =  self.distance_transform_weight*self.loss(flood_pred, flood_dt_with_background)
            _loss += distance_transform_loss
            #print(f"Distance Transform Loss: {distance_transform_loss.item()}")
        if class_pred is not None:
            class_mask = self._get_class_mask(flood)
            class_loss_value =  self.flood_classification_weight*self.class_loss(class_pred, class_mask)
            _loss += class_loss_value
            #print(f"Classification Loss: {class_loss_value.item()}")

        #print(f"Total Loss: {_loss.item()}")
        return _loss, flood_pred, flood

    def _increase_flood_classification_weight(self):
        if (self.flood_classification_enabled and
                self.flood_classification_weight < self.flood_classification_max_weight and
                self.epoch > 0 and
                self.epoch % self.flood_classification_increase_every_n_epochs == 0):
            self.flood_classification_weight *= self.flood_classification_increase_factor
            if self.flood_classification_weight > self.flood_classification_max_weight:
                self.flood_classification_weight = self.flood_classification_max_weight
            self.main_weight = 1 - (self.distance_transform_weight + self.flood_classification_weight)
            print(f"\nNew flood classification weight: {self.flood_classification_weight}")
            print(f"\nNew main weight: {self.main_weight}")

    # I tried with kornia but this implementation is faster despite moving tensor to cpu
    @staticmethod
    def _get_distance_transform_flood_mask(flood_batch):
        flood_np = flood_batch.cpu().numpy()
        distance_transforms = np.zeros_like(flood_np)
        for i in range(flood_np.shape[0]):  # iterate over batch
            for j in range(flood_np.shape[1]):  # iterate over the 4 masks
                distance_transform = distance_transform_edt(flood_np[i, j])
                # Normalize distance transform
                max_distance = distance_transform.max()
                if max_distance > 0:
                    distance_transform /= max_distance
                distance_transforms[i, j] = distance_transform
                #distance_transforms[i, j] = distance_transform_edt(flood_np[i, j])
        distance_transforms_tensor = torch.from_numpy(distance_transforms).to(flood_batch.device)
        return distance_transforms_tensor

    @staticmethod
    def _get_flood_with_background(flood_batch):
        background_mask = torch.sum(flood_batch, dim=1, keepdim=True) == 0
        background_mask = background_mask.long()
        combined_flood_mask = torch.cat((background_mask, flood_batch), dim=1)
        return combined_flood_mask

    @staticmethod
    def _get_class_mask(flood_batch):
        flooded_channels = flood_batch[:, [1, 3], :, :] # only flooded channels
        summed_spatial_tensor = torch.sum(flooded_channels, dim=[2, 3])
        summed_channel_tensor = torch.sum(summed_spatial_tensor, dim=1)
        class_mask = (summed_channel_tensor > 0).float().unsqueeze(1)
        return class_mask

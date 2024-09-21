import pytorch_lightning as pl
import torch

from metrics.metrics import get_val_metrics
from metrics.metrics import get_train_metrics
from torch import nn

from distance_transform.weighted_distance_transform import WeightedDistanceTransform
from schedulers.get_scheduler import get_scheduler


class FloodTrainer(pl.LightningModule):
    def __init__(self, loss, model, cfg):
        super(FloodTrainer, self).__init__()
        self.loss = loss
        self.class_loss = nn.BCEWithLogitsLoss()
        self.model = model
        self.cfg = cfg
        self.train_loss_sum = 0.0
        self.train_sample_count = 0
        self.distance_transform_enabled = cfg.distance_transform.enabled
        if self.distance_transform_enabled:
            self.distance_transform = WeightedDistanceTransform(
                weights=cfg.distance_transform.weights, inverted=cfg.distance_transform.inverted)
        self.flood_classification_enabled = cfg.flood_classification.enabled
        self.flood_classification_weight = cfg.flood_classification.initial_weight if self.flood_classification_enabled else 0
        self.flood_classification_increase_every_n_epochs = cfg.flood_classification.increase_every_n_epochs if self.flood_classification_enabled else None
        self.flood_classification_increase_factor = cfg.flood_classification.increase_factor if self.flood_classification_enabled else None
        self.flood_classification_max_weight = cfg.flood_classification.max_weight if self.flood_classification_enabled else None
        print(f"Initial flood classification weight: {self.flood_classification_weight}")
        self.main_weight = 1 - self.flood_classification_weight
        print(f"Main weight: {self.main_weight}")
        self.epoch = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _loss, _, _, _, _, _ = self._do_step(batch)
        self.train_loss_sum += _loss.item()
        self.train_sample_count += 1
        return _loss

    def validation_step(self, batch, batch_idx):
        _loss, flood_pred, flood, class_loss, class_pred, class_mask = self._do_step(batch)
        metrics_by_class = self.cfg.logger.metrics_by_class
        metrics = get_val_metrics(_loss, flood_pred, flood, class_loss, class_pred, class_mask, self.distance_transform_enabled, metrics_by_class)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = get_scheduler(optimizer, self.cfg)
        return [optimizer], [scheduler]

    def _do_step(self, batch):
        preimg, postimg, building, road, roadspeed, flood = batch
        flood_pred, class_pred = self.model(preimg, postimg)
        flood_with_background = self._get_flood_with_background(flood)
        if self.distance_transform_enabled:
            distance_transform_flood = self.distance_transform(flood)
            flood_with_background = torch.cat((flood_with_background, distance_transform_flood), dim=1) # 5 -> 9 channels
        _loss = self.main_weight * self.loss(flood_pred, flood_with_background)

        if self.flood_classification_enabled:
            if class_pred is None:
                raise ValueError(f"Model {self.cfg.model.name} does not have a classification head!")
            class_mask = self._get_class_mask(flood)
            class_loss_value = self.flood_classification_weight*self.class_loss(class_pred, class_mask)
            _loss += class_loss_value
        else:
            class_loss_value = None
            class_pred = None
            class_mask = None
        return _loss, flood_pred, flood_with_background, class_loss_value, class_pred, class_mask

    def _increase_flood_classification_weight(self):
        if (self.flood_classification_enabled and
                self.flood_classification_weight < self.flood_classification_max_weight and
                self.epoch > 0 and
                self.epoch % self.flood_classification_increase_every_n_epochs == 0):
            self.flood_classification_weight *= self.flood_classification_increase_factor
            if self.flood_classification_weight > self.flood_classification_max_weight:
                self.flood_classification_weight = self.flood_classification_max_weight
            self.main_weight = 1 - self.flood_classification_weight
            print(f"\nNew flood classification weight: {self.flood_classification_weight}")
            print(f"New main weight: {self.main_weight}")

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

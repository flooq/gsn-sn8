import pytorch_lightning as pl
import torch

from metrics.metrics import get_val_metrics
from metrics.metrics import get_train_metrics

from experiments.schedulers.get_scheduler import get_scheduler


class FloodTrainer(pl.LightningModule):
    def __init__(self, loss, model, cfg):
        super(FloodTrainer, self).__init__()
        self.loss = loss
        self.model = model
        self.cfg = cfg
        self.train_loss_sum = 0.0
        self.train_sample_count = 0

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

        flood_pred = self.model(preimg, postimg)
        if self.cfg.distance_transform:
            distance_transform_flood = self._get_distance_transform_flood_mask(flood)
            flood = self._get_flood_mask(flood)
            flood_dt = self._get_flood_mask(distance_transform_flood)
            _loss = self.loss(flood_pred, flood) + self.loss(flood_pred, flood_dt)
        else:
            flood = self._get_flood_mask(flood)
            _loss = self.loss(flood_pred, flood)
        return _loss, flood_pred, flood

    # I tried with kornia but this implementation is faster despite moving tensor to cpu
    def _get_distance_transform_flood_mask(self, flood_batch):
        flood_np = flood_batch.cpu().numpy()
        distance_transforms = np.zeros_like(flood_np)
        for i in range(flood_np.shape[0]):  # iterate over batch
            for j in range(flood_np.shape[1]):  # iterate over the 4 masks
                distance_transforms[i, j] = distance_transform_edt(flood_np[i, j])
        distance_transforms_tensor = torch.from_numpy(distance_transforms).to(flood_batch.device)
        return distance_transforms_tensor

    def _get_flood_mask(self, flood_batch):
        background_mask = torch.sum(flood_batch, dim=1, keepdim=True) == 0
        background_mask = background_mask.long() # ^ 0
        combined_flood_mask = torch.cat((background_mask, flood_batch), dim=1)
        return combined_flood_mask




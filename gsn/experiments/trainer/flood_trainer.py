import pytorch_lightning as pl
import torch

from metrics.metrics import get_val_metrics
from metrics.metrics import get_train_metrics


class FloodTrainer(pl.LightningModule):
    def __init__(self, loss, model, **kwargs):
        super(FloodTrainer, self).__init__()
        self.loss = loss
        self.model = model
        self.lr = kwargs.get("lr", 1e-3)
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
        metrics = get_val_metrics(_loss, flood_pred, flood)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
        return [optimizer], [scheduler]

    def _do_step(self, batch):
        preimg, postimg, building, road, roadspeed, flood = batch
        flood = self._get_flood_mask(flood)
        flood_pred = self.model(preimg, postimg)
        _loss = self.loss(flood_pred, flood)
        return _loss, flood_pred, flood

    def _get_flood_mask(self, flood_batch):
        mask = torch.sum(flood_batch, dim=1, keepdim=True) == 0
        additional_dim = mask.long() ^ 0
        flood = torch.cat((additional_dim, flood_batch), dim=1)
        return flood




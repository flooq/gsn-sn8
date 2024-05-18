import pytorch_lightning as pl
import torch

from models.flood_model import FloodModel


class FloodTrainer(pl.LightningModule):
    def __init__(self, loss, encoder_name: str = "resnet50", **kwargs):
        super(FloodTrainer, self).__init__()
        self.model = FloodModel(encoder_name=encoder_name, device="cuda") # TODO configure cuda
        self.lr = kwargs.get("lr", 1e-3)
        self.loss = loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.__calculate_loss(batch)

    def validation_step(self, batch, batch_idx):
        loss = self.__calculate_loss(batch)
        values = {"loss": loss}
        self.log_dict(values)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return [optimizer], [scheduler]

    def __calculate_loss(self, batch):
        preimg, postimg, building, road, roadspeed, flood = batch
        flood_buildings_pred, flood_roads_pred = self.model(preimg, postimg)
        loss = self.loss(flood_buildings_pred, flood[:, :2, :, :]) + self.loss(flood_roads_pred, flood[:, 2:, :, :])
        return loss

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models.baseline_unet import UNet, UNetSiamese
from models.losses import focal, soft_dice_loss, focal_loss_weight, soft_dice_loss_weight, \
    road_loss_weight, building_loss_weight, bceloss, celoss
from datasets.datasets import get_flood_mask


class LightningUNet(pl.LightningModule):
    def __init__(self, in_channels, n_classes, bilinear=True, **kwargs):
        super(LightningUNet, self).__init__()
        self.model = UNet(in_channels, n_classes, bilinear)
        self.lr = kwargs.get("lr", 1e-3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preimg, postimg, building, road, roadspeed, flood = batch
        building_pred, road_pred = self.model(preimg)
        building_loss = bceloss(building_pred, building)
        y_pred = F.sigmoid(road_pred)

        focal_l = focal(y_pred, roadspeed)
        dice_soft_l = soft_dice_loss(y_pred, roadspeed)

        road_loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
        loss = road_loss_weight * road_loss + building_loss_weight * building_loss
        return loss

    def validation_step(self, batch, batch_idx):
        preimg, postimg, building, road, roadspeed, flood = batch
        building_pred, road_pred = self.model(preimg)
        building_loss = bceloss(building_pred, building)
        y_pred = F.sigmoid(road_pred)

        focal_l = focal(y_pred, roadspeed)
        dice_soft_l = soft_dice_loss(y_pred, roadspeed)

        road_loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
        loss = road_loss_weight * road_loss + building_loss_weight * building_loss
        values = {"loss": loss, "bce_loss": building_loss, "focal_loss": focal_l, "dice_soft_loss": dice_soft_l}
        self.log_dict(values)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LightningUNetSiamese(pl.LightningModule):
    def __init__(self, in_channels, n_classes, bilinear=True, **kwargs):
        super(LightningUNetSiamese, self).__init__()
        self.model = UNetSiamese(in_channels, n_classes, bilinear)
        self.lr = kwargs.get("lr", 1e-3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preimg, postimg, building, road, roadspeed, flood = batch
        flood = get_flood_mask(flood)
        # flood_pred = model(combinedimg) # this is for resnet34 with stacked preimg+postimg input
        flood_pred = self.model(preimg, postimg)  # this is for siamese resnet34 with stacked preimg+postimg input
        loss = celoss(flood_pred, flood)
        return loss

    def validation_step(self, batch, batch_idx):
        preimg, postimg, building, road, roadspeed, flood = batch
        flood = get_flood_mask(flood)
        # flood_pred = model(combinedimg) # this is for resnet34 with stacked preimg+postimg input
        flood_pred = self.model(preimg, postimg)  # this is for siamese resnet34 with stacked preimg+postimg input
        loss = celoss(flood_pred, flood)
        values = {"loss": loss}
        self.log_dict(values)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return [optimizer], [scheduler]

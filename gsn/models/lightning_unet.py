import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from baseline_unet import UNet
from losses import focal, soft_dice_loss, focal_loss_weight, soft_dice_loss_weight, road_loss_weight, building_loss_weight


class LightningUNet(pl.LightningModule):
    def __init__(self, in_channels, n_classes, bilinear=True):
        super(LightningUNet, self).__init__()
        self.model = UNet(in_channels, n_classes, bilinear)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        preimg, postimg, building, road, roadspeed, flood = batch
        building_pred, road_pred = self.model(preimg)
        bce_l = nn.BCEWithLogitsLoss(building_pred, building)
        y_pred = F.sigmoid(road_pred)

        focal_l = focal(y_pred, roadspeed)
        dice_soft_l = soft_dice_loss(y_pred, roadspeed)

        road_loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
        building_loss = bce_l
        loss = road_loss_weight * road_loss + building_loss_weight * building_loss
        return loss

    def validation_step(self, batch, batch_idx):
        preimg, postimg, building, road, roadspeed, flood = batch
        building_pred, road_pred = self.model(preimg)
        bce_l = nn.BCEWithLogitsLoss(building_pred, building)
        y_pred = F.sigmoid(road_pred)

        focal_l = focal(y_pred, roadspeed)
        dice_soft_l = soft_dice_loss(y_pred, roadspeed)

        road_loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
        building_loss = bce_l
        loss = road_loss_weight * road_loss + building_loss_weight * building_loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


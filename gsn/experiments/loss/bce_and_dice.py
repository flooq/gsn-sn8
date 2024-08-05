from torch import nn
import torch

from experiments.loss.soft_dice_loss import SoftDiceLoss


class BceAndDice(nn.Module):
    __name__ = 'BceAndDice'

    def __init__(self, weight_focal=0.75, weight_dice=0.25):
        super().__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.soft_dice = SoftDiceLoss(apply_nonlin=torch.sigmoid, square=True, batch_dice=True)

    def forward(self, input, target, loss_mask=None):
        loss = (self.weight_focal * self.bce(input, target)) + (self.weight_dice* self.soft_dice(input, target, loss_mask=loss_mask))
        return loss.mean()

from torch import nn
import torch

from experiments.loss.binary_focal_loss import BinaryFocalLoss
from experiments.loss.soft_dice_loss import SoftDiceLoss


class FocalAndDice(nn.Module):
    __name__ = 'FocalAndDice'

    def __init__(self, alpha=None, weight_focal=0.75, weight_dice=0.25, gamma=2):
        super().__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.soft_dice = SoftDiceLoss(apply_nonlin=torch.sigmoid, square=True, batch_dice=True)

    def forward(self, input, target, loss_mask=None):
        input = input.contiguous().float()
        target = target.contiguous().float()
        loss = (self.weight_focal * self.focal(input, target, loss_mask=loss_mask)) + (self.weight_dice * self.soft_dice(input, target, loss_mask=loss_mask))
        return loss.mean()

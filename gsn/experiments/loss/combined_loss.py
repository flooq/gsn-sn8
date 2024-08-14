from torch import nn

from experiments.loss.dice_loss import Dice
from experiments.loss.focal_loss import Focal
from experiments.loss.lovasz_loss import Lovasz


class CombinedLoss(nn.Module):

    def __init__(self, cross_entropy: dict, dice: dict, focal:dict, lovasz:dict, mode: str = 'binary'):
        super().__init__()
        self.cross_entropy_weight = cross_entropy.weight
        self.dice_weight = dice.weight
        self.focal_weight = focal.weight
        self.lovasz_weight = lovasz.weight

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dice_loss = Dice()
        self.focal_loss = Focal(gamma=focal.gamma)
        self.lovasz_loss = Lovasz()

    def forward(self, inputs, targets):
        cross_entropy = self.cross_entropy_loss(inputs, targets) if self.cross_entropy_weight > 0 else 0
        dice = self.dice_loss(inputs, targets) if self.dice_weight > 0 else 0
        focal = self.focal_loss(inputs, targets) if self.focal_weight > 0 else 0
        lovasz = self.lovasz_loss(inputs, targets) if self.lovasz_weight > 0 else 0
        return (self.cross_entropy_weight * cross_entropy +
                self.dice_weight * dice +
                self.lovasz_weight * lovasz +
                self.focal_weight * focal)

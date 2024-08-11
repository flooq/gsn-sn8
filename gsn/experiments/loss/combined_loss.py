import segmentation_models_pytorch as smp
from torch import nn


class CombinedLoss(nn.Module):

    def __init__(self, cross_entropy: dict, dice: dict, focal:dict, lovasz:dict, mode: str = 'binary'):
        super().__init__()
        self.cross_entropy_weight = cross_entropy.weight
        self.dice_weight = dice.weight
        self.focal_weight = focal.weight
        self.lovasz_weight = lovasz.weight

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dice_loss = smp.losses.DiceLoss(mode=mode, log_loss=dice.log_loss)
        self.focal_loss = smp.losses.FocalLoss(mode=mode, alpha=focal.alpha, gamma=focal.gamma, reduction=focal.reduction)
        self.lovasz_loss = smp.losses.LovaszLoss(mode=mode, per_image=lovasz.per_image)

    def forward(self, inputs, targets):
        cross_entropy = self.cross_entropy_loss(inputs, targets) if self.cross_entropy_weight > 0 else 0
        dice = self.dice_loss(inputs, targets) if self.dice_weight > 0 else 0
        focal = self.focal_loss(inputs, targets) if self.focal_weight > 0 else 0
        lovasz = self.lovasz_loss(inputs, targets) if self.lovasz_weight > 0 else 0
        return (self.cross_entropy_weight * cross_entropy +
                self.dice_weight * dice +
                self.lovasz_weight * lovasz +
                self.focal_weight * focal)

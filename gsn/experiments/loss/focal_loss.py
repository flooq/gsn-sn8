import segmentation_models_pytorch as smp
from torch import nn

class Focal(nn.Module):

    def __init__(self, mode: str, alpha: int, gamma: int, reduction='mean'):
        super().__init__()
        self.loss = smp.losses.FocalLoss(mode=mode, alpha=alpha, gamma=gamma, reduction=reduction)

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

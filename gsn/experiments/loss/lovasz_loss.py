from torch import nn
import segmentation_models_pytorch as smp

class Lovasz(nn.Module):

    def __init__(self, mode: str):
        super().__init__()
        self.loss = smp.losses.LovaszLoss(mode=mode)

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss
import segmentation_models_pytorch as smp
from torch import nn

class Dice(nn.Module):

    def __init__(self, mode: str = 'binary', log_loss: bool = False):
        super().__init__()
        self.loss = smp.losses.DiceLoss(mode=mode, log_loss=log_loss)

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

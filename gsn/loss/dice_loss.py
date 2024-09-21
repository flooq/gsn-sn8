import torch
from torch import nn

class Dice(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        input = torch.sigmoid(inputs)
        smooth = 1.0

        iflat = input.view(-1)
        tflat = targets.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1-((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        return loss

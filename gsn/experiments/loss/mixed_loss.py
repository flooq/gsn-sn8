from torch import nn
import torch
import torch.nn.functional as F


class MixedLoss(nn.Module):
    __name__ = '_mixed_loss_'

    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss3(gamma)
        self.soft_dice = dice_loss3

    def forward(self, input, target):
        input = input.contiguous().float()
        target = target.contiguous().float()
        loss = self.alpha * self.focal(input, target) + ((1-self.alpha)*self.soft_dice(input, target))
        return loss.mean()


def dice_loss3(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    loss = 1-((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    return loss


class FocalLoss3(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

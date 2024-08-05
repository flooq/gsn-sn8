from torch import nn

class BCE(nn.Module):
    __name__ = '_bce_'

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        loss = self.loss(input, target)
        return loss.mean()
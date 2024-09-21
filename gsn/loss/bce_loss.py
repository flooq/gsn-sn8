from torch import nn

class BCE(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss.mean()
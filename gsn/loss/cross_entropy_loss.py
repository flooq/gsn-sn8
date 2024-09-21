from torch import nn


class CrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss.mean()
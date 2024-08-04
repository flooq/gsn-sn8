import torch
from omegaconf import DictConfig

from loss.loss import MixedLoss

def get_loss(cfg: DictConfig):
    if cfg.loss == 'mixed':
        print("mixed loss chosen")
        return MixedLoss()
    print("cell loss chosen")
    return torch.nn.CrossEntropyLoss()
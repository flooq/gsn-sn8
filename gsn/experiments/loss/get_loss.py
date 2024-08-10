import inspect

from omegaconf import DictConfig

from experiments.loss.bce_and_dice import BceAndDice
from experiments.loss.bce_loss import BCE
from experiments.loss.binary_focal_loss import BinaryFocalLoss
from experiments.loss.cross_entropy_loss import CrossEntropy
from experiments.loss.focal_and_dice import FocalAndDice
from experiments.loss.lovasz_loss import Lovasz
from experiments.loss.mixed_loss import MixedLoss
from experiments.loss.soft_dice_loss import SoftDiceLoss


def get_loss(cfg: DictConfig):

    losses = {
        'bce_and_dice': BceAndDice,
        'bce': BCE,
        'binary_focal': BinaryFocalLoss,
        'cross_entropy': CrossEntropy,
        'focal_and_dice': FocalAndDice,
        'lovasz': Lovasz,
        'mixed': MixedLoss,
        'soft_dice': SoftDiceLoss,
    }

    if cfg.loss.name not in losses:
        raise ValueError(f"Invalid loss name: {cfg.loss.name}. Choose from {list(losses.keys())}")

    classname = losses[cfg.loss.name]
    filtered_data = _filter_dict_for_constructor(classname, cfg.loss)
    print(f"Loss {cfg.loss.name} with parameters {filtered_data}")

    return classname(**filtered_data)


def _filter_dict_for_constructor(cls, data):
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.keys())[1:]
    return {key: data[key] for key in params if key in data}

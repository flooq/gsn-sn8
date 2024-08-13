import inspect

from omegaconf import DictConfig

from experiments.loss.bce_loss import BCE
from experiments.loss.combined_loss import CombinedLoss
from experiments.loss.cross_entropy_loss import CrossEntropy
from experiments.loss.dice_loss import Dice
from experiments.loss.focal_loss import Focal
from experiments.loss.lovasz_loss import Lovasz


def get_loss(cfg: DictConfig):

    losses = {
        'bce': BCE,
        'combined': CombinedLoss,
        'cross_entropy': CrossEntropy,
        'dice': Dice,
        'focal': Focal,
        'lovasz': Lovasz
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

import inspect

from omegaconf import DictConfig

from experiments.loss.bce_loss import BCE
from experiments.loss.regularization_loss_decorator import RegularizationLossDecorator
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
        'lovasz': Lovasz,
        'focal_lovasz': CombinedLoss
    }

    if cfg.loss.name not in losses:
        raise ValueError(f"Invalid loss name: {cfg.loss.name}. Choose from {list(losses.keys())}")

    classname = losses[cfg.loss.name]
    filtered_data = _filter_dict_for_constructor(classname, cfg.loss)
    print(f"Loss {cfg.loss.name} with parameters {filtered_data}")

    base_loss = classname(**filtered_data)
    if cfg.regularization_loss.enabled:
        return RegularizationLossDecorator(base_loss=base_loss,
                                     gradient_alpha=cfg.regularization_loss.gradient_alpha,
                                     shape_alpha=cfg.regularization_loss.shape_alpha,
                                     corner_alpha=cfg.regularization_loss.corner_alpha,
                                     channels_to_apply=cfg.regularization_loss.channels_to_apply)
    else:
        return base_loss

def _filter_dict_for_constructor(cls, data):
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.keys())[1:]
    return {key: data[key] for key in params if key in data}

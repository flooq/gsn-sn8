import inspect

import torch
from omegaconf import DictConfig

from experiments.models.baseline_unet import UNetSiameseBaseline
from experiments.models.unet_siamese import UnetSiamese
from experiments.models.unet_siamese_fused import UnetSiameseFused


def get_model(cfg: DictConfig):

    models = {
        'baseline': UNetSiameseBaseline,
        'siamese': UnetSiamese,
        'siamese_fused': UnetSiameseFused
    }

    if cfg.model.name not in models:
        raise ValueError(f"Invalid model name: {cfg.model.name}. Choose from {list(models.keys())}")

    classname = models[cfg.model.name]
    filtered_data = _filter_dict_for_constructor(classname, cfg.model)
    print(f"Model {cfg.model.name} with parameters {filtered_data}")

    return classname(**filtered_data)


def load_model_from_checkpoint(cfg, checkpoint_path):
    model = get_model(cfg)
    model.load_state_dict(_state_dict(checkpoint_path))
    return model


def _state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            new_key = key[len('model.'):]  # Remove the 'model.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value

    return new_state_dict

def _filter_dict_for_constructor(cls, data):
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.keys())[1:]
    return {key: data[key] for key in params if key in data}

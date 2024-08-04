import torch
from omegaconf import DictConfig

from models.baseline_unet import UNetSiamese
from models.flood_model import FloodModel
from models.flood_simple_model import FloodSimpleModel

# def get_model(model_name):
#     model_classes = {
#         'baseline': UNetSiamese(in_channels=3, n_classes=5, bilinear=True),
#         'simple': FloodSimpleModel(encoder_name="resnet50", device="cpu"),
#         'complex': FloodModel(encoder_name="resnet50"),
#     }
#
#     if model_name not in model_classes:
#         raise ValueError(f"Invalid model name: {model_name}. Choose from {list(model_classes.keys())}")
#
#     model = model_classes[model_name]
#     return model


def get_model(cfg: DictConfig):
    if cfg.model == 'baseline':
        print("baseline model chosen")
        return UNetSiamese(in_channels=3, n_classes=5, bilinear=True)
    elif cfg.model == 'complex':
        print("complex model chosen")
        return FloodModel(encoder_name="resnet50")
    else:
        print("simple model chosen")
        return FloodSimpleModel(encoder_name="resnet50", device="cuda")


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


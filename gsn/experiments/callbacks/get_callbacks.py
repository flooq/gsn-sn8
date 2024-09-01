from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

from experiments.callbacks.model_checkpoints import get_model_checkpoints


def get_callbacks(cfg: DictConfig):
    early_stopping_callback = EarlyStopping(
        monitor='val/iou',
        patience=250,
        mode='max'
    )
    return get_model_checkpoints(cfg) + [early_stopping_callback, TQDMProgressBar(refresh_rate=10)]
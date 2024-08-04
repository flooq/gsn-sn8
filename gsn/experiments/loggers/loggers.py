import os

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl


def get_logger(cfg: DictConfig):
    tags = list(OmegaConf.to_container(cfg.logger.neptune.tags, resolve=True))
    neptune_logger = pl.loggers.neptune.NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project=cfg.logger.neptune.project,
        log_model_checkpoints=False,
        tags=tags
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    neptune_logger.log_hyperparams(cfg_dict)
    return neptune_logger

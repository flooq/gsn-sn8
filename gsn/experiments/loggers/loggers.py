import os

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl


def get_logger(cfg: DictConfig):
    tags = [
        cfg.model.name,
        cfg.loss.name,
        'augment' if cfg.augment else None,
        f"lr={cfg.learning_rate}",
        f"batch={cfg.batch_size}"
    ]
    tags = [tag for tag in tags if tag is not None]

    logger = pl.loggers.neptune.NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project=cfg.logger.neptune.project,
        log_model_checkpoints=False,
        tags=tags
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparams(cfg_dict)
    return logger

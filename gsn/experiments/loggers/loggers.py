import os

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl


def get_logger(cfg: DictConfig):
    if cfg.logger.name == 'csv':
        return pl.loggers.CSVLogger(save_dir=cfg.output_dir, name='csv_logger')

    tags = [
        cfg.model.name,
        cfg.loss.name,
        'augment' if cfg.augment.enabled else None,
        'augment_color' if cfg.augment.color.enabled else None,
        'augment_spatial' if cfg.augment.spatial.enabled else None,
        'distance_transform' if is_distance_transform(cfg) else None,
        'flood_classification' if is_flood_classification(cfg) else None,
        f"lr={cfg.learning_rate}",
        f"batch={cfg.batch_size}"
    ]
    tags = [tag for tag in tags if tag is not None]

    logger = pl.loggers.neptune.NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project=cfg.logger.project,
        log_model_checkpoints=cfg.logger.log_model_checkpoints,
        tags=tags
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparams(cfg_dict)
    return logger

def is_distance_transform(cfg):
    if hasattr(cfg.model, 'distance_transform'):
        return cfg.model.distance_transform.enabled
    else:
        return False

def is_flood_classification(cfg):
    if hasattr(cfg.model, 'flood_classification'):
        return cfg.model.flood_classification.enabled
    else:
        return False
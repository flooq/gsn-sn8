import os

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl


def get_logger(cfg: DictConfig):
    if cfg.logger.name == 'csv':
        return pl.loggers.CSVLogger(save_dir=cfg.output_dir, name='csv_logger')

    tags = [
        cfg.model.name,
        'combined_loss' if cfg.loss.name == 'combined' else cfg.loss.name,
        'augment' if cfg.augment.enabled else None,
        'augment_color' if cfg.augment.color.enabled else None,
        'augment_spatial' if cfg.augment.spatial.enabled else None,
        'distance_transform' if is_distance_transform(cfg) else None,
        'flood_classification' if is_flood_classification(cfg) else None,
        'from_checkpoint' if cfg.model.load_from_checkpoint else None,
        f"cross_entropy={cfg.loss.cross_entropy.weight}" if cfg.loss.name == 'combined' and cfg.loss.cross_entropy.weight > 0 else None,
        f"dice_weight={cfg.loss.dice.weight}" if cfg.loss.name == 'combined' and cfg.loss.dice.weight > 0 else None,
        f"focal_weight={cfg.loss.focal.weight}" if cfg.loss.name == 'combined' and cfg.loss.focal.weight > 0 else None,
        f"lovasz_weight={cfg.loss.lovasz.weight}" if cfg.loss.name == 'combined' and cfg.loss.lovasz.weight > 0 else None,
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
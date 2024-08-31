import os

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import logging

def get_logger(cfg: DictConfig):
    if cfg.logger.name == 'csv':
        return pl.loggers.CSVLogger(save_dir=cfg.output_dir, name='csv_logger')

    tags = [
        cfg.model.name if cfg.model.name != 'siamese_fused' or not cfg.model.unet_plus_plus else 'siamese++_fused',
        cfg.model.encoder_name if cfg.model.name != 'baseline' else None,
        'combined_loss' if cfg.loss.name == 'combined' else cfg.loss.name,
        'aug' if cfg.augment.enabled else None,
        'aug_color' if cfg.augment.color.enabled else None,
        'aug_spatial' if cfg.augment.spatial.enabled else None,
        f"aug_factor={_calculate_number_of_pictures(cfg.augment)}" if cfg.augment.enabled else None,
        'dist_trans' if cfg.distance_transform.enabled and not cfg.distance_transform.inverted else None,
        'dist_trans_inv' if cfg.distance_transform.enabled and cfg.distance_transform.inverted else None,
        'flood_class' if cfg.flood_classification.enabled else None,
        'attention' if cfg.attention.enabled else None,
        f"attention_n={cfg.attention.pab_channels}" if cfg.attention.enabled else None,
        'ckpt' if cfg.load_from_checkpoint else None,
        f"cross_entropy={cfg.loss.cross_entropy.weight}" if cfg.loss.name == 'combined' and cfg.loss.cross_entropy.weight > 0 else None,
        f"dice_weight={cfg.loss.dice.weight}" if cfg.loss.name == 'combined' and cfg.loss.dice.weight > 0 else None,
        f"focal_weight={cfg.loss.focal.weight}" if cfg.loss.name == 'combined' and cfg.loss.focal.weight > 0 else None,
        f"lovasz_weight={cfg.loss.lovasz.weight}" if cfg.loss.name == 'combined' and cfg.loss.lovasz.weight > 0 else None,
        f"lr={cfg.learning_rate}",
        f"batch={cfg.batch_size}",
        f"lr_step={cfg.scheduler.step_size}" if cfg.scheduler.name == 'step_lr' else None,
        f"lr_gamma={cfg.scheduler.gamma}" if cfg.scheduler.name == 'step_lr' else None,
    ]
    tags = [tag for tag in tags if tag is not None]

    logger = pl.loggers.neptune.NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project=cfg.logger.project,
        log_model_checkpoints=cfg.logger.log_model_checkpoints,
        tags=tags
    )
    logging.getLogger("neptune").setLevel(logging.CRITICAL) # to ignore annoying warnings see https://github.com/neptune-ai/neptune-client/issues/1702
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparams(cfg_dict)
    return logger


def _calculate_number_of_pictures(augment_config):
    if not augment_config['enabled']:
        return 1
    if augment_config['color']['enabled']:
        color_transforms = augment_config['color']['n_transforms'] + 1
    else:
        color_transforms = 1
    spatial_transforms = 8 if augment_config['spatial']['enabled'] else 1
    number_of_pictures = color_transforms * spatial_transforms
    return number_of_pictures
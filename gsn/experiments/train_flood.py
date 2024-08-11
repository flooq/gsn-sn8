import env
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from experiments.callbacks.get_callbacks import get_callbacks
from experiments.callbacks.model_checkpoints import get_model_best_iou_checkpoint_pattern
from experiments.datasets.datamodule import SN8DataModule
from experiments.visualize.visualize import save_eval_fig_on_disk, save_eval_fig_in_neptune
from loggers.loggers import get_logger
from loss.get_loss import get_loss
from models.get_model import get_model, load_model_from_checkpoint_pattern, load_model_from_checkpoint
from trainer.flood_trainer import FloodTrainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_flood(cfg: DictConfig) -> None:
    torch.manual_seed(12)
    torch.set_float32_matmul_precision('medium')

    trainer_const_params = dict(
        devices=1,
        accelerator="gpu",
        log_every_n_steps=1,
    )
    logger = get_logger(cfg)
    if cfg.model.load_from_checkpoint:
        model = load_model_from_checkpoint(cfg, cfg.model.checkpoint_path)
    else:
        model = get_model(cfg)
    flood_trainer = FloodTrainer(loss=get_loss(cfg), model=model, cfg=cfg)
    trainer = pl.Trainer(
        **trainer_const_params,
        max_epochs=cfg.max_epochs,
        default_root_dir=cfg.output_dir,
        logger=logger,
        callbacks=get_callbacks(cfg)
    )

    data_module = SN8DataModule(train_csv=cfg.train_csv, val_csv=cfg.val_csv, batch_size=cfg.batch_size, augment=cfg.augment)
    trainer.fit(flood_trainer, datamodule=data_module)

    best_checkpoint_path = get_model_best_iou_checkpoint_pattern()
    print(f'Loading from checkpoint {best_checkpoint_path}')
    model_from_checkpoint = load_model_from_checkpoint_pattern(cfg, cfg.checkpoints_dir, best_checkpoint_path)
    if cfg.save_images_on_disk:
        save_eval_fig_on_disk(cfg, model_from_checkpoint, 'flood_eval_fig')

    if cfg.logger.name == 'neptune' and cfg.logger.save_images:
        save_eval_fig_in_neptune(cfg, model_from_checkpoint, logger)


if __name__ == "__main__":
    train_flood()


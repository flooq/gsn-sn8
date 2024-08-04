import os
import env
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import TQDMProgressBar

from experiments.datasets.datamodule import SN8DataModule
from experiments.callbacks.model_checkpoints import get_checkpoint
from experiments.visualize.visualize import save_eval_fig
from loggers.loggers import get_logger
from loss.get_loss import get_loss
from models.get_model import get_model, load_model_from_checkpoint
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
    flood_trainer = FloodTrainer(loss=get_loss(cfg), model=get_model(cfg), lr=cfg.learning_rate)
    trainer = pl.Trainer(
        **trainer_const_params,
        max_epochs=cfg.max_epochs,
        default_root_dir=cfg.output_dir,
        logger=logger,
        callbacks=[get_checkpoint(cfg), TQDMProgressBar(refresh_rate=10)]
    )

    data_module = SN8DataModule(train_csv=cfg.train_csv, val_csv=cfg.val_csv, batch_size=cfg.batch_size, augment=cfg.augment)
    trainer.fit(flood_trainer, datamodule=data_module)

    checkpoint_path = os.path.join(cfg.checkpoints_dir, 'checkpoint.ckpt')
    model_from_checkpoint = load_model_from_checkpoint(cfg, checkpoint_path)

    save_eval_fig(cfg, model_from_checkpoint, logger)


if __name__ == "__main__":
    train_flood()


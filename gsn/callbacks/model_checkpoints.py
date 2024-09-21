from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint


def get_model_best_iou_checkpoint_pattern():
    return 'best-iou-checkpoint-*-*.ckpt'

def get_model_checkpoints(cfg: DictConfig):
    best = ModelCheckpoint(
        monitor='val/loss',
        dirpath=cfg.checkpoints_dir,
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    best_iou = ModelCheckpoint(
        monitor='val/iou',
        dirpath=cfg.checkpoints_dir,
        filename='best-iou-checkpoint-{epoch:02d}-{val_iou:.2f}',
        save_top_k=1,
        mode='max'
    )

    last = ModelCheckpoint(
        dirpath=cfg.checkpoints_dir,
        filename='last-checkpoint-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=1
    )
    return [best, best_iou, last]
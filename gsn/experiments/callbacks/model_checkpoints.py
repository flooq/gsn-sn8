from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

def get_checkpoint(cfg: DictConfig):
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoints_dir,  # Directory where checkpoints will be saved
        # filename='{epoch}-{val_loss:.3f}',  # Checkpoint filename pattern
        filename='checkpoint',
        #save_top_k=3,  # Number of best checkpoints to keep
        # monitor='val/loss',  # Metric to monitor
        #mode='min',  # Mode for the monitored metric ('min' for minimizing, 'max' for maximizing),
        every_n_epochs=1
    )
    return checkpoint_callback
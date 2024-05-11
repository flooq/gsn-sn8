import os
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl

from datasets.datamodule import SN8DataModule
from loss.loss import MixedLoss

from trainer.flood_trainer import FloodTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",
                        type=str,
                        required=True)
    parser.add_argument("--val_csv",
                         type=str,
                         required=True)
    parser.add_argument("--save_dir",
                         type=str,
                         required=True)
    parser.add_argument("--model_name",
                         type=str,
                         required=True)
    parser.add_argument("--lr",
                         type=float,
                        default=0.0001)
    parser.add_argument("--batch_size",
                         type=int,
                        default=1)
    parser.add_argument("--n_epochs",
                         type=int,
                         default=50)
    parser.add_argument("--gpu",
                        type=int,
                        default=0)
    args = parser.parse_args()
    return args

trainer_const_params = dict(
    devices=1,
    accelerator="gpu",
    log_every_n_steps=1,
)


def main():
    args = parse_args()
    train_csv = args.train_csv
    val_csv = args.val_csv
    save_dir = args.save_dir
    model_name = args.model_name
    initial_lr = args.lr
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    SEED = 12
    torch.manual_seed(SEED)

    assert(os.path.exists(save_dir))
    now = datetime.now()
    date_total = str(now.strftime("%d-%m-%Y-%H-%M"))

    save_dir = os.path.join(save_dir, f"{model_name}_lr{'{:.2e}'.format(initial_lr)}_bs{batch_size}_{date_total}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.chmod(save_dir, 0o777)

    data_module = SN8DataModule(train_csv=train_csv,
                                val_csv=val_csv,
                                data_to_load=["preimg", "postimg", "flood"],
                                batch_size= int(batch_size))

    logger = pl.loggers.CSVLogger(save_dir=save_dir, name=model_name)

    model = FloodTrainer(loss=MixedLoss(), lr=initial_lr)
    trainer = pl.Trainer(
        **trainer_const_params,
        max_epochs=n_epochs, default_root_dir=save_dir, logger=logger
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()

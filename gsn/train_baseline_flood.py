import os
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl

from models.lightning_unet import LightningUNetSiamese
from datasets.datasets import SN8Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",
                        type=str,
                        required=True)
    parser.add_argument("--val_csv",
                         type=str,
                         required=True)
    parser.add_argument("--load_checkpoint",
                        type=str),
    parser.add_argument("--save_dir",
                         type=str)
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
    parser.add_argument("--precision",
                        type=str,
                        default="medium")
    args = parser.parse_args()
    return args

trainer_const_params = dict(
    devices=1,
    accelerator="gpu",
    log_every_n_steps=1,
)
num_classes = 5


def main():
    args = parse_args()
    train_csv = args.train_csv
    val_csv = args.val_csv
    save_dir = args.save_dir
    model_name = args.model_name
    initial_lr = args.lr
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    gpu = args.gpu
    precision = args.precision
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    torch.set_float32_matmul_precision(precision)

    SEED = 12
    torch.manual_seed(SEED)

    now = datetime.now()
    date_total = str(now.strftime("%d-%m-%Y-%H-%M"))
    if save_dir is None:
        save_dir = os.path.dirname(args.load_checkpoint)
    else:
        save_dir = os.path.join(save_dir, f"{model_name}_lr{'{:.2e}'.format(initial_lr)}_bs{batch_size}_{date_total}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.chmod(save_dir, 0o777)

    train_dataset = SN8Dataset(train_csv,
                               data_to_load=["preimg", "postimg", "flood"],
                               augment=True
                               )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=batch_size)
    val_dataset = SN8Dataset(val_csv,
                             data_to_load=["preimg", "postimg", "flood"]
                             )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=4, batch_size=batch_size)

    if args.load_checkpoint:
        model = LightningUNetSiamese.load_from_checkpoint(args.load_checkpoint)
    else:
        model = LightningUNetSiamese(3, num_classes, bilinear=True, lr=initial_lr)
    neptune_logger = pl.loggers.neptune.NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"], project="gsn/baseline-flood", log_model_checkpoints=False
    )
    model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=save_dir, save_top_k=-1, every_n_epochs=4)
    trainer = pl.Trainer(
        **trainer_const_params,
        max_epochs=n_epochs,
        default_root_dir=save_dir,
        callbacks=[model_checkpoint],
        logger=neptune_logger
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()

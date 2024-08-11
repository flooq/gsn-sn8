import os
import env
import click
from omegaconf import OmegaConf

from experiments.visualize.visualize import save_eval_fig_on_disk
from models.get_model import load_model_from_checkpoint

# Options:
# hydra_train_flood_out_dir - choose one from from hydra_train_flood_out_dir/checkpoints
# out_dir_name = pictures will be saved in hydra_train_flood_out_dir/out_dir_name
#   example values:
#   hydra_train_flood_out_dir = '/home/pawel/projects/flooq/gsn-sn8/outputs/gsn/train_flood/2024-08-11/07-32-45'
#   checkpoint_file = 'best-checkpoint-epoch=00-val_loss=0.00.ckpt
#   out_dir_name = 'flood_eval_fig'
@click.command()
@click.option("--hydra_train_flood_out_dir", type=str, required=True)
@click.option("--checkpoint_file", type=str, required=True)
@click.option("--out_dir_name", type=str, required=True)
@click.option("--blending_color", type=str, default=True)
@click.option("--n_images", type=int, default=5)
def visualize(hydra_train_flood_out_dir, checkpoint_file, out_dir_name, blending_color, n_images):
    cfg = _load_cfg_from_experiment(hydra_train_flood_out_dir)

    cfg.output_dir = hydra_train_flood_out_dir
    cfg.checkpoints_dir = os.path.join(cfg.output_dir, 'checkpoints')
    cfg.model.checkpoint_path = os.path.join(cfg.checkpoints_dir, checkpoint_file)
    cfg.visualize_blending_color = blending_color
    cfg.save_images_on_disk_count = n_images
    model_from_checkpoint = load_model_from_checkpoint(cfg, cfg.model.checkpoint_path)
    save_eval_fig_on_disk(cfg, model_from_checkpoint, out_dir_name)

def _load_cfg_from_experiment(output_dir: str):
    config_path = os.path.join(output_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    cfg = OmegaConf.load(config_path)
    return cfg

if __name__ == "__main__":
    visualize()


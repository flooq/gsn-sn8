import hydra
from omegaconf import DictConfig

from common import train, eval, split_train_val_dataset, get_latest_hydra_job_execution
from preprocess import preprocess_cfg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_all(cfg: DictConfig) -> None:
    preprocess_cfg(cfg)

    split_train_val_dataset(cfg)
    train(cfg, "foundation", "train_foundation_features.py")
    train(cfg, "flood", "train_flood.py")

    latest = get_latest_hydra_job_execution(["train_all"])
    eval(cfg,
         network="foundation",
         script="foundation_eval.py",
         output_train_dir=latest)
    eval(cfg,
         network="flood",
         script="flood_eval.py",
         output_train_dir=latest)


if __name__ == "__main__":
    run_all()

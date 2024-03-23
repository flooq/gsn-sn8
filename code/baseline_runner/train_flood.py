import hydra
from omegaconf import DictConfig

from train_common import train, split_train_val_dataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_flood(cfg: DictConfig) -> None:
    split_train_val_dataset(cfg)
    train(cfg, "flood", "train_flood.py")


if __name__ == "__main__":
    train_flood()


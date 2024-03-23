import hydra
from omegaconf import DictConfig

from train_common import train, split_train_val_dataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_foundation(cfg: DictConfig) -> None:
    split_train_val_dataset(cfg)
    train(cfg, "foundation", "train_foundation_features.py")


if __name__ == "__main__":
    train_foundation()

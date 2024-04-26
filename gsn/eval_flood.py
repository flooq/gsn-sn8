import hydra
from omegaconf import DictConfig

from common import eval

@hydra.main(version_base=None, config_path="conf", config_name="config")
def eval_flood(cfg: DictConfig) -> None:
    eval(cfg)

if __name__ == "__main__":
    eval_flood()


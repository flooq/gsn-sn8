import hydra
from omegaconf import DictConfig

from common import eval, get_latest_hydra_job_execution


@hydra.main(version_base=None, config_path="conf", config_name="config")
def eval_foundation(cfg: DictConfig) -> None:
    latest = get_latest_hydra_job_execution(["train_foundation", "train_all"])
    eval(cfg,
         network="foundation",
         script="foundation_eval.py",
         output_train_dir=latest)


if __name__ == "__main__":
    eval_foundation()

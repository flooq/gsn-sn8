import hydra
from omegaconf import DictConfig

from common import eval, get_latest_hydra_job_execution


@hydra.main(version_base=None, config_path="conf", config_name="config")
def eval_flood(cfg: DictConfig) -> None:
    latest = get_latest_hydra_job_execution(["train_flood", "train_all"])
    eval(cfg,
         network="flood",
         script="flood_eval.py",
         output_train_dir=latest)


if __name__ == "__main__":
    eval_flood()

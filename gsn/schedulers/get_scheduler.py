import inspect

from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR


def get_scheduler(optimizer, cfg: DictConfig):

    schedulers = {
        'step_lr': StepLR
    }

    if cfg.scheduler.name not in schedulers:
        raise ValueError(f"Invalid scheduler name: {cfg.scheduler.name}. Choose from {list(schedulers.keys())}")

    classname = schedulers[cfg.scheduler.name]
    filtered_data = _filter_dict_for_constructor(classname, cfg.scheduler)
    print(f"Scheduler {cfg.scheduler.name} with parameters {filtered_data}")

    return classname(optimizer, **filtered_data)


def _filter_dict_for_constructor(cls, data):
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.keys())[1:]
    return {key: data[key] for key in params if key in data}

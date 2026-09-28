import os
from typing import Optional

import numpy as np
import random
import torch


def _placeholder(value, placeholder: str) -> Optional[str]:
    """Return ``value`` if it is set and not equal to ``placeholder``, else ``None``."""
    if value is None:
        return None
    s = str(value)
    if not s or s == placeholder:
        return None
    return s


def init_wandb_from_config(config) -> None:
    """Initialise ``wandb`` from ``config`` and environment variables.

    The shipped YAMLs use placeholder strings (``WANDB_HOST``, ``WANDB_KEY``
    etc.) for the wandb fields so the configs can be committed without
    leaking secrets.  This helper treats those placeholders as unset and
    falls back to standard ``WANDB_API_KEY``/``WANDB_ENTITY``/``WANDB_PROJECT``
    environment variables (or any pre-existing ``wandb login``).
    """
    import wandb
    from omegaconf import OmegaConf

    wb_key = os.environ.get("WANDB_API_KEY") or _placeholder(
        getattr(config, "wandb_key", None), "WANDB_KEY"
    )
    wb_host = _placeholder(getattr(config, "wandb_host", None), "WANDB_HOST")
    wb_entity = os.environ.get("WANDB_ENTITY") or _placeholder(
        getattr(config, "wandb_entity", None), "WANDB_ENTITY"
    )
    wb_project = os.environ.get("WANDB_PROJECT") or _placeholder(
        getattr(config, "wandb_project", None), "WANDB_PROJECT"
    )
    wb_name = os.environ.get("WANDB_NAME") or getattr(config, "config_name", None)
    wb_run_group = os.environ.get("WANDB_RUN_GROUP") or _placeholder(
        getattr(config, "wandb_run_group", None), "WANDB_RUN_GROUP"
    )

    if wb_key:
        login_kwargs = {"key": wb_key}
        if wb_host:
            login_kwargs["host"] = wb_host
        wandb.login(**login_kwargs)

    init_kwargs = dict(
        config=OmegaConf.to_container(config, resolve=True),
        name=wb_name,
        mode="online",
        dir=getattr(config, "wandb_save_dir", "") or None,
    )
    if wb_entity:
        init_kwargs["entity"] = wb_entity
    if wb_project:
        init_kwargs["project"] = wb_project
    if wb_run_group:
        init_kwargs["group"] = wb_run_group
    wandb.init(**init_kwargs)


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict

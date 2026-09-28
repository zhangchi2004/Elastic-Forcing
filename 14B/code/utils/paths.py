"""Centralized filesystem layout for Self-Forcing assets.

The pretrained Wan model weights and shared training checkpoints live outside
the repository (on the large data partition).  This module is the single
source of truth for those paths so callers do not have to hard-code them.

Both can be overridden with environment variables for portability:

* ``SELF_FORCING_MODELS_DIR``   (defaults to the shared SF asset directory)
* ``SELF_FORCING_CKPT_DIR``     (defaults to the shared SF asset directory)
"""

from __future__ import annotations

import os
from pathlib import Path

_ASSET_ROOT = Path(
    os.environ.get(
        "SELF_FORCING_ASSET_ROOT",
        "assets",
    )
)
MODELS_DIR = Path(os.environ.get("SELF_FORCING_MODELS_DIR", _ASSET_ROOT / "models"))
CKPT_DIR = Path(
    os.environ.get("SELF_FORCING_CKPT_DIR", _ASSET_ROOT / "checkpoints")
)


def model_path(name: str) -> Path:
    """Return the on-disk directory for a Wan-style pretrained model bundle."""
    return MODELS_DIR / name


def ckpt_path(*parts: str) -> Path:
    """Return a path under the shared checkpoint directory."""
    return CKPT_DIR.joinpath(*parts)

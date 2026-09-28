"""Lazy public trainer exports.

Loading one trainer must not import optional dependencies belonging only to a
different experiment.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "DiffusionTrainer": (".diffusion", "Trainer"),
    "GANTrainer": (".gan", "Trainer"),
    "ODETrainer": (".ode", "Trainer"),
    "ScoreDistillationTrainer": (".distillation", "Trainer"),
    "ARLStage1Trainer": (".arl", "ARLStage1Trainer"),
    "ARLStage2Trainer": (".arl", "ARLStage2Trainer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

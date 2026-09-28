"""Compatibility module; read ``trainer.feature_gradient_replay`` instead."""

import sys
from trainer import feature_gradient_replay as _implementation

# Preserve private helpers, monkeypatch targets, and legacy import paths.
sys.modules[__name__] = _implementation

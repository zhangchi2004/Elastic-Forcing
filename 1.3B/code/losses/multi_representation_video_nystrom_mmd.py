"""Compatibility module; read ``losses.multi_encoder_mmd`` for the implementation."""

import sys
from losses import multi_encoder_mmd as _implementation

# Reuse the module itself so existing imports and module-level patches agree.
sys.modules[__name__] = _implementation

"""Compatibility import; read trainer/elastic_forcing.py for the implementation."""
import sys
from . import elastic_forcing as _implementation

# Share the module object so existing monkeypatches and private imports still work.
sys.modules[__name__] = _implementation

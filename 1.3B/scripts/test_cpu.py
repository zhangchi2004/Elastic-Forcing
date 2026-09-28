#!/usr/bin/env python3
"""Run existing CPU tensor tests without initializing the historical T5 CUDA default."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'code'))
os.chdir(ROOT/'code')
import torch
import pytest

# Only bypass import-time device discovery. No model weights or CUDA tensors load.
# Original sources remain untouched; the patch is removed before executing tests.
with patch('torch.cuda.current_device', return_value=0):
    from trainer.distillation import Trainer
    from model.wan_kv_mmd import WanKVMMD

raise SystemExit(pytest.main([
    '-q', '-p', 'no:cacheprovider',
    'tests/test_triple_partial_geometry.py',
    'tests/test_readable_config.py',
    'tests/test_triple_video_feature_bank.py',
    'tests/test_triple_feature_metadata_contract.py',
    'tests/test_triple_pixel_cotangent_accumulation.py',
    'tests/test_video_nystrom_partial_mmd.py',
    'tests/test_multi_representation_video_nystrom_mmd.py',
] + sys.argv[1:]))

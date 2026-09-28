#!/usr/bin/env bash
# Source from launchers; all generated files stay in the reproduction directory.
set -euo pipefail
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPRO_ROOT/configs/local.env"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"
CONFIG="$REPRO_ROOT/configs/local.yaml"
export SELF_FORCING_MODELS_DIR="$ELASTIC_FORCING_ASSET_ROOT/models"
export SELF_FORCING_CKPT_DIR="$ELASTIC_FORCING_ASSET_ROOT/checkpoints"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPRO_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$REPRO_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPRO_ROOT/cache/triton}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT

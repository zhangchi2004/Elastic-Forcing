#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CHECKPOINT="$(realpath "$CHECKPOINT")"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export PYTHONPATH="$REPRO_ROOT/code"
PROMPTS="${PROMPTS:-$REPRO_ROOT/code/prompts/test_8.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPRO_ROOT/outputs/demo}"
PROMPTS="$(realpath "$PROMPTS")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
cd "$REPRO_ROOT/code"
exec "$PYTHON_BIN" inference.py --config_path "$CONFIG" --checkpoint_path "$CHECKPOINT" \
  --data_path "$PROMPTS" --output_folder "$OUTPUT_DIR" \
  --num_output_frames 21 --num_samples 1 --seed 0 --save_with_index

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${CHECKPOINT:?Set CHECKPOINT to the step-80 model.pt file}"
CHECKPOINT="$(realpath "$CHECKPOINT")"
PROMPTS="$(realpath "${PROMPTS:-$REPRO_ROOT/code/prompts/test_8.txt}")"
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR:-$REPRO_ROOT/outputs/demo}")"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
cd "$REPRO_ROOT/code"
"$PYTHON_BIN" inference.py --config_path "$REPRO_ROOT/configs/inference.yaml" \
  --checkpoint_path "$CHECKPOINT" --data_path "$PROMPTS" \
  --output_folder "$OUTPUT_DIR" --num_output_frames 21 \
  --num_samples "${NUM_SAMPLES:-5}" --seed "${SEED:-0}" --save_with_index

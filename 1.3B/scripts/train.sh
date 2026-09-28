#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
extra_args=()
case "${1:-}" in
  "") ;;
  --smoke) extra_args=(--max_steps 1) ;;
  *) echo 'Usage: scripts/train.sh [--smoke]' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'Too many arguments' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
LOGDIR="${LOGDIR:-$REPRO_ROOT/runs/three-encoder}"
LOGDIR="$(realpath -m "$LOGDIR")"
mkdir -p "$(dirname "$LOGDIR")"
mkdir -p "$LOGDIR"
cp "$CONFIG" "$LOGDIR/config.yaml"
export SF_FRESH_VIDEO_P0_REPLAY=1 SF_FRESH_VIDEO_PIXEL_CACHE_DEVICE=cpu
unset SF_FRESH_VIDEO_RETAINED_GRAPH SF_FRESH_VIDEO_RETAIN_WAN_GRAPH
export PYTHONPATH="$REPRO_ROOT/code"
cd "$REPRO_ROOT/code"
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 train.py \
  --config_path "$LOGDIR/config.yaml" --logdir "$LOGDIR" \
  --disable-wandb --no_visualize "${extra_args[@]}" 2>&1 | tee "$LOGDIR/train.log"

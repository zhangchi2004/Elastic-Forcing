#!/usr/bin/env python3
"""Precompute a video-level V-JEPA2 dense Nyström teacher reference.

The extraction path deliberately matches ``WanKVMMD``:

* read the final teacher latent from the ODE LMDB;
* decode with the frozen Wan VAE in bfloat16;
* map decoded pixels from ``[-1, 1]`` to ``[0, 1]``;
* select a deterministic contiguous temporal window, then apply the V-JEPA2
  adapter's 292 -> 256 spatial preprocessing;
* execute complete V-JEPA2 blocks 0..the configured dense layer;
* retain the canonical 128 aligned token positions as one ``[128, 1024]``
  video sample.

Each torchrun rank writes one indexed FP16 feature shard to a temporary
directory. Rank 0 validates and merges the shards, builds a Nyström reference,
publishes it atomically, and removes the temporary directory only after the
published pack has passed validation.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TOOL_FORMAT_VERSION = 2
MIB = 1024 ** 2
FRAME_SAMPLING_CHOICES = (
    "center_contiguous",
    "random_contiguous_per_video",
)
TEACHER_TEMPORAL_WINDOW_DOMAIN = "teacher"
WINDOW_STARTS_HASH_ENCODING = "lmdb_index_ordered_uint64_big_endian"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a video-level Nyström RBF-MMD teacher reference from an "
            "ODE teacher LMDB using the exact Wan-VAE/V-JEPA2 dense pipeline."
        )
    )
    parser.add_argument(
        "--teacher-lmdb",
        type=Path,
        required=True,
        help="ODE LMDB containing latents and prompts arrays.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Readable local Hugging Face V-JEPA2 model directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Final atomic torch reference-pack path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Use the first N LMDB rows; 0 traverses the complete actual LMDB.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-rank extraction batch size; 1 is the memory-safe default.",
    )

    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument(
        "--frame-sampling",
        choices=FRAME_SAMPLING_CHOICES,
        default="center_contiguous",
        help=(
            "center_contiguous preserves the legacy adapter-selected center "
            "window; random_contiguous_per_video uses one stable LMDB-row-keyed "
            "contiguous window."
        ),
    )
    parser.add_argument(
        "--source-num-frames",
        type=int,
        default=81,
        help=(
            "Expected decoded source length for random contiguous sampling "
            "(default: 81). It is ignored by the legacy center path."
        ),
    )
    parser.add_argument(
        "--window-seed",
        type=str,
        default="0",
        help=(
            "Non-empty string seed for stateless per-video temporal-window "
            "selection."
        ),
    )
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--resize-short-side", type=int, default=292)
    parser.add_argument("--crop-size", type=int, default=256)

    parser.add_argument("--num-landmarks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--median-max-samples",
        type=int,
        default=2048,
        help="Deterministic subset cap for automatic median bandwidth.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="Fixed Gaussian bandwidth; omitted lets the builder estimate it.",
    )
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Sample chunk size used by rank-0 Nyström construction.",
    )

    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help=(
            "Parent for FP16 rank shards. By default /dev/shm is preferred "
            "when writable and large enough, then the platform temp directory."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output pack.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Rank-local extraction progress interval.",
    )
    parser.add_argument(
        "--distributed-timeout-minutes",
        type=int,
        default=360,
        help="Long timeout while rank 0 builds the reference after extraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Inspect LMDB/model geometry, distribution and disk budgets without "
            "loading either model or writing files."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "source_num_frames": args.source_num_frames,
        "num_tokens": args.num_tokens,
        "resize_short_side": args.resize_short_side,
        "crop_size": args.crop_size,
        "num_landmarks": args.num_landmarks,
        "median_max_samples": args.median_max_samples,
        "kmeans_iterations": args.kmeans_iterations,
        "chunk_size": args.chunk_size,
        "distributed_timeout_minutes": args.distributed_timeout_minutes,
    }
    invalid = {name: value for name, value in positive.items() if int(value) <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.layer < 0:
        raise ValueError("--layer must be non-negative")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if not isinstance(args.window_seed, str) or not args.window_seed:
        raise ValueError("--window-seed must be a non-empty string")
    if "\0" in args.window_seed:
        raise ValueError("--window-seed must not contain NUL characters")
    if (
        args.frame_sampling == "random_contiguous_per_video"
        and args.source_num_frames < args.num_frames
    ):
        raise ValueError(
            "--source-num-frames must be >= --num-frames for random "
            "contiguous sampling"
        )
    if args.seed != 0:
        raise ValueError(
            "--seed currently must be 0; landmark selection is deterministic")
    if args.resize_short_side < args.crop_size:
        raise ValueError("--resize-short-side must be >= --crop-size")
    if args.ridge <= 0.0 or not math.isfinite(float(args.ridge)):
        raise ValueError("--ridge must be finite and positive")
    if args.bandwidth is not None and (
        args.bandwidth <= 0.0 or not math.isfinite(float(args.bandwidth))
    ):
        raise ValueError("--bandwidth must be finite and positive")
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative")


def _read_model_geometry(model_id: str) -> Dict[str, int]:
    model_path = Path(os.path.expanduser(model_id))
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"--model-id must be a readable local directory, got {model_id!r}"
        )
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"V-JEPA2 config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = (
        "hidden_size",
        "num_hidden_layers",
        "patch_size",
        "tubelet_size",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            f"V-JEPA2 config {config_path} is missing fields {missing}")
    return {key: int(config[key]) for key in required}


def _inspect_lmdb(path: Path) -> Dict[str, Any]:
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError(
            "python-lmdb is required; use the same Python/PYTHONPATH as training"
        ) from exc

    from utils.lmdb import get_array_shape_from_lmdb

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"teacher LMDB directory not found: {resolved}")
    env = lmdb.open(
        str(resolved),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        latents_shape = tuple(
            int(value) for value in get_array_shape_from_lmdb(env, "latents"))
        prompts_shape = tuple(
            int(value) for value in get_array_shape_from_lmdb(env, "prompts"))
    finally:
        env.close()
    if len(latents_shape) != 6:
        raise ValueError(
            "teacher LMDB latents must be "
            "[samples,trajectory_steps,frames,channels,height,width], got "
            f"{latents_shape}"
        )
    if prompts_shape != (latents_shape[0],):
        raise ValueError(
            "teacher LMDB prompts/latents count mismatch: "
            f"{prompts_shape} vs {latents_shape}")
    data_path = resolved / "data.mdb"
    stat = data_path.stat()
    return {
        "path": str(resolved),
        "latents_shape": latents_shape,
        "prompts_shape": prompts_shape,
        "data_mdb_size_bytes": int(stat.st_size),
        "data_mdb_mtime_ns": int(stat.st_mtime_ns),
    }


def _selected_count(total: int, max_samples: int) -> int:
    count = int(total) if int(max_samples) == 0 else min(int(total), int(max_samples))
    if count <= 0:
        raise ValueError("the selected teacher reference set is empty")
    return count


def _rank_indices(count: int, rank: int, world_size: int) -> range:
    return range(int(rank), int(count), int(world_size))


def _window_starts_sha256(starts: Sequence[int]) -> str:
    """Hash starts in selected-LMDB-index order using fixed-width integers."""
    digest = hashlib.sha256()
    for start in starts:
        value = int(start)
        if value < 0:
            raise ValueError("temporal window starts must be non-negative")
        digest.update(value.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _random_teacher_window_starts(
    args: argparse.Namespace,
    indices: Sequence[int],
) -> list[int]:
    from video_fd.encoders.vjepa2_temporal_sampling import (
        stateless_temporal_window_start,
    )

    return [
        stateless_temporal_window_start(
            str(args.window_seed),
            TEACHER_TEMPORAL_WINDOW_DOMAIN,
            int(index),
            source_num_frames=int(args.source_num_frames),
            window_num_frames=int(args.num_frames),
        )
        for index in indices
    ]


def _temporal_sampling_plan(
    args: argparse.Namespace,
    selected_count: int,
) -> Dict[str, Any]:
    """Return JSON-safe temporal provenance visible in plan/dry-run output."""
    sampling = str(args.frame_sampling)
    common = {
        "frame_sampling": sampling,
        "window_num_frames": int(args.num_frames),
        "window_starts_sha256_encoding": WINDOW_STARTS_HASH_ENCODING,
    }
    if sampling == "center_contiguous":
        # Preserve the legacy path: the adapter derives the center from the
        # actual decoded length. source_num_frames/window_seed are random-mode
        # inputs and intentionally do not change the old center contract.
        return {
            **common,
            "source_num_frames": None,
            "window_seed": None,
            "window_domain": None,
            "window_start_min": None,
            "window_start_max": None,
            "teacher_window_starts_sha256": None,
            "selection": "legacy_adapter_center",
        }

    starts = _random_teacher_window_starts(
        args, list(range(int(selected_count))))
    return {
        **common,
        "source_num_frames": int(args.source_num_frames),
        "window_seed": str(args.window_seed),
        "window_domain": TEACHER_TEMPORAL_WINDOW_DOMAIN,
        "window_start_min": 0,
        "window_start_max": (
            int(args.source_num_frames) - int(args.num_frames)),
        "teacher_window_starts_sha256": _window_starts_sha256(starts),
        "selection": (
            "sha256(seed\\0domain\\0decimal_lmdb_index)"
            "_first_u64_big_endian_mod_num_windows"
        ),
    }


def _select_teacher_temporal_windows(
    video_clip_01: torch.Tensor,
    sample_indices: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, list[int]]:
    """Select teacher windows while retaining the legacy center behavior."""
    if video_clip_01.ndim != 5:
        raise ValueError(
            "decoded teacher video must be [B,T,3,H,W], got "
            f"{tuple(video_clip_01.shape)}")
    if video_clip_01.shape[0] != len(sample_indices):
        raise ValueError(
            "decoded teacher batch/sample index count mismatch: "
            f"{video_clip_01.shape[0]} != {len(sample_indices)}")
    total_frames = int(video_clip_01.shape[1])
    window_frames = int(args.num_frames)
    if total_frames < window_frames:
        raise ValueError(
            f"decoded teacher video has {total_frames} frames, fewer than "
            f"--num-frames={window_frames}")

    if str(args.frame_sampling) == "center_contiguous":
        start = (total_frames - window_frames) // 2
        # Do not crop here: forwarding the original decoded video preserves the
        # historical adapter-centered preprocessing byte for byte.
        return video_clip_01, [start] * len(sample_indices)

    expected_source_frames = int(args.source_num_frames)
    if total_frames != expected_source_frames:
        raise ValueError(
            "random contiguous teacher sampling requires the decoded source "
            f"length to equal --source-num-frames: {total_frames} != "
            f"{expected_source_frames}")
    starts = _random_teacher_window_starts(args, sample_indices)
    clips = torch.stack([
        video_clip_01[batch_index].narrow(0, start, window_frames)
        for batch_index, start in enumerate(starts)
    ], dim=0)
    return clips, starts


def _required_temp_bytes(
    count: int,
    num_tokens: int,
    feature_dim: int,
) -> int:
    feature_bytes = int(count) * int(num_tokens) * int(feature_dim) * 2
    return int(math.ceil(feature_bytes * 1.15)) + 64 * MIB


def _required_output_bytes(
    num_landmarks: int,
    num_tokens: int,
    feature_dim: int,
) -> int:
    landmark_bytes = (
        int(num_landmarks) * int(num_tokens) * int(feature_dim) * 4)
    return int(math.ceil(landmark_bytes * 1.25)) + 32 * MIB


def _choose_temp_parent(
    requested: Optional[Path],
    required_bytes: int,
) -> Tuple[Path, int]:
    candidates: Sequence[Path]
    if requested is not None:
        candidates = (requested.expanduser(),)
    else:
        candidates = (Path("/dev/shm"), Path(tempfile.gettempdir()))

    errors = []
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
            if not candidate.is_dir():
                errors.append(f"{candidate}: not a directory")
                continue
            if not os.access(candidate, os.W_OK | os.X_OK):
                errors.append(f"{candidate}: not writable")
                continue
            free = int(shutil.disk_usage(candidate).free)
            if free < required_bytes:
                errors.append(
                    f"{candidate}: {free / MIB:.1f} MiB free, "
                    f"{required_bytes / MIB:.1f} MiB required")
                continue
            return candidate, free
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "no temporary shard directory has sufficient capacity: "
        + "; ".join(errors)
    )


def _check_output(
    output: Path,
    *,
    required_bytes: int,
    overwrite: bool,
) -> Dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it")
    parent = output.parent
    if not parent.exists():
        existing = parent
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not existing.is_dir():
            raise FileNotFoundError(
                f"no existing output parent for capacity check: {parent}")
        capacity_parent = existing
    else:
        capacity_parent = parent
    free = int(shutil.disk_usage(capacity_parent).free)
    if free < required_bytes:
        raise RuntimeError(
            f"output filesystem has only {free / MIB:.1f} MiB free; "
            f"at least {required_bytes / MIB:.1f} MiB is required")
    return {
        "path": str(output),
        "capacity_parent": str(capacity_parent),
        "free_bytes": free,
    }


def _plan(args: argparse.Namespace, world_size: int) -> Dict[str, Any]:
    lmdb_info = _inspect_lmdb(args.teacher_lmdb)
    geometry = _read_model_geometry(args.model_id)
    if args.layer >= geometry["num_hidden_layers"]:
        raise ValueError(
            f"--layer={args.layer} exceeds model's "
            f"{geometry['num_hidden_layers']} blocks")
    if args.num_frames % geometry["tubelet_size"] != 0:
        raise ValueError("--num-frames must be divisible by model tubelet_size")
    if args.crop_size % geometry["patch_size"] != 0:
        raise ValueError("--crop-size must be divisible by model patch_size")
    raw_tokens = (
        args.num_frames // geometry["tubelet_size"]
        * (args.crop_size // geometry["patch_size"]) ** 2
    )
    if args.num_tokens > raw_tokens:
        raise ValueError(
            f"--num-tokens={args.num_tokens} exceeds raw grid {raw_tokens}")
    count = _selected_count(lmdb_info["latents_shape"][0], args.max_samples)
    if args.num_landmarks > count:
        raise ValueError(
            f"--num-landmarks={args.num_landmarks} exceeds {count} samples")

    temp_required = _required_temp_bytes(
        count, args.num_tokens, geometry["hidden_size"])
    temp_parent, temp_free = _choose_temp_parent(args.tmp_dir, temp_required)
    output_required = _required_output_bytes(
        args.num_landmarks, args.num_tokens, geometry["hidden_size"])
    output_info = _check_output(
        args.output,
        required_bytes=output_required,
        overwrite=args.overwrite,
    )
    per_rank = [
        len(_rank_indices(count, rank, world_size))
        for rank in range(world_size)
    ]
    return {
        "tool_format_version": TOOL_FORMAT_VERSION,
        "teacher_lmdb": lmdb_info,
        "model_id": str(Path(args.model_id).expanduser().resolve()),
        "model_geometry": geometry,
        "feature_geometry": {
            "raw_tokens": raw_tokens,
            "sample_shape": [args.num_tokens, geometry["hidden_size"]],
        },
        "selected_samples": count,
        "temporal_sampling": _temporal_sampling_plan(args, count),
        "world_size": int(world_size),
        "samples_per_rank": per_rank,
        "num_landmarks": int(args.num_landmarks),
        "temporary": {
            "parent": str(temp_parent),
            "required_bytes": temp_required,
            "free_bytes": temp_free,
        },
        "output": {
            **output_info,
            "required_bytes": output_required,
        },
    }


def _setup_distributed(timeout_minutes: int) -> Tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "reference feature extraction requires CUDA; use --dry-run for "
            "CPU-only validation")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=int(timeout_minutes)),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return torch.device("cuda", local_rank), rank, world_size


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _broadcast_rank0_object(value: Any, rank: int) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    payload = [value if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prompt_entry_digest(index: int, prompt: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(int(index).to_bytes(8, byteorder="little", signed=False))
    digest.update(b"\0")
    digest.update(str(prompt).encode("utf-8"))
    return digest.digest()


@torch.no_grad()
def _extract_rank_shard(
    args: argparse.Namespace,
    plan: Dict[str, Any],
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    shard_path: Path,
) -> None:
    from utils.dataset import ODERegressionLMDBDataset
    from utils.wan_wrapper import WanVAEWrapper
    from video_fd.encoders.vjepa2_adapter import VJEPA2Adapter
    from video_fd.encoders.vjepa2_dense_metadata import (
        uniform_dense_token_indices,
    )

    selected_count = int(plan["selected_samples"])
    assigned = list(_rank_indices(selected_count, rank, world_size))
    feature_dim = int(plan["model_geometry"]["hidden_size"])
    raw_tokens = int(plan["feature_geometry"]["raw_tokens"])

    dataset = ODERegressionLMDBDataset(
        str(args.teacher_lmdb.expanduser().resolve()),
        max_pair=selected_count,
    )
    if len(dataset) != selected_count:
        raise RuntimeError(
            f"LMDB length changed after planning: {len(dataset)} != "
            f"{selected_count}")

    vae = (
        WanVAEWrapper()
        .to(device=device, dtype=torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )
    encoder = (
        VJEPA2Adapter({
            "name": "vjepa2_dense",
            "model_id": str(Path(args.model_id).expanduser().resolve()),
            "input_frames": int(args.num_frames),
            "dense_num_frames": int(args.num_frames),
            "input_size": int(args.crop_size),
            "resize_short_side": int(args.resize_short_side),
            "crop_mode": "center",
            "dense_only": True,
            "dense_layer_index": int(args.layer),
        })
        .to(device=device, dtype=torch.float32)
        .eval()
        .requires_grad_(False)
    )
    actual = {
        "raw_tokens": int(encoder.dense_raw_tokens),
        "feature_dim": int(encoder.dense_feature_dim),
        "retained_blocks": int(encoder.retained_dense_blocks),
    }
    expected = {
        "raw_tokens": raw_tokens,
        "feature_dim": feature_dim,
        "retained_blocks": int(args.layer) + 1,
    }
    if actual != expected:
        raise RuntimeError(
            f"loaded V-JEPA2 geometry mismatch: actual={actual}, "
            f"expected={expected}")

    token_indices = uniform_dense_token_indices(
        raw_tokens,
        args.num_tokens,
        device=device,
    )
    feature_chunks = []
    index_chunks = []
    prompt_digests = []
    temporal_window_starts = []
    started = time.monotonic()

    for offset in range(0, len(assigned), args.batch_size):
        batch_indices = assigned[offset:offset + args.batch_size]
        items = [dataset[index] for index in batch_indices]
        trajectories = [item["ode_latent"] for item in items]
        expected_tail = tuple(plan["teacher_lmdb"]["latents_shape"][1:])
        for index, trajectory in zip(batch_indices, trajectories):
            if tuple(trajectory.shape) != expected_tail:
                raise ValueError(
                    f"LMDB row {index} shape {tuple(trajectory.shape)} != "
                    f"{expected_tail}")
            if not torch.isfinite(trajectory).all():
                raise FloatingPointError(
                    f"LMDB row {index} contains non-finite latents")

        latent = torch.stack(
            [trajectory[-1] for trajectory in trajectories],
            dim=0,
        ).to(device=device, dtype=torch.bfloat16)
        pixels = vae.decode_to_pixel(latent.detach())
        video01 = ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)
        encoder_video01, batch_window_starts = (
            _select_teacher_temporal_windows(
                video01,
                batch_indices,
                args,
            )
        )
        features = encoder.forward_dense(
            encoder_video01,
            layer_index=int(args.layer),
            token_indices=token_indices,
            gradient_checkpointing=False,
        ).float()
        expected_shape = (
            len(batch_indices),
            int(args.num_tokens),
            feature_dim,
        )
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"V-JEPA2 features {tuple(features.shape)} != "
                f"{expected_shape}")
        if not torch.isfinite(features).all():
            raise FloatingPointError(
                f"non-finite V-JEPA2 features for indices {batch_indices}")
        features_fp16 = features.to(dtype=torch.float16, device="cpu")
        if not torch.isfinite(features_fp16).all():
            raise FloatingPointError(
                "FP16 shard conversion overflowed for indices "
                f"{batch_indices}")
        feature_chunks.append(features_fp16)
        index_chunks.append(torch.tensor(batch_indices, dtype=torch.long))
        prompt_digests.extend(
            _prompt_entry_digest(index, item["prompts"])
            for index, item in zip(batch_indices, items)
        )
        temporal_window_starts.extend(batch_window_starts)

        del (
            latent,
            pixels,
            video01,
            encoder_video01,
            features,
            features_fp16,
            items,
            trajectories,
        )
        completed = min(offset + len(batch_indices), len(assigned))
        if args.log_every > 0 and (
            completed == len(assigned) or completed % args.log_every == 0
        ):
            print(
                f"[extract rank {rank}] {completed}/{len(assigned)} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )

    if feature_chunks:
        shard_features = torch.cat(feature_chunks, dim=0).contiguous()
        shard_indices = torch.cat(index_chunks, dim=0).contiguous()
        digest_tensor = torch.tensor(
            [list(value) for value in prompt_digests],
            dtype=torch.uint8,
        )
        window_starts_tensor = torch.tensor(
            temporal_window_starts, dtype=torch.long)
    else:
        shard_features = torch.empty(
            0, args.num_tokens, feature_dim, dtype=torch.float16)
        shard_indices = torch.empty(0, dtype=torch.long)
        digest_tensor = torch.empty(0, 32, dtype=torch.uint8)
        window_starts_tensor = torch.empty(0, dtype=torch.long)

    payload = {
        "shard_format_version": TOOL_FORMAT_VERSION,
        "rank": int(rank),
        "world_size": int(world_size),
        "indices": shard_indices,
        "features": shard_features,
        "prompt_entry_sha256": digest_tensor,
        "temporal_window_starts": window_starts_tensor,
        "sample_shape": [int(args.num_tokens), feature_dim],
        "feature_dtype": "float16",
    }
    _atomic_torch_save(payload, shard_path)
    print(
        f"[extract rank {rank}] wrote {shard_path} "
        f"{tuple(shard_features.shape)}",
        flush=True,
    )
    del encoder, vae, dataset, payload, feature_chunks, shard_features
    gc.collect()
    torch.cuda.empty_cache()


def _load_and_merge_shards(
    shard_dir: Path,
    *,
    world_size: int,
    selected_count: int,
    sample_shape: Sequence[int],
) -> Tuple[torch.Tensor, str, str, str]:
    features_parts = []
    indices_parts = []
    digest_parts = []
    window_start_parts = []
    for rank in range(world_size):
        path = shard_dir / f"rank_{rank:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing rank shard: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("shard_format_version", -1)) != TOOL_FORMAT_VERSION:
            raise ValueError(f"incompatible shard format in {path}")
        if int(payload.get("rank", -1)) != rank:
            raise ValueError(f"rank identity mismatch in {path}")
        if int(payload.get("world_size", -1)) != world_size:
            raise ValueError(f"world-size mismatch in {path}")
        features = payload.get("features")
        indices = payload.get("indices")
        digests = payload.get("prompt_entry_sha256")
        window_starts = payload.get("temporal_window_starts")
        expected_feature_tail = tuple(int(value) for value in sample_shape)
        if (
            not isinstance(features, torch.Tensor)
            or features.dtype != torch.float16
            or features.ndim != 3
            or tuple(features.shape[1:]) != expected_feature_tail
        ):
            raise ValueError(
                f"invalid FP16 feature shard {path}: "
                f"{getattr(features, 'shape', None)}")
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.long
            or tuple(indices.shape) != (features.shape[0],)
        ):
            raise ValueError(f"invalid indices in {path}")
        if (
            not isinstance(digests, torch.Tensor)
            or digests.dtype != torch.uint8
            or tuple(digests.shape) != (features.shape[0], 32)
        ):
            raise ValueError(f"invalid prompt hashes in {path}")
        if (
            not isinstance(window_starts, torch.Tensor)
            or window_starts.dtype != torch.long
            or tuple(window_starts.shape) != (features.shape[0],)
            or bool((window_starts < 0).any().item())
        ):
            raise ValueError(f"invalid temporal window starts in {path}")
        if not torch.isfinite(features).all():
            raise FloatingPointError(f"non-finite FP16 features in {path}")
        features_parts.append(features)
        indices_parts.append(indices)
        digest_parts.append(digests)
        window_start_parts.append(window_starts)

    features = torch.cat(features_parts, dim=0)
    indices = torch.cat(indices_parts, dim=0)
    digests = torch.cat(digest_parts, dim=0)
    window_starts = torch.cat(window_start_parts, dim=0)
    if features.shape[0] != selected_count:
        raise ValueError(
            f"rank shards contain {features.shape[0]} samples, expected "
            f"{selected_count}")
    order = torch.argsort(indices, stable=True)
    indices = indices.index_select(0, order)
    expected_indices = torch.arange(selected_count, dtype=torch.long)
    if not torch.equal(indices, expected_indices):
        raise ValueError(
            "rank shards do not cover each selected LMDB index exactly once")
    features = features.index_select(0, order).contiguous()
    digests = digests.index_select(0, order).contiguous()
    window_starts = window_starts.index_select(0, order).contiguous()
    prompt_index_hash = hashlib.sha256(
        digests.numpy().tobytes()).hexdigest()
    feature_bank_hash = hashlib.sha256(
        features.view(torch.uint8).numpy().tobytes()).hexdigest()
    window_starts_hash = _window_starts_sha256(window_starts.tolist())
    return (
        features,
        prompt_index_hash,
        feature_bank_hash,
        window_starts_hash,
    )


def _reference_metadata(
    args: argparse.Namespace,
    plan: Dict[str, Any],
    *,
    prompt_index_hash: str,
    feature_bank_hash: str,
    teacher_window_starts_hash: Optional[str] = None,
) -> Dict[str, Any]:
    from video_fd.encoders.vjepa2_dense_metadata import (
        canonical_vjepa2_dense_feature_metadata,
    )

    geometry = plan["model_geometry"]
    feature_kwargs = {
        "model_id": str(Path(args.model_id).expanduser().resolve()),
        "layer_index": args.layer,
        "num_frames": args.num_frames,
        "resize_short_side": args.resize_short_side,
        "crop_size": args.crop_size,
        "raw_tokens": plan["feature_geometry"]["raw_tokens"],
        "num_tokens": args.num_tokens,
        "feature_dim": geometry["hidden_size"],
    }
    if args.frame_sampling == "random_contiguous_per_video":
        feature_kwargs.update({
            "frame_sampling": args.frame_sampling,
            "source_num_frames": args.source_num_frames,
            "window_seed": args.window_seed,
        })
    feature_metadata = canonical_vjepa2_dense_feature_metadata(
        **feature_kwargs)
    source_metadata = {
        "kind": "ode_teacher_lmdb_final_latent",
        "lmdb_path": plan["teacher_lmdb"]["path"],
        "latents_shape": list(plan["teacher_lmdb"]["latents_shape"]),
        "prompts_shape": list(plan["teacher_lmdb"]["prompts_shape"]),
        "selected_samples": int(plan["selected_samples"]),
        "selected_index_start": 0,
        "selected_index_stop_exclusive": int(
            plan["selected_samples"]),
        "trajectory_index": -1,
        "lmdb_storage_dtype": "float16",
        "data_mdb_size_bytes": int(
            plan["teacher_lmdb"]["data_mdb_size_bytes"]),
        "data_mdb_mtime_ns": int(
            plan["teacher_lmdb"]["data_mdb_mtime_ns"]),
        "prompt_index_sha256": prompt_index_hash,
        "feature_bank_fp16_sha256": feature_bank_hash,
    }
    temporal_sampling = plan["temporal_sampling"]
    if temporal_sampling["frame_sampling"] == (
        "random_contiguous_per_video"
    ):
        expected_starts_hash = temporal_sampling[
            "teacher_window_starts_sha256"]
        if teacher_window_starts_hash != expected_starts_hash:
            raise ValueError(
                "merged teacher temporal-window starts do not match the "
                "dry-run/plan identity: "
                f"{teacher_window_starts_hash!r} != {expected_starts_hash!r}"
            )
        source_metadata.update({
            "frame_sampling": temporal_sampling["frame_sampling"],
            "source_num_frames": temporal_sampling["source_num_frames"],
            "window_num_frames": temporal_sampling["window_num_frames"],
            "window_seed": temporal_sampling["window_seed"],
            "window_domain": temporal_sampling["window_domain"],
            "window_start_min": temporal_sampling["window_start_min"],
            "window_start_max": temporal_sampling["window_start_max"],
            "teacher_window_starts_sha256": teacher_window_starts_hash,
            "window_starts_sha256_encoding": (
                temporal_sampling["window_starts_sha256_encoding"]),
        })

    return {
        "feature": feature_metadata,
        "reference": {
            "source": source_metadata,
            "extraction": {
                "tool": (
                    "tools/precompute_vjepa2_dense_nystrom_reference.py"),
                "tool_format_version": TOOL_FORMAT_VERSION,
                "world_size": int(plan["world_size"]),
                "vae_compute_dtype": "bfloat16",
                "vae_decode_use_cache": False,
                "decoded_pixel_transform": "((pixels+1)*0.5).clamp(0,1)",
                "vjepa2_compute_dtype": "float32",
                "rank_shard_dtype": "float16",
            },
            "build_request": {
                "num_landmarks": int(args.num_landmarks),
                "seed": int(args.seed),
                "bandwidth": (
                    None if args.bandwidth is None else float(args.bandwidth)),
                "median_max_samples": int(args.median_max_samples),
                "kmeans_iterations": int(args.kmeans_iterations),
                "ridge": float(args.ridge),
                "chunk_size": int(args.chunk_size),
            },
        },
    }


def _build_and_save_reference(
    args: argparse.Namespace,
    plan: Dict[str, Any],
    shard_dir: Path,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    from losses.video_nystrom_mmd import (
        build_video_nystrom_reference,
        load_reference_pack,
        save_reference_pack,
        validate_reference_pack,
    )

    (
        features,
        prompt_index_hash,
        feature_bank_hash,
        teacher_window_starts_hash,
    ) = _load_and_merge_shards(
        shard_dir,
        world_size=int(plan["world_size"]),
        selected_count=int(plan["selected_samples"]),
        sample_shape=plan["feature_geometry"]["sample_shape"],
    )
    metadata = _reference_metadata(
        args,
        plan,
        prompt_index_hash=prompt_index_hash,
        feature_bank_hash=feature_bank_hash,
        teacher_window_starts_hash=teacher_window_starts_hash,
    )
    print(
        f"[merge] building m={args.num_landmarks} Nyström reference from "
        f"{tuple(features.shape)} FP16 video matrices on {device}",
        flush=True,
    )
    reference = features.to(device=device, non_blocking=False)
    pack = build_video_nystrom_reference(
        reference,
        num_landmarks=int(args.num_landmarks),
        metadata=metadata,
        bandwidth=args.bandwidth,
        median_max_samples=int(args.median_max_samples),
        kmeans_iterations=int(args.kmeans_iterations),
        ridge=float(args.ridge),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
    )
    validate_reference_pack(pack)
    del reference
    torch.cuda.empty_cache()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        save_reference_pack(pack, temporary)
        published = load_reference_pack(temporary, map_location="cpu")
        validate_reference_pack(published)
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"output appeared during construction: {output}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output": str(output),
        "samples": int(features.shape[0]),
        "sample_shape": list(features.shape[1:]),
        "num_landmarks": int(pack["landmarks"].shape[0]),
        "sigma": float(torch.as_tensor(pack["sigma"]).item()),
        "content_hash": str(pack["content_hash"]),
        "prompt_index_sha256": prompt_index_hash,
        "feature_bank_fp16_sha256": feature_bank_hash,
        "teacher_window_starts_sha256": teacher_window_starts_hash,
    }


def _run(args: argparse.Namespace) -> None:
    _validate_args(args)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    plan = _plan(args, env_world_size)
    if args.dry_run:
        if env_rank == 0:
            print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return

    device, rank, world_size = _setup_distributed(
        args.distributed_timeout_minutes)
    if world_size != int(plan["world_size"]):
        raise RuntimeError(
            f"distributed world size changed after planning: {world_size} != "
            f"{plan['world_size']}")

    temp_result = None
    if rank == 0:
        try:
            parent = Path(plan["temporary"]["parent"])
            shard_dir = Path(tempfile.mkdtemp(
                prefix="vjepa2_dense_nystrom_",
                dir=parent,
            ))
            temp_result = {"path": str(shard_dir), "error": None}
        except Exception as exc:
            temp_result = {
                "path": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    temp_result = _broadcast_rank0_object(temp_result, rank)
    if temp_result["error"] is not None:
        raise RuntimeError(
            f"rank 0 could not create shard directory: "
            f"{temp_result['error']}")
    shard_dir = Path(temp_result["path"])

    success = False
    try:
        _extract_rank_shard(
            args,
            plan,
            device=device,
            rank=rank,
            world_size=world_size,
            shard_path=shard_dir / f"rank_{rank:05d}.pt",
        )
        _barrier()
        summary = None
        build_error = None
        if rank == 0:
            try:
                summary = _build_and_save_reference(
                    args,
                    plan,
                    shard_dir,
                    device=device,
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                build_error = f"{type(exc).__name__}: {exc}"
        outcome = _broadcast_rank0_object(
            {"summary": summary, "error": build_error},
            rank,
        )
        if outcome["error"] is not None:
            raise RuntimeError(
                f"rank 0 reference construction failed: {outcome['error']}")
        _barrier()
        success = True
        if rank == 0:
            print(json.dumps(
                {"status": "ok", **outcome["summary"]},
                indent=2,
                sort_keys=True,
            ), flush=True)
    finally:
        if rank == 0:
            if success:
                shutil.rmtree(shard_dir)
                print(f"[cleanup] removed temporary shards {shard_dir}", flush=True)
            else:
                print(
                    f"[cleanup] preserving failed-run shards at {shard_dir}",
                    flush=True,
                )
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _run(args)


if __name__ == "__main__":
    main()

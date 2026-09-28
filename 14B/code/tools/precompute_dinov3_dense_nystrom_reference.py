#!/usr/bin/env python3
"""Build the fixed DINOv3 ViT-L/16 whole-video Nyström reference.

The teacher and objective are unchanged from the paired V-JEPA2 experiment:
the final latent from every row of the 1,537-video CausVid ODE LMDB is decoded
with the frozen Wan VAE, then represented by a frozen DINOv3 image encoder.
The DINOv3 feature map is fixed to 16 uniformly distributed source frames and
an 8x8 subset of the 16x16 patch grid per frame.  Frame-major concatenation
therefore produces one aligned ``[16 * 64, 1024]`` matrix per video.

Every torchrun rank atomically writes one indexed FP16 feature shard.  Rank 0
validates and merges all shards in LMDB-index order, constructs the existing
video-matrix Nyström objective (m=256, seed=0), atomically publishes the pack,
and reloads the *final path* for validation.  Temporary shards are removed only
after that final validation succeeds; failed runs retain them for diagnosis.
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
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TOOL_FORMAT_VERSION = 1
MIB = 1024 ** 2
EXPECTED_TEACHER_COUNT = 1537
EXPECTED_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
SOURCE_NUM_FRAMES = 81
FRAME_INDICES = (0, 5, 11, 16, 21, 27, 32, 37, 43, 48, 53, 59, 64, 69, 75, 80)
INPUT_SIZE = 256
PATCH_SIZE = 16
PATCH_GRID_SIZE = 16
SPATIAL_AXIS_INDICES = (0, 2, 4, 6, 9, 11, 13, 15)
SPATIAL_TOKEN_INDICES = tuple(
    row * PATCH_GRID_SIZE + column
    for row in SPATIAL_AXIS_INDICES
    for column in SPATIAL_AXIS_INDICES
)
PER_FRAME_TOKENS = 64
TOTAL_TOKENS = len(FRAME_INDICES) * PER_FRAME_TOKENS
FEATURE_DIM = 1024
FORWARD_CHUNK_FRAMES = 4
NUM_LANDMARKS = 256
NYSTROM_SEED = 0
MEDIAN_MAX_SAMPLES = EXPECTED_TEACHER_COUNT
KMEANS_ITERATIONS = 20
NYSTROM_RIDGE = 1.0e-6
KERNEL_CHUNK_SIZE = 32
WAN_VAE_MODEL_NAME = "Wan2.1-T2V-1.3B"
WAN_VAE_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
WAN_VAE_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed full-1537 DINOv3 ViT-L/16 video-matrix "
            "Nyström teacher reference."
        )
    )
    parser.add_argument("--teacher-lmdb", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument(
        "--source-commit", type=str, default=EXPECTED_SOURCE_COMMIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--expected-teacher-count", type=int, default=EXPECTED_TEACHER_COUNT)
    parser.add_argument(
        "--forward-chunk-frames", type=int, default=FORWARD_CHUNK_FRAMES)
    parser.add_argument("--num-landmarks", type=int, default=NUM_LANDMARKS)
    parser.add_argument("--seed", type=int, default=NYSTROM_SEED)
    parser.add_argument(
        "--median-max-samples", type=int, default=MEDIAN_MAX_SAMPLES)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument(
        "--kmeans-iterations", type=int, default=KMEANS_ITERATIONS)
    parser.add_argument("--ridge", type=float, default=NYSTROM_RIDGE)
    parser.add_argument("--chunk-size", type=int, default=KERNEL_CHUNK_SIZE)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--distributed-timeout-minutes", type=int, default=360)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch_size": args.batch_size,
        "expected_teacher_count": args.expected_teacher_count,
        "forward_chunk_frames": args.forward_chunk_frames,
        "num_landmarks": args.num_landmarks,
        "median_max_samples": args.median_max_samples,
        "kmeans_iterations": args.kmeans_iterations,
        "chunk_size": args.chunk_size,
        "distributed_timeout_minutes": args.distributed_timeout_minutes,
    }
    invalid = {name: value for name, value in positive.items() if int(value) <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if int(args.expected_teacher_count) != EXPECTED_TEACHER_COUNT:
        raise ValueError(
            "the canonical reference requires exactly "
            f"{EXPECTED_TEACHER_COUNT} teachers")
    if int(args.forward_chunk_frames) != FORWARD_CHUNK_FRAMES:
        raise ValueError(
            f"--forward-chunk-frames must be {FORWARD_CHUNK_FRAMES}")
    if int(args.num_landmarks) != NUM_LANDMARKS:
        raise ValueError(f"--num-landmarks must be {NUM_LANDMARKS}")
    if int(args.seed) != NYSTROM_SEED:
        raise ValueError(f"--seed must be {NYSTROM_SEED}")
    if int(args.median_max_samples) != MEDIAN_MAX_SAMPLES:
        raise ValueError(
            f"--median-max-samples must be {MEDIAN_MAX_SAMPLES}")
    if args.bandwidth is not None:
        raise ValueError(
            "the canonical reference requires median-estimated bandwidth")
    if int(args.kmeans_iterations) != KMEANS_ITERATIONS:
        raise ValueError(
            f"--kmeans-iterations must be {KMEANS_ITERATIONS}")
    if not math.isclose(
        float(args.ridge), NYSTROM_RIDGE, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(f"--ridge must be {NYSTROM_RIDGE}")
    if int(args.chunk_size) != KERNEL_CHUNK_SIZE:
        raise ValueError(f"--chunk-size must be {KERNEL_CHUNK_SIZE}")
    if str(args.source_commit) != EXPECTED_SOURCE_COMMIT:
        raise ValueError(
            "--source-commit must pin the approved DINOv3 source commit "
            f"{EXPECTED_SOURCE_COMMIT}")
    if float(args.ridge) <= 0.0 or not math.isfinite(float(args.ridge)):
        raise ValueError("--ridge must be finite and positive")
    if args.bandwidth is not None and (
        float(args.bandwidth) <= 0.0
        or not math.isfinite(float(args.bandwidth))
    ):
        raise ValueError("--bandwidth must be finite and positive")
    if int(args.log_every) < 0:
        raise ValueError("--log-every must be non-negative")


def _git_head(repo_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"failed to inspect DINOv3 source checkout {repo_path}") from exc
    return result.stdout.strip()


def _git_source_status(repo_path: Path) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"failed to inspect DINOv3 source worktree {repo_path}") from exc
    return result.stdout


def _inspect_assets(args: argparse.Namespace) -> Dict[str, Any]:
    repo_path = args.repo_path.expanduser().resolve()
    weight_path = args.weight_path.expanduser().resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"DINOv3 source directory not found: {repo_path}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"DINOv3 weight file not found: {weight_path}")
    source_commit = _git_head(repo_path)
    if source_commit != str(args.source_commit):
        raise ValueError(
            "DINOv3 source checkout commit mismatch: "
            f"{source_commit} != {args.source_commit}")
    source_status = _git_source_status(repo_path)
    if source_status.strip():
        raise ValueError(
            "DINOv3 source checkout has worktree changes; refusing "
            "to build a reference from unpinned source")
    stat = weight_path.stat()
    return {
        "repo_path": str(repo_path),
        "weight_path": str(weight_path),
        "source_commit": source_commit,
        "weight_size_bytes": int(stat.st_size),
        "weight_mtime_ns": int(stat.st_mtime_ns),
    }


def _inspect_lmdb(path: Path) -> Dict[str, Any]:
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError(
            "python-lmdb is required; use the same Python environment as training"
        ) from exc
    from utils.lmdb import get_array_shape_from_lmdb

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"teacher LMDB directory not found: {resolved}")
    env = lmdb.open(
        str(resolved), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        latents_shape = tuple(
            int(value) for value in get_array_shape_from_lmdb(env, "latents"))
        prompts_shape = tuple(
            int(value) for value in get_array_shape_from_lmdb(env, "prompts"))
    finally:
        env.close()
    if len(latents_shape) != 6:
        raise ValueError(
            "teacher LMDB latents must be [N,trajectory,T,C,H,W], got "
            f"{latents_shape}")
    if prompts_shape != (latents_shape[0],):
        raise ValueError(
            f"teacher prompts/latents count mismatch: {prompts_shape} vs "
            f"{latents_shape}")
    data_path = resolved / "data.mdb"
    stat = data_path.stat()
    return {
        "path": str(resolved),
        "latents_shape": latents_shape,
        "prompts_shape": prompts_shape,
        "data_mdb_size_bytes": int(stat.st_size),
        "data_mdb_mtime_ns": int(stat.st_mtime_ns),
    }


def _rank_indices(count: int, rank: int, world_size: int) -> range:
    return range(int(rank), int(count), int(world_size))


def _required_temp_bytes(count: int) -> int:
    feature_bytes = int(count) * TOTAL_TOKENS * FEATURE_DIM * 2
    return int(math.ceil(feature_bytes * 1.15)) + 64 * MIB


def _required_output_bytes() -> int:
    landmark_bytes = NUM_LANDMARKS * TOTAL_TOKENS * FEATURE_DIM * 4
    return int(math.ceil(landmark_bytes * 1.25)) + 32 * MIB


def _choose_temp_parent(
    requested: Optional[Path], required_bytes: int
) -> Tuple[Path, int]:
    candidates: Sequence[Path] = (
        (requested.expanduser(),)
        if requested is not None
        else (Path("/dev/shm"), Path(tempfile.gettempdir()))
    )
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
        + "; ".join(errors))


def _check_output(
    output: Path, *, required_bytes: int, overwrite: bool
) -> Dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it")
    existing = output.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.is_dir():
        raise FileNotFoundError(
            f"no existing output parent for capacity check: {output.parent}")
    free = int(shutil.disk_usage(existing).free)
    if free < required_bytes:
        raise RuntimeError(
            f"output filesystem has only {free / MIB:.1f} MiB free; "
            f"{required_bytes / MIB:.1f} MiB required")
    return {
        "path": str(output),
        "capacity_parent": str(existing),
        "free_bytes": free,
        "required_bytes": required_bytes,
    }


def _plan(args: argparse.Namespace, world_size: int) -> Dict[str, Any]:
    lmdb = _inspect_lmdb(args.teacher_lmdb)
    count = int(lmdb["latents_shape"][0])
    if count != EXPECTED_TEACHER_COUNT:
        raise ValueError(
            "canonical DINOv3 reference requires the complete 1,537-row LMDB; "
            f"found {count} rows")
    if int(args.num_landmarks) > count:
        raise ValueError("number of landmarks exceeds teacher population")
    assets = _inspect_assets(args)
    temp_required = _required_temp_bytes(count)
    temp_parent, temp_free = _choose_temp_parent(args.tmp_dir, temp_required)
    output = _check_output(
        args.output,
        required_bytes=_required_output_bytes(),
        overwrite=bool(args.overwrite),
    )
    return {
        "tool_format_version": TOOL_FORMAT_VERSION,
        "teacher_lmdb": lmdb,
        "assets": assets,
        "selected_samples": count,
        "world_size": int(world_size),
        "samples_per_rank": [
            len(_rank_indices(count, rank, world_size))
            for rank in range(world_size)
        ],
        "feature_geometry": {
            "source_num_frames": SOURCE_NUM_FRAMES,
            "frame_indices": list(FRAME_INDICES),
            "input_size": INPUT_SIZE,
            "patch_size": PATCH_SIZE,
            "patch_grid_size": PATCH_GRID_SIZE,
            "spatial_axis_indices": list(SPATIAL_AXIS_INDICES),
            "spatial_token_indices": list(SPATIAL_TOKEN_INDICES),
            "raw_tokens_per_frame": PATCH_GRID_SIZE ** 2,
            "per_frame_tokens": PER_FRAME_TOKENS,
            "num_tokens": TOTAL_TOKENS,
            "feature_dim": FEATURE_DIM,
            "sample_shape": [TOTAL_TOKENS, FEATURE_DIM],
            "concat_order": "frame_major_then_spatial_row_major",
        },
        "num_landmarks": NUM_LANDMARKS,
        "temporary": {
            "parent": str(temp_parent),
            "required_bytes": temp_required,
            "free_bytes": temp_free,
        },
        "output": output,
    }


def _setup_distributed(timeout_minutes: int) -> Tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "reference extraction requires CUDA; use --dry-run on CPU")
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


def _canonical_sha256(value: Dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"identity target is not a regular file: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            digest.update(block)
    after = resolved.stat()
    if (
        int(before.st_size) != int(after.st_size)
        or int(before.st_mtime_ns) != int(after.st_mtime_ns)
    ):
        raise RuntimeError(f"identity target changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _vae_identity() -> Dict[str, Any]:
    """Resolve the exact Wan decoder used before DINOv3 extraction."""
    from utils.paths import model_path

    weights = model_path(WAN_VAE_MODEL_NAME) / "Wan2.1_VAE.pth"
    identity = {
        "wrapper": "utils.wan_wrapper.WanVAEWrapper",
        "wrapper_source": _file_identity(REPO_ROOT / "utils" / "wan_wrapper.py"),
        "model_name": WAN_VAE_MODEL_NAME,
        "weights": _file_identity(weights),
        "latent_channels": 16,
        "latent_mean": list(WAN_VAE_MEAN),
        "latent_std": list(WAN_VAE_STD),
        "mean_std_contract": "WanVAEWrapper.inline_constants.v1",
        "compute_dtype": "bfloat16",
        "decode_use_cache": False,
        "decode_output_range": "[-1,1]",
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _validate_reference_current_assets(
    pack: Mapping[str, Any],
    *,
    repo_path: Path,
    weight_path: Path,
    teacher_lmdb: Path,
) -> Dict[str, Any]:
    """Require a validated reference to match every current production asset.

    Callers must first run ``load_reference_pack`` (which verifies schema and
    content hash). This second gate prevents a valid but stale pack from being
    accepted after the DINO checkpoint, Wan VAE, or teacher LMDB changes.
    """
    if not isinstance(pack, Mapping):
        raise TypeError("reference pack must be a mapping")
    metadata = pack.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("reference pack metadata is missing")
    feature = metadata.get("feature")
    reference = metadata.get("reference")
    if not isinstance(feature, Mapping) or not isinstance(reference, Mapping):
        raise ValueError("reference feature/source metadata is missing")
    source = reference.get("source")
    saved_vae = reference.get("vae")
    if not isinstance(source, Mapping) or not isinstance(saved_vae, Mapping):
        raise ValueError("reference source or Wan VAE identity is missing")

    from video_fd.encoders.dinov3_dense_metadata import file_sha256

    resolved_repo = repo_path.expanduser().resolve(strict=True)
    resolved_weight = weight_path.expanduser().resolve(strict=True)
    current_weight_sha256 = file_sha256(resolved_weight)
    expected_feature_identity = {
        "repo_path": str(resolved_repo),
        "weight_path": str(resolved_weight),
        "weight_sha256": current_weight_sha256,
    }
    feature_mismatches = {
        key: {"saved": feature.get(key), "current": value}
        for key, value in expected_feature_identity.items()
        if feature.get(key) != value
    }
    if feature_mismatches:
        raise ValueError(
            "reference DINOv3 asset identity does not match current assets: "
            f"{feature_mismatches}")

    current_vae = _vae_identity()
    if dict(saved_vae) != current_vae:
        raise ValueError(
            "reference Wan VAE identity does not match current assets")

    current_lmdb = _inspect_lmdb(teacher_lmdb)
    expected_source_identity = {
        "lmdb_path": current_lmdb["path"],
        "latents_shape": list(current_lmdb["latents_shape"]),
        "prompts_shape": list(current_lmdb["prompts_shape"]),
        "selected_samples": int(current_lmdb["latents_shape"][0]),
        "data_mdb_size_bytes": int(current_lmdb["data_mdb_size_bytes"]),
        "data_mdb_mtime_ns": int(current_lmdb["data_mdb_mtime_ns"]),
    }
    source_mismatches = {
        key: {"saved": source.get(key), "current": value}
        for key, value in expected_source_identity.items()
        if source.get(key) != value
    }
    if source_mismatches:
        raise ValueError(
            "reference teacher LMDB identity does not match current assets: "
            f"{source_mismatches}")
    return {
        "dinov3_weight_sha256": current_weight_sha256,
        "vae_identity_sha256": current_vae["identity_sha256"],
        "teacher_lmdb": current_lmdb,
    }


def _prompt_entry_digest(index: int, prompt: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(int(index).to_bytes(8, byteorder="little", signed=False))
    digest.update(b"\0")
    digest.update(str(prompt).encode("utf-8"))
    return digest.digest()


def _load_dense_encoder(args: argparse.Namespace, device: torch.device):
    """The sole runtime boundary with the separately-owned DINOv3 adapter."""
    from video_fd.encoders.dinov3_adapter import DINOv3DenseAdapter

    encoder = DINOv3DenseAdapter({
        "name": "dinov3_dense",
        "repo_path": str(args.repo_path.expanduser().resolve()),
        "weight_path": str(args.weight_path.expanduser().resolve()),
        "source_commit": str(args.source_commit),
        "source_num_frames": SOURCE_NUM_FRAMES,
        "frame_indices": list(FRAME_INDICES),
        "input_size": INPUT_SIZE,
        "patch_size": PATCH_SIZE,
        "patch_grid_size": PATCH_GRID_SIZE,
        "spatial_axis_indices": list(SPATIAL_AXIS_INDICES),
        "forward_chunk_frames": int(args.forward_chunk_frames),
        "compute_dtype": "float32",
    })
    return encoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


@torch.no_grad()
def _extract_rank_shard(
    args: argparse.Namespace,
    plan: Dict[str, Any],
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    shard_path: Path,
    vae_identity: Dict[str, Any],
) -> None:
    from utils.dataset import ODERegressionLMDBDataset
    from utils.wan_wrapper import WanVAEWrapper

    selected_count = int(plan["selected_samples"])
    assigned = list(_rank_indices(selected_count, rank, world_size))
    dataset = ODERegressionLMDBDataset(
        str(args.teacher_lmdb.expanduser().resolve()), max_pair=selected_count)
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
    if not torch.equal(vae.mean, torch.tensor(WAN_VAE_MEAN)):
        raise RuntimeError("WanVAEWrapper latent mean changed from the reference contract")
    if not torch.equal(vae.std, torch.tensor(WAN_VAE_STD)):
        raise RuntimeError("WanVAEWrapper latent std changed from the reference contract")
    encoder = _load_dense_encoder(args, device)
    feature_metadata = encoder.feature_metadata()
    if not isinstance(feature_metadata, dict):
        raise TypeError("DINOv3 adapter feature_metadata() must return a dict")
    spatial_indices = torch.tensor(
        SPATIAL_TOKEN_INDICES, device=device, dtype=torch.long)

    feature_chunks = []
    index_chunks = []
    prompt_digests = []
    expected_tail = tuple(plan["teacher_lmdb"]["latents_shape"][1:])
    started = time.monotonic()
    for offset in range(0, len(assigned), int(args.batch_size)):
        batch_indices = assigned[offset:offset + int(args.batch_size)]
        items = [dataset[index] for index in batch_indices]
        trajectories = [item["ode_latent"] for item in items]
        for index, trajectory in zip(batch_indices, trajectories):
            if tuple(trajectory.shape) != expected_tail:
                raise ValueError(
                    f"LMDB row {index} shape {tuple(trajectory.shape)} != "
                    f"{expected_tail}")
            if not bool(torch.isfinite(trajectory).all().item()):
                raise FloatingPointError(
                    f"LMDB row {index} contains non-finite latents")
        latent = torch.stack(
            [trajectory[-1] for trajectory in trajectories], dim=0
        ).to(device=device, dtype=torch.bfloat16)
        pixels = vae.decode_to_pixel(latent.detach())
        video01 = ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)
        if (
            video01.ndim != 5
            or int(video01.shape[1]) != SOURCE_NUM_FRAMES
            or int(video01.shape[2]) != 3
        ):
            raise ValueError(
                "Wan VAE must decode [B,81,3,H,W], got "
                f"{tuple(video01.shape)}")
        features = encoder.forward_dense(
            video01,
            token_indices=spatial_indices,
            gradient_checkpointing=False,
        ).float()
        expected_shape = (len(batch_indices), TOTAL_TOKENS, FEATURE_DIM)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"DINOv3 features {tuple(features.shape)} != {expected_shape}")
        if not bool(torch.isfinite(features).all().item()):
            raise FloatingPointError(
                f"non-finite DINOv3 features for indices {batch_indices}")
        features_fp16 = features.to(dtype=torch.float16, device="cpu")
        if not bool(torch.isfinite(features_fp16).all().item()):
            raise FloatingPointError(
                f"FP16 conversion overflowed for indices {batch_indices}")
        feature_chunks.append(features_fp16)
        index_chunks.append(torch.tensor(batch_indices, dtype=torch.long))
        prompt_digests.extend(
            _prompt_entry_digest(index, item["prompts"])
            for index, item in zip(batch_indices, items)
        )
        del latent, pixels, video01, features, features_fp16, items, trajectories
        completed = min(offset + len(batch_indices), len(assigned))
        if int(args.log_every) > 0 and (
            completed == len(assigned) or completed % int(args.log_every) == 0
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
            [list(value) for value in prompt_digests], dtype=torch.uint8)
    else:
        shard_features = torch.empty(
            0, TOTAL_TOKENS, FEATURE_DIM, dtype=torch.float16)
        shard_indices = torch.empty(0, dtype=torch.long)
        digest_tensor = torch.empty(0, 32, dtype=torch.uint8)
    payload = {
        "shard_format_version": TOOL_FORMAT_VERSION,
        "rank": int(rank),
        "world_size": int(world_size),
        "indices": shard_indices,
        "features": shard_features,
        "prompt_entry_sha256": digest_tensor,
        "sample_shape": [TOTAL_TOKENS, FEATURE_DIM],
        "feature_dtype": "float16",
        "feature_metadata": feature_metadata,
        "vae_identity": vae_identity,
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
) -> Tuple[torch.Tensor, str, str, Dict[str, Any], Dict[str, Any]]:
    feature_parts = []
    index_parts = []
    digest_parts = []
    feature_metadata = None
    vae_identity = None
    expected_tail = tuple(int(value) for value in sample_shape)
    for rank in range(int(world_size)):
        path = shard_dir / f"rank_{rank:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing rank shard: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("shard_format_version", -1)) != TOOL_FORMAT_VERSION:
            raise ValueError(f"incompatible shard format in {path}")
        if int(payload.get("rank", -1)) != rank:
            raise ValueError(f"rank identity mismatch in {path}")
        if int(payload.get("world_size", -1)) != int(world_size):
            raise ValueError(f"world-size mismatch in {path}")
        features = payload.get("features")
        indices = payload.get("indices")
        digests = payload.get("prompt_entry_sha256")
        shard_feature_metadata = payload.get("feature_metadata")
        shard_vae_identity = payload.get("vae_identity")
        if (
            not isinstance(features, torch.Tensor)
            or features.dtype != torch.float16
            or features.ndim != 3
            or tuple(features.shape[1:]) != expected_tail
            or not bool(torch.isfinite(features).all().item())
        ):
            raise ValueError(f"invalid FP16 feature shard {path}")
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
        if not isinstance(shard_feature_metadata, dict):
            raise ValueError(f"missing DINOv3 feature metadata in {path}")
        if feature_metadata is None:
            feature_metadata = shard_feature_metadata
        elif shard_feature_metadata != feature_metadata:
            raise ValueError(
                f"DINOv3 feature metadata differs across rank shards: {path}")
        if not isinstance(shard_vae_identity, dict):
            raise ValueError(f"missing Wan VAE identity in {path}")
        if vae_identity is None:
            vae_identity = shard_vae_identity
        elif shard_vae_identity != vae_identity:
            raise ValueError(
                f"Wan VAE identity differs across rank shards: {path}")
        feature_parts.append(features)
        index_parts.append(indices)
        digest_parts.append(digests)
    features = torch.cat(feature_parts, dim=0)
    indices = torch.cat(index_parts, dim=0)
    digests = torch.cat(digest_parts, dim=0)
    if int(features.shape[0]) != int(selected_count):
        raise ValueError(
            f"rank shards contain {features.shape[0]} samples, expected "
            f"{selected_count}")
    order = torch.argsort(indices, stable=True)
    indices = indices.index_select(0, order)
    if not torch.equal(indices, torch.arange(selected_count, dtype=torch.long)):
        raise ValueError(
            "rank shards do not cover every selected LMDB index exactly once")
    features = features.index_select(0, order).contiguous()
    digests = digests.index_select(0, order).contiguous()
    prompt_hash = hashlib.sha256(digests.numpy().tobytes()).hexdigest()
    feature_hash = hashlib.sha256(
        features.view(torch.uint8).numpy().tobytes()).hexdigest()
    if feature_metadata is None:
        raise ValueError("rank shards contained no DINOv3 feature metadata")
    if vae_identity is None:
        raise ValueError("rank shards contained no Wan VAE identity")
    return features, prompt_hash, feature_hash, feature_metadata, vae_identity


def _validate_feature_metadata(
    args: argparse.Namespace,
    feature_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Rebuild the canonical schema and reject even consistent bad shards."""
    from video_fd.encoders.dinov3_dense_metadata import (
        canonical_dinov3_dense_feature_metadata,
        file_sha256,
    )

    repo_path = args.repo_path.expanduser().resolve()
    if _git_head(repo_path) != str(args.source_commit):
        raise RuntimeError("DINOv3 source commit changed during extraction")
    if _git_source_status(repo_path).strip():
        raise RuntimeError("DINOv3 source worktree changed during extraction")
    current_weight_sha256 = file_sha256(args.weight_path)
    expected = canonical_dinov3_dense_feature_metadata(
        repo_path=str(repo_path),
        weight_path=str(args.weight_path.expanduser().resolve()),
        source_commit=str(args.source_commit),
        weight_sha256=current_weight_sha256,
        source_num_frames=SOURCE_NUM_FRAMES,
        frame_indices=list(FRAME_INDICES),
        input_size=INPUT_SIZE,
        patch_size=PATCH_SIZE,
        patch_grid_size=PATCH_GRID_SIZE,
        spatial_axis_indices=list(SPATIAL_AXIS_INDICES),
        spatial_token_indices=list(SPATIAL_TOKEN_INDICES),
        per_frame_tokens=PER_FRAME_TOKENS,
        num_tokens=TOTAL_TOKENS,
        forward_chunk_frames=FORWARD_CHUNK_FRAMES,
        feature_dim=FEATURE_DIM,
    )
    if feature_metadata != expected:
        raise ValueError(
            "rank-shard DINOv3 metadata does not equal the canonical schema")
    return expected


def _reference_metadata(
    args: argparse.Namespace,
    plan: Dict[str, Any],
    *,
    feature_metadata: Dict[str, Any],
    vae_identity: Dict[str, Any],
    prompt_index_hash: str,
    feature_bank_hash: str,
) -> Dict[str, Any]:
    feature_metadata = _validate_feature_metadata(args, feature_metadata)
    if vae_identity != _vae_identity():
        raise RuntimeError(
            "Wan VAE asset or wrapper changed during teacher extraction")
    data_mdb = Path(plan["teacher_lmdb"]["path"]) / "data.mdb"
    data_stat = data_mdb.stat()
    if (
        int(data_stat.st_size)
        != int(plan["teacher_lmdb"]["data_mdb_size_bytes"])
        or int(data_stat.st_mtime_ns)
        != int(plan["teacher_lmdb"]["data_mdb_mtime_ns"])
    ):
        raise RuntimeError("teacher LMDB changed during DINOv3 extraction")
    source = {
        "kind": "causvid_ode_teacher_lmdb_final_latent",
        "lmdb_path": plan["teacher_lmdb"]["path"],
        "latents_shape": list(plan["teacher_lmdb"]["latents_shape"]),
        "prompts_shape": list(plan["teacher_lmdb"]["prompts_shape"]),
        "selected_samples": EXPECTED_TEACHER_COUNT,
        "selected_index_start": 0,
        "selected_index_stop_exclusive": EXPECTED_TEACHER_COUNT,
        "trajectory_index": -1,
        "lmdb_storage_dtype": "float16",
        "data_mdb_size_bytes": int(
            plan["teacher_lmdb"]["data_mdb_size_bytes"]),
        "data_mdb_mtime_ns": int(
            plan["teacher_lmdb"]["data_mdb_mtime_ns"]),
        "prompt_index_sha256": prompt_index_hash,
        "feature_bank_fp16_sha256": feature_bank_hash,
    }
    source["identity_sha256"] = _canonical_sha256(source)
    return {
        "feature": feature_metadata,
        "reference": {
            "source": source,
            "vae": vae_identity,
            "extraction": {
                "tool": "tools/precompute_dinov3_dense_nystrom_reference.py",
                "tool_format_version": TOOL_FORMAT_VERSION,
                "world_size": int(plan["world_size"]),
                "vae_compute_dtype": "bfloat16",
                "vae_decode_use_cache": False,
                "decoded_pixel_transform": "((pixels+1)*0.5).clamp(0,1)",
                "dinov3_compute_dtype": "float32",
                "forward_chunk_frames": FORWARD_CHUNK_FRAMES,
                "rank_shard_dtype": "float16",
            },
            "build_request": {
                "num_landmarks": NUM_LANDMARKS,
                "seed": NYSTROM_SEED,
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

    features, prompt_hash, feature_hash, feature_metadata, vae_identity = (
        _load_and_merge_shards(
            shard_dir,
            world_size=int(plan["world_size"]),
            selected_count=EXPECTED_TEACHER_COUNT,
            sample_shape=(TOTAL_TOKENS, FEATURE_DIM),
        )
    )
    metadata = _reference_metadata(
        args,
        plan,
        feature_metadata=feature_metadata,
        vae_identity=vae_identity,
        prompt_index_hash=prompt_hash,
        feature_bank_hash=feature_hash,
    )
    print(
        f"[merge] building m={NUM_LANDMARKS} reference from "
        f"{tuple(features.shape)} FP16 matrices on {device}",
        flush=True,
    )
    reference = features.to(device=device, non_blocking=False)
    pack = build_video_nystrom_reference(
        reference,
        num_landmarks=NUM_LANDMARKS,
        metadata=metadata,
        bandwidth=args.bandwidth,
        median_max_samples=int(args.median_max_samples),
        kmeans_iterations=int(args.kmeans_iterations),
        ridge=float(args.ridge),
        chunk_size=int(args.chunk_size),
        seed=NYSTROM_SEED,
    )
    validate_reference_pack(pack)
    del reference
    torch.cuda.empty_cache()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        save_reference_pack(pack, temporary)
        staged = load_reference_pack(temporary, map_location="cpu")
        validate_reference_pack(staged)
        if output.exists() and not bool(args.overwrite):
            raise FileExistsError(
                f"output appeared during construction: {output}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    # This final-path reload is the gate for temporary-shard deletion.
    published = load_reference_pack(output, map_location="cpu")
    validate_reference_pack(published)
    _validate_published_pack_identity(pack, published)
    return {
        "output": str(output),
        "samples": int(features.shape[0]),
        "sample_shape": list(features.shape[1:]),
        "num_landmarks": int(pack["landmarks"].shape[0]),
        "sigma": float(torch.as_tensor(pack["sigma"]).item()),
        "content_hash": str(pack["content_hash"]),
        "prompt_index_sha256": prompt_hash,
        "feature_bank_fp16_sha256": feature_hash,
        "final_path_validated": True,
    }


def _validate_published_pack_identity(
    pack: Mapping[str, Any],
    published: Mapping[str, Any],
) -> None:
    """Require the final-path reload to preserve the built pack identity."""

    if published.get("content_hash") != pack.get("content_hash"):
        raise ValueError("published reference content hash changed after rename")
    # The builder canonicalizes user metadata and adds its reserved ``nystrom``
    # block. Compare against that exact pack payload, not the pre-build input.
    if published.get("metadata") != pack.get("metadata"):
        raise ValueError("published reference metadata changed after rename")


def _remove_validated_shards(
    shard_dir: Path,
    build_summary: Dict[str, Any],
) -> None:
    """Delete shards only with the explicit final-path validation receipt."""
    if build_summary.get("final_path_validated") is not True:
        raise RuntimeError(
            "refusing to delete shards before final reference validation")
    output = Path(str(build_summary.get("output", ""))).expanduser().resolve()
    if not output.is_file():
        raise FileNotFoundError(
            f"refusing to delete shards; published reference is missing: {output}")
    shutil.rmtree(shard_dir)


def _run(args: argparse.Namespace) -> None:
    _validate_args(args)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    plan = _plan(args, env_world_size)
    if bool(args.dry_run):
        if env_rank == 0:
            print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return

    device, rank, world_size = _setup_distributed(
        int(args.distributed_timeout_minutes))
    if world_size != int(plan["world_size"]):
        raise RuntimeError(
            f"world size changed after planning: {world_size} != "
            f"{plan['world_size']}")
    shard_dir = None
    success = False
    validated_summary = None
    try:
        vae_result = None
        if rank == 0:
            try:
                vae_result = {"identity": _vae_identity(), "error": None}
            except Exception as exc:
                vae_result = {
                    "identity": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        vae_result = _broadcast_rank0_object(vae_result, rank)
        if vae_result["error"] is not None:
            raise RuntimeError(
                "rank 0 could not resolve the Wan VAE identity: "
                f"{vae_result['error']}")
        vae_identity = vae_result["identity"]

        temp_result = None
        if rank == 0:
            try:
                created = Path(tempfile.mkdtemp(
                    prefix="dinov3_dense_nystrom_",
                    dir=Path(plan["temporary"]["parent"]),
                ))
                temp_result = {"path": str(created), "error": None}
            except Exception as exc:
                temp_result = {
                    "path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        temp_result = _broadcast_rank0_object(temp_result, rank)
        if temp_result["error"] is not None:
            raise RuntimeError(
                "rank 0 could not create shard directory: "
                f"{temp_result['error']}")
        shard_dir = Path(temp_result["path"])

        _extract_rank_shard(
            args,
            plan,
            device=device,
            rank=rank,
            world_size=world_size,
            shard_path=shard_dir / f"rank_{rank:05d}.pt",
            vae_identity=vae_identity,
        )
        _barrier()
        summary = None
        build_error = None
        if rank == 0:
            try:
                summary = _build_and_save_reference(
                    args, plan, shard_dir, device=device)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                build_error = f"{type(exc).__name__}: {exc}"
        outcome = _broadcast_rank0_object(
            {"summary": summary, "error": build_error}, rank)
        if outcome["error"] is not None:
            raise RuntimeError(
                f"rank 0 reference construction failed: {outcome['error']}")
        _barrier()
        validated_summary = outcome["summary"]
        success = True
        if rank == 0:
            print(json.dumps(
                {"status": "ok", **outcome["summary"]},
                indent=2,
                sort_keys=True,
            ), flush=True)
    finally:
        if rank == 0 and shard_dir is not None:
            if success:
                _remove_validated_shards(shard_dir, validated_summary)
                print(
                    f"[cleanup] removed validated temporary shards {shard_dir}",
                    flush=True,
                )
            else:
                print(
                    f"[cleanup] preserving failed-run shards at {shard_dir}",
                    flush=True,
                )
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[Iterable[str]] = None) -> None:
    _run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()

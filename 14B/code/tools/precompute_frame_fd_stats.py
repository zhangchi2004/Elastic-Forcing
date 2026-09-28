#!/usr/bin/env python3
"""Precompute frame-level SIM-FD teacher statistics.

The preferred input is an ODE teacher LMDB. For every selected LMDB row this
tool decodes only the final trajectory latent, samples the same deterministic
random frame positions for Inception, SigLIP and MAE, and accumulates reference
moments without retaining a feature bank. The legacy ``--latent_dir`` and
``--video_dir`` inputs remain supported.

Example (all rows, eight GPUs)::

    torchrun --nproc_per_node=8 tools/precompute_frame_fd_stats.py \
        --teacher-lmdb /path/to/teacher_ode_lmdb \
        --output-dir /path/to/frame_fd_stats \
        --max-videos 0 --num-frames 16 --seed 0

The output contains the three legacy ``*_frame_stats.npz`` files plus a strict
``manifest.json``. Each frame is one FD sample; this is intentionally
different from the video-level V-JEPA2 MMD sample axis.

Passing ``--train-indices-file`` selects only that explicit teacher-train
subset and switches to the FiDEL v4 workflow. The first stage writes
``manifest.pending.json`` and ``fidel_scores.template.json``. Expensive
real-floor and initial-student evaluation may be resumed independently; once
the measured template is complete, ``--finalize-fidel-manifest-only`` validates
all split/sampling/encoder/statistics identities and publishes ``manifest.json``.
Null or non-finite score placeholders are never accepted.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TOOL_FORMAT_VERSION = 3
FIDEL_TOOL_FORMAT_VERSION = 4
FIDEL_SCORE_FORMAT_VERSION = 1
DEFAULT_LATENT_DIR = Path("assets/latents")
DEFAULT_OUTPUT_DIR = Path("assets/frame_stats")
DEFAULT_VAE_MODEL = "Wan2.1-T2V-1.3B"

# (config key, load_repr_model name, target_size, pool_type)
DEFAULT_REPS: Sequence[Tuple[str, str, int, str]] = (
    ("inception", "inception", 299, "cls"),
    ("siglip", "vit_base_patch16_siglip_224.webli", 224, "avg"),
    ("mae", "vit_base_patch16_224.mae", 224, "avg"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frame-level Inception/SigLIP/MAE teacher FD statistics.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--teacher-lmdb",
        type=Path,
        default=None,
        help=(
            "ODE LMDB with [sample, trajectory, T, C, H, W] latents. The final "
            "trajectory latent is decoded for every selected row."
        ),
    )
    source.add_argument(
        "--latent-dir",
        "--latent_dir",
        dest="latent_dir",
        type=Path,
        default=None,
        help="Legacy directory of .pt files containing a final `latent` tensor.",
    )
    source.add_argument(
        "--video-dir",
        "--video_dir",
        dest="video_dir",
        type=Path,
        default=None,
        help="Legacy directory of .pt files containing a uint8 `video` tensor.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--max-videos",
        "--max_videos",
        dest="max_videos",
        type=int,
        default=0,
        help="Use the first N source rows/files; 0 traverses the actual source.",
    )
    parser.add_argument(
        "--num-frames",
        "--num_frames",
        dest="num_frames",
        type=int,
        default=16,
        help="Number of decoded pixel frames sampled without replacement/video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic row-keyed frame sampling.",
    )
    parser.add_argument(
        "--frame-sampling",
        choices=("random", "uniform"),
        default="random",
        help=(
            "random uses a deterministic independent permutation per source "
            "row; uniform preserves the old evenly-spaced behavior."
        ),
    )
    parser.add_argument(
        "--exclude-first-frame",
        action="store_true",
        help=(
            "Exclude decoded frame zero from the teacher sampling population. "
            "The training config must use the same setting."
        ),
    )
    parser.add_argument(
        "--fd-decode-frames",
        "--fd_decode_frames",
        dest="fd_decode_frames",
        type=int,
        default=0,
        help="Decode only the first N latent frames; 0 decodes the full latent.",
    )
    parser.add_argument(
        "--reps",
        default="",
        help=(
            "Comma-separated key:timm_name:target_size:pool specs. Empty uses "
            "Inception, SigLIP and MAE."
        ),
    )
    parser.add_argument(
        "--repr-batch-size",
        "--repr_batch_size",
        dest="repr_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--vae-model-name",
        default=DEFAULT_VAE_MODEL,
        help="Identity recorded for the fixed Wan VAE used by WanVAEWrapper.",
    )
    parser.add_argument(
        "--refresh-manifest-only",
        action="store_true",
        help=(
            "Upgrade/refresh manifest hashes and strict encoder identities "
            "from existing statistics without decoding the teacher set again."
        ),
    )
    parser.add_argument(
        "--train-indices-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON train-index list (or object with `train_indices` "
            "and `train_indices_sha256`). Only these teacher rows contribute "
            "to moments. This enables the FiDEL v4 1409/1537 split."
        ),
    )
    parser.add_argument(
        "--train-indices-sha256",
        default="",
        help=(
            "Expected SHA-256 of the canonical int64 train-index vector. "
            "Required either here or inside --train-indices-file."
        ),
    )
    parser.add_argument(
        "--expected-train-count",
        type=int,
        default=0,
        help=(
            "Require exactly this many explicit train indices; 0 accepts any "
            "non-empty strict subset. Use 1409 for the 1537-video FiDEL split."
        ),
    )
    parser.add_argument(
        "--fidel-scores-json",
        type=Path,
        default=None,
        help=(
            "Strict per-encoder real-floor/student-start score payload. If "
            "omitted with explicit train indices, moments and a resumable "
            "manifest.pending.json/template are written without inventing "
            "score values."
        ),
    )
    parser.add_argument(
        "--finalize-fidel-manifest-only",
        action="store_true",
        help=(
            "Validate manifest.pending.json plus --fidel-scores-json and "
            "atomically publish the complete v4 manifest without decoding or "
            "loading encoders again."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_videos < 0:
        raise ValueError("--max-videos must be non-negative")
    for name in ("num_frames", "repr_batch_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.fd_decode_frames < 0:
        raise ValueError("--fd-decode-frames must be non-negative")
    if args.expected_train_count < 0:
        raise ValueError("--expected-train-count must be non-negative")
    if args.train_indices_file is None and (
        args.train_indices_sha256
        or args.expected_train_count
        or args.fidel_scores_json is not None
        or args.finalize_fidel_manifest_only
    ):
        raise ValueError(
            "FiDEL split/score options require --train-indices-file")
    if args.refresh_manifest_only and args.finalize_fidel_manifest_only:
        raise ValueError(
            "--refresh-manifest-only and --finalize-fidel-manifest-only are "
            "mutually exclusive")
    if args.refresh_manifest_only and args.train_indices_file is not None:
        raise ValueError(
            "legacy --refresh-manifest-only cannot rewrite a FiDEL v4 split; "
            "use --finalize-fidel-manifest-only for a pending v4 manifest")
    if (
        args.finalize_fidel_manifest_only
        and args.fidel_scores_json is None
    ):
        raise ValueError(
            "--finalize-fidel-manifest-only requires --fidel-scores-json")
    if args.vae_model_name != DEFAULT_VAE_MODEL:
        raise ValueError(
            "WanVAEWrapper currently has fixed weights; --vae-model-name must "
            f"be {DEFAULT_VAE_MODEL!r}"
        )


def setup_distributed() -> Tuple[torch.device, int, int]:
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return (
            torch.device(f"cuda:{local_rank}"),
            dist.get_rank(),
            dist.get_world_size(),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, 1


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{name} must be a lowercase 64-character SHA-256 hex string")
    return value


def train_indices_sha256(indices: Sequence[int]) -> str:
    """Hash canonical sorted indices exactly as the V-JEPA2 split manifest."""
    values = torch.tensor(list(indices), dtype=torch.int64).contiguous()
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("train indices must be a non-empty 1-D sequence")
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def validate_train_indices(
    indices: Sequence[int],
    *,
    teacher_count: int,
    expected_sha256: Optional[str] = None,
    expected_count: int = 0,
) -> Dict[str, Any]:
    """Canonicalize an explicit teacher-train subset and bind its hash."""
    if (
        isinstance(indices, (str, bytes))
        or not isinstance(indices, Sequence)
        or not indices
    ):
        raise ValueError("train_indices must be a non-empty sequence")
    if any(type(value) is not int for value in indices):
        raise ValueError("train_indices must contain only integers")
    values = list(indices)
    if any(value < 0 or value >= int(teacher_count) for value in values):
        raise ValueError(
            "train_indices must lie in [0, teacher_count)")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(
            "train_indices must be strictly increasing and unique")
    if int(expected_count) > 0 and len(values) != int(expected_count):
        raise ValueError(
            f"train_indices count mismatch: {len(values)} != "
            f"{int(expected_count)}")
    calculated = train_indices_sha256(values)
    if expected_sha256:
        expected = _validate_sha256(
            expected_sha256, "train_indices_sha256")
        if not hmac.compare_digest(expected, calculated):
            raise ValueError(
                "train_indices_sha256 mismatch: "
                f"expected={expected}, calculated={calculated}")
    return {
        "teacher_count": int(teacher_count),
        "train_indices": values,
        "train_indices_sha256": calculated,
        "train_video_count": len(values),
    }


def load_train_indices(
    path: Path,
    *,
    teacher_count: int,
    expected_sha256: str = "",
    expected_count: int = 0,
) -> Dict[str, Any]:
    """Load either a raw JSON list or a split object with an embedded hash."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"train index file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    embedded_hash = ""
    embedded_teacher_count = None
    if isinstance(payload, list):
        indices = payload
    elif isinstance(payload, Mapping):
        indices = payload.get("train_indices")
        embedded_hash = str(payload.get("train_indices_sha256", ""))
        embedded_teacher_count = payload.get("teacher_count")
    else:
        raise ValueError(
            "train index file must contain a JSON list or mapping")
    if embedded_teacher_count is not None and (
        type(embedded_teacher_count) is not int
        or embedded_teacher_count != int(teacher_count)
    ):
        raise ValueError(
            "train index file teacher_count does not match source")
    requested_hash = str(expected_sha256 or embedded_hash)
    if not requested_hash:
        raise ValueError(
            "explicit train indices require train_indices_sha256 either in "
            "the JSON object or --train-indices-sha256")
    if embedded_hash and expected_sha256 and not hmac.compare_digest(
        _validate_sha256(embedded_hash, "embedded train_indices_sha256"),
        _validate_sha256(
            expected_sha256, "--train-indices-sha256"),
    ):
        raise ValueError(
            "embedded and command-line train_indices_sha256 disagree")
    result = validate_train_indices(
        indices,
        teacher_count=teacher_count,
        expected_sha256=requested_hash,
        expected_count=expected_count,
    )
    result["indices_file"] = str(resolved)
    result["indices_file_sha256"] = _file_sha256(resolved)
    return result


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path: Path, *, content_sha256: bool = False) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    identity: Dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_sha256:
        identity["sha256"] = _file_sha256(resolved)
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _inspect_lmdb(
    path: Path,
    *,
    content_sha256: bool = False,
) -> Dict[str, Any]:
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError(
            "python-lmdb is required for --teacher-lmdb; use the training "
            "Python/PYTHONPATH"
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
            int(v) for v in get_array_shape_from_lmdb(env, "latents"))
        prompts_shape = tuple(
            int(v) for v in get_array_shape_from_lmdb(env, "prompts"))
    finally:
        env.close()
    if len(latents_shape) != 6:
        raise ValueError(
            "teacher LMDB latents must be [samples, trajectory_steps, T, C, "
            f"H, W], got {latents_shape}"
        )
    if prompts_shape != (latents_shape[0],):
        raise ValueError(
            "teacher LMDB prompts/latents count mismatch: "
            f"{prompts_shape} vs {latents_shape}"
        )
    data_identity = _path_identity(
        resolved / "data.mdb", content_sha256=content_sha256)
    public = {
        "type": "teacher_lmdb",
        "path": str(resolved),
        "latents_shape": list(latents_shape),
        "prompts_shape": list(prompts_shape),
        "actual_video_count": int(latents_shape[0]),
        "trajectory_selection": "last",
        "data_mdb": data_identity,
    }
    public["identity_sha256"] = _canonical_sha256(public)
    return public


def _selected_count(actual_count: int, max_videos: int) -> int:
    count = (
        int(actual_count)
        if int(max_videos) == 0
        else min(int(actual_count), int(max_videos))
    )
    if count <= 0:
        raise ValueError("selected frame-FD teacher set is empty")
    return count


def _inspect_file_source(path: Path, source_type: str) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{source_type} directory not found: {resolved}")
    files = [Path(value).resolve() for value in sorted(
        glob.glob(str(resolved / "*.pt")))]
    if not files:
        raise FileNotFoundError(f"no .pt files under {resolved}")
    file_entries = [
        {
            "relative_path": str(file.relative_to(resolved)),
            "size_bytes": int(file.stat().st_size),
            "mtime_ns": int(file.stat().st_mtime_ns),
        }
        for file in files
    ]
    public = {
        "type": source_type,
        "path": str(resolved),
        "actual_video_count": len(files),
        "files_identity_sha256": _canonical_sha256(file_entries),
    }
    public["identity_sha256"] = _canonical_sha256(public)
    return {**public, "_files": files}


def inspect_source(
    args: argparse.Namespace,
    *,
    content_sha256: bool = False,
) -> Dict[str, Any]:
    """Return a source plan while preserving the legacy no-source default."""
    if args.teacher_lmdb is not None:
        source = _inspect_lmdb(
            args.teacher_lmdb, content_sha256=content_sha256)
    elif args.video_dir is not None:
        source = _inspect_file_source(args.video_dir, "video_dir")
    else:
        source = _inspect_file_source(
            args.latent_dir or DEFAULT_LATENT_DIR, "latent_dir")
    source["selected_video_count"] = _selected_count(
        source["actual_video_count"], args.max_videos)
    return source


def _row_random_seed(seed: int, row_index: int) -> int:
    payload = (
        int(seed).to_bytes(16, byteorder="little", signed=True)
        + int(row_index).to_bytes(16, byteorder="little", signed=False)
    )
    # torch.Generator.manual_seed accepts a signed-64-safe positive integer.
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little"
    ) & ((1 << 63) - 1)


def sample_frame_indices(
    num_frames: int,
    num_samples: int,
    device: Optional[torch.device] = None,
    *,
    seed: int = 0,
    row_index: int = 0,
    method: str = "random",
    exclude_first_frame: bool = False,
) -> torch.Tensor:
    """Select sorted frame positions deterministically for one source row."""
    if num_frames <= 0:
        raise ValueError("decoded video must contain at least one frame")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    start = 1 if bool(exclude_first_frame) and num_frames > 1 else 0
    available = num_frames - start
    if num_samples > available:
        raise ValueError(
            f"cannot sample {num_samples} unique frames from {available}")
    count = int(num_samples)
    if count == available:
        result = torch.arange(start, num_frames, dtype=torch.long)
    elif method == "uniform":
        result = torch.linspace(
            start, num_frames - 1, steps=count).round().long()
    elif method == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_row_random_seed(seed, row_index))
        result = (
            torch.randperm(available, generator=generator)[:count] + start)
        result = result.sort().values
    else:
        raise ValueError(f"unsupported frame sampling method {method!r}")
    return result.to(device=device) if device is not None else result


def _select_final_latent(trajectory: torch.Tensor, row_index: int) -> torch.Tensor:
    if trajectory.ndim != 5:
        raise ValueError(
            f"LMDB row {row_index} trajectory must be [steps,T,C,H,W], got "
            f"{tuple(trajectory.shape)}"
        )
    latent = trajectory[-1]
    if not torch.isfinite(latent).all():
        raise FloatingPointError(
            f"LMDB row {row_index} final latent contains non-finite values")
    return latent


def _tensor_entry_digest(
    row_index: int,
    tensor: torch.Tensor,
    prompt: str = "",
) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(int(row_index).to_bytes(8, byteorder="little", signed=False))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.view(torch.uint8).numpy())
    digest.update(str(prompt).encode("utf-8"))
    return digest.hexdigest()


def parse_rep_specs(value: str) -> List[Tuple[str, str, Optional[int], str]]:
    if not value:
        return list(DEFAULT_REPS)
    specs: List[Tuple[str, str, Optional[int], str]] = []
    for item in value.split(","):
        parts = item.split(":")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(f"invalid --reps item {item!r}")
        target = int(parts[2]) if len(parts) > 2 and parts[2] else None
        pool = parts[3] if len(parts) > 3 and parts[3] else "cls"
        if pool not in ("cls", "avg"):
            raise ValueError(f"invalid pool type in --reps item {item!r}")
        specs.append((parts[0], parts[1], target, pool))
    keys = [spec[0] for spec in specs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate representation keys: {keys}")
    return specs


def extract_frames(
    repr_model: torch.nn.Module,
    frames: torch.Tensor,
    pool_type: str,
    batch: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, frames.shape[0], batch):
        output = repr_model(frames[start:start + batch])
        if isinstance(output, tuple):
            primary, secondary = output
            output = (
                secondary
                if pool_type == "avg" and secondary is not None
                else primary
            )
        chunks.append(output.detach().float())
    return torch.cat(chunks, dim=0)


class Moments:
    """Incremental float64 mean/covariance accumulator."""

    def __init__(self, dim: int, device: torch.device):
        self.dim = int(dim)
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self.s = torch.zeros(dim, dtype=torch.float64, device=device)
        self.ss = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        values = feats.double()
        self.count += values.shape[0]
        self.s += values.sum(0)
        self.ss += values.t() @ values

    def all_reduce(self) -> None:
        if dist.is_available() and dist.is_initialized():
            for value in (self.count, self.s, self.ss):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, int]:
        count = int(self.count.item())
        if count < 2:
            raise ValueError(
                f"at least two frame features are required, got {count}")
        mean = self.s / self.count
        covariance = (
            self.ss - self.count * torch.outer(mean, mean)
        ) / (self.count - 1)
        covariance = 0.5 * (covariance + covariance.t())
        if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
            raise FloatingPointError("non-finite frame-FD reference moments")
        return mean.cpu().numpy(), covariance.cpu().numpy(), count


def _merge_record_metadata(
    records: Iterable[Mapping[str, Any]],
    selected_count: int,
    *,
    expected_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["row_index"]))
    indices = [int(item["row_index"]) for item in ordered]
    expected = (
        list(range(selected_count))
        if expected_indices is None
        else list(expected_indices)
    )
    if len(expected) != int(selected_count):
        raise ValueError(
            "expected_indices length must equal selected_count")
    if indices != expected:
        raise RuntimeError(
            "distributed source rows were missing or duplicated: "
            f"got {indices[:8]}... expected {expected[:8]}... for "
            f"selected_count={selected_count}"
        )
    row_digest = hashlib.sha256()
    frame_digest = hashlib.sha256()
    decoded_counts = []
    for item in ordered:
        row_digest.update(bytes.fromhex(str(item["input_sha256"])))
        frame_payload = {
            "row_index": int(item["row_index"]),
            "decoded_frame_count": int(item["decoded_frame_count"]),
            "frame_indices": [int(v) for v in item["frame_indices"]],
        }
        frame_digest.update(json.dumps(
            frame_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        decoded_counts.append(int(item["decoded_frame_count"]))
    return {
        "selected_rows_sha256": row_digest.hexdigest(),
        "frame_indices_sha256": frame_digest.hexdigest(),
        "decoded_frame_counts": sorted(set(decoded_counts)),
    }


def _gather_records(
    local_records: List[Dict[str, Any]],
    world_size: int,
) -> List[Dict[str, Any]]:
    if not (dist.is_available() and dist.is_initialized()):
        return local_records
    gathered: List[Optional[List[Dict[str, Any]]]] = [
        None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_records)
    return [
        record
        for rank_records in gathered
        for record in (rank_records or [])
    ]


def _atomic_npz(path: Path, *, mu: np.ndarray, sigma: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, mu=mu, sigma=sigma)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _vae_identity(
    model_name: str,
    *,
    compute_dtype: str,
) -> Dict[str, Any]:
    from utils.paths import model_path

    weights = model_path(model_name) / "Wan2.1_VAE.pth"
    identity = {
        "wrapper": "utils.wan_wrapper.WanVAEWrapper",
        "model_name": model_name,
        "compute_dtype": str(compute_dtype),
        # Hashing this fixed file is cheap relative to full teacher decode and
        # makes a copied reference pack independently auditable.
        "weights": _path_identity(weights, content_sha256=True),
        "decode_output_range": "[-1,1]",
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _encoder_inference_identity() -> Dict[str, Any]:
    # ``model.fd._resize_normalize`` and ``extract_frames`` explicitly produce
    # FP32 inputs/outputs, and this tool intentionally does not autocast the
    # representation forward. The hybrid training config is required to match.
    return {
        "autocast_enabled": False,
        "compute_dtype": "float32",
        "output_dtype": "float32",
    }


def build_manifest(
    *,
    source: Mapping[str, Any],
    vae: Mapping[str, Any],
    rep_specs: Sequence[Tuple[str, str, Optional[int], str]],
    rep_results: Sequence[Mapping[str, Any]],
    sampling: Mapping[str, Any],
    encoder_inference: Mapping[str, Any],
    selected_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    if len(rep_specs) != len(rep_results):
        raise ValueError(
            "representation spec/result count mismatch: "
            f"{len(rep_specs)} != {len(rep_results)}")
    public_source = {
        key: value for key, value in source.items() if not key.startswith("_")}
    representations = []
    for spec, result in zip(rep_specs, rep_results):
        key, model_name, target_size, pool_type = spec
        representations.append({
            "key": key,
            "model_name": model_name,
            "pool_type": pool_type,
            "target_size": target_size,
            "feature_dim": int(result["feature_dim"]),
            "frame_sample_count": int(result["frame_sample_count"]),
            "stats_file": str(result["stats_file"]),
            "stats_sha256": str(result["stats_sha256"]),
            "encoder_identity": dict(result["encoder_identity"]),
        })
    manifest: Dict[str, Any] = {
        "format": "self_forcing_frame_fd_stats",
        "format_version": TOOL_FORMAT_VERSION,
        "feature_aggregation": "frame",
        "source": {
            **public_source,
            **dict(selected_metadata),
        },
        "vae": dict(vae),
        "sampling": dict(sampling),
        "encoder_inference": dict(encoder_inference),
        "representations": representations,
    }
    manifest["reference_bundle_sha256"] = _canonical_sha256({
        "source_identity_sha256": manifest["source"]["identity_sha256"],
        "selected_rows_sha256": manifest["source"]["selected_rows_sha256"],
        "frame_indices_sha256": manifest["source"]["frame_indices_sha256"],
        "vae_identity_sha256": manifest["vae"]["identity_sha256"],
        "sampling": manifest["sampling"],
        "encoder_inference": manifest["encoder_inference"],
        "stats": [
            {
                "key": item["key"],
                "sha256": item["stats_sha256"],
                "encoder_state_sha256": item[
                    "encoder_identity"]["state_sha256"],
            }
            for item in representations
        ],
    })
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def _fidel_split_public(train_split: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "teacher_count",
        "train_indices",
        "train_indices_sha256",
        "train_video_count",
    )
    if any(key not in train_split for key in required):
        raise ValueError("FiDEL train split metadata is incomplete")
    validated = validate_train_indices(
        train_split["train_indices"],
        teacher_count=int(train_split["teacher_count"]),
        expected_sha256=str(train_split["train_indices_sha256"]),
        expected_count=int(train_split["train_video_count"]),
    )
    public = dict(validated)
    if "indices_file" in train_split:
        public["indices_file"] = str(train_split["indices_file"])
        public["indices_file_sha256"] = _validate_sha256(
            train_split.get("indices_file_sha256"),
            "indices_file_sha256",
        )
    return public


def build_fidel_pending_manifest(
    *,
    source: Mapping[str, Any],
    vae: Mapping[str, Any],
    rep_specs: Sequence[Tuple[str, str, Optional[int], str]],
    rep_results: Sequence[Mapping[str, Any]],
    sampling: Mapping[str, Any],
    encoder_inference: Mapping[str, Any],
    selected_metadata: Mapping[str, Any],
    train_split: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a resumable v4 manifest that is deliberately not trainable yet."""
    split = _fidel_split_public(train_split)
    if int(source["actual_video_count"]) != split["teacher_count"]:
        raise ValueError(
            "FiDEL teacher_count does not match source actual_video_count")
    if int(source["selected_video_count"]) != split["train_video_count"]:
        raise ValueError(
            "FiDEL selected_video_count does not match train index count")
    if (
        sampling.get("method") != "random"
        or int(sampling.get("frames_per_video", -1)) != 16
        or sampling.get("without_replacement") is not True
        or sampling.get("shared_across_representations") is not True
        or isinstance(sampling.get("seed"), bool)
        or not isinstance(sampling.get("seed"), int)
    ):
        raise ValueError(
            "FiDEL v4 requires random shared sampling of exactly 16 unique "
            "frames with an explicit integer seed")

    legacy = build_manifest(
        source=source,
        vae=vae,
        rep_specs=rep_specs,
        rep_results=rep_results,
        sampling=sampling,
        encoder_inference=encoder_inference,
        selected_metadata=selected_metadata,
    )
    legacy.pop("manifest_sha256")
    legacy.pop("reference_bundle_sha256")
    legacy["format_version"] = FIDEL_TOOL_FORMAT_VERSION
    legacy["completion_status"] = "awaiting_fidel_scores"
    legacy["source"].update(split)
    expected_frames = split["train_video_count"] * 16
    for entry in legacy["representations"]:
        if int(entry["frame_sample_count"]) != expected_frames:
            raise ValueError(
                f"{entry['key']} frame_sample_count must equal "
                f"train_video_count*16 ({expected_frames})")
    legacy["pending_manifest_sha256"] = _canonical_sha256(legacy)
    return legacy


def build_fidel_score_template(
    pending_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create an auditable template for the expensive floor/start stage."""
    validate_fidel_pending_manifest(pending_manifest)
    source = pending_manifest["source"]
    return {
        "format": "self_forcing_frame_fd_fidel_scores",
        "format_version": FIDEL_SCORE_FORMAT_VERSION,
        "teacher_count": int(source["teacher_count"]),
        "train_indices_sha256": str(source["train_indices_sha256"]),
        "sampling": dict(pending_manifest["sampling"]),
        "representations": [
            {
                "key": entry["key"],
                "stats_sha256": entry["stats_sha256"],
                "encoder_state_sha256": entry[
                    "encoder_identity"]["state_sha256"],
                # These must be populated by real validation/initial-student
                # inference. The finalizer rejects null and fabricated
                # non-finite placeholders.
                "real_floor": None,
                "student_start_score": None,
                "student_start_std": None,
            }
            for entry in pending_manifest["representations"]
        ],
    }


def validate_fidel_pending_manifest(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("pending FiDEL manifest must be a mapping")
    saved_hash = _validate_sha256(
        manifest.get("pending_manifest_sha256"),
        "pending_manifest_sha256",
    )
    unhashed = dict(manifest)
    unhashed.pop("pending_manifest_sha256", None)
    calculated = _canonical_sha256(unhashed)
    if not hmac.compare_digest(saved_hash, calculated):
        raise ValueError("pending_manifest_sha256 mismatch")
    if (
        manifest.get("format") != "self_forcing_frame_fd_stats"
        or manifest.get("format_version") != FIDEL_TOOL_FORMAT_VERSION
        or manifest.get("feature_aggregation") != "frame"
        or manifest.get("completion_status") != "awaiting_fidel_scores"
    ):
        raise ValueError("invalid pending FiDEL frame-FD manifest format")
    source = manifest.get("source", {})
    split = validate_train_indices(
        source.get("train_indices"),
        teacher_count=int(source.get("teacher_count", -1)),
        expected_sha256=source.get("train_indices_sha256"),
        expected_count=int(source.get("train_video_count", -1)),
    )
    if (
        split["teacher_count"] != int(source.get("actual_video_count", -1))
        or split["train_video_count"]
        != int(source.get("selected_video_count", -1))
    ):
        raise ValueError("pending FiDEL teacher/train counts are inconsistent")
    sampling = manifest.get("sampling", {})
    if (
        sampling.get("method") != "random"
        or sampling.get("frames_per_video") != 16
        or sampling.get("without_replacement") is not True
        or sampling.get("shared_across_representations") is not True
        or isinstance(sampling.get("seed"), bool)
        or not isinstance(sampling.get("seed"), int)
    ):
        raise ValueError("pending FiDEL frame sampling is invalid")
    entries = manifest.get("representations")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "pending FiDEL manifest needs representation entries")
    for entry in entries:
        _validate_sha256(entry.get("stats_sha256"), "stats_sha256")
        identity = entry.get("encoder_identity", {})
        _validate_sha256(
            identity.get("state_sha256"), "encoder state_sha256")
        if int(entry.get("frame_sample_count", -1)) != (
            split["train_video_count"] * 16
        ):
            raise ValueError(
                "representation frame count does not match train split")
    return dict(manifest)


def _validate_fidel_score_payload(
    payload: Mapping[str, Any],
    pending_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_fidel_pending_manifest(pending_manifest)
    if not isinstance(payload, Mapping):
        raise TypeError("FiDEL score payload must be a mapping")
    source = pending_manifest["source"]
    if (
        payload.get("format") != "self_forcing_frame_fd_fidel_scores"
        or payload.get("format_version") != FIDEL_SCORE_FORMAT_VERSION
        or payload.get("teacher_count") != source["teacher_count"]
        or payload.get("train_indices_sha256")
        != source["train_indices_sha256"]
        or payload.get("sampling") != pending_manifest["sampling"]
    ):
        raise ValueError(
            "FiDEL score payload does not match teacher split/sampling")
    score_entries = payload.get("representations")
    reference_entries = pending_manifest["representations"]
    if (
        not isinstance(score_entries, list)
        or len(score_entries) != len(reference_entries)
    ):
        raise ValueError(
            "FiDEL score representation count does not match moments")
    validated_entries = []
    exact_keys = {
        "key",
        "stats_sha256",
        "encoder_state_sha256",
        "real_floor",
        "student_start_score",
        "student_start_std",
    }
    for score, reference in zip(score_entries, reference_entries):
        if not isinstance(score, Mapping) or set(score) != exact_keys:
            raise ValueError(
                "each FiDEL score entry must contain exactly "
                f"{sorted(exact_keys)}")
        if (
            score["key"] != reference["key"]
            or score["stats_sha256"] != reference["stats_sha256"]
            or score["encoder_state_sha256"]
            != reference["encoder_identity"]["state_sha256"]
        ):
            raise ValueError(
                f"FiDEL score identity mismatch for {reference['key']}")
        scalars = {}
        for name in (
            "real_floor",
            "student_start_score",
            "student_start_std",
        ):
            value = score[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"FiDEL {reference['key']} {name} must be finite")
            scalars[name] = float(value)
        if scalars["student_start_std"] < 0.0:
            raise ValueError("FiDEL student_start_std must be non-negative")
        if (
            scalars["student_start_score"] - scalars["real_floor"]
            <= 1.0e-8
        ):
            raise ValueError(
                f"FiDEL {reference['key']} student-start score must exceed "
                "its real floor by more than 1e-8")
        validated_entries.append({
            "key": reference["key"],
            "stats_sha256": reference["stats_sha256"],
            "encoder_state_sha256": reference[
                "encoder_identity"]["state_sha256"],
            **scalars,
        })
    validated = {
        "format": "self_forcing_frame_fd_fidel_scores",
        "format_version": FIDEL_SCORE_FORMAT_VERSION,
        "teacher_count": int(source["teacher_count"]),
        "train_indices_sha256": str(source["train_indices_sha256"]),
        "sampling": dict(pending_manifest["sampling"]),
        "representations": validated_entries,
    }
    provenance = payload.get("score_provenance")
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ValueError("FiDEL score_provenance must be a mapping")
        # Canonical JSON round-tripping rejects tensors, NaN and other
        # process-local values while preserving tool-specific audit metadata.
        try:
            encoded = json.dumps(
                provenance,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            validated["score_provenance"] = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "FiDEL score_provenance must be finite JSON data") from exc
    return validated


def _fidel_bundle_payload(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_identity_sha256": manifest["source"]["identity_sha256"],
        "teacher_count": manifest["source"]["teacher_count"],
        "train_indices_sha256": manifest[
            "source"]["train_indices_sha256"],
        "selected_rows_sha256": manifest[
            "source"]["selected_rows_sha256"],
        "frame_indices_sha256": manifest[
            "source"]["frame_indices_sha256"],
        "vae_identity_sha256": manifest["vae"]["identity_sha256"],
        "sampling": manifest["sampling"],
        "encoder_inference": manifest["encoder_inference"],
        "fidel_score_payload_sha256": manifest[
            "fidel_score_payload_sha256"],
        "stats": [
            {
                "key": item["key"],
                "sha256": item["stats_sha256"],
                "encoder_state_sha256": item[
                    "encoder_identity"]["state_sha256"],
                "real_floor": item["real_floor"],
                "student_start_score": item["student_start_score"],
                "student_start_std": item["student_start_std"],
            }
            for item in manifest["representations"]
        ],
    }


def finalize_fidel_manifest(
    pending_manifest: Mapping[str, Any],
    score_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach measured floor/start scores and publish a strict v4 manifest."""
    pending = validate_fidel_pending_manifest(pending_manifest)
    scores = _validate_fidel_score_payload(score_payload, pending)
    completed = dict(pending)
    completed.pop("pending_manifest_sha256")
    completed.pop("completion_status")
    completed["fidel_score_payload_sha256"] = _canonical_sha256(scores)
    if "score_provenance" in scores:
        completed["score_provenance"] = scores["score_provenance"]
    by_key = {
        entry["key"]: entry for entry in scores["representations"]}
    completed["representations"] = [
        {
            **dict(entry),
            "real_floor": by_key[entry["key"]]["real_floor"],
            "student_start_score": by_key[
                entry["key"]]["student_start_score"],
            "student_start_std": by_key[
                entry["key"]]["student_start_std"],
        }
        for entry in pending["representations"]
    ]
    completed["reference_bundle_sha256"] = _canonical_sha256(
        _fidel_bundle_payload(completed))
    completed["manifest_sha256"] = _canonical_sha256(completed)
    validate_fidel_manifest_v4(completed)
    return completed


def validate_fidel_manifest_v4(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Pure-CPU structural/hash validation for a complete FiDEL reference."""
    if not isinstance(manifest, Mapping):
        raise TypeError("FiDEL manifest must be a mapping")
    saved_hash = _validate_sha256(
        manifest.get("manifest_sha256"), "manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if not hmac.compare_digest(saved_hash, _canonical_sha256(unhashed)):
        raise ValueError("FiDEL manifest_sha256 mismatch")
    if (
        manifest.get("format") != "self_forcing_frame_fd_stats"
        or manifest.get("format_version") != FIDEL_TOOL_FORMAT_VERSION
        or manifest.get("feature_aggregation") != "frame"
        or "completion_status" in manifest
    ):
        raise ValueError("invalid complete FiDEL frame-FD manifest format")
    # Reconstruct a pending form so the common split/sampling/representation
    # identity checks remain exactly aligned.
    pending = {
        key: value
        for key, value in manifest.items()
        if key not in (
            "manifest_sha256",
            "reference_bundle_sha256",
            "fidel_score_payload_sha256",
        )
    }
    pending["completion_status"] = "awaiting_fidel_scores"
    pending["representations"] = [
        {
            key: value for key, value in entry.items()
            if key not in (
                "real_floor",
                "student_start_score",
                "student_start_std",
            )
        }
        for entry in manifest.get("representations", [])
    ]
    pending["pending_manifest_sha256"] = _canonical_sha256(pending)
    validate_fidel_pending_manifest(pending)
    scores = {
        "format": "self_forcing_frame_fd_fidel_scores",
        "format_version": FIDEL_SCORE_FORMAT_VERSION,
        "teacher_count": manifest["source"]["teacher_count"],
        "train_indices_sha256": manifest[
            "source"]["train_indices_sha256"],
        "sampling": manifest["sampling"],
        "representations": [
            {
                "key": entry["key"],
                "stats_sha256": entry["stats_sha256"],
                "encoder_state_sha256": entry[
                    "encoder_identity"]["state_sha256"],
                "real_floor": entry.get("real_floor"),
                "student_start_score": entry.get(
                    "student_start_score"),
                "student_start_std": entry.get("student_start_std"),
            }
            for entry in manifest["representations"]
        ],
    }
    if "score_provenance" in manifest:
        scores["score_provenance"] = manifest["score_provenance"]
    validated_scores = _validate_fidel_score_payload(scores, pending)
    if manifest.get("fidel_score_payload_sha256") != _canonical_sha256(
        validated_scores
    ):
        raise ValueError("fidel_score_payload_sha256 mismatch")
    expected_bundle = _canonical_sha256(_fidel_bundle_payload(manifest))
    if manifest.get("reference_bundle_sha256") != expected_bundle:
        raise ValueError("FiDEL reference_bundle_sha256 mismatch")
    return dict(manifest)


def _load_source_item(
    source: Mapping[str, Any],
    row_index: int,
    dataset: Optional[Any],
) -> Tuple[torch.Tensor, str]:
    source_type = source["type"]
    if source_type == "teacher_lmdb":
        if dataset is None:
            raise RuntimeError("teacher LMDB dataset was not initialized")
        item = dataset[row_index]
        return _select_final_latent(
            item["ode_latent"], row_index), str(item["prompts"])

    file_path = source["_files"][row_index]
    item = torch.load(file_path, map_location="cpu")
    if source_type == "video_dir":
        video = item["video"]
        if video.ndim != 4:
            raise ValueError(
                f"{file_path} video must be [T,C,H,W], got {tuple(video.shape)}")
        return video, str(item.get("prompt", ""))

    latent = item["latent"]
    # Accept both old final-latent shards and ODE trajectory shards.
    if latent.ndim == 5:
        latent = _select_final_latent(latent, row_index)
    if latent.ndim != 4:
        raise ValueError(
            f"{file_path} latent must be [T,C,H,W], got {tuple(latent.shape)}")
    if not torch.isfinite(latent).all():
        raise FloatingPointError(f"{file_path} contains non-finite latent")
    return latent, str(item.get("prompt", ""))


def _run_on_rank_zero(
    operation: str,
    function: Callable[[], Any],
    *,
    rank: int,
) -> Any:
    """Run one operation on rank zero and broadcast success or failure."""
    result: Optional[Dict[str, Any]] = None
    if rank == 0:
        try:
            result = {
                "ok": True,
                "value": function(),
            }
        except BaseException as exc:
            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if dist.is_available() and dist.is_initialized():
        payload = [result]
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
    elif rank != 0:
        raise RuntimeError(
            f"{operation} requires an initialized process group on rank {rank}")

    if not isinstance(result, dict) or not result.get("ok"):
        error = result.get("error", "unknown error") if result else "no result"
        raise RuntimeError(f"{operation} failed on rank zero: {error}")
    return result.get("value")


def _inspect_source_on_rank_zero(
    args: argparse.Namespace,
    *,
    rank: int,
) -> Dict[str, Any]:
    """Hash a large LMDB once, then give every rank the identical identity."""
    return _run_on_rank_zero(
        "source inspection",
        lambda: inspect_source(args, content_sha256=True),
        rank=rank,
    )


def _sampling_identity(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "method": args.frame_sampling,
        "seed": int(args.seed),
        "seed_derivation": "sha256(base_seed_signed128,row_index_uint128)",
        "frames_per_video": int(args.num_frames),
        "without_replacement": True,
        "shared_across_representations": True,
        "exclude_first_frame": bool(args.exclude_first_frame),
        "fd_decode_latent_frames": int(args.fd_decode_frames),
    }


def _refresh_existing_manifest(
    *,
    args: argparse.Namespace,
    source: Mapping[str, Any],
    rep_specs: Sequence[Tuple[str, str, Optional[int], str]],
    device: torch.device,
) -> None:
    """Upgrade strict identities while reusing already-computed moments."""
    from model.fd import (
        load_repr_model,
        representation_runtime_identity,
    )

    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"--refresh-manifest-only needs {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        old = json.load(handle)
    if (
        old.get("format") != "self_forcing_frame_fd_stats"
        or old.get("feature_aggregation") != "frame"
    ):
        raise ValueError("existing frame FD manifest has an invalid format")
    old_source = old.get("source", {})
    if (
        os.path.realpath(str(old_source.get("path", "")))
        != os.path.realpath(str(source.get("path", "")))
        or int(old_source.get("actual_video_count", -1))
        != int(source.get("actual_video_count", -2))
        or int(old_source.get("selected_video_count", -1))
        != int(source.get("selected_video_count", -2))
        or old_source.get("trajectory_selection")
        != source.get("trajectory_selection")
    ):
        raise ValueError(
            "existing statistics do not belong to the selected teacher source")

    sampling = _sampling_identity(args)
    old_sampling = old.get("sampling", {})
    old_exclude_first = bool(
        old_sampling.get("exclude_first_frame", False))
    for key in (
        "method",
        "seed",
        "frames_per_video",
        "without_replacement",
        "shared_across_representations",
        "fd_decode_latent_frames",
    ):
        if old_sampling.get(key) != sampling.get(key):
            raise ValueError(
                f"existing statistics sampling mismatch for {key!r}")
    if old_exclude_first != sampling["exclude_first_frame"]:
        raise ValueError(
            "existing statistics exclude_first_frame does not match")

    old_entries = old.get("representations", [])
    if len(old_entries) != len(rep_specs):
        raise ValueError(
            "existing statistics representation count does not match")
    rep_results = []
    for entry, spec in zip(old_entries, rep_specs):
        key, name, target_size, pool_type = spec
        if (
            entry.get("key") != key
            or entry.get("model_name") != name
            or entry.get("target_size") != target_size
            or entry.get("pool_type") != pool_type
        ):
            raise ValueError(
                f"existing representation {entry.get('key')!r} does not "
                f"match requested {key!r}")
        stats_path = output_dir / str(entry["stats_file"])
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"existing frame FD statistics are missing: {stats_path}")
        stats_sha256 = _file_sha256(stats_path)
        if stats_sha256 != entry.get("stats_sha256"):
            raise ValueError(
                f"existing frame FD statistics hash changed: {stats_path}")
        with np.load(stats_path) as stats:
            feature_dim = int(np.asarray(stats["mu"]).size)
            sigma_shape = tuple(np.asarray(stats["sigma"]).shape)
        if sigma_shape != (feature_dim, feature_dim):
            raise ValueError(
                f"invalid covariance shape in {stats_path}: {sigma_shape}")
        model = load_repr_model(
            name, device=device, target_size=target_size)
        if int(model.feat_dim) != feature_dim:
            raise ValueError(
                f"{name} feature dim changed: {model.feat_dim} != "
                f"{feature_dim}")
        rep_results.append({
            "stats_file": stats_path.name,
            "stats_sha256": stats_sha256,
            "feature_dim": feature_dim,
            "frame_sample_count": int(entry["frame_sample_count"]),
            "encoder_identity": representation_runtime_identity(model),
        })
        del model

    selected_metadata = {
        key: old_source[key]
        for key in (
            "selected_rows_sha256",
            "frame_indices_sha256",
            "decoded_frame_counts",
        )
    }
    manifest = build_manifest(
        source=source,
        vae=_vae_identity(
            args.vae_model_name, compute_dtype=args.dtype),
        rep_specs=rep_specs,
        rep_results=rep_results,
        sampling=sampling,
        encoder_inference=_encoder_inference_identity(),
        selected_metadata=selected_metadata,
    )
    _atomic_json(manifest_path, manifest)
    print(
        f"[stats] refreshed strict manifest {manifest_path}; "
        f"bundle_sha256={manifest['reference_bundle_sha256']}",
        flush=True,
    )


def _load_json_mapping(path: Path, name: str) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _finalize_existing_fidel_manifest(
    *,
    args: argparse.Namespace,
    source: Mapping[str, Any],
    train_split: Mapping[str, Any],
) -> None:
    """Complete a pending v4 manifest without re-running expensive encoders."""
    output_dir = args.output_dir.expanduser().resolve()
    pending_path = output_dir / "manifest.pending.json"
    pending = _load_json_mapping(
        pending_path, "pending FiDEL manifest")
    validate_fidel_pending_manifest(pending)
    pending_source = pending["source"]
    if (
        pending_source.get("identity_sha256")
        != source.get("identity_sha256")
        or int(pending_source.get("teacher_count", -1))
        != int(train_split["teacher_count"])
        or pending_source.get("train_indices_sha256")
        != train_split["train_indices_sha256"]
        or pending_source.get("train_indices")
        != train_split["train_indices"]
        or pending.get("sampling") != _sampling_identity(args)
    ):
        raise ValueError(
            "pending FiDEL manifest does not match active teacher split or "
            "frame sampling")
    for entry in pending["representations"]:
        stats_path = output_dir / str(entry["stats_file"])
        if (
            not stats_path.is_file()
            or _file_sha256(stats_path) != entry["stats_sha256"]
        ):
            raise ValueError(
                f"pending statistics changed for {entry['key']}")
    score_payload = _load_json_mapping(
        args.fidel_scores_json, "FiDEL score payload")
    completed = finalize_fidel_manifest(pending, score_payload)
    _atomic_json(output_dir / "manifest.json", completed)
    print(
        f"[stats] finalized FiDEL v4 manifest "
        f"{output_dir / 'manifest.json'}; "
        f"bundle_sha256={completed['reference_bundle_sha256']}",
        flush=True,
    )


def _run_after_setup(
    args: argparse.Namespace,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    source = _inspect_source_on_rank_zero(args, rank=rank)
    train_split = None
    if args.train_indices_file is not None:
        if source["type"] != "teacher_lmdb":
            raise ValueError(
                "explicit FiDEL train indices require --teacher-lmdb")
        train_split = load_train_indices(
            args.train_indices_file,
            teacher_count=int(source["actual_video_count"]),
            expected_sha256=args.train_indices_sha256,
            expected_count=args.expected_train_count,
        )
        source["selected_video_count"] = int(
            train_split["train_video_count"])
    selected_count = int(source["selected_video_count"])
    rep_specs = parse_rep_specs(args.reps)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    from model.fd import load_repr_model, representation_runtime_identity
    from utils.wan_wrapper import WanVAEWrapper

    if args.finalize_fidel_manifest_only:
        _run_on_rank_zero(
            "FiDEL manifest finalization",
            lambda: _finalize_existing_fidel_manifest(
                args=args,
                source=source,
                train_split=train_split,
            ),
            rank=rank,
        )
        return

    if args.refresh_manifest_only:
        _run_on_rank_zero(
            "manifest refresh",
            lambda: _refresh_existing_manifest(
                args=args,
                source=source,
                rep_specs=rep_specs,
                device=device,
            ),
            rank=rank,
        )
        return

    dataset = None
    if source["type"] == "teacher_lmdb":
        from utils.dataset import ODERegressionLMDBDataset

        dataset = ODERegressionLMDBDataset(
            source["path"],
            max_pair=int(source["actual_video_count"]),
        )
        if len(dataset) != int(source["actual_video_count"]):
            raise RuntimeError(
                "teacher LMDB length changed after inspection: "
                f"{len(dataset)} != {source['actual_video_count']}"
            )

    vae = None
    if source["type"] != "video_dir":
        vae = (
            WanVAEWrapper()
            .to(device=device, dtype=dtype)
            .eval()
            .requires_grad_(False)
        )

    reps = []
    for key, name, target_size, pool in rep_specs:
        model = load_repr_model(name, device=device, target_size=target_size)
        reps.append((
            key,
            model,
            pool,
            Moments(int(model.feat_dim), device),
        ))
        if rank == 0:
            print(
                f"[stats] rep {key}: {name} dim={model.feat_dim} "
                f"target={target_size} pool={pool}",
                flush=True,
            )

    selected_indices = (
        list(range(selected_count))
        if train_split is None
        else list(train_split["train_indices"])
    )
    assigned = selected_indices[rank::world_size]
    local_records: List[Dict[str, Any]] = []
    started = time.time()
    iterator = tqdm(assigned, disable=rank != 0, desc="decode+extract")
    for row_index in iterator:
        source_tensor, prompt = _load_source_item(
            source, row_index, dataset)
        input_sha256 = _tensor_entry_digest(
            row_index, source_tensor, prompt)

        if source["type"] == "video_dir":
            pixels = (
                source_tensor.float() / 255.0
            ).unsqueeze(0).to(device)
        else:
            latent = source_tensor
            if args.fd_decode_frames > 0:
                latent = latent[:args.fd_decode_frames]
            latent = latent.to(device=device, dtype=dtype).unsqueeze(0)
            pixels = vae.decode_to_pixel(latent)
            pixels = ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)

        decoded_frame_count = int(pixels.shape[1])
        indices_cpu = sample_frame_indices(
            decoded_frame_count,
            args.num_frames,
            seed=args.seed,
            row_index=row_index,
            method=args.frame_sampling,
            exclude_first_frame=args.exclude_first_frame,
        )
        indices = indices_cpu.to(device=device)
        frames = pixels.index_select(1, indices).reshape(
            -1, *pixels.shape[2:])
        for _key, model, pool, moments in reps:
            features = extract_frames(
                model, frames, pool, args.repr_batch_size)
            moments.update(features)

        local_records.append({
            "row_index": row_index,
            "input_sha256": input_sha256,
            "decoded_frame_count": decoded_frame_count,
            "frame_indices": indices_cpu.tolist(),
        })
        del source_tensor, pixels, frames
        if source["type"] != "video_dir":
            del latent

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    all_records = _gather_records(local_records, world_size)
    selected_metadata = _merge_record_metadata(
        all_records,
        selected_count,
        expected_indices=selected_indices,
    )

    output_dir = args.output_dir.expanduser().resolve()
    _run_on_rank_zero(
        "output directory creation",
        lambda: output_dir.mkdir(parents=True, exist_ok=True),
        rank=rank,
    )

    rep_results: List[Dict[str, Any]] = []
    for key, model, _pool, moments in reps:
        moments.all_reduce()

        def _publish_representation_stats() -> Dict[str, Any]:
            mu, sigma, frame_count = moments.finalize()
            output_path = output_dir / f"{key}_frame_stats.npz"
            _atomic_npz(output_path, mu=mu, sigma=sigma)
            result = {
                "stats_file": output_path.name,
                "stats_sha256": _file_sha256(output_path),
                "feature_dim": int(mu.shape[0]),
                "frame_sample_count": frame_count,
                "encoder_identity": representation_runtime_identity(model),
            }
            print(
                f"[stats] wrote {output_path}: {frame_count} frame samples, "
                f"dim={mu.shape[0]} sha256={result['stats_sha256']}",
                flush=True,
            )
            return result

        result = _run_on_rank_zero(
            f"{key} statistics publication",
            _publish_representation_stats,
            rank=rank,
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                f"{key} statistics publication returned an invalid result")
        rep_results.append(result)

    def _publish_output_manifests() -> None:
        sampling = _sampling_identity(args)
        manifest_kwargs = {
            "source": source,
            "vae": _vae_identity(
                args.vae_model_name, compute_dtype=args.dtype),
            "rep_specs": rep_specs,
            "rep_results": rep_results,
            "sampling": sampling,
            "encoder_inference": _encoder_inference_identity(),
            "selected_metadata": selected_metadata,
        }
        if train_split is None:
            manifest = build_manifest(**manifest_kwargs)
            _atomic_json(output_dir / "manifest.json", manifest)
        else:
            pending = build_fidel_pending_manifest(
                **manifest_kwargs,
                train_split=train_split,
            )
            _atomic_json(output_dir / "manifest.pending.json", pending)
            template = build_fidel_score_template(pending)
            _atomic_json(
                output_dir / "fidel_scores.template.json", template)
            if args.fidel_scores_json is not None:
                score_payload = _load_json_mapping(
                    args.fidel_scores_json, "FiDEL score payload")
                manifest = finalize_fidel_manifest(
                    pending, score_payload)
                _atomic_json(output_dir / "manifest.json", manifest)
            else:
                manifest = pending
                print(
                    "[stats] FiDEL moments complete; measured real-floor and "
                    "student-start scores are still required. Run "
                    "tools/precompute_frame_fd_fidel_scores.py (the template "
                    f"is {output_dir / 'fidel_scores.template.json'}) and "
                    "then rerun with --finalize-fidel-manifest-only. No "
                    "deployable manifest.json was written.",
                    flush=True,
                )
        # Preserve the old lightweight metadata filename for external scripts.
        legacy_meta = {
            item["key"]: {
                "path": str(output_dir / item["stats_file"]),
                "dim": item["feature_dim"],
                "num_frames": item["frame_sample_count"],
                "sha256": item["stats_sha256"],
            }
            for item in manifest["representations"]
        }
        _atomic_json(output_dir / "meta.json", legacy_meta)
        elapsed = time.time() - started
        output_manifest_name = (
            "manifest.json"
            if train_split is None or args.fidel_scores_json is not None
            else "manifest.pending.json"
        )
        output_bundle_hash = manifest.get(
            "reference_bundle_sha256",
            manifest.get("pending_manifest_sha256"),
        )
        print(
            f"[stats] done in {elapsed:.1f}s from {selected_count}/"
            f"{source['actual_video_count']} videos x {args.num_frames} frames; "
            f"manifest={output_dir / output_manifest_name} "
            f"bundle_sha256={output_bundle_hash}",
            flush=True,
        )

    _run_on_rank_zero(
        "manifest and metadata publication",
        _publish_output_manifests,
        rank=rank,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    device, rank, world_size = setup_distributed()
    try:
        _run_after_setup(
            args,
            device=device,
            rank=rank,
            world_size=world_size,
        )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

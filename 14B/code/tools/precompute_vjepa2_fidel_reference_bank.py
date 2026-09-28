#!/usr/bin/env python3
"""Build a random-mask V-JEPA2 FiDEL Nyström reference bank.

The expensive work is explicitly staged and resumable:

1. extract every teacher video as a full ``[8192,1024]`` FP16 dense matrix;
2. generate 512 fixed ``ode_init`` student videos and extract the same matrix;
3. build one train-reference pack per random token mask and measure its real
   validation floor plus four fixed 128-video student-start scores;
4. publish a strict, hashed bank manifest only after every measured slot is
   present and valid.

No floor or student baseline is synthesized from metadata.  A slot is
publishable only after both scores have actually been evaluated.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses.video_nystrom_reference_bank import (  # noqa: E402
    build_reference_bank_manifest,
    load_reference_bank_manifest,
    save_reference_bank_manifest,
)
from video_fd.encoders.vjepa2_dense_metadata import (  # noqa: E402
    canonical_vjepa2_dense_feature_metadata,
    token_indices_sha256,
)
from tools.precompute_vjepa2_dense_nystrom_reference import (  # noqa: E402
    _broadcast_rank0_object,
    _inspect_lmdb,
    _read_model_geometry,
    _setup_distributed,
)


TOOL_FORMAT_VERSION = 2
STAGE_STATE_VERSION = 2
RANK_MARKER_VERSION = 2
STUDENT_LATENT_FORMAT_VERSION = 1
STUDENT_LATENT_BUNDLE_MANIFEST_VERSION = 1
TEACHER_SPLIT_MANIFEST_VERSION = 1

EXPECTED_TEACHERS = 1537
TRAIN_TEACHERS = 1409
VAL_TEACHERS = 128
RAW_TOKENS = 8192
SAMPLED_TOKENS = 128
FEATURE_DIM = 1024
MASK_SLOTS = 100
LANDMARKS = 256
STUDENT_SAMPLES = 512
STUDENT_BATCHES = 4
STUDENT_BATCH_SIZE = 128
REFERENCE_WORLD_SIZE = 8
GLOBAL_MICROBATCHES = STUDENT_SAMPLES // REFERENCE_WORLD_SIZE
SUPPORTED_REFERENCE_WORLD_SIZES = (4, REFERENCE_WORLD_SIZE)

SPLIT_SEED = "sf-fidel-split-v1"
MASK_SEED = "sf-vjepa2-token-mask-v1"
BASELINE_PROMPT_SEED = "sf-fidel-student-prompts-v1"
BASELINE_NOISE_SEED = "sf-fidel-student-noise-v1"
BASELINE_EXIT_SEED = "sf-fidel-student-exit-v1"
PROMPT_NORMALIZATION = "unicode_nfkc_strip_collapse_whitespace"

GIB = 1024 ** 3
MIB = 1024 ** 2
_STATE_KEYS = {
    "format_version",
    "request",
    "request_hash",
    "teacher_bank",
    "student_bank",
    "slots",
    "content_hash",
}
_BANK_RECORD_KEYS = {
    "kind",
    "path",
    "shape",
    "dtype",
    "sha256",
    "auxiliary",
}
_STUDENT_BANK_AUXILIARY_KEYS = {
    "student_init_latent_dir",
    "student_init_latent_count",
    "student_init_latent_bundle_sha256",
    "student_init_latent_bundle_manifest",
    "student_init_latent_bundle_manifest_sha256",
    "student_init_latent_bundle_manifest_content_hash",
}
_STUDENT_LATENT_KEYS = {
    "format_version",
    "index",
    "latent",
    "latent_shape",
    "latent_dtype",
    "prompt_index",
    "prompt_sha256",
    "noise_seed",
    "global_microbatch_index",
    "exit_seed",
    "exit_indices",
    "exit_index",
    "sim_steps",
    "denoised_timestep_from",
    "denoised_timestep_to",
    "checkpoint_sha256",
    "request_hash",
    "sampling_path",
    "content_hash",
}
_STUDENT_LATENT_BUNDLE_MANIFEST_KEYS = {
    "format_version",
    "artifact_format_version",
    "request_hash",
    "generator_checkpoint_identity",
    "text_conditioning_identity",
    "student_count",
    "world_size",
    "global_microbatches",
    "latent_shape",
    "latent_dtype",
    "ordered_prompt_indices_sha256",
    "noise_seeds_sha256",
    "exit_indices_sha256",
    "artifacts",
    "bundle_sha256",
    "content_hash",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute a strict random-token V-JEPA2 FiDEL "
            "Nyström reference bank."
        )
    )
    parser.add_argument("--teacher-lmdb", type=Path)
    parser.add_argument("--model-id", type=str)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--student-config", type=Path)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument(
        "--sampled-tokens",
        type=int,
        default=SAMPLED_TOKENS,
        help="Random V-JEPA2 tokens selected per reference-bank slot.",
    )
    parser.add_argument(
        "--mask-slots",
        type=int,
        default=MASK_SLOTS,
        help="Number of independently sampled token-mask slots.",
    )
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--resize-short-side", type=int, default=292)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--teacher-batch-size", type=int, default=1)
    parser.add_argument("--student-batch-size", type=int, default=1)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--median-max-samples", type=int, default=1409)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--kernel-chunk-size", type=int, default=32)
    parser.add_argument("--build-chunk-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--distributed-timeout-minutes", type=int, default=1440)
    parser.add_argument(
        "--stage",
        choices=("all", "teacher", "student", "slots"),
        default="all",
        help="Run one resumable stage or the complete pipeline.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect identities, split, masks and capacity without writes.",
    )
    modes.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an already published manifest and all configured packs.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    sampled_tokens = int(args.sampled_tokens)
    mask_slots = int(args.mask_slots)
    if not 1 < sampled_tokens <= RAW_TOKENS:
        raise ValueError(
            f"--sampled-tokens must be in [2, {RAW_TOKENS}]")
    if mask_slots <= 0:
        raise ValueError("--mask-slots must be positive")
    if mask_slots > math.comb(RAW_TOKENS, sampled_tokens):
        raise ValueError(
            "--mask-slots exceeds the number of distinct token masks")
    if args.validate_only:
        return
    required = {
        "--teacher-lmdb": args.teacher_lmdb,
        "--model-id": args.model_id,
        "--vae-checkpoint": args.vae_checkpoint,
        "--student-config": args.student_config,
        "--student-checkpoint": args.student_checkpoint,
        "--scratch-dir": args.scratch_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"the selected mode requires arguments: {', '.join(missing)}")
    positive = {
        "teacher_batch_size": args.teacher_batch_size,
        "student_batch_size": args.student_batch_size,
        "sampled_tokens": sampled_tokens,
        "mask_slots": mask_slots,
        "num_frames": args.num_frames,
        "resize_short_side": args.resize_short_side,
        "crop_size": args.crop_size,
        "kmeans_iterations": args.kmeans_iterations,
        "median_max_samples": args.median_max_samples,
        "kernel_chunk_size": args.kernel_chunk_size,
        "build_chunk_size": args.build_chunk_size,
        "distributed_timeout_minutes": args.distributed_timeout_minutes,
    }
    invalid = {
        name: value for name, value in positive.items()
        if int(value) <= 0
    }
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.layer != 15:
        raise ValueError("the FiDEL bank contract requires --layer=15")
    if args.num_frames != 64:
        raise ValueError("the FiDEL bank contract requires --num-frames=64")
    if args.resize_short_side != 292 or args.crop_size != 256:
        raise ValueError(
            "the FiDEL bank contract requires preprocessing 292 -> 256")
    if args.student_batch_size != 1:
        raise ValueError(
            "--student-batch-size must be 1 so every sample has an isolated "
            "fixed RNG stream")
    if args.teacher_batch_size != 1:
        raise ValueError(
            "--teacher-batch-size must be 1 so resumed rows use one fixed "
            "VAE/V-JEPA2 numerical batch shape")
    if not math.isfinite(float(args.ridge)) or args.ridge <= 0:
        raise ValueError("--ridge must be finite and positive")
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative")


def canonicalize_prompt(prompt: str) -> str:
    """Canonical prompt used only for deterministic split/ranking hashes.

    Student conditioning deliberately receives the exact raw LMDB string,
    matching training; ``WanTextEncoder`` then applies its own tokenizer
    cleaning.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return " ".join(unicodedata.normalize("NFKC", prompt).split())


def _seeded_prompt_digest(seed: str, index: int, prompt: str) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(str(seed).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(canonicalize_prompt(prompt).encode("utf-8"))
    return hasher.digest()


def deterministic_prompt_split(
    prompts: Sequence[str],
    *,
    val_count: int,
    seed: str = SPLIT_SEED,
) -> Tuple[list[int], list[int]]:
    """Hash-sort normalized prompts into disjoint train and validation sets."""
    if not isinstance(seed, str) or not seed:
        raise ValueError("split seed must be non-empty")
    count = len(prompts)
    if not 0 < int(val_count) < count:
        raise ValueError("val_count must be in (0, len(prompts))")
    ranked = sorted(
        range(count),
        key=lambda index: (
            _seeded_prompt_digest(seed, index, prompts[index]),
            index,
        ),
    )
    val_indices = sorted(ranked[:int(val_count)])
    train_indices = sorted(ranked[int(val_count):])
    return train_indices, val_indices


def deterministic_token_masks(
    *,
    raw_tokens: int = RAW_TOKENS,
    num_tokens: int = SAMPLED_TOKENS,
    count: int = MASK_SLOTS,
    seed: str = MASK_SEED,
) -> list[list[int]]:
    """Return unique, sorted, per-slot random-without-replacement masks."""
    raw_tokens = int(raw_tokens)
    num_tokens = int(num_tokens)
    count = int(count)
    if raw_tokens <= 0 or not 1 < num_tokens <= raw_tokens:
        raise ValueError("invalid raw/sample token counts")
    if count <= 0 or not isinstance(seed, str) or not seed:
        raise ValueError("mask count and seed must be positive/non-empty")
    masks = []
    seen = set()
    for slot in range(count):
        attempt = 0
        while True:
            digest = hashlib.sha256(
                f"{seed}\0{slot}\0{attempt}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(digest, "big"))
            mask = tuple(sorted(rng.sample(range(raw_tokens), num_tokens)))
            if mask not in seen:
                seen.add(mask)
                masks.append(list(mask))
                break
            attempt += 1
            if attempt > 1024:
                raise RuntimeError("could not generate a unique token mask")
    return masks


def deterministic_baseline_prompt_indices(
    prompts: Sequence[str],
    train_indices: Sequence[int],
    *,
    count: int = STUDENT_SAMPLES,
    seed: str = BASELINE_PROMPT_SEED,
) -> list[int]:
    if int(count) <= 0 or int(count) > len(train_indices):
        raise ValueError(
            "baseline count must fit in the teacher-train split")
    ranked = sorted(
        (int(index) for index in train_indices),
        key=lambda index: (
            _seeded_prompt_digest(seed, index, prompts[index]),
            index,
        ),
    )
    return ranked[:int(count)]


def deterministic_noise_seeds(
    count: int = STUDENT_SAMPLES,
    *,
    seed: str = BASELINE_NOISE_SEED,
) -> list[int]:
    if int(count) <= 0 or not isinstance(seed, str) or not seed:
        raise ValueError("noise count and seed must be positive/non-empty")
    output = []
    for index in range(int(count)):
        digest = hashlib.sha256(
            f"{seed}\0{index}".encode("utf-8")).digest()
        output.append(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
    if len(set(output)) != len(output):
        raise RuntimeError("deterministic noise seed collision")
    return output


def deterministic_exit_seeds(
    count: int = GLOBAL_MICROBATCHES,
    *,
    seed: str = BASELINE_EXIT_SEED,
) -> list[int]:
    return deterministic_noise_seeds(count, seed=seed)


def deterministic_exit_indices(
    seed: int,
    *,
    num_blocks: int = 7,
    num_steps: int = 4,
) -> list[int]:
    if int(num_blocks) <= 0 or int(num_steps) <= 0:
        raise ValueError("exit descriptor geometry must be positive")
    rng = random.Random(int(seed))
    return [rng.randrange(int(num_steps)) for _ in range(int(num_blocks))]


def _global_microbatches_for_world_size(world_size: int) -> int:
    world_size = int(world_size)
    if world_size not in SUPPORTED_REFERENCE_WORLD_SIZES:
        supported = ", ".join(
            str(value) for value in SUPPORTED_REFERENCE_WORLD_SIZES)
        raise ValueError(
            f"FiDEL student baseline world_size must be one of "
            f"({supported}), got {world_size}")
    if STUDENT_SAMPLES % world_size:
        raise ValueError(
            "student sample count must be divisible by world_size")
    return STUDENT_SAMPLES // world_size


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite canonical JSON data") from exc


def _sha256_json(value: Any, domain: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(domain)
    hasher.update(_canonical_json(value))
    return hasher.hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 16 * MIB) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _indices_sha256(indices: Sequence[int]) -> str:
    return token_indices_sha256(torch.tensor(indices, dtype=torch.int64))


def _prompt_sha256(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("conditioning prompt must be a string")
    hasher = hashlib.sha256()
    hasher.update(b"sf-fidel-raw-conditioning-prompt-v1\0")
    hasher.update(prompt.encode("utf-8"))
    return hasher.hexdigest()


def _prompt_list_sha256(prompts: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for index, prompt in enumerate(prompts):
        hasher.update(int(index).to_bytes(8, "little", signed=False))
        hasher.update(_seeded_prompt_digest(
            "sf-canonical-prompt-list-v1", index, prompt))
    return hasher.hexdigest()


def _raw_prompt_list_sha256(prompts: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"sf-fidel-raw-prompt-list-v1\0")
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            raise TypeError("raw prompt list must contain strings")
        encoded = prompt.encode("utf-8")
        hasher.update(int(index).to_bytes(8, "little", signed=False))
        hasher.update(len(encoded).to_bytes(8, "little", signed=False))
        hasher.update(encoded)
    return hasher.hexdigest()


def _student_latent_dir(scratch: Path) -> Path:
    return scratch / "student_init_latents"


def _student_latent_path(scratch: Path, index: int) -> Path:
    if not 0 <= int(index) < STUDENT_SAMPLES:
        raise IndexError("student latent index is out of range")
    return _student_latent_dir(scratch) / f"{int(index):06d}.pt"


def _student_latent_tensor_sha256(latent: torch.Tensor) -> str:
    if (
        not isinstance(latent, torch.Tensor)
        or latent.device.type != "cpu"
        or latent.dtype != torch.float16
        or not latent.is_contiguous()
    ):
        raise ValueError(
            "student latent hashing requires contiguous CPU float16")
    hasher = hashlib.sha256()
    hasher.update(b"sf-fidel-student-latent-tensor-v1\0")
    hasher.update(_canonical_json({
        "shape": [int(value) for value in latent.shape],
        "dtype": "float16",
    }))
    hasher.update(latent.view(torch.uint8).numpy().tobytes(order="C"))
    return hasher.hexdigest()


def compute_student_latent_content_hash(
    artifact: Mapping[str, Any],
) -> str:
    if not isinstance(artifact, Mapping):
        raise ValueError("student latent artifact must be a mapping")
    expected_keys = _STUDENT_LATENT_KEYS - {"content_hash"}
    actual_keys = set(artifact) - {"content_hash"}
    if actual_keys != expected_keys:
        raise ValueError(
            "student latent artifact hash keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}")
    metadata = {
        key: artifact[key]
        for key in _STUDENT_LATENT_KEYS
        if key not in {"latent", "content_hash"}
    }
    hasher = hashlib.sha256()
    hasher.update(b"sf-fidel-student-latent-artifact-v1\0")
    hasher.update(_canonical_json(metadata))
    hasher.update(
        _student_latent_tensor_sha256(artifact["latent"]).encode("ascii"))
    return hasher.hexdigest()


def _expected_student_latent_metadata(
    request_wrapper: Mapping[str, Any],
    prompts: Sequence[str],
    index: int,
) -> Dict[str, Any]:
    request = request_wrapper["request"]
    request_hash = _validate_hex_sha256(
        request_wrapper["request_hash"], "request_hash")
    index = int(index)
    if not 0 <= index < STUDENT_SAMPLES:
        raise IndexError("student baseline index is out of range")
    student = request["student"]
    world_size = int(student["world_size"])
    expected_global_microbatches = _global_microbatches_for_world_size(
        world_size)
    if int(student["global_microbatches"]) != expected_global_microbatches:
        raise ValueError(
            "request student global_microbatches/world_size mismatch")
    prompt_index = int(student["prompt_indices"][index])
    global_microbatch_index = index // world_size
    exit_seed = int(student["exit_seeds"][global_microbatch_index])
    exit_indices = [
        int(value)
        for value in student["exit_indices"][global_microbatch_index]
    ]
    if exit_indices != deterministic_exit_indices(exit_seed):
        raise ValueError("request exit descriptor does not match its seed")
    return {
        "format_version": STUDENT_LATENT_FORMAT_VERSION,
        "index": index,
        "latent_shape": [21, 16, 60, 104],
        "latent_dtype": "float16",
        "prompt_index": prompt_index,
        "prompt_sha256": _prompt_sha256(prompts[prompt_index]),
        "noise_seed": int(student["noise_seeds"][index]),
        "global_microbatch_index": global_microbatch_index,
        "exit_seed": exit_seed,
        "exit_indices": exit_indices,
        "exit_index": int(exit_indices[0]),
        "sim_steps": int(exit_indices[0]) + 1,
        "checkpoint_sha256": request["student"]["checkpoint"]["sha256"],
        "request_hash": request_hash,
        "sampling_path": (
            "SelfForcingTrainingPipeline.inference_with_trajectory"),
    }


def validate_student_latent_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(artifact, Mapping) or set(artifact) != _STUDENT_LATENT_KEYS:
        raise ValueError("student latent artifact keys mismatch")
    for key, expected in expected_metadata.items():
        if artifact.get(key) != expected:
            raise ValueError(
                f"student latent artifact {key} does not match request")
    for key in ("denoised_timestep_from", "denoised_timestep_to"):
        value = artifact[key]
        if value is not None and type(value) is not int:
            raise ValueError(f"student latent artifact {key} must be int/null")
    if artifact["sim_steps"] != artifact["exit_index"] + 1:
        raise ValueError("student latent sim_steps/exit_index mismatch")
    latent = artifact["latent"]
    if (
        not isinstance(latent, torch.Tensor)
        or latent.device.type != "cpu"
        or latent.dtype != torch.float16
        or tuple(latent.shape) != (21, 16, 60, 104)
        or not bool(torch.isfinite(latent).all().item())
    ):
        raise ValueError(
            "student latent must be finite CPU float16 [21,16,60,104]")
    latent = latent.contiguous()
    normalized = dict(artifact)
    normalized["latent"] = latent
    saved_hash = _validate_hex_sha256(
        normalized["content_hash"], "student latent content_hash")
    calculated_hash = compute_student_latent_content_hash(normalized)
    if not hmac.compare_digest(saved_hash, calculated_hash):
        raise ValueError("student latent artifact content_hash mismatch")
    return normalized


def _load_student_latent_artifact(
    path: Path,
    *,
    expected_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        artifact = torch.load(
            path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load student latent artifact {path}") from exc
    return validate_student_latent_artifact(
        artifact, expected_metadata=expected_metadata)


def _atomic_student_latent_save(
    artifact: Mapping[str, Any],
    path: Path,
    *,
    expected_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = validate_student_latent_artifact(
        artifact, expected_metadata=expected_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(normalized, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _load_student_latent_artifact(
            temporary, expected_metadata=expected_metadata)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return normalized


def _student_latent_bundle_sha256(
    content_hashes: Sequence[Tuple[int, str]],
) -> str:
    normalized = []
    for index, content_hash in content_hashes:
        normalized.append([
            int(index),
            _validate_hex_sha256(
                content_hash, "student latent content_hash"),
        ])
    if normalized != sorted(normalized):
        raise ValueError("student latent bundle entries must be index-sorted")
    if len({index for index, _ in normalized}) != len(normalized):
        raise ValueError("student latent bundle indices must be unique")
    return _sha256_json(
        normalized, b"sf-fidel-student-latent-bundle-v1\0")


def _student_latent_bundle_entries(
    scratch: Path,
    request_wrapper: Mapping[str, Any],
    prompts: Sequence[str],
    indices: Sequence[int],
) -> list[Dict[str, Any]]:
    ordered = sorted(int(index) for index in indices)
    if ordered != list(indices):
        raise ValueError(
            "student latent bundle indices must be sorted")
    entries = []
    for index in ordered:
        expected_metadata = _expected_student_latent_metadata(
            request_wrapper, prompts, index)
        artifact = _load_student_latent_artifact(
            _student_latent_path(scratch, index),
            expected_metadata=expected_metadata,
        )
        entries.append({
            "index": index,
            "path": f"{index:06d}.pt",
            "content_hash": artifact["content_hash"],
            "prompt_index": artifact["prompt_index"],
            "prompt_sha256": artifact["prompt_sha256"],
            "noise_seed": artifact["noise_seed"],
            "global_microbatch_index":
                artifact["global_microbatch_index"],
            "exit_seed": artifact["exit_seed"],
            "exit_indices": artifact["exit_indices"],
            "exit_index": artifact["exit_index"],
            "sim_steps": artifact["sim_steps"],
        })
    return entries


def _validate_student_latent_bundle(
    scratch: Path,
    request_wrapper: Mapping[str, Any],
    prompts: Sequence[str],
    indices: Sequence[int],
    *,
    expected_hash: Optional[str] = None,
) -> str:
    entries = _student_latent_bundle_entries(
        scratch, request_wrapper, prompts, indices)
    actual_hash = _student_latent_bundle_sha256([
        (entry["index"], entry["content_hash"])
        for entry in entries
    ])
    if expected_hash is not None and not hmac.compare_digest(
        actual_hash,
        _validate_hex_sha256(
            expected_hash, "student latent bundle sha256"),
    ):
        raise ValueError("student latent bundle SHA mismatch")
    return actual_hash


def _student_latent_bundle_manifest_path(scratch: Path) -> Path:
    return _student_latent_dir(scratch) / "manifest.json"


def compute_student_latent_bundle_manifest_hash(
    manifest: Mapping[str, Any],
) -> str:
    if not isinstance(manifest, Mapping):
        raise ValueError("student latent bundle manifest must be a mapping")
    expected_keys = (
        _STUDENT_LATENT_BUNDLE_MANIFEST_KEYS - {"content_hash"})
    actual_keys = set(manifest) - {"content_hash"}
    if actual_keys != expected_keys:
        raise ValueError(
            "student latent bundle manifest hash keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}")
    payload = {
        key: manifest[key]
        for key in _STUDENT_LATENT_BUNDLE_MANIFEST_KEYS
        if key != "content_hash"
    }
    return _sha256_json(
        payload, b"sf-fidel-student-latent-bundle-manifest-v1\0")


def _build_student_latent_bundle_manifest(
    scratch: Path,
    request_wrapper: Mapping[str, Any],
    prompts: Sequence[str],
) -> Dict[str, Any]:
    student = request_wrapper["request"]["student"]
    entries = _student_latent_bundle_entries(
        scratch,
        request_wrapper,
        prompts,
        list(range(STUDENT_SAMPLES)),
    )
    bundle_hash = _student_latent_bundle_sha256([
        (entry["index"], entry["content_hash"])
        for entry in entries
    ])
    manifest = {
        "format_version": STUDENT_LATENT_BUNDLE_MANIFEST_VERSION,
        "artifact_format_version": STUDENT_LATENT_FORMAT_VERSION,
        "request_hash": request_wrapper["request_hash"],
        "generator_checkpoint_identity": student["checkpoint"],
        "text_conditioning_identity": student["text_conditioning"],
        "student_count": STUDENT_SAMPLES,
        "world_size": int(student["world_size"]),
        "global_microbatches": int(student["global_microbatches"]),
        "latent_shape": [21, 16, 60, 104],
        "latent_dtype": "float16",
        "ordered_prompt_indices_sha256":
            student["ordered_prompt_indices_sha256"],
        "noise_seeds_sha256": student["noise_seeds_sha256"],
        "exit_indices_sha256": student["exit_indices_sha256"],
        "artifacts": entries,
        "bundle_sha256": bundle_hash,
        "content_hash": "0" * 64,
    }
    manifest["content_hash"] = compute_student_latent_bundle_manifest_hash(
        manifest)
    return validate_student_latent_bundle_manifest(
        manifest,
        scratch=scratch,
        request_wrapper=request_wrapper,
        prompts=prompts,
    )


def validate_student_latent_bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    scratch: Path,
    request_wrapper: Mapping[str, Any],
    prompts: Sequence[str],
) -> Dict[str, Any]:
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != _STUDENT_LATENT_BUNDLE_MANIFEST_KEYS
    ):
        raise ValueError("student latent bundle manifest keys mismatch")
    student = request_wrapper["request"]["student"]
    expected_top = {
        "format_version": STUDENT_LATENT_BUNDLE_MANIFEST_VERSION,
        "artifact_format_version": STUDENT_LATENT_FORMAT_VERSION,
        "request_hash": request_wrapper["request_hash"],
        "generator_checkpoint_identity": student["checkpoint"],
        "text_conditioning_identity": student["text_conditioning"],
        "student_count": STUDENT_SAMPLES,
        "world_size": int(student["world_size"]),
        "global_microbatches": int(student["global_microbatches"]),
        "latent_shape": [21, 16, 60, 104],
        "latent_dtype": "float16",
        "ordered_prompt_indices_sha256":
            student["ordered_prompt_indices_sha256"],
        "noise_seeds_sha256": student["noise_seeds_sha256"],
        "exit_indices_sha256": student["exit_indices_sha256"],
    }
    expected_global_microbatches = _global_microbatches_for_world_size(
        expected_top["world_size"])
    if expected_top["global_microbatches"] != expected_global_microbatches:
        raise ValueError(
            "request student global_microbatches/world_size mismatch")
    for key, expected in expected_top.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"student latent bundle manifest {key} mismatch")
    actual_entries = _student_latent_bundle_entries(
        scratch,
        request_wrapper,
        prompts,
        list(range(STUDENT_SAMPLES)),
    )
    if manifest["artifacts"] != actual_entries:
        raise ValueError(
            "student latent bundle manifest artifacts mismatch")
    actual_bundle_hash = _student_latent_bundle_sha256([
        (entry["index"], entry["content_hash"])
        for entry in actual_entries
    ])
    saved_bundle_hash = _validate_hex_sha256(
        manifest["bundle_sha256"], "student latent bundle sha256")
    if not hmac.compare_digest(actual_bundle_hash, saved_bundle_hash):
        raise ValueError(
            "student latent bundle manifest bundle_sha256 mismatch")
    saved_content_hash = _validate_hex_sha256(
        manifest["content_hash"],
        "student latent bundle manifest content_hash",
    )
    actual_content_hash = compute_student_latent_bundle_manifest_hash(
        manifest)
    if not hmac.compare_digest(saved_content_hash, actual_content_hash):
        raise ValueError(
            "student latent bundle manifest content_hash mismatch")
    return json.loads(_canonical_json({
        key: value
        for key, value in manifest.items()
        if key != "artifacts"
    })) | {"artifacts": actual_entries}


def _load_student_latent_bundle_manifest(
    path: Path,
    *,
    scratch: Path,
    request_wrapper: Mapping[str, Any],
    prompts: Sequence[str],
) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load student latent bundle manifest {path}") from exc
    return validate_student_latent_bundle_manifest(
        manifest,
        scratch=scratch,
        request_wrapper=request_wrapper,
        prompts=prompts,
    )


def _model_identity(model_id: str) -> Dict[str, Any]:
    root = Path(model_id).expanduser().resolve()
    geometry = _read_model_geometry(str(root))
    candidates = [root / "config.json"]
    candidates.extend(sorted(
        path for pattern in ("*.safetensors", "*.bin", "*.pth")
        for path in root.glob(pattern)
    ))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < 2:
        raise FileNotFoundError(
            f"no V-JEPA2 weight file found under {root}")
    files = []
    aggregate = hashlib.sha256()
    aggregate.update(b"vjepa2-model-identity-v1\0")
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(root).as_posix()
        digest = _sha256_file(path)
        record = {
            "path": relative,
            "size_bytes": int(path.stat().st_size),
            "sha256": digest,
        }
        files.append(record)
        aggregate.update(_canonical_json(record))
    return {
        "path": str(root),
        "geometry": geometry,
        "files": files,
        "aggregate_sha256": aggregate.hexdigest(),
    }


def _file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(resolved),
    }


def _directory_identity(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = []
    aggregate = hashlib.sha256()
    aggregate.update(b"sf-directory-identity-v1\0")
    for candidate in sorted(
        value for value in resolved.rglob("*") if value.is_file()
    ):
        relative = candidate.relative_to(resolved).as_posix()
        record = {
            "path": relative,
            "size_bytes": int(candidate.stat().st_size),
            "sha256": _sha256_file(candidate),
        }
        files.append(record)
        aggregate.update(_canonical_json(record))
    if not files:
        raise ValueError(f"identity directory contains no files: {resolved}")
    return {
        "path": str(resolved),
        "files": files,
        "aggregate_sha256": aggregate.hexdigest(),
    }


def _text_conditioning_identity() -> Dict[str, Any]:
    from utils.paths import model_path

    root = model_path("Wan2.1-T2V-1.3B").resolve()
    return {
        "encoder_weights": _file_identity(
            root / "models_t5_umt5-xxl-enc-bf16.pth"),
        "tokenizer": _directory_identity(
            root / "google" / "umt5-xxl"),
    }


def _load_prompts(path: Path, expected_count: int) -> list[str]:
    from utils.dataset import ODERegressionLMDBDataset

    dataset = ODERegressionLMDBDataset(
        str(path.expanduser().resolve()),
        max_pair=int(expected_count),
        load_latents=False,
    )
    if len(dataset) != int(expected_count):
        raise ValueError(
            f"teacher LMDB has {len(dataset)} prompt rows, "
            f"expected {expected_count}")
    prompts = [dataset[index]["prompts"] for index in range(len(dataset))]
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise TypeError("teacher LMDB prompts must all be strings")
    del dataset
    return prompts


def _effective_student_config(args: argparse.Namespace) -> Tuple[Any, Dict[str, Any]]:
    from omegaconf import OmegaConf
    from utils.config import load_config

    default_path = REPO_ROOT / "configs/default_config.yaml"
    config = OmegaConf.merge(
        load_config(default_path),
        load_config(args.student_config),
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise ValueError("merged student config must be a mapping")
    contract = {
        "image_or_video_shape": [
            int(value) for value in config.image_or_video_shape
        ],
        "denoising_step_list": [
            int(value) for value in config.denoising_step_list
        ],
        "warp_denoising_step": bool(config.warp_denoising_step),
        "num_frame_per_block": int(config.num_frame_per_block),
        "num_training_frames": int(config.num_training_frames),
        "independent_first_frame": bool(config.independent_first_frame),
        "same_step_across_blocks": bool(config.same_step_across_blocks),
        "last_step_only": bool(config.last_step_only),
        "context_noise": int(config.context_noise),
        "causal": bool(config.causal),
        "backward_simulation": bool(
            getattr(config, "backward_simulation", True)),
        "i2v": bool(config.i2v),
        "mixed_precision": bool(config.mixed_precision),
        "teacher_latent_count": int(config.teacher_latent_count),
        "batch_size": int(config.batch_size),
        "fresh_video_local_samples": int(
            config.fresh_video_local_samples),
        "total_batch_size": int(config.total_batch_size),
    }
    if contract != {
        "image_or_video_shape": [1, 21, 16, 60, 104],
        "denoising_step_list": [1000, 750, 500, 250],
        "warp_denoising_step": True,
        "num_frame_per_block": 3,
        "num_training_frames": 21,
        "independent_first_frame": False,
        "same_step_across_blocks": True,
        "last_step_only": False,
        "context_noise": 0,
        "causal": True,
        "backward_simulation": True,
        "i2v": False,
        "mixed_precision": True,
        "teacher_latent_count": EXPECTED_TEACHERS,
        "batch_size": 1,
        "fresh_video_local_samples": 16,
        "total_batch_size": STUDENT_BATCH_SIZE,
    }:
        raise ValueError(
            "student config does not match the canonical FiDEL training "
            f"trajectory contract: {contract}")
    configured_generator = Path(
        str(config.generator_ckpt)).expanduser().resolve()
    requested_generator = args.student_checkpoint.expanduser().resolve()
    if configured_generator != requested_generator:
        raise ValueError(
            "student config generator_ckpt must resolve to the exact "
            f"--student-checkpoint: {configured_generator} != "
            f"{requested_generator}")
    if getattr(config, "resume_checkpoint", None) is not None:
        raise ValueError(
            "student config resume_checkpoint must be null when measuring "
            "the ode_init baseline")
    configured_teacher_lmdb = Path(
        str(config.teacher_latent_data_path)).expanduser().resolve()
    requested_teacher_lmdb = args.teacher_lmdb.expanduser().resolve()
    if configured_teacher_lmdb != requested_teacher_lmdb:
        raise ValueError(
            "student config teacher_latent_data_path must resolve to the "
            f"exact --teacher-lmdb: {configured_teacher_lmdb} != "
            f"{requested_teacher_lmdb}")
    contract["generator_ckpt"] = str(configured_generator)
    contract["resume_checkpoint"] = None
    contract["teacher_latent_data_path"] = str(
        configured_teacher_lmdb)
    identity = {
        "default_config": _file_identity(default_path),
        "override_config": _file_identity(args.student_config),
        "merged_config_sha256": _sha256_json(
            resolved, b"sf-fidel-merged-student-config-v1\0"),
        "training_contract": contract,
    }
    return config, identity


def _build_request(
    args: argparse.Namespace,
    *,
    world_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    world_size = int(world_size)
    if world_size != 1 and world_size not in SUPPORTED_REFERENCE_WORLD_SIZES:
        supported = ", ".join(
            str(value) for value in SUPPORTED_REFERENCE_WORLD_SIZES)
        raise ValueError(
            f"FiDEL student baseline requires a ({supported})-rank launch; "
            "world_size=1 is accepted only for dry-run planning")
    # Keep world_size=1 planning byte-for-byte compatible with the historical
    # default 8-rank request identity.
    request_world_size = (
        REFERENCE_WORLD_SIZE if world_size == 1 else world_size)
    global_microbatches = _global_microbatches_for_world_size(
        request_world_size)
    lmdb_info = _inspect_lmdb(args.teacher_lmdb)
    teacher_count = int(lmdb_info["latents_shape"][0])
    if teacher_count != EXPECTED_TEACHERS:
        raise ValueError(
            f"FiDEL bank requires exactly {EXPECTED_TEACHERS} teacher rows, "
            f"got {teacher_count}")
    teacher_latent_shape = tuple(
        int(value) for value in lmdb_info["latents_shape"])
    if (
        teacher_latent_shape[1] <= 0
        or teacher_latent_shape[2:] != (21, 16, 60, 104)
    ):
        raise ValueError(
            "teacher LMDB must contain at least one trajectory step with "
            "latent geometry [21,16,60,104], got "
            f"{teacher_latent_shape}")
    prompts = _load_prompts(args.teacher_lmdb, teacher_count)
    train_indices, val_indices = deterministic_prompt_split(
        prompts, val_count=VAL_TEACHERS)
    if (len(train_indices), len(val_indices)) != (
        TRAIN_TEACHERS, VAL_TEACHERS
    ):
        raise AssertionError("unexpected deterministic split sizes")
    masks = deterministic_token_masks(
        num_tokens=int(args.sampled_tokens),
        count=int(args.mask_slots),
    )
    baseline_prompt_indices = deterministic_baseline_prompt_indices(
        prompts, train_indices)
    noise_seeds = deterministic_noise_seeds()
    exit_seeds = deterministic_exit_seeds(count=global_microbatches)
    exit_indices = [
        deterministic_exit_indices(seed) for seed in exit_seeds
    ]

    model = _model_identity(args.model_id)
    geometry = model["geometry"]
    if (
        geometry["hidden_size"] != FEATURE_DIM
        or args.layer >= geometry["num_hidden_layers"]
    ):
        raise ValueError(
            "V-JEPA2 model geometry does not match layer-15 ViT-L contract")
    raw_tokens = (
        args.num_frames // geometry["tubelet_size"]
        * (args.crop_size // geometry["patch_size"]) ** 2
    )
    if raw_tokens != RAW_TOKENS:
        raise ValueError(
            f"V-JEPA2 preprocessing yields {raw_tokens} tokens, "
            f"expected {RAW_TOKENS}")

    data_path = Path(lmdb_info["path"]) / "data.mdb"
    teacher_data_identity = _file_identity(data_path)
    vae_identity = _file_identity(args.vae_checkpoint)
    _, student_config_identity = _effective_student_config(args)
    student_checkpoint_identity = _file_identity(args.student_checkpoint)
    text_conditioning_identity = _text_conditioning_identity()
    request = {
        "tool_format_version": TOOL_FORMAT_VERSION,
        "world_size": request_world_size,
        "teacher": {
            "lmdb": {
                **lmdb_info,
                "data_mdb_sha256": teacher_data_identity["sha256"],
            },
            "teacher_count": teacher_count,
            "prompt_normalization": PROMPT_NORMALIZATION,
            "split_hash_order": (
                "sha256(seed\\0canonical_prompt), lmdb_index_tiebreak"),
            "canonical_prompts_sha256": _prompt_list_sha256(prompts),
            "raw_prompts_sha256": _raw_prompt_list_sha256(prompts),
            "split_seed": SPLIT_SEED,
            "train_indices": train_indices,
            "train_indices_sha256": _indices_sha256(train_indices),
            "val_indices": val_indices,
            "val_indices_sha256": _indices_sha256(val_indices),
        },
        "feature": {
            "model": model,
            "vae": vae_identity,
            "layer_index": int(args.layer),
            "prefix_blocks": int(args.layer) + 1,
            "num_frames": int(args.num_frames),
            "resize_short_side": int(args.resize_short_side),
            "crop_size": int(args.crop_size),
            "raw_tokens": RAW_TOKENS,
            "sampled_tokens": int(args.sampled_tokens),
            "feature_dim": FEATURE_DIM,
            "teacher_dense_dtype": "float16_little_endian",
            "student_dense_dtype": "float16_little_endian",
            "teacher_extraction_batch_size": 1,
            "student_extraction_batch_size": 1,
            "cuda_math": {
                "matmul_allow_tf32": True,
                "cudnn_allow_tf32": True,
            },
        },
        "masks": {
            "seed": MASK_SEED,
            "count": int(args.mask_slots),
            "indices": masks,
            "indices_sha256": [
                _indices_sha256(mask) for mask in masks
            ],
        },
        "student": {
            "config": student_config_identity,
            "checkpoint": student_checkpoint_identity,
            "text_conditioning": text_conditioning_identity,
            "samples": STUDENT_SAMPLES,
            "batches": STUDENT_BATCHES,
            "batch_size": STUDENT_BATCH_SIZE,
            "prompt_seed": BASELINE_PROMPT_SEED,
            "prompt_input": (
                "exact_raw_lmdb_string_then_WanTextEncoder_tokenizer_clean"),
            "prompt_indices": baseline_prompt_indices,
            "prompt_indices_sha256": _indices_sha256(
                sorted(baseline_prompt_indices)),
            "ordered_prompt_indices_sha256": _sha256_json(
                baseline_prompt_indices,
                b"ordered-baseline-prompt-indices\0",
            ),
            "noise_seed": BASELINE_NOISE_SEED,
            "noise_seeds": noise_seeds,
            "noise_seeds_sha256": _sha256_json(
                noise_seeds, b"baseline-noise-seeds\0"),
            "sampling_steps": [1000, 750, 500, 250],
            "warp_denoising_step": True,
            "num_frame_per_block": 3,
            "num_training_frames": 21,
            "latent_shape": [21, 16, 60, 104],
            "latent_dtype": "float16",
            "generator_compute_dtype": "bfloat16",
            "noise_dtype": "bfloat16",
            "vae_decode_dtype": "bfloat16",
            "vjepa2_compute_dtype": "float32",
            "trajectory_api": (
                "SelfForcingTrainingPipeline.inference_with_trajectory"),
            "global_microbatches": global_microbatches,
            "world_size": request_world_size,
            "exit_seed": BASELINE_EXIT_SEED,
            "exit_seeds": exit_seeds,
            "exit_seeds_sha256": _sha256_json(
                exit_seeds, b"baseline-exit-seeds\0"),
            "exit_indices": exit_indices,
            "exit_indices_sha256": _sha256_json(
                exit_indices, b"baseline-exit-indices\0"),
            "exit_policy": (
                "rank0_seeded_uniform_4way_broadcast_per_global_microbatch"),
        },
        "nystrom": {
            "num_landmarks": LANDMARKS,
            "median_max_samples": int(args.median_max_samples),
            "kmeans_iterations": int(args.kmeans_iterations),
            "ridge": float(args.ridge),
            "build_chunk_size": int(args.build_chunk_size),
            "kernel_chunk_size": int(args.kernel_chunk_size),
        },
    }
    return request, {"prompts": prompts}


def _teacher_split_manifest(request: Mapping[str, Any]) -> Dict[str, Any]:
    teacher = request["teacher"]
    manifest = {
        "format_version": TEACHER_SPLIT_MANIFEST_VERSION,
        "teacher_count": EXPECTED_TEACHERS,
        "train_indices": teacher["train_indices"],
        "train_indices_sha256": teacher["train_indices_sha256"],
        "val_indices": teacher["val_indices"],
        "val_indices_sha256": teacher["val_indices_sha256"],
        "split_seed": teacher["split_seed"],
        "split_hash_order": teacher["split_hash_order"],
        "prompt_normalization": teacher["prompt_normalization"],
        "canonical_prompts_sha256":
            teacher["canonical_prompts_sha256"],
        "raw_prompts_sha256": teacher["raw_prompts_sha256"],
        "teacher_lmdb_identity": teacher["lmdb"],
        "content_hash": "0" * 64,
    }
    manifest["content_hash"] = _sha256_json(
        {
            key: value
            for key, value in manifest.items()
            if key != "content_hash"
        },
        b"sf-fidel-teacher-split-manifest-v1\0",
    )
    return manifest


def _publish_or_validate_teacher_split(
    path: Path,
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = _teacher_split_manifest(request)
    if path.is_file():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot load existing teacher split {path}") from exc
        if actual != expected:
            raise ValueError(
                "existing teacher split does not match current request")
    else:
        _atomic_json_save(expected, path)
    file_hash = _sha256_file(path)
    return {
        "path": path.name,
        "sha256": file_hash,
        "content_hash": expected["content_hash"],
    }


def compute_stage_state_hash(state: Mapping[str, Any]) -> str:
    missing = (_STATE_KEYS - {"content_hash"}) - set(state)
    if missing:
        raise ValueError(f"stage state is missing keys: {sorted(missing)}")
    payload = {
        key: state[key] for key in _STATE_KEYS if key != "content_hash"
    }
    return _sha256_json(payload, b"vjepa2-fidel-stage-state-v1\0")


def new_stage_state(
    request: Mapping[str, Any],
    *,
    slot_count: int = MASK_SLOTS,
) -> Dict[str, Any]:
    canonical_request = json.loads(_canonical_json(request))
    request_hash = _sha256_json(
        canonical_request, b"vjepa2-fidel-request-v1\0")
    state = {
        "format_version": STAGE_STATE_VERSION,
        "request": canonical_request,
        "request_hash": request_hash,
        "teacher_bank": None,
        "student_bank": None,
        "slots": [None] * int(slot_count),
    }
    state["content_hash"] = compute_stage_state_hash(state)
    return validate_stage_state(
        state, expected_request=canonical_request, slot_count=slot_count)


def _validate_bank_record(
    value: Any,
    *,
    expected_kind: str,
) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _BANK_RECORD_KEYS:
        raise ValueError(f"invalid {expected_kind} bank stage record")
    if value["kind"] != expected_kind:
        raise ValueError(
            f"bank kind {value['kind']!r} != {expected_kind!r}")
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError("bank stage path must be non-empty")
    if (
        not isinstance(value["shape"], list)
        or len(value["shape"]) != 3
        or any(type(item) is not int or item <= 0 for item in value["shape"])
    ):
        raise ValueError("bank stage shape must be three positive integers")
    if value["dtype"] != "float16_little_endian":
        raise ValueError("bank stage dtype mismatch")
    _validate_hex_sha256(value["sha256"], "bank sha256")
    auxiliary = value["auxiliary"]
    if not isinstance(auxiliary, Mapping):
        raise ValueError("bank auxiliary metadata must be a mapping")
    if expected_kind == "teacher_full_dense":
        if auxiliary:
            raise ValueError("teacher bank auxiliary metadata must be empty")
        normalized_auxiliary: Dict[str, Any] = {}
    else:
        if set(auxiliary) != _STUDENT_BANK_AUXILIARY_KEYS:
            raise ValueError("student bank auxiliary metadata keys mismatch")
        latent_dir = auxiliary["student_init_latent_dir"]
        if (
            not isinstance(latent_dir, str)
            or not latent_dir
            or Path(latent_dir).is_absolute()
            or ".." in Path(latent_dir).parts
        ):
            raise ValueError(
                "student latent artifact directory must remain relative")
        if auxiliary["student_init_latent_count"] != STUDENT_SAMPLES:
            raise ValueError("student latent artifact count mismatch")
        bundle_hash = _validate_hex_sha256(
            auxiliary["student_init_latent_bundle_sha256"],
            "student latent bundle sha256",
        )
        manifest_path = auxiliary["student_init_latent_bundle_manifest"]
        if (
            not isinstance(manifest_path, str)
            or not manifest_path
            or Path(manifest_path).is_absolute()
            or ".." in Path(manifest_path).parts
        ):
            raise ValueError(
                "student latent bundle manifest path must remain relative")
        manifest_sha256 = _validate_hex_sha256(
            auxiliary["student_init_latent_bundle_manifest_sha256"],
            "student latent bundle manifest file sha256",
        )
        manifest_content_hash = _validate_hex_sha256(
            auxiliary[
                "student_init_latent_bundle_manifest_content_hash"],
            "student latent bundle manifest content_hash",
        )
        normalized_auxiliary = {
            "student_init_latent_dir": latent_dir,
            "student_init_latent_count": STUDENT_SAMPLES,
            "student_init_latent_bundle_sha256": bundle_hash,
            "student_init_latent_bundle_manifest": manifest_path,
            "student_init_latent_bundle_manifest_sha256":
                manifest_sha256,
            "student_init_latent_bundle_manifest_content_hash":
                manifest_content_hash,
        }
    return {
        "kind": value["kind"],
        "path": value["path"],
        "shape": list(value["shape"]),
        "dtype": value["dtype"],
        "sha256": value["sha256"],
        "auxiliary": normalized_auxiliary,
    }


def _validate_hex_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return value


def validate_stage_state(
    state: Mapping[str, Any],
    *,
    expected_request: Optional[Mapping[str, Any]] = None,
    slot_count: int = MASK_SLOTS,
) -> Dict[str, Any]:
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        raise ValueError("stage state keys mismatch")
    if state["format_version"] != STAGE_STATE_VERSION:
        raise ValueError("stage state version mismatch")
    request = json.loads(_canonical_json(state["request"]))
    request_hash = _validate_hex_sha256(
        state["request_hash"], "request_hash")
    calculated_request_hash = _sha256_json(
        request, b"vjepa2-fidel-request-v1\0")
    if not hmac.compare_digest(request_hash, calculated_request_hash):
        raise ValueError("stage request_hash mismatch")
    if expected_request is not None:
        expected = json.loads(_canonical_json(expected_request))
        expected_hash = _sha256_json(
            expected, b"vjepa2-fidel-request-v1\0")
        if not hmac.compare_digest(request_hash, expected_hash):
            raise ValueError(
                "stage request does not match the current inputs/configuration")
    teacher_bank = _validate_bank_record(
        state["teacher_bank"], expected_kind="teacher_full_dense")
    student_bank = _validate_bank_record(
        state["student_bank"], expected_kind="student_init_full_dense")
    slots = state["slots"]
    if not isinstance(slots, list) or len(slots) != int(slot_count):
        raise ValueError("stage slots length mismatch")
    normalized_slots = []
    for slot, record in enumerate(slots):
        if record is None:
            normalized_slots.append(None)
            continue
        if not isinstance(record, Mapping) or int(record.get("slot", -1)) != slot:
            raise ValueError(f"invalid completed stage slot {slot}")
        # Final strict validation is delegated to build_reference_bank_manifest.
        normalized_slots.append(json.loads(_canonical_json(record)))
    saved_hash = _validate_hex_sha256(
        state["content_hash"], "stage content_hash")
    normalized = {
        "format_version": STAGE_STATE_VERSION,
        "request": request,
        "request_hash": request_hash,
        "teacher_bank": teacher_bank,
        "student_bank": student_bank,
        "slots": normalized_slots,
        "content_hash": saved_hash,
    }
    calculated_hash = compute_stage_state_hash(normalized)
    if not hmac.compare_digest(saved_hash, calculated_hash):
        raise ValueError("stage state content_hash mismatch")
    return normalized


def update_stage_state(
    state: Mapping[str, Any],
    *,
    teacher_bank: Any = ...,
    student_bank: Any = ...,
    slot: Optional[int] = None,
    slot_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    updated = validate_stage_state(
        state, slot_count=len(state["slots"]))
    if teacher_bank is not ...:
        updated["teacher_bank"] = teacher_bank
    if student_bank is not ...:
        updated["student_bank"] = student_bank
    if slot is not None:
        if not 0 <= int(slot) < len(updated["slots"]):
            raise IndexError("stage slot is out of range")
        if slot_record is None:
            raise ValueError("completed slot requires a measured slot record")
        updated["slots"][int(slot)] = dict(slot_record)
    updated["content_hash"] = compute_stage_state_hash(updated)
    return validate_stage_state(
        updated, slot_count=len(updated["slots"]))


def _atomic_json_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_or_initialize_stage_state(
    path: Path,
    request: Mapping[str, Any],
    *,
    slot_count: int = MASK_SLOTS,
) -> Tuple[Dict[str, Any], bool]:
    if not path.is_file():
        return new_stage_state(request, slot_count=slot_count), False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid stage state {path}") from exc
    return validate_stage_state(
        raw,
        expected_request=request,
        slot_count=slot_count,
    ), True


def capacity_estimate(
    *,
    sampled_tokens: int = SAMPLED_TOKENS,
    mask_slots: int = MASK_SLOTS,
) -> Dict[str, int]:
    sampled_tokens = int(sampled_tokens)
    mask_slots = int(mask_slots)
    if not 1 < sampled_tokens <= RAW_TOKENS:
        raise ValueError("invalid sampled token count for capacity estimate")
    if mask_slots <= 0:
        raise ValueError("mask slot count must be positive")
    teacher_bytes = (
        EXPECTED_TEACHERS * RAW_TOKENS * FEATURE_DIM * 2)
    student_bytes = STUDENT_SAMPLES * RAW_TOKENS * FEATURE_DIM * 2
    student_latent_bytes = STUDENT_SAMPLES * 21 * 16 * 60 * 104 * 2
    one_pack_bytes = LANDMARKS * sampled_tokens * FEATURE_DIM * 4
    scratch_bytes = int(math.ceil(
        (
            teacher_bytes
            + student_bytes
            + student_latent_bytes
        ) * 1.03
    )) + 512 * MIB
    output_bytes = int(math.ceil(
        mask_slots * one_pack_bytes * 1.08)) + 1024 * MIB
    return {
        "teacher_dense_bytes": teacher_bytes,
        "student_dense_bytes": student_bytes,
        "student_latent_bytes": student_latent_bytes,
        "one_pack_estimated_bytes": one_pack_bytes,
        "scratch_required_bytes": scratch_bytes,
        "output_required_bytes": output_bytes,
        "combined_required_bytes": scratch_bytes + output_bytes,
    }


def remaining_capacity_estimate(
    *,
    scratch: Path,
    output: Path,
    world_size: int,
    stage: str = "all",
    sampled_tokens: int = SAMPLED_TOKENS,
    mask_slots: int = MASK_SLOTS,
) -> Dict[str, int]:
    """Estimate additional bytes, accounting for resumable large artifacts."""
    sampled_tokens = int(sampled_tokens)
    mask_slots = int(mask_slots)
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    full = capacity_estimate(
        sampled_tokens=sampled_tokens,
        mask_slots=mask_slots,
    )
    teacher_path = _bank_path(scratch, "teacher")
    student_path = _bank_path(scratch, "student")
    teacher_missing = (
        full["teacher_dense_bytes"]
        if not teacher_path.is_file()
        or teacher_path.stat().st_size != full["teacher_dense_bytes"]
        else 0
    )
    student_missing = (
        full["student_dense_bytes"]
        if not student_path.is_file()
        or student_path.stat().st_size != full["student_dense_bytes"]
        else 0
    )
    one_latent_bytes = 21 * 16 * 60 * 104 * 2
    existing_latent_payload_bytes = 0
    for index in range(STUDENT_SAMPLES):
        path = _student_latent_path(scratch, index)
        if path.is_file():
            # Count no more than the raw payload.  Metadata/serialization
            # overhead must never make capacity planning optimistic.
            existing_latent_payload_bytes += min(
                int(path.stat().st_size), one_latent_bytes)
    missing_latent_bytes = max(
        0,
        full["student_latent_bytes"] - existing_latent_payload_bytes,
    )
    existing_pack_bytes = 0
    for slot in range(mask_slots):
        path = output / "packs" / f"slot_{slot:03d}.pt"
        if path.is_file():
            # Do not allow unrelated or corrupted oversized files to erase
            # the reservation for remaining packs.
            existing_pack_bytes += min(
                int(path.stat().st_size),
                full["one_pack_estimated_bytes"],
            )
    total_pack_estimate = (
        mask_slots * full["one_pack_estimated_bytes"])
    missing_pack_bytes = max(
        0, total_pack_estimate - existing_pack_bytes)
    # Atomic replacement can temporarily coexist with one pack per rank.
    replacement_headroom = (
        min(world_size, mask_slots)
        * full["one_pack_estimated_bytes"]
    )
    latent_replacement_headroom = (
        min(world_size, STUDENT_SAMPLES) * one_latent_bytes)
    scratch_remaining = (
        teacher_missing
        + student_missing
        + missing_latent_bytes
        + latent_replacement_headroom
        + 512 * MIB
    )
    output_remaining = missing_pack_bytes + replacement_headroom + 512 * MIB
    if stage == "teacher":
        scratch_remaining = teacher_missing + 256 * MIB
        output_remaining = 0
    elif stage == "student":
        scratch_remaining = (
            student_missing
            + missing_latent_bytes
            + latent_replacement_headroom
            + 256 * MIB
        )
        output_remaining = 0
    elif stage == "slots":
        scratch_remaining = 256 * MIB
    elif stage != "all":
        raise ValueError(f"unknown capacity stage {stage!r}")
    return {
        **full,
        "scratch_remaining_bytes": scratch_remaining,
        "output_remaining_bytes": output_remaining,
    }


def _verify_completed_bank(
    record: Optional[Mapping[str, Any]],
    *,
    scratch: Path,
    expected_kind: str,
    expected_shape: Sequence[int],
    request_wrapper: Optional[Mapping[str, Any]] = None,
    prompts: Optional[Sequence[str]] = None,
) -> None:
    if record is None:
        return
    validated = _validate_bank_record(
        record, expected_kind=expected_kind)
    relative = Path(validated["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("stage bank path must remain inside scratch")
    path = (scratch / relative).resolve()
    try:
        path.relative_to(scratch.resolve())
    except ValueError as exc:
        raise ValueError("stage bank path escapes scratch") from exc
    if validated["shape"] != [int(value) for value in expected_shape]:
        raise ValueError(f"{expected_kind} stage shape mismatch")
    expected_bytes = math.prod(expected_shape) * 2
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"completed stage bank is missing/truncated: {path}")
    actual_hash = _sha256_file(path)
    if not hmac.compare_digest(actual_hash, validated["sha256"]):
        raise ValueError(
            f"completed stage bank SHA mismatch: {path}")
    if expected_kind == "student_init_full_dense":
        if request_wrapper is None or prompts is None:
            raise ValueError(
                "student bank verification requires request and prompts")
        auxiliary = validated["auxiliary"]
        latent_dir = (
            scratch / auxiliary["student_init_latent_dir"]
        ).resolve()
        try:
            latent_dir.relative_to(scratch.resolve())
        except ValueError as exc:
            raise ValueError(
                "student latent directory escapes scratch") from exc
        if latent_dir != _student_latent_dir(scratch).resolve():
            raise ValueError("student latent directory is not canonical")
        manifest_path = (
            scratch / auxiliary[
                "student_init_latent_bundle_manifest"]
        ).resolve()
        try:
            manifest_path.relative_to(scratch.resolve())
        except ValueError as exc:
            raise ValueError(
                "student latent bundle manifest escapes scratch") from exc
        if (
            manifest_path
            != _student_latent_bundle_manifest_path(scratch).resolve()
        ):
            raise ValueError(
                "student latent bundle manifest path is not canonical")
        if not manifest_path.is_file():
            raise ValueError(
                "student latent bundle manifest is missing")
        manifest_file_hash = _sha256_file(manifest_path)
        if not hmac.compare_digest(
            manifest_file_hash,
            auxiliary[
                "student_init_latent_bundle_manifest_sha256"],
        ):
            raise ValueError(
                "student latent bundle manifest file SHA mismatch")
        manifest = _load_student_latent_bundle_manifest(
            manifest_path,
            scratch=scratch,
            request_wrapper=request_wrapper,
            prompts=prompts,
        )
        if (
            manifest["bundle_sha256"]
            != auxiliary["student_init_latent_bundle_sha256"]
            or manifest["content_hash"]
            != auxiliary[
                "student_init_latent_bundle_manifest_content_hash"]
        ):
            raise ValueError(
                "student latent bundle manifest/stage identity mismatch")


def _existing_parent(path: Path) -> Path:
    current = path.expanduser().absolute()
    while not current.exists() and current != current.parent:
        current = current.parent
    if not current.is_dir():
        raise FileNotFoundError(
            f"no existing capacity-check parent for {path}")
    return current


def _check_grouped_capacity(
    scratch: Path,
    output: Path,
    *,
    scratch_required: int,
    output_required: int,
) -> Dict[str, Any]:
    entries = [
        ("scratch", _existing_parent(scratch), int(scratch_required)),
        ("output", _existing_parent(output), int(output_required)),
    ]
    grouped: Dict[int, Dict[str, Any]] = {}
    for name, parent, required in entries:
        device = int(parent.stat().st_dev)
        group = grouped.setdefault(device, {
            "paths": [],
            "required_bytes": 0,
            "free_bytes": int(shutil.disk_usage(parent).free),
        })
        group["paths"].append({"kind": name, "path": str(parent)})
        group["required_bytes"] += required
        group["free_bytes"] = min(
            group["free_bytes"], int(shutil.disk_usage(parent).free))
    for group in grouped.values():
        if group["free_bytes"] < group["required_bytes"]:
            raise RuntimeError(
                "insufficient disk capacity for FiDEL reference bank: "
                f"paths={group['paths']}, "
                f"free={group['free_bytes'] / GIB:.2f} GiB, "
                f"required={group['required_bytes'] / GIB:.2f} GiB")
    return {"filesystems": list(grouped.values())}


def _synchronized_stage_call(
    label: str,
    function,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Run rank-local work and turn any one-rank failure into all-rank failure."""
    error = None
    try:
        function()
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = [None] * int(world_size)
    if dist.is_available() and dist.is_initialized():
        dist.all_gather_object(errors, error)
    else:
        errors[0] = error
    failures = [value for value in errors if value is not None]
    if failures:
        raise RuntimeError(
            f"{label} failed on at least one rank: {'; '.join(failures)}")


def _rank0_result(
    label: str,
    function,
    *,
    rank: int,
) -> Any:
    result = None
    if rank == 0:
        try:
            result = {"value": function(), "error": None}
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            result = {
                "value": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    result = _broadcast_rank0_object(result, rank)
    if result["error"] is not None:
        raise RuntimeError(f"{label} failed on rank 0: {result['error']}")
    return result["value"]


def _bank_shape(kind: str) -> Tuple[int, int, int]:
    count = EXPECTED_TEACHERS if kind == "teacher" else STUDENT_SAMPLES
    return count, RAW_TOKENS, FEATURE_DIM


def _bank_path(scratch: Path, kind: str) -> Path:
    return scratch / f"{kind}_full_dense_fp16.bin"


def _create_or_validate_memmap(path: Path, shape: Sequence[int]) -> None:
    required = math.prod(int(value) for value in shape) * 2
    if path.exists():
        if path.stat().st_size != required:
            raise ValueError(
                f"existing dense bank {path} has size {path.stat().st_size}, "
                f"expected {required}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.truncate(required)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _open_memmap(path: Path, shape: Sequence[int], mode: str):
    import numpy as np

    return np.memmap(
        path,
        dtype=np.dtype("<f2"),
        mode=mode,
        shape=tuple(int(value) for value in shape),
    )


def _rank_indices(count: int, rank: int, world_size: int) -> list[int]:
    return list(range(int(rank), int(count), int(world_size)))


def _rank_marker_path(
    scratch: Path, stage: str, rank: int
) -> Path:
    return scratch / "rank_markers" / f"{stage}_rank_{rank:05d}.json"


def _feature_rows_sha256(bank, indices: Sequence[int]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"vjepa2-dense-rank-rows-v1\0")
    for index in indices:
        hasher.update(int(index).to_bytes(8, "little", signed=False))
        hasher.update(bank[int(index)].tobytes(order="C"))
    return hasher.hexdigest()


def _valid_rank_marker(
    path: Path,
    *,
    request_hash: str,
    stage: str,
    rank: int,
    world_size: int,
    indices: Sequence[int],
    bank,
    student_latent_bundle_sha256: Optional[str] = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "format_version": RANK_MARKER_VERSION,
            "stage": stage,
            "request_hash": request_hash,
            "rank": int(rank),
            "world_size": int(world_size),
            "indices": list(indices),
            "indices_sha256": _indices_sha256(indices),
        }
        for key, value in expected.items():
            if marker.get(key) != value:
                return False
        if stage == "student":
            if student_latent_bundle_sha256 is None:
                return False
            saved_bundle = _validate_hex_sha256(
                marker.get("student_latent_bundle_sha256"),
                "student rank latent bundle sha256",
            )
            if not hmac.compare_digest(
                saved_bundle,
                _validate_hex_sha256(
                    student_latent_bundle_sha256,
                    "current student rank latent bundle sha256",
                ),
            ):
                return False
        elif "student_latent_bundle_sha256" in marker:
            return False
        saved = _validate_hex_sha256(
            marker.get("feature_rows_sha256"), "feature_rows_sha256")
        return hmac.compare_digest(
            saved, _feature_rows_sha256(bank, indices))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _write_rank_marker(
    path: Path,
    *,
    request_hash: str,
    stage: str,
    rank: int,
    world_size: int,
    indices: Sequence[int],
    feature_rows_sha256: str,
    student_latent_bundle_sha256: Optional[str] = None,
) -> None:
    marker = {
        "format_version": RANK_MARKER_VERSION,
        "stage": stage,
        "request_hash": request_hash,
        "rank": int(rank),
        "world_size": int(world_size),
        "indices": list(indices),
        "indices_sha256": _indices_sha256(indices),
        "feature_rows_sha256": feature_rows_sha256,
    }
    if stage == "student":
        marker["student_latent_bundle_sha256"] = _validate_hex_sha256(
            student_latent_bundle_sha256,
            "student rank latent bundle sha256",
        )
    elif student_latent_bundle_sha256 is not None:
        raise ValueError(
            "student latent bundle hash is only valid for student markers")
    _atomic_json_save(marker, path)


def _load_vjepa2_and_vae(
    args: argparse.Namespace,
    device: torch.device,
):
    from utils.paths import model_path
    from utils.wan_wrapper import WanVAEWrapper
    from video_fd.encoders.vjepa2_adapter import VJEPA2Adapter

    active_vae_path = (
        model_path("Wan2.1-T2V-1.3B") / "Wan2.1_VAE.pth"
    ).resolve()
    configured_vae_path = args.vae_checkpoint.expanduser().resolve()
    if active_vae_path != configured_vae_path:
        raise ValueError(
            "configured --vae-checkpoint is not the file loaded by "
            f"WanVAEWrapper: {configured_vae_path} != {active_vae_path}")
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
    actual = (
        int(encoder.dense_raw_tokens),
        int(encoder.dense_feature_dim),
        int(encoder.retained_dense_blocks),
    )
    expected = (RAW_TOKENS, FEATURE_DIM, args.layer + 1)
    if actual != expected:
        raise RuntimeError(
            f"loaded V-JEPA2 geometry {actual} != {expected}")
    return vae, encoder


@torch.no_grad()
def _extract_teacher_rank(
    args: argparse.Namespace,
    request: Mapping[str, Any],
    *,
    scratch: Path,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    from utils.dataset import ODERegressionLMDBDataset

    shape = _bank_shape("teacher")
    path = _bank_path(scratch, "teacher")
    bank = _open_memmap(path, shape, "r+")
    assigned = _rank_indices(shape[0], rank, world_size)
    marker_path = _rank_marker_path(scratch, "teacher", rank)
    if _valid_rank_marker(
        marker_path,
        request_hash=request["request_hash"],
        stage="teacher",
        rank=rank,
        world_size=world_size,
        indices=assigned,
        bank=bank,
    ):
        print(f"[teacher rank {rank}] resume marker valid; skipping", flush=True)
        return

    dataset = ODERegressionLMDBDataset(
        str(args.teacher_lmdb.expanduser().resolve()),
        max_pair=EXPECTED_TEACHERS,
    )
    vae, encoder = _load_vjepa2_and_vae(args, device)
    started = time.monotonic()
    for offset in range(0, len(assigned), args.teacher_batch_size):
        batch_indices = assigned[offset:offset + args.teacher_batch_size]
        items = [dataset[index] for index in batch_indices]
        latent = torch.stack([
            item["ode_latent"][-1] for item in items
        ]).to(device=device, dtype=torch.bfloat16)
        pixels = vae.decode_to_pixel(latent, use_cache=False)
        video01 = ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)
        features = encoder.forward_dense(
            video01,
            layer_index=args.layer,
            token_indices=None,
            gradient_checkpointing=False,
        )
        if tuple(features.shape) != (
            len(batch_indices), RAW_TOKENS, FEATURE_DIM
        ):
            raise RuntimeError(
                f"teacher dense shape mismatch: {tuple(features.shape)}")
        if not bool(torch.isfinite(features).all().item()):
            raise FloatingPointError("non-finite teacher dense features")
        features_fp16 = features.detach().to(
            device="cpu", dtype=torch.float16)
        if not bool(torch.isfinite(features_fp16).all().item()):
            raise FloatingPointError(
                "teacher dense features overflowed during FP16 conversion")
        values = features_fp16.numpy()
        for row, index in enumerate(batch_indices):
            bank[index] = values[row]
        bank.flush()
        del items, latent, pixels, video01, features, features_fp16, values
        completed = min(offset + len(batch_indices), len(assigned))
        if args.log_every and (
            completed == len(assigned)
            or completed % args.log_every == 0
        ):
            print(
                f"[teacher rank {rank}] {completed}/{len(assigned)} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )
    rows_hash = _feature_rows_sha256(bank, assigned)
    _write_rank_marker(
        marker_path,
        request_hash=request["request_hash"],
        stage="teacher",
        rank=rank,
        world_size=world_size,
        indices=assigned,
        feature_rows_sha256=rows_hash,
    )
    del bank, dataset, encoder, vae
    gc.collect()
    torch.cuda.empty_cache()


def _strip_wrapper_prefixes(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    output = {}
    for key, value in state_dict.items():
        cleaned = (
            key.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        output[cleaned] = value
    return output


def _load_student_pipeline(
    args: argparse.Namespace,
    device: torch.device,
):
    from pipeline import CausalInferencePipeline
    from pipeline.self_forcing_training import SelfForcingTrainingPipeline
    from utils.paths import model_path

    active_vae_path = (
        model_path("Wan2.1-T2V-1.3B") / "Wan2.1_VAE.pth"
    ).resolve()
    if active_vae_path != args.vae_checkpoint.expanduser().resolve():
        raise ValueError(
            "student pipeline VAE does not match --vae-checkpoint")

    config, _ = _effective_student_config(args)
    if list(config.denoising_step_list) != [1000, 750, 500, 250]:
        raise ValueError(
            "student config must use the fixed four-step denoising schedule")
    shape = list(config.image_or_video_shape)
    if shape[1:] != [21, 16, 60, 104]:
        raise ValueError(
            "student config image_or_video_shape must be [B,21,16,60,104]")
    pipeline = CausalInferencePipeline(config, device=device)
    raw = torch.load(
        args.student_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if "generator" not in raw:
        raise KeyError("student checkpoint has no 'generator' state")
    cleaned = _strip_wrapper_prefixes(raw["generator"])
    missing, unexpected = pipeline.generator.load_state_dict(
        cleaned, strict=False)
    real_missing = [
        name for name in missing
        if not name.endswith(".kernel_projection")
    ]
    if real_missing or unexpected:
        raise RuntimeError(
            "student checkpoint/model mismatch: "
            f"missing={real_missing}, unexpected={unexpected}")
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval().requires_grad_(False)
    training_pipeline = SelfForcingTrainingPipeline(
        denoising_step_list=pipeline.denoising_step_list.clone(),
        scheduler=pipeline.scheduler,
        generator=pipeline.generator,
        num_frame_per_block=int(config.num_frame_per_block),
        independent_first_frame=bool(config.independent_first_frame),
        same_step_across_blocks=bool(config.same_step_across_blocks),
        last_step_only=bool(config.last_step_only),
        num_max_frames=int(config.num_training_frames),
        context_noise=int(config.context_noise),
    )
    if not training_pipeline.same_step_across_blocks:
        raise ValueError(
            "student baseline requires same_step_across_blocks=true")
    if training_pipeline.last_step_only:
        raise ValueError("student baseline requires last_step_only=false")
    return pipeline, training_pipeline, config


def run_training_trajectory(
    training_pipeline: Any,
    *,
    noise: torch.Tensor,
    conditional_dict: Mapping[str, torch.Tensor],
    exit_indices: Sequence[int],
) -> Tuple[torch.Tensor, Optional[int], Optional[int], int]:
    """Run the exact training trajectory with an explicit early-exit draw.

    The temporary override removes the pipeline's ambient ``torch.randint``
    and its internal distributed broadcast.  Callers are responsible for
    broadcasting ``exit_indices`` before entering this helper.
    """
    if not hasattr(training_pipeline, "inference_with_trajectory"):
        raise TypeError(
            "training pipeline has no inference_with_trajectory API")
    explicit_indices = [int(value) for value in exit_indices]
    if not explicit_indices:
        raise ValueError("trajectory exit descriptor must not be empty")
    denoising_steps = len(training_pipeline.denoising_step_list)
    if any(
        value < 0 or value >= denoising_steps
        for value in explicit_indices
    ):
        raise ValueError("trajectory exit index is out of range")

    had_instance_override = (
        "generate_and_sync_list" in vars(training_pipeline))
    previous_override = vars(training_pipeline).get(
        "generate_and_sync_list")

    def explicit_generate_and_sync_list(
        num_blocks: int,
        num_denoising_steps: int,
        device: torch.device,
    ) -> list[int]:
        del device
        if int(num_blocks) != len(explicit_indices):
            raise ValueError(
                "explicit exit descriptor block count mismatch")
        if int(num_denoising_steps) != denoising_steps:
            raise ValueError(
                "explicit exit descriptor denoising-step mismatch")
        return list(explicit_indices)

    setattr(
        training_pipeline,
        "generate_and_sync_list",
        explicit_generate_and_sync_list,
    )
    try:
        result = training_pipeline.inference_with_trajectory(
            noise=noise,
            return_sim_step=True,
            **dict(conditional_dict),
        )
    finally:
        if had_instance_override:
            setattr(
                training_pipeline,
                "generate_and_sync_list",
                previous_override,
            )
        else:
            delattr(training_pipeline, "generate_and_sync_list")
    if not isinstance(result, tuple) or len(result) != 4:
        raise RuntimeError(
            "training trajectory must return "
            "(x0, timestep_from, timestep_to, sim_steps)")
    latent, timestep_from, timestep_to, sim_steps = result
    sim_steps = int(sim_steps)
    if sim_steps != explicit_indices[0] + 1:
        raise RuntimeError(
            "training trajectory returned the wrong early-exit step")
    normalized_from = (
        None if timestep_from is None else int(timestep_from))
    normalized_to = None if timestep_to is None else int(timestep_to)
    return latent, normalized_from, normalized_to, sim_steps


def _broadcast_exit_table(
    *,
    student_request: Mapping[str, Any],
    device: torch.device,
    rank: int,
) -> list[list[int]]:
    num_blocks = 21 // int(student_request["num_frame_per_block"])
    num_steps = len(student_request["sampling_steps"])
    world_size = int(student_request["world_size"])
    global_microbatches = _global_microbatches_for_world_size(world_size)
    if (
        int(student_request["global_microbatches"]) != global_microbatches
        or len(student_request["exit_seeds"]) != global_microbatches
        or len(student_request["exit_indices"]) != global_microbatches
    ):
        raise ValueError(
            "student request exit table geometry is inconsistent")
    if rank == 0:
        indices = torch.tensor(
            [
                deterministic_exit_indices(
                    int(exit_seed),
                    num_blocks=num_blocks,
                    num_steps=num_steps,
                )
                for exit_seed in student_request["exit_seeds"]
            ],
            dtype=torch.long,
            device=device,
        )
    else:
        indices = torch.empty(
            global_microbatches,
            num_blocks,
            dtype=torch.long,
            device=device,
        )
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(indices, src=0)
    elif rank != 0:
        raise RuntimeError(
            "nonzero rank cannot receive exit descriptor without dist")
    result = [
        [int(value) for value in row]
        for row in indices.cpu().tolist()
    ]
    if (
        len(result) != global_microbatches
        or any(len(row) != num_blocks for row in result)
        or any(
            value < 0 or value >= num_steps
            for row in result
            for value in row
        )
    ):
        raise ValueError("broadcast trajectory exit descriptor is invalid")
    requested = [
        [int(value) for value in row]
        for row in student_request["exit_indices"]
    ]
    if result != requested:
        raise ValueError(
            "broadcast trajectory exit table differs from request")
    return result


def _build_student_latent_artifact(
    latent: torch.Tensor,
    *,
    expected_metadata: Mapping[str, Any],
    denoised_timestep_from: Optional[int],
    denoised_timestep_to: Optional[int],
    sim_steps: int,
) -> Dict[str, Any]:
    canonical_latent = latent.detach().to(
        device="cpu", dtype=torch.float16).contiguous()
    artifact = {
        **dict(expected_metadata),
        "latent": canonical_latent,
        "sim_steps": int(sim_steps),
        "denoised_timestep_from": (
            None
            if denoised_timestep_from is None
            else int(denoised_timestep_from)
        ),
        "denoised_timestep_to": (
            None
            if denoised_timestep_to is None
            else int(denoised_timestep_to)
        ),
        "content_hash": "0" * 64,
    }
    artifact["content_hash"] = compute_student_latent_content_hash(
        artifact)
    return validate_student_latent_artifact(
        artifact, expected_metadata=expected_metadata)


@contextlib.contextmanager
def _fixed_sample_rng(device: torch.device, seed: int):
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        yield


@torch.no_grad()
def _extract_student_rank(
    args: argparse.Namespace,
    request: Mapping[str, Any],
    prompts: Sequence[str],
    *,
    scratch: Path,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    shape = _bank_shape("student")
    if int(request["request"]["student"]["world_size"]) != int(world_size):
        raise ValueError(
            "student request world_size differs from distributed launch")
    path = _bank_path(scratch, "student")
    bank = _open_memmap(path, shape, "r+")
    assigned = _rank_indices(shape[0], rank, world_size)
    marker_path = _rank_marker_path(scratch, "student", rank)
    try:
        existing_latent_bundle_hash = _validate_student_latent_bundle(
            scratch,
            request,
            prompts,
            assigned,
        )
    except (FileNotFoundError, ValueError, OSError):
        existing_latent_bundle_hash = None
    marker_valid = _valid_rank_marker(
        marker_path,
        request_hash=request["request_hash"],
        stage="student",
        rank=rank,
        world_size=world_size,
        indices=assigned,
        bank=bank,
        student_latent_bundle_sha256=existing_latent_bundle_hash,
    )
    all_markers_valid = torch.tensor(
        int(marker_valid), dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(all_markers_valid, op=dist.ReduceOp.MIN)
    if bool(all_markers_valid.item()):
        print(f"[student rank {rank}] resume marker valid; skipping", flush=True)
        return

    # Broadcast the complete rank-0-seeded table before any model/file work.
    # Keeping collectives out of the fallible per-sample loop prevents one
    # rank's local exception from desynchronizing the NCCL collective order.
    exit_table = _broadcast_exit_table(
        student_request=request["request"]["student"],
        device=device,
        rank=rank,
    )
    needs_rebuild = not marker_valid
    pipeline = training_pipeline = config = encoder = None
    if needs_rebuild:
        pipeline, training_pipeline, config = _load_student_pipeline(
            args, device)
    from video_fd.encoders.vjepa2_adapter import VJEPA2Adapter
    if needs_rebuild:
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
    started = time.monotonic()
    for completed, sample_index in enumerate(assigned, start=1):
        expected_metadata = _expected_student_latent_metadata(
            request, prompts, sample_index)
        exit_indices = exit_table[
            expected_metadata["global_microbatch_index"]]
        if exit_indices != expected_metadata["exit_indices"]:
            raise RuntimeError(
                "broadcast exit descriptor differs from request metadata")
        if not needs_rebuild:
            continue

        latent_path = _student_latent_path(scratch, sample_index)
        if latent_path.is_file():
            artifact = _load_student_latent_artifact(
                latent_path, expected_metadata=expected_metadata)
        else:
            prompt_index = expected_metadata["prompt_index"]
            conditional_dict = pipeline.text_encoder(
                text_prompts=[prompts[prompt_index]])
            with _fixed_sample_rng(
                device, expected_metadata["noise_seed"]
            ):
                noise = torch.randn(
                    1, 21, 16, 60, 104,
                    device=device,
                    dtype=torch.bfloat16,
                )
                (
                    latent,
                    timestep_from,
                    timestep_to,
                    sim_steps,
                ) = run_training_trajectory(
                    training_pipeline,
                    noise=noise,
                    conditional_dict=conditional_dict,
                    exit_indices=exit_indices,
                )
            if (
                not isinstance(latent, torch.Tensor)
                or tuple(latent.shape) != (1, 21, 16, 60, 104)
                or not bool(torch.isfinite(latent).all().item())
            ):
                raise RuntimeError(
                    "training trajectory did not produce finite "
                    "[1,21,16,60,104] x0 latent")
            artifact = _build_student_latent_artifact(
                latent[0],
                expected_metadata=expected_metadata,
                denoised_timestep_from=timestep_from,
                denoised_timestep_to=timestep_to,
                sim_steps=sim_steps,
            )
            artifact = _atomic_student_latent_save(
                artifact,
                latent_path,
                expected_metadata=expected_metadata,
            )

        # Feature extraction always decodes the canonical saved FP16 x0.
        # The frame-FiDEL scorer consumes this exact tensor later.
        canonical_latent = artifact["latent"].unsqueeze(0).to(
            device=device, dtype=torch.bfloat16)
        pixels = pipeline.vae.decode_to_pixel(
            canonical_latent, use_cache=False)
        video01 = ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)
        features = encoder.forward_dense(
            video01,
            layer_index=args.layer,
            token_indices=None,
            gradient_checkpointing=False,
        )
        if tuple(features.shape) != (1, RAW_TOKENS, FEATURE_DIM):
            raise RuntimeError(
                f"student dense shape mismatch: {tuple(features.shape)}")
        if not bool(torch.isfinite(features).all().item()):
            raise FloatingPointError("non-finite student dense features")
        features_fp16 = features[0].to(
            device="cpu", dtype=torch.float16)
        if not bool(torch.isfinite(features_fp16).all().item()):
            raise FloatingPointError(
                "student dense features overflowed during FP16 conversion")
        bank[sample_index] = features_fp16.numpy()
        bank.flush()
        if hasattr(pipeline.vae.model, "clear_cache"):
            pipeline.vae.model.clear_cache()
        del (
            artifact,
            canonical_latent,
            pixels,
            video01,
            features,
            features_fp16,
        )
        if args.log_every and (
            completed == len(assigned)
            or completed % args.log_every == 0
        ):
            print(
                f"[student rank {rank}] {completed}/{len(assigned)} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )
    if not needs_rebuild:
        print(
            f"[student rank {rank}] local marker remained valid while "
            "other ranks repaired their rows",
            flush=True,
        )
        del bank
        return
    rows_hash = _feature_rows_sha256(bank, assigned)
    latent_bundle_hash = _validate_student_latent_bundle(
        scratch, request, prompts, assigned)
    _write_rank_marker(
        marker_path,
        request_hash=request["request_hash"],
        stage="student",
        rank=rank,
        world_size=world_size,
        indices=assigned,
        feature_rows_sha256=rows_hash,
        student_latent_bundle_sha256=latent_bundle_hash,
    )
    del bank, encoder, training_pipeline, pipeline, config
    gc.collect()
    torch.cuda.empty_cache()


def _bank_record(
    *,
    kind: str,
    path: Path,
    scratch: Path,
    shape: Sequence[int],
    auxiliary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if path.stat().st_size != math.prod(shape) * 2:
        raise ValueError(f"dense bank size mismatch: {path}")
    record = {
        "kind": kind,
        "path": path.relative_to(scratch).as_posix(),
        "shape": [int(value) for value in shape],
        "dtype": "float16_little_endian",
        "sha256": _sha256_file(path),
        "auxiliary": dict(auxiliary or {}),
    }
    return _validate_bank_record(record, expected_kind=kind)


def _select_dense_matrices(
    bank,
    sample_indices: Sequence[int],
    token_indices: Sequence[int],
) -> torch.Tensor:
    output = torch.empty(
        len(sample_indices),
        len(token_indices),
        FEATURE_DIM,
        dtype=torch.float16,
    )
    for row, sample_index in enumerate(sample_indices):
        # Index one video first so NumPy never materializes an
        # [all_samples,8192,1024] advanced-index copy.
        values = bank[int(sample_index)][list(token_indices)]
        output[row].copy_(torch.from_numpy(values.copy()))
    if not bool(torch.isfinite(output).all().item()):
        raise FloatingPointError("selected dense feature matrix is non-finite")
    return output


def _slot_pack_metadata(
    args: argparse.Namespace,
    request: Mapping[str, Any],
    *,
    slot: int,
    token_indices: Sequence[int],
) -> Dict[str, Any]:
    feature = canonical_vjepa2_dense_feature_metadata(
        model_id=str(Path(args.model_id).expanduser().resolve()),
        layer_index=args.layer,
        num_frames=args.num_frames,
        resize_short_side=args.resize_short_side,
        crop_size=args.crop_size,
        raw_tokens=RAW_TOKENS,
        num_tokens=int(request["feature"]["sampled_tokens"]),
        feature_dim=FEATURE_DIM,
        token_indices=list(token_indices),
    )
    teacher = request["teacher"]
    return {
        "feature": feature,
        "reference": {
            "source": {
                "kind": "teacher_train_full_vjepa2_dense_bank",
                "teacher_count": EXPECTED_TEACHERS,
                "train_count": TRAIN_TEACHERS,
                "train_indices_sha256": teacher[
                    "train_indices_sha256"],
                "val_count": VAL_TEACHERS,
                "val_indices_sha256": teacher["val_indices_sha256"],
                "teacher_full_dense_bank_fp16_sha256": request[
                    "teacher_full_dense_bank_fp16_sha256"],
            },
            "slot": {
                "index": int(slot),
                "token_indices_sha256": _indices_sha256(token_indices),
            },
            "extraction": {
                "tool": (
                    "tools/precompute_vjepa2_fidel_reference_bank.py"),
                "tool_format_version": TOOL_FORMAT_VERSION,
                "prompt_normalization": PROMPT_NORMALIZATION,
            },
            "student_start_calibration": {
                "student_init_count": STUDENT_SAMPLES,
                "student_init_batches": STUDENT_BATCHES,
                "student_init_full_dense_bank_fp16_sha256": request[
                    "student_init_full_dense_bank_fp16_sha256"],
                "student_init_latent_bundle_sha256": request[
                    "student_init_latent_bundle_sha256"],
            },
        },
    }


@torch.no_grad()
def _build_slot(
    args: argparse.Namespace,
    request: Mapping[str, Any],
    *,
    slot: int,
    scratch: Path,
    output: Path,
    device: torch.device,
) -> Dict[str, Any]:
    from losses.video_nystrom_mmd import (
        VideoNystromMMD,
        build_video_nystrom_reference,
        load_reference_pack,
        save_reference_pack,
    )

    mask = request["masks"]["indices"][slot]
    teacher_bank = _open_memmap(
        _bank_path(scratch, "teacher"), _bank_shape("teacher"), "r")
    student_bank = _open_memmap(
        _bank_path(scratch, "student"), _bank_shape("student"), "r")
    teacher_train = _select_dense_matrices(
        teacher_bank, request["teacher"]["train_indices"], mask)
    teacher_val = _select_dense_matrices(
        teacher_bank, request["teacher"]["val_indices"], mask)
    student = _select_dense_matrices(
        student_bank, range(STUDENT_SAMPLES), mask)
    del teacher_bank, student_bank

    metadata = _slot_pack_metadata(
        args, request, slot=slot, token_indices=mask)
    train_device = teacher_train.to(device=device, dtype=torch.float16)
    pack = build_video_nystrom_reference(
        train_device,
        num_landmarks=LANDMARKS,
        metadata=metadata,
        bandwidth=None,
        median_max_samples=int(args.median_max_samples),
        kmeans_iterations=int(args.kmeans_iterations),
        ridge=float(args.ridge),
        chunk_size=int(args.build_chunk_size),
        seed=0,
    )
    del train_device, teacher_train
    objective = VideoNystromMMD(
        pack, kernel_chunk_size=int(args.kernel_chunk_size)).to(device)
    val_score, _ = objective(
        teacher_val.to(device=device, dtype=torch.float16))
    real_floor = float(val_score.detach().cpu())
    student_scores = []
    for batch in range(STUDENT_BATCHES):
        start = batch * STUDENT_BATCH_SIZE
        end = start + STUDENT_BATCH_SIZE
        score, _ = objective(
            student[start:end].to(device=device, dtype=torch.float16))
        student_scores.append(float(score.detach().cpu()))
    scores = torch.tensor(student_scores, dtype=torch.float64)
    start_mean = float(scores.mean())
    start_std = float(scores.std(unbiased=False))
    values = [real_floor, start_mean, start_std, *student_scores]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError(
            f"slot {slot} produced non-finite measured scores")
    if start_mean - real_floor <= 1.0e-8:
        raise ValueError(
            f"slot {slot} invalid student/reference gap: "
            f"start={start_mean}, floor={real_floor}")

    pack_relative = f"packs/slot_{slot:03d}.pt"
    pack_path = output / pack_relative
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    save_reference_pack(pack, pack_path, atomic=True)
    restored = load_reference_pack(
        pack_path,
        expected_metadata={
            "feature": metadata["feature"],
            "nystrom": {"num_landmarks": LANDMARKS},
        },
    )
    del objective, teacher_val, student, pack
    torch.cuda.empty_cache()
    return {
        "slot": int(slot),
        "pack_path": pack_relative,
        "pack_content_hash": restored["content_hash"],
        "token_indices": list(mask),
        "token_indices_sha256": _indices_sha256(mask),
        "video_real_floor": real_floor,
        "video_student_start_score": start_mean,
        "video_student_start_std": start_std,
    }


def _slot_marker_path(scratch: Path, slot: int) -> Path:
    return scratch / "slot_records" / f"slot_{slot:03d}.json"


def _load_slot_marker(
    path: Path,
    *,
    slot: int,
    request: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {
            "request_hash",
            "bank_identity",
            "record",
            "content_hash",
        }:
            return None
        saved_hash = _validate_hex_sha256(
            payload["content_hash"], "slot marker content_hash")
        calculated_hash = _sha256_json(
            {
                "request_hash": payload["request_hash"],
                "bank_identity": payload["bank_identity"],
                "record": payload["record"],
            },
            b"vjepa2-fidel-slot-marker-v2\0",
        )
        if not hmac.compare_digest(saved_hash, calculated_hash):
            return None
        if payload.get("request_hash") != request["request_hash"]:
            return None
        expected_bank_identity = {
            key: request[key] for key in (
                "teacher_bank_sha256",
                "student_bank_sha256",
                "student_latent_bundle_sha256",
            )
        }
        if payload["bank_identity"] != expected_bank_identity:
            return None
        record = payload["record"]
        if (
            record["slot"] != slot
            or record["token_indices"]
            != request["request"]["masks"]["indices"][slot]
        ):
            return None
        from losses.video_nystrom_reference_bank import _validate_slot_record
        _validate_slot_record(record, expected_slot=slot)
        return record
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_slot_marker(
    path: Path,
    request_wrapper: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    payload = {
        "request_hash": request_wrapper["request_hash"],
        "bank_identity": {
            key: request_wrapper[key] for key in (
                "teacher_bank_sha256",
                "student_bank_sha256",
                "student_latent_bundle_sha256",
            )
        },
        "record": dict(record),
    }
    payload["content_hash"] = _sha256_json(
        payload, b"vjepa2-fidel-slot-marker-v2\0")
    _atomic_json_save(payload, path)


def _build_slots_rank(
    args: argparse.Namespace,
    request_wrapper: Mapping[str, Any],
    *,
    scratch: Path,
    output: Path,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    request = request_wrapper["request"]
    calibrated_request = {
        **request,
        "teacher_full_dense_bank_fp16_sha256":
            request_wrapper["teacher_bank_sha256"],
        "student_init_full_dense_bank_fp16_sha256":
            request_wrapper["student_bank_sha256"],
        "student_init_latent_bundle_sha256":
            request_wrapper["student_latent_bundle_sha256"],
    }
    mask_slots = int(request["masks"]["count"])
    if (
        mask_slots <= 0
        or len(request["masks"]["indices"]) != mask_slots
        or len(request["masks"]["indices_sha256"]) != mask_slots
    ):
        raise ValueError("request token-mask slot geometry is inconsistent")
    assigned = _rank_indices(mask_slots, rank, world_size)
    for completed, slot in enumerate(assigned, start=1):
        marker_path = _slot_marker_path(scratch, slot)
        existing = _load_slot_marker(
            marker_path, slot=slot, request=request_wrapper)
        if existing is not None:
            pack_path = output / existing["pack_path"]
            if pack_path.is_file():
                from losses.video_nystrom_mmd import load_reference_pack
                pack = load_reference_pack(pack_path)
                feature = pack["metadata"].get("feature", {})
                expected_metadata = _slot_pack_metadata(
                    args,
                    calibrated_request,
                    slot=slot,
                    token_indices=request["masks"]["indices"][slot],
                )
                if (
                    hmac.compare_digest(
                    pack["content_hash"],
                    existing["pack_content_hash"],
                    )
                    and feature == expected_metadata["feature"]
                    and pack["metadata"].get("reference")
                    == expected_metadata["reference"]
                ):
                    print(
                        f"[slot rank {rank}] resume slot {slot}", flush=True)
                    continue
        record = _build_slot(
            args,
            calibrated_request,
            slot=slot,
            scratch=scratch,
            output=output,
            device=device,
        )
        _write_slot_marker(
            marker_path, request_wrapper, record)
        print(
            f"[slot rank {rank}] {completed}/{len(assigned)} "
            f"slot={slot} floor={record['video_real_floor']:.6g} "
            f"start={record['video_student_start_score']:.6g}",
            flush=True,
        )


def _manifest_metadata(
    request: Mapping[str, Any],
    *,
    teacher_bank_hash: str,
    student_bank_hash: str,
    student_latent_bundle_hash: str,
    student_latent_bundle_manifest: Mapping[str, str],
    teacher_split_manifest: Mapping[str, str],
) -> Dict[str, Any]:
    teacher = request["teacher"]
    if set(student_latent_bundle_manifest) != {
        "path", "sha256", "content_hash"
    }:
        raise ValueError(
            "student latent bundle manifest identity keys mismatch")
    latent_manifest_identity = {
        "path": str(student_latent_bundle_manifest["path"]),
        "sha256": _validate_hex_sha256(
            student_latent_bundle_manifest["sha256"],
            "student latent bundle manifest sha256",
        ),
        "content_hash": _validate_hex_sha256(
            student_latent_bundle_manifest["content_hash"],
            "student latent bundle manifest content_hash",
        ),
    }
    if set(teacher_split_manifest) != {"path", "sha256", "content_hash"}:
        raise ValueError("teacher split manifest identity keys mismatch")
    split_manifest_identity = {
        "path": str(teacher_split_manifest["path"]),
        "sha256": _validate_hex_sha256(
            teacher_split_manifest["sha256"],
            "teacher split manifest sha256",
        ),
        "content_hash": _validate_hex_sha256(
            teacher_split_manifest["content_hash"],
            "teacher split manifest content_hash",
        ),
    }
    return {
        "teacher_count": EXPECTED_TEACHERS,
        "student_init_count": STUDENT_SAMPLES,
        "student_init_batches": STUDENT_BATCHES,
        "train_indices": teacher["train_indices"],
        "train_indices_sha256": teacher["train_indices_sha256"],
        "val_indices": teacher["val_indices"],
        "val_indices_sha256": teacher["val_indices_sha256"],
        "split_seed": SPLIT_SEED,
        "split_hash_order": teacher["split_hash_order"],
        "token_mask_seed": MASK_SEED,
        "prompt_normalization": PROMPT_NORMALIZATION,
        "canonical_prompts_sha256": teacher[
            "canonical_prompts_sha256"],
        "raw_prompts_sha256": teacher["raw_prompts_sha256"],
        "teacher_split_manifest": split_manifest_identity,
        "teacher_lmdb": teacher["lmdb"],
        "teacher_lmdb_identity": teacher["lmdb"],
        "vjepa2_model": request["feature"]["model"],
        "vae": request["feature"]["vae"],
        "vae_identity": request["feature"]["vae"],
        "student_config": request["student"]["config"],
        "student_checkpoint": request["student"]["checkpoint"],
        "student_checkpoint_sha256": request[
            "student"]["checkpoint"]["sha256"],
        "generator_checkpoint_identity":
            request["student"]["checkpoint"],
        "generator_checkpoint_sha256": request[
            "student"]["checkpoint"]["sha256"],
        "generator_effective_config_identity":
            request["student"]["config"],
        "text_conditioning_identity":
            request["student"]["text_conditioning"],
        "student_baseline": {
            key: request["student"][key] for key in (
                "samples",
                "batches",
                "batch_size",
                "prompt_seed",
                "prompt_input",
                "prompt_indices",
                "prompt_indices_sha256",
                "ordered_prompt_indices_sha256",
                "noise_seed",
                "noise_seeds",
                "noise_seeds_sha256",
                "sampling_steps",
                "warp_denoising_step",
                "num_frame_per_block",
                "num_training_frames",
                "latent_shape",
                "latent_dtype",
                "generator_compute_dtype",
                "noise_dtype",
                "vae_decode_dtype",
                "vjepa2_compute_dtype",
                "trajectory_api",
                "global_microbatches",
                "world_size",
                "exit_seed",
                "exit_seeds",
                "exit_seeds_sha256",
                "exit_indices",
                "exit_indices_sha256",
                "exit_policy",
            )
        },
        "token_masks": request["masks"],
        "feature_geometry": {
            key: request["feature"][key] for key in (
                "layer_index",
                "prefix_blocks",
                "num_frames",
                "resize_short_side",
                "crop_size",
                "raw_tokens",
                "sampled_tokens",
                "feature_dim",
                "teacher_extraction_batch_size",
                "student_extraction_batch_size",
                "cuda_math",
            )
        },
        "teacher_full_dense_bank_fp16_sha256": teacher_bank_hash,
        "student_init_full_dense_bank_fp16_sha256": student_bank_hash,
        "student_init_latent_bundle_sha256":
            student_latent_bundle_hash,
        "student_init_latent_scratch_relative_dir":
            "student_init_latents",
        "student_init_latent_bundle_manifest":
            latent_manifest_identity,
        "nystrom_build": request["nystrom"],
        "tool": (
            "tools/precompute_vjepa2_fidel_reference_bank.py"),
        "tool_format_version": TOOL_FORMAT_VERSION,
    }


def _validate_published(
    output: Path,
    *,
    sampled_tokens: int = SAMPLED_TOKENS,
    mask_slots: int = MASK_SLOTS,
) -> Dict[str, Any]:
    from losses.video_nystrom_reference_bank import (
        VideoNystromReferenceBank,
    )

    sampled_tokens = int(sampled_tokens)
    mask_slots = int(mask_slots)
    if not 1 < sampled_tokens <= RAW_TOKENS or mask_slots <= 0:
        raise ValueError("invalid runtime token-mask geometry")
    manifest_path = output / "manifest.json"
    manifest = load_reference_bank_manifest(
        manifest_path, expected_slot_count=mask_slots)
    bank = VideoNystromReferenceBank(
        manifest_path, expected_slot_count=mask_slots)
    metadata = manifest["metadata"]
    token_masks = metadata.get("token_masks")
    feature_geometry = metadata.get("feature_geometry")
    if (
        not isinstance(token_masks, Mapping)
        or token_masks.get("count") != mask_slots
        or not isinstance(token_masks.get("indices"), list)
        or len(token_masks["indices"]) != mask_slots
        or not isinstance(token_masks.get("indices_sha256"), list)
        or len(token_masks["indices_sha256"]) != mask_slots
        or not isinstance(feature_geometry, Mapping)
        or feature_geometry.get("sampled_tokens") != sampled_tokens
    ):
        raise ValueError(
            "published token-mask geometry does not match runtime values")
    for slot, record in enumerate(manifest["slots"]):
        if (
            len(record["token_indices"]) != sampled_tokens
            or record["token_indices"] != token_masks["indices"][slot]
            or record["token_indices_sha256"]
            != token_masks["indices_sha256"][slot]
        ):
            raise ValueError(
                f"published token-mask identity mismatch at slot {slot}")
    for name, expected_filename in (
        ("teacher_split_manifest", "teacher_split.json"),
        (
            "student_init_latent_bundle_manifest",
            "student_init_latent_manifest.json",
        ),
    ):
        identity = metadata.get(name)
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"path", "sha256", "content_hash"}
            or identity["path"] != expected_filename
        ):
            raise ValueError(
                f"published {name} identity is invalid")
        companion_path = output / expected_filename
        if not companion_path.is_file():
            raise FileNotFoundError(companion_path)
        if not hmac.compare_digest(
            _sha256_file(companion_path),
            _validate_hex_sha256(
                identity["sha256"], f"published {name} sha256"),
        ):
            raise ValueError(f"published {name} file SHA mismatch")
        payload = json.loads(
            companion_path.read_text(encoding="utf-8"))
        if payload.get("content_hash") != identity["content_hash"]:
            raise ValueError(
                f"published {name} content_hash mismatch")
        if name == "student_init_latent_bundle_manifest":
            calculated = compute_student_latent_bundle_manifest_hash(
                payload)
            student_baseline = metadata.get("student_baseline")
            if (
                calculated != payload["content_hash"]
                or payload.get("bundle_sha256")
                != metadata["student_init_latent_bundle_sha256"]
                or payload.get("generator_checkpoint_identity")
                != metadata["generator_checkpoint_identity"]
                or payload.get("text_conditioning_identity")
                != metadata["text_conditioning_identity"]
                or not isinstance(student_baseline, Mapping)
                or payload.get("world_size")
                != student_baseline.get("world_size")
                or payload.get("global_microbatches")
                != student_baseline.get("global_microbatches")
            ):
                raise ValueError(
                    "published student latent manifest provenance mismatch")
        else:
            calculated = _sha256_json(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "content_hash"
                },
                b"sf-fidel-teacher-split-manifest-v1\0",
            )
            if (
                calculated != payload["content_hash"]
                or payload.get("train_indices")
                != metadata["train_indices"]
                or payload.get("val_indices")
                != metadata["val_indices"]
            ):
                raise ValueError(
                    "published teacher split provenance mismatch")
    hashes = bank.validate_all_packs()
    return {
        "manifest": str(manifest_path),
        "manifest_hash": manifest["content_hash"],
        "slot_count": len(hashes),
    }


def _dry_run(args: argparse.Namespace) -> None:
    request, _ = _build_request(args, world_size=int(
        os.environ.get("WORLD_SIZE", "1")))
    estimate = capacity_estimate(
        sampled_tokens=args.sampled_tokens,
        mask_slots=args.mask_slots,
    )
    remaining = remaining_capacity_estimate(
        scratch=args.scratch_dir,
        output=args.output_dir,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        stage=args.stage,
        sampled_tokens=args.sampled_tokens,
        mask_slots=args.mask_slots,
    )
    capacity = _check_grouped_capacity(
        args.scratch_dir,
        args.output_dir,
        scratch_required=remaining["scratch_remaining_bytes"],
        output_required=remaining["output_remaining_bytes"],
    )
    print(json.dumps({
        "status": "dry-run",
        "request_hash": _sha256_json(
            request, b"vjepa2-fidel-request-v1\0"),
        "teacher_count": EXPECTED_TEACHERS,
        "train_count": TRAIN_TEACHERS,
        "val_count": VAL_TEACHERS,
        "mask_count": int(args.mask_slots),
        "mask_shape": [int(args.sampled_tokens)],
        "student_samples": STUDENT_SAMPLES,
        "capacity": {**estimate, **remaining, **capacity},
    }, indent=2, sort_keys=True), flush=True)


def _run(args: argparse.Namespace) -> None:
    _validate_args(args)
    output = args.output_dir.expanduser().absolute()
    if args.validate_only:
        print(json.dumps(
            {
                "status": "ok",
                **_validate_published(
                    output,
                    sampled_tokens=args.sampled_tokens,
                    mask_slots=args.mask_slots,
                ),
            },
            indent=2,
            sort_keys=True,
        ), flush=True)
        return
    if args.dry_run:
        _dry_run(args)
        return

    device, rank, world_size = _setup_distributed(
        args.distributed_timeout_minutes)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if world_size not in SUPPORTED_REFERENCE_WORLD_SIZES:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        supported = ", ".join(
            str(value) for value in SUPPORTED_REFERENCE_WORLD_SIZES)
        raise RuntimeError(
            f"FiDEL reference precompute must run with {supported} ranks; "
            f"got {world_size}")
    plan_result = None
    if rank == 0:
        try:
            request, aux = _build_request(args, world_size=world_size)
            estimate = capacity_estimate(
                sampled_tokens=args.sampled_tokens,
                mask_slots=args.mask_slots,
            )
            plan_result = {
                "request": request,
                "prompts": aux["prompts"],
                "estimate": estimate,
                "error": None,
            }
        except Exception as exc:
            plan_result = {
                "error": f"{type(exc).__name__}: {exc}"}
    plan_result = _broadcast_rank0_object(plan_result, rank)
    if plan_result["error"] is not None:
        raise RuntimeError(f"rank-0 planning failed: {plan_result['error']}")
    request = plan_result["request"]
    prompts = plan_result["prompts"]
    request_hash = _sha256_json(
        request, b"vjepa2-fidel-request-v1\0")
    wrapper = {
        "request": request,
        "request_hash": request_hash,
    }
    scratch = args.scratch_dir.expanduser().absolute()
    state_path = scratch / "stage_state.json"

    init_result = None
    if rank == 0:
        try:
            scratch.mkdir(parents=True, exist_ok=True)
            output.mkdir(parents=True, exist_ok=True)
            teacher_split_identity = (
                _publish_or_validate_teacher_split(
                    output / "teacher_split.json", request))
            state, resumed = load_or_initialize_stage_state(
                state_path,
                request,
                slot_count=int(args.mask_slots),
            )
            _verify_completed_bank(
                state["teacher_bank"],
                scratch=scratch,
                expected_kind="teacher_full_dense",
                expected_shape=_bank_shape("teacher"),
            )
            _verify_completed_bank(
                state["student_bank"],
                scratch=scratch,
                expected_kind="student_init_full_dense",
                expected_shape=_bank_shape("student"),
                request_wrapper=wrapper,
                prompts=prompts,
            )
            remaining = remaining_capacity_estimate(
                scratch=scratch,
                output=output,
                world_size=world_size,
                stage=args.stage,
                sampled_tokens=args.sampled_tokens,
                mask_slots=args.mask_slots,
            )
            capacity = _check_grouped_capacity(
                scratch,
                output,
                scratch_required=remaining["scratch_remaining_bytes"],
                output_required=remaining["output_remaining_bytes"],
            )
            if not resumed:
                _atomic_json_save(state, state_path)
            init_result = {
                "state": state,
                "resumed": resumed,
                "remaining": remaining,
                "capacity": capacity,
                "teacher_split_identity": teacher_split_identity,
                "error": None,
            }
        except Exception as exc:
            init_result = {
                "error": f"{type(exc).__name__}: {exc}"}
    init_result = _broadcast_rank0_object(init_result, rank)
    if init_result["error"] is not None:
        raise RuntimeError(
            f"stage-state initialization failed: {init_result['error']}")
    state = init_result["state"]
    teacher_split_identity = init_result["teacher_split_identity"]
    run_teacher = args.stage in ("all", "teacher")
    run_student = args.stage in ("all", "student")
    run_slots = args.stage in ("all", "slots")
    try:
        if run_teacher and state["teacher_bank"] is None:
            def prepare_teacher_bank():
                _create_or_validate_memmap(
                    _bank_path(scratch, "teacher"), _bank_shape("teacher"))
            _rank0_result(
                "teacher bank allocation",
                prepare_teacher_bank,
                rank=rank,
            )

            _synchronized_stage_call(
                "teacher extraction",
                lambda: _extract_teacher_rank(
                    args,
                    wrapper,
                    scratch=scratch,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                ),
                rank=rank,
                world_size=world_size,
            )

            def finalize_teacher():
                nonlocal state
                record = _bank_record(
                    kind="teacher_full_dense",
                    path=_bank_path(scratch, "teacher"),
                    scratch=scratch,
                    shape=_bank_shape("teacher"),
                )
                state = update_stage_state(state, teacher_bank=record)
                _atomic_json_save(state, state_path)
                return state

            state = _rank0_result(
                "teacher bank finalization",
                finalize_teacher,
                rank=rank,
            )

        if run_student and state["student_bank"] is None:
            def prepare_student_bank():
                _create_or_validate_memmap(
                    _bank_path(scratch, "student"), _bank_shape("student"))
                _student_latent_dir(scratch).mkdir(
                    parents=True, exist_ok=True)
            _rank0_result(
                "student bank allocation",
                prepare_student_bank,
                rank=rank,
            )

            _synchronized_stage_call(
                "student baseline extraction",
                lambda: _extract_student_rank(
                    args,
                    wrapper,
                    prompts,
                    scratch=scratch,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                ),
                rank=rank,
                world_size=world_size,
            )

            def finalize_student():
                nonlocal state
                latent_bundle_manifest = (
                    _build_student_latent_bundle_manifest(
                        scratch,
                        wrapper,
                        prompts,
                    )
                )
                latent_manifest_path = (
                    _student_latent_bundle_manifest_path(scratch))
                _atomic_json_save(
                    latent_bundle_manifest, latent_manifest_path)
                restored_latent_manifest = (
                    _load_student_latent_bundle_manifest(
                        latent_manifest_path,
                        scratch=scratch,
                        request_wrapper=wrapper,
                        prompts=prompts,
                    )
                )
                latent_bundle_hash = restored_latent_manifest[
                    "bundle_sha256"]
                record = _bank_record(
                    kind="student_init_full_dense",
                    path=_bank_path(scratch, "student"),
                    scratch=scratch,
                    shape=_bank_shape("student"),
                    auxiliary={
                        "student_init_latent_dir":
                            "student_init_latents",
                        "student_init_latent_count":
                            STUDENT_SAMPLES,
                        "student_init_latent_bundle_sha256":
                            latent_bundle_hash,
                        "student_init_latent_bundle_manifest":
                            "student_init_latents/manifest.json",
                        "student_init_latent_bundle_manifest_sha256":
                            _sha256_file(latent_manifest_path),
                        "student_init_latent_bundle_manifest_content_hash":
                            restored_latent_manifest["content_hash"],
                    },
                )
                state = update_stage_state(state, student_bank=record)
                _atomic_json_save(state, state_path)
                return state

            state = _rank0_result(
                "student bank finalization",
                finalize_student,
                rank=rank,
            )

        if run_slots:
            if state["teacher_bank"] is None or state["student_bank"] is None:
                raise RuntimeError(
                    "slot stage requires completed teacher and student banks")
            wrapper["teacher_bank_sha256"] = state[
                "teacher_bank"]["sha256"]
            wrapper["student_bank_sha256"] = state[
                "student_bank"]["sha256"]
            wrapper["student_latent_bundle_sha256"] = state[
                "student_bank"]["auxiliary"][
                    "student_init_latent_bundle_sha256"]
            _synchronized_stage_call(
                "Nyström slot construction",
                lambda: _build_slots_rank(
                    args,
                    wrapper,
                    scratch=scratch,
                    output=output,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                ),
                rank=rank,
                world_size=world_size,
            )

            def publish_manifest():
                nonlocal state
                records = []
                mask_slots = int(request["masks"]["count"])
                for slot in range(mask_slots):
                    record = _load_slot_marker(
                        _slot_marker_path(scratch, slot),
                        slot=slot,
                        request=wrapper,
                    )
                    if record is None:
                        raise RuntimeError(
                            f"missing measured slot record {slot}")
                    records.append(record)
                    state = update_stage_state(
                        state, slot=slot, slot_record=record)
                _atomic_json_save(state, state_path)
                student_auxiliary = state[
                    "student_bank"]["auxiliary"]
                scratch_latent_manifest_path = (
                    scratch / student_auxiliary[
                        "student_init_latent_bundle_manifest"]
                )
                if not scratch_latent_manifest_path.is_file():
                    raise RuntimeError(
                        "student latent bundle manifest disappeared")
                if not hmac.compare_digest(
                    _sha256_file(scratch_latent_manifest_path),
                    student_auxiliary[
                        "student_init_latent_bundle_manifest_sha256"],
                ):
                    raise RuntimeError(
                        "student latent bundle manifest file SHA changed")
                compact_latent_manifest = json.loads(
                    scratch_latent_manifest_path.read_text(
                        encoding="utf-8"))
                if (
                    compact_latent_manifest.get("bundle_sha256")
                    != student_auxiliary[
                        "student_init_latent_bundle_sha256"]
                    or compact_latent_manifest.get("content_hash")
                    != student_auxiliary[
                        "student_init_latent_bundle_manifest_content_hash"]
                    or compute_student_latent_bundle_manifest_hash(
                        compact_latent_manifest)
                    != compact_latent_manifest.get("content_hash")
                ):
                    raise RuntimeError(
                        "student latent bundle compact manifest mismatch")
                published_latent_manifest_path = (
                    output / "student_init_latent_manifest.json")
                _atomic_json_save(
                    compact_latent_manifest,
                    published_latent_manifest_path,
                )
                published_latent_manifest_identity = {
                    "path": published_latent_manifest_path.name,
                    "sha256": _sha256_file(
                        published_latent_manifest_path),
                    "content_hash": compact_latent_manifest[
                        "content_hash"],
                }
                metadata = _manifest_metadata(
                    request,
                    teacher_bank_hash=state["teacher_bank"]["sha256"],
                    student_bank_hash=state["student_bank"]["sha256"],
                    student_latent_bundle_hash=state[
                        "student_bank"]["auxiliary"][
                            "student_init_latent_bundle_sha256"],
                    student_latent_bundle_manifest=
                        published_latent_manifest_identity,
                    teacher_split_manifest=teacher_split_identity,
                )
                manifest = build_reference_bank_manifest(
                    metadata=metadata,
                    slots=records,
                    expected_slot_count=mask_slots,
                )
                save_reference_bank_manifest(
                    manifest,
                    output / "manifest.json",
                    expected_slot_count=mask_slots,
                    atomic=True,
                )
                summary = _validate_published(
                    output,
                    sampled_tokens=int(
                        request["feature"]["sampled_tokens"]),
                    mask_slots=mask_slots,
                )
                return summary

            summary = _rank0_result(
                "reference-bank publication",
                publish_manifest,
                rank=rank,
            )
            if rank == 0:
                print(json.dumps(
                    {"status": "ok", **summary},
                    indent=2,
                    sort_keys=True,
                ), flush=True)
        elif rank == 0:
            print(json.dumps({
                "status": "stage-complete",
                "stage": args.stage,
                "request_hash": request_hash,
                "stage_state": str(state_path),
            }, indent=2, sort_keys=True), flush=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _run(args)


if __name__ == "__main__":
    main()

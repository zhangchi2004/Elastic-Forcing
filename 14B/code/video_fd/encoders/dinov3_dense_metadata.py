"""Strict identity metadata for the fixed DINOv3 dense-video feature.

The DINOv3 feature used by this project is intentionally a single immutable
contract rather than a family of loosely compatible preprocessing choices.
One 81-frame video is sampled at sixteen fixed frame indices.  Every selected
frame is resized directly to 256 square pixels and contributes the Cartesian
product of eight fixed patch rows and columns.  The resulting frame-major
matrix has shape ``[1024, 1024]`` for the Meta DINOv3 ViT-L/16 backbone.

Offline reference builders and online training both use the helpers in this
module.  Keeping this metadata separate from the former V-JEPA2 contract makes
cross-encoder reference reuse fail loudly instead of silently changing the
MMD's feature identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import torch


FEATURE_METADATA_VERSION = 1
MODEL_ID = "dinov3_vitl16"
FEATURE_TYPE = "dinov3_dense_latent"
DEFAULT_WAN_VAE_ID = "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
SOURCE_COMMIT: Final[str] = "6876159a11b4df116f30f667f8c9888617df0751"

SOURCE_NUM_FRAMES = 81
FRAME_INDICES = (0, 5, 11, 16, 21, 27, 32, 37, 43, 48, 53, 59, 64, 69, 75, 80)
INPUT_SIZE = 256
PATCH_SIZE = 16
PATCH_GRID_SIZE = INPUT_SIZE // PATCH_SIZE
SPATIAL_AXIS_INDICES = (0, 2, 4, 6, 9, 11, 13, 15)
FORWARD_CHUNK_FRAMES = 4
FEATURE_DIM = 1024
STORAGE_TOKENS = 4


def spatial_patch_indices(
    *,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return the canonical 64 per-frame flattened patch indices.

    The row is the outer loop and the column is the inner loop, matching the
    native ViT patch flattening order and the final frame-major matrix order.
    """

    return torch.tensor(
        [row * PATCH_GRID_SIZE + column
         for row in SPATIAL_AXIS_INDICES
         for column in SPATIAL_AXIS_INDICES],
        dtype=torch.long,
        device=device,
    )


def dense_token_indices(
    *,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return canonical indices into the full ``16 * 256`` patch population."""

    per_frame = spatial_patch_indices(device=device)
    offsets = (
        torch.arange(len(FRAME_INDICES), dtype=torch.long, device=device)
        * (PATCH_GRID_SIZE * PATCH_GRID_SIZE)
    )
    return (offsets[:, None] + per_frame[None, :]).reshape(-1)


def integer_indices_sha256(values: torch.Tensor | Sequence[int]) -> str:
    """Hash one ordered integer index sequence using explicit little endian i64."""

    tensor = (
        values.detach()
        if isinstance(values, torch.Tensor)
        else torch.tensor(tuple(values))
    )
    if (
        tensor.dtype == torch.bool
        or tensor.is_floating_point()
        or tensor.is_complex()
    ):
        raise TypeError("index hashes require integers")
    tensor = tensor.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("index hashes require a non-empty one-dimensional sequence")
    payload = b"".join(
        int(value).to_bytes(8, byteorder="little", signed=True)
        for value in tensor.tolist()
    )
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | os.PathLike[str], *, chunk_bytes: int = 8 << 20) -> str:
    """Stream a local checkpoint hash without loading a second copy in memory."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"checkpoint is not a regular file: {resolved}")
    if int(chunk_bytes) <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            block = handle.read(int(chunk_bytes))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash an injected test model's state in a deterministic tensor format."""

    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"module state {name!r} is not a tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        # Viewing as bytes also supports bfloat16, which NumPy cannot always
        # expose directly.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    """Hash a JSON payload using the one canonical serialization."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_location(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if "\0" in value:
        raise ValueError(f"{name} must not contain NUL characters")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "file"}:
        return value
    return os.path.realpath(os.path.expanduser(value))


def _sha256_hex(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a hexadecimal string")
    normalized = value.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
    return normalized


def _strict_tuple(
    actual: Sequence[int],
    expected: tuple[int, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    if isinstance(actual, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence")
    values = tuple(actual)
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} must contain Python integers")
    if values != expected:
        raise ValueError(f"{name} must equal the canonical contract {list(expected)}")
    return values


def canonical_dinov3_dense_feature_metadata(
    *,
    repo_path: str,
    weight_path: str,
    source_commit: str,
    weight_sha256: str,
    source_num_frames: int = SOURCE_NUM_FRAMES,
    frame_indices: Sequence[int] = FRAME_INDICES,
    input_size: int = INPUT_SIZE,
    patch_size: int = PATCH_SIZE,
    patch_grid_size: int = PATCH_GRID_SIZE,
    spatial_axis_indices: Sequence[int] = SPATIAL_AXIS_INDICES,
    spatial_token_indices: Sequence[int] | torch.Tensor | None = None,
    per_frame_tokens: int = len(SPATIAL_AXIS_INDICES) ** 2,
    num_tokens: int = len(FRAME_INDICES) * len(SPATIAL_AXIS_INDICES) ** 2,
    forward_chunk_frames: int = FORWARD_CHUNK_FRAMES,
    feature_dim: int = FEATURE_DIM,
    model_id: str = MODEL_ID,
    vae_id: str = DEFAULT_WAN_VAE_ID,
) -> dict[str, object]:
    """Build and hash the canonical DINOv3 dense feature identity.

    Geometry arguments are exposed so callers can pass their live adapter
    values, but every one is checked against the immutable contract.  This is
    deliberate: changing any of them requires a new metadata version and new
    reference bank, not a permissive payload under the old identity.
    """

    if model_id != MODEL_ID:
        raise ValueError(f"model_id must be {MODEL_ID!r}")
    if type(source_num_frames) is not int or source_num_frames != SOURCE_NUM_FRAMES:
        raise ValueError(f"source_num_frames must be {SOURCE_NUM_FRAMES}")
    resolved_frames = _strict_tuple(frame_indices, FRAME_INDICES, name="frame_indices")
    if type(input_size) is not int or input_size != INPUT_SIZE:
        raise ValueError(f"input_size must be {INPUT_SIZE}")
    if type(patch_size) is not int or patch_size != PATCH_SIZE:
        raise ValueError(f"patch_size must be {PATCH_SIZE}")
    if type(patch_grid_size) is not int or patch_grid_size != PATCH_GRID_SIZE:
        raise ValueError(f"patch_grid_size must be {PATCH_GRID_SIZE}")
    resolved_axes = _strict_tuple(
        spatial_axis_indices,
        SPATIAL_AXIS_INDICES,
        name="spatial_axis_indices",
    )
    if type(forward_chunk_frames) is not int or forward_chunk_frames != FORWARD_CHUNK_FRAMES:
        raise ValueError(f"forward_chunk_frames must be {FORWARD_CHUNK_FRAMES}")
    if type(feature_dim) is not int or feature_dim != FEATURE_DIM:
        raise ValueError(f"feature_dim must be {FEATURE_DIM}")
    if source_commit != SOURCE_COMMIT:
        raise ValueError(
            "source_commit must equal the pinned DINOv3 source commit "
            f"{SOURCE_COMMIT}"
        )
    if not isinstance(vae_id, str) or not vae_id or "\0" in vae_id:
        raise ValueError("vae_id must be a non-empty string without NUL characters")

    per_frame = spatial_patch_indices()
    if spatial_token_indices is not None:
        supplied_tokens = (
            spatial_token_indices.detach()
            if isinstance(spatial_token_indices, torch.Tensor)
            else torch.tensor(tuple(spatial_token_indices))
        )
        if (
            supplied_tokens.dtype == torch.bool
            or supplied_tokens.is_floating_point()
            or supplied_tokens.is_complex()
        ):
            raise TypeError("spatial_token_indices must contain integers")
        supplied_tokens = supplied_tokens.detach().to(
            device="cpu", dtype=torch.long).contiguous()
        if not torch.equal(supplied_tokens, per_frame):
            raise ValueError(
                "spatial_token_indices must equal the canonical 64 patch indices"
            )
    across_video = dense_token_indices()
    matrix_rows = len(resolved_frames) * int(per_frame.numel())
    raw_tokens = len(resolved_frames) * PATCH_GRID_SIZE * PATCH_GRID_SIZE
    if type(per_frame_tokens) is not int or per_frame_tokens != int(per_frame.numel()):
        raise ValueError(f"per_frame_tokens must be {int(per_frame.numel())}")
    if type(num_tokens) is not int or num_tokens != matrix_rows:
        raise ValueError(f"num_tokens must be {matrix_rows}")
    metadata: dict[str, object] = {
        "feature_metadata_version": FEATURE_METADATA_VERSION,
        "feature_type": FEATURE_TYPE,
        "backend": "meta_native_pinned_repo_direct_import",
        "model_id": MODEL_ID,
        "repo_path": _canonical_location(repo_path, name="repo_path"),
        "weight_path": _canonical_location(weight_path, name="weight_path"),
        "source_commit": source_commit,
        "weight_sha256": _sha256_hex(weight_sha256, name="weight_sha256"),
        "vae_id": vae_id,
        "source_num_frames": SOURCE_NUM_FRAMES,
        "num_frames": len(resolved_frames),
        "frame_sampling": "fixed_uniform_linspace_round_81_to_16",
        "frame_indices": list(resolved_frames),
        "frame_indices_sha256": integer_indices_sha256(resolved_frames),
        "input_size": INPUT_SIZE,
        "resize_mode": "direct_square",
        "interpolation": "bilinear_antialias_align_corners_false",
        "normalization": "imagenet_default",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "patch_size": PATCH_SIZE,
        "patch_grid": [PATCH_GRID_SIZE, PATCH_GRID_SIZE],
        "prefix_tokens": 1 + STORAGE_TOKENS,
        "output_field": "x_norm_patchtokens",
        "final_norm": True,
        "raw_patch_tokens_per_frame": PATCH_GRID_SIZE * PATCH_GRID_SIZE,
        "spatial_axis_indices": list(resolved_axes),
        "spatial_patch_indices": [int(value) for value in per_frame.tolist()],
        "spatial_token_indices": [int(value) for value in per_frame.tolist()],
        "spatial_patch_indices_sha256": integer_indices_sha256(per_frame),
        "spatial_token_indices_sha256": integer_indices_sha256(per_frame),
        "spatial_sampling": "fixed_cartesian_rows_then_columns",
        "selected_tokens_per_frame": int(per_frame.numel()),
        "raw_tokens": raw_tokens,
        "num_tokens": matrix_rows,
        "token_sampling": "fixed_spatial_cartesian_per_fixed_frame",
        "token_order": "frame_major_then_patch_row_then_patch_column",
        "dense_token_indices_sha256": integer_indices_sha256(across_video),
        "forward_chunk_frames": FORWARD_CHUNK_FRAMES,
        "compute_dtype": "float32",
        "feature_dim": FEATURE_DIM,
        "sample_axis": "video",
        "raw_matrix_shape": [raw_tokens, FEATURE_DIM],
        "sampled_matrix_shape": [matrix_rows, FEATURE_DIM],
        "matrix_shape": [matrix_rows, FEATURE_DIM],
        "kernel": "single_scale_gaussian_rbf",
        "kernel_metric": "frobenius_rbf_aligned_tokens",
    }
    metadata["feature_contract_sha256"] = canonical_json_sha256(metadata)
    return metadata


__all__ = [
    "DEFAULT_WAN_VAE_ID",
    "FEATURE_DIM",
    "FEATURE_METADATA_VERSION",
    "FEATURE_TYPE",
    "FORWARD_CHUNK_FRAMES",
    "FRAME_INDICES",
    "INPUT_SIZE",
    "MODEL_ID",
    "PATCH_GRID_SIZE",
    "PATCH_SIZE",
    "SOURCE_NUM_FRAMES",
    "SOURCE_COMMIT",
    "SPATIAL_AXIS_INDICES",
    "canonical_dinov3_dense_feature_metadata",
    "canonical_json_sha256",
    "dense_token_indices",
    "file_sha256",
    "integer_indices_sha256",
    "module_state_sha256",
    "spatial_patch_indices",
]

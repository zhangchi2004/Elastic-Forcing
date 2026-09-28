"""Canonical geometry and identity helpers for V-JEPA2 dense video features."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Dict

import torch

from video_fd.encoders.vjepa2_temporal_sampling import WINDOW_SAMPLING_HASH


FEATURE_METADATA_VERSION = 1
EXPLICIT_TOKEN_FEATURE_METADATA_VERSION = 2
RANDOM_CONTIGUOUS_FEATURE_METADATA_VERSION = 3
LAST_CONTIGUOUS_FEATURE_METADATA_VERSION = 4
LAST_CONTIGUOUS_FRAME_SAMPLING = "last_contiguous"
DEFAULT_WAN_VAE_ID = "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
RANDOM_CONTIGUOUS_FRAME_SAMPLING = "random_contiguous_per_video"


def uniform_dense_token_indices(
    raw_tokens: int,
    num_tokens: int,
    *,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return the one canonical deterministic token subset."""
    raw_tokens = int(raw_tokens)
    num_tokens = int(num_tokens)
    if raw_tokens <= 0:
        raise ValueError("raw_tokens must be positive")
    if num_tokens <= 1 or num_tokens > raw_tokens:
        raise ValueError(
            f"num_tokens must be in [2,{raw_tokens}], got {num_tokens}")
    return torch.linspace(
        0,
        raw_tokens - 1,
        steps=num_tokens,
        device=device,
        dtype=torch.float64,
    ).round().to(torch.long)


def token_indices_sha256(indices: torch.Tensor) -> str:
    values = torch.as_tensor(
        indices, dtype=torch.int64, device="cpu").contiguous()
    if values.ndim != 1 or values.numel() <= 1:
        raise ValueError("token indices must be a non-empty 1-D sequence")
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def validate_explicit_dense_token_indices(
    indices: torch.Tensor | Sequence[int],
    *,
    raw_tokens: int,
    num_tokens: int,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Canonicalize an explicit aligned-token subset.

    Explicit subsets are deliberately stricter than the legacy uniform
    helper: the serialized order is the matrix-row alignment contract shared
    by student features and their frozen teacher reference.
    """
    raw_tokens = int(raw_tokens)
    num_tokens = int(num_tokens)
    if raw_tokens <= 0:
        raise ValueError("raw_tokens must be positive")
    if num_tokens <= 1 or num_tokens > raw_tokens:
        raise ValueError(
            f"num_tokens must be in [2,{raw_tokens}], got {num_tokens}")
    if isinstance(indices, torch.Tensor):
        if (
            indices.dtype == torch.bool
            or indices.is_floating_point()
            or indices.is_complex()
        ):
            raise TypeError("explicit token indices must be integers")
        values = indices.detach().to(
            device="cpu", dtype=torch.int64).contiguous()
    else:
        if any(type(value) is not int for value in indices):
            raise TypeError("explicit token indices must be integers")
        values = torch.tensor(
            indices, dtype=torch.int64, device="cpu").contiguous()
    if values.ndim != 1:
        raise ValueError("explicit token indices must be one-dimensional")
    if values.numel() != num_tokens:
        raise ValueError(
            "explicit token index count mismatch: "
            f"got {values.numel()}, expected {num_tokens}")
    if int(values[0]) < 0 or int(values[-1]) >= raw_tokens:
        raise ValueError(
            f"explicit token indices must be in [0,{raw_tokens})")
    if not bool(torch.all(values[1:] > values[:-1]).item()):
        raise ValueError(
            "explicit token indices must be strictly increasing and unique")
    return values.to(device=device)


def canonical_vjepa2_dense_feature_metadata(
    *,
    model_id: str,
    layer_index: int,
    num_frames: int,
    resize_short_side: int,
    crop_size: int,
    raw_tokens: int,
    num_tokens: int,
    feature_dim: int,
    vae_id: str = DEFAULT_WAN_VAE_ID,
    token_indices: torch.Tensor | Sequence[int] | None = None,
    frame_sampling: str = "center_contiguous",
    source_num_frames: int | None = None,
    window_seed: str | None = None,
) -> Dict[str, object]:
    """Build metadata shared by offline reference creation and training."""
    expanded_model_id = os.path.expanduser(str(model_id))
    canonical_model_id = (
        os.path.realpath(expanded_model_id)
        if os.path.exists(expanded_model_id)
        else str(model_id)
    )
    if frame_sampling == "center_contiguous" and (
        source_num_frames is not None or window_seed is not None
    ):
        raise ValueError(
            "source_num_frames and window_seed are only valid for "
            f"{RANDOM_CONTIGUOUS_FRAME_SAMPLING}"
        )

    if frame_sampling == "center_contiguous" and token_indices is None:
        # Keep this legacy payload byte-for-byte stable for existing reference
        # packs and strict checkpoint metadata comparisons.
        indices = uniform_dense_token_indices(raw_tokens, num_tokens)
        return {
            "feature_metadata_version": FEATURE_METADATA_VERSION,
            "feature_type": "vjepa2_dense_latent",
            "model_id": canonical_model_id,
            "vae_id": str(vae_id),
            "layer_index": int(layer_index),
            "prefix_blocks": int(layer_index) + 1,
            "num_frames": int(num_frames),
            "frame_sampling": "center_contiguous",
            "resize_short_side": int(resize_short_side),
            "crop_size": int(crop_size),
            "interpolation": "bilinear_antialias",
            "normalization": "imagenet_default",
            "raw_tokens": int(raw_tokens),
            "num_tokens": int(num_tokens),
            "token_sampling": "uniform_linspace_round",
            "token_alignment": "same_deterministic_indices",
            "token_indices_sha256": token_indices_sha256(indices),
            "feature_dim": int(feature_dim),
            "sample_axis": "video",
            "matrix_shape": [int(num_tokens), int(feature_dim)],
            "kernel": "single_scale_gaussian_rbf",
            "kernel_metric": "frobenius_rbf_aligned_tokens",
        }

    if frame_sampling == "center_contiguous":
        indices = validate_explicit_dense_token_indices(
            token_indices,
            raw_tokens=raw_tokens,
            num_tokens=num_tokens,
        )
        return {
            "feature_metadata_version": EXPLICIT_TOKEN_FEATURE_METADATA_VERSION,
            "feature_type": "vjepa2_dense_latent",
            "model_id": canonical_model_id,
            "vae_id": str(vae_id),
            "layer_index": int(layer_index),
            "prefix_blocks": int(layer_index) + 1,
            "num_frames": int(num_frames),
            "frame_sampling": "center_contiguous",
            "resize_short_side": int(resize_short_side),
            "crop_size": int(crop_size),
            "interpolation": "bilinear_antialias",
            "normalization": "imagenet_default",
            "raw_tokens": int(raw_tokens),
            "num_tokens": int(num_tokens),
            "token_sampling": "explicit_sorted_unique",
            "token_alignment": "same_explicit_indices_and_order",
            "token_indices": [int(value) for value in indices.tolist()],
            "token_indices_sha256": token_indices_sha256(indices),
            "feature_dim": int(feature_dim),
            "sample_axis": "video",
            "matrix_shape": [int(num_tokens), int(feature_dim)],
            "kernel": "single_scale_gaussian_rbf",
            "kernel_metric": "frobenius_rbf_aligned_tokens",
        }

    if frame_sampling == LAST_CONTIGUOUS_FRAME_SAMPLING:
        if type(source_num_frames) is not int:
            raise TypeError(
                "source_num_frames must be an integer for last_contiguous")
        if window_seed is not None:
            raise ValueError("last_contiguous does not use window_seed")
        if source_num_frames < int(num_frames):
            raise ValueError(
                "source_num_frames must be >= num_frames for last_contiguous")
        indices = (
            uniform_dense_token_indices(raw_tokens, num_tokens)
            if token_indices is None
            else validate_explicit_dense_token_indices(
                token_indices,
                raw_tokens=raw_tokens,
                num_tokens=num_tokens,
            )
        )
        start = int(source_num_frames) - int(num_frames)
        metadata = {
            "feature_metadata_version": LAST_CONTIGUOUS_FEATURE_METADATA_VERSION,
            "feature_type": "vjepa2_dense_latent",
            "model_id": canonical_model_id,
            "vae_id": str(vae_id),
            "layer_index": int(layer_index),
            "prefix_blocks": int(layer_index) + 1,
            "num_frames": int(num_frames),
            "frame_sampling": LAST_CONTIGUOUS_FRAME_SAMPLING,
            "source_num_frames": int(source_num_frames),
            "selected_frame_start": start,
            "selected_frame_end_inclusive": int(source_num_frames) - 1,
            "resize_short_side": int(resize_short_side),
            "crop_size": int(crop_size),
            "interpolation": "bilinear_antialias",
            "normalization": "imagenet_default",
            "raw_tokens": int(raw_tokens),
            "num_tokens": int(num_tokens),
            "token_sampling": (
                "uniform_linspace_round"
                if token_indices is None else "explicit_sorted_unique"),
            "token_alignment": (
                "same_deterministic_indices"
                if token_indices is None
                else "same_explicit_indices_and_order"),
            "token_indices_sha256": token_indices_sha256(indices),
            "feature_dim": int(feature_dim),
            "sample_axis": "video",
            "matrix_shape": [int(num_tokens), int(feature_dim)],
            "kernel": "single_scale_gaussian_rbf",
            "kernel_metric": "frobenius_rbf_aligned_tokens",
        }
        if token_indices is not None:
            metadata["token_indices"] = [
                int(value) for value in indices.tolist()
            ]
        return metadata

    if frame_sampling != RANDOM_CONTIGUOUS_FRAME_SAMPLING:
        raise ValueError(
            "frame_sampling must be center_contiguous, last_contiguous, or "
            f"{RANDOM_CONTIGUOUS_FRAME_SAMPLING!r}, got {frame_sampling!r}"
        )
    if type(source_num_frames) is not int:
        raise TypeError(
            "source_num_frames must be an integer for "
            f"{RANDOM_CONTIGUOUS_FRAME_SAMPLING}"
        )
    if source_num_frames < int(num_frames):
        raise ValueError(
            "source_num_frames must be >= num_frames, got "
            f"{source_num_frames} < {int(num_frames)}"
        )
    if not isinstance(window_seed, str) or not window_seed:
        raise ValueError(
            "window_seed must be a non-empty string for "
            f"{RANDOM_CONTIGUOUS_FRAME_SAMPLING}"
        )
    if "\0" in window_seed:
        raise ValueError("window_seed must not contain NUL characters")

    if token_indices is None:
        indices = uniform_dense_token_indices(raw_tokens, num_tokens)
        token_metadata = {
            "token_sampling": "uniform_linspace_round",
            "token_alignment": "same_deterministic_indices",
            "token_indices_sha256": token_indices_sha256(indices),
        }
    else:
        indices = validate_explicit_dense_token_indices(
            token_indices,
            raw_tokens=raw_tokens,
            num_tokens=num_tokens,
        )
        token_metadata = {
            "token_sampling": "explicit_sorted_unique",
            "token_alignment": "same_explicit_indices_and_order",
            "token_indices": [int(value) for value in indices.tolist()],
            "token_indices_sha256": token_indices_sha256(indices),
        }

    window_start_max = int(source_num_frames) - int(num_frames)
    return {
        "feature_metadata_version":
            RANDOM_CONTIGUOUS_FEATURE_METADATA_VERSION,
        "feature_type": "vjepa2_dense_latent",
        "model_id": canonical_model_id,
        "vae_id": str(vae_id),
        "layer_index": int(layer_index),
        "prefix_blocks": int(layer_index) + 1,
        "num_frames": int(num_frames),
        "frame_sampling": RANDOM_CONTIGUOUS_FRAME_SAMPLING,
        "source_num_frames": int(source_num_frames),
        "window_seed": window_seed,
        "window_start_min": 0,
        "window_start_max": window_start_max,
        "window_num_choices": window_start_max + 1,
        "window_sampling_hash": WINDOW_SAMPLING_HASH,
        "resize_short_side": int(resize_short_side),
        "crop_size": int(crop_size),
        "interpolation": "bilinear_antialias",
        "normalization": "imagenet_default",
        "raw_tokens": int(raw_tokens),
        "num_tokens": int(num_tokens),
        **token_metadata,
        "feature_dim": int(feature_dim),
        "sample_axis": "video",
        "matrix_shape": [int(num_tokens), int(feature_dim)],
        "kernel": "single_scale_gaussian_rbf",
        "kernel_metric": "frobenius_rbf_aligned_tokens",
    }

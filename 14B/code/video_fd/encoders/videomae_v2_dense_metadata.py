"""Canonical geometry and identity for five-segment VideoMAE V2 features.

An 81-frame Wan clip is partitioned by dropping frame zero and splitting the
remaining 80 frames into five contiguous 16-frame segments.  The same relative
VideoMAE token mask is applied independently in every segment and the selected
matrices are concatenated in segment-major order.  Hashes are domain-separated
from V-JEPA2 so the two feature families cannot share an identity accidentally.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Dict

import torch


FEATURE_METADATA_VERSION = 1
FEATURE_TYPE = "videomae_v2_five_segment_dense_latent"
DEFAULT_WAN_VAE_ID = "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
NUM_SEGMENTS = 5
FRAMES_PER_SEGMENT = 16
SOURCE_NUM_FRAMES = 81
SKIPPED_PREFIX_FRAMES = 1
RAW_TOKENS_PER_SEGMENT = 1568
FEATURE_DIM = 1024
CROP_SIZE = 224
RESIZE_SHORT_SIDE = 224
FRAME_SAMPLING = "skip_first_then_five_contiguous_segments"
SEGMENT_CONCAT_ORDER = "segment_major_then_relative_token_order"

_RELATIVE_TOKEN_HASH_DOMAIN = b"part-grad/videomae-v2/relative-token-mask/v1\0"
_AGGREGATE_TOKEN_HASH_DOMAIN = b"part-grad/videomae-v2/aggregate-token-order/v1\0"
_FEATURE_IDENTITY_HASH_DOMAIN = b"part-grad/videomae-v2/feature-metadata/v1\0"


def _validated_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _validated_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def uniform_segment_token_indices(
    raw_tokens_per_segment: int,
    num_tokens: int,
    *,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return the canonical uniformly spaced relative mask for one segment."""
    raw_tokens_per_segment = _validated_positive_int(
        raw_tokens_per_segment, "raw_tokens_per_segment"
    )
    num_tokens = _validated_positive_int(num_tokens, "num_tokens")
    if num_tokens <= 1 or num_tokens > raw_tokens_per_segment:
        raise ValueError(
            "num_tokens must be in "
            f"[2,{raw_tokens_per_segment}], got {num_tokens}"
        )
    return torch.linspace(
        0,
        raw_tokens_per_segment - 1,
        steps=num_tokens,
        device=device,
        dtype=torch.float64,
    ).round().to(torch.long)


def validate_segment_token_indices(
    indices: torch.Tensor | Sequence[int],
    *,
    raw_tokens_per_segment: int,
    num_tokens: int,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Validate the sorted relative token-row alignment contract."""
    raw_tokens_per_segment = _validated_positive_int(
        raw_tokens_per_segment, "raw_tokens_per_segment"
    )
    num_tokens = _validated_positive_int(num_tokens, "num_tokens")
    if num_tokens <= 1 or num_tokens > raw_tokens_per_segment:
        raise ValueError(
            "num_tokens must be in "
            f"[2,{raw_tokens_per_segment}], got {num_tokens}"
        )
    if isinstance(indices, torch.Tensor):
        if (
            indices.dtype == torch.bool
            or indices.is_floating_point()
            or indices.is_complex()
        ):
            raise TypeError("relative token indices must contain only integers")
        values = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    else:
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in indices
        ):
            raise TypeError("relative token indices must contain only integers")
        values = torch.tensor(indices, dtype=torch.int64, device="cpu")
    if values.ndim != 1:
        raise ValueError("relative token indices must be one-dimensional")
    if int(values.numel()) != num_tokens:
        raise ValueError(
            "relative token index count mismatch: "
            f"got {values.numel()}, expected {num_tokens}"
        )
    if int(values[0]) < 0 or int(values[-1]) >= raw_tokens_per_segment:
        raise ValueError(
            "relative token indices must be in "
            f"[0,{raw_tokens_per_segment})"
        )
    if not bool(torch.all(values[1:] > values[:-1]).item()):
        raise ValueError(
            "relative token indices must be strictly increasing and unique"
        )
    return values.to(device=device)


def _indices_sha256(domain: bytes, indices: torch.Tensor) -> str:
    values = torch.as_tensor(
        indices, dtype=torch.int64, device="cpu"
    ).contiguous()
    if values.ndim != 1 or values.numel() <= 1:
        raise ValueError("token indices must be a non-empty 1-D sequence")
    digest = hashlib.sha256(domain)
    for value in values.tolist():
        digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def _integer_index_values(
    indices: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        if (
            indices.dtype == torch.bool
            or indices.is_floating_point()
            or indices.is_complex()
        ):
            raise TypeError("token indices must contain only integers")
        values = indices.detach().to(
            device="cpu", dtype=torch.int64
        ).contiguous()
    else:
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in indices
        ):
            raise TypeError("token indices must contain only integers")
        values = torch.tensor(indices, dtype=torch.int64, device="cpu")
    if values.ndim != 1 or values.numel() <= 1:
        raise ValueError("token indices must contain at least two rows")
    return values


def relative_token_indices_sha256(indices: torch.Tensor | Sequence[int]) -> str:
    """Hash a one-segment mask in the VideoMAE-specific identity domain."""
    values = _integer_index_values(indices)
    return _indices_sha256(_RELATIVE_TOKEN_HASH_DOMAIN, values)


def repeat_segment_token_indices(
    indices: torch.Tensor | Sequence[int],
    *,
    num_segments: int = NUM_SEGMENTS,
    raw_tokens_per_segment: int = RAW_TOKENS_PER_SEGMENT,
    global_offsets: bool = False,
) -> torch.Tensor:
    """Repeat a relative mask into a ``[segments,K]`` alignment table.

    With ``global_offsets=False`` every row is identical.  With
    ``global_offsets=True`` row ``s`` is offset by
    ``s * raw_tokens_per_segment`` and therefore indexes the conceptual full
    unselected ``[segments * raw_tokens_per_segment, D]`` matrix.
    """
    num_segments = _validated_positive_int(num_segments, "num_segments")
    raw_tokens_per_segment = _validated_positive_int(
        raw_tokens_per_segment, "raw_tokens_per_segment"
    )
    target_device = indices.device if isinstance(indices, torch.Tensor) else None
    values = _integer_index_values(indices)
    values = validate_segment_token_indices(
        values,
        raw_tokens_per_segment=raw_tokens_per_segment,
        num_tokens=int(values.numel()),
        device=target_device,
    )
    repeated = values.unsqueeze(0).repeat(num_segments, 1)
    if global_offsets:
        offsets = torch.arange(
            num_segments, dtype=torch.int64, device=repeated.device
        ).unsqueeze(1)
        repeated = repeated + offsets * raw_tokens_per_segment
    return repeated


def aggregate_segment_token_indices(
    indices: torch.Tensor | Sequence[int],
    *,
    num_segments: int = NUM_SEGMENTS,
    raw_tokens_per_segment: int = RAW_TOKENS_PER_SEGMENT,
) -> torch.Tensor:
    """Return the flattened segment-major global token-row order."""
    return repeat_segment_token_indices(
        indices,
        num_segments=num_segments,
        raw_tokens_per_segment=raw_tokens_per_segment,
        global_offsets=True,
    ).reshape(-1)


def aggregate_token_indices_sha256(
    indices: torch.Tensor | Sequence[int],
    *,
    num_segments: int = NUM_SEGMENTS,
    raw_tokens_per_segment: int = RAW_TOKENS_PER_SEGMENT,
) -> str:
    aggregate = aggregate_segment_token_indices(
        indices,
        num_segments=num_segments,
        raw_tokens_per_segment=raw_tokens_per_segment,
    )
    return _indices_sha256(_AGGREGATE_TOKEN_HASH_DOMAIN, aggregate)


def _canonical_model_id(model_id: str) -> str:
    expanded = os.path.expanduser(str(model_id))
    resolved = (
        os.path.realpath(expanded) if os.path.exists(expanded) else expanded)
    local_root = "/path/to/local-assets"
    logical_root = "/path/to/elastic-forcing-assets"
    if (
        resolved == local_root
        or resolved.startswith(local_root + "/")
    ):
        return logical_root + resolved[len(local_root):]
    return resolved


def _segment_frame_ranges(
    *,
    num_segments: int = NUM_SEGMENTS,
    frames_per_segment: int = FRAMES_PER_SEGMENT,
    skipped_prefix_frames: int = SKIPPED_PREFIX_FRAMES,
) -> list[list[int]]:
    return [
        [
            skipped_prefix_frames + segment * frames_per_segment,
            skipped_prefix_frames + (segment + 1) * frames_per_segment,
        ]
        for segment in range(num_segments)
    ]


def _feature_identity_sha256(metadata_without_identity: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(metadata_without_identity),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(_FEATURE_IDENTITY_HASH_DOMAIN + payload).hexdigest()


def canonical_videomae_v2_dense_feature_metadata(
    *,
    model_id: str,
    layer_index: int,
    model_id_is_reference: bool = False,
    num_tokens: int,
    token_indices: torch.Tensor | Sequence[int] | None = None,
    vae_id: str = DEFAULT_WAN_VAE_ID,
    num_segments: int = NUM_SEGMENTS,
    frames_per_segment: int = FRAMES_PER_SEGMENT,
    raw_tokens_per_segment: int = RAW_TOKENS_PER_SEGMENT,
    feature_dim: int = FEATURE_DIM,
    resize_short_side: int = RESIZE_SHORT_SIDE,
    crop_size: int = CROP_SIZE,
    skip_first_frame: bool = True,
) -> Dict[str, object]:
    """Build the strict five-segment feature/reference identity payload.

    A validated frozen reference identity is opaque: it must not trigger a
    filesystem lookup of the machine on which the features were extracted.
    """
    layer_index = _validated_nonnegative_int(layer_index, "layer_index")
    num_tokens = _validated_positive_int(num_tokens, "num_tokens")
    num_segments = _validated_positive_int(num_segments, "num_segments")
    frames_per_segment = _validated_positive_int(
        frames_per_segment, "frames_per_segment"
    )
    raw_tokens_per_segment = _validated_positive_int(
        raw_tokens_per_segment, "raw_tokens_per_segment"
    )
    feature_dim = _validated_positive_int(feature_dim, "feature_dim")
    resize_short_side = _validated_positive_int(
        resize_short_side, "resize_short_side"
    )
    crop_size = _validated_positive_int(crop_size, "crop_size")
    if num_segments != NUM_SEGMENTS:
        raise ValueError(f"num_segments must be {NUM_SEGMENTS}, got {num_segments}")
    if frames_per_segment != FRAMES_PER_SEGMENT:
        raise ValueError(
            f"frames_per_segment must be {FRAMES_PER_SEGMENT}, got {frames_per_segment}"
        )
    if raw_tokens_per_segment != RAW_TOKENS_PER_SEGMENT:
        raise ValueError(
            "raw_tokens_per_segment must be "
            f"{RAW_TOKENS_PER_SEGMENT}, got {raw_tokens_per_segment}"
        )
    if feature_dim != FEATURE_DIM:
        raise ValueError(f"feature_dim must be {FEATURE_DIM}, got {feature_dim}")
    if resize_short_side != RESIZE_SHORT_SIDE or crop_size != CROP_SIZE:
        raise ValueError(
            "VideoMAE V2 preprocessing must use resize_short_side=224 and crop_size=224"
        )
    if skip_first_frame is not True:
        raise ValueError("skip_first_frame must be True for the five-segment contract")

    if token_indices is None:
        relative_indices = uniform_segment_token_indices(
            raw_tokens_per_segment, num_tokens
        )
        token_sampling = "uniform_linspace_round"
    else:
        relative_indices = validate_segment_token_indices(
            token_indices,
            raw_tokens_per_segment=raw_tokens_per_segment,
            num_tokens=num_tokens,
        )
        token_sampling = "explicit_sorted_unique"

    selected_tokens = num_segments * num_tokens
    raw_tokens = num_segments * raw_tokens_per_segment
    processed_num_frames = num_segments * frames_per_segment
    source_num_frames = processed_num_frames + SKIPPED_PREFIX_FRAMES
    metadata: Dict[str, object] = {
        "feature_metadata_version": FEATURE_METADATA_VERSION,
        "feature_type": FEATURE_TYPE,
        "model_family": "OpenGVLab VideoMAE V2",
        "model_id": str(model_id) if model_id_is_reference else _canonical_model_id(model_id),
        "vae_id": str(vae_id),
        "layer_index": layer_index,
        "prefix_blocks": layer_index + 1,
        "layer_index_semantics": "zero_based_complete_post_residual_block",
        "feature_stage": "post_residual_pre_pool_fc_norm",
        "num_segments": num_segments,
        "frames_per_segment": frames_per_segment,
        "num_frames": processed_num_frames,
        "source_num_frames": source_num_frames,
        "skip_first_frame": True,
        "skipped_prefix_frames": SKIPPED_PREFIX_FRAMES,
        "frame_sampling": FRAME_SAMPLING,
        "segment_frame_ranges": _segment_frame_ranges(),
        "resize_short_side": resize_short_side,
        "crop_size": crop_size,
        "interpolation": "bilinear_antialias",
        "normalization": "imagenet_default",
        "encoder_input_layout": "BCTHW",
        "raw_tokens_per_segment": raw_tokens_per_segment,
        "raw_tokens": raw_tokens,
        "num_tokens_per_segment": num_tokens,
        "num_tokens": selected_tokens,
        "token_sampling": token_sampling,
        "token_mask_reuse": "same_relative_indices_every_segment",
        "token_alignment": "same_relative_indices_and_order",
        "segment_concat_order": SEGMENT_CONCAT_ORDER,
        "relative_token_indices": [
            int(value) for value in relative_indices.tolist()
        ],
        "relative_token_indices_sha256": relative_token_indices_sha256(
            relative_indices
        ),
        "aggregate_token_indices_sha256": aggregate_token_indices_sha256(
            relative_indices,
            num_segments=num_segments,
            raw_tokens_per_segment=raw_tokens_per_segment,
        ),
        "feature_dim": feature_dim,
        "sample_axis": "video",
        "matrix_shape": [selected_tokens, feature_dim],
        "kernel": "single_scale_gaussian_rbf",
        "kernel_metric": "frobenius_rbf_aligned_tokens",
    }
    metadata["feature_identity_sha256"] = _feature_identity_sha256(metadata)
    return metadata


def validate_videomae_v2_dense_feature_metadata(
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    """Validate a serialized reference identity and reject other families.

    A shallow canonical copy is returned so callers can retain their usual
    ``validated = validate_...(loaded_metadata)`` pattern.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("VideoMAE V2 feature metadata must be a mapping")
    result = dict(metadata)
    if result.get("feature_type") != FEATURE_TYPE:
        raise ValueError(
            "feature_type mismatch: expected "
            f"{FEATURE_TYPE!r}, got {result.get('feature_type')!r}; "
            "V-JEPA2 and VideoMAE V2 reference packs are not interchangeable"
        )

    required_exact = {
        "feature_metadata_version": FEATURE_METADATA_VERSION,
        "model_family": "OpenGVLab VideoMAE V2",
        "feature_stage": "post_residual_pre_pool_fc_norm",
        "layer_index_semantics": "zero_based_complete_post_residual_block",
        "num_segments": NUM_SEGMENTS,
        "frames_per_segment": FRAMES_PER_SEGMENT,
        "num_frames": NUM_SEGMENTS * FRAMES_PER_SEGMENT,
        "source_num_frames": SOURCE_NUM_FRAMES,
        "skip_first_frame": True,
        "skipped_prefix_frames": SKIPPED_PREFIX_FRAMES,
        "frame_sampling": FRAME_SAMPLING,
        "segment_frame_ranges": _segment_frame_ranges(),
        "resize_short_side": RESIZE_SHORT_SIDE,
        "crop_size": CROP_SIZE,
        "interpolation": "bilinear_antialias",
        "normalization": "imagenet_default",
        "encoder_input_layout": "BCTHW",
        "raw_tokens_per_segment": RAW_TOKENS_PER_SEGMENT,
        "raw_tokens": NUM_SEGMENTS * RAW_TOKENS_PER_SEGMENT,
        "token_mask_reuse": "same_relative_indices_every_segment",
        "token_alignment": "same_relative_indices_and_order",
        "segment_concat_order": SEGMENT_CONCAT_ORDER,
        "feature_dim": FEATURE_DIM,
        "sample_axis": "video",
        "kernel": "single_scale_gaussian_rbf",
        "kernel_metric": "frobenius_rbf_aligned_tokens",
    }
    for key, expected in required_exact.items():
        if result.get(key) != expected:
            raise ValueError(
                f"invalid VideoMAE V2 metadata field {key!r}: "
                f"expected {expected!r}, got {result.get(key)!r}"
            )

    layer_index = result.get("layer_index")
    if isinstance(layer_index, bool) or not isinstance(layer_index, Integral):
        raise ValueError("layer_index must be a non-negative integer")
    if int(layer_index) < 0 or result.get("prefix_blocks") != int(layer_index) + 1:
        raise ValueError("prefix_blocks must equal layer_index + 1")
    num_tokens_per_segment = result.get("num_tokens_per_segment")
    if (
        isinstance(num_tokens_per_segment, bool)
        or not isinstance(num_tokens_per_segment, Integral)
    ):
        raise ValueError("num_tokens_per_segment must be an integer")
    num_tokens_per_segment = int(num_tokens_per_segment)
    if not 2 <= num_tokens_per_segment <= RAW_TOKENS_PER_SEGMENT:
        raise ValueError(
            "num_tokens_per_segment must be in "
            f"[2,{RAW_TOKENS_PER_SEGMENT}]"
        )
    selected_tokens = NUM_SEGMENTS * num_tokens_per_segment
    if result.get("num_tokens") != selected_tokens:
        raise ValueError("num_tokens must equal num_segments * num_tokens_per_segment")
    if result.get("matrix_shape") != [selected_tokens, FEATURE_DIM]:
        raise ValueError(
            f"matrix_shape must be [{selected_tokens},{FEATURE_DIM}]"
        )

    relative_indices = validate_segment_token_indices(
        result.get("relative_token_indices", []),
        raw_tokens_per_segment=RAW_TOKENS_PER_SEGMENT,
        num_tokens=num_tokens_per_segment,
    )
    expected_relative_hash = relative_token_indices_sha256(relative_indices)
    if result.get("relative_token_indices_sha256") != expected_relative_hash:
        raise ValueError("relative_token_indices_sha256 does not match the mask")
    expected_aggregate_hash = aggregate_token_indices_sha256(relative_indices)
    if result.get("aggregate_token_indices_sha256") != expected_aggregate_hash:
        raise ValueError("aggregate_token_indices_sha256 does not match segment order")
    if result.get("token_sampling") not in (
        "uniform_linspace_round",
        "explicit_sorted_unique",
    ):
        raise ValueError("unsupported token_sampling in VideoMAE V2 metadata")

    identity = result.pop("feature_identity_sha256", None)
    expected_identity = _feature_identity_sha256(result)
    result["feature_identity_sha256"] = identity
    if identity != expected_identity:
        raise ValueError("feature_identity_sha256 does not match metadata contents")
    return result


# Familiar wrappers for call sites that use the V-JEPA helper vocabulary.  The
# hashes remain VideoMAE-domain-separated despite the compatibility names.
uniform_dense_token_indices = uniform_segment_token_indices
token_indices_sha256 = relative_token_indices_sha256


def validate_explicit_dense_token_indices(
    indices: torch.Tensor | Sequence[int],
    *,
    raw_tokens: int,
    num_tokens: int,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    return validate_segment_token_indices(
        indices,
        raw_tokens_per_segment=raw_tokens,
        num_tokens=num_tokens,
        device=device,
    )


__all__ = [
    "FEATURE_METADATA_VERSION",
    "FEATURE_TYPE",
    "DEFAULT_WAN_VAE_ID",
    "NUM_SEGMENTS",
    "FRAMES_PER_SEGMENT",
    "SOURCE_NUM_FRAMES",
    "SKIPPED_PREFIX_FRAMES",
    "RAW_TOKENS_PER_SEGMENT",
    "FEATURE_DIM",
    "CROP_SIZE",
    "RESIZE_SHORT_SIDE",
    "uniform_segment_token_indices",
    "validate_segment_token_indices",
    "relative_token_indices_sha256",
    "repeat_segment_token_indices",
    "aggregate_segment_token_indices",
    "aggregate_token_indices_sha256",
    "canonical_videomae_v2_dense_feature_metadata",
    "validate_videomae_v2_dense_feature_metadata",
    "uniform_dense_token_indices",
    "validate_explicit_dense_token_indices",
    "token_indices_sha256",
]

"""Identity metadata for the 20-frame DINOv3 temporal-pool feature.

Twenty uniformly sampled frames are encoded independently with the frozen
DINOv3 ViT-L/16.  The complete 16x16 final-normalized patch grid is retained
for every frame, then each consecutive group of five frame features is
averaged in FP32.  The four pooled grids are flattened group-major to one
``[1024, 1024]`` matrix per video.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import torch

from video_fd.encoders.dinov3_dense_metadata import (
    DEFAULT_WAN_VAE_ID,
    FEATURE_DIM,
    FEATURE_TYPE,
    INPUT_SIZE,
    MODEL_ID,
    PATCH_GRID_SIZE,
    PATCH_SIZE,
    SOURCE_COMMIT,
    SOURCE_NUM_FRAMES,
    canonical_dinov3_dense_feature_metadata,
    canonical_json_sha256,
    file_sha256,
    integer_indices_sha256,
    module_state_sha256,
)


FEATURE_METADATA_VERSION = 2
GEOMETRY: Final[str] = "temporal_mean5_full256"
FRAME_INDICES = (
    0, 4, 8, 13, 17,
    21, 25, 29, 34, 38,
    42, 46, 51, 55, 59,
    63, 67, 72, 76, 80,
)
SPATIAL_AXIS_INDICES = tuple(range(PATCH_GRID_SIZE))
SPATIAL_PATCH_INDICES = tuple(range(PATCH_GRID_SIZE * PATCH_GRID_SIZE))
TEMPORAL_GROUPS = 4
TEMPORAL_POOL_FRAMES = 5
FORWARD_CHUNK_FRAMES = 4
PER_FRAME_TOKENS = PATCH_GRID_SIZE * PATCH_GRID_SIZE
RAW_TOKENS = len(FRAME_INDICES) * PER_FRAME_TOKENS
NUM_TOKENS = TEMPORAL_GROUPS * PER_FRAME_TOKENS


def _strict_integer_tuple(
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
        raise ValueError(f"{name} must equal {list(expected)}")
    return values


def full_patch_indices(
    *, device: torch.device | str | int | None = None,
) -> torch.Tensor:
    return torch.arange(PER_FRAME_TOKENS, dtype=torch.long, device=device)


def temporal_group_frame_indices() -> tuple[tuple[int, ...], ...]:
    return tuple(
        FRAME_INDICES[start:start + TEMPORAL_POOL_FRAMES]
        for start in range(0, len(FRAME_INDICES), TEMPORAL_POOL_FRAMES)
    )


def canonical_dinov3_temporal_pool_feature_metadata(
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
    geometry: str = GEOMETRY,
    temporal_groups: int = TEMPORAL_GROUPS,
    temporal_pool_frames: int = TEMPORAL_POOL_FRAMES,
    per_frame_tokens: int = PER_FRAME_TOKENS,
    num_tokens: int = NUM_TOKENS,
    forward_chunk_frames: int = FORWARD_CHUNK_FRAMES,
    feature_dim: int = FEATURE_DIM,
    model_id: str = MODEL_ID,
    vae_id: str = DEFAULT_WAN_VAE_ID,
) -> dict[str, object]:
    """Return the strict online/offline identity for this feature geometry."""

    resolved_frames = _strict_integer_tuple(
        frame_indices, FRAME_INDICES, name="frame_indices")
    resolved_axes = _strict_integer_tuple(
        spatial_axis_indices, SPATIAL_AXIS_INDICES,
        name="spatial_axis_indices")
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
        if not torch.equal(supplied_tokens, full_patch_indices()):
            raise ValueError(
                "spatial_token_indices must contain all 256 patch indices")
    exact_values = {
        "source_num_frames": (source_num_frames, SOURCE_NUM_FRAMES),
        "input_size": (input_size, INPUT_SIZE),
        "patch_size": (patch_size, PATCH_SIZE),
        "patch_grid_size": (patch_grid_size, PATCH_GRID_SIZE),
        "temporal_groups": (temporal_groups, TEMPORAL_GROUPS),
        "temporal_pool_frames": (temporal_pool_frames, TEMPORAL_POOL_FRAMES),
        "per_frame_tokens": (per_frame_tokens, PER_FRAME_TOKENS),
        "num_tokens": (num_tokens, NUM_TOKENS),
        "forward_chunk_frames": (forward_chunk_frames, FORWARD_CHUNK_FRAMES),
        "feature_dim": (feature_dim, FEATURE_DIM),
    }
    for name, (actual, expected) in exact_values.items():
        if type(actual) is not int or actual != expected:
            raise ValueError(f"{name} must be {expected}")
    if geometry != GEOMETRY:
        raise ValueError(f"geometry must be {GEOMETRY!r}")
    if model_id != MODEL_ID:
        raise ValueError(f"model_id must be {MODEL_ID!r}")

    # Reuse the original contract builder for checkpoint/source/path and
    # preprocessing validation, then replace every geometry-dependent field.
    metadata = canonical_dinov3_dense_feature_metadata(
        repo_path=repo_path,
        weight_path=weight_path,
        source_commit=source_commit,
        weight_sha256=weight_sha256,
        model_id=model_id,
        vae_id=vae_id,
    )
    patches = full_patch_indices()
    groups = temporal_group_frame_indices()
    metadata.pop("feature_contract_sha256", None)
    metadata.pop("dense_token_indices_sha256", None)
    metadata.update({
        "feature_metadata_version": FEATURE_METADATA_VERSION,
        "geometry": GEOMETRY,
        "source_num_frames": SOURCE_NUM_FRAMES,
        "num_frames": len(resolved_frames),
        "frame_sampling": "fixed_uniform_linspace_round_81_to_20",
        "frame_indices": list(resolved_frames),
        "frame_indices_sha256": integer_indices_sha256(resolved_frames),
        "spatial_axis_indices": list(resolved_axes),
        "spatial_patch_indices": patches.tolist(),
        "spatial_token_indices": patches.tolist(),
        "spatial_patch_indices_sha256": integer_indices_sha256(patches),
        "spatial_token_indices_sha256": integer_indices_sha256(patches),
        "spatial_sampling": "all_patch_tokens_row_major",
        "selected_tokens_per_frame": PER_FRAME_TOKENS,
        "raw_tokens": RAW_TOKENS,
        "num_tokens": NUM_TOKENS,
        "token_sampling": "all_patches_then_temporal_feature_mean",
        "token_order": (
            "temporal_group_major_then_patch_row_then_patch_column"),
        "temporal_groups": TEMPORAL_GROUPS,
        "temporal_pool_frames": TEMPORAL_POOL_FRAMES,
        "temporal_pool_size": TEMPORAL_POOL_FRAMES,
        "temporal_pool": "fp32_mean_over_each_consecutive_5_frames",
        "temporal_group_frame_indices": [list(group) for group in groups],
        "temporal_group_frame_indices_sha256": [
            integer_indices_sha256(group) for group in groups
        ],
        "tokens_per_group": PER_FRAME_TOKENS,
        "forward_chunk_frames": FORWARD_CHUNK_FRAMES,
        "raw_matrix_shape": [RAW_TOKENS, FEATURE_DIM],
        "sampled_matrix_shape": [NUM_TOKENS, FEATURE_DIM],
        "matrix_shape": [NUM_TOKENS, FEATURE_DIM],
    })
    metadata["feature_contract_sha256"] = canonical_json_sha256(metadata)
    return metadata


__all__ = [
    "FEATURE_DIM",
    "FEATURE_TYPE",
    "FORWARD_CHUNK_FRAMES",
    "FRAME_INDICES",
    "GEOMETRY",
    "INPUT_SIZE",
    "MODEL_ID",
    "NUM_TOKENS",
    "PATCH_GRID_SIZE",
    "PATCH_SIZE",
    "PER_FRAME_TOKENS",
    "RAW_TOKENS",
    "SOURCE_COMMIT",
    "SOURCE_NUM_FRAMES",
    "SPATIAL_AXIS_INDICES",
    "TEMPORAL_GROUPS",
    "TEMPORAL_POOL_FRAMES",
    "canonical_dinov3_temporal_pool_feature_metadata",
    "file_sha256",
    "full_patch_indices",
    "module_state_sha256",
    "temporal_group_frame_indices",
]

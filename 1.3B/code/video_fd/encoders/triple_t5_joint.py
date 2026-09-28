"""Canonical three-video-encoder + UMT5 joint feature construction.

This module is model-free and deterministic so offline reference construction
and online training share exactly the same row geometry. Selected UMT5 tokens
are reshaped from 4096 to four 1024-dimensional rows in row-major order and
appended after video rows, without projection, normalization, or scaling.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch


JOINT_FEATURE_METADATA_VERSION: Final[int] = 1
FEATURE_TYPE: Final[str] = (
    "triple_dinov3_vjepa2_videomae_v2_t5_joint_dense_latent"
)
VIDEO_FEATURE_DIM: Final[int] = 1024
TEXT_RAW_TOKENS: Final[int] = 512
TEXT_FEATURE_DIM: Final[int] = 4096
TEXT_ROWS_PER_TOKEN: Final[int] = TEXT_FEATURE_DIM // VIDEO_FEATURE_DIM
TEXT_TOKEN_SAMPLING: Final[str] = "uniform_valid_linspace_round_zero_pad"
JOINT_LAYOUT: Final[str] = "reshape_4096_to_4x1024_then_row_concat"


@dataclass(frozen=True)
class JointBranchSpec:
    """Immutable row and identity contract for one video branch."""

    branch_name: str
    video_rows: int
    text_tokens: int
    video_feature_type: str
    encoder_digest_key: str

    @property
    def text_rows(self) -> int:
        return self.text_tokens * TEXT_ROWS_PER_TOKEN

    @property
    def joint_rows(self) -> int:
        return self.video_rows + self.text_rows

    @property
    def text_flat_fraction(self) -> float:
        return self.text_rows / self.joint_rows


BRANCH_SPECS: Final[dict[str, JointBranchSpec]] = {
    "dinov3": JointBranchSpec(
        "dinov3", 1024, 85, "dinov3_dense_latent", "weight_sha256"
    ),
    "vjepa2": JointBranchSpec(
        "vjepa2", 1024, 85, "vjepa2_dense_latent", "model_sha256"
    ),
    "videomae_v2": JointBranchSpec(
        "videomae_v2",
        1280,
        107,
        "videomae_v2_five_segment_dense_latent",
        "download_manifest_content_hash",
    ),
}


def _branch_spec(branch_name: str) -> JointBranchSpec:
    if not isinstance(branch_name, str):
        raise TypeError("branch_name must be a string")
    try:
        return BRANCH_SPECS[branch_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported joint branch {branch_name!r}; expected one of "
            f"{sorted(BRANCH_SPECS)}"
        ) from exc


def _require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != rank:
        raise ValueError(
            f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}"
        )


def _validate_sha256_digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _canonical_path_or_id(value: str | os.PathLike[str], name: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{name} must be a string or path-like value")
    expanded = os.path.expanduser(os.fspath(value))
    if not expanded or "\0" in expanded:
        raise ValueError(f"{name} must be non-empty and contain no NUL")
    return os.path.realpath(expanded) if os.path.exists(expanded) else expanded


def _json_safe(value: Any, *, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        return _json_safe(tensor.tolist(), path=path)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _json_safe(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_uniform_valid_text_tokens(
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Uniformly select compact valid rows; retain-once and pad if short."""
    _require_rank("prompt_embeds", prompt_embeds, 3)
    _require_rank("attention_mask", attention_mask, 2)
    if type(num_tokens) is not int:
        raise TypeError("num_tokens must be an integer")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if prompt_embeds.shape[:2] != attention_mask.shape:
        raise ValueError(
            "attention_mask must match the first two prompt_embeds "
            f"dimensions: got {tuple(attention_mask.shape)} versus "
            f"{tuple(prompt_embeds.shape[:2])}"
        )
    if attention_mask.device != prompt_embeds.device:
        raise ValueError(
            "attention_mask and prompt_embeds must be on the same device"
        )

    batch_size, _, feature_dim = prompt_embeds.shape
    selected = prompt_embeds.new_zeros(batch_size, num_tokens, feature_dim)
    valid_mask = attention_mask.gt(0)
    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(
            valid_mask[batch_index], as_tuple=False
        ).flatten()
        valid_count = int(valid_indices.numel())
        if valid_count == 0:
            continue
        if valid_count > num_tokens:
            compact_positions = torch.linspace(
                0,
                valid_count - 1,
                steps=num_tokens,
                dtype=torch.float64,
                device=prompt_embeds.device,
            ).round().to(torch.long)
            valid_indices = valid_indices.index_select(0, compact_positions)
            selected[batch_index] = prompt_embeds[batch_index].index_select(
                0, valid_indices
            )
        else:
            selected[batch_index, :valid_count] = (
                prompt_embeds[batch_index].index_select(0, valid_indices)
            )
    return selected


def fuse_video_t5_features(
    video_features: torch.Tensor,
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    branch_name: str,
) -> torch.Tensor:
    """Append the branch's canonical raw UMT5 rows after video features."""
    spec = _branch_spec(branch_name)
    _require_rank("video_features", video_features, 3)
    _require_rank("prompt_embeds", prompt_embeds, 3)
    _require_rank("attention_mask", attention_mask, 2)
    if video_features.shape[1:] != (spec.video_rows, VIDEO_FEATURE_DIM):
        raise ValueError(
            f"{branch_name} video_features must have shape "
            f"[B,{spec.video_rows},{VIDEO_FEATURE_DIM}], got "
            f"{tuple(video_features.shape)}"
        )
    if prompt_embeds.shape[0] != video_features.shape[0]:
        raise ValueError(
            "prompt_embeds and video_features must have the same batch size"
        )
    if prompt_embeds.shape[-1] != TEXT_FEATURE_DIM:
        raise ValueError(
            f"prompt_embeds feature dimension must be {TEXT_FEATURE_DIM}, "
            f"got {prompt_embeds.shape[-1]}"
        )
    if prompt_embeds.device != video_features.device:
        raise ValueError(
            "prompt_embeds and video_features must be on the same device"
        )

    selected_text = select_uniform_valid_text_tokens(
        prompt_embeds, attention_mask, num_tokens=spec.text_tokens
    )
    text_rows = selected_text.reshape(
        video_features.shape[0], spec.text_rows, VIDEO_FEATURE_DIM
    )
    if text_rows.dtype != video_features.dtype:
        text_rows = text_rows.to(dtype=video_features.dtype)
    return torch.cat((video_features, text_rows), dim=1)


def split_video_t5_joint_features(
    joint_features: torch.Tensor,
    *,
    branch_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a joint matrix into raw video rows and selected UMT5 tokens."""
    spec = _branch_spec(branch_name)
    _require_rank("joint_features", joint_features, 3)
    if joint_features.shape[1:] != (spec.joint_rows, VIDEO_FEATURE_DIM):
        raise ValueError(
            f"{branch_name} joint_features must have shape "
            f"[B,{spec.joint_rows},{VIDEO_FEATURE_DIM}], got "
            f"{tuple(joint_features.shape)}"
        )
    video_features = joint_features[:, : spec.video_rows]
    text_features = joint_features[:, spec.video_rows :].reshape(
        joint_features.shape[0], spec.text_tokens, TEXT_FEATURE_DIM
    )
    return video_features, text_features


def canonical_triple_t5_joint_feature_metadata(
    *,
    branch_name: str,
    video_feature_metadata: Mapping[str, object],
    branch_encoder_identity: Mapping[str, object],
    t5_model_id: str | os.PathLike[str],
    tokenizer_id: str | os.PathLike[str],
    t5_weights_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, object]:
    """Build the strict JSON-safe feature identity for one joint branch."""
    spec = _branch_spec(branch_name)
    if not isinstance(video_feature_metadata, Mapping):
        raise TypeError("video_feature_metadata must be a mapping")
    if not isinstance(branch_encoder_identity, Mapping):
        raise TypeError("branch_encoder_identity must be a mapping")
    video_metadata = _json_safe(
        video_feature_metadata, path="video_feature_metadata"
    )
    encoder_identity = _json_safe(
        branch_encoder_identity, path="branch_encoder_identity"
    )
    if video_metadata.get("feature_type") != spec.video_feature_type:
        raise ValueError(
            f"{branch_name} video feature_type must be "
            f"{spec.video_feature_type!r}"
        )
    if video_metadata.get("num_tokens") != spec.video_rows:
        raise ValueError(
            f"{branch_name} video metadata must contain "
            f"num_tokens={spec.video_rows}"
        )
    if video_metadata.get("feature_dim") != VIDEO_FEATURE_DIM:
        raise ValueError(
            f"{branch_name} video metadata must contain "
            f"feature_dim={VIDEO_FEATURE_DIM}"
        )
    if video_metadata.get("matrix_shape") != [
        spec.video_rows,
        VIDEO_FEATURE_DIM,
    ]:
        raise ValueError(
            f"{branch_name} video metadata matrix_shape must be "
            f"[{spec.video_rows},{VIDEO_FEATURE_DIM}]"
        )
    encoder_digest = _validate_sha256_digest(
        encoder_identity.get(spec.encoder_digest_key),
        f"branch_encoder_identity.{spec.encoder_digest_key}",
    )
    embedded_digest = video_metadata.get(spec.encoder_digest_key)
    if embedded_digest is not None and embedded_digest != encoder_digest:
        raise ValueError(
            f"{branch_name} encoder digest differs between video metadata "
            "and branch_encoder_identity"
        )
    t5_weights_sha256 = _validate_sha256_digest(
        t5_weights_sha256, "t5_weights_sha256"
    )
    tokenizer_sha256 = _validate_sha256_digest(
        tokenizer_sha256, "tokenizer_sha256"
    )
    t5_model_id = _canonical_path_or_id(t5_model_id, "t5_model_id")
    tokenizer_id = _canonical_path_or_id(tokenizer_id, "tokenizer_id")

    video_flat_dim = spec.video_rows * VIDEO_FEATURE_DIM
    text_flat_dim = spec.text_tokens * TEXT_FEATURE_DIM
    joint_flat_dim = video_flat_dim + text_flat_dim
    metadata: dict[str, object] = {
        "joint_feature_metadata_version": JOINT_FEATURE_METADATA_VERSION,
        "feature_type": FEATURE_TYPE,
        "branch_name": branch_name,
        "video": {
            "feature": video_metadata,
            "encoder_identity": encoder_identity,
            "matrix_shape": [spec.video_rows, VIDEO_FEATURE_DIM],
        },
        "text": {
            "model_id": t5_model_id,
            "tokenizer_id": tokenizer_id,
            "weights_sha256": t5_weights_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "raw_tokens": TEXT_RAW_TOKENS,
            "num_tokens": spec.text_tokens,
            "feature_dim": TEXT_FEATURE_DIM,
            "attention_mask": "tokenizer_mask_gt_zero",
            "token_sampling": TEXT_TOKEN_SAMPLING,
            "short_prompt_policy": "retain_valid_once_then_zero_pad",
        },
        "fusion": {
            "layout": JOINT_LAYOUT,
            "normalization": "none",
            "modality_scaling": "none",
            "video_matrix_shape": [spec.video_rows, VIDEO_FEATURE_DIM],
            "text_matrix_shape_before_reshape": [
                spec.text_tokens,
                TEXT_FEATURE_DIM,
            ],
            "text_matrix_shape_after_reshape": [
                spec.text_rows,
                VIDEO_FEATURE_DIM,
            ],
            "matrix_shape": [spec.joint_rows, VIDEO_FEATURE_DIM],
            "video_flat_dim": video_flat_dim,
            "text_flat_dim": text_flat_dim,
            "joint_flat_dim": joint_flat_dim,
            "text_flat_fraction": text_flat_dim / joint_flat_dim,
        },
        "sample_axis": "video_caption_pair",
        "kernel": "single_scale_gaussian_rbf",
        "kernel_metric": "frobenius_rbf_joint_aligned_rows",
    }
    metadata["joint_feature_contract_sha256"] = _canonical_json_sha256(
        metadata
    )
    json.dumps(metadata, allow_nan=False, sort_keys=True)
    return metadata


def validate_triple_t5_joint_feature_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate a serialized joint identity and detect any field mutation."""
    if not isinstance(metadata, Mapping):
        raise TypeError("joint feature metadata must be a mapping")
    result = _json_safe(metadata)
    if result.get("joint_feature_metadata_version") != (
        JOINT_FEATURE_METADATA_VERSION
    ):
        raise ValueError("unsupported joint_feature_metadata_version")
    if result.get("feature_type") != FEATURE_TYPE:
        raise ValueError(f"feature_type must be {FEATURE_TYPE!r}")
    spec = _branch_spec(result.get("branch_name"))
    video = result.get("video")
    text = result.get("text")
    fusion = result.get("fusion")
    if not all(isinstance(value, Mapping) for value in (video, text, fusion)):
        raise ValueError("video, text, and fusion metadata must be mappings")
    if video.get("matrix_shape") != [spec.video_rows, VIDEO_FEATURE_DIM]:
        raise ValueError("joint video matrix_shape does not match branch contract")
    if text.get("raw_tokens") != TEXT_RAW_TOKENS:
        raise ValueError("joint text raw_tokens does not match UMT5 contract")
    if text.get("num_tokens") != spec.text_tokens:
        raise ValueError("joint text num_tokens does not match branch contract")
    if text.get("feature_dim") != TEXT_FEATURE_DIM:
        raise ValueError("joint text feature_dim does not match UMT5 contract")
    _validate_sha256_digest(text.get("weights_sha256"), "text.weights_sha256")
    _validate_sha256_digest(
        text.get("tokenizer_sha256"), "text.tokenizer_sha256"
    )
    encoder_identity = video.get("encoder_identity")
    if not isinstance(encoder_identity, Mapping):
        raise ValueError("video.encoder_identity must be a mapping")
    _validate_sha256_digest(
        encoder_identity.get(spec.encoder_digest_key),
        f"video.encoder_identity.{spec.encoder_digest_key}",
    )
    expected_fusion = {
        "layout": JOINT_LAYOUT,
        "normalization": "none",
        "modality_scaling": "none",
        "video_matrix_shape": [spec.video_rows, VIDEO_FEATURE_DIM],
        "text_matrix_shape_before_reshape": [
            spec.text_tokens,
            TEXT_FEATURE_DIM,
        ],
        "text_matrix_shape_after_reshape": [
            spec.text_rows,
            VIDEO_FEATURE_DIM,
        ],
        "matrix_shape": [spec.joint_rows, VIDEO_FEATURE_DIM],
        "video_flat_dim": spec.video_rows * VIDEO_FEATURE_DIM,
        "text_flat_dim": spec.text_tokens * TEXT_FEATURE_DIM,
        "joint_flat_dim": spec.joint_rows * VIDEO_FEATURE_DIM,
        "text_flat_fraction": spec.text_flat_fraction,
    }
    if dict(fusion) != expected_fusion:
        raise ValueError("fusion metadata does not match the branch contract")
    claimed_hash = result.pop("joint_feature_contract_sha256", None)
    _validate_sha256_digest(
        claimed_hash, "joint_feature_contract_sha256"
    )
    actual_hash = _canonical_json_sha256(result)
    if claimed_hash != actual_hash:
        raise ValueError("joint feature metadata contract hash mismatch")
    result["joint_feature_contract_sha256"] = claimed_hash
    return result


__all__ = [
    "BRANCH_SPECS",
    "FEATURE_TYPE",
    "JOINT_FEATURE_METADATA_VERSION",
    "JOINT_LAYOUT",
    "JointBranchSpec",
    "TEXT_FEATURE_DIM",
    "TEXT_RAW_TOKENS",
    "TEXT_ROWS_PER_TOKEN",
    "TEXT_TOKEN_SAMPLING",
    "VIDEO_FEATURE_DIM",
    "canonical_triple_t5_joint_feature_metadata",
    "fuse_video_t5_features",
    "select_uniform_valid_text_tokens",
    "split_video_t5_joint_features",
    "validate_triple_t5_joint_feature_metadata",
]

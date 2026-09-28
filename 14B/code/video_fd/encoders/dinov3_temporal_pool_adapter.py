"""Frozen DINOv3 adapter for 20-frame feature-space temporal pooling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn

from video_fd.encoders.dinov3_adapter import DINOv3DenseAdapter
from video_fd.encoders.dinov3_dense_metadata import (
    FORWARD_CHUNK_FRAMES as BASE_FORWARD_CHUNK_FRAMES,
    FRAME_INDICES as BASE_FRAME_INDICES,
    SPATIAL_AXIS_INDICES as BASE_SPATIAL_AXIS_INDICES,
)
from video_fd.encoders.dinov3_temporal_pool_metadata import (
    FEATURE_DIM,
    FORWARD_CHUNK_FRAMES,
    FRAME_INDICES,
    GEOMETRY,
    NUM_TOKENS,
    PER_FRAME_TOKENS,
    RAW_TOKENS,
    SPATIAL_AXIS_INDICES,
    TEMPORAL_GROUPS,
    TEMPORAL_POOL_FRAMES,
    canonical_dinov3_temporal_pool_feature_metadata,
    full_patch_indices,
)


def _strict_tuple(config: dict, key: str, expected: tuple[int, ...]) -> tuple[int, ...]:
    raw = config.get(key, expected)
    if isinstance(raw, (str, bytes)):
        raise TypeError(f"{key} must be an integer sequence")
    values = tuple(raw)
    if any(type(value) is not int for value in values):
        raise TypeError(f"{key} must contain Python integers")
    if values != expected:
        raise ValueError(f"{key} must equal {list(expected)}")
    return values


class DINOv3TemporalPoolAdapter(DINOv3DenseAdapter):
    """Encode 20 frames, mean features by five, and retain all patches."""

    def __init__(self, config, model: Optional[nn.Module] = None):
        requested = dict(config)
        geometry = str(requested.get("geometry", GEOMETRY))
        if geometry != GEOMETRY:
            raise ValueError(f"geometry must be {GEOMETRY!r}")
        frames = _strict_tuple(requested, "frame_indices", FRAME_INDICES)
        axes = _strict_tuple(
            requested, "spatial_axis_indices", SPATIAL_AXIS_INDICES)
        temporal_groups = int(requested.get(
            "temporal_groups", TEMPORAL_GROUPS))
        temporal_pool_frames = int(requested.get(
            "temporal_pool_frames",
            requested.get("temporal_pool_size", TEMPORAL_POOL_FRAMES),
        ))
        if temporal_groups != TEMPORAL_GROUPS:
            raise ValueError(f"temporal_groups must be {TEMPORAL_GROUPS}")
        if temporal_pool_frames != TEMPORAL_POOL_FRAMES:
            raise ValueError(
                f"temporal_pool_frames must be {TEMPORAL_POOL_FRAMES}")
        if int(requested.get("per_frame_tokens", PER_FRAME_TOKENS)) != PER_FRAME_TOKENS:
            raise ValueError(f"per_frame_tokens must be {PER_FRAME_TOKENS}")
        if int(requested.get("num_tokens", NUM_TOKENS)) != NUM_TOKENS:
            raise ValueError(f"num_tokens must be {NUM_TOKENS}")
        if int(requested.get(
            "forward_chunk_frames", FORWARD_CHUNK_FRAMES
        )) != FORWARD_CHUNK_FRAMES:
            raise ValueError(
                f"forward_chunk_frames must be {FORWARD_CHUNK_FRAMES}")

        # The parent owns the pinned native model load, FP32/frozen checks,
        # resize and normalization implementation.  Initialize that machinery
        # with its immutable compatibility geometry, then replace only the
        # sampling buffers and public geometry before any feature extraction.
        base_config = dict(requested)
        base_config.update({
            "frame_indices": BASE_FRAME_INDICES,
            "spatial_axis_indices": BASE_SPATIAL_AXIS_INDICES,
            "forward_chunk_frames": BASE_FORWARD_CHUNK_FRAMES,
            "per_frame_tokens": 64,
            "num_tokens": 1024,
        })
        for key in (
            "geometry",
            "temporal_groups",
            "temporal_pool_frames",
            "temporal_pool_size",
            "spatial_token_indices",
        ):
            base_config.pop(key, None)
        super().__init__(base_config, model=model)

        self.config = requested
        self.geometry = GEOMETRY
        self.frame_indices = frames
        self.dense_num_frames = len(frames)
        self.spatial_axis_indices = axes
        self.spatial_patch_indices = tuple(range(PER_FRAME_TOKENS))
        self.spatial_token_indices = self.spatial_patch_indices
        self.temporal_groups = TEMPORAL_GROUPS
        self.temporal_pool_frames = TEMPORAL_POOL_FRAMES
        self.temporal_pool_size = TEMPORAL_POOL_FRAMES
        self.per_frame_tokens = PER_FRAME_TOKENS
        self.dense_raw_tokens = RAW_TOKENS
        self.dense_num_tokens = NUM_TOKENS
        self.dense_feature_dim = FEATURE_DIM
        self.raw_tokens = RAW_TOKENS
        self.num_tokens = NUM_TOKENS
        self.feature_dim = FEATURE_DIM
        self.forward_chunk_frames = FORWARD_CHUNK_FRAMES
        self._frame_indices_tensor = torch.tensor(frames, dtype=torch.long)
        self._spatial_patch_indices_tensor = full_patch_indices()

    @property
    def dense_token_indices(self) -> tuple[int, ...]:
        # Pooled rows are means, not a subset of raw frame-patch rows.
        return tuple(range(NUM_TOKENS))

    def _resolve_token_indices(
        self,
        token_indices: Optional[torch.Tensor | Sequence[int]],
    ) -> torch.Tensor:
        canonical = self._spatial_patch_indices_tensor
        if token_indices is None:
            return canonical
        supplied = (
            token_indices.detach()
            if isinstance(token_indices, torch.Tensor)
            else torch.tensor(tuple(token_indices))
        )
        if (
            supplied.dtype == torch.bool
            or supplied.is_floating_point()
            or supplied.is_complex()
        ):
            raise TypeError("token_indices must contain integers")
        supplied = supplied.detach().to(
            device=canonical.device, dtype=torch.long).contiguous()
        if not torch.equal(supplied, canonical):
            raise ValueError(
                "token_indices must contain all canonical 256 patch indices")
        return canonical

    def _pool_raw_features(
        self, raw_features: torch.Tensor, *, batch_size: int
    ) -> torch.Tensor:
        expected = (
            batch_size * self.dense_num_frames,
            PER_FRAME_TOKENS,
            FEATURE_DIM,
        )
        if tuple(raw_features.shape) != expected:
            raise RuntimeError(
                f"raw DINOv3 features must have shape {expected}, got "
                f"{tuple(raw_features.shape)}")
        return raw_features.reshape(
            batch_size,
            TEMPORAL_GROUPS,
            TEMPORAL_POOL_FRAMES,
            PER_FRAME_TOKENS,
            FEATURE_DIM,
        ).mean(dim=2).reshape(batch_size, NUM_TOKENS, FEATURE_DIM)

    def forward_dense(
        self,
        video_clip_01: torch.Tensor,
        token_indices: Optional[torch.Tensor | Sequence[int]] = None,
        gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        if type(gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing must be a bool")
        self._verify_model_fp32()
        self._validate_video(video_clip_01)
        resolved_tokens = self._resolve_token_indices(token_indices)
        batch_size = int(video_clip_01.shape[0])
        flat_frames = self._selected_flat_frames(video_clip_01)
        chunks = []
        for start in range(0, int(flat_frames.shape[0]), self.forward_chunk_frames):
            chunks.append(self._encode_frame_batch(
                flat_frames[start:start + self.forward_chunk_frames],
                resolved_tokens,
            ))
        return self._pool_raw_features(
            torch.cat(chunks, dim=0), batch_size=batch_size)

    @torch.no_grad()
    def forward_dense_no_grad(
        self,
        video_clip_01: torch.Tensor,
        token_indices: Optional[torch.Tensor | Sequence[int]] = None,
        gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        return self.forward_dense(
            video_clip_01,
            token_indices=token_indices,
            gradient_checkpointing=gradient_checkpointing,
        ).detach().float().clone()

    extract_dense_no_grad = forward_dense_no_grad

    def replay_dense_vjp(
        self,
        video_clip_01: torch.Tensor,
        feature_cotangent: torch.Tensor,
        *,
        expected_features: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor | Sequence[int]] = None,
        gradient_checkpointing: bool = False,
        replay_atol: float = 1.0e-6,
        replay_rtol: float = 1.0e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if type(gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing must be a bool")
        self._verify_model_fp32()
        self._validate_video(video_clip_01)
        batch_size = int(video_clip_01.shape[0])
        expected_shape = (batch_size, NUM_TOKENS, FEATURE_DIM)
        if (
            not isinstance(feature_cotangent, torch.Tensor)
            or not feature_cotangent.is_floating_point()
            or tuple(feature_cotangent.shape) != expected_shape
            or not bool(torch.isfinite(feature_cotangent).all().item())
        ):
            raise ValueError(
                "feature_cotangent must be finite floating with shape "
                f"{expected_shape}")
        if replay_atol < 0 or replay_rtol < 0:
            raise ValueError("replay tolerances must be non-negative")

        resolved_tokens = self._resolve_token_indices(token_indices)
        flat_frames = self._selected_flat_frames(video_clip_01.detach())
        pooled_cotangent = feature_cotangent.detach().float().reshape(
            batch_size,
            TEMPORAL_GROUPS,
            PER_FRAME_TOKENS,
            FEATURE_DIM,
        )
        raw_cotangent = (
            pooled_cotangent[:, :, None]
            .expand(
                batch_size,
                TEMPORAL_GROUPS,
                TEMPORAL_POOL_FRAMES,
                PER_FRAME_TOKENS,
                FEATURE_DIM,
            )
            .reshape(
                batch_size * self.dense_num_frames,
                PER_FRAME_TOKENS,
                FEATURE_DIM,
            )
            / float(TEMPORAL_POOL_FRAMES)
        )

        replay_raw_chunks = []
        gradient_chunks = []
        for start in range(0, int(flat_frames.shape[0]), self.forward_chunk_frames):
            end = min(start + self.forward_chunk_frames, int(flat_frames.shape[0]))
            frame_leaf = flat_frames[start:end].detach().clone().requires_grad_(True)
            raw_features = self._encode_frame_batch(frame_leaf, resolved_tokens)
            surrogate = (
                raw_features
                * raw_cotangent[start:end].to(raw_features.device)
            ).sum()
            (chunk_gradient,) = torch.autograd.grad(
                surrogate,
                frame_leaf,
                retain_graph=False,
                create_graph=False,
            )
            replay_raw_chunks.append(raw_features.detach().float())
            gradient_chunks.append(chunk_gradient.detach().float())

        replay_features = self._pool_raw_features(
            torch.cat(replay_raw_chunks, dim=0), batch_size=batch_size)
        if expected_features is not None:
            if (
                not isinstance(expected_features, torch.Tensor)
                or tuple(expected_features.shape) != expected_shape
                or not bool(torch.isfinite(expected_features).all().item())
            ):
                raise ValueError(
                    f"expected_features must be finite with shape {expected_shape}")
            comparison = expected_features.detach().to(
                device=replay_features.device, dtype=torch.float32)
            if not torch.allclose(
                replay_features,
                comparison,
                atol=float(replay_atol),
                rtol=float(replay_rtol),
            ):
                maximum = float((replay_features - comparison).abs().max().item())
                raise RuntimeError(
                    "DINOv3 temporal-pool replay differs from first pass: "
                    f"max_abs={maximum:.8g}")

        selected_gradient = torch.cat(gradient_chunks, dim=0).reshape(
            batch_size,
            self.dense_num_frames,
            *video_clip_01.shape[2:],
        )
        full_pixel_cotangent = torch.zeros(
            video_clip_01.shape,
            device=video_clip_01.device,
            dtype=torch.float32,
        )
        full_pixel_cotangent.index_copy_(
            1,
            self._frame_indices_tensor.to(video_clip_01.device),
            selected_gradient,
        )
        if not bool(torch.isfinite(full_pixel_cotangent).all().item()):
            raise FloatingPointError(
                "DINOv3 temporal-pool replay returned non-finite pixels")
        return replay_features, full_pixel_cotangent

    def feature_metadata(self) -> dict[str, object]:
        return canonical_dinov3_temporal_pool_feature_metadata(
            repo_path=self.repo_path,
            weight_path=self.weight_path,
            source_commit=self.source_commit,
            weight_sha256=self.weight_sha256,
            source_num_frames=self.source_num_frames,
            frame_indices=self.frame_indices,
            input_size=self.input_size,
            patch_size=self.patch_size,
            patch_grid_size=self.patch_grid_size,
            spatial_axis_indices=self.spatial_axis_indices,
            geometry=self.geometry,
            temporal_groups=self.temporal_groups,
            temporal_pool_frames=self.temporal_pool_frames,
            per_frame_tokens=self.per_frame_tokens,
            num_tokens=self.dense_num_tokens,
            forward_chunk_frames=self.forward_chunk_frames,
            feature_dim=self.dense_feature_dim,
            model_id=self.model_id,
            vae_id=self.vae_id,
        )


__all__ = ["DINOv3TemporalPoolAdapter"]

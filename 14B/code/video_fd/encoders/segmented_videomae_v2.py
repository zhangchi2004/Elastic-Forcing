"""Five-segment VideoMAE-v2 encoder for an 81-frame Wan video.

Frame zero is intentionally excluded.  The remaining 80 frames are split into
five fixed, adjacent 16-frame segments.  Production training may reuse one
frozen checkpoint instance for every segment; the legacy five-replica layout
remains available for compatibility.  Selected dense tokens are concatenated
in chronological segment order.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn

from video_fd.encoders.videomae_v2_adapter import VideoMAEv2Adapter


class SegmentedVideoMAEv2Ensemble(nn.Module):
    """Five logical same-checkpoint segments with a V-JEPA-compatible API."""

    def __init__(
        self,
        config,
        *,
        adapter_factory: Optional[Callable[[dict], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.encoder_count = int(self.config.get("encoder_count", 5))
        self.share_encoder_weights = bool(
            self.config.get("share_encoder_weights", False)
        )
        self.skip_leading_frames = int(
            self.config.get("skip_leading_frames", 1))
        self.segment_num_frames = int(
            self.config.get("segment_num_frames", 16))
        self.source_num_frames = int(
            self.config.get("source_num_frames", 81))
        if self.encoder_count != 5:
            raise ValueError(
                "the VideoMAE-v2 whole-video contract requires exactly five "
                f"encoders, got {self.encoder_count}")
        if self.skip_leading_frames != 1:
            raise ValueError(
                "the VideoMAE-v2 whole-video contract skips exactly one "
                f"leading frame, got {self.skip_leading_frames}")
        if self.segment_num_frames != 16:
            raise ValueError(
                "each VideoMAE-v2 segment must contain 16 frames, got "
                f"{self.segment_num_frames}")
        expected_source = (
            self.skip_leading_frames
            + self.encoder_count * self.segment_num_frames
        )
        if self.source_num_frames != expected_source:
            raise ValueError(
                "source_num_frames must exactly cover the skipped frame and "
                f"five segments: {self.source_num_frames} != {expected_source}")

        factory = adapter_factory or VideoMAEv2Adapter
        adapter_config = dict(self.config)
        adapter_config.update({
            "input_frames": self.segment_num_frames,
            "dense_num_frames": self.segment_num_frames,
        })
        physical_encoder_count = (
            1 if self.share_encoder_weights else self.encoder_count
        )
        self.encoders = nn.ModuleList([
            factory(dict(adapter_config))
            for _ in range(physical_encoder_count)
        ])
        self._validate_replicas()

    def _validate_replicas(self) -> None:
        expected_physical = 1 if self.share_encoder_weights else self.encoder_count
        if len(self.encoders) != expected_physical:
            raise RuntimeError("VideoMAE-v2 replica construction was incomplete")
        if len({id(encoder) for encoder in self.encoders}) != expected_physical:
            raise RuntimeError(
                "VideoMAE-v2 physical encoder instances are not distinct")
        first = self.encoders[0]
        geometry = (
            int(first.dense_layer_index),
            int(first.dense_num_frames),
            int(first.resize_short_side),
            int(first.input_size),
            int(first.retained_dense_blocks),
            int(first.dense_raw_tokens),
            int(first.dense_feature_dim),
        )
        for index, encoder in enumerate(self.encoders[1:], start=1):
            candidate = (
                int(encoder.dense_layer_index),
                int(encoder.dense_num_frames),
                int(encoder.resize_short_side),
                int(encoder.input_size),
                int(encoder.retained_dense_blocks),
                int(encoder.dense_raw_tokens),
                int(encoder.dense_feature_dim),
            )
            if candidate != geometry:
                raise RuntimeError(
                    "VideoMAE-v2 replica geometry differs at index "
                    f"{index}: {candidate} != {geometry}")
        storage_sets = []
        for encoder in self.encoders:
            storage_sets.append({
                parameter.untyped_storage().data_ptr()
                for parameter in encoder.parameters()
                if parameter.numel() > 0
            })
        for left in range(len(storage_sets)):
            for right in range(left + 1, len(storage_sets)):
                if storage_sets[left].intersection(storage_sets[right]):
                    raise RuntimeError(
                        "VideoMAE-v2 replicas unexpectedly share parameter "
                        f"storage between segments {left} and {right}")

    def _encoder_for_segment(self, index: int) -> nn.Module:
        if index < 0 or index >= self.encoder_count:
            raise IndexError(f"segment index outside [0,{self.encoder_count})")
        return self.encoders[0 if self.share_encoder_weights else index]

    @property
    def dense_layer_index(self) -> int:
        return int(self.encoders[0].dense_layer_index)

    @property
    def dense_num_frames(self) -> int:
        return self.segment_num_frames

    @property
    def resize_short_side(self) -> int:
        return int(self.encoders[0].resize_short_side)

    @property
    def input_size(self) -> int:
        return int(self.encoders[0].input_size)

    @property
    def retained_dense_blocks(self) -> int:
        return int(self.encoders[0].retained_dense_blocks)

    @property
    def dense_raw_tokens(self) -> int:
        """Raw tokens per 16-frame segment, not the five-segment total."""
        return int(self.encoders[0].dense_raw_tokens)

    @property
    def aggregate_dense_raw_tokens(self) -> int:
        return self.encoder_count * self.dense_raw_tokens

    @property
    def dense_feature_dim(self) -> int:
        return int(self.encoders[0].dense_feature_dim)

    def _validate_video(self, video_clip_01: torch.Tensor) -> None:
        if video_clip_01.ndim != 5 or int(video_clip_01.shape[2]) != 3:
            raise ValueError(
                "video_clip_01 must have shape [B,81,3,H,W], got "
                f"{tuple(video_clip_01.shape)}")
        if int(video_clip_01.shape[1]) != self.source_num_frames:
            raise ValueError(
                "segmented VideoMAE-v2 requires exactly "
                f"{self.source_num_frames} decoded frames, got "
                f"{int(video_clip_01.shape[1])}")

    def segment(self, video_clip_01: torch.Tensor, index: int) -> torch.Tensor:
        self._validate_video(video_clip_01)
        index = int(index)
        if index < 0 or index >= self.encoder_count:
            raise IndexError(f"segment index outside [0,{self.encoder_count})")
        start = self.skip_leading_frames + index * self.segment_num_frames
        return video_clip_01.narrow(1, start, self.segment_num_frames)

    def forward_dense(
        self,
        video_clip_01: torch.Tensor,
        layer_index: Optional[int] = None,
        token_indices: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
        temporal_window_starts: Optional[torch.Tensor] = None,
        temporal_window_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if temporal_window_starts is not None or temporal_window_indices is not None:
            raise ValueError(
                "fixed five-segment VideoMAE-v2 does not accept temporal windows")
        outputs = [
            self._encoder_for_segment(index).forward_dense(
                self.segment(video_clip_01, index),
                layer_index=layer_index,
                token_indices=token_indices,
                gradient_checkpointing=gradient_checkpointing,
            )
            for index in range(self.encoder_count)
        ]
        return torch.cat(outputs, dim=1)

    def forward_dense_layers(
        self,
        video_clip_01: torch.Tensor,
        layer_indices: Sequence[int],
        token_indices: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
        temporal_window_starts: Optional[torch.Tensor] = None,
        temporal_window_indices: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        if temporal_window_starts is not None or temporal_window_indices is not None:
            raise ValueError(
                "fixed five-segment VideoMAE-v2 does not accept temporal windows")
        layers = tuple(int(layer) for layer in layer_indices)
        by_segment = [
            self._encoder_for_segment(index).forward_dense_layers(
                self.segment(video_clip_01, index),
                layer_indices=layers,
                token_indices=token_indices,
                gradient_checkpointing=gradient_checkpointing,
            )
            for index in range(self.encoder_count)
        ]
        return {
            layer: torch.cat(
                [segment_features[layer] for segment_features in by_segment],
                dim=1,
            )
            for layer in layers
        }

    def replay_dense_vjp(
        self,
        video_clip_01: torch.Tensor,
        *,
        layer_indices: Sequence[int],
        token_indices: torch.Tensor,
        expected_features: torch.Tensor,
        feature_cotangent: torch.Tensor,
        gradient_checkpointing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay five encoder VJPs sequentially to cap activation memory.

        Feature transport is layer-major and, within a layer, segment-major.
        The returned pixel cotangent has a zero first frame by construction.
        """
        self._validate_video(video_clip_01)
        layers = tuple(int(layer) for layer in layer_indices)
        if not layers:
            raise ValueError("layer_indices must not be empty")
        tokens_per_segment = int(token_indices.numel())
        tokens_per_layer = self.encoder_count * tokens_per_segment
        expected_shape = (
            int(video_clip_01.shape[0]),
            len(layers) * tokens_per_layer,
            self.dense_feature_dim,
        )
        if tuple(expected_features.shape) != expected_shape:
            raise ValueError(
                "expected_features has the wrong segmented shape: "
                f"{tuple(expected_features.shape)} != {expected_shape}")
        if tuple(feature_cotangent.shape) != expected_shape:
            raise ValueError(
                "feature_cotangent has the wrong segmented shape: "
                f"{tuple(feature_cotangent.shape)} != {expected_shape}")

        expected_by_layer = expected_features.split(tokens_per_layer, dim=1)
        cotangent_by_layer = feature_cotangent.split(tokens_per_layer, dim=1)
        replay_by_layer = [[] for _ in layers]
        video_cotangent = torch.zeros_like(video_clip_01, dtype=torch.float32)

        for segment_index in range(self.encoder_count):
            encoder = self._encoder_for_segment(segment_index)
            segment_leaf = self.segment(
                video_clip_01, segment_index).detach().requires_grad_(True)
            if len(layers) == 1:
                outputs = {
                    layers[0]: encoder.forward_dense(
                        segment_leaf,
                        layer_index=layers[0],
                        token_indices=token_indices,
                        gradient_checkpointing=gradient_checkpointing,
                    ).float()
                }
            else:
                outputs = encoder.forward_dense_layers(
                    segment_leaf,
                    layer_indices=layers,
                    token_indices=token_indices,
                    gradient_checkpointing=gradient_checkpointing,
                )
                outputs = {layer: value.float() for layer, value in outputs.items()}

            surrogate = segment_leaf.new_zeros((), dtype=torch.float32)
            for layer_offset, layer in enumerate(layers):
                start = segment_index * tokens_per_segment
                expected_part = expected_by_layer[layer_offset].narrow(
                    1, start, tokens_per_segment)
                cotangent_part = cotangent_by_layer[layer_offset].narrow(
                    1, start, tokens_per_segment)
                output = outputs[layer]
                if tuple(output.shape) != tuple(expected_part.shape):
                    raise RuntimeError(
                        "VideoMAE-v2 segment replay shape changed for "
                        f"segment={segment_index}, layer={layer}")
                replay_by_layer[layer_offset].append(output.detach())
                surrogate = surrogate + (output * cotangent_part).sum()
            (segment_cotangent,) = torch.autograd.grad(
                surrogate,
                segment_leaf,
                retain_graph=False,
                create_graph=False,
            )
            start_frame = (
                self.skip_leading_frames
                + segment_index * self.segment_num_frames
            )
            video_cotangent[:, start_frame:start_frame + self.segment_num_frames] = (
                segment_cotangent.detach().float())
            del surrogate, segment_leaf, segment_cotangent, outputs

        replay_features = torch.cat([
            torch.cat(parts, dim=1)
            for parts in replay_by_layer
        ], dim=1)
        return replay_features, video_cotangent


__all__ = ["SegmentedVideoMAEv2Ensemble"]

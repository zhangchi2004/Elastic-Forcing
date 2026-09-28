"""Five-replica VideoMAE-v2 encoder for an 81-frame Wan video.

Frame zero is intentionally excluded.  The remaining 80 frames are split into
five fixed, adjacent 16-frame segments.  Each segment is processed by its own
frozen VideoMAE-v2 module instance loaded from the same checkpoint.  Selected
dense tokens are concatenated in chronological segment order.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn

from video_fd.encoders.videomae_v2_adapter import VideoMAEv2Adapter


class SegmentedVideoMAEv2Ensemble(nn.Module):
    """Five physical same-checkpoint encoders with a V-JEPA-compatible API."""

    def __init__(
        self,
        config,
        *,
        adapter_factory: Optional[Callable[[dict], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.config = dict(config)
        self.encoder_count = int(self.config.get("encoder_count", 5))
        self.skip_leading_frames = int(
            self.config.get("skip_leading_frames", 1))
        self.segment_num_frames = int(
            self.config.get("segment_num_frames", 16))
        self.source_num_frames = int(
            self.config.get("source_num_frames", 81))
        expected_source = (
            self.skip_leading_frames
            + self.encoder_count * self.segment_num_frames
        )

        factory = adapter_factory or VideoMAEv2Adapter
        adapter_config = dict(self.config)
        adapter_config.update({
            "input_frames": self.segment_num_frames,
            "dense_num_frames": self.segment_num_frames,
        })
        # These are deliberately five independent Module objects with distinct
        # parameter storages.  The checkpoint identity is nevertheless shared.
        self.encoders = nn.ModuleList([
            factory(dict(adapter_config))
            for _ in range(self.encoder_count)
        ])
        self._validate_replicas()

    def _validate_replicas(self) -> None:
        return None

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
        return None

    def segment(self, video_clip_01: torch.Tensor, index: int) -> torch.Tensor:
        self._validate_video(video_clip_01)
        index = int(index)
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
        outputs = [
            encoder.forward_dense(
                self.segment(video_clip_01, index),
                layer_index=layer_index,
                token_indices=token_indices,
                gradient_checkpointing=gradient_checkpointing,
            )
            for index, encoder in enumerate(self.encoders)
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
        layers = tuple(int(layer) for layer in layer_indices)
        by_segment = [
            encoder.forward_dense_layers(
                self.segment(video_clip_01, index),
                layer_indices=layers,
                token_indices=token_indices,
                gradient_checkpointing=gradient_checkpointing,
            )
            for index, encoder in enumerate(self.encoders)
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
        tokens_per_segment = int(token_indices.numel())
        tokens_per_layer = self.encoder_count * tokens_per_segment
        expected_shape = (
            int(video_clip_01.shape[0]),
            len(layers) * tokens_per_layer,
            self.dense_feature_dim,
        )

        expected_by_layer = expected_features.split(tokens_per_layer, dim=1)
        cotangent_by_layer = feature_cotangent.split(tokens_per_layer, dim=1)
        replay_by_layer = [[] for _ in layers]
        video_cotangent = torch.zeros_like(video_clip_01, dtype=torch.float32)

        for segment_index, encoder in enumerate(self.encoders):
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

"""OpenGVLab VideoMAE V2 sparse and intermediate-dense adapter.

The OpenGVLab Hugging Face checkpoints use remote code and wrap their
``VisionTransformer`` in ``model.model``.  That implementation consumes
``[B, C, T, H, W]`` and returns a pooled video embedding from its public
forward method.  Feature-distribution training instead needs the complete
post-residual token state of an encoder block, before mean pooling and
``fc_norm``.  This adapter therefore executes the verified OpenGVLab backbone
prefix explicitly.

There is deliberately no VideoMAE-v1 fallback: silently changing model family
would make reference-bank identity and token geometry incorrect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from video_fd.encoders.base_sparse_video_encoder import (
    SparseVideoEncoderAdapter,
    differentiable_resize_crop,
)


class VideoMAEv2Adapter(SparseVideoEncoderAdapter):
    """Frozen OpenGVLab VideoMAE V2 adapter with a dense-prefix API.

    Dense mode handles one native checkpoint segment (normally 16 frames) per
    call.  Five-segment assembly is intentionally owned by the caller so five
    separately placed/frozen encoder instances can be used by the training
    model.

    ``model`` is an injection seam for tiny tests.  Injected models must expose
    the same contract as the OpenGVLab remote-code wrapper: nested
    ``config.model_config`` and a ``model`` backbone with ``patch_embed``,
    ``pos_embed``, ``pos_drop``, and ``blocks``.
    """

    def __init__(self, config, model: Optional[nn.Module] = None):
        config = dict(config)
        explicit_dense_num_frames = "dense_num_frames" in config
        explicit_resize_short_side = (
            "resize_short_side" in config
            or "dense_resize_short_side" in config
        )
        config.setdefault("name", "videomae_v2")
        config.setdefault("input_size", config.get("crop_size", 224))
        config.setdefault("input_frames", 16)
        config.setdefault("norm_mean", (0.485, 0.456, 0.406))
        config.setdefault("norm_std", (0.229, 0.224, 0.225))
        super().__init__(config)

        self.model_id = str(
            config.get("model_id", "OpenGVLab/VideoMAEv2-Large")
        )
        self.patch_size = int(config.get("patch_size", 16))
        self.tubelet_size = int(config.get("tubelet_size", 2))
        self.dense_layer_index = int(config.get("dense_layer_index", 15))
        self.dense_num_frames = int(config.get("dense_num_frames", 16))
        self.resize_short_side = int(
            config.get(
                "resize_short_side",
                config.get("dense_resize_short_side", self.input_size),
            )
        )
        self.dense_only = bool(config.get("dense_only", False))
        self._explicit_dense_num_frames = explicit_dense_num_frames
        self._explicit_resize_short_side = explicit_resize_short_side
        self._model: Optional[nn.Module] = None
        self._configured_depth = 0
        self._embed_dim = 0
        self._dense_pruned = False

        if model is None:
            self._load()
        else:
            self._adopt_model(model)

    # -- loading and verified OpenGVLab architecture ----------------------
    def _load(self) -> None:
        try:
            from transformers import AutoModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OpenGVLab VideoMAE V2 requires transformers.AutoModel with "
                "trust_remote_code support"
            ) from exc

        load_kwargs: Dict[str, object] = {"trust_remote_code": True}
        for key in ("revision", "cache_dir", "local_files_only", "token"):
            if key in self.config:
                load_kwargs[key] = self.config[key]
        try:
            model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
        except Exception as exc:  # pragma: no cover - needs real checkpoint
            raise RuntimeError(
                f"failed to load OpenGVLab VideoMAE V2 checkpoint "
                f"{self.model_id!r}; no VideoMAE-v1 fallback is permitted"
            ) from exc
        self._adopt_model(model)

    @staticmethod
    def _config_value(model_config: object, name: str) -> object:
        if isinstance(model_config, Mapping):
            if name not in model_config:
                raise RuntimeError(
                    f"OpenGVLab config.model_config is missing {name!r}"
                )
            return model_config[name]
        if not hasattr(model_config, name):
            raise RuntimeError(
                f"OpenGVLab config.model_config is missing {name!r}"
            )
        return getattr(model_config, name)

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if isinstance(value, bool):
            raise RuntimeError(f"OpenGVLab {name} must be a positive integer")
        if isinstance(value, (tuple, list)):
            if len(value) != 2 or int(value[0]) != int(value[1]):
                raise RuntimeError(
                    f"OpenGVLab {name} must describe a square geometry, got {value!r}"
                )
            value = value[0]
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"OpenGVLab {name} must be a positive integer, got {value!r}"
            ) from exc
        if result <= 0:
            raise RuntimeError(
                f"OpenGVLab {name} must be positive, got {result}"
            )
        return result

    def _adopt_model(self, model: nn.Module) -> None:
        cfg = getattr(model, "config", None)
        model_config = getattr(cfg, "model_config", None)
        if model_config is None:
            raise RuntimeError(
                "model is not an OpenGVLab VideoMAE V2 wrapper: expected "
                "config.model_config (VideoMAE-v1 models are unsupported)"
            )

        patch_size = self._positive_int(
            self._config_value(model_config, "patch_size"),
            "model_config.patch_size",
        )
        tubelet_size = self._positive_int(
            self._config_value(model_config, "tubelet_size"),
            "model_config.tubelet_size",
        )
        input_size = self._positive_int(
            self._config_value(model_config, "img_size"),
            "model_config.img_size",
        )
        input_frames = self._positive_int(
            self._config_value(model_config, "num_frames"),
            "model_config.num_frames",
        )
        embed_dim = self._positive_int(
            self._config_value(model_config, "embed_dim"),
            "model_config.embed_dim",
        )
        depth = self._positive_int(
            self._config_value(model_config, "depth"),
            "model_config.depth",
        )

        backbone = getattr(model, "model", None)
        if not isinstance(backbone, nn.Module):
            raise RuntimeError(
                "OpenGVLab VideoMAE V2 wrapper must expose an nn.Module at model.model"
            )
        for name in ("patch_embed", "pos_drop", "blocks"):
            if not hasattr(backbone, name):
                raise RuntimeError(
                    "OpenGVLab VideoMAE V2 backbone is missing "
                    f"model.{name}"
                )
        blocks = getattr(backbone, "blocks")
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            raise RuntimeError(
                "OpenGVLab VideoMAE V2 model.blocks must be a ModuleList or Sequential"
            )
        if len(blocks) != depth:
            raise RuntimeError(
                "OpenGVLab VideoMAE V2 depth mismatch: config.model_config.depth="
                f"{depth}, but model.blocks has {len(blocks)} entries"
            )
        if not hasattr(backbone, "pos_embed"):
            raise RuntimeError(
                "OpenGVLab VideoMAE V2 backbone is missing model.pos_embed"
            )
        if input_frames % tubelet_size:
            raise RuntimeError(
                f"num_frames={input_frames} is not divisible by tubelet_size={tubelet_size}"
            )
        if input_size % patch_size:
            raise RuntimeError(
                f"img_size={input_size} is not divisible by patch_size={patch_size}"
            )

        if (
            self._explicit_dense_num_frames
            and self.dense_num_frames != input_frames
        ):
            raise ValueError(
                "dense_num_frames must equal the native OpenGVLab checkpoint "
                f"length ({input_frames}), got {self.dense_num_frames}"
            )
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.input_size = input_size
        self.input_frames = input_frames
        self.dense_num_frames = input_frames
        self._embed_dim = embed_dim
        self._configured_depth = depth
        if not self._explicit_resize_short_side:
            self.resize_short_side = input_size
        if self.resize_short_side < self.input_size:
            raise ValueError(
                "resize_short_side must be >= the checkpoint crop size, got "
                f"{self.resize_short_side} < {self.input_size}"
            )

        self._model = model.to(dtype=torch.float32)
        self._validate_dense_layer(self.dense_layer_index)
        if self.dense_only:
            self.prune_for_dense_prefix(self.dense_layer_index)
        self.freeze()

    @property
    def backbone(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("VideoMAE V2 model is not loaded")
        backbone = getattr(self._model, "model", None)
        if not isinstance(backbone, nn.Module):
            raise RuntimeError("VideoMAE V2 backbone is unavailable")
        return backbone

    # -- geometry ---------------------------------------------------------
    @property
    def dense_feature_dim(self) -> int:
        return self._embed_dim

    @property
    def dense_raw_tokens(self) -> int:
        return (
            (self.dense_num_frames // self.tubelet_size)
            * (self.input_size // self.patch_size)
            * (self.input_size // self.patch_size)
        )

    @property
    def retained_dense_blocks(self) -> int:
        return len(self.backbone.blocks)

    def _grid_shape(self, n_tokens: int) -> Optional[Tuple[int, int, int]]:
        temporal = self.input_frames // self.tubelet_size
        spatial = self.input_size // self.patch_size
        return (
            (temporal, spatial, spatial)
            if temporal * spatial * spatial == n_tokens
            else None
        )

    # -- official differentiable preprocessing ---------------------------
    def _preprocess_clip(
        self,
        video_clip_01: torch.Tensor,
        *,
        crop_mode: str,
    ) -> torch.Tensor:
        if video_clip_01.ndim != 5:
            raise ValueError(
                "video_clip_01 must have shape [B,T,3,H,W], "
                f"got {tuple(video_clip_01.shape)}"
            )
        if int(video_clip_01.shape[2]) != 3:
            raise ValueError(
                f"video_clip_01 must have 3 channels, got {video_clip_01.shape[2]}"
            )
        if int(video_clip_01.shape[1]) != self.input_frames:
            raise ValueError(
                "VideoMAE V2 requires exactly one native segment of "
                f"{self.input_frames} frames, got {video_clip_01.shape[1]}"
            )
        x = differentiable_resize_crop(
            video_clip_01,
            self.input_size,
            crop_mode=crop_mode,
            resize_short_side=self.resize_short_side,
            interpolation="bilinear",
            round_center_crop=False,
        )
        batch, frames, channels, height, width = x.shape
        flat = x.reshape(batch * frames, channels, height, width)
        flat = (
            flat - self.norm_mean.to(dtype=flat.dtype, device=flat.device)
        ) / self.norm_std.to(dtype=flat.dtype, device=flat.device)
        return flat.reshape(batch, frames, channels, height, width)

    def preprocess(self, video_clip_01: torch.Tensor) -> torch.Tensor:
        """Official bilinear/ImageNet preprocessing for the sparse API."""
        return self._preprocess_clip(video_clip_01, crop_mode=self.crop_mode)

    def preprocess_dense(
        self,
        video_clip_01: torch.Tensor,
        temporal_window_starts: Optional[torch.Tensor] = None,
        temporal_window_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Preprocess exactly one native segment, preserving ``[B,T,C,H,W]``.

        Temporal window arguments exist for call-site compatibility with the
        V-JEPA2 adapter.  VideoMAE V2 segment selection is external, so any
        non-``None`` value is rejected.
        """
        if temporal_window_starts is not None or temporal_window_indices is not None:
            raise ValueError(
                "VideoMAE V2 accepts one already-selected 16-frame segment; "
                "temporal window selection must be performed by the caller"
            )
        return self._preprocess_clip(video_clip_01, crop_mode="center")

    # -- OpenGVLab embedding and blocks -----------------------------------
    def _embed_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # OpenGVLab's remote implementation expects BCTHW, unlike the public
        # adapter API (and Hugging Face VideoMAE-v1), which uses BTCHW.
        bcthw = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = self.backbone.patch_embed(bcthw)
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            raise RuntimeError(
                "OpenGVLab model.patch_embed must return a [B,N,D] tensor"
            )
        if int(hidden_states.shape[1]) != self.dense_raw_tokens:
            raise RuntimeError(
                "unexpected VideoMAE V2 token grid: expected "
                f"{self.dense_raw_tokens}, got {hidden_states.shape[1]}"
            )
        if int(hidden_states.shape[2]) != self.dense_feature_dim:
            raise RuntimeError(
                "unexpected VideoMAE V2 feature dimension: expected "
                f"{self.dense_feature_dim}, got {hidden_states.shape[2]}"
            )

        pos_embed = getattr(self.backbone, "pos_embed")
        if pos_embed is not None:
            if not isinstance(pos_embed, torch.Tensor):
                raise RuntimeError("OpenGVLab model.pos_embed must be a tensor or None")
            if pos_embed.ndim == 2:
                pos_embed = pos_embed.unsqueeze(0)
            if (
                pos_embed.ndim != 3
                or int(pos_embed.shape[1]) != int(hidden_states.shape[1])
                or int(pos_embed.shape[2]) != int(hidden_states.shape[2])
                or int(pos_embed.shape[0]) not in (1, int(hidden_states.shape[0]))
            ):
                raise RuntimeError(
                    "OpenGVLab model.pos_embed is incompatible with patch tokens: "
                    f"{tuple(pos_embed.shape)} vs {tuple(hidden_states.shape)}"
                )
            if int(pos_embed.shape[0]) == 1:
                pos_embed = pos_embed.expand(int(hidden_states.shape[0]), -1, -1)
            # This mirrors the remote forward exactly.  In official checkpoints
            # sinusoidal position embeddings may be a plain CPU Tensor rather
            # than a registered buffer.
            hidden_states = hidden_states + (
                pos_embed.type_as(hidden_states)
                .to(hidden_states.device)
                .clone()
                .detach()
            )

        hidden_states = self.backbone.pos_drop(hidden_states)
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError("OpenGVLab model.pos_drop must return a tensor")
        self._raw_dim = int(hidden_states.shape[-1])
        return hidden_states

    @staticmethod
    def _run_block(block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        output = block(hidden_states)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "OpenGVLab VideoMAE V2 blocks must return the post-residual tensor"
            )
        return output

    def _encode_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return final post-block tokens for the historical sparse API."""
        if self._dense_pruned:
            raise RuntimeError(
                "forward_sparse is unavailable after prune_for_dense_prefix(); "
                "construct VideoMAEv2Adapter with dense_only=False to retain it"
            )
        self._validate_preprocessed(pixel_values)
        hidden_states = self._embed_tokens(pixel_values)
        for block in self.backbone.blocks:
            hidden_states = self._run_block(block, hidden_states)
        # Deliberately before mean pooling and fc_norm.
        return hidden_states

    # -- intermediate dense feature extraction ---------------------------
    def forward_dense(
        self,
        video_clip_01: torch.Tensor,
        layer_index: Optional[int] = None,
        token_indices: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
        temporal_window_starts: Optional[torch.Tensor] = None,
        temporal_window_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pixel_values = self.preprocess_dense(
            video_clip_01,
            temporal_window_starts=temporal_window_starts,
            temporal_window_indices=temporal_window_indices,
        )
        return self.encode_dense_preprocessed(
            pixel_values,
            layer_index=layer_index,
            token_indices=token_indices,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward_dense_layers(
        self,
        video_clip_01: torch.Tensor,
        layer_indices: Sequence[int],
        token_indices: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
        temporal_window_starts: Optional[torch.Tensor] = None,
        temporal_window_indices: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        pixel_values = self.preprocess_dense(
            video_clip_01,
            temporal_window_starts=temporal_window_starts,
            temporal_window_indices=temporal_window_indices,
        )
        return self.encode_dense_preprocessed_layers(
            pixel_values,
            layer_indices=layer_indices,
            token_indices=token_indices,
            gradient_checkpointing=gradient_checkpointing,
        )

    def encode_dense_preprocessed(
        self,
        pixel_values: torch.Tensor,
        layer_index: Optional[int] = None,
        token_indices: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        target_layer = (
            self.dense_layer_index if layer_index is None else self._layer_int(layer_index)
        )
        return self.encode_dense_preprocessed_layers(
            pixel_values,
            layer_indices=(target_layer,),
            token_indices=token_indices,
            gradient_checkpointing=gradient_checkpointing,
        )[target_layer]

    def _validate_preprocessed(self, pixel_values: torch.Tensor) -> None:
        if pixel_values.ndim != 5:
            raise ValueError(
                "pixel_values must have shape [B,T,3,H,W], "
                f"got {tuple(pixel_values.shape)}"
            )
        if int(pixel_values.shape[2]) != 3:
            raise ValueError(
                f"pixel_values must have 3 channels, got {pixel_values.shape[2]}"
            )
        if int(pixel_values.shape[1]) != self.dense_num_frames:
            raise ValueError(
                f"pixel_values must contain {self.dense_num_frames} frames, "
                f"got {pixel_values.shape[1]}"
            )
        if tuple(pixel_values.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                f"pixel_values must be {self.input_size}x{self.input_size}, "
                f"got {tuple(pixel_values.shape[-2:])}"
            )

    @staticmethod
    def _layer_int(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"layer index must be an integer, got {value!r}")
        return int(value)

    @staticmethod
    def _token_index_tensor(
        values: torch.Tensor | Sequence[int],
        *,
        device: torch.device,
        num_tokens: int,
    ) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            if (
                values.dtype == torch.bool
                or values.is_floating_point()
                or values.is_complex()
            ):
                raise TypeError("token_indices must contain only integers")
            indices = values.to(device=device, dtype=torch.long)
        else:
            if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
                raise TypeError("token_indices must contain only integers")
            indices = torch.tensor(values, device=device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError(
                f"token_indices must be 1-D, got shape {tuple(indices.shape)}"
            )
        if indices.numel() == 0:
            raise ValueError("token_indices must not be empty")
        if int(indices.min()) < 0 or int(indices.max()) >= num_tokens:
            raise IndexError(f"token_indices must be in [0,{num_tokens - 1}]")
        return indices

    def encode_dense_preprocessed_layers(
        self,
        pixel_values: torch.Tensor,
        layer_indices: Sequence[int],
        token_indices: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """Run one prefix and capture complete post-residual block outputs."""
        self._validate_preprocessed(pixel_values)
        try:
            requested_layers = tuple(self._layer_int(layer) for layer in layer_indices)
        except TypeError:
            raise
        if not requested_layers:
            raise ValueError("layer_indices must not be empty")
        if len(set(requested_layers)) != len(requested_layers):
            raise ValueError("layer_indices must contain unique indices")
        for layer in requested_layers:
            self._validate_dense_layer(layer)
        max_target_layer = max(requested_layers)
        if max_target_layer >= self.retained_dense_blocks:
            raise ValueError(
                f"layer_index {max_target_layer} was pruned; retained blocks are "
                f"0..{self.retained_dense_blocks - 1}"
            )

        hidden_states = self._embed_tokens(pixel_values)
        indices = None
        if token_indices is not None:
            indices = self._token_index_tensor(
                token_indices,
                device=hidden_states.device,
                num_tokens=int(hidden_states.shape[1]),
            )

        requested_set = set(requested_layers)
        captured: Dict[int, torch.Tensor] = {}
        for block_index in range(max_target_layer + 1):
            block = self.backbone.blocks[block_index]

            def run_block(x: torch.Tensor, module=block) -> torch.Tensor:
                return self._run_block(module, x)

            if (
                gradient_checkpointing
                and torch.is_grad_enabled()
                and hidden_states.requires_grad
            ):
                hidden_states = checkpoint(
                    run_block,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = run_block(hidden_states)
            if block_index in requested_set:
                captured[block_index] = (
                    hidden_states
                    if indices is None
                    else hidden_states.index_select(1, indices)
                )

        return {layer: captured[layer] for layer in requested_layers}

    def prune_for_dense_prefix(
        self,
        layer_index: Optional[int] = None,
    ) -> "VideoMAEv2Adapter":
        """Irreversibly retain embeddings and blocks through ``layer_index``."""
        target_layer = (
            self.dense_layer_index if layer_index is None else self._layer_int(layer_index)
        )
        self._validate_dense_layer(target_layer)
        if target_layer >= self.retained_dense_blocks:
            raise ValueError(
                f"cannot restore pruned layer {target_layer}; only "
                f"{self.retained_dense_blocks} blocks remain"
            )
        self.backbone.blocks = nn.ModuleList(
            list(self.backbone.blocks[: target_layer + 1])
        )
        # Dense prefix exits before all of these official final-path modules.
        for name in ("norm", "fc_norm", "head_dropout", "head"):
            if hasattr(self.backbone, name):
                setattr(self.backbone, name, None)
        self._dense_pruned = True
        self.dense_only = True
        self.dense_layer_index = target_layer
        self.freeze()
        return self

    def _validate_dense_layer(self, layer_index: int) -> None:
        layer_index = self._layer_int(layer_index)
        if layer_index < 0:
            raise ValueError(
                f"dense layer index must be non-negative, got {layer_index}"
            )
        if self._configured_depth <= 0:
            raise RuntimeError("VideoMAE V2 model depth is unavailable")
        if layer_index >= self._configured_depth:
            raise ValueError(
                f"dense layer index {layer_index} out of range for "
                f"{self._configured_depth} encoder blocks"
            )

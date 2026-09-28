"""V-JEPA2 sparse and intermediate-dense video encoder adapter.

The historical sparse API remains available.  The dense API additionally
returns the complete post-residual hidden state of an intermediate encoder
block, before V-JEPA2's final LayerNorm and predictor.  It is intended for
feature-distribution losses whose gradient must replay through preprocessing
and a frozen encoder prefix.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from video_fd.encoders.base_sparse_video_encoder import (
    SparseVideoEncoderAdapter,
    differentiable_resize_crop,
)


class VJEPA2Adapter(SparseVideoEncoderAdapter):
    """Hugging Face V-JEPA2 adapter with an optional memory-pruned dense mode.

    Dense-mode configuration keys:

    * ``dense_layer_index``: zero-based complete encoder block (default 15)
    * ``dense_num_frames``: contiguous input frames (default 64)
    * ``resize_short_side``: differentiable resize short edge (default 292)
    * ``dense_only``: discard predictor, final norm, and blocks after the dense
      target once the checkpoint is loaded

    ``model`` is an injection seam for tiny unit tests.  Production callers
    normally omit it and load ``model_id`` with ``from_pretrained``.
    """

    def __init__(self, config, model: Optional[nn.Module] = None):
        config = dict(config)
        config.setdefault("name", "vjepa2")
        config.setdefault("input_size", config.get("crop_size", 256))
        config.setdefault("input_frames", 16)
        config.setdefault("norm_mean", (0.485, 0.456, 0.406))
        config.setdefault("norm_std", (0.229, 0.224, 0.225))
        super().__init__(config)
        self.model_id = str(config.get("model_id", "facebook/vjepa2-vitl-fpc64-256"))
        self.allow_temporal_pos_interpolation = bool(config.get("allow_temporal_pos_interpolation", True))
        self.tubelet_size = int(config.get("tubelet_size", 2))
        self.patch_size = int(config.get("patch_size", 16))
        self.dense_layer_index = int(config.get("dense_layer_index", 15))
        self.dense_num_frames = int(config.get("dense_num_frames", 64))
        self.resize_short_side = int(
            config.get("resize_short_side", config.get("dense_resize_short_side", 292))
        )
        self.dense_only = bool(config.get("dense_only", False))
        if self.dense_num_frames <= 0:
            raise ValueError(
                f"dense_num_frames must be positive, got {self.dense_num_frames}"
            )
        if self.resize_short_side < self.input_size:
            raise ValueError(
                "resize_short_side must be >= input_size, got "
                f"{self.resize_short_side} < {self.input_size}"
            )
        self._model = None
        self._dense_pruned = False
        if model is None:
            self._load()
        else:
            self._adopt_model(model)

    def _load(self):
        try:
            from transformers import VJEPA2Model  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "V-JEPA2 requires `transformers` with VJEPA2Model. Original error: %r" % (exc,)
            ) from exc
        model = VJEPA2Model.from_pretrained(self.model_id)
        self._adopt_model(model)

    def _adopt_model(self, model: nn.Module) -> None:
        """Install a loaded model, derive its geometry, optionally prune it."""
        cfg = getattr(model, "config", None)
        if cfg is not None:
            self.patch_size = int(getattr(cfg, "patch_size", self.patch_size))
            self.tubelet_size = int(getattr(cfg, "tubelet_size", self.tubelet_size))
            self.input_size = int(getattr(cfg, "crop_size", getattr(cfg, "image_size", self.input_size)))
            cfg_frames = getattr(cfg, "frames_per_clip", None)
            if cfg_frames is not None and int(cfg_frames) != self.input_frames and not self.allow_temporal_pos_interpolation:
                print(f"[vjepa2] checkpoint frames_per_clip={cfg_frames} != {self.input_frames}; "
                      "temporal interpolation disabled -> will fall back to global pooling.")
        self._model = model.to(torch.float32)
        self._validate_dense_layer(self.dense_layer_index)
        if self.dense_only:
            self.prune_for_dense_prefix(self.dense_layer_index)
        self.freeze()

    def _grid_shape(self, n_tokens: int) -> Optional[Tuple[int, int, int]]:
        Tp = self.input_frames // self.tubelet_size
        Hp = self.input_size // self.patch_size
        Wp = self.input_size // self.patch_size
        return (Tp, Hp, Wp) if Tp * Hp * Wp == n_tokens else None

    def _encode_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._dense_pruned:
            raise RuntimeError(
                "forward_sparse is unavailable after prune_for_dense_prefix(); "
                "construct VJEPA2Adapter with dense_only=False to retain it"
            )
        # transformers VJEPA2 expects pixel_values_videos [B, T, 3, H, W]
        # Predictor output is unused by the sparse encoder and can be skipped
        # without changing encoder.last_hidden_state.
        out = self._model(pixel_values_videos=pixel_values, skip_predictor=True)
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return tokens  # [B, N, D]

    # -- intermediate dense feature extraction -----------------------------
    @property
    def dense_feature_dim(self) -> int:
        cfg = getattr(self._model, "config", None)
        if cfg is None or not hasattr(cfg, "hidden_size"):
            raise RuntimeError("loaded V-JEPA2 model does not expose config.hidden_size")
        return int(cfg.hidden_size)

    @property
    def dense_raw_tokens(self) -> int:
        return (
            (self.dense_num_frames // self.tubelet_size)
            * (self.input_size // self.patch_size)
            * (self.input_size // self.patch_size)
        )

    @property
    def retained_dense_blocks(self) -> int:
        return len(self._model.encoder.layer)

    @staticmethod
    def _integer_index_tensor(
        values: torch.Tensor,
        *,
        name: str,
        device: torch.device,
    ) -> torch.Tensor:
        indices = torch.as_tensor(values)
        if (
            indices.dtype == torch.bool
            or indices.is_floating_point()
            or indices.is_complex()
        ):
            raise TypeError(f"{name} must contain only integers")
        return indices.to(device=device, dtype=torch.long)

    def _temporal_window_indices(
        self,
        video_clip_01: torch.Tensor,
        *,
        temporal_window_starts: Optional[torch.Tensor],
        temporal_window_indices: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Validate and canonicalize optional per-video contiguous windows."""
        if (
            temporal_window_starts is not None
            and temporal_window_indices is not None
        ):
            raise ValueError(
                "provide either temporal_window_starts or "
                "temporal_window_indices, not both"
            )
        if (
            temporal_window_starts is None
            and temporal_window_indices is None
        ):
            return None

        batch_size = int(video_clip_01.shape[0])
        total_frames = int(video_clip_01.shape[1])
        max_start = total_frames - self.dense_num_frames
        device = video_clip_01.device

        if temporal_window_starts is not None:
            starts = self._integer_index_tensor(
                temporal_window_starts,
                name="temporal_window_starts",
                device=device,
            )
            if starts.ndim != 1 or int(starts.shape[0]) != batch_size:
                raise ValueError(
                    "temporal_window_starts must have shape [B], got "
                    f"{tuple(starts.shape)} for B={batch_size}"
                )
            if starts.numel() and (
                int(starts.min()) < 0 or int(starts.max()) > max_start
            ):
                raise ValueError(
                    "temporal_window_starts must be in "
                    f"[0,{max_start}]"
                )
            offsets = torch.arange(
                self.dense_num_frames,
                dtype=torch.long,
                device=device,
            )
            return starts[:, None] + offsets[None, :]

        indices = self._integer_index_tensor(
            temporal_window_indices,
            name="temporal_window_indices",
            device=device,
        )
        expected_shape = (batch_size, self.dense_num_frames)
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                "temporal_window_indices must have shape "
                f"[B,{self.dense_num_frames}], got {tuple(indices.shape)} "
                f"for B={batch_size}"
            )
        if indices.numel():
            starts = indices[:, 0]
            if int(starts.min()) < 0 or int(starts.max()) > max_start:
                raise ValueError(
                    "temporal_window_indices must describe windows starting "
                    f"in [0,{max_start}]"
                )
            offsets = torch.arange(
                self.dense_num_frames,
                dtype=torch.long,
                device=device,
            )
            expected = starts[:, None] + offsets[None, :]
            if not bool(torch.equal(indices, expected)):
                raise ValueError(
                    "each temporal_window_indices row must be a contiguous "
                    "strictly increasing window"
                )
        return indices

    def preprocess_dense(
        self,
        video_clip_01: torch.Tensor,
        temporal_window_starts: Optional[torch.Tensor] = None,
        temporal_window_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Official differentiable V-JEPA2 preprocessing.

        The input is ``[B,T,3,H,W]`` in ``[0,1]``.  By default a centered
        contiguous ``dense_num_frames`` window is selected.  Callers may
        instead supply explicit integer starts with shape ``[B]`` or explicit
        contiguous indices with shape ``[B,dense_num_frames]``.  Sampling is
        intentionally external: this adapter never consumes ambient RNG
        state.  The short side is then resized to 292 (configurable), a 256
        center crop is taken, and ImageNet normalization is applied.  The
        returned layout remains ``[B,T,3,256,256]``.
        """
        if video_clip_01.ndim != 5:
            raise ValueError(
                "video_clip_01 must have shape [B,T,3,H,W], "
                f"got {tuple(video_clip_01.shape)}"
            )
        if video_clip_01.shape[2] != 3:
            raise ValueError(
                f"video_clip_01 must have 3 channels, got {video_clip_01.shape[2]}"
            )
        total_frames = int(video_clip_01.shape[1])
        if total_frames < self.dense_num_frames:
            raise ValueError(
                f"V-JEPA2 dense mode needs at least {self.dense_num_frames} frames, "
                f"got {total_frames}"
            )
        frame_indices = self._temporal_window_indices(
            video_clip_01,
            temporal_window_starts=temporal_window_starts,
            temporal_window_indices=temporal_window_indices,
        )
        if frame_indices is None:
            start = (total_frames - self.dense_num_frames) // 2
            clip = video_clip_01.narrow(1, start, self.dense_num_frames)
        else:
            gather_indices = frame_indices[:, :, None, None, None].expand(
                -1,
                -1,
                int(video_clip_01.shape[2]),
                int(video_clip_01.shape[3]),
                int(video_clip_01.shape[4]),
            )
            clip = torch.gather(video_clip_01, 1, gather_indices)
        x = differentiable_resize_crop(
            clip,
            self.input_size,
            crop_mode="center",
            resize_short_side=self.resize_short_side,
            interpolation="bilinear",
            round_center_crop=True,
        )
        B, T, C, H, W = x.shape
        flat = x.reshape(B * T, C, H, W)
        flat = (flat - self.norm_mean.to(flat.dtype)) / self.norm_std.to(flat.dtype)
        return flat.reshape(B, T, C, H, W)

    def forward_dense(
        self,
        video_clip_01: torch.Tensor,
        layer_index: Optional[int] = None,
        token_indices: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
        temporal_window_starts: Optional[torch.Tensor] = None,
        temporal_window_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a raw intermediate dense token matrix from an input video.

        No final encoder LayerNorm, predictor, pooling, or flattening is applied.
        Gradients are retained with respect to ``video_clip_01`` even though all
        V-JEPA2 parameters are frozen.
        """
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
        """Return several raw intermediate token matrices in one prefix pass.

        The encoder is executed exactly once through ``max(layer_indices)``.
        Each returned value is the complete post-residual output of its
        zero-based block, before V-JEPA2's final LayerNorm.  The dictionary
        follows the caller-provided layer order.
        """
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
        """Run embeddings and complete blocks ``0..layer_index``.

        ``pixel_values`` must already have V-JEPA2 layout and normalization.
        This lower-level entrypoint lets replay code avoid preprocessing twice.
        """
        target_layer = self.dense_layer_index if layer_index is None else int(layer_index)
        return self.encode_dense_preprocessed_layers(
            pixel_values,
            layer_indices=(target_layer,),
            token_indices=token_indices,
            gradient_checkpointing=gradient_checkpointing,
        )[target_layer]

    def encode_dense_preprocessed_layers(
        self,
        pixel_values: torch.Tensor,
        layer_indices: Sequence[int],
        token_indices: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """Run one encoder prefix and return all requested block outputs.

        ``pixel_values`` must already have V-JEPA2 layout and normalization.
        Blocks are evaluated only once, through the largest requested
        zero-based layer.  Token selection happens as each requested layer is
        captured so that later blocks do not retain unneeded token outputs.
        """
        if pixel_values.ndim != 5:
            raise ValueError(
                "pixel_values must have shape [B,T,3,H,W], "
                f"got {tuple(pixel_values.shape)}"
            )
        if pixel_values.shape[2] != 3:
            raise ValueError(f"pixel_values must have 3 channels, got {pixel_values.shape[2]}")
        if pixel_values.shape[1] != self.dense_num_frames:
            raise ValueError(
                f"pixel_values must contain {self.dense_num_frames} frames, "
                f"got {pixel_values.shape[1]}"
            )
        if tuple(pixel_values.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(
                f"pixel_values must be {self.input_size}x{self.input_size}, "
                f"got {tuple(pixel_values.shape[-2:])}"
            )

        try:
            requested_layers = tuple(int(layer) for layer in layer_indices)
        except TypeError as exc:
            raise TypeError("layer_indices must be a sequence of integers") from exc
        if not requested_layers:
            raise ValueError("layer_indices must not be empty")
        if len(set(requested_layers)) != len(requested_layers):
            raise ValueError("layer_indices must contain unique indices")
        for target_layer in requested_layers:
            self._validate_dense_layer(target_layer)
        max_target_layer = max(requested_layers)
        if max_target_layer >= self.retained_dense_blocks:
            raise ValueError(
                f"layer_index {max_target_layer} was pruned; retained blocks are "
                f"0..{self.retained_dense_blocks - 1}"
            )

        encoder = self._model.encoder
        hidden_states = encoder.embeddings(pixel_values)
        expected_tokens = self.dense_raw_tokens
        if hidden_states.shape[1] != expected_tokens:
            raise RuntimeError(
                "unexpected V-JEPA2 dense token grid: expected "
                f"{expected_tokens}, got {hidden_states.shape[1]}"
            )
        self._raw_dim = int(hidden_states.shape[-1])

        indices = None
        if token_indices is not None:
            indices = torch.as_tensor(token_indices, device=hidden_states.device)
            if indices.ndim != 1:
                raise ValueError(
                    f"token_indices must be 1-D, got shape {tuple(indices.shape)}"
                )
            if indices.numel() == 0:
                raise ValueError("token_indices must not be empty")
            if indices.dtype != torch.long:
                indices = indices.to(torch.long)
            if int(indices.min()) < 0 or int(indices.max()) >= hidden_states.shape[1]:
                raise IndexError(
                    f"token_indices must be in [0,{hidden_states.shape[1] - 1}]"
                )

        requested_set = set(requested_layers)
        captured: Dict[int, torch.Tensor] = {}
        for block_index in range(max_target_layer + 1):
            block = encoder.layer[block_index]

            def run_block(x: torch.Tensor, module=block) -> torch.Tensor:
                return module(
                    x,
                    position_mask=None,
                    head_mask=None,
                    output_attentions=False,
                )[0]

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
                # This is deliberately before encoder.layernorm:
                # HF hidden_states[i+1] is the complete post-residual result
                # of zero-based block i.
                captured[block_index] = (
                    hidden_states
                    if indices is None
                    else hidden_states.index_select(1, indices)
                )

        return {layer: captured[layer] for layer in requested_layers}

    def prune_for_dense_prefix(self, layer_index: Optional[int] = None):
        """Irreversibly retain only the modules required by dense-prefix mode."""
        target_layer = self.dense_layer_index if layer_index is None else int(layer_index)
        self._validate_dense_layer(target_layer)
        if self._dense_pruned and target_layer >= len(self._model.encoder.layer):
            raise ValueError(
                f"cannot restore pruned layer {target_layer}; only "
                f"{len(self._model.encoder.layer)} blocks remain"
            )
        self._model.encoder.layer = nn.ModuleList(
            list(self._model.encoder.layer[: target_layer + 1])
        )
        # Dense prefix never executes either module.  Replacing them with None
        # unregisters their parameters immediately, which matters before FSDP.
        self._model.encoder.layernorm = None
        self._model.predictor = None
        self._dense_pruned = True
        self.dense_only = True
        self.dense_layer_index = target_layer
        return self

    def _validate_dense_layer(self, layer_index: int) -> None:
        if layer_index < 0:
            raise ValueError(f"dense layer index must be non-negative, got {layer_index}")
        encoder = getattr(self._model, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            raise RuntimeError("loaded V-JEPA2 model does not expose encoder.layer")
        configured_layers = int(
            getattr(getattr(self._model, "config", None), "num_hidden_layers", len(layers))
        )
        if layer_index >= configured_layers:
            raise ValueError(
                f"dense layer index {layer_index} out of range for "
                f"{configured_layers} encoder blocks"
            )

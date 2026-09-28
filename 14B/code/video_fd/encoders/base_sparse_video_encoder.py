"""Base sparse video-encoder adapter + shared differentiable preprocessing.

The replay pass requires every preprocessing op to be differentiable w.r.t. the
input clip ``video_clip_01 ∈ [0,1]`` of shape ``[B, T, 3, H, W]``:

* spatial resize  -> ``F.interpolate`` (bicubic, antialias)
* spatial crop    -> differentiable narrow/index (center or three-crop)
* normalize       -> affine
* layout          -> view/permute

Concrete encoders subclass :class:`SparseVideoEncoderAdapter` and implement
``_encode_tokens`` (raw token/grid output) + ``_grid_shape`` (token grid), and the
base class turns tokens into sparse features via spatial mean + temporal segment
pooling.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def differentiable_resize_crop(
    video_clip_01: torch.Tensor,
    size: int,
    crop_mode: str = "center",
    *,
    resize_short_side: Optional[int] = None,
    interpolation: str = "bicubic",
    round_center_crop: bool = False,
) -> torch.Tensor:
    """Resize+crop ``[B, T, 3, H, W]`` in ``[0,1]`` to ``size x size`` (differentiable).

    Returns ``[n_crops*B, T, 3, size, size]``; ``crop_mode='center'`` gives
    ``n_crops=1``, ``'three_crops'`` gives ``n_crops=3`` (left/center/right for
    wide videos, top/center/bottom for tall ones). Aspect ratio is preserved:
    the shorter side is resized to ``resize_short_side`` (or ``size`` when it is
    omitted) then square crops are taken.

    ``interpolation`` and ``round_center_crop`` exist so an adapter can reproduce
    an upstream processor exactly without introducing a non-differentiable PIL
    round-trip.  Existing sparse adapters retain the historical bicubic,
    floor-offset behavior.
    """
    if video_clip_01.ndim != 5:
        raise ValueError(
            "video_clip_01 must have shape [B,T,3,H,W], "
            f"got {tuple(video_clip_01.shape)}"
        )
    B, T, C, H, W = video_clip_01.shape
    if C != 3:
        raise ValueError(f"video_clip_01 must have 3 channels, got {C}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    resize_size = size if resize_short_side is None else int(resize_short_side)
    if resize_size < size:
        raise ValueError(
            f"resize_short_side ({resize_size}) must be >= crop size ({size})"
        )
    if interpolation not in ("bilinear", "bicubic"):
        raise ValueError(
            f"unsupported interpolation {interpolation!r}; use 'bilinear' or 'bicubic'"
        )

    x = video_clip_01.reshape(B * T, C, H, W)
    # interpolate needs float32+; upcast half/bf16 but preserve float32/float64.
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()

    # Resize the shorter side while preserving aspect ratio.  Hugging Face's
    # shortest-edge processors truncate the scaled long edge; keep the legacy
    # round-to-nearest behavior when no explicit resize size was requested.
    if H <= W:
        scaled_w = W * resize_size / H
        new_h = resize_size
        new_w = int(scaled_w) if resize_short_side is not None else int(round(scaled_w))
    else:
        scaled_h = H * resize_size / W
        new_h = int(scaled_h) if resize_short_side is not None else int(round(scaled_h))
        new_w = resize_size
    new_h, new_w = max(size, new_h), max(size, new_w)
    if (new_h, new_w) != (H, W):
        x = F.interpolate(
            x,
            size=(new_h, new_w),
            mode=interpolation,
            align_corners=False,
            antialias=True,
        )

    crops = _square_crops(
        x,
        size,
        crop_mode,
        round_center_crop=round_center_crop,
    )  # list of [B*T, C, size, size]
    out = torch.stack([c.reshape(B, T, C, size, size) for c in crops], dim=0)  # [n_crops, B, T, C, s, s]
    n_crops = out.shape[0]
    return out.reshape(n_crops * B, T, C, size, size)


def _square_crops(
    x: torch.Tensor,
    size: int,
    crop_mode: str,
    *,
    round_center_crop: bool = False,
) -> List[torch.Tensor]:
    _, _, h, w = x.shape
    if crop_mode == "center":
        offs = [_center_offset(h, w, size, round_offset=round_center_crop)]
    elif crop_mode == "three_crops":
        offs = _three_crop_offsets(h, w, size)
    else:
        raise ValueError(f"unknown crop_mode {crop_mode!r} (use 'center' or 'three_crops')")
    return [x[:, :, top: top + size, left: left + size] for (top, left) in offs]


def _center_offset(
    h: int,
    w: int,
    size: int,
    *,
    round_offset: bool = False,
) -> Tuple[int, int]:
    if round_offset:
        # torchvision.transforms.functional.center_crop uses round rather than
        # floor for odd differences.  V-JEPA2's official processor uses it.
        return max(0, int(round((h - size) / 2.0))), max(
            0, int(round((w - size) / 2.0))
        )
    return max(0, (h - size) // 2), max(0, (w - size) // 2)


def _three_crop_offsets(h: int, w: int, size: int) -> List[Tuple[int, int]]:
    if w >= h:  # wide: left / center / right
        return [(max(0, (h - size) // 2), 0),
                (max(0, (h - size) // 2), max(0, (w - size) // 2)),
                (max(0, (h - size) // 2), max(0, w - size))]
    # tall: top / center / bottom
    return [(0, max(0, (w - size) // 2)),
            (max(0, (h - size) // 2), max(0, (w - size) // 2)),
            (max(0, h - size), max(0, (w - size) // 2))]


def temporal_segment_mean(x: torch.Tensor, segments: int) -> torch.Tensor:
    """Pool ``[B, T, C]`` into ``[B, segments, C]`` by averaging contiguous segments.

    If ``T`` is not divisible by ``segments``, segment boundaries are spread as
    evenly as possible. ``segments==1`` reduces to a global temporal mean.
    """
    B, T, C = x.shape
    if segments <= 1:
        return x.mean(dim=1, keepdim=True)
    if segments >= T:
        return x
    bounds = torch.linspace(0, T, steps=segments + 1).round().long().tolist()
    chunks = [x[:, bounds[i]: bounds[i + 1]].mean(dim=1) for i in range(segments)]
    return torch.stack(chunks, dim=1)


class SparseVideoEncoderAdapter(nn.Module):
    """Unified frozen sparse video encoder.

    Subclasses implement :meth:`_encode_tokens` and :meth:`_grid_shape`. The base
    class owns preprocessing (resize/crop/normalize) and the tokens -> sparse
    ``[B, S, D]`` pooling, with a safe global-pooling fallback when the token grid
    cannot be inferred.
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.name = str(config.get("name", "sparse_video_encoder"))
        self.input_size = int(config.get("input_size", 224))
        self.input_frames = int(config.get("input_frames", config.get("clip_frames", 16)))
        self.sparse_tokens = int(config.get("sparse_tokens", 4))
        self.use_patch_tokens = bool(config.get("use_patch_tokens", True))
        self.sparse_mode = str(config.get("sparse_mode", "temporal"))  # "temporal" | "global"
        self.crop_mode = str(config.get("crop_mode", "center"))
        # Global encoders (e.g. VideoPrism LvT, which pools to a single [B,D]
        # video embedding) produce S>1 segments via segment-forward: the clip is
        # split into S contiguous temporal segments, each temporally resized to
        # the encoder's native frame count and encoded separately, then stacked.
        self.global_encoder = bool(config.get("global_encoder", False))
        # normalization (ImageNet default; subclasses override)
        mean = config.get("norm_mean", (0.485, 0.456, 0.406))
        std = config.get("norm_std", (0.229, 0.224, 0.225))
        self.register_buffer("norm_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("norm_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self._warned_grid = False
        self._raw_dim: Optional[int] = None

    # -- to be provided by concrete adapters --------------------------------
    def _encode_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[B, T, 3, size, size]`` (normalized) -> raw tokens ``[B, N, D]``."""
        raise NotImplementedError

    def _grid_shape(self, n_tokens: int) -> Optional[Tuple[int, int, int]]:
        """Return ``(T', H', W')`` if ``T'·H'·W' == n_tokens``, else ``None``."""
        return None

    def _encode_global(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[nB, T, 3, size, size]`` (normalized) -> global embedding ``[nB, D]``.

        Implemented by global encoders (``global_encoder=True``); ``T`` equals the
        encoder's native frame count (segment-forward resizes to it)."""
        raise NotImplementedError

    # -- shared preprocessing ----------------------------------------------
    def preprocess(self, video_clip_01: torch.Tensor) -> torch.Tensor:
        """``[B, T, 3, H, W]`` in [0,1] -> normalized ``[n_crops*B, T, 3, s, s]``."""
        x = differentiable_resize_crop(video_clip_01, self.input_size, self.crop_mode)
        nB, T, C, h, w = x.shape
        x = x.reshape(nB * T, C, h, w)
        x = (x - self.norm_mean.to(x.dtype)) / self.norm_std.to(x.dtype)
        return x.reshape(nB, T, C, h, w)

    # -- sparse feature extraction -----------------------------------------
    def forward_sparse(self, video_clip_01: torch.Tensor) -> torch.Tensor:
        """``[B, T, 3, H, W]`` in [0,1] -> sparse features ``[B, S, D]``.

        When ``crop_mode='three_crops'`` the crops are averaged back into the
        per-video feature (so the FD sample count is unchanged); this keeps the
        crop policy identical between stats and training.
        """
        pixel_values = self.preprocess(video_clip_01)          # [nB, T, 3, s, s]
        nB = pixel_values.shape[0]
        B = video_clip_01.shape[0]
        if self.global_encoder:
            sparse = self._segment_forward(pixel_values)       # [nB, S, D]
        else:
            tokens = self._encode_tokens(pixel_values)         # [nB, N, D]
            self._raw_dim = int(tokens.shape[-1])
            sparse = self._tokens_to_sparse(tokens)            # [nB, S, D]
        if nB != B:                                            # average crops
            n_crops = nB // B
            sparse = sparse.reshape(n_crops, B, *sparse.shape[1:]).mean(dim=0)
        return sparse

    def _segment_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Split ``[nB, T, 3, s, s]`` into S temporal segments, each resized to the
        native frame count, encode globally, and stack to ``[nB, S, D]``."""
        nB, T, C, s, _ = pixel_values.shape
        S = max(1, self.sparse_tokens) if self.sparse_mode != "global" else 1
        bounds = torch.linspace(0, T, steps=S + 1).round().long().tolist()
        feats = []
        for i in range(S):
            seg = pixel_values[:, bounds[i]: max(bounds[i] + 1, bounds[i + 1])]   # [nB, t_i, 3, s, s]
            seg = self._resize_time(seg, self.input_frames)                        # [nB, native, 3, s, s]
            feats.append(self._encode_global(seg))                                 # [nB, D]
        out = torch.stack(feats, dim=1)                                            # [nB, S, D]
        self._raw_dim = int(out.shape[-1])
        return out

    @staticmethod
    def _resize_time(seg: torch.Tensor, target_t: int) -> torch.Tensor:
        """Differentiably resample the temporal axis of ``[nB, t, 3, s, s]`` to ``target_t``."""
        nB, t, C, h, w = seg.shape
        if t == target_t:
            return seg
        x = seg.permute(0, 2, 3, 4, 1).reshape(nB * C * h * w, 1, t)      # [.., 1, t]
        x = F.interpolate(x, size=target_t, mode="linear", align_corners=False)
        return x.reshape(nB, C, h, w, target_t).permute(0, 4, 1, 2, 3).contiguous()

    def _tokens_to_sparse(self, tokens: torch.Tensor) -> torch.Tensor:
        nB, N, D = tokens.shape
        if self.sparse_mode == "global" or not self.use_patch_tokens:
            return tokens.mean(dim=1, keepdim=True)            # [nB, 1, D]
        grid = self._grid_shape(N)
        if grid is None:
            if not self._warned_grid:
                print(f"[{self.name}] grid inference failed for N={N}; using global token pooling.")
                self._warned_grid = True
            return tokens.mean(dim=1, keepdim=True)
        Tp, Hp, Wp = grid
        x = tokens.reshape(nB, Tp, Hp, Wp, D).mean(dim=(2, 3))  # spatial mean -> [nB, T', D]
        return temporal_segment_mean(x, self.sparse_tokens)     # [nB, S, D]

    @property
    def raw_dim(self) -> Optional[int]:
        return self._raw_dim

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self


class SyntheticSparseVideoEncoder(SparseVideoEncoderAdapter):
    """Tiny deterministic token encoder for tests (no external weights).

    Produces a ``(T', H', W')`` token grid exactly like a tubelet ViT, so the
    base-class reshape -> spatial-mean -> temporal-segment path and the replay VJP
    are exercised end-to-end. Frozen and differentiable w.r.t. the input clip.
    """

    def __init__(self, config):
        config = dict(config)
        config.setdefault("input_size", 64)
        config.setdefault("input_frames", 16)
        super().__init__(config)
        self.tubelet = int(config.get("tubelet_size", 2))
        self.patch = int(config.get("patch_size", 16))
        self.embed_dim = int(config.get("raw_dim", 96))
        # a small frozen 3D patch embed -> tokens
        self.proj = nn.Conv3d(3, self.embed_dim, kernel_size=(self.tubelet, self.patch, self.patch),
                              stride=(self.tubelet, self.patch, self.patch))
        self.act = nn.Tanh()
        torch.manual_seed(0)
        for p in self.parameters():
            nn.init.normal_(p, std=0.05)
        self.freeze()

    def _grid_shape(self, n_tokens: int):
        Tp = self.input_frames // self.tubelet
        Hp = self.input_size // self.patch
        Wp = self.input_size // self.patch
        return (Tp, Hp, Wp) if Tp * Hp * Wp == n_tokens else None

    def _encode_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [nB, T, 3, s, s] -> [nB, 3, T, s, s]
        x = pixel_values.permute(0, 2, 1, 3, 4)
        x = self.act(self.proj(x))                  # [nB, D, T', H', W']
        nB, D, Tp, Hp, Wp = x.shape
        return x.reshape(nB, D, Tp * Hp * Wp).permute(0, 2, 1)  # [nB, N, D]

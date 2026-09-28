"""Chunked VAE decoding for replay video FD-loss.

The decoder is treated as an opaque callable ``vae_decode_fn(z_chunk) ->
video_chunk`` so the same code path works for a tiny synthetic decoder (unit
tests) and the real Wan VAE (training).  Both tensors follow the channels-first
temporal convention ``[B, C, T, H, W]`` (``time_dim=2``).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional

import torch

from utils.video_chunking import TemporalChunk, scatter_add_latent_grad, select_latent_chunk

VaeDecodeFn = Callable[[torch.Tensor], torch.Tensor]


def _amp_context(use_amp: bool, amp_dtype: torch.dtype, device_type: str):
    if use_amp and device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def default_preprocess_fn(video: torch.Tensor) -> torch.Tensor:
    """Convert decoded video from [-1,1] to [0,1] as in model/fd.py."""
    return ((video + 1.0) * 0.5).clamp(0.0, 1.0)


def make_wan_vae_decode_fn(vae_decoder) -> VaeDecodeFn:
    """Wrap WanVAEWrapper (latent [B,T,C,H,W]) as BCTHW decode_fn."""

    def decode_fn(z_bcthw: torch.Tensor) -> torch.Tensor:
        z_btchw = z_bcthw.permute(0, 2, 1, 3, 4)
        video_btchw = vae_decoder.decode_to_pixel(z_btchw)
        return video_btchw.permute(0, 2, 1, 3, 4)

    return decode_fn


def btchw_to_bcthw(z: torch.Tensor) -> torch.Tensor:
    return z.permute(0, 2, 1, 3, 4)


def bcthw_to_btchw(z: torch.Tensor) -> torch.Tensor:
    return z.permute(0, 2, 1, 3, 4)


def vae_decode_chunk(
    z_chunk: torch.Tensor,
    vae_decode_fn: VaeDecodeFn,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode a single latent chunk ``[B, C_latent, T_ctx, H, W]`` to pixels."""
    device_type = z_chunk.device.type
    with _amp_context(use_amp, amp_dtype, device_type):
        video_chunk = vae_decode_fn(z_chunk)
    return video_chunk


def crop_center_pixels(
    video_chunk: torch.Tensor,
    chunk: TemporalChunk,
    time_dim: int = 2,
) -> torch.Tensor:
    """Crop a decoded context chunk to its core window's pixel frames."""
    t_total = video_chunk.shape[time_dim]
    start = max(0, min(chunk.center_t_start, t_total))
    end = max(start, min(chunk.center_t_end, t_total))
    if start == 0 and end == t_total:
        return video_chunk
    if end <= start:
        end = min(t_total, start + 1)
    return video_chunk.narrow(time_dim, start, end - start)


def decode_window(
    z_chunk: torch.Tensor,
    chunk: TemporalChunk,
    vae_decode_fn: VaeDecodeFn,
    preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    time_dim: int = 2,
) -> torch.Tensor:
    """Decode a latent context chunk and return the preprocessed center window."""
    video_chunk = vae_decode_chunk(z_chunk, vae_decode_fn, use_amp=use_amp, amp_dtype=amp_dtype)
    video_center = crop_center_pixels(video_chunk, chunk, time_dim=time_dim)
    if preprocess_fn is not None:
        video_center = preprocess_fn(video_center)
    return video_center


__all__ = [
    "VaeDecodeFn",
    "default_preprocess_fn",
    "make_wan_vae_decode_fn",
    "btchw_to_bcthw",
    "bcthw_to_btchw",
    "vae_decode_chunk",
    "decode_window",
    "crop_center_pixels",
    "select_latent_chunk",
    "scatter_add_latent_grad",
]

"""Chunked video feature extraction and the replay VJP.

Two passes share one decode+crop+extract routine so their features are bit-for-bit
identical (the determinism that makes replay exact):

* :func:`extract_video_features_no_grad` -- *Stage A's* memory-safe first pass.
  Decodes each temporal window (with optional overlap context), crops the center
  region, extracts one window feature, all under ``torch.no_grad``.  Returns the
  flattened features ``[N, D]`` and the ``feature_map`` (one entry per window).

* :func:`replay_feature_vjp_to_latents` -- *Stage B*.  Re-decodes each window
  **with gradients enabled on the latent chunk only**, forms the linear surrogate
  ``<f_k, sg(g_{f,k})>``, takes its gradient w.r.t. the latent chunk, and
  scatter-adds it into ``grad_z``.  Summed over windows this equals
  ``(dF/dz)^T g_f`` -- the latent gradient the generator needs.

:func:`extract_window_features_from_video` is the naive full-graph reference used
by ``losses/video_fd.py`` and the gradient-equivalence test.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple

import torch

from utils.chunked_vae_decode import crop_center_pixels, decode_window, vae_decode_chunk
from utils.video_chunking import (
    TemporalChunk,
    TemporalChunker,
    scatter_add_latent_grad,
    select_latent_chunk,
)


def flatten_feature_output(feat: torch.Tensor) -> torch.Tensor:
    """Normalise a feature-extractor output to 2D ``[rows, D]``.

    * ``[B, D]``      -> unchanged (one feature per batch item / window).
    * ``[B, N, D]``   -> ``[B*N, D]`` (N sub-features per window), row-major.
    * ``[B, ..., D]`` -> all leading dims flattened, row-major.

    The flatten order is row-major and identical across passes, so the produced
    rows always line up with the feature map.
    """
    if feat.ndim < 2:
        raise ValueError(f"feature output must be >=2D [.., D], got shape {tuple(feat.shape)}")
    if feat.ndim == 2:
        return feat
    return feat.reshape(-1, feat.shape[-1])


def select_feature_grad(grad_features: torch.Tensor, chunk: TemporalChunk) -> torch.Tensor:
    """Slice the feature-gradient rows produced by ``chunk`` (window-major)."""
    return grad_features[chunk.feature_start:chunk.feature_end]


@torch.no_grad()
def extract_video_features_no_grad(
    z: torch.Tensor,
    vae_decoder: Callable[[torch.Tensor], torch.Tensor],
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    chunker: TemporalChunker,
    preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    time_dim: int = 2,
) -> Tuple[torch.Tensor, List[TemporalChunk]]:
    """Decode + extract per-window features with no autograd graph.

    Returns ``(features [N, D], feature_map)`` where ``feature_map`` is a list of
    :class:`TemporalChunk` whose ``feature_start/end`` exactly index the returned
    rows (recomputed from the actual extractor output, so any features-per-window
    is supported).
    """
    batch_size = z.shape[0]
    geometry = chunker.chunks(batch_size=batch_size, features_per_window=1)

    features: List[torch.Tensor] = []
    feature_map: List[TemporalChunk] = []
    offset = 0
    for chunk in geometry:
        z_chunk = select_latent_chunk(z, chunk, time_dim=time_dim)
        video_center = decode_window(
            z_chunk, chunk, vae_decoder, preprocess_fn,
            use_amp=use_amp, amp_dtype=amp_dtype, time_dim=time_dim,
        )
        feat_chunk = flatten_feature_output(feature_extractor(video_center)).float()
        rows = feat_chunk.shape[0]
        chunk.feature_start = offset
        chunk.feature_end = offset + rows
        offset += rows
        features.append(feat_chunk)
        feature_map.append(chunk)

    features_all = torch.cat(features, dim=0)
    return features_all, feature_map


def replay_feature_vjp_to_latents(
    z_detached: torch.Tensor,
    grad_features: torch.Tensor,
    vae_decoder: Callable[[torch.Tensor], torch.Tensor],
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    chunker: TemporalChunker,
    preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    feature_map: Optional[List[TemporalChunk]] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    time_dim: int = 2,
) -> torch.Tensor:
    """Replay the feature-space VJP back to a full latent gradient ``grad_z``.

    For each window: decode the latent context chunk with grad, crop+preprocess
    the center, extract the feature, form ``replay_loss = <f_k, sg(g_{f,k})>`` and
    accumulate ``d(replay_loss)/d(z_chunk)`` into ``grad_z``.  No
    ``torch.no_grad`` is used here -- gradients must flow through the (frozen)
    decoder/extractor ops down to the latent chunk.
    """
    if feature_map is None:
        feature_map = chunker.chunks(batch_size=z_detached.shape[0], features_per_window=1)

    grad_z = torch.zeros_like(z_detached)
    grad_features = grad_features.detach()

    for chunk in feature_map:
        # Independent latent leaf for this window (z_detached carries no graph).
        z_chunk = select_latent_chunk(z_detached, chunk, time_dim=time_dim)
        z_chunk = z_chunk.detach().clone().requires_grad_(True)

        video_center = decode_window(
            z_chunk, chunk, vae_decoder, preprocess_fn,
            use_amp=use_amp, amp_dtype=amp_dtype, time_dim=time_dim,
        )
        feat_chunk = flatten_feature_output(feature_extractor(video_center)).float()

        g_chunk = select_feature_grad(grad_features, chunk).to(feat_chunk.dtype)
        if g_chunk.shape != feat_chunk.shape:
            raise RuntimeError(
                f"feature/grad shape mismatch for window {chunk.index}: "
                f"feat {tuple(feat_chunk.shape)} vs grad {tuple(g_chunk.shape)}. "
                "Check that the replay feature map matches the no-grad pass."
            )

        replay_loss = (feat_chunk * g_chunk).sum()
        grad_chunk = torch.autograd.grad(
            replay_loss,
            z_chunk,
            retain_graph=False,
            create_graph=False,
        )[0]
        scatter_add_latent_grad(
            grad_z,
            grad_chunk.to(grad_z.dtype),
            chunk,
            time_dim=time_dim,
            center_only=chunker.use_center_crop,
        )

    return grad_z


def extract_window_features_from_video(
    video_full: torch.Tensor,
    chunker: TemporalChunker,
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    time_dim: int = 2,
    enable_grad: bool = True,
) -> Tuple[torch.Tensor, List[TemporalChunk]]:
    """Naive reference: window features from an already-decoded full video.

    Unlike the chunked path this crops each window's **global** pixel range out of
    the full decode (the exact receptive field), so it is the ground-truth the
    chunked replay is validated against.  Used by ``losses/video_fd.py`` and the
    gradient-equivalence test.  Keeps the full graph when ``enable_grad`` so a
    single ``loss.backward()`` reaches the latents.
    """
    batch_size = video_full.shape[0]
    geometry = chunker.chunks(batch_size=batch_size, features_per_window=1)
    ctx = nullcontext() if enable_grad else torch.no_grad()

    features: List[torch.Tensor] = []
    feature_map: List[TemporalChunk] = []
    offset = 0
    with ctx:
        for chunk in geometry:
            start = chunk.pixel_t_start
            end = min(chunk.pixel_t_end, video_full.shape[time_dim])
            window = video_full.narrow(time_dim, start, max(1, end - start))
            if preprocess_fn is not None:
                window = preprocess_fn(window)
            feat_chunk = flatten_feature_output(feature_extractor(window)).float()
            rows = feat_chunk.shape[0]
            chunk.feature_start = offset
            chunk.feature_end = offset + rows
            offset += rows
            features.append(feat_chunk)
            feature_map.append(chunk)

    return torch.cat(features, dim=0), feature_map

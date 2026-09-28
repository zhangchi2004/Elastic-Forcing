"""Temporal chunking utilities for memory-efficient ("replay") video FD-loss.

The replay FD-loss never keeps the full decoder / feature-extractor graph alive.
Instead it decodes the latent video in *temporal windows* (chunks), extracts one
feature per window, and -- during the gradient replay -- recomputes each window's
decode/feature graph one at a time.  This module owns everything related to the
temporal geometry of those windows:

* :class:`TemporalMap`     -- the latent <-> pixel temporal-frame mapping.
* :class:`TemporalChunk`   -- one window's bookkeeping (mirrors the ``feature_map``
                              dict described in the design spec).
* :class:`TemporalChunker` -- enumerates the windows for a given latent length.
* :func:`select_latent_chunk` / :func:`scatter_add_latent_grad` -- the two
                              axis-aware tensor ops used by the replay VJP.

Tensor convention (matches the design spec *and* the native Wan VAE ``decode``):

    latent video : ``[B, C_latent, T_latent, H_latent, W_latent]``   (time_dim=2)
    decoded video: ``[B, C_rgb,    T_pixel,  H,        W]``           (time_dim=2)

The repo's :class:`~utils.wan_wrapper.WanVAEWrapper` stores latents as
``[B, T, C, H, W]``; convert with a single ``permute(0, 2, 1, 3, 4)`` before
handing tensors to these utilities (see ``experiments/train_video_fd_replay.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch


@dataclass
class TemporalMap:
    """Maps latent temporal indices to decoded-pixel temporal indices.

    Two regimes are supported:

    * ``independent_first_frame=False`` (default): every latent frame expands to
      ``ratio`` pixel frames, so latent ``j`` -> pixels ``[ratio*j, ratio*(j+1))``
      and ``T_pixel == ratio * T_latent``.  With ``ratio == 1`` this is the
      identity used by the gradient-equivalence test.
    * ``independent_first_frame=True`` (Wan-style causal VAE): the first latent
      frame decodes to a single pixel frame and every subsequent latent frame to
      ``ratio`` pixel frames, so ``T_pixel == 1 + ratio*(T_latent-1)``.

    The mapping is only an *exact* description of an isolated chunk decode when the
    decoder is frame-local; for causal decoders that re-initialise temporal state
    on each isolated chunk it is approximate at chunk boundaries (see the module
    docstring of :mod:`utils.chunked_vae_decode`).
    """

    ratio: int = 1
    independent_first_frame: bool = False

    def latent_start_to_pixel_start(self, t_latent: int) -> int:
        if self.independent_first_frame:
            if t_latent <= 0:
                return 0
            return 1 + self.ratio * (t_latent - 1)
        return self.ratio * t_latent

    def latent_count_to_pixel_count(self, n_latent: int, includes_first_frame: bool) -> int:
        if n_latent <= 0:
            return 0
        if self.independent_first_frame and includes_first_frame:
            return 1 + self.ratio * (n_latent - 1)
        return self.ratio * n_latent

    def total_pixel_frames(self, t_latent: int) -> int:
        return self.latent_count_to_pixel_count(t_latent, includes_first_frame=True)


@dataclass
class TemporalChunk:
    """Bookkeeping for one temporal window.

    Field names follow the ``feature_map`` schema from the design spec so a list
    of these objects (or their :meth:`to_dict`) *is* the feature map.

    * ``latent_t_start/end``     -- the *core* latent window this feature "owns".
    * ``ctx_latent_t_start/end`` -- the (possibly larger, overlap-padded) latent
                                     range that is actually decoded.
    * ``center_t_start/end``     -- pixel-frame range, **local to the decoded
                                     context chunk**, used to crop the window video
                                     before feature extraction.
    * ``pixel_t_start/end``      -- the *global* pixel range of the core (for
                                     real-video alignment / debugging).
    * ``feature_start/end``      -- rows in the flattened feature tensor produced
                                     by this window (window-major layout).
    """

    index: int
    feature_start: int
    feature_end: int
    latent_t_start: int
    latent_t_end: int
    ctx_latent_t_start: int
    ctx_latent_t_end: int
    center_t_start: int
    center_t_end: int
    pixel_t_start: int
    pixel_t_end: int
    batch_size: int
    batch_indices: Optional[List[int]] = None

    @property
    def ctx_latent_len(self) -> int:
        return self.ctx_latent_t_end - self.ctx_latent_t_start

    @property
    def latent_len(self) -> int:
        return self.latent_t_end - self.latent_t_start

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalChunker:
    """Enumerate temporal windows over the latent time axis.

    Parameters
    ----------
    T_latent:
        number of latent temporal frames.
    window_size:
        core window length, in **latent** frames.
    stride:
        hop between consecutive window cores, in latent frames.  ``stride ==
        window_size`` tiles the axis exactly (recommended / exact); ``stride <
        window_size`` produces overlapping windows (a larger FD population).
    overlap:
        extra latent frames decoded on *each* side of the core to cover the
        decoder's temporal receptive field.  Set ``overlap >= receptive_radius``
        for an exact chunked replay; ``overlap=0`` is fine for frame-independent
        decoders.
    use_center_crop:
        if ``True`` only the core region's gradient is scattered back (the spec's
        approximate boundary handling).  Default ``False`` scatters the full
        decoded-context gradient, which is the *exact* per-window VJP.
    temporal_ratio / independent_first_frame:
        forwarded to :class:`TemporalMap`.
    latent_time_dim:
        which tensor axis is the latent temporal axis (2 for ``[B,C,T,H,W]``).
    drop_last:
        drop a trailing window shorter than ``window_size`` (default keeps it,
        clamped to ``T_latent``).
    """

    def __init__(
        self,
        T_latent: int,
        window_size: int,
        stride: int,
        overlap: int = 0,
        use_center_crop: bool = False,
        temporal_ratio: int = 1,
        independent_first_frame: bool = False,
        latent_time_dim: int = 2,
        drop_last: bool = False,
    ):
        if T_latent <= 0:
            raise ValueError(f"T_latent must be positive, got {T_latent}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")

        self.T_latent = int(T_latent)
        self.window_size = int(min(window_size, T_latent))
        self.stride = int(stride)
        self.overlap = int(overlap)
        self.use_center_crop = bool(use_center_crop)
        self.latent_time_dim = int(latent_time_dim)
        self.drop_last = bool(drop_last)
        self.temporal_map = TemporalMap(
            ratio=int(temporal_ratio),
            independent_first_frame=bool(independent_first_frame),
        )

    # -- window enumeration -------------------------------------------------
    def _core_starts(self) -> List[int]:
        starts: List[int] = []
        t = 0
        while t < self.T_latent:
            if t + self.window_size > self.T_latent:
                if self.drop_last and starts:
                    break
                # final clamped window so the tail is always covered exactly once
                start = max(0, self.T_latent - self.window_size)
                if not starts or start != starts[-1]:
                    starts.append(start)
                break
            starts.append(t)
            t += self.stride
        if not starts:
            starts.append(0)
        return starts

    def chunks(self, batch_size: int = 1, features_per_window: int = 1) -> List[TemporalChunk]:
        """Return the ordered list of windows (a.k.a. the feature map).

        ``features_per_window`` is the number of feature rows the extractor emits
        per batch item for one window (1 for a ``[B, D]`` extractor).  Feature
        rows use a **window-major** layout: window ``k`` occupies the contiguous
        block ``[k * B * fpw, (k+1) * B * fpw)`` so :func:`select_feature_grad`
        is a simple slice.
        """
        rows_per_window = batch_size * features_per_window
        chunks: List[TemporalChunk] = []
        for k, t0 in enumerate(self._core_starts()):
            t1 = min(t0 + self.window_size, self.T_latent)
            ctx0 = max(0, t0 - self.overlap)
            ctx1 = min(self.T_latent, t1 + self.overlap)

            # pixel range of the core, both globally and local to the ctx decode.
            pixel_core_start = self.temporal_map.latent_start_to_pixel_start(t0)
            pixel_core_end = self.temporal_map.latent_start_to_pixel_start(t1)
            pixel_ctx_start = self.temporal_map.latent_start_to_pixel_start(ctx0)
            center_start = pixel_core_start - pixel_ctx_start
            center_end = pixel_core_end - pixel_ctx_start

            chunks.append(
                TemporalChunk(
                    index=k,
                    feature_start=k * rows_per_window,
                    feature_end=(k + 1) * rows_per_window,
                    latent_t_start=t0,
                    latent_t_end=t1,
                    ctx_latent_t_start=ctx0,
                    ctx_latent_t_end=ctx1,
                    center_t_start=center_start,
                    center_t_end=center_end,
                    pixel_t_start=pixel_core_start,
                    pixel_t_end=pixel_core_end,
                    batch_size=batch_size,
                )
            )
        return chunks

    @property
    def num_windows(self) -> int:
        return len(self._core_starts())

    def feature_map(self, batch_size: int = 1, features_per_window: int = 1) -> List[dict]:
        return [c.to_dict() for c in self.chunks(batch_size, features_per_window)]


def select_latent_chunk(
    z: torch.Tensor,
    chunk: TemporalChunk,
    time_dim: int = 2,
) -> torch.Tensor:
    """Slice the decoded-context latent range for ``chunk`` along ``time_dim``."""
    start = chunk.ctx_latent_t_start
    length = chunk.ctx_latent_len
    return z.narrow(time_dim, start, length)


def scatter_add_latent_grad(
    grad_z: torch.Tensor,
    grad_chunk: torch.Tensor,
    chunk: TemporalChunk,
    time_dim: int = 2,
    center_only: bool = False,
) -> torch.Tensor:
    """Accumulate ``grad_chunk`` into ``grad_z`` at ``chunk``'s temporal range.

    ``grad_chunk`` is the gradient w.r.t. the decoded-context latent slice (i.e.
    the output of :func:`select_latent_chunk`).  With ``center_only=False``
    (default) the full context-range gradient is accumulated, which is the exact
    per-window VJP; with ``center_only=True`` only the core region is kept (the
    spec's approximate boundary handling).
    """
    if not center_only:
        grad_z.narrow(time_dim, chunk.ctx_latent_t_start, chunk.ctx_latent_len).add_(grad_chunk)
        return grad_z

    # keep only the core [latent_t_start, latent_t_end) slice of the context grad
    core_offset = chunk.latent_t_start - chunk.ctx_latent_t_start
    core_len = chunk.latent_len
    core_grad = grad_chunk.narrow(time_dim, core_offset, core_len)
    grad_z.narrow(time_dim, chunk.latent_t_start, core_len).add_(core_grad)
    return grad_z

"""Stride-4 latent-aligned contiguous-clip sampler.

From a ``num_pixel_frames``-frame, ``source_fps``-fps video, select a phase
``o ~ Uniform{0,1,2,3}`` and take every ``stride``-th frame
(``I = {o + stride*k}``), giving an effective ``source_fps/stride`` fps stream;
then take a random contiguous ``clip_frames``-frame window of that stream.

For the canonical setting (81 frames @ 16 fps, stride 4, clip 16) this yields 16
frames at an effective 4 fps. The *same* sampler is used for target-statistic
precomputation, generated training, and evaluation (random phase + random valid
start), with an optional deterministic mode for tests/debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


def stride4_candidate_indices(phase: int, stride: int, num_pixel_frames: int) -> List[int]:
    """All frame indices ``phase + stride*k`` within ``[0, num_pixel_frames-1]``."""
    last = num_pixel_frames - 1
    return [phase + stride * k for k in range(num_pixel_frames) if phase + stride * k <= last]


@dataclass
class ClipSpec:
    """One sampled clip: a set of frame indices applied across the batch.

    ``crop_id`` / ``encoder_name`` are optional bookkeeping fields set by the
    caller (the spatial crop is applied by the encoder adapter, not here).
    """

    phase: int
    start: int
    frame_indices: List[int]
    stride: int = 4
    clip_frames: int = 16
    crop_id: Optional[str] = None
    encoder_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "phase": int(self.phase),
            "start": int(self.start),
            "frame_indices": list(self.frame_indices),
            "stride": int(self.stride),
            "clip_frames": int(self.clip_frames),
            "crop_id": self.crop_id,
            "encoder_name": self.encoder_name,
        }

    def index_tensor(self, device=None) -> torch.Tensor:
        return torch.tensor(self.frame_indices, dtype=torch.long, device=device)


class Stride4ClipSampler:
    """Sampler for stride-4 contiguous clips.

    Parameters
    ----------
    num_pixel_frames, source_fps, stride, clip_frames:
        temporal geometry (defaults: 81, 16, 4, 16 -> 16 frames @ 4 fps).
    random_phase / random_start:
        if ``False`` and a ``deterministic`` value is given, use it.
    deterministic / phase / start:
        debug mode -- always return ``frame_indices`` for the fixed phase/start.
    """

    def __init__(
        self,
        num_pixel_frames: int = 81,
        source_fps: int = 16,
        stride: int = 4,
        clip_frames: int = 16,
        random_phase: bool = True,
        random_start: bool = True,
        deterministic: bool = False,
        phase: int = 0,
        start: int = 0,
    ):
        if clip_frames <= 0:
            raise ValueError("clip_frames must be positive")
        self.num_pixel_frames = int(num_pixel_frames)
        self.source_fps = int(source_fps)
        self.stride = int(stride)
        self.clip_frames = int(clip_frames)
        self.random_phase = bool(random_phase)
        self.random_start = bool(random_start)
        self.deterministic = bool(deterministic)
        self.fixed_phase = int(phase)
        self.fixed_start = int(start)

        # validate feasibility: the shortest phase stream must still hold a clip
        min_candidates = min(
            len(stride4_candidate_indices(p, self.stride, self.num_pixel_frames))
            for p in range(self.stride)
        )
        if min_candidates < self.clip_frames:
            raise ValueError(
                f"stride={self.stride}, clip_frames={self.clip_frames} infeasible: "
                f"some phase yields only {min_candidates} candidate frames "
                f"(< {self.clip_frames}) for {self.num_pixel_frames} pixel frames"
            )

    @property
    def effective_fps(self) -> float:
        return self.source_fps / self.stride

    def _sample_one(self, generator: Optional[torch.Generator]) -> ClipSpec:
        if self.deterministic:
            phase = self.fixed_phase
            start = self.fixed_start
        else:
            if self.random_phase:
                phase = int(torch.randint(0, self.stride, (), generator=generator).item())
            else:
                phase = self.fixed_phase
            candidate = stride4_candidate_indices(phase, self.stride, self.num_pixel_frames)
            max_start = len(candidate) - self.clip_frames
            if self.random_start and max_start > 0:
                start = int(torch.randint(0, max_start + 1, (), generator=generator).item())
            else:
                start = min(self.fixed_start, max_start) if not self.random_start else 0
        candidate = stride4_candidate_indices(phase, self.stride, self.num_pixel_frames)
        max_start = len(candidate) - self.clip_frames
        start = max(0, min(start, max_start))
        frame_indices = candidate[start: start + self.clip_frames]
        return ClipSpec(
            phase=phase,
            start=start,
            frame_indices=frame_indices,
            stride=self.stride,
            clip_frames=self.clip_frames,
        )

    def sample(
        self,
        batch_size: int = 1,
        num_clips: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> List[ClipSpec]:
        """Return ``num_clips`` clip specs (each applied across the whole batch).

        ``batch_size`` is accepted for API symmetry; the same frame indices are
        applied to all batch items of a given clip (see ``gather_frames``).
        """
        return [self._sample_one(generator) for _ in range(max(1, num_clips))]

    def config_dict(self) -> dict:
        """Sampler config to embed in every statistics file (spec section 1.2)."""
        return {
            "source_fps": self.source_fps,
            "num_pixel_frames": self.num_pixel_frames,
            "stride": self.stride,
            "effective_fps": self.effective_fps,
            "clip_frames": self.clip_frames,
            "phase_distribution": "uniform_0_1_2_3" if self.stride == 4 else f"uniform_0_{self.stride - 1}",
            "start_distribution": "uniform_valid_contiguous_start",
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "Stride4ClipSampler":
        return cls(
            num_pixel_frames=int(cfg.get("num_pixel_frames", 81)),
            source_fps=int(cfg.get("source_fps", 16)),
            stride=int(cfg.get("stride", 4)),
            clip_frames=int(cfg.get("clip_frames", 16)),
            random_phase=bool(cfg.get("random_phase", True)),
            random_start=bool(cfg.get("random_start", True)),
            deterministic=bool(cfg.get("deterministic", cfg.get("deterministic_debug", False))),
            phase=int(cfg.get("phase", 0)),
            start=int(cfg.get("start", 0)),
        )


def gather_frames(video: torch.Tensor, frame_indices, time_dim: int = 1) -> torch.Tensor:
    """Select ``frame_indices`` along ``time_dim`` of ``video`` (differentiable)."""
    if not torch.is_tensor(frame_indices):
        frame_indices = torch.tensor(list(frame_indices), dtype=torch.long, device=video.device)
    else:
        frame_indices = frame_indices.to(device=video.device, dtype=torch.long)
    return video.index_select(time_dim, frame_indices)


def scatter_add_frames(target: torch.Tensor, frame_indices, values: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
    """Accumulate ``values`` into ``target`` at ``frame_indices`` along ``time_dim``.

    Uses ``index_add`` so that a pixel frame selected by multiple clips / crops /
    encoders correctly accumulates gradients (spec section 11.3).
    """
    if not torch.is_tensor(frame_indices):
        frame_indices = torch.tensor(list(frame_indices), dtype=torch.long, device=target.device)
    else:
        frame_indices = frame_indices.to(device=target.device, dtype=torch.long)
    target.index_add_(time_dim, frame_indices, values.to(target.dtype))
    return target

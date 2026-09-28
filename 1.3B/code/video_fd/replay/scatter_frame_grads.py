"""Frame gather / scatter-add for replay over arbitrary selected frame indices.

The stride-4 sampler selects interleaved pixel frames; the replay must (a) gather
those frames into a 16-frame clip and (b) scatter the clip's pixel gradient back
into the full 81-frame buffer, **accumulating** when a frame is selected by
multiple clips / crops / encoders (spec section 11.3).
"""

from __future__ import annotations

import torch

# Re-export the canonical implementations from the sampler module so there is a
# single source of truth for the gather/scatter rule.
from video_fd.samplers.stride4_clip_sampler import gather_frames, scatter_add_frames

__all__ = ["gather_frames", "scatter_add_frames", "new_frame_grad_buffer"]


def new_frame_grad_buffer(video: torch.Tensor) -> torch.Tensor:
    """Zero gradient buffer matching a ``[B, T, 3, H, W]`` video (float32)."""
    return torch.zeros_like(video, dtype=torch.float32)

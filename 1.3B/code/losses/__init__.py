"""Video FD-loss implementations.

* :mod:`losses.video_fd`        -- the naive (full-graph) video FD loss, kept as
  the ground-truth reference for tiny-scale gradient-equivalence tests.
* :mod:`losses.replay_video_fd` -- the memory-efficient feature-gradient *replay*
  FD loss, plus the generated-feature queue.
"""

from losses.video_fd import VideoFDLoss, naive_video_fd_loss
from losses.replay_video_fd import (
    FeatureQueue,
    MultiRepReplayVideoFDLoss,
    ReplayVideoFDLoss,
)

__all__ = [
    "VideoFDLoss",
    "naive_video_fd_loss",
    "FeatureQueue",
    "ReplayVideoFDLoss",
    "MultiRepReplayVideoFDLoss",
]

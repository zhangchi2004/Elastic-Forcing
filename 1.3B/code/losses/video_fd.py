"""Naive (full-graph) video FD loss.

This is the straightforward implementation that backprops through the *entire*
``generator -> VAE decode -> feature extractor -> FD`` graph.  It OOMs on real
videos, but is exact and is therefore retained as the ground-truth reference for
the gradient-equivalence test and any tiny-scale debugging.  Do **not** use it
for full-scale training -- use :class:`losses.replay_video_fd.ReplayVideoFDLoss`.

It deliberately reuses the same building blocks as the replay path (the same
``TemporalChunker`` window definition, the same ``fd_loss_from_features``) so the
only difference under test is *how* the latent gradient is obtained, not *what*
features/objective are used.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional, Tuple

import torch

from utils.fd_feature_grad import fd_loss_from_features
from utils.video_chunking import TemporalChunker
from utils.video_feature_extract import extract_window_features_from_video


class VideoFDLoss:
    """Naive full-graph video FD loss (reference implementation).

    Parameters mirror :class:`losses.replay_video_fd.ReplayVideoFDLoss`.  The
    decoder/extractor are frozen and put in eval mode; only the latent input (and
    hence the generator) carries gradients.
    """

    def __init__(
        self,
        vae_decoder: Callable[[torch.Tensor], torch.Tensor],
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        real_stats: dict,
        chunker: TemporalChunker,
        preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        cov_eps: float = 1e-4,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        time_dim: int = 2,
    ):
        self.vae_decoder = vae_decoder
        self.feature_extractor = feature_extractor
        self.real_stats = real_stats
        self.chunker = chunker
        self.preprocess_fn = preprocess_fn
        self.cov_eps = float(real_stats.get("cov_eps", cov_eps))
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype
        self.time_dim = int(time_dim)
        self._freeze(vae_decoder)
        self._freeze(feature_extractor)

    @staticmethod
    def _freeze(module) -> None:
        if isinstance(module, torch.nn.Module):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    def extract_features(self, z: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Decode the full video and extract windowed features (full graph)."""
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if (self.use_amp and z.device.type == "cuda")
            else nullcontext()
        )
        with amp_ctx:
            video_full = self.vae_decoder(z)
        features, feature_map = extract_window_features_from_video(
            video_full=video_full,
            chunker=self.chunker,
            feature_extractor=self.feature_extractor,
            preprocess_fn=self.preprocess_fn,
            time_dim=self.time_dim,
            enable_grad=True,
        )
        return features, feature_map

    def compute(
        self,
        z: torch.Tensor,
        queue_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Return ``(loss, log_dict)``; ``loss.backward()`` reaches ``z``."""
        features, feature_map = self.extract_features(z)
        device = features.device
        real_mean = self.real_stats["mean"].to(device=device, dtype=torch.float32)
        real_cov = self.real_stats["cov"].to(device=device, dtype=torch.float32)
        real_cov_sqrt = self.real_stats.get("cov_sqrt")
        if real_cov_sqrt is not None:
            real_cov_sqrt = real_cov_sqrt.to(device=device, dtype=torch.float32)

        loss = fd_loss_from_features(
            current_features=features,
            real_mean=real_mean,
            real_cov=real_cov,
            queue_features=queue_features,
            cov_eps=self.cov_eps,
            real_cov_sqrt=real_cov_sqrt,
        )
        log_dict = {
            "loss/fd": loss.detach(),
            "num_features": features.shape[0],
        }
        return loss, log_dict


def naive_video_fd_loss(
    z: torch.Tensor,
    vae_decoder: Callable[[torch.Tensor], torch.Tensor],
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    real_stats: dict,
    chunker: TemporalChunker,
    preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    cov_eps: float = 1e-4,
    queue_features: Optional[torch.Tensor] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    time_dim: int = 2,
) -> Tuple[torch.Tensor, dict]:
    """Functional wrapper around :class:`VideoFDLoss`."""
    loss_fn = VideoFDLoss(
        vae_decoder=vae_decoder,
        feature_extractor=feature_extractor,
        real_stats=real_stats,
        chunker=chunker,
        preprocess_fn=preprocess_fn,
        cov_eps=cov_eps,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        time_dim=time_dim,
    )
    return loss_fn.compute(z, queue_features=queue_features)

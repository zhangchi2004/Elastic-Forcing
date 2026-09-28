"""Memory-efficient video FD loss via feature-gradient replay.

The replay loss computes the latent gradient

    g_z = (dF/dz)^T g_f,      F(z) = phi(D_VAE(z)),   g_f = dL_FD/df

*without* ever keeping the full ``decode -> feature`` graph alive.  Three stages:

  Stage A.1  features = F(z)               (chunked, ``torch.no_grad``)
  Stage A.2  g_f       = dL_FD / df        (feature space only, no decoder graph)
  Stage B    g_z       = (dF/dz)^T g_f     (per-window decode replay + VJP)

The generator graph stays alive from ``z = generator(...)`` until the caller runs
``z.backward(grad_z)``; only small per-window decoder graphs are ever built, and
they are freed immediately after each window's VJP.

See the module docstrings of :mod:`utils.video_feature_extract`,
:mod:`utils.fd_feature_grad` and :mod:`utils.chunked_vae_decode` for the details.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from utils.fd_feature_grad import compute_fd_feature_gradient, tensor_health
from utils.video_chunking import TemporalChunker
from utils.video_feature_extract import (
    extract_video_features_no_grad,
    replay_feature_vjp_to_latents,
)


class FeatureQueue:
    """Ring buffer of detached generated features (one per representation).

    Stores features on ``device`` (CPU by default to save VRAM).  :meth:`get`
    returns all currently-held features moved to ``compute_device`` for the FD
    feature-gradient computation.  Queue features never require grad.
    """

    def __init__(
        self,
        max_size: int,
        feature_dim: int,
        device: str = "cpu",
        compute_device: Optional[str] = None,
    ):
        self.max_size = int(max_size)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.compute_device = torch.device(compute_device) if compute_device is not None else None
        self._buf = torch.empty(0, self.feature_dim, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, features: torch.Tensor) -> None:
        if self.max_size <= 0 or features is None or features.numel() == 0:
            return
        feats = features.detach().to(device=self.device, dtype=torch.float32)
        if feats.ndim != 2 or feats.shape[1] != self.feature_dim:
            raise ValueError(
                f"FeatureQueue expects [N, {self.feature_dim}] features, got {tuple(feats.shape)}"
            )
        self._buf = torch.cat([self._buf, feats], dim=0)
        if self._buf.shape[0] > self.max_size:
            self._buf = self._buf[-self.max_size:].contiguous()

    @torch.no_grad()
    def get(self) -> Optional[torch.Tensor]:
        if self._buf.shape[0] == 0:
            return None
        out = self._buf
        if self.compute_device is not None:
            out = out.to(self.compute_device)
        return out

    def __len__(self) -> int:
        return int(self._buf.shape[0])

    def state_dict(self) -> dict:
        return {"buf": self._buf.cpu(), "max_size": self.max_size, "feature_dim": self.feature_dim}

    def load_state_dict(self, state: dict) -> None:
        self._buf = state["buf"].to(self.device)
        self.max_size = int(state.get("max_size", self.max_size))
        self.feature_dim = int(state.get("feature_dim", self.feature_dim))


class ReplayVideoFDLoss:
    """Single-representation replay video FD loss.

    Parameters
    ----------
    vae_decoder / feature_extractor:
        frozen, eval-mode callables ``z_chunk -> video`` and ``video -> feature``.
        They are frozen in ``__init__`` but gradients still flow *through* their
        ops to the latent chunk during replay.
    real_stats:
        dict with ``mean`` ``[D]`` and ``cov`` ``[D, D]`` (and optional
        ``cov_eps`` / ``cov_sqrt``), as produced by
        ``precompute_real_video_fd_stats.py``.
    chunker:
        :class:`utils.video_chunking.TemporalChunker` defining the temporal
        windows and the latent/pixel geometry.
    feature_queue:
        optional :class:`FeatureQueue`; its detached features stabilise the
        generated-moment estimate while only current features carry gradients.
    grad_clip_z:
        optional max-norm clip applied to ``grad_z`` before it is returned.
    """

    def __init__(
        self,
        vae_decoder: Callable[[torch.Tensor], torch.Tensor],
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        real_stats: dict,
        chunker: TemporalChunker,
        preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        feature_queue: Optional[FeatureQueue] = None,
        cov_eps: Optional[float] = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        time_dim: int = 2,
        grad_clip_z: Optional[float] = None,
        update_queue: bool = True,
    ):
        self.vae_decoder = vae_decoder.eval() if isinstance(vae_decoder, torch.nn.Module) else vae_decoder
        self.feature_extractor = (
            feature_extractor.eval() if isinstance(feature_extractor, torch.nn.Module) else feature_extractor
        )
        self._freeze(self.vae_decoder)
        self._freeze(self.feature_extractor)

        self.real_stats = real_stats
        self.chunker = chunker
        self.preprocess_fn = preprocess_fn
        self.feature_queue = feature_queue
        self.cov_eps = cov_eps if cov_eps is not None else float(real_stats.get("cov_eps", 1e-4))
        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype
        self.time_dim = int(time_dim)
        self.grad_clip_z = grad_clip_z
        self.update_queue = bool(update_queue)
        self.last_log_dict: Dict[str, float] = {}

    @staticmethod
    def _freeze(module) -> None:
        if isinstance(module, torch.nn.Module):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    def compute_latent_grad(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss_value, grad_z)`` for latents ``z`` ``[B, C, T, H, W]``.

        ``loss_value`` is a detached scalar tensor; ``grad_z`` is the latent
        gradient to feed to ``z.backward(grad_z)``.  Extra metrics are stored in
        :attr:`last_log_dict`.
        """
        z_detached = z.detach()

        # Stage A.1: chunked no-grad feature extraction.
        with torch.no_grad():
            features, feature_map = extract_video_features_no_grad(
                z=z_detached,
                vae_decoder=self.vae_decoder,
                feature_extractor=self.feature_extractor,
                chunker=self.chunker,
                preprocess_fn=self.preprocess_fn,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
                time_dim=self.time_dim,
            )

        queue_features = self.feature_queue.get() if self.feature_queue is not None else None

        # Stage A.2: FD feature gradient (no decoder/extractor graph).
        loss_value, grad_features = compute_fd_feature_gradient(
            features_no_grad=features,
            real_stats=self.real_stats,
            queue_features=queue_features,
            cov_eps=self.cov_eps,
        )

        # Stage B: replay the VJP through chunked decode/extract.
        grad_z = replay_feature_vjp_to_latents(
            z_detached=z_detached,
            grad_features=grad_features,
            vae_decoder=self.vae_decoder,
            feature_extractor=self.feature_extractor,
            chunker=self.chunker,
            preprocess_fn=self.preprocess_fn,
            feature_map=feature_map,
            use_amp=self.use_amp,
            amp_dtype=self.amp_dtype,
            time_dim=self.time_dim,
        )

        if self.grad_clip_z is not None and self.grad_clip_z > 0:
            grad_z = _clip_grad_tensor(grad_z, self.grad_clip_z)

        if self.update_queue and self.feature_queue is not None:
            self.feature_queue.update(features.detach())

        self.last_log_dict = {
            "loss/fd": float(loss_value.item()),
            "grad_z_norm": float(grad_z.detach().float().norm().item()),
            "grad_features_norm": float(grad_features.float().norm().item()),
            "num_features": int(features.shape[0]),
            "queue_size": int(len(self.feature_queue)) if self.feature_queue is not None else 0,
        }
        self.last_log_dict.update(tensor_health("grad_z", grad_z))
        return loss_value, grad_z

    def compute_and_backward(self, z: torch.Tensor) -> Dict[str, float]:
        """Convenience: run :meth:`compute_latent_grad` then ``z.backward(grad_z)``."""
        loss_value, grad_z = self.compute_latent_grad(z)
        z.backward(grad_z)
        return dict(self.last_log_dict)


class MultiRepReplayVideoFDLoss:
    """Multiple representations summed: ``g_z = sum_r weight_r * g_{z,r}``.

    Each representation is a fully-configured :class:`ReplayVideoFDLoss` (its own
    feature extractor, chunker, real stats and queue).  ``compute_latent_grad``
    runs them in turn and accumulates the weighted latent gradients.  Start with a
    single representation; add more once it works (spec section 13).
    """

    def __init__(self, reps: Sequence[Tuple[str, ReplayVideoFDLoss, float]]):
        if not reps:
            raise ValueError("MultiRepReplayVideoFDLoss requires at least one representation")
        self.reps: List[Tuple[str, ReplayVideoFDLoss, float]] = list(reps)
        self.last_log_dict: Dict[str, float] = {}

    def compute_latent_grad(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_z = torch.zeros_like(z.detach())
        total_loss = torch.zeros((), device=z.device, dtype=torch.float32)
        log: Dict[str, float] = {}
        for name, rep, weight in self.reps:
            loss_r, grad_r = rep.compute_latent_grad(z)
            grad_z = grad_z + weight * grad_r
            total_loss = total_loss + weight * loss_r.to(total_loss.device)
            for k, v in rep.last_log_dict.items():
                log[f"{name}/{k}"] = v
        log["loss/fd_total"] = float(total_loss.item())
        log["grad_z_norm"] = float(grad_z.detach().float().norm().item())
        self.last_log_dict = log
        return total_loss.detach(), grad_z

    def compute_and_backward(self, z: torch.Tensor) -> Dict[str, float]:
        _, grad_z = self.compute_latent_grad(z)
        z.backward(grad_z)
        return dict(self.last_log_dict)


def _clip_grad_tensor(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = grad.detach().float().norm()
    if torch.isfinite(norm) and norm > max_norm:
        grad = grad * (max_norm / (norm + 1e-12))
    return grad

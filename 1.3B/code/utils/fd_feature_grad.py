"""Feature-space Fréchet-Distance loss and its feature gradient.

This is *Stage A* of replay FD: given generated window features (extracted with
no grad), compute the FD loss and ``g_f = dL/df`` **without** ever touching the
VAE decoder or feature-extractor graph.  ``g_f`` is then replayed back to the
latents in Stage B (see :mod:`utils.video_feature_extract`).

All statistics are computed in ``float32`` with covariance jitter ``Sigma <-
Sigma + eps*I`` for a numerically stable, differentiable matrix square root.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _covariance(feats: torch.Tensor) -> torch.Tensor:
    """Unbiased covariance ``[D, D]`` of rows in ``feats`` ``[N, D]``."""
    n = feats.shape[0]
    if n < 2:
        return feats.new_zeros(feats.shape[1], feats.shape[1])
    centered = feats - feats.mean(dim=0, keepdim=True)
    return centered.t() @ centered / (n - 1)


def _matrix_sqrt_psd(mat: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition (float32)."""
    mat = 0.5 * (mat + mat.t())
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp_min(0)
    return (evecs * evals.sqrt().unsqueeze(0)) @ evecs.t()


def _safe_psd_eigvals(mat: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of a symmetric PSD matrix, robust to ill-conditioning.

    ``eigvalsh`` can fail to converge when the matrix has many (near-)repeated
    eigenvalues -- common when the generated covariance is estimated from fewer
    samples than its dimension (cold feature queue). We try ``eigvalsh`` in
    float64, then fall back to ``svdvals`` (which equals the eigenvalues for a
    PSD matrix and effectively always converges)."""
    mat = 0.5 * (mat + mat.t())
    mat64 = mat.double()
    try:
        return torch.linalg.eigvalsh(mat64).clamp_min(0).to(mat.dtype)
    except Exception:
        pass
    try:
        return torch.linalg.svdvals(mat64).clamp_min(0).to(mat.dtype)
    except Exception:
        return mat.diagonal().clamp_min(0)


def frechet_distance_from_moments(
    mu_gen: torch.Tensor,
    cov_gen: torch.Tensor,
    mu_real: torch.Tensor,
    cov_real: torch.Tensor,
    cov_eps: float = 1e-4,
    cov_real_sqrt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fréchet distance ``||mu_g-mu_r||^2 + Tr(Cg+Cr-2(Cr^.5 Cg Cr^.5)^.5)``.

    Differentiable w.r.t. ``mu_gen`` / ``cov_gen``.  ``cov_real_sqrt`` may be
    precomputed (it only depends on the frozen real stats) to avoid an eigh per
    call; jitter ``cov_eps`` is added to *both* covariances for stability.
    """
    dim = mu_gen.shape[0]
    eye = torch.eye(dim, device=mu_gen.device, dtype=torch.float32)
    cov_gen = cov_gen.float() + cov_eps * eye
    cov_real = cov_real.float() + cov_eps * eye

    diff = mu_gen.float() - mu_real.float()
    mean_term = diff.dot(diff)

    if cov_real_sqrt is None:
        cov_real_sqrt = _matrix_sqrt_psd(cov_real)
    cov_real_sqrt = cov_real_sqrt.float()

    product = cov_real_sqrt @ cov_gen @ cov_real_sqrt
    product = 0.5 * (product + product.t())
    eigvals = _safe_psd_eigvals(product)
    trace_term = torch.diagonal(cov_gen).sum() + torch.diagonal(cov_real).sum() - 2.0 * eigvals.sqrt().sum()
    return mean_term + trace_term


def fd_loss_from_features(
    current_features: torch.Tensor,
    real_mean: torch.Tensor,
    real_cov: torch.Tensor,
    queue_features: Optional[torch.Tensor] = None,
    cov_eps: float = 1e-5,
    real_cov_sqrt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ordinary FD loss from features.

    Parameters
    ----------
    current_features: ``[N_cur, D]`` -- gradient-carrying generated features.
    queue_features:   ``[N_queue, D]`` -- detached generated features from past
        steps; concatenated with ``current_features`` to estimate the generated
        moments but contributing **no** gradient.
    real_mean / real_cov: ``[D]`` / ``[D, D]`` -- frozen real-video statistics.

    Returns a scalar FD loss whose gradient flows only through
    ``current_features``.
    """
    current = current_features.float()
    if queue_features is not None and queue_features.numel() > 0:
        queue = queue_features.detach().float().to(current.device)
        features_all = torch.cat([current, queue], dim=0)
    else:
        features_all = current

    mu_gen = features_all.mean(dim=0)
    cov_gen = _covariance(features_all)
    return frechet_distance_from_moments(
        mu_gen=mu_gen,
        cov_gen=cov_gen,
        mu_real=real_mean,
        cov_real=real_cov,
        cov_eps=cov_eps,
        cov_real_sqrt=real_cov_sqrt,
    )


def compute_fd_feature_gradient(
    features_no_grad: torch.Tensor,
    real_stats: dict,
    queue_features: Optional[torch.Tensor] = None,
    cov_eps: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(loss_value, g_f = dL/df)`` for the current features.

    ``features_no_grad`` are the detached features from the no-grad
    decode/extract pass.  No decoder/extractor graph is involved here, so this is
    cheap compared with full video decoding.

    ``real_stats`` is the dict produced by ``precompute_real_video_fd_stats.py``
    and must contain ``mean`` ``[D]`` and ``cov`` ``[D, D]`` (optionally
    ``cov_eps`` and a precomputed ``cov_sqrt``).
    """
    device = features_no_grad.device
    real_mean = real_stats["mean"].to(device=device, dtype=torch.float32)
    real_cov = real_stats["cov"].to(device=device, dtype=torch.float32)
    real_cov_sqrt = real_stats.get("cov_sqrt")
    if real_cov_sqrt is not None:
        real_cov_sqrt = real_cov_sqrt.to(device=device, dtype=torch.float32)
    eps = float(real_stats.get("cov_eps", 1e-4)) if cov_eps is None else float(cov_eps)

    queue = None
    if queue_features is not None and queue_features.numel() > 0:
        queue = queue_features.detach().to(device=device, dtype=torch.float32)

    features_leaf = features_no_grad.detach().float().requires_grad_(True)
    loss = fd_loss_from_features(
        current_features=features_leaf,
        real_mean=real_mean,
        real_cov=real_cov,
        queue_features=queue,
        cov_eps=eps,
        real_cov_sqrt=real_cov_sqrt,
    )
    grad_features = torch.autograd.grad(
        loss,
        features_leaf,
        retain_graph=False,
        create_graph=False,
    )[0]
    return loss.detach(), grad_features.detach()


def tensor_health(name: str, tensor: torch.Tensor) -> dict:
    """Small NaN/Inf/norm health report for logging (see spec sections 15 & 19)."""
    with torch.no_grad():
        finite = torch.isfinite(tensor)
        return {
            f"{name}/has_nan": bool(torch.isnan(tensor).any().item()),
            f"{name}/has_inf": bool(torch.isinf(tensor).any().item()),
            f"{name}/finite_frac": float(finite.float().mean().item()),
            f"{name}/norm": float(tensor.detach().float().norm().item()),
        }

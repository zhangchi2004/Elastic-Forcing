"""Single-encoder distribution matching: exact repulsion and reference attraction.

Reading map: offline helpers build the persistent Nyström reference; training
starts at ``VideoNystromMMD.forward``. Paper Sec. 3.2 calls its combination
with the Monte Carlo reference minibatch the hybrid reference kernel mean.
The released multi-encoder path uses full-population partial gradients with
FIFO and attraction EMA disabled; those optional modes are retained below.

The released estimator includes diagonal pairs (B * B denominator) and the
reference-reference constant. This differs from the paper's self-excluding
repulsion and constant-free optimization expression; preserve both here,
including their effect on the release's subsequent detached loss scaling.

Every input sample is a complete ``[N, D]`` dense-video feature matrix.  For
generated samples ``G``, frozen landmarks ``Z`` and a frozen reference pack,
the legacy objective is

    mean(K_gg) - 2 * mean(K_gz @ alpha) + k_rr.

An opt-in matched-teacher path additionally accepts a fresh teacher batch
``T``. It computes the complete biased Gaussian-kernel terms ``mean(K_gt)``
and ``mean(K_tt)`` without sampling or approximation. The fixed-reference and
fresh-teacher MMDs are combined with normalized non-negative group weights, so
enabling a second teacher target does not duplicate the student repulsion or
silently double the total attraction coefficient.

The explicit partial-gradient interface uses gradient students ``G`` and a
detached full student population ``S``. Without explicit student indices it
preserves the rectangular estimator ``mean K(G,S)`` and evaluates attractions
only for ``G``. Supplying the positions of ``G`` in ``S`` opts into the
full-student estimator: selected differentiable values are reinserted into
detached ``S``, then the loss uses ``mean K(S,S)``, fixed attraction over
``S``, and matched-teacher attraction over ``S x T``. Thus only ``G`` gets
gradients, including both sides of the full self-kernel. Both partial modes
reject FIFO and attraction EMA because either stateful transform would change
their stated estimators.

By default, ``K_gg`` is the exact, biased Gaussian-RBF Gram matrix (including
its diagonal).  An optional transactional FIFO changes only this repulsion
term to the current-current off-diagonal sum plus the current-history cross
sum, normalized by the exact number of ordered pairs.  FIFO entries are
detached FP32 feature matrices and are published only after optimizer success.
Only the generated samples receive gradients.  The reference attraction is
represented by a kernel-ridge/Nyström fit and ``k_rr`` is the exact biased
reference-reference constant.

An optional transactional path applies an EMA to the complete scalar
attraction term only.  The exact current-batch repulsion is never averaged.
The current attraction keeps its natural ``1-beta`` derivative after the
first committed update; historical attraction values are detached.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


REFERENCE_PACK_VERSION = 1
_PACK_KEYS = {
    "format_version",
    "landmarks",
    "alpha",
    "k_rr",
    "sigma",
    "metadata",
    "content_hash",
}


def canonicalize_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deterministic, JSON-safe metadata mapping."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    try:
        encoded = json.dumps(
            dict(metadata),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "metadata must contain only finite JSON-serializable values"
        ) from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("metadata must canonicalize to a dictionary")
    return decoded


def _metadata_contains(actual: Any, expected: Any, path: str = "metadata") -> None:
    """Require ``expected`` to be a recursively exact subset of ``actual``."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"{path} is not a mapping")
        for key, expected_value in expected.items():
            if key not in actual:
                raise ValueError(f"{path} is missing required key {key!r}")
            _metadata_contains(
                actual[key], expected_value, f"{path}.{key}")
        return
    if actual != expected:
        raise ValueError(
            f"{path} mismatch: saved={actual!r}, expected={expected!r}")


_FINITE_VALIDATION_MAX_ELEMENTS = 64 * 1024 * 1024


def _validate_video_tensor(tensor: torch.Tensor, name: str) -> Tuple[int, int, int]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3 or min(tensor.shape) <= 0:
        raise ValueError(f"{name} must have non-empty shape [B, N, D]")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")

    # A full isfinite mask is one byte per element.  The production reference
    # has 12,212 * 1,024 * 1,024 elements, so materializing that mask needs
    # 11.93 GiB and can OOM an otherwise valid H200 consolidation.  Validate
    # complete rows in bounded slices instead; every value is still checked.
    row_elements = int(tensor.shape[1]) * int(tensor.shape[2])
    rows_per_chunk = max(
        1, int(_FINITE_VALIDATION_MAX_ELEMENTS) // row_elements)
    detached = tensor.detach()
    for start in range(0, int(tensor.shape[0]), rows_per_chunk):
        block = detached[start:start + rows_per_chunk]
        if not bool(torch.isfinite(block).all().item()):
            raise FloatingPointError(f"{name} contains non-finite values")
    return int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2])


def _flat_row_norm_sq(flattened: torch.Tensor) -> torch.Tensor:
    """FP32 squared row norms without a full elementwise-square temporary."""
    if flattened.ndim != 2 or not flattened.is_floating_point():
        raise ValueError("flattened video features must be a floating [B, F] tensor")
    return torch.linalg.vector_norm(flattened, ord=2, dim=1).square()


def _pairwise_flat_sq(
    left_flat: torch.Tensor,
    right_flat: torch.Tensor,
    left_sq: torch.Tensor,
    right_sq: torch.Tensor,
) -> torch.Tensor:
    return (
        left_sq[:, None]
        + right_sq[None, :]
        - 2.0 * (left_flat @ right_flat.t())
    ).clamp_min(0)


def _inverse_two_sigma_sq(
    sigma: torch.Tensor | float,
    *,
    device: torch.device,
) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(sigma, device=device, dtype=torch.float32)
    if sigma_tensor.numel() != 1:
        raise ValueError("sigma must be scalar")
    if not bool(torch.isfinite(sigma_tensor).item()) or float(sigma_tensor) <= 0:
        raise ValueError("sigma must be finite and positive")
    return 0.5 / sigma_tensor.square()


def _rbf_flat_precomputed(
    left_flat: torch.Tensor,
    right_flat: torch.Tensor,
    left_sq: torch.Tensor,
    right_sq: torch.Tensor,
    inverse_two_sigma_sq: torch.Tensor,
) -> torch.Tensor:
    distances_sq = _pairwise_flat_sq(
        left_flat, right_flat, left_sq, right_sq)
    return torch.exp(-distances_sq * inverse_two_sigma_sq)


def pairwise_frobenius_sq(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Squared Frobenius distances between ``[B,N,D]`` video matrices."""
    _, left_n, left_d = _validate_video_tensor(left, "left")
    _, right_n, right_d = _validate_video_tensor(right, "right")
    if (left_n, left_d) != (right_n, right_d):
        raise ValueError(
            "left/right video matrix shapes differ: "
            f"{(left_n, left_d)} != {(right_n, right_d)}")
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    return _pairwise_flat_sq(
        left_flat,
        right_flat,
        _flat_row_norm_sq(left_flat),
        _flat_row_norm_sq(right_flat),
    )


def rbf_frobenius_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Gaussian RBF kernel using complete video-matrix samples."""
    _, left_n, left_d = _validate_video_tensor(left, "left")
    _, right_n, right_d = _validate_video_tensor(right, "right")
    if (left_n, left_d) != (right_n, right_d):
        raise ValueError(
            "left/right video matrix shapes differ: "
            f"{(left_n, left_d)} != {(right_n, right_d)}")
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    return _rbf_flat_precomputed(
        left_flat,
        right_flat,
        _flat_row_norm_sq(left_flat),
        _flat_row_norm_sq(right_flat),
        _inverse_two_sigma_sq(sigma, device=left.device),
    )


@torch.no_grad()
def median_frobenius_sigma(
    reference: torch.Tensor,
    *,
    max_samples: Optional[int] = 2048,
    chunk_size: int = 256,
    min_sigma: float = 1.0e-6,
) -> torch.Tensor:
    """Deterministic median-distance bandwidth over video samples.

    A deterministic linspace subset bounds the number of pair distances.
    Pair blocks are constructed in chunks; only the final one-dimensional
    vector of upper-triangular distances is retained for the exact median of
    that subset.
    """
    count, _, _ = _validate_video_tensor(reference, "reference")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not math.isfinite(float(min_sigma)) or float(min_sigma) <= 0:
        raise ValueError("min_sigma must be finite and positive")
    samples = reference.detach().float()
    if max_samples is not None:
        if int(max_samples) < 2:
            raise ValueError("max_samples must be at least 2 or None")
        if count > int(max_samples):
            indices = torch.linspace(
                0,
                count - 1,
                steps=int(max_samples),
                device=samples.device,
                dtype=torch.float64,
            ).round().long()
            samples = samples.index_select(0, indices)
    count = int(samples.shape[0])
    if count < 2:
        return samples.new_tensor(1.0)

    flattened = samples.flatten(1)
    norms_sq = _flat_row_norm_sq(flattened)
    pieces = []
    for row_start in range(0, count, chunk_size):
        row_end = min(row_start + chunk_size, count)
        for col_start in range(row_start, count, chunk_size):
            col_end = min(col_start + chunk_size, count)
            block = _pairwise_flat_sq(
                flattened[row_start:row_end],
                flattened[col_start:col_end],
                norms_sq[row_start:row_end],
                norms_sq[col_start:col_end],
            )
            if col_start == row_start:
                upper = torch.triu_indices(
                    row_end - row_start,
                    col_end - col_start,
                    offset=1,
                    device=block.device,
                )
                values = block[upper[0], upper[1]]
            else:
                values = block.reshape(-1)
            positive = values[values > 0]
            if positive.numel():
                pieces.append(positive)
    if not pieces:
        return samples.new_tensor(1.0)
    median_sq = torch.cat(pieces).median().clamp_min(float(min_sigma) ** 2)
    return median_sq.sqrt()


def _nearest_centroid_labels(
    samples: torch.Tensor,
    centroids: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sample_flat = samples.float().flatten(1)
    centroid_flat = centroids.float().flatten(1)
    sample_sq = _flat_row_norm_sq(sample_flat)
    centroid_sq = _flat_row_norm_sq(centroid_flat)
    labels = []
    minimum_sq = []
    for start in range(0, samples.shape[0], chunk_size):
        distances = _pairwise_flat_sq(
            sample_flat[start:start + chunk_size],
            centroid_flat,
            sample_sq[start:start + chunk_size],
            centroid_sq,
        )
        values, indices = distances.min(dim=1)
        labels.append(indices)
        minimum_sq.append(values)
    return torch.cat(labels), torch.cat(minimum_sq)


def _deterministic_cluster_sums(
    flattened: torch.Tensor,
    labels: torch.Tensor,
    num_clusters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Segment-sum rows without CUDA atomic ``index_add_``.

    Sorting once and differencing a fixed-order prefix sum makes a rebuilt
    reference pack bitwise reproducible on the same hardware/software stack.
    CUDA ``index_add_`` uses repeated-index atomics whose accumulation order
    is intentionally unspecified.
    """
    counts = torch.bincount(labels, minlength=int(num_clusters))
    order = torch.argsort(labels, stable=True)
    sorted_values = flattened.index_select(0, order)
    prefix = sorted_values.cumsum(dim=0)
    sums = flattened.new_zeros(
        (int(num_clusters), flattened.shape[1]))
    nonempty = counts > 0
    ends = counts.cumsum(0) - 1
    selected_ends = ends[nonempty]
    cluster_sums = prefix.index_select(0, selected_ends)
    starts = selected_ends - counts[nonempty] + 1
    has_predecessor = starts > 0
    if bool(has_predecessor.any()):
        cluster_sums[has_predecessor] -= prefix.index_select(
            0, starts[has_predecessor] - 1)
    sums[nonempty] = cluster_sums
    return sums, counts


@torch.no_grad()
def deterministic_video_kmeans(
    reference: torch.Tensor,
    num_landmarks: int,
    *,
    iterations: int = 20,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Deterministic farthest-first initialized Lloyd k-means."""
    count, num_tokens, feature_dim = _validate_video_tensor(
        reference, "reference")
    num_landmarks = int(num_landmarks)
    if num_landmarks <= 0 or num_landmarks > count:
        raise ValueError(
            f"num_landmarks must be in [1, {count}], got {num_landmarks}")
    if int(iterations) < 0:
        raise ValueError("iterations must be non-negative")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")

    samples = reference.detach().float()
    flattened = samples.flatten(1)
    norms_sq = _flat_row_norm_sq(flattened)
    global_mean = samples.mean(dim=0, keepdim=True).flatten(1)
    mean_sq = _flat_row_norm_sq(global_mean)
    first = int(
        _pairwise_flat_sq(flattened, global_mean, norms_sq, mean_sq)
        .squeeze(1)
        .argmax()
        .item()
    )
    selected = [first]
    closest_sq = _pairwise_flat_sq(
        flattened,
        flattened[first:first + 1],
        norms_sq,
        norms_sq[first:first + 1],
    ).squeeze(1)
    for _ in range(1, num_landmarks):
        next_index = int(closest_sq.argmax().item())
        selected.append(next_index)
        next_sq = _pairwise_flat_sq(
            flattened,
            flattened[next_index:next_index + 1],
            norms_sq,
            norms_sq[next_index:next_index + 1],
        ).squeeze(1)
        closest_sq = torch.minimum(closest_sq, next_sq)

    centroids = samples.index_select(
        0,
        torch.tensor(selected, device=samples.device, dtype=torch.long),
    ).clone()
    previous_labels = None
    for _ in range(int(iterations)):
        labels, _ = _nearest_centroid_labels(
            samples, centroids, int(chunk_size))
        if previous_labels is not None and torch.equal(labels, previous_labels):
            break
        previous_labels = labels
        flattened = samples.flatten(1)
        sums, cluster_counts = _deterministic_cluster_sums(
            flattened, labels, num_landmarks)
        nonempty = cluster_counts > 0
        updated = centroids.flatten(1).clone()
        updated[nonempty] = (
            sums[nonempty]
            / cluster_counts[nonempty, None].to(sums.dtype)
        )
        centroids = updated.reshape(
            num_landmarks, num_tokens, feature_dim)
    return centroids


@torch.no_grad()
def compute_nystrom_alpha(
    reference: torch.Tensor,
    landmarks: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    ridge: float = 1.0e-6,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Fit coefficients for the persistent Nyström reference kernel mean.

    Solve (K_ZZ + ridge * I) alpha = mean_R K_ZR once, offline. Training
    evaluates K(x, Z) @ alpha without revisiting the full reference bank R.
    Code ``alpha`` is the paper's coefficient vector beta (Appendix C.2),
    not the scalar mixture weight alpha_e in the hybrid mean (Eq. 9).
    """
    ref_count, ref_n, ref_d = _validate_video_tensor(reference, "reference")
    _, landmark_n, landmark_d = _validate_video_tensor(
        landmarks, "landmarks")
    if (ref_n, ref_d) != (landmark_n, landmark_d):
        raise ValueError("reference and landmarks have different matrix shapes")
    if not math.isfinite(float(ridge)) or float(ridge) < 0:
        raise ValueError("ridge must be finite and non-negative")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")

    reference_flat = reference.detach().float().flatten(1)
    landmark_flat = landmarks.detach().float().flatten(1)
    reference_sq = _flat_row_norm_sq(reference_flat)
    landmark_sq = _flat_row_norm_sq(landmark_flat)
    inverse = _inverse_two_sigma_sq(sigma, device=landmarks.device)
    target = landmarks.new_zeros(landmarks.shape[0], dtype=torch.float64)
    for start in range(0, ref_count, int(chunk_size)):
        end = min(start + int(chunk_size), ref_count)
        block = _rbf_flat_precomputed(
            landmark_flat,
            reference_flat[start:end],
            landmark_sq,
            reference_sq[start:end],
            inverse,
        )
        target.add_(block.double().sum(dim=1))
    target.div_(float(ref_count))

    kernel_zz = _rbf_flat_precomputed(
        landmark_flat,
        landmark_flat,
        landmark_sq,
        landmark_sq,
        inverse,
    ).double()
    kernel_zz = 0.5 * (kernel_zz + kernel_zz.t())
    kernel_zz.fill_diagonal_(1.0)
    regularized = kernel_zz + float(ridge) * torch.eye(
        landmarks.shape[0],
        device=landmarks.device,
        dtype=torch.float64,
    )
    try:
        alpha = torch.linalg.solve(regularized, target)
    except RuntimeError as exc:
        raise ValueError(
            "Nyström landmark kernel is singular; increase ridge") from exc
    return alpha.float()


@torch.no_grad()
def compute_reference_krr(
    reference: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Exact biased ``mean(K_rr)``, including the reference diagonal."""
    count, _, _ = _validate_video_tensor(reference, "reference")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    flattened = reference.detach().float().flatten(1)
    norms_sq = _flat_row_norm_sq(flattened)
    inverse = _inverse_two_sigma_sq(sigma, device=reference.device)
    total = reference.new_zeros((), dtype=torch.float64)
    chunk_size = int(chunk_size)
    # The kernel is symmetric.  Evaluate the diagonal and strict upper block
    # triangle once, doubling only off-diagonal blocks.  This is the same full
    # biased Gram mean while avoiding millions of duplicate tiny GEMMs.
    for row_start in range(0, count, chunk_size):
        row_end = min(row_start + chunk_size, count)
        for col_start in range(row_start, count, chunk_size):
            col_end = min(col_start + chunk_size, count)
            kernel = _rbf_flat_precomputed(
                flattened[row_start:row_end],
                flattened[col_start:col_end],
                norms_sq[row_start:row_end],
                norms_sq[col_start:col_end],
                inverse,
            )
            block_sum = kernel.double().sum()
            if row_start == col_start:
                block_sum.add_(
                    (1.0 - kernel.diagonal()).double().sum())
                total.add_(block_sum)
            else:
                total.add_(2.0 * block_sum)
    return (total / float(count * count)).float()


def _hash_tensor(hasher: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    descriptor = {
        "name": name,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }
    hasher.update(json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    hasher.update(raw)


def compute_reference_pack_hash(pack: Mapping[str, Any]) -> str:
    """Hash pack contents, deliberately excluding ``content_hash`` itself."""
    if not isinstance(pack, Mapping):
        raise TypeError("reference pack must be a mapping")
    required_without_hash = _PACK_KEYS - {"content_hash"}
    missing = required_without_hash - set(pack)
    if missing:
        raise ValueError(f"reference pack is missing keys: {sorted(missing)}")
    metadata = canonicalize_metadata(pack["metadata"])
    hasher = hashlib.sha256()
    hasher.update(b"video-nystrom-reference-pack\0")
    hasher.update(str(int(pack["format_version"])).encode("ascii"))
    hasher.update(json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8"))
    for name in ("landmarks", "alpha", "k_rr", "sigma"):
        value = pack[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"reference pack {name!r} must be a tensor")
        _hash_tensor(hasher, name, value)
    return hasher.hexdigest()


def validate_reference_pack(
    pack: Mapping[str, Any],
    *,
    expected_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate schema, geometry, recursive metadata subset, and SHA-256."""
    if not isinstance(pack, Mapping):
        raise TypeError("reference pack must be a mapping")
    keys = set(pack)
    if keys != _PACK_KEYS:
        raise ValueError(
            "reference pack keys mismatch: "
            f"missing={sorted(_PACK_KEYS - keys)}, "
            f"unexpected={sorted(keys - _PACK_KEYS)}")
    if int(pack["format_version"]) != REFERENCE_PACK_VERSION:
        raise ValueError(
            "unsupported reference pack version "
            f"{pack['format_version']!r}; expected {REFERENCE_PACK_VERSION}")

    landmarks = pack["landmarks"]
    alpha = pack["alpha"]
    k_rr = pack["k_rr"]
    sigma = pack["sigma"]
    landmark_count, num_tokens, feature_dim = _validate_video_tensor(
        landmarks, "landmarks")
    if not isinstance(alpha, torch.Tensor) or alpha.ndim != 1:
        raise ValueError("alpha must be a one-dimensional tensor")
    if alpha.shape[0] != landmark_count or not alpha.is_floating_point():
        raise ValueError("alpha length/dtype is incompatible with landmarks")
    if not bool(torch.isfinite(alpha).all().item()):
        raise FloatingPointError("alpha contains non-finite values")
    for name, value in (("k_rr", k_rr), ("sigma", sigma)):
        if (
            not isinstance(value, torch.Tensor)
            or value.numel() != 1
            or not value.is_floating_point()
        ):
            raise ValueError(f"{name} must be a floating scalar tensor")
        if not bool(torch.isfinite(value).item()):
            raise FloatingPointError(f"{name} is non-finite")
    if float(sigma) <= 0:
        raise ValueError("sigma must be positive")
    if float(k_rr) < 0:
        raise ValueError("k_rr must be non-negative")

    metadata = canonicalize_metadata(pack["metadata"])
    if metadata != pack["metadata"]:
        raise ValueError("reference pack metadata is not canonical")
    nystrom_metadata = metadata.get("nystrom")
    required_nystrom = {
        "sample_axis": "video",
        "sample_shape": [num_tokens, feature_dim],
        "kernel": "gaussian_rbf",
        "kernel_metric": "frobenius",
        "generated_gram": "biased_exact_including_diagonal",
        "num_landmarks": landmark_count,
    }
    if not isinstance(nystrom_metadata, Mapping):
        raise ValueError("metadata.nystrom is missing")
    _metadata_contains(
        nystrom_metadata, required_nystrom, "metadata.nystrom")
    if expected_metadata is not None:
        expected = canonicalize_metadata(expected_metadata)
        _metadata_contains(metadata, expected)

    saved_hash = pack["content_hash"]
    if not isinstance(saved_hash, str) or len(saved_hash) != 64:
        raise ValueError("content_hash must be a 64-character SHA-256 hex string")
    calculated_hash = compute_reference_pack_hash(pack)
    if not hmac.compare_digest(saved_hash, calculated_hash):
        raise ValueError(
            "reference pack content_hash mismatch: "
            f"saved={saved_hash}, calculated={calculated_hash}")

    return {
        "format_version": REFERENCE_PACK_VERSION,
        "landmarks": landmarks,
        "alpha": alpha,
        "k_rr": k_rr.reshape(()),
        "sigma": sigma.reshape(()),
        "metadata": metadata,
        "content_hash": saved_hash,
    }


@torch.no_grad()
def build_video_nystrom_reference(
    reference: torch.Tensor,
    *,
    num_landmarks: int,
    metadata: Mapping[str, Any],
    bandwidth: Optional[float] = None,
    median_max_samples: Optional[int] = 2048,
    kmeans_iterations: int = 20,
    ridge: float = 1.0e-6,
    chunk_size: int = 256,
    seed: int = 0,
) -> Dict[str, Any]:
    """Construct a frozen, portable Nyström reference pack."""
    reference_count, num_tokens, feature_dim = _validate_video_tensor(
        reference, "reference")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    seed = int(seed)
    reference = reference.detach().float()
    if bandwidth is None:
        sigma = median_frobenius_sigma(
            reference,
            max_samples=median_max_samples,
            chunk_size=int(chunk_size),
        )
        bandwidth_source = "median_frobenius_distance"
    else:
        if not math.isfinite(float(bandwidth)) or float(bandwidth) <= 0:
            raise ValueError("bandwidth must be finite and positive")
        sigma = reference.new_tensor(float(bandwidth))
        bandwidth_source = "provided"

    landmarks = deterministic_video_kmeans(
        reference,
        int(num_landmarks),
        iterations=int(kmeans_iterations),
        chunk_size=int(chunk_size),
    )
    alpha = compute_nystrom_alpha(
        reference,
        landmarks,
        sigma,
        ridge=float(ridge),
        chunk_size=int(chunk_size),
    )
    k_rr = compute_reference_krr(
        reference, sigma, chunk_size=int(chunk_size))

    canonical_metadata = canonicalize_metadata(metadata)
    technical_metadata = {
        "sample_axis": "video",
        "sample_shape": [num_tokens, feature_dim],
        "kernel": "gaussian_rbf",
        "kernel_metric": "frobenius",
        "generated_gram": "biased_exact_including_diagonal",
        "reference_count": reference_count,
        "num_landmarks": int(num_landmarks),
        "bandwidth_source": bandwidth_source,
        "median_max_samples": (
            None if median_max_samples is None else int(median_max_samples)
        ),
        "kmeans": "deterministic_farthest_first_lloyd",
        "seed": seed,
        "kmeans_iterations": int(kmeans_iterations),
        "ridge": float(ridge),
    }
    if (
        "nystrom" in canonical_metadata
        and canonical_metadata["nystrom"] != technical_metadata
    ):
        raise ValueError(
            "metadata.nystrom is reserved and does not match builder settings")
    canonical_metadata["nystrom"] = technical_metadata
    canonical_metadata = canonicalize_metadata(canonical_metadata)

    pack: Dict[str, Any] = {
        "format_version": REFERENCE_PACK_VERSION,
        "landmarks": landmarks.detach().cpu().contiguous(),
        "alpha": alpha.detach().cpu().contiguous(),
        "k_rr": k_rr.detach().cpu().reshape(()),
        "sigma": sigma.detach().cpu().reshape(()),
        "metadata": canonical_metadata,
    }
    pack["content_hash"] = compute_reference_pack_hash(pack)
    return validate_reference_pack(pack)


def save_reference_pack(
    pack: Mapping[str, Any],
    path: str | os.PathLike,
    *,
    atomic: bool = True,
) -> None:
    """Validate and save a portable pack, atomically by default."""
    validated = validate_reference_pack(pack)
    portable = {
        **validated,
        "landmarks": validated["landmarks"].detach().cpu().contiguous(),
        "alpha": validated["alpha"].detach().cpu().contiguous(),
        "k_rr": validated["k_rr"].detach().cpu().reshape(()),
        "sigma": validated["sigma"].detach().cpu().reshape(()),
    }
    # Moving tensors to CPU is part of the serialized representation, so hash
    # the exact portable payload rather than copying the incoming hash blindly.
    portable["content_hash"] = compute_reference_pack_hash(portable)
    destination = os.path.abspath(os.fspath(path))
    if not atomic:
        torch.save(portable, destination)
        return
    directory = os.path.dirname(destination)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=directory,
    )
    os.close(descriptor)
    try:
        torch.save(portable, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_reference_pack(
    path: str | os.PathLike,
    *,
    expected_metadata: Optional[Mapping[str, Any]] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load a pack and strictly validate its metadata and content hash."""
    try:
        pack = torch.load(
            os.fspath(path),
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        pack = torch.load(os.fspath(path), map_location=map_location)
    return validate_reference_pack(
        pack, expected_metadata=expected_metadata)


def _kernel_sum_chunked(
    left: torch.Tensor,
    right: torch.Tensor,
    sigma: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    total = left.new_zeros(())
    self_kernel = left is right
    for row_start in range(0, left.shape[0], chunk_size):
        rows = left[row_start:row_start + chunk_size]
        for col_start in range(0, right.shape[0], chunk_size):
            cols = right[col_start:col_start + chunk_size]
            kernel = rbf_frobenius_kernel(rows, cols, sigma)
            block_sum = kernel.sum()
            if self_kernel and row_start == col_start:
                # The RBF self-kernel is analytically one.  Explicitly replace
                # it to avoid norm/Gram cancellation error in FP32.
                diagonal = kernel.diagonal()
                block_sum = block_sum + (1.0 - diagonal).sum()
            total = total + block_sum
    return total


def _kernel_matvec_chunked(
    samples: torch.Tensor,
    landmarks: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    outputs = []
    for row_start in range(0, samples.shape[0], chunk_size):
        rows = samples[row_start:row_start + chunk_size]
        values = rows.new_zeros(rows.shape[0])
        for col_start in range(0, landmarks.shape[0], chunk_size):
            cols = landmarks[col_start:col_start + chunk_size]
            kernel = rbf_frobenius_kernel(rows, cols, sigma)
            values = values + kernel @ alpha[
                col_start:col_start + chunk_size]
        outputs.append(values)
    return torch.cat(outputs)


class VideoNystromMMD(nn.Module):
    """One encoder's MMD: repulsion - 2 * attraction + reference constant.

    The generated-sample kernel is evaluated exactly. Nyström approximates
    only attraction to the persistent reference bank; attraction to the
    current reference minibatch is also exact when enabled.
    """

    _STATE_VERSION = 1
    _ATTRACTION_EMA_STATE_VERSION = 1
    _REPULSION_FIFO_STATE_VERSION = 1

    def __init__(
        self,
        reference_pack: Mapping[str, Any],
        *,
        expected_metadata: Optional[Mapping[str, Any]] = None,
        kernel_chunk_size: int = 256,
        attraction_ema_enabled: bool = False,
        attraction_ema_beta: float = 0.99,
        attraction_weight: float = 1.0,
        current_teacher_attraction_enabled: bool = False,
        current_teacher_attraction_weight: float = 1.0,
        repulsion_fifo_size: int = 0,
    ):
        super().__init__()
        if int(kernel_chunk_size) <= 0:
            raise ValueError("kernel_chunk_size must be positive")
        if (
            not math.isfinite(float(attraction_ema_beta))
            or not 0.0 <= float(attraction_ema_beta) < 1.0
        ):
            raise ValueError("attraction EMA beta must be in [0, 1)")
        if (
            not math.isfinite(float(attraction_weight))
            or float(attraction_weight) < 0.0
        ):
            raise ValueError("attraction weight must be finite and non-negative")
        if (
            not math.isfinite(float(current_teacher_attraction_weight))
            or float(current_teacher_attraction_weight) < 0.0
        ):
            raise ValueError(
                "current teacher attraction weight must be finite and "
                "non-negative")
        current_teacher_attraction_enabled = bool(
            current_teacher_attraction_enabled)
        if (
            current_teacher_attraction_enabled
            and float(attraction_weight)
            + float(current_teacher_attraction_weight) <= 0.0
        ):
            raise ValueError(
                "fixed/current teacher attraction weights must have a "
                "positive sum when current teacher attraction is enabled")
        repulsion_fifo_size = int(repulsion_fifo_size)
        if repulsion_fifo_size < 0:
            raise ValueError("repulsion FIFO size must be non-negative")
        if repulsion_fifo_size > 0 and attraction_ema_enabled:
            raise ValueError(
                "repulsion FIFO is incompatible with attraction EMA")
        pack = validate_reference_pack(
            reference_pack, expected_metadata=expected_metadata)
        self.kernel_chunk_size = int(kernel_chunk_size)
        self.attraction_ema_enabled = bool(attraction_ema_enabled)
        self.attraction_ema_beta = float(attraction_ema_beta)
        self.attraction_weight = float(attraction_weight)
        self.current_teacher_attraction_enabled = (
            current_teacher_attraction_enabled)
        self.current_teacher_attraction_weight = float(
            current_teacher_attraction_weight)
        self.repulsion_fifo_size = repulsion_fifo_size
        self.repulsion_fifo_enabled = bool(repulsion_fifo_size > 0)
        self.reference_metadata = pack["metadata"]
        self.reference_hash = pack["content_hash"]
        self.register_buffer(
            "landmarks", pack["landmarks"].detach().float().clone(),
            persistent=True)
        self.register_buffer(
            "alpha", pack["alpha"].detach().float().clone(),
            persistent=True)
        self.register_buffer(
            "k_rr", pack["k_rr"].detach().float().reshape(()).clone(),
            persistent=True)
        self.register_buffer(
            "sigma", pack["sigma"].detach().float().reshape(()).clone(),
            persistent=True)
        # FIFO contents have their own strict trainer-checkpoint payload.  A
        # non-persistent buffer still follows ``module.to(device)`` without
        # silently duplicating the large history in the module state_dict.
        self.register_buffer(
            "repulsion_fifo",
            torch.empty(
                (0, int(self.landmarks.shape[1]), int(self.landmarks.shape[2])),
                dtype=torch.float32,
                device=self.landmarks.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "repulsion_fifo_commits",
            torch.zeros((), dtype=torch.long, device=self.landmarks.device),
            persistent=False,
        )
        # Keep the disabled module's state_dict byte-schema compatible with
        # the original exact objective.  Enabled modules persist these two
        # scalars and advertise the changed schema through extra state.
        self.register_buffer(
            "attraction_ema",
            torch.zeros((), dtype=torch.float32),
            persistent=self.attraction_ema_enabled,
        )
        self.register_buffer(
            "attraction_ema_updates",
            torch.zeros((), dtype=torch.long),
            persistent=self.attraction_ema_enabled,
        )
        # Pending state is deliberately not a buffer: an uncommitted optimizer
        # update must never enter a checkpoint.
        self._pending_attraction: Optional[torch.Tensor] = None
        self._pending_repulsion_fifo: Optional[torch.Tensor] = None
        # The final FIFO allocation is prepared before optimizer.step().
        # Post-step commit only publishes this already-owned tensor.
        self._prepared_repulsion_fifo: Optional[torch.Tensor] = None

    @property
    def sample_shape(self) -> Tuple[int, int]:
        return int(self.landmarks.shape[1]), int(self.landmarks.shape[2])

    @property
    def attraction_ema_pending(self) -> bool:
        return self._pending_attraction is not None

    @property
    def repulsion_fifo_pending(self) -> bool:
        return self._pending_repulsion_fifo is not None

    @property
    def repulsion_fifo_fill(self) -> int:
        return int(self.repulsion_fifo.shape[0])

    def forward(
        self,
        generated: Optional[torch.Tensor] = None,
        current_teacher: Optional[torch.Tensor] = None,
        *,
        gradient_students: Optional[torch.Tensor] = None,
        student_reference: Optional[torch.Tensor] = None,
        gradient_student_indices: Optional[
            torch.Tensor | Sequence[int]
        ] = None,
        stage: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate the legacy full-batch or explicit partial-gradient loss.

        Legacy callers pass ``generated`` and retain the original two-sided
        ``K(generated, generated)`` derivative exactly. Partial-gradient
        callers instead pass both ``gradient_students`` (the row-side samples
        whose derivatives are required) and ``student_reference`` (the full
        detached column-side student population). The latter computes

        ``mean K(G,S) - 2 * attraction(G) + reference_constant``

        with a rectangular ``G x S`` student population. The matched
        teacher population must have exactly ``|G|`` samples. Neither
        ``student_reference`` nor ``current_teacher`` can receive gradients.

        Passing ``gradient_student_indices`` opts into the full-student
        partial-gradient estimator. The indices identify the positions of
        ``gradient_students`` in ``student_reference``. Those differentiable
        values are functionally inserted into the otherwise detached student
        population, yielding complete ``mean K(S,S)``, fixed attraction over
        all ``S``, and exact current-teacher attraction over ``S x T``. Only
        indexed students retain gradients; ``|T|`` must still equal ``|G|``.
        """
        # 1. Choose the generated population and samples that retain a graph.
        # ``student_reference`` is generated data, not the persistent reference
        # bank; these input names remain compatible with existing callers.
        partial_mode = (
            gradient_students is not None or student_reference is not None)
        full_student_partial_mode = gradient_student_indices is not None
        if full_student_partial_mode and not partial_mode:
            raise ValueError(
                "gradient_student_indices is only valid with the explicit "
                "partial-gradient interface")
        if generated is not None and gradient_students is not None:
            raise ValueError(
                "generated and gradient_students are mutually exclusive")
        if partial_mode:
            if gradient_students is None or student_reference is None:
                raise ValueError(
                    "partial-gradient MMD requires both gradient_students and "
                    "student_reference")
            if generated is not None:
                raise ValueError(
                    "generated cannot be combined with the explicit "
                    "partial-gradient interface")
            if self.repulsion_fifo_enabled:
                raise ValueError(
                    "partial-gradient student_reference is incompatible with "
                    "repulsion FIFO")
            if self.attraction_ema_enabled:
                raise ValueError(
                    "partial-gradient student_reference is incompatible with "
                    "attraction EMA")
            gradient_input = gradient_students
        else:
            if generated is None:
                raise ValueError(
                    "generated is required for legacy MMD calls")
            gradient_input = generated

        gradient_batch, num_tokens, feature_dim = _validate_video_tensor(
            gradient_input,
            "gradient_students" if partial_mode else "generated",
        )
        if (num_tokens, feature_dim) != self.sample_shape:
            raise ValueError(
                "student video matrix shape does not match frozen reference: "
                f"{(num_tokens, feature_dim)} != {self.sample_shape}")
        if gradient_input.device != self.landmarks.device:
            raise ValueError(
                "gradient students and frozen reference must be on the same "
                "device")

        samples = gradient_input.float()
        selected_indices = None
        if partial_mode:
            reference_batch, reference_tokens, reference_dim = (
                _validate_video_tensor(
                    student_reference, "student_reference"))
            if (reference_tokens, reference_dim) != (num_tokens, feature_dim):
                raise ValueError(
                    "student_reference matrix shape must match "
                    "gradient_students: "
                    f"{(reference_tokens, reference_dim)} != "
                    f"{(num_tokens, feature_dim)}")
            if student_reference.device != gradient_input.device:
                raise ValueError(
                    "gradient_students and student_reference must share a "
                    "device")
            reference_samples = student_reference.detach().float()
            if full_student_partial_mode:
                try:
                    selected_indices = torch.as_tensor(
                        gradient_student_indices,
                        device=gradient_input.device,
                    )
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "gradient_student_indices must be an integer "
                        "tensor or sequence") from exc
                integer_dtypes = (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
                if (
                    selected_indices.dtype == torch.bool
                    or selected_indices.dtype not in integer_dtypes
                ):
                    raise TypeError(
                        "gradient_student_indices must contain integers")
                selected_indices = selected_indices.to(dtype=torch.long)
                if selected_indices.ndim != 1:
                    raise ValueError(
                        "gradient_student_indices must be one-dimensional")
                if int(selected_indices.numel()) != gradient_batch:
                    raise ValueError(
                        "gradient_student_indices must contain exactly one "
                        "index per gradient student: "
                        f"{int(selected_indices.numel())} != {gradient_batch}")
                if bool(
                    ((selected_indices < 0)
                     | (selected_indices >= reference_batch)).any().item()
                ):
                    raise ValueError(
                        "gradient_student_indices contains an out-of-range "
                        "student_reference position")
                if (
                    int(torch.unique(selected_indices).numel())
                    != gradient_batch
                ):
                    raise ValueError(
                        "gradient_student_indices must be unique")
                selected_reference = reference_samples.index_select(
                    0, selected_indices)
                if not torch.equal(samples.detach(), selected_reference):
                    raise ValueError(
                        "gradient_students must exactly match "
                        "student_reference at gradient_student_indices")
                # Functionally replace only selected detached values. Reusing
                # this same object on both sides of K(S,S) gives selected
                # students both row- and column-side derivatives.
                objective_samples = reference_samples.index_copy(
                    0, selected_indices, samples)
                repulsion_reference_samples = objective_samples
                attraction_samples = objective_samples
                attraction_batch = reference_batch
            else:
                objective_samples = samples
                repulsion_reference_samples = reference_samples
                attraction_samples = samples
                attraction_batch = gradient_batch
        else:
            reference_batch = gradient_batch
            # Preserve object identity here. The legacy self-kernel helper
            # relies on it both for its exact diagonal replacement and for the
            # original two-sided K_ss derivative.
            reference_samples = samples
            objective_samples = samples
            repulsion_reference_samples = reference_samples
            attraction_samples = samples
            attraction_batch = gradient_batch

        # 2. Read the frozen reference minibatch for the Monte Carlo term.
        # These features are targets; gradients flow through generated samples.
        teacher_samples = None
        teacher_batch = None
        if self.current_teacher_attraction_enabled:
            if current_teacher is None:
                raise ValueError(
                    "current_teacher is required when exact current-teacher "
                    "attraction is enabled")
            teacher_batch, teacher_tokens, teacher_dim = (
                _validate_video_tensor(current_teacher, "current_teacher"))
            if (
                teacher_batch != gradient_batch
                or teacher_tokens != num_tokens
                or teacher_dim != feature_dim
            ):
                student_name = (
                    "gradient_students" if partial_mode else "generated")
                raise ValueError(
                    "current teacher features must exactly match "
                    f"{student_name} "
                    "[B,N,D]: "
                    f"{(teacher_batch, teacher_tokens, teacher_dim)} != "
                    f"{(gradient_batch, num_tokens, feature_dim)}")
            if current_teacher.device != gradient_input.device:
                raise ValueError(
                    "gradient students and current teacher must be on the same "
                    "device")
            # The current teacher is a matched frozen target. Detach before
            # every kernel operation so callers can never accidentally create
            # a teacher-side gradient graph.
            teacher_samples = current_teacher.detach().float()
        elif current_teacher is not None:
            raise ValueError(
                "current_teacher was provided while current teacher "
                "attraction is disabled")

        # 3. Exact generated-sample repulsion: sum every kernel pair in
        # chunks. Chunking controls memory without approximating the sum.
        exact_kgg_sum = _kernel_sum_chunked(
            objective_samples,
            repulsion_reference_samples,
            self.sigma,
            self.kernel_chunk_size,
        )
        if self.repulsion_fifo_enabled:
            history_size = self.repulsion_fifo_fill
            denominator = gradient_batch * (
                gradient_batch - 1 + history_size)
            if denominator <= 0:
                raise ValueError(
                    "repulsion FIFO needs at least one current-current "
                    "off-diagonal or current-history pair")
            # ``_kernel_sum_chunked`` analytically replaces each self-kernel
            # by one.  Subtracting B therefore removes the complete diagonal,
            # including its autograd path, without forming a dense mask.
            current_offdiag_sum = exact_kgg_sum - float(gradient_batch)
            if history_size:
                history_cross_sum = _kernel_sum_chunked(
                    samples,
                    self.repulsion_fifo,
                    self.sigma,
                    self.kernel_chunk_size,
                )
            else:
                history_cross_sum = samples.new_zeros(())
            repulsion = (
                current_offdiag_sum + history_cross_sum
            ) / float(denominator)
        else:
            history_size = 0
            # Full-population replay keeps B * B pairs even when only a
            # subset is differentiated. The rectangular mode uses |G| * B.
            # Gradient-subset compensation is handled by the replay caller.
            if full_student_partial_mode:
                denominator = reference_batch * reference_batch
            else:
                denominator = gradient_batch * reference_batch
            current_offdiag_sum = None
            history_cross_sum = None
            repulsion = exact_kgg_sum / float(denominator)
        # 4. Persistent Nyström attraction (paper Eq. 8): K(x, Z) @ alpha.
        # The saved ``alpha`` vector is paper beta, not the mixture weight.
        fixed_attraction_values = _kernel_matvec_chunked(
            attraction_samples,
            self.landmarks,
            self.alpha,
            self.sigma,
            self.kernel_chunk_size,
        )
        fixed_current_attraction = fixed_attraction_values.mean()
        current_teacher_attraction = None
        current_teacher_self_kernel = None
        current_teacher_exact_mmd = None
        # 5. Hybrid reference mean (paper Eq. 9): mix the persistent term
        # with exact minibatch attraction over all generated x reference pairs.
        # Group weights 2:1 correspond to paper alpha_e = 1/3.
        if self.current_teacher_attraction_enabled:
            fixed_weight = self.attraction_weight
            current_weight = self.current_teacher_attraction_weight
            teacher_weight_sum = fixed_weight + current_weight
            fixed_normalized_weight = fixed_weight / teacher_weight_sum
            current_normalized_weight = current_weight / teacher_weight_sum
            current_teacher_attraction = _kernel_sum_chunked(
                attraction_samples,
                teacher_samples,
                self.sigma,
                self.kernel_chunk_size,
            ) / float(attraction_batch * teacher_batch)
            # The reference self-kernel has no generator gradient. Its value
            # still affects the release's later detached-scale division.
            with torch.no_grad():
                current_teacher_self_kernel = _kernel_sum_chunked(
                    teacher_samples,
                    teacher_samples,
                    self.sigma,
                    self.kernel_chunk_size,
                ) / float(teacher_batch * teacher_batch)
            current_attraction = (
                fixed_normalized_weight * fixed_current_attraction
                + current_normalized_weight * current_teacher_attraction
            )
            reference_constant = (
                fixed_normalized_weight * self.k_rr
                + current_normalized_weight * current_teacher_self_kernel
            )
            current_teacher_exact_mmd = (
                repulsion
                - 2.0 * current_teacher_attraction
                + current_teacher_self_kernel
            )
        else:
            fixed_normalized_weight = None
            current_normalized_weight = None
            teacher_weight_sum = None
            current_attraction = fixed_current_attraction
            reference_constant = self.k_rr
        # 6. Optional historical attraction/FIFO state. These branches are
        # inactive in the released multi-encoder configuration.
        history_updates = int(self.attraction_ema_updates.item())
        if self.attraction_ema_enabled and history_updates > 0:
            ema_blend_current_coefficient = 1.0 - self.attraction_ema_beta
            ema_attraction = (
                self.attraction_ema_beta * self.attraction_ema.detach()
                + ema_blend_current_coefficient * current_attraction
            )
            if self.current_teacher_attraction_enabled:
                # Keep the transactional EMA's smoothed forward value while
                # preserving the full derivative of this update's normalized
                # fixed/current teacher mixture. Without this straight-through
                # form beta=0.99 would silently shrink all teacher gradients
                # to one percent relative to K_ss.
                attraction = current_attraction + (
                    ema_attraction - current_attraction).detach()
                current_coefficient = 1.0
            else:
                attraction = ema_attraction
                current_coefficient = ema_blend_current_coefficient
        else:
            ema_blend_current_coefficient = 1.0
            current_coefficient = 1.0
            attraction = current_attraction

        if self.attraction_ema_enabled and stage:
            if self._pending_attraction is not None:
                raise RuntimeError(
                    "an attraction EMA update is already staged; "
                    "commit or discard it before another staged forward")
            self._pending_attraction = current_attraction.detach().clone()

        if self.repulsion_fifo_enabled and stage:
            if self._pending_repulsion_fifo is not None:
                raise RuntimeError(
                    "a repulsion FIFO update is already staged; "
                    "commit or discard it before another staged forward")
            # The fresh GradCache supplies the identical global batch on every
            # rank.  Cloning it here gives each rank the same independently
            # owned FIFO candidate and prevents all gradients/aliasing.
            self._pending_repulsion_fifo = (
                samples.detach().to(dtype=torch.float32).clone())

        # 7. Assemble the raw single-encoder loss. MultiEncoderMMD applies
        # detached-scale normalization and encoder weighting after this.
        if self.current_teacher_attraction_enabled:
            # This is the weighted average of the fixed-reference MMD and the
            # current exact MMD. K_ss is shared, so it appears once and the
            # total attraction coefficient remains -2 for every raw scale.
            loss = repulsion - 2.0 * attraction + reference_constant
            effective_attraction_weight = 1.0
        else:
            # Preserve the legacy value and gradient exactly when the opt-in
            # current teacher path is disabled.
            if self.attraction_weight == 1.0:
                loss = repulsion - 2.0 * attraction + self.k_rr
            else:
                loss = (
                    repulsion
                    - 2.0 * self.attraction_weight * attraction
                    + self.k_rr
                )
            effective_attraction_weight = self.attraction_weight
        metadata = {
            "repulsion": repulsion.detach(),
            "attraction": attraction.detach(),
            "attraction_weight": loss.new_tensor(
                effective_attraction_weight),
            "reference_constant": reference_constant.detach(),
            "sigma": self.sigma.detach(),
            "global_samples": torch.tensor(
                reference_batch, device=loss.device, dtype=torch.long),
        }
        if partial_mode:
            metadata.update({
                "partial_student_reference_enabled": loss.new_ones(
                    (), dtype=torch.long),
                "gradient_student_samples": loss.new_tensor(
                    gradient_batch, dtype=torch.long),
                "student_reference_samples": loss.new_tensor(
                    reference_batch, dtype=torch.long),
                "student_reference_detached": loss.new_ones(
                    (), dtype=torch.long),
                "repulsion_denominator": loss.new_tensor(
                    denominator, dtype=torch.long),
                "student_repulsion_denominator": loss.new_tensor(
                    denominator, dtype=torch.long),
                "fixed_teacher_attraction_denominator": loss.new_tensor(
                    attraction_batch, dtype=torch.long),
            })
            if full_student_partial_mode:
                metadata.update({
                    "full_student_partial_gradient": loss.new_ones(
                        (), dtype=torch.long),
                    "student_repulsion_gradient_sides": loss.new_tensor(
                        2, dtype=torch.long),
                })
        if self.current_teacher_attraction_enabled:
            metadata.update({
                "current_teacher_attraction_enabled": loss.new_ones(
                    (), dtype=torch.long),
                "fixed_teacher_attraction": (
                    fixed_current_attraction.detach()),
                "current_teacher_attraction": (
                    current_teacher_attraction.detach()),
                "fixed_teacher_attraction_weight": loss.new_tensor(
                    self.attraction_weight),
                "current_teacher_attraction_weight": loss.new_tensor(
                    self.current_teacher_attraction_weight),
                "teacher_attraction_weight_sum": loss.new_tensor(
                    teacher_weight_sum),
                "fixed_teacher_attraction_normalized_weight": (
                    loss.new_tensor(fixed_normalized_weight)),
                "current_teacher_attraction_normalized_weight": (
                    loss.new_tensor(current_normalized_weight)),
                "mixed_current_attraction": current_attraction.detach(),
                "fixed_reference_constant": self.k_rr.detach(),
                "current_teacher_self_kernel": (
                    current_teacher_self_kernel.detach()),
                "current_teacher_exact_mmd": (
                    current_teacher_exact_mmd.detach()),
                "current_teacher_samples": loss.new_tensor(
                    teacher_batch, dtype=torch.long),
            })
            if partial_mode:
                metadata.update({
                    "current_teacher_attraction_denominator": (
                        loss.new_tensor(
                            attraction_batch * teacher_batch,
                            dtype=torch.long,
                        )
                    ),
                    "current_teacher_self_kernel_denominator": (
                        loss.new_tensor(
                            teacher_batch * teacher_batch,
                            dtype=torch.long,
                        )
                    ),
                })
        if self.attraction_ema_enabled:
            metadata.update({
                "current_attraction": current_attraction.detach(),
                "attraction_ema_enabled": loss.new_ones(
                    (), dtype=torch.long),
                "attraction_ema_updates": (
                    self.attraction_ema_updates.detach().clone()),
                "attraction_current_coefficient": loss.new_tensor(
                    current_coefficient),
            })
            if self.current_teacher_attraction_enabled:
                metadata.update({
                    "attraction_ema_blend_current_coefficient": (
                        loss.new_tensor(
                            ema_blend_current_coefficient)),
                    "attraction_gradient_current_coefficient": (
                        loss.new_ones(())),
                    "attraction_ema_smoothed_value": attraction.detach(),
                    "effective_ema_loss": loss.detach(),
                    "attraction_gradient_mode": (
                        "straight_through_full_current_mix"),
                })
        if self.repulsion_fifo_enabled:
            fill_after_commit = min(
                self.repulsion_fifo_size,
                history_size + gradient_batch,
            )
            metadata.update({
                "repulsion_fifo_enabled": loss.new_ones(
                    (), dtype=torch.long),
                "repulsion_fifo_capacity": loss.new_tensor(
                    self.repulsion_fifo_size, dtype=torch.long),
                "repulsion_fifo_fill_before": loss.new_tensor(
                    history_size, dtype=torch.long),
                # Compatibility spelling used by the trainer logger.
                "repulsion_fifo_filled_before": loss.new_tensor(
                    history_size, dtype=torch.long),
                "repulsion_fifo_fill_after_commit": loss.new_tensor(
                    fill_after_commit, dtype=torch.long),
                "repulsion_fifo_commits": (
                    self.repulsion_fifo_commits.detach().clone()),
                "repulsion_fifo_committed_updates": (
                    self.repulsion_fifo_commits.detach().clone()),
                "repulsion_current_offdiag_kernel_sum": (
                    current_offdiag_sum.detach()),
                "repulsion_fifo_current_offdiag_sum": (
                    current_offdiag_sum.detach()),
                "repulsion_history_kernel_sum": (
                    history_cross_sum.detach()),
                "repulsion_fifo_history_cross_sum": (
                    history_cross_sum.detach()),
                "repulsion_denominator": loss.new_tensor(
                    denominator, dtype=torch.long),
                "repulsion_fifo_denominator": loss.new_tensor(
                    denominator, dtype=torch.long),
            })
        return loss, metadata

    @torch.no_grad()
    def prepare_commit(self) -> None:
        """Validate all staged distribution state before optimizer mutation."""
        if self.attraction_ema_enabled:
            if self._pending_attraction is None:
                raise RuntimeError("no attraction EMA update is staged")
            if (
                self._pending_attraction.ndim != 0
                or not bool(torch.isfinite(self._pending_attraction).item())
            ):
                raise FloatingPointError(
                    "refusing to commit a non-finite attraction EMA update")
        if self.repulsion_fifo_enabled:
            pending = self._pending_repulsion_fifo
            if pending is None:
                raise RuntimeError("no repulsion FIFO update is staged")
            if (
                pending.ndim != 3
                or tuple(pending.shape[1:]) != self.sample_shape
                or pending.shape[0] <= 0
                or pending.dtype != torch.float32
                or pending.device != self.repulsion_fifo.device
                or pending.requires_grad
                or pending.grad_fn is not None
            ):
                raise ValueError(
                    "staged repulsion FIFO values must be detached FP32 "
                    f"[B,{self.sample_shape[0]},{self.sample_shape[1]}] "
                    "on the FIFO device")
            if not bool(torch.isfinite(pending).all().item()):
                raise FloatingPointError(
                    "refusing to commit non-finite repulsion FIFO values")
            if self._prepared_repulsion_fifo is None:
                combined = torch.cat((self.repulsion_fifo, pending), dim=0)
                if combined.shape[0] > self.repulsion_fifo_size:
                    combined = combined[-self.repulsion_fifo_size:]
                # Allocate the right-sized, independently owned result while
                # the trainer can still abort before optimizer mutation.
                self._prepared_repulsion_fifo = (
                    combined.detach().contiguous().clone())

    @torch.no_grad()
    def commit(self) -> None:
        """Publish staged state only after a successful optimizer update."""
        if (
            self._pending_attraction is None
            and self._pending_repulsion_fifo is None
        ):
            return
        self.prepare_commit()
        if self._pending_attraction is not None:
            pending_attraction = self._pending_attraction
            if int(self.attraction_ema_updates.item()) == 0:
                self.attraction_ema.copy_(pending_attraction)
            else:
                self.attraction_ema.mul_(self.attraction_ema_beta).add_(
                    pending_attraction,
                    alpha=1.0 - self.attraction_ema_beta,
                )
            self.attraction_ema_updates.add_(1)
        if self._pending_repulsion_fifo is not None:
            if self._prepared_repulsion_fifo is None:
                raise RuntimeError("repulsion FIFO commit was not prepared")
            self.repulsion_fifo = self._prepared_repulsion_fifo
            self.repulsion_fifo_commits.add_(1)
        self.discard()

    def discard(self) -> None:
        """Drop all staged state after a failed or skipped update."""
        self._pending_attraction = None
        self._pending_repulsion_fifo = None
        self._prepared_repulsion_fifo = None

    def repulsion_fifo_state_dict(self) -> Optional[Dict[str, Any]]:
        """Return the committed FIFO in oldest-to-newest order."""
        if not self.repulsion_fifo_enabled:
            return None
        if self.repulsion_fifo_pending:
            raise RuntimeError(
                "refusing to checkpoint a pending repulsion FIFO update")
        return {
            "version": self._REPULSION_FIFO_STATE_VERSION,
            "capacity": self.repulsion_fifo_size,
            "sample_shape": list(self.sample_shape),
            "dtype": "float32",
            "count": self.repulsion_fifo_fill,
            "commits": int(self.repulsion_fifo_commits.item()),
            "values": self.repulsion_fifo.detach().cpu().clone(),
        }
    def validate_repulsion_fifo_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Validate and normalize a portable FIFO checkpoint payload."""
        if not self.repulsion_fifo_enabled:
            raise ValueError(
                "cannot load repulsion FIFO state while it is disabled")
        if not isinstance(state, Mapping):
            raise TypeError("repulsion FIFO state must be a mapping")
        expected_keys = {
            "version", "capacity", "sample_shape", "dtype",
            "count", "commits", "values",
        }
        if set(state) != expected_keys:
            raise ValueError(
                "repulsion FIFO state keys mismatch: "
                f"saved={sorted(state)}, expected={sorted(expected_keys)}")
        if (
            state["version"] != self._REPULSION_FIFO_STATE_VERSION
            or int(state["capacity"]) != self.repulsion_fifo_size
            or tuple(state["sample_shape"]) != self.sample_shape
            or state["dtype"] != "float32"
        ):
            raise ValueError("repulsion FIFO checkpoint/config mismatch")
        count = int(state["count"])
        commits = int(state["commits"])
        values = state["values"]
        if not 0 <= count <= self.repulsion_fifo_size or commits < 0:
            raise ValueError("repulsion FIFO counters are invalid")
        if (
            not isinstance(values, torch.Tensor)
            or values.dtype != torch.float32
            or values.layout != torch.strided
            or values.is_meta
            or tuple(values.shape) != (count, *self.sample_shape)
            or not bool(torch.isfinite(values).all().item())
        ):
            raise ValueError(
                "repulsion FIFO values must be finite float32 with shape "
                f"{(count, *self.sample_shape)}")
        return {
            "version": self._REPULSION_FIFO_STATE_VERSION,
            "capacity": self.repulsion_fifo_size,
            "sample_shape": list(self.sample_shape),
            "dtype": "float32",
            "count": count,
            "commits": commits,
            "values": values.detach().cpu().clone(),
        }

    @torch.no_grad()
    def load_repulsion_fifo_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        validated = self.validate_repulsion_fifo_state_dict(state)
        self.repulsion_fifo = validated["values"].to(
            device=self.landmarks.device,
            dtype=torch.float32,
        ).contiguous().clone()
        self.repulsion_fifo_commits.fill_(validated["commits"])
        self.discard()

    def attraction_ema_state_dict(self) -> Optional[Dict[str, Any]]:
        """Return the small trainer checkpoint payload for the opt-in EMA."""
        if not self.attraction_ema_enabled:
            return None
        if self.attraction_ema_pending:
            raise RuntimeError(
                "refusing to checkpoint a pending attraction EMA update")
        return {
            "version": self._ATTRACTION_EMA_STATE_VERSION,
            "beta": self.attraction_ema_beta,
            "value": self.attraction_ema.detach().cpu().clone(),
            "updates": self.attraction_ema_updates.detach().cpu().clone(),
        }

    def validate_attraction_ema_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Validate and normalize a small attraction EMA checkpoint payload."""
        if not self.attraction_ema_enabled:
            raise ValueError(
                "cannot load attraction EMA state while the path is disabled")
        if not isinstance(state, Mapping):
            raise TypeError("attraction EMA state must be a mapping")
        expected_keys = {"version", "beta", "value", "updates"}
        if set(state) != expected_keys:
            raise ValueError(
                "attraction EMA state keys mismatch: "
                f"saved={sorted(state)}, expected={sorted(expected_keys)}")
        if (
            state["version"] != self._ATTRACTION_EMA_STATE_VERSION
            or float(state["beta"]) != self.attraction_ema_beta
        ):
            raise ValueError(
                "attraction EMA checkpoint/config mismatch: "
                f"saved version/beta={state['version']!r}/{state['beta']!r}, "
                "expected "
                f"{self._ATTRACTION_EMA_STATE_VERSION}/"
                f"{self.attraction_ema_beta}")
        value = state["value"]
        updates = state["updates"]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 0
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or value.is_meta
            or not bool(torch.isfinite(value).item())
        ):
            raise ValueError(
                "attraction EMA checkpoint value must be a finite scalar "
                "torch.float32 tensor")
        if (
            not isinstance(updates, torch.Tensor)
            or updates.ndim != 0
            or updates.dtype != torch.long
            or updates.layout != torch.strided
            or updates.is_meta
            or int(updates.item()) < 0
        ):
            raise ValueError(
                "attraction EMA checkpoint updates must be a non-negative "
                "scalar torch.long tensor")
        if int(updates.item()) == 0 and float(value.item()) != 0.0:
            raise ValueError(
                "an uninitialized attraction EMA must have value zero")
        return {
            "version": self._ATTRACTION_EMA_STATE_VERSION,
            "beta": self.attraction_ema_beta,
            "value": value.detach().cpu().clone(),
            "updates": updates.detach().cpu().clone(),
        }

    @torch.no_grad()
    def load_attraction_ema_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        """Strictly restore only the transactional attraction EMA scalars."""
        validated = self.validate_attraction_ema_state_dict(state)
        self.attraction_ema.copy_(
            validated["value"].to(
                device=self.attraction_ema.device,
                dtype=self.attraction_ema.dtype,
            )
        )
        self.attraction_ema_updates.copy_(
            validated["updates"].to(
                device=self.attraction_ema_updates.device)
        )
        self.discard()

    def get_extra_state(self) -> Dict[str, Any]:
        state = {
            "version": self._STATE_VERSION,
            "reference_metadata": self.reference_metadata,
            "reference_hash": self.reference_hash,
        }
        if self.attraction_ema_enabled:
            state = {
                **state,
                "version": self._STATE_VERSION + 1,
                "attraction_ema": {
                    "version": self._ATTRACTION_EMA_STATE_VERSION,
                    "beta": self.attraction_ema_beta,
                },
            }
        if self.current_teacher_attraction_enabled:
            state = {
                **state,
                "version": (4 if self.attraction_ema_enabled else 3),
                "current_teacher_attraction": {
                    "enabled": True,
                    "fixed_weight": self.attraction_weight,
                    "current_weight": (
                        self.current_teacher_attraction_weight),
                    "normalization": "divide_by_group_weight_sum",
                    "estimator": "exact_biased_gaussian_mmd",
                },
            }
        return state

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        expected = self.get_extra_state()
        if state != expected:
            raise ValueError(
                "Video Nyström MMD checkpoint/reference mismatch: "
                f"saved={state}, current={expected}")

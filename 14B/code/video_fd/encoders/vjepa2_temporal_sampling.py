"""Shared stateless temporal-window sampling for V-JEPA2 dense features.

The helper in this module is deliberately independent of Python, NumPy, and
PyTorch random-number-generator state. Training and offline teacher tooling
can therefore derive the same window from stable sample identities without
changing any ambient RNG stream.
"""

from __future__ import annotations

import hashlib

import torch


WINDOW_SAMPLING_HASH = "sha256_first_u64_big_endian_mod_num_windows"


def _validate_identity(seed: str, domain: str, sample_key: int) -> None:
    if not isinstance(seed, str) or not seed:
        raise ValueError("temporal window seed must be a non-empty string")
    if "\0" in seed:
        raise ValueError("temporal window seed must not contain NUL characters")
    if not isinstance(domain, str) or not domain:
        raise ValueError("temporal window domain must be a non-empty string")
    if "\0" in domain:
        raise ValueError(
            "temporal window domain must not contain NUL characters")
    if type(sample_key) is not int:
        raise TypeError("temporal window sample_key must be an integer")
    if sample_key < 0:
        raise ValueError("temporal window sample_key must be non-negative")


def _validate_geometry(
    source_num_frames: int,
    window_num_frames: int,
) -> tuple[int, int]:
    if type(source_num_frames) is not int:
        raise TypeError("source_num_frames must be an integer")
    if type(window_num_frames) is not int:
        raise TypeError("window_num_frames must be an integer")
    if source_num_frames <= 0:
        raise ValueError("source_num_frames must be positive")
    if window_num_frames <= 0:
        raise ValueError("window_num_frames must be positive")
    if window_num_frames > source_num_frames:
        raise ValueError(
            "window_num_frames must not exceed source_num_frames, got "
            f"{window_num_frames} > {source_num_frames}"
        )
    return source_num_frames, window_num_frames


def temporal_window_sample_sha256(
    seed: str,
    domain: str,
    sample_key: int,
) -> str:
    """Return the canonical hash for one stable temporal-window identity.

    The exact payload is ``seed UTF-8 + NUL + domain UTF-8 + NUL + the
    unsigned decimal sample key in UTF-8``.
    """
    _validate_identity(seed, domain, sample_key)
    payload = (
        seed.encode("utf-8")
        + b"\0"
        + domain.encode("utf-8")
        + b"\0"
        + str(sample_key).encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def stateless_temporal_window_start(
    seed: str,
    domain: str,
    sample_key: int,
    *,
    source_num_frames: int,
    window_num_frames: int,
) -> int:
    """Map a stable sample identity to one valid contiguous-window start."""
    source_num_frames, window_num_frames = _validate_geometry(
        source_num_frames,
        window_num_frames,
    )
    digest = bytes.fromhex(
        temporal_window_sample_sha256(seed, domain, sample_key))
    num_windows = source_num_frames - window_num_frames + 1
    return int.from_bytes(digest[:8], byteorder="big") % num_windows


def stateless_temporal_window_indices(
    seed: str,
    domain: str,
    sample_key: int,
    *,
    source_num_frames: int,
    window_num_frames: int,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return the selected contiguous frame indices as a 1-D LongTensor."""
    start = stateless_temporal_window_start(
        seed,
        domain,
        sample_key,
        source_num_frames=source_num_frames,
        window_num_frames=window_num_frames,
    )
    return torch.arange(
        start,
        start + window_num_frames,
        dtype=torch.long,
        device=device,
    )

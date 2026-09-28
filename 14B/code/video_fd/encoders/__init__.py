"""Sparse video-encoder adapters.

All adapters expose a unified API (``preprocess`` / ``forward_sparse``), are frozen
and in eval mode, downsample internally to the encoder's native size, and emit
*sparse* features ``[B, S, D]`` (S in {1,4,8}) -- never dense patch populations.
The replay forward pass keeps gradients w.r.t. the input clip (no ``no_grad``).
"""

from video_fd.encoders.base_sparse_video_encoder import (
    SparseVideoEncoderAdapter,
    SyntheticSparseVideoEncoder,
    differentiable_resize_crop,
    temporal_segment_mean,
)


def build_encoder(config) -> SparseVideoEncoderAdapter:
    """Factory: build a sparse video-encoder adapter from a config dict.

    ``config['name']`` selects the adapter. ``synthetic`` is always available
    (used by the replay-gradient tests); the real adapters are imported lazily so
    a missing dependency only errors when that encoder is actually requested.
    """
    name = str(config.get("name", "synthetic")).lower()
    if name == "synthetic":
        return SyntheticSparseVideoEncoder(config)
    if name in ("videoprism_b", "videoprism"):
        from video_fd.encoders.videoprism_b_adapter import VideoPrismBAdapter
        return VideoPrismBAdapter(config)
    if name in ("vjepa2", "v-jepa2", "vjepa"):
        from video_fd.encoders.vjepa2_adapter import VJEPA2Adapter
        return VJEPA2Adapter(config)
    if name in ("videomae_v2", "videomae", "videomaev2"):
        from video_fd.encoders.videomae_v2_adapter import VideoMAEv2Adapter
        return VideoMAEv2Adapter(config)
    raise ValueError(f"unknown encoder name {name!r}")


__all__ = [
    "SparseVideoEncoderAdapter",
    "SyntheticSparseVideoEncoder",
    "differentiable_resize_crop",
    "temporal_segment_mean",
    "build_encoder",
]

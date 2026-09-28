"""VideoPrism-B (LvT base) sparse video-encoder adapter.

Uses the PyTorch port ``sposiboh/videoprism-lvt-base-f16r288-pt`` (Google
VideoPrism LvT base, ~0.138B params), loaded via
``AutoModel(trust_remote_code=True)``. The model forward takes a **channels-last**
``[B, T, H, W, 3]`` tensor in ``[0, 1]`` (native 16 frames @ 288) and returns a
single L2-normalized global ``video_embeds [B, D]``.

Since the LvT base pools to a global embedding, sparse ``[B, S, D]`` segments are
produced by the base class's segment-forward (S temporal segments, each resized
to 16 frames and encoded). Set ``sparse_mode='global'`` for S=1.
"""

from __future__ import annotations

import torch

from video_fd.encoders.base_sparse_video_encoder import SparseVideoEncoderAdapter


class VideoPrismBAdapter(SparseVideoEncoderAdapter):
    def __init__(self, config):
        config = dict(config)
        config.setdefault("name", "videoprism_b")
        config.setdefault("input_size", 288)
        config.setdefault("input_frames", 16)
        # VideoPrism expects pixels in [0,1] (no mean/std normalization).
        config.setdefault("norm_mean", (0.0, 0.0, 0.0))
        config.setdefault("norm_std", (1.0, 1.0, 1.0))
        config["global_encoder"] = True
        super().__init__(config)
        self.model_id = str(config.get("model_id", "sposiboh/videoprism-lvt-base-f16r288-pt"))
        self._model = None
        self._load()

    def _load(self):
        try:
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover
            raise ImportError(f"VideoPrism needs transformers: {exc!r}") from exc
        try:
            model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as exc:
            raise ImportError(
                f"Could not load VideoPrism port {self.model_id!r} "
                f"(AutoModel trust_remote_code). Original error: {exc!r}"
            ) from exc
        self._model = model.eval().to(torch.float32)
        self.freeze()

    def _encode_global(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [nB, T, 3, s, s] in [0,1]; the port wants channels-last [nB, T, s, s, 3].
        x = pixel_values.permute(0, 1, 3, 4, 2).contiguous()
        out = self._model(pixel_values=x)
        if hasattr(out, "video_embeds") and out.video_embeds is not None:
            return out.video_embeds.float()
        if hasattr(out, "last_hidden_state"):  # token output fallback -> global mean
            return out.last_hidden_state.float().mean(dim=1)
        if torch.is_tensor(out):
            return out.float() if out.dim() == 2 else out.float().mean(dim=1)
        raise RuntimeError("Unexpected VideoPrism output structure")

"""Strict safetensors cache for matched-manifest live-teacher features."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from utils.matched_video_manifest import MatchedVideoManifestEntry


CACHE_FORMAT = "sf_matched_manifest_video_vjepa2_teacher_feature"
CACHE_FORMAT_VERSION = 1
CACHE_METADATA_KEY = "sf_matched_video_teacher_feature"
CACHE_TENSOR_KEY = "features"

# Independent from the fixed-reference central-64 protocol.  Both new student
# and live teacher use the last 64 frames, i.e. 17..80 for the 81-frame source.
MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT: dict[str, Any] = {
    "adapter_contract_version": 1,
    "contract_version": 1,
    "crop_size": 256,
    "expected_source_frame_count": 81,
    "feature_dim": 1024,
    "feature_type": "vjepa2_dense_latent",
    "frame_indices": [17, 80],
    "frame_sampling": "last_contiguous",
    "interpolation": "bilinear_antialias",
    "layer_index": 15,
    "model_family": "facebook/vjepa2-vitl-fpc64-256",
    "model_config_sha256": (
        "3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9"
    ),
    "normalization": "imagenet_default",
    "num_frames": 64,
    "num_tokens": 1024,
    "output_dtype": "float32",
    "output_shape": [1024, 1024],
    "processor_config_sha256": (
        "d2fab4418fc0390b62c4cd72ade56908a7929f80c62288adbe10dd8d23421227"
    ),
    "resize_short_side": 292,
    "sample_axis": "video",
    "token_alignment": "same_deterministic_indices",
    "token_indices_sha256": (
        "a77ee1f9d20cba6b2f2a9a971c3d36610ca7f8aa2a55c616192582665be69d10"
    ),
    "token_sampling": "uniform_linspace_round",
    "transformers_version": "4.53.3",
}

MATCHED_PYAV_LAST64_DECODER_CONTRACT: dict[str, Any] = {
    "backend": "pyav",
    "backend_version": "13.1.0",
    "contract_version": 1,
    "expected_frame_indices": [17, 80],
    "expected_source_frame_count": 81,
    "frame_count_source": "container_decode_order_with_stream_frames_hint",
    "num_frames": 64,
    "output_dtype": "float32",
    "output_layout": "TCHW",
    "output_range": [0.0, 1.0],
    "pixel_format": "rgb24",
    "selection": "last_contiguous",
}


class MatchedVideoFeatureCacheValidationError(ValueError):
    """An existing cache file is stale, corrupt, or identity-incompatible."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT_SHA256 = canonical_sha256(
    MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT
)
MATCHED_PYAV_LAST64_DECODER_CONTRACT_SHA256 = canonical_sha256(
    MATCHED_PYAV_LAST64_DECODER_CONTRACT
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _entry_context(
    entry: MatchedVideoManifestEntry | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(entry, MatchedVideoManifestEntry):
        result = {
            "sample_id": entry.sample_id,
            "video_mtime_ns": int(entry.video_mtime_ns),
            "video_name": entry.video_name,
            "video_size_bytes": int(entry.video_size_bytes),
        }
    elif isinstance(entry, Mapping):
        required = {
            "sample_id", "video_mtime_ns", "video_name", "video_size_bytes",
        }
        missing = required - set(entry)
        if missing:
            raise ValueError(f"cache sample context is missing {sorted(missing)}")
        result = {
            "sample_id": str(entry["sample_id"]),
            "video_mtime_ns": int(entry["video_mtime_ns"]),
            "video_name": str(entry["video_name"]),
            "video_size_bytes": int(entry["video_size_bytes"]),
        }
    else:
        raise TypeError("entry must be MatchedVideoManifestEntry or mapping")
    if not result["sample_id"] or result["sample_id"] != result["video_name"]:
        raise ValueError("cache sample_id must equal video_name")
    if result["video_size_bytes"] <= 0 or result["video_mtime_ns"] <= 0:
        raise ValueError("cache video stat must be positive")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


class MatchedVideoFeatureCache:
    """Lazy one-file-per-video FP32 cache with immutable identity checks."""

    def __init__(
        self,
        cache_root: str | os.PathLike,
        *,
        expected_model_sha256: str,
        expected_feature_contract_sha256: str,
        feature_contract: Mapping[str, Any] | None = None,
        decoder_contract: Mapping[str, Any] | None = None,
        expected_shape: tuple[int, int] | None = None,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.model_sha256 = str(expected_model_sha256).lower()
        if not _is_sha256(self.model_sha256):
            raise ValueError("expected_model_sha256 must be lowercase SHA-256")
        self.feature_contract = json.loads(canonical_json(
            feature_contract or MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT
        ))
        self.feature_contract_sha256 = canonical_sha256(self.feature_contract)
        expected_hash = str(expected_feature_contract_sha256).lower()
        if not _is_sha256(expected_hash):
            raise ValueError(
                "expected_feature_contract_sha256 must be lowercase SHA-256"
            )
        if self.feature_contract_sha256 != expected_hash:
            raise ValueError(
                "feature contract SHA-256 mismatch: "
                f"expected={expected_hash}, actual={self.feature_contract_sha256}"
            )
        self.decoder_contract = json.loads(canonical_json(
            decoder_contract or MATCHED_PYAV_LAST64_DECODER_CONTRACT
        ))
        self.decoder_contract_sha256 = canonical_sha256(self.decoder_contract)
        contract_shape = tuple(
            int(value) for value in self.feature_contract.get("output_shape", ())
        )
        self.expected_shape = (
            tuple(int(value) for value in expected_shape)
            if expected_shape is not None else contract_shape
        )
        if len(self.expected_shape) != 2 or min(self.expected_shape) <= 0:
            raise ValueError("expected feature shape must be positive [N,D]")
        if contract_shape and self.expected_shape != contract_shape:
            raise ValueError(
                "expected_shape must match feature_contract.output_shape"
            )

    def _key_payload(
        self, entry: MatchedVideoManifestEntry | Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "decoder_contract": self.decoder_contract,
            "decoder_contract_sha256": self.decoder_contract_sha256,
            "feature_contract": self.feature_contract,
            "feature_contract_sha256": self.feature_contract_sha256,
            "format": CACHE_FORMAT,
            "format_version": CACHE_FORMAT_VERSION,
            "model_sha256": self.model_sha256,
            "sample": _entry_context(entry),
        }

    def key_for(
        self, entry: MatchedVideoManifestEntry | Mapping[str, Any],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"sf-matched-video-teacher-feature-cache\0")
        digest.update(canonical_json(self._key_payload(entry)).encode("utf-8"))
        return digest.hexdigest()

    def path_for_key(self, cache_key: str) -> Path:
        key = str(cache_key).lower()
        if not _is_sha256(key):
            raise ValueError("cache_key must be lowercase SHA-256")
        return self.cache_root / key[:2] / key[2:4] / f"{key}.safetensors"

    def _expected_metadata(
        self,
        entry: MatchedVideoManifestEntry | Mapping[str, Any],
        cache_key: str,
        tensor_sha256: str,
    ) -> dict[str, Any]:
        result = self._key_payload(entry)
        result.update({
            "cache_key": cache_key,
            "tensor": {
                "dtype": "float32", "key": CACHE_TENSOR_KEY,
                "sha256": tensor_sha256, "shape": list(self.expected_shape),
            },
        })
        return result

    def _validate_tensor(self, feature: torch.Tensor) -> torch.Tensor:
        if not isinstance(feature, torch.Tensor):
            raise TypeError("teacher feature must be a torch.Tensor")
        if feature.ndim == 3 and int(feature.shape[0]) == 1:
            feature = feature[0]
        if feature.ndim != 2 or tuple(feature.shape) != self.expected_shape:
            raise ValueError(
                f"teacher feature shape mismatch: {tuple(feature.shape)} "
                f"!= {self.expected_shape}"
            )
        if not feature.is_floating_point():
            raise TypeError("teacher feature must be floating point")
        result = feature.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(result).all().item()):
            raise FloatingPointError("teacher feature contains non-finite values")
        return result

    def load(
        self,
        entry: MatchedVideoManifestEntry | Mapping[str, Any],
        *,
        cache_key: str | None = None,
    ) -> torch.Tensor | None:
        key = self.key_for(entry)
        if cache_key is not None and str(cache_key) != key:
            raise MatchedVideoFeatureCacheValidationError(
                "caller cache_key does not match sample/model/contracts"
            )
        path = self.path_for_key(key)
        if not path.exists():
            return None
        if not path.is_file():
            raise MatchedVideoFeatureCacheValidationError(
                f"cache path is not a regular file: {path}"
            )
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if list(handle.keys()) != [CACHE_TENSOR_KEY]:
                    raise MatchedVideoFeatureCacheValidationError(
                        f"cache tensor keys mismatch in {path}"
                    )
                serialized = (handle.metadata() or {}).get(CACHE_METADATA_KEY)
                if serialized is None:
                    raise MatchedVideoFeatureCacheValidationError(
                        f"cache metadata is missing in {path}"
                    )
                metadata = json.loads(serialized)
                feature = handle.get_tensor(CACHE_TENSOR_KEY).clone()
        except MatchedVideoFeatureCacheValidationError:
            raise
        except Exception as exc:
            raise MatchedVideoFeatureCacheValidationError(
                f"failed to read safetensors cache {path}: {exc}"
            ) from exc
        feature = self._validate_tensor(feature)
        expected = self._expected_metadata(entry, key, _tensor_sha256(feature))
        if metadata != expected:
            raise MatchedVideoFeatureCacheValidationError(
                f"cache metadata/identity mismatch (stale or corrupt) in {path}"
            )
        return feature

    def write(
        self,
        entry: MatchedVideoManifestEntry | Mapping[str, Any],
        feature: torch.Tensor,
        *,
        cache_key: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        key = self.key_for(entry)
        if cache_key is not None and str(cache_key) != key:
            raise MatchedVideoFeatureCacheValidationError(
                "caller cache_key does not match sample/model/contracts"
            )
        feature = self._validate_tensor(feature)
        tensor_hash = _tensor_sha256(feature)
        destination = self.path_for_key(key)
        if destination.exists() and not overwrite:
            existing = self.load(entry, cache_key=key)
            if existing is None or _tensor_sha256(existing) != tensor_hash:
                raise MatchedVideoFeatureCacheValidationError(
                    "refusing to replace a valid cache entry with different features: "
                    f"{destination}"
                )
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._expected_metadata(entry, key, tensor_hash)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.unlink()
            save_file(
                {CACHE_TENSOR_KEY: feature}, str(temporary),
                metadata={CACHE_METADATA_KEY: canonical_json(metadata)},
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(temporary, destination)
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    existing = self.load(entry, cache_key=key)
                    if existing is None or _tensor_sha256(existing) != tensor_hash:
                        raise MatchedVideoFeatureCacheValidationError(
                            "concurrent cache writer produced different features "
                            f"for {key}"
                        )
                temporary.unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
        self.load(entry, cache_key=key)
        return destination


__all__ = [
    "CACHE_FORMAT", "CACHE_FORMAT_VERSION", "CACHE_METADATA_KEY",
    "CACHE_TENSOR_KEY", "MATCHED_PYAV_LAST64_DECODER_CONTRACT",
    "MATCHED_PYAV_LAST64_DECODER_CONTRACT_SHA256",
    "MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT",
    "MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT_SHA256",
    "MatchedVideoFeatureCache", "MatchedVideoFeatureCacheValidationError",
    "canonical_json", "canonical_sha256",
]

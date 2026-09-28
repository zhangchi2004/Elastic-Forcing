"""Strict, per-sample safetensors cache for OpenVid teacher features."""

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

from utils.openvid_manifest import OpenVidManifestEntry


CACHE_FORMAT = "sf_openvid_vjepa2_teacher_feature"
CACHE_FORMAT_VERSION = 1
CACHE_METADATA_KEY = "sf_openvid_teacher_feature"
CACHE_TENSOR_KEY = "features"

# This is the exact geometry/identity contract carried by the uploaded layer-15
# Nyström reference.  Model *weights* are identified independently by their
# SHA-256 so replacing a checkpoint cannot alias an existing cache entry.
OPENVID_VJEPA2_L15_FEATURE_CONTRACT: dict[str, Any] = {
    "adapter_contract_version": 1,
    "contract_version": 1,
    "crop_size": 256,
    "feature_dim": 1024,
    "feature_type": "vjepa2_dense_latent",
    "frame_sampling": "center_contiguous",
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

OPENVID_PYAV_DECODER_CONTRACT: dict[str, Any] = {
    "backend": "pyav",
    "backend_version": "13.1.0",
    "center_tie_break": "floor",
    "contract_version": 1,
    "frame_count_source": "container_decode_order_with_stream_frames_hint",
    "num_frames": 64,
    "output_dtype": "float32",
    "output_layout": "TCHW",
    "output_range": [0.0, 1.0],
    "pixel_format": "rgb24",
    "selection": "center_contiguous",
}


class OpenVidFeatureCacheValidationError(ValueError):
    """A cache file exists but does not satisfy its immutable contract."""


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


OPENVID_VJEPA2_L15_FEATURE_CONTRACT_SHA256 = canonical_sha256(
    OPENVID_VJEPA2_L15_FEATURE_CONTRACT
)
OPENVID_PYAV_DECODER_CONTRACT_SHA256 = canonical_sha256(
    OPENVID_PYAV_DECODER_CONTRACT
)


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _entry_context(
    entry: OpenVidManifestEntry | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(entry, OpenVidManifestEntry):
        result = {
            "sample_id": entry.sample_id,
            "video_mtime_ns": int(entry.video_mtime_ns),
            "video_name": entry.video_name,
            "video_size_bytes": int(entry.video_size_bytes),
        }
    elif isinstance(entry, Mapping):
        required = {
            "sample_id",
            "video_mtime_ns",
            "video_name",
            "video_size_bytes",
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
        raise TypeError("entry must be OpenVidManifestEntry or mapping")
    if not result["sample_id"] or not result["video_name"]:
        raise ValueError("cache sample/video identifiers must be non-empty")
    if result["video_size_bytes"] <= 0 or result["video_mtime_ns"] <= 0:
        raise ValueError("cache video stat must be positive")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


class OpenVidFeatureCache:
    """Lazy one-file-per-video FP32 cache with strict identity validation."""

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
        model_sha256 = str(expected_model_sha256).lower()
        if not _is_sha256(model_sha256):
            raise ValueError("expected_model_sha256 must be lowercase SHA-256")
        self.model_sha256 = model_sha256

        self.feature_contract = json.loads(canonical_json(
            feature_contract
            if feature_contract is not None
            else OPENVID_VJEPA2_L15_FEATURE_CONTRACT
        ))
        self.feature_contract_sha256 = canonical_sha256(self.feature_contract)
        expected_contract_hash = str(
            expected_feature_contract_sha256).lower()
        if not _is_sha256(expected_contract_hash):
            raise ValueError(
                "expected_feature_contract_sha256 must be lowercase SHA-256"
            )
        if self.feature_contract_sha256 != expected_contract_hash:
            raise ValueError(
                "feature contract SHA-256 mismatch: "
                f"expected={expected_contract_hash}, "
                f"actual={self.feature_contract_sha256}"
            )

        self.decoder_contract = json.loads(canonical_json(
            decoder_contract
            if decoder_contract is not None
            else OPENVID_PYAV_DECODER_CONTRACT
        ))
        self.decoder_contract_sha256 = canonical_sha256(
            self.decoder_contract)
        contract_shape = tuple(
            int(value)
            for value in self.feature_contract.get("output_shape", ())
        )
        self.expected_shape = (
            tuple(int(value) for value in expected_shape)
            if expected_shape is not None
            else contract_shape
        )
        if len(self.expected_shape) != 2 or min(self.expected_shape) <= 0:
            raise ValueError("expected feature shape must be positive [N,D]")
        if contract_shape and self.expected_shape != contract_shape:
            raise ValueError(
                "expected_shape must match feature_contract.output_shape"
            )

    def _key_payload(
        self,
        entry: OpenVidManifestEntry | Mapping[str, Any],
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
        self,
        entry: OpenVidManifestEntry | Mapping[str, Any],
    ) -> str:
        payload = self._key_payload(entry)
        digest = hashlib.sha256()
        digest.update(b"sf-openvid-teacher-feature-cache\0")
        digest.update(canonical_json(payload).encode("utf-8"))
        return digest.hexdigest()

    def path_for_key(self, cache_key: str) -> Path:
        cache_key = str(cache_key).lower()
        if not _is_sha256(cache_key):
            raise ValueError("cache_key must be lowercase SHA-256")
        return (
            self.cache_root
            / cache_key[:2]
            / cache_key[2:4]
            / f"{cache_key}.safetensors"
        )

    def _expected_metadata(
        self,
        entry: OpenVidManifestEntry | Mapping[str, Any],
        cache_key: str,
        tensor_sha256: str,
    ) -> dict[str, Any]:
        payload = self._key_payload(entry)
        payload.update({
            "cache_key": cache_key,
            "tensor": {
                "dtype": "float32",
                "key": CACHE_TENSOR_KEY,
                "sha256": tensor_sha256,
                "shape": list(self.expected_shape),
            },
        })
        return payload

    def _validate_tensor(self, feature: torch.Tensor) -> torch.Tensor:
        if not isinstance(feature, torch.Tensor):
            raise TypeError("teacher feature must be a torch.Tensor")
        if feature.ndim == 3 and int(feature.shape[0]) == 1:
            feature = feature[0]
        if feature.ndim != 2 or tuple(feature.shape) != self.expected_shape:
            raise ValueError(
                "teacher feature shape mismatch: "
                f"{tuple(feature.shape)} != {self.expected_shape}"
            )
        if not feature.is_floating_point():
            raise TypeError("teacher feature must be floating point")
        feature = feature.detach().to(
            device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(feature).all().item()):
            raise FloatingPointError("teacher feature contains non-finite values")
        return feature

    def load(
        self,
        entry: OpenVidManifestEntry | Mapping[str, Any],
        *,
        cache_key: str | None = None,
    ) -> torch.Tensor | None:
        expected_key = self.key_for(entry)
        if cache_key is not None and str(cache_key) != expected_key:
            raise OpenVidFeatureCacheValidationError(
                "caller cache_key does not match sample/model/contracts"
            )
        path = self.path_for_key(expected_key)
        if not path.exists():
            return None
        if not path.is_file():
            raise OpenVidFeatureCacheValidationError(
                f"cache path is not a regular file: {path}"
            )
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if list(handle.keys()) != [CACHE_TENSOR_KEY]:
                    raise OpenVidFeatureCacheValidationError(
                        f"cache tensor keys mismatch in {path}"
                    )
                header = handle.metadata() or {}
                serialized = header.get(CACHE_METADATA_KEY)
                if serialized is None:
                    raise OpenVidFeatureCacheValidationError(
                        f"cache metadata is missing in {path}"
                    )
                metadata = json.loads(serialized)
                # Own the storage after ``safe_open`` closes its mmap.
                feature = handle.get_tensor(CACHE_TENSOR_KEY).clone()
        except OpenVidFeatureCacheValidationError:
            raise
        except Exception as exc:
            raise OpenVidFeatureCacheValidationError(
                f"failed to read safetensors cache {path}: {exc}"
            ) from exc

        feature = self._validate_tensor(feature)
        tensor_hash = _tensor_sha256(feature)
        expected_metadata = self._expected_metadata(
            entry, expected_key, tensor_hash)
        if metadata != expected_metadata:
            raise OpenVidFeatureCacheValidationError(
                f"cache metadata/identity mismatch in {path}"
            )
        return feature

    def write(
        self,
        entry: OpenVidManifestEntry | Mapping[str, Any],
        feature: torch.Tensor,
        *,
        cache_key: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        expected_key = self.key_for(entry)
        if cache_key is not None and str(cache_key) != expected_key:
            raise OpenVidFeatureCacheValidationError(
                "caller cache_key does not match sample/model/contracts"
            )
        feature = self._validate_tensor(feature)
        tensor_hash = _tensor_sha256(feature)
        destination = self.path_for_key(expected_key)
        if destination.exists() and not overwrite:
            existing = self.load(entry, cache_key=expected_key)
            if _tensor_sha256(existing) != tensor_hash:
                raise OpenVidFeatureCacheValidationError(
                    "refusing to replace a valid cache entry with different "
                    f"features: {destination}"
                )
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._expected_metadata(entry, expected_key, tensor_hash)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            # safetensors refuses to overwrite an already-open descriptor, so
            # remove the reserved empty name and let it create the payload.
            temporary.unlink()
            save_file(
                {CACHE_TENSOR_KEY: feature},
                str(temporary),
                metadata={CACHE_METADATA_KEY: canonical_json(metadata)},
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(temporary, destination)
            else:
                try:
                    # A hard link publishes a fully-written inode without ever
                    # exposing a partial destination and without clobbering a
                    # concurrent writer.
                    os.link(temporary, destination)
                except FileExistsError:
                    existing = self.load(entry, cache_key=expected_key)
                    if _tensor_sha256(existing) != tensor_hash:
                        raise OpenVidFeatureCacheValidationError(
                            "concurrent cache writer produced different "
                            f"features for {expected_key}"
                        )
                temporary.unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
        # Re-open through the strict reader before reporting success.
        self.load(entry, cache_key=expected_key)
        return destination


__all__ = [
    "CACHE_FORMAT",
    "CACHE_FORMAT_VERSION",
    "CACHE_METADATA_KEY",
    "CACHE_TENSOR_KEY",
    "OPENVID_PYAV_DECODER_CONTRACT",
    "OPENVID_PYAV_DECODER_CONTRACT_SHA256",
    "OPENVID_VJEPA2_L15_FEATURE_CONTRACT",
    "OPENVID_VJEPA2_L15_FEATURE_CONTRACT_SHA256",
    "OpenVidFeatureCache",
    "OpenVidFeatureCacheValidationError",
    "canonical_json",
    "canonical_sha256",
]

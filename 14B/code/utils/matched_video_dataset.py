"""Metadata-only matched-video dataset with explicitly lazy teacher I/O."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.matched_video_feature_cache import (
    MATCHED_PYAV_LAST64_DECODER_CONTRACT,
    MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT,
    MatchedVideoFeatureCache,
)
from utils.matched_video_manifest import (
    MatchedVideoManifestEntry,
    load_matched_video_manifest,
)


@dataclass(frozen=True)
class DecodedMatchedVideo:
    """The final contiguous RGB frame window in V-JEPA2 input range."""

    video01: torch.Tensor
    frame_indices: torch.Tensor
    source_frame_count: int


def _frames_to_video01(frames: Sequence[Any]) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    shape: tuple[int, int, int] | None = None
    for frame in frames:
        array = frame.to_ndarray(format="rgb24")
        if not isinstance(array, np.ndarray):
            raise TypeError("PyAV frame.to_ndarray did not return ndarray")
        if array.ndim != 3 or int(array.shape[2]) != 3:
            raise ValueError(
                f"decoded RGB frame must be [H,W,3], got {array.shape}"
            )
        if array.dtype != np.uint8:
            raise ValueError(f"decoded RGB frame must be uint8, got {array.dtype}")
        current_shape = tuple(int(value) for value in array.shape)
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError(
                "video changes resolution inside selected frame window: "
                f"{current_shape} != {shape}"
            )
        tensors.append(
            torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        )
    if not tensors:
        raise ValueError("decoded frame window is empty")
    return torch.stack(tensors).to(dtype=torch.float32).div_(255.0)


def decode_matched_video_last_contiguous_pyav(
    video_path: str | os.PathLike,
    *,
    decoder_contract: Mapping[str, Any] | None = None,
) -> DecodedMatchedVideo:
    """Decode exactly frames 17..80 from a contractually 81-frame MP4."""

    contract = dict(
        decoder_contract
        if decoder_contract is not None
        else MATCHED_PYAV_LAST64_DECODER_CONTRACT
    )
    if contract.get("backend") != "pyav":
        raise ValueError("decoder contract backend must be pyav")
    if contract.get("selection") != "last_contiguous":
        raise ValueError("decoder contract selection must be last_contiguous")
    num_frames = int(contract.get("num_frames", 0))
    expected_count = int(contract.get("expected_source_frame_count", 0))
    if num_frames <= 0 or expected_count < num_frames:
        raise ValueError("decoder contract frame geometry is invalid")
    expected_bounds = [expected_count - num_frames, expected_count - 1]
    if list(contract.get("expected_frame_indices", ())) != expected_bounds:
        raise ValueError(
            "decoder contract expected_frame_indices do not describe last64"
        )
    try:
        import av  # Imported only for cache misses.
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyAV is required for a matched-video teacher cache miss; "
            "install the pinned av==13.1.0 dependency"
        ) from exc
    expected_version = str(contract.get("backend_version", ""))
    actual_version = str(getattr(av, "__version__", ""))
    if actual_version != expected_version:
        raise RuntimeError(
            "PyAV version differs from the cache decoder contract: "
            f"{actual_version!r} != {expected_version!r}"
        )

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    selected: list[Any] = []
    decoded_count = 0
    start = expected_count - num_frames
    with av.open(str(path), mode="r") as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise ValueError(
                f"expected exactly one video stream in {path}, got {len(streams)}"
            )
        stream = streams[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            decoded_count = frame_index + 1
            if start <= frame_index < expected_count:
                selected.append(frame)
    if decoded_count != expected_count:
        raise ValueError(
            "matched video source frame count violates the last64 contract: "
            f"sample={path.name}, actual={decoded_count}, "
            f"expected={expected_count}"
        )
    if len(selected) != num_frames:
        raise RuntimeError(
            f"failed to decode frames {start}..{expected_count - 1} from {path}"
        )
    video01 = _frames_to_video01(selected)
    if tuple(video01.shape[:2]) != (num_frames, 3):
        raise RuntimeError(f"decoded video has invalid shape {tuple(video01.shape)}")
    return DecodedMatchedVideo(
        video01=video01,
        frame_indices=torch.arange(start, expected_count, dtype=torch.long),
        source_frame_count=decoded_count,
    )


DecodeFunction = Callable[[Path, Mapping[str, Any]], DecodedMatchedVideo]


def _default_decode(
    path: Path,
    contract: Mapping[str, Any],
) -> DecodedMatchedVideo:
    return decode_matched_video_last_contiguous_pyav(
        path, decoder_contract=contract
    )


class MatchedManifestVideoDataset(Dataset):
    """Return metadata eagerly; teacher cache/decode only on explicit request."""

    def __init__(
        self,
        manifest_path: str | os.PathLike,
        video_root: str | os.PathLike,
        cache_root: str | os.PathLike,
        expected_model_sha256: str,
        expected_feature_contract_sha256: str,
        *,
        expected_source_csv_sha256: str | None = None,
        expected_count: int | None = None,
        feature_contract: Mapping[str, Any] | None = None,
        decoder_contract: Mapping[str, Any] | None = None,
        decode_fn: DecodeFunction | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.video_root = Path(video_root).expanduser().resolve()
        if not self.video_root.is_dir():
            raise NotADirectoryError(self.video_root)
        self.entries, self.manifest_metadata = load_matched_video_manifest(
            self.manifest_path,
            expected_source_csv_sha256=expected_source_csv_sha256,
            expected_count=expected_count,
        )
        if not self.entries:
            raise ValueError("matched-video manifest is empty")
        self.feature_cache = MatchedVideoFeatureCache(
            cache_root,
            expected_model_sha256=expected_model_sha256,
            expected_feature_contract_sha256=expected_feature_contract_sha256,
            feature_contract=(
                feature_contract
                if feature_contract is not None
                else MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT
            ),
            decoder_contract=(
                decoder_contract
                if decoder_contract is not None
                else MATCHED_PYAV_LAST64_DECODER_CONTRACT
            ),
        )
        self.decoder_contract = self.feature_cache.decoder_contract
        self.decode_fn = decode_fn if decode_fn is not None else _default_decode
        self._cache_key_to_index: dict[str, int] = {}
        self._sample_id_to_index: dict[str, int] = {}
        for index, entry in enumerate(self.entries):
            cache_key = self.feature_cache.key_for(entry)
            if cache_key in self._cache_key_to_index:
                raise ValueError(f"duplicate matched-video cache key: {cache_key}")
            if entry.sample_id in self._sample_id_to_index:
                raise ValueError(f"duplicate sample_id: {entry.sample_id}")
            self._cache_key_to_index[cache_key] = index
            self._sample_id_to_index[entry.sample_id] = index

    def __len__(self) -> int:
        return len(self.entries)

    def _entry_for_sample(self, sample_id: str) -> MatchedVideoManifestEntry:
        try:
            return self.entries[self._sample_id_to_index[str(sample_id)]]
        except KeyError as exc:
            raise KeyError(f"unknown matched-video sample_id: {sample_id!r}") from exc

    def _video_path_and_verify_stat(
        self,
        entry: MatchedVideoManifestEntry,
    ) -> Path:
        path = self.video_root / entry.video_name
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        actual = (int(stat.st_size), int(stat.st_mtime_ns))
        expected = (entry.video_size_bytes, entry.video_mtime_ns)
        if actual != expected:
            raise ValueError(
                "matched video is stale after manifest freeze for "
                f"sample_id={entry.sample_id}: actual={actual}, "
                f"expected={expected}"
            )
        return path

    def cache_key_for_sample(self, sample_id: str) -> str:
        return self.feature_cache.key_for(self._entry_for_sample(sample_id))

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return only prompt/sample/path metadata; never touch teacher data."""

        entry = self.entries[int(index)]
        video_path = self._video_path_and_verify_stat(entry)
        return {
            "idx": entry.manifest_index,
            "prompts": entry.caption,
            "sample_id": entry.sample_id,
            "teacher_cache_key": self.feature_cache.key_for(entry),
            "video_name": entry.video_name,
            "video_path": str(video_path),
        }

    def load_teacher_payload(self, sample_id: str) -> dict[str, Any]:
        """Load/decode teacher input only after suffix replay selects a sample."""

        entry = self._entry_for_sample(sample_id)
        video_path = self._video_path_and_verify_stat(entry)
        cache_key = self.feature_cache.key_for(entry)
        cached = self.feature_cache.load(entry, cache_key=cache_key)
        payload: dict[str, Any] = {
            "sample_id": entry.sample_id,
            "teacher_cache_key": cache_key,
            "video_name": entry.video_name,
            "video_path": str(video_path),
        }
        if cached is not None:
            payload.update({
                "current_teacher_cache_hit": True,
                "current_teacher_decode_seconds": 0.0,
                "current_teacher_features": cached,
            })
            return payload

        decode_started = time.perf_counter()
        decoded = self.decode_fn(video_path, self.decoder_contract)
        decode_seconds = time.perf_counter() - decode_started
        if not isinstance(decoded, DecodedMatchedVideo):
            raise TypeError("decode_fn must return DecodedMatchedVideo")
        expected_count = int(self.decoder_contract["expected_source_frame_count"])
        expected_frames = int(self.decoder_contract["num_frames"])
        expected_indices = torch.arange(
            expected_count - expected_frames,
            expected_count,
            dtype=torch.long,
        )
        if (
            int(decoded.source_frame_count) != expected_count
            or not isinstance(decoded.frame_indices, torch.Tensor)
            or decoded.frame_indices.dtype != torch.long
            or not torch.equal(
                decoded.frame_indices.detach().cpu(), expected_indices
            )
        ):
            raise ValueError(
                "decode_fn violated the strict last-contiguous 17..80 contract"
            )
        video = decoded.video01
        if (
            not isinstance(video, torch.Tensor)
            or video.dtype != torch.float32
            or video.ndim != 4
            or tuple(video.shape[:2]) != (expected_frames, 3)
            or not bool(torch.isfinite(video).all().item())
            or float(video.min()) < 0.0
            or float(video.max()) > 1.0
        ):
            raise ValueError(
                "decode_fn returned invalid video01; expected finite FP32 "
                f"[{expected_frames},3,H,W] in [0,1]"
            )
        payload.update({
            "current_teacher_cache_hit": False,
            "current_teacher_decode_seconds": decode_seconds,
            "current_teacher_frame_indices": decoded.frame_indices,
            "current_teacher_source_frame_count": expected_count,
            "current_teacher_video01": video.contiguous(),
        })
        return payload

    def write_feature_cache(
        self,
        cache_key: str,
        feature: torch.Tensor,
        *,
        sample_id: str,
    ) -> Path:
        try:
            index = self._cache_key_to_index[str(cache_key)]
        except KeyError as exc:
            raise KeyError(f"unknown teacher cache key: {cache_key!r}") from exc
        entry = self.entries[index]
        if entry.sample_id != str(sample_id):
            raise ValueError(
                "sample_id does not own teacher cache key: "
                f"{sample_id!r} != {entry.sample_id!r}"
            )
        self._video_path_and_verify_stat(entry)
        return self.feature_cache.write(
            entry, feature, cache_key=str(cache_key)
        )

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collate_matched_video_metadata(batch)


def collate_matched_video_metadata(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collate only metadata; teacher payloads intentionally have no collator."""

    if not batch:
        raise ValueError("cannot collate an empty matched-video batch")
    forbidden = {
        "current_teacher_features",
        "current_teacher_video01",
        "current_teacher_cache_hit",
    }
    for item in batch:
        present = forbidden.intersection(item)
        if present:
            raise ValueError(
                "metadata collator received eager teacher payload fields: "
                f"{sorted(present)}"
            )
    return {
        "idx": torch.tensor([int(item["idx"]) for item in batch]),
        "prompts": [str(item["prompts"]) for item in batch],
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "teacher_cache_keys": [str(item["teacher_cache_key"]) for item in batch],
        "video_names": [str(item["video_name"]) for item in batch],
        "video_paths": [str(item["video_path"]) for item in batch],
    }


__all__ = [
    "DecodedMatchedVideo",
    "MatchedManifestVideoDataset",
    "collate_matched_video_metadata",
    "decode_matched_video_last_contiguous_pyav",
]

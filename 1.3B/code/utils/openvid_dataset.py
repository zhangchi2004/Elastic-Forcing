"""OpenVid prompt/video dataset with lazy teacher-feature caching."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.openvid_feature_cache import (
    OPENVID_PYAV_DECODER_CONTRACT,
    OPENVID_VJEPA2_L15_FEATURE_CONTRACT,
    OpenVidFeatureCache,
)
from utils.openvid_manifest import (
    OpenVidManifestEntry,
    load_openvid_manifest,
)


@dataclass(frozen=True)
class DecodedOpenVidVideo:
    """Central contiguous RGB frame window in V-JEPA2 input range."""

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
            raise ValueError(
                f"decoded RGB frame must be uint8, got {array.dtype}"
            )
        current_shape = tuple(int(value) for value in array.shape)
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError(
                "video changes resolution inside selected frame window: "
                f"{current_shape} != {shape}"
            )
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        tensors.append(tensor)
    if not tensors:
        raise ValueError("decoded frame window is empty")
    return torch.stack(tensors).to(dtype=torch.float32).div_(255.0)


def _decode_window_and_count(
    av_module: Any,
    video_path: Path,
    *,
    start: int,
    num_frames: int,
) -> tuple[list[Any], int]:
    selected: list[Any] = []
    decoded_count = 0
    with av_module.open(str(video_path), mode="r") as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise ValueError(
                f"expected exactly one video stream in {video_path}, "
                f"got {len(streams)}"
            )
        stream = streams[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            decoded_count = frame_index + 1
            if start <= frame_index < start + num_frames:
                selected.append(frame)
    return selected, decoded_count


def decode_openvid_center_contiguous_pyav(
    video_path: str | os.PathLike,
    *,
    decoder_contract: Mapping[str, Any] | None = None,
) -> DecodedOpenVidVideo:
    """Decode the exact central contiguous frame window using PyAV order."""

    contract = dict(
        decoder_contract
        if decoder_contract is not None
        else OPENVID_PYAV_DECODER_CONTRACT
    )
    if contract.get("backend") != "pyav":
        raise ValueError("decoder contract backend must be pyav")
    if contract.get("selection") != "center_contiguous":
        raise ValueError("decoder contract selection must be center_contiguous")
    if contract.get("center_tie_break") != "floor":
        raise ValueError("decoder contract center_tie_break must be floor")
    num_frames = int(contract.get("num_frames", 0))
    if num_frames <= 0:
        raise ValueError("decoder contract num_frames must be positive")
    try:
        import av  # Imported lazily so complete cache hits need no decoder.
    except ImportError as exc:  # pragma: no cover - exercised without PyAV
        raise ImportError(
            "PyAV is required for an OpenVid teacher cache miss; install "
            "the pinned av==13.1.0 dependency"
        ) from exc
    expected_version = str(contract.get("backend_version", ""))
    actual_version = str(getattr(av, "__version__", ""))
    if actual_version != expected_version:
        raise RuntimeError(
            "PyAV version differs from the cache decoder contract: "
            f"{actual_version!r} != {expected_version!r}"
        )

    video_path = Path(video_path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    with av.open(str(video_path), mode="r") as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise ValueError(
                f"expected exactly one video stream in {video_path}, "
                f"got {len(streams)}"
            )
        frame_count_hint = int(getattr(streams[0], "frames", 0) or 0)
    provisional_start = max((frame_count_hint - num_frames) // 2, 0)
    selected, actual_count = _decode_window_and_count(
        av, video_path, start=provisional_start, num_frames=num_frames)
    if actual_count < num_frames:
        raise ValueError(
            f"OpenVid video has only {actual_count} decoded frames; "
            f"need {num_frames}: {video_path}"
        )
    actual_start = (actual_count - num_frames) // 2
    if actual_start != provisional_start or len(selected) != num_frames:
        selected, repeated_count = _decode_window_and_count(
            av, video_path, start=actual_start, num_frames=num_frames)
        if repeated_count != actual_count:
            raise RuntimeError(
                "PyAV returned a different frame count on deterministic "
                f"re-decode: {repeated_count} != {actual_count}"
            )
    if len(selected) != num_frames:
        raise RuntimeError(
            f"failed to decode central {num_frames}-frame window from {video_path}"
        )
    video01 = _frames_to_video01(selected)
    if tuple(video01.shape[:2]) != (num_frames, 3):
        raise RuntimeError(
            f"decoded video has invalid shape {tuple(video01.shape)}"
        )
    frame_indices = torch.arange(
        actual_start,
        actual_start + num_frames,
        dtype=torch.long,
    )
    return DecodedOpenVidVideo(
        video01=video01,
        frame_indices=frame_indices,
        source_frame_count=actual_count,
    )


DecodeFunction = Callable[[Path, Mapping[str, Any]], DecodedOpenVidVideo]


def _default_decode(
    path: Path,
    contract: Mapping[str, Any],
) -> DecodedOpenVidVideo:
    return decode_openvid_center_contiguous_pyav(
        path, decoder_contract=contract)


class OpenVidPromptVideoDataset(Dataset):
    """Return a prompt plus either cached teacher features or decoded video."""

    def __init__(
        self,
        manifest_path: str | os.PathLike,
        video_root: str | os.PathLike,
        cache_root: str | os.PathLike,
        expected_model_sha256: str,
        expected_feature_contract_sha256: str,
        *,
        feature_contract: Mapping[str, Any] | None = None,
        decoder_contract: Mapping[str, Any] | None = None,
        decode_fn: DecodeFunction | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.video_root = Path(video_root).expanduser().resolve()
        if not self.video_root.is_dir():
            raise NotADirectoryError(self.video_root)
        self.entries, self.manifest_metadata = load_openvid_manifest(
            self.manifest_path)
        if not self.entries:
            raise ValueError("OpenVid manifest is empty")
        self.feature_cache = OpenVidFeatureCache(
            cache_root,
            expected_model_sha256=expected_model_sha256,
            expected_feature_contract_sha256=(
                expected_feature_contract_sha256),
            feature_contract=(
                feature_contract
                if feature_contract is not None
                else OPENVID_VJEPA2_L15_FEATURE_CONTRACT
            ),
            decoder_contract=(
                decoder_contract
                if decoder_contract is not None
                else OPENVID_PYAV_DECODER_CONTRACT
            ),
        )
        self.decoder_contract = self.feature_cache.decoder_contract
        self.decode_fn = decode_fn if decode_fn is not None else _default_decode
        self._cache_key_to_index: dict[str, int] = {}
        self._sample_id_to_index: dict[str, int] = {}
        for index, entry in enumerate(self.entries):
            cache_key = self.feature_cache.key_for(entry)
            if cache_key in self._cache_key_to_index:
                raise ValueError(f"duplicate OpenVid teacher cache key: {cache_key}")
            self._cache_key_to_index[cache_key] = index
            self._sample_id_to_index[entry.sample_id] = index

    def __len__(self) -> int:
        return len(self.entries)

    def _video_path_and_verify_stat(
        self,
        entry: OpenVidManifestEntry,
    ) -> Path:
        path = self.video_root / entry.video_name
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        actual = (int(stat.st_size), int(stat.st_mtime_ns))
        expected = (entry.video_size_bytes, entry.video_mtime_ns)
        if actual != expected:
            raise ValueError(
                "OpenVid video stat changed after manifest freeze for "
                f"{entry.video_name}: actual={actual}, expected={expected}"
            )
        return path

    def cache_key_for_sample(self, sample_id: str) -> str:
        try:
            entry = self.entries[self._sample_id_to_index[str(sample_id)]]
        except KeyError as exc:
            raise KeyError(f"unknown OpenVid sample_id: {sample_id!r}") from exc
        return self.feature_cache.key_for(entry)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[int(index)]
        video_path = self._video_path_and_verify_stat(entry)
        cache_key = self.feature_cache.key_for(entry)
        cached = self.feature_cache.load(entry, cache_key=cache_key)
        item: dict[str, Any] = {
            "idx": entry.manifest_index,
            "prompts": entry.caption,
            "sample_id": entry.sample_id,
            "teacher_cache_key": cache_key,
            "video_name": entry.video_name,
            "video_path": str(video_path),
        }
        if cached is not None:
            item["current_teacher_cache_hit"] = True
            item["current_teacher_decode_seconds"] = 0.0
            item["current_teacher_features"] = cached
            return item

        decode_started = time.perf_counter()
        decoded = self.decode_fn(video_path, self.decoder_contract)
        decode_seconds = time.perf_counter() - decode_started
        if not isinstance(decoded, DecodedOpenVidVideo):
            raise TypeError("decode_fn must return DecodedOpenVidVideo")
        expected_frames = int(self.decoder_contract["num_frames"])
        source_frame_count = int(decoded.source_frame_count)
        expected_start = (source_frame_count - expected_frames) // 2
        expected_indices = torch.arange(
            expected_start,
            expected_start + expected_frames,
            dtype=torch.long,
        )
        if (
            source_frame_count < expected_frames
            or not isinstance(decoded.frame_indices, torch.Tensor)
            or decoded.frame_indices.dtype != torch.long
            or not torch.equal(
                decoded.frame_indices.detach().cpu(), expected_indices)
        ):
            raise ValueError(
                "decode_fn violated the central-contiguous frame-index "
                "contract"
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
        item.update({
            "current_teacher_cache_hit": False,
            "current_teacher_decode_seconds": decode_seconds,
            "current_teacher_frame_indices": decoded.frame_indices,
            "current_teacher_source_frame_count": source_frame_count,
            "current_teacher_video01": video.contiguous(),
        })
        return item

    def write_feature_cache(
        self,
        cache_key: str,
        feature: torch.Tensor,
        *,
        sample_id: str,
    ) -> Path:
        """Atomically publish one extracted teacher feature by issued key."""

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
            entry,
            feature,
            cache_key=str(cache_key),
        )

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collate_openvid_prompt_video(batch)


def collate_openvid_prompt_video(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collate the cache-hit/cache-miss union used by batch-size-one training."""

    if not batch:
        raise ValueError("cannot collate an empty OpenVid batch")
    hits = torch.tensor(
        [bool(item["current_teacher_cache_hit"]) for item in batch],
        dtype=torch.bool,
    )
    result: dict[str, Any] = {
        "idx": torch.tensor([int(item["idx"]) for item in batch]),
        "prompts": [str(item["prompts"]) for item in batch],
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "teacher_cache_keys": [
            str(item["teacher_cache_key"]) for item in batch
        ],
        "video_names": [str(item["video_name"]) for item in batch],
        "video_paths": [str(item["video_path"]) for item in batch],
        "current_teacher_cache_hit": hits,
        "current_teacher_decode_seconds": sum(
            float(item.get("current_teacher_decode_seconds", 0.0))
            for item in batch
        ),
    }
    if bool(hits.all().item()):
        result["current_teacher_features"] = torch.stack([
            item["current_teacher_features"] for item in batch
        ])
        return result
    if bool((~hits).all().item()):
        videos = [item["current_teacher_video01"] for item in batch]
        shapes = {tuple(video.shape) for video in videos}
        if len(shapes) != 1:
            raise ValueError(
                "OpenVid raw videos have different spatial shapes; this "
                "collate path requires batch_size=1"
            )
        result.update({
            "current_teacher_frame_indices": torch.stack([
                item["current_teacher_frame_indices"] for item in batch
            ]),
            "current_teacher_source_frame_count": torch.tensor([
                int(item["current_teacher_source_frame_count"])
                for item in batch
            ]),
            "current_teacher_video01": torch.stack(videos),
        })
        return result
    raise ValueError(
        "mixed OpenVid teacher cache hits/misses cannot be collated; use the "
        "configured batch_size=1 while lazily populating the cache"
    )


__all__ = [
    "DecodedOpenVidVideo",
    "OpenVidPromptVideoDataset",
    "collate_openvid_prompt_video",
    "decode_openvid_center_contiguous_pyav",
]

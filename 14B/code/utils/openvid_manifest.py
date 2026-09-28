"""Deterministic OpenVid prompt/video manifest construction and validation.

The training subset is selected only from the official part mapping.  Caption
metadata is joined by the OpenVid video identifier (the MP4 filename), never by
caption text: captions are not unique in OpenVid.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_FORMAT = "sf_openvid_prompt_video_manifest"
MANIFEST_FORMAT_VERSION = 1
DEFAULT_SELECTION_SEED = "0"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: str | os.PathLike, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_digest(seed: str, video_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(seed.encode("utf-8"))
    digest.update(b"\0")
    digest.update(video_name.encode("utf-8"))
    return digest.hexdigest()


def _manifest_meta_path(manifest_path: Path) -> Path:
    if manifest_path.suffix == ".jsonl":
        return manifest_path.with_suffix(".meta.json")
    return manifest_path.with_name(manifest_path.name + ".meta.json")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class OpenVidManifestEntry:
    """One immutable prompt/video identity from a frozen manifest."""

    manifest_index: int
    sample_id: str
    video_name: str
    caption: str
    video_size_bytes: int
    video_mtime_ns: int
    selection_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OpenVidManifestEntry":
        required = {
            "manifest_index",
            "sample_id",
            "video_name",
            "caption",
            "video_size_bytes",
            "video_mtime_ns",
            "selection_sha256",
        }
        missing = required - set(value)
        unexpected = set(value) - required
        if missing or unexpected:
            raise ValueError(
                "OpenVid manifest entry schema mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        entry = cls(
            manifest_index=int(value["manifest_index"]),
            sample_id=str(value["sample_id"]),
            video_name=str(value["video_name"]),
            caption=str(value["caption"]),
            video_size_bytes=int(value["video_size_bytes"]),
            video_mtime_ns=int(value["video_mtime_ns"]),
            selection_sha256=str(value["selection_sha256"]),
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if self.manifest_index < 0:
            raise ValueError("manifest_index must be non-negative")
        if not self.sample_id or self.sample_id != self.video_name:
            raise ValueError(
                "sample_id must be the non-empty OpenVid video identifier"
            )
        if Path(self.video_name).name != self.video_name:
            raise ValueError("video_name must be a basename")
        if not self.video_name.lower().endswith(".mp4"):
            raise ValueError("video_name must identify an MP4")
        if not self.caption.strip():
            raise ValueError("caption must be non-empty")
        if self.video_size_bytes <= 0 or self.video_mtime_ns <= 0:
            raise ValueError("video stat must contain positive size and mtime")
        if not _is_sha256(self.selection_sha256):
            raise ValueError("selection_sha256 must be lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "manifest_index": self.manifest_index,
            "sample_id": self.sample_id,
            "selection_sha256": self.selection_sha256,
            "video_mtime_ns": self.video_mtime_ns,
            "video_name": self.video_name,
            "video_size_bytes": self.video_size_bytes,
        }


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_part_video_names(mapping_csv: Path, expected_zip: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    with mapping_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"zip_file", "video"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"mapping CSV must contain {sorted(required)}; "
                f"got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            if row["zip_file"] != expected_zip:
                raise ValueError(
                    f"unexpected zip_file at mapping row {row_number}: "
                    f"{row['zip_file']!r} != {expected_zip!r}"
                )
            name = row["video"].strip()
            if not name or Path(name).name != name:
                raise ValueError(
                    f"invalid video identifier at mapping row {row_number}"
                )
            if name in seen:
                raise ValueError(f"duplicate mapping video identifier: {name}")
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError("mapping CSV is empty")
    return names


def _join_captions(
    metadata_csv: Path,
    video_names: Sequence[str],
) -> dict[str, str]:
    wanted = set(video_names)
    captions: dict[str, str] = {}
    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"video", "caption"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"metadata CSV must contain {sorted(required)}; "
                f"got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            name = row["video"].strip()
            if name not in wanted:
                continue
            if name in captions:
                raise ValueError(
                    "metadata CSV contains duplicate rows for selected video "
                    f"{name!r} (second occurrence at row {row_number})"
                )
            caption = row["caption"].strip()
            if not caption:
                raise ValueError(f"empty caption for selected video {name!r}")
            captions[name] = caption
    missing = sorted(wanted - set(captions))
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        raise ValueError(
            f"metadata CSV is missing {len(missing)} mapped videos: {preview}"
        )
    return captions


def _serialize_entries(entries: Iterable[OpenVidManifestEntry]) -> bytes:
    return (
        "".join(_canonical_json(entry.to_dict()) + "\n" for entry in entries)
        .encode("utf-8")
    )


def build_openvid_part_manifest(
    *,
    mapping_csv: str | os.PathLike,
    metadata_csv: str | os.PathLike,
    video_root: str | os.PathLike,
    manifest_path: str | os.PathLike,
    target_count: int = 12_800,
    seed: str = DEFAULT_SELECTION_SEED,
    expected_zip: str = "OpenVid_part1.zip",
) -> dict[str, Any]:
    """Build and atomically publish a deterministic, stat-frozen manifest."""

    mapping_csv = Path(mapping_csv).expanduser().resolve()
    metadata_csv = Path(metadata_csv).expanduser().resolve()
    video_root = Path(video_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if not seed:
        raise ValueError("selection seed must be non-empty")
    for source in (mapping_csv, metadata_csv):
        if not source.is_file():
            raise FileNotFoundError(source)
    if not video_root.is_dir():
        raise NotADirectoryError(video_root)

    video_names = _read_part_video_names(mapping_csv, expected_zip)
    if target_count > len(video_names):
        raise ValueError(
            f"target_count={target_count} exceeds mapped videos={len(video_names)}"
        )
    captions = _join_captions(metadata_csv, video_names)
    ranked = sorted(
        ((_selection_digest(seed, name), name) for name in video_names),
        key=lambda item: (item[0], item[1]),
    )[:target_count]

    entries: list[OpenVidManifestEntry] = []
    for manifest_index, (selection_hash, video_name) in enumerate(ranked):
        path = video_root / video_name
        if not path.is_file():
            raise FileNotFoundError(
                f"mapped OpenVid video is absent from video_root: {path}"
            )
        stat = path.stat()
        entry = OpenVidManifestEntry(
            manifest_index=manifest_index,
            sample_id=video_name,
            video_name=video_name,
            caption=captions[video_name],
            video_size_bytes=int(stat.st_size),
            video_mtime_ns=int(stat.st_mtime_ns),
            selection_sha256=selection_hash,
        )
        entry.validate()
        entries.append(entry)

    manifest_payload = _serialize_entries(entries)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    selected_identity_sha256 = hashlib.sha256(
        b"".join(
            bytes.fromhex(entry.selection_sha256) + b"\0"
            + entry.video_name.encode("utf-8") + b"\0"
            for entry in entries
        )
    ).hexdigest()
    metadata = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_FORMAT_VERSION,
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "mapped_video_count": len(video_names),
        "selected_identity_sha256": selected_identity_sha256,
        "selected_video_count": len(entries),
        "selection": {
            "algorithm": "sort_sha256(seed_utf8 + NUL + video_name_utf8)",
            "seed": seed,
            "tie_breaker": "video_name_utf8",
            "take": target_count,
        },
        "source": {
            "expected_zip": expected_zip,
            "mapping_csv": str(mapping_csv),
            "mapping_csv_sha256": sha256_file(mapping_csv),
            "metadata_csv": str(metadata_csv),
            "metadata_csv_sha256": sha256_file(metadata_csv),
            "video_root": str(video_root),
        },
    }
    metadata_payload = (
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(manifest_path, manifest_payload)
    _atomic_write_bytes(_manifest_meta_path(manifest_path), metadata_payload)
    return metadata


def load_openvid_manifest(
    manifest_path: str | os.PathLike,
) -> tuple[list[OpenVidManifestEntry], dict[str, Any]]:
    """Load a frozen manifest and strictly verify its sidecar checksum."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    meta_path = _manifest_meta_path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("format") != MANIFEST_FORMAT:
        raise ValueError("unsupported OpenVid manifest format")
    if int(metadata.get("format_version", -1)) != MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported OpenVid manifest format_version")
    if metadata.get("manifest_file") != manifest_path.name:
        raise ValueError("manifest sidecar points to a different filename")
    actual_hash = sha256_file(manifest_path)
    expected_hash = metadata.get("manifest_sha256")
    if not _is_sha256(str(expected_hash)) or actual_hash != expected_hash:
        raise ValueError(
            "OpenVid manifest SHA-256 mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )

    entries: list[OpenVidManifestEntry] = []
    seen_sample_ids: set[str] = set()
    seen_video_names: set[str] = set()
    previous_rank: tuple[str, str] | None = None
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank manifest line {line_number}")
            entry = OpenVidManifestEntry.from_mapping(json.loads(line))
            if entry.manifest_index != len(entries):
                raise ValueError(
                    f"non-contiguous manifest_index at line {line_number}"
                )
            rank = (entry.selection_sha256, entry.video_name)
            if previous_rank is not None and rank <= previous_rank:
                raise ValueError("manifest is not in deterministic selection order")
            previous_rank = rank
            if entry.sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id: {entry.sample_id}")
            if entry.video_name in seen_video_names:
                raise ValueError(f"duplicate video_name: {entry.video_name}")
            seen_sample_ids.add(entry.sample_id)
            seen_video_names.add(entry.video_name)
            entries.append(entry)
    expected_count = int(metadata.get("selected_video_count", -1))
    if len(entries) != expected_count:
        raise ValueError(
            f"manifest row count mismatch: {len(entries)} != {expected_count}"
        )
    return entries, metadata


__all__ = [
    "DEFAULT_SELECTION_SEED",
    "MANIFEST_FORMAT",
    "MANIFEST_FORMAT_VERSION",
    "OpenVidManifestEntry",
    "build_openvid_part_manifest",
    "load_openvid_manifest",
    "sha256_file",
]

"""Frozen prompt/video manifests for the generic matched-teacher data source.

Rows are joined to videos exclusively by the MP4 basename.  Prompt text is
never an identity key because prompts need not be unique.  The two bookkeeping
columns used by the source dataset (``used_by_sim`` and ``used_by_vjepa2``) are
deliberately ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


MANIFEST_FORMAT = "sf_matched_manifest_video"
MANIFEST_FORMAT_VERSION = 1
DEFAULT_EXPECTED_SOURCE_CSV_SHA256 = (
    "fe8ee308715e60c35856d0c7a4df71759f4969801312befacfa3ba3309381397"
)
DEFAULT_EXPECTED_COUNT = 5_000


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


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
class MatchedVideoManifestEntry:
    """One immutable prompt/video pair in CSV row order."""

    manifest_index: int
    source_index: int
    sample_id: str
    video_name: str
    caption: str
    video_size_bytes: int
    video_mtime_ns: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "MatchedVideoManifestEntry":
        required = {
            "manifest_index",
            "source_index",
            "sample_id",
            "video_name",
            "caption",
            "video_size_bytes",
            "video_mtime_ns",
        }
        missing = required - set(value)
        unexpected = set(value) - required
        if missing or unexpected:
            raise ValueError(
                "matched-video manifest entry schema mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        entry = cls(
            manifest_index=int(value["manifest_index"]),
            source_index=int(value["source_index"]),
            sample_id=str(value["sample_id"]),
            video_name=str(value["video_name"]),
            caption=str(value["caption"]),
            video_size_bytes=int(value["video_size_bytes"]),
            video_mtime_ns=int(value["video_mtime_ns"]),
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if self.manifest_index < 0 or self.source_index < 0:
            raise ValueError("manifest/source indices must be non-negative")
        if not self.sample_id or self.sample_id != self.video_name:
            raise ValueError("sample_id must equal the MP4 basename")
        if Path(self.video_name).name != self.video_name:
            raise ValueError("video_name must be a basename")
        if not self.video_name.lower().endswith(".mp4"):
            raise ValueError("video_name must identify an MP4")
        if not self.caption.strip():
            raise ValueError("caption must be non-empty")
        if self.video_size_bytes <= 0 or self.video_mtime_ns <= 0:
            raise ValueError("video stat must contain positive size and mtime")

    def to_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "manifest_index": self.manifest_index,
            "sample_id": self.sample_id,
            "source_index": self.source_index,
            "video_mtime_ns": self.video_mtime_ns,
            "video_name": self.video_name,
            "video_size_bytes": self.video_size_bytes,
        }


def _serialize_entries(entries: Iterable[MatchedVideoManifestEntry]) -> bytes:
    return (
        "".join(_canonical_json(entry.to_dict()) + "\n" for entry in entries)
        .encode("utf-8")
    )


def build_matched_video_manifest(
    *,
    source_csv: str | os.PathLike,
    video_root: str | os.PathLike,
    manifest_path: str | os.PathLike,
    expected_source_csv_sha256: str | None = (
        DEFAULT_EXPECTED_SOURCE_CSV_SHA256
    ),
    expected_count: int | None = DEFAULT_EXPECTED_COUNT,
) -> dict[str, Any]:
    """Freeze every CSV prompt/MP4 pair and publish JSONL plus sidecar."""

    source_csv = Path(source_csv).expanduser().resolve()
    video_root = Path(video_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    if not video_root.is_dir():
        raise NotADirectoryError(video_root)
    source_hash = sha256_file(source_csv)
    if expected_source_csv_sha256 is not None:
        expected_hash = str(expected_source_csv_sha256).lower()
        if not _is_sha256(expected_hash):
            raise ValueError(
                "expected_source_csv_sha256 must be lowercase SHA-256"
            )
        if source_hash != expected_hash:
            raise ValueError(
                "source CSV SHA-256 mismatch: "
                f"expected={expected_hash}, actual={source_hash}"
            )
    if expected_count is not None and int(expected_count) <= 0:
        raise ValueError("expected_count must be positive or None")

    entries: list[MatchedVideoManifestEntry] = []
    seen_sample_ids: set[str] = set()
    with source_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"idx", "mp4", "prompt"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"source CSV must contain {sorted(required)}; "
                f"got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                source_index = int(row["idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid idx at source CSV row {row_number}"
                ) from exc
            if source_index != len(entries):
                raise ValueError(
                    "source CSV idx must be contiguous in row order: "
                    f"row={row_number}, idx={source_index}, "
                    f"expected={len(entries)}"
                )
            video_name = str(row["mp4"]).strip()
            if (
                not video_name
                or Path(video_name).name != video_name
                or not video_name.lower().endswith(".mp4")
            ):
                raise ValueError(f"invalid MP4 basename at row {row_number}")
            if video_name in seen_sample_ids:
                raise ValueError(f"duplicate MP4/sample_id: {video_name}")
            caption = str(row["prompt"]).strip()
            if not caption:
                raise ValueError(f"empty prompt for {video_name!r}")
            path = video_root / video_name
            if not path.is_file():
                raise FileNotFoundError(
                    f"CSV video is absent from video_root: {path}"
                )
            stat = path.stat()
            entry = MatchedVideoManifestEntry(
                manifest_index=len(entries),
                source_index=source_index,
                sample_id=video_name,
                video_name=video_name,
                caption=caption,
                video_size_bytes=int(stat.st_size),
                video_mtime_ns=int(stat.st_mtime_ns),
            )
            entry.validate()
            entries.append(entry)
            seen_sample_ids.add(video_name)

    if expected_count is not None and len(entries) != int(expected_count):
        raise ValueError(
            f"source row count mismatch: {len(entries)} != {int(expected_count)}"
        )
    if not entries:
        raise ValueError("source CSV is empty")

    manifest_payload = _serialize_entries(entries)
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    identity_hash = hashlib.sha256(
        b"".join(
            entry.sample_id.encode("utf-8")
            + b"\0"
            + str(entry.video_size_bytes).encode("ascii")
            + b"\0"
            + str(entry.video_mtime_ns).encode("ascii")
            + b"\0"
            for entry in entries
        )
    ).hexdigest()
    metadata = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_FORMAT_VERSION,
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_hash,
        "sample_count": len(entries),
        "sample_identity_sha256": identity_hash,
        "source": {
            "csv": str(source_csv),
            "csv_sha256": source_hash,
            "ignored_columns": ["used_by_sim", "used_by_vjepa2"],
            "join_key": "mp4_basename",
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


def load_matched_video_manifest(
    manifest_path: str | os.PathLike,
    *,
    expected_source_csv_sha256: str | None = None,
    expected_count: int | None = None,
) -> tuple[list[MatchedVideoManifestEntry], dict[str, Any]]:
    """Load a frozen manifest and strictly verify its immutable identity."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    meta_path = _manifest_meta_path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("format") != MANIFEST_FORMAT:
        raise ValueError("unsupported matched-video manifest format")
    if int(metadata.get("format_version", -1)) != MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported matched-video manifest format_version")
    if metadata.get("manifest_file") != manifest_path.name:
        raise ValueError("manifest sidecar points to a different filename")
    expected_manifest_hash = str(metadata.get("manifest_sha256", ""))
    actual_manifest_hash = sha256_file(manifest_path)
    if (
        not _is_sha256(expected_manifest_hash)
        or actual_manifest_hash != expected_manifest_hash
    ):
        raise ValueError(
            "matched-video manifest SHA-256 mismatch: "
            f"expected={expected_manifest_hash}, actual={actual_manifest_hash}"
        )
    source_hash = str(metadata.get("source", {}).get("csv_sha256", ""))
    if not _is_sha256(source_hash):
        raise ValueError("manifest sidecar source CSV hash is invalid")
    if (
        expected_source_csv_sha256 is not None
        and source_hash != str(expected_source_csv_sha256).lower()
    ):
        raise ValueError(
            "manifest source CSV SHA-256 mismatch: "
            f"expected={expected_source_csv_sha256}, actual={source_hash}"
        )

    entries: list[MatchedVideoManifestEntry] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank manifest line {line_number}")
            entry = MatchedVideoManifestEntry.from_mapping(json.loads(line))
            if entry.manifest_index != len(entries):
                raise ValueError(
                    f"non-contiguous manifest_index at line {line_number}"
                )
            if entry.sample_id in seen:
                raise ValueError(f"duplicate sample_id: {entry.sample_id}")
            seen.add(entry.sample_id)
            entries.append(entry)
    sidecar_count = int(metadata.get("sample_count", -1))
    if len(entries) != sidecar_count:
        raise ValueError(
            f"manifest row count mismatch: {len(entries)} != {sidecar_count}"
        )
    if expected_count is not None and len(entries) != int(expected_count):
        raise ValueError(
            f"expected manifest row count mismatch: "
            f"{len(entries)} != {int(expected_count)}"
        )
    return entries, metadata


__all__ = [
    "DEFAULT_EXPECTED_COUNT",
    "DEFAULT_EXPECTED_SOURCE_CSV_SHA256",
    "MANIFEST_FORMAT",
    "MANIFEST_FORMAT_VERSION",
    "MatchedVideoManifestEntry",
    "build_matched_video_manifest",
    "load_matched_video_manifest",
    "sha256_file",
]

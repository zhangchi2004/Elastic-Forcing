"""Immutable OpenVid row ledger shared by frozen video references.

Unlike prompt datasets, a feature-reference ledger retains videos whose caption
is empty.  Feature rows are identified exclusively by the contiguous source
sequence and MP4 basename; captions are descriptive payloads, never keys.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterable, Mapping


LEDGER_KIND = "openvid_video_reference_ledger"
LEDGER_SCHEMA_VERSION = 1


def file_sha256(path: str | os.PathLike, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(int(chunk_size)):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
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
class OpenVidReferenceRow:
    sequence: int
    role: str
    video_name: str
    caption: str
    video_size_bytes: int
    video_mtime_ns: int

    def validate(self, *, anchor_cohort_size: int) -> None:
        if self.sequence < 0:
            raise ValueError("OpenVid reference sequence must be non-negative")
        expected_role = (
            "anchor" if self.sequence < int(anchor_cohort_size) else "reference")
        if self.role != expected_role:
            raise ValueError(
                f"row {self.sequence} role {self.role!r} != {expected_role!r}")
        if (
            Path(self.video_name).name != self.video_name
            or not self.video_name.lower().endswith(".mp4")
        ):
            raise ValueError(f"invalid MP4 basename: {self.video_name!r}")
        if self.video_size_bytes <= 0 or self.video_mtime_ns <= 0:
            raise ValueError("video size/mtime identity must be positive")

    def to_dict(self) -> dict:
        return {
            "caption": self.caption,
            "role": self.role,
            "sequence": self.sequence,
            "video_mtime_ns": self.video_mtime_ns,
            "video_name": self.video_name,
            "video_size_bytes": self.video_size_bytes,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping, *, anchor_cohort_size: int
    ) -> "OpenVidReferenceRow":
        row = cls(
            sequence=int(value["sequence"]),
            role=str(value["role"]),
            video_name=str(value["video_name"]),
            caption=str(value.get("caption", "")),
            video_size_bytes=int(value["video_size_bytes"]),
            video_mtime_ns=int(value["video_mtime_ns"]),
        )
        if set(value) != set(row.to_dict()):
            raise ValueError("OpenVid reference row schema mismatch")
        row.validate(anchor_cohort_size=anchor_cohort_size)
        return row


class OpenVidReferencePromptDataset:
    """Prompt-only view of a frozen feature ledger.

    The nine captionless videos remain in the 77,615-row feature reference but
    are intentionally absent from text-conditioned student sampling.
    """

    def __init__(
        self,
        ledger_path: str | os.PathLike,
        *,
        expected_count: int = 77_615,
    ) -> None:
        rows, metadata = load_openvid_reference_ledger(
            ledger_path, expected_count=expected_count)
        self.metadata = metadata
        self.rows = tuple(row for row in rows if row.caption.strip())
        skipped = len(rows) - len(self.rows)
        if skipped != int(metadata["caption_empty_count"]):
            raise ValueError("OpenVid empty-caption ledger count mismatch")
        if not self.rows:
            raise ValueError("OpenVid ledger has no usable prompts")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[int(index)]
        return {
            "idx": row.sequence,
            "prompts": row.caption,
            "sample_ids": row.video_name,
        }


def _video_name_sequence_sha256(rows: Iterable[OpenVidReferenceRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.video_name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _local_source_identity_sha256(rows: Iterable[OpenVidReferenceRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.video_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.video_size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row.video_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def freeze_openvid_reference_ledger(
    *,
    source_csv: str | os.PathLike,
    video_root: str | os.PathLike,
    ledger_path: str | os.PathLike,
    expected_count: int = 77_615,
    anchor_cohort_size: int = 1_024,
) -> dict:
    """Validate the CSV/video mirror and atomically publish JSONL + metadata."""
    source_csv = Path(source_csv).expanduser().resolve(strict=True)
    video_root = Path(video_root).expanduser().resolve(strict=True)
    ledger_path = Path(ledger_path).expanduser().resolve()
    expected_count = int(expected_count)
    anchor_cohort_size = int(anchor_cohort_size)
    if not video_root.is_dir():
        raise NotADirectoryError(video_root)
    if not 0 < anchor_cohort_size <= expected_count:
        raise ValueError("invalid OpenVid anchor cohort size")

    rows: list[OpenVidReferenceRow] = []
    seen: set[str] = set()
    with source_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence", "role", "video", "caption"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"OpenVid reference CSV requires {sorted(required)}")
        for csv_line, value in enumerate(reader, start=2):
            try:
                sequence = int(value["sequence"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid sequence at CSV line {csv_line}") from exc
            if sequence != len(rows):
                raise ValueError(
                    "OpenVid sequence is not contiguous in CSV order: "
                    f"{sequence} != {len(rows)}")
            video_name = str(value["video"]).strip()
            if video_name in seen:
                raise ValueError(f"duplicate OpenVid video: {video_name}")
            video_path = video_root / video_name
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            stat = video_path.stat()
            row = OpenVidReferenceRow(
                sequence=sequence,
                role=str(value["role"]).strip(),
                video_name=video_name,
                # Empty captions are valid reference rows.
                caption=str(value["caption"] or ""),
                video_size_bytes=int(stat.st_size),
                video_mtime_ns=int(stat.st_mtime_ns),
            )
            row.validate(anchor_cohort_size=anchor_cohort_size)
            rows.append(row)
            seen.add(video_name)
    if len(rows) != expected_count:
        raise ValueError(
            f"OpenVid reference count {len(rows)} != {expected_count}")

    lines = "".join(
        json.dumps(
            row.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ) + "\n"
        for row in rows
    ).encode("utf-8")
    manifest_sha256 = hashlib.sha256(lines).hexdigest()
    metadata = {
        "anchor_cohort_size": anchor_cohort_size,
        "anchor_video_name_sequence_sha256": (
            _video_name_sequence_sha256(rows[:anchor_cohort_size])),
        "caption_empty_count": sum(not row.caption.strip() for row in rows),
        "csv_path": str(source_csv),
        "csv_sha256": file_sha256(source_csv),
        "kind": LEDGER_KIND,
        "ledger_file": ledger_path.name,
        "ledger_sha256": manifest_sha256,
        "local_source_identity_sha256": (
            _local_source_identity_sha256(rows)),
        "reference_count": len(rows),
        "schema_version": LEDGER_SCHEMA_VERSION,
        "video_name_sequence_sha256": _video_name_sequence_sha256(rows),
        "video_root": str(video_root),
    }
    _atomic_bytes(ledger_path, lines)
    _atomic_bytes(
        ledger_path.with_suffix(".meta.json"),
        (json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, indent=2,
            allow_nan=False,
        ) + "\n").encode("utf-8"),
    )
    return metadata


def load_openvid_reference_ledger(
    ledger_path: str | os.PathLike,
    *,
    expected_count: int | None = None,
) -> tuple[list[OpenVidReferenceRow], dict]:
    path = Path(ledger_path).expanduser().resolve(strict=True)
    meta_path = path.with_suffix(".meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        metadata.get("kind") != LEDGER_KIND
        or int(metadata.get("schema_version", -1)) != LEDGER_SCHEMA_VERSION
        or metadata.get("ledger_file") != path.name
        or metadata.get("ledger_sha256") != file_sha256(path)
    ):
        raise ValueError("OpenVid reference ledger metadata/hash mismatch")
    anchor_count = int(metadata["anchor_cohort_size"])
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank OpenVid ledger line {line_number}")
            row = OpenVidReferenceRow.from_mapping(
                json.loads(line), anchor_cohort_size=anchor_count)
            if row.sequence != len(rows):
                raise ValueError("OpenVid ledger sequence is not contiguous")
            rows.append(row)
    if len(rows) != int(metadata.get("reference_count", -1)):
        raise ValueError("OpenVid ledger row count mismatch")
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(
            f"OpenVid ledger count {len(rows)} != {int(expected_count)}")
    if (
        _video_name_sequence_sha256(rows)
        != metadata.get("video_name_sequence_sha256")
        or _video_name_sequence_sha256(rows[:anchor_count])
        != metadata.get("anchor_video_name_sequence_sha256")
        or _local_source_identity_sha256(rows)
        != metadata.get("local_source_identity_sha256")
    ):
        raise ValueError("OpenVid ledger row identity mismatch")
    return rows, metadata


__all__ = [
    "LEDGER_KIND",
    "LEDGER_SCHEMA_VERSION",
    "OpenVidReferenceRow",
    "OpenVidReferencePromptDataset",
    "canonical_sha256",
    "file_sha256",
    "freeze_openvid_reference_ledger",
    "load_openvid_reference_ledger",
]

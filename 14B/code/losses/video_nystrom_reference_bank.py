"""Strict multi-mask reference bank for video-level Nyström MMD.

The JSON manifest is intentionally small: it contains the exact teacher
train/validation split, an ordered list of token masks, and relative paths to
the independently hashed ``VideoNystromMMD`` reference packs.  Large tensors
remain in the pack files and only the active slot is resident in the loss.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Dict, Optional

import torch

from losses.video_nystrom_mmd import (
    VideoNystromMMD,
    canonicalize_metadata,
    load_reference_pack,
)
from video_fd.encoders.vjepa2_dense_metadata import (
    EXPLICIT_TOKEN_FEATURE_METADATA_VERSION,
    token_indices_sha256,
    validate_explicit_dense_token_indices,
)


REFERENCE_BANK_FORMAT_VERSION = 1
REFERENCE_BANK_KIND = "vjepa2_dense_video_nystrom_reference_bank"
DEFAULT_REFERENCE_BANK_SLOTS = 100
MIN_STUDENT_REFERENCE_GAP = 1.0e-8

_MANIFEST_KEYS = {
    "format_version",
    "kind",
    "slot_count",
    "metadata",
    "slots",
    "content_hash",
}
_SLOT_KEYS = {
    "slot",
    "pack_path",
    "pack_content_hash",
    "token_indices",
    "token_indices_sha256",
    "video_real_floor",
    "video_student_start_score",
    "video_student_start_std",
}
_SPLIT_METADATA_KEYS = {
    "teacher_count",
    "train_indices",
    "train_indices_sha256",
    "val_indices",
    "val_indices_sha256",
}


def _canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reference-bank manifest must be finite JSON data") from exc
    return encoded.encode("utf-8")


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{name} must be a lowercase 64-character SHA-256 hex string")
    return value


def _canonical_indices(values: Any, name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list of integers")
    if any(type(value) is not int for value in values):
        raise ValueError(f"{name} must contain only integers")
    if values[0] < 0:
        raise ValueError(f"{name} must contain non-negative indices")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return list(values)


def _indices_hash(values: Sequence[int]) -> str:
    return token_indices_sha256(torch.tensor(values, dtype=torch.int64))


def _validate_split_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    missing = _SPLIT_METADATA_KEYS - set(metadata)
    if missing:
        raise ValueError(
            "reference-bank metadata is missing split keys: "
            f"{sorted(missing)}")
    train_indices = _canonical_indices(
        metadata["train_indices"], "metadata.train_indices")
    val_indices = _canonical_indices(
        metadata["val_indices"], "metadata.val_indices")
    teacher_count = metadata["teacher_count"]
    if type(teacher_count) is not int or teacher_count <= 0:
        raise ValueError("metadata.teacher_count must be a positive integer")
    if set(train_indices).intersection(val_indices):
        raise ValueError(
            "metadata train_indices and val_indices must be disjoint")
    combined = sorted(train_indices + val_indices)
    if combined != list(range(teacher_count)):
        raise ValueError(
            "metadata train_indices and val_indices must partition every "
            "dataset index in [0, teacher_count)")
    for name, values in (
        ("train_indices", train_indices),
        ("val_indices", val_indices),
    ):
        saved_hash = _validate_sha256(
            metadata[f"{name}_sha256"],
            f"metadata.{name}_sha256",
        )
        calculated_hash = _indices_hash(values)
        if not hmac.compare_digest(saved_hash, calculated_hash):
            raise ValueError(
                f"metadata.{name}_sha256 mismatch: "
                f"saved={saved_hash}, calculated={calculated_hash}")
    return canonicalize_metadata(metadata)


def _validate_relative_pack_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = PurePath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or value == "."
        or "\\" in value
    ):
        raise ValueError(
            f"{name} must be relative to the manifest without '..'")
    if "." in path.parts or value.endswith(("/", os.sep)):
        raise ValueError(f"{name} is not a canonical file path")
    canonical = os.path.normpath(value)
    if canonical != value:
        raise ValueError(f"{name} is not normalized: {value!r}")
    return value


def _validate_slot_record(
    record: Any,
    *,
    expected_slot: int,
) -> Dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"slots[{expected_slot}] must be a mapping")
    keys = set(record)
    if keys != _SLOT_KEYS:
        raise ValueError(
            f"slots[{expected_slot}] keys mismatch: "
            f"missing={sorted(_SLOT_KEYS - keys)}, "
            f"unexpected={sorted(keys - _SLOT_KEYS)}")
    if type(record["slot"]) is not int or record["slot"] != expected_slot:
        raise ValueError(
            f"slots[{expected_slot}].slot must equal {expected_slot}")
    token_values = _canonical_indices(
        record["token_indices"],
        f"slots[{expected_slot}].token_indices",
    )
    saved_token_hash = _validate_sha256(
        record["token_indices_sha256"],
        f"slots[{expected_slot}].token_indices_sha256",
    )
    calculated_token_hash = _indices_hash(token_values)
    if not hmac.compare_digest(saved_token_hash, calculated_token_hash):
        raise ValueError(
            f"slots[{expected_slot}] token_indices_sha256 mismatch: "
            f"saved={saved_token_hash}, calculated={calculated_token_hash}")
    pack_path = _validate_relative_pack_path(
        record["pack_path"], f"slots[{expected_slot}].pack_path")
    pack_hash = _validate_sha256(
        record["pack_content_hash"],
        f"slots[{expected_slot}].pack_content_hash",
    )
    scalars: Dict[str, float] = {}
    for name in (
        "video_real_floor",
        "video_student_start_score",
        "video_student_start_std",
    ):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"slots[{expected_slot}].{name} must be a finite scalar")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(
                f"slots[{expected_slot}].{name} must be finite")
        if name == "video_student_start_std" and value < 0:
            raise ValueError(
                f"slots[{expected_slot}].{name} must be non-negative")
        scalars[name] = value
    if (
        scalars["video_student_start_score"]
        - scalars["video_real_floor"]
        <= MIN_STUDENT_REFERENCE_GAP
    ):
        raise ValueError(
            f"slots[{expected_slot}] student start score must exceed "
            f"the real floor by more than {MIN_STUDENT_REFERENCE_GAP}")
    return {
        "slot": expected_slot,
        "pack_path": pack_path,
        "pack_content_hash": pack_hash,
        "token_indices": token_values,
        "token_indices_sha256": saved_token_hash,
        **scalars,
    }


def compute_reference_bank_hash(manifest: Mapping[str, Any]) -> str:
    """Hash the canonical manifest while excluding ``content_hash`` itself."""
    if not isinstance(manifest, Mapping):
        raise TypeError("reference-bank manifest must be a mapping")
    missing = (_MANIFEST_KEYS - {"content_hash"}) - set(manifest)
    if missing:
        raise ValueError(
            f"reference-bank manifest is missing keys: {sorted(missing)}")
    payload = {
        key: manifest[key]
        for key in _MANIFEST_KEYS
        if key != "content_hash"
    }
    hasher = hashlib.sha256()
    hasher.update(b"vjepa2-video-nystrom-reference-bank\0")
    hasher.update(_canonical_json(payload))
    return hasher.hexdigest()


def validate_reference_bank_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_slot_count: Optional[int] = DEFAULT_REFERENCE_BANK_SLOTS,
) -> Dict[str, Any]:
    """Validate exact schema, split, masks, slot order, and manifest SHA."""
    if not isinstance(manifest, Mapping):
        raise TypeError("reference-bank manifest must be a mapping")
    keys = set(manifest)
    if keys != _MANIFEST_KEYS:
        raise ValueError(
            "reference-bank manifest keys mismatch: "
            f"missing={sorted(_MANIFEST_KEYS - keys)}, "
            f"unexpected={sorted(keys - _MANIFEST_KEYS)}")
    if type(manifest["format_version"]) is not int:
        raise ValueError("format_version must be an integer")
    if manifest["format_version"] != REFERENCE_BANK_FORMAT_VERSION:
        raise ValueError(
            "unsupported reference-bank format_version "
            f"{manifest['format_version']!r}")
    if manifest["kind"] != REFERENCE_BANK_KIND:
        raise ValueError(
            f"reference-bank kind must be {REFERENCE_BANK_KIND!r}")
    if type(manifest["slot_count"]) is not int:
        raise ValueError("slot_count must be an integer")
    slot_count = manifest["slot_count"]
    if slot_count <= 0:
        raise ValueError("slot_count must be positive")
    if (
        expected_slot_count is not None
        and slot_count != int(expected_slot_count)
    ):
        raise ValueError(
            f"reference-bank slot_count is {slot_count}, "
            f"expected {int(expected_slot_count)}")
    if not isinstance(manifest["metadata"], Mapping):
        raise TypeError("reference-bank metadata must be a mapping")
    metadata = _validate_split_metadata(manifest["metadata"])
    if metadata != manifest["metadata"]:
        raise ValueError("reference-bank metadata is not canonical")
    if (
        not isinstance(manifest["slots"], list)
        or len(manifest["slots"]) != slot_count
    ):
        raise ValueError(
            f"slots must contain exactly slot_count={slot_count} records")
    slots = [
        _validate_slot_record(record, expected_slot=slot)
        for slot, record in enumerate(manifest["slots"])
    ]
    token_counts = {len(record["token_indices"]) for record in slots}
    if len(token_counts) != 1:
        raise ValueError("all reference-bank slots must use one token count")
    token_hashes = [record["token_indices_sha256"] for record in slots]
    if len(set(token_hashes)) != slot_count:
        raise ValueError(
            "every reference-bank slot must use a distinct token mask")
    pack_paths = [record["pack_path"] for record in slots]
    if len(set(pack_paths)) != slot_count:
        raise ValueError(
            "every reference-bank slot must use a distinct pack_path")

    saved_hash = _validate_sha256(
        manifest["content_hash"], "content_hash")
    normalized = {
        "format_version": REFERENCE_BANK_FORMAT_VERSION,
        "kind": REFERENCE_BANK_KIND,
        "slot_count": slot_count,
        "metadata": metadata,
        "slots": slots,
        "content_hash": saved_hash,
    }
    calculated_hash = compute_reference_bank_hash(normalized)
    if not hmac.compare_digest(saved_hash, calculated_hash):
        raise ValueError(
            "reference-bank content_hash mismatch: "
            f"saved={saved_hash}, calculated={calculated_hash}")
    return normalized


def build_reference_bank_manifest(
    *,
    metadata: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    expected_slot_count: Optional[int] = DEFAULT_REFERENCE_BANK_SLOTS,
) -> Dict[str, Any]:
    """Build and hash a strict JSON-portable reference-bank manifest."""
    slot_records = [
        _validate_slot_record(record, expected_slot=slot)
        for slot, record in enumerate(slots)
    ]
    manifest: Dict[str, Any] = {
        "format_version": REFERENCE_BANK_FORMAT_VERSION,
        "kind": REFERENCE_BANK_KIND,
        "slot_count": len(slot_records),
        "metadata": canonicalize_metadata(metadata),
        "slots": slot_records,
    }
    manifest["content_hash"] = compute_reference_bank_hash(manifest)
    return validate_reference_bank_manifest(
        manifest, expected_slot_count=expected_slot_count)


def save_reference_bank_manifest(
    manifest: Mapping[str, Any],
    path: str | os.PathLike,
    *,
    expected_slot_count: Optional[int] = DEFAULT_REFERENCE_BANK_SLOTS,
    atomic: bool = True,
) -> None:
    """Save a validated manifest as canonical JSON."""
    validated = validate_reference_bank_manifest(
        manifest, expected_slot_count=expected_slot_count)
    destination = Path(path).expanduser().absolute()
    encoded = _canonical_json(validated) + b"\n"
    if not atomic:
        destination.write_bytes(encoded)
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_reference_bank_manifest(
    path: str | os.PathLike,
    *,
    expected_slot_count: Optional[int] = DEFAULT_REFERENCE_BANK_SLOTS,
) -> Dict[str, Any]:
    """Load canonical JSON and validate its schema and content hash."""
    source = Path(path).expanduser().absolute()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"failed to read reference-bank manifest {source}") from exc
    validated = validate_reference_bank_manifest(
        manifest, expected_slot_count=expected_slot_count)
    return validated


@dataclass(frozen=True)
class ActivatedReferenceSlot:
    """The objective and immutable identity of one activated bank slot."""

    slot: int
    record: Dict[str, Any]
    token_indices: torch.Tensor
    token_indices_sha256: str
    pack_path: Path
    pack_content_hash: str
    video_real_floor: float
    video_student_start_score: float
    video_student_start_std: float
    objective: VideoNystromMMD


class VideoNystromReferenceBank:
    """Lazy reference-pack loader with in-place objective switching."""

    def __init__(
        self,
        manifest_path: str | os.PathLike,
        *,
        expected_slot_count: Optional[int] = DEFAULT_REFERENCE_BANK_SLOTS,
        expected_pack_metadata: Optional[Mapping[str, Any]] = None,
        map_location: str | torch.device = "cpu",
        kernel_chunk_size: int = 256,
    ):
        if int(kernel_chunk_size) <= 0:
            raise ValueError("kernel_chunk_size must be positive")
        self.manifest_path = Path(
            manifest_path).expanduser().absolute()
        self._manifest = load_reference_bank_manifest(
            self.manifest_path,
            expected_slot_count=expected_slot_count,
        )
        self._manifest_hash = self._manifest["content_hash"]
        self._metadata = copy.deepcopy(self._manifest["metadata"])
        self._train_indices = tuple(self._metadata["train_indices"])
        self._val_indices = tuple(self._metadata["val_indices"])
        self._expected_pack_metadata = (
            None
            if expected_pack_metadata is None
            else canonicalize_metadata(expected_pack_metadata)
        )
        self.map_location = map_location
        self.kernel_chunk_size = int(kernel_chunk_size)
        self._objective: Optional[VideoNystromMMD] = None
        self._current_slot: Optional[int] = None
        for slot in range(self.slot_count):
            path = self.pack_path(slot)
            if not path.is_file():
                raise FileNotFoundError(
                    f"reference-bank pack does not exist: {path}")

    @property
    def slot_count(self) -> int:
        return int(self._manifest["slot_count"])

    @property
    def manifest(self) -> Dict[str, Any]:
        return copy.deepcopy(self._manifest)

    @property
    def metadata(self) -> Dict[str, Any]:
        return copy.deepcopy(self._metadata)

    @property
    def manifest_hash(self) -> str:
        return self._manifest_hash

    @property
    def train_indices(self) -> tuple[int, ...]:
        return self._train_indices

    @property
    def val_indices(self) -> tuple[int, ...]:
        return self._val_indices

    @property
    def expected_pack_metadata(self) -> Optional[Dict[str, Any]]:
        return (
            None
            if self._expected_pack_metadata is None
            else copy.deepcopy(self._expected_pack_metadata)
        )

    @property
    def current_slot(self) -> Optional[int]:
        return self._current_slot

    @property
    def objective(self) -> Optional[VideoNystromMMD]:
        return self._objective

    def slot_record(self, slot: int) -> Dict[str, Any]:
        slot = self._validate_slot_number(slot)
        return copy.deepcopy(self._manifest["slots"][slot])

    def token_indices(
        self,
        slot: int,
        *,
        device: torch.device | str | int | None = None,
    ) -> torch.Tensor:
        record = self._manifest["slots"][self._validate_slot_number(slot)]
        return torch.tensor(
            record["token_indices"],
            dtype=torch.long,
            device=device,
        )

    def token_indices_hash(self, slot: int) -> str:
        return str(self._manifest["slots"][
            self._validate_slot_number(slot)
        ]["token_indices_sha256"])

    def pack_hash(self, slot: int) -> str:
        return str(self._manifest["slots"][
            self._validate_slot_number(slot)
        ]["pack_content_hash"])

    def video_real_floor(self, slot: int) -> float:
        return float(self._manifest["slots"][
            self._validate_slot_number(slot)
        ]["video_real_floor"])

    def video_student_start_score(self, slot: int) -> float:
        return float(self._manifest["slots"][
            self._validate_slot_number(slot)
        ]["video_student_start_score"])

    def video_student_start_std(self, slot: int) -> float:
        return float(self._manifest["slots"][
            self._validate_slot_number(slot)
        ]["video_student_start_std"])

    @property
    def current_token_indices(self) -> torch.Tensor:
        return self.token_indices(self._require_current_slot())

    @property
    def current_token_indices_hash(self) -> str:
        return self.token_indices_hash(self._require_current_slot())

    @property
    def current_pack_hash(self) -> str:
        return self.pack_hash(self._require_current_slot())

    @property
    def current_video_real_floor(self) -> float:
        return self.video_real_floor(self._require_current_slot())

    @property
    def current_video_student_start_score(self) -> float:
        return self.video_student_start_score(
            self._require_current_slot())

    @property
    def current_video_student_start_std(self) -> float:
        return self.video_student_start_std(
            self._require_current_slot())

    def pack_path(self, slot: int) -> Path:
        record = self._manifest["slots"][self._validate_slot_number(slot)]
        candidate = (
            self.manifest_path.parent / record["pack_path"]).resolve()
        try:
            candidate.relative_to(self.manifest_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                "reference-bank pack resolves outside the manifest directory: "
                f"{record['pack_path']!r}") from exc
        return candidate

    def load_slot_pack(
        self,
        slot: int,
        *,
        map_location: str | torch.device | None = None,
    ) -> Dict[str, Any]:
        """Load and validate a slot pack against its manifest record."""
        slot = self._validate_slot_number(slot)
        record = self._manifest["slots"][slot]
        pack = load_reference_pack(
            self.pack_path(slot),
            expected_metadata=self._expected_pack_metadata,
            map_location=(
                self.map_location if map_location is None else map_location),
        )
        if not hmac.compare_digest(
            pack["content_hash"], record["pack_content_hash"]
        ):
            raise ValueError(
                f"slot {slot} pack hash mismatch: "
                f"manifest={record['pack_content_hash']}, "
                f"pack={pack['content_hash']}")
        feature = pack["metadata"].get("feature")
        if not isinstance(feature, Mapping):
            raise ValueError(f"slot {slot} pack metadata.feature is missing")
        required_feature = {
            "feature_metadata_version":
                EXPLICIT_TOKEN_FEATURE_METADATA_VERSION,
            "feature_type": "vjepa2_dense_latent",
            "num_tokens": len(record["token_indices"]),
            "token_sampling": "explicit_sorted_unique",
            "token_alignment": "same_explicit_indices_and_order",
            "token_indices": record["token_indices"],
            "token_indices_sha256": record["token_indices_sha256"],
            "sample_axis": "video",
            "kernel_metric": "frobenius_rbf_aligned_tokens",
        }
        for key, expected in required_feature.items():
            if feature.get(key) != expected:
                raise ValueError(
                    f"slot {slot} pack feature metadata mismatch for {key}: "
                    f"saved={feature.get(key)!r}, expected={expected!r}")
        raw_tokens = feature.get("raw_tokens")
        if type(raw_tokens) is not int:
            raise ValueError(
                f"slot {slot} pack feature raw_tokens must be an integer")
        validate_explicit_dense_token_indices(
            record["token_indices"],
            raw_tokens=raw_tokens,
            num_tokens=len(record["token_indices"]),
        )
        sample_shape = pack["metadata"]["nystrom"]["sample_shape"]
        if (
            feature.get("matrix_shape") != sample_shape
            or sample_shape[0] != len(record["token_indices"])
        ):
            raise ValueError(
                f"slot {slot} pack feature/Nyström matrix shape mismatch")
        return pack

    @torch.no_grad()
    def activate_slot_(
        self,
        slot: int,
        objective: Optional[VideoNystromMMD] = None,
    ) -> ActivatedReferenceSlot:
        """Activate ``slot``, reusing objective and buffer storage if possible."""
        slot = self._validate_slot_number(slot)
        pack = self.load_slot_pack(slot)
        target = objective if objective is not None else self._objective
        if target is None:
            target = VideoNystromMMD(
                pack,
                expected_metadata=self._expected_pack_metadata,
                kernel_chunk_size=self.kernel_chunk_size,
            )
        else:
            if not isinstance(target, VideoNystromMMD):
                raise TypeError("objective must be a VideoNystromMMD")
            tensors = {
                "landmarks": pack["landmarks"].detach().float(),
                "alpha": pack["alpha"].detach().float(),
                "k_rr": pack["k_rr"].detach().float().reshape(()),
                "sigma": pack["sigma"].detach().float().reshape(()),
            }
            for name, source in tensors.items():
                destination = getattr(target, name)
                if destination.shape != source.shape:
                    raise ValueError(
                        f"slot {slot} cannot switch {name} in place: "
                        f"{tuple(source.shape)} != "
                        f"{tuple(destination.shape)}")
            # Complete every potentially allocating conversion before the
            # first mutation.  The following same-shape copy_ calls require no
            # allocation, so an OOM cannot leave a partially switched loss.
            prepared = {}
            for name, source in tensors.items():
                destination = getattr(target, name)
                prepared[name] = source.to(
                    device=destination.device,
                    dtype=destination.dtype,
                ).contiguous()
                if not bool(torch.isfinite(prepared[name]).all().item()):
                    raise FloatingPointError(
                        f"slot {slot} converted {name} is non-finite")
            for name, source in prepared.items():
                getattr(target, name).copy_(
                    source
                )
            target.reference_metadata = pack["metadata"]
            target.reference_hash = pack["content_hash"]
        self._objective = target
        self._current_slot = slot
        record = self._manifest["slots"][slot]
        return ActivatedReferenceSlot(
            slot=slot,
            record=copy.deepcopy(record),
            token_indices=self.token_indices(slot),
            token_indices_sha256=record["token_indices_sha256"],
            pack_path=self.pack_path(slot),
            pack_content_hash=record["pack_content_hash"],
            video_real_floor=float(record["video_real_floor"]),
            video_student_start_score=float(
                record["video_student_start_score"]),
            video_student_start_std=float(
                record["video_student_start_std"]),
            objective=target,
        )

    def validate_all_packs(self) -> list[str]:
        """Validate every referenced pack without retaining its tensors."""
        return [
            self.load_slot_pack(slot)["content_hash"]
            for slot in range(self.slot_count)
        ]

    def _validate_slot_number(self, slot: int) -> int:
        if type(slot) is not int or not 0 <= slot < self.slot_count:
            raise IndexError(
                f"slot must be an integer in [0,{self.slot_count}), "
                f"got {slot!r}")
        return slot

    def _require_current_slot(self) -> int:
        if self._current_slot is None:
            raise RuntimeError("no reference-bank slot is active")
        return self._current_slot

"""Strict mmap-backed teacher bank for V-JEPA2 + VideoMAE-v2 features."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from torch.utils.data import Dataset


BANK_KIND = "dual_vjepa2_videomae_v2_matched_teacher_bank"
BANK_SCHEMA_VERSION = 1
BRANCH_SHAPES = {
    "vjepa2": (1024, 1024),
    "videomae_v2": (1280, 1024),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict_sha256(value: object, label: str) -> str:
    normalized = str(value)
    if (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return normalized


class DualVideoFeatureBank(Dataset):
    """Metadata dataset plus random-access frozen FP16 feature rows."""

    def __init__(
        self,
        manifest_path: str | os.PathLike,
        *,
        expected_manifest_sha256: str,
        expected_content_hash: str,
        expected_source_manifest_sha256: str,
        expected_sample_count: int = 5000,
        expected_videomae_revision: str | None = None,
        expected_videomae_download_manifest_content_hash: str | None = None,
        expected_vjepa_model_sha256: str | None = None,
        validate_all_shards: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        actual_manifest_sha = file_sha256(self.manifest_path)
        if actual_manifest_sha != _strict_sha256(
            expected_manifest_sha256, "expected_manifest_sha256"
        ):
            raise ValueError("dual teacher bank manifest SHA-256 mismatch")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise TypeError("dual teacher bank manifest must be an object")
        required = {
            "schema_version",
            "kind",
            "sample_count",
            "source_manifest",
            "branches",
            "samples",
            "shards",
            "content_hash",
        }
        if set(manifest) != required:
            raise ValueError(
                "dual teacher bank manifest keys mismatch: "
                f"{sorted(manifest)} != {sorted(required)}")
        if (
            manifest["schema_version"] != BANK_SCHEMA_VERSION
            or manifest["kind"] != BANK_KIND
            or int(manifest["sample_count"]) != int(expected_sample_count)
        ):
            raise ValueError("dual teacher bank kind/schema/sample count mismatch")
        payload = dict(manifest)
        content_hash = payload.pop("content_hash")
        if canonical_sha256(payload) != content_hash:
            raise ValueError("dual teacher bank content_hash is invalid")
        if content_hash != _strict_sha256(
            expected_content_hash, "expected_content_hash"
        ):
            raise ValueError("dual teacher bank content identity mismatch")
        source = manifest["source_manifest"]
        if source.get("sha256") != _strict_sha256(
            expected_source_manifest_sha256,
            "expected_source_manifest_sha256",
        ):
            raise ValueError("dual bank source manifest SHA-256 mismatch")
        branches = manifest["branches"]
        if set(branches) != set(BRANCH_SHAPES):
            raise ValueError("dual teacher bank branch set mismatch")
        for name, expected_shape in BRANCH_SHAPES.items():
            branch = branches[name]
            if (
                tuple(branch.get("sample_shape", ())) != expected_shape
                or branch.get("dtype") != "float16"
                or not isinstance(branch.get("feature_contract"), Mapping)
                or not isinstance(branch.get("model_identity"), Mapping)
            ):
                raise ValueError(f"dual teacher branch {name!r} contract mismatch")
        if expected_videomae_revision is not None and (
            branches["videomae_v2"]["model_identity"].get("revision")
            != str(expected_videomae_revision)
        ):
            raise ValueError("VideoMAE-v2 bank model revision mismatch")
        if expected_videomae_download_manifest_content_hash is not None and (
            branches["videomae_v2"]["model_identity"].get(
                "download_manifest_content_hash"
            )
            != _strict_sha256(
                expected_videomae_download_manifest_content_hash,
                "expected_videomae_download_manifest_content_hash",
            )
        ):
            raise ValueError(
                "VideoMAE-v2 bank download manifest identity mismatch")
        if expected_vjepa_model_sha256 is not None and (
            branches["vjepa2"]["model_identity"].get("sha256")
            != _strict_sha256(
                expected_vjepa_model_sha256,
                "expected_vjepa_model_sha256",
            )
        ):
            raise ValueError("V-JEPA2 bank model SHA-256 mismatch")

        samples = manifest["samples"]
        if len(samples) != int(expected_sample_count):
            raise ValueError("dual bank sample table length mismatch")
        self.samples = []
        self._sample_index = {}
        for expected_index, item in enumerate(samples):
            if (
                not isinstance(item, dict)
                or set(item) != {"index", "sample_id", "prompt"}
                or int(item["index"]) != expected_index
                or not isinstance(item["sample_id"], str)
                or not item["sample_id"]
                or not isinstance(item["prompt"], str)
            ):
                raise ValueError(
                    f"invalid dual bank sample record at {expected_index}")
            if item["sample_id"] in self._sample_index:
                raise ValueError("dual bank sample IDs must be unique")
            self._sample_index[item["sample_id"]] = expected_index
            self.samples.append(dict(item))

        self.root = self.manifest_path.parent
        self.shards = []
        self._row_to_shard: list[tuple[int, int]] = [None] * len(samples)
        expected_start = 0
        for shard_index, item in enumerate(manifest["shards"]):
            required_shard = {
                "index", "path", "sample_start", "sample_end",
                "sample_ids_sha256", "sha256", "content_hash",
            }
            if not isinstance(item, dict) or set(item) != required_shard:
                raise ValueError(f"invalid shard record {shard_index}")
            start = int(item["sample_start"])
            end = int(item["sample_end"])
            if (
                int(item["index"]) != shard_index
                or start != expected_start
                or end <= start
                or end > len(samples)
            ):
                raise ValueError("dual bank shard ranges are not contiguous")
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("dual bank shard path must stay under bank root")
            path = (self.root / relative).resolve()
            if self.root not in path.parents or not path.is_file():
                raise FileNotFoundError(path)
            ids = [samples[row]["sample_id"] for row in range(start, end)]
            if canonical_sha256(ids) != item["sample_ids_sha256"]:
                raise ValueError("dual bank shard sample ID hash mismatch")
            if validate_all_shards and file_sha256(path) != item["sha256"]:
                raise ValueError(f"dual bank shard SHA-256 mismatch: {path}")
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("content_hash") != item["content_hash"]:
                    raise ValueError("dual bank shard embedded content hash mismatch")
                for name, shape in BRANCH_SHAPES.items():
                    tensor_shape = tuple(handle.get_slice(name).get_shape())
                    if tensor_shape != (end - start, *shape):
                        raise ValueError(
                            f"dual bank shard tensor {name!r} shape mismatch")
            for row in range(start, end):
                self._row_to_shard[row] = (shard_index, row - start)
            self.shards.append({**item, "resolved_path": str(path)})
            expected_start = end
        if expected_start != len(samples):
            raise ValueError("dual bank shards do not cover all samples")
        self.manifest = manifest
        self.content_hash = content_hash
        self.manifest_sha256 = actual_manifest_sha

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.samples[int(index)]
        return {
            "prompts": item["prompt"],
            "sample_ids": item["sample_id"],
            "teacher_bank_rows": int(item["index"]),
        }

    @staticmethod
    def collate_fn(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "prompts": [str(item["prompts"]) for item in items],
            "sample_ids": [str(item["sample_ids"]) for item in items],
            "teacher_bank_rows": torch.tensor([
                int(item["teacher_bank_rows"]) for item in items
            ], dtype=torch.long),
        }

    def load_rows(
        self,
        sample_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Read only requested rows, preserving order and duplicate IDs."""
        ids = tuple(str(value) for value in sample_ids)
        if not ids or any(value not in self._sample_index for value in ids):
            raise KeyError("dual bank request contains an unknown sample ID")
        output = {name: [None] * len(ids) for name in BRANCH_SHAPES}
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for output_index, sample_id in enumerate(ids):
            row = self._sample_index[sample_id]
            shard_index, offset = self._row_to_shard[row]
            by_shard.setdefault(shard_index, []).append((output_index, offset))
        for shard_index, requests in by_shard.items():
            path = self.shards[shard_index]["resolved_path"]
            with safe_open(path, framework="pt", device="cpu") as handle:
                slices = {name: handle.get_slice(name) for name in BRANCH_SHAPES}
                for output_index, offset in requests:
                    for name in BRANCH_SHAPES:
                        output[name][output_index] = slices[name][
                            offset:offset + 1
                        ].squeeze(0)
        return {
            name: torch.stack(rows, dim=0)
            for name, rows in output.items()
        }


__all__ = [
    "BANK_KIND",
    "BANK_SCHEMA_VERSION",
    "BRANCH_SHAPES",
    "DualVideoFeatureBank",
    "canonical_sha256",
    "file_sha256",
]

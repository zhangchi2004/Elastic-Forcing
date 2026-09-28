"""Mmap-backed teacher bank for DINOv3 + V-JEPA2 + VideoMAE-v2 features."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from torch.utils.data import Dataset


BANK_KIND = "triple_dinov3_vjepa2_videomae_v2_matched_teacher_bank"
BANK_SCHEMA_VERSION = 1
BRANCH_SHAPES = {
    "dinov3": (1024, 1024),
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
    return normalized


class TripleVideoFeatureBank(Dataset):
    """Metadata dataset plus random-access frozen FP16 feature rows."""

    def __init__(
        self,
        manifest_path: str | os.PathLike,
        *,
        expected_manifest_sha256: str,
        expected_content_hash: str,
        expected_source_manifest_sha256: str,
        expected_source_csv_sha256: str | None = None,
        expected_sample_identity_sha256: str | None = None,
        expected_sample_count: int = 5000,
        expected_videomae_revision: str | None = None,
        expected_videomae_download_manifest_content_hash: str | None = None,
        expected_vjepa_model_sha256: str | None = None,
        expected_dinov3_checkpoint_sha256: str | None = None,
        expected_dinov3_source_commit: str | None = None,
        expected_shard_size: int = 32,
        validate_all_shards: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        actual_manifest_sha = file_sha256(self.manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        content_hash = manifest["content_hash"]

        samples = manifest["samples"]
        self.samples = []
        self._sample_index = {}
        for expected_index, item in enumerate(samples):
            self._sample_index[item["sample_id"]] = expected_index
            self.samples.append(dict(item))

        self.root = self.manifest_path.parent
        self.shards = []
        self._row_to_shard: list[tuple[int, int]] = [None] * len(samples)
        for shard_index, item in enumerate(manifest["shards"]):
            start = int(item["sample_start"])
            end = int(item["sample_end"])
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("triple bank shard path must stay under bank root")
            path = (self.root / relative).resolve()
            if self.root not in path.parents or not path.is_file():
                raise FileNotFoundError(path)
            for row in range(start, end):
                self._row_to_shard[row] = (shard_index, row - start)
            self.shards.append({**item, "resolved_path": str(path)})
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
    "TripleVideoFeatureBank",
    "canonical_sha256",
    "file_sha256",
]

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from utils.triple_video_feature_bank import (
    BANK_KIND,
    BANK_SCHEMA_VERSION,
    TripleVideoFeatureBank,
    canonical_sha256,
    file_sha256,
)
from trainer.distillation import Trainer


VJEPA_SHA = "1" * 64
VIDEOMAE_REVISION = "9981a9c8f77118c421e5228e1b219468a4b0238d"
VIDEOMAE_DOWNLOAD_HASH = "2" * 64
SOURCE_SHA = "3" * 64
DINOV3_SHA = "4" * 64
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def _write_bank(root: Path):
    samples = [
        {"index": index, "sample_id": f"sample-{index}", "prompt": f"p{index}"}
        for index in range(3)
    ]
    branch_metadata = {
        "dinov3": {
            "sample_shape": [1024, 1024],
            "dtype": "float16",
            "feature_contract": {"name": "dinov3-test"},
            "model_identity": {
                "checkpoint_sha256": DINOV3_SHA,
                "source_commit": DINOV3_COMMIT,
            },
        },
        "vjepa2": {
            "sample_shape": [1024, 1024],
            "dtype": "float16",
            "feature_contract": {"name": "vjepa2-test"},
            "model_identity": {"sha256": VJEPA_SHA},
        },
        "videomae_v2": {
            "sample_shape": [1280, 1024],
            "dtype": "float16",
            "feature_contract": {"name": "videomae-test"},
            "model_identity": {
                "revision": VIDEOMAE_REVISION,
                "download_manifest_content_hash": VIDEOMAE_DOWNLOAD_HASH,
            },
        },
    }
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)
    shards = []
    for shard_index, (start, end) in enumerate(((0, 2), (2, 3))):
        shard_content_hash = canonical_sha256({
            "index": shard_index,
            "sample_start": start,
            "sample_end": end,
        })
        path = shard_dir / f"{shard_index:05d}.safetensors"
        save_file(
            {
                "dinov3": torch.stack([
                    torch.full((1024, 1024), float(row + 20), dtype=torch.float16)
                    for row in range(start, end)
                ]),
                "vjepa2": torch.stack([
                    torch.full((1024, 1024), float(row), dtype=torch.float16)
                    for row in range(start, end)
                ]),
                "videomae_v2": torch.stack([
                    torch.full((1280, 1024), float(row + 10), dtype=torch.float16)
                    for row in range(start, end)
                ]),
            },
            path,
            metadata={"content_hash": shard_content_hash},
        )
        shards.append({
            "index": shard_index,
            "path": str(path.relative_to(root)),
            "sample_start": start,
            "sample_end": end,
            "sample_ids_sha256": canonical_sha256([
                item["sample_id"] for item in samples[start:end]
            ]),
            "sha256": file_sha256(path),
            "content_hash": shard_content_hash,
        })
    payload = {
        "schema_version": BANK_SCHEMA_VERSION,
        "kind": BANK_KIND,
        "sample_count": len(samples),
        "source_manifest": {"sha256": SOURCE_SHA},
        "branches": branch_metadata,
        "samples": samples,
        "shards": shards,
    }
    manifest = {**payload, "content_hash": canonical_sha256(payload)}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, manifest


def _open(manifest_path: Path, manifest: dict, **overrides):
    kwargs = {
        "expected_manifest_sha256": file_sha256(manifest_path),
        "expected_content_hash": manifest["content_hash"],
        "expected_source_manifest_sha256": SOURCE_SHA,
        "expected_sample_count": 3,
        "expected_videomae_revision": VIDEOMAE_REVISION,
        "expected_videomae_download_manifest_content_hash": (
            VIDEOMAE_DOWNLOAD_HASH),
        "expected_vjepa_model_sha256": VJEPA_SHA,
        "expected_dinov3_checkpoint_sha256": DINOV3_SHA,
        "expected_dinov3_source_commit": DINOV3_COMMIT,
        "expected_shard_size": 2,
        "validate_all_shards": True,
    }
    kwargs.update(overrides)
    return TripleVideoFeatureBank(manifest_path, **kwargs)


def test_random_access_preserves_order_and_duplicate_sample_ids(tmp_path):
    manifest_path, manifest = _write_bank(tmp_path)
    bank = _open(manifest_path, manifest)
    assert [bank[index]["sample_ids"] for index in (2, 0, 2)] == [
        "sample-2", "sample-0", "sample-2"]
    loaded = bank.load_rows(["sample-2", "sample-0", "sample-2"])
    assert tuple(loaded["dinov3"].shape) == (3, 1024, 1024)
    assert tuple(loaded["vjepa2"].shape) == (3, 1024, 1024)
    assert tuple(loaded["videomae_v2"].shape) == (3, 1280, 1024)
    torch.testing.assert_close(
        loaded["dinov3"][:, 0, 0], torch.tensor([22, 20, 22]).half())
    torch.testing.assert_close(
        loaded["vjepa2"][:, 0, 0], torch.tensor([2, 0, 2]).half())
    torch.testing.assert_close(
        loaded["videomae_v2"][:, 0, 0],
        torch.tensor([12, 10, 12]).half(),
    )


@pytest.mark.parametrize(
    "override, message",
    [
        ({"expected_videomae_revision": "wrong"}, "revision"),
        ({
            "expected_videomae_download_manifest_content_hash": "4" * 64,
        }, "download manifest"),
        ({"expected_vjepa_model_sha256": "5" * 64}, "V-JEPA2"),
        ({"expected_dinov3_checkpoint_sha256": "5" * 64}, "DINOv3"),
        ({"expected_dinov3_source_commit": "wrong"}, "DINOv3"),
        ({"expected_content_hash": "6" * 64}, "content identity"),
    ],
)
def test_model_and_content_identity_mismatches_fail_closed(
    tmp_path, override, message
):
    manifest_path, manifest = _write_bank(tmp_path)
    with pytest.raises(ValueError, match=message):
        _open(manifest_path, manifest, **override)


def test_changed_shard_hash_fails_before_row_loading(tmp_path):
    manifest_path, manifest = _write_bank(tmp_path)
    shard = tmp_path / manifest["shards"][0]["path"]
    with shard.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="shard SHA-256 mismatch"):
        _open(manifest_path, manifest)


def test_counter_sampler_is_rank_major_ordered_and_with_replacement():
    class Dataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"sample_ids": f"sample-{index}", "prompts": f"p{index}"}

        @staticmethod
        def collate_fn(items):
            return {
                "sample_ids": [item["sample_ids"] for item in items],
                "prompts": [item["prompts"] for item in items],
            }

    trainer = Trainer.__new__(Trainer)
    trainer.matched_counter_sampling_enabled = True
    trainer.dataset = Dataset()
    trainer.config = SimpleNamespace(seed=7, batch_size=1)
    trainer.step = 11
    trainer.data_counter = 0
    trainer.fresh_video_total_samples = 8
    trainer.fresh_video_local_samples = 2
    expected = tuple(
        int.from_bytes(hashlib.sha256(
            f"sf-matched-counter-v1\0{trainer.config.seed}\0{trainer.step}\0{position}"
            .encode("ascii")
        ).digest()[:8], "little") % len(trainer.dataset)
        for position in range(trainer.fresh_video_total_samples)
    )
    assert trainer._matched_global_indices_for_step() == expected
    # Eight with-replacement draws from a three-row bank must contain repeats.
    assert len(set(expected)) < len(expected)
    with mock.patch("trainer.distillation.dist.get_rank", return_value=2):
        first = trainer._matched_batch_for_microbatch(0)
        second = trainer._matched_batch_for_microbatch(1)
    assert first["sample_ids"] == [f"sample-{expected[4]}"]
    assert second["sample_ids"] == [f"sample-{expected[5]}"]
    assert trainer.data_counter == 2

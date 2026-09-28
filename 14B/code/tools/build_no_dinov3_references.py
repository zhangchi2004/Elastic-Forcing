#!/usr/bin/env python3
"""Build the two new Nyström references without changing either estimator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from losses.video_nystrom_mmd import (
    build_video_nystrom_reference,
    compute_reference_pack_hash,
    load_reference_pack,
)
from utils.triple_video_feature_bank import (
    TripleVideoFeatureBank,
    file_sha256,
)


OUTPUT_NAMES = {
    "vjepa2": "vjepa2_l15_last64_k1024_m1024_vbench8852.pt",
    "videomae_v2": "videomae_v2_l15_fixed5x16_k256_m256_vbench8852.pt",
}
LANDMARKS = {"dinov3": 256, "vjepa2": 1024, "videomae_v2": 256}


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_branch(bank: TripleVideoFeatureBank, branch: str) -> torch.Tensor:
    parts = []
    for shard in bank.shards:
        with safe_open(
            shard["resolved_path"], framework="pt", device="cpu"
        ) as handle:
            parts.append(handle.get_tensor(branch))
    value = torch.cat(parts, dim=0)
    expected = (len(bank), *bank.manifest["branches"][branch]["sample_shape"])
    if value.dtype != torch.float16 or tuple(value.shape) != expected:
        raise ValueError(f"bank branch {branch} geometry/dtype mismatch")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-manifest", required=True)
    parser.add_argument("--bank-manifest-sha256", required=True)
    parser.add_argument("--bank-content-hash", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--videomae-revision", default=(
        "9981a9c8f77118c421e5228e1b219468a4b0238d"))
    parser.add_argument(
        "--videomae-download-manifest-content-hash", required=True)
    parser.add_argument("--vjepa-model-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--expected-sample-count", type=int, default=5000)
    parser.add_argument("--expected-shard-size", type=int, default=32)
    parser.add_argument("--dataset-name", default="fdwan13b_vidprom5k_matched")
    parser.add_argument("--skip-bank-shard-hashes", action="store_true")
    args = parser.parse_args()
    bank = TripleVideoFeatureBank(
        args.bank_manifest,
        expected_manifest_sha256=args.bank_manifest_sha256,
        expected_content_hash=args.bank_content_hash,
        expected_source_manifest_sha256=args.source_manifest_sha256,
        expected_sample_count=args.expected_sample_count,
        expected_videomae_revision=args.videomae_revision,
        expected_videomae_download_manifest_content_hash=(
            args.videomae_download_manifest_content_hash),
        expected_vjepa_model_sha256=args.vjepa_model_sha256,
        active_branches=("vjepa2", "videomae_v2"),
        expected_shard_size=args.expected_shard_size,
        validate_all_shards=not args.skip_bank_shard_hashes,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_request = {
        "chunk_size": int(args.chunk_size),
        "kmeans_iterations": int(args.kmeans_iterations),
        "ridge": float(args.ridge),
        "seed": 0,
    }
    summaries = {}
    for branch in ("vjepa2", "videomae_v2"):
        print(json.dumps({'stage': 'reference_start', 'branch': branch, 'samples': len(bank)}), flush=True)
        output = output_dir / OUTPUT_NAMES[branch]
        if output.is_file():
            existing = load_reference_pack(output, map_location="cpu")
            reference_metadata = existing["metadata"].get("reference", {})
            source = reference_metadata.get("source", {})
            nystrom = existing["metadata"].get("nystrom", {})
            if (
                source.get("triple_teacher_bank_content_hash")
                != bank.content_hash
                or int(nystrom.get("num_landmarks", -1)) != LANDMARKS[branch]
                or int(nystrom.get("kmeans_iterations", -1))
                != int(args.kmeans_iterations)
                or float(nystrom.get("ridge", float("nan")))
                != float(args.ridge)
                or int(nystrom.get("seed", -1)) != 0
                or reference_metadata.get("build_request") != build_request
            ):
                raise ValueError(
                    f"existing reference has another identity: {output}")
            summaries[branch] = {
                "path": str(output),
                "sha256": file_sha256(output),
                "content_hash": existing["content_hash"],
                "resumed": True,
            }
            continue
        cpu_reference = load_branch(bank, branch)
        print(json.dumps({'stage': 'reference_rows_loaded', 'branch': branch}), flush=True)
        reference = cpu_reference.to(
            device=torch.device(args.device), dtype=torch.float32)
        del cpu_reference
        branch_manifest = bank.manifest["branches"][branch]
        metadata = {
            "feature": branch_manifest["feature_contract"],
            "reference": {
                "dataset": args.dataset_name,
                "build_request": build_request,
                "source": {
                    "triple_teacher_bank_content_hash": bank.content_hash,
                    "branch_name": branch,
                    "triple_teacher_bank_manifest_sha256": (
                        bank.manifest_sha256),
                    "source_manifest_sha256": (
                        args.source_manifest_sha256),
                    "selected_samples": len(bank),
                    "sample_order": "frozen_manifest_order",
                    "feature_bank_fp16_sha256": branch_manifest[
                        "feature_bank_fp16_sha256"],
                    "model_identity": branch_manifest["model_identity"],
                },
            },
        }
        pack = build_video_nystrom_reference(
            reference,
            num_landmarks=LANDMARKS[branch],
            metadata=metadata,
            kmeans_iterations=args.kmeans_iterations,
            ridge=args.ridge,
            chunk_size=args.chunk_size,
            seed=0,
        )
        print(json.dumps({'stage': 'reference_computed', 'branch': branch}), flush=True)
        del reference
        cpu_pack = {
            key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            for key, value in pack.items()
        }
        if compute_reference_pack_hash(cpu_pack) != cpu_pack["content_hash"]:
            raise RuntimeError("reference hash changed during CPU transfer")
        atomic_torch_save(cpu_pack, output)
        validated = load_reference_pack(output, map_location="cpu")
        summaries[branch] = {
            "path": str(output),
            "sha256": file_sha256(output),
            "content_hash": validated["content_hash"],
            "resumed": False,
        }
        del pack, cpu_pack, validated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest = {
        "schema_version": 1,
        "bank_manifest": str(Path(args.bank_manifest).resolve()),
        "bank_manifest_sha256": args.bank_manifest_sha256,
        "bank_content_hash": bank.content_hash,
        "build_request": build_request,
        "references": summaries,
    }
    path = output_dir / "no_dinov3_reference_manifest.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

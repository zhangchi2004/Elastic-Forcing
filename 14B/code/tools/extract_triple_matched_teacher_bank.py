#!/usr/bin/env python3
"""Decode each frozen 81-frame MP4 once and write a resumable triple bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from utils.triple_video_feature_bank import (
    BANK_KIND,
    BANK_SCHEMA_VERSION,
    canonical_sha256,
    file_sha256,
)
from utils.matched_video_manifest import (
    load_matched_video_manifest,
    sha256_file,
)
from video_fd.encoders.dinov3_temporal_pool_adapter import (
    DINOv3TemporalPoolAdapter,
)
from video_fd.encoders.segmented_videomae_v2 import (
    SegmentedVideoMAEv2Ensemble,
)
from video_fd.encoders.videomae_v2_dense_metadata import (
    canonical_videomae_v2_dense_feature_metadata,
    uniform_segment_token_indices,
)
from video_fd.encoders.vjepa2_adapter import VJEPA2Adapter
from video_fd.encoders.vjepa2_dense_metadata import (
    canonical_vjepa2_dense_feature_metadata,
    uniform_dense_token_indices,
)


VIDEOMAE_REVISION = "9981a9c8f77118c421e5228e1b219468a4b0238d"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def decode_81(path: Path, *, size: int, mtime_ns: int) -> torch.Tensor:
    stat = path.stat()
    if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
        raise ValueError(f"frozen video stat changed: {path}")
    import av

    frames = []
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"expected one video stream: {path}")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            array = frame.to_ndarray(format="rgb24")
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"invalid RGB frame in {path}")
            frames.append(torch.from_numpy(
                np.ascontiguousarray(array)).permute(2, 0, 1))
    if len(frames) != 81:
        raise ValueError(f"{path.name} has {len(frames)} frames, expected 81")
    shape = tuple(frames[0].shape)
    if any(tuple(frame.shape) != shape for frame in frames):
        raise ValueError(f"resolution changes within {path}")
    return torch.stack(frames).float().div_(255.0).unsqueeze(0)


def load_download_manifest(model_dir: Path) -> dict:
    path = model_dir / "download_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    payload = dict(manifest)
    saved = payload.pop("content_hash")
    if (
        manifest.get("repo_id") != "OpenGVLab/VideoMAEv2-Large"
        or manifest.get("revision") != VIDEOMAE_REVISION
        or canonical_sha256(payload) != saved
    ):
        raise ValueError("invalid pinned VideoMAE download manifest")
    for item in manifest["files"]:
        path = (model_dir / item["path"]).resolve()
        if (
            model_dir not in path.parents
            or path.stat().st_size != int(item["size_bytes"])
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"VideoMAE file identity mismatch: {path}")
    return manifest


def build_encoders(args, device):
    dino = DINOv3TemporalPoolAdapter({
        "repo_path": str(args.dinov3_repo),
        "weight_path": str(args.dinov3_weights),
        "source_commit": str(args.dinov3_source_commit),
        "checkpoint_sha256": str(args.dinov3_checkpoint_sha256),
        "source_num_frames": 81,
        "frame_indices": (
            0, 4, 8, 13, 17, 21, 25, 29, 34, 38,
            42, 46, 51, 55, 59, 63, 67, 72, 76, 80),
        "input_size": 256,
        "patch_size": 16,
        "spatial_axis_indices": tuple(range(16)),
        "forward_chunk_frames": 4,
        "geometry": "temporal_mean5_full256",
        "temporal_groups": 4,
        "temporal_pool_frames": 5,
    }).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    vj = VJEPA2Adapter({
        "name": "vjepa2_dense",
        "model_id": str(args.vjepa_model),
        "input_frames": 64,
        "dense_num_frames": 64,
        "input_size": 256,
        "resize_short_side": 292,
        "crop_mode": "center",
        "dense_only": True,
        "dense_layer_index": 15,
    }).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    vm = SegmentedVideoMAEv2Ensemble({
        "name": "videomae_v2_five_segment_dense",
        "model_id": str(args.videomae_model),
        "revision": VIDEOMAE_REVISION,
        "local_files_only": True,
        "encoder_count": 5,
        "skip_leading_frames": 1,
        "segment_num_frames": 16,
        "source_num_frames": 81,
        "input_frames": 16,
        "dense_num_frames": 16,
        "input_size": 224,
        "resize_short_side": 224,
        "crop_mode": "center",
        "dense_only": True,
        "dense_layer_index": 15,
    }).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    if (dino.dense_feature_dim, vj.dense_feature_dim, vm.dense_feature_dim) != (1024, 1024, 1024):
        raise RuntimeError("encoder feature dimensions differ from contract")
    return dino, vj, vm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-csv-sha256", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--dinov3-repo", required=True)
    parser.add_argument("--dinov3-weights", required=True)
    parser.add_argument("--dinov3-source-commit", required=True)
    parser.add_argument("--dinov3-checkpoint-sha256", required=True)
    parser.add_argument("--vjepa-model", required=True)
    parser.add_argument("--vjepa-reference-model-id", required=True)
    parser.add_argument("--vjepa-model-sha256", required=True)
    parser.add_argument("--videomae-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-rows", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-count", type=int, default=8000)
    args = parser.parse_args()
    if args.shard_rows != 32:
        raise ValueError("triple teacher bank requires shard-rows=32")
    manifest_path = Path(args.manifest).expanduser().resolve()
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise ValueError("source manifest SHA-256 mismatch")
    entries, source_metadata = load_matched_video_manifest(
        manifest_path,
        expected_source_csv_sha256=args.source_csv_sha256,
        expected_count=args.expected_count,
    )
    video_root = Path(args.video_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    final_path = output / "manifest.json"
    vm_download = load_download_manifest(
        Path(args.videomae_model).expanduser().resolve())
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        device = torch.device(args.device)
    dino, vj, vm = build_encoders(args, device)
    vj_indices = uniform_dense_token_indices(
        vj.dense_raw_tokens, 1024, device=device)
    vm_indices = uniform_segment_token_indices(
        vm.dense_raw_tokens, 256, device=device)
    vj_contract = canonical_vjepa2_dense_feature_metadata(
        model_id=args.vjepa_reference_model_id,
        layer_index=15,
        num_frames=64,
        resize_short_side=292,
        crop_size=256,
        raw_tokens=vj.dense_raw_tokens,
        num_tokens=1024,
        feature_dim=1024,
        token_indices=vj_indices.cpu(),
        frame_sampling="last_contiguous",
        source_num_frames=81,
    )
    vm_contract = canonical_videomae_v2_dense_feature_metadata(
        model_id=str(Path(args.videomae_model).expanduser().resolve()),
        layer_index=15,
        num_tokens=256,
        token_indices=vm_indices.cpu(),
    )
    fixed = {
        "schema_version": BANK_SCHEMA_VERSION,
        "kind": BANK_KIND,
        "sample_count": len(entries),
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": args.manifest_sha256,
            "source_csv_sha256": args.source_csv_sha256,
            "sample_identity_sha256": source_metadata.get(
                "sample_identity_sha256"),
        },
        "branches": {
            "dinov3": {
                "sample_shape": [1024, 1024],
                "dtype": "float16",
                "feature_contract": dino.feature_metadata(),
                "model_identity": {
                    "model_id": "dinov3_vitl16",
                    "repo_path": str(Path(args.dinov3_repo).resolve()),
                    "weight_path": str(Path(args.dinov3_weights).resolve()),
                    "source_commit": args.dinov3_source_commit,
                    "checkpoint_sha256": args.dinov3_checkpoint_sha256,
                },
            },
            "vjepa2": {
                "sample_shape": [1024, 1024],
                "dtype": "float16",
                "feature_contract": vj_contract,
                "model_identity": {
                    "model_id": args.vjepa_reference_model_id,
                    "load_path": str(Path(args.vjepa_model).resolve()),
                    "sha256": args.vjepa_model_sha256,
                },
            },
            "videomae_v2": {
                "sample_shape": [1280, 1024],
                "dtype": "float16",
                "feature_contract": vm_contract,
                "model_identity": {
                    "model_id": "OpenGVLab/VideoMAEv2-Large",
                    "revision": VIDEOMAE_REVISION,
                    "download_manifest_content_hash": vm_download[
                        "content_hash"],
                },
            },
        },
        "samples": [{
            "index": entry.manifest_index,
            "sample_id": entry.sample_id,
            "prompt": entry.caption,
        } for entry in entries],
    }
    progress_identity = canonical_sha256(fixed)
    receipt_dir = output / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    shard_count = (len(entries) + args.shard_rows - 1) // args.shard_rows
    started = time.time()
    for shard_index in range(rank, shard_count, world_size):
        start = shard_index * args.shard_rows
        end = min(start + args.shard_rows, len(entries))
        relative = f"shards/shard-{shard_index:05d}.safetensors"
        path = output / relative
        receipt_path = receipt_dir / f"shard-{shard_index:05d}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("progress_identity") != progress_identity
                or int(receipt.get("index", -1)) != shard_index
                or int(receipt.get("sample_start", -1)) != start
                or int(receipt.get("sample_end", -1)) != end
                or not path.is_file()
                or file_sha256(path) != receipt.get("sha256")
            ):
                raise ValueError(
                    f"resumable triple shard identity mismatch: {path}")
            print(json.dumps({
                "rank": rank,
                "resumed_shard": shard_index,
                "completed": end,
            }), flush=True)
            continue

        branch_rows = {
            "dinov3": [],
            "vjepa2": [],
            "videomae_v2": [],
        }
        for entry in entries[start:end]:
            video = decode_81(
                video_root / entry.video_name,
                size=entry.video_size_bytes,
                mtime_ns=entry.video_mtime_ns,
            ).to(device=device, dtype=torch.float32)
            temporal = torch.arange(
                17, 81, device=device, dtype=torch.long).unsqueeze(0)
            with torch.inference_mode():
                dino_feature = dino.forward_dense_no_grad(
                    video, gradient_checkpointing=False)
                vj_feature = vj.forward_dense(
                    video,
                    layer_index=15,
                    token_indices=vj_indices,
                    gradient_checkpointing=False,
                    temporal_window_indices=temporal,
                )
                vm_feature = vm.forward_dense(
                    video,
                    layer_index=15,
                    token_indices=vm_indices,
                    gradient_checkpointing=False,
                )
            observed = {
                "dinov3": tuple(dino_feature.shape),
                "vjepa2": tuple(vj_feature.shape),
                "videomae_v2": tuple(vm_feature.shape),
            }
            expected = {
                "dinov3": (1, 1024, 1024),
                "vjepa2": (1, 1024, 1024),
                "videomae_v2": (1, 1280, 1024),
            }
            if observed != expected:
                raise RuntimeError(
                    f"triple bank row shape mismatch: {observed}")
            branch_rows["dinov3"].append(
                dino_feature.squeeze(0).half().cpu().contiguous())
            branch_rows["vjepa2"].append(
                vj_feature.squeeze(0).half().cpu().contiguous())
            branch_rows["videomae_v2"].append(
                vm_feature.squeeze(0).half().cpu().contiguous())
            del video, dino_feature, vj_feature, vm_feature

        tensors = {
            name: torch.stack(rows, dim=0)
            for name, rows in branch_rows.items()
        }
        ids = [entry.sample_id for entry in entries[start:end]]
        shard_payload = {
            "index": shard_index,
            "sample_start": start,
            "sample_end": end,
            "sample_ids_sha256": canonical_sha256(ids),
            "tensor_sha256": {
                name: tensor_sha256(value)
                for name, value in tensors.items()
            },
            "branches": fixed["branches"],
            "source_manifest_sha256": args.manifest_sha256,
            "progress_identity": progress_identity,
        }
        shard_content_hash = canonical_sha256(shard_payload)
        temporary = path.with_name(f".{path.name}.rank{rank}.tmp")
        save_file(tensors, str(temporary), metadata={
            "content_hash": shard_content_hash,
            "sample_start": str(start),
            "sample_end": str(end),
            "source_manifest_sha256": args.manifest_sha256,
            "progress_identity": progress_identity,
        })
        os.replace(temporary, path)
        shard = {
            "index": shard_index,
            "path": relative,
            "sample_start": start,
            "sample_end": end,
            "sample_ids_sha256": canonical_sha256(ids),
            "sha256": file_sha256(path),
            "content_hash": shard_content_hash,
        }
        atomic_json(receipt_path, {
            **shard,
            "progress_identity": progress_identity,
            "rank": rank,
        })
        print(json.dumps({
            "rank": rank,
            "completed": end,
            "total": len(entries),
            "elapsed_seconds": time.time() - started,
            "shard": relative,
        }), flush=True)

    if distributed:
        dist.barrier()
    if rank == 0:
        shards = []
        for shard_index in range(shard_count):
            receipt_path = receipt_dir / f"shard-{shard_index:05d}.json"
            if not receipt_path.is_file():
                raise FileNotFoundError(
                    f"missing completed shard receipt: {receipt_path}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.pop("progress_identity", None) != progress_identity:
                raise ValueError("triple shard progress identity mismatch")
            receipt.pop("rank", None)
            path = output / receipt["path"]
            if file_sha256(path) != receipt["sha256"]:
                raise ValueError(f"triple shard SHA-256 mismatch: {path}")
            shards.append(receipt)

        branch_hashers = {
            name: hashlib.sha256() for name in fixed["branches"]
        }
        for shard in shards:
            with safe_open(
                str(output / shard["path"]),
                framework="pt",
                device="cpu",
            ) as handle:
                for name in branch_hashers:
                    value = handle.get_tensor(name).contiguous()
                    branch_hashers[name].update(value.numpy().tobytes())
        for name, hasher in branch_hashers.items():
            fixed["branches"][name]["feature_bank_fp16_sha256"] = (
                hasher.hexdigest())
        final_payload = {**fixed, "shards": shards}
        final_payload["content_hash"] = canonical_sha256(final_payload)
        atomic_json(final_path, final_payload)
        print(json.dumps({
            "manifest": str(final_path),
            "manifest_sha256": file_sha256(final_path),
            "content_hash": final_payload["content_hash"],
            "samples": len(entries),
            "world_size": world_size,
        }, sort_keys=True), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create local asset paths without changing the published training settings."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from utils.config import resolve_config_aliases
PREFIX = "/path/to/elastic-forcing-assets"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    asset_root = str(args.asset_root.expanduser().resolve())
    cfg = resolve_config_aliases(yaml.safe_load((ROOT / "configs/train.yaml").read_text()))
    cfg = {k: asset_root + v[len(PREFIX):]
           if isinstance(v, str) and v.startswith(PREFIX + "/") else v
           for k, v in cfg.items()}
    if not args.inference_only:
        bank_path = Path(cfg["triple_teacher_bank_manifest"])
        data = bank_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != cfg["triple_teacher_bank_manifest_sha256"]:
            raise ValueError("Feature bank manifest does not match the published configuration")
        bank = json.loads(data)
        # Frozen metadata identities are separate from local model load paths.
        dino = bank["branches"]["dinov3"]["feature_contract"]
        cfg["dinov3_reference_paths"] = {k: dino[k] for k in ("repo_path", "weight_path")}
        cfg["videomae_v2_reference_model_id"] = bank["branches"]["videomae_v2"]["feature_contract"]["model_id"]
    if args.resume_checkpoint:
        cfg["resume_checkpoint"] = str(args.resume_checkpoint.expanduser().resolve())
    (ROOT / "configs/local.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (ROOT / "configs/local.env").write_text(
        "export ELASTIC_FORCING_ASSET_ROOT=" + shlex.quote(asset_root) + "\n")
    print("Local configuration written. Asset paths are excluded from Git.")

if __name__ == "__main__":
    main()

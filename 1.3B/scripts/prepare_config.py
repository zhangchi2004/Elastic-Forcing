#!/usr/bin/env python3
"""Relocate asset paths, preserving every scientific setting and content hash."""
import argparse
import json
from pathlib import Path
import shlex
import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from utils.config import resolve_config_aliases
ORIGINAL_ROOT = '/path/to/elastic-forcing-assets'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', required=True, type=Path,
                        help='Directory containing models/, checkpoints/, teacher_banks/, experiments/')
    args = parser.parse_args()
    original = ROOT / 'configs/three_encoder.yaml'
    base = resolve_config_aliases(yaml.safe_load(original.read_text()))
    asset_root = str(args.asset_root.expanduser().resolve())
    cfg = {k: asset_root + v[len(ORIGINAL_ROOT):]
           if isinstance(v, str) and v.startswith(ORIGINAL_ROOT + '/') else v
           for k, v in base.items()}
    # matched_video_root is an identity field; bank-based training does not decode it.
    output = ROOT / 'configs/local.yaml'
    output.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (ROOT / 'configs/local.env').write_text(
        'export ELASTIC_FORCING_ASSET_ROOT=' + shlex.quote(asset_root) + '\n')
    changes = {k: {'original': base[k], 'local': v} for k, v in cfg.items() if base[k] != v}
    (ROOT / 'configs/path-remapping.json').write_text(json.dumps(changes, indent=2) + '\n')
    print(f'Wrote {output}; {len(changes)} asset paths remapped, zero setting changes.')

if __name__ == '__main__':
    main()

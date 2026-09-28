#!/usr/bin/env python3
"""Replace only dataset/reference identity in the verified no-DINO config."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf
from utils.config import load_config
from utils.triple_video_feature_bank import file_sha256, NO_DINO_BANK_KIND
def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-config', required=True)
    p.add_argument('--bank-manifest', required=True)
    p.add_argument('--reference-manifest', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--expected-count', type=int, required=True)
    args = p.parse_args()
    config = load_config(args.base_config)
    original = OmegaConf.to_container(config)
    bank_path = Path(args.bank_manifest).resolve()
    refs_path = Path(args.reference_manifest).resolve()
    bank = json.loads(bank_path.read_text())
    refs = json.loads(refs_path.read_text())
    names = {'vjepa2', 'videomae_v2'}
    if bank['kind'] != NO_DINO_BANK_KIND or set(bank['branches']) != names or bank['sample_count'] != args.expected_count:
        raise ValueError('unexpected two-branch bank identity')
    bank_sha = file_sha256(bank_path)
    if refs['bank_content_hash'] != bank['content_hash'] or refs['bank_manifest_sha256'] != bank_sha:
        raise ValueError('references and bank disagree')
    if set(refs['references']) != names:
        raise ValueError('references must contain exactly two encoders')
    for item in refs['references'].values():
        if file_sha256(item['path']) != item['sha256']:
            raise ValueError('reference file SHA256 mismatch')
    source = bank['source_manifest']
    updates = {
        'triple_teacher_bank_manifest': str(bank_path),
        'triple_teacher_bank_manifest_sha256': bank_sha,
        'triple_teacher_bank_content_hash': bank['content_hash'],
        'matched_manifest_path': source['path'], 'matched_manifest_sha256': source['sha256'],
        'matched_manifest_samples': args.expected_count,
        'videomae_v2_reference_model_id': bank['branches']['videomae_v2']['feature_contract'].get('model_id', config.videomae_v2_model_id),
        'matched_source_csv_sha256': source['source_csv_sha256'],
        'matched_sample_identity_sha256': source['sample_identity_sha256'],
        'matched_video_root': str(Path(args.dataset_root).resolve() / 'videos'),
        'triple_nystrom_reference_path': refs['references']['vjepa2']['path'],
        'triple_nystrom_reference_sha256': refs['references']['vjepa2']['sha256'],
        'videomae_v2_nystrom_reference_path': refs['references']['videomae_v2']['path'],
        'videomae_v2_nystrom_reference_sha256': refs['references']['videomae_v2']['sha256'],
        'feature_protocol_identity': original['feature_protocol_identity'] + '_rebuilt',
        'dataset_identity': {
            **original['dataset_identity'], 'name': 'reference_videos',
            'sample_count': args.expected_count, 'source_manifest_sha256': source['sha256'],
            'teacher_bank_content_hash': bank['content_hash'],
            'reference_manifest_sha256': file_sha256(refs_path),
        },
    }
    for key, value in updates.items():
        config[key] = value
    actual = OmegaConf.to_container(config)
    if {k: v for k, v in actual.items() if k not in updates} != {k: v for k, v in original.items() if k not in updates}:
        raise ValueError('non-data hyperparameter changed')
    if actual['triple_active_branches'] != ['vjepa2', 'videomae_v2'] or actual['triple_aggregation_divisor'] != 3:
        raise ValueError('loss definition changed')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, suffix='.yaml.tmp')
    os.close(descriptor)
    try:
        OmegaConf.save(config, temporary)
        with open(temporary, 'rb') as handle:
            os.fsync(handle.fileno())
        if output.exists():
            if file_sha256(output) != file_sha256(temporary):
                raise ValueError('refusing to replace a different existing run config')
        else:
            os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    atomic_json(output.with_suffix('.identity.json'), dict(
        config_sha256=file_sha256(output), baseline_config_sha256=file_sha256(args.base_config),
        changed_data_keys=sorted(updates), sample_count=args.expected_count,
        bank_sha256=bank_sha, reference_manifest_sha256=file_sha256(refs_path),
        source_manifest_sha256=source['sha256'], resume_checkpoint=None,
        loss='(M_vj/(stopgrad(M_vj)+1e-4) + M_vm/(stopgrad(M_vm)+1e-4))/3'))
    print(str(output), flush=True)


if __name__ == '__main__':
    main()

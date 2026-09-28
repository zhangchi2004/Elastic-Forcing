import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from utils.config import load_config
from safetensors.torch import load_file, save_file

from tests.test_triple_video_feature_bank import _write_bank, _open
from utils.active_mmd_branches import NO_DINO_BRANCHES
from utils.triple_video_feature_bank import canonical_sha256, file_sha256, NO_DINO_BANK_KIND


def _pair_bank(tmp_path):
    path, manifest = _write_bank(tmp_path)
    manifest['kind'] = NO_DINO_BANK_KIND
    del manifest['branches']['dinov3']
    for shard in manifest['shards']:
        tensor_path = tmp_path / shard['path']
        tensors = load_file(tensor_path)
        del tensors['dinov3']
        save_file(tensors, tensor_path, metadata={'content_hash': shard['content_hash']})
        shard['sha256'] = file_sha256(tensor_path)
    manifest['content_hash'] = canonical_sha256({k: v for k, v in manifest.items() if k != 'content_hash'})
    path.write_text(json.dumps(manifest, sort_keys=True) + '\n')
    return path, manifest


def test_real_pair_bank_without_placeholder_dino(tmp_path):
    path, manifest = _pair_bank(tmp_path)
    pair = _open(path, manifest, active_branches=NO_DINO_BRANCHES,
                 expected_dinov3_checkpoint_sha256=None, expected_dinov3_source_commit=None)
    data = pair.load_rows(['sample-2', 'sample-0'])
    assert tuple(data) == NO_DINO_BRANCHES
    assert torch.equal(data['vjepa2'][:, 0, 0], torch.tensor([2., 0.], dtype=torch.float16))
    assert torch.equal(data['videomae_v2'][:, 0, 0], torch.tensor([12., 10.], dtype=torch.float16))
    with pytest.raises(ValueError, match='explicit no-DINO'):
        _open(path, manifest)
    with pytest.raises(ValueError, match='DINOv3 identity requested'):
        _open(path, manifest, active_branches=NO_DINO_BRANCHES)


def test_materializer_changes_only_data_and_preserves_all_hyperparameters(tmp_path, monkeypatch):
    from tools.materialize_no_dinov3_filtered_config import main
    bank_path, bank = _pair_bank(tmp_path)
    bank['source_manifest'].update(path='/data/new.jsonl', source_csv_sha256='c' * 64,
                                   sample_identity_sha256='d' * 64)
    bank['content_hash'] = canonical_sha256({k: v for k, v in bank.items() if k != 'content_hash'})
    bank_path.write_text(json.dumps(bank, sort_keys=True) + '\n')
    refs = {}
    for name in NO_DINO_BRANCHES:
        path = tmp_path / f'{name}.pt'
        path.write_bytes(b'test pack')
        refs[name] = {'path': str(path), 'sha256': file_sha256(path)}
    refs_path = tmp_path / 'references.json'
    refs_path.write_text(json.dumps({'bank_content_hash': bank['content_hash'],
                                    'bank_manifest_sha256': file_sha256(bank_path), 'references': refs}))
    # This helper is an optional two-encoder ablation, not the released recipe.
    recipe = Path(__file__).resolve().parents[2] / 'configs/train.yaml'
    ablation = load_config(recipe)
    ablation.triple_active_branches = ['vjepa2', 'videomae_v2']
    base = tmp_path / 'ablation.yaml'
    OmegaConf.save(ablation, base)
    output = tmp_path / 'train.yaml'
    monkeypatch.setattr('sys.argv', ['materialize', '--base-config', str(base),
        '--bank-manifest', str(bank_path), '--reference-manifest', str(refs_path),
        '--dataset-root', str(tmp_path), '--output', str(output), '--expected-count', '3'])
    main()
    old = OmegaConf.to_container(load_config(base))
    new = OmegaConf.to_container(load_config(output))
    for key in ('lr', 'lr_final', 'lr_schedule_total_steps', 'max_steps', 'log_iters',
                'checkpoint_keep_last', 'generator_ckpt', 'resume_checkpoint', 'triple_active_branches',
                'triple_aggregation_divisor', 'triple_vjepa2_weight', 'triple_videomae_v2_weight',
                'fresh_video_generator_grad_sample_scale', 'partial_mmd_geometry'):
        assert new[key] == old[key]
    assert new['matched_manifest_samples'] == 3
    assert new['triple_teacher_bank_manifest'] == str(bank_path)
    assert new['triple_teacher_bank_content_hash'] != old['triple_teacher_bank_content_hash']

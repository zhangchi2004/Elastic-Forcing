#!/usr/bin/env python3
"""Compare baseline and readable MMD values, feature gradients and state on CPU."""
import argparse
import importlib
import subprocess
import sys
import types
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--release", choices=("1.3B", "14B"), required=True)
parser.add_argument("--baseline", default="7985874ff49de326b8350a66516c50645a604f45")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
release = args.release
sys.path.insert(0, str(root / release / 'code'))
new_module = importlib.import_module('losses.multi_encoder_mmd')
legacy_module = importlib.import_module('losses.multi_representation_video_nystrom_mmd')
assert legacy_module is new_module
assert legacy_module.MultiRepresentationVideoNystromMMD is new_module.MultiEncoderMMD
assert legacy_module.fd_normalize_raw_mmd is new_module.normalize_mmd_with_detached_scale
old_module = types.ModuleType('baseline_multi_encoder')
old_source = subprocess.check_output(['git', 'show', f'{args.baseline}:{release}/code/losses/multi_representation_video_nystrom_mmd.py'], cwd=root, text=True)
exec(compile(old_source, '<baseline>', 'exec'), old_module.__dict__)
from losses.video_nystrom_mmd import VideoNystromMMD, build_video_nystrom_reference

def same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right

generator = torch.Generator().manual_seed(1862643)
packs = {}
shapes = {'dino': (1, 2), 'vjepa': (2, 3), 'videomae': (1, 4)}
for name, shape in shapes.items():
    reference = torch.randn(12, *shape, generator=generator)
    packs[name] = build_video_nystrom_reference(reference, num_landmarks=5, metadata={'feature': {'name': name, 'matrix_shape': list(shape)}, 'reference': {'dataset': 'readability-parity'}}, bandwidth=1.7, kmeans_iterations=3, ridge=1e-5, chunk_size=16)

def objective(cls):
    branches = {name: VideoNystromMMD(pack, kernel_chunk_size=3, attraction_weight=2.0, current_teacher_attraction_enabled=True, current_teacher_attraction_weight=1.0) for name, pack in packs.items()}
    return cls(branches, weights={'dino': 10.0, 'vjepa': 1.0, 'videomae': 1.0}, normalization_eps=1e-4, aggregation_divisor=3.0)

for mode in ('full', 'partial'):
    old = objective(old_module.MultiRepresentationVideoNystromMMD)
    new = objective(new_module.MultiEncoderMMD)
    generated = {name: torch.randn(4, *shape, generator=generator) for name, shape in shapes.items()}
    teacher = {name: torch.randn(4 if mode == 'full' else 2, *shape, generator=generator) for name, shape in shapes.items()}
    results = []
    for module in (old, new):
        if mode == 'full':
            leaves = {name: value.clone().requires_grad_() for name, value in generated.items()}
            loss, metadata = module(leaves, current_teacher=teacher)
        else:
            indices = torch.tensor([0, 2])
            leaves = {name: value.index_select(0, indices).clone().requires_grad_() for name, value in generated.items()}
            loss, metadata = module(gradient_students=leaves, student_reference=generated, current_teacher=teacher, gradient_student_indices=indices)
        loss.backward()
        results.append((loss.detach(), metadata, {name: leaf.grad for name, leaf in leaves.items()}, module.state_dict()))
    same(*results)
    print(release, mode, 'loss, metadata, input gradients, and state_dict are bitwise equal')
for value in (-1.0, 0.0, 0.1, 7.0):
    results = []
    for function in (old_module.fd_normalize_raw_mmd, new_module.normalize_mmd_with_detached_scale):
        raw = torch.tensor(value, requires_grad=True)
        normalized, denominator = function(raw)
        normalized.backward()
        results.append((normalized.detach(), denominator, raw.grad))
    same(*results)
print(release, 'normalization parity and legacy module/class/function identity passed')

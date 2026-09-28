"""Regression coverage for the missing disabled-DINO offload entry."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from model.wan_kv_mmd import WanKVMMD


def _model(names, device="cpu"):
    model = object.__new__(WanKVMMD)
    torch.nn.Module.__init__(model)
    model.device = torch.device(device)
    model.generator_replay_frozen_encoder_cpu_offload = True
    model._generator_replay_frozen_encoder_offload_state = None
    model.dual_branch_registry = {
        name: {"encoder_attribute": f"{name}_encoder"} for name in names}
    model.vae = torch.nn.Linear(3, 3).to(device).eval().requires_grad_(False)
    for name in names:
        setattr(model, f"{name}_encoder", torch.nn.Linear(3, 3).to(device).eval().requires_grad_(False))
    return model


@pytest.mark.parametrize("names", [
    ("vjepa2", "videomae_v2"), ("dinov3", "vjepa2", "videomae_v2")])
def test_active_offload_restore_repeats_without_changing_weights(names):
    model = _model(names)
    modules = ("vae",) + tuple(f"{name}_encoder" for name in names)
    before = {name: {k: v.clone() for k, v in getattr(model, name).state_dict().items()} for name in modules}
    with patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "empty_cache"):
        for _ in range(2):
            moved = model.offload_frozen_modules_for_generator_replay()
            assert moved > 0
            assert model._generator_replay_frozen_encoder_offload_state["module_names"] == modules
            with pytest.raises(RuntimeError, match="already offloaded"):
                model.offload_frozen_modules_for_generator_replay()
            assert model.restore_frozen_modules_before_feature_replay() == moved
            assert model._generator_replay_frozen_encoder_offload_state is None
            for name in modules:
                module = getattr(model, name)
                assert not module.training and not any(p.requires_grad for p in module.parameters())
                for key, value in module.state_dict().items():
                    assert torch.equal(value, before[name][key])
    if "dinov3" not in names:
        assert not hasattr(model, "dinov3_encoder")


def test_missing_active_encoder_still_fails_closed_and_rolls_back():
    model = _model(("vjepa2", "videomae_v2"))
    del model.vjepa2_encoder
    with patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "empty_cache"):
        with pytest.raises(RuntimeError, match="vjepa2_encoder is unavailable"):
            model.offload_frozen_modules_for_generator_replay()
    assert model._generator_replay_frozen_encoder_offload_state is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_cuda_two_step_offload_restore_is_bitwise_exact():
    model = _model(("vjepa2", "videomae_v2"), "cuda:0")
    modules = (model.vae, model.vjepa2_encoder, model.videomae_v2_encoder)
    inputs = torch.randn(4, 3, device="cuda:0")
    expected = [module(inputs).detach() for module in modules]
    for _ in range(2):
        model.offload_frozen_modules_for_generator_replay()
        assert all(next(module.parameters()).device.type == "cpu" for module in modules)
        model.restore_frozen_modules_before_feature_replay()
        assert all(next(module.parameters()).device.type == "cuda" for module in modules)
        for module, baseline in zip(modules, expected):
            assert torch.equal(module(inputs), baseline)

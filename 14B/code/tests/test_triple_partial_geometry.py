from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trainer.distillation import Trainer


def _branch():
    return {
        "repulsion_denominator": torch.tensor(65536),
        "fixed_teacher_attraction_denominator": torch.tensor(256),
        "current_teacher_attraction_denominator": torch.tensor(32768),
        "current_teacher_self_kernel_denominator": torch.tensor(16384),
        "full_student_partial_gradient": torch.tensor(1),
    }


def _trainer():
    trainer = object.__new__(Trainer)
    trainer.fresh_video_total_samples = 256
    trainer.fresh_video_generator_grad_global_samples = 128
    trainer.fresh_video_full_student_partial_gradient = True
    trainer.fresh_video_generator_grad_sample_scale = 2.0
    trainer.frame_fd_fusion_enabled = False
    trainer.is_main_process = False
    trainer.step = 0
    trainer.config = SimpleNamespace(
        partial_mmd_normalization_identity=(
            "full_student_population_denominators_v1"),
        partial_mmd_kss_denominator=65536,
        partial_mmd_fixed_attraction_denominator=256,
        partial_mmd_live_cross_denominator=32768,
        partial_mmd_teacher_self_denominator=16384,
        partial_mmd_gradient_compensation=2.0,
    )
    trainer.model = SimpleNamespace(
        feature_type="triple_dinov3_vjepa2_videomae_v2_dense_latent",
        current_teacher_attraction_enabled=True,
        nystrom_repulsion_fifo_enabled=False,
        nystrom_attraction_ema_enabled=False,
        vjepa2_dense_layers=(15,),
        dense_layer=15,
    )
    return trainer


def _result(branch_names=("dinov3", "vjepa2", "videomae_v2")):
    branches = {name: _branch() for name in branch_names}
    metadata = {
        "branches": branches,
        "gradcache_kss_denominator": torch.tensor(65536),
        "gradcache_full_student_partial_gradient": torch.tensor(1),
    }
    return SimpleNamespace(
        metadata=metadata,
        global_samples=256,
        gradient_global_samples=128,
        current_teacher_global_samples=128,
    )


def test_triple_geometry_validator_accepts_exact_three_branch_contract():
    _trainer()._validate_partial_mmd_geometry(_result())


def test_triple_geometry_validator_rejects_legacy_dual_branch_set():
    with pytest.raises(RuntimeError, match="branch set mismatch"):
        _trainer()._validate_partial_mmd_geometry(
            _result(("vjepa2", "videomae_v2")))


def test_release_config_matches_execution_geometry():
    import yaml
    from utils.config import resolve_config_aliases
    config = resolve_config_aliases(yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/train.yaml").read_text()))
    assert config["max_steps"] == 80
    assert config["lr_schedule_total_steps"] == 200
    assert config["triple_active_branches"] == ["dinov3", "vjepa2", "videomae_v2"]
    assert config["triple_dinov3_weight"] == 10.0
    assert config["matched_manifest_samples"] == 8000
    assert config["triple_aggregation_divisor"] == 3.0
    assert config["batch_size"] == 2
    assert config["fresh_video_logical_micro_batch_size"] == 16
    assert config["fresh_video_physical_replay_batch_size"] == 1
    assert config["partial_mmd_geometry"] == [256, 128, 128]
    assert config["optimizer_state_cpu_offload_during_backward"]
    assert config["generator_replay_activation_cpu_offload"]


def test_logical_cotangent_is_sliced_without_rescaling_for_b1_replay():
    trainer = object.__new__(Trainer)
    trainer.fresh_video_physical_replay_batch_size = 1
    trainer.fresh_video_logical_micro_batch_size = 2
    descriptor = {
        "rng_state": {"opaque": True},
        "latent_shape": (2, 3, 1, 1, 1),
        "generation_descriptor": {
            "version": 2,
            "num_generated_blocks": 1,
            "noise": torch.arange(6).reshape(2, 3, 1, 1, 1),
            "trajectory_seeds": torch.tensor([11, 22]),
        },
        "student_latent": torch.arange(6).reshape(2, 3, 1, 1, 1),
    }
    physical_entry = {
        "replay_descriptor": descriptor,
        "image_or_video_shape": [2, 3, 1, 1, 1],
        "conditional_dict": {
            "prompt_embeds": torch.arange(8).reshape(2, 4),
        },
        "unconditional_dict": {
            "prompt_embeds": torch.arange(8, 16).reshape(2, 4),
        },
    }
    logical_entry = {
        "features": {"vjepa2": torch.arange(12).reshape(2, 2, 3)},
        "physical_entries": (physical_entry,),
    }
    cotangent = {
        "vjepa2": torch.arange(12, 24).reshape(2, 2, 3),
    }
    units = trainer._split_fresh_video_replay_units(
        logical_entry, cotangent)
    assert len(units) == 2
    for index, unit in enumerate(units):
        assert unit["replay_descriptor"]["latent_shape"][0] == 1
        assert unit["image_or_video_shape"][0] == 1
        torch.testing.assert_close(
            unit["cotangent"]["vjepa2"],
            cotangent["vjepa2"][index:index + 1],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            unit["replay_descriptor"]["generation_descriptor"][
                "trajectory_seeds"],
            descriptor["generation_descriptor"][
                "trajectory_seeds"][index:index + 1],
            rtol=0,
            atol=0,
        )

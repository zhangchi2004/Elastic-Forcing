import copy
from types import SimpleNamespace

import pytest

from model.wan_kv_mmd import WanKVMMD


def _triple_state_only():
    model = object.__new__(WanKVMMD)
    model.feature_type = "triple_dinov3_vjepa2_videomae_v2_dense_latent"
    model.is_dinov3_dense = False
    model.is_triple_dense = True
    model.uses_fresh_video_gradcache = True
    model.dense_feature_prefix = "triple"
    model.dense_encoder_name = "triple"
    model.dense_log_prefix = "triple_dense"
    model.fidel_two_level_enabled = False
    model.vjepa2_multilayer_enabled = False
    model.frame_fd_fusion_enabled = False
    model.mmd_estimator = "nystrom_exact_repulsion"
    model.triple_teacher_bank_content_hash = "a" * 64
    model.dual_teacher_bank_content_hash = "a" * 64
    model.dual_branch_registry = {
        "dinov3": {
            "reference_path": "/data/dino.pt",
            "reference_hash": "b" * 64,
            "nystrom_landmarks": 256,
        },
        "vjepa2": {
            "reference_path": "/data/vj.pt",
            "reference_hash": "c" * 64,
            "nystrom_landmarks": 1024,
        },
        "videomae_v2": {
            "reference_path": "/data/vm.pt",
            "reference_hash": "d" * 64,
            "nystrom_landmarks": 256,
        },
    }
    model.latent_mmd = SimpleNamespace(
        attraction_ema_enabled=False,
        repulsion_fifo_enabled=False,
    )
    model.distribution_state_pending = lambda: False
    model._fd_metadata = lambda: {
        "feature_type": model.feature_type,
        "teacher_bank_content_hash": model.triple_teacher_bank_content_hash,
    }
    return model


def test_triple_fd_state_round_trip_uses_own_namespace_and_version_10():
    model = _triple_state_only()
    state = model.fd_state_dict()
    assert state["triple_dense_nystrom_mmd_version"] == 10
    assert "vjepa2_dense_nystrom_mmd_version" not in state
    model.load_fd_state_dict(copy.deepcopy(state))
    assert model.can_load_initialization_fd_state(state)


def test_triple_fd_state_rejects_foreign_or_wrong_version():
    model = _triple_state_only()
    state = model.fd_state_dict()
    wrong = copy.deepcopy(state)
    wrong["triple_dense_nystrom_mmd_version"] = 9
    with pytest.raises(ValueError, match="triple_dense_nystrom_mmd_version"):
        model.load_fd_state_dict(wrong)
    foreign = copy.deepcopy(state)
    foreign["vjepa2_dense_nystrom_mmd_version"] = 10
    with pytest.raises(ValueError, match="another dense feature namespace"):
        model.load_fd_state_dict(foreign)

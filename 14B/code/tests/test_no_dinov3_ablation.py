"""Two-branch ablation: same loss scale, teacher mixture and replay geometry."""
import ast
import copy
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from model.wan_kv_mmd import WanKVMMD
from utils.active_mmd_branches import active_mmd_branches, NO_DINO_BRANCHES
from losses.multi_representation_video_nystrom_mmd import MultiRepresentationVideoNystromMMD
from tests.test_multi_representation_video_nystrom_mmd import _triple
from tests.test_triple_video_feature_bank import _write_bank, _open
from tests.test_triple_partial_geometry import _trainer, _result
from tests.test_triple_fd_state import _triple_state_only


def test_branch_selector_is_explicit_and_fail_closed():
    assert active_mmd_branches() == ("dinov3", *NO_DINO_BRANCHES)
    assert active_mmd_branches(list(NO_DINO_BRANCHES)) == NO_DINO_BRANCHES
    for invalid in ([], ["vjepa2"], ["vjepa2", "vjepa2"], list(reversed(NO_DINO_BRANCHES)), "vjepa2"):
        with pytest.raises(ValueError):
            active_mmd_branches(invalid)


def test_bank_pair_is_bitwise_same_without_reading_dino_tensors(tmp_path, monkeypatch):
    path, manifest = _write_bank(tmp_path)
    full = _open(path, manifest)
    pair = _open(path, manifest, active_branches=NO_DINO_BRANCHES,
                 expected_dinov3_checkpoint_sha256=None, expected_dinov3_source_commit=None)
    ids = ["sample-2", "sample-0", "sample-2"]
    expected = full.load_rows(ids)
    import utils.triple_video_feature_bank as bank_module
    original_open = bank_module.safe_open
    accesses = []

    class ActiveOnlyOpen:
        def __init__(self, *args, **kwargs):
            self.handle = original_open(*args, **kwargs)
        def __enter__(self):
            self.handle.__enter__()
            return self
        def get_slice(self, name):
            accesses.append(name)
            assert name != "dinov3", "disabled DINO tensor accessed"
            return self.handle.get_slice(name)
        def __exit__(self, *args):
            return self.handle.__exit__(*args)

    monkeypatch.setattr(bank_module, "safe_open", ActiveOnlyOpen)
    actual = pair.load_rows(ids)
    assert tuple(actual) == NO_DINO_BRANCHES
    assert set(accesses) == set(NO_DINO_BRANCHES)
    for name in NO_DINO_BRANCHES:
        assert torch.equal(actual[name], expected[name])
    assert pair.content_hash == full.content_hash


def test_encoder_initialization_never_calls_dino():
    model = object.__new__(WanKVMMD)
    # Populate only cheap frozen-prefix interface fixtures, no real weights.
    body = ast.parse(textwrap.dedent(inspect.getsource(
        WanKVMMD._initialize_triple_dinov3_vjepa2_videomae_v2_features)))
    attributes = next(ast.literal_eval(node.value) for node in ast.walk(body)
                      if isinstance(node, ast.Assign) and any(
                          isinstance(t, ast.Name) and t.id == "vjepa_attribute_names"
                          for t in node.targets))
    for name in attributes:
        setattr(model, name, 0)
    for name, value in {
        "vjepa2_dense_layers": (15,), "vjepa2_dense_num_frames": 64,
        "vjepa2_student_source_num_frames": 81,
        "vjepa2_student_frame_sampling": "last_contiguous",
        "vjepa2_dense_num_tokens": 1024, "vjepa2_dense_feature_dim": 1024,
        "videomae_v2_dense_layers": (15,), "videomae_v2_encoder_count": 5,
        "videomae_v2_skip_leading_frames": 1, "videomae_v2_segment_num_frames": 16,
        "videomae_v2_source_num_frames": 81, "videomae_v2_dense_num_tokens": 256,
        "videomae_v2_model_id": "fake-vm", "videomae_v2_reference_model_id": "fake-vm", "videomae_v2_download_manifest_content_hash": "a" * 64,
    }.items():
        setattr(model, name, value)
    model.vjepa2_encoder = SimpleNamespace()
    model.videomae_v2_encoder = SimpleNamespace(dense_feature_dim=1024)
    model._initialize_vjepa2_features = Mock()
    model._initialize_videomae_v2_features = Mock()
    model._bind_dense_runtime_aliases = Mock()
    model._initialize_dinov3_features = Mock(side_effect=AssertionError("DINO loaded"))
    model._initialize_triple_dinov3_vjepa2_videomae_v2_features(
        SimpleNamespace(total_batch_size=256, triple_active_branches=NO_DINO_BRANCHES), "cpu")
    model._initialize_dinov3_features.assert_not_called()
    assert tuple(model.dual_branch_registry) == NO_DINO_BRANCHES
    assert not hasattr(model, "dinov3_encoder")


def test_loss_and_gradients_equal_original_minus_dino_keep_divisor_three():
    old = _triple()
    pair = MultiRepresentationVideoNystromMMD(
        {name: copy.deepcopy(old.branches[name]) for name in NO_DINO_BRANCHES},
        weights=dict.fromkeys(NO_DINO_BRANCHES, 1.0),
        normalization_eps=1e-4, aggregation_divisor=3)
    rng = torch.Generator().manual_seed(4311)
    reference = {name: torch.randn(256, *branch.sample_shape, generator=rng)
                 for name, branch in old.branches.items()}
    indices = torch.arange(128, 256)
    leaves = {name: value[128:].clone().requires_grad_(True) for name, value in reference.items()}
    teacher = {name: torch.randn(128, *branch.sample_shape, generator=rng, requires_grad=True)
               for name, branch in old.branches.items()}
    args = dict(gradient_students=leaves, student_reference=reference,
                current_teacher=teacher, gradient_student_indices=indices, stage=False)
    old_loss, old_metadata = old(**args)
    pair_args = {key: ({name: value[name] for name in NO_DINO_BRANCHES}
                      if isinstance(value, dict) else value) for key, value in args.items()}
    new_loss, metadata = pair(**pair_args)
    raw_dino, _ = old.branches["dinov3"](**{
        key: (value["dinov3"] if isinstance(value, dict) else value) for key, value in args.items()})
    expected = old_loss - (10 / 3) * raw_dino / (raw_dino.detach() + 1e-4)
    torch.testing.assert_close(new_loss, expected, atol=1e-6, rtol=1e-6)
    # ×2 remains a separate replay compensation, not an encoder weight.
    new_grads = torch.autograd.grad(2 * new_loss, tuple(leaves.values()) + tuple(teacher.values()),
                                    allow_unused=True, retain_graph=True)
    old_grads = torch.autograd.grad(2 * old_loss, tuple(leaves.values()), retain_graph=True)
    assert new_grads[0] is None
    for idx, name in enumerate(NO_DINO_BRANCHES, start=1):
        torch.testing.assert_close(new_grads[idx], old_grads[idx], rtol=1e-6, atol=1e-7)
        assert torch.isfinite(new_grads[idx]).all() and new_grads[idx].norm() > 0
        branch = metadata["branches"][name]
        assert torch.equal(branch["raw_mmd"], old_metadata["branches"][name]["raw_mmd"])
        assert int(branch["repulsion_denominator"]) == 65536
        assert int(branch["current_teacher_attraction_denominator"]) == 32768
        assert float(branch["fixed_teacher_attraction_normalized_weight"]) == pytest.approx(2 / 3)
        assert float(branch["current_teacher_attraction_normalized_weight"]) == pytest.approx(1 / 3)
    assert all(value is None for value in new_grads[3:])
    assert tuple(metadata["branches"]) == NO_DINO_BRANCHES
    assert not any(parameter.requires_grad for parameter in pair.parameters())


def test_partial_geometry_uses_actual_pair_registry():
    trainer = _trainer()
    trainer.model.dual_branch_registry = dict.fromkeys(NO_DINO_BRANCHES, {})
    trainer._validate_partial_mmd_geometry(_result(NO_DINO_BRANCHES))
    with pytest.raises(RuntimeError, match="branch set mismatch"):
        trainer._validate_partial_mmd_geometry(_result())


def test_exact_resume_rejects_original_three_branch_state():
    old = _triple_state_only()
    pair = _triple_state_only()
    del pair.dual_branch_registry["dinov3"]
    own_state = pair.fd_state_dict()
    pair.load_fd_state_dict(copy.deepcopy(own_state))
    with pytest.raises(ValueError):
        pair.load_fd_state_dict(old.fd_state_dict())


def test_objective_binding_does_not_require_any_dino_reference(tmp_path, monkeypatch):
    import hashlib
    import losses.video_nystrom_mmd as single_module
    import losses.multi_encoder_mmd as multi_module
    paths = {name: tmp_path / f"{name}.pt" for name in NO_DINO_BRANCHES}
    for path in paths.values():
        path.write_bytes(b"small test reference stub")
    digest = hashlib.sha256(b"small test reference stub").hexdigest()
    sources = {name: {"triple_teacher_bank_content_hash": "a" * 64,
                      "source_manifest_sha256": "b" * 64,
                      "selected_samples": 8000, "branch_name": name}
               for name in NO_DINO_BRANCHES}

    class ReferenceStub:
        def __init__(self, name, **kwargs):
            self.sample_shape = (1280, 1024) if name == "videomae_v2" else (1024, 1024)
            self.reference_metadata = {"reference": {"source": sources[name]}}
            self.reference_hash = name
            self.landmarks = torch.empty(1)
        def to(self, **kwargs):
            return self

    monkeypatch.setattr(single_module, "VideoNystromMMD", ReferenceStub)
    loaded = []
    def load_pack(path, **kwargs):
        name = Path(path).stem
        assert name == "videomae_v2", "unexpected reference (DINO) was loaded"
        loaded.append(name)
        return name
    monkeypatch.setattr(single_module, "load_reference_pack", load_pack)
    captured = {}
    def bind(branches, **kwargs):
        captured.update(branches=branches, **kwargs)
        return SimpleNamespace(to=lambda **unused: SimpleNamespace())
    monkeypatch.setattr(multi_module, "MultiEncoderMMD", bind)
    model = object.__new__(WanKVMMD)
    for name, value in {
        "uses_fresh_video_gradcache": True, "latent_self_normalize": True,
        "nystrom_attraction_ema_enabled": False, "nystrom_repulsion_fifo_enabled": False,
        "frame_fd_fusion_enabled": False, "current_teacher_attraction_enabled": True,
        "nystrom_attraction_weight": 2.0, "current_teacher_attraction_weight": 1.0,
        "expected_current_teacher_samples": 128, "latent_norm_eps": 1e-4,
        "vjepa2_nystrom_reference_path": str(paths["vjepa2"]),
        "videomae_v2_encoder": SimpleNamespace(dense_raw_tokens=1568, dense_feature_dim=1024),
        "videomae_v2_dense_num_tokens": 256, "videomae_v2_model_id": "fake-vm", "videomae_v2_reference_model_id": "fake-vm",
        "latent_mmd": ReferenceStub("vjepa2"),
        "dual_branch_registry": {name: {} for name in NO_DINO_BRANCHES},
    }.items():
        setattr(model, name, value)
    model._distributed_file_sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    model._dense_config_value = lambda *args: digest
    model._dinov3_feature_metadata = Mock(side_effect=AssertionError("DINO metadata requested"))
    args = SimpleNamespace(
        triple_active_branches=NO_DINO_BRANCHES,
        videomae_v2_nystrom_reference_path=str(paths["videomae_v2"]),
        videomae_v2_nystrom_reference_sha256=digest,
        triple_teacher_bank_content_hash="a" * 64, matched_manifest_sha256="b" * 64,
        matched_manifest_samples=8000)
    model._replace_with_triple_nystrom_objective(args, "cpu")
    assert loaded == ["videomae_v2"]
    assert tuple(captured["branches"]) == NO_DINO_BRANCHES
    assert captured["weights"] == dict.fromkeys(NO_DINO_BRANCHES, 1)
    assert captured["aggregation_divisor"] == 3
    assert captured["normalization_eps"] == 1e-4
    assert not hasattr(model, "dinov3_nystrom_reference_path")


def test_replay_sums_two_encoder_cotangents_and_calls_vae_once(monkeypatch):
    import model.wan_kv_mmd as model_module
    model = object.__new__(WanKVMMD)
    model.frame_fd_fusion_enabled = False
    model.vjepa2_dense_gradient_checkpointing = True
    model.videomae_v2_dense_gradient_checkpointing = True
    model.independent_first_frame = False
    model.dense_log_prefix = "triple_dense"
    model.vae = object()
    model.dual_branch_registry = {name: {"sample_shape": (1, 2)} for name in NO_DINO_BRANCHES}
    model._dual_token_indices = lambda device: {name: torch.arange(1) for name in NO_DINO_BRANCHES}
    expected = {name: torch.ones(1, 1, 2) for name in NO_DINO_BRANCHES}
    cotangents = {name: torch.ones(1, 1, 2) for name in NO_DINO_BRANCHES}
    pixels = torch.zeros(1, 81, 3, 1, 1)
    latent = torch.zeros(1, 1, 1, 1, 1)
    model._replay_dinov3_dense_to_pixels = Mock(side_effect=AssertionError("DINO replay called"))
    model._replay_dense_encoder_vjp = Mock(return_value=(expected["vjepa2"], torch.ones_like(pixels)))
    vm_replay = Mock(return_value=(expected["videomae_v2"], 2 * torch.ones_like(pixels)))
    model.videomae_v2_encoder = SimpleNamespace(replay_dense_vjp=vm_replay)
    def decoder(vae, z, combined, **kwargs):
        assert torch.equal(combined, 3 * torch.ones_like(pixels))
        return torch.ones_like(z) * combined.sum()
    decoder_mock = Mock(side_effect=decoder)
    monkeypatch.setattr(model_module, "replay_decoder_vjp", decoder_mock)
    descriptor = dict(rng_state={}, latent_shape=tuple(latent.shape),
                      generation_descriptor={}, student_latent=latent,
                      cached_student_video01=pixels)
    result = model._dual_gradcache_prepare_replay(
        replay_descriptor=descriptor, expected_features=expected,
        feature_cotangent=cotangents, frame_fd_feature_cotangents=None,
        replay_atol=1e-5, replay_rtol=1e-4, collect_branch_latent_diagnostics=False)
    assert decoder_mock.call_count == vm_replay.call_count == model._replay_dense_encoder_vjp.call_count == 1
    model._replay_dinov3_dense_to_pixels.assert_not_called()
    assert result["replay_log"]["triple_dense_encoder_vjp_count"].item() == 2
    assert result["replay_log"]["triple_dense_vae_vjp_count"].item() == 1
    assert result["latent_cotangent"].isfinite().all()
    assert not any("dinov3" in key for key in result["replay_log"])

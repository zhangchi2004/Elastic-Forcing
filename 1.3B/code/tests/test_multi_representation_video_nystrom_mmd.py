import torch

from losses.multi_representation_video_nystrom_mmd import (
    MultiRepresentationVideoNystromMMD,
    fd_normalize_raw_mmd,
)
from losses.video_nystrom_mmd import (
    VideoNystromMMD,
    build_video_nystrom_reference,
)
from trainer.multi_representation_fresh_video_gradcache import (
    MultiRepresentationFreshVideoGradCache,
)


def _objective(name: str, shape: tuple[int, int], seed: int):
    generator = torch.Generator().manual_seed(seed)
    reference = torch.randn(12, *shape, generator=generator)
    pack = build_video_nystrom_reference(
        reference,
        num_landmarks=5,
        metadata={
            "feature": {"name": name, "matrix_shape": list(shape)},
            "reference": {"dataset": "triple-unit-test"},
        },
        bandwidth=1.7,
        kmeans_iterations=3,
        ridge=1.0e-5,
        chunk_size=16,
    )
    return VideoNystromMMD(
        pack,
        kernel_chunk_size=64,
        attraction_weight=2.0,
        current_teacher_attraction_enabled=True,
        current_teacher_attraction_weight=1.0,
    )


def _triple():
    return MultiRepresentationVideoNystromMMD(
        {
            "dinov3": _objective("tiny-dinov3", (1, 3), 4199),
            "vjepa2": _objective("tiny-vjepa2", (1, 2), 4200),
            "videomae_v2": _objective("tiny-videomae", (2, 3), 4201),
        },
        weights={"dinov3": 15.0, "vjepa2": 1.0, "videomae_v2": 2.0},
        normalization_eps=1.0e-4,
        aggregation_divisor=10.0,
    )


def test_three_raw_mmds_are_normalized_weighted_then_divided_by_ten():
    objective = _triple()
    generator = torch.Generator().manual_seed(4202)
    reference = {
        "dinov3": torch.randn(256, 1, 3, generator=generator),
        "vjepa2": torch.randn(256, 1, 2, generator=generator),
        "videomae_v2": torch.randn(256, 2, 3, generator=generator),
    }
    indices = torch.arange(128, 256)
    gradient = {
        name: value.index_select(0, indices).clone().requires_grad_(True)
        for name, value in reference.items()
    }
    teacher = {
        "dinov3": torch.randn(
            128, 1, 3, generator=generator, requires_grad=True),
        "vjepa2": torch.randn(
            128, 1, 2, generator=generator, requires_grad=True),
        "videomae_v2": torch.randn(
            128, 2, 3, generator=generator, requires_grad=True),
    }
    loss, metadata = objective(
        gradient_students=gradient,
        student_reference=reference,
        current_teacher=teacher,
        gradient_student_indices=indices,
        stage=False,
    )

    expected_terms = []
    for name, branch in objective.branches.items():
        raw, branch_metadata = branch(
            gradient_students=gradient[name],
            student_reference=reference[name],
            current_teacher=teacher[name],
            gradient_student_indices=indices,
            stage=False,
        )
        normalized = raw / (raw.detach() + 1.0e-4)
        expected_terms.append(objective.branch_weights[name] * normalized)
        observed = metadata["branches"][name]
        torch.testing.assert_close(observed["raw_mmd"], raw.detach())
        assert int(observed["repulsion_denominator"]) == 65536
        assert int(observed["fixed_teacher_attraction_denominator"]) == 256
        assert int(observed["current_teacher_attraction_denominator"]) == 32768
        assert int(observed["current_teacher_self_kernel_denominator"]) == 16384
        assert int(observed["full_student_partial_gradient"]) == 1
        assert abs(float(observed[
            "fixed_teacher_attraction_normalized_weight"]) - 2.0 / 3.0) < 1e-7
        assert abs(float(observed[
            "current_teacher_attraction_normalized_weight"]) - 1.0 / 3.0) < 1e-7
        assert branch_metadata["repulsion_denominator"].item() == 65536
    expected = torch.stack(expected_terms).sum() / 10.0
    torch.testing.assert_close(loss, expected, rtol=2.0e-5, atol=2.0e-6)
    # Aggregate exactly as (15 * DINOv3 + V-JEPA2 + 2 * VideoMAE-v2) / 10.
    torch.testing.assert_close(
        metadata["normalized_mmd_sum"], expected.detach(),
        rtol=2.0e-5, atol=2.0e-6)
    assert metadata["branches"]["dinov3"]["branch_weight"].item() == 15.0
    assert metadata["branches"]["vjepa2"]["branch_weight"].item() == 1.0
    assert metadata["branches"]["videomae_v2"]["branch_weight"].item() == 2.0
    assert metadata["aggregation_divisor"].item() == 10.0

    gradients = torch.autograd.grad(
        loss,
        tuple(gradient.values()) + tuple(teacher.values()),
        allow_unused=True,
    )
    assert all(value is not None and value.norm() > 0 for value in gradients[:3])
    assert all(value is None for value in gradients[3:])


def test_dual_dinov3_weight_is_applied_after_fd_normalization():
    objective = MultiRepresentationVideoNystromMMD(
        {
            "dinov3": _objective("tiny-dinov3-weighted", (1, 3), 4210),
            "vjepa2": _objective("tiny-vjepa2-weighted", (1, 2), 4211),
        },
        weights={"dinov3": 10.0, "vjepa2": 1.0},
        normalization_eps=1.0e-4,
    )
    generator = torch.Generator().manual_seed(4212)
    generated = {
        "dinov3": torch.randn(
            8, 1, 3, generator=generator, requires_grad=True),
        "vjepa2": torch.randn(
            8, 1, 2, generator=generator, requires_grad=True),
    }
    teacher = {
        "dinov3": torch.randn(8, 1, 3, generator=generator),
        "vjepa2": torch.randn(8, 1, 2, generator=generator),
    }

    loss, metadata = objective(
        generated,
        current_teacher=teacher,
        stage=False,
    )

    dino = metadata["branches"]["dinov3"]
    vjepa = metadata["branches"]["vjepa2"]
    expected = 10.0 * dino["normalized_mmd"] + vjepa["normalized_mmd"]
    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(
        dino["weighted_normalized_mmd"],
        10.0 * dino["normalized_mmd"],
    )
    assert dino["branch_weight"].item() == 10.0
    assert vjepa["branch_weight"].item() == 1.0

    gradients = torch.autograd.grad(loss, tuple(generated.values()))
    assert all(torch.isfinite(value).all() and value.norm() > 0
               for value in gradients)


def test_mapping_gradcache_suffix_and_cotangent_compensation():
    objective = _triple()
    cache = MultiRepresentationFreshVideoGradCache(
        objective,
        target_local_samples=4,
        gradient_local_samples=2,
        full_student_partial_gradient=True,
        require_sample_id_alignment=True,
    )
    generator = torch.Generator().manual_seed(4203)
    batches = []
    for index in range(4):
        features = {
            "dinov3": torch.randn(1, 1, 3, generator=generator),
            "vjepa2": torch.randn(1, 1, 2, generator=generator),
            "videomae_v2": torch.randn(1, 2, 3, generator=generator),
        }
        batches.append(features)
        selected = index >= 2
        teacher = ({
            "dinov3": torch.randn(1, 1, 3, generator=generator),
            "vjepa2": torch.randn(1, 1, 2, generator=generator),
            "videomae_v2": torch.randn(1, 2, 3, generator=generator),
        } if selected else None)
        sample_id = (f"sample-{index}",)
        cache.add(
            features,
            teacher,
            gradient_selected=selected,
            sample_ids=sample_id,
            current_teacher_sample_ids=sample_id if selected else None,
        )
    result = cache.finalize()
    assert result.replay_batch_indices == (2, 3)
    assert len(result.fsdp_cotangents) == 2
    assert set(result.fsdp_cotangents[0]) == {
        "dinov3", "vjepa2", "videomae_v2"}
    assert int(result.metadata["gradcache_kss_denominator"]) == 16
    assert int(result.metadata["gradcache_gradient_global_samples"]) == 2
    original_loss = result.loss.clone()
    compensated = tuple({
        name: value * 2.0 for name, value in cotangent.items()
    } for cotangent in result.fsdp_cotangents)
    for before, after in zip(result.fsdp_cotangents, compensated):
        for name in before:
            torch.testing.assert_close(after[name], before[name] * 2.0)
    torch.testing.assert_close(result.loss, original_loss, rtol=0, atol=0)


def test_fd_normalization_preserves_exact_signed_and_zero_neighborhood_behavior():
    eps = 1.0e-4
    cases = (
        (-2.0e-4, 2.0, -1.0e4),
        (0.0, 0.0, 1.0e4),
        (1.0e-8, 1.0e-8 / (eps + 1.0e-8), 1.0 / (eps + 1.0e-8)),
    )
    for value, expected_forward, expected_gradient in cases:
        raw = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        normalized, denominator = fd_normalize_raw_mmd(raw, eps)
        torch.testing.assert_close(
            normalized.detach(), torch.tensor(expected_forward, dtype=torch.float64))
        (gradient,) = torch.autograd.grad(normalized, raw)
        torch.testing.assert_close(
            gradient, torch.tensor(expected_gradient, dtype=torch.float64))
        torch.testing.assert_close(
            denominator, raw.detach() + eps, rtol=0, atol=0)

    singular = torch.tensor(-eps, dtype=torch.float64, requires_grad=True)
    normalized, denominator = fd_normalize_raw_mmd(singular, eps)
    assert denominator.item() == 0.0
    assert torch.isinf(normalized) and normalized.item() < 0.0

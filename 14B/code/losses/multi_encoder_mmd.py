"""Combine distribution-matching losses across frozen video encoders.

Each encoder defines its own video feature space and Gaussian kernel. Compute
one MMD per encoder, apply the release's detached per-encoder loss scaling,
then apply the configured weights and aggregation divisor. ``VideoNystromMMD``
implements generated repulsion and persistent/minibatch reference attraction.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from losses.video_nystrom_mmd import VideoNystromMMD


def normalize_mmd_with_detached_scale(
    raw_mmd: torch.Tensor,
    normalization_eps: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return M / (stop_gradient(M) + eps) and its detached scale.

    Detaching only the denominator preserves gradients through M while
    balancing the different numerical scales of the encoder objectives.
    This release-specific scaling is additional to the paper's MMD objective;
    the expression deliberately has no absolute value or clamping.
    """
    if not isinstance(raw_mmd, torch.Tensor) or raw_mmd.numel() != 1:
        raise TypeError("raw_mmd must be a scalar tensor")
    if not math.isfinite(float(normalization_eps)) or normalization_eps <= 0:
        raise ValueError("normalization_eps must be finite and positive")
    denominator = raw_mmd.detach() + float(normalization_eps)
    return raw_mmd / denominator, denominator


class MultiEncoderMMD(nn.Module):
    """Weighted distribution matching in multiple frozen feature spaces.

    A branch is one encoder, its kernel bandwidth, and its persistent Nyström
    reference. For raw branch losses M_e, the combined objective is
    sum_e weight_e * M_e / (stop_gradient(M_e) + eps), divided by
    ``aggregation_divisor``. That divisor is explicit; it need not equal the
    number of branches or the sum of weights.
    """

    def __init__(
        self,
        branches: Mapping[str, VideoNystromMMD],
        *,
        weights: Optional[Mapping[str, float]] = None,
        normalization_eps: float = 1.0e-4,
        aggregation_divisor: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(branches, Mapping) or not branches:
            raise ValueError("branches must be a non-empty mapping")
        if not math.isfinite(float(normalization_eps)) or normalization_eps <= 0:
            raise ValueError("normalization_eps must be finite and positive")
        if (
            not math.isfinite(float(aggregation_divisor))
            or aggregation_divisor <= 0
        ):
            raise ValueError("aggregation_divisor must be finite and positive")
        ordered = OrderedDict()
        current_teacher_modes = set()
        for name, objective in branches.items():
            if not isinstance(name, str) or not name:
                raise ValueError("branch names must be non-empty strings")
            if not isinstance(objective, VideoNystromMMD):
                raise TypeError(
                    f"branch {name!r} must be a VideoNystromMMD instance")
            current_teacher_modes.add(bool(
                objective.current_teacher_attraction_enabled))
            if any(parameter.requires_grad for parameter in objective.parameters()):
                raise ValueError(f"branch {name!r} reference must be frozen")
            ordered[name] = objective
        if len(current_teacher_modes) != 1:
            raise ValueError(
                "all branches must agree on current-teacher attraction mode")
        self.branches = nn.ModuleDict(ordered)
        configured_weights = weights or {name: 1.0 for name in ordered}
        if set(configured_weights) != set(ordered):
            raise ValueError("branch weights must exactly match branch names")
        self.branch_weights = {}
        for name in ordered:
            weight = float(configured_weights[name])
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"branch weight for {name!r} must be finite and non-negative")
            self.branch_weights[name] = weight
        if sum(self.branch_weights.values()) <= 0.0:
            raise ValueError("at least one branch weight must be positive")
        self.normalization_eps = float(normalization_eps)
        self.aggregation_divisor = float(aggregation_divisor)
        self.current_teacher_attraction_enabled = current_teacher_modes.pop()
        self.attraction_ema_enabled = False
        self.repulsion_fifo_enabled = False

    @property
    def branch_names(self) -> tuple[str, ...]:
        return tuple(self.branches.keys())

    @property
    def sample_shapes(self) -> dict[str, tuple[int, int]]:
        return {
            name: objective.sample_shape
            for name, objective in self.branches.items()
        }

    @property
    def landmarks(self) -> torch.Tensor:
        """Legacy diagnostics use the first branch; branch logs stay exact."""
        return self.branches[self.branch_names[0]].landmarks

    @property
    def branch_registry(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "sample_shape": objective.sample_shape,
                "reference_hash": objective.reference_hash,
                "weight": self.branch_weights[name],
            }
            for name, objective in self.branches.items()
        }

    def _validate_mapping(
        self,
        value: Optional[Mapping[str, torch.Tensor]],
        label: str,
        *,
        required: bool,
    ) -> Optional[Mapping[str, torch.Tensor]]:
        if value is None:
            if required:
                raise ValueError(f"{label} is required")
            return None
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a branch tensor mapping")
        if set(value) != set(self.branches):
            raise ValueError(
                f"{label} branch keys differ: got {sorted(value)}, "
                f"expected {sorted(self.branches)}")
        return value

    def forward(
        self,
        generated: Optional[Mapping[str, torch.Tensor]] = None,
        current_teacher: Optional[Mapping[str, torch.Tensor]] = None,
        *,
        gradient_students: Optional[Mapping[str, torch.Tensor]] = None,
        student_reference: Optional[Mapping[str, torch.Tensor]] = None,
        gradient_student_indices: Optional[
            torch.Tensor | Sequence[int]
        ] = None,
        stage: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Evaluate each encoder loss, then balance and combine their scales.

        ``gradient_students`` selects generated samples that retain gradients;
        ``student_reference`` holds the full generated feature population.
        ``current_teacher`` is the current frozen reference minibatch.
        These input names are retained for existing callers and checkpoints.
        """
        # Select full-gradient or cached partial-gradient inputs. Every encoder
        # receives the same sample selection.
        partial = gradient_students is not None or student_reference is not None
        generated = self._validate_mapping(
            generated, "generated", required=not partial)
        gradient_students = self._validate_mapping(
            gradient_students, "gradient_students", required=partial)
        student_reference = self._validate_mapping(
            student_reference, "student_reference", required=partial)
        current_teacher = self._validate_mapping(
            current_teacher,
            "current_teacher",
            required=self.current_teacher_attraction_enabled,
        )
        if (
            not self.current_teacher_attraction_enabled
            and current_teacher is not None
        ):
            raise ValueError(
                "current_teacher was provided while every branch has "
                "current-teacher attraction disabled")

        # Each feature space has its own kernel objective; encoder features
        # and persistent reference landmarks remain separate.
        normalized_terms = []
        branch_metadata: dict[str, dict[str, Any]] = {}
        for name, objective in self.branches.items():
            if partial:
                raw, metadata = objective(
                    gradient_students=gradient_students[name],
                    student_reference=student_reference[name],
                    current_teacher=(
                        current_teacher[name]
                        if current_teacher is not None else None
                    ),
                    gradient_student_indices=gradient_student_indices,
                    stage=stage,
                )
            else:
                raw, metadata = objective(
                    generated[name],
                    current_teacher=(
                        current_teacher[name]
                        if current_teacher is not None else None
                    ),
                    stage=stage,
                )
            # Balance the branch scale before applying its configured weight.
            normalized, denominator = normalize_mmd_with_detached_scale(
                raw, self.normalization_eps)
            weighted = self.branch_weights[name] * normalized
            normalized_terms.append(weighted)
            branch_metadata[name] = {
                **metadata,
                "raw_mmd": raw.detach(),
                "normalization_denominator": denominator.detach(),
                "normalized_mmd": normalized.detach(),
                "branch_weight": raw.new_tensor(self.branch_weights[name]),
                "weighted_normalized_mmd": weighted.detach(),
                "sample_shape": objective.sample_shape,
                "reference_hash": objective.reference_hash,
            }
        # Preserve the explicit reduction order and configured divisor.
        total = (
            torch.stack(normalized_terms).sum() / self.aggregation_divisor
        )
        # Detached values below are diagnostics; ``total`` retains the graph.
        first = branch_metadata[self.branch_names[0]]
        metadata = {
            "branches": branch_metadata,
            "branch_names": self.branch_names,
            "raw_mmd_sum": torch.stack([
                branch_metadata[name]["raw_mmd"] for name in self.branch_names
            ]).sum(),
            "normalized_mmd_sum": total.detach(),
            "normalization_eps": total.new_tensor(self.normalization_eps),
            "normalization": "raw_div_stop_gradient_raw_plus_eps",
            "aggregation": "weighted_sum_then_divide",
            "aggregation_divisor": total.new_tensor(
                self.aggregation_divisor),
            # Shared batch counts are used by feature-gradient replay;
            # per-encoder diagnostics remain available under ``branches``.
            "global_samples": first["global_samples"],
            "gradient_student_samples": first.get(
                "gradient_student_samples", first["global_samples"]),
            "repulsion_denominator": first.get("repulsion_denominator"),
            "repulsion": torch.stack([
                branch_metadata[name]["repulsion"]
                for name in self.branch_names
            ]).sum(),
            "attraction": torch.stack([
                branch_metadata[name]["attraction"]
                for name in self.branch_names
            ]).sum(),
            "attraction_weight": total.new_ones(()),
            "reference_constant": torch.stack([
                branch_metadata[name]["reference_constant"]
                for name in self.branch_names
            ]).sum(),
        }
        return total, metadata

    def prepare_commit(self) -> None:
        for objective in self.branches.values():
            objective.prepare_commit()

    def commit(self) -> None:
        for objective in self.branches.values():
            objective.commit()

    def discard(self) -> None:
        for objective in self.branches.values():
            objective.discard()


# Historical imports and serialized class references use these aliases.
MultiRepresentationVideoNystromMMD = MultiEncoderMMD
fd_normalize_raw_mmd = normalize_mmd_with_detached_scale

__all__ = [
    "MultiEncoderMMD",
    "normalize_mmd_with_detached_scale",
    "MultiRepresentationVideoNystromMMD",
    "fd_normalize_raw_mmd",
]

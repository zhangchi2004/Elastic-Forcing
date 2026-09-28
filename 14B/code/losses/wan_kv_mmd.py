"""RBF-MMD on Wan KV matrices with a mathematically valid EMA distribution.

An RBF kernel has an (infinite-dimensional) feature map ``phi`` and

    MMD²(P, Q) = || E_P[phi(x)] - E_Q[phi(y)] ||².

We approximate ``phi`` with fixed random Fourier features (RFF), then keep an
EMA of the *kernel mean embeddings*.  This represents an EMA mixture of whole
historical distributions; averaging raw KV rows across unrelated videos does
not.  Historical embeddings are detached, so gradients only reach the latest
student KV matrix.  The natural ``1-beta`` factor is intentionally preserved.

History updates are transactional: ``forward(stage=True)`` only stages detached
current embeddings.  ``commit()`` must be called after a successful optimizer
step; ``discard()`` is used when the optimizer step is skipped.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn


def _sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = x.square().sum(-1, keepdim=True)
    y2 = y.square().sum(-1).unsqueeze(0)
    return (x2 + y2 - 2.0 * x @ y.t()).clamp_min(0)


def exact_rbf_mmd(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    bandwidth: float,
    bandwidth_multipliers: Iterable[float] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Biased multi-bandwidth RBF-MMD² reference implementation."""
    if student.ndim != 2 or teacher.ndim != 2:
        raise ValueError("exact_rbf_mmd expects [samples, features] matrices")
    if student.shape[1] != teacher.shape[1] or min(student.shape[0], teacher.shape[0]) < 1:
        raise ValueError("student/teacher matrices need matching feature dims and samples")
    if not math.isfinite(float(bandwidth)) or float(bandwidth) <= 0:
        raise ValueError("bandwidth must be finite and positive")
    multipliers = tuple(float(v) for v in bandwidth_multipliers)
    if not multipliers or any((not math.isfinite(v) or v <= 0) for v in multipliers):
        raise ValueError("bandwidth multipliers must be finite and positive")

    x, y = student.float(), teacher.float()
    dxx, dyy, dxy = _sq_dist(x, x), _sq_dist(y, y), _sq_dist(x, y)
    bandwidth_sq = x.new_tensor(float(bandwidth) ** 2)
    terms = []
    for multiplier in multipliers:
        denom = 2.0 * bandwidth_sq * multiplier * multiplier
        terms.append(
            torch.exp(-dxx / denom).mean()
            + torch.exp(-dyy / denom).mean()
            - 2.0 * torch.exp(-dxy / denom).mean())
    return torch.stack(terms).mean()


class EMARBFMMD(nn.Module):
    """EMA RBF kernel-mean MMD using fixed random Fourier features.

    ``sample_mode="rows"`` preserves the original behavior, where every
    ``[D]`` row in a ``[B, N, D]`` input is one sample.  In
    ``sample_mode="video_matrix"``, the only sample axis is ``B`` and every
    position-sensitive ``[N, D]`` matrix is one sample.  Its RBF distance is
    the Frobenius distance, equivalently the Euclidean distance after
    vectorizing the matrix.
    """

    _STATE_VERSION = 2
    _SAMPLE_MODES = ("rows", "video_matrix")

    def __init__(
        self,
        beta: float = 0.99,
        num_rff_features: int = 256,
        bandwidth: Optional[float] = None,
        bandwidth_multipliers: Iterable[float] = (0.5, 1.0, 2.0),
        seed: int = 0,
        row_normalize: bool = False,
        sample_mode: str = "rows",
    ):
        super().__init__()
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("EMA beta must be in [0, 1)")
        if int(num_rff_features) <= 0:
            raise ValueError("num_rff_features must be positive")
        multipliers = tuple(float(v) for v in bandwidth_multipliers)
        if not multipliers or any((not math.isfinite(v) or v <= 0) for v in multipliers):
            raise ValueError("bandwidth multipliers must be finite and positive")
        if bandwidth is not None and (
            not math.isfinite(float(bandwidth)) or float(bandwidth) <= 0
        ):
            raise ValueError("bandwidth must be finite and positive")
        if sample_mode not in self._SAMPLE_MODES:
            raise ValueError(
                f"sample_mode must be one of {self._SAMPLE_MODES}, "
                f"got {sample_mode!r}")

        self.beta = float(beta)
        self.num_rff_features = int(num_rff_features)
        self.bandwidth_multipliers = multipliers
        self.seed = int(seed)
        self.row_normalize = bool(row_normalize)
        self.sample_mode = sample_mode

        initial_bw_sq = (
            float(bandwidth) ** 2 if bandwidth is not None else float("nan"))
        self.register_buffer(
            "bandwidth_sq", torch.tensor(initial_bw_sq, dtype=torch.float32),
            persistent=True)
        self.register_buffer("rff_weight", torch.empty(0), persistent=True)
        self.register_buffer("rff_phase", torch.empty(0), persistent=True)
        self.register_buffer("student_ema", torch.empty(0), persistent=True)
        self.register_buffer("teacher_ema", torch.empty(0), persistent=True)
        self.register_buffer(
            "num_updates", torch.zeros((), dtype=torch.long), persistent=True)

        # Deliberately not buffers: an uncommitted optimizer step must never enter
        # a checkpoint.
        self._pending_student: Optional[torch.Tensor] = None
        self._pending_teacher: Optional[torch.Tensor] = None

    @property
    def embedding_dim(self) -> int:
        return self.num_rff_features * len(self.bandwidth_multipliers)

    @staticmethod
    def _distributed() -> bool:
        return (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    @classmethod
    @torch.no_grad()
    def _all_ranks_true(cls, local_value: bool, device: torch.device) -> bool:
        """Return one consensus bit before entering later collectives.

        A rank-local exception here would leave the other ranks blocked in the
        RFF broadcast/all-reduce.  Every input-validation failure therefore
        becomes an all-rank failure.
        """
        if not cls._distributed():
            return bool(local_value)
        flag = torch.tensor(
            int(bool(local_value)), device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    @classmethod
    def _global_mean(
        cls,
        local_sum: torch.Tensor,
        local_count: int,
        *,
        return_count: bool = False,
    ):
        """Autograd-aware, sample-weighted mean across all ranks.

        The differentiable all-reduce is intentional.  Since every rank
        backpropagates the same global loss, its backward supplies the right
        scaling before FSDP averages the replicated parameter gradients.
        """
        if not cls._distributed():
            count = local_sum.new_tensor(float(local_count))
            mean = local_sum / float(local_count)
        else:
            global_sum = dist_nn.all_reduce(
                local_sum, op=dist.ReduceOp.SUM)
            count = local_sum.new_tensor(float(local_count))
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            mean = global_sum / count
        if return_count:
            return mean, count
        return mean

    @torch.no_grad()
    def _estimate_bandwidth_sq(
        self, student: torch.Tensor, teacher: torch.Tensor, max_samples: int = 256
    ) -> torch.Tensor:
        if self.sample_mode == "rows":
            merged = torch.cat([
                student.detach().flatten(0, 1),
                teacher.detach().flatten(0, 1),
            ], dim=0).float()
        else:
            # Keep each [N, D] video matrix intact: B is the sole sample axis.
            merged = torch.cat([
                student.detach(),
                teacher.detach(),
            ], dim=0).float()
        if merged.shape[0] > max_samples:
            indices = torch.linspace(
                0, merged.shape[0] - 1, max_samples, device=merged.device
            ).round().long()
            merged = merged.index_select(0, indices)
        if self._distributed():
            # Use all ranks for the one-time median heuristic.  Rows are
            # padded only for the collective and unpadded immediately after.
            local_size = torch.tensor(
                [merged.shape[0]], device=merged.device, dtype=torch.long)
            gathered_sizes = [
                torch.zeros_like(local_size) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_sizes, local_size)
            sizes = [int(size.item()) for size in gathered_sizes]
            max_size = max(sizes)
            padded = merged.new_zeros((max_size, *merged.shape[1:]))
            padded[:merged.shape[0]].copy_(merged)
            gathered_samples = [
                torch.empty_like(padded) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, padded)
            merged = torch.cat([
                samples[:size]
                for samples, size in zip(gathered_samples, sizes)
            ], dim=0)

        if merged.shape[0] < 2:
            estimate = merged.new_tensor(1.0)
        else:
            if not self._distributed() or dist.get_rank() == 0:
                if self.sample_mode == "rows":
                    distances_sq = torch.pdist(merged, p=2).square()
                else:
                    # ||X-Y||_F^2 without collapsing [N, D] into a token axis.
                    squared_norm = merged.square().sum(dim=(1, 2))
                    gram = torch.einsum(
                        "ind,jnd->ij", merged, merged)
                    pairwise_sq = (
                        squared_norm[:, None]
                        + squared_norm[None, :]
                        - 2.0 * gram
                    ).clamp_min(0)
                    upper = torch.triu_indices(
                        merged.shape[0],
                        merged.shape[0],
                        offset=1,
                        device=merged.device,
                    )
                    distances_sq = pairwise_sq[upper[0], upper[1]]
                positive = distances_sq[distances_sq > 0]
                estimate = (
                    positive.median()
                    if positive.numel()
                    else distances_sq.new_tensor(1.0)
                ).clamp_min(1e-6)
            else:
                estimate = merged.new_tensor(1.0)
        if self._distributed():
            # All ranks must use the same fixed, checkpointed RFF map.
            dist.broadcast(estimate, src=0)
        return estimate

    @torch.no_grad()
    def _initialize_rff(self, student: torch.Tensor, teacher: torch.Tensor) -> None:
        if self.sample_mode == "rows":
            sample_shape = (int(student.shape[-1]),)
        else:
            sample_shape = (
                int(student.shape[-2]),
                int(student.shape[-1]),
            )
        weight_shape = (*sample_shape, self.num_rff_features)
        if self.rff_weight.numel() > 0:
            if self.rff_weight.shape != weight_shape:
                raise ValueError(
                    "KV sample shape changed: RFF map expects "
                    f"{tuple(self.rff_weight.shape[:-1])}, got {sample_shape}")
            return
        if not bool(torch.isfinite(self.bandwidth_sq).item()):
            self.bandwidth_sq.copy_(self._estimate_bandwidth_sq(student, teacher))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        weight = torch.randn(
            weight_shape, generator=generator,
            dtype=torch.float32)
        phase = 2.0 * math.pi * torch.rand(
            self.num_rff_features, generator=generator, dtype=torch.float32)
        self.rff_weight.resize_(weight.shape).copy_(
            weight.to(device=student.device))
        self.rff_phase.resize_(phase.shape).copy_(
            phase.to(device=student.device))

    def _kernel_sum(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, int]:
        samples = matrix.float()
        if self.row_normalize:
            samples = torch.nn.functional.normalize(samples, dim=-1)
        if self.sample_mode == "rows":
            samples = samples.flatten(0, 1)
            base_projection = samples @ self.rff_weight
        else:
            # This is exactly
            # samples.flatten(1) @ rff_weight.flatten(0, 1), while retaining
            # the [N, D] axes and their positional identity.
            base_projection = torch.einsum(
                "bnd,ndr->br", samples, self.rff_weight)
        bandwidth = self.bandwidth_sq.sqrt().clamp_min(1e-6)
        scale = math.sqrt(
            2.0 / (self.num_rff_features * len(self.bandwidth_multipliers)))
        features = [
            torch.cos(
                base_projection / (bandwidth * multiplier) + self.rff_phase
            ) * scale
            for multiplier in self.bandwidth_multipliers
        ]
        return torch.cat(features, dim=-1).sum(dim=0), int(samples.shape[0])

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        *,
        stage: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        local_shape_ok = (
            student.ndim == 3
            and teacher.ndim == 3
            and student.shape == teacher.shape
            and student.device == teacher.device
            and min(student.shape[0], student.shape[1], student.shape[2]) > 0
        )
        if not self._all_ranks_true(local_shape_ok, student.device):
            raise ValueError(
                "invalid Wan KV shape/device on at least one rank; each local "
                "student/teacher pair must match [B, tokens, features]")

        # A mismatched feature width would initialize incompatible RFF maps.
        # Matrix samples additionally require a shared N because every token
        # position has its own slice of the position-sensitive random weight.
        if self._distributed():
            rff_sample_shape = (
                student.shape[-1:]
                if self.sample_mode == "rows"
                else student.shape[1:]
            )
            shape_bounds = torch.tensor(
                [*rff_sample_shape, *(-value for value in rff_sample_shape)],
                device=student.device,
                dtype=torch.long,
            )
            dist.all_reduce(shape_bounds, op=dist.ReduceOp.MIN)
            num_dimensions = len(rff_sample_shape)
            if any(
                int(shape_bounds[index].item())
                != -int(shape_bounds[index + num_dimensions].item())
                for index in range(num_dimensions)
            ):
                raise ValueError(
                    "Wan KV RFF sample shape differs across distributed ranks")

        rff_initialized = self.rff_weight.numel() > 0
        expected_rff_shape = (
            (student.shape[-1], self.num_rff_features)
            if self.sample_mode == "rows"
            else (
                student.shape[-2],
                student.shape[-1],
                self.num_rff_features,
            )
        )
        local_rff_shape_ok = (
            not rff_initialized
            or self.rff_weight.shape == expected_rff_shape
        )
        if not self._all_ranks_true(local_rff_shape_ok, student.device):
            raise ValueError(
                "RFF map shape is incompatible on at least one rank")
        if self._distributed():
            bandwidth_initialized = bool(
                torch.isfinite(self.bandwidth_sq).item())
            state_bounds = torch.tensor(
                [
                    int(rff_initialized),
                    -int(rff_initialized),
                    int(bandwidth_initialized),
                    -int(bandwidth_initialized),
                ],
                device=student.device,
                dtype=torch.int32,
            )
            dist.all_reduce(state_bounds, op=dist.ReduceOp.MIN)
            if (
                int(state_bounds[0].item()) != -int(state_bounds[1].item())
                or int(state_bounds[2].item()) != -int(state_bounds[3].item())
            ):
                raise RuntimeError(
                    "RFF/bandwidth initialization state differs across ranks")

        local_finite = bool(
            torch.isfinite(student).all().item()
            and torch.isfinite(teacher).all().item()
        )
        if not self._all_ranks_true(local_finite, student.device):
            raise FloatingPointError(
                "non-finite Wan KV feature on at least one rank")

        teacher = teacher.detach()
        self._initialize_rff(student, teacher)
        student_sum, student_count = self._kernel_sum(student)
        teacher_sum, teacher_count = self._kernel_sum(teacher)
        current_student, global_count = self._global_mean(
            student_sum, student_count, return_count=True)
        current_teacher = self._global_mean(teacher_sum, teacher_count)

        if self.student_ema.numel() == 0:
            current_coefficient = 1.0
            effective_student = current_student
            effective_teacher = current_teacher.detach()
        else:
            current_coefficient = 1.0 - self.beta
            # History has no graph. Consequently only current_student receives
            # gradients, with the mathematically correct (1-beta) coefficient.
            effective_student = (
                self.beta * self.student_ema.detach()
                + (1.0 - self.beta) * current_student)
            effective_teacher = (
                self.beta * self.teacher_ema.detach()
                + (1.0 - self.beta) * current_teacher.detach())

        loss = (effective_student - effective_teacher).square().sum()
        if stage:
            self._pending_student = current_student.detach()
            self._pending_teacher = current_teacher.detach()
        return loss, {
            "bandwidth": self.bandwidth_sq.detach().sqrt(),
            "history_updates": self.num_updates.detach().clone(),
            "current_coefficient": loss.new_tensor(current_coefficient),
            "local_samples": torch.tensor(
                student_count, device=loss.device, dtype=torch.long),
            "global_samples": global_count.detach().to(dtype=torch.long),
        }

    @torch.no_grad()
    def commit(self) -> None:
        """Commit staged kernel means after a successful optimizer step."""
        if self._pending_student is None or self._pending_teacher is None:
            return
        if not (
            torch.isfinite(self._pending_student).all()
            and torch.isfinite(self._pending_teacher).all()
        ):
            self.discard()
            raise FloatingPointError("refusing to commit non-finite KV kernel mean")
        if self.student_ema.numel() == 0:
            self.student_ema.resize_as_(self._pending_student).copy_(
                self._pending_student)
            self.teacher_ema.resize_as_(self._pending_teacher).copy_(
                self._pending_teacher)
        else:
            self.student_ema.mul_(self.beta).add_(
                self._pending_student, alpha=1.0 - self.beta)
            self.teacher_ema.mul_(self.beta).add_(
                self._pending_teacher, alpha=1.0 - self.beta)
        self.num_updates.add_(1)
        self.discard()

    def discard(self) -> None:
        """Drop staged state when backward/optimizer is skipped."""
        self._pending_student = None
        self._pending_teacher = None

    def get_extra_state(self):
        state = {
            "version": self._STATE_VERSION,
            "beta": self.beta,
            "num_rff_features": self.num_rff_features,
            "bandwidth_multipliers": self.bandwidth_multipliers,
            "seed": self.seed,
            "row_normalize": self.row_normalize,
        }
        # Preserve the exact legacy rows-mode payload.  Wan v3 checkpoints
        # contain this dict and must remain loadable without migration.
        if self.sample_mode != "rows":
            state["sample_mode"] = self.sample_mode
        return state

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        if state != expected:
            raise ValueError(
                f"Wan KV MMD checkpoint/config mismatch: saved={state}, "
                f"current={expected}")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        for name in (
            "rff_weight", "rff_phase", "student_ema", "teacher_ema",
        ):
            key = prefix + name
            if key in state_dict:
                getattr(self, name).resize_as_(state_dict[key])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

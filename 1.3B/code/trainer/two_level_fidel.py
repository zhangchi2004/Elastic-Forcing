"""Transactional two-level FiDEL weighting for video and frame objectives.

The controller deliberately operates on detached scalar diagnostics.  It
returns one FP64 cotangent multiplier for each raw loss; callers apply those
multipliers to the corresponding feature cotangents before replay.

All persistent updates are transactional.  ``stage`` computes the weights for
the current optimizer update from the last committed state and prepares the
next EMA/hysteresis state.  ``commit`` publishes it only after optimizer
success, while ``discard`` leaves persistent state unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional

import torch
from torch import nn


BRANCHES = ("video", "inception", "siglip", "mae")
FRAME_BRANCHES = ("inception", "siglip", "mae")
GROUPS = ("video", "frame")
_BRANCH_INDEX = {name: index for index, name in enumerate(BRANCHES)}


@dataclass(frozen=True)
class TwoLevelFiDELStage:
    """Detached FP64 result for one staged optimizer update."""

    raw_losses: Mapping[str, torch.Tensor]
    violations: Mapping[str, torch.Tensor]
    loss_scales: Mapping[str, torch.Tensor]
    frame_q: Mapping[str, torch.Tensor]
    frame_alpha: Mapping[str, torch.Tensor]
    frame_violation: torch.Tensor
    group_q: Mapping[str, torch.Tensor]
    group_weights: Mapping[str, torch.Tensor]
    cotangent_scales: Mapping[str, torch.Tensor]
    total_loss_value: torch.Tensor
    effective_theta_frame: torch.Tensor
    effective_theta_group: torch.Tensor
    frame_active: Mapping[str, bool]
    group_active: Mapping[str, bool]
    committed_updates_before: int


@dataclass(frozen=True)
class _PendingState:
    next_loss_scale_ema: torch.Tensor
    next_loss_scale_initialized: torch.Tensor
    next_theta_ema: torch.Tensor
    next_frame_active: torch.Tensor
    next_group_active: torch.Tensor


class TwoLevelFiDELController(nn.Module):
    """Two-level FiDEL controller with delayed, transactional EMA state.

    The four raw objectives are ``video``, ``inception``, ``siglip`` and
    ``mae``.  The three image objectives are first combined into ``frame``;
    ``video`` and ``frame`` are then combined with the configured group priors.

    ``stage`` accepts three independent reference diagnostics per branch:

    * ``current_scores``: score of the current student population;
    * ``floors``: real validation floor;
    * ``start_scores``: score of the frozen initial student.

    The violation is ``max((current-floor)/(start-floor+eps), 0)``.  A
    non-positive (up to epsilon) initial gap is rejected because it cannot
    define a meaningful normalized violation.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        loss_scale_beta: float = 0.99,
        theta_beta: float = 0.99,
        frame_temperature: float = 1.0,
        group_temperature: float = 1.0,
        frame_budget: float = 1.0,
        video_prior: float = 0.7,
        frame_prior: float = 0.3,
        warmup_commits: int = 5,
        hysteresis_enter: float = 0.02,
        epsilon: float = 1.0e-12,
    ) -> None:
        super().__init__()
        scalar_values = (
            loss_scale_beta,
            theta_beta,
            frame_temperature,
            group_temperature,
            frame_budget,
            video_prior,
            frame_prior,
            hysteresis_enter,
            epsilon,
        )
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("all FiDEL hyperparameters must be finite")
        if not 0.0 <= loss_scale_beta < 1.0:
            raise ValueError("loss_scale_beta must be in [0, 1)")
        if not 0.0 <= theta_beta < 1.0:
            raise ValueError("theta_beta must be in [0, 1)")
        if frame_temperature <= 0.0 or group_temperature <= 0.0:
            raise ValueError("FiDEL temperatures must be positive")
        if frame_budget <= 0.0:
            raise ValueError("frame_budget must be positive")
        if video_prior <= 0.0 or frame_prior <= 0.0:
            raise ValueError("both FiDEL group priors must be positive")
        if int(warmup_commits) != warmup_commits or warmup_commits < 0:
            raise ValueError("warmup_commits must be a non-negative integer")
        if hysteresis_enter < 0.0:
            raise ValueError("hysteresis_enter must be non-negative")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.loss_scale_beta = float(loss_scale_beta)
        self.theta_beta = float(theta_beta)
        self.frame_temperature = float(frame_temperature)
        self.group_temperature = float(group_temperature)
        self.frame_budget = float(frame_budget)
        self.video_prior = float(video_prior)
        self.frame_prior = float(frame_prior)
        self.warmup_commits = int(warmup_commits)
        self.hysteresis_enter = float(hysteresis_enter)
        self.epsilon = float(epsilon)

        self.register_buffer(
            "loss_scale_ema",
            torch.zeros(len(BRANCHES), dtype=torch.float64),
        )
        self.register_buffer(
            "loss_scale_initialized",
            torch.zeros(len(BRANCHES), dtype=torch.bool),
        )
        # The chemical-potential EMA starts at one and is slowly updated even
        # during warmup.  Warmup only fixes the *effective* threshold at one.
        self.register_buffer(
            "theta_ema",
            torch.ones(len(GROUPS), dtype=torch.float64),
        )
        self.register_buffer(
            "frame_active",
            torch.zeros(len(FRAME_BRANCHES), dtype=torch.bool),
        )
        self.register_buffer(
            "group_active",
            torch.zeros(len(GROUPS), dtype=torch.bool),
        )
        self.register_buffer(
            "committed_updates",
            torch.zeros((), dtype=torch.long),
        )
        self._stage_result: Optional[TwoLevelFiDELStage] = None
        self._pending_state: Optional[_PendingState] = None

    @property
    def staged(self) -> bool:
        return self._stage_result is not None

    def _metadata(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "branches": BRANCHES,
            "frame_branches": FRAME_BRANCHES,
            "groups": GROUPS,
            "loss_scale_beta": self.loss_scale_beta,
            "theta_beta": self.theta_beta,
            "frame_temperature": self.frame_temperature,
            "group_temperature": self.group_temperature,
            "frame_budget": self.frame_budget,
            "video_prior": self.video_prior,
            "frame_prior": self.frame_prior,
            "warmup_commits": self.warmup_commits,
            "hysteresis_enter": self.hysteresis_enter,
            "epsilon": self.epsilon,
        }

    def get_extra_state(self) -> dict[str, Any]:
        return self._metadata()

    def set_extra_state(self, state: Any) -> None:
        expected = self._metadata()
        if state != expected:
            raise RuntimeError(
                "incompatible two-level FiDEL controller metadata: "
                f"expected {expected}, got {state}")

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        result = super().load_state_dict(
            state_dict, strict=strict, assign=assign)
        self.discard()
        return result

    def _as_scalar(self, value: Any, *, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be a scalar")
            scalar = value.detach().to(
                device=self.loss_scale_ema.device,
                dtype=torch.float64,
            ).reshape(())
        else:
            try:
                scalar = torch.tensor(
                    float(value),
                    device=self.loss_scale_ema.device,
                    dtype=torch.float64,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a real scalar") from exc
        if not bool(torch.isfinite(scalar).item()):
            raise FloatingPointError(f"{name} must be finite")
        return scalar

    @staticmethod
    def _require_branches(
        values: Mapping[str, Any],
        *,
        name: str,
    ) -> None:
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a mapping")
        actual = set(values)
        expected = set(BRANCHES)
        if actual != expected:
            raise ValueError(
                f"{name} must contain exactly {BRANCHES}; got "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}")

    @staticmethod
    def _median(values: list[torch.Tensor]) -> Optional[torch.Tensor]:
        if not values:
            return None
        ordered = torch.sort(torch.stack(values)).values
        middle = len(values) // 2
        if len(values) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) * 0.5

    def _next_active(
        self,
        previous: torch.Tensor,
        violations: torch.Tensor,
    ) -> torch.Tensor:
        # Active branches remain active for every strictly positive violation;
        # inactive branches need to cross the guard band to re-enter.
        return torch.where(
            previous,
            violations > 0.0,
            violations > self.hysteresis_enter,
        )

    @staticmethod
    def _detached_mapping(
        names: tuple[str, ...],
        values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            name: values[index].detach().clone()
            for index, name in enumerate(names)
        }

    @torch.no_grad()
    def stage(
        self,
        raw_losses: Mapping[str, Any],
        current_scores: Mapping[str, Any],
        floors: Mapping[str, Any],
        start_scores: Mapping[str, Any],
    ) -> TwoLevelFiDELStage:
        """Stage weights and the next persistent state for one update."""
        if self.staged:
            raise RuntimeError(
                "a FiDEL update is already staged; commit or discard it")
        for name, values in (
            ("raw_losses", raw_losses),
            ("current_scores", current_scores),
            ("floors", floors),
            ("start_scores", start_scores),
        ):
            self._require_branches(values, name=name)

        raw = torch.stack([
            self._as_scalar(raw_losses[name], name=f"raw_losses[{name}]")
            for name in BRANCHES
        ])
        current = torch.stack([
            self._as_scalar(
                current_scores[name], name=f"current_scores[{name}]")
            for name in BRANCHES
        ])
        floor = torch.stack([
            self._as_scalar(floors[name], name=f"floors[{name}]")
            for name in BRANCHES
        ])
        start = torch.stack([
            self._as_scalar(
                start_scores[name], name=f"start_scores[{name}]")
            for name in BRANCHES
        ])

        gaps = start - floor
        invalid = gaps <= self.epsilon
        if bool(invalid.any().item()):
            names = [
                BRANCHES[index]
                for index in invalid.nonzero(as_tuple=False).flatten().tolist()
            ]
            raise ValueError(
                "each FiDEL student-start score must exceed its real floor "
                f"by more than epsilon; invalid branches: {names}")
        violations = ((current - floor) / (gaps + self.epsilon)).clamp_min(0.0)

        absolute_raw = raw.abs()
        current_scales = torch.where(
            self.loss_scale_initialized,
            self.loss_scale_ema,
            absolute_raw,
        )
        next_scales = torch.where(
            self.loss_scale_initialized,
            self.loss_scale_beta * self.loss_scale_ema
            + (1.0 - self.loss_scale_beta) * absolute_raw,
            absolute_raw,
        )
        next_scale_initialized = torch.ones_like(
            self.loss_scale_initialized)

        frame_violations = torch.stack([
            violations[_BRANCH_INDEX[name]] for name in FRAME_BRANCHES
        ])
        next_frame_active = self._next_active(
            self.frame_active, frame_violations)
        if int(self.committed_updates.item()) < self.warmup_commits:
            effective_theta_frame = self.theta_ema.new_tensor(1.0)
            effective_theta_group = self.theta_ema.new_tensor(1.0)
        else:
            effective_theta_frame = self.theta_ema[0].clone()
            effective_theta_group = self.theta_ema[1].clone()

        frame_q = next_frame_active.to(torch.float64) * torch.sigmoid(
            (frame_violations - effective_theta_frame)
            / self.frame_temperature
        )
        frame_q_sum = frame_q.sum()
        if bool((frame_q_sum > 0.0).item()):
            frame_alpha = self.frame_budget * frame_q / frame_q_sum
            frame_violation = (
                frame_alpha * frame_violations
            ).sum() / frame_alpha.sum()
        else:
            frame_alpha = torch.zeros_like(frame_q)
            frame_violation = frame_q_sum.clone()

        group_violations = torch.stack([
            violations[_BRANCH_INDEX["video"]],
            frame_violation,
        ])
        next_group_active = self._next_active(
            self.group_active, group_violations)
        group_q = next_group_active.to(torch.float64) * torch.sigmoid(
            (group_violations - effective_theta_group)
            / self.group_temperature
        )
        q_video, q_frame = group_q.unbind()
        has_video = bool((q_video > 0.0).item())
        has_frame = bool((q_frame > 0.0).item())
        if has_video and has_frame:
            weighted_video = self.video_prior * q_video
            weighted_frame = self.frame_prior * q_frame
            denominator = weighted_video + weighted_frame
            lambda_video = weighted_video / denominator
            lambda_frame = weighted_frame / denominator
        elif has_video:
            lambda_video = group_q.new_tensor(1.0)
            lambda_frame = group_q.new_tensor(0.0)
        elif has_frame:
            lambda_video = group_q.new_tensor(0.0)
            lambda_frame = group_q.new_tensor(1.0)
        else:
            lambda_video = group_q.new_tensor(0.0)
            lambda_frame = group_q.new_tensor(0.0)

        cotangent = torch.zeros_like(raw)
        cotangent[_BRANCH_INDEX["video"]] = (
            lambda_video
            / (current_scales[_BRANCH_INDEX["video"]] + self.epsilon)
        )
        for frame_index, name in enumerate(FRAME_BRANCHES):
            cotangent[_BRANCH_INDEX[name]] = (
                lambda_frame
                * frame_alpha[frame_index]
                / (current_scales[_BRANCH_INDEX[name]] + self.epsilon)
            )
        total_loss_value = (cotangent * raw).sum()

        positive_frame = [
            value for value in frame_violations.unbind()
            if bool((value > 0.0).item())
        ]
        frame_observation = self._median(positive_frame)
        positive_groups = [
            value for value in group_violations.unbind()
            if bool((value > 0.0).item())
        ]
        group_observation = self._median(positive_groups)
        next_theta = self.theta_ema.clone()
        if frame_observation is not None:
            next_theta[0] = (
                self.theta_beta * self.theta_ema[0]
                + (1.0 - self.theta_beta) * frame_observation
            )
        if group_observation is not None:
            next_theta[1] = (
                self.theta_beta * self.theta_ema[1]
                + (1.0 - self.theta_beta) * group_observation
            )

        if not bool(torch.isfinite(torch.cat([
            violations,
            current_scales,
            next_scales,
            frame_q,
            frame_alpha,
            frame_violation.reshape(1),
            group_q,
            torch.stack([lambda_video, lambda_frame]),
            cotangent,
            total_loss_value.reshape(1),
            next_theta,
        ])).all().item()):
            raise FloatingPointError("non-finite two-level FiDEL state")

        result = TwoLevelFiDELStage(
            raw_losses=self._detached_mapping(BRANCHES, raw),
            violations=self._detached_mapping(BRANCHES, violations),
            loss_scales=self._detached_mapping(BRANCHES, current_scales),
            frame_q=self._detached_mapping(FRAME_BRANCHES, frame_q),
            frame_alpha=self._detached_mapping(FRAME_BRANCHES, frame_alpha),
            frame_violation=frame_violation.detach().clone(),
            group_q=self._detached_mapping(GROUPS, group_q),
            group_weights={
                "video": lambda_video.detach().clone(),
                "frame": lambda_frame.detach().clone(),
            },
            cotangent_scales=self._detached_mapping(BRANCHES, cotangent),
            total_loss_value=total_loss_value.detach().clone(),
            effective_theta_frame=effective_theta_frame.detach().clone(),
            effective_theta_group=effective_theta_group.detach().clone(),
            frame_active={
                name: bool(next_frame_active[index].item())
                for index, name in enumerate(FRAME_BRANCHES)
            },
            group_active={
                name: bool(next_group_active[index].item())
                for index, name in enumerate(GROUPS)
            },
            committed_updates_before=int(self.committed_updates.item()),
        )
        self._pending_state = _PendingState(
            next_loss_scale_ema=next_scales.detach().clone(),
            next_loss_scale_initialized=next_scale_initialized.detach().clone(),
            next_theta_ema=next_theta.detach().clone(),
            next_frame_active=next_frame_active.detach().clone(),
            next_group_active=next_group_active.detach().clone(),
        )
        self._stage_result = result
        return result

    @torch.no_grad()
    def prepare_commit(self) -> TwoLevelFiDELStage:
        """Validate the complete pending transaction before optimizer mutation."""
        if self._stage_result is None or self._pending_state is None:
            raise RuntimeError("cannot prepare an unstaged FiDEL update")
        pending = self._pending_state
        if (
            pending.next_loss_scale_ema.shape != self.loss_scale_ema.shape
            or pending.next_loss_scale_initialized.shape
            != self.loss_scale_initialized.shape
            or pending.next_theta_ema.shape != self.theta_ema.shape
            or pending.next_frame_active.shape != self.frame_active.shape
            or pending.next_group_active.shape != self.group_active.shape
            or not bool(torch.isfinite(
                pending.next_loss_scale_ema).all().item())
            or not bool(torch.isfinite(pending.next_theta_ema).all().item())
        ):
            raise RuntimeError("invalid pending two-level FiDEL transaction")
        return self._stage_result

    @torch.no_grad()
    def commit(self) -> TwoLevelFiDELStage:
        """Publish a prepared transaction after optimizer success."""
        result = self.prepare_commit()
        pending = self._pending_state
        self.loss_scale_ema.copy_(pending.next_loss_scale_ema)
        self.loss_scale_initialized.copy_(
            pending.next_loss_scale_initialized)
        self.theta_ema.copy_(pending.next_theta_ema)
        self.frame_active.copy_(pending.next_frame_active)
        self.group_active.copy_(pending.next_group_active)
        self.committed_updates.add_(1)
        self._clear_stage()
        return result

    def _clear_stage(self) -> None:
        self._stage_result = None
        self._pending_state = None

    def discard(self) -> None:
        """Discard update-local state without changing persistent statistics."""
        self._clear_stage()

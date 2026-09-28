"""Separate distribution estimation from generator backpropagation (Sec. 3.3).

The distribution batch B supplies all cross-video interactions. Only the smaller
configured differentiation batch b needs feature gradients and rollout replay.
This module caches features and returns their cotangents (dL/d feature); the
trainer later replays the encoders, VAE, and generator to obtain dL/d parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from losses.multi_encoder_mmd import (
    MultiEncoderMMD,
)


def _detach(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    return value


def _detach_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", non_blocking=False)
    if isinstance(value, Mapping):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    return value


@dataclass(frozen=True)
class FeatureGradientReplayResult:
    """Detached loss, per-microbatch feature gradients, and replay bookkeeping.

    ``local_cotangents`` are the derivatives of the global distribution loss.
    ``fsdp_cotangents`` additionally compensate for FSDP's gradient averaging.
    Field names stay compatible with existing training logs and checkpoints.
    """

    loss: torch.Tensor
    optimized_loss: torch.Tensor
    metadata: Mapping[str, Any]
    local_cotangents: tuple[Mapping[str, torch.Tensor], ...]
    fsdp_cotangents: tuple[Mapping[str, torch.Tensor], ...]
    replay_batch_indices: tuple[int, ...]
    replay_sample_ids: tuple[tuple[str, ...], ...]
    local_samples: int
    global_samples: int
    gradient_local_samples: int
    gradient_global_samples: int
    current_teacher_local_samples: int
    current_teacher_global_samples: int
    world_size: int


class FeatureGradientReplay:
    """Cache B generated feature sets and differentiate the selected b samples.

    B and b denote global sample counts; constructor counts are per rank. Each
    encoder keeps its own [tokens, channels] layout while sharing sample order.
    This implementation selects a per-rank suffix. All B samples still define
    the distribution loss, including interactions with unselected samples.
    """

    def __init__(
        self,
        loss_module: MultiEncoderMMD,
        *,
        target_local_samples: int,
        gradient_local_samples: int,
        loss_scale: float = 1.0,
        require_sample_id_alignment: bool = True,
        full_student_partial_gradient: bool = True,
    ) -> None:
        if not isinstance(loss_module, MultiEncoderMMD):
            raise TypeError(
                "loss_module must be MultiRepresentationVideoNystromMMD")
        if not 0 < int(gradient_local_samples) < int(target_local_samples):
            raise ValueError(
                "dual GradCache requires a strict positive gradient suffix")
        if not full_student_partial_gradient:
            raise ValueError(
                "dual GradCache requires full_student_partial_gradient=true")
        self.loss_module = loss_module
        self.target_local_samples = int(target_local_samples)
        self.gradient_local_samples = int(gradient_local_samples)
        self.loss_scale = float(loss_scale)
        self.require_sample_id_alignment = bool(require_sample_id_alignment)
        self.full_student_partial_gradient = True
        self.partial_gradient_enabled = True
        self.current_teacher_attraction_enabled = bool(
            loss_module.current_teacher_attraction_enabled)
        self._feature_batches: list[dict[str, torch.Tensor]] = []
        self._teacher_batches: list[dict[str, torch.Tensor]] = []
        self._batch_sizes: list[int] = []
        self._gradient_batch_indices: list[int] = []
        self._gradient_batch_sizes: list[int] = []
        self._gradient_sample_ids: list[tuple[str, ...]] = []
        self._selection_started = False
        self._device: torch.device | None = None
        self._result: FeatureGradientReplayResult | None = None

    @staticmethod
    def _world_size() -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def local_samples(self) -> int:
        return sum(self._batch_sizes)

    @property
    def selected_local_samples(self) -> int:
        return sum(self._gradient_batch_sizes)

    @property
    def finalized(self) -> bool:
        return self._result is not None

    def _validate_mapping(
        self,
        value: Mapping[str, torch.Tensor],
        label: str,
        *,
        expected_batch: int | None = None,
    ) -> tuple[int, torch.device]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a branch tensor mapping")
        if set(value) != set(self.loss_module.branch_names):
            raise ValueError(
                f"{label} keys must be {self.loss_module.branch_names}")
        batch_size = None
        device = None
        for name in self.loss_module.branch_names:
            tensor = value[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 3
                or not tensor.is_floating_point()
                or min(tensor.shape) <= 0
            ):
                raise ValueError(f"{label}[{name!r}] must be floating [B,N,D]")
            if tuple(tensor.shape[1:]) != self.loss_module.sample_shapes[name]:
                raise ValueError(
                    f"{label}[{name!r}] geometry {tuple(tensor.shape[1:])} "
                    f"!= {self.loss_module.sample_shapes[name]}")
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(f"{label}[{name!r}] is non-finite")
            if batch_size is None:
                batch_size = int(tensor.shape[0])
                device = tensor.device
            elif int(tensor.shape[0]) != batch_size or tensor.device != device:
                raise ValueError(
                    f"all {label} branches must share batch size and device")
        if expected_batch is not None and batch_size != expected_batch:
            raise ValueError(
                f"{label} batch size {batch_size} != {expected_batch}")
        return int(batch_size), device

    def add(
        self,
        generated_features: Mapping[str, torch.Tensor],
        current_teacher_features: Mapping[str, torch.Tensor] | None = None,
        *,
        gradient_selected: bool = True,
        sample_ids: Sequence[str] | None = None,
        current_teacher_sample_ids: Sequence[str] | None = None,
    ) -> int:
        """Store one rollout's features without retaining its generator graph."""
        if self.finalized:
            raise RuntimeError("cannot add after finalize")
        batch_size, device = self._validate_mapping(
            generated_features, "generated_features")
        if self._device is None:
            self._device = device
        elif device != self._device:
            raise ValueError("all microbatches must share a device")
        if self.local_samples + batch_size > self.target_local_samples:
            raise ValueError("dual GradCache exceeds target_local_samples")
        if gradient_selected:
            self._selection_started = True
            if self.selected_local_samples + batch_size > self.gradient_local_samples:
                raise ValueError("dual GradCache exceeds gradient suffix")
        elif self._selection_started:
            raise ValueError("gradient-selected microbatches must form a suffix")

        normalized_ids = None
        if sample_ids is not None:
            normalized_ids = tuple(str(value) for value in sample_ids)
            if len(normalized_ids) != batch_size or any(not value for value in normalized_ids):
                raise ValueError("sample_ids must contain one non-empty ID per row")
        if (
            not self.current_teacher_attraction_enabled
            and current_teacher_sample_ids is not None
        ):
            raise ValueError(
                "current_teacher_sample_ids were provided while current-teacher "
                "attraction is disabled")
        if (
            gradient_selected
            and self.require_sample_id_alignment
            and self.current_teacher_attraction_enabled
        ):
            if normalized_ids is None:
                raise ValueError("selected dual samples require sample_ids")
            teacher_ids = (
                tuple(str(value) for value in current_teacher_sample_ids)
                if current_teacher_sample_ids is not None else None
            )
            if teacher_ids != normalized_ids:
                raise ValueError("student and teacher sample IDs/order differ")

        if gradient_selected and self.current_teacher_attraction_enabled:
            if current_teacher_features is None:
                raise ValueError("selected dual samples require teacher features")
            self._validate_mapping(
                current_teacher_features,
                "current_teacher_features",
                expected_batch=batch_size,
            )
        elif current_teacher_features is not None:
            raise ValueError(
                "current teacher features were provided when this microbatch "
                "must not carry teacher payloads")

        # Collection stage (Algorithm 1): keep only feature values and sample
        # order. Replay state is stored by the trainer, outside this cache.
        staged = {
            name: generated_features[name].detach().float().clone()
            for name in self.loss_module.branch_names
        }
        self._feature_batches.append(staged)
        index = len(self._feature_batches) - 1
        if gradient_selected:
            self._gradient_batch_indices.append(index)
            self._gradient_batch_sizes.append(batch_size)
            self._gradient_sample_ids.append(normalized_ids)
            if self.current_teacher_attraction_enabled:
                self._teacher_batches.append({
                    name: current_teacher_features[name].detach().float().clone()
                    for name in self.loss_module.branch_names
                })
        self._batch_sizes.append(batch_size)
        return index

    def _gather(self, value: torch.Tensor) -> list[torch.Tensor]:
        if self._world_size() == 1:
            return [value.detach()]
        gathered = [torch.empty_like(value) for _ in range(self._world_size())]
        dist.all_gather(gathered, value.detach().contiguous())
        return gathered

    def _validate_ready(self) -> None:
        local = torch.tensor(
            [self.local_samples, self.selected_local_samples],
            device=self._device,
            dtype=torch.long,
        )
        expected = (self.target_local_samples, self.gradient_local_samples)
        if self._world_size() == 1:
            observed = tuple(int(value) for value in local.tolist())
            if observed != expected:
                raise RuntimeError(
                    f"incomplete dual GradCache: {observed} != {expected}")
            return
        gathered = [torch.empty_like(local) for _ in range(self._world_size())]
        dist.all_gather(gathered, local)
        observed = [tuple(int(value) for value in item.tolist()) for item in gathered]
        if any(item != expected for item in observed):
            raise RuntimeError(
                f"incomplete distributed dual GradCache: {observed} != {expected}")

    def finalize(self) -> FeatureGradientReplayResult:
        """Evaluate the global objective and cache dL/d feature for replay."""
        if self._result is not None:
            return self._result
        self._validate_ready()
        world_size = self._world_size()
        global_distribution_features = {}
        global_differentiation_features = {}
        global_reference_features = (
            {} if self.current_teacher_attraction_enabled else None)
        local_leaves = {}
        # Map each rank's selected suffix into the concatenated B-sample batch.
        # These indices preserve full-batch interactions while only b rows have
        # an autograd path. This function does not randomly resample the subset.
        suffix_start = self.target_local_samples - self.gradient_local_samples
        gradient_indices = torch.cat([
            torch.arange(
                rank * self.target_local_samples + suffix_start,
                (rank + 1) * self.target_local_samples,
                device=self._device,
                dtype=torch.long,
            )
            for rank in range(world_size)
        ])
        for name in self.loss_module.branch_names:
            local_distribution_features = torch.cat([
                batch[name] for batch in self._feature_batches
            ], dim=0)
            local_differentiation_features = torch.cat([
                self._feature_batches[index][name]
                for index in self._gradient_batch_indices
            ], dim=0)
            distribution_parts = self._gather(local_distribution_features)
            differentiation_parts = self._gather(local_differentiation_features)
            # Only this rank's selected features are differentiable leaves.
            # Remote features contribute values; their owning rank gets their
            # derivatives. No generator or encoder graph is retained here.
            leaf = local_differentiation_features.detach().requires_grad_(True)
            differentiation_parts[self._rank()] = leaf
            local_leaves[name] = leaf
            global_distribution_features[name] = torch.cat(distribution_parts, dim=0).detach()
            global_differentiation_features[name] = torch.cat(differentiation_parts, dim=0)
            if self.current_teacher_attraction_enabled:
                local_reference_features = torch.cat([
                    batch[name] for batch in self._teacher_batches
                ], dim=0)
                reference_parts = self._gather(local_reference_features)
                global_reference_features[name] = torch.cat(
                    reference_parts, dim=0).detach()

        # Resolve all cross-video interactions once. The loss combines the
        # persistent Nystrom summary with current reference features (Eq. 9),
        # then aggregates the encoder objectives (Eqs. 11 and 13).
        raw_loss, metadata = self.loss_module(
            gradient_students=global_differentiation_features,
            student_reference=global_distribution_features,
            current_teacher=global_reference_features,
            gradient_student_indices=gradient_indices,
        )
        optimized = raw_loss * self.loss_scale
        gradients = torch.autograd.grad(
            optimized,
            tuple(local_leaves[name] for name in self.loss_module.branch_names),
            retain_graph=False,
            create_graph=False,
        )
        # Restore the collection order so each feature gradient is replayed
        # through the exact sample and encoder that produced its feature value.
        split_by_branch = {
            name: tuple(part.contiguous() for part in gradient.detach().split(
                self._gradient_batch_sizes, dim=0))
            for name, gradient in zip(self.loss_module.branch_names, gradients)
        }
        local_parts = tuple({
            name: split_by_branch[name][index]
            for name in self.loss_module.branch_names
        } for index in range(len(self._gradient_batch_sizes)))
        # FSDP averages parameter gradients across ranks, so cancel that factor
        # here. This world_size multiplier is separate from the B/b selective
        # differentiation correction (Eq. 12), applied later by the trainer.
        fsdp_parts = tuple({
            name: (value * float(world_size)).contiguous()
            for name, value in part.items()
        } for part in local_parts)
        global_samples = self.target_local_samples * world_size
        gradient_global = self.gradient_local_samples * world_size
        kss_denominator = global_samples * global_samples
        # Keep established metadata names for checkpoint/log compatibility.
        detached_metadata = dict(_detach(metadata))
        teacher_local = (
            self.gradient_local_samples
            if self.current_teacher_attraction_enabled else 0)
        teacher_global = teacher_local * world_size
        detached_metadata.update({
            "raw_loss": raw_loss.detach(),
            "optimized_loss": optimized.detach(),
            "loss_scale": raw_loss.new_tensor(self.loss_scale),
            "gradcache_local_samples": raw_loss.new_tensor(
                self.target_local_samples, dtype=torch.long),
            "gradcache_global_samples": raw_loss.new_tensor(
                global_samples, dtype=torch.long),
            "gradcache_forward_local_samples": raw_loss.new_tensor(
                self.target_local_samples, dtype=torch.long),
            "gradcache_forward_global_samples": raw_loss.new_tensor(
                global_samples, dtype=torch.long),
            "gradcache_gradient_local_samples": raw_loss.new_tensor(
                self.gradient_local_samples, dtype=torch.long),
            "gradcache_gradient_global_samples": raw_loss.new_tensor(
                gradient_global, dtype=torch.long),
            "gradcache_current_teacher_local_samples": raw_loss.new_tensor(
                teacher_local, dtype=torch.long),
            "gradcache_current_teacher_global_samples": raw_loss.new_tensor(
                teacher_global, dtype=torch.long),
            "gradcache_kss_denominator": raw_loss.new_tensor(
                kss_denominator, dtype=torch.long),
            "gradcache_partial_gradient_enabled": raw_loss.new_ones(
                (), dtype=torch.long),
            "gradcache_full_student_partial_gradient": raw_loss.new_ones(
                (), dtype=torch.long),
            "gradcache_world_size": raw_loss.new_tensor(
                world_size, dtype=torch.long),
            "gradcache_fsdp_cotangent_scale": raw_loss.new_tensor(
                float(world_size)),
            "gradcache_branch_names": self.loss_module.branch_names,
        })
        self._result = FeatureGradientReplayResult(
            loss=raw_loss.detach(),
            optimized_loss=optimized.detach(),
            metadata=detached_metadata,
            local_cotangents=local_parts,
            fsdp_cotangents=fsdp_parts,
            replay_batch_indices=tuple(self._gradient_batch_indices),
            replay_sample_ids=tuple(self._gradient_sample_ids),
            local_samples=self.target_local_samples,
            global_samples=global_samples,
            gradient_local_samples=self.gradient_local_samples,
            gradient_global_samples=gradient_global,
            current_teacher_local_samples=teacher_local,
            current_teacher_global_samples=teacher_global,
            world_size=world_size,
        )
        return self._result

    def offload_finalized_to_cpu(
        self,
    ) -> FeatureGradientReplayResult:
        """Move finalized feature/cotangent storage to CPU transactionally."""
        if self._result is None:
            raise RuntimeError(
                "cannot offload a multi-representation cache before finalize")
        self._feature_batches = [
            _detach_to_cpu(value) for value in self._feature_batches
        ]
        self._teacher_batches = [
            _detach_to_cpu(value) for value in self._teacher_batches
        ]
        result = self._result
        self._result = FeatureGradientReplayResult(
            loss=_detach_to_cpu(result.loss),
            optimized_loss=_detach_to_cpu(result.optimized_loss),
            metadata=_detach_to_cpu(result.metadata),
            local_cotangents=_detach_to_cpu(result.local_cotangents),
            fsdp_cotangents=_detach_to_cpu(result.fsdp_cotangents),
            replay_batch_indices=result.replay_batch_indices,
            replay_sample_ids=result.replay_sample_ids,
            local_samples=result.local_samples,
            global_samples=result.global_samples,
            gradient_local_samples=result.gradient_local_samples,
            gradient_global_samples=result.gradient_global_samples,
            current_teacher_local_samples=result.current_teacher_local_samples,
            current_teacher_global_samples=result.current_teacher_global_samples,
            world_size=result.world_size,
        )
        return self._result


    def _clear(self) -> None:
        self._feature_batches.clear()
        self._teacher_batches.clear()
        self._batch_sizes.clear()
        self._gradient_batch_indices.clear()
        self._gradient_batch_sizes.clear()
        self._gradient_sample_ids.clear()
        self._selection_started = False
        self._device = None
        self._result = None

    def commit(self) -> None:
        """Release cached values after the corresponding optimizer update."""
        if self._result is None:
            raise RuntimeError("cannot commit before finalize")
        self._clear()

    def discard(self) -> None:
        """Release an abandoned batch without marking an optimizer update."""
        self._clear()


# Preserve old imports and serialized class references during the naming cleanup.
MultiRepresentationFreshVideoGradCache = FeatureGradientReplay
MultiRepresentationFreshVideoGradCacheResult = FeatureGradientReplayResult

__all__ = [
    "FeatureGradientReplay",
    "FeatureGradientReplayResult",
    "MultiRepresentationFreshVideoGradCache",
    "MultiRepresentationFreshVideoGradCacheResult",
]

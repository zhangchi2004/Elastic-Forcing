"""Fresh-update gradient cache for video-level distribution losses.

The cache deliberately has no state across optimizer updates.  It collects a
fixed number of detached ``[B, N, D]`` feature matrices on every rank, builds
the exact global fresh batch in deterministic rank order, and differentiates
the frozen reference loss only with respect to the local rank's feature leaf.

The two returned cotangents have distinct meanings:

``local_cotangents``
    The mathematical derivatives ``d L_global / d feature_local``.

``fsdp_cotangents``
    ``world_size * local_cotangents``.  These are the cotangents that replay
    should use when FSDP/DDP will average parameter gradients across ranks:

    ``mean_r(world_size * J_r.T @ dL/dF_r) = sum_r J_r.T @ dL/dF_r``.

Collection is transactional.  ``finalize`` stages a result but retains the
fresh features until the caller reports optimizer success with ``commit`` or
failure with ``discard``.  Both terminal operations clear all update-local
state, so no generated sample can leak into the next optimizer update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from losses.video_nystrom_mmd import VideoNystromMMD
from losses.multilayer_video_nystrom_mmd import (
    MultiLayerVideoNystromMMD,
)


def _detach_metadata(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {
            key: _detach_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_detach_metadata(item) for item in value)
    if isinstance(value, list):
        return [_detach_metadata(item) for item in value]
    return value

def _detach_to_cpu(value: Any) -> Any:
    """Recursively detach transactional result tensors onto host memory."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", non_blocking=False).contiguous()
    if isinstance(value, Mapping):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    return value


@dataclass(frozen=True)
class FreshVideoGradCacheResult:
    """Finalized loss and replay cotangents for one fresh optimizer update."""

    loss: torch.Tensor
    optimized_loss: torch.Tensor
    metadata: Mapping[str, Any]
    local_cotangents: tuple[torch.Tensor, ...]
    fsdp_cotangents: tuple[torch.Tensor, ...]
    replay_batch_indices: tuple[int, ...]
    replay_sample_ids: tuple[tuple[str, ...], ...]
    local_samples: int
    global_samples: int
    gradient_local_samples: int
    gradient_global_samples: int
    current_teacher_local_samples: int
    current_teacher_global_samples: int
    world_size: int


class FreshVideoGradCache:
    """Collect and differentiate one globally fresh video feature batch."""

    def __init__(
        self,
        loss_module: VideoNystromMMD | MultiLayerVideoNystromMMD,
        *,
        target_local_samples: int = 16,
        gradient_local_samples: int | None = None,
        loss_scale: float = 1.0,
        require_sample_id_alignment: bool = False,
        full_student_partial_gradient: bool = False,
    ) -> None:
        if not isinstance(
            loss_module,
            (VideoNystromMMD, MultiLayerVideoNystromMMD),
        ):
            raise TypeError(
                "loss_module must be a VideoNystromMMD or "
                "MultiLayerVideoNystromMMD instance")
        if int(target_local_samples) <= 0:
            raise ValueError("target_local_samples must be positive")
        if gradient_local_samples is None:
            gradient_local_samples = int(target_local_samples)
        if not 0 < int(gradient_local_samples) <= int(target_local_samples):
            raise ValueError(
                "gradient_local_samples must be in [1, target_local_samples]")
        if (
            not math.isfinite(float(loss_scale))
            or float(loss_scale) < 0.0
        ):
            raise ValueError("loss_scale must be finite and non-negative")
        if any(parameter.requires_grad for parameter in loss_module.parameters()):
            raise ValueError(
                "VideoNystromMMD reference parameters must be frozen")

        self.loss_module = loss_module
        self.current_teacher_attraction_enabled = bool(getattr(
            loss_module, "current_teacher_attraction_enabled", False))
        self.target_local_samples = int(target_local_samples)
        self.gradient_local_samples = int(gradient_local_samples)
        self.partial_gradient_enabled = bool(
            self.gradient_local_samples < self.target_local_samples)
        self.full_student_partial_gradient = bool(
            full_student_partial_gradient)
        if (
            self.full_student_partial_gradient
            and not self.partial_gradient_enabled
        ):
            raise ValueError(
                "full_student_partial_gradient requires "
                "gradient_local_samples < target_local_samples")
        if (
            self.partial_gradient_enabled
            and isinstance(loss_module, MultiLayerVideoNystromMMD)
        ):
            raise TypeError(
                "partial-gradient GradCache currently requires a single "
                "VideoNystromMMD layer")
        self.loss_scale = float(loss_scale)
        self.require_sample_id_alignment = bool(require_sample_id_alignment)
        self._feature_batches: list[torch.Tensor] = []
        self._teacher_feature_batches: list[torch.Tensor] = []
        self._batch_sizes: list[int] = []
        self._gradient_batch_indices: list[int] = []
        self._gradient_batch_sizes: list[int] = []
        self._gradient_sample_ids: list[tuple[str, ...]] = []
        self._selection_started = False
        self._sample_shape: tuple[int, int] | None = None
        self._device: torch.device | None = None
        self._result: FreshVideoGradCacheResult | None = None

    @staticmethod
    def _world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def _rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _all_ranks_true(local_value: bool, device: torch.device) -> bool:
        flag = torch.tensor(
            int(bool(local_value)), device=device, dtype=torch.int32)
        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    @property
    def local_samples(self) -> int:
        return sum(self._batch_sizes)

    @property
    def num_microbatches(self) -> int:
        return len(self._feature_batches)

    @property
    def selected_local_samples(self) -> int:
        return sum(self._gradient_batch_sizes)

    @property
    def ready(self) -> bool:
        return (
            self.local_samples == self.target_local_samples
            and self.selected_local_samples == self.gradient_local_samples
        )

    @property
    def finalized(self) -> bool:
        return self._result is not None

    def add(
        self,
        generated_features: torch.Tensor,
        current_teacher_features: torch.Tensor | None = None,
        *,
        gradient_selected: bool | None = None,
        sample_ids: Sequence[str] | None = None,
        current_teacher_sample_ids: Sequence[str] | None = None,
    ) -> int:
        """Add one detached microbatch and return its zero-based cache index."""
        if self.finalized:
            raise RuntimeError(
                "cannot add features after finalize; commit or discard first")
        if not isinstance(generated_features, torch.Tensor):
            raise TypeError("generated_features must be a torch.Tensor")
        if (
            generated_features.ndim != 3
            or min(generated_features.shape) <= 0
            or not generated_features.is_floating_point()
        ):
            raise ValueError(
                "generated_features must be a non-empty floating [B, N, D] "
                "tensor")
        if not bool(torch.isfinite(generated_features).all().item()):
            raise FloatingPointError(
                "generated_features contains non-finite values")

        if gradient_selected is None:
            if self.partial_gradient_enabled:
                raise TypeError(
                    "gradient_selected must be explicit when partial-gradient "
                    "GradCache is enabled")
            gradient_selected = True
        gradient_selected = bool(gradient_selected)
        if self._selection_started and not gradient_selected:
            raise ValueError(
                "gradient-selected microbatches must form a contiguous suffix")
        if gradient_selected:
            self._selection_started = True

        batch_size = int(generated_features.shape[0])
        normalized_sample_ids: tuple[str, ...] | None = None
        normalized_teacher_ids: tuple[str, ...] | None = None
        if sample_ids is not None:
            normalized_sample_ids = tuple(str(value) for value in sample_ids)
            if len(normalized_sample_ids) != batch_size or any(
                not value for value in normalized_sample_ids
            ):
                raise ValueError(
                    "sample_ids must contain one non-empty ID per student")
        if current_teacher_sample_ids is not None:
            normalized_teacher_ids = tuple(
                str(value) for value in current_teacher_sample_ids)
            if len(normalized_teacher_ids) != batch_size or any(
                not value for value in normalized_teacher_ids
            ):
                raise ValueError(
                    "current_teacher_sample_ids must contain one non-empty ID "
                    "per selected student")
        if self.require_sample_id_alignment and normalized_sample_ids is None:
            raise ValueError(
                "sample_ids are required by this GradCache experiment")

        if self.current_teacher_attraction_enabled and gradient_selected:
            if not isinstance(current_teacher_features, torch.Tensor):
                raise TypeError(
                    "current_teacher_features must be a torch.Tensor for each "
                    "selected student microbatch")
            if (
                current_teacher_features.ndim != 3
                or min(current_teacher_features.shape) <= 0
                or not current_teacher_features.is_floating_point()
            ):
                raise ValueError(
                    "current_teacher_features must be a non-empty floating "
                    "[B, N, D] tensor")
            if tuple(current_teacher_features.shape) != tuple(
                generated_features.shape
            ):
                raise ValueError(
                    "current teacher features must exactly match selected "
                    "student [B,N,D]: "
                    f"{tuple(current_teacher_features.shape)} != "
                    f"{tuple(generated_features.shape)}")
            if current_teacher_features.device != generated_features.device:
                raise ValueError(
                    "generated and current teacher features must share a device")
            if not bool(torch.isfinite(
                current_teacher_features.detach()
            ).all().item()):
                raise FloatingPointError(
                    "current_teacher_features contains non-finite values")
            if self.require_sample_id_alignment and normalized_teacher_ids is None:
                raise ValueError(
                    "current_teacher_sample_ids are required for selected "
                    "students")
            if (
                normalized_teacher_ids is not None
                and normalized_sample_ids != normalized_teacher_ids
            ):
                raise ValueError(
                    "matched teacher sample IDs must exactly equal selected "
                    "student sample IDs in order")
        elif current_teacher_features is not None:
            raise ValueError(
                "current_teacher_features may only be provided for a selected "
                "microbatch while exact current-teacher attraction is enabled")
        elif current_teacher_sample_ids is not None:
            raise ValueError(
                "current_teacher_sample_ids may only be provided for a "
                "selected current-teacher microbatch")

        if self.local_samples + batch_size > self.target_local_samples:
            raise ValueError(
                "fresh feature cache would exceed target_local_samples: "
                f"{self.local_samples} + {batch_size} > "
                f"{self.target_local_samples}")
        if (
            gradient_selected
            and self.selected_local_samples + batch_size
            > self.gradient_local_samples
        ):
            raise ValueError(
                "fresh feature cache would exceed gradient_local_samples: "
                f"{self.selected_local_samples} + {batch_size} > "
                f"{self.gradient_local_samples}")

        sample_shape = (
            int(generated_features.shape[1]),
            int(generated_features.shape[2]),
        )
        if self._sample_shape is None:
            self._sample_shape = sample_shape
            self._device = generated_features.device
        elif (
            sample_shape != self._sample_shape
            or generated_features.device != self._device
        ):
            raise ValueError(
                "all fresh feature batches on a rank must share [N, D] and "
                "device; expected "
                f"{self._sample_shape} on {self._device}, got "
                f"{sample_shape} on {generated_features.device}")

        staged = generated_features.detach().to(dtype=torch.float32).clone()
        self._feature_batches.append(staged)
        batch_index = len(self._feature_batches) - 1
        if gradient_selected:
            self._gradient_batch_indices.append(batch_index)
            self._gradient_batch_sizes.append(batch_size)
            self._gradient_sample_ids.append(
                normalized_sample_ids or tuple("" for _ in range(batch_size)))
        if self.current_teacher_attraction_enabled and gradient_selected:
            staged_teacher = current_teacher_features.detach().to(
                dtype=torch.float32).clone()
            self._teacher_feature_batches.append(staged_teacher)
        self._batch_sizes.append(batch_size)
        return batch_index

    def _validate_distributed_geometry(
        self,
        local_features: torch.Tensor,
        *,
        expected_samples: int,
        label: str,
    ) -> None:
        if self._world_size() == 1:
            if int(local_features.shape[0]) != int(expected_samples):
                raise ValueError(
                    f"{label} expected {expected_samples} local samples, got "
                    f"{int(local_features.shape[0])}")
            return
        signature = torch.tensor(
            [
                int(local_features.shape[0]),
                int(local_features.shape[1]),
                int(local_features.shape[2]),
            ],
            device=local_features.device,
            dtype=torch.long,
        )
        signatures = [
            torch.empty_like(signature) for _ in range(self._world_size())
        ]
        dist.all_gather(signatures, signature)
        expected = (
            int(expected_samples),
            int(local_features.shape[1]),
            int(local_features.shape[2]),
        )
        observed = [
            tuple(int(value) for value in item.tolist())
            for item in signatures
        ]
        if any(item != expected for item in observed):
            raise ValueError(
                f"{label} requires {expected_samples} samples and identical "
                f"[N,D] on every rank; expected {expected}, got {observed}")

    def _validate_ready_consensus(self) -> None:
        """Make incomplete full or gradient suffix counts an all-rank error."""
        expected = (
            self.target_local_samples,
            self.gradient_local_samples,
        )
        if self._world_size() == 1:
            observed = (self.local_samples, self.selected_local_samples)
            if observed != expected:
                raise RuntimeError(
                    "cannot finalize an incomplete fresh feature cache; "
                    f"expected forward/gradient={expected}, got {observed}")
            return
        collective_device = (
            self._device
            if self._device is not None
            else next(self.loss_module.buffers()).device
        )
        local_counts = torch.tensor(
            [self.local_samples, self.selected_local_samples],
            device=collective_device,
            dtype=torch.long,
        )
        counts = [
            torch.empty_like(local_counts) for _ in range(self._world_size())
        ]
        dist.all_gather(counts, local_counts)
        observed = [
            tuple(int(value) for value in item.tolist()) for item in counts
        ]
        if any(value != expected for value in observed):
            observed_forward = [value[0] for value in observed]
            observed_gradient = [value[1] for value in observed]
            raise RuntimeError(
                "cannot finalize incomplete distributed fresh feature "
                f"caches; expected {self.target_local_samples} samples per "
                f"rank, got {observed_forward}; expected gradient "
                f"{self.gradient_local_samples}, got {observed_gradient}")

    def _gather_detached(
        self,
        local_features: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self._world_size() == 1:
            return [local_features.detach()]
        gathered = [
            torch.empty_like(local_features)
            for _ in range(self._world_size())
        ]
        dist.all_gather(gathered, local_features.detach().contiguous())
        return gathered

    def finalize(self) -> FreshVideoGradCacheResult:
        """Compute the global loss and stage suffix-only replay cotangents."""
        if self._result is not None:
            return self._result
        self._validate_ready_consensus()

        local_reference = torch.cat(self._feature_batches, dim=0)
        self._validate_distributed_geometry(
            local_reference,
            expected_samples=self.target_local_samples,
            label="student reference",
        )
        gathered_reference = self._gather_detached(local_reference)

        if self.partial_gradient_enabled:
            selected_batches = [
                self._feature_batches[index]
                for index in self._gradient_batch_indices
            ]
            local_gradient = torch.cat(selected_batches, dim=0)
            self._validate_distributed_geometry(
                local_gradient,
                expected_samples=self.gradient_local_samples,
                label="gradient students",
            )
            gathered_gradient = self._gather_detached(local_gradient)
            local_leaf = local_gradient.detach().requires_grad_(True)
            gathered_gradient[self._rank()] = local_leaf
            global_gradient = torch.cat(gathered_gradient, dim=0)
            global_reference = torch.cat(
                gathered_reference, dim=0).detach()
            gradient_student_indices = None
            if self.full_student_partial_gradient:
                # ``global_gradient`` is rank-major, with each rank
                # contributing its contiguous selected suffix. Build the
                # matching positions in the rank-major full student batch so
                # the objective can inject every selected value back into its
                # original location without reordering either population.
                suffix_start = (
                    self.target_local_samples
                    - self.gradient_local_samples
                )
                gradient_student_indices = torch.cat([
                    torch.arange(
                        rank * self.target_local_samples + suffix_start,
                        (rank + 1) * self.target_local_samples,
                        device=global_reference.device,
                        dtype=torch.long,
                    )
                    for rank in range(self._world_size())
                ])
        else:
            local_leaf = local_reference.detach().requires_grad_(True)
            gathered_reference[self._rank()] = local_leaf
            global_gradient = torch.cat(gathered_reference, dim=0)
            global_reference = None

        global_teacher = None
        if self.current_teacher_attraction_enabled:
            if len(self._teacher_feature_batches) != len(
                self._gradient_batch_indices
            ):
                raise RuntimeError(
                    "matched teacher and selected student GradCache batch "
                    "counts differ")
            local_teacher = torch.cat(
                self._teacher_feature_batches, dim=0)
            if tuple(local_teacher.shape) != tuple(
                local_gradient.shape
                if self.partial_gradient_enabled else local_reference.shape
            ):
                raise RuntimeError(
                    "matched teacher and gradient student shapes differ")
            self._validate_distributed_geometry(
                local_teacher,
                expected_samples=self.gradient_local_samples,
                label="current teacher",
            )
            gathered_teacher = self._gather_detached(local_teacher)
            global_teacher = torch.cat(
                gathered_teacher, dim=0).detach().to(dtype=torch.float32)

        if self.partial_gradient_enabled:
            if isinstance(self.loss_module, MultiLayerVideoNystromMMD):
                raise TypeError(
                    "partial-gradient GradCache currently requires a single "
                    "VideoNystromMMD layer")
            partial_loss_kwargs = {
                "gradient_students": global_gradient,
                "student_reference": global_reference,
                "current_teacher": global_teacher,
            }
            if self.full_student_partial_gradient:
                partial_loss_kwargs["gradient_student_indices"] = (
                    gradient_student_indices)
            raw_loss, metadata = self.loss_module(**partial_loss_kwargs)
        elif self.current_teacher_attraction_enabled:
            raw_loss, metadata = self.loss_module(
                global_gradient, current_teacher=global_teacher)
        else:
            raw_loss, metadata = self.loss_module(global_gradient)
        if not isinstance(raw_loss, torch.Tensor) or raw_loss.ndim != 0:
            raise TypeError(
                "VideoNystromMMD must return a scalar tensor loss")
        if not self._all_ranks_true(
            bool(torch.isfinite(raw_loss.detach()).item()),
            raw_loss.device,
        ):
            raise FloatingPointError(
                "VideoNystromMMD returned a non-finite loss on at least one "
                "rank")
        optimized_loss = raw_loss * self.loss_scale

        (local_cotangent,) = torch.autograd.grad(
            optimized_loss,
            local_leaf,
            retain_graph=False,
            create_graph=False,
        )
        local_cotangent = local_cotangent.detach()
        if not self._all_ranks_true(
            bool(torch.isfinite(local_cotangent).all().item()),
            local_cotangent.device,
        ):
            raise FloatingPointError(
                "VideoNystromMMD returned a non-finite feature cotangent on "
                "at least one rank")

        world_size = self._world_size()
        local_parts = tuple(
            value.contiguous()
            for value in local_cotangent.split(
                self._gradient_batch_sizes, dim=0)
        )
        fsdp_parts = tuple(
            (value * float(world_size)).contiguous()
            for value in local_parts
        )
        global_samples = self.target_local_samples * world_size
        gradient_global_samples = self.gradient_local_samples * world_size
        teacher_local_samples = (
            self.gradient_local_samples
            if self.current_teacher_attraction_enabled else 0
        )
        teacher_global_samples = teacher_local_samples * world_size
        kss_denominator = global_samples * (
            global_samples
            if self.full_student_partial_gradient
            else gradient_global_samples
        )
        detached_metadata = dict(_detach_metadata(metadata))
        detached_metadata.update({
            "raw_loss": raw_loss.detach(),
            "optimized_loss": optimized_loss.detach(),
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
                gradient_global_samples, dtype=torch.long),
            "gradcache_kss_denominator": raw_loss.new_tensor(
                kss_denominator, dtype=torch.long),
            "gradcache_partial_gradient_enabled": raw_loss.new_tensor(
                int(self.partial_gradient_enabled), dtype=torch.long),
            "gradcache_full_student_partial_gradient": raw_loss.new_tensor(
                int(self.full_student_partial_gradient), dtype=torch.long),
            "gradcache_world_size": raw_loss.new_tensor(
                world_size, dtype=torch.long),
            "gradcache_fsdp_cotangent_scale": raw_loss.new_tensor(
                float(world_size)),
        })
        detached_metadata.update({
            "gradcache_current_teacher_local_samples": raw_loss.new_tensor(
                teacher_local_samples, dtype=torch.long),
            "gradcache_current_teacher_global_samples": raw_loss.new_tensor(
                teacher_global_samples, dtype=torch.long),
        })
        core_global_samples = detached_metadata.get("global_samples")
        if (
            core_global_samples is not None
            and int(torch.as_tensor(core_global_samples).item())
            != global_samples
        ):
            raise ValueError(
                "VideoNystromMMD global_samples metadata disagrees with the "
                f"fresh GradCache: {int(torch.as_tensor(core_global_samples))} "
                f"!= {global_samples}")
        core_gradient_samples = detached_metadata.get(
            "gradient_student_samples")
        if (
            self.partial_gradient_enabled
            and core_gradient_samples is not None
            and int(torch.as_tensor(core_gradient_samples).item())
            != gradient_global_samples
        ):
            raise ValueError(
                "VideoNystromMMD gradient_student_samples metadata disagrees "
                "with the fresh GradCache")
        core_repulsion_denominator = detached_metadata.get(
            "repulsion_denominator")
        if (
            self.partial_gradient_enabled
            and core_repulsion_denominator is not None
            and int(torch.as_tensor(core_repulsion_denominator).item())
            != kss_denominator
        ):
            raise ValueError(
                "VideoNystromMMD Kss denominator disagrees with the fresh "
                f"GradCache: expected {kss_denominator}")

        self._result = FreshVideoGradCacheResult(
            loss=raw_loss.detach(),
            optimized_loss=optimized_loss.detach(),
            metadata=detached_metadata,
            local_cotangents=local_parts,
            fsdp_cotangents=fsdp_parts,
            replay_batch_indices=tuple(self._gradient_batch_indices),
            replay_sample_ids=tuple(self._gradient_sample_ids),
            local_samples=self.target_local_samples,
            global_samples=global_samples,
            gradient_local_samples=self.gradient_local_samples,
            gradient_global_samples=gradient_global_samples,
            current_teacher_local_samples=teacher_local_samples,
            current_teacher_global_samples=teacher_global_samples,
            world_size=world_size,
        )
        return self._result

    def _clear(self) -> None:
        self._feature_batches.clear()
        self._teacher_feature_batches.clear()
        self._batch_sizes.clear()
        self._gradient_batch_indices.clear()
        self._gradient_batch_sizes.clear()
        self._gradient_sample_ids.clear()
        self._selection_started = False
        self._sample_shape = None
        self._device = None
        self._result = None

    def offload_finalized_to_cpu(self) -> FreshVideoGradCacheResult:
        """Move finalized replay storage to CPU without committing the update.

        ``finalize`` has already differentiated the complete global MMD and
        produced detached replay cotangents. Keeping the fresh feature copies
        and both cotangent views on CUDA after that point only increases the
        generator-replay peak. This operation deliberately preserves all
        transactional counters and the finalized result, so ``commit`` and
        ``discard`` retain their original success/failure semantics.
        """

        if self._result is None:
            raise RuntimeError(
                "cannot offload a fresh feature cache before finalize"
            )
        self._feature_batches = [
            _detach_to_cpu(value) for value in self._feature_batches
        ]
        self._teacher_feature_batches = [
            _detach_to_cpu(value) for value in self._teacher_feature_batches
        ]
        result = self._result
        self._result = FreshVideoGradCacheResult(
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

    def commit(self) -> None:
        """Clear a finalized cache after a successful optimizer update."""
        if self._result is None:
            raise RuntimeError(
                "cannot commit a fresh feature cache before finalize")
        self._clear()

    def discard(self) -> None:
        """Clear collected/finalized data after a skipped or failed update."""
        self._clear()

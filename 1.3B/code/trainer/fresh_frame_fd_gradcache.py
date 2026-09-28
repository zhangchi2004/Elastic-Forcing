"""Update-wide transactional GradCache for single-frame SIM FD.

One optimizer update collects fresh frame features from every video
microbatch. ``finalize`` evaluates the legacy normalized FD objectives once on
the global current population plus each representation's detached FIFO
history. Only current local features are leaves. Returned FSDP cotangents are
multiplied by ``world_size`` so averaged parameter gradients recover the
derivative of the single global objective.

``finalize_fidel`` exposes each representation separately: its raw
FIFO-plus-current objective/cotangent and its current-only raw progress score.
Those cotangents are intentionally unscaled so the caller can first apply
detached FiDEL weights and only then compensate FSDP averaging.

Queue mutations are staged: current global features enter the FIFO only after
``commit``. ``discard`` leaves all persistent statistics unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
from torch import nn

from model.frame_fd_head import FrameFDExtraction, FrameFDHead


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    if (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


@dataclass(frozen=True)
class FreshFrameFDGradCacheResult:
    loss: torch.Tensor
    fd_values: Mapping[str, torch.Tensor]
    local_feature_cotangents: tuple[Mapping[str, torch.Tensor], ...]
    fsdp_feature_cotangents: tuple[Mapping[str, torch.Tensor], ...]
    frame_indices: tuple[torch.Tensor, ...]
    metadata: Mapping[str, Any]
    local_videos: int
    global_videos: int
    frames_per_video: int
    world_size: int


@dataclass(frozen=True)
class FreshFrameFDFiDELResult:
    """Per-representation raw FiDEL inputs for one fresh global update.

    ``raw_fd_values`` and ``local_raw_feature_cotangents`` come from the
    differentiable FIFO-history-plus-current objective. ``fresh_fd_scores``
    score only the fixed-size current global frame population and therefore do
    not drift merely because the FIFO is filling.

    The cotangents are deliberately local and *unscaled*: no representation
    weight, normalization, static averaging, FiDEL coefficient, or FSDP
    ``world_size`` compensation has been applied.  The caller must apply its
    detached FiDEL coefficient and then multiply by ``world_size`` before
    replay under FSDP.
    """

    raw_fd_values: Mapping[str, torch.Tensor]
    fresh_fd_scores: Mapping[str, torch.Tensor]
    local_raw_feature_cotangents: tuple[
        Mapping[str, torch.Tensor], ...
    ]
    frame_indices: tuple[torch.Tensor, ...]
    metadata: Mapping[str, Any]
    local_videos: int
    global_videos: int
    frames_per_video: int
    world_size: int


@dataclass(frozen=True)
class _PreparedFIFOUpdate:
    judge_name: str
    features: torch.Tensor
    feature_sum: Optional[torch.Tensor]
    feature_outer: Optional[torch.Tensor]


class FreshFrameFDGradCache:
    """Collect one global fresh frame population and stage its FIFO update."""

    def __init__(
        self,
        head: FrameFDHead,
        *,
        target_local_videos: int = 16,
    ) -> None:
        if not isinstance(head, FrameFDHead):
            raise TypeError("head must be a FrameFDHead")
        if int(target_local_videos) <= 0:
            raise ValueError("target_local_videos must be positive")
        self.head = head
        self.target_local_videos = int(target_local_videos)
        self._extractions: list[FrameFDExtraction] = []
        self._batch_sizes: list[int] = []
        self._frames_per_video: Optional[int] = None
        self._device: Optional[torch.device] = None
        self._result: Optional[
            FreshFrameFDGradCacheResult | FreshFrameFDFiDELResult
        ] = None
        self._result_mode: Optional[str] = None
        self._staged_global_features: Optional[
            dict[str, torch.Tensor]
        ] = None
        self._prepared_fifo_updates: Optional[
            tuple[_PreparedFIFOUpdate, ...]
        ] = None

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

    @property
    def local_videos(self) -> int:
        return sum(self._batch_sizes)

    @property
    def num_microbatches(self) -> int:
        return len(self._extractions)

    @property
    def ready(self) -> bool:
        return self.local_videos == self.target_local_videos

    @property
    def finalized(self) -> bool:
        return self._result is not None

    def add(self, extraction: FrameFDExtraction) -> int:
        if self.finalized:
            raise RuntimeError(
                "cannot add frame features after finalize; commit or discard")
        if not isinstance(extraction, FrameFDExtraction):
            raise TypeError("extraction must be a FrameFDExtraction")
        if self.local_videos + extraction.batch_size > self.target_local_videos:
            raise ValueError(
                "frame FD cache would exceed target_local_videos: "
                f"{self.local_videos} + {extraction.batch_size} > "
                f"{self.target_local_videos}")
        if extraction.frame_indices.shape != (
            extraction.batch_size,
            extraction.frames_per_video,
        ):
            raise ValueError("extraction frame_indices metadata is inconsistent")
        expected_names = set(self.head.representation_names)
        if set(extraction.features) != expected_names:
            raise ValueError(
                "extraction representations do not match FrameFDHead")
        device = extraction.frame_indices.device
        rows = extraction.batch_size * extraction.frames_per_video
        cloned_features = {}
        for judge in self.head.judges:
            value = extraction.features[judge.name]
            if (
                value.ndim != 2
                or value.shape[0] != rows
                or value.shape[1] != judge.queue.feat_dim
                or value.device != device
                or not value.is_floating_point()
            ):
                raise ValueError(
                    f"invalid {judge.name} extraction shape/device: "
                    f"{tuple(value.shape)} on {value.device}")
            if not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(
                    f"{judge.name} extraction contains non-finite values")
            cloned_features[judge.name] = (
                value.detach().float().contiguous().clone())

        if self._frames_per_video is None:
            self._frames_per_video = extraction.frames_per_video
            self._device = device
        elif (
            extraction.frames_per_video != self._frames_per_video
            or device != self._device
        ):
            raise ValueError(
                "all update-local frame extractions must share K and device")

        staged = FrameFDExtraction(
            frame_indices=extraction.frame_indices.detach().clone(),
            features=cloned_features,
            batch_size=extraction.batch_size,
            frames_per_video=extraction.frames_per_video,
            total_pixel_frames=extraction.total_pixel_frames,
        )
        self._extractions.append(staged)
        self._batch_sizes.append(extraction.batch_size)
        return len(self._extractions) - 1

    def _validate_ready_consensus(self) -> None:
        device = self._device
        if device is None:
            device = next(self.head.judges[0].buffers()).device
        count = torch.tensor(
            self.local_videos, device=device, dtype=torch.long)
        if self._world_size() == 1:
            observed = [int(count)]
        else:
            gathered = [
                torch.empty_like(count) for _ in range(self._world_size())]
            dist.all_gather(gathered, count)
            observed = [int(value) for value in gathered]
        if any(value != self.target_local_videos for value in observed):
            raise RuntimeError(
                "cannot finalize incomplete distributed frame FD caches; "
                f"expected {self.target_local_videos} videos per rank, got "
                f"{observed}")

    def _validate_geometry(self, local_rows: int) -> None:
        if self._world_size() == 1:
            return
        signature = torch.tensor(
            [self.local_videos, self._frames_per_video, local_rows],
            device=self._device,
            dtype=torch.long,
        )
        gathered = [
            torch.empty_like(signature) for _ in range(self._world_size())]
        dist.all_gather(gathered, signature)
        expected = (
            self.target_local_videos,
            int(self._frames_per_video),
            self.target_local_videos * int(self._frames_per_video),
        )
        observed = [tuple(int(item) for item in value.tolist())
                    for value in gathered]
        if any(value != expected for value in observed):
            raise ValueError(
                f"frame FD distributed geometry mismatch: expected "
                f"{expected}, got {observed}")

    def _gather(self, local: torch.Tensor) -> list[torch.Tensor]:
        if self._world_size() == 1:
            return [local.detach()]
        gathered = [
            torch.empty_like(local) for _ in range(self._world_size())]
        dist.all_gather(gathered, local.detach().contiguous())
        return gathered

    def finalize(self) -> FreshFrameFDGradCacheResult:
        if self._result is not None:
            if self._result_mode != "legacy_normalized":
                raise RuntimeError(
                    "frame FD cache was already finalized in FiDEL raw mode")
            return self._result
        self._validate_ready_consensus()
        frames_per_video = int(self._frames_per_video)
        rows_per_microbatch = [
            batch_size * frames_per_video
            for batch_size in self._batch_sizes
        ]
        assembly_error = None
        try:
            local_by_name = {
                name: torch.cat(
                    [item.features[name] for item in self._extractions],
                    dim=0,
                )
                for name in self.head.representation_names
            }
        except BaseException as exc:
            assembly_error = exc
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.empty_cache()
        if not _all_ranks_true(
            assembly_error is None, self._device
        ):
            message = (
                f": {type(assembly_error).__name__}: {assembly_error}"
                if assembly_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "frame FD local feature assembly failed" + message
            ) from assembly_error
        local_rows = next(iter(local_by_name.values())).shape[0]
        self._validate_geometry(local_rows)

        global_by_name: dict[str, torch.Tensor] = {}
        loss_terms = []
        fd_values = {}
        local_gradients: dict[str, torch.Tensor] = {}
        representation_scale = (
            1.0 / len(self.head.judges)
            if self.head.average_representations
            else 1.0
        )
        for judge in self.head.judges:
            local = local_by_name[judge.name]
            gathered = self._gather(local)
            judge_error = None
            try:
                leaf = local.detach().requires_grad_(True)
                gathered[self._rank()] = leaf
                global_features = torch.cat(gathered, dim=0)
                judge_loss, fd = judge.compute(
                    global_features, self.head.norm_eps)
                scaled_loss = judge_loss * representation_scale
                if not bool(torch.isfinite(scaled_loss.detach()).item()):
                    raise FloatingPointError(
                        f"frame FD {judge.name} returned a non-finite loss")
                (gradient,) = torch.autograd.grad(
                    scaled_loss,
                    leaf,
                    retain_graph=False,
                    create_graph=False,
                )
                if not bool(torch.isfinite(gradient).all().item()):
                    raise FloatingPointError(
                        f"frame FD {judge.name} returned a non-finite "
                        "feature cotangent")
                staged_global = global_features.detach().clone()
            except BaseException as exc:
                judge_error = exc
                if local.device.type == "cuda":
                    torch.cuda.empty_cache()
            if not _all_ranks_true(
                judge_error is None, local.device
            ):
                message = (
                    f": {type(judge_error).__name__}: {judge_error}"
                    if judge_error is not None
                    else " on another rank"
                )
                raise RuntimeError(
                    f"frame FD {judge.name} finalize failed" + message
                ) from judge_error
            global_by_name[judge.name] = staged_global
            local_gradients[judge.name] = gradient.detach()
            loss_terms.append(scaled_loss.detach())
            fd_values[judge.name] = fd.detach()

        loss = torch.stack(loss_terms).sum()
        if not _all_ranks_true(
            bool(torch.isfinite(loss.detach()).item()), loss.device
        ):
            raise FloatingPointError(
                "frame FD returned a non-finite loss on at least one rank")

        split_error = None
        try:
            local_parts = []
            fsdp_parts = []
            world_size = self._world_size()
            offsets = [0]
            for rows in rows_per_microbatch:
                offsets.append(offsets[-1] + rows)
            for start, end in zip(offsets[:-1], offsets[1:]):
                local_part = {
                    name: local_gradients[name][start:end].contiguous()
                    for name in self.head.representation_names
                }
                local_parts.append(local_part)
                fsdp_parts.append({
                    name: (value * float(world_size)).contiguous()
                    for name, value in local_part.items()
                })
        except BaseException as exc:
            split_error = exc
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
        if not _all_ranks_true(split_error is None, self._device):
            message = (
                f": {type(split_error).__name__}: {split_error}"
                if split_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "frame FD cotangent split failed" + message
            ) from split_error

        global_videos = self.target_local_videos * world_size
        metadata = {
            "frame_fd_loss": loss.detach(),
            "frame_fd_local_videos": loss.new_tensor(
                self.target_local_videos, dtype=torch.long),
            "frame_fd_global_videos": loss.new_tensor(
                global_videos, dtype=torch.long),
            "frame_fd_frames_per_video": loss.new_tensor(
                frames_per_video, dtype=torch.long),
            "frame_fd_global_frames": loss.new_tensor(
                global_videos * frames_per_video, dtype=torch.long),
            "frame_fd_world_size": loss.new_tensor(
                world_size, dtype=torch.long),
            "frame_fd_fsdp_cotangent_scale": loss.new_tensor(
                float(world_size)),
            "frame_fd_queue_filled_before": {
                judge.name: int(judge.queue.filled.item())
                for judge in self.head.judges
            },
        }
        for name, value in fd_values.items():
            metadata[f"frame_fd_{name}"] = value

        self._staged_global_features = global_by_name
        self._result_mode = "legacy_normalized"
        self._result = FreshFrameFDGradCacheResult(
            loss=loss.detach(),
            fd_values=fd_values,
            local_feature_cotangents=tuple(local_parts),
            fsdp_feature_cotangents=tuple(fsdp_parts),
            frame_indices=tuple(
                item.frame_indices.detach().clone()
                for item in self._extractions
            ),
            metadata=metadata,
            local_videos=self.target_local_videos,
            global_videos=global_videos,
            frames_per_video=frames_per_video,
            world_size=world_size,
        )
        return self._result

    def finalize_fidel(self) -> FreshFrameFDFiDELResult:
        """Finalize raw per-encoder objectives for two-level FiDEL.

        Unlike :meth:`finalize`, this method does not use ``FDJudge.weight``,
        ``norm_eps``, representation averaging, or FSDP scaling.  It returns
        one raw feature cotangent per representation and microbatch, leaving
        all detached FiDEL weighting to the caller.
        """
        if self._result is not None:
            if self._result_mode != "fidel_raw":
                raise RuntimeError(
                    "frame FD cache was already finalized in legacy mode")
            return self._result
        self._validate_ready_consensus()
        frames_per_video = int(self._frames_per_video)
        rows_per_microbatch = [
            batch_size * frames_per_video
            for batch_size in self._batch_sizes
        ]
        assembly_error = None
        try:
            local_by_name = {
                name: torch.cat(
                    [item.features[name] for item in self._extractions],
                    dim=0,
                )
                for name in self.head.representation_names
            }
        except BaseException as exc:
            assembly_error = exc
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.empty_cache()
        if not _all_ranks_true(assembly_error is None, self._device):
            message = (
                f": {type(assembly_error).__name__}: {assembly_error}"
                if assembly_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "frame FD local feature assembly failed" + message
            ) from assembly_error
        local_rows = next(iter(local_by_name.values())).shape[0]
        self._validate_geometry(local_rows)

        global_by_name: dict[str, torch.Tensor] = {}
        raw_fd_values: dict[str, torch.Tensor] = {}
        fresh_fd_scores: dict[str, torch.Tensor] = {}
        local_gradients: dict[str, torch.Tensor] = {}
        for judge in self.head.judges:
            local = local_by_name[judge.name]
            gathered = self._gather(local)
            judge_error = None
            try:
                leaf = local.detach().requires_grad_(True)
                gathered[self._rank()] = leaf
                global_features = torch.cat(gathered, dim=0)
                raw_fd = judge.compute_raw(
                    global_features,
                    include_queue=True,
                    retain_full_queue_history=True,
                )
                # This score has a fixed sample population and is only used by
                # the detached FiDEL controller.  Keep it outside the raw loss
                # graph so it cannot accidentally contribute a second
                # feature-gradient path.
                fresh_score = judge.compute_raw(
                    global_features.detach(), include_queue=False)
                if (
                    not bool(torch.isfinite(raw_fd.detach()).item())
                    or not bool(torch.isfinite(fresh_score.detach()).item())
                ):
                    raise FloatingPointError(
                        f"frame FD {judge.name} returned a non-finite "
                        "raw FiDEL value")
                (gradient,) = torch.autograd.grad(
                    raw_fd,
                    leaf,
                    retain_graph=False,
                    create_graph=False,
                )
                if not bool(torch.isfinite(gradient).all().item()):
                    raise FloatingPointError(
                        f"frame FD {judge.name} returned a non-finite raw "
                        "feature cotangent")
                staged_global = global_features.detach().clone()
            except BaseException as exc:
                judge_error = exc
                if local.device.type == "cuda":
                    torch.cuda.empty_cache()
            if not _all_ranks_true(judge_error is None, local.device):
                message = (
                    f": {type(judge_error).__name__}: {judge_error}"
                    if judge_error is not None
                    else " on another rank"
                )
                raise RuntimeError(
                    f"frame FD {judge.name} FiDEL finalize failed" + message
                ) from judge_error
            global_by_name[judge.name] = staged_global
            local_gradients[judge.name] = gradient.detach()
            raw_fd_values[judge.name] = raw_fd.detach()
            fresh_fd_scores[judge.name] = fresh_score.detach()

        split_error = None
        try:
            local_parts = []
            offsets = [0]
            for rows in rows_per_microbatch:
                offsets.append(offsets[-1] + rows)
            for start, end in zip(offsets[:-1], offsets[1:]):
                local_parts.append({
                    name: local_gradients[name][start:end].contiguous()
                    for name in self.head.representation_names
                })
        except BaseException as exc:
            split_error = exc
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
        if not _all_ranks_true(split_error is None, self._device):
            message = (
                f": {type(split_error).__name__}: {split_error}"
                if split_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "frame FD raw FiDEL cotangent split failed" + message
            ) from split_error

        world_size = self._world_size()
        global_videos = self.target_local_videos * world_size
        device_value = next(iter(raw_fd_values.values()))
        metadata = {
            "frame_fd_fidel_mode": "raw_per_representation",
            "frame_fd_local_videos": device_value.new_tensor(
                self.target_local_videos, dtype=torch.long),
            "frame_fd_global_videos": device_value.new_tensor(
                global_videos, dtype=torch.long),
            "frame_fd_frames_per_video": device_value.new_tensor(
                frames_per_video, dtype=torch.long),
            "frame_fd_global_frames": device_value.new_tensor(
                global_videos * frames_per_video, dtype=torch.long),
            "frame_fd_world_size": device_value.new_tensor(
                world_size, dtype=torch.long),
            "frame_fd_raw_cotangent_scale": device_value.new_tensor(1.0),
            "frame_fd_raw_cotangents_are_fsdp_scaled": False,
            "frame_fd_queue_filled_before": {
                judge.name: int(judge.queue.filled.item())
                for judge in self.head.judges
            },
        }
        for name in self.head.representation_names:
            metadata[f"frame_fd_{name}_raw_loss"] = raw_fd_values[name]
            metadata[
                f"frame_fd_{name}_fresh_score"
            ] = fresh_fd_scores[name]

        self._staged_global_features = global_by_name
        self._result_mode = "fidel_raw"
        self._result = FreshFrameFDFiDELResult(
            raw_fd_values=raw_fd_values,
            fresh_fd_scores=fresh_fd_scores,
            local_raw_feature_cotangents=tuple(local_parts),
            frame_indices=tuple(
                item.frame_indices.detach().clone()
                for item in self._extractions
            ),
            metadata=metadata,
            local_videos=self.target_local_videos,
            global_videos=global_videos,
            frames_per_video=frames_per_video,
            world_size=world_size,
        )
        return self._result

    def _clear(self) -> None:
        self._extractions.clear()
        self._batch_sizes.clear()
        self._frames_per_video = None
        self._device = None
        self._result = None
        self._result_mode = None
        self._staged_global_features = None
        self._prepared_fifo_updates = None

    @staticmethod
    @torch.no_grad()
    def _prepare_fifo_update(judge, features: torch.Tensor) -> _PreparedFIFOUpdate:
        """Do all allocation/fallible math before mutating a FIFO."""
        queue = judge.queue
        values = features.detach().float()
        if queue.size <= 0 or not queue.online_accum:
            return _PreparedFIFOUpdate(
                judge.name, values, None, None)

        values_d = values.double()
        filled = int(queue.filled.item())
        if values.shape[0] >= queue.size:
            kept = values_d[-queue.size:]
            feature_sum = kept.sum(0)
            feature_outer = kept.T @ kept
        else:
            # Accumulators describe every currently filled slot. Determine the
            # exact rows the ring write will evict, then stage the replacement
            # moments without touching either the ring or its moments.
            evicted_count = max(0, filled + values.shape[0] - queue.size)
            if evicted_count == 0:
                evicted = values_d.new_empty((0, values.shape[1]))
            elif filled < queue.size:
                # A not-yet-full ring is contiguous and its oldest row is zero.
                evicted = queue.feats[:evicted_count].detach().double()
            else:
                evicted = queue._get_evicted(evicted_count).detach().double()
            feature_sum = (
                queue.feat_sum.detach()
                - evicted.sum(0)
                + values_d.sum(0)
            )
            feature_outer = (
                queue.feat_outer.detach()
                - evicted.T @ evicted
                + values_d.T @ values_d
            )
        return _PreparedFIFOUpdate(
            judge.name, values, feature_sum, feature_outer)

    @staticmethod
    @torch.no_grad()
    def _apply_fifo_update(judge, update: _PreparedFIFOUpdate) -> None:
        """Allocation-free publication of one already validated update."""
        queue = judge.queue
        values = update.features
        if queue.size <= 0:
            return
        if values.shape[0] >= queue.size:
            queue.feats[:queue.size].copy_(values[-queue.size:])
            queue.ptr.zero_()
            queue.filled.fill_(queue.size)
        else:
            count = int(values.shape[0])
            pointer = queue.pointer
            if pointer + count <= queue.size:
                queue.feats[pointer:pointer + count].copy_(values)
            else:
                first = queue.size - pointer
                queue.feats[pointer:].copy_(values[:first])
                queue.feats[:count - first].copy_(values[first:])
            queue.ptr.fill_((pointer + count) % queue.size)
            queue.filled.fill_(
                min(queue.size, int(queue.filled.item()) + count))
        if queue.online_accum:
            queue.feat_sum.copy_(update.feature_sum)
            queue.feat_outer.copy_(update.feature_outer)
        queue.initialized.fill_(True)

    def prepare_commit(self) -> None:
        """Perform every fallible FIFO calculation before optimizer mutation."""
        if self._result is None or self._staged_global_features is None:
            raise RuntimeError(
                "cannot prepare frame FD commit before finalize")
        if self._prepared_fifo_updates is not None:
            return
        # Validate the complete transaction before the first mutation.
        for judge in self.head.judges:
            value = self._staged_global_features[judge.name]
            if (
                judge.queue.ema_stats
                or value.ndim != 2
                or value.shape[1] != judge.queue.feat_dim
                or not bool(torch.isfinite(value).all().item())
            ):
                raise RuntimeError(
                    f"invalid staged FIFO update for {judge.name}")
        # Expensive moment products for all representations are prepared before
        # the first queue mutation. Thus an OOM/numerical failure leaves every
        # persistent queue untouched and the caller can still discard.
        self._prepared_fifo_updates = tuple(
            self._prepare_fifo_update(
                judge, self._staged_global_features[judge.name])
            for judge in self.head.judges
        )

    def commit(self) -> None:
        """Publish a prevalidated FIFO transaction after optimizer success."""
        self.prepare_commit()
        for judge, update in zip(
            self.head.judges, self._prepared_fifo_updates
        ):
            self._apply_fifo_update(judge, update)
        self._clear()

    def discard(self) -> None:
        """Drop all update-local features without changing any FIFO."""
        self._clear()


@dataclass(frozen=True)
class GradNormBalanceStage:
    weight: float
    mmd_rms: float
    fd_rms: float
    next_mmd_rms_ema: float
    next_fd_rms_ema: float


class GradNormEMABalancer(nn.Module):
    """Transactional previous-update EMA balance for two latent gradients.

    ``current_weight`` is read before replay. Calls to ``stage`` only collect
    detached squared norms for the current update; ``finalize_stage`` computes
    the next EMA but does not publish it. Only ``commit`` changes persistent
    state, while ``discard`` leaves the previous update's weight intact.
    """

    def __init__(
        self,
        *,
        fd_weight: float = 1.0,
        beta: float = 0.9,
        minimum: float = 0.1,
        maximum: float = 10.0,
        epsilon: float = 1.0e-12,
    ) -> None:
        super().__init__()
        values = (fd_weight, beta, minimum, maximum, epsilon)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("grad-norm balance values must be finite")
        if fd_weight < 0 or not 0 <= beta < 1:
            raise ValueError("fd_weight must be non-negative and beta in [0,1)")
        if minimum <= 0 or maximum < minimum or epsilon <= 0:
            raise ValueError(
                "need 0 < minimum <= maximum and epsilon > 0")
        self.fd_weight = float(fd_weight)
        self.beta = float(beta)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.epsilon = float(epsilon)
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("mmd_rms_ema", torch.zeros((), dtype=torch.float64))
        self.register_buffer("fd_rms_ema", torch.zeros((), dtype=torch.float64))
        self.register_buffer(
            "committed_updates", torch.zeros((), dtype=torch.long))
        self._sum_squares: Optional[torch.Tensor] = None
        self._stage_result: Optional[GradNormBalanceStage] = None

    @property
    def staged(self) -> bool:
        return self._sum_squares is not None

    @property
    def stage_finalized(self) -> bool:
        return self._stage_result is not None

    def current_weight(self) -> float:
        if not bool(self.initialized.item()):
            return self.fd_weight
        ratio = float(self.mmd_rms_ema) / (
            float(self.fd_rms_ema) + self.epsilon)
        ratio = min(self.maximum, max(self.minimum, ratio))
        return self.fd_weight * ratio

    @torch.no_grad()
    def stage(
        self,
        mmd_gradient: torch.Tensor,
        fd_gradient: torch.Tensor,
    ) -> float:
        """Accumulate one replay microbatch and return its fixed update weight."""
        if self._stage_result is not None:
            raise RuntimeError(
                "cannot add gradients after finalize_stage; commit or discard")
        if (
            not isinstance(mmd_gradient, torch.Tensor)
            or not isinstance(fd_gradient, torch.Tensor)
            or mmd_gradient.numel() != fd_gradient.numel()
            or mmd_gradient.numel() == 0
        ):
            raise ValueError(
                "MMD and FD gradients must be non-empty tensors with equal size")
        if (
            not bool(torch.isfinite(mmd_gradient).all().item())
            or not bool(torch.isfinite(fd_gradient).all().item())
        ):
            raise FloatingPointError(
                "cannot balance non-finite MMD/FD gradients")
        stats = torch.stack([
            mmd_gradient.detach().double().square().sum(),
            fd_gradient.detach().double().square().sum(),
            torch.tensor(
                float(mmd_gradient.numel()),
                device=mmd_gradient.device,
                dtype=torch.float64,
            ),
        ])
        if self._sum_squares is None:
            self._sum_squares = stats
        else:
            if stats.device != self._sum_squares.device:
                raise ValueError(
                    "all staged gradients must live on the same device")
            self._sum_squares.add_(stats)
        return self.current_weight()

    @torch.no_grad()
    def finalize_stage(self) -> GradNormBalanceStage:
        if self._stage_result is not None:
            return self._stage_result
        if self._sum_squares is None:
            raise RuntimeError("cannot finalize an empty grad-norm stage")
        totals = self._sum_squares.clone()
        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        count = totals[2].clamp_min(1.0)
        mmd_rms = (totals[0] / count).sqrt()
        fd_rms = (totals[1] / count).sqrt()
        if not bool(torch.isfinite(torch.stack([mmd_rms, fd_rms])).all()):
            raise FloatingPointError("non-finite global gradient RMS")
        if bool(self.initialized.item()):
            next_mmd = (
                self.beta * self.mmd_rms_ema.to(mmd_rms.device)
                + (1.0 - self.beta) * mmd_rms
            )
            next_fd = (
                self.beta * self.fd_rms_ema.to(fd_rms.device)
                + (1.0 - self.beta) * fd_rms
            )
        else:
            next_mmd = mmd_rms
            next_fd = fd_rms
        self._stage_result = GradNormBalanceStage(
            weight=self.current_weight(),
            mmd_rms=float(mmd_rms),
            fd_rms=float(fd_rms),
            next_mmd_rms_ema=float(next_mmd),
            next_fd_rms_ema=float(next_fd),
        )
        return self._stage_result

    @torch.no_grad()
    def commit(self) -> GradNormBalanceStage:
        result = self.finalize_stage()
        self.mmd_rms_ema.fill_(result.next_mmd_rms_ema)
        self.fd_rms_ema.fill_(result.next_fd_rms_ema)
        self.initialized.fill_(True)
        self.committed_updates.add_(1)
        self._sum_squares = None
        self._stage_result = None
        return result

    def discard(self) -> None:
        self._sum_squares = None
        self._stage_result = None

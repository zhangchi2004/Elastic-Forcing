"""Elastic Forcing: match generated videos in frozen encoder feature spaces.

Read the active path in this order:
  collect_rollout_features -> hybrid MMD -> prepare_latent_gradients
  -> replay_generator_gradients.

The encoders and VAE are frozen, but their input derivatives connect the
feature-space objective to the generator. Modules/attributes used in saved
state dictionaries keep their original names for checkpoint compatibility.
See docs/paper-to-code.md for the paper mapping and estimator conventions.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
import random
import time
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from losses.multilayer_video_nystrom_mmd import (
    MultiLayerVideoNystromMMD,
)
from losses.wan_kv_mmd import EMARBFMMD
from model.fd import FD
from utils.active_mmd_branches import active_mmd_branches
from video_fd.encoders.vjepa2_dense_metadata import (
    canonical_vjepa2_dense_feature_metadata,
    uniform_dense_token_indices,
)
from video_fd.encoders.vjepa2_temporal_sampling import (
    stateless_temporal_window_start,
)
from video_fd.replay.replay_video_encoder_vjp import replay_decoder_vjp


def _canonicalize_shared_asset_paths(value):
    """Normalize the local and shared mount names recursively."""
    local_root = "/path/to/local-assets"
    logical_root = "/path/to/elastic-forcing-assets"
    if isinstance(value, str):
        if value == local_root or value.startswith(local_root + "/"):
            return logical_root + value[len(local_root):]
        return value
    if isinstance(value, Mapping):
        return {
            key: _canonicalize_shared_asset_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_canonicalize_shared_asset_paths(item) for item in value)
    if isinstance(value, list):
        return [_canonicalize_shared_asset_paths(item) for item in value]
    return value


def _accumulate_fp32_pixel_cotangent(
    accumulated: Optional[torch.Tensor],
    value: torch.Tensor,
    expected_shape: Optional[tuple[int, ...]],
    *,
    branch_name: str,
) -> tuple[torch.Tensor, tuple[int, ...], torch.Tensor]:
    """Accumulate one branch immediately and retain only its scalar norm."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{branch_name} pixel cotangent must be floating")
    detached = value.detach().float()
    shape = tuple(detached.shape)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(
            "triple encoder pixel cotangent shapes differ: "
            f"{branch_name}={shape} != {expected_shape}")
    if not bool(torch.isfinite(detached).all().item()):
        raise FloatingPointError(
            f"{branch_name} pixel cotangent is non-finite")
    branch_norm = detached.norm().detach()
    if accumulated is None:
        accumulated = detached
    else:
        accumulated.add_(detached)
    return accumulated, shape, branch_norm


class ElasticForcingModel(FD):
    """Match generated and teacher latents in a configured dense feature space.

    The generator remains the only trainable model.  In ``post_block_latent``
    mode ``real_score`` is reused as the frozen Wan encoder so the trainer can
    shard it normally with FSDP.  In ``vjepa2_dense_latent`` mode ``real_score``
    is not constructed; a frozen, replicated FP32 V-JEPA2 prefix is used
    instead.
    """

    uses_critic = False
    uses_vae_for_loss = False
    uses_vae_for_visualization = True
    requires_distribution_state_on_resume = True

    _DENSE_VIDEO_FEATURE_TYPES = frozenset({
        "vjepa2_dense_latent",
        "dinov3_dense_latent",
        "videomae_v2_dense_latent",
        "dual_vjepa2_videomae_v2_dense_latent",
        "dual_dinov3_vjepa2_dense_latent",
        "triple_dinov3_vjepa2_videomae_v2_dense_latent",
    })

    @staticmethod
    def _dense_feature_prefix_for(feature_type: str) -> Optional[str]:
        return {
            "vjepa2_dense_latent": "vjepa2",
            "dinov3_dense_latent": "dinov3",
            "videomae_v2_dense_latent": "videomae_v2",
            "dual_vjepa2_videomae_v2_dense_latent": "dual",
            "dual_dinov3_vjepa2_dense_latent": "dinov3_vjepa2",
            "triple_dinov3_vjepa2_videomae_v2_dense_latent": "triple",
        }.get(str(feature_type))

    def _dense_config_value(self, args, suffix: str, default):
        """Read a backbone-specific public config key.

        The established replay implementation keeps its historical internal
        ``vjepa2_*`` attributes for compatibility. Public VideoMAE-v2 configs,
        metadata, logs, and checkpoint keys remain independently namespaced.
        """
        prefix = self.dense_feature_prefix
        return getattr(args, f"{prefix}_{suffix}", default)

    @property
    def dense_state_prefix(self) -> str:
        """Public artifact namespace for the active dense-video backbone."""
        prefix = getattr(self, "dense_feature_prefix", None)
        if prefix is None:
            prefix = self._dense_feature_prefix_for(
                getattr(self, "feature_type", ""))
        return prefix or "vjepa2"

    @property
    def dense_nystrom_version_key(self) -> str:
        return f"{self.dense_state_prefix}_dense_nystrom_mmd_version"

    @property
    def dense_ema_version_key(self) -> str:
        return f"{self.dense_state_prefix}_dense_mmd_version"

    @property
    def dense_ema_payload_key(self) -> str:
        return f"{self.dense_state_prefix}_dense_mmd"

    @property
    def dense_feature_display_name(self) -> str:
        return (
            "VideoMAE-v2"
            if self.dense_state_prefix == "videomae_v2"
            else (
                "DINOv3 + V-JEPA2 + VideoMAE-v2"
                if self.dense_state_prefix == "triple"
                else (
                    "DINOv3 + V-JEPA2"
                    if self.dense_state_prefix == "dinov3_vjepa2"
                    else (
                        "V-JEPA2 + VideoMAE-v2"
                    if self.dense_state_prefix == "dual"
                    else ("DINOv3" if self.dense_state_prefix == "dinov3" else "V-JEPA2")
                    )
                )
            )
        )

    @property
    def dense_feature_metric_prefix(self) -> str:
        return f"{self.dense_state_prefix}_dense"

    @property
    def dense_sampled_tokens_per_layer(self) -> int:
        return int(self.vjepa2_dense_num_tokens) * int(
            getattr(self, "dense_feature_segment_count", 1))

    @property
    def dense_raw_tokens_total(self) -> int:
        return int(self.vjepa2_dense_raw_tokens) * int(
            getattr(self, "dense_feature_segment_count", 1))

    def _namespace_dense_metrics(self, values: Mapping[str, object]) -> dict:
        """Translate legacy internal metric names at the public boundary."""
        if self.dense_state_prefix == "vjepa2":
            return dict(values)
        if self.dense_state_prefix in {
            "dual", "triple", "dinov3", "dinov3_vjepa2"
        }:
            return dict(values)
        return {
            (
                key.replace("vjepa2_", "videomae_v2_", 1)
                if isinstance(key, str) and key.startswith("vjepa2_")
                else key
            ): value
            for key, value in values.items()
        }

    @staticmethod
    def _all_ranks_true(local_value: bool, device) -> bool:
        flag = torch.tensor(
            int(bool(local_value)), device=device, dtype=torch.int32)
        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _synchronize_dinov3_local_failure(
        self,
        error: Optional[BaseException],
        device,
        *,
        operation: str,
    ) -> None:
        """Make a rank-local DINOv3 failure an all-rank branch decision.

        DINOv3 startup performs expensive local imports, file inspection and
        metadata validation before it hashes shared artifacts. A rank that
        raises before a peer enters one of those hash broadcasts would strand
        the peer. Every such local-only phase terminates through this helper.
        Single-process callers retain the original exception type and message.
        """
        if self._all_ranks_true(error is None, device):
            return
        is_distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if error is not None and not is_distributed:
            raise error
        detail = (
            f": {type(error).__name__}: {error}"
            if error is not None
            else " on another rank"
        )
        raise RuntimeError(operation + detail) from error
    @staticmethod
    def _configured_current_teacher_global_samples(args) -> int:
        """Resolve the exact live-teacher geometry for this process group.

        Legacy GradCache updates use every forward sample for the MMD VJP, so
        their teacher count remains ``total_batch_size``. Partial replay
        explicitly configures a per-rank gradient suffix; only that suffix has
        a matched teacher and therefore the exact teacher kernel has
        ``gradient_local * world_size`` rows.
        """
        configured_total = int(getattr(args, "total_batch_size", 0))
        gradient_local = getattr(
            args, "fresh_video_generator_grad_local_samples", None)
        if gradient_local is None:
            return configured_total
        if type(gradient_local) is not int or gradient_local <= 0:
            raise ValueError(
                "fresh_video_generator_grad_local_samples must be a "
                "positive integer")
        forward_local = getattr(args, "fresh_video_local_samples", None)
        if type(forward_local) is not int or forward_local <= 0:
            raise ValueError(
                "fresh_video_local_samples must be a positive integer when "
                "partial GradCache replay is configured")
        if gradient_local > forward_local:
            raise ValueError(
                "fresh_video_generator_grad_local_samples must not exceed "
                "fresh_video_local_samples")

        if dist.is_available() and dist.is_initialized():
            world_size = int(dist.get_world_size())
        else:
            if configured_total <= 0 or configured_total % forward_local != 0:
                raise ValueError(
                    "total_batch_size must be divisible by "
                    "fresh_video_local_samples before distributed "
                    "initialization")
            world_size = configured_total // forward_local
        expected_forward = forward_local * world_size
        if configured_total != expected_forward:
            raise ValueError(
                "total_batch_size does not match the partial GradCache "
                "forward geometry: "
                f"{configured_total} != {forward_local} * {world_size}")
        return gradient_local * world_size

    def __init__(self, args, device):
        # FD's generation path is useful, but its pixel judges are not built by
        # the base class.  The opt-in fusion branch constructs a reusable
        # FrameFDHead after the shared generator/text/VAE objects exist.
        self.frame_fd_fusion_enabled = bool(
            getattr(args, "frame_fd_fusion_enabled", False))
        self.frame_fd_fusion_mode = str(getattr(
            args, "frame_fd_fusion_mode", "grad_norm_ema"))
        if self.frame_fd_fusion_mode not in {
            "grad_norm_ema",
            "fidel_two_level",
        }:
            raise ValueError(
                "frame_fd_fusion_mode must be grad_norm_ema or "
                f"fidel_two_level, got {self.frame_fd_fusion_mode!r}")
        self.fidel_two_level_enabled = bool(
            self.frame_fd_fusion_enabled
            and self.frame_fd_fusion_mode == "fidel_two_level"
        )
        p0_values = {
            "SF_FRESH_VIDEO_P0_REPLAY": os.environ.get(
                "SF_FRESH_VIDEO_P0_REPLAY"),
            "SF_FIDEL_P0_REPLAY": os.environ.get(
                "SF_FIDEL_P0_REPLAY"),
        }
        p0_enabled = {}
        for name, value in p0_values.items():
            if value is None:
                continue
            normalized = value.strip().lower()
            if normalized not in {
                "0", "1", "false", "true", "no", "yes"
            }:
                raise ValueError(
                    f"{name} must be one of 0/1/false/true/no/yes")
            p0_enabled[name] = normalized in {"1", "true", "yes"}
        if len(set(p0_enabled.values())) > 1:
            raise ValueError(
                "SF_FRESH_VIDEO_P0_REPLAY and the legacy "
                "SF_FIDEL_P0_REPLAY setting disagree")
        self._fresh_video_p0_requested = next(
            iter(p0_enabled.values()), True)
        # Final eligibility is known only after the estimator is initialized.
        self.fresh_video_prepared_replay_enabled = False
        self.fidel_p0_replay_enabled = False
        self.fidel_p0_pixel_cache_device = os.environ.get(
            "SF_FRESH_VIDEO_PIXEL_CACHE_DEVICE", "cuda"
        ).strip().lower()
        if self.fidel_p0_pixel_cache_device not in {"cpu", "cuda"}:
            raise ValueError(
                "SF_FRESH_VIDEO_PIXEL_CACHE_DEVICE must be cpu or cuda")
        try:
            self.fidel_p0_branch_diagnostics_interval = int(os.environ.get(
                "SF_FIDEL_BRANCH_LATENT_DIAGNOSTICS_INTERVAL", "0"))
        except ValueError as exc:
            raise ValueError(
                "SF_FIDEL_BRANCH_LATENT_DIAGNOSTICS_INTERVAL must be an "
                "integer") from exc
        if self.fidel_p0_branch_diagnostics_interval < 0:
            raise ValueError(
                "SF_FIDEL_BRANCH_LATENT_DIAGNOSTICS_INTERVAL must be "
                "non-negative")
        configured_fd_representations = getattr(
            args, "fd_repr_models", [])
        requested_feature_type = str(
            getattr(args, "wan_feature_type", "post_block_latent"))
        supported_feature_types = {
            "post_block_latent",
            "vjepa2_dense_latent",
            "dinov3_dense_latent",
            "videomae_v2_dense_latent",
            "dual_vjepa2_videomae_v2_dense_latent",
            "dual_dinov3_vjepa2_dense_latent",
            "triple_dinov3_vjepa2_videomae_v2_dense_latent",
        }
        if requested_feature_type not in supported_feature_types:
            raise ValueError(
                "wan_feature_type must be one of "
                f"{sorted(supported_feature_types)}, got "
                f"{requested_feature_type!r}")
        self.dense_feature_prefix = self._dense_feature_prefix_for(
            requested_feature_type)
        is_dense_video_feature = bool(
            requested_feature_type in self._DENSE_VIDEO_FEATURE_TYPES)
        self.mmd_estimator = str(
            getattr(args, "wan_mmd_estimator", "ema_rff"))
        supported_estimators = {"ema_rff", "nystrom_exact_repulsion"}
        if self.mmd_estimator not in supported_estimators:
            raise ValueError(
                "wan_mmd_estimator must be one of "
                f"{sorted(supported_estimators)}, got "
                f"{self.mmd_estimator!r}")
        if (
            self.mmd_estimator == "nystrom_exact_repulsion"
            and not is_dense_video_feature
        ):
            raise ValueError(
                "nystrom_exact_repulsion currently requires "
                "a dense video feature backend")
        self.is_vjepa2_dense = (
            requested_feature_type == "vjepa2_dense_latent")
        self.is_dinov3_dense = (
            requested_feature_type == "dinov3_dense_latent")
        self.is_videomae_v2_dense = (
            requested_feature_type == "videomae_v2_dense_latent")
        self.is_dinov3_vjepa2_dense = (
            requested_feature_type == "dual_dinov3_vjepa2_dense_latent")
        self.is_triple_dense = (
            requested_feature_type
            == "triple_dinov3_vjepa2_videomae_v2_dense_latent")
        self.is_dense_video_feature = is_dense_video_feature
        if self.is_dinov3_dense and (
            self.mmd_estimator != "nystrom_exact_repulsion"
        ):
            raise ValueError(
                "wan_feature_type=dinov3_dense_latent currently requires "
                "wan_mmd_estimator=nystrom_exact_repulsion")
        if self.is_dense_video_feature:
            # BaseModel otherwise constructs full real/fake Wan score models.
            # Neither participates in this generator-only replay objective.
            args.uses_score_models = False
            args.uses_real_score = False
            args.uses_fake_score = False
        args.fd_repr_models = []
        self.generator_replay_activation_cpu_offload = bool(getattr(
            args, "generator_replay_activation_cpu_offload", False))
        self.generator_replay_activation_cpu_offload_block_stride = int(
            getattr(
                args,
                "generator_replay_activation_cpu_offload_block_stride",
                0,
            ))
        if self.generator_replay_activation_cpu_offload_block_stride < 0:
            raise ValueError(
                "generator_replay_activation_cpu_offload_block_stride must "
                "be non-negative")
        if (
            self.generator_replay_activation_cpu_offload
            and self.generator_replay_activation_cpu_offload_block_stride
        ):
            raise ValueError(
                "choose either full generator replay activation CPU "
                "offload or a positive block stride, not both")
        self.generator_replay_frozen_encoder_cpu_offload = bool(getattr(
            args, "generator_replay_frozen_encoder_cpu_offload", False))
        self._generator_replay_frozen_encoder_offload_state = None
        if (
            self.generator_replay_frozen_encoder_cpu_offload
            and not self.is_triple_dense
        ):
            raise ValueError(
                "generator replay frozen-encoder CPU offload is only "
                "implemented for the triple video MMD objective")
        try:
            super().__init__(args, device)
        finally:
            # Do not leave the caller's OmegaConf mutated. FrameFDHead reads
            # these representation names from the original configuration.
            args.fd_repr_models = configured_fd_representations
        self.generator.model.gradient_checkpointing_activation_cpu_offload = (
            False)
        self.generator.model.gradient_checkpointing_activation_cpu_offload_block_stride = (
            self.generator_replay_activation_cpu_offload_block_stride)
        if self.generator_replay_activation_cpu_offload and (
            not dist.is_initialized() or dist.get_rank() == 0):
            print("[memory] generator checkpoint activations offload to CPU")
        if self.generator_replay_activation_cpu_offload_block_stride and (
            not dist.is_initialized() or dist.get_rank() == 0):
            print(
                "[memory] generator checkpoint activations selectively "
                "offload to CPU every "
                f"{self.generator_replay_activation_cpu_offload_block_stride} "
                "blocks")
        self.feature_type = requested_feature_type
        self.dense_encoder_name = self.dense_state_prefix
        self.dense_log_prefix = f"{self.dense_encoder_name}_dense"

        if self.frame_fd_fusion_enabled and (
            not self.is_vjepa2_dense
            or self.mmd_estimator != "nystrom_exact_repulsion"
        ):
            raise ValueError(
                "frame_fd_fusion_enabled requires "
                "wan_feature_type=vjepa2_dense_latent and "
                "wan_mmd_estimator=nystrom_exact_repulsion")
        if (
            self.frame_fd_fusion_enabled
            and not self.fidel_two_level_enabled
            and str(getattr(
                args, "frame_fd_dynamic_weighting", "grad_norm_ema"
            )) != "grad_norm_ema"
        ):
            raise ValueError(
                "frame FD fusion currently requires "
                "frame_fd_dynamic_weighting=grad_norm_ema")
        if (
            self.frame_fd_fusion_enabled
            and int(getattr(args, "fd_decode_frames", 0)) != 0
        ):
            raise ValueError(
                "frame FD fusion requires fd_decode_frames=0 so teacher and "
                "student sample from the same complete 81-frame video")

        if self.feature_type == "post_block_latent":
            self._initialize_wan_features(args)
        elif self.is_vjepa2_dense:
            self._initialize_vjepa2_features(args, device)
        elif self.is_dinov3_dense:
            self._initialize_dinov3_features(args, device)
        elif self.feature_type == "videomae_v2_dense_latent":
            self._initialize_videomae_v2_features(args, device)
        elif self.is_dinov3_vjepa2_dense:
            self._initialize_dual_dinov3_vjepa2_features(args, device)
        else:
            self._initialize_multi_encoder_features(args, device)

        self.latent_loss_weight = float(
            getattr(args, "wan_kv_loss_weight", 1.0))
        self.latent_self_normalize = bool(
            getattr(args, "wan_kv_self_normalize", True))
        self.latent_norm_eps = float(
            getattr(args, "wan_kv_norm_eps", 1e-4))
        if (
            getattr(self, "vjepa2_multilayer_enabled", False)
            and self.mmd_estimator != "nystrom_exact_repulsion"
        ):
            raise ValueError(
                "multi-layer V-JEPA2 MMD requires "
                "wan_mmd_estimator=nystrom_exact_repulsion")
        if (
            getattr(self, "vjepa2_multilayer_enabled", False)
            and not self.latent_self_normalize
        ):
            raise ValueError(
                "multi-layer V-JEPA2 MMD requires "
                "wan_kv_self_normalize=true so layer losses are normalized "
                "independently before summation")
        if self.is_vjepa2_dense:
            self.vjepa2_token_sampling = str(getattr(
                args, "vjepa2_token_sampling", "uniform_fixed"))
            if self.vjepa2_token_sampling not in {
                "uniform_fixed",
                "random_without_replacement_per_update",
            }:
                raise ValueError(
                    "vjepa2_token_sampling must be uniform_fixed or "
                    "random_without_replacement_per_update")
            self.uses_random_token_reference_bank = bool(
                self.vjepa2_token_sampling
                == "random_without_replacement_per_update"
            )
        else:
            # DINOv3 owns an independent fixed spatial-token contract. It must
            # never inherit a V-JEPA2 token-mask/reference-bank setting.
            self.vjepa2_token_sampling = None
            self.uses_random_token_reference_bank = False
        if (
            getattr(self, "vjepa2_multilayer_enabled", False)
            and self.uses_random_token_reference_bank
        ):
            raise ValueError(
                "multi-layer V-JEPA2 MMD currently requires "
                "vjepa2_token_sampling=uniform_fixed")
        if (
            getattr(self, "vjepa2_multilayer_enabled", False)
            and self.frame_fd_fusion_enabled
        ):
            raise ValueError(
                "multi-layer V-JEPA2 MMD is video-only and requires "
                "frame_fd_fusion_enabled=false")
        if self.uses_random_token_reference_bank != self.fidel_two_level_enabled:
            raise ValueError(
                "random per-update dense tokens and two-level FiDEL must be "
                "enabled together; legacy estimators keep uniform_fixed")
        self.vjepa2_reference_bank = None
        self.dense_reference_bank = None
        self.vjepa2_active_reference_slot = None
        self.vjepa2_active_token_indices_hash = None
        self.dense_active_reference_slot = None
        self.dense_active_token_indices_hash = None
        self.uses_fresh_video_gradcache = (
            self.mmd_estimator == "nystrom_exact_repulsion")
        self.nystrom_attraction_ema_enabled = bool(
            self._dense_config_value(
                args, "nystrom_attraction_ema_enabled", False))
        self.nystrom_attraction_ema_beta = float(
            self._dense_config_value(
                args, "nystrom_attraction_ema_beta", 0.99))
        self.nystrom_attraction_weight = float(
            self._dense_config_value(
                args, "nystrom_attraction_weight", 1.0))
        self.current_teacher_attraction_enabled = bool(
            self._dense_config_value(
                args, "nystrom_current_teacher_attraction_enabled", False))
        self.current_teacher_attraction_weight = float(
            self._dense_config_value(
                args, "nystrom_current_teacher_attraction_weight", 1.0))
        self.expected_current_teacher_samples = int(getattr(
            args, "expected_current_teacher_samples", 0))
        self.current_teacher_source = str(getattr(
            args, "current_teacher_source", "auto"))
        if self.current_teacher_source not in {
            "auto", "none", "lmdb_latent", "openvid_video",
            "matched_manifest_video", "matched_dual_feature_bank",
            "matched_triple_feature_bank",
            "matched_dinov3_vjepa2_feature_bank",
        }:
            raise ValueError(
                "unsupported current_teacher_source: "
                f"{self.current_teacher_source!r}")
        self.nystrom_repulsion_fifo_size = int(
            self._dense_config_value(
                args, "nystrom_repulsion_fifo_size", 0))
        if (
            not np.isfinite(self.nystrom_attraction_weight)
            or self.nystrom_attraction_weight < 0.0
        ):
            raise ValueError(
                f"{self.dense_state_prefix}_nystrom_attraction_weight must "
                "be finite and non-negative")
        if (
            not np.isfinite(self.current_teacher_attraction_weight)
            or self.current_teacher_attraction_weight < 0.0
        ):
            raise ValueError(
                f"{self.dense_state_prefix}_nystrom_current_teacher_"
                "attraction_weight must be finite and non-negative")
        if (
            self.current_teacher_attraction_enabled
            and self.nystrom_attraction_weight
            + self.current_teacher_attraction_weight <= 0.0
        ):
            raise ValueError(
                "fixed/current teacher attraction weights must have a "
                "positive sum")
        if self.nystrom_repulsion_fifo_size < 0:
            raise ValueError(
                "dense Nyström repulsion FIFO size must be non-negative")
        self.nystrom_repulsion_fifo_enabled = bool(
            self.nystrom_repulsion_fifo_size > 0)
        if (
            not np.isfinite(self.nystrom_attraction_ema_beta)
            or not 0.0 <= self.nystrom_attraction_ema_beta < 1.0
        ):
            raise ValueError(
                "dense Nyström attraction EMA beta must be in [0, 1)")
        if (
            self.nystrom_attraction_ema_enabled
            and not self.uses_fresh_video_gradcache
        ):
            raise ValueError(
                "vjepa2_nystrom_attraction_ema_enabled requires "
                "wan_mmd_estimator=nystrom_exact_repulsion")
        if (
            self.nystrom_attraction_ema_enabled
            and self.uses_random_token_reference_bank
        ):
            raise ValueError(
                "attraction EMA requires a fixed Nyström reference; it is "
                "incompatible with random per-update reference-bank slots")
        if (
            self.current_teacher_attraction_enabled
            and not self.uses_fresh_video_gradcache
        ):
            raise ValueError(
                "current teacher attraction requires "
                "wan_mmd_estimator=nystrom_exact_repulsion")
        if (
            self.current_teacher_attraction_enabled
            and self.uses_random_token_reference_bank
        ):
            raise ValueError(
                "current teacher attraction currently requires a fixed "
                "Nyström reference")
        if (
            self.current_teacher_attraction_enabled
            and getattr(self, "vjepa2_multilayer_enabled", False)
        ):
            raise ValueError(
                "current teacher attraction currently supports one dense "
                "feature layer")
        if self.current_teacher_attraction_enabled:
            configured_teacher_total = (
                self._configured_current_teacher_global_samples(args)
            )
            if self.expected_current_teacher_samples <= 0:
                raise ValueError(
                    "expected_current_teacher_samples must be explicitly "
                    "positive when current teacher attraction is enabled")
            if configured_teacher_total != self.expected_current_teacher_samples:
                raise ValueError(
                    "the configured GradCache gradient geometry must match "
                    "expected_current_teacher_samples for exact current "
                    "teacher MMD: "
                    f"{configured_teacher_total} != "
                    f"{self.expected_current_teacher_samples}")
        if (
            self.nystrom_repulsion_fifo_enabled
            and not self.uses_fresh_video_gradcache
        ):
            raise ValueError(
                "vjepa2_nystrom_repulsion_fifo_size requires "
                "wan_mmd_estimator=nystrom_exact_repulsion")
        if (
            self.nystrom_repulsion_fifo_enabled
            and self.uses_random_token_reference_bank
        ):
            raise ValueError(
                "repulsion FIFO requires a fixed Nyström reference; queued "
                "features cannot be mixed across random token masks")
        if (
            self.nystrom_repulsion_fifo_enabled
            and getattr(self, "vjepa2_multilayer_enabled", False)
        ):
            raise ValueError(
                "repulsion FIFO currently supports one fixed V-JEPA2 layer")
        if (
            self.nystrom_repulsion_fifo_enabled
            and self.frame_fd_fusion_enabled
        ):
            raise ValueError(
                "repulsion FIFO training requires frame_fd_fusion_enabled=false")
        if (
            self.nystrom_repulsion_fifo_enabled
            and self.nystrom_attraction_ema_enabled
        ):
            raise ValueError(
                "repulsion FIFO training requires attraction EMA to be disabled")
        self.fresh_video_prepared_replay_enabled = bool(
            self._fresh_video_p0_requested
            and self.uses_fresh_video_gradcache
            and (
                not self.frame_fd_fusion_enabled
                or self.fidel_two_level_enabled
            )
        )
        self.fidel_p0_replay_enabled = bool(
            self.fresh_video_prepared_replay_enabled
            and self.fidel_two_level_enabled)
        if self.uses_fresh_video_gradcache:
            if not self.causal_generator:
                raise NotImplementedError(
                    "nystrom_exact_repulsion currently requires the causal "
                    "generator so every stochastic rollout input can be "
                    "captured explicitly")
            from losses.video_nystrom_mmd import (
                VideoNystromMMD,
                load_reference_pack,
            )
            kernel_chunk_size = int(
                getattr(
                    args,
                    "dinov3_nystrom_kernel_chunk_size",
                    32,
                )
                if self.is_dinov3_dense
                else self._dense_config_value(
                    args, "nystrom_kernel_chunk_size", 32)
            )
            if self.is_dinov3_dense:
                reference_load_error = None
                reference_path = None
                reference_pack = None
                try:
                    configured_reference = str(getattr(
                        args,
                        "dinov3_nystrom_reference_path",
                        "",
                    )).strip()
                    reference_path = (
                        os.path.realpath(os.path.expanduser(
                            configured_reference))
                        if configured_reference else ""
                    )
                    if (
                        not reference_path
                        or not os.path.isfile(reference_path)
                    ):
                        raise FileNotFoundError(
                            "DINOv3 dense Nyström requires a precomputed fixed "
                            "reference pack under "
                            "dinov3_nystrom_reference_path; "
                            f"got {reference_path!r}")
                    reference_pack = load_reference_pack(
                        reference_path, map_location="cpu")
                except BaseException as exc:
                    reference_load_error = exc
                self._synchronize_dinov3_local_failure(
                    reference_load_error,
                    device,
                    operation="DINOv3 reference-pack loading failed",
                )
                self._validate_dinov3_reference_runtime_identity(
                    args, reference_pack["metadata"])
                objective_error = None
                try:
                    reference_section = reference_pack["metadata"].get(
                        "reference", {})
                    reference_source = (
                        reference_section.get("source", {})
                        if isinstance(reference_section, dict)
                        else {}
                    )
                    feature_bank_hash = reference_source.get(
                        "feature_bank_fp16_sha256")
                    if (
                        not isinstance(feature_bank_hash, str)
                        or len(feature_bank_hash) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in feature_bank_hash
                        )
                    ):
                        raise ValueError(
                            "DINOv3 Nyström reference is missing a valid "
                            "frozen teacher feature_bank_fp16_sha256")
                    objective = VideoNystromMMD(
                        reference_pack,
                        expected_metadata={
                            "feature": self._dinov3_feature_metadata(),
                            "nystrom": {
                                "num_landmarks": int(getattr(
                                    args, "dinov3_nystrom_landmarks", 256)),
                                "seed": int(getattr(
                                    args, "dinov3_nystrom_seed", 0)),
                            },
                        },
                        kernel_chunk_size=kernel_chunk_size,
                        attraction_ema_enabled=False,
                        repulsion_fifo_size=0,
                    )
                    reference_count = int(
                        objective.reference_metadata[
                            "nystrom"]["reference_count"])
                    if int(reference_source.get(
                        "selected_samples", -1
                    )) != reference_count:
                        raise ValueError(
                            "DINOv3 Nyström reference source count does not "
                            "match the frozen kernel reference count")
                    self.dinov3_nystrom_reference_path = reference_path
                    self.dinov3_nystrom_reference_hash = str(
                        objective.reference_hash)
                    self.dinov3_teacher_feature_bank_hash = feature_bank_hash
                    self.dinov3_nystrom_reference_count = reference_count
                    self.latent_mmd = objective.to(device=device)
                except BaseException as exc:
                    objective_error = exc
                self._synchronize_dinov3_local_failure(
                    objective_error,
                    device,
                    operation="DINOv3 reference objective setup failed",
                )
            elif self.uses_random_token_reference_bank:
                from losses.video_nystrom_reference_bank import (
                    VideoNystromReferenceBank,
                )

                configured_manifest = str(getattr(
                    args,
                    "vjepa2_nystrom_reference_bank_manifest",
                    "",
                )).strip()
                manifest_path = (
                    os.path.realpath(os.path.expanduser(configured_manifest))
                    if configured_manifest else ""
                )
                if not manifest_path or not os.path.isfile(manifest_path):
                    raise FileNotFoundError(
                        "random per-update V-JEPA2 tokens require "
                        "vjepa2_nystrom_reference_bank_manifest; got "
                        f"{manifest_path!r}")
                expected_slots = int(getattr(
                    args, "vjepa2_token_mask_count", 100))
                configured_max_steps = int(getattr(args, "max_steps", 100))
                if (
                    expected_slots <= 0
                    or configured_max_steps <= 0
                    or configured_max_steps > expected_slots
                ):
                    raise ValueError(
                        "max_steps must be positive and no larger than "
                        "vjepa2_token_mask_count so reference masks never "
                        "cycle")
                self.vjepa2_reference_bank = VideoNystromReferenceBank(
                    manifest_path,
                    expected_slot_count=expected_slots,
                    expected_pack_metadata={
                        "feature": self._vjepa2_static_feature_metadata(),
                        "nystrom": {
                            "num_landmarks": int(getattr(
                                args, "vjepa2_nystrom_landmarks", 256)),
                        },
                    },
                    kernel_chunk_size=kernel_chunk_size,
                )
                self._validate_vjepa2_reference_bank_runtime_identity(args)
                activated = self.vjepa2_reference_bank.activate_slot_(0)
                self.latent_mmd = activated.objective.to(device=device)
                self.vjepa2_nystrom_reference_path = manifest_path
                self.vjepa2_nystrom_reference_bank_hash = (
                    self.vjepa2_reference_bank.manifest_hash)
                self.vjepa2_teacher_feature_bank_hash = str(
                    self.vjepa2_reference_bank.metadata.get(
                        "teacher_full_dense_bank_fp16_sha256",
                        self.vjepa2_reference_bank.manifest_hash,
                    )
                )
                self.vjepa2_nystrom_reference_count = len(
                    self.vjepa2_reference_bank.train_indices)
                self._publish_active_reference(activated)
            else:
                def load_fixed_reference(layer_index, configured_path):
                    reference_path = (
                        os.path.realpath(os.path.expanduser(
                            str(configured_path).strip()))
                        if str(configured_path).strip() else ""
                    )
                    if (
                        not reference_path
                        or not os.path.isfile(reference_path)
                    ):
                        raise FileNotFoundError(
                            "nystrom_exact_repulsion requires a precomputed "
                            "reference pack; got "
                            f"{reference_path!r} for layer {layer_index}")
                    reference_pack = load_reference_pack(
                        reference_path, map_location="cpu")
                    reference_section = reference_pack["metadata"].get(
                        "reference", {})
                    reference_source = (
                        reference_section.get("source", {})
                        if isinstance(reference_section, dict)
                        else {}
                    )
                    feature_bank_hash = reference_source.get(
                        "feature_bank_fp16_sha256")
                    if (
                        not isinstance(feature_bank_hash, str)
                        or len(feature_bank_hash) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in feature_bank_hash
                        )
                    ):
                        raise ValueError(
                            "Nyström reference is missing a valid frozen "
                            "teacher feature_bank_fp16_sha256 for layer "
                            f"{layer_index}")
                    objective = VideoNystromMMD(
                        reference_pack,
                        expected_metadata={
                            "feature": self._vjepa2_feature_metadata(
                                layer_index=layer_index),
                            "nystrom": {
                                "num_landmarks": int(
                                    self._dense_config_value(
                                        args, "nystrom_landmarks", 256)),
                                "seed": int(
                                    self._dense_config_value(
                                        args, "nystrom_seed", 0)),
                            },
                        },
                        kernel_chunk_size=kernel_chunk_size,
                        attraction_ema_enabled=(
                            self.nystrom_attraction_ema_enabled),
                        attraction_ema_beta=(
                            self.nystrom_attraction_ema_beta),
                        attraction_weight=(
                            self.nystrom_attraction_weight),
                        current_teacher_attraction_enabled=(
                            self.current_teacher_attraction_enabled),
                        current_teacher_attraction_weight=(
                            self.current_teacher_attraction_weight),
                        repulsion_fifo_size=(
                            self.nystrom_repulsion_fifo_size),
                    )
                    reference_count = int(
                        objective.reference_metadata[
                            "nystrom"]["reference_count"])
                    if int(reference_source.get(
                        "selected_samples", -1
                    )) != reference_count:
                        raise ValueError(
                            "Nyström reference source count does not match "
                            "the frozen kernel reference count for layer "
                            f"{layer_index}")
                    source_identity = {
                        key: value
                        for key, value in reference_source.items()
                        if key != "feature_bank_fp16_sha256"
                    }
                    return (
                        reference_path,
                        objective,
                        feature_bank_hash,
                        reference_count,
                        source_identity,
                    )

                if getattr(self, "vjepa2_multilayer_enabled", False):
                    configured_paths = getattr(
                        args, "vjepa2_nystrom_reference_paths", None)
                    if (
                        configured_paths is None
                        or isinstance(configured_paths, (str, bytes))
                    ):
                        raise ValueError(
                            "multi-layer V-JEPA2 MMD requires "
                            "vjepa2_nystrom_reference_paths in layer order")
                    configured_paths = tuple(configured_paths)
                    if len(configured_paths) != len(
                        self.vjepa2_dense_layers
                    ):
                        raise ValueError(
                            "vjepa2_nystrom_reference_paths must match "
                            "vjepa2_dense_layers one-for-one")
                    loaded = [
                        load_fixed_reference(layer, path)
                        for layer, path in zip(
                            self.vjepa2_dense_layers,
                            configured_paths,
                        )
                    ]
                    objectives = {
                        layer: item[1]
                        for layer, item in zip(
                            self.vjepa2_dense_layers, loaded)
                    }
                    self.latent_mmd = MultiLayerVideoNystromMMD(
                        objectives,
                        self_normalize=self.latent_self_normalize,
                        norm_eps=self.latent_norm_eps,
                    ).to(device=device)
                    self.vjepa2_nystrom_reference_paths = tuple(
                        item[0] for item in loaded)
                    self.vjepa2_nystrom_reference_hashes = tuple(
                        objective.reference_hash
                        for objective in objectives.values())
                    self.vjepa2_teacher_feature_bank_hashes = tuple(
                        item[2] for item in loaded)
                    reference_counts = tuple(
                        item[3] for item in loaded)
                    if len(set(reference_counts)) != 1:
                        raise ValueError(
                            "multi-layer Nyström references must contain "
                            "the same teacher sample count")
                    source_identities = tuple(
                        item[4] for item in loaded)
                    if any(
                        identity != source_identities[0]
                        for identity in source_identities[1:]
                    ):
                        raise ValueError(
                            "multi-layer Nyström references must use the "
                            "same teacher LMDB rows and prompt index")
                    self.vjepa2_nystrom_reference_count = (
                        reference_counts[0])
                    # Singular aliases keep generic diagnostics compatible;
                    # checkpoint identity uses the complete tuples below.
                    self.vjepa2_nystrom_reference_path = (
                        self.vjepa2_nystrom_reference_paths[-1])
                    self.vjepa2_nystrom_reference_hash = (
                        self.vjepa2_nystrom_reference_hashes[-1])
                    self.vjepa2_teacher_feature_bank_hash = (
                        self.vjepa2_teacher_feature_bank_hashes[-1])
                else:
                    configured_reference_path = self._dense_config_value(
                        args, "nystrom_reference_path", "")
                    (
                        reference_path,
                        objective,
                        feature_bank_hash,
                        reference_count,
                        _,
                    ) = load_fixed_reference(
                        self.vjepa2_dense_layer,
                        configured_reference_path,
                    )
                    self.vjepa2_nystrom_reference_path = reference_path
                    self.latent_mmd = objective.to(device=device)
                    self.vjepa2_teacher_feature_bank_hash = (
                        feature_bank_hash)
                    self.vjepa2_nystrom_reference_hash = str(
                        objective.reference_hash)
                    self.vjepa2_nystrom_reference_count = reference_count
            if self.is_dinov3_dense and (
                not dist.is_available()
                or not dist.is_initialized()
                or dist.get_rank() == 0
            ):
                print(
                    "[dinov3-nystrom] loaded frozen reference "
                    f"n={self.dinov3_nystrom_reference_count}, "
                    f"m={self.latent_mmd.landmarks.shape[0]}, "
                    f"sigma={float(self.latent_mmd.sigma):.6g}, "
                    f"hash={self.dinov3_nystrom_reference_hash}, "
                    "teacher_features="
                    f"{self.dinov3_teacher_feature_bank_hash}",
                    flush=True,
                )
            elif (
                not dist.is_available()
                or not dist.is_initialized()
                or dist.get_rank() == 0
            ):
                if getattr(self, "vjepa2_multilayer_enabled", False):
                    reference_summary = ", ".join(
                        (
                            f"layer={layer}:"
                            f"sigma={float(objective.sigma):.6g}:"
                            f"hash={objective.reference_hash}"
                        )
                        for layer, objective in zip(
                            self.vjepa2_dense_layers,
                            self.latent_mmd.objectives.values(),
                        )
                    )
                    print(
                        "[vjepa2-nystrom] loaded frozen multi-layer "
                        f"references n={self.vjepa2_nystrom_reference_count}, "
                        f"m={self.latent_mmd.landmarks.shape[0]}, "
                        f"normalized={self.latent_self_normalize}, "
                        "attraction_ema="
                        f"{self.nystrom_attraction_ema_enabled}, "
                        "attraction_weight="
                        f"{self.nystrom_attraction_weight:g}, "
                        "repulsion_fifo="
                        f"{self.nystrom_repulsion_fifo_size}, "
                        f"{reference_summary}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{self.dense_encoder_name}-nystrom] loaded frozen reference "
                        f"n={self.vjepa2_nystrom_reference_count}, "
                        f"m={self.latent_mmd.landmarks.shape[0]}, "
                        f"sigma={float(self.latent_mmd.sigma):.6g}, "
                        f"hash={self.vjepa2_nystrom_reference_hash}, "
                        "attraction_ema="
                        f"{self.nystrom_attraction_ema_enabled}, "
                        "repulsion_fifo="
                        f"{self.nystrom_repulsion_fifo_size}, "
                        "slot="
                        f"{self.vjepa2_active_reference_slot}, "
                        "teacher_features="
                        f"{self.vjepa2_teacher_feature_bank_hash}",
                        flush=True,
                    )
        else:
            self.latent_mmd = EMARBFMMD(
                beta=float(getattr(args, "wan_kv_ema_beta", 0.99)),
                num_rff_features=int(
                    getattr(args, "wan_kv_rff_features", 256)),
                bandwidth=getattr(args, "wan_kv_rbf_bandwidth", None),
                bandwidth_multipliers=getattr(
                    args, "wan_kv_rbf_multipliers", [0.5, 1.0, 2.0]),
                seed=int(getattr(args, "wan_kv_rff_seed", 0)),
                row_normalize=bool(
                    getattr(args, "wan_kv_row_normalize", False)),
                sample_mode=(
                    "video_matrix"
                    if self.is_dense_video_feature
                    else "rows"
                ),
            ).to(device)
        if self.feature_type == "triple_dinov3_vjepa2_videomae_v2_dense_latent":
            self._build_multi_encoder_objective(args, device)
        elif self.is_dinov3_vjepa2_dense:
            self._replace_with_dual_dinov3_vjepa2_nystrom_objective(
                args, device)
        # Dense video backends always decode videos. The Wan path only needs VAE
        # loss decoding for longer causal rollouts.
        self.uses_vae_for_loss = (
            self.is_dense_video_feature
            or bool(self.causal_generator and self.num_training_frames > 21)
        )
        self.frame_fd_head = None
        self.frame_fd_balancer = None
        self.fidel_controller = None
        self.frame_fd_reference_metadata = None
        if self.frame_fd_fusion_enabled:
            from model.frame_fd_head import FrameFDHead

            self.frame_fd_head = FrameFDHead.from_config(
                args, device, vae=self.vae)
            self.frame_fd_reference_metadata = (
                self._load_and_validate_frame_fd_manifest(args))
            if self.fidel_two_level_enabled:
                from trainer.two_level_fidel import TwoLevelFiDELController

                if str(getattr(
                    args,
                    "fidel_chemical_potential_mode",
                    "ema_median",
                )) != "ema_median":
                    raise ValueError(
                        "two-level FiDEL currently requires "
                        "fidel_chemical_potential_mode=ema_median")
                if str(getattr(
                    args,
                    "fidel_controller_score_source",
                    "fresh_only",
                )) != "fresh_only":
                    raise ValueError(
                        "two-level FiDEL frame violations require the fixed "
                        "fresh_only 2048-frame population")
                if int(getattr(args, "fd_num_frames", 16)) != 16:
                    raise ValueError(
                        "the calibrated two-level FiDEL reference requires "
                        "fd_num_frames=16")
                if int(getattr(args, "fd_queue_size", 50000)) != 50000:
                    raise ValueError(
                        "the FiDEL frame objective requires the configured "
                        "50,000-feature FIFO")
                if int(getattr(args, "queue_warmup_samples", 0)) != 0:
                    raise ValueError(
                        "two-level FiDEL does not prefill the frame FIFO")
                if self.frame_fd_head.replay_grad_clip_z is not None:
                    raise ValueError(
                        "two-level FiDEL requires replay_grad_clip_z=null; "
                        "branch-local clipping would change the detached "
                        "FiDEL mixture coefficients. The final generator "
                        "gradient is clipped once after all branches merge.")
                if any(
                    abs(float(judge.weight) - 1.0) > 0.0
                    for judge in self.frame_fd_head.judges
                ):
                    raise ValueError(
                        "two-level FiDEL requires every fd_repr_weight to be "
                        "1; FiDEL alpha owns all representation weighting")
                self.fidel_controller = TwoLevelFiDELController(
                    loss_scale_beta=float(getattr(
                        args, "fidel_loss_scale_beta", 0.99)),
                    theta_beta=float(getattr(
                        args, "fidel_controller_ema_beta", 0.99)),
                    frame_temperature=float(getattr(
                        args, "fidel_frame_temperature", 1.0)),
                    group_temperature=float(getattr(
                        args, "fidel_group_temperature", 1.0)),
                    frame_budget=float(getattr(
                        args, "fidel_frame_budget", 1.0)),
                    video_prior=float(getattr(
                        args, "fidel_video_prior", 0.7)),
                    frame_prior=float(getattr(
                        args, "fidel_frame_prior", 0.3)),
                    warmup_commits=int(getattr(
                        args, "fidel_controller_warmup_updates", 5)),
                    hysteresis_enter=float(getattr(
                        args, "fidel_satisfaction_hysteresis", 0.02)),
                    epsilon=float(getattr(
                        args, "fidel_floor_epsilon", 1.0e-8)),
                ).to(device)
            else:
                from trainer.fresh_frame_fd_gradcache import (
                    GradNormEMABalancer,
                )

                self.frame_fd_balancer = GradNormEMABalancer(
                    fd_weight=float(getattr(args, "frame_fd_weight", 1.0)),
                    beta=float(getattr(
                        args, "frame_fd_balance_beta", 0.9)),
                    minimum=float(getattr(
                        args, "frame_fd_balance_min", 0.1)),
                    maximum=float(getattr(
                        args, "frame_fd_balance_max", 10.0)),
                    epsilon=float(getattr(
                        args, "frame_fd_balance_eps", 1.0e-12)),
                ).to(device)

    def offload_frozen_modules_for_generator_replay(self) -> int:
        """Offload frozen representation modules after their exact VJPs.

        This mirrors the proven VACE replay schedule: the complete-batch MMD
        is evaluated first, selected samples replay the frozen encoders and
        VAE to obtain detached latent cotangents, and only then are those
        frozen modules moved to CPU for the generator-only replay.
        """
        if not self.generator_replay_frozen_encoder_cpu_offload:
            return 0
        if self._generator_replay_frozen_encoder_offload_state is not None:
            raise RuntimeError("frozen replay modules are already offloaded")
        registry = getattr(self, "dual_branch_registry", None)
        if not isinstance(registry, Mapping) or not registry:
            raise RuntimeError("frozen replay requires an active encoder registry")
        module_names = ("vae",) + tuple(
            branch["encoder_attribute"] for branch in registry.values())
        if len(set(module_names)) != len(module_names):
            raise RuntimeError("frozen replay registry contains duplicate modules")
        original_device = torch.device(self.device)
        moved = []
        tensor_bytes = 0
        try:
            torch.cuda.synchronize(original_device)
            for module_name in module_names:
                module = getattr(self, module_name, None)
                if not isinstance(module, torch.nn.Module):
                    raise RuntimeError(
                        f"frozen replay module {module_name} is unavailable")
                if module.training or any(
                    parameter.requires_grad for parameter in module.parameters()
                ):
                    raise RuntimeError(
                        f"refusing to offload non-frozen module {module_name}")
                tensors = tuple(module.parameters()) + tuple(module.buffers())
                if not tensors:
                    raise RuntimeError(
                        f"frozen replay module {module_name} is not fully CUDA-resident")
                tensor_bytes += sum(
                    tensor.numel() * tensor.element_size() for tensor in tensors
                )
                module.to(device="cpu")
                moved.append(module_name)
            torch.cuda.synchronize(original_device)
            torch.cuda.empty_cache()
        except BaseException:
            for module_name in moved:
                getattr(self, module_name).to(device=original_device)
            torch.cuda.synchronize(original_device)
            raise
        self._generator_replay_frozen_encoder_offload_state = {
            "module_names": module_names,
            "original_device": str(original_device),
            "tensor_bytes": int(tensor_bytes),
        }
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[memory] replay offloaded frozen modules={module_names} "
                f"({tensor_bytes / (1024 ** 3):.2f} GiB tensors)",
                flush=True,
            )
        return int(tensor_bytes)

    def restore_frozen_modules_before_feature_replay(self) -> int:
        """Restore frozen modules before the next complete-batch feature pass."""
        state = self._generator_replay_frozen_encoder_offload_state
        if state is None:
            return 0
        original_device = torch.device(state["original_device"])
        for module_name in state["module_names"]:
            module = getattr(self, module_name, None)
            if not isinstance(module, torch.nn.Module):
                raise RuntimeError(
                    f"offloaded replay module {module_name} disappeared")
            tensors = tuple(module.parameters()) + tuple(module.buffers())
            if not tensors or any(
                tensor.device.type != "cpu" for tensor in tensors
            ):
                raise RuntimeError(
                    f"offloaded replay module {module_name} is not fully on CPU")
            module.to(device=original_device)
            module.eval().requires_grad_(False)
        torch.cuda.synchronize(original_device)
        tensor_bytes = int(state["tensor_bytes"])
        self._generator_replay_frozen_encoder_offload_state = None
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[memory] restored frozen modules={state['module_names']} "
                "for the next MMD feature/replay pass",
                flush=True,
            )
        return tensor_bytes

    @staticmethod
    def _canonical_sha256(value) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _canonical_identity_value(value):
        """Normalize an artifact identity through strict canonical JSON.

        Offline identities are loaded from JSON, where sequence shapes are
        lists. Live inspectors naturally return tuples for those same shapes.
        Comparing the two Python containers directly creates a false mismatch
        even though their canonical artifact identity is identical.
        """
        try:
            payload = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "artifact identity must contain only finite "
                "JSON-serializable values"
            ) from exc
        return json.loads(payload)

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _distributed_file_sha256(cls, path: str) -> str:
        """Hash a shared file once on rank zero and broadcast the result."""
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() == 1
        ):
            return cls._file_sha256(path)
        payload = [None]
        if dist.get_rank() == 0:
            try:
                payload[0] = {
                    "ok": True,
                    "sha256": cls._file_sha256(path),
                }
            except BaseException as exc:
                payload[0] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if not isinstance(result, dict) or not result.get("ok"):
            error = (
                result.get("error", "unknown error")
                if isinstance(result, dict)
                else "no rank-zero result"
            )
            raise RuntimeError(f"rank-zero file hashing failed: {error}")
        return str(result["sha256"])

    @classmethod
    def _runtime_file_identity(cls, path, *, file_hasher=None) -> dict:
        """Rebuild the offline tool's strict path/stat/content identity."""
        resolved = os.path.realpath(os.path.expanduser(os.fspath(path)))
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"runtime identity target is not a file: {resolved}")
        before = os.stat(resolved)
        hasher = file_hasher or cls._distributed_file_sha256
        digest = str(hasher(resolved))
        after = os.stat(resolved)
        if (
            int(before.st_size) != int(after.st_size)
            or int(before.st_mtime_ns) != int(after.st_mtime_ns)
        ):
            raise RuntimeError(
                f"runtime identity target changed while hashing: {resolved}")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"runtime identity returned an invalid SHA256 for {resolved}")
        return {
            "path": resolved,
            "size_bytes": int(after.st_size),
            "mtime_ns": int(after.st_mtime_ns),
            "sha256": digest,
        }

    @classmethod
    def _distributed_runtime_file_identity(cls, path) -> dict:
        """Build one shared-file identity on rank zero and publish it safely.

        Unlike calling ``_runtime_file_identity`` with a distributed hasher,
        nonzero ranks do not perform a fallible ``stat`` before entering the
        broadcast. Rank zero owns the complete stat/hash/stat transaction and
        broadcasts either its validated identity or a serialized failure.
        """
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() == 1
        ):
            return cls._runtime_file_identity(
                path, file_hasher=cls._file_sha256)
        payload = [None]
        if dist.get_rank() == 0:
            try:
                payload[0] = {
                    "ok": True,
                    "identity": cls._runtime_file_identity(
                        path, file_hasher=cls._file_sha256),
                }
            except BaseException as exc:
                payload[0] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if not isinstance(result, dict) or not result.get("ok"):
            error = (
                result.get("error", "unknown error")
                if isinstance(result, dict)
                else "no rank-zero result"
            )
            raise RuntimeError(
                f"rank-zero runtime file identity failed: {error}")
        identity = result.get("identity")
        if not isinstance(identity, dict):
            raise RuntimeError(
                "rank-zero runtime file identity returned invalid data")
        return dict(identity)

    def _validate_dinov3_reference_runtime_identity(
        self,
        args,
        reference_metadata: Mapping,
        *,
        inspect_lmdb=None,
        model_path_resolver=None,
        wrapper_source_path=None,
        file_hasher=None,
    ) -> dict:
        """Bind a DINOv3 reference to the active CausVid LMDB and Wan VAE.

        All fallible rank-local inspection completes and reaches consensus
        before any rank enters a shared-file hash broadcast. The large VAE
        files are then identified entirely on rank zero. Injection seams keep
        the artifact checks unit-testable without loading a real LMDB or VAE.
        """
        from pathlib import Path

        collective_device = getattr(self, "device", None)
        if collective_device is None:
            collective_device = torch.device("cpu")
        preflight = None
        preflight_error = None
        try:
            if not isinstance(reference_metadata, Mapping):
                raise TypeError("DINOv3 reference metadata must be a mapping")
            reference = reference_metadata.get("reference")
            if not isinstance(reference, Mapping):
                raise ValueError(
                    "DINOv3 reference metadata is missing reference identity")
            source = reference.get("source")
            saved_vae = reference.get("vae")
            if not isinstance(source, Mapping) or not isinstance(
                saved_vae, Mapping
            ):
                raise ValueError(
                    "DINOv3 reference must contain source and VAE identities")
            source = dict(source)
            saved_vae = dict(saved_vae)

            teacher_path = os.path.realpath(os.path.expanduser(str(getattr(
                args, "teacher_latent_data_path", ""))))
            if not teacher_path or not os.path.isdir(teacher_path):
                raise FileNotFoundError(
                    "DINOv3 reference validation requires the configured "
                    f"teacher LMDB directory; got {teacher_path!r}")
            active_inspector = inspect_lmdb
            if active_inspector is None:
                from tools.precompute_dinov3_dense_nystrom_reference import (
                    _inspect_lmdb,
                )

                active_inspector = _inspect_lmdb
            live_lmdb = dict(active_inspector(Path(teacher_path)))
            live_lmdb_path = os.path.realpath(str(
                live_lmdb.get("path", "")))
            live_latents_shape = [
                int(value)
                for value in live_lmdb.get("latents_shape", ())
            ]
            live_prompts_shape = [
                int(value)
                for value in live_lmdb.get("prompts_shape", ())
            ]
            if (
                live_lmdb_path != teacher_path
                or len(live_latents_shape) != 6
                or live_prompts_shape != [live_latents_shape[0]]
            ):
                raise ValueError(
                    "active teacher LMDB has an invalid CausVid array identity")
            teacher_count = int(getattr(
                args, "teacher_latent_count", live_latents_shape[0]))
            data_mdb_path = os.path.join(teacher_path, "data.mdb")
            if not os.path.isfile(data_mdb_path):
                raise FileNotFoundError(
                    f"teacher LMDB data file is missing: {data_mdb_path}")
            data_stat = os.stat(data_mdb_path)
            expected_source_keys = {
                "kind",
                "lmdb_path",
                "latents_shape",
                "prompts_shape",
                "selected_samples",
                "selected_index_start",
                "selected_index_stop_exclusive",
                "trajectory_index",
                "lmdb_storage_dtype",
                "data_mdb_size_bytes",
                "data_mdb_mtime_ns",
                "prompt_index_sha256",
                "feature_bank_fp16_sha256",
                "identity_sha256",
            }
            source_without_hash = dict(source)
            saved_source_hash = source_without_hash.pop(
                "identity_sha256", None)
            source_hash_valid = (
                isinstance(saved_source_hash, str)
                and saved_source_hash
                == self._canonical_sha256(source_without_hash)
            )
            source_sha_fields_valid = all(
                isinstance(source.get(key), str)
                and len(source[key]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in source[key]
                )
                for key in (
                    "prompt_index_sha256",
                    "feature_bank_fp16_sha256",
                )
            )
            if (
                set(source) != expected_source_keys
                or source.get("kind")
                != "causvid_ode_teacher_lmdb_final_latent"
                or os.path.realpath(str(source.get("lmdb_path", "")))
                != teacher_path
                or source.get("latents_shape") != live_latents_shape
                or source.get("prompts_shape") != live_prompts_shape
                or int(source.get("selected_samples", -1))
                != live_latents_shape[0]
                or live_latents_shape[0] != 1537
                or teacher_count != live_latents_shape[0]
                or int(source.get("selected_index_start", -1)) != 0
                or int(source.get("selected_index_stop_exclusive", -1))
                != live_latents_shape[0]
                or int(source.get("trajectory_index", -2)) != -1
                or source.get("lmdb_storage_dtype") != "float16"
                or int(source.get("data_mdb_size_bytes", -1))
                != int(data_stat.st_size)
                or int(source.get("data_mdb_mtime_ns", -1))
                != int(data_stat.st_mtime_ns)
                or int(live_lmdb.get("data_mdb_size_bytes", -1))
                != int(data_stat.st_size)
                or int(live_lmdb.get("data_mdb_mtime_ns", -1))
                != int(data_stat.st_mtime_ns)
                or not source_sha_fields_valid
                or not source_hash_valid
            ):
                raise ValueError(
                    "DINOv3 reference teacher LMDB identity does not match "
                    "the active CausVid dataset")

            active_model_path_resolver = model_path_resolver
            if active_model_path_resolver is None:
                from utils.paths import model_path

                active_model_path_resolver = model_path
            vae_model_name = "Wan2.1-T2V-1.3B"
            weights_path = (
                Path(active_model_path_resolver(vae_model_name))
                / "Wan2.1_VAE.pth"
            )
            active_wrapper_source_path = wrapper_source_path
            if active_wrapper_source_path is None:
                active_wrapper_source_path = (
                    Path(__file__).resolve().parents[1]
                    / "utils" / "wan_wrapper.py"
                )
            # Prove every rank can resolve/stat both identities before rank zero
            # enters either hash transaction.
            for identity_path in (weights_path, active_wrapper_source_path):
                resolved = os.path.realpath(os.path.expanduser(
                    os.fspath(identity_path)))
                if not os.path.isfile(resolved):
                    raise FileNotFoundError(
                        f"runtime identity target is not a file: {resolved}")
                os.stat(resolved)

            vae_mean = getattr(self.vae, "mean", None)
            vae_std = getattr(self.vae, "std", None)
            if not isinstance(vae_mean, torch.Tensor) or not isinstance(
                vae_std, torch.Tensor
            ):
                raise TypeError(
                    "active Wan VAE wrapper is missing tensor mean/std "
                    "constants")
            runtime_vae_base = {
                "wrapper": (
                    f"{type(self.vae).__module__}."
                    f"{type(self.vae).__qualname__}"),
                "model_name": vae_model_name,
                "latent_channels": int(vae_mean.numel()),
                "latent_mean": [
                    round(float(value), 4) for value in
                    vae_mean.detach().float().cpu().reshape(-1).tolist()
                ],
                "latent_std": [
                    round(float(value), 4) for value in
                    vae_std.detach().float().cpu().reshape(-1).tolist()
                ],
                "mean_std_contract": "WanVAEWrapper.inline_constants.v1",
                "compute_dtype": str(self.dtype).replace("torch.", ""),
                "decode_use_cache": False,
                "decode_output_range": "[-1,1]",
            }
            preflight = {
                "saved_source_hash": saved_source_hash,
                "saved_vae": saved_vae,
                "weights_path": weights_path,
                "wrapper_source_path": active_wrapper_source_path,
                "runtime_vae_base": runtime_vae_base,
            }
        except BaseException as exc:
            preflight_error = exc
        self._synchronize_dinov3_local_failure(
            preflight_error,
            collective_device,
            operation="DINOv3 runtime identity preflight failed",
        )

        identity_error = None
        weights_identity = None
        wrapper_identity = None
        try:
            if file_hasher is None:
                weights_identity = self._distributed_runtime_file_identity(
                    preflight["weights_path"])
                wrapper_identity = self._distributed_runtime_file_identity(
                    preflight["wrapper_source_path"])
            else:
                weights_identity = self._runtime_file_identity(
                    preflight["weights_path"], file_hasher=file_hasher)
                wrapper_identity = self._runtime_file_identity(
                    preflight["wrapper_source_path"],
                    file_hasher=file_hasher,
                )
        except BaseException as exc:
            identity_error = exc
        self._synchronize_dinov3_local_failure(
            identity_error,
            collective_device,
            operation="DINOv3 shared-file identity failed",
        )

        receipt = None
        comparison_error = None
        try:
            runtime_vae = dict(preflight["runtime_vae_base"])
            runtime_vae["wrapper_source"] = wrapper_identity
            runtime_vae["weights"] = weights_identity
            runtime_vae["identity_sha256"] = self._canonical_sha256(
                runtime_vae)
            if preflight["saved_vae"] != runtime_vae:
                raise ValueError(
                    "DINOv3 reference Wan VAE identity does not match the "
                    "active decoder")
            receipt = {
                "source_identity_sha256": preflight["saved_source_hash"],
                "vae_identity_sha256": runtime_vae["identity_sha256"],
            }
        except BaseException as exc:
            comparison_error = exc
        self._synchronize_dinov3_local_failure(
            comparison_error,
            collective_device,
            operation="DINOv3 runtime identity comparison failed",
        )
        return receipt

    @classmethod
    def _resolve_vjepa2_model_configuration(
        cls,
        args,
    ) -> Tuple[str, str, str]:
        """Resolve the load source independently from reference metadata.

        vjepa2_model_id identifies the model that transformers actually opens.
        vjepa2_reference_model_id is the canonical identity stored in an
        already-created feature/reference pack; it can legitimately be a path
        from the machine that produced that pack. Keeping the two separate
        lets a byte-identical local mirror satisfy strict reference metadata
        without asking transformers to open the producer's path.

        When configured, vjepa2_model_sha256 authenticates the local
        model.safetensors before the adapter is constructed.
        """
        configured_load_id = str(getattr(
            args,
            "vjepa2_model_id",
            "facebook/vjepa2-vitl-fpc64-256",
        )).strip()
        if not configured_load_id:
            raise ValueError("vjepa2_model_id must be a non-empty string")
        expanded_load_id = os.path.expanduser(configured_load_id)
        model_load_path = (
            os.path.realpath(expanded_load_id)
            if os.path.exists(expanded_load_id)
            else configured_load_id
        )

        configured_reference_id = getattr(
            args, "vjepa2_reference_model_id", None)
        reference_model_id = (
            "" if configured_reference_id is None
            else str(configured_reference_id).strip()
        )
        if not reference_model_id:
            # Backward compatibility: old configs used one field for both
            # loading and canonical reference metadata.
            reference_model_id = model_load_path

        expected_sha256 = str(getattr(
            args, "vjepa2_model_sha256", "")).strip().lower()
        if expected_sha256:
            if (
                len(expected_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_sha256
                )
            ):
                raise ValueError(
                    "vjepa2_model_sha256 must be a 64-character SHA-256")
            if not os.path.isdir(model_load_path):
                raise ValueError(
                    "vjepa2_model_sha256 requires vjepa2_model_id to be a "
                    "local model directory")
            weight_path = os.path.join(
                model_load_path, "model.safetensors")
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(
                    "vjepa2_model_sha256 requires the local weight file "
                    f"{weight_path}")
            actual_sha256 = cls._distributed_file_sha256(weight_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "V-JEPA2 model.safetensors SHA-256 mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}")

        return model_load_path, reference_model_id, expected_sha256

    @staticmethod
    def _distributed_representation_identity(module) -> dict:
        """Hash one replicated encoder once and share its strict identity."""
        from model.fd import representation_runtime_identity

        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() == 1
        ):
            return representation_runtime_identity(module)
        payload = [None]
        if dist.get_rank() == 0:
            try:
                payload[0] = {
                    "ok": True,
                    "identity": representation_runtime_identity(module),
                }
            except BaseException as exc:
                payload[0] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if not isinstance(result, dict) or not result.get("ok"):
            error = (
                result.get("error", "unknown error")
                if isinstance(result, dict)
                else "no rank-zero result"
            )
            raise RuntimeError(
                f"rank-zero representation hashing failed: {error}")
        return dict(result["identity"])

    @staticmethod
    def _config_list(value) -> list:
        if isinstance(value, str):
            return [value]
        return list(value)

    def _dense_config(self, args, suffix: str, default):
        """Read an encoder-scoped dense option without changing old YAMLs."""
        return getattr(
            args, f"{self._dense_encoder_kind()}_{suffix}", default)

    def _dense_key(self, suffix: str) -> str:
        prefix = getattr(
            self,
            "dense_log_prefix",
            f"{self._dense_encoder_kind()}_dense",
        )
        return f"{prefix}_{suffix}"

    def _dense_encoder_kind(self) -> str:
        return getattr(
            self,
            "dense_encoder_name",
            (
                "motionformer"
                if getattr(self, "feature_type", None)
                == "motionformer_dense_latent"
                else "vjepa2"
            ),
        )

    def _dense_encoder_display_name(self) -> str:
        return (
            "MotionFormer"
            if self._dense_encoder_kind() == "motionformer"
            else "V-JEPA2"
        )

    def _dense_attr(self, suffix: str):
        generic_name = f"dense_{suffix}"
        if hasattr(self, generic_name):
            return getattr(self, generic_name)
        return getattr(self, f"vjepa2_dense_{suffix}")

    def _dense_bank(self):
        bank = getattr(self, "dense_reference_bank", None)
        if bank is None:
            bank = getattr(self, "vjepa2_reference_bank", None)
        return bank

    def _dense_ema_version_key(self) -> str:
        return self._dense_key("mmd_version")

    def _dense_ema_state_key(self) -> str:
        return self._dense_key("mmd")

    def _dense_nystrom_version_key(self) -> str:
        return self._dense_key("nystrom_mmd_version")

    def _validate_vjepa2_reference_bank_runtime_identity(self, args) -> None:
        """Bind an offline FiDEL bank to every live artifact that created it.

        The compact bank intentionally does not retain the full teacher or
        student feature matrices.  Consequently, accepting a bank whose
        encoder, VAE, teacher LMDB, generator, text stack, or effective YAML
        changed would silently make its floors and start scores meaningless.
        Hash the large shared artifacts once on rank zero and broadcast only
        success/failure so every rank enters training with the same verdict.
        """
        if self.vjepa2_reference_bank is None:
            raise RuntimeError("reference-bank identity validation needs a bank")

        payload = [None]
        is_distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        is_rank_zero = not is_distributed or dist.get_rank() == 0
        if is_rank_zero:
            try:
                from pathlib import Path
                from types import SimpleNamespace

                from tools.precompute_vjepa2_fidel_reference_bank import (
                    EXPECTED_TEACHERS,
                    MASK_SEED,
                    SPLIT_SEED,
                    STUDENT_BATCHES,
                    STUDENT_SAMPLES,
                    _effective_student_config,
                    _file_identity,
                    _inspect_lmdb,
                    _model_identity,
                    _text_conditioning_identity,
                )
                from utils.paths import model_path

                metadata = self.vjepa2_reference_bank.metadata
                config_path = Path(str(
                    getattr(args, "config_path", "")
                )).expanduser().resolve()
                if not config_path.is_file():
                    raise FileNotFoundError(
                        "strict FiDEL runtime validation requires the exact "
                        f"config_path used for precompute; got {config_path}")
                generator_path = Path(str(
                    getattr(args, "generator_ckpt", "")
                )).expanduser().resolve()
                teacher_path = Path(str(
                    getattr(args, "teacher_latent_data_path", "")
                )).expanduser().resolve()
                vae_path = (
                    model_path(str(getattr(
                        args, "fake_name", "Wan2.1-T2V-1.3B"
                    ))) / "Wan2.1_VAE.pth"
                ).expanduser().resolve()

                teacher_identity = _inspect_lmdb(teacher_path)
                teacher_identity["data_mdb_sha256"] = _file_identity(
                    teacher_path / "data.mdb")["sha256"]
                _, config_identity = _effective_student_config(
                    SimpleNamespace(
                        student_config=config_path,
                        student_checkpoint=generator_path,
                        teacher_lmdb=teacher_path,
                    )
                )
                expected = self._canonical_identity_value({
                    "vjepa2_model": _model_identity(self.vjepa2_model_id),
                    "vae_identity": _file_identity(vae_path),
                    "teacher_lmdb_identity": teacher_identity,
                    "generator_checkpoint_identity":
                        _file_identity(generator_path),
                    "generator_effective_config_identity": config_identity,
                    "text_conditioning_identity":
                        _text_conditioning_identity(),
                })
                aliases = {
                    "vae": "vae_identity",
                    "teacher_lmdb": "teacher_lmdb_identity",
                    "student_checkpoint":
                        "generator_checkpoint_identity",
                    "student_config":
                        "generator_effective_config_identity",
                }
                mismatched = {
                    key: {
                        "saved": metadata.get(key),
                        "runtime": value,
                    }
                    for key, value in expected.items()
                    if metadata.get(key) != value
                }
                for alias, canonical in aliases.items():
                    if metadata.get(alias) != expected[canonical]:
                        mismatched[alias] = {
                            "saved": metadata.get(alias),
                            "runtime": expected[canonical],
                        }

                token_masks = metadata.get("token_masks", {})
                configured_mask_count = int(getattr(
                    args,
                    "vjepa2_token_mask_count",
                    self.vjepa2_reference_bank.slot_count,
                ))
                fixed_contract = {
                    "teacher_count": EXPECTED_TEACHERS,
                    "student_init_count": STUDENT_SAMPLES,
                    "student_init_batches": STUDENT_BATCHES,
                    "split_seed": SPLIT_SEED,
                    "token_mask_seed": MASK_SEED,
                }
                for key, expected_value in fixed_contract.items():
                    if metadata.get(key) != expected_value:
                        mismatched[key] = {
                            "saved": metadata.get(key),
                            "runtime": expected_value,
                        }
                if (
                    not isinstance(token_masks, Mapping)
                    or token_masks.get("seed") != MASK_SEED
                    or int(token_masks.get("count", -1))
                    != configured_mask_count
                    or int(token_masks.get("count", -1))
                    != int(self.vjepa2_reference_bank.slot_count)
                    or str(getattr(
                        args, "vjepa2_token_mask_seed", ""
                    )) != MASK_SEED
                ):
                    mismatched["token_masks"] = {
                        "saved": token_masks,
                        "runtime": {
                            "seed": MASK_SEED,
                            "count": configured_mask_count,
                            "config_seed": str(getattr(
                                args, "vjepa2_token_mask_seed", "")),
                        },
                    }
                if mismatched:
                    raise ValueError(
                        "FiDEL reference bank does not match live training "
                        f"artifacts/configuration: {mismatched}")
                payload[0] = {"ok": True}
            except BaseException as exc:
                payload[0] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if is_distributed:
            dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if not isinstance(result, dict) or not result.get("ok"):
            error = (
                result.get("error", "unknown error")
                if isinstance(result, dict)
                else "rank zero returned no identity result"
            )
            raise RuntimeError(
                "strict V-JEPA2 FiDEL reference validation failed: "
                f"{error}")

    def _load_and_validate_frame_fd_manifest(self, args) -> dict:
        """Validate that runtime SIM-FD settings exactly match teacher stats."""
        configured_path = str(getattr(
            args, "frame_fd_stats_manifest_path", "")).strip()
        manifest_path = (
            os.path.realpath(os.path.expanduser(configured_path))
            if configured_path else "")
        if not manifest_path or not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                "frame FD fusion requires frame_fd_stats_manifest_path; got "
                f"{manifest_path!r}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("frame FD reference manifest must be a mapping")
        saved_manifest_hash = manifest.get("manifest_sha256")
        unhashed = dict(manifest)
        unhashed.pop("manifest_sha256", None)
        computed_manifest_hash = self._canonical_sha256(unhashed)
        if saved_manifest_hash != computed_manifest_hash:
            raise ValueError(
                "frame FD reference manifest_sha256 does not match contents")
        expected_manifest_version = (
            4 if self.fidel_two_level_enabled else 3)
        if (
            manifest.get("format") != "self_forcing_frame_fd_stats"
            or int(manifest.get("format_version", -1))
            != expected_manifest_version
            or manifest.get("feature_aggregation") != "frame"
        ):
            raise ValueError(
                "frame FD reference must be a "
                f"v{expected_manifest_version} frame-aggregation manifest")
        if getattr(self, "fidel_two_level_enabled", False):
            # Reuse the exact pure-data validator used by the precompute tool;
            # this proves the floor/start payload and all bundle hashes before
            # any values are admitted to the live controller.
            from tools.precompute_frame_fd_stats import (
                validate_fidel_manifest_v4,
            )

            validate_fidel_manifest_v4(manifest)

        sampling = manifest.get("sampling", {})
        expected_frames = int(getattr(args, "fd_num_frames", 16))
        expected_sampling = (
            "random"
            if bool(getattr(args, "fd_random_frame_sample", True))
            else "uniform"
        )
        if (
            sampling.get("method") != expected_sampling
            or int(sampling.get("frames_per_video", -1)) != expected_frames
            or sampling.get("without_replacement") is not True
            or sampling.get("shared_across_representations") is not True
            or bool(sampling.get("exclude_first_frame", False))
            != bool(getattr(args, "fd_exclude_first_frame", False))
            or int(sampling.get("fd_decode_latent_frames", -1)) != int(
                getattr(args, "fd_decode_frames", 0))
        ):
            raise ValueError(
                "frame FD teacher sampling does not match the active config")

        source = manifest.get("source", {})
        teacher_path = os.path.realpath(os.path.expanduser(str(
            getattr(args, "teacher_latent_data_path", ""))))
        source_selection_ok = (
            int(source.get("selected_video_count", -1))
            == int(source.get("actual_video_count", -2))
        )
        if getattr(self, "fidel_two_level_enabled", False):
            bank_metadata = self.vjepa2_reference_bank.metadata
            source_selection_ok = bool(
                int(source.get("teacher_count", -1))
                == int(bank_metadata["teacher_count"])
                == int(source.get("actual_video_count", -2))
                and int(source.get("train_video_count", -1))
                == len(self.vjepa2_reference_bank.train_indices)
                == int(source.get("selected_video_count", -2))
                and tuple(source.get("train_indices", ()))
                == self.vjepa2_reference_bank.train_indices
                and source.get("train_indices_sha256")
                == bank_metadata["train_indices_sha256"]
            )
        if (
            source.get("type") != "teacher_lmdb"
            or os.path.realpath(str(source.get("path", ""))) != teacher_path
            or source.get("trajectory_selection") != "last"
            or not source_selection_ok
        ):
            detail = (
                "the exact teacher-train split shared with the V-JEPA2 "
                "reference bank"
                if self.fidel_two_level_enabled
                else "the full active teacher LMDB"
            )
            raise ValueError(
                "frame FD reference must cover "
                f"{detail} using each trajectory's final latent")
        data_mdb_path = os.path.join(teacher_path, "data.mdb")
        if not os.path.isfile(data_mdb_path):
            raise FileNotFoundError(
                f"teacher LMDB data file is missing: {data_mdb_path}")
        data_identity = source.get("data_mdb", {})
        data_stat = os.stat(data_mdb_path)
        data_sha256 = self._distributed_file_sha256(data_mdb_path)
        if (
            os.path.realpath(str(data_identity.get("path", "")))
            != os.path.realpath(data_mdb_path)
            or int(data_identity.get("size_bytes", -1))
            != int(data_stat.st_size)
            or int(data_identity.get("mtime_ns", -1))
            != int(data_stat.st_mtime_ns)
            or data_identity.get("sha256") != data_sha256
            or data_identity.get("identity_sha256")
            != self._canonical_sha256({
                "path": os.path.realpath(data_mdb_path),
                "size_bytes": int(data_stat.st_size),
                "mtime_ns": int(data_stat.st_mtime_ns),
                "sha256": data_sha256,
            })
        ):
            raise ValueError(
                "teacher LMDB identity no longer matches the frame FD manifest")
        expected_source_identity = self._canonical_sha256({
            "type": "teacher_lmdb",
            "path": teacher_path,
            "latents_shape": source.get("latents_shape"),
            "prompts_shape": source.get("prompts_shape"),
            "actual_video_count": int(source.get("actual_video_count", -1)),
            "trajectory_selection": "last",
            "data_mdb": data_identity,
        })
        if source.get("identity_sha256") != expected_source_identity:
            raise ValueError(
                "frame FD teacher source identity hash is invalid")
        if source.get("decoded_frame_counts") != [81]:
            raise ValueError(
                "frame FD teacher reference must contain 81-frame videos")

        names = self._config_list(getattr(args, "fd_repr_models", ()))
        pools = self._config_list(getattr(args, "fd_repr_pool_types", ()))
        sizes = self._config_list(getattr(args, "fd_target_sizes", ()))
        stats_paths = [
            os.path.realpath(os.path.expanduser(str(path)))
            for path in self._config_list(
                getattr(args, "fd_repr_stats_paths", ()))
        ]
        entries = manifest.get("representations")
        if (
            not isinstance(entries, list)
            or not (
                len(entries)
                == len(names)
                == len(pools)
                == len(sizes)
                == len(stats_paths)
                == len(self.frame_fd_head.judges)
            )
        ):
            raise ValueError(
                "frame FD representation lists do not match the manifest")
        selected_videos = int(source["selected_video_count"])
        compact_entries = []
        for entry, name, pool, size, stats_path, judge in zip(
            entries,
            names,
            pools,
            sizes,
            stats_paths,
            self.frame_fd_head.judges,
        ):
            if not os.path.isfile(stats_path):
                raise FileNotFoundError(
                    f"frame FD statistics file not found: {stats_path}")
            stats_hash = self._distributed_file_sha256(stats_path)
            encoder_identity = self._distributed_representation_identity(
                judge.repr_model)
            if (
                entry.get("key") != judge.name
                or entry.get("model_name") != str(name)
                or entry.get("pool_type") != str(pool)
                or int(entry.get("target_size", -1)) != int(size)
                or int(entry.get("feature_dim", -1))
                != int(judge.queue.feat_dim)
                or int(entry.get("frame_sample_count", -1))
                != selected_videos * expected_frames
                or os.path.basename(stats_path) != entry.get("stats_file")
                or stats_hash != entry.get("stats_sha256")
                or encoder_identity != entry.get("encoder_identity")
            ):
                raise ValueError(
                    f"frame FD representation {judge.name!r} does not match "
                    "the frozen reference manifest")
            compact_entries.append({
                "key": judge.name,
                "model_name": str(name),
                "pool_type": str(pool),
                "target_size": int(size),
                "feature_dim": int(judge.queue.feat_dim),
                "stats_path": stats_path,
                "stats_sha256": stats_hash,
                "encoder_identity": encoder_identity,
                **({
                    "real_floor": float(entry["real_floor"]),
                    "student_start_score": float(
                        entry["student_start_score"]),
                    "student_start_std": float(
                        entry["student_start_std"]),
                } if self.fidel_two_level_enabled else {}),
            })

        vae = manifest.get("vae", {})
        expected_vae_name = str(getattr(
            args, "fake_name", "Wan2.1-T2V-1.3B"))
        from utils.paths import model_path

        expected_weights_path = os.path.realpath(str(
            model_path(expected_vae_name) / "Wan2.1_VAE.pth"))
        weights_identity = vae.get("weights", {})
        weights_path = os.path.realpath(str(
            weights_identity.get("path", "")))
        if (
            vae.get("model_name") != expected_vae_name
            or weights_path != expected_weights_path
            or not os.path.isfile(weights_path)
        ):
            raise ValueError(
                "frame FD manifest VAE identity does not match the active Wan "
                "VAE")
        weights_stat = os.stat(weights_path)
        weights_sha256 = self._distributed_file_sha256(weights_path)
        expected_vae_dtype = (
            "bfloat16"
            if bool(getattr(args, "mixed_precision", False))
            else "float32"
        )
        if (
            int(weights_identity.get("size_bytes", -1))
            != int(weights_stat.st_size)
            or int(weights_identity.get("mtime_ns", -1))
            != int(weights_stat.st_mtime_ns)
            or weights_identity.get("sha256") != weights_sha256
            or vae.get("compute_dtype") != expected_vae_dtype
        ):
            raise ValueError(
                "Wan VAE weights no longer match the frame FD manifest")
        expected_weights_identity = {
            "path": weights_path,
            "size_bytes": int(weights_stat.st_size),
            "mtime_ns": int(weights_stat.st_mtime_ns),
            "sha256": weights_sha256,
        }
        expected_weights_identity["identity_sha256"] = (
            self._canonical_sha256(expected_weights_identity))
        expected_vae_identity = {
            "wrapper": "utils.wan_wrapper.WanVAEWrapper",
            "model_name": expected_vae_name,
            "compute_dtype": expected_vae_dtype,
            "weights": expected_weights_identity,
            "decode_output_range": "[-1,1]",
        }
        expected_vae_identity["identity_sha256"] = self._canonical_sha256(
            expected_vae_identity)
        if vae != expected_vae_identity:
            raise ValueError(
                "frame FD manifest VAE identity hash is invalid")

        encoder_inference = manifest.get("encoder_inference", {})
        expected_encoder_inference = {
            "autocast_enabled": bool(
                self.frame_fd_head.replay_use_amp),
            "compute_dtype": (
                str(self.frame_fd_head.replay_amp_dtype).replace(
                    "torch.", "")
                if self.frame_fd_head.replay_use_amp
                else "float32"
            ),
            "output_dtype": "float32",
        }
        if encoder_inference != expected_encoder_inference:
            raise ValueError(
                "frame FD encoder inference dtype does not match the frozen "
                "teacher reference")

        if getattr(self, "fidel_two_level_enabled", False):
            expected_bundle_payload = {
                "source_identity_sha256": source.get("identity_sha256"),
                "teacher_count": int(source["teacher_count"]),
                "train_indices_sha256": source[
                    "train_indices_sha256"],
                "selected_rows_sha256": source.get(
                    "selected_rows_sha256"),
                "frame_indices_sha256": source.get(
                    "frame_indices_sha256"),
                "vae_identity_sha256": vae.get("identity_sha256"),
                "sampling": sampling,
                "encoder_inference": encoder_inference,
                "fidel_score_payload_sha256": manifest[
                    "fidel_score_payload_sha256"],
                "stats": [
                    {
                        "key": item["key"],
                        "sha256": item["stats_sha256"],
                        "encoder_state_sha256": item[
                            "encoder_identity"]["state_sha256"],
                        "real_floor": item["real_floor"],
                        "student_start_score": item[
                            "student_start_score"],
                        "student_start_std": item[
                            "student_start_std"],
                    }
                    for item in compact_entries
                ],
            }
        else:
            expected_bundle_payload = {
                "source_identity_sha256": source.get("identity_sha256"),
                "selected_rows_sha256": source.get(
                    "selected_rows_sha256"),
                "frame_indices_sha256": source.get(
                    "frame_indices_sha256"),
                "vae_identity_sha256": vae.get("identity_sha256"),
                "sampling": sampling,
                "encoder_inference": encoder_inference,
                "stats": [
                    {
                        "key": item["key"],
                        "sha256": item["stats_sha256"],
                        "encoder_state_sha256": item[
                            "encoder_identity"]["state_sha256"],
                    }
                    for item in compact_entries
                ],
            }
        expected_bundle_hash = self._canonical_sha256(
            expected_bundle_payload)
        if manifest.get("reference_bundle_sha256") != expected_bundle_hash:
            raise ValueError(
                "frame FD reference_bundle_sha256 does not match its teacher, "
                "sampling, VAE and statistics identities")

        fidel_baseline_provenance = None
        if getattr(self, "fidel_two_level_enabled", False):
            provenance = manifest.get("score_provenance")
            bank_metadata = self.vjepa2_reference_bank.metadata
            latent_manifest_identity = bank_metadata.get(
                "student_init_latent_bundle_manifest")
            if (
                not isinstance(provenance, dict)
                or not isinstance(latent_manifest_identity, dict)
            ):
                raise ValueError(
                    "FiDEL frame scores are missing V-JEPA2 student-baseline "
                    "provenance")
            expected_provenance = {
                "reference_bank_manifest_path": os.path.realpath(str(
                    self.vjepa2_reference_bank.manifest_path)),
                "reference_bank_manifest_sha256":
                    self._distributed_file_sha256(str(
                        self.vjepa2_reference_bank.manifest_path)),
                "reference_bank_content_hash":
                    self.vjepa2_reference_bank.manifest_hash,
                "student_init_latent_bundle_sha256":
                    bank_metadata.get(
                        "student_init_latent_bundle_sha256"),
                "student_init_latent_manifest_sha256":
                    latent_manifest_identity.get("sha256"),
                "student_init_latent_manifest_content_hash":
                    latent_manifest_identity.get("content_hash"),
                "generator_checkpoint_sha256":
                    bank_metadata.get("generator_checkpoint_sha256"),
                "text_encoder_sha256":
                    bank_metadata.get(
                        "text_conditioning_identity", {}
                    ).get("encoder_weights", {}).get("sha256"),
                "tokenizer_aggregate_sha256":
                    bank_metadata.get(
                        "text_conditioning_identity", {}
                    ).get("tokenizer", {}).get("aggregate_sha256"),
            }
            mismatched = {
                key: (provenance.get(key), expected)
                for key, expected in expected_provenance.items()
                if provenance.get(key) != expected
            }
            if mismatched:
                raise ValueError(
                    "frame FiDEL student-start scores were not measured from "
                    "the exact 512-latent baseline used by the V-JEPA2 "
                    f"reference bank: {mismatched}")
            fidel_baseline_provenance = expected_provenance

        metadata = {
            "manifest_path": manifest_path,
            "manifest_sha256": saved_manifest_hash,
            "reference_bundle_sha256": manifest.get(
                "reference_bundle_sha256"),
            "teacher_source_identity_sha256": source.get(
                "identity_sha256"),
            "teacher_selected_rows_sha256": source.get(
                "selected_rows_sha256"),
            "teacher_frame_indices_sha256": source.get(
                "frame_indices_sha256"),
            "teacher_video_count": selected_videos,
            **({
                "teacher_count": int(source["teacher_count"]),
                "train_indices_sha256": source[
                    "train_indices_sha256"],
                "val_indices_sha256": (
                    self.vjepa2_reference_bank.metadata[
                        "val_indices_sha256"]),
                "fidel_score_payload_sha256": manifest[
                    "fidel_score_payload_sha256"],
                "student_baseline_provenance":
                    fidel_baseline_provenance,
            } if self.fidel_two_level_enabled else {}),
            "teacher_frame_samples_per_representation": (
                selected_videos * expected_frames),
            "vae_identity_sha256": vae.get("identity_sha256"),
            "sampling": sampling,
            "encoder_inference": encoder_inference,
            "representations": compact_entries,
        }
        if (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        ):
            print(
                "[frame-fd] loaded frozen reference "
                f"videos={selected_videos}, "
                f"frames/encoder={selected_videos * expected_frames}, "
                f"hash={metadata['reference_bundle_sha256']}, "
                "encoder_dtype="
                f"{encoder_inference['compute_dtype']}",
                flush=True,
            )
        return metadata

    def _initialize_wan_features(self, args) -> None:
        if self.real_score is None:
            raise ValueError(
                "post_block_latent requires uses_real_score: true")
        self.latent_layer = int(getattr(args, "wan_latent_layer", 15))
        num_encoder_blocks = len(self.real_score.model.blocks)
        if self.latent_layer < 0 or self.latent_layer >= num_encoder_blocks:
            raise ValueError(
                f"wan_latent_layer={self.latent_layer} outside frozen Wan's "
                f"0..{num_encoder_blocks - 1} range")
        if bool(getattr(args, "wan_latent_prune_unused_blocks", True)):
            # The feature path returns after the requested block, so later
            # frozen blocks can never be used.  Removing them before FSDP avoids
            # sharding/loading another ~650M parameters for layers 16..29.
            keep = self.latent_layer + 1
            self.real_score.model.blocks = self.real_score.model.blocks[:keep]
        self.latent_num_tokens = int(
            getattr(args, "wan_latent_num_tokens", 128))
        if self.latent_num_tokens <= 1:
            raise ValueError("wan_latent_num_tokens must be > 1")
        self.latent_timestep = float(
            getattr(args, "wan_latent_timestep", 0.0))
        self.latent_patch_size = tuple(
            int(v) for v in self.real_score.model.patch_size)
        self.latent_feature_dim = int(self.real_score.model.dim)
        # We need input gradients through the frozen encoder, and checkpointing is
        # the main activation-memory control for deeper prefixes.
        if bool(getattr(args, "wan_latent_gradient_checkpointing", True)):
            self.real_score.enable_gradient_checkpointing()

    def _initialize_vjepa2_features(self, args, device) -> None:
        # Lazy import keeps the transformers/V-JEPA2 dependency completely out
        # of the unchanged Wan feature path.
        from video_fd.encoders.vjepa2_adapter import VJEPA2Adapter

        (
            self.vjepa2_model_load_path,
            self.vjepa2_reference_model_id,
            self.vjepa2_model_sha256,
        ) = self._resolve_vjepa2_model_configuration(args)
        # Retain the historical runtime attribute for reference-bank identity
        # checks and downstream code that expects the actual load source.
        self.vjepa2_model_id = self.vjepa2_model_load_path
        self.vjepa2_reference_token_contract = str(getattr(
            args,
            "vjepa2_reference_token_contract",
            "explicit_uniform_v2",
        )).strip()
        if self.vjepa2_reference_token_contract not in {
            "explicit_uniform_v2",
            "legacy_uniform_implicit_v1",
        }:
            raise ValueError(
                "vjepa2_reference_token_contract must be "
                "explicit_uniform_v2 or legacy_uniform_implicit_v1")
        configured_dense_layers = getattr(
            args, "vjepa2_dense_layers", None)
        if configured_dense_layers is None:
            dense_layers = (
                int(getattr(args, "vjepa2_dense_layer", 15)),
            )
        else:
            if isinstance(configured_dense_layers, (str, bytes)):
                raise TypeError(
                    "vjepa2_dense_layers must be a sequence of integers")
            dense_layers = tuple(
                int(layer) for layer in configured_dense_layers)
            if not dense_layers:
                raise ValueError(
                    "vjepa2_dense_layers must not be empty")
        if len(set(dense_layers)) != len(dense_layers):
            raise ValueError(
                "vjepa2_dense_layers must contain unique indices")
        if tuple(sorted(dense_layers)) != dense_layers:
            raise ValueError(
                "vjepa2_dense_layers must be sorted")
        self.vjepa2_dense_layers = dense_layers
        self.vjepa2_multilayer_enabled = len(dense_layers) > 1
        # The maximum layer controls prefix pruning. Single-layer callers keep
        # the historical scalar attribute unchanged.
        self.vjepa2_dense_layer = dense_layers[-1]
        self.vjepa2_dense_num_frames = int(
            getattr(args, "vjepa2_dense_num_frames", 64))
        self.vjepa2_dense_frame_sampling = str(getattr(
            args, "vjepa2_dense_frame_sampling", "center_contiguous"))
        # Keep the immutable reference contract separate from the student
        # temporal window.  The historical ``last_contiguous`` alias denotes
        # matched5k references that were actually extracted from frames 17..80;
        # explicit ``center_contiguous`` + student ``last_contiguous`` is the
        # legacy OpenVid-77,615 center64-reference protocol.
        self.vjepa2_reference_frame_sampling = (
            self.vjepa2_dense_frame_sampling)
        self.vjepa2_dense_source_num_frames = int(getattr(
            args,
            "vjepa2_dense_source_num_frames",
            self.vjepa2_dense_num_frames,
        ))
        configured_student_frame_sampling = getattr(
            args, "vjepa2_student_frame_sampling", None)
        if self.vjepa2_dense_frame_sampling == "last_contiguous":
            if (
                configured_student_frame_sampling is not None
                and str(configured_student_frame_sampling) != "last_contiguous"
            ):
                raise ValueError(
                    "vjepa2_dense_frame_sampling=last_contiguous conflicts "
                    "with vjepa2_student_frame_sampling")
            # The dense contract identifies the frozen center64 reference.
            # Treat this alias as a student-only last64 request.
            self.vjepa2_dense_frame_sampling = "center_contiguous"
            self.vjepa2_student_frame_sampling = "last_contiguous"
        else:
            self.vjepa2_student_frame_sampling = str(
                configured_student_frame_sampling
                if configured_student_frame_sampling is not None
                else self.vjepa2_dense_frame_sampling
            )
        default_student_source_frames = (
            81
            if self.vjepa2_student_frame_sampling == "last_contiguous"
            else self.vjepa2_dense_source_num_frames
        )
        self.vjepa2_student_source_num_frames = int(getattr(
            args, "vjepa2_student_source_num_frames",
            default_student_source_frames,
        ))
        self.vjepa2_dense_window_seed = str(getattr(
            args, "vjepa2_dense_window_seed", "sf-vjepa2-window-v1"))
        self.vjepa2_dense_num_tokens = int(
            getattr(args, "vjepa2_dense_num_tokens", 128))
        self.vjepa2_dense_resize_short_side = int(
            getattr(args, "vjepa2_dense_resize_short_side", 292))
        self.vjepa2_dense_crop_size = int(
            getattr(args, "vjepa2_dense_crop_size", 256))
        self.vjepa2_dense_gradient_checkpointing = bool(
            getattr(args, "vjepa2_dense_gradient_checkpointing", True))

        if any(layer < 0 for layer in self.vjepa2_dense_layers):
            raise ValueError(
                "V-JEPA2 dense layer indices must be non-negative")
        if self.vjepa2_dense_num_frames <= 0:
            raise ValueError("vjepa2_dense_num_frames must be positive")
        if self.vjepa2_dense_frame_sampling not in {
            "center_contiguous",
            "random_contiguous_per_video",
        }:
            raise ValueError(
                "vjepa2_dense_frame_sampling must be center_contiguous or "
                "random_contiguous_per_video")
        if self.vjepa2_student_frame_sampling not in {
            "center_contiguous",
            "random_contiguous_per_video",
            "last_contiguous",
        }:
            raise ValueError(
                "vjepa2_student_frame_sampling must be center_contiguous, "
                "random_contiguous_per_video, or last_contiguous")
        if (
            self.vjepa2_student_frame_sampling
            != self.vjepa2_dense_frame_sampling
            and self.vjepa2_student_frame_sampling != "last_contiguous"
        ):
            raise ValueError(
                "only last_contiguous may intentionally differ from the "
                "frozen-reference frame sampling")
        if (
            self.vjepa2_student_frame_sampling == "last_contiguous"
            and self.vjepa2_dense_frame_sampling != "center_contiguous"
        ):
            raise ValueError("last_contiguous requires a center64 reference")
        if (
            self.vjepa2_student_source_num_frames
            < self.vjepa2_dense_num_frames
        ):
            raise ValueError(
                "vjepa2_student_source_num_frames must be at least "
                "vjepa2_dense_num_frames")
        if self.vjepa2_student_frame_sampling == "last_contiguous" and (
            self.vjepa2_dense_num_frames != 64
            or self.vjepa2_student_source_num_frames != 81
        ):
            raise ValueError(
                "last_contiguous is the strict 81-frame last64 contract and "
                "requires vjepa2_dense_num_frames=64 plus "
                "vjepa2_student_source_num_frames=81")
        if self.vjepa2_dense_source_num_frames < self.vjepa2_dense_num_frames:
            raise ValueError(
                "vjepa2_dense_source_num_frames must be at least "
                "vjepa2_dense_num_frames")
        if not self.vjepa2_dense_window_seed:
            raise ValueError(
                "vjepa2_dense_window_seed must be a non-empty string")
        if (
            self.vjepa2_student_frame_sampling
            == "random_contiguous_per_video"
            and self.mmd_estimator != "nystrom_exact_repulsion"
        ):
            raise ValueError(
                "random V-JEPA2 temporal windows require "
                "wan_mmd_estimator=nystrom_exact_repulsion")
        if self.vjepa2_dense_num_tokens <= 1:
            raise ValueError("vjepa2_dense_num_tokens must be > 1")
        if (
            self.vjepa2_dense_crop_size <= 0
            or self.vjepa2_dense_resize_short_side
            < self.vjepa2_dense_crop_size
        ):
            raise ValueError(
                "V-JEPA2 resize_short_side must be >= positive crop_size")

        adapter_config = {
            "name": "vjepa2_dense",
            "model_id": self.vjepa2_model_load_path,
            "input_frames": self.vjepa2_dense_num_frames,
            "dense_num_frames": self.vjepa2_dense_num_frames,
            "input_size": self.vjepa2_dense_crop_size,
            "resize_short_side": self.vjepa2_dense_resize_short_side,
            "crop_mode": "center",
            "dense_only": True,
            "dense_layer_index": self.vjepa2_dense_layer,
        }
        self.vjepa2_encoder = VJEPA2Adapter(adapter_config)
        self.vjepa2_encoder.to(device=device, dtype=torch.float32)
        self.vjepa2_encoder.eval().requires_grad_(False)

        actual_geometry = (
            int(self.vjepa2_encoder.dense_layer_index),
            int(self.vjepa2_encoder.dense_num_frames),
            int(self.vjepa2_encoder.resize_short_side),
            int(self.vjepa2_encoder.input_size),
            int(self.vjepa2_encoder.retained_dense_blocks),
        )
        expected_geometry = (
            self.vjepa2_dense_layer,
            self.vjepa2_dense_num_frames,
            self.vjepa2_dense_resize_short_side,
            self.vjepa2_dense_crop_size,
            self.vjepa2_dense_layer + 1,
        )
        if actual_geometry != expected_geometry:
            raise RuntimeError(
                "loaded V-JEPA2 adapter geometry does not match the requested "
                "dense-prefix configuration: "
                f"actual={actual_geometry}, expected={expected_geometry}")
        self.vjepa2_dense_raw_tokens = int(
            self.vjepa2_encoder.dense_raw_tokens)
        self.vjepa2_dense_feature_dim = int(
            self.vjepa2_encoder.dense_feature_dim)
        if self.vjepa2_dense_num_tokens > self.vjepa2_dense_raw_tokens:
            raise ValueError(
                "vjepa2_dense_num_tokens cannot exceed the full dense grid: "
                f"{self.vjepa2_dense_num_tokens} > "
                f"{self.vjepa2_dense_raw_tokens}")
        self._bind_dense_runtime_aliases()

    def _initialize_motionformer_features(self, args, device) -> None:
        """Build the optional frozen MotionFormer dense-prefix extractor."""
        from video_fd.encoders.motionformer_adapter import (
            MotionFormerAdapter,
            SSV2_MOTIONFORMER_224_16X4_SHA256,
        )

        configured_path = str(getattr(
            args, "motionformer_checkpoint_path", "")).strip()
        checkpoint_path = os.path.realpath(
            os.path.expanduser(configured_path)) if configured_path else ""
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "wan_feature_type=motionformer_dense_latent requires "
                "motionformer_checkpoint_path; got "
                f"{checkpoint_path!r}")
        checkpoint_sha256 = str(getattr(
            args, "motionformer_checkpoint_sha256", "")).strip().lower()
        if (
            len(checkpoint_sha256) != 64
            or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
        ):
            raise ValueError(
                "motionformer_checkpoint_sha256 must be a lowercase SHA-256")
        if checkpoint_sha256 != SSV2_MOTIONFORMER_224_16X4_SHA256:
            raise ValueError(
                "motionformer_checkpoint_sha256 must identify the official "
                "SSv2 Base 16x4 checkpoint")

        self.motionformer_checkpoint_path = checkpoint_path
        self.motionformer_checkpoint_sha256 = checkpoint_sha256
        self.motionformer_dense_layer = int(getattr(
            args, "motionformer_dense_layer", 11))
        self.motionformer_dense_num_frames = int(getattr(
            args, "motionformer_dense_num_frames", 16))
        self.motionformer_dense_num_tokens = int(getattr(
            args, "motionformer_dense_num_tokens", 128))
        self.motionformer_dense_resize_short_side = int(getattr(
            args, "motionformer_dense_resize_short_side", 224))
        self.motionformer_dense_crop_size = int(getattr(
            args, "motionformer_dense_crop_size", 224))
        self.motionformer_dense_gradient_checkpointing = bool(getattr(
            args, "motionformer_dense_gradient_checkpointing", True))
        if self.motionformer_dense_layer != 11:
            raise ValueError(
                "the supported SSv2 MotionFormer feature contract fixes "
                "motionformer_dense_layer=11")
        if self.motionformer_dense_num_frames != 16:
            raise ValueError(
                "the supported SSv2 MotionFormer feature contract fixes "
                "motionformer_dense_num_frames=16")
        if (
            self.motionformer_dense_resize_short_side != 224
            or self.motionformer_dense_crop_size != 224
        ):
            raise ValueError(
                "the supported SSv2 MotionFormer preprocessing contract fixes "
                "resize_short_side=crop_size=224")
        if self.motionformer_dense_num_tokens <= 1:
            raise ValueError("motionformer_dense_num_tokens must be > 1")

        adapter_config = {
            "name": "motionformer_dense",
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "input_frames": self.motionformer_dense_num_frames,
            "dense_num_frames": self.motionformer_dense_num_frames,
            "input_size": self.motionformer_dense_crop_size,
            "resize_short_side": self.motionformer_dense_resize_short_side,
            "crop_mode": "center",
            "dense_only": True,
            "dense_layer_index": self.motionformer_dense_layer,
            "use_original_trajectory_attention": True,
        }
        self.motionformer_encoder = MotionFormerAdapter(adapter_config)
        self.motionformer_encoder.to(device=device, dtype=torch.float32)
        self.motionformer_encoder.eval().requires_grad_(False)
        actual_geometry = (
            int(self.motionformer_encoder.dense_layer_index),
            int(self.motionformer_encoder.dense_num_frames),
            int(self.motionformer_encoder.input_size),
            int(self.motionformer_encoder.retained_dense_blocks),
        )
        expected_geometry = (
            self.motionformer_dense_layer,
            self.motionformer_dense_num_frames,
            self.motionformer_dense_crop_size,
            self.motionformer_dense_layer + 1,
        )
        if actual_geometry != expected_geometry:
            raise RuntimeError(
                "loaded MotionFormer adapter geometry does not match the "
                "requested dense-prefix configuration: "
                f"actual={actual_geometry}, expected={expected_geometry}")
        self.motionformer_dense_raw_tokens = int(
            self.motionformer_encoder.dense_raw_tokens)
        self.motionformer_dense_feature_dim = int(
            self.motionformer_encoder.dense_feature_dim)
        if self.motionformer_dense_raw_tokens != 1568:
            raise RuntimeError(
                "SSv2 MotionFormer must expose 1568 non-CLS dense tokens, got "
                f"{self.motionformer_dense_raw_tokens}")
        if self.motionformer_dense_feature_dim != 768:
            raise RuntimeError(
                "SSv2 MotionFormer must expose 768-dimensional features, got "
                f"{self.motionformer_dense_feature_dim}")
        if (
            self.motionformer_dense_num_tokens
            > self.motionformer_dense_raw_tokens
        ):
            raise ValueError(
                "motionformer_dense_num_tokens cannot exceed the full dense "
                f"grid: {self.motionformer_dense_num_tokens} > "
                f"{self.motionformer_dense_raw_tokens}")

        # The replay/MMD implementation below deliberately consumes one common
        # dense-encoder contract.  These private aliases keep the already
        # validated V-JEPA2 implementation and its legacy tests byte-compatible;
        # public generic aliases are what the trainer uses.
        self.vjepa2_model_id = checkpoint_path
        self.vjepa2_dense_layer = self.motionformer_dense_layer
        self.vjepa2_dense_layers = (self.motionformer_dense_layer,)
        self.vjepa2_multilayer_enabled = False
        self.vjepa2_dense_num_frames = self.motionformer_dense_num_frames
        # MotionFormer receives the complete decoded video and performs its
        # official 16-segment midpoint sampling inside the adapter.  Mark this
        # as non-random so the V-JEPA2-only contiguous-window machinery stays
        # dormant.
        self.vjepa2_dense_frame_sampling = "center_contiguous"
        self.vjepa2_dense_source_num_frames = (
            self.motionformer_dense_num_frames)
        self.vjepa2_dense_window_seed = ""
        self.vjepa2_dense_num_tokens = self.motionformer_dense_num_tokens
        self.vjepa2_dense_resize_short_side = (
            self.motionformer_dense_resize_short_side)
        self.vjepa2_dense_crop_size = self.motionformer_dense_crop_size
        self.vjepa2_dense_gradient_checkpointing = (
            self.motionformer_dense_gradient_checkpointing)
        self.vjepa2_encoder = self.motionformer_encoder
        self.vjepa2_dense_raw_tokens = self.motionformer_dense_raw_tokens
        self.vjepa2_dense_feature_dim = self.motionformer_dense_feature_dim
        self._bind_dense_runtime_aliases()

    def _bind_dense_runtime_aliases(self) -> None:
        """Publish the encoder-neutral runtime interface used by ElasticForcingTrainer."""
        self.dense_encoder = self.vjepa2_encoder
        self.dense_layer = self.vjepa2_dense_layer
        self.dense_num_frames = self.vjepa2_dense_num_frames
        self.dense_num_tokens = self.vjepa2_dense_num_tokens
        self.dense_resize_short_side = self.vjepa2_dense_resize_short_side
        self.dense_crop_size = self.vjepa2_dense_crop_size
        self.dense_gradient_checkpointing = (
            self.vjepa2_dense_gradient_checkpointing)
        self.dense_raw_tokens = self.vjepa2_dense_raw_tokens
        self.dense_feature_dim = self.vjepa2_dense_feature_dim

    def _initialize_dual_dinov3_vjepa2_features(
        self,
        args,
        device,
    ) -> None:
        """Build only the frozen DINOv3 and V-JEPA2 prefixes."""
        self.fresh_video_total_samples = int(getattr(
            args, "total_batch_size", 0))
        if self.fresh_video_total_samples <= 0:
            raise ValueError(
                "DINOv3+V-JEPA2 MMD requires positive total_batch_size")
        self._initialize_vjepa2_features(args, device)
        expected_vjepa = {
            "layers": (15,),
            "frames": 64,
            "source_frames": 81,
            "student_sampling": "last_contiguous",
            "tokens": 1024,
            "feature_dim": 1024,
        }
        actual_vjepa = {
            "layers": tuple(self.vjepa2_dense_layers),
            "frames": int(self.vjepa2_dense_num_frames),
            "source_frames": int(self.vjepa2_student_source_num_frames),
            "student_sampling": str(self.vjepa2_student_frame_sampling),
            "tokens": int(self.vjepa2_dense_num_tokens),
            "feature_dim": int(self.vjepa2_dense_feature_dim),
        }
        if actual_vjepa != expected_vjepa:
            raise ValueError(
                "dual DINOv3+V-JEPA2 contract fixes V-JEPA2 layer15, "
                "frames17..80, K=1024 and D=1024: "
                f"actual={actual_vjepa}, expected={expected_vjepa}")
        self._initialize_dinov3_features(args, device)
        actual_dinov3 = {
            "geometry": self.dinov3_dense_geometry,
            "source_frames": self.dinov3_dense_source_num_frames,
            "sample_shape": (
                self.dinov3_dense_num_tokens,
                self.dinov3_dense_feature_dim,
            ),
        }
        expected_dinov3 = {
            "geometry": "temporal_mean5_full256",
            "source_frames": 81,
            "sample_shape": (1024, 1024),
        }
        if actual_dinov3 != expected_dinov3:
            raise ValueError(
                "dual DINOv3+V-JEPA2 contract fixes DINOv3 to 20 frames, "
                "mean5, K=1024 and D=1024: "
                f"actual={actual_dinov3}, expected={expected_dinov3}")
        self.dual_branch_registry = {
            "dinov3": {
                "encoder_attribute": "dinov3_encoder",
                "source_frame_indices": list(
                    self.dinov3_dense_frame_indices),
                "sample_shape": (1024, 1024),
                "tokens_per_temporal_group": 256,
                "temporal_groups": 4,
                "model_id": self.dinov3_model_id,
                "source_commit": self.dinov3_source_commit,
                "checkpoint_sha256": getattr(
                    self.dinov3_encoder, "checkpoint_sha256", ""),
            },
            "vjepa2": {
                "encoder_attribute": "vjepa2_encoder",
                "layer": 15,
                "source_frame_range": [17, 80],
                "sample_shape": (1024, 1024),
                "tokens_per_segment": 1024,
                "segments": 1,
                "model_id": self.vjepa2_reference_model_id,
                "model_sha256": self.vjepa2_model_sha256,
            },
        }

    def _replace_with_dual_dinov3_vjepa2_nystrom_objective(
        self, args, device
    ) -> None:
        """Bind two Wan-video references and sum normalized branch MMDs."""
        from losses.multi_encoder_mmd import (
            MultiEncoderMMD,
        )
        from losses.video_nystrom_mmd import (
            VideoNystromMMD,
            load_reference_pack,
        )

        self.fresh_video_gradient_global_samples = (
            self._configured_current_teacher_global_samples(args))

        if not self.uses_fresh_video_gradcache:
            raise ValueError(
                "DINOv3+V-JEPA2 MMD requires nystrom_exact_repulsion")
        if not self.latent_self_normalize:
            raise ValueError(
                "DINOv3+V-JEPA2 MMD requires self normalization")
        if (
            self.nystrom_attraction_ema_enabled
            or self.nystrom_repulsion_fifo_enabled
            or self.frame_fd_fusion_enabled
        ):
            raise ValueError(
                "DINOv3+V-JEPA2 MMD requires EMA, FIFO and frame FD disabled")
        if self.current_teacher_attraction_enabled:
            if (
                self.nystrom_attraction_weight != 2.0
                or self.current_teacher_attraction_weight != 1.0
            ):
                raise ValueError(
                    "DINOv3+V-JEPA2 fixed/random teacher weights must be 2:1")
            if self.expected_current_teacher_samples != 128:
                raise ValueError(
                    "DINOv3+V-JEPA2 current-teacher mode requires 128 "
                    "matched teachers")
        else:
            if self.current_teacher_source != "none":
                raise ValueError(
                    "fixed-only DINOv3+V-JEPA2 requires "
                    "current_teacher_source=none")
            if self.nystrom_attraction_weight != 1.0:
                raise ValueError(
                    "fixed-only DINOv3+V-JEPA2 requires fixed attraction "
                    "weight 1.0")
            if self.current_teacher_attraction_weight != 0.0:
                raise ValueError(
                    "fixed-only DINOv3+V-JEPA2 requires current-teacher "
                    "weight 0.0")
            if self.expected_current_teacher_samples != 0:
                raise ValueError(
                    "fixed-only DINOv3+V-JEPA2 requires zero expected "
                    "current-teacher samples")
            if (
                self.vjepa2_reference_frame_sampling != "center_contiguous"
                or self.vjepa2_reference_token_contract
                != "legacy_uniform_implicit_v1"
            ):
                raise ValueError(
                    "fixed-only OpenVid V-JEPA2 requires the approved "
                    "center64 legacy_uniform_implicit_v1 reference contract")

        def strict_sha256(value, label):
            normalized = str(value).strip().lower()
            if (
                len(normalized) != 64
                or any(c not in "0123456789abcdef" for c in normalized)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256")
            return normalized

        def checked_reference_path(path_value, sha_value, label):
            path = os.path.realpath(os.path.expanduser(str(path_value)))
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"DINOv3+V-JEPA2 {label} reference missing: {path!r}")
            expected_sha = strict_sha256(sha_value, f"{label} reference SHA-256")
            actual_sha = self._distributed_file_sha256(path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"{label} reference SHA-256 mismatch: "
                    f"{actual_sha} != {expected_sha}")
            return path, expected_sha, load_reference_pack(
                path, map_location="cpu")

        # Each encoder owns its reference summary; the wrapper combines normalized losses.
        vjepa_objective = self.latent_mmd
        if not isinstance(vjepa_objective, VideoNystromMMD):
            raise TypeError(
                "dual bootstrap V-JEPA2 objective must be VideoNystromMMD")
        vjepa_path = self.vjepa2_nystrom_reference_path
        vjepa_file_sha = strict_sha256(
            self._dense_config_value(
                args, "nystrom_reference_sha256", ""),
            "V-JEPA2 reference SHA-256",
        )
        if self._distributed_file_sha256(vjepa_path) != vjepa_file_sha:
            raise ValueError(
                "dual V-JEPA2 reference file SHA-256 mismatch")

        dinov3_path, dinov3_file_sha, dinov3_pack = checked_reference_path(
            getattr(args, "dinov3_nystrom_reference_path", ""),
            getattr(args, "dinov3_nystrom_reference_sha256", ""),
            "DINOv3",
        )
        dinov3_objective = VideoNystromMMD(
            dinov3_pack,
            expected_metadata={
                "feature": self._dinov3_feature_metadata(),
                "nystrom": {
                    "num_landmarks": int(getattr(
                        args, "dinov3_nystrom_landmarks", 256)),
                    "seed": int(getattr(args, "dinov3_nystrom_seed", 0)),
                },
            },
            kernel_chunk_size=int(getattr(
                args, "dinov3_nystrom_kernel_chunk_size", 32)),
            attraction_weight=self.nystrom_attraction_weight,
            current_teacher_attraction_enabled=(
                self.current_teacher_attraction_enabled),
            current_teacher_attraction_weight=(
                self.current_teacher_attraction_weight),
            attraction_ema_enabled=False,
            repulsion_fifo_size=0,
        ).to(device=device)
        objectives = {
            "dinov3": dinov3_objective,
            "vjepa2": vjepa_objective,
        }
        expected_shapes = {
            "dinov3": (1024, 1024),
            "vjepa2": (1024, 1024),
        }
        actual_shapes = {
            name: tuple(objective.sample_shape)
            for name, objective in objectives.items()
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                "dual reference sample-shape mismatch: "
                f"actual={actual_shapes}, expected={expected_shapes}")

        if self.current_teacher_attraction_enabled:
            configured_reference_identity = strict_sha256(
                getattr(args, "dinov3_vjepa2_teacher_bank_content_hash", ""),
                "dinov3_vjepa2_teacher_bank_content_hash",
            )
            expected_manifest_hash = strict_sha256(
                getattr(args, "matched_manifest_sha256", ""),
                "matched_manifest_sha256",
            )
            for name, objective in objectives.items():
                source = objective.reference_metadata.get(
                    "reference", {}).get("source", {})
                if (
                    source.get("dinov3_vjepa2_teacher_bank_content_hash")
                    != configured_reference_identity
                    or source.get("source_manifest_sha256")
                    != expected_manifest_hash
                    or int(source.get("selected_samples", -1)) != 5000
                    or source.get("branch_name") != name
                ):
                    raise ValueError(
                        f"{name} reference is not bound to the configured "
                        "frozen 5000-Wan-video DINOv3+V-JEPA2 bank")
            self.dinov3_vjepa2_teacher_bank_content_hash = (
                configured_reference_identity)
        else:
            bundle_path = os.path.realpath(os.path.expanduser(str(getattr(
                args, "dinov3_vjepa2_reference_bundle_manifest", ""))))
            if not os.path.isfile(bundle_path):
                raise FileNotFoundError(
                    "fixed-only OpenVid reference bundle is missing: "
                    f"{bundle_path!r}")
            configured_reference_identity = strict_sha256(
                getattr(
                    args,
                    "dinov3_vjepa2_reference_bundle_manifest_sha256",
                    "",
                ),
                "dinov3_vjepa2_reference_bundle_manifest_sha256",
            )
            actual_bundle_sha = self._distributed_file_sha256(bundle_path)
            if actual_bundle_sha != configured_reference_identity:
                raise ValueError(
                    "OpenVid dual reference bundle SHA-256 mismatch: "
                    f"{actual_bundle_sha} != {configured_reference_identity}")
            with open(bundle_path, "r", encoding="utf-8") as handle:
                bundle = json.load(handle)
            expected_count = int(getattr(
                args, "openvid_reference_count", 77615))
            expected_sequence = strict_sha256(
                getattr(args, "openvid_video_name_sequence_sha256", ""),
                "openvid_video_name_sequence_sha256",
            )
            if (
                bundle.get("kind")
                != "openvid_dual_nystrom_reference_bundle"
                or int(bundle.get("schema_version", -1)) != 1
                or int(bundle.get("reference_count", -1)) != expected_count
                or bundle.get("video_name_sequence_sha256")
                != expected_sequence
                or set(bundle.get("branches", {}))
                != {"dinov3", "vjepa2"}
            ):
                raise ValueError(
                    "OpenVid dual reference bundle identity/schema mismatch")
            bridge = bundle.get("legacy_vjepa_provenance_bridge", {})
            if (
                bridge.get("accepted") is not True
                or bridge.get("limitation")
                != "legacy_pack_has_no_embedded_row_name_sequence"
            ):
                raise ValueError(
                    "OpenVid bundle must explicitly attest the accepted legacy "
                    "V-JEPA provenance limitation")
            expected_files = {
                "dinov3": dinov3_file_sha,
                "vjepa2": vjepa_file_sha,
            }
            for name, objective in objectives.items():
                entry = bundle["branches"].get(name, {})
                nystrom = objective.reference_metadata.get("nystrom", {})
                if (
                    entry.get("file_sha256") != expected_files[name]
                    or entry.get("content_hash") != objective.reference_hash
                    or entry.get("feature")
                    != objective.reference_metadata.get("feature")
                    or int(entry.get("reference_count", -1))
                    != expected_count
                    or int(nystrom.get("reference_count", -1))
                    != expected_count
                ):
                    raise ValueError(
                        f"OpenVid bundle does not bind the loaded {name} "
                        "reference pack")
            self.dinov3_vjepa2_reference_bundle_manifest = bundle_path
            self.dinov3_vjepa2_reference_bundle_manifest_sha256 = (
                configured_reference_identity)
            self.openvid_video_name_sequence_sha256 = expected_sequence
        # Paper Eq. (3) motivates encoder weighting; this release also uses detached-scale normalization.
        branch_weights = {
            "dinov3": float(getattr(
                args, "dinov3_vjepa2_dinov3_weight", 10.0)),
            "vjepa2": float(getattr(
                args, "dinov3_vjepa2_vjepa2_weight", 1.0)),
        }
        if branch_weights != {"dinov3": 10.0, "vjepa2": 1.0}:
            raise ValueError(
                "DINOv3+V-JEPA2 representation weights must be exactly 10:1")
        self.latent_mmd = MultiEncoderMMD(
            objectives,
            weights=branch_weights,
            normalization_eps=self.latent_norm_eps,
        ).to(device=device)
        self.dinov3_nystrom_reference_path = dinov3_path
        self.dinov3_nystrom_reference_hash = dinov3_objective.reference_hash
        self.dual_teacher_bank_content_hash = configured_reference_identity
        reference_files = {
            "dinov3": dinov3_file_sha,
            "vjepa2": vjepa_file_sha,
        }
        reference_paths = {
            "dinov3": dinov3_path,
            "vjepa2": vjepa_path,
        }
        for name, objective in objectives.items():
            self.dual_branch_registry[name].update({
                "reference_path": reference_paths[name],
                "reference_file_sha256": reference_files[name],
                "reference_hash": objective.reference_hash,
                "nystrom_landmarks": int(objective.landmarks.shape[0]),
            })

    def _initialize_multi_encoder_features(
        self,
        args,
        device,
    ) -> None:
        """Build the frozen feature encoders and their video/token geometry.

        The released 14B recipe uses all three encoders. The explicit list also
        supports ablations; saved attribute names stay compatible."""
        self.dinov3_reference_paths = dict(getattr(args, "dinov3_reference_paths", {}))
        self.triple_active_branches = active_mmd_branches(
            getattr(args, "triple_active_branches", None))
        self.fresh_video_total_samples = int(getattr(
            args, "total_batch_size", 0))
        if self.fresh_video_total_samples <= 0:
            raise ValueError(
                "triple video MMD requires positive total_batch_size")
        self._initialize_vjepa2_features(args, device)
        vjepa_attribute_names = (
            "vjepa2_model_load_path",
            "vjepa2_reference_model_id",
            "vjepa2_model_sha256",
            "vjepa2_model_id",
            "vjepa2_dense_layers",
            "vjepa2_multilayer_enabled",
            "vjepa2_dense_layer",
            "vjepa2_dense_num_frames",
            "vjepa2_dense_frame_sampling",
            "vjepa2_reference_frame_sampling",
            "vjepa2_reference_token_contract",
            "vjepa2_dense_source_num_frames",
            "vjepa2_student_frame_sampling",
            "vjepa2_student_source_num_frames",
            "vjepa2_dense_window_seed",
            "vjepa2_dense_num_tokens",
            "vjepa2_dense_resize_short_side",
            "vjepa2_dense_crop_size",
            "vjepa2_dense_gradient_checkpointing",
            "vjepa2_dense_raw_tokens",
            "vjepa2_dense_feature_dim",
        )
        vjepa_state = {
            name: getattr(self, name) for name in vjepa_attribute_names
        }
        vjepa_encoder = self.vjepa2_encoder
        expected_vjepa = {
            "layers": (15,),
            "frames": 64,
            "source_frames": 81,
            "student_sampling": "last_contiguous",
            "tokens": 1024,
            "feature_dim": 1024,
        }
        actual_vjepa = {
            "layers": tuple(self.vjepa2_dense_layers),
            "frames": int(self.vjepa2_dense_num_frames),
            "source_frames": int(self.vjepa2_student_source_num_frames),
            "student_sampling": str(self.vjepa2_student_frame_sampling),
            "tokens": int(self.vjepa2_dense_num_tokens),
            "feature_dim": int(self.vjepa2_dense_feature_dim),
        }
        if actual_vjepa != expected_vjepa:
            raise ValueError(
                "triple V-JEPA2 contract is fixed to layer15, frames17..80, "
                "K=1024 and D=1024: "
                f"actual={actual_vjepa}, expected={expected_vjepa}")

        self._initialize_videomae_v2_features(args, device)
        expected_videomae = {
            "layers": (15,),
            "segments": 5,
            "skip": 1,
            "frames_per_segment": 16,
            "source_frames": 81,
            "tokens_per_segment": 256,
            "feature_dim": 1024,
        }
        actual_videomae = {
            "layers": tuple(self.videomae_v2_dense_layers),
            "segments": int(self.videomae_v2_encoder_count),
            "skip": int(self.videomae_v2_skip_leading_frames),
            "frames_per_segment": int(self.videomae_v2_segment_num_frames),
            "source_frames": int(self.videomae_v2_source_num_frames),
            "tokens_per_segment": int(self.videomae_v2_dense_num_tokens),
            "feature_dim": int(self.videomae_v2_encoder.dense_feature_dim),
        }
        if actual_videomae != expected_videomae:
            raise ValueError(
                "triple VideoMAE-v2 contract is fixed to layer15, frames1..80 "
                "as 5x16, K=256 per segment and D=1024: "
                f"actual={actual_videomae}, expected={expected_videomae}")

        for name, value in vjepa_state.items():
            setattr(self, name, value)
        self.vjepa2_encoder = vjepa_encoder
        self.dense_feature_segment_count = 1
        self._bind_dense_runtime_aliases()
        self.dual_branch_registry = {}
        if "dinov3" in self.triple_active_branches:
            self._initialize_dinov3_features(args, device)
            actual_dinov3 = {
                "geometry": self.dinov3_dense_geometry,
                "source_frames": self.dinov3_dense_source_num_frames,
                "sample_shape": (
                    self.dinov3_dense_num_tokens,
                    self.dinov3_dense_feature_dim,
                ),
            }
            expected_dinov3 = {
                "geometry": "temporal_mean5_full256",
                "source_frames": 81,
                "sample_shape": (1024, 1024),
            }
            if actual_dinov3 != expected_dinov3:
                raise ValueError(
                    "triple DINOv3 contract is fixed to 20 frames, mean-pool "
                    "consecutive groups of 5, K=1024 and D=1024: "
                    f"actual={actual_dinov3}, expected={expected_dinov3}")
            self.dual_branch_registry["dinov3"] = {
                "encoder_attribute": "dinov3_encoder",
                "source_frame_indices": list(
                    self.dinov3_dense_frame_indices),
                "sample_shape": (1024, 1024),
                "tokens_per_temporal_group": 256,
                "temporal_groups": 4,
                "model_id": self.dinov3_model_id,
                "source_commit": self.dinov3_source_commit,
                "checkpoint_sha256": getattr(
                    self.dinov3_encoder, "checkpoint_sha256", ""),
            }
        self.dual_branch_registry.update({
            "vjepa2": {
                "encoder_attribute": "vjepa2_encoder",
                "layer": 15,
                "source_frame_range": [17, 80],
                "sample_shape": (1024, 1024),
                "tokens_per_segment": 1024,
                "segments": 1,
                "model_id": self.vjepa2_reference_model_id,
                "model_sha256": self.vjepa2_model_sha256,
            },
            "videomae_v2": {
                "encoder_attribute": "videomae_v2_encoder",
                "layer": 15,
                "source_frame_range": [1, 80],
                "sample_shape": (1280, 1024),
                "tokens_per_segment": 256,
                "segments": 5,
                "model_id": self.videomae_v2_model_id,
                "model_revision": str(getattr(
                    args, "videomae_v2_model_revision", "")),
                "download_manifest_content_hash": (
                    self.videomae_v2_download_manifest_content_hash),
            },
        })

    def _build_multi_encoder_objective(self, args, device) -> None:
        """Bind each encoder to its fixed reference summary and combine its loss.

        The hybrid reference uses both Nyström statistics and a sampled reference
        minibatch. Encoder weights follow the release's detached-scale normalization."""
        active_branches = active_mmd_branches(
            getattr(args, "triple_active_branches", None))
        from losses.multi_encoder_mmd import (
            MultiEncoderMMD,
        )
        from losses.video_nystrom_mmd import (
            VideoNystromMMD,
            load_reference_pack,
        )
        from video_fd.encoders.videomae_v2_dense_metadata import (
            canonical_videomae_v2_dense_feature_metadata,
            uniform_segment_token_indices,
        )

        if not self.uses_fresh_video_gradcache:
            raise ValueError(
                "triple video MMD requires nystrom_exact_repulsion")
        if not self.latent_self_normalize:
            raise ValueError(
                "triple video MMD requires wan_kv_self_normalize=true")
        if (
            self.nystrom_attraction_ema_enabled
            or self.nystrom_repulsion_fifo_enabled
            or self.frame_fd_fusion_enabled
        ):
            raise ValueError(
                "triple video MMD requires EMA, FIFO, and frame FD disabled")
        if not self.current_teacher_attraction_enabled:
            raise ValueError(
                "triple video MMD requires exact matched teacher MMD")
        if (
            self.nystrom_attraction_weight != 2.0
            or self.current_teacher_attraction_weight != 1.0
        ):
            raise ValueError(
                "triple fixed/current teacher weights must be exactly 2:1")
        if self.expected_current_teacher_samples != 128:
            raise ValueError(
                "triple objective requires exactly 128 matched teachers")

        def strict_sha256(value, label):
            normalized = str(value).strip().lower()
            if (
                len(normalized) != 64
                or any(c not in "0123456789abcdef" for c in normalized)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256")
            return normalized

        def checked_reference_path(path_value, sha_value, label):
            path = os.path.realpath(os.path.expanduser(str(path_value)))
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"triple {label} reference pack is missing: {path!r}")
            expected_sha = strict_sha256(sha_value, f"{label} reference SHA-256")
            actual_sha = self._distributed_file_sha256(path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"triple {label} reference file SHA-256 mismatch: "
                    f"{actual_sha} != {expected_sha}")
            return path, expected_sha, load_reference_pack(
                path, map_location="cpu")

        # Each encoder owns its reference summary; the wrapper combines normalized losses.
        vjepa_objective = self.latent_mmd
        if not isinstance(vjepa_objective, VideoNystromMMD):
            raise TypeError(
                "triple bootstrap V-JEPA2 objective must be VideoNystromMMD")
        vjepa_path = self.vjepa2_nystrom_reference_path
        configured_vjepa_reference_sha = strict_sha256(
            self._dense_config_value(
                args, "nystrom_reference_sha256", ""),
            "V-JEPA2 reference SHA-256",
        )
        if self._distributed_file_sha256(
            vjepa_path
        ) != configured_vjepa_reference_sha:
            raise ValueError(
                "triple V-JEPA2 reference file SHA-256 mismatch")

        objectives = {}
        reference_files = {}
        reference_paths = {}
        if "dinov3" in active_branches:
            dinov3_path, dinov3_file_sha, dinov3_pack = checked_reference_path(
                getattr(args, "dinov3_nystrom_reference_path", ""),
                getattr(args, "dinov3_nystrom_reference_sha256", ""),
                "DINOv3",
            )
            dinov3_objective = VideoNystromMMD(
                dinov3_pack,
                expected_metadata={
                    "feature": self._dinov3_feature_metadata(),
                    "nystrom": {
                        "num_landmarks": int(getattr(
                            args, "dinov3_nystrom_landmarks", 256)),
                        "seed": int(getattr(args, "dinov3_nystrom_seed", 0)),
                    },
                },
                kernel_chunk_size=int(getattr(
                    args, "dinov3_nystrom_kernel_chunk_size", 32)),
                attraction_weight=2.0,
                current_teacher_attraction_enabled=True,
                current_teacher_attraction_weight=1.0,
                attraction_ema_enabled=False,
                repulsion_fifo_size=0,
            ).to(device=device)
            objectives["dinov3"] = dinov3_objective
            reference_files["dinov3"] = dinov3_file_sha
            reference_paths["dinov3"] = dinov3_path
            self.dinov3_nystrom_reference_path = dinov3_path
            self.dinov3_nystrom_reference_hash = dinov3_objective.reference_hash

        videomae_path, videomae_file_sha, videomae_pack = checked_reference_path(
            getattr(args, "videomae_v2_nystrom_reference_path", ""),
            getattr(args, "videomae_v2_nystrom_reference_sha256", ""),
            "VideoMAE-v2",
        )
        video_tokens = uniform_segment_token_indices(
            self.videomae_v2_encoder.dense_raw_tokens,
            self.videomae_v2_dense_num_tokens,
        )
        video_feature_metadata = canonical_videomae_v2_dense_feature_metadata(
            model_id=self.videomae_v2_reference_model_id,
            model_id_is_reference=True,
            layer_index=15,
            num_tokens=256,
            token_indices=video_tokens,
            num_segments=5,
            frames_per_segment=16,
            raw_tokens_per_segment=(
                self.videomae_v2_encoder.dense_raw_tokens),
            feature_dim=self.videomae_v2_encoder.dense_feature_dim,
            resize_short_side=224,
            crop_size=224,
            skip_first_frame=True,
        )
        videomae_objective = VideoNystromMMD(
            videomae_pack,
            expected_metadata={
                "feature": video_feature_metadata,
                "nystrom": {
                    "num_landmarks": int(getattr(
                        args, "videomae_v2_nystrom_landmarks", 256)),
                    "seed": int(getattr(
                        args, "videomae_v2_nystrom_seed", 0)),
                },
            },
            kernel_chunk_size=int(getattr(
                args, "videomae_v2_nystrom_kernel_chunk_size", 32)),
            attraction_weight=2.0,
            current_teacher_attraction_enabled=True,
            current_teacher_attraction_weight=1.0,
            attraction_ema_enabled=False,
            repulsion_fifo_size=0,
        ).to(device=device)

        objectives.update({
            "vjepa2": vjepa_objective,
            "videomae_v2": videomae_objective,
        })
        expected_shapes = {
            "dinov3": (1024, 1024),
            "vjepa2": (1024, 1024),
            "videomae_v2": (1280, 1024),
        }
        expected_shapes = {name: expected_shapes[name] for name in active_branches}
        actual_shapes = {
            name: tuple(objective.sample_shape)
            for name, objective in objectives.items()
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                "triple reference sample-shape mismatch: "
                f"actual={actual_shapes}, expected={expected_shapes}")

        configured_bank_hash = strict_sha256(
            getattr(args, "triple_teacher_bank_content_hash", ""),
            "triple_teacher_bank_content_hash",
        )
        expected_manifest_hash = strict_sha256(
            getattr(args, "matched_manifest_sha256", ""),
            "matched_manifest_sha256",
        )
        expected_reference_samples = int(getattr(
            args, "matched_manifest_samples", 5000))
        if expected_reference_samples <= 0:
            raise ValueError("matched_manifest_samples must be positive")
        sources = {
            name: objective.reference_metadata.get(
                "reference", {}).get("source", {})
            for name, objective in objectives.items()
        }
        for name, source in sources.items():
            if (
                source.get("triple_teacher_bank_content_hash")
                != configured_bank_hash
                or source.get("source_manifest_sha256")
                != expected_manifest_hash
                or int(source.get("selected_samples", -1))
                != expected_reference_samples
                or source.get("branch_name") != name
            ):
                raise ValueError(
                    f"{name} reference is not bound to the configured frozen "
                    f"{expected_reference_samples}-video triple teacher bank")

        # Paper Eq. (3) motivates encoder weighting; this release also uses detached-scale normalization.
        branch_weights = {
            "dinov3": float(getattr(args, "triple_dinov3_weight", 10.0)),
            "vjepa2": float(getattr(args, "triple_vjepa2_weight", 1.0)),
            "videomae_v2": float(getattr(
                args, "triple_videomae_v2_weight", 1.0)),
        }
        branch_weights = {name: branch_weights[name] for name in active_branches}
        expected_weights = {
            "dinov3": 10.0,
            "vjepa2": 1.0,
            "videomae_v2": 1.0,
        }
        if branch_weights != {name: expected_weights[name] for name in active_branches}:
            raise ValueError(
                "active representation weights must retain their original values")
        aggregation_divisor = float(getattr(
            args, "triple_aggregation_divisor", 3.0))
        if aggregation_divisor != 3.0:
            raise ValueError(
                "triple representation aggregation divisor must be exactly 3")
        self.latent_mmd = MultiEncoderMMD(
            objectives,
            weights=branch_weights,
            normalization_eps=self.latent_norm_eps,
            aggregation_divisor=aggregation_divisor,
        ).to(device=device)

        self.videomae_v2_nystrom_reference_path = videomae_path
        self.videomae_v2_nystrom_reference_hash = (
            videomae_objective.reference_hash)
        self.triple_teacher_bank_content_hash = configured_bank_hash
        # Compatibility aliases are internal only; public metadata is triple.
        self.dual_teacher_bank_content_hash = configured_bank_hash
        reference_files.update({
            "vjepa2": configured_vjepa_reference_sha,
            "videomae_v2": videomae_file_sha,
        })
        reference_paths.update({
            "vjepa2": vjepa_path,
            "videomae_v2": videomae_path,
        })
        for name, objective in objectives.items():
            self.dual_branch_registry[name].update({
                "reference_path": reference_paths[name],
                "reference_file_sha256": reference_files[name],
                "reference_hash": objective.reference_hash,
                "nystrom_landmarks": int(objective.landmarks.shape[0]),
            })

    def _initialize_videomae_v2_features(self, args, device) -> None:
        """Build the fixed five-replica VideoMAE-v2 Large feature map.

        Public state is VideoMAE-namespaced.  A small set of historical
        ``vjepa2_*`` aliases is retained only inside this class and its trainer
        so the already validated GradCache/VAE replay implementation can be
        reused without changing the distribution-loss formula.
        """
        from video_fd.encoders.segmented_videomae_v2 import (
            SegmentedVideoMAEv2Ensemble,
        )

        configured_model_id = str(getattr(
            args,
            "videomae_v2_model_id",
            "OpenGVLab/VideoMAEv2-Large",
        ))
        expanded_model_id = os.path.expanduser(configured_model_id)
        self.videomae_v2_model_id = (
            os.path.realpath(expanded_model_id)
            if os.path.exists(expanded_model_id)
            else configured_model_id
        )
        # Keep frozen metadata identity separate from the model's load location.
        # The local preparation script reads this identity from the feature bank.
        self.videomae_v2_reference_model_id = str(getattr(
            args, "videomae_v2_reference_model_id", self.videomae_v2_model_id))
        self.videomae_v2_local_files_only = bool(getattr(
            args, "videomae_v2_local_files_only", True))
        if (
            self.videomae_v2_local_files_only
            and not os.path.isdir(self.videomae_v2_model_id)
        ):
            raise FileNotFoundError(
                "VideoMAE-v2 training requires the pinned local checkpoint "
                "directory; download OpenGVLab/VideoMAEv2-Large first: "
                f"{self.videomae_v2_model_id}")
        self.videomae_v2_model_revision = str(getattr(
            args, "videomae_v2_model_revision", ""))
        self.videomae_v2_download_manifest_content_hash = str(getattr(
            args, "videomae_v2_download_manifest_content_hash", ""))
        if self.dual_video_mmd_enabled:
            expected_revision = (
                "9981a9c8f77118c421e5228e1b219468a4b0238d")
            if self.videomae_v2_model_revision != expected_revision:
                raise ValueError(
                    "VideoMAE-v2 revision must be pinned to "
                    f"{expected_revision}")
            download_manifest_path = os.path.join(
                self.videomae_v2_model_id, "download_manifest.json")
            if not os.path.isfile(download_manifest_path):
                raise FileNotFoundError(
                    "VideoMAE-v2 download_manifest.json is required")
            with open(download_manifest_path, "r", encoding="utf-8") as handle:
                download_manifest = json.load(handle)
            saved_content_hash = download_manifest.get("content_hash")
            manifest_payload = dict(download_manifest)
            manifest_payload.pop("content_hash", None)
            if (
                download_manifest.get("repo_id")
                != "OpenGVLab/VideoMAEv2-Large"
                or download_manifest.get("revision") != expected_revision
                or saved_content_hash
                != self._canonical_sha256(manifest_payload)
                or self.videomae_v2_download_manifest_content_hash
                != saved_content_hash
            ):
                raise ValueError(
                    "VideoMAE-v2 download manifest identity mismatch")
            for file_identity in download_manifest.get("files", []):
                relative = str(file_identity.get("path", ""))
                path = os.path.realpath(os.path.join(
                    self.videomae_v2_model_id, relative))
                if (
                    not relative
                    or not path.startswith(self.videomae_v2_model_id + os.sep)
                    or not os.path.isfile(path)
                    or os.path.getsize(path) != int(
                        file_identity.get("size_bytes", -1))
                    or self._distributed_file_sha256(path)
                    != file_identity.get("sha256")
                ):
                    raise ValueError(
                        "VideoMAE-v2 downloaded file identity mismatch: "
                        f"{relative}")

        configured_dense_layers = getattr(
            args, "videomae_v2_dense_layers", None)
        if configured_dense_layers is None:
            dense_layers = (
                int(getattr(args, "videomae_v2_dense_layer", 15)),
            )
        else:
            if isinstance(configured_dense_layers, (str, bytes)):
                raise TypeError(
                    "videomae_v2_dense_layers must be a sequence of integers")
            dense_layers = tuple(
                int(layer) for layer in configured_dense_layers)
            if not dense_layers:
                raise ValueError(
                    "videomae_v2_dense_layers must not be empty")
        if len(set(dense_layers)) != len(dense_layers):
            raise ValueError(
                "videomae_v2_dense_layers must contain unique indices")
        if tuple(sorted(dense_layers)) != dense_layers:
            raise ValueError(
                "videomae_v2_dense_layers must be sorted")
        if any(layer < 0 or layer >= 24 for layer in dense_layers):
            raise ValueError(
                "VideoMAE-v2 Large dense layers use zero-based indices in "
                "[0,24)")

        self.videomae_v2_dense_layers = dense_layers
        self.videomae_v2_multilayer_enabled = len(dense_layers) > 1
        self.videomae_v2_dense_layer = dense_layers[-1]
        self.videomae_v2_encoder_count = int(getattr(
            args, "videomae_v2_encoder_count", 5))
        self.videomae_v2_share_encoder_weights = bool(getattr(
            args, "videomae_v2_share_encoder_weights", False))
        self.videomae_v2_skip_leading_frames = int(getattr(
            args, "videomae_v2_skip_leading_frames", 1))
        self.videomae_v2_segment_num_frames = int(getattr(
            args, "videomae_v2_segment_num_frames", 16))
        derived_source_num_frames = (
            self.videomae_v2_skip_leading_frames
            + self.videomae_v2_encoder_count
            * self.videomae_v2_segment_num_frames
        )
        self.videomae_v2_source_num_frames = int(getattr(
            args,
            "videomae_v2_source_num_frames",
            derived_source_num_frames,
        ))
        self.videomae_v2_dense_num_tokens = int(getattr(
            args, "videomae_v2_dense_num_tokens", 128))
        self.videomae_v2_dense_resize_short_side = int(getattr(
            args, "videomae_v2_dense_resize_short_side", 224))
        self.videomae_v2_dense_crop_size = int(getattr(
            args, "videomae_v2_dense_crop_size", 224))
        self.videomae_v2_dense_gradient_checkpointing = bool(getattr(
            args, "videomae_v2_dense_gradient_checkpointing", True))

        fixed_geometry = (
            self.videomae_v2_encoder_count,
            self.videomae_v2_skip_leading_frames,
            self.videomae_v2_segment_num_frames,
            self.videomae_v2_source_num_frames,
            self.videomae_v2_dense_resize_short_side,
            self.videomae_v2_dense_crop_size,
        )
        expected_geometry = (5, 1, 16, 81, 224, 224)
        if fixed_geometry != expected_geometry:
            raise ValueError(
                "VideoMAE-v2 whole-video geometry is fixed to five physical "
                "16-frame encoders over decoded frames [1:81]: "
                f"actual={fixed_geometry}, expected={expected_geometry}")
        if self.videomae_v2_dense_num_tokens <= 1:
            raise ValueError(
                "videomae_v2_dense_num_tokens must be > 1 per segment")

        adapter_config = {
            "name": "videomae_v2_five_segment_dense",
            "model_id": self.videomae_v2_model_id,
            "encoder_count": self.videomae_v2_encoder_count,
            "share_encoder_weights": (
                self.videomae_v2_share_encoder_weights),
            "skip_leading_frames": self.videomae_v2_skip_leading_frames,
            "segment_num_frames": self.videomae_v2_segment_num_frames,
            "source_num_frames": self.videomae_v2_source_num_frames,
            "input_frames": self.videomae_v2_segment_num_frames,
            "dense_num_frames": self.videomae_v2_segment_num_frames,
            "input_size": self.videomae_v2_dense_crop_size,
            "resize_short_side": (
                self.videomae_v2_dense_resize_short_side),
            "crop_mode": "center",
            "dense_only": True,
            # ``15`` means the 16th block and captures its post-residual
            # dense state, before pooling/fc_norm.
            "dense_layer_index": self.videomae_v2_dense_layer,
            "local_files_only": self.videomae_v2_local_files_only,
        }
        if self.videomae_v2_model_revision:
            adapter_config["revision"] = self.videomae_v2_model_revision
        self.videomae_v2_encoder = SegmentedVideoMAEv2Ensemble(
            adapter_config)
        self.videomae_v2_encoder.to(device=device, dtype=torch.float32)
        self.videomae_v2_encoder.eval().requires_grad_(False)

        actual_adapter_geometry = (
            int(self.videomae_v2_encoder.dense_layer_index),
            int(self.videomae_v2_encoder.dense_num_frames),
            int(self.videomae_v2_encoder.resize_short_side),
            int(self.videomae_v2_encoder.input_size),
            int(self.videomae_v2_encoder.retained_dense_blocks),
            int(self.videomae_v2_encoder.dense_raw_tokens),
            int(self.videomae_v2_encoder.dense_feature_dim),
            len(self.videomae_v2_encoder.encoders),
        )
        expected_adapter_geometry = (
            self.videomae_v2_dense_layer,
            16,
            224,
            224,
            self.videomae_v2_dense_layer + 1,
            1568,
            1024,
            1 if self.videomae_v2_share_encoder_weights else 5,
        )
        if actual_adapter_geometry != expected_adapter_geometry:
            raise RuntimeError(
                "loaded VideoMAE-v2 Large ensemble does not match the fixed "
                "dense-prefix geometry: "
                f"actual={actual_adapter_geometry}, "
                f"expected={expected_adapter_geometry}")
        configured_depths = {
            int(getattr(encoder, "_configured_depth", -1))
            for encoder in self.videomae_v2_encoder.encoders
        }
        if configured_depths != {24}:
            raise RuntimeError(
                "all five encoders must be VideoMAE-v2 Large with 24 blocks; "
                f"got depths={sorted(configured_depths)}")

        # Historical internal aliases.  Public metadata, config, logs, and
        # checkpoints are translated back to ``videomae_v2_*``.
        self.dense_feature_segment_count = 5
        self.vjepa2_model_id = self.videomae_v2_model_id
        self.vjepa2_dense_layers = self.videomae_v2_dense_layers
        self.vjepa2_multilayer_enabled = (
            self.videomae_v2_multilayer_enabled)
        self.vjepa2_dense_layer = self.videomae_v2_dense_layer
        self.vjepa2_dense_num_frames = 80
        self.vjepa2_dense_source_num_frames = 81
        self.vjepa2_dense_frame_sampling = "center_contiguous"
        self.vjepa2_dense_window_seed = "fixed-five-segment-videomae-v2"
        self.vjepa2_dense_num_tokens = (
            self.videomae_v2_dense_num_tokens)
        self.vjepa2_dense_resize_short_side = (
            self.videomae_v2_dense_resize_short_side)
        self.vjepa2_dense_crop_size = self.videomae_v2_dense_crop_size
        self.vjepa2_dense_gradient_checkpointing = (
            self.videomae_v2_dense_gradient_checkpointing)
        self.vjepa2_encoder = self.videomae_v2_encoder
        self.vjepa2_dense_raw_tokens = int(
            self.videomae_v2_encoder.dense_raw_tokens)
        self.vjepa2_dense_feature_dim = int(
            self.videomae_v2_encoder.dense_feature_dim)
        if self.vjepa2_dense_num_tokens > self.vjepa2_dense_raw_tokens:
            raise ValueError(
                "videomae_v2_dense_num_tokens cannot exceed the native "
                "per-segment token grid: "
                f"{self.vjepa2_dense_num_tokens} > "
                f"{self.vjepa2_dense_raw_tokens}")

    def _initialize_dinov3_features(self, args, device) -> None:
        """Load the independent frozen DINOv3 dense-image backend.

        The compatibility geometry consumes sixteen fixed frames and retains a
        row-major 8x8 spatial patch subset from each frame.  The optional
        ``temporal_mean5_full256`` geometry consumes twenty fixed frames,
        averages every five consecutive frame-feature grids, and retains the
        complete 16x16 patch grid for each of the four temporal groups.  Both
        geometries produce one aligned [1024, 1024] matrix per video; their
        adapter metadata keeps references and checkpoints mutually exclusive.
        """
        compatibility_frames = (
            0, 5, 11, 16, 21, 27, 32, 37,
            43, 48, 53, 59, 64, 69, 75, 80,
        )
        temporal_pool_frames = (
            0, 4, 8, 13, 17, 21, 25, 29, 34, 38,
            42, 46, 51, 55, 59, 63, 67, 72, 76, 80,
        )
        compatibility_spatial_axis = (0, 2, 4, 6, 9, 11, 13, 15)
        full_spatial_axis = tuple(range(16))
        initialization_error = None
        try:
            geometry = str(getattr(
                args, "dinov3_dense_geometry", "f16x64")).strip()
            if geometry == "f16x64":
                from video_fd.encoders.dinov3_adapter import (
                    DINOv3DenseAdapter as Adapter,
                )

                canonical_frames = compatibility_frames
                canonical_spatial_axis = compatibility_spatial_axis
            elif geometry == "temporal_mean5_full256":
                from video_fd.encoders.dinov3_temporal_pool_adapter import (
                    DINOv3TemporalPoolAdapter as Adapter,
                )

                canonical_frames = temporal_pool_frames
                canonical_spatial_axis = full_spatial_axis
            else:
                raise ValueError(
                    "dinov3_dense_geometry must be 'f16x64' or "
                    "'temporal_mean5_full256', got "
                    f"{geometry!r}")

            configured_frames = tuple(int(value) for value in getattr(
                args, "dinov3_dense_frame_indices", canonical_frames))
            configured_spatial_axis = tuple(int(value) for value in getattr(
                args, "dinov3_dense_spatial_axis_indices",
                canonical_spatial_axis))
            adapter_config = {
                "repo_path": str(getattr(args, "dinov3_repo_path", "")),
                "weight_path": str(getattr(args, "dinov3_weight_path", "")),
                "source_commit": str(getattr(
                    args, "dinov3_source_commit", "")),
                "source_num_frames": int(getattr(
                    args, "dinov3_dense_source_num_frames", 81)),
                "frame_indices": configured_frames,
                "input_size": int(getattr(
                    args, "dinov3_dense_input_size", 256)),
                "patch_size": int(getattr(
                    args, "dinov3_dense_patch_size", 16)),
                "spatial_axis_indices": configured_spatial_axis,
                "forward_chunk_frames": int(getattr(
                    args, "dinov3_dense_forward_chunk_frames", 4)),
            }
            if geometry == "temporal_mean5_full256":
                adapter_config.update({
                    "geometry": "temporal_mean5_full256",
                    "temporal_groups": 4,
                    "temporal_pool_frames": 5,
                })
            configured_checkpoint_sha256 = str(getattr(
                args, "dinov3_checkpoint_sha256", "")).strip()
            if configured_checkpoint_sha256:
                adapter_config["checkpoint_sha256"] = (
                    configured_checkpoint_sha256)

            self.dinov3_encoder = Adapter(adapter_config)
            self.dinov3_encoder.to(device=device, dtype=torch.float32)
            self.dinov3_encoder.eval().requires_grad_(False)

            self.dinov3_dense_geometry = geometry
            self.dinov3_model_id = str(self.dinov3_encoder.model_id)
            self.dinov3_repo_path = str(self.dinov3_encoder.repo_path)
            self.dinov3_weight_path = str(self.dinov3_encoder.weight_path)
            self.dinov3_source_commit = str(
                self.dinov3_encoder.source_commit)
            self.dinov3_dense_source_num_frames = int(
                self.dinov3_encoder.source_num_frames)
            self.dinov3_dense_frame_indices = tuple(
                int(value) for value in self.dinov3_encoder.frame_indices)
            self.dinov3_dense_num_frames = int(
                self.dinov3_encoder.dense_num_frames)
            self.dinov3_dense_input_size = int(
                self.dinov3_encoder.input_size)
            self.dinov3_dense_patch_size = int(
                self.dinov3_encoder.patch_size)
            self.dinov3_dense_spatial_axis_indices = tuple(
                int(value) for value in
                self.dinov3_encoder.spatial_axis_indices)
            self.dinov3_dense_forward_chunk_frames = int(
                self.dinov3_encoder.forward_chunk_frames)
            self.dinov3_dense_raw_tokens = int(
                self.dinov3_encoder.dense_raw_tokens)
            self.dinov3_dense_num_tokens = int(
                self.dinov3_encoder.dense_num_tokens)
            self.dinov3_dense_feature_dim = int(
                self.dinov3_encoder.dense_feature_dim)
            self.dinov3_dense_gradient_checkpointing = bool(getattr(
                args, "dinov3_dense_gradient_checkpointing", False))

            if geometry == "temporal_mean5_full256":
                self.dinov3_dense_temporal_groups = int(
                    self.dinov3_encoder.temporal_groups)
                self.dinov3_dense_temporal_pool_size = int(
                    self.dinov3_encoder.temporal_pool_frames)
                self.dinov3_dense_tokens_per_group = int(
                    self.dinov3_encoder.per_frame_tokens)

            actual_contract = {
                "model_id": self.dinov3_model_id,
                "source_num_frames": self.dinov3_dense_source_num_frames,
                "frame_indices": self.dinov3_dense_frame_indices,
                "num_frames": self.dinov3_dense_num_frames,
                "input_size": self.dinov3_dense_input_size,
                "patch_size": self.dinov3_dense_patch_size,
                "spatial_axis_indices": (
                    self.dinov3_dense_spatial_axis_indices),
                "forward_chunk_frames": (
                    self.dinov3_dense_forward_chunk_frames),
                "raw_tokens": self.dinov3_dense_raw_tokens,
                "num_tokens": self.dinov3_dense_num_tokens,
                "feature_dim": self.dinov3_dense_feature_dim,
            }
            if geometry == "temporal_mean5_full256":
                actual_contract.update({
                    "geometry": str(self.dinov3_encoder.geometry),
                    "temporal_groups": (
                        self.dinov3_dense_temporal_groups),
                    "temporal_pool_size": (
                        self.dinov3_dense_temporal_pool_size),
                    "tokens_per_group": (
                        self.dinov3_dense_tokens_per_group),
                })
                expected_contract = {
                    "model_id": "dinov3_vitl16",
                    "source_num_frames": 81,
                    "frame_indices": canonical_frames,
                    "num_frames": 20,
                    "input_size": 256,
                    "patch_size": 16,
                    "spatial_axis_indices": canonical_spatial_axis,
                    "forward_chunk_frames": 4,
                    "raw_tokens": 5120,
                    "num_tokens": 1024,
                    "feature_dim": 1024,
                    "geometry": "temporal_mean5_full256",
                    "temporal_groups": 4,
                    "temporal_pool_size": 5,
                    "tokens_per_group": 256,
                }
            else:
                expected_contract = {
                    "model_id": "dinov3_vitl16",
                    "source_num_frames": 81,
                    "frame_indices": canonical_frames,
                    "num_frames": 16,
                    "input_size": 256,
                    "patch_size": 16,
                    "spatial_axis_indices": canonical_spatial_axis,
                    "forward_chunk_frames": 4,
                    "raw_tokens": 4096,
                    "num_tokens": 1024,
                    "feature_dim": 1024,
                }
            if actual_contract != expected_contract:
                raise ValueError(
                    "loaded DINOv3 adapter does not satisfy the public dense "
                    f"feature contract: actual={actual_contract}, "
                    f"expected={expected_contract}")
        except BaseException as exc:
            initialization_error = exc
        self._synchronize_dinov3_local_failure(
            initialization_error,
            device,
            operation="DINOv3 adapter initialization failed",
        )

    def _dinov3_feature_metadata(self) -> dict:
        metadata = self.dinov3_encoder.feature_metadata()
        if not isinstance(metadata, Mapping):
            raise TypeError("DINOv3 feature_metadata() must return a mapping")
        metadata = dict(metadata)
        required_identity = {
            "feature_type": "dinov3_dense_latent",
            "model_id": self.dinov3_model_id,
            "raw_tokens": self.dinov3_dense_raw_tokens,
            "num_tokens": self.dinov3_dense_num_tokens,
            "feature_dim": self.dinov3_dense_feature_dim,
            "matrix_shape": [
                self.dinov3_dense_num_tokens,
                self.dinov3_dense_feature_dim,
            ],
        }
        mismatched = {
            key: {"saved": metadata.get(key), "runtime": value}
            for key, value in required_identity.items()
            if metadata.get(key) != value
        }
        if mismatched:
            raise ValueError(
                "DINOv3 adapter metadata violates the runtime feature "
                f"contract: {mismatched}")
        # The frozen bank may record its original asset locations. Use those
        # strings only as metadata identities; encoder I/O uses relocated paths.
        reference_paths = getattr(self, "dinov3_reference_paths", {})
        if reference_paths:
            for key in ("repo_path", "weight_path"):
                metadata[key] = reference_paths[key]
            metadata.pop("feature_contract_sha256", None)
            payload = json.dumps(
                metadata, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("utf-8")
            metadata["feature_contract_sha256"] = hashlib.sha256(payload).hexdigest()
        return metadata

    def _dinov3_descriptor_frame_indices(
        self,
        replay_descriptor: Mapping,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        saved = replay_descriptor.get("dinov3_frame_indices")
        if not isinstance(saved, torch.Tensor):
            raise TypeError("DINOv3 replay frame indices must be a tensor")
        saved = saved.to(device=device, dtype=torch.long)
        expected = torch.tensor(
            self.dinov3_dense_frame_indices,
            device=device,
            dtype=torch.long,
        )
        if saved.ndim != 1 or not torch.equal(saved, expected):
            raise ValueError(
                "DINOv3 replay frame indices changed from the fixed public "
                "sampling contract")
        return saved

    def _extract_dinov3_dense(self, video01: torch.Tensor) -> torch.Tensor:
        if (
            video01.ndim != 5
            or int(video01.shape[1])
            != self.dinov3_dense_source_num_frames
            or int(video01.shape[2]) != 3
        ):
            raise ValueError(
                "DINOv3 dense extraction requires full decoded videos with "
                f"shape [B,{self.dinov3_dense_source_num_frames},3,H,W], "
                f"got {tuple(video01.shape)}")
        extract = (
            self.dinov3_encoder.forward_dense_no_grad
            if not torch.is_grad_enabled()
            else self.dinov3_encoder.forward_dense
        )
        features = extract(
            video01,
            gradient_checkpointing=(
                self.dinov3_dense_gradient_checkpointing),
        )
        expected_shape = (
            int(video01.shape[0]),
            self.dinov3_dense_num_tokens,
            self.dinov3_dense_feature_dim,
        )
        if features.ndim != 3 or tuple(features.shape) != expected_shape:
            raise ValueError(
                "frozen DINOv3 returned an invalid dense feature matrix: "
                f"got {tuple(features.shape)}, expected {expected_shape}")
        return features

    def _replay_dinov3_dense_to_pixels(
        self,
        video01: torch.Tensor,
        *,
        expected_features: torch.Tensor,
        feature_cotangent: torch.Tensor,
        replay_atol: float,
        replay_rtol: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        replay_features, video_cotangent = (
            self.dinov3_encoder.replay_dense_vjp(
                video01.detach().float(),
                feature_cotangent.detach().float(),
                expected_features=expected_features.detach().float(),
                gradient_checkpointing=(
                    self.dinov3_dense_gradient_checkpointing),
                replay_atol=float(replay_atol),
                replay_rtol=float(replay_rtol),
            )
        )
        expected_feature_shape = tuple(expected_features.shape)
        if tuple(replay_features.shape) != expected_feature_shape:
            raise RuntimeError(
                "DINOv3 feature replay changed matrix shape: "
                f"{tuple(replay_features.shape)} != "
                f"{expected_feature_shape}")
        if tuple(video_cotangent.shape) != tuple(video01.shape):
            raise RuntimeError(
                "DINOv3 replay returned an invalid full-video pixel "
                f"cotangent: {tuple(video_cotangent.shape)} != "
                f"{tuple(video01.shape)}")
        if not bool(torch.isfinite(video_cotangent).all().item()):
            raise FloatingPointError(
                "DINOv3 replay returned a non-finite pixel cotangent")
        replay_max_abs = (
            replay_features.detach().float()
            - expected_features.detach().float()
        ).abs().max()
        return video_cotangent.detach().float(), replay_max_abs.detach()

    def _dense_metric_prefix(self) -> str:
        return (
            "dinov3_dense"
            if self._uses_dinov3_dense_backend()
            else "vjepa2_dense"
        )

    def _uses_dinov3_dense_backend(self) -> bool:
        """Support legacy/test instances created without the new init flags."""
        return bool(getattr(
            self,
            "is_dinov3_dense",
            getattr(self, "feature_type", None) == "dinov3_dense_latent",
        ))

    def _token_indices(self, latent: torch.Tensor) -> torch.Tensor:
        patch = self.latent_patch_size
        tokens = (
            (latent.shape[1] // int(patch[0]))
            * (latent.shape[3] // int(patch[1]))
            * (latent.shape[4] // int(patch[2]))
        )
        n = min(tokens, self.latent_num_tokens)
        return torch.linspace(
            0, tokens - 1, steps=n, device=latent.device).round().long()

    def _extract(self, latent, conditional_dict, token_indices):
        timestep = latent.new_full(
            (latent.shape[0], latent.shape[1]),
            self.latent_timestep,
            dtype=torch.float32,
        )
        features = self.real_score(
            noisy_image_or_video=latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
            return_latent_features=True,
            latent_feature_layer=self.latent_layer,
            latent_token_indices=token_indices,
        )
        expected_shape = (
            latent.shape[0], token_indices.numel(), self.latent_feature_dim)
        if features.ndim != 3 or tuple(features.shape) != expected_shape:
            raise ValueError(
                "frozen Wan returned an invalid post-block latent shape: "
                f"got {tuple(features.shape)}, expected {expected_shape}")
        return features

    def _vjepa2_token_indices(self, device: torch.device) -> torch.Tensor:
        if self.uses_random_token_reference_bank:
            if (
                self.vjepa2_reference_bank is None
                or self.vjepa2_active_reference_slot is None
            ):
                raise RuntimeError(
                    "no random-token Nyström reference slot is active")
            return self.vjepa2_reference_bank.token_indices(
                self.vjepa2_active_reference_slot, device=device)
        return uniform_dense_token_indices(
            self.vjepa2_dense_raw_tokens,
            self.vjepa2_dense_num_tokens,
            device=device,
        )

    def _vjepa2_feature_metadata(
        self,
        token_indices: Optional[torch.Tensor] = None,
        *,
        layer_index: Optional[int] = None,
    ) -> dict:
        resolved_layer = (
            (
                self.motionformer_dense_layer
                if self._dense_encoder_kind() == "motionformer"
                else self.vjepa2_dense_layer
            )
            if layer_index is None
            else int(layer_index)
        )
        if self._dense_encoder_kind() == "motionformer":
            if resolved_layer != self.motionformer_dense_layer:
                raise ValueError(
                    "requested feature metadata for an inactive MotionFormer "
                    f"layer: {resolved_layer}")
            from video_fd.encoders.motionformer_dense_metadata import (
                canonical_motionformer_dense_feature_metadata,
            )

            return canonical_motionformer_dense_feature_metadata(
                checkpoint_path=self.motionformer_checkpoint_path,
                checkpoint_sha256=self.motionformer_checkpoint_sha256,
                layer_index=self.motionformer_dense_layer,
                num_frames=self.motionformer_dense_num_frames,
                raw_tokens=self.motionformer_dense_raw_tokens,
                num_tokens=self.motionformer_dense_num_tokens,
                feature_dim=self.motionformer_dense_feature_dim,
                token_indices=token_indices,
            )
        if resolved_layer not in getattr(
            self, "vjepa2_dense_layers", (self.vjepa2_dense_layer,)
        ):
            raise ValueError(
                "requested feature metadata for an inactive V-JEPA2 layer: "
                f"{resolved_layer}")
        if (
            token_indices is None
            and self.dense_state_prefix in {
                "dual", "triple", "dinov3_vjepa2"
            }
            and getattr(
                self,
                "vjepa2_reference_token_contract",
                "explicit_uniform_v2",
            ) != "legacy_uniform_implicit_v1"
        ):
            # Multi-encoder Wan banks serialize the exact uniform subset so
            # the reference and replay contracts attest row order explicitly.
            token_indices = uniform_dense_token_indices(
                self.vjepa2_dense_raw_tokens,
                self.vjepa2_dense_num_tokens,
            )
        metadata_kwargs = {
            "model_id": getattr(
                self, "vjepa2_reference_model_id", self.vjepa2_model_id),
            "layer_index": resolved_layer,
            "num_frames": self.vjepa2_dense_num_frames,
            "resize_short_side": self.vjepa2_dense_resize_short_side,
            "crop_size": self.vjepa2_dense_crop_size,
            "raw_tokens": self.vjepa2_dense_raw_tokens,
            "num_tokens": self.vjepa2_dense_num_tokens,
            "feature_dim": self.vjepa2_dense_feature_dim,
            "token_indices": token_indices,
        }
        default_reference_sampling = getattr(
            self,
            "vjepa2_dense_frame_sampling",
            "center_contiguous",
        )
        if (
            not hasattr(self, "vjepa2_reference_frame_sampling")
            and self.dense_state_prefix in {
                "dual", "triple", "dinov3_vjepa2"
            }
        ):
            # Preserve pre-existing checkpoint/test fixtures that predate the
            # explicit reference/student split: multi-encoder references were
            # last64 unless an initialized model now says otherwise.
            default_reference_sampling = "last_contiguous"
        frame_sampling = getattr(
            self,
            "vjepa2_reference_frame_sampling",
            default_reference_sampling,
        )
        if frame_sampling == "last_contiguous":
            metadata_kwargs.update({
                "frame_sampling": frame_sampling,
                "source_num_frames": self.vjepa2_dense_source_num_frames,
            })
        elif frame_sampling == "random_contiguous_per_video":
            metadata_kwargs.update({
                "frame_sampling": frame_sampling,
                "source_num_frames": self.vjepa2_dense_source_num_frames,
                "window_seed": self.vjepa2_dense_window_seed,
            })
        return canonical_vjepa2_dense_feature_metadata(**metadata_kwargs)

    def _vjepa2_static_feature_metadata(self) -> dict:
        """Metadata subset shared by every explicit-token reference slot."""
        if getattr(self, "vjepa2_multilayer_enabled", False):
            raise RuntimeError(
                "random-token reference banks do not support multi-layer MMD")
        metadata = self._vjepa2_feature_metadata()
        for key in (
            "feature_metadata_version",
            "token_sampling",
            "token_alignment",
            "token_indices_sha256",
        ):
            metadata.pop(key, None)
        return metadata

    def _dense_feature_metadata(
        self,
        token_indices: Optional[torch.Tensor] = None,
        *,
        layer_index: Optional[int] = None,
    ) -> dict:
        """Encoder-neutral alias for strict dense feature identity."""
        return self._vjepa2_feature_metadata(
            token_indices, layer_index=layer_index)

    def _dense_static_feature_metadata(self) -> dict:
        """Encoder-neutral static identity for random-token bank slots."""
        return self._vjepa2_static_feature_metadata()

    def _publish_active_reference(self, activated) -> None:
        self.vjepa2_active_reference_slot = int(activated.slot)
        self.vjepa2_active_token_indices_hash = str(
            activated.token_indices_sha256)
        self.dense_active_reference_slot = int(activated.slot)
        self.dense_active_token_indices_hash = str(
            activated.token_indices_sha256)
        self.vjepa2_nystrom_reference_hash = str(
            activated.pack_content_hash)
        self.vjepa2_active_video_real_floor = float(
            activated.video_real_floor)
        self.vjepa2_active_video_student_start_score = float(
            activated.video_student_start_score)
        self.vjepa2_active_video_student_start_std = float(
            activated.video_student_start_std)
        reference_count = int(
            activated.objective.reference_metadata[
                "nystrom"]["reference_count"])
        if (
            self.vjepa2_reference_bank is not None
            and reference_count
            != len(self.vjepa2_reference_bank.train_indices)
        ):
            raise ValueError(
                "active Nyström pack reference_count does not match the "
                "teacher-train split")
        self.vjepa2_nystrom_reference_count = reference_count

    def activate_dense_reference_slot(self, slot: int) -> dict:
        """Switch the empty update pipeline to one explicit-token reference."""
        if not self.uses_random_token_reference_bank:
            raise RuntimeError(
                "reference slots require random per-update token sampling")
        if self.fidel_controller is not None and self.fidel_controller.staged:
            raise RuntimeError(
                "cannot switch V-JEPA2 reference while FiDEL state is staged")
        activated = self._dense_bank().activate_slot_(
            int(slot), self.latent_mmd)
        self._publish_active_reference(activated)
        return {
            "slot": int(activated.slot),
            "token_indices": activated.token_indices.detach().clone(),
            "token_indices_sha256": activated.token_indices_sha256,
            "pack_content_hash": activated.pack_content_hash,
            "video_real_floor": float(activated.video_real_floor),
            "video_student_start_score": float(
                activated.video_student_start_score),
            "video_student_start_std": float(
                activated.video_student_start_std),
        }

    def activate_vjepa2_reference_slot(self, slot: int) -> dict:
        """Backward-compatible alias for the dense reference selector."""
        return self.activate_dense_reference_slot(slot)

    def stage_fidel_update(
        self,
        *,
        video_result,
        frame_result,
    ) -> Mapping[str, object]:
        """Stage two-level FiDEL and scale update-local replay cotangents.

        The controller only sees detached scalar diagnostics.  Raw feature
        cotangents are multiplied here, before either frozen encoder is
        replayed, so there is exactly one derivative path per branch.  The
        additional ``world_size`` factor compensates FSDP's parameter-gradient
        averaging; there is deliberately no division by the number of replay
        microbatches because all of them are blocks of one global objective.
        """
        if not self.fidel_two_level_enabled or self.fidel_controller is None:
            raise RuntimeError(
                "stage_fidel_update requires frame_fd_fusion_mode="
                "fidel_two_level")
        if self.fidel_controller.staged:
            raise RuntimeError(
                "a two-level FiDEL update is already staged")
        if self.vjepa2_active_reference_slot is None:
            raise RuntimeError("no random-token reference slot is active")
        frame_branches = ("inception", "siglip", "mae")

        def canonical_frame_branch(judge_name: str) -> str:
            lowered = str(judge_name).lower()
            matches = [
                branch for branch in frame_branches
                if branch in lowered
            ]
            if len(matches) != 1:
                raise ValueError(
                    "cannot map frame FD judge to exactly one FiDEL branch: "
                    f"{judge_name!r}")
            return matches[0]

        judge_to_branch = {
            name: canonical_frame_branch(name)
            for name in self.frame_fd_head.representation_names
        }
        branch_to_judge = {
            branch: name for name, branch in judge_to_branch.items()
        }
        if (
            len(judge_to_branch) != len(frame_branches)
            or set(branch_to_judge) != set(frame_branches)
        ):
            raise ValueError(
                "two-level FiDEL requires one Inception, one SigLIP and one "
                "MAE frame judge")
        actual_names = set(judge_to_branch)
        if set(frame_result.raw_fd_values) != actual_names:
            raise ValueError(
                "frame raw losses do not match the canonical FiDEL branches")
        if set(frame_result.fresh_fd_scores) != actual_names:
            raise ValueError(
                "frame fresh scores do not match the canonical FiDEL branches")
        if len(video_result.local_cotangents) != len(
            frame_result.local_raw_feature_cotangents
        ):
            raise ValueError(
                "video/frame FiDEL cotangent microbatch counts differ")
        if int(video_result.world_size) != int(frame_result.world_size):
            raise ValueError("video/frame FiDEL world sizes differ")

        frame_references = {
            entry["key"]: entry
            for entry in self.frame_fd_reference_metadata[
                "representations"]
        }
        if set(frame_references) != actual_names:
            raise ValueError(
                "frame FD reference manifest does not contain exactly the "
                "three FiDEL frame branches")
        for name, entry in frame_references.items():
            for key in (
                "real_floor",
                "student_start_score",
                "student_start_std",
            ):
                if key not in entry:
                    raise ValueError(
                        f"frame FD reference {name!r} is missing {key!r}")

        raw_losses = {
            "video": video_result.loss,
            **{
                branch: frame_result.raw_fd_values[judge]
                for branch, judge in branch_to_judge.items()
            },
        }
        current_scores = {
            "video": video_result.loss,
            **{
                branch: frame_result.fresh_fd_scores[judge]
                for branch, judge in branch_to_judge.items()
            },
        }
        floors = {
            "video": self.vjepa2_active_video_real_floor,
            **{
                branch: frame_references[judge]["real_floor"]
                for branch, judge in branch_to_judge.items()
            },
        }
        start_scores = {
            "video": self.vjepa2_active_video_student_start_score,
            **{
                branch: frame_references[judge]["student_start_score"]
                for branch, judge in branch_to_judge.items()
            },
        }
        stage = self.fidel_controller.stage(
            raw_losses=raw_losses,
            current_scores=current_scores,
            floors=floors,
            start_scores=start_scores,
        )
        world_size = float(video_result.world_size)
        video_scale = float(stage.cotangent_scales["video"]) * world_size
        video_cotangents = tuple(
            (value * video_scale).contiguous()
            for value in video_result.local_cotangents
        )
        frame_cotangents = []
        for microbatch in frame_result.local_raw_feature_cotangents:
            if set(microbatch) != actual_names:
                raise ValueError(
                    "frame raw cotangent keys changed across microbatches")
            frame_cotangents.append({
                judge: (
                    value
                    * (
                        float(stage.cotangent_scales[
                            judge_to_branch[judge]])
                        * world_size
                    )
                ).contiguous()
                for judge, value in microbatch.items()
            })

        diagnostic_device = video_result.loss.device
        diagnostics = {
            "fidel_reference_slot": torch.tensor(
                int(self.vjepa2_active_reference_slot),
                device=diagnostic_device,
                dtype=torch.long,
            ),
            "fidel_committed_updates_before": torch.tensor(
                int(stage.committed_updates_before),
                device=diagnostic_device,
                dtype=torch.long,
            ),
            "fidel_theta_frame": stage.effective_theta_frame.to(
                device=diagnostic_device, dtype=torch.float32),
            "fidel_theta_group": stage.effective_theta_group.to(
                device=diagnostic_device, dtype=torch.float32),
            "fidel_frame_violation": stage.frame_violation.to(
                device=diagnostic_device, dtype=torch.float32),
            "fidel_lambda_video": stage.group_weights["video"].to(
                device=diagnostic_device, dtype=torch.float32),
            "fidel_lambda_frame": stage.group_weights["frame"].to(
                device=diagnostic_device, dtype=torch.float32),
            "vjepa2_token_indices_sha256": (
                self.vjepa2_active_token_indices_hash),
            "vjepa2_reference_pack_hash": (
                self.vjepa2_nystrom_reference_hash),
        }
        for name in ("video", "inception", "siglip", "mae"):
            diagnostics[f"fidel_{name}_raw_loss"] = (
                stage.raw_losses[name].to(
                    device=diagnostic_device, dtype=torch.float32))
            diagnostics[f"fidel_{name}_current_score"] = torch.as_tensor(
                current_scores[name],
                device=diagnostic_device,
                dtype=torch.float32,
            ).detach()
            diagnostics[f"fidel_{name}_real_floor"] = torch.as_tensor(
                floors[name],
                device=diagnostic_device,
                dtype=torch.float32,
            ).detach()
            diagnostics[f"fidel_{name}_student_start_score"] = (
                torch.as_tensor(
                    start_scores[name],
                    device=diagnostic_device,
                    dtype=torch.float32,
                ).detach()
            )
            diagnostics[f"fidel_{name}_violation"] = (
                stage.violations[name].to(
                    device=diagnostic_device, dtype=torch.float32))
            diagnostics[f"fidel_{name}_loss_scale"] = (
                stage.loss_scales[name].to(
                    device=diagnostic_device, dtype=torch.float32))
            diagnostics[f"fidel_{name}_cotangent_scale"] = (
                stage.cotangent_scales[name].to(
                    device=diagnostic_device, dtype=torch.float32))
        for branch in frame_branches:
            diagnostics[f"fidel_{branch}_q"] = (
                stage.frame_q[branch].to(
                    device=diagnostic_device, dtype=torch.float32))
            diagnostics[f"fidel_{branch}_alpha"] = (
                stage.frame_alpha[branch].to(
                    device=diagnostic_device, dtype=torch.float32))
            diagnostics[f"fidel_{branch}_fresh_score"] = (
                frame_result.fresh_fd_scores[
                    branch_to_judge[branch]].detach().to(
                    device=diagnostic_device, dtype=torch.float32))
        diagnostics["fidel_video_group_q"] = stage.group_q["video"].to(
            device=diagnostic_device, dtype=torch.float32)
        diagnostics["fidel_frame_group_q"] = stage.group_q["frame"].to(
            device=diagnostic_device, dtype=torch.float32)

        all_weights_zero = all(
            float(value) == 0.0
            for value in stage.cotangent_scales.values()
        )
        return {
            "video_fsdp_cotangents": video_cotangents,
            "frame_fsdp_feature_cotangents": tuple(frame_cotangents),
            "diagnostics": diagnostics,
            "generator_loss": stage.total_loss_value.to(
                device=diagnostic_device, dtype=torch.float32),
            "all_weights_zero": bool(all_weights_zero),
        }

    def _vjepa2_temporal_window_indices(
        self,
        sample_keys,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        sampling = getattr(
            self,
            "vjepa2_student_frame_sampling",
            getattr(self, "vjepa2_dense_frame_sampling", "center_contiguous"),
        )
        if sampling == "center_contiguous":
            if sample_keys is not None:
                raise ValueError(
                    "window sample keys are invalid for center sampling")
            return None
        if sampling == "last_contiguous":
            if sample_keys is not None:
                raise ValueError(
                    "window sample keys are invalid for last sampling")
            start = (
                int(self.vjepa2_student_source_num_frames)
                - int(self.vjepa2_dense_num_frames)
            )
            offsets = torch.arange(
                self.vjepa2_dense_num_frames,
                device=device,
                dtype=torch.long,
            )
            return (offsets + start).unsqueeze(0).expand(
                int(batch_size), -1).clone()
        if sample_keys is None:
            raise ValueError(
                "random V-JEPA2 windows require one sample key per video")
        if isinstance(sample_keys, torch.Tensor):
            if sample_keys.ndim != 1:
                raise ValueError("window sample keys must have shape [B]")
            keys = sample_keys.detach().cpu().tolist()
        else:
            keys = list(sample_keys)
        if len(keys) != int(batch_size):
            raise ValueError(
                "window sample key count does not match the video batch: "
                f"{len(keys)} != {int(batch_size)}")
        starts = [
            stateless_temporal_window_start(
                self.vjepa2_dense_window_seed,
                "student",
                int(key),
                source_num_frames=getattr(
                    self, "vjepa2_student_source_num_frames",
                    self.vjepa2_dense_source_num_frames),
                window_num_frames=self.vjepa2_dense_num_frames,
            )
            for key in keys
        ]
        start_tensor = torch.tensor(
            starts, device=device, dtype=torch.long)
        offsets = torch.arange(
            self.vjepa2_dense_num_frames,
            device=device,
            dtype=torch.long,
        )
        return start_tensor[:, None] + offsets[None, :]

    def _vjepa2_descriptor_window_indices(
        self,
        replay_descriptor: dict,
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if getattr(
            self,
            "vjepa2_student_frame_sampling",
            getattr(self, "vjepa2_dense_frame_sampling", "center_contiguous"),
        ) == "center_contiguous":
            return None
        return replay_descriptor["vjepa2_frame_indices"].to(
            device=device, dtype=torch.long)

    def _extract_vjepa2_dense(
        self,
        video01: torch.Tensor,
        token_indices: torch.Tensor,
        *,
        gradient_checkpointing: bool,
        temporal_window_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._dense_encoder_kind() == "motionformer":
            if temporal_window_indices is not None:
                raise ValueError(
                    "MotionFormer performs fixed segment-midpoint sampling "
                    "and does not accept V-JEPA2 temporal windows")
            features = self.motionformer_encoder.forward_dense(
                video01,
                layer_index=self.motionformer_dense_layer,
                token_indices=token_indices,
                gradient_checkpointing=gradient_checkpointing,
            )
            expected_shape = (
                video01.shape[0],
                token_indices.numel(),
                self.motionformer_dense_feature_dim,
            )
            if features.ndim != 3 or tuple(features.shape) != expected_shape:
                raise ValueError(
                    "frozen MotionFormer returned an invalid intermediate "
                    f"dense shape: got {tuple(features.shape)}, expected "
                    f"{expected_shape}")
            return features

        dense_layers = getattr(
            self, "vjepa2_dense_layers", (self.vjepa2_dense_layer,))
        if len(dense_layers) == 1:
            features_by_layer = {
                dense_layers[0]: self.vjepa2_encoder.forward_dense(
                    video01,
                    layer_index=dense_layers[0],
                    token_indices=token_indices,
                    gradient_checkpointing=gradient_checkpointing,
                    temporal_window_indices=temporal_window_indices,
                )
            }
        else:
            features_by_layer = self.vjepa2_encoder.forward_dense_layers(
                video01,
                layer_indices=dense_layers,
                token_indices=token_indices,
                gradient_checkpointing=gradient_checkpointing,
                temporal_window_indices=temporal_window_indices,
            )
        expected_shape = (
            video01.shape[0],
            token_indices.numel(),
            self.vjepa2_dense_feature_dim,
        )
        ordered_features = []
        for layer in dense_layers:
            features = features_by_layer[layer]
            if (
                features.ndim != 3
                or tuple(features.shape) != expected_shape
            ):
                raise ValueError(
                    "frozen V-JEPA2 returned an invalid intermediate dense "
                    f"shape at layer {layer}: got {tuple(features.shape)}, "
                    f"expected {expected_shape}")
            ordered_features.append(features)
        # The GradCache transports one tensor. The loss module splits this
        # layer-major concatenation and evaluates independent kernels.
        return (
            ordered_features[0]
            if len(ordered_features) == 1
            else torch.cat(ordered_features, dim=1)
        )

    def _replay_dense_encoder_vjp(
        self,
        video01: torch.Tensor,
        token_indices: torch.Tensor,
        expected_features: torch.Tensor,
        feature_cotangent: torch.Tensor,
        *,
        gradient_checkpointing: bool,
        temporal_window_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replay frozen V-JEPA2 and return its features and pixel VJP."""
        if expected_features.shape != feature_cotangent.shape:
            raise ValueError(
                "dense encoder replay feature/cotangent shapes differ: "
                f"{tuple(expected_features.shape)} != "
                f"{tuple(feature_cotangent.shape)}")
        student_video_leaf = video01.detach().float().requires_grad_(True)
        replay_kwargs = {
            "gradient_checkpointing": gradient_checkpointing,
        }
        if temporal_window_indices is not None:
            replay_kwargs["temporal_window_indices"] = temporal_window_indices
        replay_features = self._extract_vjepa2_dense(
            student_video_leaf, token_indices, **replay_kwargs).float()
        encoder_surrogate = (replay_features * feature_cotangent).sum()
        (video_cotangent,) = torch.autograd.grad(
            encoder_surrogate, student_video_leaf,
            retain_graph=False, create_graph=False)
        replay_features = replay_features.detach().float()
        video_cotangent = video_cotangent.detach().float()
        del encoder_surrogate, student_video_leaf
        return replay_features, video_cotangent

    def _decode_video01(self, latent: torch.Tensor) -> torch.Tensor:
        pixels = self.vae.decode_to_pixel(latent.detach())
        if pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError(
                "Wan VAE returned an invalid video shape: expected "
                f"[B,T,3,H,W], got {tuple(pixels.shape)}")
        if (
            self.is_triple_dense
            or getattr(self, "is_dinov3_vjepa2_dense", False)
            or self._uses_dinov3_dense_backend()
        ):
            if int(pixels.shape[1]) != 81:
                raise ValueError(
                    "DINOv3/triple dense features require exactly 81 "
                    f"decoded Wan frames; got {int(pixels.shape[1])}")
        elif self.dense_state_prefix == "videomae_v2":
            if int(pixels.shape[1]) != 81:
                raise ValueError(
                    "segmented VideoMAE-v2 requires exactly 81 decoded "
                    f"frames; got {int(pixels.shape[1])}")
        else:
            if pixels.shape[1] < self.vjepa2_dense_num_frames:
                raise ValueError(
                    "decoded video is too short for the dense-video clip")
            student_sampling = getattr(
                self, "vjepa2_student_frame_sampling",
                getattr(self, "vjepa2_dense_frame_sampling", "center_contiguous"))
            if (
                student_sampling in {"random_contiguous_per_video", "last_contiguous"}
                and int(pixels.shape[1]) != self.vjepa2_student_source_num_frames
            ):
                raise ValueError(
                    f"{student_sampling} V-JEPA2 sampling requires "
                    f"{self.vjepa2_student_source_num_frames} frames")
        return ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)

    def _capture_generation_rng(self) -> dict:
        return {
            "cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(self.device).clone(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    @contextmanager
    def _replay_generation_rng(self, rng_state: dict):
        if set(rng_state) != {"cpu", "cuda", "numpy", "python"}:
            raise ValueError(
                "invalid fresh-video replay RNG descriptor")
        # fork_rng restores the ambient stream afterwards.  Consequently the
        # second differentiable pass does not consume new training randomness.
        ambient_numpy = np.random.get_state()
        ambient_python = random.getstate()
        try:
            with torch.random.fork_rng(devices=[self.device], enabled=True):
                torch.set_rng_state(rng_state["cpu"])
                torch.cuda.set_rng_state(rng_state["cuda"], self.device)
                np.random.set_state(rng_state["numpy"])
                random.setstate(rng_state["python"])
                yield
        finally:
            np.random.set_state(ambient_numpy)
            random.setstate(ambient_python)

    def _generate_student_latent(
        self,
        *,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        initial_latent: Optional[torch.Tensor] = None,
        generation_descriptor: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.causal_generator:
            pred, gradient_mask, _, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=dict(conditional_dict),
                initial_latent=initial_latent,
                generation_descriptor=generation_descriptor,
            )
            return pred, gradient_mask
        if initial_latent is not None:
            raise NotImplementedError(
                "fresh V-JEPA2 GradCache currently supports T2V only")
        pred = self._run_few_step_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
        )
        return pred, None

    def _validate_current_teacher_feature_tensor(
        self,
        features: torch.Tensor,
        *,
        student_latent: torch.Tensor,
        token_indices: torch.Tensor,
        source: str,
    ) -> torch.Tensor:
        """Return one frozen FP32 teacher matrix with exact student geometry."""
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"{source} must be a torch.Tensor")
        if not features.is_floating_point():
            raise TypeError(f"{source} must be floating point")
        expected_shape = (
            int(student_latent.shape[0]),
            int(token_indices.numel()),
            int(self._dense_attr("feature_dim")),
        )
        if features.ndim != 3 or tuple(features.shape) != expected_shape:
            raise ValueError(
                f"{source} must have exact [B,N,D] shape "
                f"{expected_shape}, got {tuple(features.shape)}")
        if features.device != student_latent.device:
            raise ValueError(
                f"{source} and student latent must share a device")
        frozen = features.detach()
        if not bool(torch.isfinite(frozen).all().item()):
            raise FloatingPointError(f"{source} contains non-finite values")
        return frozen.float()

    def _current_teacher_features_for_fresh_video(
        self,
        *,
        student_latent: torch.Tensor,
        token_indices: torch.Tensor,
        extract_kwargs: Mapping[str, object],
        current_teacher_features: Optional[torch.Tensor],
        current_teacher_video01: Optional[torch.Tensor],
        current_teacher_latent: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Resolve the mutually exclusive matched-teacher input contract."""
        candidates = {
            "current_teacher_features": current_teacher_features,
            "current_teacher_video01": current_teacher_video01,
            "current_teacher_latent": current_teacher_latent,
        }
        supplied = [
            name for name, value in candidates.items() if value is not None
        ]
        if not getattr(self, "current_teacher_attraction_enabled", False):
            if supplied:
                raise ValueError(
                    f"{', '.join(supplied)} provided while exact "
                    "current-teacher attraction is disabled")
            return None
        if len(supplied) != 1:
            raise ValueError(
                "exactly one of current_teacher_features, "
                "current_teacher_video01, or current_teacher_latent must be "
                f"provided; got {supplied}")

        source = supplied[0]
        del candidates
        if source == "current_teacher_features":
            return self._validate_current_teacher_feature_tensor(
                current_teacher_features,
                student_latent=student_latent,
                token_indices=token_indices,
                source=source,
            )

        if source == "current_teacher_video01":
            teacher_video01 = current_teacher_video01
            if not isinstance(teacher_video01, torch.Tensor):
                raise TypeError(
                    "current_teacher_video01 must be a torch.Tensor")
            if not teacher_video01.is_floating_point():
                raise TypeError(
                    "current_teacher_video01 must be floating point")
            if (
                teacher_video01.ndim != 5
                or int(teacher_video01.shape[0])
                != int(student_latent.shape[0])
                or int(teacher_video01.shape[2]) != 3
                or int(teacher_video01.shape[1])
                < int(self._dense_attr("num_frames"))
                or int(teacher_video01.shape[3]) <= 0
                or int(teacher_video01.shape[4]) <= 0
            ):
                raise ValueError(
                    "current_teacher_video01 must have shape "
                    f"[B,T>={int(self._dense_attr('num_frames'))},3,H,W] "
                    "with the student batch size; got "
                    f"{tuple(teacher_video01.shape)}")
            student_sampling = getattr(
                self,
                "vjepa2_student_frame_sampling",
                getattr(
                    self, "vjepa2_dense_frame_sampling", "center_contiguous"),
            )
            if (
                student_sampling == "last_contiguous"
                and int(teacher_video01.shape[1]) not in {
                    int(self.vjepa2_dense_num_frames),
                    int(self.vjepa2_student_source_num_frames),
                }
            ):
                raise ValueError(
                    "last_contiguous teacher video must be either the exact "
                    "64-frame contract clip or its complete 81-frame source")
            if teacher_video01.device != student_latent.device:
                raise ValueError(
                    "current_teacher_video01 and student latent must share "
                    "a device")
            frozen_video01 = teacher_video01.detach()
            if not bool(torch.isfinite(frozen_video01).all().item()):
                raise FloatingPointError(
                    "current_teacher_video01 contains non-finite values")
            if (
                float(frozen_video01.min().item()) < 0.0
                or float(frozen_video01.max().item()) > 1.0
            ):
                raise ValueError(
                    "current_teacher_video01 must be normalized to [0,1]")
            extracted = self._extract_vjepa2_dense(
                frozen_video01,
                token_indices,
                **extract_kwargs,
            )
            del current_teacher_video01, frozen_video01, teacher_video01
        else:
            teacher_latent = current_teacher_latent
            if not isinstance(teacher_latent, torch.Tensor):
                raise TypeError(
                    "current_teacher_latent must be a torch.Tensor")
            if not teacher_latent.is_floating_point():
                raise TypeError(
                    "current_teacher_latent must be floating point")
            if tuple(teacher_latent.shape) != tuple(student_latent.shape):
                raise ValueError(
                    "current teacher/student latent shapes must match: "
                    f"{tuple(teacher_latent.shape)} != "
                    f"{tuple(student_latent.shape)}")
            if teacher_latent.device != student_latent.device:
                raise ValueError(
                    "current teacher/student latents must share a device")
            frozen_latent = teacher_latent.detach()
            if not bool(torch.isfinite(frozen_latent).all().item()):
                raise FloatingPointError(
                    "current teacher latent contains non-finite values")
            teacher_video01 = self._decode_video01(frozen_latent)
            extracted = self._extract_vjepa2_dense(
                teacher_video01,
                token_indices,
                **extract_kwargs,
            )
            del current_teacher_latent, teacher_latent, frozen_latent, teacher_video01

        return self._validate_current_teacher_feature_tensor(
            extracted,
            student_latent=student_latent,
            token_indices=token_indices,
            source=f"V-JEPA2 output for {source}",
        )

    @property
    def dual_video_mmd_enabled(self) -> bool:
        """Compatibility name for heterogeneous multi-encoder MMD."""
        return getattr(self, "feature_type", None) in {
            "dual_vjepa2_videomae_v2_dense_latent",
            "dual_dinov3_vjepa2_dense_latent",
            "triple_dinov3_vjepa2_videomae_v2_dense_latent",
        }

    def _dual_token_indices(
        self,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        indices = {
            "vjepa2": uniform_dense_token_indices(
                self.vjepa2_dense_raw_tokens, 1024, device=device),
        }
        if "dinov3" in self.dual_branch_registry:
            indices["dinov3"] = torch.tensor(
                self.dinov3_dense_frame_indices,
                device=device,
                dtype=torch.long,
            )
        if "videomae_v2" in self.dual_branch_registry:
            from video_fd.encoders.videomae_v2_dense_metadata import (
                uniform_segment_token_indices,
            )

            indices["videomae_v2"] = uniform_segment_token_indices(
                self.videomae_v2_encoder.dense_raw_tokens,
                256,
                device=device,
            )
        return indices

    def _validate_dual_feature_mapping(
        self,
        features: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        label: str,
    ) -> dict[str, torch.Tensor]:
        expected_keys = tuple(self.dual_branch_registry)
        if not isinstance(features, Mapping) or set(features) != set(expected_keys):
            raise ValueError(
                f"{label} must contain exactly branches {expected_keys}")
        result = {}
        for name in expected_keys:
            tensor = features[name]
            expected_shape = (
                batch_size,
                *self.dual_branch_registry[name]["sample_shape"],
            )
            if (
                not isinstance(tensor, torch.Tensor)
                or not tensor.is_floating_point()
                or tuple(tensor.shape) != expected_shape
                or tensor.device != device
            ):
                raise ValueError(
                    f"{label}[{name!r}] must be floating {expected_shape} "
                    f"on {device}; got {getattr(tensor, 'shape', None)}")
            frozen = tensor.detach().float()
            if not bool(torch.isfinite(frozen).all().item()):
                raise FloatingPointError(f"{label}[{name!r}] is non-finite")
            result[name] = frozen
        return result

    def _extract_multi_encoder_features(
        self,
        video01: torch.Tensor,
        token_indices: Mapping[str, torch.Tensor],
        *,
        gradient_checkpointing: bool,
    ) -> dict[str, torch.Tensor]:
        if int(video01.shape[1]) != 81:
            raise ValueError(
                "multi-encoder video feature extraction requires 81 frames")
        temporal_indices = torch.arange(
            17, 81, device=video01.device, dtype=torch.long
        ).unsqueeze(0).expand(int(video01.shape[0]), -1)
        features = {}
        if "dinov3" in self.dual_branch_registry:
            features["dinov3"] = self._extract_dinov3_dense(video01)
        features["vjepa2"] = self._extract_vjepa2_dense(
            video01,
            token_indices["vjepa2"],
            gradient_checkpointing=gradient_checkpointing,
            temporal_window_indices=temporal_indices,
        )
        if "videomae_v2" in self.dual_branch_registry:
            features["videomae_v2"] = (
                self.videomae_v2_encoder.forward_dense(
                    video01,
                    layer_index=15,
                    token_indices=token_indices["videomae_v2"],
                    gradient_checkpointing=gradient_checkpointing,
                )
            )
        return self._validate_dual_feature_mapping(
            features,
            batch_size=int(video01.shape[0]),
            device=video01.device,
            label="multi-encoder output",
        )

    @torch.no_grad()
    def _collect_multi_encoder_rollout_features(
        self,
        *,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        initial_latent: Optional[torch.Tensor],
        current_teacher_features: Optional[Mapping[str, torch.Tensor]],
        current_teacher_video01: Optional[torch.Tensor],
        current_teacher_latent: Optional[torch.Tensor],
        require_current_teacher: bool,
    ) -> Tuple[Mapping[str, torch.Tensor], dict]:
        if current_teacher_video01 is not None or current_teacher_latent is not None:
            raise ValueError(
                "dual training reads frozen teacher features from the bank; "
                "online teacher video/latent decoding is forbidden")
        if require_current_teacher != (current_teacher_features is not None):
            raise ValueError(
                "selected dual suffix rows require exactly one bank feature mapping")
        if not self.fresh_video_prepared_replay_enabled:
            raise ValueError(
                "dual video MMD requires prepared replay so both encoder VJPs "
                "share one decoded pixel tensor and one VAE VJP")

        generation_descriptor = self._sample_generator_input_descriptor(
            image_or_video_shape, initial_latent=initial_latent)
        saved_generation_noise = generation_descriptor["noise"].detach().clone()
        rng_state = self._capture_generation_rng()
        student_latent, _ = self._generate_student_latent(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=initial_latent,
            generation_descriptor=generation_descriptor,
        )
        token_indices = self._dual_token_indices(student_latent.device)
        teacher = None
        if require_current_teacher:
            teacher = self._validate_dual_feature_mapping(
                current_teacher_features,
                batch_size=int(student_latent.shape[0]),
                device=student_latent.device,
                label="dual teacher bank features",
            )
        student_video01 = self._decode_video01(student_latent)
        if int(student_video01.shape[1]) != 81:
            raise ValueError("dual student decode must produce exactly 81 frames")
        features = self._extract_multi_encoder_features(
            student_video01,
            token_indices,
            gradient_checkpointing=False,
        )
        descriptor = {
            "rng_state": rng_state,
            "latent_shape": tuple(int(value) for value in student_latent.shape),
            "generation_descriptor": {
                "version": int(generation_descriptor["version"]),
                "num_generated_blocks": int(
                    generation_descriptor["num_generated_blocks"]),
                "noise": saved_generation_noise,
                **({
                    "trajectory_seeds": generation_descriptor[
                        "trajectory_seeds"].detach().clone(),
                } if "trajectory_seeds" in generation_descriptor else {}),
            },
            "student_latent": student_latent.detach().clone(),
        }
        if teacher is not None:
            descriptor["current_teacher_features"] = teacher
        cached_video01 = student_video01.detach()
        if self.fidel_p0_pixel_cache_device == "cpu":
            cached_video01 = cached_video01.to(
                device="cpu", dtype=torch.float32, copy=True)
        descriptor["cached_student_video01"] = cached_video01
        return features, descriptor

    def _prepare_multi_encoder_latent_gradients(
        self,
        *,
        replay_descriptor: dict,
        expected_features: Mapping[str, torch.Tensor],
        feature_cotangent: Mapping[str, torch.Tensor],
        frame_fd_feature_cotangents: Optional[dict],
        replay_atol: float,
        replay_rtol: float,
        collect_branch_latent_diagnostics: bool,
    ) -> dict:
        if frame_fd_feature_cotangents is not None or self.frame_fd_fusion_enabled:
            raise ValueError("triple video MMD requires frame-FD disabled")
        if collect_branch_latent_diagnostics:
            raise ValueError("triple video MMD has no frame diagnostics")
        required = {
            "rng_state",
            "latent_shape",
            "generation_descriptor",
            "student_latent",
            "cached_student_video01",
        }
        if not isinstance(replay_descriptor, dict) or set(replay_descriptor) != required:
            raise ValueError("invalid cached triple-video replay descriptor")
        student_latent = replay_descriptor["student_latent"]
        cached = replay_descriptor.pop("cached_student_video01")
        student_video01 = cached.to(
            device=student_latent.device,
            dtype=torch.float32,
            non_blocking=bool(cached.is_pinned()),
        )
        batch_size = int(student_latent.shape[0])
        # The finalized global GradCache is intentionally offloaded before
        # encoder/VAE replay. Stage only this sample's cached features and
        # cotangents back to the active rank; the strict validator below still
        # enforces shape, dtype, finiteness, and the destination device.
        def _stage_replay_mapping(
            mapping: Mapping[str, torch.Tensor],
        ) -> Mapping[str, torch.Tensor]:
            if not isinstance(mapping, Mapping):
                return mapping
            return {
                name: (
                    value.to(
                        device=student_latent.device,
                        non_blocking=bool(value.is_pinned()),
                    )
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for name, value in mapping.items()
            }

        expected_features = _stage_replay_mapping(expected_features)
        feature_cotangent = _stage_replay_mapping(feature_cotangent)
        expected = self._validate_dual_feature_mapping(
            expected_features,
            batch_size=batch_size,
            device=student_latent.device,
            label="cached triple student features",
        )
        cotangent = self._validate_dual_feature_mapping(
            feature_cotangent,
            batch_size=batch_size,
            device=student_latent.device,
            label="triple feature cotangent",
        )
        tokens = self._dual_token_indices(student_latent.device)
        temporal_indices = torch.arange(
            17, 81, device=student_latent.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)

        combined_pixel = None
        pixel_shape = None
        pixel_norms = {}
        replay_maxima = {}
        if "dinov3" in self.dual_branch_registry:
            with torch.enable_grad():
                pixel_dino, dino_max_abs = self._replay_dinov3_dense_to_pixels(
                    student_video01,
                    expected_features=expected["dinov3"],
                    feature_cotangent=cotangent["dinov3"],
                    replay_atol=replay_atol,
                    replay_rtol=replay_rtol,
                )
            combined_pixel, pixel_shape, pixel_norms["dinov3"] = (
                _accumulate_fp32_pixel_cotangent(
                    combined_pixel,
                    pixel_dino,
                    pixel_shape,
                    branch_name="dinov3",
                )
            )
            replay_maxima["dinov3"] = dino_max_abs
            del pixel_dino

        with torch.enable_grad():
            replay_vj, pixel_vj = self._replay_dense_encoder_vjp(
                student_video01,
                tokens["vjepa2"],
                expected["vjepa2"],
                cotangent["vjepa2"],
                gradient_checkpointing=(
                    self.vjepa2_dense_gradient_checkpointing),
                temporal_window_indices=temporal_indices,
            )
        if not torch.allclose(
            replay_vj, expected["vjepa2"],
            atol=float(replay_atol), rtol=float(replay_rtol),
        ):
            raise RuntimeError("triple V-JEPA2 replay changed cached features")
        combined_pixel, pixel_shape, pixel_norms["vjepa2"] = (
            _accumulate_fp32_pixel_cotangent(
                combined_pixel,
                pixel_vj,
                pixel_shape,
                branch_name="vjepa2",
            )
        )
        replay_maxima["vjepa2"] = (
            replay_vj - expected["vjepa2"]).abs().max().detach()
        del replay_vj, pixel_vj

        if "videomae_v2" in self.dual_branch_registry:
            with torch.enable_grad():
                replay_vm, pixel_vm = self.videomae_v2_encoder.replay_dense_vjp(
                    student_video01.detach().float(),
                    layer_indices=(15,),
                    token_indices=tokens["videomae_v2"],
                    expected_features=expected["videomae_v2"],
                    feature_cotangent=cotangent["videomae_v2"],
                    gradient_checkpointing=(
                        self.videomae_v2_dense_gradient_checkpointing),
                )
            if not torch.allclose(
                replay_vm, expected["videomae_v2"],
                atol=float(replay_atol), rtol=float(replay_rtol),
            ):
                raise RuntimeError(
                    "VideoMAE-v2 replay changed cached features")
            combined_pixel, pixel_shape, pixel_norms["videomae_v2"] = (
                _accumulate_fp32_pixel_cotangent(
                    combined_pixel,
                    pixel_vm,
                    pixel_shape,
                    branch_name="videomae_v2",
                )
            )
            replay_maxima["videomae_v2"] = (
                replay_vm - expected["videomae_v2"]).abs().max().detach()
            del replay_vm, pixel_vm
        del student_video01

        if set(pixel_norms) != set(self.dual_branch_registry):
            raise RuntimeError(
                "triple encoder replay did not visit every branch")
        if not bool(torch.isfinite(combined_pixel).all().item()):
            raise FloatingPointError(
                "combined triple pixel cotangent is non-finite")
        latent_cotangent = replay_decoder_vjp(
            self.vae,
            student_latent.detach(),
            combined_pixel,
            independent_first_frame=self.independent_first_frame,
        ).detach().float()
        if latent_cotangent.shape != student_latent.shape:
            raise ValueError(
                "triple Wan VAE VJP returned an invalid latent shape")
        if not bool(torch.isfinite(latent_cotangent).all().item()):
            raise FloatingPointError(
                "triple Wan VAE VJP returned non-finite values")

        prefix = self.dense_log_prefix
        feature_norm = torch.sqrt(torch.stack([
            value.norm().float().square() for value in cotangent.values()
        ]).sum())
        replay_max_abs = torch.stack([
            value.float() for value in replay_maxima.values()
        ]).max()
        replay_log = {
            f"{prefix}_feature_grad_norm": feature_norm.detach(),
            f"{prefix}_video_grad_norm": combined_pixel.norm().detach(),
            f"{prefix}_latent_grad_norm": latent_cotangent.norm().detach(),
            f"{prefix}_replay_max_abs": replay_max_abs.detach(),
            "combined_latent_grad_norm": latent_cotangent.norm().detach(),
            "fresh_video_prepared_vae_vjp": latent_cotangent.new_ones(()),
            f"{prefix}_encoder_vjp_count": latent_cotangent.new_tensor(
                len(pixel_norms), dtype=torch.long),
            f"{prefix}_vae_vjp_count": latent_cotangent.new_tensor(
                1, dtype=torch.long),
        }
        for name in self.dual_branch_registry:
            replay_log[f"{name}_dense_feature_grad_norm"] = (
                cotangent[name].norm().detach())
            replay_log[f"{name}_dense_video_grad_norm"] = (
                pixel_norms[name])
            replay_log[f"{name}_dense_replay_max_abs"] = (
                replay_maxima[name].detach())
        return {
            "latent_cotangent": latent_cotangent.detach().to(
                device="cpu",
                dtype=torch.float32,
                non_blocking=False,
            ).contiguous(),
            "replay_log": replay_log,
        }

    @torch.no_grad()
    def collect_rollout_features(
        self,
        *,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        initial_latent: Optional[torch.Tensor] = None,
        current_teacher_features: Optional[torch.Tensor] = None,
        current_teacher_video01: Optional[torch.Tensor] = None,
        current_teacher_latent: Optional[torch.Tensor] = None,
        require_current_teacher: bool = True,
        vjepa2_window_sample_keys=None,
    ) -> Tuple[torch.Tensor, dict]:
        """First GradCache pass: generate and encode one fresh microbatch."""
        if not self.uses_fresh_video_gradcache:
            raise RuntimeError(
                "collect_rollout_features requires "
                "wan_mmd_estimator=nystrom_exact_repulsion")
        if self.dual_video_mmd_enabled:
            return self._collect_multi_encoder_rollout_features(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                initial_latent=initial_latent,
                current_teacher_features=current_teacher_features,
                current_teacher_video01=current_teacher_video01,
                current_teacher_latent=current_teacher_latent,
                require_current_teacher=require_current_teacher,
            )
        if type(require_current_teacher) is not bool:
            raise TypeError("require_current_teacher must be a bool")
        supplied_teacher_inputs = [
            name for name, value in {
                "current_teacher_features": current_teacher_features,
                "current_teacher_video01": current_teacher_video01,
                "current_teacher_latent": current_teacher_latent,
            }.items() if value is not None
        ]
        if not require_current_teacher and supplied_teacher_inputs:
            raise ValueError(
                "require_current_teacher=false rejects all teacher inputs; "
                f"got {supplied_teacher_inputs}")
        first_pass_error = None
        try:
            generation_descriptor = self._sample_generator_input_descriptor(
                image_or_video_shape,
                initial_latent=initial_latent,
            )
            saved_generation_noise = (
                generation_descriptor["noise"].detach().clone())
            # Initial Gaussian noise and the block count are explicit.  The
            # saved state begins *after* sampling them and drives any
            # stochastic denoise/renoise choices inside the rollout pipeline.
            rng_state = self._capture_generation_rng()
            student_latent, _ = self._generate_student_latent(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                initial_latent=initial_latent,
                generation_descriptor=generation_descriptor,
            )
            matched_teacher_features = None
            teacher_extract_seconds = None
            student_video01 = self._decode_video01(student_latent)
            token_indices = None
            temporal_window_indices = None
            if self._uses_dinov3_dense_backend():
                if vjepa2_window_sample_keys is not None:
                    raise ValueError(
                        "DINOv3 fixed-frame sampling rejects V-JEPA2 keys")
                features = self._extract_dinov3_dense(
                    student_video01).detach().float()
                if require_current_teacher:
                    if (
                        current_teacher_video01 is not None
                        or current_teacher_latent is not None
                        or not isinstance(current_teacher_features, torch.Tensor)
                    ):
                        raise ValueError(
                            "DINOv3 current teacher must be detached features")
                    matched_teacher_features = (
                        current_teacher_features.detach().float())
                    if tuple(matched_teacher_features.shape) != tuple(features.shape):
                        raise ValueError(
                            "DINOv3 teacher/student feature shapes differ")
            else:
                token_indices = self._vjepa2_token_indices(
                    student_latent.device)
                temporal_window_indices = (
                    self._vjepa2_temporal_window_indices(
                        vjepa2_window_sample_keys,
                        batch_size=int(student_latent.shape[0]),
                        device=student_latent.device,
                    )
                )
                extract_kwargs = {"gradient_checkpointing": False}
                if temporal_window_indices is not None:
                    extract_kwargs["temporal_window_indices"] = (
                        temporal_window_indices)
                # A live-video cache miss supplies an already selected 64-frame
                # clip. A latent or complete 81-frame video uses the student's
                # explicit temporal indices.
                teacher_extract_kwargs = dict(extract_kwargs)
                if (
                    current_teacher_video01 is not None
                    and temporal_window_indices is not None
                    and int(current_teacher_video01.shape[1])
                    == int(self.vjepa2_dense_num_frames)
                ):
                    teacher_extract_kwargs.pop("temporal_window_indices", None)
                teacher_requires_extraction = bool(
                    require_current_teacher
                    and (
                        current_teacher_video01 is not None
                        or current_teacher_latent is not None
                    )
                )
                if teacher_requires_extraction and student_latent.is_cuda:
                    torch.cuda.synchronize(student_latent.device)
                teacher_extract_started = time.perf_counter()
                matched_teacher_features = (
                    self._current_teacher_features_for_fresh_video(
                        student_latent=student_latent,
                        token_indices=token_indices,
                        extract_kwargs=teacher_extract_kwargs,
                        current_teacher_features=current_teacher_features,
                        current_teacher_video01=current_teacher_video01,
                        current_teacher_latent=current_teacher_latent,
                    )
                    if require_current_teacher
                    else None
                )
                if teacher_requires_extraction and student_latent.is_cuda:
                    torch.cuda.synchronize(student_latent.device)
                teacher_extract_seconds = (
                    time.perf_counter() - teacher_extract_started
                    if teacher_requires_extraction else None
                )
                # The trainer transfers ownership with dict.pop(). Drop the three
                # input aliases now so an uncached live video can be reclaimed
                # before student VAE decode/V-JEPA2 extraction begins.
                current_teacher_features = None
                current_teacher_video01 = None
                current_teacher_latent = None

                student_video01 = self._decode_video01(student_latent)
                features = self._extract_vjepa2_dense(
                    student_video01,
                    token_indices,
                    **extract_kwargs,
                ).detach().float()
                if (
                    matched_teacher_features is not None
                    and tuple(matched_teacher_features.shape)
                    != tuple(features.shape)
                ):
                    raise ValueError(
                        "matched teacher/student V-JEPA2 feature shapes must "
                        "match exactly: "
                        f"{tuple(matched_teacher_features.shape)} != "
                        f"{tuple(features.shape)}")
        except BaseException as exc:
            first_pass_error = exc
        if not self._all_ranks_true(
            first_pass_error is None, self.device
        ):
            message = (
                f": {type(first_pass_error).__name__}: {first_pass_error}"
                if first_pass_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "fresh video generation/VAE/dense-encoder first pass failed"
                + message
            ) from first_pass_error
        local_features_finite = bool(torch.isfinite(features).all().item())
        if not self._all_ranks_true(
            local_features_finite, features.device
        ):
            raise FloatingPointError(
                "non-finite dense feature in fresh GradCache first pass "
                "on at least one rank")
        if matched_teacher_features is not None:
            local_teacher_features_finite = bool(
                torch.isfinite(matched_teacher_features).all().item())
            if not self._all_ranks_true(
                local_teacher_features_finite,
                matched_teacher_features.device,
            ):
                raise FloatingPointError(
                    "non-finite matched teacher feature in fresh GradCache "
                    "first pass on at least one rank")
        descriptor = {
            "rng_state": rng_state,
            "latent_shape": tuple(int(v) for v in student_latent.shape),
            "generation_descriptor": {
                "version": int(generation_descriptor["version"]),
                "num_generated_blocks": int(
                    generation_descriptor["num_generated_blocks"]),
                "noise": saved_generation_noise,
                **({
                    "trajectory_seeds": generation_descriptor[
                        "trajectory_seeds"].detach().clone(),
                } if "trajectory_seeds" in generation_descriptor else {}),
            },
            "student_latent": student_latent.detach().clone(),
        }
        if matched_teacher_features is not None:
            # The trainer moves this detached value into the update-local
            # GradCache immediately; teacher latents/pixels are never replayed.
            descriptor["current_teacher_features"] = (
                matched_teacher_features)
            if teacher_extract_seconds is not None:
                descriptor["current_teacher_extract_seconds"] = float(
                    teacher_extract_seconds)
        if temporal_window_indices is not None:
            descriptor["vjepa2_frame_indices"] = (
                temporal_window_indices.detach().clone())
        if self._uses_dinov3_dense_backend():
            descriptor["dinov3_frame_indices"] = torch.tensor(
                self.dinov3_dense_frame_indices,
                device=student_latent.device,
                dtype=torch.long,
            )
        if self.uses_random_token_reference_bank:
            descriptor.update({
                "vjepa2_token_indices": token_indices.detach().clone(),
                "vjepa2_token_indices_sha256": (
                    self.vjepa2_active_token_indices_hash),
                "vjepa2_reference_slot": int(
                    self.vjepa2_active_reference_slot),
                "vjepa2_reference_pack_hash": (
                    self.vjepa2_nystrom_reference_hash),
            })
        if self.frame_fd_fusion_enabled:
            # Sampling may be random, but replay never consults ambient RNG:
            # the explicit seed, [B,K] positions and detached feature values
            # are all retained in this update-local descriptor.
            frame_error = None
            try:
                frame_seed = int(torch.randint(
                    0,
                    2**31 - 1,
                    (),
                    device=student_latent.device,
                    dtype=torch.long,
                ).item())
                frame_extraction = self.frame_fd_head.extract_no_grad(
                    student_video01, seed=frame_seed)
            except BaseException as exc:
                frame_error = exc
                frame_seed = -1
                frame_extraction = None
            if not self._all_ranks_true(
                frame_error is None, student_latent.device
            ):
                message = (
                    f": {type(frame_error).__name__}: {frame_error}"
                    if frame_error is not None
                    else " on another rank"
                )
                raise RuntimeError(
                    "frame FD first-pass extraction failed" + message
                ) from frame_error
            descriptor["frame_fd_seed"] = frame_seed
            descriptor["frame_fd_extraction"] = frame_extraction
        if getattr(
            self, "fresh_video_prepared_replay_enabled",
            getattr(self, "fidel_p0_replay_enabled", False),
        ):
            cache_error = None
            try:
                # Keep the exact first-pass FP32 pixels until update-wide
                # feature cotangents are available. They are consumed before
                # any differentiable generator replay, so this storage never
                # overlaps the generator graph peak.
                cached_video01 = student_video01.detach()
                if self.fidel_p0_pixel_cache_device == "cpu":
                    cached_video01 = cached_video01.to(
                        device="cpu", dtype=torch.float32, copy=True)
                descriptor["cached_student_video01"] = cached_video01
            except BaseException as exc:
                cache_error = exc
            if not self._all_ranks_true(
                cache_error is None, student_latent.device
            ):
                message = (
                    f": {type(cache_error).__name__}: {cache_error}"
                    if cache_error is not None
                    else " on another rank"
                )
                raise RuntimeError(
                    "fresh-video pixel caching failed" + message
                ) from cache_error
        return features, descriptor

    def prepare_latent_gradients(
        self,
        *,
        replay_descriptor: dict,
        expected_features: torch.Tensor,
        feature_cotangent: torch.Tensor,
        frame_fd_feature_cotangents: Optional[dict],
        replay_atol: float = 1e-5,
        replay_rtol: float = 1e-4,
        collect_branch_latent_diagnostics: bool = False,
    ) -> dict:
        """Prepare a detached latent cotangent before generator replay.

        The first-pass decoded pixels are consumed here. Frozen feature encoder,
        image-encoder and VAE VJPs therefore finish before an FSDP generator
        graph exists. On regular steps the two pixel cotangents are added before
        the linear decoder VJP, so only one causal VAE replay is required.
        """
        if self.dual_video_mmd_enabled:
            return self._prepare_multi_encoder_latent_gradients(
                replay_descriptor=replay_descriptor,
                expected_features=expected_features,
                feature_cotangent=feature_cotangent,
                frame_fd_feature_cotangents=frame_fd_feature_cotangents,
                replay_atol=replay_atol,
                replay_rtol=replay_rtol,
                collect_branch_latent_diagnostics=(
                    collect_branch_latent_diagnostics),
            )
        prepared_enabled = getattr(
            self,
            "fresh_video_prepared_replay_enabled",
            getattr(self, "fidel_p0_replay_enabled", False),
        )
        if not prepared_enabled or not self.uses_fresh_video_gradcache:
            raise RuntimeError(
                "prepared replay is unavailable for this fresh-video path")
        if collect_branch_latent_diagnostics and not self.fidel_two_level_enabled:
            raise ValueError(
                "branch latent diagnostics require two-level FiDEL")
        prepare_error = None
        prepared = None
        try:
            required_descriptor_keys = {
                "rng_state",
                "latent_shape",
                "generation_descriptor",
                "student_latent",
                "cached_student_video01",
            }
            if self.frame_fd_fusion_enabled:
                required_descriptor_keys.update({
                    "frame_fd_seed",
                    "frame_fd_extraction",
                })
            if getattr(
                self, "vjepa2_student_frame_sampling",
                getattr(self, "vjepa2_dense_frame_sampling", "center_contiguous"),
            ) != "center_contiguous":
                required_descriptor_keys.add("vjepa2_frame_indices")
            if self._uses_dinov3_dense_backend():
                required_descriptor_keys.add("dinov3_frame_indices")
            if self.uses_random_token_reference_bank:
                required_descriptor_keys.update({
                    "vjepa2_token_indices",
                    "vjepa2_token_indices_sha256",
                    "vjepa2_reference_slot",
                    "vjepa2_reference_pack_hash",
                })
            if (
                not isinstance(replay_descriptor, dict)
                or set(replay_descriptor) != required_descriptor_keys
            ):
                raise ValueError(
                    "invalid cached fresh-video replay descriptor")
            # Pop immediately so the descriptor returns to the strict legacy
            # key schema before generator replay. The trainer also removes it
            # in its finally block for skip/error paths.
            cached_student_video01 = replay_descriptor.pop(
                "cached_student_video01")
            student_latent = replay_descriptor["student_latent"]
            if (
                not isinstance(student_latent, torch.Tensor)
                or tuple(student_latent.shape)
                != tuple(replay_descriptor["latent_shape"])
            ):
                raise ValueError(
                    "cached first-pass latent has an invalid shape")
            if not isinstance(cached_student_video01, torch.Tensor):
                raise TypeError("cached first-pass pixels must be a tensor")
            student_video01 = cached_student_video01.to(
                device=student_latent.device,
                dtype=torch.float32,
                non_blocking=bool(cached_student_video01.is_pinned()),
            )
            del cached_student_video01
            if (
                student_video01.ndim != 5
                or student_video01.shape[0] != student_latent.shape[0]
                or student_video01.shape[2] != 3
            ):
                raise ValueError(
                    "cached first-pass pixels have an invalid shape")
            if not bool(torch.isfinite(student_video01).all().item()):
                raise FloatingPointError(
                    "cached first-pass pixels are non-finite")

            temporal_window_indices = None
            token_indices = None
            if self._uses_dinov3_dense_backend():
                self._dinov3_descriptor_frame_indices(
                    replay_descriptor, device=student_latent.device)
            elif self.uses_random_token_reference_bank:
                temporal_window_indices = (
                    self._vjepa2_descriptor_window_indices(
                        replay_descriptor, device=student_latent.device)
                )
                token_indices = replay_descriptor[
                    "vjepa2_token_indices"].to(
                        device=student_latent.device, dtype=torch.long)
                expected_indices = self._vjepa2_token_indices(
                    student_latent.device)
                reference_matches = bool(
                    int(replay_descriptor["vjepa2_reference_slot"])
                    == int(self.vjepa2_active_reference_slot)
                    and replay_descriptor["vjepa2_token_indices_sha256"]
                    == self.vjepa2_active_token_indices_hash
                    and replay_descriptor["vjepa2_reference_pack_hash"]
                    == self.vjepa2_nystrom_reference_hash
                    and torch.equal(token_indices, expected_indices)
                )
                if not reference_matches:
                    raise RuntimeError(
                        "random-token reference changed before frozen replay")
            else:
                temporal_window_indices = (
                    self._vjepa2_descriptor_window_indices(
                        replay_descriptor, device=student_latent.device)
                )
                token_indices = self._vjepa2_token_indices(
                    student_latent.device)

            expected_features = expected_features.to(
                device=student_latent.device, dtype=torch.float32)
            feature_cotangent = feature_cotangent.to(
                device=student_latent.device, dtype=torch.float32)
            if feature_cotangent.shape != expected_features.shape:
                raise ValueError(
                    "feature cotangent shape does not match cached features: "
                    f"{tuple(feature_cotangent.shape)} != "
                    f"{tuple(expected_features.shape)}")
            if not bool(torch.isfinite(feature_cotangent).all().item()):
                raise FloatingPointError(
                    "non-finite fresh dense feature cotangent")
            layer_feature_grad_norms = {}
            if getattr(self, "vjepa2_multilayer_enabled", False):
                expected_transport_tokens = (
                    len(self.vjepa2_dense_layers)
                    * self.vjepa2_dense_num_tokens
                )
                if int(feature_cotangent.shape[1]) != (
                    expected_transport_tokens
                ):
                    raise ValueError(
                        "multi-layer feature cotangent has an invalid "
                        "transport token count")
                layer_feature_grad_norms = {
                    f"vjepa2_dense_l{layer}_feature_grad_norm": (
                        part.norm().detach())
                    for layer, part in zip(
                        self.vjepa2_dense_layers,
                        feature_cotangent.split(
                            self.vjepa2_dense_num_tokens, dim=1),
                    )
                }

            if self._uses_dinov3_dense_backend():
                video_cotangent, replay_max_abs = (
                    self._replay_dinov3_dense_to_pixels(
                        student_video01,
                        expected_features=expected_features,
                        feature_cotangent=feature_cotangent,
                        replay_atol=float(replay_atol),
                        replay_rtol=float(replay_rtol),
                    )
                )
            else:
                with torch.enable_grad():
                    student_video_leaf = (
                        student_video01.detach().requires_grad_(True))
                    replay_extract_kwargs = {
                        "gradient_checkpointing": (
                            self.vjepa2_dense_gradient_checkpointing),
                    }
                    if temporal_window_indices is not None:
                        replay_extract_kwargs["temporal_window_indices"] = (
                            temporal_window_indices)
                    replay_features = self._extract_vjepa2_dense(
                        student_video_leaf,
                        token_indices,
                        **replay_extract_kwargs,
                    ).float()
                    if replay_features.shape != expected_features.shape:
                        raise RuntimeError(
                            "fresh feature replay changed matrix shape: "
                            f"{tuple(replay_features.shape)} != "
                            f"{tuple(expected_features.shape)}")
                    replay_max_abs = (
                        replay_features.detach()
                        - expected_features).abs().max()
                    if not bool(torch.allclose(
                        replay_features.detach(),
                        expected_features,
                        atol=float(replay_atol),
                        rtol=float(replay_rtol),
                    )):
                        raise RuntimeError(
                            "cached pixels changed V-JEPA2 features "
                            f"(max_abs={float(replay_max_abs):.6g})")
                    encoder_surrogate = (
                        replay_features * feature_cotangent).sum()
                    (video_cotangent,) = torch.autograd.grad(
                        encoder_surrogate,
                        student_video_leaf,
                        retain_graph=False,
                        create_graph=False,
                    )
                del encoder_surrogate, student_video_leaf, replay_features
            video_cotangent = video_cotangent.detach().float()
            if not bool(torch.isfinite(video_cotangent).all().item()):
                raise FloatingPointError(
                    "non-finite fresh dense-encoder pixel cotangent")
            mmd_video_grad_norm = video_cotangent.norm().detach()

            frame_pixel_cotangent = None
            frame_feature_grad_norm = video_cotangent.new_zeros(())
            frame_pixel_grad_norm = video_cotangent.new_zeros(())
            frame_dynamic_weight = video_cotangent.new_zeros(())
            if self.frame_fd_fusion_enabled:
                if frame_fd_feature_cotangents is None:
                    raise ValueError(
                        "frame fusion replay requires feature cotangents")
                frame_extraction = replay_descriptor["frame_fd_extraction"]
                frame_pixel_cotangent = (
                    self.frame_fd_head.replay_feature_cotangents_to_pixels(
                        student_video01,
                        frame_extraction.frame_indices,
                        frame_fd_feature_cotangents,
                        expected_features=frame_extraction.features,
                        replay_atol=float(replay_atol),
                        replay_rtol=float(replay_rtol),
                    ).detach().float()
                )
                if (
                    frame_pixel_cotangent.shape != video_cotangent.shape
                    or frame_pixel_cotangent.device != video_cotangent.device
                ):
                    raise ValueError(
                        "frame/video pixel cotangents are incompatible")
                if not bool(torch.isfinite(
                    frame_pixel_cotangent).all().item()
                ):
                    raise FloatingPointError(
                        "frame pixel cotangent is non-finite")
                frame_feature_grad_norm = torch.sqrt(sum(
                    value.detach().float().square().sum()
                    for value in frame_fd_feature_cotangents.values()
                ))
                frame_pixel_grad_norm = frame_pixel_cotangent.norm().detach()
                frame_dynamic_weight = video_cotangent.new_ones(())
            elif frame_fd_feature_cotangents is not None:
                raise ValueError(
                    "frame cotangents were provided while fusion is disabled")
            del student_video01

            branch_diagnostics_available = video_cotangent.new_zeros(())
            merged_vae_vjp = video_cotangent.new_zeros(())
            if collect_branch_latent_diagnostics:
                mmd_latent_cotangent = replay_decoder_vjp(
                    self.vae,
                    student_latent.detach(),
                    video_cotangent,
                    independent_first_frame=self.independent_first_frame,
                ).detach().float()
                fd_latent_cotangent = replay_decoder_vjp(
                    self.vae,
                    student_latent.detach(),
                    frame_pixel_cotangent,
                    independent_first_frame=self.independent_first_frame,
                ).detach().float()
                latent_cotangent = (
                    mmd_latent_cotangent + fd_latent_cotangent)
                mmd_norm = mmd_latent_cotangent.norm().detach()
                fd_norm = fd_latent_cotangent.norm().detach()
                cosine_denominator = mmd_norm * fd_norm
                if bool((cosine_denominator > 0).item()):
                    branch_gradient_cosine = (
                        (mmd_latent_cotangent * fd_latent_cotangent).sum()
                        / cosine_denominator
                    ).detach()
                else:
                    branch_gradient_cosine = mmd_norm.new_zeros(())
                branch_diagnostics_available = latent_cotangent.new_ones(())
                del mmd_latent_cotangent, fd_latent_cotangent
            else:
                # A decoder VJP is linear at the fixed first-pass latent. FiDEL
                # merges its scaled video/frame pixel cotangents; video-only
                # MMD replays the video cotangent directly.
                if frame_pixel_cotangent is not None:
                    video_cotangent.add_(frame_pixel_cotangent)
                    if not bool(torch.isfinite(video_cotangent).all().item()):
                        raise FloatingPointError(
                            "non-finite combined pixel cotangent")
                    merged_vae_vjp = video_cotangent.new_ones(())
                latent_cotangent = replay_decoder_vjp(
                    self.vae,
                    student_latent.detach(),
                    video_cotangent,
                    independent_first_frame=self.independent_first_frame,
                ).detach().float()
            del video_cotangent
            if frame_pixel_cotangent is not None:
                del frame_pixel_cotangent

            if latent_cotangent.shape != student_latent.shape:
                raise ValueError(
                    "combined Wan VAE VJP returned an invalid latent shape")
            if not bool(torch.isfinite(latent_cotangent).all().item()):
                raise FloatingPointError(
                    "combined Wan VAE VJP returned a non-finite cotangent")
            dense_prefix = self._dense_metric_prefix()
            replay_log = {
                self._dense_key("feature_grad_norm"): (
                    feature_cotangent.norm().detach()),
                self._dense_key("video_grad_norm"): mmd_video_grad_norm,
                "combined_latent_grad_norm": (
                    latent_cotangent.norm().detach()),
                self._dense_key("replay_max_abs"): replay_max_abs.detach(),
                "fresh_video_prepared_vae_vjp": (
                    latent_cotangent.new_ones(())),
            }
            replay_log.update(layer_feature_grad_norms)
            if self.fidel_two_level_enabled:
                replay_log.update({
                    "frame_fd_feature_grad_norm": frame_feature_grad_norm,
                    "frame_fd_pixel_grad_norm": frame_pixel_grad_norm,
                    "frame_fd_dynamic_weight": frame_dynamic_weight,
                    "fidel_branch_latent_diagnostics_available": (
                        branch_diagnostics_available),
                    "fidel_merged_vae_vjp": merged_vae_vjp,
                })
            else:
                replay_log[self._dense_key("latent_grad_norm")] = (
                    latent_cotangent.norm().detach())
            if collect_branch_latent_diagnostics:
                replay_log.update({
                    self._dense_key("latent_grad_norm"): mmd_norm,
                    "frame_fd_latent_grad_norm": fd_norm,
                    "video_frame_latent_grad_cosine": (
                        branch_gradient_cosine),
                })
            prepared = {
                "latent_cotangent": latent_cotangent.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                    non_blocking=False,
                ).contiguous(),
                "replay_log": replay_log,
            }
        except BaseException as exc:
            prepare_error = exc
        prepare_ok = self._all_ranks_true(
            prepare_error is None, self.device)
        if not prepare_ok:
            all_errors_are_nonfinite = self._all_ranks_true(
                prepare_error is None
                or isinstance(prepare_error, FloatingPointError),
                self.device,
            )
            message = (
                f": {type(prepare_error).__name__}: {prepare_error}"
                if prepare_error is not None
                else " on another rank"
            )
            if all_errors_are_nonfinite:
                raise FloatingPointError(
                    "fresh-video prepared replay produced a non-finite value" + message
                ) from prepare_error
            raise RuntimeError(
                "fresh-video replay preparation failed" + message
            ) from prepare_error
        return prepared

    def replay_generator_gradients(
        self,
        *,
        replay_descriptor: dict,
        expected_features: torch.Tensor,
        feature_cotangent: torch.Tensor,
        frame_fd_feature_cotangents: Optional[dict] = None,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        initial_latent: Optional[torch.Tensor] = None,
        prepared_replay: Optional[dict] = None,
        replay_atol: float = 1e-5,
        replay_rtol: float = 1e-4,
    ) -> Tuple[torch.Tensor, dict]:
        """Second GradCache pass: replay one video microbatch and encoder/VAE VJP."""
        if not self.uses_fresh_video_gradcache:
            raise RuntimeError(
                "replay_generator_gradients requires "
                "wan_mmd_estimator=nystrom_exact_repulsion")
        if self.dual_video_mmd_enabled and prepared_replay is None:
            raise RuntimeError("dual video MMD requires prepared replay")
        required_descriptor_keys = {
            "rng_state",
            "latent_shape",
            "generation_descriptor",
            "student_latent",
        }
        # The dual prepared path has already consumed encoder-specific frame
        # and token fields. Generator replay intentionally retains only the
        # common generator identity keys.
        if not self.dual_video_mmd_enabled:
            if self.frame_fd_fusion_enabled:
                required_descriptor_keys.update({
                    "frame_fd_seed",
                    "frame_fd_extraction",
                })
            if getattr(
                self, "vjepa2_student_frame_sampling",
                getattr(self, "vjepa2_dense_frame_sampling", "center_contiguous"),
            ) != "center_contiguous":
                required_descriptor_keys.add("vjepa2_frame_indices")
            if self._uses_dinov3_dense_backend():
                required_descriptor_keys.add("dinov3_frame_indices")
            if self.uses_random_token_reference_bank:
                required_descriptor_keys.update({
                    "vjepa2_token_indices",
                    "vjepa2_token_indices_sha256",
                    "vjepa2_reference_slot",
                    "vjepa2_reference_pack_hash",
                })
        if (
            not isinstance(replay_descriptor, dict)
            or set(replay_descriptor) != required_descriptor_keys
        ):
            raise ValueError("invalid fresh-video replay descriptor")
        generator_replay_error = None
        try:
            with self._replay_generation_rng(replay_descriptor["rng_state"]):
                activation_context = (
                    torch.autograd.graph.save_on_cpu(pin_memory=True)
                    if getattr(
                        self, "generator_replay_activation_cpu_offload", False)
                    else nullcontext()
                )
                with activation_context:
                    student_latent, gradient_mask = self._generate_student_latent(
                        image_or_video_shape=image_or_video_shape,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        initial_latent=initial_latent,
                        generation_descriptor=replay_descriptor[
                            "generation_descriptor"],
                    )
        except BaseException as exc:
            generator_replay_error = exc
        if not self._all_ranks_true(
            generator_replay_error is None, self.device
        ):
            message = (
                f": {type(generator_replay_error).__name__}: "
                f"{generator_replay_error}"
                if generator_replay_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "fresh generator replay failed" + message
            ) from generator_replay_error
        expected_latent_shape = tuple(replay_descriptor["latent_shape"])
        local_latent_shape_ok = (
            tuple(student_latent.shape) == expected_latent_shape)
        if not self._all_ranks_true(
            local_latent_shape_ok, student_latent.device
        ):
            raise RuntimeError(
                "fresh generator replay changed latent shape on at least one "
                f"rank; local={tuple(student_latent.shape)}, "
                f"expected={expected_latent_shape}")
        expected_student_latent = replay_descriptor["student_latent"].to(
            device=student_latent.device, dtype=student_latent.dtype)
        latent_replay_max_abs = (
            student_latent.detach() - expected_student_latent
        ).abs().max()
        local_latent_replay_matches = bool(torch.allclose(
            student_latent.detach(),
            expected_student_latent,
            atol=float(replay_atol),
            rtol=float(replay_rtol),
        ))
        if not self._all_ranks_true(
            local_latent_replay_matches, student_latent.device
        ):
            raise RuntimeError(
                "fresh generator replay changed the generated latent on at "
                "least one rank "
                f"(local max_abs={float(latent_replay_max_abs):.6g})")
        student_latent = self._apply_gradient_mask(
            student_latent, gradient_mask)

        if prepared_replay is not None:
            # The detached decoder VJP was evaluated at the exact first-pass
            # latent. Unlike the legacy value-side replay, an allclose-only
            # match is insufficient because J_D(z) changes with z.
            local_prepared_latent_exact = bool(torch.equal(
                student_latent.detach(), expected_student_latent))
            if not self._all_ranks_true(
                local_prepared_latent_exact, student_latent.device
            ):
                raise RuntimeError(
                    "prepared fresh-video replay requires bitwise-identical generator "
                    "latents on every rank "
                    f"(local max_abs={float(latent_replay_max_abs):.6g})")

            prepared_error = None
            latent_cotangent = None
            replay_log = None
            try:
                prepared_enabled = getattr(
                    self,
                    "fresh_video_prepared_replay_enabled",
                    getattr(self, "fidel_p0_replay_enabled", False),
                )
                if not prepared_enabled:
                    raise RuntimeError(
                        "prepared replay was supplied while P0 is disabled")
                if (
                    not isinstance(prepared_replay, dict)
                    or set(prepared_replay)
                    != {"latent_cotangent", "replay_log"}
                    or not isinstance(prepared_replay.get("replay_log"), dict)
                ):
                    raise ValueError(
                        "invalid prepared fresh-video replay payload")
                latent_cotangent = prepared_replay["latent_cotangent"]
                if (
                    not isinstance(latent_cotangent, torch.Tensor)
                    or latent_cotangent.requires_grad
                    or latent_cotangent.grad_fn is not None
                    or latent_cotangent.shape != student_latent.shape
                ):
                    raise ValueError(
                        "prepared fresh-video latent cotangent has invalid structure")
                latent_cotangent = latent_cotangent.to(
                    device=student_latent.device, dtype=torch.float32)
                if not bool(torch.isfinite(
                    latent_cotangent).all().item()
                ):
                    raise FloatingPointError(
                        "prepared fresh-video latent cotangent is non-finite")
                replay_log = dict(prepared_replay["replay_log"])
            except BaseException as exc:
                prepared_error = exc
            prepared_ok = self._all_ranks_true(
                prepared_error is None, student_latent.device)
            if not prepared_ok:
                all_errors_are_nonfinite = self._all_ranks_true(
                    prepared_error is None
                    or isinstance(prepared_error, FloatingPointError),
                    student_latent.device,
                )
                message = (
                    f": {type(prepared_error).__name__}: {prepared_error}"
                    if prepared_error is not None
                    else " on another rank"
                )
                if all_errors_are_nonfinite:
                    raise FloatingPointError(
                        "prepared fresh-video replay produced a non-finite value"
                        + message
                    ) from prepared_error
                raise RuntimeError(
                    "prepared fresh-video replay validation failed" + message
                ) from prepared_error
            replay_surrogate = (
                student_latent.float() * latent_cotangent).sum()
            loss = replay_surrogate - replay_surrogate.detach()
            dense_prefix = self._dense_metric_prefix()
            replay_log.update({
                self._dense_key("latent_replay_max_abs"): (
                    latent_replay_max_abs.detach()),
                "fresh_video_prepared_replay": (
                    latent_cotangent.new_ones(())),
            })
            if self.fidel_two_level_enabled:
                replay_log["fidel_prepared_replay"] = (
                    latent_cotangent.new_ones(()))
            return loss, replay_log

        token_indices = None
        temporal_window_indices = None
        if (
            not self._uses_dinov3_dense_backend()
            and self.uses_random_token_reference_bank
        ):
            token_indices = replay_descriptor[
                "vjepa2_token_indices"].to(
                    device=student_latent.device, dtype=torch.long)
            expected_indices = self._vjepa2_token_indices(
                student_latent.device)
            local_reference_matches = bool(
                int(replay_descriptor["vjepa2_reference_slot"])
                == int(self.vjepa2_active_reference_slot)
                and replay_descriptor["vjepa2_token_indices_sha256"]
                == self.vjepa2_active_token_indices_hash
                and replay_descriptor["vjepa2_reference_pack_hash"]
                == self.vjepa2_nystrom_reference_hash
                and torch.equal(token_indices, expected_indices)
            )
            if not self._all_ranks_true(
                local_reference_matches, student_latent.device
            ):
                raise RuntimeError(
                    "random-token reference changed between first pass and "
                    "replay on at least one rank")
        elif not self._uses_dinov3_dense_backend():
            token_indices = self._vjepa2_token_indices(
                student_latent.device)
        if not self._uses_dinov3_dense_backend():
            temporal_window_indices = (
                self._vjepa2_descriptor_window_indices(
                    replay_descriptor, device=student_latent.device)
            )

        # Recreate the value-side VAE/V-JEPA2 path and prove that the replayed
        # generator sample is exactly the sample scored in the first pass.
        with torch.no_grad():
            student_video01 = self._decode_video01(student_latent)
        expected_features = expected_features.to(
            device=student_latent.device, dtype=torch.float32)

        feature_cotangent = feature_cotangent.to(
            device=student_latent.device, dtype=torch.float32)
        local_cotangent_shape_ok = (
            feature_cotangent.shape == expected_features.shape)
        if not self._all_ranks_true(
            local_cotangent_shape_ok, student_latent.device
        ):
            raise ValueError(
                "feature cotangent shape does not match the cached video "
                "matrix on at least one rank: "
                f"{tuple(feature_cotangent.shape)} != "
                f"{tuple(expected_features.shape)}")
        local_cotangent_finite = bool(
            torch.isfinite(feature_cotangent).all().item())
        if not self._all_ranks_true(
            local_cotangent_finite, student_latent.device
        ):
            raise FloatingPointError(
                "non-finite fresh dense feature cotangent on at least one "
                "rank")

        # Frozen encoder VJP to decoded pixels. No rank enters a subsequent
        # collective until every locally fallible encoder/autograd operation
        # has either completed or been captured, preventing a one-rank replay
        # exception from stranding peers in an all-reduce.
        encoder_replay_error = None
        try:
            if self._uses_dinov3_dense_backend():
                self._dinov3_descriptor_frame_indices(
                    replay_descriptor, device=student_latent.device)
                video_cotangent, replay_max_abs = (
                    self._replay_dinov3_dense_to_pixels(
                        student_video01,
                        expected_features=expected_features,
                        feature_cotangent=feature_cotangent,
                        replay_atol=float(replay_atol),
                        replay_rtol=float(replay_rtol),
                    )
                )
            else:
                with torch.enable_grad():
                    student_video_leaf = (
                        student_video01.detach().float().requires_grad_(True))
                    replay_extract_kwargs = {
                        "gradient_checkpointing": (
                            self.vjepa2_dense_gradient_checkpointing),
                    }
                    if temporal_window_indices is not None:
                        replay_extract_kwargs["temporal_window_indices"] = (
                            temporal_window_indices)
                    replay_features = self._extract_vjepa2_dense(
                        student_video_leaf,
                        token_indices,
                        **replay_extract_kwargs,
                    ).float()
                    if replay_features.shape != expected_features.shape:
                        raise RuntimeError(
                            "fresh feature replay changed matrix shape: "
                            f"{tuple(replay_features.shape)} != "
                            f"{tuple(expected_features.shape)}")
                    replay_max_abs = (
                        replay_features.detach()
                        - expected_features).abs().max()
                    if not bool(torch.allclose(
                        replay_features.detach(),
                        expected_features,
                        atol=float(replay_atol),
                        rtol=float(replay_rtol),
                    )):
                        raise RuntimeError(
                            "fresh generator replay is not deterministic in "
                            "V-JEPA2 space "
                            f"(max_abs={float(replay_max_abs):.6g})")
                    encoder_surrogate = (
                        replay_features * feature_cotangent).sum()
                    (video_cotangent,) = torch.autograd.grad(
                        encoder_surrogate,
                        student_video_leaf,
                        retain_graph=False,
                        create_graph=False,
                    )
                del encoder_surrogate, student_video_leaf, replay_features
        except BaseException as exc:
            encoder_replay_error = exc
        if not self._all_ranks_true(
            encoder_replay_error is None, student_latent.device
        ):
            message = (
                f": {type(encoder_replay_error).__name__}: "
                f"{encoder_replay_error}"
                if encoder_replay_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "frozen dense encoder replay failed" + message
            ) from encoder_replay_error
        video_cotangent = video_cotangent.detach().float()
        local_video_finite = bool(
            torch.isfinite(video_cotangent).all().item())
        if not self._all_ranks_true(
            local_video_finite, video_cotangent.device
        ):
            raise FloatingPointError(
                "non-finite fresh dense-encoder pixel cotangent on at least one "
                "rank")
        mmd_video_grad_norm = video_cotangent.norm().detach()
        # ``encoder_surrogate`` is the autograd root and therefore keeps the
        # leaf (and its full decoded-video storage) alive even after
        # ``autograd.grad`` has consumed the graph. Drop the already-consumed
        # root together with the leaf so ``student_video01 = None`` below can
        # actually release that storage before the frame VAE VJP. No tensor
        # operation or derivative changes: all three values are dead here.
        # Frozen causal Wan VAE VJP to the generated latent, with the same
        # rank-consensus exception boundary as the encoder replay above.
        vae_replay_error = None
        try:
            mmd_latent_cotangent = replay_decoder_vjp(
                self.vae,
                student_latent.detach(),
                video_cotangent,
                independent_first_frame=self.independent_first_frame,
            ).detach().float()
            if mmd_latent_cotangent.shape != student_latent.shape:
                raise ValueError(
                    "Wan VAE decoder VJP returned an invalid latent shape: "
                    f"{tuple(mmd_latent_cotangent.shape)} != "
                    f"{tuple(student_latent.shape)}")
        except BaseException as exc:
            vae_replay_error = exc
        if not self._all_ranks_true(
            vae_replay_error is None, student_latent.device
        ):
            message = (
                f": {type(vae_replay_error).__name__}: {vae_replay_error}"
                if vae_replay_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "frozen Wan VAE replay failed" + message
            ) from vae_replay_error
        local_latent_finite = bool(
            torch.isfinite(mmd_latent_cotangent).all().item())
        if not self._all_ranks_true(
            local_latent_finite, mmd_latent_cotangent.device
        ):
            raise FloatingPointError(
                "non-finite fresh dense-feature latent cotangent on at least one "
                "rank")
        del video_cotangent

        fd_weight = 0.0
        fd_latent_cotangent = torch.zeros_like(mmd_latent_cotangent)
        fd_pixel_cotangent = torch.zeros(
            (), device=student_latent.device, dtype=torch.float32)
        if self.frame_fd_fusion_enabled:
            frame_error = None
            try:
                if frame_fd_feature_cotangents is None:
                    raise ValueError(
                        "hybrid replay requires frame FD feature cotangents")
                frame_extraction = replay_descriptor["frame_fd_extraction"]
                # FrameFDHead.replay_feature_cotangents_to_latents is exactly
                # these two calls in this order.  Keep them split here so the
                # full decoded 81-frame value tensor can die before the
                # memory-heavy one-latent-frame Wan VAE VJP.  This changes
                # tensor lifetime only; cotangents, operations, and operation
                # order are identical to the combined helper.
                fd_pixel_values = (
                    self.frame_fd_head.replay_feature_cotangents_to_pixels(
                        student_video01,
                        frame_extraction.frame_indices,
                        frame_fd_feature_cotangents,
                        expected_features=frame_extraction.features,
                        replay_atol=float(replay_atol),
                        replay_rtol=float(replay_rtol),
                    )
                )
                student_video01 = None
                if student_latent.is_cuda:
                    torch.cuda.empty_cache()
                fd_latent_cotangent = (
                    self.frame_fd_head.replay_pixel_cotangent_to_latents(
                        student_latent.detach(), fd_pixel_values)
                )
                fd_pixel_cotangent = fd_pixel_values.norm().detach()
                del fd_pixel_values
                if self.fidel_two_level_enabled:
                    # The update-wide FiDEL coefficients and FSDP world-size
                    # correction were already applied to the feature
                    # cotangents in ``stage_fidel_update``.  Replaying those
                    # four weighted branches therefore ends with a plain sum.
                    fd_weight = 1.0
                else:
                    fd_weight = float(self.frame_fd_balancer.stage(
                        mmd_latent_cotangent, fd_latent_cotangent))
            except BaseException as exc:
                frame_error = exc
            if not self._all_ranks_true(
                frame_error is None, student_latent.device
            ):
                message = (
                    f": {type(frame_error).__name__}: {frame_error}"
                    if frame_error is not None
                    else " on another rank"
                )
                raise RuntimeError(
                    "frame FD replay/balance failed" + message
                ) from frame_error
        elif frame_fd_feature_cotangents is not None:
            raise ValueError(
                "frame FD cotangents were provided while fusion is disabled")
        del student_video01

        mmd_norm = mmd_latent_cotangent.norm()
        fd_norm = fd_latent_cotangent.norm()
        cosine_denominator = mmd_norm * fd_norm
        if bool((cosine_denominator > 0).item()):
            branch_gradient_cosine = (
                (mmd_latent_cotangent * fd_latent_cotangent).sum()
                / cosine_denominator
            ).detach()
        else:
            branch_gradient_cosine = mmd_norm.new_zeros(())
        latent_cotangent = (
            mmd_latent_cotangent
            + float(fd_weight) * fd_latent_cotangent)
        local_combined_finite = bool(
            torch.isfinite(latent_cotangent).all().item())
        if not self._all_ranks_true(
            local_combined_finite, student_latent.device
        ):
            raise FloatingPointError(
                "non-finite combined MMD/frame-FD latent cotangent on at "
                "least one rank")

        # The global scalar value is logged from FreshVideoGradCache.  This
        # zero-valued surrogate carries only its exact replayed derivative.
        replay_surrogate = (
            student_latent.float() * latent_cotangent).sum()
        loss = replay_surrogate - replay_surrogate.detach()
        dense_prefix = self._dense_metric_prefix()
        return loss, {
            f"{dense_prefix}_feature_grad_norm": (
                feature_cotangent.norm().detach()),
            self._dense_key("video_grad_norm"): (
                mmd_video_grad_norm),
            self._dense_key("latent_grad_norm"): (
                mmd_latent_cotangent.norm().detach()),
            "frame_fd_feature_grad_norm": (
                torch.sqrt(sum(
                    value.detach().float().square().sum()
                    for value in (
                        frame_fd_feature_cotangents or {}).values()
                )) if frame_fd_feature_cotangents else
                latent_cotangent.new_zeros(())),
            "frame_fd_pixel_grad_norm": fd_pixel_cotangent,
            "frame_fd_latent_grad_norm": (
                fd_latent_cotangent.norm().detach()),
            "video_frame_latent_grad_cosine": branch_gradient_cosine,
            "frame_fd_dynamic_weight": latent_cotangent.new_tensor(
                float(fd_weight)),
            "combined_latent_grad_norm": (
                latent_cotangent.norm().detach()),
            self._dense_key("replay_max_abs"): replay_max_abs.detach(),
            self._dense_key("latent_replay_max_abs"): (
                latent_replay_max_abs.detach()),
        }

    def compute_latent_mmd_loss(
        self,
        student_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
        conditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if self.uses_fresh_video_gradcache:
            raise RuntimeError(
                "nystrom_exact_repulsion must be evaluated through the "
                "trainer's full fresh-video GradCache batch")
        student_latent = self._apply_gradient_mask(student_latent, gradient_mask)
        teacher_latent = teacher_latent.to(
            device=student_latent.device, dtype=student_latent.dtype)
        if teacher_latent.shape != student_latent.shape:
            raise ValueError(
                f"teacher latent {tuple(teacher_latent.shape)} != "
                f"student latent {tuple(student_latent.shape)}")

        if self.feature_type == "post_block_latent":
            return self._compute_wan_latent_mmd_loss(
                student_latent, teacher_latent, conditional_dict)
        return self._compute_vjepa2_dense_mmd_loss(
            student_latent, teacher_latent)

    def _compute_wan_latent_mmd_loss(
        self,
        student_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, dict]:
        indices = self._token_indices(student_latent)
        # Teacher first: its no-grad temporaries are released before the student
        # checkpoint graph is constructed.
        with torch.no_grad():
            teacher_features = self._extract(
                teacher_latent, conditional_dict, indices)
        student_features = self._extract(
            student_latent, conditional_dict, indices)
        mmd, mmd_meta = self.latent_mmd(
            student_features, teacher_features, stage=True)
        normalized_mmd = (
            mmd / (mmd.detach() + self.latent_norm_eps)
            if self.latent_self_normalize else mmd)
        loss = self.latent_loss_weight * normalized_mmd
        return loss, {
            "dmdtrain_gradient_norm": loss.new_zeros(()),
            "wan_latent_mmd": mmd.detach(),
            "wan_latent_loss_normalized": normalized_mmd.detach(),
            "wan_latent_bandwidth": mmd_meta["bandwidth"],
            "wan_latent_history_updates": mmd_meta["history_updates"],
            "wan_latent_current_coefficient": mmd_meta["current_coefficient"],
            "wan_latent_tokens": loss.new_tensor(student_features.shape[1]),
            "wan_latent_feature_dim": loss.new_tensor(
                student_features.shape[2]),
            "wan_latent_prefix_blocks": loss.new_tensor(
                self.latent_layer + 1),
        }

    def _compute_vjepa2_dense_mmd_loss(
        self,
        student_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        token_indices = self._vjepa2_token_indices(student_latent.device)

        # Stage A: decode and extract both sides without retaining either the
        # causal VAE or the V-JEPA2 graph.  Teacher is processed first and its
        # large decoded video is released before the student decode.
        with torch.no_grad():
            teacher_video01 = self._decode_video01(teacher_latent)
            teacher_features = self._extract_vjepa2_dense(
                teacher_video01,
                token_indices,
                gradient_checkpointing=False,
            ).detach().float()
            del teacher_video01

            student_video01 = self._decode_video01(student_latent)
            student_features_detached = self._extract_vjepa2_dense(
                student_video01,
                token_indices,
                gradient_checkpointing=False,
            ).detach().float()

        # Stage A.2: calculate the MMD and its cotangent only on a small dense
        # feature leaf.  EMARBFMMD detaches historical kernel means and teacher
        # features, so only this latest student leaf receives a gradient.
        with torch.enable_grad():
            student_features_leaf = (
                student_features_detached.detach().requires_grad_(True))
            mmd, mmd_meta = self.latent_mmd(
                student_features_leaf, teacher_features, stage=True)
            normalized_mmd = (
                mmd / (mmd.detach() + self.latent_norm_eps)
                if self.latent_self_normalize else mmd)
            objective = self.latent_loss_weight * normalized_mmd
            (feature_cotangent,) = torch.autograd.grad(
                objective,
                student_features_leaf,
                retain_graph=False,
                create_graph=False,
            )
        feature_cotangent = feature_cotangent.detach().float()
        if not torch.isfinite(feature_cotangent).all():
            raise FloatingPointError(
                "non-finite V-JEPA2 dense-feature MMD cotangent")

        # Stage B.1: replay only the frozen FP32 V-JEPA2 prefix.  The adapter
        # repeats the exact deterministic center-frame/resize/crop/normalize
        # pipeline used above; non-reentrant block checkpointing bounds memory.
        with torch.enable_grad():
            student_video_leaf = (
                student_video01.detach().float().requires_grad_(True))
            replay_features = self._extract_vjepa2_dense(
                student_video_leaf,
                token_indices,
                gradient_checkpointing=(
                    self.vjepa2_dense_gradient_checkpointing),
            ).float()
            encoder_surrogate = (
                replay_features * feature_cotangent).sum()
            (video_cotangent,) = torch.autograd.grad(
                encoder_surrogate,
                student_video_leaf,
                retain_graph=False,
                create_graph=False,
            )
        video_cotangent = video_cotangent.detach().float()
        if not torch.isfinite(video_cotangent).all():
            raise FloatingPointError(
                "non-finite V-JEPA2 replay pixel cotangent")
        del encoder_surrogate, student_video_leaf, replay_features, student_video01

        # Stage B.2: replay the frozen stateful Wan VAE decoder one latent frame
        # at a time.  Its temporal cache is carried by value and detached between
        # frames, matching the repository's validated bounded-memory replay.
        latent_cotangent = replay_decoder_vjp(
            self.vae,
            student_latent.detach(),
            video_cotangent,
            independent_first_frame=self.independent_first_frame,
        ).detach().float()
        if latent_cotangent.shape != student_latent.shape:
            raise ValueError(
                "Wan VAE decoder VJP returned an invalid latent shape: "
                f"got {tuple(latent_cotangent.shape)}, expected "
                f"{tuple(student_latent.shape)}")
        if not torch.isfinite(latent_cotangent).all():
            raise FloatingPointError(
                "non-finite V-JEPA2 replay latent cotangent")

        # Preserve the true objective as the scalar value reported by the
        # trainer while replacing its derivative with the replayed latent VJP.
        replay_surrogate = (
            student_latent.float() * latent_cotangent).sum()
        loss = replay_surrogate + (
            objective.detach() - replay_surrogate.detach())
        return loss, {
            "dmdtrain_gradient_norm": latent_cotangent.norm().detach(),
            self._dense_key("mmd"): mmd.detach(),
            self._dense_key("loss_normalized"): normalized_mmd.detach(),
            self._dense_key("bandwidth"): mmd_meta["bandwidth"],
            self._dense_key("history_updates"): mmd_meta["history_updates"],
            self._dense_key("current_coefficient"): (
                mmd_meta["current_coefficient"]),
            "vjepa2_dense_raw_tokens": loss.new_tensor(
                self.vjepa2_dense_raw_tokens),
            "vjepa2_dense_sampled_tokens": loss.new_tensor(
                student_features_detached.shape[1]),
            self._dense_key("feature_dim"): loss.new_tensor(
                student_features_detached.shape[2]),
            self._dense_key("local_video_samples"): mmd_meta["local_samples"],
            self._dense_key("global_video_samples"): mmd_meta["global_samples"],
            self._dense_key("num_frames"): loss.new_tensor(
                self.vjepa2_dense_num_frames),
            self._dense_key("layer"): loss.new_tensor(
                self.vjepa2_dense_layer),
            self._dense_key("prefix_blocks"): loss.new_tensor(
                self.vjepa2_dense_layer + 1),
            self._dense_key("feature_grad_norm"): (
                feature_cotangent.norm().detach()),
            self._dense_key("video_grad_norm"): (
                video_cotangent.norm().detach()),
            self._dense_key("latent_grad_norm"): (
                latent_cotangent.norm().detach()),
        }

    def current_frame_fd_weight(self) -> float:
        if not getattr(self, "frame_fd_fusion_enabled", False):
            return 0.0
        if getattr(self, "fidel_two_level_enabled", False):
            return 1.0
        return float(self.frame_fd_balancer.current_weight())

    def finalize_frame_fd_balance_stage(self):
        if not getattr(self, "frame_fd_fusion_enabled", False):
            return None
        if getattr(self, "fidel_two_level_enabled", False):
            if not self.fidel_controller.staged:
                raise RuntimeError("no two-level FiDEL update is staged")
            return self.fidel_controller.prepare_commit()
        return self.frame_fd_balancer.finalize_stage()

    def distribution_state_pending(self) -> bool:
        latent_mmd = getattr(self, "latent_mmd", None)
        mmd_pending = bool(
            getattr(self, "uses_fresh_video_gradcache", False)
            and (
                getattr(latent_mmd, "attraction_ema_pending", False)
                or getattr(latent_mmd, "repulsion_fifo_pending", False)
            )
        )
        if not getattr(self, "frame_fd_fusion_enabled", False):
            return mmd_pending
        if getattr(self, "fidel_two_level_enabled", False):
            return mmd_pending or bool(self.fidel_controller.staged)
        return mmd_pending or bool(self.frame_fd_balancer.staged)

    def prepare_distribution_state(self):
        """Validate staged controller state before optimizer mutation."""
        latent_mmd = getattr(self, "latent_mmd", None)
        if (
            getattr(self, "uses_fresh_video_gradcache", False)
            and (
                getattr(latent_mmd, "attraction_ema_enabled", False)
                or getattr(latent_mmd, "repulsion_fifo_enabled", False)
            )
        ):
            latent_mmd.prepare_commit()
        if not getattr(self, "frame_fd_fusion_enabled", False):
            return None
        if getattr(self, "fidel_two_level_enabled", False):
            return self.fidel_controller.prepare_commit()
        return self.frame_fd_balancer.finalize_stage()

    def commit_distribution_state(self) -> None:
        if not self.uses_fresh_video_gradcache:
            self.latent_mmd.commit()
        else:
            latent_mmd = getattr(self, "latent_mmd", None)
            if (
                getattr(latent_mmd, "attraction_ema_enabled", False)
                or getattr(latent_mmd, "repulsion_fifo_enabled", False)
            ):
                latent_mmd.commit()
        if (
            self.uses_fresh_video_gradcache
            and getattr(self, "frame_fd_fusion_enabled", False)
        ):
            if getattr(self, "fidel_two_level_enabled", False):
                self.fidel_controller.commit()
            else:
                self.frame_fd_balancer.commit()

    def discard_distribution_state(self) -> None:
        if not self.uses_fresh_video_gradcache:
            self.latent_mmd.discard()
        else:
            latent_mmd = getattr(self, "latent_mmd", None)
            if (
                getattr(latent_mmd, "attraction_ema_enabled", False)
                or getattr(latent_mmd, "repulsion_fifo_enabled", False)
            ):
                latent_mmd.discard()
        if (
            self.uses_fresh_video_gradcache
            and getattr(self, "frame_fd_fusion_enabled", False)
        ):
            if getattr(self, "fidel_two_level_enabled", False):
                self.fidel_controller.discard()
            else:
                self.frame_fd_balancer.discard()

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ):
        if self.uses_fresh_video_gradcache:
            raise RuntimeError(
                "nystrom_exact_repulsion generator_loss requires "
                "FreshVideoGradCache orchestration in ElasticForcingTrainer")
        if clean_latent is None:
            raise ValueError(
                "wan_kv_mmd needs aligned teacher latents; configure "
                "teacher_latent_data_path")
        if self.causal_generator:
            pred, gradient_mask, _, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
            )
        else:
            pred = self._run_few_step_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )
            gradient_mask = None
        return self.compute_latent_mmd_loss(
            pred, clean_latent, conditional_dict, gradient_mask=gradient_mask)

    def _fd_metadata(self) -> dict:
        if self.feature_type == "dinov3_dense_latent":
            return self._dinov3_feature_metadata()
        if self.feature_type in self._DENSE_VIDEO_FEATURE_TYPES:
            if self.dual_video_mmd_enabled:
                branches = {}
                for name, registry in self.dual_branch_registry.items():
                    branches[name] = {
                        "reference_path": registry["reference_path"],
                        "reference_hash": registry["reference_hash"],
                        "sample_shape": list(registry["sample_shape"]),
                    }
                if "vjepa2" in branches:
                    branches["vjepa2"]["feature"] = (
                        self._vjepa2_feature_metadata())
                if "dinov3" in branches:
                    branches["dinov3"]["feature"] = (
                        self._dinov3_feature_metadata())
                if "videomae_v2" in branches:
                    from video_fd.encoders.videomae_v2_dense_metadata import (
                        canonical_videomae_v2_dense_feature_metadata,
                        uniform_segment_token_indices,
                    )
                    video_indices = uniform_segment_token_indices(
                        self.videomae_v2_encoder.dense_raw_tokens, 256)
                    branches["videomae_v2"]["feature"] = (
                        canonical_videomae_v2_dense_feature_metadata(
                            model_id=self.videomae_v2_reference_model_id,
                            model_id_is_reference=True,
                            layer_index=15,
                            num_tokens=256,
                            token_indices=video_indices,
                            num_segments=5,
                            frames_per_segment=16,
                            raw_tokens_per_segment=(
                                self.videomae_v2_encoder.dense_raw_tokens),
                            feature_dim=(
                                self.videomae_v2_encoder.dense_feature_dim),
                            resize_short_side=224,
                            crop_size=224,
                            skip_first_frame=True,
                        ))
                branch_weights = {
                    name: float(self.latent_mmd.branch_weights[name])
                    for name in sorted(branches)
                }
                teacher_bank_content_hash = (
                    self.triple_teacher_bank_content_hash
                    if self.feature_type
                    == "triple_dinov3_vjepa2_videomae_v2_dense_latent"
                    else self.dual_teacher_bank_content_hash
                )
                aggregation_divisor = float(
                    self.latent_mmd.aggregation_divisor)
                combination = {
                    "operation": "sum",
                    "branch_weights": branch_weights,
                    "divide_by_branch_count": (
                        aggregation_divisor == float(len(branches))),
                    "normalization": (
                        "raw_div_stop_gradient_raw_plus_epsilon"),
                    "normalization_epsilon": self.latent_norm_eps,
                }
                if aggregation_divisor != 1.0:
                    combination["aggregation_divisor"] = aggregation_divisor
                current_enabled = bool(
                    self.current_teacher_attraction_enabled)
                forward_samples = int(self.fresh_video_total_samples)
                gradient_samples = int(
                    self.expected_current_teacher_samples
                    if current_enabled
                    else getattr(
                        self,
                        "fresh_video_gradient_global_samples",
                        forward_samples,
                    )
                )
                teacher_samples = (
                    gradient_samples if current_enabled else 0)
                result = {
                    "feature_type": self.feature_type,
                    "branches": branches,
                    "teacher_bank_content_hash": teacher_bank_content_hash,
                    "teacher_mixture": {
                        "fixed_reference_weight": (
                            self.nystrom_attraction_weight),
                        "current_teacher_weight": (
                            self.current_teacher_attraction_weight
                            if current_enabled else 0.0),
                        "normalization": (
                            "divide_by_weight_sum"
                            if current_enabled else "fixed_reference_only"),
                    },
                    "combination": combination,
                    "partial_gradient": {
                        "forward_global_samples": forward_samples,
                        "gradient_global_samples": gradient_samples,
                        "expected_teacher_global_samples": teacher_samples,
                        "full_student_population": True,
                        "replay_cotangent_compensation": (
                            float(forward_samples) / float(gradient_samples)),
                        "denominators": {
                            "kss": forward_samples * forward_samples,
                            "fixed_attraction": forward_samples,
                            "teacher_cross": (
                                forward_samples * teacher_samples),
                            "teacher_self": teacher_samples * teacher_samples,
                        },
                    },
                }
                if hasattr(
                    self, "dinov3_vjepa2_reference_bundle_manifest_sha256"
                ):
                    result["reference_bundle"] = {
                        "path": (
                            self.dinov3_vjepa2_reference_bundle_manifest),
                        "sha256": (
                            self.dinov3_vjepa2_reference_bundle_manifest_sha256),
                        "video_name_sequence_sha256": (
                            self.openvid_video_name_sequence_sha256),
                        "legacy_vjepa_provenance_bridge": True,
                    }
                return result
            if getattr(self, "vjepa2_multilayer_enabled", False):
                return {
                    "feature_type": "vjepa2_dense_multilayer_latent",
                    "features": [
                        self._vjepa2_feature_metadata(
                            layer_index=layer)
                        for layer in self.vjepa2_dense_layers
                    ],
                    "combination": {
                        "operation": "sum",
                        "per_layer_objective": (
                            "nystrom_exact_repulsion"),
                        "normalization": (
                            "mmd_over_abs_detached_mmd_plus_epsilon"
                            if self.latent_self_normalize
                            else "none"
                        ),
                        "normalization_epsilon": self.latent_norm_eps,
                        "layer_weights": [
                            1.0 for _ in self.vjepa2_dense_layers
                        ],
                    },
                }
            if getattr(self, "uses_random_token_reference_bank", False):
                metadata = self._vjepa2_static_feature_metadata()
                metadata.update({
                    "feature_metadata_version": 2,
                    "token_sampling": (
                        "random_without_replacement_per_update"),
                    "token_alignment": (
                        "same_explicit_indices_and_order_per_update"),
                    "token_mask_count": int(
                        self.vjepa2_reference_bank.slot_count),
                })
                return metadata
            metadata = self._vjepa2_feature_metadata()
            reference_sampling = getattr(
                self, "vjepa2_dense_frame_sampling", "center_contiguous")
            student_sampling = getattr(
                self,
                "vjepa2_student_frame_sampling",
                reference_sampling,
            )
            if student_sampling != reference_sampling:
                student_start = (
                    int(self.vjepa2_student_source_num_frames)
                    - int(self.vjepa2_dense_num_frames)
                )
                metadata = dict(metadata)
                metadata["student_temporal_sampling"] = {
                    "frame_sampling": student_sampling,
                    "source_num_frames": int(
                        self.vjepa2_student_source_num_frames),
                    "selected_frame_indices": [
                        student_start,
                        student_start + int(self.vjepa2_dense_num_frames) - 1,
                    ],
                    "reference_frame_sampling": (
                        reference_sampling),
                }
            if getattr(self, "nystrom_attraction_weight", 1.0) != 1.0:
                metadata = dict(metadata)
                metadata["nystrom_attraction_weight"] = (
                    self.nystrom_attraction_weight)
            if getattr(
                self, "current_teacher_attraction_enabled", False
            ):
                metadata = dict(metadata)
                metadata["nystrom_current_teacher_attraction"] = {
                    "enabled": True,
                    "estimator": "exact_biased_gaussian_mmd",
                    "fixed_weight": self.nystrom_attraction_weight,
                    "current_weight": (
                        self.current_teacher_attraction_weight),
                    "normalization": "divide_by_group_weight_sum",
                    "expected_global_samples": (
                        self.expected_current_teacher_samples),
                    "ema_gradient_mode": (
                        "straight_through_full_current_mix"
                        if self.nystrom_attraction_ema_enabled
                        else "direct_full_current_mix"),
                }
                if self.current_teacher_source == "matched_manifest_video":
                    metadata[
                        "nystrom_current_teacher_attraction"
                    ]["source"] = "matched_manifest_video"
            # Preserve the tuple used by legacy v2 checkpoints.  The new
            # canonical reference metadata stores JSON-like lists instead.
            if getattr(self, "mmd_estimator", "ema_rff") == "ema_rff":
                metadata = dict(metadata)
                metadata.pop("feature_metadata_version", None)
                metadata.pop("vae_id", None)
                metadata.pop("token_indices_sha256", None)
                metadata.pop("kernel", None)
                metadata["matrix_shape"] = tuple(metadata["matrix_shape"])
            return metadata
        return {
            "feature_type": self.feature_type,
            "layer_index": self.latent_layer,
            "prefix_blocks": self.latent_layer + 1,
            "num_tokens": self.latent_num_tokens,
            "feature_dim": self.latent_feature_dim,
            "timestep": self.latent_timestep,
            "patch_size": self.latent_patch_size,
        }

    def _reference_bank_checkpoint_identity(self) -> dict:
        if not self.uses_random_token_reference_bank:
            raise RuntimeError(
                "reference-bank checkpoint identity requires random tokens")
        metadata = self.vjepa2_reference_bank.metadata
        return {
            "manifest_path": os.path.realpath(
                str(self.vjepa2_reference_bank.manifest_path)),
            "manifest_hash": self.vjepa2_reference_bank.manifest_hash,
            "slot_count": int(self.vjepa2_reference_bank.slot_count),
            "teacher_count": int(metadata["teacher_count"]),
            "train_indices_sha256": metadata["train_indices_sha256"],
            "val_indices_sha256": metadata["val_indices_sha256"],
        }

    def _nystrom_attraction_ema_active(self) -> bool:
        return bool(
            getattr(self, "uses_fresh_video_gradcache", False)
            and getattr(
                getattr(self, "latent_mmd", None),
                "attraction_ema_enabled",
                False,
            )
        )

    def _nystrom_repulsion_fifo_active(self) -> bool:
        return bool(
            getattr(self, "uses_fresh_video_gradcache", False)
            and getattr(
                getattr(self, "latent_mmd", None),
                "repulsion_fifo_enabled",
                False,
            )
        )

    def _nystrom_fd_state_version(self) -> int:
        if getattr(self, "is_triple_dense", False):
            base_version = 10
        elif self.dual_video_mmd_enabled:
            base_version = 9
        elif getattr(self, "fidel_two_level_enabled", False):
            base_version = 3
        elif getattr(self, "vjepa2_multilayer_enabled", False):
            base_version = 4
        elif getattr(self, "frame_fd_fusion_enabled", False):
            base_version = 2
        else:
            base_version = 1
        if self._nystrom_repulsion_fifo_active():
            if base_version != 1 or self._nystrom_attraction_ema_active():
                raise ValueError(
                    "repulsion FIFO requires the single-layer fixed-reference "
                    "Nyström state without attraction EMA")
            return 8
        if not self._nystrom_attraction_ema_active():
            return base_version
        enabled_versions = {1: 5, 2: 6, 4: 7}
        if base_version not in enabled_versions:
            raise ValueError(
                "attraction EMA is incompatible with this Nyström state "
                f"variant v{base_version}")
        return enabled_versions[base_version]

    def _with_nystrom_attraction_ema_state(self, state: dict) -> dict:
        if self._nystrom_attraction_ema_active():
            state["attraction_ema"] = (
                self.latent_mmd.attraction_ema_state_dict())
        return state

    def _with_nystrom_distribution_state(self, state: dict) -> dict:
        state = self._with_nystrom_attraction_ema_state(state)
        if self._nystrom_repulsion_fifo_active():
            state["repulsion_fifo"] = (
                self.latent_mmd.repulsion_fifo_state_dict())
        return state

    def fd_state_dict(self) -> dict:
        nystrom_version_key = self._dense_nystrom_version_key()
        if getattr(self, "uses_fresh_video_gradcache", False):
            if self.distribution_state_pending():
                raise RuntimeError(
                    "refusing to checkpoint pending distribution state")
            if self._uses_dinov3_dense_backend():
                return {
                    "dinov3_dense_nystrom_mmd_version": 1,
                    "metadata": self._fd_metadata(),
                    "reference_path": self.dinov3_nystrom_reference_path,
                    "reference_hash": self.dinov3_nystrom_reference_hash,
                    "teacher_feature_bank_hash": (
                        self.dinov3_teacher_feature_bank_hash),
                    "estimator": self.mmd_estimator,
                }
            if getattr(self, "fidel_two_level_enabled", False):
                return {
                    "vjepa2_dense_nystrom_mmd_version": 3,
                    "metadata": self._fd_metadata(),
                    "reference_bank": (
                        self._reference_bank_checkpoint_identity()),
                    "teacher_feature_bank_hash": (
                        self.vjepa2_teacher_feature_bank_hash),
                    "estimator": self.mmd_estimator,
                    "frame_fd_reference": self.frame_fd_reference_metadata,
                    "frame_fd": self.frame_fd_head.frame_fd_state_dict(),
                    "fusion": {
                        "version": 2,
                        "metadata": {
                            "algorithm": (
                                "two_level_fidel_previous_commit"),
                            "controller": (
                                self.fidel_controller.get_extra_state()),
                        },
                        "state": self.fidel_controller.state_dict(),
                    },
                }
            if getattr(self, "vjepa2_multilayer_enabled", False):
                return self._with_nystrom_distribution_state({
                    "vjepa2_dense_nystrom_mmd_version": (
                        self._nystrom_fd_state_version()),
                    "metadata": self._fd_metadata(),
                    "reference_paths": list(
                        self.vjepa2_nystrom_reference_paths),
                    "reference_hashes": list(
                        self.vjepa2_nystrom_reference_hashes),
                    "teacher_feature_bank_hashes": list(
                        self.vjepa2_teacher_feature_bank_hashes),
                    "estimator": self.mmd_estimator,
                })
            if getattr(self, "frame_fd_fusion_enabled", False):
                return self._with_nystrom_distribution_state({
                    "vjepa2_dense_nystrom_mmd_version": (
                        self._nystrom_fd_state_version()),
                    "metadata": self._fd_metadata(),
                    "reference_path": self.vjepa2_nystrom_reference_path,
                    "reference_hash": self.vjepa2_nystrom_reference_hash,
                    "teacher_feature_bank_hash": (
                        self.vjepa2_teacher_feature_bank_hash),
                    "estimator": self.mmd_estimator,
                    "frame_fd_reference": self.frame_fd_reference_metadata,
                    "frame_fd": self.frame_fd_head.frame_fd_state_dict(),
                    "fusion": {
                        "version": 1,
                        "metadata": {
                            "algorithm": "grad_norm_ema_previous_commit",
                            "fd_weight": self.frame_fd_balancer.fd_weight,
                            "beta": self.frame_fd_balancer.beta,
                            "minimum": self.frame_fd_balancer.minimum,
                            "maximum": self.frame_fd_balancer.maximum,
                            "epsilon": self.frame_fd_balancer.epsilon,
                        },
                        "state": self.frame_fd_balancer.state_dict(),
                    },
                })
            if self.dual_video_mmd_enabled:
                return {
                    self.dense_nystrom_version_key: (
                        self._nystrom_fd_state_version()),
                    "metadata": self._fd_metadata(),
                    "references": {
                        name: {
                            "path": branch["reference_path"],
                            "hash": branch["reference_hash"],
                            "landmarks": branch["nystrom_landmarks"],
                        }
                        for name, branch in self.dual_branch_registry.items()
                    },
                    "teacher_bank_content_hash": (
                        self.triple_teacher_bank_content_hash
                        if self.is_triple_dense
                        else self.dual_teacher_bank_content_hash),
                    "estimator": self.mmd_estimator,
                }
            return self._with_nystrom_distribution_state({
                "vjepa2_dense_nystrom_mmd_version": (
                    self._nystrom_fd_state_version()),
                "metadata": self._fd_metadata(),
                "reference_path": self.vjepa2_nystrom_reference_path,
                "reference_hash": self.vjepa2_nystrom_reference_hash,
                "teacher_feature_bank_hash": (
                    self.vjepa2_teacher_feature_bank_hash),
                "estimator": self.mmd_estimator,
            })
        if self.feature_type == "vjepa2_dense_latent":
            return {
                "vjepa2_dense_mmd_version": 2,
                "metadata": self._fd_metadata(),
                "vjepa2_dense_mmd": self.latent_mmd.state_dict(),
            }
        return {
            "wan_kv_mmd_version": 3,
            "metadata": self._fd_metadata(),
            "wan_latent_mmd": self.latent_mmd.state_dict(),
        }

    def load_fd_state_dict(self, state_dict: dict) -> None:
        if self._uses_dinov3_dense_backend():
            if not isinstance(state_dict, dict):
                raise TypeError("DINOv3 distribution state must be a mapping")
            if any(key in state_dict for key in (
                "wan_kv_mmd_version",
                "vjepa2_dense_mmd_version",
                "vjepa2_dense_nystrom_mmd_version",
            )):
                raise ValueError(
                    "Wan/V-JEPA2 distribution state cannot be loaded into "
                    "the DINOv3 dense-latent feature space")
            version = state_dict.get(
                "dinov3_dense_nystrom_mmd_version")
            if version != 1:
                raise ValueError(
                    "resume checkpoint has no compatible DINOv3 dense "
                    f"Nyström state v1 (got {version!r})")
            expected = self.fd_state_dict()
            if set(state_dict) != set(expected):
                raise ValueError(
                    "DINOv3 Nyström checkpoint/reference keys mismatch: "
                    f"saved={sorted(state_dict)}, "
                    f"current={sorted(expected)}")
            if state_dict != expected:
                raise ValueError(
                    "DINOv3 Nyström checkpoint/reference identity mismatch: "
                    f"saved={state_dict}, current={expected}")
            return

        dense_version_key = self._dense_ema_version_key()
        dense_state_key = self._dense_ema_state_key()
        nystrom_version_key = self._dense_nystrom_version_key()
        foreign_dense_version_keys = {
            "vjepa2_dense_mmd_version",
            "motionformer_dense_mmd_version",
            "vjepa2_dense_nystrom_mmd_version",
            "motionformer_dense_nystrom_mmd_version",
            "dual_dense_mmd_version",
            "dual_dense_nystrom_mmd_version",
            "triple_dense_nystrom_mmd_version",
        } - {dense_version_key, nystrom_version_key}
        if getattr(self, "uses_fresh_video_gradcache", False):
            present_foreign = sorted(
                foreign_dense_version_keys.intersection(state_dict))
            if present_foreign:
                raise ValueError(
                    "checkpoint belongs to another dense feature namespace: "
                    f"{present_foreign}")
            if "dinov3_dense_nystrom_mmd_version" in state_dict:
                raise ValueError(
                    "DINOv3 distribution state cannot be loaded into the "
                    "V-JEPA2 dense-latent feature space")
            if (
                "wan_kv_mmd_version" in state_dict
                or dense_version_key in state_dict
            ):
                raise ValueError(
                    "EMA/RFF distribution history cannot be loaded into the "
                    "fresh Nyström exact-repulsion estimator")
            version = state_dict.get(nystrom_version_key)
            expected_version = self._nystrom_fd_state_version()
            if version != expected_version:
                raise ValueError(
                    "resume checkpoint has no compatible dense feature "
                    f"Nyström state v{expected_version} for "
                    f"{nystrom_version_key} (got {version!r})")
            validated_attraction_ema = None
            if self._nystrom_attraction_ema_active():
                if "attraction_ema" not in state_dict:
                    raise ValueError(
                        "resume checkpoint is missing attraction EMA state")
                validated_attraction_ema = (
                    self.latent_mmd.validate_attraction_ema_state_dict(
                        state_dict["attraction_ema"])
                )
            validated_repulsion_fifo = None
            if self._nystrom_repulsion_fifo_active():
                if "repulsion_fifo" not in state_dict:
                    raise ValueError(
                        "resume checkpoint is missing repulsion FIFO state")
                validated_repulsion_fifo = (
                    self.latent_mmd.validate_repulsion_fifo_state_dict(
                        state_dict["repulsion_fifo"])
                )
            if expected_version == 3:
                expected = self.fd_state_dict()
                expected_top_level_keys = set(expected)
                if set(state_dict) != expected_top_level_keys:
                    missing = sorted(expected_top_level_keys - set(state_dict))
                    unexpected = sorted(
                        set(state_dict) - expected_top_level_keys)
                    raise ValueError(
                        "V-JEPA2/two-level-FiDEL checkpoint keys mismatch: "
                        f"missing={missing}, unexpected={unexpected}")
                identity_keys = {
                    "vjepa2_dense_nystrom_mmd_version",
                    "metadata",
                    "reference_bank",
                    "teacher_feature_bank_hash",
                    "estimator",
                    "frame_fd_reference",
                }
                saved_identity = {
                    key: state_dict.get(key) for key in identity_keys}
                expected_identity = {
                    key: expected.get(key) for key in identity_keys}
                if saved_identity != expected_identity:
                    raise ValueError(
                        "V-JEPA2/two-level-FiDEL checkpoint/reference-bank "
                        "mismatch")
                saved_frame = state_dict.get("frame_fd")
                saved_fusion = state_dict.get("fusion")
                if not isinstance(saved_frame, dict):
                    raise ValueError(
                        "FiDEL checkpoint is missing frame FD FIFO state")
                if (
                    not isinstance(saved_fusion, dict)
                    or set(saved_fusion) != {"version", "metadata", "state"}
                    or saved_fusion.get("version") != 2
                    or saved_fusion.get("metadata")
                    != expected["fusion"]["metadata"]
                    or not isinstance(saved_fusion.get("state"), dict)
                ):
                    raise ValueError(
                        "FiDEL checkpoint has incompatible controller state")

                # Validate the complete restore transaction before mutating the
                # first live FIFO, controller buffer, or active reference slot.
                validated_frame = (
                    self.frame_fd_head.validate_frame_fd_state_dict(saved_frame))
                saved_controller_state = saved_fusion["state"]
                live_controller_state = self.fidel_controller.state_dict()
                if set(saved_controller_state) != set(live_controller_state):
                    raise ValueError(
                        "FiDEL controller state keys do not match the active "
                        "controller")
                for key, live_value in live_controller_state.items():
                    saved_value = saved_controller_state[key]
                    if isinstance(live_value, torch.Tensor):
                        if (
                            not isinstance(saved_value, torch.Tensor)
                            or saved_value.shape != live_value.shape
                            or saved_value.dtype != live_value.dtype
                            or saved_value.layout != live_value.layout
                            or saved_value.is_meta
                        ):
                            raise ValueError(
                                "FiDEL controller tensor is incompatible for "
                                f"{key!r}")
                    elif saved_value != live_value:
                        raise ValueError(
                            "FiDEL controller extra state is incompatible for "
                            f"{key!r}")
                temporary_controller = copy.deepcopy(self.fidel_controller)
                temporary_controller.load_state_dict(
                    saved_controller_state, strict=True)
                if (
                    not bool(torch.isfinite(
                        temporary_controller.loss_scale_ema).all().item())
                    or not bool(torch.isfinite(
                        temporary_controller.theta_ema).all().item())
                    or bool(
                        (temporary_controller.loss_scale_ema < 0.0).any().item())
                    or bool((temporary_controller.theta_ema < 0.0).any().item())
                ):
                    raise ValueError(
                        "FiDEL controller checkpoint contains invalid "
                        "persistent statistics")
                committed = int(
                    temporary_controller.committed_updates.item())
                slot_count = int(self.vjepa2_reference_bank.slot_count)
                if committed < 0 or committed > slot_count:
                    raise ValueError(
                        "FiDEL committed update count is outside the "
                        "reference-bank range")
                # A checkpoint is taken after commit.  The next successful
                # update consumes the equally numbered slot.  At the terminal
                # count there is no next slot, so retain the final valid slot.
                resume_slot = min(committed, slot_count - 1)

                # Reading the complete target pack here ensures every hash,
                # metadata and shape failure occurs before publication. The
                # subsequent activation performs only the same validated load
                # plus allocation-free fixed-buffer copies.
                prepared_pack = self.vjepa2_reference_bank.load_slot_pack(
                    resume_slot)
                for name, source in {
                    "landmarks": prepared_pack["landmarks"],
                    "alpha": prepared_pack["alpha"],
                    "k_rr": prepared_pack["k_rr"].reshape(()),
                    "sigma": prepared_pack["sigma"].reshape(()),
                }.items():
                    if getattr(self.latent_mmd, name).shape != source.shape:
                        raise ValueError(
                            "FiDEL resume reference pack cannot replace "
                            f"{name}: {tuple(source.shape)} != "
                            f"{tuple(getattr(self.latent_mmd, name).shape)}")
                prepared_reference_count = int(
                    prepared_pack["metadata"]["nystrom"][
                        "reference_count"])
                if prepared_reference_count != len(
                    self.vjepa2_reference_bank.train_indices
                ):
                    raise ValueError(
                        "FiDEL resume reference pack count does not match "
                        "the frozen teacher-train split")

                # No validation remains after this point. Activate first because
                # activation itself loads/checks the external pack before its
                # first live buffer copy; FIFO/controller publication then uses
                # only states proven compatible above.
                self.activate_vjepa2_reference_slot(resume_slot)
                self.frame_fd_head.load_validated_frame_fd_state_dict(
                    validated_frame)
                self.fidel_controller.load_state_dict(
                    temporary_controller.state_dict(), strict=True)
                return
            if expected_version in {2, 6}:
                expected = self.fd_state_dict()
                if set(state_dict) != set(expected):
                    raise ValueError(
                        "V-JEPA2/frame-FD checkpoint keys mismatch: "
                        f"saved={sorted(state_dict)}, "
                        f"current={sorted(expected)}")
                identity_keys = {
                    "vjepa2_dense_nystrom_mmd_version",
                    "metadata",
                    "reference_path",
                    "reference_hash",
                    "teacher_feature_bank_hash",
                    "estimator",
                    "frame_fd_reference",
                }
                saved_identity = {
                    key: state_dict.get(key) for key in identity_keys}
                expected_identity = {
                    key: expected.get(key) for key in identity_keys}
                if saved_identity != expected_identity:
                    raise ValueError(
                        "V-JEPA2/frame-FD hybrid checkpoint/reference "
                        "mismatch")
                saved_frame = state_dict.get("frame_fd")
                saved_fusion = state_dict.get("fusion")
                if not isinstance(saved_frame, dict):
                    raise ValueError(
                        "hybrid checkpoint is missing frame FD FIFO state")
                if (
                    not isinstance(saved_fusion, dict)
                    or set(saved_fusion) != {"version", "metadata", "state"}
                    or saved_fusion.get("version") != 1
                    or saved_fusion.get("metadata")
                    != expected["fusion"]["metadata"]
                    or not isinstance(saved_fusion.get("state"), dict)
                ):
                    raise ValueError(
                        "hybrid checkpoint has incompatible dynamic fusion "
                        "state")

                # Prove the complete hybrid restore before publishing its
                # first FIFO, balancer, or attraction EMA scalar.
                validated_frame = (
                    self.frame_fd_head.validate_frame_fd_state_dict(
                        saved_frame))
                saved_balancer_state = saved_fusion["state"]
                live_balancer_state = self.frame_fd_balancer.state_dict()
                if set(saved_balancer_state) != set(live_balancer_state):
                    raise ValueError(
                        "hybrid grad-norm balancer state keys do not match "
                        "the active controller")
                for key, live_value in live_balancer_state.items():
                    saved_value = saved_balancer_state[key]
                    if isinstance(live_value, torch.Tensor):
                        if (
                            not isinstance(saved_value, torch.Tensor)
                            or saved_value.shape != live_value.shape
                            or saved_value.dtype != live_value.dtype
                            or saved_value.layout != live_value.layout
                            or saved_value.is_meta
                        ):
                            raise ValueError(
                                "hybrid grad-norm balancer tensor is "
                                f"incompatible for {key!r}")
                    elif saved_value != live_value:
                        raise ValueError(
                            "hybrid grad-norm balancer extra state is "
                            f"incompatible for {key!r}")
                temporary_balancer = copy.deepcopy(
                    self.frame_fd_balancer)
                temporary_balancer.load_state_dict(
                    saved_balancer_state, strict=True)
                initialized = bool(
                    temporary_balancer.initialized.item())
                committed_updates = int(
                    temporary_balancer.committed_updates.item())
                if (
                    not bool(torch.isfinite(
                        temporary_balancer.mmd_rms_ema).item())
                    or not bool(torch.isfinite(
                        temporary_balancer.fd_rms_ema).item())
                    or float(temporary_balancer.mmd_rms_ema) < 0.0
                    or float(temporary_balancer.fd_rms_ema) < 0.0
                    or committed_updates < 0
                    or initialized != (committed_updates > 0)
                ):
                    raise ValueError(
                        "hybrid grad-norm balancer checkpoint contains "
                        "invalid persistent statistics")

                self.frame_fd_head.load_validated_frame_fd_state_dict(
                    validated_frame)
                self.frame_fd_balancer.load_state_dict(
                    temporary_balancer.state_dict(), strict=True)
                if validated_attraction_ema is not None:
                    self.latent_mmd.load_attraction_ema_state_dict(
                        validated_attraction_ema)
                return
            expected = self.fd_state_dict()
            if set(state_dict) != set(expected):
                raise ValueError(
                    "V-JEPA2 Nyström checkpoint/reference keys mismatch: "
                    f"saved={sorted(state_dict)}, "
                    f"current={sorted(expected)}")
            identity_keys = set(expected)
            identity_keys.discard("attraction_ema")
            identity_keys.discard("repulsion_fifo")
            saved_identity = {
                key: state_dict.get(key) for key in identity_keys}
            expected_identity = {
                key: expected.get(key) for key in identity_keys}
            # Runtime live-teacher batch size is not frozen-reference identity.
            saved_identity = copy.deepcopy(saved_identity)
            expected_identity = copy.deepcopy(expected_identity)
            for identity in (saved_identity, expected_identity):
                current_teacher = identity.get("metadata", {}).get(
                    "nystrom_current_teacher_attraction", {})
                current_teacher.pop("expected_global_samples", None)
            saved_identity = _canonicalize_shared_asset_paths(saved_identity)
            expected_identity = _canonicalize_shared_asset_paths(
                expected_identity)
            if saved_identity != expected_identity:
                raise ValueError(
                    "V-JEPA2 Nyström checkpoint/reference mismatch: "
                    f"saved={saved_identity}, current={expected_identity}")
            if validated_attraction_ema is not None:
                self.latent_mmd.load_attraction_ema_state_dict(
                    validated_attraction_ema)
            if validated_repulsion_fifo is not None:
                self.latent_mmd.load_repulsion_fifo_state_dict(
                    validated_repulsion_fifo)
            return

        if self.feature_type == "vjepa2_dense_latent":
            if "dinov3_dense_nystrom_mmd_version" in state_dict:
                raise ValueError(
                    "DINOv3 distribution state cannot be loaded into the "
                    "V-JEPA2 dense-latent feature space")
            if "wan_kv_mmd_version" in state_dict:
                raise ValueError(
                    "Wan post-block latent MMD history cannot be loaded into "
                    "the V-JEPA2 dense-latent feature space")
            version = state_dict.get("vjepa2_dense_mmd_version")
            if version != 2:
                detail = (
                    " V-JEPA2 dense MMD v1 treated individual tokens as "
                    "distribution samples and is intentionally incompatible."
                    if version == 1 else ""
                )
                raise ValueError(
                    "resume checkpoint has no compatible V-JEPA2 dense "
                    f"latent MMD v2 state (got version {version!r})."
                    f"{detail}")
            if "vjepa2_dense_mmd" not in state_dict:
                raise ValueError(
                    "resume checkpoint is missing fd.vjepa2_dense_mmd history")
            expected = self._fd_metadata()
            saved = state_dict.get("metadata")
            if saved != expected:
                raise ValueError(
                    "V-JEPA2 dense latent checkpoint/config mismatch: "
                    f"saved metadata={saved}, current={expected}")
            self.latent_mmd.load_state_dict(
                state_dict["vjepa2_dense_mmd"], strict=True)
            return

        if "dinov3_dense_nystrom_mmd_version" in state_dict:
            raise ValueError(
                "DINOv3 distribution state cannot be loaded into the Wan "
                "post-block latent feature space")
        version = state_dict.get("wan_kv_mmd_version")
        if version != 3:
            detail = (
                " Wan KV MMD v2 history is intentionally incompatible."
                if version == 2 else "")
            raise ValueError(
                "resume checkpoint has no compatible Wan post-block latent "
                f"MMD v3 state.{detail}")
        if "wan_latent_mmd" not in state_dict:
            raise ValueError(
                "resume checkpoint is missing fd.wan_latent_mmd history")
        expected = self._fd_metadata()
        saved = state_dict.get("metadata")
        if saved != expected:
            raise ValueError(
                "Wan post-block latent checkpoint/config mismatch: "
                f"saved metadata={saved}, current={expected}. "
                "KV EMA checkpoints are intentionally incompatible.")
        self.latent_mmd.load_state_dict(
            state_dict["wan_latent_mmd"], strict=True)

    def can_load_initialization_fd_state(self, state_dict: dict) -> bool:
        """Whether an initialization-only checkpoint carries our estimator."""
        if not isinstance(state_dict, dict):
            return False
        if self._uses_dinov3_dense_backend():
            return state_dict.get(
                "dinov3_dense_nystrom_mmd_version") == 1
        if getattr(self, "uses_fresh_video_gradcache", False):
            return (
                state_dict.get(self._dense_nystrom_version_key())
                == self._nystrom_fd_state_version()
            )
        if self.feature_type == "vjepa2_dense_latent":
            return state_dict.get("vjepa2_dense_mmd_version") == 2
        return state_dict.get("wan_kv_mmd_version") == 3

    # Historical entry points remain aliases for external callers.
    fresh_video_gradcache_prepare_replay = prepare_latent_gradients
    fresh_video_gradcache_replay = replay_generator_gradients
    fresh_video_gradcache_forward = collect_rollout_features

    _initialize_triple_dinov3_vjepa2_videomae_v2_features = _initialize_multi_encoder_features
    _replace_with_triple_nystrom_objective = _build_multi_encoder_objective
    _extract_dual_video_features = _extract_multi_encoder_features
    _fresh_video_dual_gradcache_forward = _collect_multi_encoder_rollout_features
    _dual_gradcache_prepare_replay = _prepare_multi_encoder_latent_gradients

# Old import/class spelling remains valid for existing callers.
WanKVMMD = ElasticForcingModel

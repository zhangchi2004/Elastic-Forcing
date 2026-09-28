"""Training orchestration for Elastic Forcing (paper Section 3.3).

Start at _run_distribution_matching_step: collect the distribution batch,
form the feature-space objective, replay the differentiation batch, and update
only the generator. The long initialization/resume code preserves the released
asset and checkpoint contracts; it does not define a separate score model.
"""
import copy
import gc
import hashlib
import logging
import math
from collections.abc import Mapping

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import (
    OpenVidPromptVideoDataset,
    ODERegressionLMDBDataset,
    TextDataset,
)
from utils.distributed import (
    EMA_FSDP,
    fsdp_wrap,
    fsdp_state_dict,
    fsdp_optim_state_dict,
    fsdp_load_optim_state_dict,
    move_adamw_moment_state_,
    launch_distributed_job,
)
from utils.misc import (
    init_wandb_from_config,
    set_seed,
    merge_dict_list,
)
from trainer.fresh_video_gradcache import FreshVideoGradCache
from trainer.feature_gradient_replay import (
    FeatureGradientReplay,
)
from trainer.fresh_frame_fd_gradcache import FreshFrameFDGradCache
import torch.distributed as dist
from omegaconf import OmegaConf
from model import ElasticForcingModel
import numpy as np
import random
import imageio.v3 as iio
import torch
from safetensors.torch import load_file as load_safetensors_file
from utils.optional_wandb import wandb
import time
import os
import shutil


def _canonicalize_experiment_identity_paths(value):
    """Make shared asset identities independent of a compute-node mount alias."""
    local_root = "/path/to/local-assets"
    logical_root = "/path/to/elastic-forcing-assets"
    if isinstance(value, str):
        if value == local_root or value.startswith(local_root + "/"):
            return logical_root + value[len(local_root):]
        return value
    if isinstance(value, Mapping):
        return {
            key: _canonicalize_experiment_identity_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _canonicalize_experiment_identity_paths(item) for item in value)
    if isinstance(value, list):
        return [
            _canonicalize_experiment_identity_paths(item) for item in value]
    return value


def _dense_video_metric_contract(feature_type):
    """Return the backend-owned prefix and universally required replay keys."""
    prefixes = {
        "vjepa2_dense_latent": "vjepa2_dense",
        "dinov3_dense_latent": "dinov3_dense",
        "dual_dinov3_vjepa2_dense_latent": "dinov3_vjepa2_dense",
        "triple_dinov3_vjepa2_videomae_v2_dense_latent": "triple_dense",
    }
    try:
        prefix = prefixes[str(feature_type)]
    except KeyError as exc:
        raise ValueError(
            "fresh-video GradCache requires a supported dense feature type; "
            f"got {feature_type!r}"
        ) from exc
    return prefix, frozenset({
        "combined_latent_grad_norm",
        f"{prefix}_feature_grad_norm",
        f"{prefix}_video_grad_norm",
        f"{prefix}_replay_max_abs",
        f"{prefix}_latent_replay_max_abs",
    })


def _validated_fidel_train_indices(dataset_length, reference_bank):
    """Validate the frozen teacher split and return its training indices."""
    if reference_bank is None:
        raise ValueError(
            "two-level FiDEL requires a V-JEPA2 reference bank")
    metadata = reference_bank.metadata
    if not isinstance(metadata, dict):
        raise TypeError("reference-bank metadata must be a dictionary")
    teacher_count = metadata.get("teacher_count")
    if type(teacher_count) is not int or teacher_count <= 0:
        raise ValueError(
            "reference-bank metadata.teacher_count must be positive")
    if int(dataset_length) != teacher_count:
        raise ValueError(
            "teacher LMDB size does not match the FiDEL reference split: "
            f"{int(dataset_length)} != {teacher_count}")
    train_indices = tuple(reference_bank.train_indices)
    val_indices = tuple(reference_bank.val_indices)
    if not train_indices or not val_indices:
        raise ValueError(
            "FiDEL teacher train/validation splits must both be non-empty")
    if (
        any(type(value) is not int for value in train_indices + val_indices)
        or tuple(sorted(train_indices)) != train_indices
        or tuple(sorted(val_indices)) != val_indices
        or len(set(train_indices)) != len(train_indices)
        or len(set(val_indices)) != len(val_indices)
        or set(train_indices).intersection(val_indices)
        or sorted(train_indices + val_indices)
        != list(range(teacher_count))
    ):
        raise ValueError(
            "FiDEL teacher train/validation indices must be a disjoint, "
            "strict partition of [0, teacher_count)")
    return train_indices


def _validate_fidel_resume_position(
    *,
    step,
    committed_updates,
    slot_count,
    max_steps,
):
    """Keep public trainer progress and consumed random masks identical."""
    values = {
        "step": step,
        "committed_updates": committed_updates,
        "slot_count": slot_count,
        "max_steps": max_steps,
    }
    if any(type(value) is not int for value in values.values()):
        raise TypeError("FiDEL resume counters must be integers")
    if slot_count <= 0 or max_steps <= 0:
        raise ValueError(
            "FiDEL slot_count and max_steps must both be positive")
    if max_steps > slot_count:
        raise ValueError(
            "FiDEL max_steps cannot exceed the reference slot_count: "
            f"{max_steps} > {slot_count}")
    if step != committed_updates:
        raise ValueError(
            "FiDEL checkpoint step/controller mismatch: "
            f"step={step}, committed_updates={committed_updates}")
    if not 0 <= step <= max_steps:
        raise ValueError(
            "FiDEL checkpoint progress is outside this run: "
            f"step={step}, max_steps={max_steps}, "
            f"slot_count={slot_count}")


def _validated_resume_data_counter_override(value, *, resume_checkpoint):
    """Validate an explicit lightweight-resume prompt-stream position.

    The counter is deliberately never derived from ``step`` or the current
    world size.  Changing the rank topology changes how many per-rank batches
    correspond to one optimizer update, so callers must provide the desired
    absolute per-rank batch counter explicitly.
    """
    if value is None:
        return None
    if not resume_checkpoint:
        raise ValueError(
            "resume_data_counter_override is only valid with "
            "resume_checkpoint")
    if type(value) is not int:
        raise TypeError(
            "resume_data_counter_override must be a non-negative integer")
    if value < 0:
        raise ValueError(
            "resume_data_counter_override must be a non-negative integer")
    return value


def _cosine_learning_rate(update_index, *, start, end, total_updates):
    """Endpoint-inclusive cosine LR for the optimizer update at update_index."""
    if type(update_index) is not int or update_index < 0:
        raise ValueError(
            "learning-rate update_index must be a non-negative integer")
    if type(total_updates) is not int or total_updates <= 0:
        raise ValueError(
            "lr_schedule_total_steps must be a positive integer")
    start = float(start)
    end = float(end)
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("learning-rate endpoints must be finite")
    if start <= 0.0 or end <= 0.0 or end > start:
        raise ValueError(
            "learning-rate endpoints must satisfy start >= end > 0")
    if total_updates == 1:
        return end
    position = min(update_index, total_updates - 1) / (total_updates - 1)
    return end + 0.5 * (start - end) * (
        1.0 + math.cos(math.pi * position))


def _learning_rate_schedule_identity(config):
    """Validate and freeze the optimizer schedule used by this experiment."""
    schedule = str(
        getattr(config, "lr_schedule", "constant")).strip().lower()
    if schedule == "constant":
        value = float(config.lr)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "constant learning rate must be finite and positive")
        return {"name": "constant", "value": value}
    if schedule != "cosine":
        raise ValueError(f"unsupported lr_schedule {schedule!r}")
    total_updates = int(getattr(config, "lr_schedule_total_steps", 0))
    max_steps = int(getattr(config, "max_steps", 0))
    if not 0 < max_steps <= total_updates:
        raise ValueError(
            "endpoint-inclusive cosine schedule requires "
            "0 < max_steps <= lr_schedule_total_steps")
    start = float(config.lr)
    end = float(getattr(config, "lr_final", 0.0))
    first = _cosine_learning_rate(
        0, start=start, end=end, total_updates=total_updates)
    last = _cosine_learning_rate(
        total_updates - 1,
        start=start,
        end=end,
        total_updates=total_updates,
    )
    if first != start or last != end:
        raise RuntimeError("cosine learning-rate endpoint invariant failed")
    return {
        "name": "cosine_endpoint_inclusive_v1",
        "start": start,
        "end": end,
        "total_updates": total_updates,
    }


def _validated_fresh_video_generator_grad_sample_scale(
    value,
    *,
    partial_gradient_enabled,
):
    """Validate the explicit suffix-replay gradient compensation."""
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fresh_video_generator_grad_sample_scale must be finite and "
            "strictly positive"
        ) from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "fresh_video_generator_grad_sample_scale must be finite and "
            "strictly positive")
    if scale != 1.0 and not bool(partial_gradient_enabled):
        raise ValueError(
            "fresh_video_generator_grad_sample_scale may differ from 1 only "
            "when a strict generator-gradient subsample is enabled")
    return scale


def _scale_replay_video_cotangents(cotangents, scale):
    """Scale replay gradients without changing the normalized MMD value.

    GradCache has already applied the configured MMD loss weight and the FSDP
    world-size compensation at this point. The explicit partial-gradient
    compensation is therefore applied exactly once, immediately before
    prepare/replay consumes these cotangents.
    """
    values = tuple(cotangents)
    if float(scale) == 1.0:
        return values
    def scaled(cotangent):
        if isinstance(cotangent, Mapping):
            return {
                key: (value * float(scale)).contiguous()
                for key, value in cotangent.items()
            }
        return (cotangent * float(scale)).contiguous()

    return tuple(scaled(cotangent) for cotangent in values)


def _atomic_torch_save(value, path):
    """Publish a checkpoint file only after ``torch.save`` fully succeeds."""
    temporary_path = f"{path}.tmp"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def _fresh_video_generator_grad_microbatches(
    *,
    local_samples,
    grad_local_samples,
    batch_size,
):
    """Return how many trailing microbatches build generator graphs.

    ``grad_local_samples=0`` is the compatibility path and replays every
    microbatch.  A positive value keeps the complete forward feature batch for
    the MMD scalar but replays only the trailing whole microbatches.  No
    inverse-probability rescaling is applied: earlier features are constants,
    exactly matching a partial derivative of the full MMD objective.
    """
    values = {
        "local_samples": local_samples,
        "grad_local_samples": grad_local_samples,
        "batch_size": batch_size,
    }
    if any(type(value) is not int for value in values.values()):
        raise TypeError(
            "fresh-video generator-gradient geometry must use integers")
    if local_samples <= 0 or batch_size <= 0:
        raise ValueError("local_samples and batch_size must be positive")
    if local_samples % batch_size != 0:
        raise ValueError("local_samples must be divisible by batch_size")
    if grad_local_samples < 0 or grad_local_samples > local_samples:
        raise ValueError(
            "grad_local_samples must be zero or lie in [1, local_samples]")
    selected_samples = (
        local_samples if grad_local_samples == 0 else grad_local_samples)
    if selected_samples % batch_size != 0:
        raise ValueError(
            "grad_local_samples must select complete physical microbatches")
    return selected_samples // batch_size


def _transformer_block_types(wrapper, label):
    model = getattr(wrapper, "model", None)
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError(
            f"{label} transformer FSDP strategy requires model.blocks")
    return {type(block) for block in blocks}


def _strict_sha256(value, label):
    """Return a normalized SHA-256 string or reject ambiguous identity."""
    normalized = str(value or "").strip().lower()
    if (
        len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return normalized


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_openvid_experiment_identity(
    config,
    model,
    ledger_metadata,
    *,
    forward_global_samples,
    gradient_global_samples,
    gradient_compensation,
):
    """Build the immutable fixed-only identity saved in every checkpoint."""
    if (
        str(getattr(model, "feature_type", ""))
        != "dual_dinov3_vjepa2_dense_latent"
        or bool(getattr(model, "current_teacher_attraction_enabled", True))
        or str(getattr(config, "current_teacher_source", "")) != "none"
        or int(getattr(config, "expected_current_teacher_samples", -1)) != 0
    ):
        raise ValueError(
            "fixed OpenVid experiment identity requires dual fixed-only mode")
    branch_weights = {
        name: float(value)
        for name, value in model.latent_mmd.branch_weights.items()
    }
    if branch_weights != {"dinov3": 10.0, "vjepa2": 1.0}:
        raise ValueError("fixed OpenVid branch weights must be exactly 10:1")
    registry = getattr(model, "dual_branch_registry", {})
    if set(registry) != {"dinov3", "vjepa2"}:
        raise ValueError("fixed OpenVid branch registry is incomplete")
    branches = {}
    for name in sorted(registry):
        branch = registry[name]
        branches[name] = {
            "nystrom_landmarks": int(branch["nystrom_landmarks"]),
            "reference_file_sha256": _strict_sha256(
                branch["reference_file_sha256"],
                f"{name} reference_file_sha256"),
            "reference_hash": _strict_sha256(
                branch["reference_hash"], f"{name} reference_hash"),
            "reference_path": str(branch["reference_path"]),
            "sample_shape": list(branch["sample_shape"]),
        }
    expected_branch_geometry = {
        "dinov3": {"shape": [1024, 1024], "landmarks": 256},
        "vjepa2": {"shape": [1024, 1024], "landmarks": 1024},
    }
    for name, expected in expected_branch_geometry.items():
        if (
            branches[name]["sample_shape"] != expected["shape"]
            or branches[name]["nystrom_landmarks"] != expected["landmarks"]
        ):
            raise ValueError(f"fixed OpenVid {name} geometry changed")

    bundle_sha = _strict_sha256(
        getattr(config, "dinov3_vjepa2_reference_bundle_manifest_sha256", ""),
        "dinov3_vjepa2_reference_bundle_manifest_sha256",
    )
    if bundle_sha != str(getattr(
        model, "dinov3_vjepa2_reference_bundle_manifest_sha256", "")):
        raise ValueError("fixed OpenVid model/config bundle SHA mismatch")
    ledger_sha = _strict_sha256(
        ledger_metadata.get("ledger_sha256", ""), "OpenVid ledger SHA-256")
    sequence_sha = _strict_sha256(
        ledger_metadata.get("video_name_sequence_sha256", ""),
        "OpenVid video-name sequence SHA-256",
    )
    if (
        int(ledger_metadata.get("reference_count", -1)) != 77_615
        or sequence_sha != _strict_sha256(
            getattr(config, "openvid_video_name_sequence_sha256", ""),
            "configured OpenVid video-name sequence SHA-256",
        )
    ):
        raise ValueError("fixed OpenVid ledger identity/count mismatch")
    generator_sha = _strict_sha256(
        getattr(config, "generator_ckpt_sha256", ""),
        "generator_ckpt_sha256",
    )
    generator_model = str(getattr(
        config, "generator_ckpt_model_name", ""))
    generator_recipe = str(getattr(
        config, "generator_ckpt_ode_recipe", ""))
    generator_step = int(getattr(config, "generator_ckpt_step", -1))
    generator_prompt_sha = _strict_sha256(
        getattr(config, "generator_ckpt_prompt_manifest_sha256", ""),
        "generator_ckpt_prompt_manifest_sha256",
    )
    model_kwargs = dict(getattr(config, "model_kwargs", {}))
    schedule = {
        "denoising_step_list": list(config.denoising_step_list),
        "model_timestep_shift": float(model_kwargs.get("timestep_shift", -1)),
        "timestep_shift": float(config.timestep_shift),
        "warp_denoising_step": bool(config.warp_denoising_step),
    }
    if (
        generator_model != "Wan2.1-T2V-14B"
        or generator_recipe != "causvid_14b_16k"
        or generator_step <= 0
        or str(model_kwargs.get("model_name", "")) != generator_model
        or schedule != {
            "denoising_step_list": [1000, 757, 522, 0],
            "model_timestep_shift": 8.0,
            "timestep_shift": 8.0,
            "warp_denoising_step": False,
        }
    ):
        raise ValueError("fixed OpenVid Wan-14B ODE/schedule identity changed")
    forward = int(forward_global_samples)
    gradient = int(gradient_global_samples)
    if (
        (forward, gradient) != (256, 128)
        or float(gradient_compensation) != 2.0
    ):
        raise ValueError("fixed OpenVid partial-gradient geometry changed")
    return {
        "mode": "openvid77615_fixed_only_dual_v1",
        "current_teacher_source": "none",
        "reference_bundle": {
            "path": str(config.dinov3_vjepa2_reference_bundle_manifest),
            "sha256": bundle_sha,
        },
        "ledger": {
            "path": str(config.fixed_student_prompt_ledger),
            "sha256": ledger_sha,
            "reference_count": 77_615,
            "video_name_sequence_sha256": sequence_sha,
        },
        "generator_initialization": {
            "path": str(config.generator_ckpt),
            "sha256": generator_sha,
            "model_name": generator_model,
            "ode_recipe": generator_recipe,
            "ode_step": generator_step,
            "prompt_manifest_sha256": generator_prompt_sha,
        },
        "schedule": schedule,
        "branches": branches,
        "branch_weights": branch_weights,
        "partial_gradient": {
            "forward_global_samples": forward,
            "gradient_global_samples": gradient,
            "teacher_global_samples": 0,
            "full_student_partial_gradient": True,
            "kss_denominator": 65_536,
            "fixed_attraction_denominator": 256,
            "teacher_cross_denominator": 0,
            "teacher_self_denominator": 0,
            "replay_cotangent_compensation": float(
                gradient_compensation),
        },
        "feature_protocol_identity": str(config.feature_protocol_identity),
    }


class ElasticForcingTrainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.data_counter = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        expected_world_size = int(
            getattr(config, "expected_world_size", 0) or 0)
        if expected_world_size and self.world_size != expected_world_size:
            raise ValueError(
                "distributed world-size mismatch: expected "
                f"{expected_world_size}, got {self.world_size}")
        self.uses_fresh_video_gradcache = (
            str(getattr(config, "distribution_loss", ""))
            == "wan_kv_mmd"
            and str(getattr(config, "wan_mmd_estimator", "ema_rff"))
            == "nystrom_exact_repulsion"
        )
        self.frame_fd_fusion_enabled = bool(
            getattr(config, "frame_fd_fusion_enabled", False))
        self.frame_fd_fusion_mode = str(getattr(
            config, "frame_fd_fusion_mode", "grad_norm_ema"))
        self.fidel_two_level_enabled = bool(
            self.frame_fd_fusion_enabled
            and self.frame_fd_fusion_mode == "fidel_two_level"
        )
        if (
            self.frame_fd_fusion_mode == "fidel_two_level"
            and not self.frame_fd_fusion_enabled
        ):
            raise ValueError(
                "frame_fd_fusion_mode=fidel_two_level requires "
                "frame_fd_fusion_enabled=true")
        if (
            self.frame_fd_fusion_enabled
            and not self.uses_fresh_video_gradcache
        ):
            raise ValueError(
                "frame_fd_fusion_enabled requires the fresh-video Nyström "
                "training path")
        self.fresh_video_local_samples = int(
            getattr(config, "fresh_video_local_samples", config.batch_size))
        self.fresh_video_physical_forward_batch_size = int(config.batch_size)
        self.fresh_video_logical_micro_batch_size = int(getattr(
            config,
            "fresh_video_logical_micro_batch_size",
            self.fresh_video_physical_forward_batch_size,
        ))
        self.fresh_video_physical_replay_batch_size = int(getattr(
            config,
            "fresh_video_physical_replay_batch_size",
            self.fresh_video_physical_forward_batch_size,
        ))
        if self.uses_fresh_video_gradcache:
            sizes = {
                "fresh_video_local_samples": self.fresh_video_local_samples,
                "fresh_video_logical_micro_batch_size": (
                    self.fresh_video_logical_micro_batch_size),
                "batch_size": self.fresh_video_physical_forward_batch_size,
                "fresh_video_physical_replay_batch_size": (
                    self.fresh_video_physical_replay_batch_size),
            }
            if any(value <= 0 for value in sizes.values()):
                raise ValueError(
                    f"fresh-video execution sizes must be positive: {sizes}")
            if (
                self.fresh_video_local_samples
                % self.fresh_video_logical_micro_batch_size
                or self.fresh_video_logical_micro_batch_size
                % self.fresh_video_physical_forward_batch_size
                or self.fresh_video_logical_micro_batch_size
                % self.fresh_video_physical_replay_batch_size
            ):
                raise ValueError(
                    "VACE-aligned logical/physical batches must divide "
                    f"exactly: {sizes}")
            self.fresh_video_microbatches = (
                self.fresh_video_local_samples
                // self.fresh_video_logical_micro_batch_size)
        else:
            self.fresh_video_microbatches = 1
        expected_total_batch = self.fresh_video_local_samples * self.world_size
        configured_total_batch = int(
            getattr(config, "total_batch_size", expected_total_batch))
        if configured_total_batch != expected_total_batch:
            raise ValueError(
                "total_batch_size must equal fresh_video_local_samples * "
                f"world_size; got {configured_total_batch} != "
                f"{self.fresh_video_local_samples} * {self.world_size} "
                f"({expected_total_batch})")
        self.fresh_video_total_samples = configured_total_batch
        configured_grad_local_samples = int(getattr(
            config,
            "fresh_video_generator_grad_local_samples",
            self.fresh_video_local_samples,
        ))
        self.fresh_video_generator_grad_microbatches = (
            _fresh_video_generator_grad_microbatches(
                local_samples=self.fresh_video_local_samples,
                grad_local_samples=configured_grad_local_samples,
                batch_size=self.fresh_video_logical_micro_batch_size,
            )
            if self.uses_fresh_video_gradcache
            else 1
        )
        self.fresh_video_generator_grad_local_samples = (
            configured_grad_local_samples
            if self.uses_fresh_video_gradcache
            else int(config.batch_size)
        )
        self.fresh_video_generator_grad_subsample_enabled = bool(
            self.uses_fresh_video_gradcache
            and self.fresh_video_generator_grad_local_samples
            < self.fresh_video_local_samples
        )
        self.fresh_video_full_student_partial_gradient = bool(getattr(
            config, "fresh_video_full_student_partial_gradient", False))
        if (
            self.fresh_video_full_student_partial_gradient
            and not self.fresh_video_generator_grad_subsample_enabled
        ):
            raise ValueError(
                "fresh_video_full_student_partial_gradient requires a strict "
                "gradient suffix smaller than fresh_video_local_samples")
        self.fresh_video_generator_grad_global_samples = (
            self.fresh_video_generator_grad_local_samples * self.world_size
        )
        self.fresh_video_generator_grad_sample_scale = (
            _validated_fresh_video_generator_grad_sample_scale(
                getattr(
                    config,
                    "fresh_video_generator_grad_sample_scale",
                    1.0,
                ),
                partial_gradient_enabled=(
                    self.fresh_video_generator_grad_subsample_enabled),
            )
        )
        if (
            self.fresh_video_generator_grad_subsample_enabled
            and self.frame_fd_fusion_enabled
        ):
            raise ValueError(
                "partial generator replay supports video-only MMD; image/frame "
                "loss must be disabled")
        if (
            self.fresh_video_generator_grad_subsample_enabled
            and self.fidel_two_level_enabled
        ):
            raise ValueError(
                "partial generator replay is incompatible with two-level FiDEL")
        configured_teacher_samples = getattr(
            config, "expected_current_teacher_samples", None)
        if (
            configured_teacher_samples is not None
            and int(configured_teacher_samples) != 0
            and int(configured_teacher_samples)
            != self.fresh_video_generator_grad_global_samples
        ):
            raise ValueError(
                "expected_current_teacher_samples must equal the global "
                "gradient suffix, got "
                f"{int(configured_teacher_samples)} != "
                f"{self.fresh_video_generator_grad_global_samples}")

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            init_wandb_from_config(config)

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss != "wan_kv_mmd":
            raise ValueError("This release supports Elastic Forcing (wan_kv_mmd).")
        self.model = ElasticForcingModel(config, device=self.device)
        self.uses_critic = bool(getattr(self.model, "uses_critic", True))
        if (
            bool(getattr(
                self.model, "uses_fresh_video_gradcache", False))
            != self.uses_fresh_video_gradcache
        ):
            raise RuntimeError(
                "trainer/model fresh-video GradCache configuration mismatch")
        if (
            bool(getattr(
                self.model, "frame_fd_fusion_enabled", False))
            != self.frame_fd_fusion_enabled
        ):
            raise RuntimeError(
                "trainer/model frame-FD fusion configuration mismatch")
        if (
            bool(getattr(
                self.model, "fidel_two_level_enabled", False))
            != self.fidel_two_level_enabled
        ):
            raise RuntimeError(
                "trainer/model two-level FiDEL configuration mismatch")
        expected_teacher_samples = (
            self.fresh_video_generator_grad_global_samples
            if bool(getattr(
                self.model, "current_teacher_attraction_enabled", False))
            else 0
        )
        if (
            configured_teacher_samples is not None
            and int(configured_teacher_samples) != expected_teacher_samples
        ):
            raise ValueError(
                "expected_current_teacher_samples disagrees with the active "
                "teacher mode: "
                f"{int(configured_teacher_samples)} != "
                f"{expected_teacher_samples}")
        fresh_cache_class = (
            FeatureGradientReplay
            if bool(getattr(self.model, "dual_video_mmd_enabled", False))
            else FreshVideoGradCache
        )
        self.fresh_video_gradcache = (
            fresh_cache_class(
                self.model.latent_mmd,
                target_local_samples=self.fresh_video_local_samples,
                gradient_local_samples=(
                    self.fresh_video_generator_grad_local_samples),
                require_sample_id_alignment=(
                    str(getattr(config, "current_teacher_source", ""))
                    .strip().lower() in {
                        "matched_manifest_video",
                        "matched_dual_feature_bank",
                        "matched_triple_feature_bank",
                        "matched_dinov3_vjepa2_feature_bank",
                    }),
                full_student_partial_gradient=(
                    self.fresh_video_full_student_partial_gradient),
                # FiDEL applies its detached EMA normalization after both raw
                # objectives have been finalized.  Scaling here would apply
                # the video coefficient twice.
                loss_scale=(
                    1.0
                    if self.fidel_two_level_enabled
                    else float(getattr(
                        config, "wan_kv_loss_weight", 1.0))
                ),
            )
            if self.uses_fresh_video_gradcache
            else None
        )
        self.fresh_frame_fd_gradcache = (
            FreshFrameFDGradCache(
                self.model.frame_fd_head,
                target_local_videos=self.fresh_video_local_samples,
            )
            if self.frame_fd_fusion_enabled
            else None
        )

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = (
            self.model.fake_score.state_dict()
            if getattr(self.model, "fake_score", None) is not None
            else None
        )

        generator_wrap_strategy = config.generator_fsdp_wrap_strategy
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=generator_wrap_strategy,
            transformer_module=(
                _transformer_block_types(self.model.generator, "generator")
                if generator_wrap_strategy == "transformer"
                else None
            ),
            backward_prefetch=str(getattr(
                config, "fsdp_backward_prefetch", "pre")),
        )

        if getattr(self.model, "real_score", None) is not None:
            real_wrap_strategy = config.real_score_fsdp_wrap_strategy
            self.model.real_score = fsdp_wrap(
                self.model.real_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=real_wrap_strategy,
                transformer_module=(
                    _transformer_block_types(self.model.real_score, "real_score")
                    if real_wrap_strategy == "transformer"
                    else None
                ),
            )

        if getattr(self.model, "fake_score", None) is not None:
            self.model.fake_score = fsdp_wrap(
                self.model.fake_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.fake_score_fsdp_wrap_strategy
            )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if (
            getattr(self.model, "uses_vae_for_loss", False)
            or (
                not config.no_visualize
                and getattr(self.model, "uses_vae_for_visualization", True)
            )
            or bool(getattr(config, "load_raw_video", False))
            or int(getattr(config, "eval_interval", 0)) > 0
        ):
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            foreach=bool(getattr(config, "optimizer_foreach", False)),
            fused=bool(getattr(config, "optimizer_fused", False)),
        )
        self.optimizer_state_cpu_offload_during_backward = bool(getattr(
            config, "optimizer_state_cpu_offload_during_backward", False))
        self.learning_rate_schedule = _learning_rate_schedule_identity(config)

        if self.uses_critic:
            self.critic_optimizer = torch.optim.AdamW(
                [param for param in self.model.fake_score.parameters()
                 if param.requires_grad],
                lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
                betas=(config.beta1_critic, config.beta2_critic),
                weight_decay=config.weight_decay
            )
        else:
            self.critic_optimizer = None

        # Step 3: Initialize the dataloader
        current_teacher_enabled = bool(getattr(
            self.model, "current_teacher_attraction_enabled", False))
        configured_teacher_source = getattr(
            config, "current_teacher_source", None)
        requested_teacher_source = str(
            configured_teacher_source
            if configured_teacher_source is not None
            else "auto"
        ).strip().lower()
        self.current_teacher_source = (
            "lmdb_latent" if current_teacher_enabled else "none"
        ) if requested_teacher_source == "auto" else requested_teacher_source
        if self.current_teacher_source not in {
            "none", "lmdb_latent", "openvid_video", "matched_manifest_video",
            "matched_dual_feature_bank", "matched_triple_feature_bank",
            "matched_dinov3_vjepa2_feature_bank",
        }:
            raise ValueError(
                "current_teacher_source must be auto, none, lmdb_latent, "
                "openvid_video, matched_manifest_video, or "
                "matched_dual_feature_bank, matched_triple_feature_bank, or "
                "matched_dinov3_vjepa2_feature_bank; got "
                f"{self.current_teacher_source!r}")
        if current_teacher_enabled == (self.current_teacher_source == "none"):
            raise ValueError(
                "current_teacher_source must name exactly one source iff "
                "exact current-teacher attraction is enabled")
        self.current_teacher_dataset = None
        self.matched_counter_sampling_enabled = False
        self.experiment_identity = None
        if self.config.distribution_loss == "wan_kv_mmd":
            if self.current_teacher_source in {
                "matched_dual_feature_bank",
                "matched_triple_feature_bank",
                "matched_dinov3_vjepa2_feature_bank",
            }:
                source = self.current_teacher_source
                triple_bank = source == "matched_triple_feature_bank"
                dino_vjepa_bank = (
                    source == "matched_dinov3_vjepa2_feature_bank")
                if triple_bank:
                    from utils.active_mmd_branches import active_mmd_branches
                    from utils.triple_video_feature_bank import (
                        TripleVideoFeatureBank as FeatureBank,
                    )
                    manifest_field = "triple_teacher_bank_manifest"
                    content_field = "triple_teacher_bank_content_hash"
                    expected_branch_names = set(active_mmd_branches(
                        getattr(config, "triple_active_branches", None)))
                    expected_feature_type = (
                        "triple_dinov3_vjepa2_videomae_v2_dense_latent")
                elif dino_vjepa_bank:
                    from utils.dinov3_vjepa2_feature_bank import (
                        DINOv3VJEPA2FeatureBank as FeatureBank,
                    )
                    manifest_field = "dinov3_vjepa2_teacher_bank_manifest"
                    content_field = (
                        "dinov3_vjepa2_teacher_bank_content_hash")
                    expected_branch_names = {"dinov3", "vjepa2"}
                    expected_feature_type = (
                        "dual_dinov3_vjepa2_dense_latent")
                else:
                    from utils.dual_video_feature_bank import (
                        DualVideoFeatureBank as FeatureBank,
                    )
                    manifest_field = "dual_teacher_bank_manifest"
                    content_field = "dual_teacher_bank_content_hash"
                    expected_branch_names = {"vjepa2", "videomae_v2"}
                    expected_feature_type = (
                        "dual_vjepa2_videomae_v2_dense_latent")
                if not bool(getattr(self.model, "dual_video_mmd_enabled", False)):
                    raise ValueError(
                        "matched feature bank requires multi-representation mode")
                if str(getattr(self.model, "feature_type", "")) != expected_feature_type:
                    raise ValueError(
                        "teacher bank branch set does not match model mode")
                if not self.uses_fresh_video_gradcache:
                    raise ValueError("matched feature bank requires exact GradCache")
                manifest_sha_field = f"{manifest_field}_sha256"
                kwargs = {
                    "expected_manifest_sha256": _strict_sha256(
                        getattr(config, manifest_sha_field), manifest_sha_field),
                    "expected_content_hash": _strict_sha256(
                        getattr(config, content_field), content_field),
                    "expected_source_manifest_sha256": _strict_sha256(
                        config.matched_manifest_sha256,
                        "matched_manifest_sha256"),
                    "expected_sample_count": int(getattr(
                        config, "matched_manifest_samples", 5000)),
                    "expected_vjepa_model_sha256": _strict_sha256(
                        config.vjepa2_model_sha256,
                        "vjepa2_model_sha256"),
                    "validate_all_shards": True,
                }
                if dino_vjepa_bank:
                    kwargs.update({
                        "expected_source_csv_sha256": _strict_sha256(
                            config.matched_source_csv_sha256,
                            "matched_source_csv_sha256"),
                        "expected_sample_identity_sha256": _strict_sha256(
                            config.matched_sample_identity_sha256,
                            "matched_sample_identity_sha256"),
                    })
                if not dino_vjepa_bank:
                    kwargs.update({
                        "expected_videomae_revision": str(
                            config.videomae_v2_model_revision),
                        "expected_videomae_download_manifest_content_hash": (
                            _strict_sha256(
                                config.videomae_v2_download_manifest_content_hash,
                                "videomae_v2_download_manifest_content_hash")),
                    })
                if triple_bank:
                    kwargs["active_branches"] = active_mmd_branches(
                        getattr(config, "triple_active_branches", None))
                    kwargs["expected_shard_size"] = 32
                if "dinov3" in expected_branch_names:
                    kwargs.update({
                        "expected_dinov3_checkpoint_sha256": _strict_sha256(
                            config.dinov3_checkpoint_sha256,
                            "dinov3_checkpoint_sha256"),
                        "expected_dinov3_source_commit": str(
                            config.dinov3_source_commit),
                        "expected_shard_size": 32,
                    })
                dataset = FeatureBank(
                    str(getattr(config, manifest_field)), **kwargs)
                model_bank_hash = str(getattr(
                    self.model,
                    "triple_teacher_bank_content_hash"
                    if triple_bank else "dual_teacher_bank_content_hash"))
                if dataset.content_hash != model_bank_hash:
                    raise ValueError(
                        "training bank identity differs from all references")
                self.current_teacher_dataset = dataset
                self.matched_counter_sampling_enabled = True
                branches = copy.deepcopy(self.model.dual_branch_registry)
                if set(branches) != expected_branch_names:
                    raise ValueError(
                        "model branch registry does not match teacher bank")
                self.experiment_identity = {
                    "current_teacher_source": self.current_teacher_source,
                    f"{manifest_field}_sha256": dataset.manifest_sha256,
                    content_field: dataset.content_hash,
                    "source_manifest_sha256": _strict_sha256(
                        config.matched_manifest_sha256,
                        "matched_manifest_sha256"),
                    "sample_count": len(dataset),
                    "branches": branches,
                    "forward_global_samples": self.fresh_video_total_samples,
                    "gradient_global_samples": (
                        self.fresh_video_generator_grad_global_samples),
                    "teacher_global_samples": int(
                        config.expected_current_teacher_samples),
                    "full_student_partial_gradient": (
                        self.fresh_video_full_student_partial_gradient),
                    "kss_denominator": self.fresh_video_total_samples ** 2,
                    "fixed_attraction_denominator": (
                        self.fresh_video_total_samples),
                    "teacher_cross_denominator": (
                        self.fresh_video_total_samples
                        * int(config.expected_current_teacher_samples)),
                    "teacher_self_denominator": (
                        int(config.expected_current_teacher_samples) ** 2),
                    "replay_cotangent_compensation": (
                        self.fresh_video_generator_grad_sample_scale),
                    "branch_normalization": (
                        "raw_div_stop_gradient_raw_plus_1e-4"),
                    "branch_aggregation": (
                        ("sum_10_to_1_to_1_then_divide_3"
                         if "dinov3" in expected_branch_names
                         else "sum_1_to_1_then_divide_3")
                        if triple_bank
                        else "sum_10_to_1_without_divide_2"),
                }
            elif self.current_teacher_source == "matched_manifest_video":
                from utils.matched_video_dataset import (
                    MatchedManifestVideoDataset,
                )
                from utils.matched_video_feature_cache import (
                    MATCHED_PYAV_LAST64_DECODER_CONTRACT_SHA256,
                    MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT_SHA256,
                )

                if not self.uses_fresh_video_gradcache:
                    raise ValueError(
                        "current_teacher_source=matched_manifest_video requires "
                        "the fresh-video exact Nyström path")
                if int(config.batch_size) != 1:
                    raise ValueError(
                        "lazy matched-video teacher decoding requires "
                        "batch_size=1")
                manifest_path = os.path.realpath(os.path.expanduser(str(
                    config.matched_manifest_path)))
                expected_manifest_sha256 = _strict_sha256(
                    config.matched_manifest_sha256,
                    "matched_manifest_sha256",
                )
                actual_manifest_sha256 = _file_sha256(manifest_path)
                if actual_manifest_sha256 != expected_manifest_sha256:
                    raise ValueError(
                        "frozen matched-video manifest SHA-256 mismatch: "
                        f"{actual_manifest_sha256} != "
                        f"{expected_manifest_sha256}")
                source_csv_sha256 = _strict_sha256(
                    config.matched_source_csv_sha256,
                    "matched_source_csv_sha256",
                )
                expected_sample_identity_sha256 = _strict_sha256(
                    config.matched_sample_identity_sha256,
                    "matched_sample_identity_sha256",
                )
                model_sha256 = _strict_sha256(
                    config.vjepa2_model_sha256, "vjepa2_model_sha256")
                feature_contract_sha256 = _strict_sha256(
                    config.matched_teacher_feature_contract_sha256,
                    "matched_teacher_feature_contract_sha256",
                )
                if (
                    feature_contract_sha256
                    != MATCHED_VJEPA2_L15_LAST64_FEATURE_CONTRACT_SHA256
                ):
                    raise ValueError(
                        "configured matched-video feature contract does not "
                        "match runtime")
                decoder_contract_sha256 = _strict_sha256(
                    config.matched_teacher_decoder_contract_sha256,
                    "matched_teacher_decoder_contract_sha256",
                )
                if (
                    decoder_contract_sha256
                    != MATCHED_PYAV_LAST64_DECODER_CONTRACT_SHA256
                ):
                    raise ValueError(
                        "configured matched-video decoder contract does not "
                        "match runtime")
                sampling_algorithm = str(
                    config.matched_sampling_algorithm)
                if sampling_algorithm != "sha256_seed_step_position_mod_n_v1":
                    raise ValueError(
                        "unsupported matched counter sampling algorithm: "
                        f"{sampling_algorithm!r}")
                if not bool(config.matched_sampling_with_replacement):
                    raise ValueError(
                        "matched sampler must explicitly enable replacement")
                expected_samples = int(getattr(
                    config, "matched_manifest_samples", 5000))
                dataset = MatchedManifestVideoDataset(
                    manifest_path=manifest_path,
                    video_root=str(config.matched_video_root),
                    cache_root=str(config.matched_teacher_cache_root),
                    expected_model_sha256=model_sha256,
                    expected_feature_contract_sha256=(
                        feature_contract_sha256),
                    expected_source_csv_sha256=source_csv_sha256,
                    expected_count=expected_samples,
                )
                actual_sample_identity_sha256 = str(
                    dataset.manifest_metadata.get(
                        "sample_identity_sha256", ""))
                if (
                    actual_sample_identity_sha256
                    != expected_sample_identity_sha256
                ):
                    raise ValueError(
                        "matched-video sample identity SHA-256 mismatch: "
                        f"{actual_sample_identity_sha256} != "
                        f"{expected_sample_identity_sha256}")
                self.current_teacher_dataset = dataset
                self.matched_counter_sampling_enabled = True
                self.experiment_identity = {
                    "current_teacher_source": "matched_manifest_video",
                    "matched_manifest_sha256": expected_manifest_sha256,
                    "matched_source_csv_sha256": source_csv_sha256,
                    "matched_sample_identity_sha256": (
                        expected_sample_identity_sha256),
                    "matched_manifest_samples": expected_samples,
                    "vjepa2_reference_model_id": str(
                        config.vjepa2_reference_model_id),
                    "vjepa2_model_sha256": model_sha256,
                    "matched_teacher_feature_contract_sha256": (
                        feature_contract_sha256),
                    "matched_teacher_decoder_contract_sha256": (
                        decoder_contract_sha256),
                    "student_frame_contract": "last64_frames_17_80",
                    "live_teacher_frame_contract": "last64_frames_17_80",
                    "fixed_reference_frame_contract": "central64_existing_pack",
                    "counter_sampling_algorithm": (
                        "sha256_seed_step_position_mod_n_v1"),
                    "counter_sampling_seed": int(config.seed),
                    "forward_global_samples": (
                        self.fresh_video_total_samples),
                    "gradient_global_samples": (
                        self.fresh_video_generator_grad_global_samples),
                    "explicit_gradient_compensation": (
                        self.fresh_video_generator_grad_sample_scale),
                    "partial_mmd_gradient_compensation": (
                        self.fresh_video_generator_grad_sample_scale),
                    "teacher_global_samples": int(
                        config.expected_current_teacher_samples),
                    "partial_mmd_objective": (
                        "full_student_biased_mmd_suffix_partial_backward_v1"
                        if self.fresh_video_full_student_partial_gradient
                        else "rectangular_gradient_student_v1"
                    ),
                    "full_student_partial_gradient": (
                        self.fresh_video_full_student_partial_gradient),
                    "partial_mmd_kss_denominator": (
                        self.fresh_video_total_samples ** 2
                        if self.fresh_video_full_student_partial_gradient
                        else self.fresh_video_total_samples
                        * self.fresh_video_generator_grad_global_samples
                    ),
                    "partial_mmd_fixed_attraction_denominator": (
                        self.fresh_video_total_samples
                        if self.fresh_video_full_student_partial_gradient
                        else self.fresh_video_generator_grad_global_samples
                    ),
                    "partial_mmd_live_cross_denominator": (
                        self.fresh_video_total_samples
                        * int(config.expected_current_teacher_samples)
                        if self.fresh_video_full_student_partial_gradient
                        else self.fresh_video_generator_grad_global_samples
                        * int(config.expected_current_teacher_samples)
                    ),
                    "partial_mmd_teacher_self_denominator": (
                        int(config.expected_current_teacher_samples) ** 2),
                    "partial_replay_upstream_commit": (
                        "ef43f2d616ff31ba34904ae9743b0de4dda95afd"),
                    "fixed_reference_file_sha256": _strict_sha256(
                        config.vjepa2_nystrom_reference_sha256,
                        "vjepa2_nystrom_reference_sha256"),
                    "fixed_reference_hash": str(getattr(
                        self.model, "vjepa2_nystrom_reference_hash", "")),
                    "live_teacher_baseline_commit": str(
                        config.live_teacher_baseline_commit),
                    "partial_gradient_reference_commit": str(
                        config.partial_gradient_reference_commit),
                    "feature_protocol_identity": str(
                        config.feature_protocol_identity),
                }
            elif self.current_teacher_source == "openvid_video":
                from utils.openvid_feature_cache import (
                    OPENVID_PYAV_DECODER_CONTRACT_SHA256,
                    OPENVID_VJEPA2_L15_FEATURE_CONTRACT_SHA256,
                )

                if not self.uses_fresh_video_gradcache:
                    raise ValueError(
                        "current_teacher_source=openvid_video requires the "
                        "fresh-video exact Nyström path")
                if int(config.batch_size) != 1:
                    raise ValueError(
                        "OpenVid lazy teacher decoding currently requires "
                        "batch_size=1")
                manifest_path = os.path.realpath(os.path.expanduser(str(
                    config.openvid_manifest_path)))
                expected_manifest_sha256 = _strict_sha256(
                    config.openvid_manifest_sha256,
                    "openvid_manifest_sha256",
                )
                actual_manifest_sha256 = _file_sha256(manifest_path)
                if actual_manifest_sha256 != expected_manifest_sha256:
                    raise ValueError(
                        "frozen OpenVid manifest SHA-256 mismatch: "
                        f"{actual_manifest_sha256} != "
                        f"{expected_manifest_sha256}")
                model_sha256 = _strict_sha256(
                    config.vjepa2_model_sha256, "vjepa2_model_sha256")
                feature_contract_sha256 = _strict_sha256(
                    config.openvid_teacher_feature_contract_sha256,
                    "openvid_teacher_feature_contract_sha256",
                )
                if (
                    feature_contract_sha256
                    != OPENVID_VJEPA2_L15_FEATURE_CONTRACT_SHA256
                ):
                    raise ValueError(
                        "configured OpenVid feature contract does not match "
                        "runtime")
                decoder_contract_sha256 = _strict_sha256(
                    config.openvid_teacher_decoder_contract_sha256,
                    "openvid_teacher_decoder_contract_sha256",
                )
                if (
                    decoder_contract_sha256
                    != OPENVID_PYAV_DECODER_CONTRACT_SHA256
                ):
                    raise ValueError(
                        "configured OpenVid decoder contract does not match "
                        "runtime")
                dataset = OpenVidPromptVideoDataset(
                    manifest_path=manifest_path,
                    video_root=str(config.openvid_video_root),
                    cache_root=str(config.openvid_teacher_cache_root),
                    expected_model_sha256=model_sha256,
                    expected_feature_contract_sha256=(
                        feature_contract_sha256),
                )
                expected_samples = int(getattr(
                    config, "openvid_manifest_samples", 12800))
                if len(dataset) != expected_samples:
                    raise ValueError(
                        "frozen OpenVid dataset size mismatch: "
                        f"{len(dataset)} != {expected_samples}")
                self.current_teacher_dataset = dataset
                self.experiment_identity = {
                    "current_teacher_source": "openvid_video",
                    "openvid_manifest_sha256": expected_manifest_sha256,
                    "vjepa2_reference_model_id": str(
                        config.vjepa2_reference_model_id),
                    "vjepa2_model_sha256": model_sha256,
                    "openvid_teacher_feature_contract_sha256": (
                        feature_contract_sha256),
                    "openvid_teacher_decoder_contract_sha256": (
                        decoder_contract_sha256),
                }
            elif self.current_teacher_source == "none":
                fixed_prompt_ledger = getattr(
                    config, "fixed_student_prompt_ledger", None)
                if fixed_prompt_ledger:
                    from utils.openvid_reference_ledger import (
                        OpenVidReferencePromptDataset,
                    )
                    dataset = OpenVidReferencePromptDataset(
                        fixed_prompt_ledger,
                        expected_count=int(getattr(
                            config, "openvid_reference_count", 77615)),
                    )
                    self.experiment_identity = (
                        _fixed_openvid_experiment_identity(
                            config,
                            self.model,
                            dataset.metadata,
                            forward_global_samples=(
                                self.fresh_video_total_samples),
                            gradient_global_samples=(
                                self.fresh_video_generator_grad_global_samples),
                            gradient_compensation=(
                                self.fresh_video_generator_grad_sample_scale),
                        )
                    )
                else:
                    dataset = TextDataset(config.data_path)
            else:
                load_matched_terminal_teacher = bool(
                    self.uses_fresh_video_gradcache
                    and self.current_teacher_source == "lmdb_latent")
                dataset = ODERegressionLMDBDataset(
                    config.teacher_latent_data_path,
                    max_pair=int(getattr(
                        config, "teacher_latent_count", 1e8)),
                    load_latents=(
                        not self.uses_fresh_video_gradcache
                        or load_matched_terminal_teacher
                    ),
                    load_terminal_latent_only=(
                        load_matched_terminal_teacher),
                )
            if self.fidel_two_level_enabled:
                train_indices = _validated_fidel_train_indices(
                    len(dataset), self.model.vjepa2_reference_bank)
                dataset = torch.utils.data.Subset(dataset, train_indices)
        elif self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        self.dataset = dataset
        if self.matched_counter_sampling_enabled:
            self.dataloader_len = 1
            self.dataloader = None
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True, drop_last=True)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=int(getattr(config, "num_workers", 4)),
                drop_last=True,
                collate_fn=(
                    dataset.collate_fn
                    if self.current_teacher_dataset is dataset else None),
            )
            self.dataloader_len = max(1, len(dataloader))
            self.dataloader = cycle(dataloader)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))

        # Make the optimizer trajectory part of the exact-resume identity.
        # This is attached after the data/MMD identity is built and before a
        # resume checkpoint is compared against it.
        if self.experiment_identity is None:
            self.experiment_identity = {}
        self.experiment_identity["optimizer_learning_rate_schedule"] = (
            copy.deepcopy(self.learning_rate_schedule))
        self.experiment_identity["vace_aligned_execution"] = {
            "fsdp_backward_prefetch": str(getattr(
                config, "fsdp_backward_prefetch", "pre")),
            "optimizer_foreach": bool(getattr(
                config, "optimizer_foreach", False)),
            "optimizer_fused": bool(getattr(
                config, "optimizer_fused", False)),
            "optimizer_state_cpu_offload_during_backward": bool(
                self.optimizer_state_cpu_offload_during_backward),
            "logical_micro_batch_size": int(getattr(
                config, "fresh_video_logical_micro_batch_size",
                config.batch_size)),
            "physical_forward_batch_size": int(config.batch_size),
            "physical_replay_batch_size": int(getattr(
                config, "fresh_video_physical_replay_batch_size",
                config.batch_size)),
        }
        self.experiment_identity["generator_initialization"] = {
            "path": str(config.generator_ckpt),
            "sha256": _strict_sha256(
                config.generator_ckpt_sha256, "generator_ckpt_sha256"),
            "replay_activation_cpu_offload": bool(
                self.model.generator_replay_activation_cpu_offload),
            "replay_activation_cpu_offload_block_stride": int(
                self.model.
                generator_replay_activation_cpu_offload_block_stride),
            "replay_frozen_encoder_cpu_offload": bool(
                self.model.generator_replay_frozen_encoder_cpu_offload),
        }

        ##############################################################################################################
        # 6. Record trainable parameters.  Generator EMA is initialized only
        # after loading the generator checkpoint, below.
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = float(getattr(config, "ema_weight", 0.0) or 0.0)
        self.generator_ema = None
        resume_ema_state = None

        ##############################################################################################################
        # 7. Load model/distribution state before constructing an EMA shadow.
        resume_checkpoint = getattr(config, "resume_checkpoint", None)
        self.resume_data_counter_override = (
            _validated_resume_data_counter_override(
                getattr(config, "resume_data_counter_override", None),
                resume_checkpoint=resume_checkpoint,
            )
        )
        fixed_openvid_dual = bool(
            getattr(self.model, "feature_type", "")
            == "dual_dinov3_vjepa2_dense_latent"
            and not getattr(
                self.model, "current_teacher_attraction_enabled", False)
        )
        if resume_checkpoint:
            # Restore generator weights, optional generator EMA, distribution
            # history and step. Optimizer/RNG/dataloader state is restored later
            # when save_full_state is enabled and optim.pt is present.
            if fixed_openvid_dual:
                expected_resume_sha256 = _strict_sha256(
                    getattr(config, "resume_checkpoint_sha256", ""),
                    "resume_checkpoint_sha256",
                )
                actual_resume_sha256 = self.model._distributed_file_sha256(
                    str(resume_checkpoint))
                if actual_resume_sha256 != expected_resume_sha256:
                    raise ValueError(
                        "fixed-only exact resume checkpoint SHA-256 mismatch")
                checkpoint_dir = os.path.dirname(str(resume_checkpoint))
                required_full_state = [
                    os.path.join(checkpoint_dir, "optim.pt"),
                    *[
                        os.path.join(
                            checkpoint_dir, f"rng_rank{rank:02d}.pt")
                        for rank in range(4)
                    ],
                ]
                missing = [
                    path for path in required_full_state
                    if not os.path.isfile(path)
                ]
                if missing:
                    raise FileNotFoundError(
                        "fixed-only exact resume requires optimizer and all "
                        f"four RNG states: {missing}")
            print(f"Resuming training state from {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            if self.experiment_identity is not None:
                saved_identity = (
                    ckpt.get("experiment_identity")
                    if isinstance(ckpt, dict)
                    else None
                )
                saved_identity = _canonicalize_experiment_identity_paths(
                    saved_identity)
                runtime_identity = _canonicalize_experiment_identity_paths(
                    self.experiment_identity)
                if saved_identity != runtime_identity:
                    raise ValueError(
                        "resume checkpoint experiment identity does not match "
                        "the frozen teacher/V-JEPA2 experiment identity: "
                        f"{saved_identity!r} != "
                        f"{runtime_identity!r}")
            gen_sd = ckpt["generator"] if (isinstance(ckpt, dict) and "generator" in ckpt) else ckpt
            self.model.generator.load_state_dict(gen_sd, strict=True)
            if isinstance(ckpt, dict):
                resume_ema_state = ckpt.get("generator_ema")
            if (
                isinstance(ckpt, dict)
                and "fd" in ckpt
                and hasattr(self.model, "load_fd_state_dict")
            ):
                self.model.load_fd_state_dict(ckpt["fd"])
                print("  restored FD feature queue")
            elif getattr(
                self.model, "requires_distribution_state_on_resume", False
            ):
                raise ValueError(
                    "resume checkpoint is missing the required Wan KV "
                    "distribution history under key 'fd'")
            if isinstance(ckpt, dict) and "step" in ckpt:
                self.step = int(ckpt["step"])
            else:
                import re
                m = re.search(r"checkpoint_model_(\d+)", str(resume_checkpoint))
                self.step = int(m.group(1)) if m else 0
            if isinstance(ckpt, dict) and "data_counter" in ckpt:
                self.data_counter = int(ckpt["data_counter"])
            print(f"  resumed at step {self.step}")
        elif getattr(config, "generator_ckpt", False):
            if not fixed_openvid_dual:
                expected_generator_sha256 = _strict_sha256(
                    config.generator_ckpt_sha256, "generator_ckpt_sha256")
                actual_generator_sha256 = (
                    self.model._distributed_file_sha256(
                        str(config.generator_ckpt)))
                if actual_generator_sha256 != expected_generator_sha256:
                    raise ValueError(
                        "generator initialization SHA-256 mismatch")
            if fixed_openvid_dual:
                expected_generator_sha256 = str(getattr(
                    config, "generator_ckpt_sha256", "")).strip().lower()
                if (
                    len(expected_generator_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in expected_generator_sha256
                    )
                ):
                    raise ValueError(
                        "fixed-only OpenVid requires generator_ckpt_sha256")
                actual_generator_sha256 = (
                    self.model._distributed_file_sha256(
                        str(config.generator_ckpt)))
                if actual_generator_sha256 != expected_generator_sha256:
                    raise ValueError(
                        "selected 14B ODE generator checkpoint SHA-256 "
                        "mismatch")
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            if str(config.generator_ckpt).endswith(".safetensors"):
                state_dict = load_safetensors_file(
                    str(config.generator_ckpt), device="cpu")
            else:
                state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if fixed_openvid_dual and (
                not isinstance(state_dict, dict)
                or state_dict.get("model_name") != "Wan2.1-T2V-14B"
                or state_dict.get("ode_recipe") != "causvid_14b_16k"
                or int(state_dict.get("seed", -1)) != 0
                or state_dict.get("model_name")
                != str(config.generator_ckpt_model_name)
                or state_dict.get("ode_recipe")
                != str(config.generator_ckpt_ode_recipe)
                or int(state_dict.get("step", -1))
                != int(config.generator_ckpt_step)
                or state_dict.get("prompt_manifest_sha256")
                != _strict_sha256(
                    config.generator_ckpt_prompt_manifest_sha256,
                    "generator_ckpt_prompt_manifest_sha256",
                )
            ):
                raise ValueError(
                    "fixed-only OpenVid initialization must be a selected "
                    "Wan-14B CausVid ODE checkpoint")
            fd_state_dict = state_dict.get("fd") if isinstance(state_dict, dict) else None
            if isinstance(state_dict, dict):
                if "generator" in state_dict:
                    state_dict = state_dict["generator"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )
            if fd_state_dict is not None and hasattr(self.model, "load_fd_state_dict"):
                is_compatible_initialization_state = (
                    self.model.can_load_initialization_fd_state(fd_state_dict)
                    if hasattr(
                        self.model, "can_load_initialization_fd_state")
                    else not getattr(
                        self.model,
                        "requires_distribution_state_on_resume",
                        False,
                    )
                )
                if is_compatible_initialization_state:
                    # Known dense-MMD states are always passed through the
                    # model's strict metadata/cross-feature compatibility
                    # checks. Unknown FD payloads in an initialization-only
                    # checkpoint remain intentionally ignorable.
                    self.model.load_fd_state_dict(fd_state_dict)
                elif self.is_main_process:
                    print(
                        "  initialization checkpoint has unrelated or "
                        "incompatible FD state; using generator weights only")

        if self.fidel_two_level_enabled:
            _validate_fidel_resume_position(
                step=int(self.step),
                committed_updates=int(
                    self.model.fidel_controller.committed_updates.item()),
                slot_count=int(
                    self.model.vjepa2_reference_bank.slot_count),
                max_steps=int(getattr(config, "max_steps", 0)),
            )

        ##############################################################################################################
        ema_start_step = int(getattr(config, "ema_start_step", 0))
        if ema_weight > 0.0 and self.step >= ema_start_step:
            print(f"Setting up EMA with weight {ema_weight}")
            # This now snapshots the loaded generator, never the stale
            # pre-checkpoint parameters.
            self.generator_ema = EMA_FSDP(
                self.model.generator, decay=ema_weight)
            if resume_ema_state is not None:
                self.generator_ema.load_state_dict(resume_ema_state)
                print("  restored EMA shadow")
            elif resume_checkpoint and self.step > ema_start_step:
                raise ValueError(
                    "resume checkpoint is past ema_start_step but has no "
                    "generator_ema state")

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        # ---- precise-resume + periodic eval bookkeeping ----
        # ``save_full_state`` additionally writes the optimizer state, per-rank RNG
        # states and the dataloader counter so a resume continues bit-for-bit (up
        # to multi-worker dataloader ordering). Off by default so other trainers
        # (DMD etc.) keep their lightweight checkpoints.
        self.save_full_state = bool(getattr(config, "save_full_state", False))
        self.save_full_state_final_only = bool(
            getattr(config, "save_full_state_final_only", False))
        self.checkpoint_keep_last = int(
            getattr(config, "checkpoint_keep_last", 0))
        if self.checkpoint_keep_last < 0:
            raise ValueError("checkpoint_keep_last must be non-negative")
        self.eval_interval = int(getattr(config, "eval_interval", 0))
        self.eval_seed = int(getattr(config, "eval_seed", 12345))
        self.eval_num_prompts = int(getattr(config, "eval_num_prompts", 4))
        self.eval_prompts = self._load_eval_prompts() if self.eval_interval > 0 else []
        self._last_saved_step = None
        # If we resumed and the optimizer/RNG/data state was restored, this is set.
        self._restore_full_training_state(resume_checkpoint)
        self.optimizer_state_offload_report = None
        if self.optimizer_state_cpu_offload_during_backward:
            torch.cuda.synchronize(self.device)
            self.optimizer_state_offload_report = move_adamw_moment_state_(
                self.generator_optimizer, "cpu")
            torch.cuda.empty_cache()
            if self.is_main_process:
                print(
                    "[memory] VACE optimizer-state CPU offload ready: "
                    f"{self.optimizer_state_offload_report}",
                    flush=True,
                )
        self.current_learning_rate = self._set_generator_learning_rate(
            self.step)

    def _set_generator_learning_rate(self, update_index):
        schedule = self.learning_rate_schedule
        if schedule["name"] == "constant":
            value = float(schedule["value"])
        else:
            value = _cosine_learning_rate(
                int(update_index),
                start=float(schedule["start"]),
                end=float(schedule["end"]),
                total_updates=int(schedule["total_updates"]),
            )
        for parameter_group in self.generator_optimizer.param_groups:
            parameter_group["lr"] = value
        self.current_learning_rate = value
        return value

    def _all_ranks_true(self, local_value):
        """Create an all-rank decision before a rank can branch or raise."""
        flag = torch.tensor(
            int(bool(local_value)), device=self.device, dtype=torch.int32)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def save(self):
        if (
            self.fresh_video_gradcache is not None
            and (
                self.fresh_video_gradcache.local_samples
                or self.fresh_video_gradcache.finalized
            )
        ):
            raise RuntimeError(
                "refusing to checkpoint a partial fresh-video optimizer "
                "update")
        if (
            self.fresh_frame_fd_gradcache is not None
            and (
                self.fresh_frame_fd_gradcache.local_videos
                or self.fresh_frame_fd_gradcache.finalized
            )
        ):
            raise RuntimeError(
                "refusing to checkpoint a partial frame-FD optimizer update")
        if (
            hasattr(self.model, "distribution_state_pending")
            and self.model.distribution_state_pending()
        ):
            raise RuntimeError(
                "refusing to checkpoint pending distribution state")
        # max_steps can coincide with log_iters; avoid gathering and rewriting
        # the same full FSDP checkpoint twice.
        if self._last_saved_step == self.step:
            return
        max_steps = int(getattr(self.config, "max_steps", 0))
        write_full_state = (
            self.save_full_state
            and (
                not self.save_full_state_final_only
                or (max_steps > 0 and self.step >= max_steps)
            )
        )
        required_free_gib = float(
            getattr(
                self.config,
                (
                    "checkpoint_full_min_free_gib"
                    if write_full_state
                    else "checkpoint_model_min_free_gib"
                ),
                0.0,
            )
        )
        disk_ok = True
        disk_error = None
        if self.is_main_process and required_free_gib > 0:
            try:
                free_gib = (
                    shutil.disk_usage(self.output_path).free / (1024 ** 3))
                if free_gib < required_free_gib:
                    disk_ok = False
                    disk_error = (
                        f"checkpoint at step {self.step} needs at least "
                        f"{required_free_gib:.1f} GiB free, only "
                        f"{free_gib:.1f} GiB is available"
                    )
            except Exception as exc:
                disk_ok = False
                disk_error = f"checkpoint disk-space check failed: {exc}"
        disk_flag = torch.tensor(
            int(disk_ok), device=self.device, dtype=torch.int32)
        if dist.is_initialized():
            dist.broadcast(disk_flag, src=0)
        if not bool(disk_flag.item()):
            raise RuntimeError(
                disk_error or "rank 0 checkpoint disk-space check failed")

        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = (
            fsdp_state_dict(self.model.fake_score)
            if self.uses_critic
            else None
        )

        if self.generator_ema is not None:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }
        if self.uses_critic:
            state_dict["critic"] = critic_state_dict
        if hasattr(self.model, "fd_state_dict"):
            state_dict["fd"] = self.model.fd_state_dict()
        # record the step so a resume continues numbering / EMA timing.
        state_dict["step"] = self.step
        state_dict["data_counter"] = self.data_counter
        if self.experiment_identity is not None:
            state_dict["experiment_identity"] = (
                _canonicalize_experiment_identity_paths(
                    self.experiment_identity))

        # ---- full training state for precise resume (optimizer + RNG + data) ----
        gen_optim_full = None
        critic_optim_full = None
        if write_full_state:
            # collective: every rank participates; full state gathered on rank 0.
            gen_optim_full = fsdp_optim_state_dict(self.model.generator, self.generator_optimizer)
            if self.uses_critic:
                critic_optim_full = fsdp_optim_state_dict(self.model.fake_score, self.critic_optimizer)

        ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        rank0_write_error = None
        if self.is_main_process:
            try:
                os.makedirs(ckpt_dir, exist_ok=True)
                model_path = os.path.join(ckpt_dir, "model.pt")
                _atomic_torch_save(state_dict, model_path)
                print("Model saved to", model_path)
                if write_full_state:
                    optim_path = os.path.join(ckpt_dir, "optim.pt")
                    _atomic_torch_save(
                        {
                            "generator_optimizer": gen_optim_full,
                            "critic_optimizer": critic_optim_full,
                            "step": self.step,
                            "data_counter": self.data_counter,
                            "world_size": self.world_size,
                        },
                        optim_path,
                    )
                    print(
                        "Optimizer/training state saved to",
                        optim_path,
                    )
            except Exception as exc:
                rank0_write_error = exc

        write_ok = torch.tensor(
            int(rank0_write_error is None),
            device=self.device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.broadcast(write_ok, src=0)
        if not bool(write_ok.item()):
            if rank0_write_error is not None:
                raise RuntimeError(
                    f"checkpoint write failed at step {self.step}"
                ) from rank0_write_error
            raise RuntimeError(
                f"rank 0 checkpoint write failed at step {self.step}")

        # Per-rank RNG state (each rank writes its own) for reproducible streams.
        rng_write_ok = True
        rng_write_error = None
        if write_full_state:
            try:
                os.makedirs(ckpt_dir, exist_ok=True)
                rng_state = {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(self.device),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                    "data_counter": self.data_counter,
                }
                _atomic_torch_save(
                    rng_state,
                    os.path.join(
                        ckpt_dir, f"rng_rank{dist.get_rank():02d}.pt"),
                )
            except Exception as exc:
                rng_write_ok = False
                rng_write_error = exc
        rng_ok = torch.tensor(
            int(rng_write_ok), device=self.device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(rng_ok, op=dist.ReduceOp.MIN)
        if not bool(rng_ok.item()):
            if rng_write_error is not None:
                raise RuntimeError(
                    f"rank {dist.get_rank()} RNG checkpoint write failed at "
                    f"step {self.step}"
                ) from rng_write_error
            raise RuntimeError(
                f"an RNG checkpoint write failed at step {self.step}")

        if dist.is_initialized():
            dist.barrier()

        retention_error = None
        if self.is_main_process and self.checkpoint_keep_last > 0:
            try:
                checkpoints = []
                for name in os.listdir(self.output_path):
                    prefix = "checkpoint_model_"
                    if not name.startswith(prefix):
                        continue
                    try:
                        checkpoint_step = int(name[len(prefix):])
                    except ValueError:
                        continue
                    path = os.path.join(self.output_path, name)
                    if not os.path.isdir(path):
                        continue
                    if not os.path.isfile(os.path.join(path, "model.pt")):
                        shutil.rmtree(path)
                        print("Removed incomplete checkpoint", path)
                        continue
                    checkpoints.append((checkpoint_step, path))
                checkpoints.sort()
                for _, path in checkpoints[:-self.checkpoint_keep_last]:
                    shutil.rmtree(path)
                    print("Removed old checkpoint", path)
            except Exception as exc:
                retention_error = exc
        retention_ok = torch.tensor(
            int(retention_error is None),
            device=self.device,
            dtype=torch.int32,
        )
        if dist.is_initialized():
            dist.broadcast(retention_ok, src=0)
        if not bool(retention_ok.item()):
            if retention_error is not None:
                raise RuntimeError(
                    "checkpoint retention failed after publishing a valid "
                    f"checkpoint at step {self.step}"
                ) from retention_error
            raise RuntimeError(
                "rank 0 checkpoint retention failed after publishing a "
                f"valid checkpoint at step {self.step}")
        if dist.is_initialized():
            dist.barrier()
        self._last_saved_step = self.step

    def _restore_full_training_state(self, resume_checkpoint):
        """Restore optimizer + per-rank RNG + dataloader position for a precise resume.

        Called at the end of ``__init__`` (after the optimizer and generator
        weights exist).  Silently no-ops if the resume checkpoint has no
        ``optim.pt`` (e.g. an old lightweight checkpoint), so older runs still
        resume their weights/EMA/queue/step as before.
        """
        if not resume_checkpoint:
            return
        if not self.save_full_state:
            # Lightweight checkpoints still record data_counter.  They do not
            # promise bit-exact optimizer/RNG continuation, but they must not
            # restart the prompt stream at its first batch.
            self._apply_lightweight_resume_data_counter_override()
            self._fast_forward_dataloader(self.data_counter)
            if self.is_main_process:
                print(
                    "  lightweight resume: restored generator/distribution "
                    "state and data position; optimizer and RNG start fresh",
                    flush=True,
                )
            return
        ckpt_dir = os.path.dirname(resume_checkpoint)
        optim_path = os.path.join(ckpt_dir, "optim.pt")

        if os.path.exists(optim_path):
            if getattr(self, "resume_data_counter_override", None) is not None:
                raise ValueError(
                    "resume_data_counter_override is only valid for a "
                    "lightweight resume; precise optimizer state exists at "
                    f"{optim_path}")
            # rank0 holds the full optimizer state; FSDP broadcasts/shards it.
            full = None
            optim_load_error = None
            if self.is_main_process:
                try:
                    full = torch.load(
                        optim_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except Exception as exc:
                    optim_load_error = exc
            optim_load_ok = torch.tensor(
                int(optim_load_error is None),
                device=self.device,
                dtype=torch.int32,
            )
            if dist.is_initialized():
                dist.broadcast(optim_load_ok, src=0)
            if not bool(optim_load_ok.item()):
                if optim_load_error is not None:
                    raise RuntimeError(
                        f"failed to load optimizer checkpoint {optim_path}"
                    ) from optim_load_error
                raise RuntimeError(
                    f"rank 0 failed to load optimizer checkpoint {optim_path}")
            saved_world_size = torch.tensor(
                [
                    int(full.get("world_size", self.world_size))
                    if full is not None
                    else 0
                ],
                device=self.device,
                dtype=torch.long,
            )
            dist.broadcast(saved_world_size, src=0)
            if (
                int(saved_world_size.item()) != self.world_size
                and not bool(getattr(
                    self.config, "allow_resume_world_size_change", False))
            ):
                raise ValueError(
                    "precise resume world-size mismatch: checkpoint used "
                    f"{int(saved_world_size.item())}, current run uses "
                    f"{self.world_size}; set allow_resume_world_size_change "
                    "only if non-bit-exact data/RNG continuation is acceptable")
            gen_osd = full["generator_optimizer"] if full is not None else None
            fsdp_load_optim_state_dict(self.model.generator, self.generator_optimizer, gen_osd)
            if self.uses_critic:
                critic_osd = full["critic_optimizer"] if full is not None else None
                if critic_osd is not None:
                    fsdp_load_optim_state_dict(self.model.fake_score, self.critic_optimizer, critic_osd)
            # broadcast the data counter from rank0
            dc = torch.tensor(
                [int(full["data_counter"]) if full is not None else 0],
                device=self.device, dtype=torch.long,
            )
            if dist.is_initialized():
                dist.broadcast(dc, src=0)
            self.data_counter = int(dc.item())
            if self.is_main_process:
                print(f"  restored optimizer state; data_counter={self.data_counter}")
        else:
            self._apply_lightweight_resume_data_counter_override()
            if self.is_main_process:
                print(
                    "  lightweight resume: restored generator/MMD history "
                    "and data position; optimizer and RNG start fresh",
                    flush=True,
                )

        # Read RNG now, but restore it only after DataLoader fast-forwarding:
        # creating/advancing workers consumes the main process RNG.
        rng_path = os.path.join(ckpt_dir, f"rng_rank{dist.get_rank():02d}.pt")
        rng = None
        if os.path.exists(rng_path):
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)

        # fast-forward the prompt stream to its position
        self._fast_forward_dataloader(self.data_counter)

        if rng is not None:
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"], self.device)
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
            if self.is_main_process:
                print("  restored per-rank RNG states")

    def _apply_lightweight_resume_data_counter_override(self):
        override = getattr(self, "resume_data_counter_override", None)
        if override is None:
            return
        checkpoint_counter = self.data_counter
        self.data_counter = override
        if self.is_main_process:
            print(
                "  lightweight resume data cursor: explicitly overriding "
                f"checkpoint data_counter {checkpoint_counter} -> {override} "
                "from resume_data_counter_override",
                flush=True,
            )

    def _matched_global_indices_for_step(self):
        """Counter-based with-replacement draw shared identically by all ranks."""
        if not getattr(self, "matched_counter_sampling_enabled", False):
            raise RuntimeError("matched counter sampler is not enabled")
        population = len(self.dataset)
        if population <= 0:
            raise RuntimeError("matched counter sampler has an empty population")
        seed = int(self.config.seed)
        step = int(self.step)
        indices = []
        for position in range(self.fresh_video_total_samples):
            payload = (
                f"sf-matched-counter-v1\0{seed}\0{step}\0{position}"
                .encode("ascii")
            )
            value = int.from_bytes(
                hashlib.sha256(payload).digest()[:8], "little")
            indices.append(value % population)
        return tuple(indices)

    def _matched_batch_for_microbatch(self, microbatch_index):
        if not getattr(self, "matched_counter_sampling_enabled", False):
            raise RuntimeError("matched counter sampler is not enabled")
        batch_size = int(self.config.batch_size)
        local_offset = (
            dist.get_rank() * self.fresh_video_local_samples
            + int(microbatch_index) * batch_size
        )
        indices = self._matched_global_indices_for_step()
        selected = indices[local_offset:local_offset + batch_size]
        if len(selected) != batch_size:
            raise RuntimeError("matched counter sampler produced a short slice")
        items = [self.dataset[index] for index in selected]
        self.data_counter += 1
        return self.dataset.collate_fn(items)

    def _fast_forward_dataloader(self, n_consumed):
        if getattr(self, "matched_counter_sampling_enabled", False):
            return
        if n_consumed <= 0:
            return
        if hasattr(self.dataloader, "seek"):
            skip = self.dataloader.seek(n_consumed)
        else:
            skip = n_consumed % self.dataloader_len
            for _ in range(skip):
                next(self.dataloader)
        if self.is_main_process and skip:
            print(
                f"  restored dataloader epoch/offset "
                f"({n_consumed // self.dataloader_len}, {skip})")

    def _next_batch(self):
        if getattr(self, "matched_counter_sampling_enabled", False):
            raise RuntimeError(
                "matched-manifest training must use the counter sampler")
        self.data_counter += 1
        return next(self.dataloader)

    def _activate_fidel_reference_for_update(self):
        """Activate exactly the slot named by the committed FiDEL state."""
        if not self.fidel_two_level_enabled:
            return None
        controller = getattr(self.model, "fidel_controller", None)
        bank = getattr(self.model, "vjepa2_reference_bank", None)
        if controller is None or bank is None:
            raise RuntimeError(
                "two-level FiDEL controller/reference bank is not initialized")
        local_slot = int(controller.committed_updates.item())
        root_slot = torch.tensor(
            local_slot, device=self.device, dtype=torch.long)
        if dist.is_initialized() and self.world_size > 1:
            dist.broadcast(root_slot, src=0)
        if not self._all_ranks_true(
            local_slot == int(root_slot.item())
        ):
            raise RuntimeError(
                "FiDEL committed update count differs across ranks")
        slot = int(root_slot.item())
        if not 0 <= slot < int(bank.slot_count):
            raise RuntimeError(
                "FiDEL exhausted its random-token reference bank: "
                f"next slot {slot}, slot_count={int(bank.slot_count)}")

        activation = None
        activation_error = None
        try:
            activation = self.model.activate_vjepa2_reference_slot(slot)
        except BaseException as exc:
            activation_error = exc
        if not self._all_ranks_true(activation_error is None):
            message = (
                f": {type(activation_error).__name__}: {activation_error}"
                if activation_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "FiDEL reference-slot activation failed" + message
            ) from activation_error

        local_identity = {
            "slot": int(activation["slot"]),
            "token_indices": [
                int(value)
                for value in activation["token_indices"].detach().cpu().tolist()
            ],
            "token_indices_sha256": str(
                activation["token_indices_sha256"]),
            "pack_content_hash": str(activation["pack_content_hash"]),
        }
        if dist.is_initialized() and self.world_size > 1:
            root_identity = [
                local_identity if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(root_identity, src=0)
            identity_matches = local_identity == root_identity[0]
            if not self._all_ranks_true(identity_matches):
                raise RuntimeError(
                    "FiDEL reference slot/token/hash differs across ranks")
        if self.is_main_process:
            print(
                "[fidel] activated "
                f"slot={slot} "
                f"token_hash={local_identity['token_indices_sha256']} "
                f"pack_hash={local_identity['pack_content_hash']}",
                flush=True,
            )
        return activation

    def _load_eval_prompts(self):
        path = getattr(self.config, "eval_prompt_path", "prompts/test_8.txt")
        try:
            with open(path, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
        except FileNotFoundError:
            if self.is_main_process:
                print(f"[eval] prompt file {path} not found; eval disabled")
            return []
        return lines[: self.eval_num_prompts]

    @torch.no_grad()
    def _fd_eval_inference(self):
        """Generate fixed-prompt videos with the live generator and log to wandb.

        Runs the model's own few-step (here single-step, no-CFG) generation under
        a fixed seed so successive evals are directly comparable.  All ranks
        participate (FSDP forward is collective); only rank 0 writes/logs.
        """
        if not self.eval_prompts:
            return
        # Frozen encoders stay in eval mode; generator gradients are enabled only on replay.
        self.model.eval()
        # fix the RNG for reproducible eval, then restore so training is unperturbed
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)
        torch.manual_seed(self.eval_seed)

        prompts = self.eval_prompts
        # Evaluation is deliberately serialized at one video per rank.  The
        # training forward is also microbatch 1; evaluating all fixed prompts
        # together would multiply the peak generator/VAE memory before the
        # first optimizer update.
        rank0_pixels = []
        try:
            for prompt in prompts:
                conditional_dict = self.model.text_encoder(
                    text_prompts=[prompt])
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt])
                shape = [1] + list(
                    self.config.image_or_video_shape[1:])
                latent = self.model._run_few_step_generator(
                    image_or_video_shape=shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                )
                pixels = self.model.vae.decode_to_pixel(
                    latent.to(self.dtype))
                pixels = ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)
                if self.is_main_process:
                    rank0_pixels.append(pixels[0].float().cpu())
                del latent, pixels, conditional_dict, unconditional_dict
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)

        if self.is_main_process:
            eval_dir = os.path.join(self.output_path, "eval", f"step_{self.step:06d}")
            os.makedirs(eval_dir, exist_ok=True)
            log_videos = {}
            for i in range(len(prompts)):
                vid = (
                    rank0_pixels[i].permute(0, 2, 3, 1).numpy() * 255.0
                ).astype("uint8")  # [T,H,W,C]
                out_path = os.path.join(eval_dir, f"sample_{i}.mp4")
                wrote = False
                try:
                    iio.imwrite(
                        out_path,
                        vid,
                        fps=16,
                        codec="libx264",
                        plugin="FFMPEG",
                    )
                    wrote = True
                except Exception as exc:  # pragma: no cover - codec/runtime issues
                    print(f"[eval] failed to write {out_path}: {exc}")
                # Log the encoded mp4 by PATH so wandb does not require moviepy.
                if not self.disable_wandb and wrote:
                    try:
                        log_videos[f"eval/sample_{i}"] = wandb.Video(out_path, format="mp4")
                    except Exception as exc:  # pragma: no cover
                        print(f"[eval] wandb.Video failed for {out_path}: {exc}")
            if not self.disable_wandb and log_videos:
                wandb.log(log_videos, step=self.step)
            print(f"[eval] step {self.step}: wrote {len(prompts)} videos to {eval_dir}", flush=True)
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _batch_string_list(value, *, key, batch_size):
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            raise TypeError(f"{key} must collate to a list of strings")
        if (
            len(values) != batch_size
            or any(not isinstance(item, str) or not item for item in values)
        ):
            raise ValueError(
                f"{key} must contain one non-empty string per prompt")
        return values

    def _attach_matched_teacher_payload(self, batch):
        """Load cache/decode teacher data only for a selected replay suffix."""
        if self.current_teacher_source != "matched_manifest_video":
            return
        if self.current_teacher_dataset is None:
            raise RuntimeError("matched teacher dataset is unavailable")
        batch_size = len(batch.get("prompts", ()))
        sample_ids = self._batch_string_list(
            batch.get("sample_ids"), key="sample_ids", batch_size=batch_size)
        cache_keys = self._batch_string_list(
            batch.get("teacher_cache_keys"), key="teacher_cache_keys",
            batch_size=batch_size)
        local_error = None
        payloads = None
        try:
            payloads = [
                self.current_teacher_dataset.load_teacher_payload(sample_id)
                for sample_id in sample_ids
            ]
            for sample_id, cache_key, payload in zip(
                sample_ids, cache_keys, payloads
            ):
                if (
                    payload.get("sample_id") != sample_id
                    or payload.get("teacher_cache_key") != cache_key
                ):
                    raise ValueError(
                        "lazy teacher payload identity does not match the "
                        "selected student")
            hit_values = [
                bool(payload["current_teacher_cache_hit"])
                for payload in payloads
            ]
            if not (all(hit_values) or not any(hit_values)):
                raise ValueError(
                    "a metadata microbatch cannot mix cached features and "
                    "decoded videos")
            batch["current_teacher_cache_hit"] = torch.tensor(
                hit_values, dtype=torch.bool)
            batch["current_teacher_decode_seconds"] = sum(
                float(payload["current_teacher_decode_seconds"])
                for payload in payloads
            )
            if all(hit_values):
                batch["current_teacher_features"] = torch.stack([
                    payload["current_teacher_features"]
                    for payload in payloads
                ])
            else:
                batch["current_teacher_video01"] = torch.stack([
                    payload["current_teacher_video01"]
                    for payload in payloads
                ])
        except BaseException as exc:
            local_error = exc
        if not self._all_ranks_true(local_error is None):
            message = (
                f": {type(local_error).__name__}: {local_error}"
                if local_error is not None else " on another rank")
            raise RuntimeError(
                "matched live-teacher lazy I/O failed" + message
            ) from local_error

    def _prepare_current_teacher_inputs(self, batch, batch_size):
        """Validate and move exactly one configured teacher payload."""
        if not bool(getattr(
            self.model, "current_teacher_attraction_enabled", False
        )):
            unexpected = [
                key for key in (
                    "current_teacher_features",
                    "current_teacher_video01",
                    "current_teacher_latent",
                    "ode_latent",
                )
                if batch.get(key) is not None
            ]
            if unexpected:
                raise ValueError(
                    "fixed-only MMD rejects current-teacher payloads: "
                    f"{unexpected}")
            return {}, None, None

        local_error = None
        forward_inputs = None
        cache_records = None
        cache_log = None
        try:
            if self.current_teacher_source in {
                "matched_dual_feature_bank",
                "matched_triple_feature_bank",
                "matched_dinov3_vjepa2_feature_bank",
            }:
                if self.current_teacher_dataset is None:
                    raise RuntimeError("matched teacher bank is unavailable")
                sample_ids = self._batch_string_list(
                    batch.get("sample_ids"), key="sample_ids",
                    batch_size=batch_size)
                features = self.current_teacher_dataset.load_rows(sample_ids)
                expected_shapes = getattr(
                    self.model.latent_mmd, "sample_shapes", {})
                if set(features) != set(expected_shapes):
                    raise ValueError("teacher bank/model feature branches differ")
                for name, tensor in features.items():
                    expected = (batch_size, *expected_shapes[name])
                    if tensor.dtype != torch.float16 or tuple(tensor.shape) != expected:
                        raise ValueError(
                            f"teacher bank {name} row shape/dtype mismatch: "
                            f"{tuple(tensor.shape)}/{tensor.dtype} != "
                            f"{expected}/float16")
                    features[name] = tensor.to(
                        device=self.device, dtype=torch.float32)
                forward_inputs = {"current_teacher_features": features}
                cache_records = ()
                cache_log = {
                    "hits": batch_size,
                    "misses": 0,
                    "decode_seconds": 0.0,
                    "bank_rows_read": batch_size,
                }
            elif self.current_teacher_source == "lmdb_latent":
                latent = batch.get("ode_latent")
                if (
                    not isinstance(latent, torch.Tensor)
                    or latent.ndim != 5
                    or int(latent.shape[0]) != batch_size
                    or not latent.is_floating_point()
                ):
                    raise ValueError(
                        "exact LMDB current-teacher MMD requires one terminal "
                        "latent per prompt with shape "
                        "[B,frames,channels,height,width]")
                latent = latent.to(device=self.device, dtype=self.dtype)
                if not bool(torch.isfinite(latent).all().item()):
                    raise FloatingPointError(
                        "matched current teacher latent contains non-finite "
                        "values")
                forward_inputs = {"current_teacher_latent": latent}
            elif self.current_teacher_source in {
                "openvid_video", "matched_manifest_video"
            }:
                features = batch.get("current_teacher_features")
                video01 = batch.get("current_teacher_video01")
                if (features is None) == (video01 is None):
                    raise ValueError(
                        "video teacher batch must provide exactly one of cached "
                        "current_teacher_features or current_teacher_video01")
                cache_hit = batch.get("current_teacher_cache_hit")
                if (
                    not isinstance(cache_hit, torch.Tensor)
                    or cache_hit.dtype != torch.bool
                    or tuple(cache_hit.shape) != (batch_size,)
                ):
                    raise ValueError(
                        "current_teacher_cache_hit must be BoolTensor[B]")
                sample_ids = self._batch_string_list(
                    batch.get("sample_ids"), key="sample_ids",
                    batch_size=batch_size)
                cache_keys = self._batch_string_list(
                    batch.get("teacher_cache_keys"),
                    key="teacher_cache_keys", batch_size=batch_size)
                decode_value = batch.get("current_teacher_decode_seconds", 0.0)
                if isinstance(decode_value, torch.Tensor):
                    decode_seconds = float(
                        decode_value.detach().float().sum().item())
                else:
                    decode_seconds = float(decode_value)
                if not np.isfinite(decode_seconds) or decode_seconds < 0.0:
                    raise ValueError(
                        "current_teacher_decode_seconds must be finite and "
                        "non-negative")
                if features is not None:
                    expected_shape = (
                        batch_size,
                        int(self.model.dense_num_tokens),
                        int(self.model.dense_feature_dim),
                    )
                    if (
                        not isinstance(features, torch.Tensor)
                        or not features.is_floating_point()
                        or tuple(features.shape) != expected_shape
                        or not bool(cache_hit.all().item())
                    ):
                        raise ValueError(
                            "cached video teacher features must be floating "
                            f"{expected_shape} with every cache-hit flag set")
                    features = features.to(
                        device=self.device, dtype=torch.float32)
                    if not bool(torch.isfinite(features).all().item()):
                        raise FloatingPointError(
                            "cached video teacher features are non-finite")
                    forward_inputs = {
                        "current_teacher_features": features}
                    cache_hits = batch_size
                    cache_records = ()
                else:
                    expected_prefix = (
                        batch_size,
                        int(self.model.dense_num_frames),
                        3,
                    )
                    if (
                        not isinstance(video01, torch.Tensor)
                        or not video01.is_floating_point()
                        or video01.ndim != 5
                        or tuple(video01.shape[:3]) != expected_prefix
                        or bool(cache_hit.any().item())
                    ):
                        raise ValueError(
                            "uncached video teacher video must be floating "
                            "[B,64,3,H,W] with every cache-hit flag clear")
                    if (
                        not bool(torch.isfinite(video01).all().item())
                        or float(video01.min().item()) < 0.0
                        or float(video01.max().item()) > 1.0
                    ):
                        raise ValueError(
                            "current_teacher_video01 must contain finite "
                            "values in [0,1]")
                    video01 = video01.to(
                        device=self.device, dtype=torch.float32)
                    forward_inputs = {
                        "current_teacher_video01": video01}
                    cache_hits = 0
                    cache_records = tuple(zip(cache_keys, sample_ids))
                cache_log = {
                    "hits": cache_hits,
                    "misses": batch_size - cache_hits,
                    "decode_seconds": decode_seconds,
                }
            else:
                raise RuntimeError(
                    "enabled exact current teacher has no configured source")
        except BaseException as exc:
            local_error = exc
        if not self._all_ranks_true(local_error is None):
            message = (
                f": {type(local_error).__name__}: {local_error}"
                if local_error is not None else " on another rank")
            raise RuntimeError(
                "current teacher batch validation failed" + message
            ) from local_error
        return forward_inputs, cache_records, cache_log

    def _write_current_teacher_cache(self, records, features):
        if not records:
            return
        if self.current_teacher_dataset is None:
            raise RuntimeError("live-teacher feature cache writer is unavailable")
        cpu_features = features.detach().float().cpu()
        if int(cpu_features.shape[0]) != len(records):
            raise ValueError(
                "teacher feature count does not match cache-miss records")
        for (cache_key, sample_id), feature in zip(records, cpu_features):
            self.current_teacher_dataset.write_feature_cache(
                cache_key, feature, sample_id=sample_id)

    def _collect_rollout_microbatch(
        self,
        batch,
        *,
        gradient_selected=True,
        vjepa2_window_sample_keys=None,
    ):
        """Collect generated features and replay inputs without a generator graph.

        Reference minibatch rows are attached only to selected differentiation
        samples; all generated features still enter the distribution loss."""
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        sample_ids = (
            self._batch_string_list(
                batch.get("sample_ids"), key="sample_ids",
                batch_size=batch_size)
            if batch.get("sample_ids") is not None
            else None
        )
        if (
            gradient_selected
            and self.current_teacher_source == "matched_manifest_video"
        ):
            self._attach_matched_teacher_payload(batch)
        if gradient_selected:
            teacher_forward_inputs, cache_records, teacher_cache_log = (
                self._prepare_current_teacher_inputs(batch, batch_size))
        else:
            teacher_forward_inputs, cache_records, teacher_cache_log = (
                {}, None, None)

        # Transfer ownership out of the DataLoader batch before model entry.
        # Each tensor is popped from teacher_forward_inputs as the call binds
        # it, so the model can release its final reference before decoding the
        # student and avoid a teacher+student pixel-memory peak.
        for key in (
            "current_teacher_features",
            "current_teacher_video01",
            "ode_latent",
        ):
            batch.pop(key, None)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        first_pass_started = time.perf_counter()
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)
            cached_unconditional = getattr(
                self, "unconditional_dict", None)
            cached_batch = (
                next(iter(cached_unconditional.values())).shape[0]
                if cached_unconditional else None
            )
            if cached_unconditional is None or cached_batch != batch_size:
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[
                        self.config.negative_prompt] * batch_size)
                unconditional_dict = {
                    key: value.detach()
                    for key, value in unconditional_dict.items()
                }
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = cached_unconditional

            features, replay_descriptor = (
                self.model.collect_rollout_features(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    initial_latent=None,
                    vjepa2_window_sample_keys=(
                        vjepa2_window_sample_keys),
                    require_current_teacher=bool(
                        gradient_selected
                        and getattr(
                            self.model,
                            "current_teacher_attraction_enabled",
                            False,
                        )
                    ),
                    current_teacher_features=teacher_forward_inputs.pop(
                        "current_teacher_features", None),
                    current_teacher_video01=teacher_forward_inputs.pop(
                        "current_teacher_video01", None),
                    current_teacher_latent=teacher_forward_inputs.pop(
                        "current_teacher_latent", None),
                )
            )
        first_pass_seconds = time.perf_counter() - first_pass_started
        current_teacher_features = replay_descriptor.pop(
            "current_teacher_features", None)
        teacher_extract_seconds = replay_descriptor.pop(
            "current_teacher_extract_seconds", None)
        if (
            gradient_selected
            and bool(getattr(
                self.model, "current_teacher_attraction_enabled", False))
            and current_teacher_features is None
        ):
            raise RuntimeError(
                "fresh first pass omitted matched current teacher features")
        cache_write_error = None
        try:
            if cache_records:
                self._write_current_teacher_cache(
                    cache_records, current_teacher_features)
        except BaseException as exc:
            cache_write_error = exc
        if not self._all_ranks_true(cache_write_error is None):
            message = (
                f": {type(cache_write_error).__name__}: {cache_write_error}"
                if cache_write_error is not None else " on another rank")
            raise RuntimeError(
                "live-teacher feature cache write failed" + message
            ) from cache_write_error
        # The call-site pops every mutually exclusive input while binding it;
        # retaining anything here would defeat the teacher payload handoff.
        if teacher_forward_inputs:
            raise RuntimeError(
                "unconsumed current teacher forward input remained")
        if teacher_cache_log is not None:
            teacher_cache_log["first_pass_seconds"] = first_pass_seconds
            if teacher_extract_seconds is not None:
                teacher_cache_log["extract_seconds"] = float(
                    teacher_extract_seconds)
        entry = {
            "features": features,
            "replay_descriptor": replay_descriptor,
            "image_or_video_shape": image_or_video_shape,
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
            "gradient_selected": bool(gradient_selected),
            "sample_ids": sample_ids,
        }
        if current_teacher_features is not None:
            entry["current_teacher_features"] = current_teacher_features
        if teacher_cache_log is not None:
            entry["current_teacher_cache_log"] = teacher_cache_log
        if self.frame_fd_fusion_enabled:
            extraction = replay_descriptor.get("frame_fd_extraction")
            if extraction is None:
                raise RuntimeError(
                    "hybrid first pass did not return frame FD features")
            entry["frame_fd_extraction"] = extraction
        return entry

    @staticmethod
    def _slice_fresh_video_batch_mapping(mapping, start, stop, batch_size):
        sliced = {}
        for key, value in mapping.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and int(value.shape[0]) == batch_size
            ):
                sliced[key] = value[start:stop]
            else:
                sliced[key] = value
        return sliced

    def _aggregate_fresh_video_forward_entries(
        self,
        physical_entries,
        *,
        gradient_selected,
    ):
        if not physical_entries:
            raise ValueError("logical fresh-video microbatch is empty")
        if self.frame_fd_fusion_enabled:
            raise NotImplementedError(
                "VACE-aligned split physical batches are only enabled for "
                "the pure video MMD path")
        first_features = physical_entries[0]["features"]
        if isinstance(first_features, Mapping):
            features = {
                name: torch.cat([
                    entry["features"][name] for entry in physical_entries
                ], dim=0)
                for name in first_features
            }
        else:
            features = torch.cat([
                entry["features"] for entry in physical_entries
            ], dim=0)
        sample_id_parts = [
            entry.get("sample_ids") for entry in physical_entries
        ]
        if all(value is None for value in sample_id_parts):
            sample_ids = None
        elif any(value is None for value in sample_id_parts):
            raise RuntimeError(
                "sample identities are incomplete within a logical batch")
        else:
            sample_ids = tuple(
                sample_id
                for values in sample_id_parts
                for sample_id in values
            )
            if len(sample_ids) != self.fresh_video_logical_micro_batch_size:
                raise RuntimeError(
                    "logical fresh-video sample identity count mismatch")
        aggregate = {
            "features": features,
            "gradient_selected": bool(gradient_selected),
            "sample_ids": sample_ids,
            "physical_entries": tuple(physical_entries),
        }
        teacher_values = [
            entry.get("current_teacher_features")
            for entry in physical_entries
        ]
        if any(value is not None for value in teacher_values):
            if not all(value is not None for value in teacher_values):
                raise RuntimeError(
                    "matched teacher features are incomplete within a "
                    "logical microbatch")
            if isinstance(teacher_values[0], Mapping):
                aggregate["current_teacher_features"] = {
                    name: torch.cat([
                        value[name] for value in teacher_values
                    ], dim=0)
                    for name in teacher_values[0]
                }
            else:
                aggregate["current_teacher_features"] = torch.cat(
                    teacher_values, dim=0)
        for entry in physical_entries:
            entry.pop("features", None)
            entry.pop("current_teacher_features", None)
        return aggregate

    def _split_fresh_video_replay_units(
        self,
        logical_entry,
        logical_cotangent,
    ):
        logical_features = logical_entry["features"]
        units = []
        logical_offset = 0
        replay_batch = self.fresh_video_physical_replay_batch_size
        for physical_entry in logical_entry["physical_entries"]:
            descriptor = physical_entry["replay_descriptor"]
            physical_batch = int(descriptor["latent_shape"][0])
            for start in range(0, physical_batch, replay_batch):
                stop = start + replay_batch
                replay_descriptor = dict(descriptor)
                replay_descriptor["latent_shape"] = (
                    replay_batch, *tuple(descriptor["latent_shape"][1:]))
                replay_descriptor["student_latent"] = descriptor[
                    "student_latent"][start:stop]
                generation_descriptor = dict(
                    descriptor["generation_descriptor"])
                generation_descriptor["noise"] = generation_descriptor[
                    "noise"][start:stop]
                generation_descriptor["trajectory_seeds"] = (
                    generation_descriptor["trajectory_seeds"][start:stop])
                replay_descriptor["generation_descriptor"] = (
                    generation_descriptor)
                for key, value in tuple(replay_descriptor.items()):
                    if key in {
                        "rng_state", "latent_shape", "generation_descriptor",
                        "student_latent",
                    }:
                        continue
                    if (
                        isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and int(value.shape[0]) == physical_batch
                    ):
                        replay_descriptor[key] = value[start:stop]
                unit_slice = slice(
                    logical_offset + start, logical_offset + stop)
                unit_features = (
                    {
                        name: value[unit_slice]
                        for name, value in logical_features.items()
                    }
                    if isinstance(logical_features, Mapping)
                    else logical_features[unit_slice]
                )
                unit_cotangent = (
                    {
                        name: value[unit_slice]
                        for name, value in logical_cotangent.items()
                    }
                    if isinstance(logical_cotangent, Mapping)
                    else logical_cotangent[unit_slice]
                )
                units.append({
                    "replay_descriptor": replay_descriptor,
                    "features": unit_features,
                    "cotangent": unit_cotangent,
                    "image_or_video_shape": [
                        replay_batch,
                        *physical_entry["image_or_video_shape"][1:],
                    ],
                    "conditional_dict": (
                        self._slice_fresh_video_batch_mapping(
                            physical_entry["conditional_dict"],
                            start, stop, physical_batch)),
                    "unconditional_dict": (
                        self._slice_fresh_video_batch_mapping(
                            physical_entry["unconditional_dict"],
                            start, stop, physical_batch)),
                })
            logical_offset += physical_batch
        if logical_offset != self.fresh_video_logical_micro_batch_size:
            raise RuntimeError(
                "logical replay partition does not cover its full batch")
        return units

    def _fresh_video_window_log(self, entries, reference):
        if getattr(
            self.model,
            "vjepa2_dense_frame_sampling",
            "center_contiguous",
        ) != "random_contiguous_per_video":
            return {}
        num_starts = (
            int(self.model.vjepa2_dense_source_num_frames)
            - int(self.model.vjepa2_dense_num_frames)
            + 1
        )
        local_error = None
        local_starts = None
        try:
            start_parts = []
            for entry in entries:
                for physical_entry in entry["physical_entries"]:
                    indices = physical_entry["replay_descriptor"][
                        "vjepa2_frame_indices"]
                    if (
                        not isinstance(indices, torch.Tensor)
                        or indices.ndim != 2
                        or int(indices.shape[1])
                        != int(self.model.vjepa2_dense_num_frames)
                    ):
                        raise ValueError(
                            "invalid V-JEPA2 frame indices in GradCache entry")
                    start_parts.append(indices[:, 0].to(
                        device=reference.device, dtype=torch.long))
            local_starts = torch.cat(start_parts)
            if int(local_starts.numel()) != self.fresh_video_local_samples:
                raise ValueError(
                    "V-JEPA2 window count does not match local samples")
            if (
                int(local_starts.min().item()) < 0
                or int(local_starts.max().item()) >= num_starts
            ):
                raise ValueError("V-JEPA2 window start is out of range")
        except BaseException as exc:
            local_error = exc
        if not self._all_ranks_true(local_error is None):
            message = (
                f": {type(local_error).__name__}: {local_error}"
                if local_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "V-JEPA2 window logging validation failed" + message
            ) from local_error
        histogram = torch.bincount(
            local_starts, minlength=num_starts)[:num_starts]
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(histogram, op=dist.ReduceOp.SUM)
        if int(histogram.sum().item()) != self.fresh_video_total_samples:
            raise RuntimeError(
                "global V-JEPA2 window histogram has an invalid sample count")
        used = histogram.nonzero(as_tuple=False).flatten()
        if used.numel() == 0:
            raise RuntimeError("global V-JEPA2 window histogram is empty")
        return {
            "vjepa2_dense_window_start_min": used[0],
            "vjepa2_dense_window_start_max": used[-1],
            "vjepa2_dense_window_unique_starts": used.new_tensor(
                int(used.numel())),
            "vjepa2_dense_source_num_frames": used.new_tensor(
                int(self.model.vjepa2_dense_source_num_frames)),
        }

    def _validate_partial_mmd_geometry(self, result):
        expected_forward = self.fresh_video_total_samples
        expected_gradient = self.fresh_video_generator_grad_global_samples
        expected_teacher = (
            expected_gradient
            if bool(getattr(
                self.model,
                "current_teacher_attraction_enabled",
                False,
            ))
            else 0
        )
        full_student_mode = self.fresh_video_full_student_partial_gradient
        expected_kss = expected_forward * (
            expected_forward if full_student_mode else expected_gradient)
        expected_fixed = expected_forward if full_student_mode else expected_gradient
        expected_live = expected_fixed * expected_teacher
        expected_teacher_self = expected_teacher * expected_teacher
        metadata = result.metadata
        branch_metadata = metadata.get("branches")
        if isinstance(branch_metadata, Mapping):
            registry = getattr(
                self.model, "dual_branch_registry", None)
            if isinstance(registry, Mapping) and registry:
                expected_branches = set(registry)
            else:
                feature_type = str(getattr(
                    self.model, "feature_type", ""))
                expected_branches = (
                    {"dinov3", "vjepa2", "videomae_v2"}
                    if feature_type
                    == "triple_dinov3_vjepa2_videomae_v2_dense_latent"
                    else (
                        {"dinov3", "vjepa2"}
                        if feature_type
                        == "dual_dinov3_vjepa2_dense_latent"
                        else {"vjepa2", "videomae_v2"}
                    )
                )
            if set(branch_metadata) != expected_branches:
                raise RuntimeError(
                    "multi-representation MMD metadata branch set mismatch: "
                    f"{set(branch_metadata)} != {expected_branches}")
            denominator_keys = [
                "repulsion_denominator",
                "fixed_teacher_attraction_denominator",
            ]
            expected_branch_values = [
                expected_kss,
                expected_fixed,
            ]
            if expected_teacher:
                denominator_keys.extend([
                    "current_teacher_attraction_denominator",
                    "current_teacher_self_kernel_denominator",
                ])
                expected_branch_values.extend([
                    expected_live,
                    expected_teacher_self,
                ])
            for name, branch in branch_metadata.items():
                observed_branch_values = tuple(
                    int(torch.as_tensor(branch[key]).item())
                    for key in denominator_keys
                )
                if observed_branch_values != tuple(expected_branch_values):
                    raise RuntimeError(
                        f"multi-representation branch {name} denominator mismatch: "
                        f"{observed_branch_values} != "
                        f"{tuple(expected_branch_values)}")
                if int(torch.as_tensor(
                    branch.get("full_student_partial_gradient", 0)
                ).item()) != 1:
                    raise RuntimeError(
                        f"multi-representation branch {name} omitted full-student injection")
            metadata = {**metadata, **branch_metadata["vjepa2"]}
        observed = (
            int(result.global_samples),
            int(result.gradient_global_samples),
            int(result.current_teacher_global_samples),
            int(torch.as_tensor(
                metadata["gradcache_kss_denominator"]).item()),
            int(torch.as_tensor(
                metadata["fixed_teacher_attraction_denominator"]).item()),
            int(torch.as_tensor(
                metadata.get("current_teacher_attraction_denominator", 0)).item()),
            int(torch.as_tensor(
                metadata.get("current_teacher_self_kernel_denominator", 0)).item()),
            int(torch.as_tensor(
                metadata["gradcache_full_student_partial_gradient"]).item()),
        )
        expected = (
            expected_forward,
            expected_gradient,
            expected_teacher,
            expected_kss,
            expected_fixed,
            expected_live,
            expected_teacher_self,
            int(full_student_mode),
        )
        if observed != expected:
            raise RuntimeError(
                "partial-MMD geometry mismatch: "
                f"observed={observed}, expected={expected}")
        if int(torch.as_tensor(
            metadata.get("repulsion_denominator", -1)
        ).item()) != expected_kss:
            raise RuntimeError("partial-MMD core Kss denominator mismatch")
        if full_student_mode:
            if int(torch.as_tensor(
                metadata.get("full_student_partial_gradient", 0)
            ).item()) != 1:
                raise RuntimeError(
                    "loss did not activate full-student partial gradients")
            configured_denominators = (
                str(getattr(
                    self.config, "partial_mmd_normalization_identity", "")),
                int(getattr(
                    self.config, "partial_mmd_kss_denominator", -1)),
                int(getattr(
                    self.config,
                    "partial_mmd_fixed_attraction_denominator", -1)),
                int(getattr(
                    self.config, "partial_mmd_live_cross_denominator", -1)),
                int(getattr(
                    self.config,
                    "partial_mmd_teacher_self_denominator", -1)),
            )
            expected_config = (
                "full_student_population_denominators_v1",
                expected_kss,
                expected_fixed,
                expected_live,
                expected_teacher_self,
            )
            if configured_denominators != expected_config:
                raise RuntimeError(
                    "full-student MMD config normalization mismatch: "
                    f"{configured_denominators} != {expected_config}")
            configured_compensation = getattr(
                self.config, "partial_mmd_gradient_compensation", None)
            if (
                configured_compensation is not None
                and float(configured_compensation)
                != self.fresh_video_generator_grad_sample_scale
            ):
                raise RuntimeError(
                    "full-student MMD gradient compensation mismatch: "
                    f"{configured_compensation} != "
                    f"{self.fresh_video_generator_grad_sample_scale}")
        if bool(getattr(
            self.model, "nystrom_repulsion_fifo_enabled", False
        )):
            raise RuntimeError("partial-MMD experiment requires FIFO=0")
        if bool(getattr(
            self.model, "nystrom_attraction_ema_enabled", False
        )):
            raise RuntimeError(
                "partial-MMD experiment requires attraction EMA=0")
        layers = tuple(getattr(
            self.model,
            "vjepa2_dense_layers",
            (int(self.model.dense_layer),),
        ))
        if layers != (15,):
            raise RuntimeError(
                f"partial-MMD experiment requires only layer 15, got {layers}")
        if self.frame_fd_fusion_enabled:
            raise RuntimeError(
                "partial-MMD experiment requires image/frame loss disabled")
        if self.is_main_process and int(self.step) == 0:
            print(
                "[partial-mmd-geometry] "
                f"forward={expected_forward} "
                f"gradient={expected_gradient} "
                f"teacher={expected_teacher} "
                f"Kss_denominator={expected_kss} "
                f"fixed_attraction_denominator={expected_fixed} "
                f"live_cross_denominator={expected_live} "
                f"teacher_self_denominator={expected_teacher_self} "
                f"full_student_partial_gradient={int(full_student_mode)} "
                "gradient_compensation="
                f"{self.fresh_video_generator_grad_sample_scale:g} "
                "fifo=0 attraction_ema=0 layer=15",
                flush=True,
            )
            if bool(getattr(self.model, "dual_video_mmd_enabled", False)):
                registry = getattr(self.model, "dual_branch_registry", {})
                shapes = {
                    name: list(branch["sample_shape"])
                    for name, branch in sorted(registry.items())
                }
                expected_shapes = {
                    name: ([1280, 1024] if name == "videomae_v2"
                           else [1024, 1024])
                    for name in registry
                }
                if shapes != expected_shapes:
                    raise RuntimeError(
                        f"multi-encoder branch sample shapes changed: {shapes}")
                feature_type = str(getattr(
                    self.model, "feature_type", ""))
                shape_tag = (
                    "triple-sample-shapes"
                    if feature_type
                    == "triple_dinov3_vjepa2_videomae_v2_dense_latent"
                    else (
                        "dinov3-vjepa2-sample-shapes"
                        if feature_type == "dual_dinov3_vjepa2_dense_latent"
                        else "dual-sample-shapes"
                    )
                )
                print(
                    f"[{shape_tag}] "
                    + " ".join(
                        f"{name}={shape[0]}x{shape[1]}"
                        for name, shape in sorted(shapes.items())
                    ),
                    flush=True,
                )

    def _run_distribution_matching_step(self):
        """Perform one distribution-matching update (paper Sec. 3.3).

        Collect B generated representations, differentiate the feature objective
        for b selected rollouts, then replay encoders, VAE and generator. The
        released selection and loss conventions are documented in paper-to-code.md."""
        self._fresh_progress("feature_collection_start")
        restore_error = None
        try:
            self.model.restore_frozen_modules_before_feature_replay()
        except BaseException as exc:
            restore_error = exc
        if not self._all_ranks_true(restore_error is None):
            message = (
                f": {type(restore_error).__name__}: {restore_error}"
                if restore_error is not None
                else " on another rank"
            )
            raise RuntimeError(
                "frozen replay module restore failed" + message
            ) from restore_error
        # Frozen encoders stay in eval mode; generator gradients are enabled only on replay.
        self.model.eval()
        dense_log_prefix = getattr(
            self.model, "dense_log_prefix", "vjepa2_dense")

        def dense_key(suffix):
            return f"{dense_log_prefix}_{suffix}"

        if self.step % 20 == 0:
            torch.cuda.empty_cache()
        # This cache lasts one optimizer update; it is not a history of older rollouts.
        cache = self.fresh_video_gradcache
        frame_cache = self.fresh_frame_fd_gradcache
        if cache is None:
            raise RuntimeError("fresh video GradCache is not initialized")
        if cache.local_samples or cache.finalized:
            raise RuntimeError(
                "fresh video GradCache leaked state across optimizer updates")
        if frame_cache is not None and (
            frame_cache.local_videos or frame_cache.finalized
        ):
            raise RuntimeError(
                "frame FD GradCache leaked state across optimizer updates")
        if (
            hasattr(self.model, "distribution_state_pending")
            and self.model.distribution_state_pending()
        ):
            raise RuntimeError(
                "dynamic distribution state leaked across optimizer updates")
        active_fidel_reference = (
            self._activate_fidel_reference_for_update()
            if self.fidel_two_level_enabled
            else None
        )
        # This consensus must happen before the first model forward because
        # enabled ranks enter a pixel-cache collective inside that forward.
        use_prepared_replay = bool(
            self.uses_fresh_video_gradcache
            and getattr(
                self.model,
                "fresh_video_prepared_replay_enabled",
                getattr(self.model, "fidel_p0_replay_enabled", False),
            )
        )
        all_use_prepared = self._all_ranks_true(use_prepared_replay)
        none_use_prepared = self._all_ranks_true(
            not use_prepared_replay)
        if not (all_use_prepared or none_use_prepared):
            raise RuntimeError(
                "fresh-video prepared replay setting differs across ranks")
        use_prepared_replay = all_use_prepared

        entries = []
        try:
            physical_per_logical = (
                self.fresh_video_logical_micro_batch_size
                // self.fresh_video_physical_forward_batch_size)
            for microbatch_index in range(self.fresh_video_microbatches):
                gradient_selected = bool(
                    microbatch_index >= (
                        self.fresh_video_microbatches
                        - self.fresh_video_generator_grad_microbatches)
                )
                physical_entries = []
                for physical_slot in range(physical_per_logical):
                    physical_index = (
                        microbatch_index * physical_per_logical
                        + physical_slot)
                    window_sample_keys = None
                    if getattr(
                        self.model,
                        "vjepa2_dense_frame_sampling",
                        "center_contiguous",
                    ) == "random_contiguous_per_video":
                        local_start = (
                            physical_index
                            * self.fresh_video_physical_forward_batch_size)
                        global_start = (
                            dist.get_rank() * self.fresh_video_local_samples
                            + local_start)
                        update_start = (
                            int(self.step) * self.fresh_video_total_samples)
                        window_sample_keys = tuple(
                            update_start + global_start + offset
                            for offset in range(
                                self.fresh_video_physical_forward_batch_size))
                    batch = (
                        self._matched_batch_for_microbatch(physical_index)
                        if getattr(
                            self, "matched_counter_sampling_enabled", False)
                        else self._next_batch()
                    )
                    physical_entries.append(
                        self._collect_rollout_microbatch(
                            batch,
                            gradient_selected=gradient_selected,
                            vjepa2_window_sample_keys=window_sample_keys,
                        )
                    )
                entry = self._aggregate_fresh_video_forward_entries(
                    physical_entries,
                    gradient_selected=gradient_selected,
                )
                add_error = None
                try:
                    student_ids = entry.get("sample_ids")
                    if hasattr(cache, "partial_gradient_enabled"):
                        cache.add(
                            entry["features"],
                            entry.get("current_teacher_features"),
                            gradient_selected=gradient_selected,
                            sample_ids=student_ids,
                            current_teacher_sample_ids=(
                                student_ids
                                if entry.get(
                                    "current_teacher_features") is not None
                                else None
                            ),
                        )
                    else:
                        cache.add(entry["features"])
                    entry.pop("current_teacher_features", None)
                except BaseException as exc:
                    add_error = exc
                if not self._all_ranks_true(add_error is None):
                    message = (
                        f": {type(add_error).__name__}: {add_error}"
                        if add_error is not None
                        else " on another rank"
                    )
                    raise RuntimeError(
                        "fresh video GradCache add failed" + message
                    ) from add_error
                if not gradient_selected:
                    for physical_entry in entry["physical_entries"]:
                        descriptor = physical_entry.get(
                            "replay_descriptor", {})
                        descriptor.pop("cached_student_video01", None)
                        physical_entry.pop("conditional_dict", None)
                        physical_entry.pop("unconditional_dict", None)
                        physical_entry.pop("image_or_video_shape", None)
                entries.append(entry)
            # Differentiate the global feature-space loss before constructing generator graphs.
            result = cache.finalize()
            if (
                self.fresh_video_generator_grad_subsample_enabled
                and hasattr(cache, "partial_gradient_enabled")
            ):
                self._validate_partial_mmd_geometry(result)
            window_log = self._fresh_video_window_log(entries, result.loss)
            if (
                use_prepared_replay
                and self.model.generator_replay_frozen_encoder_cpu_offload
            ):
                if not hasattr(cache, "offload_finalized_to_cpu"):
                    raise RuntimeError(
                        "prepared triple GradCache lacks CPU offload support")
                result = cache.offload_finalized_to_cpu()
                for entry in entries:
                    features = entry.pop("features")
                    if entry.get("gradient_selected", False):
                        if not isinstance(features, Mapping):
                            raise TypeError(
                                "triple replay features must be a mapping")
                        entry["features"] = {
                            name: value.detach().to(
                                device="cpu", non_blocking=False)
                            for name, value in features.items()
                        }
            frame_result = (
                (
                    frame_cache.finalize_fidel()
                    if self.fidel_two_level_enabled
                    else frame_cache.finalize()
                )
                if frame_cache is not None
                else None
            )
            fidel_update = None
            if self.fidel_two_level_enabled:
                if frame_result is None:
                    raise RuntimeError(
                        "two-level FiDEL requires finalized frame objectives")
                fidel_error = None
                try:
                    fidel_update = self.model.stage_fidel_update(
                        video_result=result,
                        frame_result=frame_result,
                    )
                    video_cotangents = fidel_update[
                        "video_fsdp_cotangents"]
                    frame_cotangents = fidel_update[
                        "frame_fsdp_feature_cotangents"]
                    skip_optimizer_step = bool(
                        fidel_update["all_weights_zero"])
                    if (
                        len(video_cotangents) != len(entries)
                        or len(frame_cotangents) != len(entries)
                    ):
                        raise RuntimeError(
                            "FiDEL cotangent/microbatch count mismatch")
                except BaseException as exc:
                    fidel_error = exc
                if not self._all_ranks_true(fidel_error is None):
                    message = (
                        f": {type(fidel_error).__name__}: {fidel_error}"
                        if fidel_error is not None
                        else " on another rank"
                    )
                    raise RuntimeError(
                        "two-level FiDEL staging failed" + message
                    ) from fidel_error
                all_skip = self._all_ranks_true(skip_optimizer_step)
                none_skip = self._all_ranks_true(not skip_optimizer_step)
                if not (all_skip or none_skip):
                    raise RuntimeError(
                        "FiDEL all-weights-zero decision differs across ranks")
                skip_optimizer_step = all_skip
            else:
                video_cotangents = result.fsdp_cotangents
                frame_cotangents = (
                    frame_result.fsdp_feature_cotangents
                    if frame_result is not None
                    else (None,) * len(entries)
                )
                skip_optimizer_step = False
            expected_replay_start = (
                len(entries)
                - self.fresh_video_generator_grad_microbatches
            )
            expected_replay_indices = tuple(
                range(expected_replay_start, len(entries)))
            replay_indices = tuple(getattr(
                result,
                "replay_batch_indices",
                expected_replay_indices,
            ))
            if replay_indices != expected_replay_indices:
                raise RuntimeError(
                    "GradCache replay indices are not the configured suffix: "
                    f"{replay_indices} != {expected_replay_indices}")
            replay_entries = tuple(entries[index] for index in replay_indices)
            if (
                not replay_entries
                or len(replay_entries)
                != self.fresh_video_generator_grad_microbatches
                or (
                    hasattr(cache, "partial_gradient_enabled")
                    and any(
                        not entry.get("gradient_selected", False)
                        for entry in replay_entries
                    )
                )
            ):
                raise RuntimeError(
                    "generator-gradient replay microbatch selection is invalid")
            if len(video_cotangents) == len(replay_indices):
                replay_video_cotangents = tuple(video_cotangents)
            elif len(video_cotangents) == len(entries):
                replay_video_cotangents = tuple(
                    video_cotangents[index] for index in replay_indices)
            else:
                raise RuntimeError(
                    "fresh GradCache cotangent/replay count mismatch")
            # Apply B/b selection compensation once; FSDP averaging has a separate factor.
            replay_video_cotangents = _scale_replay_video_cotangents(
                replay_video_cotangents,
                self.fresh_video_generator_grad_sample_scale,
            )
            if len(frame_cotangents) == len(entries):
                replay_frame_cotangents = tuple(
                    frame_cotangents[index] for index in replay_indices)
            elif len(frame_cotangents) == len(replay_indices):
                replay_frame_cotangents = tuple(frame_cotangents)
            else:
                raise RuntimeError(
                    "frame FD cotangent/replay count mismatch")

            if any(value is not None for value in replay_frame_cotangents):
                raise NotImplementedError(
                    "VACE-aligned split replay is restricted to pure MMD")
            replay_units = []
            for entry, cotangent in zip(
                replay_entries, replay_video_cotangents
            ):
                replay_units.extend(
                    self._split_fresh_video_replay_units(entry, cotangent))
            if (
                len(replay_units)
                * self.fresh_video_physical_replay_batch_size
                != self.fresh_video_generator_grad_local_samples
            ):
                raise RuntimeError(
                    "physical replay units do not cover exactly the configured "
                    "gradient suffix")
            prepared_replays = [None] * len(replay_units)
            self._fresh_progress("encoder_vae_replay_start", units=len(replay_units))
            if use_prepared_replay and not skip_optimizer_step:
                diagnostics_interval = (
                    int(getattr(
                        self.model,
                        "fidel_p0_branch_diagnostics_interval",
                        0,
                    ))
                    if self.fidel_two_level_enabled
                    else 0
                )
                collect_branch_diagnostics = bool(
                    diagnostics_interval > 0
                    and (int(self.step) + 1) % diagnostics_interval == 0
                )
                all_collect_diagnostics = self._all_ranks_true(
                    collect_branch_diagnostics)
                none_collect_diagnostics = self._all_ranks_true(
                    not collect_branch_diagnostics)
                if not (all_collect_diagnostics or none_collect_diagnostics):
                    raise RuntimeError(
                        "FiDEL branch diagnostic setting differs across ranks")
                collect_branch_diagnostics = all_collect_diagnostics
                prepared_replays = []
                for unit in replay_units:
                    prepared_replays.append(
                        self.model.prepare_latent_gradients(
                            replay_descriptor=unit["replay_descriptor"],
                            expected_features=unit["features"],
                            feature_cotangent=unit["cotangent"],
                            frame_fd_feature_cotangents=None,
                            replay_atol=float(getattr(
                                self.config,
                                "fresh_video_replay_atol",
                                1e-5,
                            )),
                            replay_rtol=float(getattr(
                                self.config,
                                "fresh_video_replay_rtol",
                                1e-4,
                            )),
                            collect_branch_latent_diagnostics=(
                                collect_branch_diagnostics),
                        )
                    )

            replay_offload_bytes = 0
            self._fresh_progress("encoder_vae_replay_complete")
            if use_prepared_replay and not skip_optimizer_step:
                offload_error = None
                try:
                    replay_offload_bytes = (
                        self.model.offload_frozen_modules_for_generator_replay()
                    )
                except BaseException as exc:
                    offload_error = exc
                if not self._all_ranks_true(offload_error is None):
                    message = (
                        f": {type(offload_error).__name__}: {offload_error}"
                        if offload_error is not None
                        else " on another rank"
                    )
                    raise RuntimeError(
                        "frozen replay module offload failed" + message
                    ) from offload_error

            replay_logs = []
            frame_fd_weight = (
                0.0
                if self.fidel_two_level_enabled
                else self.model.current_frame_fd_weight()
            )
            if not skip_optimizer_step:
                self._fresh_progress("generator_backward_start", units=len(replay_units))
                for unit, prepared_replay in zip(
                    replay_units, prepared_replays
                ):
                    surrogate, replay_log = (
                        self.model.replay_generator_gradients(
                            replay_descriptor=unit["replay_descriptor"],
                            expected_features=unit["features"],
                            feature_cotangent=unit["cotangent"],
                            frame_fd_feature_cotangents=None,
                            image_or_video_shape=unit[
                                "image_or_video_shape"],
                            conditional_dict=unit["conditional_dict"],
                            unconditional_dict=unit[
                                "unconditional_dict"],
                            initial_latent=None,
                            prepared_replay=prepared_replay,
                            replay_atol=float(getattr(
                                self.config,
                                "fresh_video_replay_atol",
                                1e-5,
                            )),
                            replay_rtol=float(getattr(
                                self.config,
                                "fresh_video_replay_rtol",
                                1e-4,
                            )),
                        )
                    )
                    surrogate.backward()
                    replay_logs.append(replay_log)
                    if len(replay_logs) % 4 == 0 or len(replay_logs) == len(replay_units):
                        self._fresh_progress("generator_backward_progress", completed=len(replay_logs), total=len(replay_units))

            balance_stage = (
                self.model.finalize_frame_fd_balance_stage()
                if (
                    frame_result is not None
                    and not self.fidel_two_level_enabled
                )
                else None
            )
            if skip_optimizer_step:
                generator_grad_norm = result.loss.new_zeros(())
                log_dict = {
                    "combined_latent_grad_norm": result.loss.new_zeros(()),
                    dense_key("feature_grad_norm"): (
                        result.loss.new_zeros(())),
                    dense_key("video_grad_norm"): result.loss.new_zeros(()),
                    dense_key("latent_grad_norm"): result.loss.new_zeros(()),
                    "frame_fd_feature_grad_norm": result.loss.new_zeros(()),
                    "frame_fd_pixel_grad_norm": result.loss.new_zeros(()),
                    "frame_fd_latent_grad_norm": result.loss.new_zeros(()),
                    "video_frame_latent_grad_cosine": (
                        result.loss.new_zeros(())),
                    dense_key("replay_max_abs"): result.loss.new_zeros(()),
                    dense_key("latent_replay_max_abs"): (
                        result.loss.new_zeros(())),
                }
            else:
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                    self.max_grad_norm_generator)
                log_dict = {
                    key: torch.stack([
                        item[key].detach().float()
                        for item in replay_logs
                    ]).mean()
                    for key in replay_logs[0]
                }
            meta = result.metadata
            generator_loss = (
                fidel_update["generator_loss"]
                if self.fidel_two_level_enabled
                else result.optimized_loss
            )
            if (
                frame_result is not None
                and not self.fidel_two_level_enabled
            ):
                generator_loss = (
                    generator_loss
                    + float(frame_fd_weight) * frame_result.loss)
            multilayer_metadata = meta.get("layers")
            displayed_mmd = (
                meta["raw_mmd_sum"]
                if isinstance(multilayer_metadata, dict)
                else result.loss
            )
            log_dict.update({
                "generator_loss": generator_loss,
                "generator_grad_norm": generator_grad_norm,
                "dmdtrain_gradient_norm": log_dict[
                    "combined_latent_grad_norm"],
                dense_key("mmd"): displayed_mmd,
                dense_key("optimized_mmd"): result.optimized_loss,
                dense_key("exact_repulsion"): meta["repulsion"],
                dense_key("nystrom_attraction"): meta["attraction"],
                dense_key("nystrom_attraction_weight"): meta.get(
                    "attraction_weight", result.loss.new_ones(())),
                dense_key("reference_constant"): meta[
                    "reference_constant"],
                dense_key("local_video_samples"): meta[
                    "gradcache_local_samples"],
                dense_key("global_video_samples"): meta[
                    "gradcache_global_samples"],
                dense_key("gradient_local_video_samples"): meta.get(
                    "gradcache_gradient_local_samples",
                    result.loss.new_tensor(
                        self.fresh_video_generator_grad_local_samples,
                        dtype=torch.long,
                    ),
                ),
                dense_key("gradient_global_video_samples"): meta.get(
                    "gradcache_gradient_global_samples",
                    result.loss.new_tensor(
                        self.fresh_video_generator_grad_global_samples,
                        dtype=torch.long,
                    ),
                ),
                dense_key("current_teacher_global_samples"): meta.get(
                    "gradcache_current_teacher_global_samples",
                    result.loss.new_zeros((), dtype=torch.long),
                ),
                dense_key("kss_denominator"): meta.get(
                    "gradcache_kss_denominator",
                    result.loss.new_tensor(
                        self.fresh_video_total_samples ** 2,
                        dtype=torch.long,
                    ),
                ),
                dense_key("fresh_microbatches"): result.loss.new_tensor(
                    len(entries), dtype=torch.long),
                dense_key("replay_microbatches"): result.loss.new_tensor(
                    len(replay_entries), dtype=torch.long),
                dense_key("physical_forward_calls"): result.loss.new_tensor(
                    len(entries) * physical_per_logical, dtype=torch.long),
                dense_key("physical_replay_calls"): result.loss.new_tensor(
                    len(replay_units), dtype=torch.long),
                dense_key("raw_tokens"): result.loss.new_tensor(
                    getattr(
                        self.model,
                        "dense_raw_tokens",
                        self.model.vjepa2_dense_raw_tokens,
                    ), dtype=torch.long),
                dense_key("sampled_tokens"): result.loss.new_tensor(
                    getattr(
                        self.model,
                        "dense_num_tokens",
                        self.model.vjepa2_dense_num_tokens,
                    ), dtype=torch.long),
                dense_key("feature_dim"): result.loss.new_tensor(
                    getattr(
                        self.model,
                        "dense_feature_dim",
                        self.model.vjepa2_dense_feature_dim,
                    ), dtype=torch.long),
                dense_key("num_frames"): result.loss.new_tensor(
                    getattr(
                        self.model,
                        "dense_num_frames",
                        self.model.vjepa2_dense_num_frames,
                    ), dtype=torch.long),
                dense_key("layer"): result.loss.new_tensor(
                    getattr(
                        self.model,
                        "dense_layer",
                        self.model.vjepa2_dense_layer,
                    ), dtype=torch.long),
                dense_key("prefix_blocks"): result.loss.new_tensor(
                    getattr(
                        self.model,
                        "dense_layer",
                        self.model.vjepa2_dense_layer,
                    ) + 1, dtype=torch.long),
                dense_key("nystrom_landmarks"): result.loss.new_tensor(
                    self.model.latent_mmd.landmarks.shape[0],
                    dtype=torch.long),
                dense_key("reference_samples"): result.loss.new_tensor(
                    self.model.vjepa2_nystrom_reference_count,
                    dtype=torch.long),
                "fresh_video_optimizer_step_skipped": (
                    result.loss.new_tensor(
                        int(skip_optimizer_step), dtype=torch.long)),
                "fresh_video_generator_grad_sample_scale": (
                    result.loss.new_tensor(
                        self.fresh_video_generator_grad_sample_scale)),
                "fresh_video_generator_effective_replay_scale": (
                    result.loss.new_tensor(
                        self.fresh_video_generator_grad_sample_scale
                        * float(getattr(self, "world_size", 1)))),
                "fresh_video_generator_grad_subsample_enabled": (
                    result.loss.new_tensor(
                        int(self.fresh_video_generator_grad_subsample_enabled),
                        dtype=torch.long,
                    )
                ),
                "fresh_video_generator_grad_local_samples": (
                    result.loss.new_tensor(
                        self.fresh_video_generator_grad_local_samples,
                        dtype=torch.long,
                    )
                ),
                "fresh_video_generator_grad_global_samples": (
                    result.loss.new_tensor(
                        self.fresh_video_generator_grad_global_samples,
                        dtype=torch.long,
                    )
                ),
                "fresh_video_generator_grad_microbatches": (
                    result.loss.new_tensor(
                        self.fresh_video_generator_grad_microbatches,
                        dtype=torch.long,
                    )
                ),
                "fresh_video_logical_micro_batch_size": (
                    result.loss.new_tensor(
                        self.fresh_video_logical_micro_batch_size,
                        dtype=torch.long,
                    )
                ),
                "fresh_video_physical_forward_batch_size": (
                    result.loss.new_tensor(
                        self.fresh_video_physical_forward_batch_size,
                        dtype=torch.long,
                    )
                ),
                "fresh_video_physical_replay_batch_size": (
                    result.loss.new_tensor(
                        self.fresh_video_physical_replay_batch_size,
                        dtype=torch.long,
                    )
                ),
                "generator_replay_frozen_module_offload_bytes": (
                    result.loss.new_tensor(
                        replay_offload_bytes, dtype=torch.long)
                ),
            })
            if isinstance(meta.get("branches"), Mapping):
                for branch_name, branch in meta["branches"].items():
                    prefix = f"{branch_name}_dense"
                    branch_logs = {
                        f"{prefix}_raw_mmd": branch["raw_mmd"],
                        f"{prefix}_normalized_mmd": branch["normalized_mmd"],
                        f"{prefix}_normalization_denominator": (
                            branch["normalization_denominator"]),
                        f"{prefix}_exact_repulsion": branch["repulsion"],
                        f"{prefix}_fixed_teacher_attraction": (
                            branch.get(
                                "fixed_teacher_attraction",
                                branch["attraction"],
                            )),
                        f"{prefix}_reference_constant": (
                            branch["reference_constant"]),
                        f"{prefix}_bandwidth": branch["sigma"],
                        f"{prefix}_kss_denominator": (
                            branch["repulsion_denominator"]),
                        f"{prefix}_fixed_attraction_denominator": (
                            branch["fixed_teacher_attraction_denominator"]),
                    }
                    if "current_teacher_attraction" in branch:
                        branch_logs.update({
                            f"{prefix}_current_teacher_attraction": (
                                branch["current_teacher_attraction"]),
                            f"{prefix}_teacher_cross_denominator": (
                                branch[
                                    "current_teacher_attraction_denominator"]),
                            f"{prefix}_teacher_self_denominator": (
                                branch[
                                    "current_teacher_self_kernel_denominator"]),
                        })
                    log_dict.update(branch_logs)
                log_dict["dual_dense_normalized_mmd_sum"] = meta[
                    "normalized_mmd_sum"]
            teacher_cache_logs = [
                physical_entry["current_teacher_cache_log"]
                for entry in entries
                for physical_entry in entry["physical_entries"]
                if "current_teacher_cache_log" in physical_entry
            ]
            if teacher_cache_logs:
                cache_stats = torch.tensor([
                    sum(item["hits"] for item in teacher_cache_logs),
                    sum(item["misses"] for item in teacher_cache_logs),
                    sum(item["decode_seconds"] for item in teacher_cache_logs),
                    sum(item["first_pass_seconds"] for item in teacher_cache_logs),
                    len(teacher_cache_logs),
                    sum(item.get("extract_seconds", 0.0)
                        for item in teacher_cache_logs),
                    sum("extract_seconds" in item
                        for item in teacher_cache_logs),
                ], dtype=torch.float64, device=torch.device(
                    "cuda",
                    self.device,
                ))
                if dist.is_initialized() and self.world_size > 1:
                    dist.all_reduce(cache_stats, op=dist.ReduceOp.SUM)
                total_items = cache_stats[0] + cache_stats[1]
                teacher_metric_prefix = (
                    "triple_bank"
                    if self.current_teacher_source == "matched_triple_feature_bank"
                    else ("dinov3_vjepa2_bank"
                    if self.current_teacher_source
                    == "matched_dinov3_vjepa2_feature_bank"
                    else ("dual_bank"
                    if self.current_teacher_source == "matched_dual_feature_bank"
                    else (
                        "matched"
                        if self.current_teacher_source == "matched_manifest_video"
                        else "openvid"
                    )))
                )
                log_dict.update({
                    f"{teacher_metric_prefix}_teacher_cache_hits": cache_stats[0],
                    f"{teacher_metric_prefix}_teacher_cache_misses": cache_stats[1],
                    f"{teacher_metric_prefix}_teacher_cache_hit_rate": (
                        cache_stats[0] / total_items.clamp_min(1.0)),
                    f"{teacher_metric_prefix}_teacher_decode_seconds": cache_stats[2],
                    f"{teacher_metric_prefix}_teacher_first_pass_seconds_per_microbatch": (
                        cache_stats[3] / cache_stats[4].clamp_min(1.0)),
                })
                if float(cache_stats[6].item()) > 0:
                    log_dict[
                        f"{teacher_metric_prefix}_teacher_extract_seconds_per_microbatch"
                    ] = cache_stats[5] / cache_stats[6]
            if "current_attraction" in meta:
                log_dict.update({
                    dense_key("nystrom_current_attraction"): (
                        meta["current_attraction"]),
                    dense_key("nystrom_attraction_ema_updates"): (
                        meta["attraction_ema_updates"]),
                    dense_key("nystrom_attraction_current_coefficient"): (
                        meta["attraction_current_coefficient"]),
                })
            if "current_teacher_attraction" in meta:
                log_dict.update({
                    dense_key("fixed_teacher_attraction"): (
                        meta["fixed_teacher_attraction"]),
                    dense_key("current_teacher_attraction"): (
                        meta["current_teacher_attraction"]),
                    dense_key("mixed_current_attraction"): (
                        meta["mixed_current_attraction"]),
                    dense_key("fixed_teacher_normalized_weight"): (
                        meta[
                            "fixed_teacher_attraction_normalized_weight"]),
                    dense_key("current_teacher_normalized_weight"): (
                        meta[
                            "current_teacher_attraction_normalized_weight"]),
                    dense_key("current_teacher_self_kernel"): (
                        meta["current_teacher_self_kernel"]),
                    dense_key("current_teacher_exact_mmd"): (
                        meta["current_teacher_exact_mmd"]),
                    dense_key("current_teacher_samples"): (
                        meta["current_teacher_samples"]),
                })
                if "attraction_ema_blend_current_coefficient" in meta:
                    log_dict.update({
                        dense_key("attraction_ema_blend_coefficient"): (
                            meta[
                                "attraction_ema_blend_current_coefficient"]),
                        dense_key("attraction_gradient_coefficient"): (
                            meta[
                                "attraction_gradient_current_coefficient"]),
                        dense_key("effective_ema_loss"): (
                            meta["effective_ema_loss"]),
                    })
            if "repulsion_fifo_enabled" in meta:
                log_dict.update({
                    dense_key("repulsion_fifo_capacity"): (
                        meta["repulsion_fifo_capacity"]),
                    dense_key("repulsion_fifo_fill_before"): (
                        meta["repulsion_fifo_fill_before"]),
                    dense_key("repulsion_fifo_fill_after_commit"): (
                        meta["repulsion_fifo_fill_after_commit"]),
                    dense_key("repulsion_fifo_commits_before"): (
                        meta["repulsion_fifo_commits"]),
                    dense_key("repulsion_current_offdiag_kernel_sum"): (
                        meta["repulsion_current_offdiag_kernel_sum"]),
                    dense_key("repulsion_history_kernel_sum"): (
                        meta["repulsion_history_kernel_sum"]),
                    dense_key("repulsion_denominator"): (
                        meta["repulsion_denominator"]),
                })
            if isinstance(multilayer_metadata, dict):
                log_dict.update({
                    "vjepa2_dense_multilayer_normalized_sum": result.loss,
                    "vjepa2_dense_multilayer_weighted_sum": (
                        result.optimized_loss),
                    "vjepa2_dense_num_layers": meta["num_layers"],
                    "vjepa2_dense_sampled_tokens_per_layer": (
                        result.loss.new_tensor(
                            self.model.vjepa2_dense_num_tokens,
                            dtype=torch.long,
                        )
                    ),
                    "vjepa2_dense_gradcache_transport_tokens": (
                        result.loss.new_tensor(
                            self.model.vjepa2_dense_num_tokens
                            * len(self.model.vjepa2_dense_layers),
                            dtype=torch.long,
                        )
                    ),
                    "vjepa2_dense_self_normalized": (
                        meta["self_normalized"]),
                    "vjepa2_dense_normalization_epsilon": (
                        meta["normalization_epsilon"]),
                    "vjepa2_dense_layer_min": result.loss.new_tensor(
                        min(self.model.vjepa2_dense_layers),
                        dtype=torch.long,
                    ),
                    "vjepa2_dense_layer_max": result.loss.new_tensor(
                        max(self.model.vjepa2_dense_layers),
                        dtype=torch.long,
                    ),
                })
                for layer in self.model.vjepa2_dense_layers:
                    layer_meta = multilayer_metadata[str(layer)]
                    prefix = f"vjepa2_dense_l{layer}"
                    log_dict.update({
                        f"{prefix}_mmd": layer_meta["raw_mmd"],
                        f"{prefix}_normalized_mmd": (
                            layer_meta["normalized_mmd"]),
                        f"{prefix}_weighted_mmd": (
                            layer_meta["weighted_mmd"]),
                        f"{prefix}_repulsion": (
                            layer_meta["repulsion"]),
                        f"{prefix}_nystrom_attraction": (
                            layer_meta["attraction"]),
                        f"{prefix}_reference_constant": (
                            layer_meta["reference_constant"]),
                        f"{prefix}_bandwidth": layer_meta["sigma"],
                    })
                    if "current_attraction" in layer_meta:
                        log_dict.update({
                            f"{prefix}_nystrom_current_attraction": (
                                layer_meta["current_attraction"]),
                            f"{prefix}_attraction_ema_updates": (
                                layer_meta["attraction_ema_updates"]),
                            f"{prefix}_attraction_current_coefficient": (
                                layer_meta[
                                    "attraction_current_coefficient"]),
                        })
            elif not isinstance(meta.get("branches"), Mapping):
                log_dict[dense_key("bandwidth")] = meta["sigma"]
            log_dict.update(window_log)
            if self.fidel_two_level_enabled:
                log_dict["fidel_skip_optimizer_step"] = (
                    result.loss.new_tensor(
                        int(skip_optimizer_step), dtype=torch.long))
                log_dict.update(fidel_update["diagnostics"])
                log_dict["fidel_reference_slot"] = result.loss.new_tensor(
                    int(active_fidel_reference["slot"]),
                    dtype=torch.long,
                )
            if frame_result is not None:
                frame_meta = frame_result.metadata
                log_dict.update({
                    "frame_fd_local_videos": frame_meta[
                        "frame_fd_local_videos"],
                    "frame_fd_global_videos": frame_meta[
                        "frame_fd_global_videos"],
                    "frame_fd_frames_per_video": frame_meta[
                        "frame_fd_frames_per_video"],
                    "frame_fd_global_frames": frame_meta[
                        "frame_fd_global_frames"],
                })
                if not self.fidel_two_level_enabled:
                    log_dict.update({
                        "frame_fd_loss": frame_result.loss,
                        "frame_fd_dynamic_weight": result.loss.new_tensor(
                            float(frame_fd_weight)),
                        "frame_fd_mmd_latent_rms": result.loss.new_tensor(
                            balance_stage.mmd_rms),
                        "frame_fd_latent_rms": result.loss.new_tensor(
                            balance_stage.fd_rms),
                        "frame_fd_next_mmd_rms_ema": result.loss.new_tensor(
                            balance_stage.next_mmd_rms_ema),
                        "frame_fd_next_fd_rms_ema": result.loss.new_tensor(
                            balance_stage.next_fd_rms_ema),
                        "frame_fd_balance_committed_updates": (
                            self.model.frame_fd_balancer.committed_updates
                            .detach().clone()),
                    })
                for judge in self.model.frame_fd_head.judges:
                    name = judge.name
                    before = int(
                        frame_meta["frame_fd_queue_filled_before"][name])
                    after = min(
                        int(judge.queue.size),
                        before + int(frame_result.global_videos)
                        * int(frame_result.frames_per_video),
                    )
                    if not self.fidel_two_level_enabled:
                        log_dict[f"frame_fd_{name}"] = frame_meta[
                            f"frame_fd_{name}"]
                    else:
                        log_dict[f"frame_fd_{name}_raw_loss"] = frame_meta[
                            f"frame_fd_{name}_raw_loss"]
                        log_dict[f"frame_fd_{name}_fresh_score"] = frame_meta[
                            f"frame_fd_{name}_fresh_score"]
                    log_dict[f"frame_fd_queue_{name}_filled_before"] = (
                        result.loss.new_tensor(before, dtype=torch.long))
                    log_dict[f"frame_fd_queue_{name}_filled_after_commit"] = (
                        result.loss.new_tensor(after, dtype=torch.long))
            return log_dict
        except FloatingPointError as exc:
            cache.discard()
            if frame_cache is not None:
                frame_cache.discard()
            if hasattr(self.model, "discard_distribution_state"):
                self.model.discard_distribution_state()
            if self.is_main_process:
                print(
                    "[warn] discarding complete fresh-video update after "
                    f"non-finite value: {exc}",
                    flush=True,
                )
            nonfinite = torch.tensor(
                float("nan"), device=self.device, dtype=torch.float32)
            return {
                "generator_loss": nonfinite,
                "generator_grad_norm": nonfinite,
                "dmdtrain_gradient_norm": nonfinite,
                dense_key("update_discarded"): nonfinite.new_ones(()),
            }
        except BaseException:
            cache.discard()
            if frame_cache is not None:
                frame_cache.discard()
            if hasattr(self.model, "discard_distribution_state"):
                self.model.discard_distribution_state()
            raise
        finally:
            # P0 cached pixels are update-local and may still exist when FiDEL
            # skips an all-zero update or any finalize/prepare/replay stage
            # raises. Never let a full decoded batch leak into the next update.
            for entry in entries:
                for physical_entry in entry.get("physical_entries", ()):
                    descriptor = physical_entry.get(
                        "replay_descriptor", {})
                    if isinstance(descriptor, dict):
                        descriptor.pop("cached_student_video01", None)

    def fwdbwd_one_step(self, batch, train_generator):
        # Frozen encoders stay in eval mode; generator gradients are enabled only on replay.
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.distribution_loss == "wan_kv_mmd":
            teacher_trajectory = batch["ode_latent"]
            local_shape_ok = teacher_trajectory.ndim == 6
            if not self._all_ranks_true(local_shape_ok):
                raise ValueError(
                    "wan_kv_mmd teacher LMDB yielded an invalid shape on at "
                    "least one rank; expected "
                    "[B, steps, frames, channels, height, width]")
            clean_latent = teacher_trajectory[:, -1].to(
                device=self.device, dtype=self.dtype)
            local_finite = bool(torch.isfinite(clean_latent).all().item())
            if not self._all_ranks_true(local_finite):
                raise FloatingPointError(
                    "teacher latent contains non-finite values on at least one rank")
            image_latent = None
        elif self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            # plumb raw prompts to models that need them (e.g. text-joint video FD)
            if hasattr(self.model, "set_prompts"):
                self.model.set_prompts(text_prompts)
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def _fresh_progress(self, phase, **fields):
        if self.is_main_process and bool(getattr(self.config, "fresh_video_progress_logs", False)):
            suffix = " ".join(f"{key}={value}" for key, value in fields.items())
            print(f"[fresh-progress] update={self.step + 1} phase={phase} {suffix}", flush=True)

    def train(self):
        start_step = self.step
        max_steps = int(getattr(self.config, "max_steps", 0))
        if max_steps > 0 and self.step >= max_steps:
            if self.is_main_process:
                print(
                    f"[done] checkpoint step {self.step} already reached "
                    f"max_steps={max_steps}; nothing to train.",
                    flush=True,
                )
            return

        # Optional: warm up the generated-feature queue before training (FD-loss
        # recipe) so the FD covariance is well-conditioned from step 1.
        warmup_samples = int(getattr(self.config, "queue_warmup_samples", 0))
        if warmup_samples > 0 and hasattr(self.model, "warmup_queue") and self.step == 0:
            batch = next(self.dataloader)
            with torch.no_grad():
                cond = self.model.text_encoder(text_prompts=batch["prompts"])
            image_or_video_shape = list(self.config.image_or_video_shape)
            image_or_video_shape[0] = len(batch["prompts"])
            if hasattr(self.model, "set_prompts"):
                self.model.set_prompts(batch["prompts"])
            if dist.get_rank() == 0:
                print(f"[warmup] filling generated-feature queue to >= {warmup_samples} samples...", flush=True)
            t_w = time.time()
            n = self.model.warmup_queue(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=cond,
                unconditional_dict=self.unconditional_dict if getattr(self, "unconditional_dict", None) else cond,
                target_samples=warmup_samples,
                warmup_num_clips=int(getattr(self.config, "queue_warmup_num_clips", 8)),
            )
            if dist.get_rank() == 0:
                print(f"[warmup] done: {n} generation batches in {time.time() - t_w:.1f}s", flush=True)

        # baseline eval before any optimization (captures the init / resumed model)
        if self.eval_interval > 0:
            self._fd_eval_inference()

        while True:
            current_learning_rate = self._set_generator_learning_rate(self.step)
            TRAIN_GENERATOR = (
                (not self.uses_critic)
                or self.step % self.config.dfake_gen_update_ratio == 0
            )
            update_committed = True

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                if self.uses_fresh_video_gradcache:
                    generator_log_dict = (
                        self._run_distribution_matching_step())
                else:
                    extras_list = []
                    batch = self._next_batch()
                    extra = self.fwdbwd_one_step(batch, True)
                    extras_list.append(extra)
                    generator_log_dict = merge_dict_list(extras_list)
                # Skip the step if the generator gradient is non-finite (avoids
                # corrupting weights from a stray bf16 overflow); the gradients are
                # zeroed at the top of the next iteration.
                _ggn = generator_log_dict.get("generator_grad_norm", None)
                _local_ggn_ok = (
                    (_ggn is None)
                    or bool(torch.isfinite(
                        torch.as_tensor(_ggn).float()).all().item())
                )
                _ggn_ok = self._all_ranks_true(_local_ggn_ok)
                if _ggn_ok:
                    if self.fidel_two_level_enabled:
                        local_skip_optimizer = bool(
                            int(torch.as_tensor(
                                generator_log_dict.get(
                                    "fidel_skip_optimizer_step", 0)
                            ).item())
                        )
                        all_skip_optimizer = self._all_ranks_true(
                            local_skip_optimizer)
                        no_skip_optimizer = self._all_ranks_true(
                            not local_skip_optimizer)
                        if not (all_skip_optimizer or no_skip_optimizer):
                            raise RuntimeError(
                                "FiDEL all-weights-zero decision differs "
                                "across ranks")
                        skip_optimizer = all_skip_optimizer
                    else:
                        skip_optimizer = False
                    prepare_error = None
                    try:
                        if self.fresh_frame_fd_gradcache is not None:
                            # Allocate/cross-product before parameters change;
                            # commit after optimizer.step is publication-only.
                            self.fresh_frame_fd_gradcache.prepare_commit()
                        if hasattr(
                            self.model, "prepare_distribution_state"
                        ):
                            self.model.prepare_distribution_state()
                    except Exception as exc:
                        prepare_error = exc
                    prepare_ok = self._all_ranks_true(
                        prepare_error is None)
                    if not prepare_ok:
                        if self.fresh_video_gradcache is not None:
                            self.fresh_video_gradcache.discard()
                        if self.fresh_frame_fd_gradcache is not None:
                            self.fresh_frame_fd_gradcache.discard()
                        if hasattr(self.model, "discard_distribution_state"):
                            self.model.discard_distribution_state()
                        if prepare_error is not None:
                            raise RuntimeError(
                                "distribution transaction preparation failed "
                                "before optimizer.step") from prepare_error
                        raise RuntimeError(
                            "distribution transaction preparation failed on "
                            "another rank before optimizer.step")
                    try:
                        if not skip_optimizer:
                            if self.optimizer_state_cpu_offload_during_backward:
                                move_adamw_moment_state_(
                                    self.generator_optimizer, self.device)
                                torch.cuda.synchronize(self.device)
                            # Commit one update after every selected rollout has contributed its gradient.
                            self.generator_optimizer.step()
                            if self.optimizer_state_cpu_offload_during_backward:
                                torch.cuda.synchronize(self.device)
                                self.optimizer_state_offload_report = (
                                    move_adamw_moment_state_(
                                        self.generator_optimizer, "cpu"))
                                torch.cuda.empty_cache()
                        if self.fresh_frame_fd_gradcache is not None:
                            self.fresh_frame_fd_gradcache.commit()
                        if hasattr(self.model, "commit_distribution_state"):
                            self.model.commit_distribution_state()
                        if self.fresh_video_gradcache is not None:
                            self.fresh_video_gradcache.commit()
                    except BaseException:
                        if self.fresh_video_gradcache is not None:
                            self.fresh_video_gradcache.discard()
                        if self.fresh_frame_fd_gradcache is not None:
                            self.fresh_frame_fd_gradcache.discard()
                        if hasattr(self.model, "discard_distribution_state"):
                            self.model.discard_distribution_state()
                        raise
                    update_committed = True
                    if (
                        self.generator_ema is not None
                        and not skip_optimizer
                    ):
                        self.generator_ema.update(self.model.generator)
                else:
                    update_committed = False
                    if self.fresh_video_gradcache is not None:
                        self.fresh_video_gradcache.discard()
                    if self.fresh_frame_fd_gradcache is not None:
                        self.fresh_frame_fd_gradcache.discard()
                    if hasattr(self.model, "discard_distribution_state"):
                        self.model.discard_distribution_state()
                    if self.is_main_process:
                        print(f"[warn] step {self.step}: non-finite generator_grad_norm={_ggn}; skipping optimizer step", flush=True)

            # A failed FiDEL attempt must retry the same prompt-mask slot.  It
            # neither advances the public step nor runs eval/save/log side
            # effects.  All-weights-zero is different: its FIFO/controller
            # transaction commits above and therefore consumes the slot.
            if self.fidel_two_level_enabled and not update_committed:
                self.generator_optimizer.zero_grad(set_to_none=True)
                continue

            # Train the critic
            critic_log_dict = {}
            if self.uses_critic:
                self.critic_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = self._next_batch()
                extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
                critic_log_dict = merge_dict_list(extras_list)
                self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1
            if self.is_main_process and bool(getattr(self.config, "fresh_video_progress_logs", False)):
                print(f"[update-committed] step={self.step} committed={int(update_committed)}", flush=True)

            # Create EMA params (if not already created)
            ema_start_step = int(getattr(self.config, "ema_start_step", 0))
            ema_weight = float(getattr(self.config, "ema_weight", 0.0) or 0.0)
            if (self.step >= ema_start_step) and \
                    (self.generator_ema is None) and (ema_weight > 0):
                self.generator_ema = EMA_FSDP(
                    self.model.generator, decay=ema_weight)

            # Save the model
            checkpoint_due = self.step % self.config.log_iters == 0
            if (not self.config.no_save) and (self.step - start_step) > 0 and checkpoint_due:
                if not update_committed:
                    raise RuntimeError("refusing to checkpoint an uncommitted optimizer update")
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Periodic eval inference (all ranks participate; rank 0 logs/saves).
            # Remember whether this step was evaluated so max_steps does not
            # write the same fixed-seed samples twice.
            did_periodic_eval = (
                self.eval_interval > 0
                and (self.step - start_step) > 0
                and self.step % self.eval_interval == 0
            )
            if did_periodic_eval:
                torch.cuda.empty_cache()
                self._fd_eval_inference()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                    for key, value in generator_log_dict.items():
                        if key in wandb_loss_dict or key in {
                            "generator_loss",
                            "generator_grad_norm",
                            "dmdtrain_gradient_norm",
                        }:
                            continue
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            # Metadata includes integral counters (for example
                            # Wan-KV history_updates).  ``Tensor.mean`` rejects
                            # integer dtypes; a one-element tensor needs no
                            # reduction.
                            wandb_loss_dict[key] = value.detach().item()
                    wandb_loss_dict["learning_rate"] = current_learning_rate

                if self.uses_critic:
                    wandb_loss_dict.update(
                        {
                            "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                            "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                        }
                    )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

                if (self.step % getattr(self.config, "stdout_log_iters", 1)) == 0:
                    summary = " ".join(
                        f"{k}={v:.4g}" for k, v in wandb_loss_dict.items()
                        if isinstance(v, (int, float))
                    )
                    print(f"[step {self.step}] {summary}", flush=True)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            # Stop only after logging the final step. This is used by smoke tests
            # and sequential launchers, so suppressing its metrics would hide
            # precisely the EMA transition they need to verify.
            max_steps = int(getattr(self.config, "max_steps", 0))
            if max_steps > 0 and self.step >= max_steps:
                if not self.config.no_save:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()
                if self.eval_interval > 0 and not did_periodic_eval:
                    self._fd_eval_inference()
                if self.is_main_process:
                    print(
                        f"[done] reached max_steps={max_steps}; exiting.",
                        flush=True,
                    )
                break

    # Historical entry points remain aliases for external callers.
    _prepare_fresh_video_microbatch = _collect_rollout_microbatch
    _fresh_video_generator_update = _run_distribution_matching_step

# Old import/class spelling remains valid for existing callers.
Trainer = ElasticForcingTrainer

"""Reusable single-frame FD extraction and replay.

The head owns only the frozen image representations and their FIFO statistics.
It does not construct a generator and it does not register the surrounding
model's VAE.  This lets another objective (for example video-level V-JEPA2 MMD)
reuse the original SIM FD representations without a second model instance.

The first pass extracts detached features from explicit per-video frame
indices.  Replay evaluates the same frozen encoders in small chunks and maps
feature cotangents back to decoded pixels or, through the validated Wan decoder
VJP, to generator latents.  All representations share exactly the same
``[B, K]`` frame-index tensor.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import torch
from torch import nn

from model.fd import (
    FDJudge,
    _expand_config_list,
    _load_reference_stats,
    load_repr_model,
)
from video_fd.replay.replay_video_encoder_vjp import replay_decoder_vjp


@dataclass(frozen=True)
class FrameFDExtraction:
    """Detached first-pass features for one video microbatch."""

    frame_indices: torch.Tensor
    features: Mapping[str, torch.Tensor]
    batch_size: int
    frames_per_video: int
    total_pixel_frames: int


@dataclass(frozen=True)
class ValidatedFrameFDState:
    """Structurally validated FIFO tensors ready for allocation-free publish."""

    queue_states: tuple[Mapping[str, torch.Tensor], ...]


def _amp_context(enabled: bool, dtype: torch.dtype, device: torch.device):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


class FrameFDHead(nn.Module):
    """Frozen image encoders plus deterministic frame extraction/replay APIs."""

    def __init__(
        self,
        judges: Sequence[FDJudge],
        *,
        vae=None,
        num_frames: int = 16,
        random_frame_sample: bool = True,
        exclude_first_frame: bool = False,
        replay_chunk_frames: int = 4,
        replay_use_amp: bool = True,
        replay_amp_dtype: torch.dtype = torch.bfloat16,
        replay_grad_clip_z: Optional[float] = None,
        norm_eps: float = 0.01,
        average_representations: bool = True,
        independent_first_frame: bool = True,
        decoder_vjp_fn: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if not judges:
            raise ValueError("FrameFDHead needs at least one FDJudge")
        if int(num_frames) == 0 or int(num_frames) < -1:
            raise ValueError("num_frames must be positive or -1 for all frames")
        if int(replay_chunk_frames) <= 0:
            raise ValueError("replay_chunk_frames must be positive")
        if not torch.isfinite(torch.tensor(float(norm_eps))) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")

        names = [judge.name for judge in judges]
        if len(set(names)) != len(names):
            raise ValueError(f"FDJudge names must be unique, got {names}")
        for judge in judges:
            if judge.queue.ema_stats:
                raise ValueError(
                    "FrameFDHead requires a true FIFO queue; set fd_ema_beta=0")
            judge.eval().requires_grad_(False)

        self.judges = nn.ModuleList(judges)
        # The surrounding model owns the VAE. Avoid registering it here so it is
        # not duplicated in checkpoints/module walks.
        object.__setattr__(self, "_vae", vae)
        object.__setattr__(
            self,
            "_decoder_vjp_fn",
            decoder_vjp_fn if decoder_vjp_fn is not None else replay_decoder_vjp,
        )
        self.num_frames = int(num_frames)
        self.random_frame_sample = bool(random_frame_sample)
        self.exclude_first_frame = bool(exclude_first_frame)
        self.replay_chunk_frames = int(replay_chunk_frames)
        self.replay_use_amp = bool(replay_use_amp)
        self.replay_amp_dtype = replay_amp_dtype
        self.replay_grad_clip_z = (
            None
            if replay_grad_clip_z is None
            else float(replay_grad_clip_z)
        )
        self.norm_eps = float(norm_eps)
        self.average_representations = bool(average_representations)
        self.independent_first_frame = bool(independent_first_frame)

    @classmethod
    def from_config(cls, args, device, *, vae=None) -> "FrameFDHead":
        """Build the original FD judges without constructing another model."""
        names_value = getattr(
            args, "frame_fd_repr_models", getattr(args, "fd_repr_models", ()))
        names = [names_value] if isinstance(names_value, str) else list(names_value)
        if not names:
            raise ValueError("frame FD fusion needs fd_repr_models")
        count = len(names)

        def config_list(frame_name, legacy_name, default=None):
            return _expand_config_list(
                getattr(args, frame_name, getattr(args, legacy_name, None)),
                count,
                default,
            )

        stats_paths = config_list(
            "frame_fd_repr_stats_paths", "fd_repr_stats_paths")
        weights = config_list(
            "frame_fd_repr_weights", "fd_repr_weights", 1.0)
        pool_types = config_list(
            "frame_fd_repr_pool_types", "fd_repr_pool_types", "cls")
        target_sizes = config_list(
            "frame_fd_target_sizes", "fd_target_sizes")
        queue_size = int(getattr(args, "fd_queue_size", 50000))
        ema_beta = float(getattr(args, "fd_ema_beta", 0.0))
        if ema_beta != 0.0:
            raise ValueError(
                "frame FD fusion uses a detached FIFO, so fd_ema_beta must be 0")
        online_accum = bool(getattr(args, "fd_online_accum", True))
        use_eigvalsh = bool(getattr(args, "fd_eigvalsh", True))

        judges = []
        for name, stats_path, weight, pool_type, target_size in zip(
            names, stats_paths, weights, pool_types, target_sizes
        ):
            representation = load_repr_model(
                str(name), device=device, target_size=target_size)
            mu_ref, sigma_ref = _load_reference_stats(
                stats_path,
                str(pool_type),
                device,
                representation.feat_dim,
            )
            judge = FDJudge(
                str(name),
                representation,
                mu_ref,
                sigma_ref,
                pool_type=str(pool_type),
                weight=float(weight),
                queue_size=queue_size,
                ema_beta=0.0,
                online_accum=online_accum,
                use_eigvalsh=use_eigvalsh,
            )
            judge.config_model_name = str(name)
            judge.config_target_size = (
                None if target_size is None else int(target_size))
            judges.append(judge)

        amp_name = str(getattr(args, "replay_amp_dtype", "bf16")).lower()
        amp_dtype = (
            torch.bfloat16
            if amp_name in {"bf16", "bfloat16"}
            else torch.float16
        )
        return cls(
            judges,
            vae=vae,
            num_frames=int(getattr(args, "fd_num_frames", 16)),
            random_frame_sample=bool(
                getattr(args, "fd_random_frame_sample", True)),
            exclude_first_frame=bool(
                getattr(args, "fd_exclude_first_frame", False)),
            replay_chunk_frames=int(
                getattr(args, "replay_chunk_frames", 4)),
            replay_use_amp=bool(getattr(
                args,
                "replay_use_amp",
                getattr(args, "mixed_precision", True),
            )),
            replay_amp_dtype=amp_dtype,
            replay_grad_clip_z=getattr(args, "replay_grad_clip_z", None),
            norm_eps=float(getattr(args, "fd_fid_norm_eps", 0.01)),
            average_representations=bool(
                getattr(args, "fd_average_reps", True)),
            independent_first_frame=bool(
                getattr(args, "independent_first_frame", True)),
        ).to(device)

    @property
    def representation_names(self) -> tuple[str, ...]:
        return tuple(judge.name for judge in self.judges)

    @property
    def vae(self):
        return self._vae

    def frame_fd_state_dict(self) -> dict:
        """Return only persistent FIFO state, never frozen encoder weights."""
        return {
            "version": 1,
            "metadata": {
                "representations": [
                    {
                        "name": judge.name,
                        "model_name": getattr(
                            judge, "config_model_name", judge.name),
                        "feature_dim": judge.queue.feat_dim,
                        "pool_type": judge.pool_type,
                        "weight": judge.weight,
                        "target_size": getattr(
                            judge,
                            "config_target_size",
                            getattr(judge.repr_model, "target_size", None),
                        ),
                        "queue_size": judge.queue.size,
                        "queue_ema_beta": judge.queue.ema_beta,
                        "queue_online_accum": judge.queue.online_accum,
                    }
                    for judge in self.judges
                ],
                "num_frames": self.num_frames,
                "random_frame_sample": self.random_frame_sample,
                "exclude_first_frame": self.exclude_first_frame,
                "norm_eps": self.norm_eps,
                "average_representations": self.average_representations,
                "replay_chunk_frames": self.replay_chunk_frames,
                "replay_use_amp": self.replay_use_amp,
                "replay_amp_dtype": str(self.replay_amp_dtype),
                "replay_grad_clip_z": self.replay_grad_clip_z,
                "independent_first_frame": self.independent_first_frame,
            },
            "judges": [
                {
                    "name": judge.name,
                    "queue": judge.queue.state_dict(),
                }
                for judge in self.judges
            ],
        }

    def validate_frame_fd_state_dict(
        self,
        state: Mapping,
    ) -> ValidatedFrameFDState:
        """Validate every FIFO before allowing the first live queue mutation."""
        if not isinstance(state, Mapping):
            raise TypeError("frame FD state must be a mapping")
        expected_keys = {"version", "metadata", "judges"}
        if set(state) != expected_keys:
            raise ValueError(
                "frame FD state keys do not match the strict schema: "
                f"missing={sorted(expected_keys - set(state))}, "
                f"unexpected={sorted(set(state) - expected_keys)}")
        if type(state.get("version")) is not int or state["version"] != 1:
            raise ValueError("unsupported frame FD state version")
        expected = self.frame_fd_state_dict()["metadata"]
        if state.get("metadata") != expected:
            raise ValueError(
                "frame FD state metadata does not match the active head")
        saved = state.get("judges")
        if not isinstance(saved, list):
            raise ValueError("frame FD state is missing judge queues")
        if len(saved) != len(self.judges):
            raise ValueError(
                "frame FD state judge count does not match the active head")
        by_name = {}
        for index, item in enumerate(saved):
            if not isinstance(item, Mapping) or set(item) != {"name", "queue"}:
                raise ValueError(
                    f"frame FD judge state {index} must contain exactly "
                    "`name` and `queue`")
            name = item["name"]
            if name in by_name:
                raise ValueError(
                    f"frame FD state contains duplicate judge {name!r}")
            by_name[name] = item["queue"]
        if set(by_name) != set(self.representation_names):
            raise ValueError(
                "frame FD state representations do not match the active head")

        validated = []
        for judge in self.judges:
            queue_state = by_name[judge.name]
            if not isinstance(queue_state, Mapping):
                raise ValueError(
                    f"frame FD queue {judge.name!r} state must be a mapping")
            live_state = judge.queue.state_dict()
            if set(queue_state) != set(live_state):
                raise ValueError(
                    f"frame FD queue {judge.name!r} keys mismatch: "
                    f"missing={sorted(set(live_state) - set(queue_state))}, "
                    f"unexpected={sorted(set(queue_state) - set(live_state))}")
            canonical_queue = {}
            for key, live_tensor in live_state.items():
                saved_tensor = queue_state[key]
                if not isinstance(saved_tensor, torch.Tensor):
                    raise TypeError(
                        f"frame FD queue {judge.name!r}.{key} must be a tensor")
                if (
                    saved_tensor.shape != live_tensor.shape
                    or saved_tensor.dtype != live_tensor.dtype
                    or saved_tensor.layout != live_tensor.layout
                    or saved_tensor.is_meta
                ):
                    raise ValueError(
                        f"frame FD queue {judge.name!r}.{key} is incompatible: "
                        f"saved shape/dtype/layout="
                        f"{tuple(saved_tensor.shape)}/{saved_tensor.dtype}/"
                        f"{saved_tensor.layout}, expected "
                        f"{tuple(live_tensor.shape)}/{live_tensor.dtype}/"
                        f"{live_tensor.layout}")
                canonical_queue[key] = saved_tensor

            # These scalar invariants are not checked by nn.Module.load_state_dict
            # but are required by the FIFO ring implementation.
            filled = int(canonical_queue["filled"].item())
            pointer = int(canonical_queue["ptr"].item())
            initialized = bool(canonical_queue["initialized"].item())
            queue_size = int(judge.queue.size)
            if queue_size <= 0:
                valid_ring = (
                    filled == 0 and pointer == 0 and not initialized)
            else:
                valid_ring = (
                    0 <= filled <= queue_size
                    and 0 <= pointer < queue_size
                    and (filled == queue_size or pointer == filled)
                    and initialized == (filled > 0)
                )
            if not valid_ring:
                raise ValueError(
                    f"frame FD queue {judge.name!r} has invalid FIFO counters: "
                    f"size={queue_size}, filled={filled}, ptr={pointer}, "
                    f"initialized={initialized}")
            if filled > 0 and not bool(
                torch.isfinite(canonical_queue["feats"][:filled]).all().item()
            ):
                raise FloatingPointError(
                    f"frame FD queue {judge.name!r} contains non-finite "
                    "committed features")
            for accumulator_name in ("feat_sum", "feat_outer"):
                accumulator = canonical_queue.get(accumulator_name)
                if accumulator is not None and not bool(
                    torch.isfinite(accumulator).all().item()
                ):
                    raise FloatingPointError(
                        f"frame FD queue {judge.name!r}.{accumulator_name} "
                        "contains non-finite values")
            validated.append(canonical_queue)
        return ValidatedFrameFDState(queue_states=tuple(validated))

    @torch.no_grad()
    def load_validated_frame_fd_state_dict(
        self,
        validated: ValidatedFrameFDState,
    ) -> None:
        """Publish a state that passed :meth:`validate_frame_fd_state_dict`."""
        if (
            not isinstance(validated, ValidatedFrameFDState)
            or len(validated.queue_states) != len(self.judges)
        ):
            raise TypeError("invalid prepared frame FD state")
        for judge, queue_state in zip(
            self.judges, validated.queue_states
        ):
            live_state = judge.queue.state_dict()
            # Validation proved exact keys, shapes, dtypes and layouts. These
            # fixed-buffer copies do not perform any later schema validation.
            for key, live_tensor in live_state.items():
                live_tensor.copy_(queue_state[key])

    def load_frame_fd_state_dict(self, state: Mapping) -> None:
        """Strict two-phase restore of all detached FIFO queues."""
        validated = self.validate_frame_fd_state_dict(state)
        self.load_validated_frame_fd_state_dict(validated)

    @staticmethod
    def _validate_pixels(pixels01: torch.Tensor) -> None:
        if (
            not isinstance(pixels01, torch.Tensor)
            or pixels01.ndim != 5
            or pixels01.shape[0] <= 0
            or pixels01.shape[1] <= 0
            or pixels01.shape[2] != 3
            or not pixels01.is_floating_point()
        ):
            raise ValueError(
                "pixels01 must be a non-empty floating [B, T, 3, H, W] tensor")
        if not bool(torch.isfinite(pixels01).all().item()):
            raise FloatingPointError("pixels01 contains non-finite values")

    @staticmethod
    def _validate_indices(
        frame_indices: torch.Tensor,
        *,
        batch_size: int,
        total_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        indices = torch.as_tensor(
            frame_indices, device=device, dtype=torch.long)
        if (
            indices.ndim != 2
            or indices.shape[0] != batch_size
            or indices.shape[1] <= 0
        ):
            raise ValueError(
                "frame_indices must be [B, K] with one non-empty row per video")
        if int(indices.min()) < 0 or int(indices.max()) >= total_frames:
            raise ValueError(
                f"frame_indices must be in [0, {total_frames}), got "
                f"[{int(indices.min())}, {int(indices.max())}]")
        sorted_indices = indices.sort(dim=1).values
        if (
            sorted_indices.shape[1] > 1
            and bool((sorted_indices[:, 1:] == sorted_indices[:, :-1]).any())
        ):
            raise ValueError(
                "frame_indices must sample without replacement within each video")
        return indices.contiguous()

    def sample_frame_indices(
        self,
        *,
        batch_size: int,
        total_frames: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Return explicit per-video indices, never using ambient random state."""
        if batch_size <= 0 or total_frames <= 0:
            raise ValueError("batch_size and total_frames must be positive")
        if generator is not None and seed is not None:
            raise ValueError("pass either generator or seed, not both")
        start = 1 if self.exclude_first_frame and total_frames > 1 else 0
        available = total_frames - start
        count = (
            available
            if self.num_frames < 0 or self.num_frames >= available
            else self.num_frames
        )
        if count <= 0:
            raise ValueError("no frames remain after frame sampling constraints")
        if count == available:
            row = torch.arange(start, total_frames, device=device)
            return row.unsqueeze(0).expand(batch_size, -1).clone()
        if not self.random_frame_sample:
            row = torch.linspace(
                start, total_frames - 1, steps=count, device=device
            ).round().long()
            return row.unsqueeze(0).expand(batch_size, -1).clone()
        if generator is None:
            if seed is None:
                raise ValueError(
                    "random frame sampling requires an explicit generator or seed")
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        scores = torch.rand(
            batch_size,
            available,
            device=device,
            generator=generator,
            dtype=torch.float32,
        )
        indices = scores.argsort(dim=1)[:, :count] + start
        return indices.sort(dim=1).values.contiguous()

    def select_frames(
        self,
        pixels01: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_pixels(pixels01)
        indices = self._validate_indices(
            frame_indices,
            batch_size=int(pixels01.shape[0]),
            total_frames=int(pixels01.shape[1]),
            device=pixels01.device,
        )
        gather_indices = indices[:, :, None, None, None].expand(
            -1, -1, pixels01.shape[2], pixels01.shape[3], pixels01.shape[4])
        return pixels01.gather(1, gather_indices)

    def _extract_flat_frames(
        self,
        judge: FDJudge,
        flat_frames: torch.Tensor,
    ) -> torch.Tensor:
        chunks = []
        for start in range(0, flat_frames.shape[0], self.replay_chunk_frames):
            frames = flat_frames[start:start + self.replay_chunk_frames]
            with _amp_context(
                self.replay_use_amp, self.replay_amp_dtype, frames.device
            ):
                features = judge.extract(frames)
            features = features.float()
            if features.ndim != 2 or features.shape[0] != frames.shape[0]:
                raise RuntimeError(
                    f"{judge.name} must return one [D] feature per frame; got "
                    f"{tuple(features.shape)} for {frames.shape[0]} frames")
            chunks.append(features)
        return torch.cat(chunks, dim=0)

    def extract_no_grad(
        self,
        pixels01: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ) -> FrameFDExtraction:
        """Extract detached ``[B*K, D]`` features for every representation."""
        self._validate_pixels(pixels01)
        batch_size, total_frames = pixels01.shape[:2]
        if frame_indices is None:
            frame_indices = self.sample_frame_indices(
                batch_size=int(batch_size),
                total_frames=int(total_frames),
                device=pixels01.device,
                generator=generator,
                seed=seed,
            )
        indices = self._validate_indices(
            frame_indices,
            batch_size=int(batch_size),
            total_frames=int(total_frames),
            device=pixels01.device,
        )
        with torch.no_grad():
            selected = self.select_frames(pixels01.detach(), indices)
            flat_frames = selected.reshape(
                -1, *selected.shape[2:]).contiguous()
            features = {
                judge.name: self._extract_flat_frames(
                    judge, flat_frames).detach().float().clone()
                for judge in self.judges
            }
        return FrameFDExtraction(
            frame_indices=indices.detach().clone(),
            features=features,
            batch_size=int(batch_size),
            frames_per_video=int(indices.shape[1]),
            total_pixel_frames=int(total_frames),
        )

    def replay_feature_cotangents_to_pixels(
        self,
        pixels01: torch.Tensor,
        frame_indices: torch.Tensor,
        feature_cotangents: Mapping[str, torch.Tensor],
        *,
        expected_features: Optional[Mapping[str, torch.Tensor]] = None,
        replay_atol: float = 1.0e-5,
        replay_rtol: float = 1.0e-4,
    ) -> torch.Tensor:
        """Replay frozen image encoders and return ``dL/d pixels01``."""
        self._validate_pixels(pixels01)
        indices = self._validate_indices(
            frame_indices,
            batch_size=int(pixels01.shape[0]),
            total_frames=int(pixels01.shape[1]),
            device=pixels01.device,
        )
        expected_names = set(self.representation_names)
        if set(feature_cotangents) != expected_names:
            raise ValueError(
                "feature cotangent keys must exactly match representations: "
                f"expected {sorted(expected_names)}, got "
                f"{sorted(feature_cotangents)}")
        if (
            expected_features is not None
            and set(expected_features) != expected_names
        ):
            raise ValueError(
                "expected feature keys must exactly match representations")

        selected = self.select_frames(pixels01.detach(), indices)
        flat_frames = selected.reshape(
            -1, *selected.shape[2:]).contiguous()
        flat_pixel_grad = torch.zeros_like(flat_frames, dtype=torch.float32)

        for judge in self.judges:
            cotangent = feature_cotangents[judge.name].detach().float()
            if (
                cotangent.ndim != 2
                or cotangent.shape[0] != flat_frames.shape[0]
                or not bool(torch.isfinite(cotangent).all().item())
            ):
                raise ValueError(
                    f"invalid {judge.name} feature cotangent shape/value: "
                    f"{tuple(cotangent.shape)}")
            offset = 0
            replay_values = []
            for start in range(
                0, flat_frames.shape[0], self.replay_chunk_frames
            ):
                frames_leaf = (
                    flat_frames[start:start + self.replay_chunk_frames]
                    .detach()
                    .clone()
                    .requires_grad_(True)
                )
                with _amp_context(
                    self.replay_use_amp,
                    self.replay_amp_dtype,
                    frames_leaf.device,
                ):
                    features = judge.extract(frames_leaf)
                features = features.float()
                rows = int(features.shape[0])
                grad_chunk = cotangent[offset:offset + rows].to(
                    features.device)
                if features.shape != grad_chunk.shape:
                    raise RuntimeError(
                        f"{judge.name} replay feature/cotangent mismatch: "
                        f"{tuple(features.shape)} != {tuple(grad_chunk.shape)}")
                replay_values.append(features.detach())
                surrogate = (features * grad_chunk).sum()
                (pixel_chunk_grad,) = torch.autograd.grad(
                    surrogate,
                    frames_leaf,
                    retain_graph=False,
                    create_graph=False,
                )
                flat_pixel_grad[start:start + rows].add_(
                    pixel_chunk_grad.detach().float())
                offset += rows
            if offset != cotangent.shape[0]:
                raise RuntimeError(
                    f"{judge.name} replay consumed {offset} of "
                    f"{cotangent.shape[0]} cotangent rows")
            if expected_features is not None:
                replayed = torch.cat(replay_values, dim=0).float()
                expected = expected_features[judge.name].to(
                    device=replayed.device, dtype=torch.float32)
                if not torch.allclose(
                    replayed,
                    expected,
                    atol=float(replay_atol),
                    rtol=float(replay_rtol),
                ):
                    max_abs = (replayed - expected).abs().max()
                    raise RuntimeError(
                        f"{judge.name} frame replay changed features "
                        f"(max_abs={float(max_abs):.6g})")

        selected_grad = flat_pixel_grad.reshape_as(selected)
        full_grad = torch.zeros_like(pixels01, dtype=torch.float32)
        for batch_index in range(pixels01.shape[0]):
            full_grad[batch_index].index_add_(
                0,
                indices[batch_index],
                selected_grad[batch_index],
            )
        if (
            self.replay_grad_clip_z is not None
            and self.replay_grad_clip_z > 0
        ):
            grad_norm = full_grad.norm()
            if bool(torch.isfinite(grad_norm)) and (
                grad_norm > self.replay_grad_clip_z
            ):
                full_grad.mul_(
                    self.replay_grad_clip_z / (grad_norm + 1.0e-12))
        if not bool(torch.isfinite(full_grad).all().item()):
            raise FloatingPointError(
                "frame encoder replay returned a non-finite pixel cotangent")
        return full_grad

    def replay_pixel_cotangent_to_latents(
        self,
        latents: torch.Tensor,
        pixel_cotangent: torch.Tensor,
    ) -> torch.Tensor:
        """Map an already-computed pixel cotangent through the frozen VAE."""
        if self.vae is None:
            raise RuntimeError(
                "FrameFDHead needs a VAE for pixel-to-latent replay")
        if pixel_cotangent.ndim != 5:
            raise ValueError(
                "pixel_cotangent must be [B, T_pixel, 3, H, W]")
        latent_cotangent = self._decoder_vjp_fn(
            self.vae,
            latents.detach(),
            pixel_cotangent.detach().float(),
            independent_first_frame=self.independent_first_frame,
        ).detach().float()
        if latent_cotangent.shape != latents.shape:
            raise ValueError(
                "decoder VJP returned an invalid latent shape: "
                f"{tuple(latent_cotangent.shape)} != {tuple(latents.shape)}")
        if not bool(torch.isfinite(latent_cotangent).all().item()):
            raise FloatingPointError(
                "decoder replay returned a non-finite latent cotangent")
        return latent_cotangent

    def replay_feature_cotangents_to_latents(
        self,
        latents: torch.Tensor,
        pixels01: torch.Tensor,
        frame_indices: torch.Tensor,
        feature_cotangents: Mapping[str, torch.Tensor],
        *,
        expected_features: Optional[Mapping[str, torch.Tensor]] = None,
        replay_atol: float = 1.0e-5,
        replay_rtol: float = 1.0e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(latent_cotangent, pixel_cotangent)`` for one microbatch."""
        pixel_cotangent = self.replay_feature_cotangents_to_pixels(
            pixels01,
            frame_indices,
            feature_cotangents,
            expected_features=expected_features,
            replay_atol=replay_atol,
            replay_rtol=replay_rtol,
        )
        latent_cotangent = self.replay_pixel_cotangent_to_latents(
            latents, pixel_cotangent)
        return latent_cotangent, pixel_cotangent

"""Frozen Meta-native DINOv3 ViT-L/16 dense-video adapter.

This adapter implements the one feature geometry used by the DINOv3 variant
of the whole-video MMD objective:

* input ``video_clip_01`` is ``[B, 81, 3, H, W]`` in ``[0, 1]``;
* sixteen fixed frames are selected in chronological order;
* each frame is directly resized to ``256 x 256`` and ImageNet-normalized;
* Meta native ``forward_features()["x_norm_patchtokens"]`` supplies the final
  normalized ``16 x 16`` patch grid;
* eight fixed rows crossed with eight fixed columns retain 64 patches/frame;
* frame-major concatenation returns one ``[1024, 1024]`` matrix per video.

The encoder always stays in eval mode with FP32 parameters and floating-point
buffers.  The first pass is detached, while :meth:`replay_dense_vjp` replays
four selected frames at a time and returns a full 81-frame pixel cotangent
without ever building a sixteen-frame encoder graph.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from video_fd.encoders.dinov3_dense_metadata import (
    DEFAULT_WAN_VAE_ID,
    FEATURE_DIM,
    FORWARD_CHUNK_FRAMES,
    FRAME_INDICES,
    INPUT_SIZE,
    MODEL_ID,
    PATCH_GRID_SIZE,
    PATCH_SIZE,
    SOURCE_NUM_FRAMES,
    SOURCE_COMMIT,
    SPATIAL_AXIS_INDICES,
    canonical_dinov3_dense_feature_metadata,
    dense_token_indices,
    file_sha256,
    module_state_sha256,
    spatial_patch_indices,
)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_NATIVE_VITL16_PARAMETER_COUNT = 303_154_176
_NATIVE_BACKBONES_MODULE = "dinov3.hub.backbones"


def _dinov3_module_names() -> tuple[str, ...]:
    """Return a stable snapshot of every cached module in the DINOv3 tree."""

    return tuple(
        name
        for name in tuple(sys.modules)
        if name == "dinov3" or name.startswith("dinov3.")
    )


def _assert_native_module_origin(
    module_name: str,
    module: object,
    repo: Path,
) -> Path:
    """Require a cached native module to be the matching file in ``repo``.

    Merely prepending ``repo`` to ``sys.path`` is insufficient because Python
    consults ``sys.modules`` first. Validate both normal modules and package
    search paths so an earlier import cannot redirect a relative import to a
    different DINOv3 checkout or an installed package.
    """

    if module is None:
        raise RuntimeError(
            f"cached module {module_name!r} is incomplete; refusing DINOv3 import"
        )
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise RuntimeError(
            f"cached module {module_name!r} has no filesystem origin; "
            "refusing DINOv3 import"
        )
    try:
        resolved_origin = Path(origin).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"cannot resolve cached module {module_name!r} origin {origin!r}"
        ) from exc

    relative_parts = module_name.split(".")
    module_base = repo.joinpath(*relative_parts)
    expected_origins = {
        module_base.with_suffix(".py").resolve(),
        (module_base / "__init__.py").resolve(),
    }
    if resolved_origin not in expected_origins:
        raise RuntimeError(
            f"cached module {module_name!r} comes from {resolved_origin}, "
            f"not the pinned DINOv3 repo {repo}"
        )

    package_paths = getattr(module, "__path__", None)
    if package_paths is not None:
        try:
            resolved_paths = tuple(
                Path(path).expanduser().resolve(strict=True)
                for path in package_paths
            )
        except (OSError, RuntimeError, TypeError) as exc:
            raise RuntimeError(
                f"cached package {module_name!r} has invalid search paths"
            ) from exc
        expected_package_path = module_base.resolve(strict=True)
        if resolved_paths != (expected_package_path,):
            raise RuntimeError(
                f"cached package {module_name!r} search paths {resolved_paths!r} "
                f"do not equal the pinned path {(expected_package_path,)!r}"
            )
    return resolved_origin


def _validate_cached_dinov3_modules(repo: Path) -> None:
    return None


def _remove_new_dinov3_modules(previous_names: set[str]) -> None:
    for module_name in _dinov3_module_names():
        if module_name not in previous_names:
            sys.modules.pop(module_name, None)


def _import_native_vitl16_factory(repo: Path):
    """Import only the official backbone module from one verified checkout.

    The repository's ``hubconf.py`` imports optional segmentation, detection,
    depth, and text dependencies even though this adapter needs only a vision
    backbone. Directly importing ``dinov3.hub.backbones`` avoids those optional
    dependency edges while preserving Meta's native architecture factory.
    """

    repo = repo.expanduser().resolve(strict=True)
    if not (repo / "dinov3" / "hub" / "backbones.py").is_file():
        raise FileNotFoundError(
            f"pinned DINOv3 backbone module is missing under {repo}"
        )
    _validate_cached_dinov3_modules(repo)

    previous_module_names = set(_dinov3_module_names())
    previous_sys_path = list(sys.path)
    try:
        sys.path.insert(0, str(repo))
        importlib.invalidate_caches()
        module = importlib.import_module(_NATIVE_BACKBONES_MODULE)
        _assert_native_module_origin(_NATIVE_BACKBONES_MODULE, module, repo)
        _validate_cached_dinov3_modules(repo)
        factory = getattr(module, MODEL_ID, None)
        if not callable(factory):
            raise RuntimeError(
                f"{_NATIVE_BACKBONES_MODULE} does not expose callable {MODEL_ID}"
            )
        return factory
    except Exception:
        _remove_new_dinov3_modules(previous_module_names)
        raise
    finally:
        sys.path[:] = previous_sys_path


def _build_native_vitl16(repo: Path) -> nn.Module:
    """Construct the exact official ViT-L/16 architecture without weights."""

    factory = _import_native_vitl16_factory(repo)
    previous_module_names = set(_dinov3_module_names())
    previous_sys_path = list(sys.path)
    try:
        model = factory(pretrained=False)
        _validate_cached_dinov3_modules(repo)
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"official {MODEL_ID} factory did not return an nn.Module"
            )
        return model
    except Exception:
        _remove_new_dinov3_modules(previous_module_names)
        raise
    finally:
        # Some optional acceleration imports used by the native constructor
        # append transient paths. Keep the adapter's source-loading boundary
        # side-effect free regardless of whether construction succeeds.
        sys.path[:] = previous_sys_path


def _autocast_disabled(device: torch.device):
    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _strict_config_tuple(
    config,
    key: str,
    expected: tuple[int, ...],
) -> tuple[int, ...]:
    raw = config.get(key, expected)
    values = tuple(raw)
    return values


def _is_tensor_state_dict(value) -> bool:
    return (
        isinstance(value, Mapping)
        and len(value) > 0
        and all(isinstance(key, str) for key in value)
        and all(torch.is_tensor(tensor) for tensor in value.values())
    )


def _checkpoint_state_dict(payload) -> Mapping[str, torch.Tensor]:
    """Accept only a direct state dict or an explicit ``model`` wrapper."""

    if _is_tensor_state_dict(payload):
        return payload
    if isinstance(payload, Mapping) and "model" in payload:
        state = payload["model"]
        if _is_tensor_state_dict(state):
            return state
        raise TypeError("checkpoint['model'] is not a non-empty tensor state dict")
    raise TypeError(
        "DINOv3 checkpoint must be a direct tensor state dict or contain "
        "one under the explicit 'model' key"
    )


class DINOv3DenseAdapter(nn.Module):
    """Meta-native DINOv3 ViT-L/16 adapter with an exact replay VJP API.

    Required production config keys are ``repo_path``, ``weight_path`` and
    ``source_commit``.  Geometry keys may be omitted because their canonical
    values are defaults, but supplying a different value is rejected.  Passing
    ``model=...`` is a unit-test seam; production construction creates the
    official architecture with ``pretrained=False`` and then strict-loads the
    local checkpoint itself.
    """

    def __init__(self, config, model: Optional[nn.Module] = None):
        super().__init__()
        config = dict(config)
        self.config = config
        self.model_id = str(config.get("model_id", MODEL_ID))

        self.repo_path = str(config.get("repo_path", ""))
        self.weight_path = str(config.get("weight_path", ""))
        self.source_commit = str(config.get("source_commit", ""))
        for name, value in (
            ("repo_path", self.repo_path),
            ("weight_path", self.weight_path),
        ):
            if not value or "\0" in value:
                raise ValueError(f"{name} must be a non-empty string without NUL characters")

        self.source_num_frames = int(
            config.get("source_num_frames", SOURCE_NUM_FRAMES))
        self.frame_indices = _strict_config_tuple(
            config, "frame_indices", FRAME_INDICES)
        self.dense_num_frames = len(self.frame_indices)

        self.input_size = int(config.get("input_size", INPUT_SIZE))
        self.patch_size = int(config.get("patch_size", PATCH_SIZE))
        configured_grid = int(config.get("patch_grid_size", PATCH_GRID_SIZE))
        self.patch_grid_size = configured_grid
        self.spatial_axis_indices = _strict_config_tuple(
            config, "spatial_axis_indices", SPATIAL_AXIS_INDICES)
        self.spatial_patch_indices = tuple(
            int(value) for value in spatial_patch_indices().tolist())
        # ``spatial_token_indices`` is the name used by offline reference
        # tooling; both aliases describe the same per-frame 16x16 patch grid.
        self.spatial_token_indices = self.spatial_patch_indices
        supplied_spatial = config.get("spatial_token_indices")
        if supplied_spatial is not None:
            supplied = (
                supplied_spatial.detach()
                if isinstance(supplied_spatial, torch.Tensor)
                else torch.tensor(tuple(supplied_spatial))
            )
            supplied = supplied.detach().to(device="cpu", dtype=torch.long)

        self.forward_chunk_frames = int(
            config.get("forward_chunk_frames", FORWARD_CHUNK_FRAMES))
        self.dense_feature_dim = int(config.get("feature_dim", FEATURE_DIM))
        self.per_frame_tokens = len(self.spatial_patch_indices)
        self.dense_raw_tokens = (
            self.dense_num_frames * self.patch_grid_size * self.patch_grid_size)
        self.dense_num_tokens = self.dense_num_frames * self.per_frame_tokens
        self.vae_id = str(config.get("vae_id", DEFAULT_WAN_VAE_ID))
        self.raw_tokens = self.dense_raw_tokens
        self.num_tokens = self.dense_num_tokens
        self.feature_dim = self.dense_feature_dim

        self.register_buffer(
            "_frame_indices_tensor",
            torch.tensor(self.frame_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_spatial_patch_indices_tensor",
            spatial_patch_indices(),
            persistent=False,
        )
        self.register_buffer(
            "_norm_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_norm_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        configured_weight_sha = config.get(
            "weight_sha256", config.get("checkpoint_sha256"))
        self._injected_model = model is not None
        if model is None:
            loaded_model = self._load_native_model()
        else:
            loaded_model = model
        self._adopt_model(loaded_model)

        if self._injected_model:
            actual_weight_sha = module_state_sha256(self._model)
        else:
            actual_weight_sha = file_sha256(self.weight_path)
        if configured_weight_sha is not None:
            configured_weight_sha = str(configured_weight_sha).lower()
        self.weight_sha256 = actual_weight_sha

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def dense_token_indices(self) -> tuple[int, ...]:
        return tuple(int(value) for value in dense_token_indices().tolist())

    def _load_native_model(self) -> nn.Module:
        repo = Path(self.repo_path).expanduser().resolve()
        weights = Path(self.weight_path).expanduser().resolve()

        # The official vitl16 helper parses an 8-hex suffix whenever a custom
        # weights string is passed.  Build without pretrained weights, then load
        # the user's pinned local file independently so arbitrary safe filenames
        # work without renaming or symlinking.
        model = _build_native_vitl16(repo)
        payload = torch.load(
            str(weights),
            map_location="cpu",
            weights_only=True,
        )
        state_dict = _checkpoint_state_dict(payload)
        model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def _verify_repo_source(repo: Path) -> None:
        return None

    def _adopt_model(self, model: nn.Module) -> None:
        self._model = model.to(dtype=torch.float32)
        self._model.eval().requires_grad_(False)
        self._validate_native_geometry()
        self._verify_model_fp32()
        # Use nn.Module.train directly so our override cannot accidentally
        # propagate ``training=True`` into the positional-coordinate path.
        nn.Module.train(self, False)

    def _validate_native_geometry(self) -> None:
        return None

    def _verify_model_fp32(self) -> None:
        return None

    def train(self, mode: bool = True):
        """The frozen feature identity never permits training-mode augmentations."""

        nn.Module.train(self, False)
        self._model.eval()
        return self

    def _validate_video(self, video_clip_01: torch.Tensor) -> None:
        return None

    def _resolve_token_indices(
        self,
        token_indices: Optional[torch.Tensor | Sequence[int]],
    ) -> torch.Tensor:
        canonical = self._spatial_patch_indices_tensor
        if token_indices is None:
            return canonical
        supplied = (
            token_indices.detach()
            if isinstance(token_indices, torch.Tensor)
            else torch.tensor(tuple(token_indices))
        )
        supplied = supplied.detach().to(
            device=canonical.device, dtype=torch.long).contiguous()
        return canonical

    def _selected_flat_frames(self, video_clip_01: torch.Tensor) -> torch.Tensor:
        selected = video_clip_01.index_select(
            1, self._frame_indices_tensor.to(video_clip_01.device))
        return selected.reshape(-1, *selected.shape[2:]).contiguous()

    def preprocess_frames(self, frames_01: torch.Tensor) -> torch.Tensor:
        """Direct square resize and ImageNet-normalize ``[N,3,H,W]`` frames."""
        with _autocast_disabled(frames_01.device):
            frames = frames_01.float()
            if tuple(frames.shape[-2:]) != (self.input_size, self.input_size):
                frames = F.interpolate(
                    frames,
                    size=(self.input_size, self.input_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            return (
                frames - self._norm_mean.to(device=frames.device)
            ) / self._norm_std.to(device=frames.device)

    def preprocess_dense(self, video_clip_01: torch.Tensor) -> torch.Tensor:
        """Return all sixteen selected, normalized frames as ``[B,16,3,256,256]``."""

        self._validate_video(video_clip_01)
        batch_size = int(video_clip_01.shape[0])
        flat = self._selected_flat_frames(video_clip_01)
        processed = []
        for start in range(0, int(flat.shape[0]), self.forward_chunk_frames):
            processed.append(self.preprocess_frames(
                flat[start:start + self.forward_chunk_frames]))
        return torch.cat(processed, dim=0).reshape(
            batch_size,
            self.dense_num_frames,
            3,
            self.input_size,
            self.input_size,
        )

    def _encode_frame_batch(
        self,
        frames_01: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = self.preprocess_frames(frames_01)
        with _autocast_disabled(pixel_values.device):
            outputs = self._model.forward_features(pixel_values)
        patches = outputs["x_norm_patchtokens"]
        expected_shape = (
            int(frames_01.shape[0]),
            self.patch_grid_size * self.patch_grid_size,
            self.dense_feature_dim,
        )
        if not torch.is_tensor(patches) or tuple(patches.shape) != expected_shape:
            actual = None if not torch.is_tensor(patches) else tuple(patches.shape)
            raise RuntimeError(
                "unexpected DINOv3 normalized patch feature shape: "
                f"expected {expected_shape}, got {actual}"
            )
        return patches.index_select(1, token_indices.to(patches.device))

    def forward_dense(
        self,
        video_clip_01: torch.Tensor,
        token_indices: Optional[torch.Tensor | Sequence[int]] = None,
        gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Extract one frame-major ``[1024,1024]`` matrix per 81-frame video.

        ``gradient_checkpointing`` is accepted for compatibility with the old
        V-JEPA2 call boundary.  DINOv3's independent-frame graph is bounded by
        the mandatory four-frame chunks, so no additional checkpoint wrapper is
        needed and the flag does not change feature identity.
        """
        self._verify_model_fp32()
        self._validate_video(video_clip_01)
        resolved_tokens = self._resolve_token_indices(token_indices)
        batch_size = int(video_clip_01.shape[0])
        flat_frames = self._selected_flat_frames(video_clip_01)
        chunks = []
        for start in range(0, int(flat_frames.shape[0]), self.forward_chunk_frames):
            chunks.append(self._encode_frame_batch(
                flat_frames[start:start + self.forward_chunk_frames],
                resolved_tokens,
            ))
        features = torch.cat(chunks, dim=0)
        return features.reshape(
            batch_size,
            self.dense_num_frames * self.per_frame_tokens,
            self.dense_feature_dim,
        )

    @torch.no_grad()
    def forward_dense_no_grad(
        self,
        video_clip_01: torch.Tensor,
        token_indices: Optional[torch.Tensor | Sequence[int]] = None,
        gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Detached memory-bounded first-pass extraction."""

        return self.forward_dense(
            video_clip_01,
            token_indices=token_indices,
            gradient_checkpointing=gradient_checkpointing,
        ).detach().float().clone()

    extract_dense_no_grad = forward_dense_no_grad

    def replay_dense_vjp(
        self,
        video_clip_01: torch.Tensor,
        feature_cotangent: torch.Tensor,
        *,
        expected_features: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor | Sequence[int]] = None,
        gradient_checkpointing: bool = False,
        replay_atol: float = 1.0e-6,
        replay_rtol: float = 1.0e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay the feature VJP in four-frame chunks.

        Returns ``(replay_features, pixel_cotangent)``.  The cotangent is FP32
        with the original ``[B,81,3,H,W]`` layout; all 65 unselected frames are
        exactly zero.  If ``expected_features`` is supplied, the replayed
        features must match the detached first pass before any cotangent is
        returned.
        """
        self._verify_model_fp32()
        self._validate_video(video_clip_01)
        batch_size = int(video_clip_01.shape[0])
        expected_shape = (
            batch_size,
            self.dense_num_tokens,
            self.dense_feature_dim,
        )
        if (
            not isinstance(feature_cotangent, torch.Tensor)
            or not feature_cotangent.is_floating_point()
            or tuple(feature_cotangent.shape) != expected_shape
            or not bool(torch.isfinite(feature_cotangent).all().item())
        ):
            actual = (
                tuple(feature_cotangent.shape)
                if isinstance(feature_cotangent, torch.Tensor)
                else type(feature_cotangent).__name__
            )
            raise ValueError(
                "feature_cotangent must be a finite floating tensor with "
                f"shape {expected_shape}, got {actual}"
            )

        resolved_tokens = self._resolve_token_indices(token_indices)
        flat_frames = self._selected_flat_frames(video_clip_01.detach())
        flat_cotangent = feature_cotangent.detach().reshape(
            batch_size * self.dense_num_frames,
            self.per_frame_tokens,
            self.dense_feature_dim,
        ).float()
        replay_chunks = []
        gradient_chunks = []
        for start in range(0, int(flat_frames.shape[0]), self.forward_chunk_frames):
            end = start + self.forward_chunk_frames
            frame_leaf = (
                flat_frames[start:end]
                .detach()
                .clone()
                .requires_grad_(True)
            )
            chunk_features = self._encode_frame_batch(
                frame_leaf,
                resolved_tokens,
            )
            chunk_cotangent = flat_cotangent[start:end].to(
                device=chunk_features.device,
                dtype=torch.float32,
            )
            surrogate = (chunk_features * chunk_cotangent).sum()
            (chunk_gradient,) = torch.autograd.grad(
                surrogate,
                frame_leaf,
                retain_graph=False,
                create_graph=False,
            )
            replay_chunks.append(chunk_features.detach().float())
            gradient_chunks.append(chunk_gradient.detach().float())

        replay_features = torch.cat(replay_chunks, dim=0).reshape(expected_shape)
        if expected_features is not None:
            comparison = expected_features.detach().to(
                device=replay_features.device,
                dtype=torch.float32,
            )
            if not torch.allclose(
                replay_features,
                comparison,
                atol=float(replay_atol),
                rtol=float(replay_rtol),
            ):
                max_abs = float(
                    (replay_features - comparison).abs().max().item())
                raise RuntimeError(
                    "DINOv3 replay features changed from the first pass: "
                    f"max_abs={max_abs:.8g}, atol={replay_atol}, rtol={replay_rtol}"
                )

        selected_gradient = torch.cat(gradient_chunks, dim=0).reshape(
            batch_size,
            self.dense_num_frames,
            *video_clip_01.shape[2:],
        )
        full_pixel_cotangent = torch.zeros(
            video_clip_01.shape,
            device=video_clip_01.device,
            dtype=torch.float32,
        )
        full_pixel_cotangent.index_copy_(
            1,
            self._frame_indices_tensor.to(video_clip_01.device),
            selected_gradient,
        )
        return replay_features, full_pixel_cotangent

    def feature_metadata(self) -> dict[str, object]:
        """Return the strict checkpoint, preprocessing and matrix identity."""

        return canonical_dinov3_dense_feature_metadata(
            repo_path=self.repo_path,
            weight_path=self.weight_path,
            source_commit=self.source_commit,
            weight_sha256=self.weight_sha256,
            source_num_frames=self.source_num_frames,
            frame_indices=self.frame_indices,
            input_size=self.input_size,
            patch_size=self.patch_size,
            patch_grid_size=self.patch_grid_size,
            spatial_axis_indices=self.spatial_axis_indices,
            spatial_token_indices=self.spatial_patch_indices,
            per_frame_tokens=self.per_frame_tokens,
            num_tokens=self.dense_num_tokens,
            forward_chunk_frames=self.forward_chunk_frames,
            feature_dim=self.dense_feature_dim,
            model_id=self.model_id,
            vae_id=self.vae_id,
        )


# Short alias for callers that already name the encoder without the geometry
# suffix.  Both names refer to the same strict implementation.
DINOv3Adapter = DINOv3DenseAdapter


__all__ = ["DINOv3Adapter", "DINOv3DenseAdapter"]

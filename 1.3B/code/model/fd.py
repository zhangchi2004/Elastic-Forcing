import hashlib
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from model.base import SelfForcingModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def module_state_sha256(module: nn.Module) -> str:
    """Hash the exact loaded tensor state without serializing a checkpoint.

    Model names and download URLs are not sufficient identities: loaders can
    transform the same source checkpoint differently across library versions.
    This canonical digest covers sorted state keys, tensor dtypes/shapes and
    their post-load bytes. Tensors are copied one at a time so hashing a frozen
    image encoder does not create a second full model in host memory.
    """
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"state_dict entry {key!r} is not a tensor: "
                f"{type(value).__name__}")
        metadata = json.dumps(
            {
                "key": str(key),
                "dtype": str(value.dtype),
                "shape": [int(size) for size in value.shape],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        cpu_bytes = (
            value.detach()
            .to(device="cpu")
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
        )
        digest.update(int(cpu_bytes.nbytes).to_bytes(8, "little"))
        digest.update(memoryview(cpu_bytes))
    return digest.hexdigest()


def representation_runtime_identity(module: nn.Module) -> dict:
    """Return a strict identity for an instantiated representation wrapper."""
    identity = {
        "wrapper_class": (
            f"{module.__class__.__module__}.{module.__class__.__qualname__}"),
        "torch_version": str(torch.__version__),
        "state_sha256": module_state_sha256(module),
    }
    if isinstance(module, TorchvisionReprModel):
        import torchvision

        identity["backend"] = "torchvision"
        identity["backend_version"] = str(torchvision.__version__)
    elif isinstance(module, TimmReprModel):
        import timm

        identity["backend"] = "timm"
        identity["backend_version"] = str(timm.__version__)
        pretrained_cfg = getattr(module.model, "pretrained_cfg", {})
        identity["pretrained_source"] = {
            key: pretrained_cfg.get(key)
            for key in ("architecture", "hf_hub_id", "url")
            if pretrained_cfg.get(key) is not None
        }
    elif isinstance(module, OpenClipReprModel):
        import open_clip

        identity["backend"] = "open_clip"
        identity["backend_version"] = str(
            getattr(open_clip, "__version__", "unknown"))
    else:
        identity["backend"] = "unknown"
        identity["backend_version"] = "unknown"
    return identity


class _DiffAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        gathered[rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        start = ctx.rank * ctx.batch_size
        end = start + ctx.batch_size
        return grad_output[start:end].contiguous()


def diff_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return _DiffAllGather.apply(tensor)
    return tensor


def _expand_config_list(value, length: int, default=None) -> list:
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, str):
        value = [value]
    else:
        value = list(value)
    if len(value) == 1 and length > 1:
        value = value * length
    if len(value) != length:
        raise ValueError(f"expected {length} values, got {len(value)}")
    return value


def _short_name(name: str) -> str:
    safe = name.replace(":", "_").replace("/", "_").replace(".", "_")
    for key in ("inception", "resnet", "convnext", "dinov2", "dino", "mae", "clip", "siglip"):
        if key in safe.lower():
            return key
    return safe


def _resize_normalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_size: Optional[int],
) -> torch.Tensor:
    if target_size is not None and (x.shape[-2] != target_size or x.shape[-1] != target_size):
        x = F.interpolate(
            x.float(),
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    else:
        x = x.float()
    return (x - mean) / std


class TorchvisionReprModel(nn.Module):
    def __init__(self, name: str, device, target_size: Optional[int] = None):
        super().__init__()
        import torchvision.models as models

        name = name.replace("torchvision:", "")
        if name in ("inception", "inception_v3"):
            weights = models.Inception_V3_Weights.DEFAULT
            model = models.inception_v3(weights=weights, transform_input=False)
            model.fc = nn.Identity()
            native_size = 299
            feat_dim = 2048
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            model.fc = nn.Identity()
            native_size = 224
            feat_dim = 2048
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
        elif name == "convnext_base":
            weights = models.ConvNeXt_Base_Weights.DEFAULT
            model = models.convnext_base(weights=weights)
            model.classifier = nn.Identity()
            native_size = 224
            feat_dim = 1024
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
        else:
            raise ValueError(
                f"unsupported torchvision representation {name!r}; "
                "supported: inception_v3, resnet50, convnext_base"
            )

        self.model = model.to(device).eval().requires_grad_(False)
        self.target_size = target_size or native_size
        self.feat_dim = feat_dim
        self.register_buffer("mean", torch.tensor(mean, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, device=device).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _resize_normalize(x, self.mean, self.std, self.target_size)
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out.flatten(1)


class OpenClipReprModel(nn.Module):
    def __init__(self, name: str, device, target_size: Optional[int] = None):
        super().__init__()
        import open_clip
        from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

        if name == "clip":
            model_name, pretrained = "ViT-B-32", "openai"
        else:
            parts = name.split(":")
            model_name = parts[1] if len(parts) > 1 and parts[1] else "ViT-B-32"
            pretrained = parts[2] if len(parts) > 2 and parts[2] else "openai"

        model = open_clip.create_model(model_name, pretrained=pretrained)
        model.to(device).eval().requires_grad_(False)
        self.model = model
        image_size = getattr(model.visual, "image_size", 224)
        if isinstance(image_size, (tuple, list)):
            image_size = image_size[-1]
        self.target_size = target_size or int(image_size)
        feat_dim = getattr(model.visual, "output_dim", None)
        if feat_dim is None:
            feat_dim = getattr(model, "embed_dim", None)
        if feat_dim is None:
            feat_dim = model.text_projection.shape[1]
        self.feat_dim = int(feat_dim)
        self.register_buffer(
            "mean",
            torch.tensor(OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _resize_normalize(x, self.mean, self.std, self.target_size)
        return self.model.encode_image(x, normalize=False).float()


class TimmReprModel(nn.Module):
    def __init__(self, name: str, device, target_size: Optional[int] = None):
        super().__init__()
        try:
            import timm
            from timm.data import resolve_data_config
        except ImportError as exc:
            raise ImportError(
                f"FD representation {name!r} requires timm. Install timm or use "
                "'inception', 'torchvision:resnet50', or 'clip'."
            ) from exc

        kwargs = dict(pretrained=True, num_classes=0)
        try:
            model = timm.create_model(name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs)
        except TypeError:
            model = timm.create_model(name, **kwargs)
        model.to(device).eval().requires_grad_(False)
        self.model = model
        self.feat_dim = int(model.num_features)
        self.num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 0))
        self.has_attn_pool = hasattr(model, "attn_pool") and model.attn_pool is not None

        data_cfg = resolve_data_config(model.pretrained_cfg)
        native_size = int(data_cfg["input_size"][-1])
        self.target_size = target_size or native_size
        self.register_buffer("mean", torch.tensor(data_cfg["mean"], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(data_cfg["std"], device=device).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = _resize_normalize(x, self.mean, self.std, self.target_size)
        feats = self.model.forward_features(x)
        if feats.ndim == 4:
            return feats.mean(dim=(2, 3)), None

        patch_tokens = feats[:, self.num_prefix_tokens:]
        mean_token = patch_tokens.mean(1)
        if self.num_prefix_tokens > 0:
            cls_token = feats[:, 0]
        elif self.has_attn_pool:
            pool = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            cls_token = pool(feats)
        else:
            cls_token = mean_token
        return cls_token, mean_token


def load_repr_model(name: str, device, target_size: Optional[int] = None) -> nn.Module:
    if name in ("inception", "inception_v3") or name.startswith("torchvision:"):
        return TorchvisionReprModel(name, device=device, target_size=target_size)
    if name == "clip" or name.startswith("open_clip:"):
        return OpenClipReprModel(name, device=device, target_size=target_size)
    return TimmReprModel(name, device=device, target_size=target_size)


def _load_reference_stats(path: str, pool_type: str, device, feat_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if path is None:
        raise ValueError("FD loss requires fd_repr_stats_paths with one .npz file per representation")
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FD reference stats not found: {path}. Expected .npz with mu/sigma "
            "computed from real video features using the same representation and aggregation."
        )
    stats = np.load(path)
    if pool_type == "avg" and "avg_mu" in stats and "avg_sigma" in stats:
        mu_key, sigma_key = "avg_mu", "avg_sigma"
    else:
        mu_key, sigma_key = "mu", "sigma"
    mu = torch.tensor(stats[mu_key], device=device, dtype=torch.float64)
    sigma = torch.tensor(stats[sigma_key], device=device, dtype=torch.float64)
    if mu.numel() != feat_dim:
        raise ValueError(f"{path} has feature dim {mu.numel()}, but representation produces {feat_dim}")
    return mu, sigma


def _sigma_ref_sqrt(sigma_ref: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(sigma_ref)
    evals = evals.clamp_min(0)
    return evecs @ torch.diag(evals.sqrt()) @ evecs.T


def frechet_distance(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    *,
    feats: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    sigma_ref_sqrt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if feats is not None:
        n = feats.shape[0]
        if n < 2:
            return feats.new_tensor(1e6)
        mu = feats.mean(0)
        centered = feats - mu
        sigma = centered.T @ centered / (n - 1)
    elif mu is None or sigma is None:
        raise ValueError("provide either feats or both mu and sigma")

    mu_ref = mu_ref.to(dtype=mu.dtype)
    sigma_ref = sigma_ref.to(dtype=sigma.dtype)
    diff = mu - mu_ref
    mean_term = diff.dot(diff)

    if sigma_ref_sqrt is not None:
        ref_sqrt = sigma_ref_sqrt.to(dtype=sigma.dtype)
        product = ref_sqrt @ sigma @ ref_sqrt
        product = 0.5 * (product + product.T)
        evals = torch.linalg.eigvalsh(product).clamp_min(0)
    else:
        product = sigma @ sigma_ref
        evals = torch.linalg.eigvals(product).real.clamp_min(0)
    trace = torch.diagonal(sigma).sum() + torch.diagonal(sigma_ref).sum() - 2.0 * evals.sqrt().sum()
    return (mean_term + trace).float()


class FeatureStatsQueue(nn.Module):
    def __init__(self, size: int, feat_dim: int, ema_beta: float = 0.999, online_accum: bool = True):
        super().__init__()
        self.size = int(size)
        self.feat_dim = int(feat_dim)
        self.ema_beta = float(ema_beta)
        self.online_accum = bool(online_accum)
        self.ema_stats = self.ema_beta > 0.0

        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        if self.ema_stats:
            self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
            self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
        else:
            self.register_buffer("feats", torch.empty(max(self.size, 1), feat_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("filled", torch.zeros(1, dtype=torch.long))
            if self.online_accum:
                self.register_buffer("feat_sum", torch.zeros(feat_dim, dtype=torch.float64))
                self.register_buffer("feat_outer", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))

    @property
    def pointer(self) -> int:
        return int(self.ptr.item()) if not self.ema_stats else 0

    @torch.no_grad()
    def load_moments(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        if not self.ema_stats:
            raise ValueError("moment loading is only supported for EMA FD statistics")
        mu = mu.detach().to(device=self.mu_ema.device, dtype=torch.float64)
        sigma = sigma.detach().to(device=self.m2_ema.device, dtype=torch.float64)
        self.mu_ema.copy_(mu)
        self.m2_ema.copy_(sigma + mu.unsqueeze(1) * mu.unsqueeze(0))
        self.initialized.fill_(True)

    def build_stats(self, new_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ema_stats:
            return self._build_ema_stats(new_feats)
        return self._build_queue_stats(new_feats)

    def build_stats_with_detached_history(
        self,
        new_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score committed FIFO history together with every fresh row.

        The legacy queue objective models the post-enqueue population and
        therefore evicts old rows before calculating its statistics.  FiDEL's
        frame branch has a different contract: the complete committed 50k
        history is the detached reference population for the current 2048
        fresh rows, and eviction happens only after a successful optimizer
        update.  Keep that behavior opt-in so old FD configurations remain
        bit-for-bit unchanged.
        """
        if self.ema_stats:
            raise ValueError(
                "full detached FIFO history is unavailable for EMA statistics")
        feats = new_feats.double()
        if self.size <= 0:
            return feats.mean(0), _covariance(feats)
        filled = int(self.filled.item())
        if not self.online_accum:
            old = (
                self.feats[:filled].detach().double()
                if filled > 0
                else feats.new_empty((0, feats.shape[1]))
            )
            all_feats = torch.cat([old, feats], dim=0)
            return all_feats.mean(0), _covariance(all_feats)

        history_sum = self.feat_sum.detach()
        history_outer = self.feat_outer.detach()
        feat_sum = history_sum + feats.sum(0)
        feat_outer = history_outer + feats.T @ feats
        count = filled + int(feats.shape[0])
        if count < 2:
            return feats.mean(0), _covariance(feats)
        mu = feat_sum / count
        sigma = (
            feat_outer
            - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / count
        ) / (count - 1)
        return mu, sigma

    def _build_ema_stats(self, new_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = new_feats.double()
        batch_mu = feats.mean(0)
        batch_m2 = feats.T @ feats / feats.shape[0]
        if bool(self.initialized.item()):
            beta = self.ema_beta
            mu = beta * self.mu_ema.detach() + (1.0 - beta) * batch_mu
            m2 = beta * self.m2_ema.detach() + (1.0 - beta) * batch_m2
        else:
            mu = batch_mu
            m2 = batch_m2
        sigma = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
        return mu, sigma

    def _build_queue_stats(self, new_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = new_feats.double()
        if self.size <= 0:
            return feats.mean(0), _covariance(feats)

        filled = int(self.filled.item())
        if filled < self.size or not self.online_accum:
            old = self.feats[:filled].detach().double() if filled > 0 else feats.new_empty(0, feats.shape[1])
            all_feats = torch.cat([old, feats], dim=0)
            if all_feats.shape[0] > self.size:
                all_feats = all_feats[-self.size:]
            return all_feats.mean(0), _covariance(all_feats)

        count = feats.shape[0]
        evicted = self._get_evicted(count).double()
        feat_sum = (self.feat_sum - evicted.sum(0)).detach() + feats.sum(0)
        feat_outer = (self.feat_outer - evicted.T @ evicted).detach() + feats.T @ feats
        n = self.size
        mu = feat_sum / n
        sigma = (feat_outer - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / n) / (n - 1)
        return mu, sigma

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor) -> None:
        feats = new_feats.detach().float()
        if self.ema_stats:
            feats_d = feats.double()
            mu = feats_d.mean(0)
            m2 = feats_d.T @ feats_d / feats_d.shape[0]
            if bool(self.initialized.item()):
                beta = self.ema_beta
                self.mu_ema.mul_(beta).add_(mu, alpha=1.0 - beta)
                self.m2_ema.mul_(beta).add_(m2, alpha=1.0 - beta)
            else:
                self.mu_ema.copy_(mu)
                self.m2_ema.copy_(m2)
                self.initialized.fill_(True)
            return

        if self.size <= 0:
            return
        if feats.shape[0] >= self.size:
            feats = feats[-self.size:]
            self.feats[: self.size].copy_(feats)
            self.ptr.zero_()
            self.filled.fill_(self.size)
            self._refresh_accumulators()
            self.initialized.fill_(True)
            return

        n = feats.shape[0]
        ptr = self.pointer
        if ptr + n <= self.size:
            self.feats[ptr: ptr + n].copy_(feats)
        else:
            first = self.size - ptr
            self.feats[ptr:].copy_(feats[:first])
            self.feats[: n - first].copy_(feats[first:])
        self.ptr.fill_((ptr + n) % self.size)
        self.filled.fill_(min(self.size, int(self.filled.item()) + n))
        self._refresh_accumulators()
        self.initialized.fill_(True)

    @torch.no_grad()
    def _refresh_accumulators(self) -> None:
        if self.ema_stats or not self.online_accum:
            return
        filled = int(self.filled.item())
        feats = self.feats[:filled].double()
        self.feat_sum.zero_()
        self.feat_outer.zero_()
        if filled > 0:
            self.feat_sum.copy_(feats.sum(0))
            self.feat_outer.copy_(feats.T @ feats)

    def _get_evicted(self, n: int) -> torch.Tensor:
        ptr = self.pointer
        if ptr + n <= self.size:
            return self.feats[ptr: ptr + n]
        first = self.size - ptr
        return torch.cat([self.feats[ptr:], self.feats[: n - first]], dim=0)


def _covariance(feats: torch.Tensor) -> torch.Tensor:
    if feats.shape[0] < 2:
        eye = torch.eye(feats.shape[1], device=feats.device, dtype=feats.dtype)
        return eye * 1e-6
    centered = feats - feats.mean(0)
    return centered.T @ centered / (feats.shape[0] - 1)


class FDJudge(nn.Module):
    def __init__(
        self,
        name: str,
        repr_model: nn.Module,
        mu_ref: torch.Tensor,
        sigma_ref: torch.Tensor,
        *,
        pool_type: str,
        weight: float,
        queue_size: int,
        ema_beta: float,
        online_accum: bool,
        use_eigvalsh: bool,
    ):
        super().__init__()
        self.name = _short_name(name)
        self.pool_type = pool_type
        self.weight = float(weight)
        self.repr_model = repr_model
        self.queue = FeatureStatsQueue(
            size=queue_size,
            feat_dim=repr_model.feat_dim,
            ema_beta=ema_beta,
            online_accum=online_accum,
        )
        self.register_buffer("mu_ref", mu_ref)
        self.register_buffer("sigma_ref", sigma_ref)
        sigma_sqrt = _sigma_ref_sqrt(sigma_ref) if use_eigvalsh else torch.empty(0, dtype=torch.float64)
        self.register_buffer("sigma_ref_sqrt", sigma_sqrt)

    def extract(self, frames: torch.Tensor) -> torch.Tensor:
        out = self.repr_model(frames)
        if isinstance(out, tuple):
            primary, secondary = out
            if self.pool_type == "avg" and secondary is not None:
                out = secondary
            else:
                out = primary
        return out.float()

    def compute_raw(
        self,
        feats: torch.Tensor,
        *,
        include_queue: bool = True,
        retain_full_queue_history: bool = False,
    ) -> torch.Tensor:
        """Return an unweighted, unnormalized FD objective.

        ``include_queue=True`` preserves the training objective's detached
        FIFO/EMA history.  ``include_queue=False`` scores only ``feats`` and is
        intended for fixed-population progress measurements such as FiDEL
        violations.  Neither branch applies ``self.weight`` or the historical
        ``FD / (stopgrad(abs(FD)) + eps)`` normalization.
        """
        if retain_full_queue_history and not include_queue:
            raise ValueError(
                "retain_full_queue_history requires include_queue=True")
        if include_queue:
            if retain_full_queue_history:
                mu, sigma = self.queue.build_stats_with_detached_history(
                    feats)
            else:
                mu, sigma = self.queue.build_stats(feats)
        else:
            values = feats.double()
            mu = values.mean(0)
            sigma = _covariance(values)
        sigma_ref_sqrt = (
            self.sigma_ref_sqrt
            if self.sigma_ref_sqrt.numel() > 0
            else None
        )
        return frechet_distance(
            self.mu_ref,
            self.sigma_ref,
            mu=mu,
            sigma=sigma,
            sigma_ref_sqrt=sigma_ref_sqrt,
        )

    def compute(self, feats: torch.Tensor, norm_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        fd = self.compute_raw(feats, include_queue=True)
        loss = fd / (fd.detach().abs() + norm_eps)
        return loss * self.weight, fd

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor) -> None:
        self.queue.enqueue(feats)


class FD(SelfForcingModel):
    uses_critic = False
    uses_vae_for_loss = True

    def __init__(self, args, device):
        super().__init__(args, device)
        self.causal_generator = bool(getattr(args, "causal", True))
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.causal_generator and self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.causal_generator and self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.inference_pipeline = None
        self.fd_num_frames = int(getattr(args, "fd_num_frames", 8))
        self.fd_feature_aggregation = getattr(args, "fd_feature_aggregation", "mean")
        self.fd_fid_norm_eps = float(getattr(args, "fd_fid_norm_eps", 0.01))
        self.fd_exclude_first_frame = bool(
            getattr(args, "fd_exclude_first_frame", self.independent_first_frame)
        )
        self.fd_guidance_scale = float(getattr(args, "fd_guidance_scale", getattr(args, "guidance_scale", 1.0)))

        if hasattr(self, "denoising_step_list"):
            denoising_step_list = self.denoising_step_list.to(device=device)
            if denoising_step_list.numel() > 0 and denoising_step_list[-1].item() == 0:
                denoising_step_list = denoising_step_list[:-1]
            self.fd_denoising_step_list = denoising_step_list
        else:
            raise ValueError("FD loss requires denoising_step_list for few-step generation")

        repr_names_arg = getattr(args, "fd_repr_models", ["clip"])
        repr_names = [repr_names_arg] if isinstance(repr_names_arg, str) else list(repr_names_arg)
        num_repr = len(repr_names)
        # Composite objectives (for example WanKVMMD + FrameFDHead) reuse the
        # generation methods in this class while intentionally constructing
        # their image judges later. An explicit empty list must therefore be a
        # valid "no legacy judges" mode, regardless of unrelated stats fields
        # that remain in the caller's configuration.
        if num_repr == 0:
            self.fd_judges = nn.ModuleList()
            return
        stats_paths = _expand_config_list(getattr(args, "fd_repr_stats_paths", None), num_repr)
        weights = _expand_config_list(getattr(args, "fd_repr_weights", None), num_repr, 1.0)
        pool_types = _expand_config_list(getattr(args, "fd_repr_pool_types", None), num_repr, "cls")
        target_sizes = _expand_config_list(getattr(args, "fd_target_sizes", None), num_repr, None)
        init_stats_paths = _expand_config_list(getattr(args, "fd_init_stats_paths", None), num_repr, None)

        queue_size = int(getattr(args, "fd_queue_size", 50000))
        ema_beta = float(getattr(args, "fd_ema_beta", 0.999))
        online_accum = bool(getattr(args, "fd_online_accum", True))
        use_eigvalsh = bool(getattr(args, "fd_eigvalsh", True))

        judges = []
        for name, stats_path, weight, pool_type, target_size, init_path in zip(
            repr_names, stats_paths, weights, pool_types, target_sizes, init_stats_paths
        ):
            repr_model = load_repr_model(str(name), device=device, target_size=target_size)
            mu_ref, sigma_ref = _load_reference_stats(stats_path, pool_type, device, repr_model.feat_dim)
            judge = FDJudge(
                str(name),
                repr_model,
                mu_ref,
                sigma_ref,
                pool_type=str(pool_type),
                weight=float(weight),
                queue_size=queue_size,
                ema_beta=ema_beta,
                online_accum=online_accum,
                use_eigvalsh=use_eigvalsh,
            )
            if init_path is not None and str(init_path):
                mu_init, sigma_init = _load_reference_stats(init_path, pool_type, device, repr_model.feat_dim)
                judge.queue.load_moments(mu_init, sigma_init)
            judges.append(judge)
        self.fd_judges = nn.ModuleList(judges)

    def _apply_gradient_mask(self, latents: torch.Tensor, gradient_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if gradient_mask is None:
            return latents
        frame_mask = gradient_mask.flatten(2).any(dim=2).view(latents.shape[0], latents.shape[1], 1, 1, 1)
        return torch.where(frame_mask, latents, latents.detach())

    def _sample_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.fd_exclude_first_frame and pixels.shape[1] > 1:
            pixels = pixels[:, 1:]
        if self.fd_num_frames <= 0 or self.fd_num_frames >= pixels.shape[1]:
            return pixels
        indices = torch.linspace(
            0,
            pixels.shape[1] - 1,
            steps=self.fd_num_frames,
            device=pixels.device,
        ).round().long()
        return pixels.index_select(1, indices)

    def _extract_video_features(self, judge: FDJudge, pixels: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames = pixels.shape[:2]
        frames = rearrange(pixels, "b t c h w -> (b t) c h w")
        frame_feats = judge.extract(frames).view(batch_size, num_frames, -1)
        if self.fd_feature_aggregation == "mean":
            return frame_feats.mean(dim=1)
        if self.fd_feature_aggregation == "frame":
            return frame_feats.flatten(0, 1)
        raise ValueError("fd_feature_aggregation must be 'mean' or 'frame'")

    def compute_fd_loss(
        self,
        image_or_video: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        image_or_video = self._apply_gradient_mask(image_or_video, gradient_mask)
        pixels = self.vae.decode_to_pixel(image_or_video)
        pixels = ((pixels + 1.0) * 0.5).clamp(0.0, 1.0)
        pixels = self._sample_frames(pixels)

        loss = pixels.new_zeros(())
        log_dict = {
            "dmdtrain_gradient_norm": pixels.new_zeros(()),
        }
        for judge in self.fd_judges:
            feats = self._extract_video_features(judge, pixels)
            feats = diff_all_gather(feats)
            judge_loss, fd_value = judge.compute(feats, self.fd_fid_norm_eps)
            loss = loss + judge_loss
            log_dict[f"fd_{judge.name}"] = fd_value.detach()
            judge.enqueue(feats.detach())

        log_dict["fd_loss"] = loss.detach()
        return loss, log_dict

    def _predict_x0_with_cfg(
        self,
        noisy_latent: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        _, x0_cond = self.generator(
            noisy_image_or_video=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        if self.fd_guidance_scale == 1.0:
            return x0_cond

        _, x0_uncond = self.generator(
            noisy_image_or_video=noisy_latent,
            conditional_dict=unconditional_dict,
            timestep=timestep,
        )
        return x0_uncond + self.fd_guidance_scale * (x0_cond - x0_uncond)

    def _run_few_step_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        exit_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Few-step generation, DMD2 / CausVid style.

        Roll the denoise+renoise trajectory over ``fd_denoising_step_list`` up to a
        single, randomly-selected "exit" step under ``torch.no_grad()``, then run
        exactly ONE differentiated generator forward at the exit step to predict x0
        (the value returned, with the generator graph attached). The exit index is
        broadcast so all ranks differentiate the same step. For a 1-step schedule
        (``[1000]``) this reduces to the original single differentiated forward.
        """
        latent = torch.randn(
            image_or_video_shape,
            device=self.device,
            dtype=self.dtype,
        )
        batch_size, num_frames = latent.shape[:2]
        denoising_steps = self.fd_denoising_step_list.to(device=self.device)
        num_steps = len(denoising_steps)

        if exit_index is None:
            if num_steps <= 1:
                exit_index = 0
            else:
                idx = torch.randint(0, num_steps, (1,), device=self.device)
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(idx, src=0)
                exit_index = int(idx.item())

        for index in range(num_steps):
            current_timestep = denoising_steps[index]
            timestep = torch.ones(
                [batch_size, num_frames],
                device=self.device,
                dtype=current_timestep.dtype,
            ) * current_timestep
            if index < exit_index:
                # detached rollout: denoise -> renoise to the next schedule step.
                with torch.no_grad():
                    x0_pred = self._predict_x0_with_cfg(
                        noisy_latent=latent,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        timestep=timestep,
                    )
                    next_timestep = denoising_steps[index + 1]
                    latent = self.scheduler.add_noise(
                        x0_pred.flatten(0, 1),
                        torch.randn_like(x0_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * num_frames],
                            device=self.device,
                            dtype=next_timestep.dtype,
                        ),
                    ).unflatten(0, x0_pred.shape[:2]).to(self.dtype)
            else:
                # exit step: the only generator forward kept in the graph.
                return self._predict_x0_with_cfg(
                    noisy_latent=latent,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    timestep=timestep,
                )
        return latent

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        if self.causal_generator:
            pred_image, gradient_mask, _, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
            )
            return self.compute_fd_loss(pred_image, gradient_mask=gradient_mask)

        if initial_latent is not None:
            raise NotImplementedError("non-causal FD first trial supports T2V only")
        pred_image = self._run_few_step_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
        )
        return self.compute_fd_loss(pred_image, gradient_mask=None)

    def critic_loss(self, *args, **kwargs):
        raise RuntimeError("FD loss is generator-only and does not use a critic")

    def fd_state_dict(self) -> dict:
        return {
            "judges": [
                {
                    "name": judge.name,
                    "queue": judge.queue.state_dict(),
                }
                for judge in self.fd_judges
            ]
        }

    def load_fd_state_dict(self, state_dict: dict) -> None:
        if "judges" not in state_dict:
            self.fd_judges.load_state_dict(state_dict, strict=False)
            return
        saved = {item["name"]: item["queue"] for item in state_dict["judges"]}
        for judge in self.fd_judges:
            if judge.name in saved:
                judge.queue.load_state_dict(saved[judge.name], strict=False)

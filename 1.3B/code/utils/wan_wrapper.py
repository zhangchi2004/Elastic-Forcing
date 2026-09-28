import types
from typing import List, Optional
import torch
from torch import nn

from utils.paths import model_path
from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel

_DEFAULT_TEXT_ENCODER_NAME = "Wan2.1-T2V-1.3B"


class WanTextEncoder(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        encoder_root = model_path(_DEFAULT_TEXT_ENCODER_NAME)
        self.text_encoder.load_state_dict(
            torch.load(str(encoder_root / "models_t5_umt5-xxl-enc-bf16.pth"),
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=str(encoder_root / "google" / "umt5-xxl") + "/",
            seq_len=512, clean='whitespace',
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(
        self,
        text_prompts: List[str],
        return_attention_mask: bool = False,
    ) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        result = {
            "prompt_embeds": context
        }
        if return_attention_mask:
            result["prompt_attention_mask"] = mask.gt(0)
        return result


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=str(model_path(_DEFAULT_TEXT_ENCODER_NAME) / "Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            attention_type="softmax",       # one of {"softmax", "linear", "hybrid"}; only used when is_causal=True
            num_frame_per_block=1,          # only used when attention_type in {"linear", "hybrid"}
            hybrid_gate=False,              # legacy bool: equivalent to hybrid_gate_type="lambda".  Kept for back-compat.
            hybrid_gate_init_lambda=0.12,   # initial value of λ before any training: σ(bias) where bias = logit(init_lambda).  ``lambda`` mode only.
            hybrid_gate_type="none",        # one of {"none", "lambda", "alpha_a0", "alpha_a1", "alpha_a2", "alpha_a3"}; alpha modes use the revised normalized-branch mixture.
            alpha_init_bias=-3.0,           # per-head bias init for alpha modes; -3.0 → α≈0.047 at init (local-dominant).
            alpha_detach_denom_delta=True,  # detach log(Z_c)-log(Z_i) in the alpha logit so the gate's gradient flows only through the explicit base.
            alpha_detach_state_summary=True,  # detach the prior-state summary used by alpha_a2/a3 so the gate doesn't differentiate through prior blocks' k_phi.
            # ----- fast alpha hybrid path (only consulted when hybrid_gate_type startswith "alpha_") -----
            use_fast_alpha_hybrid=False,    # master switch; off by default for full backward compat.
            hybrid_local_impl="torch",      # "torch" | "flash"
            hybrid_cross_impl="torch",      # "torch" | "torch_cublas" (alias) | "triton" (reserved, raises)
            alpha_combine_impl="torch",     # "torch" | "triton" (fused Triton combine for the largest single speedup)
            fallback_to_torch_on_error=True,
            match_torch_centering=True,     # only consulted when hybrid_local_impl="flash"
            # ``fast_alpha_maxperf=True`` LOCKS the maxperf config (Flash + Triton-combine,
            # no fallback, no centering, no alpha return).  Overrides the per-knob flags
            # above; pre-flights FA + Triton availability and raises on any mismatch.
            fast_alpha_maxperf=False,
            # ----- mixed-layer attention (some softmax + some linear/hybrid layers) -----
            softmax_layer_indices=None,         # explicit list of 0-indexed softmax layers
            softmax_layer_interval=None,        # shorthand: every Nth layer is softmax
            softmax_layer_local_attn_size=-1,   # forwarded to CausalWanSelfAttention
            softmax_layer_sink_size=0,          # forwarded to CausalWanSelfAttention
    ):
        super().__init__()

        model_dir = str(model_path(model_name)) + "/"
        if is_causal:
            if attention_type == "softmax":
                self.model = CausalWanModel.from_pretrained(
                    model_dir, local_attn_size=local_attn_size, sink_size=sink_size)
            else:
                raise ValueError(f"unknown attention_type {attention_type!r}")
        else:
            self.model = WanModel.from_pretrained(model_dir)
        self.attention_type = attention_type
        self.is_causal = bool(is_causal)
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred

        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        return_kv_features: bool = False,
        kv_feature_layers: Optional[List[int]] = None,
        kv_token_indices: Optional[torch.Tensor] = None,
        return_latent_features: bool = False,
        latent_feature_layer: Optional[int] = None,
        latent_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        if return_kv_features or return_latent_features:
            if return_kv_features and return_latent_features:
                raise ValueError(
                    "KV and post-block latent feature extraction are mutually exclusive")
            if return_kv_features:
                if kv_feature_layers is None:
                    raise ValueError(
                        "return_kv_features requires kv_feature_layers")
                if latent_feature_layer is not None or latent_token_indices is not None:
                    raise ValueError(
                        "latent feature arguments cannot be used in KV feature mode")
            else:
                if latent_feature_layer is None:
                    raise ValueError(
                        "return_latent_features requires latent_feature_layer")
                if kv_feature_layers is not None or kv_token_indices is not None:
                    raise ValueError(
                        "KV feature arguments cannot be used in latent feature mode")
            if self.is_causal:
                raise ValueError(
                    "Wan feature extraction requires the frozen non-causal Wan encoder")
            if getattr(self.model, "model_type", "t2v") != "t2v":
                raise ValueError("Wan feature extraction currently supports T2V only")
            if kv_cache is not None or crossattn_cache is not None or clean_x is not None:
                raise ValueError(
                    "Wan feature extraction is mutually exclusive with "
                    "generation caches/teacher forcing")
            if noisy_image_or_video.ndim != 5:
                raise ValueError(
                    "Wan feature input must have shape [batch, frames, channels, height, width]")
            patch = tuple(int(v) for v in self.model.patch_size)
            _, frames, _, height, width = noisy_image_or_video.shape
            feature_seq_len = (
                ((frames - patch[0]) // patch[0] + 1)
                * ((height - patch[1]) // patch[1] + 1)
                * ((width - patch[2]) // patch[2] + 1)
            )
            return self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=feature_seq_len,
                kv_feature_layers=kv_feature_layers,
                kv_token_indices=kv_token_indices,
                latent_feature_layer=latent_feature_layer,
                latent_token_indices=latent_token_indices,
            )
        if latent_feature_layer is not None or latent_token_indices is not None:
            raise ValueError(
                "latent feature arguments require return_latent_features=True")

        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

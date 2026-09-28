from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        # ``model_kwargs`` already carries timestep_shift; we additionally route
        # the (optional) linear-attention configuration from the top-level
        # config into the wrapper so a single config file controls everything.
        gen_kwargs = dict(getattr(args, "model_kwargs", {}))
        attn_type = getattr(args, "attention_type", "softmax")
        # ``gdn`` (ARL² GDN hybrid) is selected like any other attention type,
        # but it is realized by injecting GDN layers into a softmax backbone
        # (see wan/modules/gdn.py), so the underlying CausalWanModel is softmax.
        is_gdn = attn_type == "gdn"
        backbone_attn_type = "softmax" if is_gdn else attn_type
        gen_kwargs.setdefault("attention_type", backbone_attn_type)
        if backbone_attn_type in ("linear", "hybrid"):
            gen_kwargs.setdefault(
                "num_frame_per_block", getattr(args, "num_frame_per_block", 1),
            )
        if backbone_attn_type == "hybrid":
            gen_kwargs.setdefault(
                "hybrid_gate", bool(getattr(args, "hybrid_gate", False)),
            )
            gen_kwargs.setdefault(
                "hybrid_gate_init_lambda", float(getattr(args, "hybrid_gate_init_lambda", 0.12)),
            )
            gen_kwargs.setdefault(
                "hybrid_gate_type", str(getattr(args, "hybrid_gate_type", "none")),
            )
            gen_kwargs.setdefault(
                "alpha_init_bias", float(getattr(args, "alpha_init_bias", -3.0)),
            )
            gen_kwargs.setdefault(
                "alpha_detach_denom_delta", bool(getattr(args, "alpha_detach_denom_delta", True)),
            )
            gen_kwargs.setdefault(
                "alpha_detach_state_summary", bool(getattr(args, "alpha_detach_state_summary", True)),
            )
            # Fast alpha-hybrid path knobs (all default-off so this is a no-op
            # for existing configs).  Honour only when the active gate type
            # is an alpha_* mode; the wrapper itself silently ignores them
            # for "none" / "lambda" / non-hybrid attention.
            gen_kwargs.setdefault(
                "use_fast_alpha_hybrid", bool(getattr(args, "use_fast_alpha_hybrid", False)),
            )
            gen_kwargs.setdefault(
                "hybrid_local_impl", str(getattr(args, "hybrid_local_impl", "torch")),
            )
            gen_kwargs.setdefault(
                "hybrid_cross_impl", str(getattr(args, "hybrid_cross_impl", "torch")),
            )
            gen_kwargs.setdefault(
                "alpha_combine_impl", str(getattr(args, "alpha_combine_impl", "torch")),
            )
            gen_kwargs.setdefault(
                "fallback_to_torch_on_error", bool(getattr(args, "fallback_to_torch_on_error", True)),
            )
            gen_kwargs.setdefault(
                "match_torch_centering", bool(getattr(args, "match_torch_centering", True)),
            )
            gen_kwargs.setdefault(
                "fast_alpha_maxperf", bool(getattr(args, "fast_alpha_maxperf", False)),
            )
            # ---- mixed-layer attention ----
            sm_indices = getattr(args, "softmax_layer_indices", None)
            sm_interval = getattr(args, "softmax_layer_interval", None)
            if sm_indices is not None:
                gen_kwargs.setdefault("softmax_layer_indices", list(sm_indices))
            if sm_interval is not None:
                gen_kwargs.setdefault("softmax_layer_interval", int(sm_interval))
            gen_kwargs.setdefault(
                "softmax_layer_local_attn_size",
                int(getattr(args, "softmax_layer_local_attn_size", -1)),
            )
            gen_kwargs.setdefault(
                "softmax_layer_sink_size",
                int(getattr(args, "softmax_layer_sink_size", 0)),
            )
        generator_is_causal = bool(getattr(args, "causal", True))
        self.generator = WanDiffusionWrapper(**gen_kwargs, is_causal=generator_is_causal)

        # ── GDN (ARL²) hybrid attention ──────────────────────────────────────
        # When ``attention_type: gdn`` (or a legacy ``arl2_*`` layer spec is
        # present) the softmax generator is converted into a GDN hybrid model:
        # HybridBlockAttention + GDN/FFN LoRA are injected at the selected
        # layers and an optional Stage-2 checkpoint is warm-started.  The whole
        # model is then trained by DMD (the ARL² DMD-continuation recipe).
        gdn_configured = is_gdn or any(
            getattr(args, name, None) is not None
            for name in (
                "gdn_layer_indices",
                "gdn_layer_indices_json",
                "arl2_hybrid_layers",
                "arl2_hybrid_layers_json",
            )
        )
        gdn_layers = None
        if gdn_configured:
            # GDN depends on the optional ``fla`` package.  Keep that
            # dependency optional for ordinary softmax/linear Wan training.
            from wan.modules.gdn import (
                gdn_layers_from_config,
                setup_gdn_generator,
            )
            gdn_layers = gdn_layers_from_config(args)

        if generator_is_causal and gdn_configured:
            assert backbone_attn_type == "softmax", (
                "GDN injection expects a softmax CausalWanModel backbone; got "
                "attention_type=%r" % attn_type)
            setup_gdn_generator(self.generator, args, load_stage2=True)
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                print(f"[base] GDN hybrid generator: injected at layers {gdn_layers}")
        elif not generator_is_causal and gdn_configured:
            raise ValueError("GDN/hybrid injection is only supported for causal generators")

        self.generator.model.requires_grad_(True)

        self.uses_score_models = bool(
            getattr(args, "uses_score_models", getattr(args, "distribution_loss", None) != "fd")
        )
        self.uses_real_score = bool(getattr(args, "uses_real_score", self.uses_score_models))
        self.uses_fake_score = bool(getattr(args, "uses_fake_score", self.uses_score_models))
        if self.uses_real_score:
            self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
            self.real_score.model.requires_grad_(False)
        else:
            self.real_score = None
        if self.uses_fake_score:
            self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
            self.fake_score.model.requires_grad_(True)
        else:
            self.fake_score = None

        # Construct the frozen 5.7B T5 encoder directly in the training dtype.
        # Building four temporary fp32 copies can otherwise exceed host memory
        # before FSDP gets a chance to shard them.
        text_dtype = (
            torch.bfloat16 if bool(getattr(args, "mixed_precision", False))
            else torch.float32
        )
        self.text_encoder = WanTextEncoder(dtype=text_dtype)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _sample_generator_input_descriptor(
        self,
        image_or_video_shape,
        initial_latent: torch.Tensor = None,
    ) -> dict:
        """Explicitly sample `_run_generator`'s block count and initial noise.

        The returned descriptor is reusable: passing it back to
        ``_run_generator`` avoids consulting the ambient RNG for those direct
        inputs.  A caller replaying the full rollout must additionally capture
        the post-sampling RNG state used by stochastic pipeline internals.
        """
        if self.args.i2v:
            noise_shape = [
                image_or_video_shape[0],
                image_or_video_shape[1] - 1,
                *image_or_video_shape[2:],
            ]
        else:
            noise_shape = list(image_or_video_shape)

        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = (
            self.num_training_frames - 1
            if self.args.independent_first_frame
            else self.num_training_frames
        )
        if max_num_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "num_training_frames is not divisible by "
                "num_frame_per_block")
        if min_num_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "minimum generated frames is not divisible by "
                "num_frame_per_block")
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        block_count = torch.randint(
            min_num_blocks,
            max_num_blocks + 1,
            (1,),
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(block_count, src=0)
        num_generated_blocks = int(block_count.item())
        num_generated_frames = (
            num_generated_blocks * self.num_frame_per_block)
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
        noise_shape[1] = num_generated_frames
        noise = torch.randn(
            noise_shape, device=self.device, dtype=self.dtype)
        return {
            "version": 2,
            "num_generated_blocks": num_generated_blocks,
            "noise": noise,
            "trajectory_seeds": torch.randint(
                0, 2**63 - 1, (noise_shape[0],),
                device=self.device, dtype=torch.int64),
        }

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        generation_descriptor: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        if generation_descriptor is None:
            generation_descriptor = (
                self._sample_generator_input_descriptor(
                    image_or_video_shape,
                    initial_latent=initial_latent,
                )
            )
        if (
            not isinstance(generation_descriptor, dict)
            or set(generation_descriptor)
            != {"version", "num_generated_blocks", "noise", "trajectory_seeds"}
            or int(generation_descriptor["version"]) != 2
        ):
            raise ValueError("invalid explicit generator input descriptor")
        num_generated_blocks = int(
            generation_descriptor["num_generated_blocks"])
        if not min_num_blocks <= num_generated_blocks <= max_num_blocks:
            raise ValueError(
                "generator descriptor block count is outside the configured "
                f"range [{min_num_blocks}, {max_num_blocks}]")
        if dist.is_available() and dist.is_initialized():
            block_min = torch.tensor(
                num_generated_blocks, device=self.device, dtype=torch.long)
            block_max = block_min.clone()
            dist.all_reduce(block_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(block_max, op=dist.ReduceOp.MAX)
            if int(block_min.item()) != int(block_max.item()):
                raise ValueError(
                    "generator descriptor block count differs across ranks")
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        expected_noise_shape = list(image_or_video_shape)
        if self.args.i2v:
            expected_noise_shape[1] -= 1
        expected_noise_shape[1] = num_generated_frames
        generation_noise = generation_descriptor["noise"]
        expected_device = (
            torch.device("cuda", int(self.device))
            if isinstance(self.device, int)
            else torch.device(self.device)
        )
        if (
            not isinstance(generation_noise, torch.Tensor)
            or tuple(generation_noise.shape) != tuple(expected_noise_shape)
            or generation_noise.device != expected_device
            or generation_noise.dtype != self.dtype
            or not bool(torch.isfinite(generation_noise).all().item())
        ):
            raise ValueError(
                "generator descriptor noise is incompatible with the "
                f"expected {tuple(expected_noise_shape)} {self.dtype} on "
                f"{expected_device}")
        trajectory_seeds = generation_descriptor["trajectory_seeds"]
        if (
            not isinstance(trajectory_seeds, torch.Tensor)
            or tuple(trajectory_seeds.shape) != (expected_noise_shape[0],)
            or trajectory_seeds.device != expected_device
            or trajectory_seeds.dtype != torch.int64
            or bool((trajectory_seeds < 0).any().item())
        ):
            raise ValueError(
                "generator descriptor trajectory_seeds must be one "
                "non-negative int64 CUDA seed per sample")

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=generation_noise,
            trajectory_seeds=trajectory_seeds,
            **conditional_dict,
        )
        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        trajectory_seeds: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            trajectory_seeds=trajectory_seeds,
            **conditional_dict,
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise
        )

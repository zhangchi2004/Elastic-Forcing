from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


def _wan_cache_geometry(generator: WanDiffusionWrapper):
    """Return the cache geometry exposed by the loaded Wan backbone."""
    model = generator.model
    try:
        num_transformer_blocks = len(model.blocks)
        num_heads = int(model.num_heads)
        dim = int(model.dim)
    except AttributeError as exc:
        raise AttributeError(
            "Wan generator.model must expose blocks, num_heads, and dim"
        ) from exc

    if num_transformer_blocks <= 0:
        raise ValueError("Wan generator.model.blocks must not be empty")
    if num_heads <= 0 or dim <= 0 or dim % num_heads != 0:
        raise ValueError(
            f"invalid Wan attention geometry: dim={dim}, num_heads={num_heads}"
        )
    return num_transformer_blocks, num_heads, dim // num_heads


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        if generator is None:
            # Mirror ``model/base.py:_initialize_models`` so that ``attention_type``
            # and ``num_frame_per_block`` from the config flow into the wrapper
            # (necessary for the block-causal linear-attention student).
            gen_kwargs = dict(getattr(args, "model_kwargs", {}))
            attn_type = getattr(args, "attention_type", "softmax")
            # ``gdn`` is injected into a softmax backbone (see wan/modules/gdn.py).
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
                # Fast alpha-hybrid path knobs (see ``model/base.py``).
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
            self.generator = WanDiffusionWrapper(**gen_kwargs, is_causal=True)
            # ── GDN (ARL²) injection — happens BEFORE the checkpoint load in
            # inference.py so the injected keys line up.  load_stage2=False:
            # inference.py loads the full generator (Stage-2 or post-DMD) itself.
            # Keep the ordinary softmax inference path independent of the
            # optional GDN/FLA stack. Importing ``wan.modules.gdn`` eagerly
            # would require ``fla`` even when this config does not request any
            # GDN layers.
            gdn_configured = is_gdn or any(
                getattr(args, name, None) is not None
                for name in (
                    "gdn_layer_indices",
                    "gdn_layer_indices_json",
                    "arl2_hybrid_layers",
                    "arl2_hybrid_layers_json",
                )
            )
            if gdn_configured:
                from wan.modules.gdn import (
                    gdn_layers_from_config,
                    setup_gdn_generator,
                )
                gdn_layers = gdn_layers_from_config(args)
                setup_gdn_generator(self.generator, args, load_stage2=False)
                print(f"[causal_inference] GDN hybrid injected at layers {gdn_layers}")
        else:
            self.generator = generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Evaluated 14B sampler: preserve the four trained non-zero
        # stages, then append Krea's terminal t=0 cleanup after warping.
        self.inference_terminal_zero_cleanup = bool(
            getattr(args, "inference_terminal_zero_cleanup", False)
        )
        if (
            self.inference_terminal_zero_cleanup
            and float(self.denoising_step_list[-1]) != 0.0
        ):
            self.denoising_step_list = torch.cat(
                (self.denoising_step_list, self.denoising_step_list.new_zeros(1))
            )

        # MMD training called scheduler.add_noise(..., t=0) before each cache
        # write.  Because extra_one_step=True, that is sigma ~= 0.00498 rather
        # than exact x0.  Match that cache-input distribution without adding
        # noise to the decoded output.
        self.inference_train_matched_context_cache = bool(
            getattr(args, "inference_train_matched_context_cache", False)
        )
        self.inference_aux_noise_seed = int(
            getattr(args, "inference_aux_noise_seed", 0)
        )
        self._inference_call_index = 0

        (
            self.num_transformer_blocks,
            self.num_heads,
            self.head_dim,
        ) = _wan_cache_geometry(self.generator)
        self.frame_seq_length = 1560

        self.attention_type = getattr(self.generator, "attention_type", "softmax")
        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        # Linear-attention student does not have a (sliding) ``local_attn_size``;
        # treat it as unbounded so the recurrent state is allocated for the full
        # video.
        self.local_attn_size = getattr(self.generator.model, "local_attn_size", -1)

        print(
            f"KV inference with {self.num_frame_per_block} frames per block "
            f"(attention_type={self.attention_type})"
        )

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _set_advance_state(self, value: bool) -> None:
        """Toggle ``advance_state`` on every linear/hybrid kv-cache dict.

        See ``SelfForcingTrainingPipeline._set_advance_state``: the flag
        controls whether the linear / hybrid recurrent state is updated in
        place on a given forward.  Pipelines set ``False`` for every
        denoising-loop pass and ``True`` only on the clean re-run so each
        block contributes EXACTLY ONCE to the persistent state, matching
        the chunkwise training semantics and aligning train / inference
        cache distributions.  No-op for softmax caches.
        """
        if self.kv_cache1 is None:
            return
        for cache in self.kv_cache1:
            if "state" in cache:
                cache["advance_state"] = value
            # ARL2 HybridBlockAttention layers: clean-pass GDN update flag.
            if "gdn_state" in cache:
                cache["update_state"] = value

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        decode: bool = True,
        standard_noise_override=None,
        aux_noise_override=None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        aux_generator = torch.Generator(device=noise.device)
        aux_generator.manual_seed(
            self.inference_aux_noise_seed + self._inference_call_index
        )
        self._inference_call_index += 1

        def aux_randn_like(tensor: torch.Tensor) -> torch.Tensor:
            if aux_noise_override is not None:
                value = next(aux_noise_override).to(tensor.device)
                return value
            return torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
                generator=aux_generator,
            )
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                cache = self.kv_cache1[block_index]
                cache["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                cache["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                # Linear/hybrid layers carry a recurrent state; softmax
                # layers carry a K/V cache.  In a MIXED model each layer
                # has only one of the two -- zero whichever is present.
                if "state" in cache:
                    cache["state"].zero_()
                if "normalizer" in cache:
                    cache["normalizer"].zero_()
                if "k" in cache:
                    cache["k"].zero_()
                if "v" in cache:
                    cache["v"].zero_()
                # ARL2 GDN layers: drop the leaked recurrent state so the next
                # video starts from a clean (lazily re-allocated) state.
                if "gdn_state" in cache:
                    cache["gdn_state"] = None
                    cache["update_state"] = True

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Initial-latent context forwards write clean K/V into the cache
            # exactly once per block -> advance_state=True.
            self._set_advance_state(True)
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # All denoising-loop forwards on this block read the same cross-block
            # cache and do NOT advance it; only the clean re-run (Step 3.3)
            # advances the cache.  No-op for softmax caches.
            self._set_advance_state(False)

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    if (
                        self.inference_terminal_zero_cleanup
                        and float(next_timestep) == 0.0
                    ):
                        transition_noise = aux_randn_like(
                            denoised_pred.flatten(0, 1)
                        )
                    else:
                        target = denoised_pred.flatten(0, 1)
                        if standard_noise_override is None:
                            transition_noise = torch.randn_like(target)
                        else:
                            transition_noise = next(standard_noise_override).to(target.device)
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        transition_noise,
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: update KV cache at t=0.  Hybrid mode noises only this
            # cache input to match training; the stored output remains clean.
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            cache_input = denoised_pred
            if self.inference_train_matched_context_cache:
                flat_cache_input = cache_input.flatten(0, 1)
                cache_input = self.scheduler.add_noise(
                    flat_cache_input,
                    aux_randn_like(flat_cache_input),
                    context_timestep.flatten(0, 1),
                ).unflatten(0, denoised_pred.shape[:2])
            # This is the single write the linear/hybrid cache gets for this
            # block -- restore the default ``advance_state=True`` before the
            # forward so the clean K/V actually lands in the cache.
            self._set_advance_state(True)
            self.generator(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output (skippable for latent-only generation, e.g.
        # ARL² Stage-2 teacher-video latent synthesis where pixels are not needed)
        if not decode:
            return (None, output) if return_latents else output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Per-layer cache shape depends on that layer's attention type:

        * ``softmax`` layers carry a sliding-window K/V tensor
          ``[B, kv_cache_size, H, Dv]`` in the activation dtype (bf16).
          The cache size matches the all-softmax baseline.
        * ``linear`` / ``hybrid`` layers carry a recurrent state
          ``[B, H, M, Dv]`` plus denominator ``[B, H, M]`` in fp32 for
          numerical stability over the long causal rollout.

        For a MIXED model (some softmax, some linear/hybrid) we walk the
        ``per_layer_self_attn_types`` list and pick the right shape per
        layer.  Pure-linear / pure-hybrid models use the same code path
        with every layer set to ``"linear_or_hybrid"``.
        """
        kv_cache1 = []
        model = self.generator.model

        # ── ARL2 path: softmax CausalWanModel with some blocks swapped to
        # HybridBlockAttention (GDN).  GDN layers use a lazily-allocated
        # gdn_state (+ update_state flag); other layers keep K/V.
        def is_gdn_block(block):
            cls = type(block.self_attn)
            return (
                cls.__name__ == "HybridBlockAttention"
                and cls.__module__ == "wan.modules.hybrid_attention"
            )

        if any(is_gdn_block(b) for b in model.blocks):
            sm_size = (self.local_attn_size * self.frame_seq_length
                       if self.local_attn_size != -1 else 32760)
            for b in model.blocks:
                entry = {
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
                if is_gdn_block(b):
                    entry["gdn_state"] = None
                    entry["update_state"] = True
                else:
                    entry["k"] = torch.zeros([batch_size, sm_size, self.num_heads, self.head_dim], dtype=dtype, device=device)
                    entry["v"] = torch.zeros([batch_size, sm_size, self.num_heads, self.head_dim], dtype=dtype, device=device)
                kv_cache1.append(entry)
            self.kv_cache1 = kv_cache1
            return

        if self.attention_type in ("linear", "hybrid"):
            # Default: every layer is linear/hybrid.  A mixed model exposes
            # an explicit per-layer types list that overrides this.
            per_layer_types = getattr(
                model, "per_layer_self_attn_types",
                ["linear_or_hybrid"] * self.num_transformer_blocks,
            )
            # Softmax-layer KV cache size mirrors the all-softmax baseline.
            softmax_cache_size = (
                self.local_attn_size * self.frame_seq_length
                if self.local_attn_size != -1
                else 32760
            )
            for layer_idx in range(self.num_transformer_blocks):
                ltype = per_layer_types[layer_idx] if layer_idx < len(per_layer_types) else "linear_or_hybrid"
                if ltype == "softmax":
                    kv_cache1.append({
                        "k": torch.zeros(
                            [batch_size, softmax_cache_size, self.num_heads, self.head_dim],
                            dtype=dtype, device=device,
                        ),
                        "v": torch.zeros(
                            [batch_size, softmax_cache_size, self.num_heads, self.head_dim],
                            dtype=dtype, device=device,
                        ),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    })
                else:
                    # linear / hybrid recurrent state.  Kernel dim defaults
                    # to head_dim but may be overridden via num_kernel_features.
                    kernel_dim = self.head_dim
                    try:
                        first_block = model.blocks[layer_idx]
                        kernel_dim = first_block.self_attn.kernel_dim
                    except (AttributeError, IndexError):
                        pass
                    kv_cache1.append({
                        "state": torch.zeros(
                            [batch_size, self.num_heads, kernel_dim, self.head_dim],
                            dtype=torch.float32, device=device,
                        ),
                        "normalizer": torch.zeros(
                            [batch_size, self.num_heads, kernel_dim],
                            dtype=torch.float32, device=device,
                        ),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    })
        else:
            # All-softmax model (CausalWanModel path; no mixing).
            if self.local_attn_size != -1:
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                kv_cache_size = 32760
            for _ in range(self.num_transformer_blocks):
                kv_cache1.append({
                    "k": torch.zeros([batch_size, kv_cache_size, self.num_heads, self.head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, self.num_heads, self.head_dim], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, self.num_heads, self.head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_heads, self.head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

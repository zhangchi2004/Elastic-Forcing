"""Autoregressive training rollouts with a sampled denoising exit step.

Each block conditions on earlier generated blocks. Training stops at a sampled
exit and differentiates that denoising call when gradients are enabled. Released
four-step inference runs the complete schedule in ``causal_inference.py``.
"""

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    """Generate on-policy training videos and reproduce their rollout trajectory."""

    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def _set_advance_state(self, value: bool) -> None:
        """Toggle the ``advance_state`` flag on every layer's kv-cache dict.

        Only the linear / hybrid attention recurrent caches read this flag
        (their kv-cache dicts have ``"state"`` and ``"normalizer"`` fields);
        the softmax cache uses overwrite-style writes, so this is a no-op
        for those layers.  See ``wan/modules/linear_attention.py`` for the
        rationale (avoid per-pass over-accumulation across the denoising
        loop's multiple forwards on the same block).
        """
        if self.kv_cache1 is None:
            return
        for cache in self.kv_cache1:
            if "state" in cache:
                cache["advance_state"] = value
            # ARL2 HybridBlockAttention layers use a GDN recurrent state and
            # read ``update_state`` instead of ``advance_state`` (clean-pass
            # update: skip writes during denoising, write on the clean rerun).
            if "gdn_state" in cache:
                cache["update_state"] = value

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        """Choose denoising exit indices on rank zero and share them across ranks."""
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Sampling an exit bounds the training graph to one denoising call
            # per block; last_step_only optionally fixes it to the final call.
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        """Return a training rollout plus its selected denoising interval.

        During feature collection the caller disables gradients. During replay,
        the same random choices reconstruct the selected calls with gradients.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
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
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            # Initial-latent context forward is a legit "clean K/V" write -> advance cache.
            self._set_advance_state(True)
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        # Retain generator gradients only for the final 21 latent frames.
        # Earlier context still affects the rollout through cached values.
        start_gradient_frame_index = num_output_frames - 21

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # All denoising-loop forwards on this block read the same cross-block
            # cache and do NOT advance it; only the clean re-run (Step 3.3)
            # advances the cache.  No-op for the softmax attention path.
            self._set_advance_state(False)

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    # Earlier denoising calls construct the selected step's
                    # input; their graphs are deliberately not retained.
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # The selected exit produces the block's clean prediction.
                    # Replay differentiates this call within the gradient window.
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: publish this generated block as context for the next.
            # This cache write has no graph, preserving the training truncation.
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            # This is the single write the linear/hybrid cache gets for this
            # block -- restore the default ``advance_state=True`` before the
            # forward so the clean K/V actually lands in the cache.
            self._set_advance_state(True)
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Per-layer cache shape depends on that layer's attention type
        (mirrors :meth:`pipeline.causal_inference.CausalInferencePipeline._initialize_kv_cache`):

          * ``softmax`` layers          -> ``k`` / ``v`` in bf16, shape
                                           ``[B, cache_size, H, Dv]``.
          * ``linear`` / ``hybrid``     -> ``state`` / ``normalizer`` in fp32,
                                           shape ``[B, H, M, Dv]`` / ``[B, H, M]``.

        For a MIXED model (some softmax + some linear/hybrid layers) we
        walk the model's ``per_layer_self_attn_types`` list and allocate
        the matching shape per layer.  Pure-linear / pure-hybrid /
        pure-softmax models still work because the type list is uniform
        in those cases.

        Crashing on the wrong key (the previous "always-linear" allocator
        handed a softmax layer a dict without ``"k"``, raising
        ``KeyError: 'k'`` deep inside the attention forward) was exactly
        the failure mode this fix addresses.
        """
        attention_type = getattr(self.generator, "attention_type", "softmax")
        model = self.generator.model
        num_heads = getattr(model, "num_heads", 12)
        head_dim = model.dim // num_heads
        kv_cache1 = []

        # ── ARL2 path: a softmax CausalWanModel with some blocks' self_attn
        # replaced by HybridBlockAttention.  Those blocks use a lazily-
        # allocated GDN state (gdn_state/update_state); the rest keep K/V.
        def is_gdn_block(block):
            cls = type(block.self_attn)
            return (
                cls.__name__ == "HybridBlockAttention"
                and cls.__module__ == "wan.modules.hybrid_attention"
            )

        if any(is_gdn_block(b) for b in model.blocks):
            for b in model.blocks:
                entry = {
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
                if is_gdn_block(b):
                    entry["gdn_state"] = None        # lazily allocated H x D x D
                    entry["update_state"] = True
                else:
                    entry["k"] = torch.zeros(
                        [batch_size, self.kv_cache_size, num_heads, head_dim], dtype=dtype, device=device)
                    entry["v"] = torch.zeros(
                        [batch_size, self.kv_cache_size, num_heads, head_dim], dtype=dtype, device=device)
                kv_cache1.append(entry)
            self.kv_cache1 = kv_cache1
            return

        if attention_type in ("linear", "hybrid"):
            # Default to "every layer is linear/hybrid"; mixed models
            # override via ``per_layer_self_attn_types``.
            per_layer_types = getattr(
                model, "per_layer_self_attn_types",
                ["linear_or_hybrid"] * self.num_transformer_blocks,
            )
            # Softmax-layer cache size.  Use the model's
            # ``softmax_layer_local_attn_size`` knob so a windowed config
            # only allocates as much storage as it actually uses.
            sm_local = int(getattr(model, "softmax_layer_local_attn_size", -1))
            if sm_local != -1:
                softmax_cache_size = sm_local * self.frame_seq_length
            else:
                softmax_cache_size = self.kv_cache_size
            for layer_idx in range(self.num_transformer_blocks):
                ltype = (
                    per_layer_types[layer_idx]
                    if layer_idx < len(per_layer_types)
                    else "linear_or_hybrid"
                )
                if ltype == "softmax":
                    kv_cache1.append({
                        "k": torch.zeros(
                            [batch_size, softmax_cache_size, num_heads, head_dim],
                            dtype=dtype, device=device,
                        ),
                        "v": torch.zeros(
                            [batch_size, softmax_cache_size, num_heads, head_dim],
                            dtype=dtype, device=device,
                        ),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    })
                else:
                    # Linear/hybrid recurrent state.  ``kernel_dim`` may
                    # be overridden via ``num_kernel_features``; default
                    # is ``head_dim`` (= 128 for Wan-1.3B).
                    kernel_dim = head_dim
                    try:
                        first_linhyb_block = model.blocks[layer_idx]
                        kernel_dim = first_linhyb_block.self_attn.kernel_dim
                    except (AttributeError, IndexError):
                        pass
                    kv_cache1.append({
                        "state": torch.zeros(
                            [batch_size, num_heads, kernel_dim, head_dim],
                            dtype=torch.float32, device=device,
                        ),
                        "normalizer": torch.zeros(
                            [batch_size, num_heads, kernel_dim],
                            dtype=torch.float32, device=device,
                        ),
                        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    })
        else:
            # All-softmax (Wan baseline) path.
            for _ in range(self.num_transformer_blocks):
                kv_cache1.append({
                    "k": torch.zeros([batch_size, self.kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, self.kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
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
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

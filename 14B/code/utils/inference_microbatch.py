"""Keep the evaluated sample-batch noise order with single-video execution."""
import torch


def install_microbatch_adapter(pipeline):
    """Draw all transition noises before inference, as in the step-80 run.

    Five samples is the evaluated setting. Generalizing the batch shape permits
    smaller demos while preserving the original draw order for five samples.
    """
    original = pipeline.inference
    prompt_index = 0

    def inference(noise, text_prompts, return_latents=True, initial_latent=None, low_memory=False):
        nonlocal prompt_index
        samples, frames, channels, height, width = noise.shape
        block_frames = pipeline.num_frame_per_block
        blocks = frames // block_frames
        template = torch.empty((samples * block_frames, channels, height, width),
                               device=noise.device, dtype=noise.dtype)
        # Three transitions between the four trained stages per block.
        standard = [torch.randn_like(template).cpu() for _ in range(blocks * 3)]
        auxiliary_rng = torch.Generator(device=noise.device).manual_seed(
            pipeline.inference_aux_noise_seed + prompt_index)
        # Terminal cleanup and context-cache writes each consume one auxiliary draw.
        auxiliary = [torch.randn(template.shape, dtype=template.dtype, device=template.device,
                                 generator=auxiliary_rng).cpu() for _ in range(blocks * 2)]
        videos, latents = [], []
        for sample in range(samples):
            start, end = sample * block_frames, (sample + 1) * block_frames
            main_it = iter([x[start:end] for x in standard])
            aux_it = iter([x[start:end] for x in auxiliary])
            video, latent = original(
                noise=noise[sample:sample + 1], text_prompts=[text_prompts[sample]],
                return_latents=True, low_memory=low_memory,
                initial_latent=None if initial_latent is None else initial_latent[sample:sample + 1],
                standard_noise_override=main_it, aux_noise_override=aux_it)
            videos.append(video.cpu())
            latents.append(latent.cpu())
            del video, latent
            pipeline.vae.model.clear_cache()
            print(f'[inference] prompt={prompt_index} sample={sample} complete', flush=True)
        prompt_index += 1
        return torch.cat(videos), torch.cat(latents)

    pipeline.inference = inference

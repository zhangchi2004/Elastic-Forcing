import argparse
import torch
import os
from omegaconf import OmegaConf
from utils.config import load_config
from tqdm import tqdm
from torchvision import transforms
import imageio.v3 as iio  # torchvision.io.write_video was removed in torchvision 0.26
from einops import rearrange


def write_video(path: str, frames: torch.Tensor, fps: int = 16) -> None:
    """Drop-in replacement for ``torchvision.io.write_video``.

    Accepts a ``[T, H, W, C]`` tensor in ``[0, 255]`` (float or uint8) and
    writes it to ``path`` as an H.264 MP4 via the imageio-ffmpeg plugin.
    """
    if frames.dtype != torch.uint8:
        frames = frames.clamp(0, 255).to(torch.uint8)
    iio.imwrite(
        path,
        frames.cpu().numpy(),
        fps=fps,
        codec="libx264",
        plugin="FFMPEG",
    )
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

import demo_utils.memory as _demo_memory
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from pipeline import causal_inference as _causal_inference_module

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

# ``demo_utils.memory.gpu`` is computed at import time as
# ``cuda:<current_device>`` and is also re-exported into
# ``pipeline.causal_inference``.  In a distributed launch every rank imports
# before ``set_device`` runs, so they all see ``cuda:0`` and end up loading
# every model onto rank 0's GPU.  Patch both module-level bindings so the
# downstream ``.to(device=gpu)`` calls land on the per-rank device.
_demo_memory.gpu = device
_causal_inference_module.gpu = device
gpu = device

print(f'[rank {local_rank}] Free VRAM on {device}: {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = load_config(args.config_path, "configs/default_config.yaml")

# Initialize pipeline
pipeline = CausalInferencePipeline(config, device=device)

def _strip_wrapper_prefixes(state_dict):
    """Strip FSDP / activation-checkpointing / torch.compile wrapper prefixes
    that may be embedded in saved state_dict keys.  ``EMA_FSDP.state_dict()``
    in particular keeps ``_fsdp_wrapped_module.`` prefixes."""
    out = {}
    for k, v in state_dict.items():
        new_k = (k.replace("_fsdp_wrapped_module.", "")
                  .replace("_checkpoint_wrapped_module.", "")
                  .replace("_orig_mod.", ""))
        out[new_k] = v
    return out


if args.checkpoint_path:
    raw_state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    key = "generator_ema" if args.use_ema else "generator"
    if key not in raw_state_dict:
        available = list(raw_state_dict.keys())
        raise KeyError(
            f"Checkpoint {args.checkpoint_path!r} has no '{key}' entry "
            f"(found: {available}).  Pass --use_ema only for DMD checkpoints saved "
            f"after ema_start_step."
        )
    cleaned = _strip_wrapper_prefixes(raw_state_dict[key])
    missing, unexpected = pipeline.generator.load_state_dict(cleaned, strict=False)
    # ``kernel_projection`` is registered persistent=False, so it is NEVER in
    # the saved state_dict and we MUST tolerate it being missing here.  Any
    # other missing/unexpected key indicates a real architecture mismatch and
    # should fail loudly.
    expected_missing = {n for n in missing if n.endswith(".kernel_projection")}
    real_missing = [n for n in missing if n not in expected_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Unexpected state-dict mismatch loading {args.checkpoint_path!r} "
            f"(key='{key}'): missing={real_missing}, unexpected={unexpected}"
        )
    if local_rank == 0:
        skipped = len(missing) - len(real_missing)
        print(
            f"Loaded {args.checkpoint_path!r} (key='{key}', "
            f"{len(cleaned)} tensors; skipped {skipped} non-persistent buffers)"
        )

if args.checkpoint_path:
    del raw_state_dict, cleaned
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)

# Preserve the evaluated sample-batch noise draws while executing one video at a time.
from utils.inference_microbatch import install_microbatch_adapter
install_microbatch_adapter(pipeline)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # Generate 81 frames
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)

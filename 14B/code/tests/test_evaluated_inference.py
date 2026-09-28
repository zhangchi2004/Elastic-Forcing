"""Exercise the actual sampler without loading the 14B weights."""
from types import SimpleNamespace
import torch
from pipeline.causal_inference import CausalInferencePipeline
from utils.scheduler import FlowMatchScheduler


class RecordingGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(blocks=[SimpleNamespace(self_attn=SimpleNamespace())], num_heads=1, dim=1, local_attn_size=1)
        self.attention_type = 'softmax'
        self.calls = []
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

    def get_scheduler(self):
        return self.scheduler

    def forward(self, noisy_image_or_video, timestep, **kwargs):
        self.calls.append((timestep.clone(), noisy_image_or_video.clone()))
        prediction = noisy_image_or_video * 0.5
        return torch.zeros_like(prediction), prediction


def make_pipeline():
    args = SimpleNamespace(denoising_step_list=[1000, 750, 500, 250], warp_denoising_step=True,
        independent_first_frame=False, num_frame_per_block=3, context_noise=0,
        inference_terminal_zero_cleanup=True, inference_train_matched_context_cache=True,
        inference_aux_noise_seed=12345)
    generator = RecordingGenerator()
    pipe = CausalInferencePipeline(args, 'cpu', generator=generator,
        text_encoder=lambda **kw: {}, vae=SimpleNamespace())
    pipe.frame_seq_length = 1
    return pipe, generator


def test_terminal_cleanup_and_noised_cache_are_executed():
    pipe, generator = make_pipeline()
    assert len(pipe.denoising_step_list) == 5
    assert pipe.denoising_step_list[-1] == 0
    noise = torch.ones(1, 6, 1, 1, 1)
    result = pipe.inference(noise=noise, text_prompts=['test'], decode=False)
    # Each block has five denoising calls and a separate cache write.
    assert len(generator.calls) == 12
    for block in range(2):
        calls = generator.calls[block * 6:(block + 1) * 6]
        assert [float(t[0, 0]) for t, _ in calls] == pipe.denoising_step_list.tolist() + [0.0]
        clean = result[:, block * 3:(block + 1) * 3]
        assert torch.equal(clean, calls[4][1] * 0.5)
        assert not torch.equal(calls[5][1], clean)  # Cache noise never overwrites output.



def test_microbatch_adapter_preserves_five_sample_noise_order():
    from utils.inference_microbatch import install_microbatch_adapter
    records = []
    def record(noise, text_prompts, standard_noise_override, aux_noise_override, **kwargs):
        records.append((noise.clone(), list(standard_noise_override), list(aux_noise_override)))
        return noise.clone(), noise.clone()
    pipe = SimpleNamespace(inference=record, num_frame_per_block=3, inference_aux_noise_seed=12345,
        vae=SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None)))
    install_microbatch_adapter(pipe)
    torch.manual_seed(0)
    noise = torch.randn(5, 21, 1, 2, 2)
    state = torch.get_rng_state().clone()
    videos, latents = pipe.inference(noise, ['test'] * 5)
    final_state = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    main = [torch.randn(15, 1, 2, 2) for _ in range(21)]
    rng = torch.Generator().manual_seed(12345)
    aux = [torch.randn(15, 1, 2, 2, generator=rng) for _ in range(14)]
    assert torch.equal(final_state, torch.get_rng_state())
    assert torch.equal(videos, noise) and torch.equal(latents, noise)
    for i, (initial, actual_main, actual_aux) in enumerate(records):
        assert torch.equal(initial, noise[i:i+1])
        assert all(torch.equal(a, b[i*3:(i+1)*3]) for a, b in zip(actual_main, main))
        assert all(torch.equal(a, b[i*3:(i+1)*3]) for a, b in zip(actual_aux, aux))

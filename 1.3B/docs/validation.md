# Validation

The GPU results below describe commit `676d5a5ec9eb3cd65152b23826f028f388886dd5`, before validation gates were removed. They were obtained on four NVIDIA H200 GPUs using the original three-encoder training assets. The gate-removal revision has separate CPU regression checks; it has not been rerun on GPUs.

| Component | Version |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| FlashAttention | 2.7.4.post1 |
| Transformers | 4.53.3 |

## Current CPU regression checks

After removing the launcher preflight gates, core reference checks, and selected encoder checks, all **15 retained CPU numerical tests passed**. They cover MMD gradients, partial-batch normalization, feature sampling, and pixel-gradient accumulation. Tests that only enforced removed gates were retired. The current revision has not been rerun on GPUs.

## Historical GPU checks

- **31 CPU tests passed**, covering MMD gradients, partial-batch normalization, feature contracts, pixel-gradient accumulation, and relocated reference identities.
- **One full-size optimizer update completed** with 256 generated videos, 128 differentiated videos, and 128 sampled references on four workers. All 272 reference feature shards were validated during initialization.
- The logged generator loss was **3.995** and the gradient norm before clipping was **44.62**. The update was not skipped. DINOv3, V-JEPA 2, and VideoMAEv2 feature replay errors were all zero in this run.
- Generator weights were finite, and the model, optimizer, and four per-worker random states were saved successfully. All 825 saved Adam parameter states recorded one update, and their gradients were not all zero.
- The new checkpoint was loaded by the shared inference implementation and generated a decodable **81-frame, 832 × 480, 16 FPS** video.

Machine-readable results are saved in [three-encoder-smoke.json](../results/validation/three-encoder-smoke.json).

The run used `bash scripts/train.sh --smoke`. This overrides only the training length to one update; the distribution batch, differentiation batch, encoders, and reference assets remain those of the released recipe.

That revision included reference-content and relocation checks. These historical results should not be interpreted as validation of the current gate-removal changes.

This is an execution check. It does not repeat the 150-update training run, rerun VBench, establish minimum hardware requirements, or validate installation in a fresh environment.

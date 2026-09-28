# Elastic Forcing — 14B

[Project overview](../README.md) · [1.3B release](../1.3B/README.md)

This directory provides the Wan2.1-T2V-14B implementation for **80 updates** of Elastic Forcing with **DINOv3, V-JEPA 2, and VideoMAEv2**. It includes the causal generator, frozen feature encoders, distribution-matching objective, gradient replay, CPU offload, checkpoint saving/resume, and inference code.

This is the non-EMA checkpoint used for the human study and VLM evaluations. See [evaluated settings](docs/evaluated-settings.md) for the model identity and distinct sampling protocols.

The step-80 recipe starts from the **Krea-provided causal ODE initialization**. It uses the first 80 updates of a 200-step cosine learning-rate schedule; shortening the learning-rate schedule to 80 steps changes the recipe.

For an implementation walkthrough, see [Reading the code](../docs/paper-to-code.md). The configuration uses the paper's distribution and differentiation terminology; `code/utils/config.py` preserves legacy configuration compatibility.

## 1. Environment

Run the following commands from `14B/`. Use Linux, Python 3.10, and CUDA-capable GPUs with BF16 support. The original training used **8 H200 GPUs** with CPU offload and approximately **1 TB of host RAM** allocated. Inference uses one GPU; this release does not establish a minimum VRAM requirement.

```bash
conda create -n elastic-forcing-14b python=3.10 -y
conda activate elastic-forcing-14b
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
export PYTHON_BIN="$(command -v python)"
```

## 2. Models and Data

See [Asset preparation](docs/assets.md) for the directory layout, frozen encoder revisions, and reference-preparation tools.

Inference requires the Wan2.1-T2V-14B base model, shared T5/tokenizer/VAE, and a trained Elastic Forcing `model.pt`. Training additionally requires the original causal ODE initialization and the **8,000-video three-encoder reference collection**. These experiment-specific weights and reference assets are not bundled in this source repository; downloading the public Wan model alone does not reproduce the trained checkpoint.

```bash
export ASSET_ROOT=/path/to/elastic-forcing-assets
huggingface-cli download Wan-AI/Wan2.1-T2V-14B \
  --local-dir "$ASSET_ROOT/models/Wan2.1-T2V-14B"

# After placing all training assets according to docs/assets.md:
python scripts/prepare_config.py --asset-root "$ASSET_ROOT"

# For inference alone, omit the training-bank requirement:
# python scripts/prepare_config.py --asset-root "$ASSET_ROOT" --inference-only
```

The setup script writes `configs/local.yaml` and `configs/local.env`, which are excluded from Git. It preserves the frozen feature-bank identity while separating the VideoMAEv2 load location from the identity stored in its reference metadata.

## 3. Inference

```bash
GPUS=0 \
CHECKPOINT=/path/to/checkpoint_model_000080/model.pt \
PROMPTS=code/prompts/test_8.txt \
OUTPUT_DIR=outputs/step80 \
bash scripts/infer.sh
```

The default matches the previous inference run: seed `0`, **five samples per prompt**, regular generator weights, and **no classifier-free guidance**. All five samples' initial and transition noises are drawn together; the generator then executes one video at a time to limit memory. Set `NUM_SAMPLES=1` for a single-video demo; this changes the random-number sequence. The seed advances continuously across prompts. Exact output reproduction also requires the same prompt order and shard split. The supplied pipeline performs four causal denoising steps followed by a terminal `t=0` cleanup, and uses the training-matched context-cache lifetime. The auxiliary noise seed is separately fixed in `configs/inference.yaml`.

The `21` latent frames requested by the launcher decode to **81 frames at 832 × 480 and 16 FPS**.

## 4. Training

After preparing the training assets and local configuration:

```bash
GPUS=0,1,2,3,4,5,6,7 \
LOGDIR=runs/three-encoder \
bash scripts/train.sh
```

Training stops after update 80. Full model, optimizer, and per-rank random-state checkpoints are saved every 10 updates. Use `runs/three-encoder/checkpoint_model_000080/model.pt` for inference. Keep all companion files for resuming training. Use a fresh `LOGDIR` for each new experiment; eight full 14B checkpoints require substantial disk space.

| Setting | Value |
|---|---|
| Generator | Wan2.1-T2V-14B, causal softmax attention |
| Initialization | Krea causal ODE checkpoint |
| GPUs | 8 |
| Distribution / differentiation / reference batch | 256 / 128 / 128 |
| Per-rank generated / differentiated videos | 32 / 16 |
| Physical generation / logical feature / physical replay batch | 2 / 16 / 1 |
| Reference collection | 8,000 videos |
| Frozen feature encoders | DINOv3 final block; V-JEPA 2 and VideoMAEv2 block index 15 |
| Loss aggregation | `(10 × DINOv3 + V-JEPA 2 + VideoMAEv2) / 3`, after per-branch normalization |
| Fixed / sampled reference attraction weights | 2 / 1 |
| Optimizer | AdamW, betas `(0, 0.999)`, weight decay `0` |
| Learning rate | Cosine `2.5e-6 → 5e-7` over **200** updates; stop at **80** |
| Generator gradient clipping | 10 |
| EMA | Disabled |
| Memory management | FSDP full sharding, activation checkpointing, optimizer/activation/frozen-encoder CPU offload |

All three encoders are frozen, but their input gradients are retained for generator training. The original configuration uses `seed: 0`, which selects an effective seed at launch; set an explicit positive training seed for a repeatable new run. Matching the recipe alone does not recover the historical RNG state.

For a one-update smoke test with the full batch geometry:

```bash
LOGDIR=runs/smoke bash scripts/train.sh --smoke
```

Resume a checkpoint produced by this release using the same eight-rank topology and asset identities:

```bash
python scripts/prepare_config.py --asset-root "$ASSET_ROOT" \
  --resume-checkpoint "$PWD/runs/three-encoder/checkpoint_model_000070/model.pt"
LOGDIR=runs/resumed bash scripts/train.sh
```

This continues through update 80. Historical checkpoints may contain deployment-specific resume identities; the inference loader accepts their generator weights, but exact training resume across repackaged identities is not guaranteed.

To run in tmux after activating the environment:

```bash
tmux new-session -s elastic-forcing-14b
bash scripts/train.sh
# Detach: Ctrl-b, then d. Reattach: tmux attach -t elastic-forcing-14b
```

## 5. Validation and Evaluation

```bash
python -m pip install pytest
python scripts/test_cpu.py
```

The CPU suite covers three-encoder configuration and feature-bank loading, partial-gradient geometry, MMD gradients, replay, offload state restoration, terminal cleanup/context-cache inference, sample-batch noise ordering, and the 200-step cosine curve. See [validation notes](docs/validation.md) for the distinction between the original GPU run and checks of this source package.

For VBench, supply the benchmark prompt list to `scripts/infer.sh` and evaluate the resulting videos with the official [VBench](https://github.com/Vchitect/VBench) implementation. A sample-video smoke test is not a benchmark reproduction.

## Acknowledgments

This implementation builds on [Self-Forcing](https://github.com/guandeh17/Self-Forcing), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [DINOv3](https://github.com/facebookresearch/dinov3), [V-JEPA 2](https://github.com/facebookresearch/vjepa2), and [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2). We thank Krea AI for providing the 14B causal ODE initialization. Upstream source notices are preserved in `code/LICENSE`; model weights and datasets retain their respective terms.

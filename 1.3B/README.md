# Elastic Forcing — 1.3B

[Project overview](../README.md) · [14B release](../14B/README.md)

---

For a guided implementation walkthrough, see [Reading the code](../docs/paper-to-code.md).
The YAML uses the paper's distribution batch, differentiation batch, and reference minibatch terminology; `code/utils/config.py` keeps legacy configurations compatible.

Elastic Forcing post-trains few-step autoregressive video generators by matching generated and reference videos in frozen representation spaces. It combines a hybrid Nyström–Monte Carlo distribution estimator with gradient replay, allowing distribution estimation to use more samples than backpropagation. Only the generator is optimized; diffusion score models are not needed during post-training.

This repository provides the **Wan2.1-T2V-1.3B three-encoder configuration**, using **DINOv3, V-JEPA 2, and VideoMAEv2**. This configuration achieves **84.25 VBench Total** in the paper.

## Supported Configuration

| Backbone | Feature encoders | Sampling | Training GPUs | Updates |
|---|---|---|---|---:|
| Wan2.1-T2V-1.3B | DINOv3 + V-JEPA 2 + VideoMAEv2 | Causal, 4 steps per chunk | 4 | 150 |

Training uses a distribution batch of **256 generated videos**, a differentiation batch of **128 generated videos**, and a reference minibatch of **128 videos**. Inference produces **81-frame, 832 × 480 videos at 16 FPS**.

## 1. Environment

Use Linux, Python 3.10, and NVIDIA GPUs with BF16 support. Run the commands below from the `1.3B/` directory.

```bash
conda create -n elastic-forcing python=3.10 -y
conda activate elastic-forcing

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation

export PYTHON_BIN="$(command -v python)"
```

## 2. Models and Data

### Inference assets

Inference needs the Wan2.1 base model, text encoder, tokenizer, VAE, and a trained Elastic Forcing checkpoint. Download the public Wan2.1 assets:

```bash
export ASSET_ROOT=/path/to/elastic-forcing-assets

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir "$ASSET_ROOT/models/Wan2.1-T2V-1.3B"
```

The base model is stored under `$ASSET_ROOT/models/Wan2.1-T2V-1.3B/`. The trained generator checkpoint can be stored separately and selected with `CHECKPOINT` when running inference.

### Training assets

Training starts from the Self-Forcing ODE initialization:

```bash
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt \
  --local-dir "$ASSET_ROOT"
```

It additionally requires:

- **Frozen feature encoders:** DINOv3 ViT-L/16, V-JEPA 2 ViT-L, and VideoMAEv2-Large.
- **Reference features:** precomputed features for the 8,682-video reference collection, with the aligned prompt/video manifest.
- **Nyström references:** one fixed summary per encoder, built from the same collection.

Reference preparation consists of extracting the three representations and building their Nyström summaries. During training, reference features are read directly from disk. See [Asset Preparation](docs/assets.md) for model sources, the directory layout, and preprocessing tools.

**The trained Elastic Forcing weights and reference collection are not distributed with this source release.** Reproducing the reported training run requires the corresponding reference assets in addition to the public model downloads.

### Configure asset paths

Generate the local configuration once the asset root is set:

```bash
"$PYTHON_BIN" scripts/prepare_config.py --asset-root "$ASSET_ROOT"
```

This creates `configs/local.yaml` from [the three-encoder configuration](configs/three_encoder.yaml). Local asset paths are excluded from version control. Download dependencies before launching a run; the training and inference scripts use local models in offline mode.

## 3. Quick Inference

Create a text file with one prompt per line, then select a trained checkpoint:

```bash
export CHECKPOINT=/path/to/elastic-forcing/model.pt

GPUS=0 \
  PROMPTS=/path/to/prompts.txt \
  OUTPUT_DIR=outputs/demo \
  bash scripts/infer.sh
```

The launcher generates one video per prompt with seed 0, using the generator weights without EMA. The four-step inference path has no additional classifier-free guidance branch. Example prompts are provided in [test_8.txt](code/prompts/test_8.txt).

## 4. Training

Prepare the environment and training assets above, then start the three-encoder run on four GPUs:

```bash
export LOGDIR=runs/elastic-forcing
GPUS=0,1,2,3 bash scripts/train.sh
```

The launcher uses FSDP and gradient checkpointing. Each worker generates 64 videos for distribution estimation and replays 32 with gradients. The feature encoders remain frozen, but gradients pass through them to the generated videos. W&B logging is disabled by the launcher.

Use a separate `LOGDIR` for each run. Training starts from the ODE initialization and runs for **150 updates**, saving model weights, optimizer state, and per-worker random states every five updates. All checkpoints are retained. The final generator checkpoint is:

```text
runs/elastic-forcing/checkpoint_model_000150/model.pt
```

Monitor progress with `tail -f "$LOGDIR/train.log"`. To generate videos from the trained model:

```bash
export CHECKPOINT="$LOGDIR/checkpoint_model_000150/model.pt"
GPUS=0 PROMPTS=/path/to/prompts.txt \
  OUTPUT_DIR=outputs/elastic-forcing bash scripts/infer.sh
```

<details>
<summary><b>Training configuration</b></summary>

| Setting | Value |
|---|---|
| Initialization | Self-Forcing ODE checkpoint |
| Distribution batch | 256 generated videos |
| Differentiation batch | 128 generated videos |
| Reference minibatch | 128 videos, sampled with replacement |
| Encoder weights | DINOv3 : V-JEPA 2 : VideoMAEv2 = 10 : 1 : 1 |
| Loss aggregation | Sum of weighted, normalized branch losses divided by 3 |
| Reference estimator | Nyström + Monte Carlo, weights 2 : 1 |
| Optimizer | AdamW, betas (0, 0.999), weight decay 0 |
| Gradient clipping | Maximum norm 10 |
| Learning rate | Cosine decay from 2.5e-6 to 5e-7 over 125 updates, then constant |
| Training length | 150 updates |
| Checkpoint interval | Every 5 updates |

The full configuration is in [configs/three_encoder.yaml](configs/three_encoder.yaml).

</details>

### Short training run

To check the complete training and checkpoint path before a full run:

```bash
export LOGDIR=runs/elastic-forcing-smoke
GPUS=0,1,2,3 bash scripts/train.sh --smoke
```

This runs **one optimizer update with the same batch sizes and all three encoders**. The resulting checkpoint is `$LOGDIR/checkpoint_model_000001/model.pt` and can be loaded with the inference command above.

<details>
<summary><b>Running in tmux and CPU tests</b></summary>

On a machine with four allocated GPUs, start a persistent session:

```bash
tmux new -s elastic-forcing
```

Inside the session, activate the environment, set `PYTHON_BIN` and `LOGDIR`, and run the training command. Detach with `Ctrl-b`, then `d`; reconnect with `tmux attach -t elastic-forcing`.

Run the CPU numerical tests separately:

```bash
"$PYTHON_BIN" -m pip install pytest
"$PYTHON_BIN" scripts/test_cpu.py
```

See the [validation record](docs/validation.md) for revision-specific test results and the GPU environment used.

</details>

## 5. Evaluation

The paper reports the following result for the three-encoder generator after 150 updates:

| Model | VBench Total ↑ | Quality ↑ | Semantic ↑ |
|---|---:|---:|---:|
| Elastic Forcing, 1.3B, three encoders | **84.25** | **85.06** | **80.99** |

The per-dimension scores are available in the [reference summary](results/reference/three-encoder-summary.json). Use the upstream [VBench](https://github.com/Vchitect/VBench) evaluator to evaluate generated videos; VBench and its evaluation models are maintained separately from this training and inference release. A short training run verifies execution and does not reproduce the reported benchmark score.

## Acknowledgments

This implementation builds on [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and [Wan2.1](https://github.com/Wan-Video/Wan2.1). We also thank the authors of [DINOv3](https://github.com/facebookresearch/dinov3), [V-JEPA 2](https://github.com/facebookresearch/vjepa2), [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), and [VBench](https://github.com/Vchitect/VBench) for making their work available. Bundled source retains its upstream license notices; model weights and datasets follow their respective upstream terms.

We are also deeply grateful to the Krea AI team for generously providing the ODE-initialized checkpoint for
Wan2.1-T2V-14B. Their support allowed us to avoid reproducing an exceptionally computationally intensive
initialization procedure at this scale, saving a substantial amount of GPU time and making our large-scale
experiments possible.

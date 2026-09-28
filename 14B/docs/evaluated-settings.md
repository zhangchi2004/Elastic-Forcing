# Evaluated 14B step-80 settings

The released configuration corresponds to the **three-encoder step-80 non-EMA model used in the human and VLM evaluations**. The earlier two-encoder / 8,852-reference configuration described a different experiment.

| Setting | Value |
|---|---|
| Generator | Wan2.1-T2V-14B, causal softmax attention |
| Initialization | Krea causal ODE; SHA-256 `01a21da77adf0e1e803d74ae73ac9c2bc397b1a05a74cbf89709a653282f7ab6` |
| Frozen encoders | DINOv3 + V-JEPA 2 + VideoMAEv2 |
| Loss weights / divisor | **10 : 1 : 1 / 3**, after detached per-encoder normalization |
| Reference videos | **8,000** |
| Distribution / differentiation / reference minibatch | **256 / 128 / 128** |
| GPUs / per-rank generated / per-rank differentiated | **8 / 32 / 16** |
| Physical generation / logical feature / physical replay batch | **2 / 16 / 1** |
| AdamW | LR `2.5e-6`, betas `(0, 0.999)`, epsilon `1e-8`, weight decay `0`, gradient clipping `10` |
| Schedule | Cosine to `5e-7` over **200** updates; take checkpoint **80** |
| Nyström / Monte Carlo mixture | **2 : 1** |
| Training video / causal chunk | 81 RGB frames at 480 × 832; 21 latent frames; 3 latent frames per chunk |
| Inference weights | `checkpoint_model_000080/model.pt`, regular `generator`, no EMA or CFG |

The original training config SHA-256 is `fba9c9abd26f9b6b08ed5450c97d01a250da2747c7325db469a2e2f204f694ab`. Public asset paths and descriptive identities are relocated; the three reference-pack hashes and numerical training settings are retained. The public recipe stops at update 80 without shortening the original 200-update learning-rate curve. `seed: 0` chooses an effective seed at startup; exact historical training replay additionally requires its effective seed and RNG state.

## Inference

The pipeline and sample-batch noise order follow the previous step-80 inference run:

- Four trained stages `[1000, 750, 500, 250]`, warped to approximately `[1000, 937.5, 833.3333, 625]`, then a terminal `t=0` cleanup.
- Training-matched context-cache writes, with auxiliary noise seed `12345`. Scheduler `t=0` has a small nonzero noise level; cache input is noised separately from the saved clean output.
- **Seed 0, five samples per prompt**, continuous random stream. Draw all five samples' initial/transition noises together, then execute one video at a time.
- 81 output frames, 832 × 480, 16 FPS.

```bash
CHECKPOINT=/path/to/checkpoint_model_000080/model.pt \
PROMPTS=/path/to/prompts.txt \
SEED=0 NUM_SAMPLES=5 OUTPUT_DIR=outputs/step80 \
bash scripts/infer.sh
```

Set `NUM_SAMPLES=1` for a smaller demo. This changes the random-number sequence. Exact historical video reproduction requires the same prompts, ordering and shard assignment in addition to the same checkpoint. The original VBench source used four contiguous shards with 237 / 237 / 236 / 236 prompt rows, restarting seed 0 for each shard. This package provides generation, not human/VLM scoring infrastructure.

See [asset preparation](assets.md) for encoder geometry and reference bandwidths, and [validation](validation.md) for checks and limits.

> Historical refactor audit: the 14B results below describe the earlier two-encoder package. The current release restores the evaluated three-encoder recipe and sampler; see [14B validation](../14B/docs/validation.md). The original baseline Git history is not bundled in this repository, so the historical comparison scripts require a separate checkout containing that revision.

# Readability refactor

Baseline: public commit `7985874ff49de326b8350a66516c50645a604f45`.
The goal is to make the released training procedure readable alongside the
paper while preserving its calculations, resolved settings and state dictionaries.
Start with [the paper-to-code guide](paper-to-code.md).

## Main names

Both model directories use the following implementation paths. The previous
paths are small compatibility shims pointing to the same modules.

| Previous implementation | Readable implementation |
|---|---|
| `model/wan_kv_mmd.py`: `WanKVMMD` | `model/elastic_forcing.py`: `ElasticForcingModel` |
| `trainer/distillation.py`: `Trainer` | `trainer/elastic_forcing.py`: `ElasticForcingTrainer` |
| `trainer/multi_representation_fresh_video_gradcache.py` | `trainer/feature_gradient_replay.py`: `FeatureGradientReplay` |
| `losses/multi_representation_video_nystrom_mmd.py` | `losses/multi_encoder_mmd.py`: `MultiEncoderMMD` |
| `fd_normalize_raw_mmd` | `normalize_mmd_with_detached_scale` |

The training stages are `_collect_rollout_microbatch`,
`_run_distribution_matching_step`, `collect_rollout_features`,
`prepare_latent_gradients`, and `replay_generator_gradients`. Multi-encoder
helper names now describe their work instead of calling a three-encoder path
"dual". Cached feature variables distinguish generated distribution features,
selected differentiation features, and reference features.

Public YAML has 32 supported readable key aliases, with concise comments at
logical boundaries. Examples include `distribution_batch_size`,
`differentiation_samples_per_rank`, `reference_minibatch_size`,
`initial_generator_checkpoint`, and `training_updates`. The resolver accepts
legacy YAML as well. An explicit readable key overrides the matching legacy
key; defaults are merged after resolving the public input. Unknown metadata
and key order are preserved. Generated local configs keep the runtime schema.
The CLI also accepts `--training-updates`, `--distribution-samples-per-rank`
and `--distribution-batch-size`, alongside the original flags.

Saved module attributes, tensor keys, reference-pack fields and training-state
identities retain their established spelling. Method interception must use the
new names, as explained in the guide. Existing `provenance/source-sha256.json`
files remain historical release records; they are not hashes of the refactored
source. The Git baseline and audit below provide the refactor comparison.

## Validation

- 1.3B CPU suite: **20 passed**.
- 14B CPU suite: **48 passed, 1 skipped**; the skipped test requires CUDA.
- All 173 resolved 1.3B training fields, 165 resolved 14B training fields, and
  19 resolved 14B inference fields match the baseline, including key order.
- Fourteen core implementation files have identical executable ASTs after
  accounting for the explicitly listed renames, documentation and compatibility aliases.
- Full-batch and selected-sample MMD toy comparisons give bitwise-identical
  losses, metadata tensors, feature gradients, normalization gradients and state dictionaries.
- Training entry-point tests exercise both the readable and legacy flags.
  Asset preparation is checked in temporary directories, including paths with spaces.

The renamed 1.3B source also completed a full first optimizer update on four
H200 GPUs with the same effective seed, assets and resolved settings as the
previous public-source run. All 424 tensor archives were compared. Initial
weights, RNG, forward inputs/features, MMD feature gradients and post-step RNG
match bitwise. Generator gradients and Adam updates do not: relative L2
differences are **0.299963%** and **7.580502%**, compared with **0.312999%** and
**7.707839%** between two earlier runs of the same original source. The first
observed difference occurs in encoder/VAE backward. The new run exited normally
and saved model, optimizer and RNG checkpoints.

See the [first-step comparison](first-step-comparison.md) for source identities,
full-vector metrics and limitations. This is one 1.3B training update; it does
not establish bitwise or long-run equivalence, nor validate 14B GPU training,
inference quality or VBench.

Run from the repository root with the release's dependencies installed:

```bash
python scripts/check_readability.py
python scripts/check_loss_parity.py --release 1.3B
python scripts/check_loss_parity.py --release 14B
python 1.3B/scripts/test_cpu.py
python 14B/scripts/test_cpu.py
```

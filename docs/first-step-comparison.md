# Readability refactor: first optimizer-step comparison

The refactored 1.3B three-encoder code completed one full optimizer update on four H200 GPUs. Initial state, sampled inputs, forward features, loss and feature gradients match the earlier public-source run bitwise. The first recorded numerical difference is in `encoder_and_vae_backward`. Compare against the same-source repeated-run control below.

## Source and run identity

- Refactor base: `7985874ff49de326b8350a66516c50645a604f45`; the GPU comparison used a source snapshot captured before committing the refactor.
- Tested source snapshot manifest SHA-256: `a806daa28aa2aa75c5adbe1e6bb930dbae822aee07cbfaf830980f6f8111b001`.
- Previous public trace commit: `985ebec542b1c2bdb7af95a4a7404b3069cc15e7`. Its code tree is byte-identical to the pre-refactor 1.3B code tree: `60b92b4230405641d8d65a048d7b7cf842a653cf`.
- Original-source control commit: `e8ec87e10c4825dfaaec993643724c5e8a2e28e9`.
- Base seed **1862643**, with rank seeds **1862643–1862646**, matching the recovered historical effective seed. A legacy setting of `seed: 0` requests seed randomization and is not the effective seed.
- Distribution / differentiation / reference batches: **256 / 128 / 128**. Four ranks, 64 forward samples and 32 backward samples per rank, microbatch size 1.
- Encoder weights **10:1:1**, loss divisor **3**, fixed-reference / Monte Carlo reference weights **2:1**.
- Same physical model/reference assets and four physical GPUs as the earlier runs. Independently queried software versions match.
- All **173** readable input configuration fields resolve to the previous values in the same order. The new config loader and `--training-updates 1` alias were exercised. The configured cosine schedule remains steps 0–125, initial learning-rate setting 2.5e-6, floor 5e-7.
- Observation hooks use the renamed methods and otherwise preserve the earlier observer. No production computation was changed for this run.

## Scalar observations

| Quantity | Previous public source | Readable source | Original A | Original B |
|---|---:|---:|---:|---:|
| Loss | 3.994718074798584 | 3.994718074798584 | 3.994718074798584 | 3.994718074798584 |
| Gradient norm before clipping | 48.80884552001953 | 48.82303619384766 | 48.82280349731445 | 48.81880187988281 |

## Full-vector differences

| Quantity | Public → readable | Original A → readable | Original A → original B |
|---|---:|---:|---:|
| Latent gradient — relative L2 | 0.507891% | 0.508414% | 0.508167% |
| Latent gradient — cosine similarity | 0.999987102368 | 0.99998707571 | 0.999987088324 |
| Latent gradient — maximum absolute difference | 4.95910644531e-05 | 6.103515625e-05 | 6.103515625e-05 |
| Generator gradient before clipping — relative L2 | 0.299963% | 0.301090% | 0.312999% |
| Generator gradient before clipping — cosine similarity | 0.999995544683 | 0.999995467272 | 0.999995104525 |
| Generator gradient before clipping — maximum absolute difference | 0.0146789550781 | 0.00958251953125 | 0.0114288330078 |
| Generator gradient after clipping — relative L2 | 0.298507% | 0.301089% | 0.312905% |
| Generator gradient after clipping — cosine similarity | 0.999995544684 | 0.999995467273 | 0.999995104526 |
| Generator gradient after clipping — maximum absolute difference | 0.00274252891541 | 0.0019583106041 | 0.00238353013992 |
| Actual Adam parameter update — relative L2 | 7.580502% | 7.667867% | 7.707839% |
| Actual Adam parameter update — cosine similarity | 0.997126798637 | 0.997060191614 | 0.99702946052 |
| Actual Adam parameter update — maximum absolute difference | 5.00679016113e-06 | 5.00679016113e-06 | 5.00679016113e-06 |

Relative L2 means `||right - left||₂ / ||left||₂`, accumulated across all four ranks. Adam update differences compare `weights_after - initial_weights`; they are not normalized by the much larger model weight norm. Initial parameters are byte-identical across all variants.

## Exact comparison with the previous public source

| Recorded stage | Bitwise equality |
|---|---|
| effective_configuration | Exact |
| initial_weights | Exact |
| initial_rng | Exact |
| forward_inputs_and_features | Exact |
| mmd_result | Exact |
| encoder_and_vae_backward | Different (645/2304 tensors) |
| gradients_before_clip | Different (918/918 tensors) |
| gradients_after_clip | Different (918/918 tensors) |
| optimizer_state | Different (1836/2754 tensors) |
| updated_weights | Different (916/3300 tensors) |
| post_step_rng | Exact |

All **424 tensor archives** (106 per rank) were compared, including all 256 forward samples and 128 selected backward samples. Equality checks compare raw bytes, including signed zeros. Configuration comparison excludes only output-directory and input-file-location fields. Timings and the final process-completion event are excluded from numerical comparison. All non-timing tensor values and structural fields remain compared.

## Interpretation and scope

The first recorded numerical difference is in `encoder_and_vae_backward`. The earlier same-source repeat has nonidentical encoder/VAE backward, generator gradients and Adam updates. The control demonstrates backward execution variability in this GPU configuration; it does not identify a specific nondeterministic kernel or establish a statistical tolerance from one control pair.

The public-to-readable gradient relative L2 difference is 0.9583 times the measured same-source control difference; for the Adam update it is 0.9835 times. These ratios contextualize the observed discrepancies; they do not define a pass/fail tolerance or guarantee matching long-run trajectories.

The readable-source run completed backward, clipping, Adam, native checkpoint saving, and all four ranks successfully (exit code 0). The previous public run also completed successfully. Original A completed the first update and every compared trace, then failed at checkpoint saving because its audit output directory was absent; original B completed normally.

This comparison concerns reruns under the historical settings, not tensors captured during the original historical first step. It covers 1.3B only and does not evaluate 14B training, all training updates, generated-video quality, or VBench.

## Software environment

`{"cuda": "12.1", "cudnn": 90100, "flash_attn": "2.7.4.post1", "numpy": "1.26.4", "python": "3.10.12", "torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121", "transformers": "4.53.3"}`

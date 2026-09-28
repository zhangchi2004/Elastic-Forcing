# Reading the Elastic Forcing implementation

Start with the training update, then follow the loss and replay calls. The paper's
Sections 3.1–3.3 describe the objective and the separation between estimating a
distribution and differentiating selected rollouts. Appendix C.4 and Algorithm 1
describe the replay procedure. The notes below distinguish that mathematical
presentation from the numerical choices retained in this release.

## Reading order

The two model directories have the same main entry points. Start with 1.3B for the
three-encoder configuration; the released 14B configuration also uses DINOv3,
V-JEPA 2, and VideoMAEv2.

| Read | Purpose | 1.3B | 14B |
|---|---|---|---|
| Configuration | Batch sizes, encoders, loss weights, initialization and optimizer | [three_encoder.yaml](../1.3B/configs/three_encoder.yaml) | [train.yaml](../14B/configs/train.yaml) |
| Training entry | Load configuration and start distributed training | [train.py](../1.3B/code/train.py) | [train.py](../14B/code/train.py) |
| `ElasticForcingTrainer` | Assemble one complete update: collect rollouts, compute feature gradients, replay and update the generator | [trainer/elastic_forcing.py](../1.3B/code/trainer/elastic_forcing.py) | [trainer/elastic_forcing.py](../14B/code/trainer/elastic_forcing.py) |
| `FeatureGradientReplay` | Gather the distribution batch and compute gradients with respect to selected features | [trainer/feature_gradient_replay.py](../1.3B/code/trainer/feature_gradient_replay.py) | [trainer/feature_gradient_replay.py](../14B/code/trainer/feature_gradient_replay.py) |
| `MultiEncoderMMD` | Combine the independently scaled encoder losses | [losses/multi_encoder_mmd.py](../1.3B/code/losses/multi_encoder_mmd.py) | [losses/multi_encoder_mmd.py](../14B/code/losses/multi_encoder_mmd.py) |
| `VideoNystromMMD` | Evaluate generated–generated repulsion and hybrid reference attraction | [losses/video_nystrom_mmd.py](../1.3B/code/losses/video_nystrom_mmd.py) | [losses/video_nystrom_mmd.py](../14B/code/losses/video_nystrom_mmd.py) |
| `ElasticForcingModel` | Connect rollout generation, frozen encoders, VAE decoding and generator replay | [model/elastic_forcing.py](../1.3B/code/model/elastic_forcing.py) | [model/elastic_forcing.py](../14B/code/model/elastic_forcing.py) |

The previous module paths remain compatibility shims. Read the implementations
linked above rather than the shims. Low-level network definitions in `wan/` are
useful after the training flow is clear.

In `ElasticForcingTrainer`, start at `_run_distribution_matching_step` and
follow `_collect_rollout_microbatch`. In `ElasticForcingModel`, the corresponding
stages are `collect_rollout_features`, `prepare_latent_gradients` and
`replay_generator_gradients`.

## Three different batch sizes

| Paper symbol | Meaning | Released settings |
|---|---|---|
| $B$ | **Distribution batch:** generated videos that participate in the loss | 256 globally; 64 per GPU on 1.3B, 32 per GPU on 14B |
| $b$ | **Differentiation batch:** generated videos replayed for generator backpropagation | 128 globally; 32 per GPU on 1.3B, 16 per GPU on 14B |
| $B_{\mathrm{ref}}$ | **Reference minibatch:** stored reference-video features used by the Monte Carlo term | 128 globally |

The readable configuration calls these `distribution_batch_size`,
`distribution_samples_per_rank`, `differentiation_samples_per_rank` and
`reference_minibatch_size`. `differentiation_gradient_scale` is $B/b$;
`encoder_loss_divisor` is the final loss divisor. `utils/config.py` translates
these names into the compatible runtime fields without changing their values.

A microbatch is the smaller group processed at once; it is not $B$ or $b$.
All 256 generated videos affect the loss, including the 128 whose generator
computation is not replayed. Computing independent MMD losses on microbatches
would omit their cross-microbatch interactions.

Compatibility names can be misleading: `student_reference` means the full
**generated** population, while `current_teacher` means the **reference
minibatch**. Reference features are read from a frozen feature bank; this path
does not run an online diffusion teacher.

## One optimizer update

```text
Prompts + noise
    → generate B videos without retained generator graphs
    → extract frozen-encoder features and retain replay state
    → gather the complete distribution batch across GPUs
    → combine persistent Nyström statistics with a reference minibatch
    → compute feature gradients for the b selected videos
    → replay encoders: feature gradients → pixel gradients
    → replay VAE: pixel gradients → latent gradients
    → replay generator: latent gradients → parameter gradients
    → accumulate contributions, clip gradients, update once
```

A **cotangent** is the incoming gradient used in a vector–Jacobian product. For
example, a feature cotangent is $g_i^e=\partial L/\partial z_i^e$; replay passes it
back through the corresponding encoder and generator. Frozen encoder and VAE
parameters are not optimized, but derivatives with respect to their inputs are
still required. Replay restores the original noise, random choices and segment
inputs so that the gradient is paired with the intended computation.

Two scaling factors have different purposes. The configured $B/b=2$ compensates
for replaying only part of the distribution batch. Multiplication by the number
of GPUs compensates for FSDP averaging parameter gradients across ranks.

## Connecting the loss to the paper

Section 3.1 compares generated and reference distributions in frozen encoder
spaces. Section 3.2 combines two estimates of the reference kernel mean:

- **Persistent Nyström mean:** landmark kernels multiplied by precomputed
  coefficients, as in Equation (8) and Appendix C.2, Equations (18)–(21).
- **Monte Carlo mean:** average kernel similarity to a sampled reference
  minibatch, as in Equation (6).

The released Nyström:Monte Carlo weights are 2:1, so the mixture in Equation (9)
has $\alpha_e=1/3$. The tensor saved under the legacy name `alpha` is instead
the **Nyström coefficient vector**, denoted $\boldsymbol\beta_e$ in Equation
(19). These are different quantities.

For each encoder, the released implementation evaluates

$$
M_e=\frac{1}{B^2}\sum_{i,j}k_e(z_i,z_j)
-\frac{2}{B}\sum_i\left[\frac{2}{3}m^e_{\mathrm{Nys}}(z_i)
+\frac{1}{3}\widehat m^e_{\mathrm{MC}}(z_i)\right]+C_e,
$$

where $C_e$ is the retained reference–reference contribution. It then applies
detached per-encoder loss scaling:

$$
L=\frac{1}{d}\sum_e w_e
\frac{M_e}{\operatorname{stopgrad}(M_e)+\epsilon}.
$$

Here $d$ is the explicit aggregation divisor; it is not inferred from the sum
of the weights. Both 1.3B and the human/VLM-evaluated 14B use
DINOv3:V-JEPA 2:VideoMAEv2 = 10:1:1. Both configurations use $d=3$ and
$\epsilon=10^{-4}$.

## Details to preserve when reviewing or refactoring

- **Self-pairs and constants.** Paper Equations (11)/(13) use the off-diagonal
  denominator $B(B-1)$ and omit reference–reference constants. This release
  retains self-pairs with denominator $B^2$ and retains the constants. Although
  a reference constant has no direct generator derivative, it affects the
  detached normalization denominator above. Changing either convention changes
  the released numerical objective.
- **Replay selection.** Algorithm 1 uniformly samples a subset. The released
  implementation selects a fixed suffix of each rank's generated batch and
  creates feature-gradient leaves for those selected samples. Their gradients
  still include interactions with the full distribution batch. The conditional
  unbiasedness statement in Equation (27) assumes uniform subset sampling;
  it should not be claimed for this fixed-suffix implementation.
- **VAE temporal derivatives.** The active model paths call `replay_decoder_vjp`,
  which carries temporal-cache values but detaches their gradients between latent
  frames. This preserves decoded values while truncating cross-frame cache
  derivatives. The 14B tree also provides `replay_decoder_vjp_exact` and
  `wan_recurrent_decoder_vjp.py`, which propagate future-state cotangents; their
  presence does not mean the released training path selects them. Appendix C.4
  distinguishes prescribed gradient truncation from exact replay.
- **Compatibility.** Resource-pack keys, checkpoint parameter names and legacy
  configuration fields remain supported. Their historical wording is not an
  additional training objective or a requirement to instantiate score models.

## Instrumentation and compatibility names

Old imports and direct calls remain available as aliases. New training code
calls the readable method names, so observers that intercept methods should
patch `_collect_rollout_microbatch`, `_run_distribution_matching_step`,
`collect_rollout_features`, `prepare_latent_gradients`, and
`replay_generator_gradients`. Patching an old method alias does not intercept
calls dispatched through the new name.

Edit the public YAML when changing experiment settings. `prepare_config.py`
resolves those readable fields and writes the legacy runtime schema to
`configs/local.yaml`; this preserves resource and checkpoint identities.

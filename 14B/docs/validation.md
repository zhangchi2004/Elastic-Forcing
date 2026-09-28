# Validation

## Evaluated checkpoint

The restored recipe is the **three-encoder, 8,000-reference-video run**. The original checkpoint header records step **80**, **1,095 generator parameter tensors**, and all three branches: DINOv3, V-JEPA 2, and VideoMAEv2. Its bank identity matches the released training config. The historical human-evaluation generation records and the VLM source-video run point to this same checkpoint.

The original step-80 log reports loss **3.854**, generator gradient norm **10.98**, three encoder VJPs, batch geometry **256 / 128 / 128**, zero reported feature replay error for all three encoders, and a completed optimizer update. These are historical measurements, not a new run of this package. See [`original-step80.json`](../results/original-step80.json).

## Checks of the corrected release

- **51 CPU tests passed; 1 CUDA-only test skipped.** Coverage includes three-encoder configuration, bank loading, MMD gradients, replay/offload, the 200-update cosine curve, DINOv3 metadata relocation, terminal cleanup and context-cache inference, and five-sample noise ordering.
- All **184 fields** from the original training configuration were accounted for. Asset paths and descriptive identities are public equivalents; the public stopping point is 80 rather than 200, while the cosine horizon remains 200. The three active encoders are specified explicitly.
- The public sampler was compared with the actual historical evaluation sampler using a small recording generator and the real scheduler. Outputs and every generator input/timestep matched exactly for two consecutive prompts, both with ordinary random draws and with supplied noise tensors.
- The five-sample adapter draws the same 21 main transition tensors and 14 auxiliary tensors per prompt as the original 81-frame inference adapter, before single-video execution. CPU tests verify noise slices and final RNG state.
- DINOv3 metadata relocation reproduces the complete actual frozen-bank metadata and digest without loading models from the original paths.
- The single-encoder MMD kernel, training rollout, and encoder VJP implementation match the original three-encoder snapshot's executable AST after removing documentation and restoring readability aliases.
- Python parsing, shell syntax, and diff whitespace checks passed.

Path-free evidence is in [`release-validation.json`](../results/release-validation.json). Source hashes now refer to the correct three-encoder training snapshot in [`source-sha256.json`](../provenance/source-sha256.json); hashes do not imply all files are identical.

**Full 14B GPU training, GPU video generation, and VBench were not rerun for this correction.** CPU sampler comparisons use a small test generator, not the 14B network, and do not establish identical generated videos or a new benchmark score.

Run the CPU suite from `14B/`:

```bash
python scripts/test_cpu.py
```

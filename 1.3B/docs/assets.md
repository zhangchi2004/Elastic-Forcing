# Asset Preparation

The source release includes training and evaluation code, the training configuration, and example prompts. Model weights, training reference data, and generated videos are stored separately.

## Required assets

| Use | Assets |
|---|---|
| Inference | Wan2.1-T2V-1.3B base weights, text encoder, tokenizer, VAE, and a trained Elastic Forcing checkpoint |
| Training | Inference base assets, the Self-Forcing ODE initialization, three frozen feature encoders, reference features, and Nyström summaries |

The trained Elastic Forcing checkpoint and reference collection are not distributed in this repository. Public encoder downloads alone are insufficient to reproduce training with the provided configuration.

## Public model sources

- [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- [Self-Forcing initialization](https://huggingface.co/gdhe17/Self-Forcing)
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [V-JEPA 2 ViT-L](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)
- [VideoMAEv2-Large](https://huggingface.co/OpenGVLab/VideoMAEv2-Large)

For the supplied configuration, DINOv3 uses ViT-L/16 features from its final layer; V-JEPA 2 and VideoMAEv2 use layer 15. VideoMAEv2 is pinned to revision `9981a9c8f77118c421e5228e1b219468a4b0238d`, and the DINOv3 source is pinned to commit `6876159a11b4df116f30f667f8c9888617df0751`.

## Directory layout

Place assets under a common root. The following filenames are retained for compatibility with the supplied configuration:

```text
ASSET_ROOT/
├── checkpoints/ode_init.pt
├── models/
│   ├── Wan2.1-T2V-1.3B/
│   ├── dinov3-src/
│   ├── dinov3_vitl16_lvd1689m.pth
│   ├── dinov3_vitl16_lvd1689m.pth.provenance.json
│   ├── vjepa2-vitl-fpc64-256/
│   └── VideoMAEv2-Large/
├── teacher_banks/triple_video_new8682_824/
│   ├── manifest.json
│   └── ...                         # All feature shards referenced by the manifest
└── experiments/triple_video_new8682_824/
    ├── source/new8682_824_matched.jsonl
    └── references/
        ├── dinov3_vitl16_f20mean5_k1024_m256_matched5k.pt
        ├── vjepa2_l15_last64_k1024_m1024_matched5k.pt
        └── videomae_v2_l15_fixed5x16_k256_m256_matched5k.pt
```

The reference collection contains 8,682 videos stored in 272 feature shards. The `matched5k` suffix is a legacy filename; the manifest determines the actual collection size.

Keep the feature shards, manifests, and summaries together without changing their contents or sample ordering. Preserve the DINOv3 provenance file and the VideoMAEv2 `download_manifest.json` alongside their weights.

## Configure asset paths

From the `1.3B/` directory:

```bash
export ASSET_ROOT=/path/to/elastic-forcing-assets
python scripts/prepare_config.py --asset-root "$ASSET_ROOT"
```

The setup script generates a local configuration with relocated asset paths. Generated configurations are excluded from version control. Asset and source-hash preflight checks are not run by the launchers.

## Reference feature preparation

Preprocessing is performed once, before generator training. It requires the raw captioned reference videos, the frozen encoders, and a CUDA device. Training subsequently reads the saved features instead of decoding those reference videos again.

1. **Prepare the prompt/video manifest.** Each entry binds a caption to an MP4 basename in a fixed sample order. The loader in [`matched_video_manifest.py`](../code/utils/matched_video_manifest.py) expects a JSONL manifest and its `.meta.json` sidecar, including the source-data identity. A plain prompt text file is not a training manifest.
2. **Extract all three representations.** [`extract_triple_matched_teacher_bank.py`](../code/tools/extract_triple_matched_teacher_bank.py) reads the manifest and raw videos and writes feature shards plus a `manifest.json`. Supply the source manifest, video root, encoder paths, and model identities. For the released collection, set `--expected-count 8682` and `--shard-rows 32`; the default sample count in this lower-level tool is different.
3. **Build the fixed summaries.** [`build_triple_nystrom_references.py`](../code/tools/build_triple_nystrom_references.py) reads the extracted feature bank and writes one Nyström reference per encoder. Supply the bank and model identities and set `--expected-sample-count 8682`. Keep the reference files together with the feature bank from which they were built.

Run the preprocessing tools from the `1.3B/` directory with `PYTHONPATH="$PWD/code"`. Their `--help` output lists the required manifests, encoder identities, and output paths.

The feature protocol used by the training configuration is:

| Encoder | Video sampling | Representation | Nyström landmarks |
|---|---|---|---:|
| DINOv3 ViT-L/16 | 20 frames across the 81-frame video, pooled in groups of 5 | Final-layer features, 1,024 tokens per video | 256 |
| V-JEPA 2 ViT-L | Last 64 contiguous frames | Layer 15, 1,024 tokens per video | 1,024 |
| VideoMAEv2-Large | Skip the first frame, then use five 16-frame clips | Layer 15, 256 tokens per clip | 256 |

Encoder revisions, feature sampling, source ordering, and preprocessing metadata must agree between the reference assets and training configuration. Re-extraction can change serialized hashes; it should not be assumed to recreate the supplied assets byte for byte.

## Using another reference collection

A new collection requires a new manifest, feature bank, Nyström summaries, and a training configuration with matching paths and identities.

Update the local configuration for a new experiment. Changing the reference collection should not be expected to reproduce the reported VBench score.

# 14B Asset Preparation

The evaluated model is the **three-encoder step-80 non-EMA generator**, trained on **8,000 reference videos**. Its asset identities are recorded in [`configs/train.yaml`](../configs/train.yaml). The former two-encoder / 8,852-video bank is a different experiment and cannot reproduce this model.

Download locations are listed in the project [Hugging Face collection](https://huggingface.co/collections/liuyueyi-8/elastic-forcing-6aba0e62a4416b13ae7b13bf). Use the matching three-encoder checkpoint and reference assets; filenames or the step number alone do not identify the experiment. This source repository does not include model weights or reference tensors. The official Krea final distilled model is not interchangeable with the Krea-provided **causal ODE initialization** needed for training.

## Directory layout

```text
ASSET_ROOT/
├── models/
│   ├── Wan2.1-T2V-14B/          # Base generator, T5, tokenizer, and VAE
│   ├── dinov3-src/             # Official DINOv3 source at the pinned commit
│   ├── dinov3_vitl16_lvd1689m.pth
│   ├── vjepa2-vitl-fpc64-256/
│   └── VideoMAEv2-Large/       # Includes download_manifest.json
├── checkpoints/
│   └── krea-realtime-14b-causal-ode-init.safetensors
├── teacher_bank/
│   ├── manifest.json
│   └── ...                    # Preserve all shard-relative paths
├── source/
│   ├── matched.jsonl
│   └── matched.meta.json
├── references/
│   ├── dinov3.pt
│   ├── vjepa2.pt
│   └── videomae_v2.pt
└── videos/                    # Feature extraction only; not decoded during bank-based training
```

Preserve original serialized manifests, reference packs, and shards. The three reference files may be renamed as above without changing their bytes. `prepare_config.py` reads DINOv3 and VideoMAEv2 metadata identities from the bank and keeps them separate from local encoder load paths. These local settings are excluded from Git.

## Frozen encoders

Wan2.1-T2V-14B includes the shared T5 weights, VAE weights, and tokenizer. Download it as described in the model README. Before offline training, prepare the other encoders:

```bash
git clone https://github.com/facebookresearch/dinov3.git "$ASSET_ROOT/models/dinov3-src"
git -C "$ASSET_ROOT/models/dinov3-src" checkout 6876159a11b4df116f30f667f8c9888617df0751
# Obtain dinov3_vitl16_lvd1689m.pth through the official DINOv3 model access process.
huggingface-cli download facebook/vjepa2-vitl-fpc64-256 \
  --local-dir "$ASSET_ROOT/models/vjepa2-vitl-fpc64-256"
PYTHONPATH="$PWD/code" python code/tools/download_videomae_v2_large.py \
  --output-dir "$ASSET_ROOT/models/VideoMAEv2-Large"
```

The exact encoder hashes are in the training config; a newer checkpoint is not interchangeable with a frozen reference bank. VideoMAEv2 uses revision `9981a9c8f77118c421e5228e1b219468a4b0238d`.

| Encoder | Input and representation | Nyström landmarks | RBF bandwidth |
|---|---|---:|---:|
| DINOv3 ViT-L/16 | 20 sampled frames resized to 256²; final normalized patch features; mean each group of five frames; 1,024 × 1,024 matrix | 256 | 341.2794189453125 |
| V-JEPA 2 ViT-L | Frames 17–80; short side 292, center crop 256²; block index 15; 1,024 × 1,024 matrix | 1,024 | 3025.305908203125 |
| VideoMAEv2-Large | Frames 1–80 split into five 16-frame clips; short side/crop 224; block index 15; 256 tokens per clip, 1,280 × 1,024 per video | 256 | 3204.123291015625 |

VideoMAEv2 shares one frozen encoder across the five clips. All branches use ImageNet pixel normalization. The video-level kernel is `exp(-||X-Y||_F² / (2 sigma²))`, without dividing the distance by the number of tokens or channels. The fixed Nyström references use ridge `1e-6` and 20 k-means iterations. The loss normalization epsilon is separately `1e-4`. Full metadata is in [`reference-settings.json`](../provenance/reference-settings.json).

## Rebuilding a reference bank

Rebuilding requires the original captioned videos and matched manifest. Re-extraction may change serialized hashes; another video collection defines a new experiment. The included three-encoder tools are:

```bash
PYTHONPATH="$PWD/code" python code/tools/extract_triple_matched_teacher_bank.py --help
PYTHONPATH="$PWD/code" python code/tools/build_triple_nystrom_references.py --help
```

Use `--expected-count 8000 --shard-rows 32` for extraction. Pass all three pinned encoder locations and identities, the source manifest and CSV hashes, and the video root. For reference construction use `--expected-sample-count 8000 --expected-shard-size 32 --kmeans-iterations 20 --ridge 1e-6`, together with the resulting bank identities. The builder fixes the landmark counts to DINOv3 256, V-JEPA 2 1,024, and VideoMAEv2 256.

For a rebuilt bank, update a local copy of the training config with its new manifest/content hashes and all three reference filenames/hashes before preparing local paths. The optional `*_no_dinov3_*` tools are two-encoder ablation utilities and are not part of this recipe.

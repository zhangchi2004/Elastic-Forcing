<div align="center">

# Elastic Forcing

### From Scores to Samples: Elastic Forcing for Autoregressive Video Generation

Chi Zhang<sup>1 &#42;</sup> · Yueyi Liu<sup>1,2 &#42;</sup> · Haoyang Shi<sup>1,3 &#42;</sup> · Ruichuan An<sup>4</sup><br>
Haoyu Li<sup>2</sup> · Yuhang Wu<sup>1</sup> · Sen Cui<sup>5</sup> · Miao Liu<sup>1 †</sup>

<sup>1</sup> College of AI, Tsinghua University<br>
<sup>2</sup> IAIR, Xi’an Jiaotong University · <sup>3</sup> Xianghui Academy, Fudan University<br>
<sup>4</sup> Peking University · <sup>5</sup> BAAI

<sup>&#42;</sup> Equal contribution. &nbsp; <sup>†</sup> Corresponding author.

[Contact Chi Zhang](mailto:imzc.2004@gmail.com) · [Contact Miao Liu](mailto:miaoliu@mail.tsinghua.edu.cn)

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arxiv&logoColor=white)](docs/elastic-forcing.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-orange?style=flat&logo=googlechrome&logoColor=white)](https://video-examples-m8r2v6.pages.dev/)
[![GitHub Code](https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white)](https://github.com/zhangchi2004/Elastic-Forcing)
[![Hugging Face Collection](https://img.shields.io/badge/Hugging%20Face-Collection-FFD21E?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/collections/liuyueyi-8/elastic-forcing-6aba0e62a4416b13ae7b13bf)

[1.3B Training & Inference](1.3B/README.md) · [14B Training & Inference](14B/README.md)

</div>

---

Elastic Forcing post-trains few-step autoregressive video generators by matching generated and reference videos in frozen representation spaces. It combines a hybrid Nyström–Monte Carlo distribution estimator with gradient replay. Only the generator is optimized; diffusion score models are not needed during post-training.

Read the implementation through the [paper-to-code guide](docs/paper-to-code.md), which maps the paper's notation to configuration fields, the training loop, the hybrid MMD objective, and gradient replay.

## Models

Each model directory contains its own implementation, configuration, dependency list, asset guide, and training/inference scripts. Run its commands from that directory.

| Release | Frozen feature encoders | Training | Inference | Guide |
|---|---|---|---|---|
| **1.3B** | DINOv3 + V-JEPA 2 + VideoMAEv2 | 4 GPUs, 150 updates | 4 causal steps per chunk | [Training and inference](1.3B/README.md) |
| **14B** | DINOv3 + V-JEPA 2 + VideoMAEv2 | 8 GPUs, 80 updates | 4 trained steps + terminal zero-step cleanup | [Training and inference](14B/README.md) |

Both configurations use a distribution batch of 256 generated videos, a differentiation batch of 128 videos, and a reference minibatch of 128 videos. They generate 81-frame, 832 × 480 videos at 16 FPS.

The 1.3B three-encoder configuration achieves **84.25 VBench Total** in the paper. The 14B release targets the **step-80 checkpoint**, retaining the first 80 updates of the original **200-step cosine learning-rate schedule**. This is the **three-encoder, 8,000-reference-video model used in the human and VLM evaluations**; see [evaluated settings](14B/docs/evaluated-settings.md).

```text
Elastic-Forcing/
├── 1.3B/                # Three-encoder training and inference
│   ├── code/
│   ├── configs/
│   ├── scripts/
│   └── README.md
├── 14B/                 # Three-encoder training and step-80 inference
│   ├── code/
│   ├── configs/
│   ├── scripts/
│   └── README.md
└── docs/elastic-forcing.pdf
```

## Getting Started

```bash
git clone https://github.com/zhangchi2004/Elastic-Forcing.git
cd Elastic-Forcing

# Choose one model, then follow its README.
cd 14B
# Or: cd 1.3B
```

See the [1.3B asset guide](1.3B/docs/assets.md) or [14B asset guide](14B/docs/assets.md) for model and reference-data requirements. This GitHub release contains source code and the paper; trained generator weights, the 14B ODE initialization, and frozen reference assets are stored separately and are not bundled here. Download the released model checkpoints and training data from our [Hugging Face collection](https://huggingface.co/collections/liuyueyi-8/elastic-forcing-6aba0e62a4416b13ae7b13bf).

Validation is documented separately for [1.3B](1.3B/docs/validation.md) and [14B](14B/docs/validation.md).

## Acknowledgments

This implementation builds on [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and [Wan2.1](https://github.com/Wan-Video/Wan2.1). We also thank the authors of [DINOv3](https://github.com/facebookresearch/dinov3), [V-JEPA 2](https://github.com/facebookresearch/vjepa2), [VideoMAEv2](https://github.com/OpenGVLab/VideoMAEv2), and [VBench](https://github.com/Vchitect/VBench) for making their work available. Bundled source retains its upstream license notices; see [third-party sources](THIRD_PARTY_NOTICES.md) for attribution and license references. Model weights and datasets follow their respective upstream terms.

We are also deeply grateful to the Krea AI team for generously providing the ODE-initialized checkpoint for
Wan2.1-T2V-14B. Their support allowed us to avoid reproducing an exceptionally computationally intensive
initialization procedure at this scale, saving a substantial amount of GPU time and making our large-scale
experiments possible.

# Third-party sources

Elastic Forcing builds on the following projects. The repository's Apache-2.0
license is provided in [LICENSE](LICENSE), with copies in each model's `code/`
directory. Existing upstream copyright headers are retained.

| Project | Use in this repository | Upstream license |
|---|---|---|
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | Base autoregressive video training and inference implementation, adapted for Elastic Forcing | [Apache-2.0](https://github.com/guandeh17/Self-Forcing/blob/main/LICENSE) |
| [Wan2.1](https://github.com/Wan-Video/Wan2.1) | Wan model components under `1.3B/code/wan/` and `14B/code/wan/`, including adapted causal components | [Apache-2.0](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt) |
| [FramePack](https://github.com/lllyasviel/FramePack) | Memory helpers in both `code/demo_utils/memory.py` files, as identified by their existing source comments | [Apache-2.0](https://github.com/lllyasviel/FramePack/blob/main/LICENSE) |

DINOv3, V-JEPA 2, VideoMAEv2, and optional encoder implementations are external
model/source dependencies. Their checkpoints and datasets are not included in
this Git repository. Obtain them from their official sources and follow their
own licenses and access conditions; this repository's license does not relicense
those external assets. See the [1.3B asset guide](1.3B/docs/assets.md) and
[14B asset guide](14B/docs/assets.md).

The Krea-provided ODE initialization is also a separate asset. Its availability
and redistribution terms must be established with its provider independently of
the source-code license.

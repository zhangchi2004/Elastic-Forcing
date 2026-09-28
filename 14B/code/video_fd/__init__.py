"""Stride-4 video-encoder FD with replay gradients.

A self-contained subsystem for computing a video FD-loss on *sparse* features from
real video encoders (VideoPrism-B / V-JEPA2 / VideoMAE-v2), supervised at an
effective 4 fps via a stride-4 latent-aligned clip sampler, with memory-efficient
feature-gradient replay (encoder VJP + VAE decoder VJP).

Submodules:
* ``samplers``  -- stride-4 contiguous-clip sampler.
* ``encoders``  -- unified sparse video-encoder adapters (+ a synthetic encoder).
* ``losses``    -- stride-4 video FD loss (content / delta / text-joint).
* ``replay``    -- encoder VJP + frame gather/scatter + decoder VJP.
* ``stats``     -- target-statistic precomputation.
"""

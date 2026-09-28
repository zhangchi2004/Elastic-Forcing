"""Replay VJP: feature gradients -> pixel-frame gradients -> latent gradient.

Stage B1 (encoder VJP): for each sampled clip, re-run the frozen encoder (+ the
differentiable normalize/project) **with gradients on the clip**, form the linear
surrogate ``<F_proj, sg(g_F_proj)>``, take its gradient w.r.t. the clip, and
scatter-add into the full 81-frame pixel-gradient buffer.

Stage B2 exposes two decoder helpers. The released feature-matching model calls
``replay_decoder_vjp``, which detaches temporal caches between latent frames.
The optional ``replay_decoder_vjp_exact`` helper preserves temporal derivatives
using recurrent boundary replay; it is not selected by that training path.

Frozen parameter gradients remain disabled; replay builds only the input and
explicit recurrent-state graphs required for the requested VJP.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from video_fd.replay.scatter_frame_grads import gather_frames, scatter_add_frames, new_frame_grad_buffer
from video_fd.replay.wan_recurrent_decoder_vjp import (
    replay_wan_decoder_vjp_exact,
    wan_recurrent_components,
)


def _amp_ctx(use_amp: bool, amp_dtype: torch.dtype, device_type: str):
    if use_amp and device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def replay_encoder_vjp(
    encoder,
    v_clip_detached: torch.Tensor,             # [B, T, 3, H, W] in [0,1], detached
    grad_feat_proj: torch.Tensor,              # [B, S, D_proj], dL/d(projected feature)
    projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """One-clip encoder VJP: returns ``g_clip`` = (dF_proj/d v_clip)^T grad_feat_proj."""
    # Frozen encoder parameters still allow derivatives with respect to pixels.
    # Recreate only this clip's graph; grad_feat_proj is the cached cotangent.
    v_leaf = v_clip_detached.detach().clone().requires_grad_(True)
    with _amp_ctx(use_amp, amp_dtype, v_leaf.device.type):
        feat = encoder.forward_sparse(v_leaf)              # [B, S, D_raw]
        feat_proj = projector(feat) if projector is not None else feat
    feat_proj = feat_proj.float()
    g = grad_feat_proj.detach().to(feat_proj.dtype)
    if g.shape != feat_proj.shape:
        raise RuntimeError(
            f"encoder VJP shape mismatch: feat_proj {tuple(feat_proj.shape)} vs "
            f"grad {tuple(g.shape)}"
        )
    # The inner product injects dL/d feature without recomputing the MMD loss:
    # its pixel derivative is J_encoder^T (dL/d feature), the required VJP.
    replay_loss = (feat_proj * g).sum()
    (g_clip,) = torch.autograd.grad(replay_loss, v_leaf, retain_graph=False, create_graph=False)
    return g_clip


def replay_video_encoder_vjp(
    video81_detached: torch.Tensor,            # [B, 81, 3, H, W] in [0,1], detached
    specs_and_grads: List[Tuple[object, object, torch.Tensor]],
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Accumulate per-clip encoder VJPs into a full pixel-frame gradient buffer.

    ``specs_and_grads`` is a list of ``(encoder, spec_or_projector_tuple,
    grad_feat_proj)`` where the middle element is ``(spec, projector)``. Each
    clip's gradient is scattered into the 81-frame buffer at ``spec.frame_indices``
    (accumulating overlaps).
    """
    # Multiple clips/encoders can touch the same frame. Sum their derivatives
    # in video coordinates before replaying the VAE decoder.
    g_video = new_frame_grad_buffer(video81_detached)
    for encoder, (spec, projector), grad_feat_proj in specs_and_grads:
        idx = spec.frame_indices
        v_clip = gather_frames(video81_detached, idx, time_dim=1)         # [B, 16, 3, H, W]
        g_clip = replay_encoder_vjp(
            encoder, v_clip, grad_feat_proj, projector=projector,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        scatter_add_frames(g_video, idx, g_clip, time_dim=1)
    return g_video


@torch.enable_grad()
def replay_decoder_vjp(
    vae,
    z_detached: torch.Tensor,                  # [B, T_lat, C, H, W] (repo latent layout)
    g_video01: torch.Tensor,                   # [B, T_pix, 3, H, W] dL/d(pixels in [0,1])
    independent_first_frame: bool = True,
) -> torch.Tensor:
    """Per-latent-frame decode VJP: ``grad_z = (d pixels01 / d z)^T g_video01``.

    Re-decodes one latent frame at a time with grad, carrying the VAE temporal
    cache by value but detaching it between frames so peak memory is ~one frame's
    decode graph. Frames with zero gradient still advance the cache (needed for
    exact values of later frames) but contribute no latent gradient.

    Temporal cache detachment truncates cross-frame derivatives. This historical
    path preserves decode values, but is not the full decoder Jacobian.
    """
    vae_model = vae.model
    zdim = vae_model.z_dim
    device = z_detached.device
    dtype = z_detached.dtype
    mean = vae.mean.to(device=device, dtype=dtype).view(1, zdim, 1, 1, 1)
    inv_std = (1.0 / vae.std.to(device=device, dtype=dtype)).view(1, zdim, 1, 1, 1)

    z_cf = z_detached.detach().permute(0, 2, 1, 3, 4).contiguous()   # [B, zdim, T_lat, H, W]
    T = z_cf.shape[2]
    grad_z_cf = torch.zeros_like(z_cf, dtype=torch.float32)

    # Recompute one decode segment, inject its pixel cotangent, and release the
    # segment graph. The returned latent cotangent drives generator replay next.
    vae_model.clear_cache()
    px_cursor = 0
    for i in range(T):
        vae_model._conv_idx = [0]
        z_i = z_cf[:, :, i:i + 1].clone().requires_grad_(True)
        z_i_unscaled = z_i / inv_std + mean
        x_i = vae_model.conv2(z_i_unscaled)
        out_i = vae_model.decoder(x_i, feat_cache=vae_model._feat_map, feat_idx=vae_model._conv_idx)
        out_i = out_i.float().clamp(-1.0, 1.0)
        pix_i = out_i.shape[2]
        pixels01_i = ((out_i + 1.0) * 0.5).permute(0, 2, 1, 3, 4)     # [B, pix_i, 3, H, W]
        end = min(px_cursor + pix_i, g_video01.shape[1])
        # Compute the surrogate in float32 so a large pixel-frame gradient does
        # not overflow bf16 (the encoder VJP can be large).
        g_i = g_video01[:, px_cursor:end].float()
        if g_i.shape[1] == pixels01_i.shape[1] and torch.isfinite(g_i).all() and g_i.abs().sum() > 0:
            surrogate_i = (pixels01_i.float() * g_i.detach()).sum()
            (grad_zi,) = torch.autograd.grad(surrogate_i, z_i, retain_graph=False)
            grad_z_cf[:, :, i:i + 1] += grad_zi.float()
            # The gradient has already been accumulated.  Keeping its graph
            # root alive until the next decoder call would retain the old
            # frame leaf and output storage without affecting any later math.
            del surrogate_i, grad_zi
        px_cursor += pix_i
        vae_model._feat_map = [c.detach() if torch.is_tensor(c) else c for c in vae_model._feat_map]
        # The detached temporal cache above is the only state needed by the
        # next frame.  Release the consumed frame graph and views before that
        # frame's decoder reaches its allocation peak.
        del pixels01_i, out_i, x_i, z_i_unscaled, z_i, g_i
    vae_model.clear_cache()
    return grad_z_cf.permute(0, 2, 1, 3, 4)                            # [B, T_lat, C, H, W]


@torch.enable_grad()
def _replay_generic_decoder_vjp_exact(
    vae,
    z_detached: torch.Tensor,
    g_video01: torch.Tensor,
    *,
    expected_pixels01: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact full-video decoder VJP with activation replay.

    Wan's causal VAE carries temporal feature caches between all 21 latent
    frames.  Detaching those caches changes the MMD gradient.  This path
    reentrantly checkpoints the complete frozen decode instead: the first
    forward runs without an autograd graph, then backward recomputes the exact
    graph while saved activations are moved to pageable CPU memory.  Every
    cross-frame dependency remains in the autograd graph.
    """

    if z_detached.ndim != 5 or g_video01.ndim != 5:
        raise ValueError("decoder VJP expects [B,T,C,H,W] latent and pixel tensors")
    if z_detached.device.type != g_video01.device.type:
        raise ValueError("decoder latent and pixel cotangent must share a device")
    if not torch.isfinite(g_video01).all():
        raise ValueError("decoder pixel cotangent contains non-finite values")
    z_leaf = z_detached.detach().clone().requires_grad_(True)

    def decode_pixels01(latent: torch.Tensor) -> torch.Tensor:
        # With reentrant checkpointing the first call is under no_grad; only
        # the backward recompute saves tensors.  Keeping this hook *inside*
        # the function is required so those saved activations move to CPU.
        with torch.autograd.graph.save_on_cpu(
            pin_memory=False, device_type=latent.device.type
        ):
            with _amp_ctx(True, torch.bfloat16, latent.device.type):
                decoded = vae.decode_to_pixel(latent, use_cache=False)
                pixels01 = ((decoded + 1.0) * 0.5).clamp(0.0, 1.0)
            return pixels01

    pixels01 = checkpoint(
        decode_pixels01,
        z_leaf,
        use_reentrant=True,
        preserve_rng_state=False,
    )
    if tuple(pixels01.shape) != tuple(g_video01.shape):
        raise ValueError(
            f"decoder cotangent shape {tuple(g_video01.shape)} differs from "
            f"decoded pixels {tuple(pixels01.shape)}"
        )
    surrogate = (pixels01.float() * g_video01.detach().float()).sum()
    if expected_pixels01 is not None and not torch.equal(
        pixels01.detach(), expected_pixels01
    ):
        raise RuntimeError("generic decoder replay differs from expected pixels")
    surrogate.backward()
    gradient = z_leaf.grad
    if gradient is None:
        raise RuntimeError("exact decoder VJP did not produce a latent gradient")
    if not torch.isfinite(gradient).all():
        raise RuntimeError("exact decoder VJP produced non-finite latent gradients")
    return gradient.detach()


@torch.enable_grad()
def replay_decoder_vjp_exact(
    vae,
    z_detached: torch.Tensor,
    g_video01: torch.Tensor,
    *,
    expected_pixels01: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact decoder VJP with bounded-memory Wan recurrent BPTT.

    The official Wan decoder uses an explicit 32-cache recurrent state, K=4
    pageable-CPU boundaries, and future-state cotangent propagation.  This
    retains the complete temporal Jacobian without materializing or offloading
    the whole 21-frame graph.  Stateless toy/non-Wan decoders retain the
    generic reentrant checkpoint fallback for test and API compatibility.
    """

    if z_detached.ndim != 5 or g_video01.ndim != 5:
        raise ValueError("decoder VJP expects [B,T,C,H,W] latent and pixel tensors")
    if z_detached.device != g_video01.device:
        raise ValueError("decoder latent and pixel cotangent must share a device")
    if not torch.isfinite(g_video01).all():
        raise ValueError("decoder pixel cotangent contains non-finite values")
    # This optional Wan path carries future-state cotangents across temporal
    # boundaries; the generic fallback preserves the full decode graph as well.
    if wan_recurrent_components(vae) is not None:
        return replay_wan_decoder_vjp_exact(
            vae,
            z_detached,
            g_video01,
            expected_pixels01=expected_pixels01,
        )
    return _replay_generic_decoder_vjp_exact(
        vae,
        z_detached,
        g_video01,
        expected_pixels01=expected_pixels01,
    )

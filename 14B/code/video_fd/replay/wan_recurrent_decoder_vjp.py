"""Bounded-memory exact recurrent VJP for the official Wan VAE decoder.

Wan's decoder mutates a Python feature-cache list while consuming one latent
frame at a time.  This module makes that list an explicit recurrent state,
stores only K=4 boundaries on CPU, and propagates the future-state cotangent
back through every transition.  The resulting VJP is the full temporal
Jacobian, not the historical detached-cache approximation.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


CHUNK_LATENT_FRAMES = 4
REP_SENTINEL = "Rep"


def _amp(device_type: str):
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def wan_recurrent_components(vae: Any) -> tuple[Any, Any] | None:
    """Duck-type the official Wan wrapper without creating an import cycle."""

    model = getattr(vae, "model", None)
    decoder = getattr(model, "decoder", None)
    if (
        model is None
        or decoder is None
        or not hasattr(model, "conv2")
        or not hasattr(model, "z_dim")
        or not hasattr(decoder, "num_res_blocks")
        or not hasattr(decoder, "dim_mult")
        or not hasattr(decoder, "temperal_upsample")
        or not hasattr(vae, "mean")
        or not hasattr(vae, "std")
    ):
        return None
    return model, decoder


def _active_cache_slots(decoder: Any) -> int:
    stages = len(tuple(decoder.dim_mult))
    residual_convs = 4 + stages * (int(decoder.num_res_blocks) + 1) * 2
    temporal_resamplers = sum(
        bool(value) for value in tuple(decoder.temperal_upsample)[: stages - 1]
    )
    # decoder.conv1 + residual convs + temporal resamplers + decoder.head
    return 1 + residual_convs + temporal_resamplers + 1


def _is_formal_480p(model: Any, decoder: Any, latent: torch.Tensor) -> bool:
    return (
        int(model.z_dim) == 16
        and int(decoder.dim) == 96
        and int(decoder.num_res_blocks) == 2
        and tuple(decoder.dim_mult) == (1, 2, 4, 4)
        and tuple(bool(v) for v in decoder.temperal_upsample)
        == (True, True, False)
        and tuple(latent.shape) == (1, 21, 16, 60, 104)
    )


def _model_cache_signature(model: Any) -> tuple[Any, ...]:
    """Identity-only proof that the functional path never touched globals."""

    values = getattr(model, "_feat_map", None)
    index = getattr(model, "_conv_idx", None)
    value_signature = None
    if isinstance(values, list):
        value_signature = tuple(
            ("tensor", id(value)) if torch.is_tensor(value) else ("value", value)
            for value in values
        )
    index_signature = tuple(index) if isinstance(index, list) else index
    return id(values), value_signature, id(index), index_signature


def _formal_shapes(
    *, batch: int, height: int, width: int, frames_consumed: int
) -> dict[int, tuple[int, ...] | str]:
    temporal = 1 if frames_consumed == 1 else 2
    result: dict[int, tuple[int, ...] | str] = {
        0: (batch, 16, temporal, height, width),
        12: (batch, 192, temporal, height * 2, width * 2),
    }
    result.update(
        {
            index: (batch, 384, temporal, height, width)
            for index in range(1, 11)
        }
    )
    result[11] = (
        REP_SENTINEL
        if frames_consumed == 1
        else (batch, 384, 2, height, width)
    )
    result.update(
        {
            index: (batch, 384, temporal, height * 2, width * 2)
            for index in range(13, 18)
        }
    )
    result[18] = (
        REP_SENTINEL
        if frames_consumed == 1
        else (batch, 384, 2, height * 2, width * 2)
    )
    result.update(
        {
            index: (batch, 192, temporal, height * 4, width * 4)
            for index in range(19, 25)
        }
    )
    result.update(
        {
            index: (batch, 96, temporal, height * 8, width * 8)
            for index in range(25, 32)
        }
    )
    return result


def _formal_cache_dtype(index: int) -> torch.dtype:
    if index not in range(32):
        raise ValueError(f"formal Wan cache index out of range: {index}")
    return torch.bfloat16 if index in (0, 11, 18) else torch.float32


def _validate_state(
    state: tuple[Any, ...],
    *,
    slots: int,
    frames_consumed: int,
    batch: int,
    height: int,
    width: int,
    formal: bool,
    device_type: str | None = None,
) -> None:
    if len(state) != slots:
        raise RuntimeError(f"Wan state has {len(state)} slots, expected {slots}")
    for index, value in enumerate(state):
        if torch.is_tensor(value):
            if value.ndim != 5 or value.shape[0] != batch:
                raise RuntimeError(
                    f"Wan cache slot {index} has invalid shape {tuple(value.shape)}"
                )
            if not value.is_floating_point():
                raise TypeError(f"Wan cache slot {index} is not floating point")
            if device_type is not None and value.device.type != device_type:
                raise RuntimeError(
                    f"Wan cache slot {index} is on {value.device.type}, "
                    f"expected {device_type}"
                )
        elif value is None:
            if frames_consumed != 0:
                raise RuntimeError(f"Wan cache slot {index} unexpectedly remained empty")
        elif value == REP_SENTINEL:
            if frames_consumed != 1 or (formal and index not in (11, 18)):
                raise RuntimeError(
                    f"unexpected Rep sentinel in slot {index} after "
                    f"{frames_consumed} frames"
                )
        else:
            raise TypeError(
                f"unsupported Wan cache value {type(value).__name__} in slot {index}"
            )

    if not formal:
        return
    if slots != 32:
        raise RuntimeError(f"formal Wan decoder needs 32 cache slots, got {slots}")
    if frames_consumed == 0:
        if any(value is not None for value in state):
            raise RuntimeError("initial formal Wan state is not empty")
        return
    expected = _formal_shapes(
        batch=batch,
        height=height,
        width=width,
        frames_consumed=frames_consumed,
    )
    if set(expected) != set(range(32)):
        raise AssertionError("formal cache layout does not cover all 32 slots")
    for index, specification in expected.items():
        value = state[index]
        if specification == REP_SENTINEL:
            if value != REP_SENTINEL:
                raise RuntimeError(f"formal Wan slot {index} must be Rep")
            continue
        if not torch.is_tensor(value) or tuple(value.shape) != specification:
            actual = tuple(value.shape) if torch.is_tensor(value) else value
            raise RuntimeError(
                f"formal Wan slot {index} has {actual}, expected {specification}"
            )
        expected_dtype = _formal_cache_dtype(index)
        if value.dtype != expected_dtype:
            raise TypeError(
                f"formal Wan slot {index} must be {expected_dtype}, "
                f"got {value.dtype}"
            )


def _state_to_cpu(state: tuple[Any, ...], *, formal: bool) -> tuple[Any, ...]:
    result: list[Any] = []
    for index, value in enumerate(state):
        if torch.is_tensor(value):
            if formal and value.dtype != _formal_cache_dtype(index):
                raise TypeError(
                    f"formal boundary slot {index} must be "
                    f"{_formal_cache_dtype(index)}, got {value.dtype}"
                )
            result.append(
                value.detach().to(device="cpu", non_blocking=False).contiguous()
            )
        else:
            result.append(value)
    return tuple(result)


def _state_to_device(
    state: tuple[Any, ...], *, device: torch.device
) -> tuple[Any, ...]:
    return tuple(
        value.to(device=device, non_blocking=False).detach()
        if torch.is_tensor(value)
        else value
        for value in state
    )


def _state_as_leaves(state: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        value.detach().requires_grad_(True) if torch.is_tensor(value) else value
        for value in state
    )


def _transition(
    *,
    model: Any,
    preconv_frame: torch.Tensor,
    state: tuple[Any, ...],
    slots: int,
) -> tuple[torch.Tensor, tuple[Any, ...]]:
    if preconv_frame.ndim != 5 or preconv_frame.shape[2] != 1:
        raise ValueError("Wan transition requires one pre-convolved frame")
    local_cache = list(state)
    local_index = [0]
    with _amp(preconv_frame.device.type):
        decoded_cf = model.decoder(
            preconv_frame,
            feat_cache=local_cache,
            feat_idx=local_index,
        )
        # Preserve WanVAEWrapper exactly: FP32 [-1,1] clamp, repository
        # time-first layout, then the [0,1] affine and clamp.
        decoded = decoded_cf.float().clamp(-1.0, 1.0).permute(0, 2, 1, 3, 4)
        pixels01 = ((decoded + 1.0) * 0.5).clamp(0.0, 1.0)
    if local_index[0] != slots:
        raise RuntimeError(
            f"Wan transition consumed {local_index[0]} cache slots, expected {slots}"
        )
    if len(local_cache) != slots:
        raise RuntimeError("Wan transition changed local cache-list length")
    return pixels01, tuple(local_cache)


def _full_preconv(vae: Any, latent: torch.Tensor) -> torch.Tensor:
    model = vae.model
    channels = int(model.z_dim)
    if latent.shape[2] != channels:
        raise ValueError(
            f"Wan latent has {latent.shape[2]} channels, expected {channels}"
        )
    mean = vae.mean.to(device=latent.device, dtype=latent.dtype).view(
        1, channels, 1, 1, 1
    )
    inverse_std = (
        1.0 / vae.std.to(device=latent.device, dtype=latent.dtype)
    ).view(1, channels, 1, 1, 1)
    unscaled = latent.permute(0, 2, 1, 3, 4) / inverse_std + mean
    return model.conv2(unscaled)


def _formal_pixel_span(frame_index: int) -> tuple[int, int]:
    if frame_index == 0:
        return 0, 1
    return 1 + 4 * (frame_index - 1), 1 + 4 * frame_index


def replay_wan_decoder_vjp_exact(
    vae: Any,
    z_detached: torch.Tensor,
    g_video01: torch.Tensor,
    *,
    expected_pixels01: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run exact K=4 recurrent BPTT through an official Wan VAE decoder."""

    components = wan_recurrent_components(vae)
    if components is None:
        raise TypeError("recurrent Wan VJP received a non-Wan decoder")
    model, decoder = components
    if z_detached.shape[0] != 1:
        raise ValueError("Wan recurrent decoder VJP requires batch size 1")
    if model.training or decoder.training:
        raise RuntimeError("Wan recurrent decoder VJP requires eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Wan recurrent decoder VJP requires frozen parameters")
    slots = _active_cache_slots(decoder)
    formal = _is_formal_480p(model, decoder, z_detached)
    if int(model.z_dim) == 16 and int(decoder.dim) == 96 and slots != 32:
        raise RuntimeError(f"production Wan exposed {slots} slots, expected 32")
    if formal and (
        z_detached.dtype != torch.bfloat16 or g_video01.dtype != torch.float32
    ):
        raise TypeError("formal 480P VJP needs BF16 latent and FP32 cotangent")
    if expected_pixels01 is not None:
        if expected_pixels01.device != z_detached.device:
            raise ValueError("expected pixels must remain on the latent device")
        if tuple(expected_pixels01.shape) != tuple(g_video01.shape):
            raise ValueError("expected pixels differ from cotangent shape")
        if not torch.isfinite(expected_pixels01).all():
            raise ValueError("expected pixels contain non-finite values")

    global_signature = _model_cache_signature(model)
    batch, latent_frames, _, height, width = z_detached.shape
    initial_state = tuple(None for _ in range(slots))
    _validate_state(
        initial_state,
        slots=slots,
        frames_consumed=0,
        batch=batch,
        height=height,
        width=width,
        formal=formal,
    )

    # Official decode scales and calls conv2 once at full temporal shape.
    # Preserve that operation, collect dL/dx_i recurrently, then run one exact
    # full-shape conv2+scale VJP at the end.
    with torch.no_grad(), _amp(z_detached.device.type):
        preconv = _full_preconv(vae, z_detached).detach()
    if tuple(preconv.shape[:3]) != (
        batch,
        int(model.z_dim),
        latent_frames,
    ):
        raise RuntimeError(f"unexpected conv2 output shape {tuple(preconv.shape)}")

    boundaries: dict[int, tuple[Any, ...]] = {0: initial_state}
    state = initial_state
    pixel_lengths: list[int] = []
    cursor = 0
    final_state: tuple[Any, ...] | None = None
    with torch.no_grad():
        for frame_index in range(latent_frames):
            if frame_index and frame_index % CHUNK_LATENT_FRAMES == 0:
                boundary = _state_to_cpu(state, formal=formal)
                _validate_state(
                    boundary,
                    slots=slots,
                    frames_consumed=frame_index,
                    batch=batch,
                    height=height,
                    width=width,
                    formal=formal,
                    device_type="cpu",
                )
                boundaries[frame_index] = boundary
            pixels, state = _transition(
                model=model,
                preconv_frame=preconv[:, :, frame_index : frame_index + 1],
                state=state,
                slots=slots,
            )
            _validate_state(
                state,
                slots=slots,
                frames_consumed=frame_index + 1,
                batch=batch,
                height=height,
                width=width,
                formal=formal,
                device_type=z_detached.device.type,
            )
            length = int(pixels.shape[1])
            if tuple(pixels.shape[2:]) != tuple(g_video01.shape[2:]):
                raise RuntimeError("Wan transition pixel shape differs from cotangent")
            if formal:
                start, end = _formal_pixel_span(frame_index)
                if cursor != start or length != end - start:
                    raise RuntimeError(
                        f"frame {frame_index} emitted {length} at {cursor}, "
                        f"expected [{start},{end})"
                    )
            if expected_pixels01 is not None:
                expected = expected_pixels01[:, cursor : cursor + length]
                if not torch.equal(pixels, expected):
                    difference = (pixels.float() - expected.float()).abs()
                    raise RuntimeError(
                        "functional Wan decode differs from official values; "
                        f"frame={frame_index}, unequal="
                        f"{int(torch.count_nonzero(pixels != expected).item())}, "
                        f"max_abs={float(difference.max().item())}"
                    )
            pixel_lengths.append(length)
            cursor += length
            final_state = state
            del pixels

    if cursor != g_video01.shape[1]:
        raise ValueError(
            f"Wan emitted {cursor} pixel frames, cotangent has {g_video01.shape[1]}"
        )
    if formal:
        if pixel_lengths != [1] + [4] * 20 or cursor != 81:
            raise RuntimeError(
                f"formal output schedule changed: {pixel_lengths}, total={cursor}"
            )
        if sorted(boundaries) != [0, 4, 8, 12, 16, 20]:
            raise RuntimeError(f"formal CPU boundaries changed: {sorted(boundaries)}")
    if final_state is None:
        raise RuntimeError("Wan decoder emitted no transitions")
    if formal and any(not torch.is_tensor(value) for value in final_state):
        raise RuntimeError("all 32 final formal cache states must be tensors")

    # There is no terminal-state loss. Each reverse transition creates the
    # exact future-state cotangent consumed by the preceding transition.
    future_cotangent: tuple[Any, ...] = (None,) * slots
    del final_state, state
    grad_preconv = torch.empty_like(preconv)
    offsets = [0]
    for length in pixel_lengths:
        offsets.append(offsets[-1] + length)
    chunk_starts = list(range(0, latent_frames, CHUNK_LATENT_FRAMES))
    if formal and list(reversed(chunk_starts)) != [20, 16, 12, 8, 4, 0]:
        raise RuntimeError(f"formal reverse chunks changed: {chunk_starts}")

    for chunk_start in reversed(chunk_starts):
        chunk_end = min(chunk_start + CHUNK_LATENT_FRAMES, latent_frames)
        if chunk_start not in boundaries:
            raise RuntimeError(f"missing CPU boundary at frame {chunk_start}")
        replay_state = _state_to_device(
            boundaries[chunk_start], device=z_detached.device
        )
        states_before: list[tuple[Any, ...]] = []
        with torch.no_grad():
            for frame_index in range(chunk_start, chunk_end):
                states_before.append(replay_state)
                replay_pixels, replay_state = _transition(
                    model=model,
                    preconv_frame=preconv[:, :, frame_index : frame_index + 1],
                    state=replay_state,
                    slots=slots,
                )
                if replay_pixels.shape[1] != pixel_lengths[frame_index]:
                    raise RuntimeError("chunk replay changed its pixel length")
                del replay_pixels
        del replay_state

        for frame_index in reversed(range(chunk_start, chunk_end)):
            input_state = _state_as_leaves(states_before.pop())
            preconv_leaf = (
                preconv[:, :, frame_index : frame_index + 1]
                .detach()
                .requires_grad_(True)
            )
            pixels, output_state = _transition(
                model=model,
                preconv_frame=preconv_leaf,
                state=input_state,
                slots=slots,
            )
            start, end = offsets[frame_index], offsets[frame_index + 1]
            pixel_cotangent = g_video01[:, start:end].detach().float()
            if tuple(pixel_cotangent.shape) != tuple(pixels.shape):
                raise RuntimeError("reverse pixel cotangent shape mismatch")
            if expected_pixels01 is not None:
                expected = expected_pixels01[:, start:end]
                if not torch.equal(pixels.detach(), expected):
                    difference = (pixels.detach().float() - expected.float()).abs()
                    raise RuntimeError(
                        "gradient Wan replay differs from forward values; "
                        f"frame={frame_index}, unequal="
                        f"{int(torch.count_nonzero(pixels.detach() != expected).item())}, "
                        f"max_abs={float(difference.max().item())}"
                    )
            outputs = [pixels]
            output_cotangents = [pixel_cotangent]
            for index, (value, cotangent) in enumerate(
                zip(output_state, future_cotangent)
            ):
                if cotangent is None:
                    continue
                if not torch.is_tensor(value):
                    raise RuntimeError(
                        f"output state {index} has a cotangent but is not a tensor"
                    )
                if tuple(value.shape) != tuple(cotangent.shape):
                    raise RuntimeError(f"state cotangent mismatch in slot {index}")
                if value.dtype != cotangent.dtype:
                    raise RuntimeError(f"state cotangent dtype mismatch in slot {index}")
                outputs.append(value)
                output_cotangents.append(cotangent.detach())

            positions = [
                index for index, value in enumerate(input_state) if torch.is_tensor(value)
            ]
            state_leaves = [input_state[index] for index in positions]
            gradients = torch.autograd.grad(
                outputs,
                [preconv_leaf, *state_leaves],
                grad_outputs=output_cotangents,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            grad_x = gradients[0]
            if grad_x is None or not torch.isfinite(grad_x).all():
                raise RuntimeError(f"transition {frame_index} has invalid conv2 gradient")
            grad_preconv[:, :, frame_index : frame_index + 1].copy_(grad_x.detach())
            incoming: list[Any] = [None] * slots
            for index, gradient in zip(positions, gradients[1:]):
                if gradient is None:
                    raise RuntimeError(
                        f"cache slot {index} lost its recurrent gradient"
                    )
                if not torch.isfinite(gradient).all():
                    raise RuntimeError(f"cache slot {index} has non-finite cotangent")
                incoming[index] = gradient.detach()
            future_cotangent = tuple(incoming)
            value = cotangent = gradient = None
            del (
                gradients,
                grad_x,
                outputs,
                output_cotangents,
                pixels,
                output_state,
                preconv_leaf,
                input_state,
                positions,
                state_leaves,
                pixel_cotangent,
                incoming,
            )
        if states_before:
            raise AssertionError("reverse chunk left unreplayed states")

    # Finish the exact chain rule through one full-shape conv2+scale replay.
    latent_leaf = z_detached.detach().clone().requires_grad_(True)
    with _amp(z_detached.device.type):
        replay_preconv = _full_preconv(vae, latent_leaf)
    if not torch.equal(replay_preconv.detach(), preconv):
        difference = (replay_preconv.detach().float() - preconv.float()).abs()
        raise RuntimeError(
            f"full-shape conv2 replay changed values; max_abs={float(difference.max())}"
        )
    (gradient,) = torch.autograd.grad(
        replay_preconv,
        latent_leaf,
        grad_outputs=grad_preconv.detach(),
        retain_graph=False,
        create_graph=False,
    )
    if not torch.isfinite(gradient).all():
        raise RuntimeError("exact recurrent Wan VJP produced non-finite gradients")
    if _model_cache_signature(model) != global_signature:
        raise RuntimeError("functional Wan VJP mutated model._feat_map/_conv_idx")
    return gradient.detach()

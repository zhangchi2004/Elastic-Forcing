"""Independent normalized Nyström MMD objectives over multiple video layers.

The V-JEPA2 caller concatenates the sampled dense matrices from every selected
layer along the token axis.  This module splits that transport tensor back into
its layer-local matrices, evaluates one frozen-reference MMD per layer, applies
detached scalar self-normalization independently, and sums the results.

Keeping the objectives separate is important: concatenating layer features
inside one kernel would change both the bandwidth and the teacher reference
distribution instead of giving each layer equal normalized influence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from losses.video_nystrom_mmd import VideoNystromMMD


class MultiLayerVideoNystromMMD(nn.Module):
    """Sum independent, optionally self-normalized per-layer video MMDs."""

    def __init__(
        self,
        objectives: Mapping[int, VideoNystromMMD],
        *,
        self_normalize: bool = True,
        norm_eps: float = 1.0e-4,
        layer_weights: Mapping[int, float] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(objectives, Mapping) or not objectives:
            raise ValueError("objectives must be a non-empty layer mapping")
        ordered_layers = tuple(int(layer) for layer in objectives)
        if len(set(ordered_layers)) != len(ordered_layers):
            raise ValueError("objective layer indices must be unique")
        if any(layer < 0 for layer in ordered_layers):
            raise ValueError("objective layer indices must be non-negative")
        if tuple(sorted(ordered_layers)) != ordered_layers:
            raise ValueError("objective layer indices must be sorted")
        if (
            not math.isfinite(float(norm_eps))
            or float(norm_eps) <= 0.0
        ):
            raise ValueError("norm_eps must be finite and positive")

        weights = (
            {layer: 1.0 for layer in ordered_layers}
            if layer_weights is None
            else {int(layer): float(value)
                  for layer, value in layer_weights.items()}
        )
        if set(weights) != set(ordered_layers):
            raise ValueError(
                "layer_weights must contain exactly the objective layers")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in weights.values()
        ):
            raise ValueError(
                "layer weights must be finite and non-negative")

        modules = {}
        sample_shapes = []
        for layer in ordered_layers:
            objective = objectives[layer]
            if not isinstance(objective, VideoNystromMMD):
                raise TypeError(
                    "every layer objective must be VideoNystromMMD")
            if any(
                parameter.requires_grad
                for parameter in objective.parameters()
            ):
                raise ValueError(
                    "Nyström reference objectives must remain frozen")
            modules[str(layer)] = objective
            sample_shapes.append(objective.sample_shape)
        feature_dims = {shape[1] for shape in sample_shapes}
        if len(feature_dims) != 1:
            raise ValueError(
                "all layer objectives must share one feature dimension")

        self.objectives = nn.ModuleDict(modules)
        self.layer_indices = ordered_layers
        self.layer_token_counts = tuple(
            int(shape[0]) for shape in sample_shapes)
        self.feature_dim = int(sample_shapes[0][1])
        self.self_normalize = bool(self_normalize)
        self.norm_eps = float(norm_eps)
        self.layer_weights = tuple(
            float(weights[layer]) for layer in ordered_layers)
        ema_settings = {
            (
                objective.attraction_ema_enabled,
                objective.attraction_ema_beta,
            )
            for objective in self.objectives.values()
        }
        if len(ema_settings) != 1:
            raise ValueError(
                "all layer objectives must use the same attraction EMA "
                "enabled/beta setting")
        (
            self.attraction_ema_enabled,
            self.attraction_ema_beta,
        ) = next(iter(ema_settings))

    @property
    def sample_shape(self) -> tuple[int, int]:
        """Shape of the concatenated GradCache transport matrix."""
        return sum(self.layer_token_counts), self.feature_dim

    @property
    def landmarks(self) -> torch.Tensor:
        """Compatibility device/landmark-count view used by GradCache logs."""
        return self.objectives[str(self.layer_indices[0])].landmarks

    @property
    def reference_hashes(self) -> tuple[str, ...]:
        return tuple(
            str(self.objectives[str(layer)].reference_hash)
            for layer in self.layer_indices
        )

    @property
    def reference_metadata(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            self.objectives[str(layer)].reference_metadata
            for layer in self.layer_indices
        )

    @property
    def attraction_ema_pending(self) -> bool:
        pending = tuple(
            objective.attraction_ema_pending
            for objective in self.objectives.values()
        )
        if pending and any(value != pending[0] for value in pending[1:]):
            raise RuntimeError(
                "per-layer attraction EMA pending state is inconsistent")
        return bool(pending[0]) if pending else False

    def split_features(
        self,
        generated: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if not isinstance(generated, torch.Tensor):
            raise TypeError("generated must be a torch.Tensor")
        if (
            generated.ndim != 3
            or min(generated.shape) <= 0
            or not generated.is_floating_point()
        ):
            raise ValueError(
                "generated must be a non-empty floating [B,N,D] tensor")
        if not bool(torch.isfinite(generated.detach()).all().item()):
            raise FloatingPointError(
                "generated contains non-finite values")
        expected_tokens, expected_dim = self.sample_shape
        actual_shape = (
            int(generated.shape[1]), int(generated.shape[2]))
        if actual_shape != (expected_tokens, expected_dim):
            raise ValueError(
                "concatenated multi-layer video matrix shape does not match "
                f"the frozen references: {actual_shape} != "
                f"{(expected_tokens, expected_dim)}")
        return tuple(torch.split(
            generated, self.layer_token_counts, dim=1))

    def forward(
        self,
        generated: torch.Tensor,
        *,
        stage: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        layer_features = self.split_features(generated)
        total = generated.new_zeros((), dtype=torch.float32)
        raw_sum = total.clone()
        repulsion_sum = total.clone()
        attraction_sum = total.clone()
        current_attraction_sum = total.clone()
        reference_constant_sum = total.clone()
        layer_metadata: dict[str, dict[str, torch.Tensor]] = {}
        global_samples = None
        ema_updates = None
        attraction_current_coefficient = None

        for layer, weight, features in zip(
            self.layer_indices,
            self.layer_weights,
            layer_features,
        ):
            objective = self.objectives[str(layer)]
            raw_mmd, metadata = (
                objective(features, stage=stage)
                if self.attraction_ema_enabled
                else objective(features)
            )
            if self.self_normalize:
                # Nyström attraction is approximate, so its finite estimate
                # can be slightly negative even though exact squared MMD is
                # non-negative. Normalize by detached magnitude to avoid both
                # a singularity near -eps and an accidental gradient reversal.
                denominator = raw_mmd.detach().abs() + self.norm_eps
                if (
                    not bool(torch.isfinite(denominator).item())
                    or float(denominator) == 0.0
                ):
                    raise FloatingPointError(
                        f"layer {layer} has an invalid MMD normalization "
                        "denominator")
                normalized_mmd = raw_mmd / denominator
            else:
                normalized_mmd = raw_mmd
            weighted_mmd = float(weight) * normalized_mmd
            total = total + weighted_mmd
            raw_sum = raw_sum + raw_mmd.detach()
            repulsion_sum = repulsion_sum + metadata["repulsion"]
            attraction_sum = attraction_sum + metadata["attraction"]
            if self.attraction_ema_enabled:
                current_attraction_sum = (
                    current_attraction_sum + metadata[
                        "current_attraction"])
            reference_constant_sum = (
                reference_constant_sum + metadata["reference_constant"])
            observed_global_samples = int(metadata["global_samples"])
            if global_samples is None:
                global_samples = observed_global_samples
            elif observed_global_samples != global_samples:
                raise RuntimeError(
                    "per-layer MMD objectives disagree on global batch size")
            if self.attraction_ema_enabled:
                observed_updates = int(
                    metadata["attraction_ema_updates"])
                if ema_updates is None:
                    ema_updates = observed_updates
                elif observed_updates != ema_updates:
                    raise RuntimeError(
                        "per-layer attraction EMA update counts disagree")
                observed_coefficient = float(
                    metadata["attraction_current_coefficient"])
                if attraction_current_coefficient is None:
                    attraction_current_coefficient = observed_coefficient
                elif observed_coefficient != attraction_current_coefficient:
                    raise RuntimeError(
                        "per-layer attraction EMA coefficients disagree")
            layer_metadata[str(layer)] = {
                **metadata,
                "raw_mmd": raw_mmd.detach(),
                "normalized_mmd": normalized_mmd.detach(),
                "weighted_mmd": weighted_mmd.detach(),
                "weight": raw_mmd.new_tensor(float(weight)),
            }

        if not bool(torch.isfinite(total.detach()).item()):
            raise FloatingPointError(
                "combined multi-layer MMD loss is non-finite")
        combined_metadata = {
            "layers": layer_metadata,
            "layer_indices": torch.tensor(
                self.layer_indices,
                device=total.device,
                dtype=torch.long,
            ),
            "num_layers": total.new_tensor(
                len(self.layer_indices), dtype=torch.long),
            "self_normalized": total.new_tensor(
                int(self.self_normalize), dtype=torch.long),
            "normalization_epsilon": total.new_tensor(self.norm_eps),
            "raw_mmd_sum": raw_sum.detach(),
            "normalized_mmd_sum": total.detach(),
            # Aggregated raw-kernel terms preserve the existing generic log
            # contract. Per-layer diagnostics above are authoritative.
            "repulsion": repulsion_sum.detach(),
            "attraction": attraction_sum.detach(),
            "reference_constant": reference_constant_sum.detach(),
            "global_samples": total.new_tensor(
                int(global_samples), dtype=torch.long),
        }
        if self.attraction_ema_enabled:
            combined_metadata.update({
                "current_attraction": current_attraction_sum.detach(),
                "attraction_ema_enabled": total.new_ones(
                    (), dtype=torch.long),
                "attraction_ema_updates": total.new_tensor(
                    int(ema_updates), dtype=torch.long),
                "attraction_current_coefficient": total.new_tensor(
                    float(attraction_current_coefficient)),
            })
        return total, combined_metadata

    def prepare_commit(self) -> None:
        """Validate every layer before publishing any EMA scalar."""
        if not self.attraction_ema_enabled:
            return
        for objective in self.objectives.values():
            objective.prepare_commit()

    def commit(self) -> None:
        """Publish one attraction EMA update for every layer."""
        if not self.attraction_ema_enabled:
            return
        if not self.attraction_ema_pending:
            return
        self.prepare_commit()
        for objective in self.objectives.values():
            objective.commit()

    def discard(self) -> None:
        for objective in self.objectives.values():
            objective.discard()

    def attraction_ema_state_dict(self) -> Mapping[str, Any] | None:
        if not self.attraction_ema_enabled:
            return None
        if self.attraction_ema_pending:
            raise RuntimeError(
                "refusing to checkpoint pending per-layer attraction EMA")
        return {
            "version": 1,
            "layers": list(self.layer_indices),
            "states": {
                str(layer): self.objectives[
                    str(layer)].attraction_ema_state_dict()
                for layer in self.layer_indices
            },
        }

    def validate_attraction_ema_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not self.attraction_ema_enabled:
            raise ValueError(
                "cannot load attraction EMA state while the path is disabled")
        if (
            not isinstance(state, Mapping)
            or set(state) != {"version", "layers", "states"}
            or state.get("version") != 1
            or tuple(state.get("layers", ())) != self.layer_indices
            or not isinstance(state.get("states"), Mapping)
            or set(state["states"])
            != {str(layer) for layer in self.layer_indices}
        ):
            raise ValueError(
                "multi-layer attraction EMA checkpoint schema mismatch")

        # Validate every layer before the first live scalar is mutated.
        validated = {}
        for layer in self.layer_indices:
            objective = self.objectives[str(layer)]
            validated[str(layer)] = (
                objective.validate_attraction_ema_state_dict(
                    state["states"][str(layer)])
            )
        update_counts = {
            int(layer_state["updates"].item())
            for layer_state in validated.values()
        }
        if len(update_counts) != 1:
            raise ValueError(
                "multi-layer attraction EMA update counts disagree")
        return {
            "version": 1,
            "layers": list(self.layer_indices),
            "states": validated,
        }

    def load_attraction_ema_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        validated = self.validate_attraction_ema_state_dict(state)
        for layer in self.layer_indices:
            self.objectives[str(layer)].load_attraction_ema_state_dict(
                validated["states"][str(layer)])

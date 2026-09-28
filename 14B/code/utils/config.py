"""Readable paper terminology at the config boundary, stable runtime keys inside.

The public YAML uses distribution/differentiation/reference batches. Translate
once before loading defaults, constructing models, or relocating assets. The
resolved config retains the released keys and values so checkpoint/reference
identities and existing scripts remain compatible. Legacy YAML is still valid.
"""
from collections.abc import Mapping
from copy import deepcopy

# New public spelling -> existing runtime/serialized spelling.
CONFIG_ALIASES = {
    'initial_generator_checkpoint': 'generator_ckpt',
    'initial_generator_checkpoint_sha256': 'generator_ckpt_sha256',
    'training_objective': 'distribution_loss',
    'training_loop': 'trainer',
    'feature_space': 'wan_feature_type',
    'mmd_estimator': 'wan_mmd_estimator',
    'objective_weight': 'wan_kv_loss_weight',
    'normalize_encoder_losses': 'wan_kv_self_normalize',
    'encoder_loss_normalization_epsilon': 'wan_kv_norm_eps',
    'reference_feature_source': 'current_teacher_source',
    'reference_feature_manifest': 'triple_teacher_bank_manifest',
    'reference_feature_manifest_sha256': 'triple_teacher_bank_manifest_sha256',
    'reference_feature_content_hash': 'triple_teacher_bank_content_hash',
    'reference_video_manifest': 'matched_manifest_path',
    'reference_video_count': 'matched_manifest_samples',
    'nystrom_reference_weight': 'triple_nystrom_attraction_weight',
    'monte_carlo_reference_weight': 'triple_nystrom_current_teacher_attraction_weight',
    'dino_loss_weight': 'triple_dinov3_weight',
    'vjepa_loss_weight': 'triple_vjepa2_weight',
    'videomae_loss_weight': 'triple_videomae_v2_weight',
    'encoder_loss_divisor': 'triple_aggregation_divisor',
    'active_feature_encoders': 'triple_active_branches',
    'per_device_microbatch_size': 'batch_size',
    'distribution_samples_per_rank': 'fresh_video_local_samples',
    'differentiation_samples_per_rank': 'fresh_video_generator_grad_local_samples',
    'differentiate_full_distribution_loss': 'fresh_video_full_student_partial_gradient',
    'differentiation_gradient_scale': 'fresh_video_generator_grad_sample_scale',
    'distribution_batch_size': 'total_batch_size',
    'reference_minibatch_size': 'expected_current_teacher_samples',
    'latent_video_shape': 'image_or_video_shape',
    'training_updates': 'max_steps',
    'generator_learning_rate': 'lr',
}

VALUE_ALIASES = {
    'training_objective': {'elastic_forcing': 'wan_kv_mmd'},
    'training_loop': {'elastic_forcing': 'score_distillation'},
    'feature_space': {'frozen_video_encoders': 'triple_dinov3_vjepa2_videomae_v2_dense_latent'},
    'mmd_estimator': {'hybrid_nystrom_monte_carlo': 'nystrom_exact_repulsion'},
    'reference_feature_source': {'matched_encoder_features': 'matched_triple_feature_bank'},
}


def resolve_config_aliases(config: Mapping) -> dict:
    """Return the released runtime schema without mutating the caller's data.

    An explicitly supplied readable key takes precedence over its legacy alias.
    This also permits overriding a legacy base config with readable settings.
    """
    result = {}
    overridden = {legacy for readable, legacy in CONFIG_ALIASES.items() if readable in config}
    for key, value in config.items():
        if key in CONFIG_ALIASES:
            legacy = CONFIG_ALIASES[key]
            replacements = VALUE_ALIASES.get(key, {})
            result[legacy] = replacements.get(value, value) if isinstance(value, str) else deepcopy(value)
        elif key not in overridden:
            result[key] = deepcopy(value)
    return result


def load_config(path, default_path=None):
    """Resolve public names before merging defaults; preserve OmegaConf behavior."""
    from omegaconf import OmegaConf

    def read(filename):
        config = OmegaConf.to_container(OmegaConf.load(filename), resolve=False)
        return OmegaConf.create(resolve_config_aliases(config))

    config = read(path)
    return OmegaConf.merge(read(default_path), config) if default_path else config

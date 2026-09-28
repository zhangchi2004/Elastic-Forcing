import torch

from model.wan_kv_mmd import WanKVMMD
from video_fd.encoders.vjepa2_dense_metadata import (
    canonical_vjepa2_dense_feature_metadata,
    uniform_dense_token_indices,
)


def test_triple_vjepa_metadata_matches_bank_last_64_contract():
    model = object.__new__(WanKVMMD)
    model.feature_type = "triple_dinov3_vjepa2_videomae_v2_dense_latent"
    model.dense_feature_prefix = "triple"
    model.dense_encoder_name = "triple"
    model.dense_log_prefix = "triple_dense"
    model.vjepa2_reference_model_id = "facebook/vjepa2-vitl-fpc64-256"
    model.vjepa2_model_id = "/unused/local/model"
    model.vjepa2_dense_layer = 15
    model.vjepa2_dense_layers = (15,)
    model.vjepa2_dense_num_frames = 64
    model.vjepa2_dense_source_num_frames = 81
    model.vjepa2_dense_frame_sampling = "last_contiguous"
    model.vjepa2_dense_resize_short_side = 292
    model.vjepa2_dense_crop_size = 256
    model.vjepa2_dense_raw_tokens = 2048
    model.vjepa2_dense_num_tokens = 1024
    model.vjepa2_dense_feature_dim = 1024
    indices = uniform_dense_token_indices(2048, 1024)
    actual = model._vjepa2_feature_metadata(indices)
    expected = canonical_vjepa2_dense_feature_metadata(
        model_id="facebook/vjepa2-vitl-fpc64-256",
        layer_index=15,
        num_frames=64,
        resize_short_side=292,
        crop_size=256,
        raw_tokens=2048,
        num_tokens=1024,
        feature_dim=1024,
        token_indices=indices,
        frame_sampling="last_contiguous",
        source_num_frames=81,
    )
    assert actual == expected
    assert actual["selected_frame_start"] == 17
    assert actual["selected_frame_end_inclusive"] == 80


def test_dual_dinov3_vjepa_metadata_uses_explicit_uniform_indices_by_default():
    model = object.__new__(WanKVMMD)
    model.feature_type = "dual_dinov3_vjepa2_dense_latent"
    model.dense_feature_prefix = "dinov3_vjepa2"
    model.dense_encoder_name = "dinov3_vjepa2"
    model.dense_log_prefix = "dinov3_vjepa2_dense"
    model.vjepa2_reference_model_id = "facebook/vjepa2-vitl-fpc64-256"
    model.vjepa2_model_id = "/unused/local/model"
    model.vjepa2_dense_layer = 15
    model.vjepa2_dense_layers = (15,)
    model.vjepa2_dense_num_frames = 64
    model.vjepa2_dense_source_num_frames = 81
    model.vjepa2_dense_frame_sampling = "center_contiguous"
    model.vjepa2_dense_resize_short_side = 292
    model.vjepa2_dense_crop_size = 256
    model.vjepa2_dense_raw_tokens = 2048
    model.vjepa2_dense_num_tokens = 1024
    model.vjepa2_dense_feature_dim = 1024
    actual = model._vjepa2_feature_metadata()
    indices = uniform_dense_token_indices(2048, 1024)
    expected = canonical_vjepa2_dense_feature_metadata(
        model_id="facebook/vjepa2-vitl-fpc64-256",
        layer_index=15,
        num_frames=64,
        resize_short_side=292,
        crop_size=256,
        raw_tokens=2048,
        num_tokens=1024,
        feature_dim=1024,
        token_indices=indices,
        frame_sampling="last_contiguous",
        source_num_frames=81,
    )
    assert actual == expected
    assert actual["token_alignment"] == "same_explicit_indices_and_order"

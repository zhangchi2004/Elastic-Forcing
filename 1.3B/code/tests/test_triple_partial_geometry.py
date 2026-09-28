from pathlib import Path


def test_released_recipe_uses_10_1_1_loss_divided_by_three():
    import yaml
    from utils.config import resolve_config_aliases
    root = Path(__file__).resolve().parents[2]
    config = resolve_config_aliases(yaml.safe_load((root / "configs/three_encoder.yaml").read_text()))
    assert config["triple_dinov3_weight"] == 10.0
    assert config["triple_vjepa2_weight"] == 1.0
    assert config["triple_videomae_v2_weight"] == 1.0
    assert config["triple_aggregation_divisor"] == 3.0
    assert config["partial_mmd_geometry"] == [256, 128, 128]

"""Exercise readable config inputs through the actual public entry points."""
import importlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml
from utils.config import load_config, resolve_config_aliases

ROOT = Path(__file__).resolve().parents[2]
RECIPE = "three_encoder.yaml" if ROOT.name == "1.3B" else "train.yaml"


def test_readable_override_wins_over_legacy_defaults_without_losing_metadata(tmp_path):
    defaults = tmp_path / "defaults.yaml"
    defaults.write_text("total_batch_size: 8\nmax_steps: 99\nmetadata: {revision: frozen}\n")
    path = tmp_path / "run.yaml"
    path.write_text("distribution_batch_size: 256\ntraining_updates: 1\nreference_minibatch_size: 128\n")
    cfg = load_config(path, defaults)
    assert (cfg.total_batch_size, cfg.max_steps, cfg.expected_current_teacher_samples) == (256, 1, 128)
    assert cfg.metadata.revision == "frozen"
    # Existing runtime YAML and arbitrary metadata survive the boundary unchanged.
    legacy = {"total_batch_size": 256, "metadata": {"shape": [1, 2, 3]}}
    resolved = resolve_config_aliases(legacy)
    resolved["metadata"]["shape"].append(4)
    assert legacy["metadata"]["shape"] == [1, 2, 3]


@pytest.mark.parametrize("step_flag", ["--max_steps", "--training-updates"])
def test_training_entry_resolves_recipe_before_constructing_trainer(monkeypatch, tmp_path, step_flag):
    captured = {}
    class RecordingTrainer:
        def __init__(self, config):
            captured["config"] = config
        def train(self):
            captured["trained"] = True

    monkeypatch.setitem(sys.modules, "trainer.elastic_forcing", SimpleNamespace(ElasticForcingTrainer=RecordingTrainer))
    spec = importlib.util.spec_from_file_location("readable_training_entry", ROOT / "code/train.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(entry.wandb, "finish", lambda: None)
    monkeypatch.setattr(entry.dist, "is_initialized", lambda: False)
    monkeypatch.chdir(ROOT / "code")
    monkeypatch.setattr(sys, "argv", ["train.py", "--config_path", str(ROOT / "configs" / RECIPE),
                                     "--logdir", str(tmp_path), "--disable-wandb", step_flag, "1"])
    entry.main()
    cfg = captured["config"]
    assert captured["trained"]
    assert cfg.trainer == "score_distillation" and cfg.distribution_loss == "wan_kv_mmd"
    assert cfg.total_batch_size == 256 and cfg.expected_current_teacher_samples == 128
    assert cfg.max_steps == 1 and cfg.lr == 2.5e-6
    assert cfg.triple_aggregation_divisor == 3.0
    # Stopping early must not shorten the scientific learning-rate schedule.
    if ROOT.name == "1.3B":
        assert cfg.generator_lr_cosine_end_step == 125
    else:
        assert cfg.lr_schedule_total_steps == 200


def test_prepare_config_preserves_recipe_and_only_relocates_assets(tmp_path):
    sandbox = tmp_path / "release"
    for folder in ("configs", "scripts", "code/utils"):
        (sandbox / folder).mkdir(parents=True, exist_ok=True)
    for source in (Path("configs") / RECIPE, Path("scripts/prepare_config.py"), Path("code/utils/config.py")):
        shutil.copy2(ROOT / source, sandbox / source)
    assets = tmp_path / "assets with spaces"
    command = [sys.executable, str(sandbox / "scripts/prepare_config.py"), "--asset-root", str(assets)]
    if ROOT.name == "14B":
        command.append("--inference-only")  # No external feature-bank files are needed to relocate paths.
    subprocess.run(command, check=True, capture_output=True, text=True)
    original = resolve_config_aliases(yaml.safe_load((ROOT / "configs" / RECIPE).read_text()))
    prefix = "/path/to/elastic-forcing-assets"
    expected = {k: str(assets) + v[len(prefix):] if isinstance(v, str) and v.startswith(prefix + "/") else v
                for k, v in original.items()}
    actual = yaml.safe_load((sandbox / "configs/local.yaml").read_text())
    assert actual == expected
    assert list(actual) == list(expected)


def test_legacy_modules_share_implementation_and_method_aliases():
    for old, new, old_class, new_class in (
        ("model.wan_kv_mmd", "model.elastic_forcing", "WanKVMMD", "ElasticForcingModel"),
        ("trainer.distillation", "trainer.elastic_forcing", "Trainer", "ElasticForcingTrainer"),
        ("losses.multi_representation_video_nystrom_mmd", "losses.multi_encoder_mmd", "MultiRepresentationVideoNystromMMD", "MultiEncoderMMD"),
        ("trainer.multi_representation_fresh_video_gradcache", "trainer.feature_gradient_replay", "MultiRepresentationFreshVideoGradCache", "FeatureGradientReplay"),
    ):
        legacy, current = importlib.import_module(old), importlib.import_module(new)
        assert legacy is current
        assert getattr(legacy, old_class) is getattr(current, new_class)
    from model.elastic_forcing import ElasticForcingModel
    from trainer.elastic_forcing import ElasticForcingTrainer
    assert ElasticForcingModel.fresh_video_gradcache_prepare_replay is ElasticForcingModel.prepare_latent_gradients
    assert ElasticForcingTrainer._fresh_video_generator_update is ElasticForcingTrainer._run_distribution_matching_step

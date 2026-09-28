from types import SimpleNamespace
import pytest
from trainer.distillation import _cosine_learning_rate, _learning_rate_schedule_identity


def test_step80_keeps_the_original_cosine_curve():
    cfg = SimpleNamespace(lr_schedule="cosine", lr=2.5e-6, lr_final=5e-7,
                          lr_schedule_total_steps=200, max_steps=80)
    early = _learning_rate_schedule_identity(cfg)
    cfg.max_steps = 200
    original = _learning_rate_schedule_identity(cfg)
    assert early == original
    for update in range(80):
        assert _cosine_learning_rate(update, start=early["start"], end=early["end"], total_updates=early["total_updates"]) == _cosine_learning_rate(update, start=2.5e-6, end=5e-7, total_updates=200)
    assert _cosine_learning_rate(79, start=2.5e-6, end=5e-7, total_updates=200) > 5e-7
    cfg.max_steps = 1
    assert _learning_rate_schedule_identity(cfg) == original
    cfg.max_steps = 201
    with pytest.raises(ValueError):
        _learning_rate_schedule_identity(cfg)

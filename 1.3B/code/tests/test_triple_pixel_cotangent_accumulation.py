import pytest
import torch

from model.wan_kv_mmd import _accumulate_fp32_pixel_cotangent


def test_pixel_cotangents_accumulate_immediately_in_fp32():
    accumulated = None
    shape = None
    expected = torch.zeros(2, 3, 4, dtype=torch.float32)
    norms = []
    for index, dtype in enumerate(
        (torch.bfloat16, torch.float16, torch.float32), start=1
    ):
        value = torch.full((2, 3, 4), float(index), dtype=dtype)
        expected.add_(value.float())
        accumulated, shape, norm = _accumulate_fp32_pixel_cotangent(
            accumulated, value, shape, branch_name=f"branch{index}")
        norms.append(norm)
    assert accumulated.dtype == torch.float32
    torch.testing.assert_close(accumulated, expected)
    assert shape == (2, 3, 4)
    assert all(item.ndim == 0 for item in norms)


def test_pixel_cotangent_accumulation_rejects_shape_and_nonfinite():
    first, shape, _ = _accumulate_fp32_pixel_cotangent(
        None, torch.ones(1, 2), None, branch_name="first")
    with pytest.raises(ValueError, match="shapes differ"):
        _accumulate_fp32_pixel_cotangent(
            first, torch.ones(1, 3), shape, branch_name="second")
    with pytest.raises(FloatingPointError, match="non-finite"):
        _accumulate_fp32_pixel_cotangent(
            None, torch.tensor([float("nan")]), None, branch_name="bad")

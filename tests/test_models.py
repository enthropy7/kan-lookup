import os
import sys

import pytest
import torch

from kantab.models import KAN

sys.path.insert(0, os.environ["KANTAB_EFFICIENT_KAN"])
import efficient_kan  # noqa: E402

SIZES = [7, 5, 3]


def pair(seed=0, **kwargs):
    torch.manual_seed(seed)
    ours = KAN(SIZES, **kwargs)
    torch.manual_seed(seed)
    return ours, efficient_kan.KAN(SIZES, **kwargs)


def test_same_seed_same_parameters():
    ours, ref = pair()
    for a, b in zip(ours.layers, ref.layers):
        torch.testing.assert_close(a.knots, b.grid[0])
        torch.testing.assert_close(a.base_weight, b.base_weight)
        torch.testing.assert_close(a.spline_weight, b.spline_weight)
        torch.testing.assert_close(a.spline_scaler, b.spline_scaler)


@pytest.mark.parametrize("grid_size,spline_order", [(5, 3), (10, 2), (3, 1)])
def test_forward_and_gradients_match(grid_size, spline_order):
    ours, ref = pair(grid_size=grid_size, spline_order=spline_order)
    with torch.no_grad():
        for a, b in zip(ours.layers, ref.layers):
            a.spline_weight.copy_(b.spline_weight)
    ours.double(), ref.double()
    torch.manual_seed(1)
    x = torch.randn(64, SIZES[0], dtype=torch.float64) * 2
    w = torch.randn(64, SIZES[-1], dtype=torch.float64)
    xs = [x.clone().requires_grad_(), x.clone().requires_grad_()]
    outs = [m(xi) for m, xi in zip((ours, ref), xs)]
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-12, atol=1e-12)
    for out in outs:
        (out * w).sum().backward()
    torch.testing.assert_close(xs[0].grad, xs[1].grad, rtol=1e-12, atol=1e-12)
    for a, b in zip(ours.layers, ref.layers):
        for name in ("base_weight", "spline_weight", "spline_scaler"):
            torch.testing.assert_close(getattr(a, name).grad, getattr(b, name).grad, rtol=1e-12, atol=1e-12)

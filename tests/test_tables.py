import torch

from kantab.tables import Int8MLP, LutKAN, edge_tables, int8_rows
from kantab.models import KAN
from kantab.models import MLP


def test_edge_tables_are_the_edge_functions():
    torch.manual_seed(0)
    layer = KAN([5, 3]).layers[0]
    grid = torch.linspace(-1, 1, 7).repeat(5, 1)
    t = edge_tables(layer, grid)
    for q in range(7):
        x = grid[:, q]
        assert torch.allclose(t[:, :, q].sum(1), layer(x.unsqueeze(0))[0], atol=1e-5)


def test_fine_lut_matches_fp32():
    torch.manual_seed(1)
    model = KAN([6, 4, 3])
    x = torch.rand(200, 6) * 2 - 1
    lut = LutKAN(model, bits=14, x_train=x, int8_tables=False)
    with torch.no_grad():
        assert (lut(x) - model(x)).abs().max() < 2e-2


def test_int8_rows_and_mlp():
    t = torch.tensor([[1.0, -0.5], [0.01, 0.02]])
    q = int8_rows(t)
    assert (q - t).abs().max() <= t.abs().max() / 127 / 2 + 1e-7
    torch.manual_seed(2)
    mlp = MLP([6, 8, 3])
    x = torch.rand(100, 6) * 2 - 1
    with torch.no_grad():
        assert (Int8MLP(mlp, x)(x) - mlp(x)).abs().max() < 5e-2


def test_int8_mlp_smooth_activation_tables():
    torch.manual_seed(3)
    x = torch.rand(300, 5) * 2 - 1
    for act in ("silu", "tanh"):
        mlp = MLP([5, 16, 16, 1], act=act)
        q = Int8MLP(mlp, x)
        assert q.act == act and len(q.pre_scale) == 2
        with torch.no_grad():
            err = (q(x) - mlp(x)).abs().max()
        assert 0 < err < 3e-2


def test_hidden_range_margin():
    torch.manual_seed(5)
    model = KAN([3, 4, 1])
    x = torch.rand(200, 3) * 2 - 1
    a, b = LutKAN(model, 6, x, interp=True), LutKAN(model, 6, x, interp=True, margin=0.05)
    (lo0, hi0, _), (lo1, hi1, _) = a.layers[1], b.layers[1]
    assert torch.allclose(lo1, lo0 - 0.05 * (hi0 - lo0)) and torch.allclose(hi1, hi0 + 0.05 * (hi0 - lo0))
    assert torch.equal(a.layers[0][0], b.layers[0][0])

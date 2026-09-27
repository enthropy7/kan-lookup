import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from kantab import maps, sweep, training
from kantab.data import formulas, physics
from kantab.export import f32_bytes, forest_bytes, lut_bytes, poly_bytes, q8_bytes, x_bytes
from kantab.tables import Int8MLP, LutKAN
from kantab.models import KAN, MLP

C_DIR = Path(__file__).resolve().parent.parent / "c"
CC = shutil.which(os.environ.get("CC", "cc"))


def test_synthetic_split():
    (xtr, ytr), (xva, yva), (xte, yte), (mean, std) = formulas.split("kan_toy2", 0)
    assert len(xtr) == 6000 and len(xva) == len(xte) == 2000
    assert xtr.min() == -1 and xtr.max() == 1 and abs(ytr.mean()) < 1e-5
    x, y, _ = formulas.load("kan_toy2")
    assert np.allclose(y, np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2))
    again = formulas.split("kan_toy2", 0)
    assert np.array_equal(again[2][0], xte) and again[3] == (mean, std)


def test_interpolated_lut_is_exact_on_grid_points_and_close_between():
    torch.manual_seed(0)
    model = KAN([3, 5, 1], grid_size=5)
    x = torch.rand(300, 3) * 2 - 1
    fine = LutKAN(model, 10, x, interp=True)
    with torch.no_grad():
        assert (fine(x) - model(x)).abs().max() < 2e-3
    grid = torch.linspace(-1, 1, 16).unsqueeze(1).repeat(1, 3)
    one = KAN([3, 2], grid_size=5)
    near, lin = LutKAN(one, 4, x, interp=False), LutKAN(one, 4, x, interp=True)
    with torch.no_grad():
        assert torch.allclose(near(grid), lin(grid), atol=1e-6)


def test_monomials_and_features():
    terms = training.monomials(3, 4)
    assert len(terms) == math.comb(3 + 4, 4)
    x = np.random.default_rng(0).uniform(-1, 1, (7, 3))
    f = training.poly_features(x, terms)
    for i, t in enumerate(terms):
        assert np.allclose(f[:, i], np.prod([x[:, v] for v in t], axis=0) if t else 1.0)


def test_sizes():
    assert training.nn_bytes("mlp1_w4", 2, "int8") == (2 * 4 + 8 * 4 + 4) + (4 + 8 + 4)
    assert training.nn_bytes("mlp1_w4", 2, "fp32") == 4 * (2 * 4 + 4 + 4 + 1)
    assert training.nn_bytes("kan1_w2_g5", 3, "lin4_i16") == (3 * 2 * 16 * 2 + 8 + 24) + (2 * 16 * 2 + 4 + 16)


@pytest.fixture(scope="module")
def c_run(tmp_path_factory):
    if CC is None:
        pytest.skip("no C compiler")
    exe = tmp_path_factory.mktemp("c") / "run"
    subprocess.run([CC, "-O2", "-std=c11", "-ffp-contract=off", "-Wall", "-Werror", "-o", str(exe),
                    str(C_DIR / "run.c"), str(C_DIR / "kernels.c"), str(C_DIR / "baselines.c"), str(C_DIR / "maps.c"), "-lm"],
                   check=True)
    return exe


def run_c(exe, kind, model_blob, x, tmp_path):
    (tmp_path / "m").write_bytes(model_blob)
    (tmp_path / "x").write_bytes(x_bytes(x))
    out = subprocess.run([str(exe), "eval", kind, str(tmp_path / "m"), str(tmp_path / "x")], capture_output=True,
                         check=True).stdout
    return np.frombuffer(out, "<f4")


@pytest.mark.parametrize("bits,interp", [(6, False), (8, False), (4, True), (6, True)])
def test_c_lut16_matches_simulator(c_run, tmp_path, bits, interp):
    torch.manual_seed(1)
    model = KAN([4, 16, 16, 1], grid_size=5)
    xtr = torch.rand(500, 4) * 2 - 1
    x = torch.rand(200, 4) * 2 - 1
    lut = LutKAN(model, bits, xtr, table_bits=16, interp=interp)
    with torch.no_grad():
        want = lut(x).reshape(-1).numpy()
    got = run_c(c_run, "lin16" if interp else "lut16", lut_bytes(model, lut), x.numpy(), tmp_path)
    ref, fast = got[:200], got[200:]
    tol = 1e-4 * max(1.0, np.abs(want).max())
    assert np.abs(ref - want).max() <= tol
    assert np.mean(np.abs(fast - want) <= tol) >= 0.98


def test_c_poly_matches_numpy(c_run, tmp_path):
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, (100, 3)).astype(np.float32)
    terms = training.monomials(3, 5)
    w = rng.normal(size=len(terms))
    want = training.poly_features(x.astype(np.float64), terms) @ w.astype(np.float32).astype(np.float64)
    got = run_c(c_run, "poly", poly_bytes(3, 5, w), x, tmp_path)
    assert np.abs(got - want).max() < 1e-4 * np.abs(want).max()


def test_c_forest_matches_sklearn(c_run, tmp_path):
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, (2000, 5)).astype(np.float32)
    y = np.sin(3 * x[:, 0]) + x[:, 1] * x[:, 2] + 0.1 * rng.normal(size=2000)
    m = HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=15, early_stopping=False, random_state=0).fit(x, y)
    xt = np.concatenate([x[:300], rng.uniform(-1.2, 1.2, (300, 5)).astype(np.float32)])
    got = run_c(c_run, "forest", forest_bytes(m), xt, tmp_path)
    assert np.abs(got - m.predict(xt)).max() < 1e-5


@pytest.mark.parametrize("d", [3, 20])
@pytest.mark.parametrize("kind,act", [(k, a) for k in ("q8", "f32") for a in ("relu", "silu", "tanh")] + [("lut8", None)])
def test_c_narrow_and_wide_layers_match_simulator(c_run, tmp_path, kind, d, act):
    torch.manual_seed(4)
    xtr, x = torch.rand(400, d) * 2 - 1, torch.rand(64, d) * 2 - 1
    for hidden in ([7], [40, 40], [37]):
        if kind == "lut8":
            model = KAN([d, *hidden, 1], grid_size=5)
            sim = LutKAN(model, 6, xtr, table_bits=8)
            blob = lut_bytes(model, sim, 8)
        else:
            model = MLP([d, *hidden, 1], act=act)
            with torch.no_grad():
                for m in model.net:
                    if isinstance(m, torch.nn.Linear):
                        m.bias.uniform_(-0.1, 0.3)
            sim = Int8MLP(model, xtr) if kind == "q8" else model
            blob = q8_bytes(sim) if kind == "q8" else f32_bytes(model)
        with torch.no_grad():
            want = sim(x).reshape(-1).numpy()
        got = run_c(c_run, kind, blob, x.numpy(), tmp_path)
        ref, fast = got[:64], got[64:]
        tol = 1e-5 * max(1.0, np.abs(want).max())
        assert np.abs(ref - want).max() <= tol and np.abs(fast - want).max() <= tol
        if kind != "f32":
            assert np.array_equal(ref, fast)


def test_shape_names():
    assert training.parse("mlp2_w16") == ("mlp", 2, 16, None, "relu")
    assert training.parse("mlp1_w8_tanh@30k") == ("mlp", 1, 8, None, "tanh")
    assert training.parse("kan1_w8_g10") == ("kan", 1, 8, 10, None)
    assert isinstance(training.build("mlp1_w8_silu", 3).net[1], torch.nn.SiLU)
    relu, silu = (training.nn_bytes(s, 3, "int8") for s in ("mlp2_w8", "mlp2_w8_silu"))
    assert silu - relu == 2 * 256
    assert sweep.spec("kan_toy2", "mlp2_w8_tanh@30k", "int8") == sweep.spec("kan_toy2", "mlp2_w8", "int8") + ":tanh"
    assert sweep.spec("kan_toy2", "mlp2_w8_silu", "fp32") == "f32:2-8-8-1:silu"


def test_train_nn_with_fine_tables():
    from kantab.experiments.fine_tables import FINE_VARIANTS
    data = formulas.split("kan_toy2", 0)
    r, _ = training.train_nn(data, "kan1_w2_g5", 1e-2, 0, steps=250, lut_variants=FINE_VARIANTS)
    assert {"lin7_i16", "lin8_i16", "lin10_i16", "lut6_i8"} <= set(r["test"])
    assert abs(r["test"]["lin10_i16"] - r["test"]["fp32"]) <= abs(r["test"]["lin6_i16"] - r["test"]["fp32"]) + 1e-6
    assert training.nn_bytes("kan1_w2_g5", 2, "lin8_i16") == (2 * 2 * 256 * 2 + 8 + 16) + (2 * 256 * 2 + 4 + 16)


def test_margin_variants_parse():
    assert training.table_bits("lin8_i16_m5") == (8, 16) and training.table_bits("lut6_i8") == (6, 8)
    assert training.nn_bytes("kan1_w2_g5", 2, "lin8_i16_m5") == training.nn_bytes("kan1_w2_g5", 2, "lin8_i16")
    assert sweep.spec("kan_toy2", "kan1_w32_g10@30k-r3", "lin8_i16_m5") == "lin16:2-32-1:256"


def test_rotkan_shapes_and_c_kernel(c_run, tmp_path):
    from kantab.export import rot_bytes
    from kantab.tables import RotLutKAN
    assert training.parse("rot2_kan1_w8_g10@30k") == ("rot", 1, 8, 10, None)
    assert training.dims("rot2_kan2_w8_g5", 3) == [3, 6, 8, 8, 1]
    assert training.nn_bytes("rot2_kan1_w8_g10", 3, "lin8_i16") == 4 * (3 * 6 + 6) + training.nn_bytes("kan1_w8_g10", 6, "lin8_i16")
    torch.manual_seed(6)
    model = training.build("rot2_kan1_w8_g10", 3)
    xtr, x = torch.rand(400, 3) * 2 - 1, torch.rand(64, 3) * 2 - 1
    sim = RotLutKAN(model, 8, xtr, table_bits=16, interp=True, margin=0.05)
    with torch.no_grad():
        want = sim(x).reshape(-1).numpy()
        assert (sim(x) - model(x)).abs().max() < 5e-3
    got = run_c(c_run, "rot16", rot_bytes(model, sim), x.numpy(), tmp_path)
    tol = 1e-4 * max(1.0, np.abs(want).max())
    assert np.abs(got[:64] - want).max() <= tol and np.mean(np.abs(got[64:] - want) <= tol) >= 0.98


def test_c_maps_match_simulator(c_run, tmp_path):
    import scipy.sparse as sp
    from kantab.export import cp_bytes, ga2m_bytes, grid_bytes
    torch.manual_seed(8)
    x = (torch.rand(200, 5) * 2.2 - 1.1).numpy()
    values = np.random.default_rng(0).normal(size=4 ** 5)
    r, c, v = maps.grid_weights(np.clip(x, -1, 1), 4)
    s = np.abs(values).max() / 32767
    want = sp.csr_matrix((v, (r, c)), shape=(200, 4 ** 5)) @ (np.round(values / s) * s)
    got = run_c(c_run, "grid", grid_bytes(values, 5, 4), x, tmp_path)
    assert np.abs(got - want).max() < 1e-5 * np.abs(want).max()
    for model, blob in ((maps.CPMap(5, 3, 16), cp_bytes), (maps.GA2M(5, 16, 8), ga2m_bytes)):
        with torch.no_grad():
            for p in model.parameters():
                p.normal_()
            want = model.deployed()(torch.from_numpy(x)).reshape(-1).numpy()
        got = run_c(c_run, "cp" if isinstance(model, maps.CPMap) else "ga2m", blob(model), x, tmp_path)
        assert np.abs(got - want).max() < 1e-4 * max(1.0, np.abs(want).max())


def test_c_gridv_matches_simulator(c_run, tmp_path):
    import scipy.sparse as sp
    from kantab.export import gridv_bytes
    x = (torch.rand(200, 4, generator=torch.Generator().manual_seed(9)) * 2.2 - 1.1).numpy()
    counts = (3, 5, 2, 7)
    values = np.random.default_rng(1).normal(size=3 * 5 * 2 * 7)
    r, c, v = maps.grid_weights(np.clip(x, -1, 1), counts)
    s = np.abs(values).max() / 32767
    want = sp.csr_matrix((v, (r, c)), shape=(200, len(values))) @ (np.round(values / s) * s)
    got = run_c(c_run, "gridv", gridv_bytes(values, counts), x, tmp_path)
    assert np.abs(got - want).max() < 1e-5 * np.abs(want).max()
    ru, cu, vu = maps.grid_weights(np.clip(x, -1, 1), 4)
    rv, cv, vv = maps.grid_weights(np.clip(x, -1, 1), (4, 4, 4, 4))
    assert (cu == cv).all() and np.allclose(vu, vv, rtol=0, atol=1e-15)
    assert maps.grid_size((4, 4, 4, 4), 4) == (256, 2 * 256 + 4 + 16) and maps.grid_size(4, 4) == (256, 516)


def test_grid_counts_follow_curvature():
    from kantab.maps import grid_counts
    n = grid_counts([1.0, 1.0, 0.0], 100)
    assert n[2] == 2 and n[0] == n[1] and n[0] * n[1] * 2 <= 100 and (n[0] + 1) * n[1] * 2 > 100
    n = grid_counts([16.0, 1.0], 64)
    assert n[0] > n[1] and n[0] * n[1] <= 64


def test_map_shapes():
    assert training.map_params("cp8_q32") == ("cp", 8, 32)
    assert training.map_params("ga2m_q32_16") == ("ga2m", 32, 16)
    from kantab.maps import CPMap, GA2M
    assert training.nn_bytes("cp8_q32", 5, "i16") == CPMap(5, 8, 32).n_bytes()
    assert training.nn_bytes("ga2m_q32_16", 5, "i16") == GA2M(5, 32, 16).n_bytes()


def test_pv_node_grid_reproduces_the_simulator_at_nodes():
    x, y, train, _, _ = physics._parts("pv_mp5")
    lo, hi = x[train].min(0), x[train].max(0)
    n = 3
    values = physics.node_values("pv_mp5", n)
    idx = (1, 0, 2, 1, 0)
    scaled = np.array([[-1 + 2 * i / (n - 1) for i in idx]])
    physical = np.array([[lo[a] + (hi[a] - lo[a]) * i / (n - 1) for a, i in enumerate(idx)]])
    r, c, v = maps.grid_weights(scaled, n)
    got = float(sum(vv * values[cc] for cc, vv in zip(c, v)))
    want = (physics.model("pv_mp5", physical)[0] - y[train].mean()) / y[train].std()
    assert abs(got - want) < 1e-9


def test_phys_node_grid_reproduces_the_simulator_at_nodes():
    x, y, train, _, _ = physics._parts("tyre4")
    lo, hi = x[train].min(0), x[train].max(0)
    counts = (3, 4, 2, 3)
    values = physics.node_values("tyre4", counts)
    idx = (1, 3, 0, 2)
    scaled = np.array([[-1 + 2 * i / (n - 1) for i, n in zip(idx, counts)]])
    physical = np.array([[lo[a] + (hi[a] - lo[a]) * i / (counts[a] - 1) for a, i in enumerate(idx)]])
    r, c, v = maps.grid_weights(scaled, counts)
    got = float(sum(vv * values[cc] for cc, vv in zip(c, v)))
    want = (physics.model("tyre4", physical)[0] - y[train].mean()) / y[train].std()
    assert abs(got - want) < 1e-9


def test_rule_msis_slices_match_msis7_and_igrf_interpolation():
    import datetime
    import ppigrf
    x = np.array([[400.0, 10.0, 15.0, 100.0, 120.0]])
    full = np.array([[400.0, 10.0, 15.0, 100.0, 157.5, 120.0, 40.0]])
    assert physics.model("msis_d5", x)[0] == physics.model("msis7", full)[0]
    lat, lon, alt = np.array([30.0, -60.0]), np.array([100.0, -20.0]), np.array([400.0, 700.0])
    ref = np.sqrt((np.stack(ppigrf.igrf(lon, lat, alt, datetime.datetime(2025, 1, 1)))[:, 0] ** 2).sum(0))
    got = physics.model("igrf4", np.stack([lat, lon, alt, np.full(2, 2025.0)], 1))
    assert np.abs(got - ref).max() < 1e-6

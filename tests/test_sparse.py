import numpy as np
import pytest

from kantab import sparse

rng = np.random.default_rng(0)
smooth = lambda x: np.exp(np.sin(2 * x[:, 0]) * np.cos(x[:, -1])) + 0.3 * x.sum(1) ** 2


def test_point_counts():
    # the modified basis has the points of a zero-boundary sparse grid: (n - 1) 2^n + 1 in two dimensions
    assert sparse.regular("mod", 2, 3).n_points == 17
    # one dimension with boundary: the full grid of 2^(n-1) + 1 points
    assert sparse.regular("bound", 1, 3).n_points == 5


@pytest.mark.parametrize("basis", ["bound", "mod"])
def test_interpolates_at_nodes(basis):
    g = sparse.regular(basis, 3, 4)
    g.interpolate(smooth)
    x = g.all_nodes()
    assert np.abs(g(x) - smooth(x)).max() < 1e-10


def test_reproduces_multilinear_and_linear():
    x = rng.uniform(-1, 1, (500, 2))
    f = lambda x: 1 + 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 0] * x[:, 1]
    g = sparse.regular("bound", 2, 1)
    g.interpolate(f)
    assert np.abs(g(x) - f(x)).max() < 1e-12
    lin = lambda x: 1 + 2 * x[:, 0] - x[:, 1]
    g = sparse.regular("mod", 2, 2)
    g.interpolate(lin)
    assert np.abs(g(x) - lin(x)).max() < 1e-12


@pytest.mark.parametrize("basis", ["bound", "mod"])
def test_adaptive_prefixes_interpolate(basis):
    g, sizes = sparse.adaptive(basis, 3, smooth, 4000, 0.9)
    assert sizes == sorted(sizes) and sizes[-1] <= 4000 and sizes[-1] == g.n_bytes(True)
    for k in (1, len(g.levels) // 2, len(g.levels)):
        p = sparse.prefix(g, k)
        assert p.n_bytes(True) == sizes[k - 1]
        x = p.all_nodes()
        assert np.abs(p(x) - smooth(x)).max() < 1e-10


@pytest.mark.parametrize("basis", ["bound", "mod"])
def test_design_matches_evaluation(basis):
    g = sparse.regular(basis, 3, 4)
    g.alpha = rng.normal(size=g.n_points)
    x = rng.uniform(-1.2, 1.2, (300, 3))
    assert np.abs(g.design(x) @ g.alpha - g(x)).max() < 1e-12


@pytest.mark.parametrize("basis", ["bound", "mod"])
def test_converges(basis):
    x = rng.uniform(-1, 1, (4000, 2))
    errors = []
    for n in (3, 5, 7):
        g = sparse.regular(basis, 2, n)
        g.interpolate(smooth)
        errors.append(np.sqrt(np.mean((g(x) - smooth(x)) ** 2)))
    assert errors[0] > 2 * errors[1] > 4 * errors[2]


def test_quantized_close():
    g = sparse.regular("mod", 3, 5)
    g.interpolate(smooth)
    x = rng.uniform(-1, 1, (500, 3))
    assert np.abs(g(x, g.quantized()) - g(x)).max() < 1e-3


def _c_maps(tmp_path):
    import ctypes
    import subprocess
    lib = tmp_path / "maps.so"
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(lib), "c/maps.c", "-lm"], check=True)
    return ctypes.CDLL(str(lib))


@pytest.mark.parametrize("basis", ["bound", "mod"])
def test_c_kernel_matches(basis, tmp_path):
    import ctypes

    class SGrid(ctypes.Structure):
        _fields_ = [("d", ctypes.c_int), ("s", ctypes.c_int), ("bound", ctypes.c_int),
                    ("levels", ctypes.POINTER(ctypes.c_uint8)), ("scale", ctypes.POINTER(ctypes.c_float)),
                    ("v", ctypes.POINTER(ctypes.c_int16))]
    lib = _c_maps(tmp_path)
    lib.sgrid_forward.restype = ctypes.c_float
    reg = sparse.regular(basis, 4, 4)
    reg.interpolate(smooth)
    for g in (reg, sparse.adaptive(basis, 4, smooth, 6000, 0.9)[0]):
        levels = np.ascontiguousarray(g.L.astype(np.uint8).ravel())
        scale = np.array([max(np.abs(g.alpha[g.offsets[k]:g.offsets[k + 1]]).max(), 1e-30) / 32767
                          for k in range(len(g.levels))], dtype=np.float32)
        v = np.concatenate([np.round(g.alpha[g.offsets[k]:g.offsets[k + 1]] / scale[k])
                            for k in range(len(g.levels))]).astype(np.int16)
        m = SGrid(4, len(g.levels), basis == "bound", levels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                  scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)))
        x = rng.uniform(-1.1, 1.1, (200, 4)).astype(np.float32)
        c = np.array([lib.sgrid_forward(ctypes.byref(m), row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for row in x])
        assert np.abs(c - g(x.astype(np.float64), g.quantized())).max() < 1e-4 * max(1, np.abs(c).max())


def test_c_plan_matches(tmp_path):
    import ctypes

    class SGrid(ctypes.Structure):
        _fields_ = [("d", ctypes.c_int), ("s", ctypes.c_int), ("bound", ctypes.c_int),
                    ("levels", ctypes.POINTER(ctypes.c_uint8)), ("scale", ctypes.POINTER(ctypes.c_float)),
                    ("v", ctypes.POINTER(ctypes.c_int16))]
    lib = _c_maps(tmp_path)
    lib.sgrid_forward.restype = ctypes.c_float
    lib.sgrid_forward_plan.restype = ctypes.c_float
    g = sparse.adaptive("mod", 5, smooth, 20000, 0.9)[0]
    levels = np.ascontiguousarray(g.L.astype(np.uint8).ravel())
    scale = np.full(len(g.levels), 1e-4, dtype=np.float32)
    v = rng.integers(-32767, 32767, g.n_points).astype(np.int16)
    m = SGrid(5, len(g.levels), 0, levels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
              scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), v.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)))
    lib.sgrid_forward_fast.restype = ctypes.c_float
    plan, fast = ctypes.create_string_buffer(256), ctypes.create_string_buffer(256)
    assert lib.sgrid_plan(ctypes.byref(m), plan) == 0 and lib.sgrid_fast(ctypes.byref(m), fast) == 0
    rows = np.concatenate([rng.uniform(-1.1, 1.1, (300, 5)), np.eye(5)[[0, 1]], -np.eye(5)[[2, 3]]]).astype(np.float32)
    for row in rows:
        px = row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        a = lib.sgrid_forward(ctypes.byref(m), px)
        for b in (lib.sgrid_forward_plan(plan, px), lib.sgrid_forward_fast(fast, px)):
            assert abs(a - b) <= 1e-4 + 1e-5 * abs(a)   # float32 over hundreds of terms of up to 3

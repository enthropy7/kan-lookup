import os
from datetime import date

import numpy as np

from kantab.splits import folds, scale_inputs

K = 5
NEW_FORMULAS = {
    "f_i27_6":   ("I.27.6", "A", [("d1", 1, 5), ("d2", 1, 5), ("n", 1, 5)], lambda d1, d2, n: 1 / (1 / d1 + n / d2)),
    "f_i34_1":   ("I.34.1", "A", [("w0", 1, 5), ("v", 1, 2), ("c", 3, 10)], lambda w0, v, c: w0 / (1 - v / c)),
    "f_ii6_15a": ("II.6.15a", "A", [("pd", 1, 3), ("eps", 1, 3), ("z", 1, 3), ("x", 1, 3), ("y", 1, 3), ("r", 1, 3)],
                  lambda pd, eps, z, x, y, r: 3 * pd * z * np.sqrt(x ** 2 + y ** 2) / (4 * np.pi * eps * r ** 5)),
    "f_i24_6":   ("I.24.6", "A", [("m", 1, 5), ("w", 1, 5), ("w0", 1, 5), ("x", 1, 5)],
                  lambda m, w, w0, x: m * (w ** 2 + w0 ** 2) * x ** 2 / 4),
    "f_ii35_18": ("II.35.18", "A", [("n0", 1, 5), ("mu", 1, 5), ("B", 1, 5), ("k", 1, 5), ("T", 1, 5)],
                  lambda n0, mu, B, k, T: n0 / (np.exp(mu * B / (k * T)) + np.exp(-mu * B / (k * T)))),
    "f_i41_16":  ("I.41.16", "A", [("w", 1, 5), ("T", 1, 5), ("h", 1, 5), ("k", 1, 5), ("c", 1, 5)],
                  lambda w, T, h, k, c: h * w ** 3 / (np.pi ** 2 * c ** 2 * (np.exp(h * w / (k * T)) - 1))),
    "f_i8_14":   ("I.8.14", "A", [("x1", 1, 5), ("x2", 1, 5), ("y1", 1, 5), ("y2", 1, 5)],
                  lambda x1, x2, y1, y2: np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)),
    "f_i11_19":  ("I.11.19", "A", [("x1", 1, 5), ("x2", 1, 5), ("x3", 1, 5), ("y1", 1, 5), ("y2", 1, 5), ("y3", 1, 5)],
                  lambda x1, x2, x3, y1, y2, y3: x1 * y1 + x2 * y2 + x3 * y3),
    "f_i12_11":  ("I.12.11", "B", [("q", 1, 5), ("Ef", 1, 5), ("B", 1, 5), ("v", 1, 5), ("th", 1, 5)],
                  lambda q, Ef, B, v, th: q * (Ef + B * v * np.sin(th))),
    "f_i13_12":  ("I.13.12", "B", [("G", 1, 5), ("m1", 1, 5), ("m2", 1, 5), ("r1", 1, 5), ("r2", 1, 5)],
                  lambda G, m1, m2, r1, r2: G * m1 * m2 * (1 / r2 - 1 / r1)),
    "f_i26_2":   ("I.26.2", "B", [("n", 0, 1), ("th2", 1, 5)], lambda n, th2: np.arcsin(n * np.sin(th2))),
    "f_i37_4":   ("I.37.4", "B", [("I1", 1, 5), ("I2", 1, 5), ("delta", 1, 5)],
                  lambda I1, I2, delta: I1 + I2 + 2 * np.sqrt(I1 * I2) * np.cos(delta)),
    "f_i44_4":   ("I.44.4", "B", [("n", 1, 5), ("k", 1, 5), ("T", 1, 5), ("V1", 1, 5), ("V2", 1, 5)],
                  lambda n, k, T, V1, V2: n * k * T * np.log(V2 / V1)),
    "f_i50_26":  ("I.50.26", "B", [("x1", 1, 3), ("w", 1, 3), ("t", 1, 3), ("alpha", 1, 3)],
                  lambda x1, w, t, alpha: x1 * (np.cos(w * t) + alpha * np.cos(w * t) ** 2)),
    "f_iii9_52": ("III.9.52", "B", [("pd", 1, 3), ("Ef", 1, 3), ("t", 1, 3), ("h", 1, 3), ("w", 1, 5), ("w0", 1, 5)],
                  lambda pd, Ef, t, h, w, w0: pd * Ef * t / (h / (2 * np.pi)) * np.sinc((w - w0) * t / 2 / np.pi) ** 2),
    "f_ii2_42":  ("II.2.42", "B", [("kappa", 1, 5), ("T1", 1, 5), ("T2", 1, 5), ("A", 1, 5), ("d", 1, 5)],
                  lambda kappa, T1, T2, A, d: kappa * (T2 - T1) * A / d),
}

SYNTHETIC = ("kan_toy2", "kan_toy4", "feyn_i6_2", "feyn_i29_16", "feyn_i40_1")
UCI = ("ccpp", "airfoil", "concrete", "energy", "airq")
TASKS = SYNTHETIC + UCI


def _synthetic(name, n=10_000):
    rng = np.random.default_rng(35_000 + SYNTHETIC.index(name))
    u = lambda lo, hi, d: rng.uniform(lo, hi, (n, d))
    if name == "kan_toy2":
        x = u(-1, 1, 2)
        y = np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2)
    elif name == "kan_toy4":
        x = u(-1, 1, 4)
        y = np.exp((np.sin(np.pi * (x[:, 0] ** 2 + x[:, 1] ** 2)) + np.sin(np.pi * (x[:, 2] ** 2 + x[:, 3] ** 2))) / 2)
    elif name == "feyn_i6_2":
        x = u(1, 3, 2)
        theta, sigma = x.T
        y = np.exp(-((theta / sigma) ** 2) / 2) / (np.sqrt(2 * np.pi) * sigma)
    elif name == "feyn_i29_16":
        x = u(1, 5, 4)
        x1, x2, t1, t2 = x.T
        y = np.sqrt(x1 ** 2 + x2 ** 2 - 2 * x1 * x2 * np.cos(t1 - t2))
    elif name == "feyn_i40_1":
        x = u(1, 5, 6)
        n0, m, xx, t, g, kb = x.T
        y = n0 * np.exp(-m * g * xx / (kb * t))
    return x, y, rng.permutation(n)


def _xlsx(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    return [r for r in ws.iter_rows(values_only=True)]


def _uci(name):
    if name == "ccpp":
        rows = _xlsx(os.environ["KANTAB_CCPP"] + "/CCPP/Folds5x2_pp.xlsx")[1:]
        a = np.array(rows, dtype=np.float64)
    elif name == "airfoil":
        a = np.loadtxt(os.environ["KANTAB_AIRFOIL"] + "/airfoil_self_noise.dat")
    elif name == "concrete":
        import xlrd
        s = xlrd.open_workbook(os.environ["KANTAB_CONCRETE"] + "/Concrete_Data.xls").sheet_by_index(0)
        a = np.array([s.row_values(i) for i in range(1, s.nrows)], dtype=np.float64)
    elif name == "energy":
        rows = [r[:9] for r in _xlsx(os.environ["KANTAB_ENERGY"] + "/ENB2012_data.xlsx")[1:] if r[0] is not None]
        a = np.array(rows, dtype=np.float64)
    else:
        return _airq()
    return a[:, :-1], a[:, -1], np.random.default_rng(35_100 + UCI.index(name)).permutation(len(a))


def _airq():
    with open(os.environ["KANTAB_AIR_QUALITY"] + "/AirQualityUCI.csv") as f:
        lines = f.read().splitlines()
    head = lines[0].split(";")
    cols = [head.index(c) for c in ("PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)",
                                     "T", "RH", "AH", "NO2(GT)")]
    xs, weeks = [], []
    for line in lines[1:]:
        f = line.split(";")
        if not f[0]:
            continue
        v = [float(f[c].replace(",", ".")) for c in cols]
        if -200.0 in v:
            continue
        d, m, y = map(int, f[0].split("/"))
        xs.append(v)
        weeks.append(date(y, m, d).toordinal() // 7)
    a = np.array(xs)
    return a[:, :-1], a[:, -1], np.array(weeks)


def _formula(name, n=10_000):
    _, _, variables, f = NEW_FORMULAS[name]
    rng = np.random.default_rng(35_200 + list(NEW_FORMULAS).index(name))
    x = np.stack([rng.uniform(lo, hi, n) for _, lo, hi in variables], axis=1)
    return x, f(*x.T), rng.permutation(n)


def load(name):
    if name in NEW_FORMULAS:
        return _formula(name)
    return _synthetic(name) if name in SYNTHETIC else _uci(name)


def split(name, fold):
    x, y, g = load(name)
    fs = folds(g, K)
    test, val = np.isin(g, fs[fold]), np.isin(g, fs[(fold + 1) % K])
    train = ~(test | val)
    xtr, xva, xte = scale_inputs(x[train], x[val], x[test])
    mean, std = y[train].mean(), y[train].std()
    z = lambda v: ((v - mean) / std).astype(np.float32)
    return (xtr, z(y[train])), (xva, z(y[val])), (xte, z(y[test])), (float(mean), float(std))

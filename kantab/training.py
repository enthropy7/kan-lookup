import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

from kantab.tables import CHUNK, Int8MLP, LutKAN, RotLutKAN, layer_ranges
from kantab.models import KAN, RotKAN
from kantab.models import MLP

STEPS, BATCH, EVAL_EVERY = 6000, 64, 250
LUT_VARIANTS = tuple((f"lut{b}_i{t}", b, t, False) for b in (6, 8) for t in (8, 16)) + \
    tuple((f"lin{b}_i16", b, 16, True) for b in (4, 6))


def parse(shape):
    parts = shape.split("@")[0].split("_")
    if parts[0].startswith("rot"):
        _, depth, width, grid, _ = parse("_".join(parts[1:]))
        return "rot", depth, width, grid, None
    fam, depth, width = parts[0][:3], int(parts[0][3:]), int(parts[1][1:])
    if fam == "mlp":
        return fam, depth, width, None, parts[2] if len(parts) > 2 else "relu"
    return fam, depth, width, int(parts[2][1:]), None


def rot_width(shape, d):
    return int(shape.split("_")[0][3:]) * d


def dims(shape, d):
    fam, depth, width, _, _ = parse(shape)
    front = [d, rot_width(shape, d)] if fam == "rot" else [d]
    return [*front, *[width] * depth, 1]


def map_params(shape):
    parts = shape.split("@")[0].split("_")
    if parts[0] == "ga2m":
        return "ga2m", int(parts[1][1:]), int(parts[2])
    return "cp", int(parts[0][2:]), int(parts[1][1:])


def is_map(shape):
    return shape.startswith(("cp", "ga2m"))


def build(shape, d):
    if is_map(shape):
        from kantab.maps import CPMap, GA2M
        kind, a, b = map_params(shape)
        return CPMap(d, a, b) if kind == "cp" else GA2M(d, a, b)
    fam, depth, width, grid, act = parse(shape)
    if fam == "rot":
        return RotKAN(d, rot_width(shape, d), [width] * depth, grid_size=grid)
    return MLP(dims(shape, d), act=act) if fam == "mlp" else KAN(dims(shape, d), grid_size=grid)


def nn_bytes(shape, d, variant):
    if is_map(shape):
        kind, a, b = map_params(shape)
        pairs = d * (d - 1) // 2
        if kind == "cp":
            entries, fp32_extra, scales = a * d * b, 4 * a + 4, 4 * a * d
        else:
            entries, fp32_extra, scales = d * a + pairs * b * b, 4, 4 * (d + pairs)
        return 4 * entries + fp32_extra if variant == "fp32" else 2 * entries + fp32_extra + scales
    sizes = dims(shape, d)
    if shape.startswith("rot"):
        m = sizes[1]
        kan = "kan{}_w{}_g{}".format(*parse(shape)[1:4])
        return 4 * (d * m + m) + nn_bytes(kan, m, variant)
    layers = list(zip(sizes[:-1], sizes[1:]))
    if variant == "fp32" and shape.startswith("mlp"):
        return sum(4 * (i * o + o) for i, o in layers)
    if variant == "int8":
        tables = 256 * (len(layers) - 1) if parse(shape)[4] != "relu" else 0
        return sum(i * o + 8 * o + 4 for i, o in layers) + tables
    if variant == "fp32":
        grid = parse(shape)[3]
        return sum(4 * i * o * (grid + 3 + 2) for i, o in layers)
    b, t = table_bits(variant)
    return sum(i * o * 2 ** b * t // 8 + 4 * o + 8 * i for i, o in layers)


def table_bits(variant):
    b, t = variant[3:].split("_m")[0].split("_i")
    return int(b), int(t)


def _predict(f, x):
    with torch.no_grad():
        return torch.cat([f(x[s:s + CHUNK]).reshape(-1) for s in range(0, len(x), CHUNK)])


def _rmse(pred, y, std):
    return float(std * torch.sqrt(torch.mean((pred.reshape(-1) - y) ** 2)))


def deployments(model, shape, xtr, lut_variants=LUT_VARIANTS):
    """The trained model and its deployed forms (tables, int8), by variant name; xtr is the training input tensor."""
    variants = {"fp32": model}
    if is_map(shape):
        variants["i16"] = model.deployed()
    elif shape.startswith("mlp"):
        variants["int8"] = Int8MLP(model, xtr)
    elif shape.startswith("rot"):
        with torch.no_grad():
            ranges = layer_ranges(model.kan, model.mix(xtr))
        for name, b, t, interp, *margin in lut_variants:
            if interp:
                variants[name] = RotLutKAN(model, b, xtr, table_bits=t, interp=True, margin=margin[0] if margin else 0.0,
                                           ranges=ranges)
    else:
        ranges = layer_ranges(model, xtr)
        for name, b, t, interp, *margin in lut_variants:
            variants[name] = LutKAN(model, b, xtr, table_bits=t, interp=interp, margin=margin[0] if margin else 0.0,
                                    ranges=ranges)
    return variants


def train_nn(data, shape, lr, seed, steps=STEPS, lut_variants=LUT_VARIANTS):
    (xtr, ytr), (xva, yva), (xte, yte), (_, std) = data
    xtr, ytr, xva, yva, xte, yte = map(torch.from_numpy, (xtr, ytr, xva, yva, xte, yte))
    torch.manual_seed(seed)
    model = build(shape, xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed + 10_000)
    perm, pos = torch.randperm(len(xtr), generator=gen), 0
    best = (math.inf, 0, None)
    for step in range(1, steps + 1):
        if pos + BATCH > len(xtr):
            perm, pos = torch.randperm(len(xtr), generator=gen), 0
        idx = perm[pos:pos + BATCH]
        pos += BATCH
        loss = F.mse_loss(model(xtr[idx]).reshape(-1), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                v = _rmse(_predict(model, xva), yva, 1.0)
            if v < best[0]:
                best = (v, step, {k: t.clone() for k, t in model.state_dict().items()})
    model.load_state_dict(best[2])
    model.eval()
    variants = deployments(model, shape, xtr, lut_variants)
    out = {"val": {}, "test": {}, "n_params": sum(p.numel() for p in model.parameters()), "best_step": best[1]}
    with torch.no_grad():
        for name, f in variants.items():
            out["val"][name] = _rmse(_predict(f, xva), yva, std)
            out["test"][name] = _rmse(_predict(f, xte), yte, std)
    return out, model


def fit_gbdt(data, n_trees, leaves, lr):
    from sklearn.ensemble import HistGradientBoostingRegressor
    (xtr, ytr), (xva, yva), (xte, yte), (_, std) = data
    m = HistGradientBoostingRegressor(max_iter=n_trees, max_leaf_nodes=leaves, learning_rate=lr,
                                      early_stopping=False, random_state=0).fit(xtr, ytr)
    rmse = lambda x, y: float(std * np.sqrt(np.mean((m.predict(x) - y) ** 2)))
    nodes = sum(len(p[0].nodes) for p in m._predictors)
    return {"val": {"gbdt": rmse(xva, yva)}, "test": {"gbdt": rmse(xte, yte)}, "nodes": nodes}, m


def monomials(d, degree):
    return [c for k in range(degree + 1) for c in itertools.combinations_with_replacement(range(d), k)]


def poly_features(x, terms):
    index = {t: i for i, t in enumerate(terms)}
    m = np.empty((len(x), len(terms)))
    m[:, 0] = 1.0
    for i, t in enumerate(terms[1:], 1):
        m[:, i] = m[:, index[t[:-1]]] * x[:, t[-1]]
    return m


def fit_poly(data, degree, alphas):
    (xtr, ytr), (xva, yva), (xte, yte), (_, std) = data
    terms = monomials(xtr.shape[1], degree)
    ftr, fva, fte = (poly_features(x.astype(np.float64), terms) for x in (xtr, xva, xte))
    a, b = ftr.T @ ftr / len(ftr), ftr.T @ ytr.astype(np.float64) / len(ftr)
    out = {}
    for alpha in alphas:
        pen = np.full(len(terms), alpha)
        pen[0] = 0.0
        w = np.linalg.lstsq(a + np.diag(pen), b, rcond=None)[0]
        rmse = lambda f, y: float(std * np.sqrt(np.mean((f @ w - y) ** 2)))
        out[alpha] = ({"val": {"poly": rmse(fva, yva)}, "test": {"poly": rmse(fte, yte)}, "terms": len(terms)}, w)
    return out

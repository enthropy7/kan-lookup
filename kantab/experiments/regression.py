import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from kantab import sweep, training
from kantab.data import formulas
from kantab.sweep import BASELINES

MLP_SHAPES = tuple(f"mlp{L}_w{W}" for L in (1, 2, 3) for W in (4, 8, 16, 32, 64, 128))
KAN_SHAPES = tuple(f"kan1_w{W}_g{G}" for W in (1, 2, 4, 8, 16, 32) for G in (5, 10, 20)) + \
    tuple(f"kan2_w{W}_g5" for W in (4, 8, 16))
NN_LRS, SEEDS = (1e-3, 3e-3, 1e-2), (0, 1)
GBDT_SHAPES = tuple(f"gbdt_n{N}_l{L}" for N in (25, 50, 100, 200, 400, 800) for L in (7, 15, 31))
GBDT_LRS = (0.05, 0.1, 0.3)
POLY_DEGREES, POLY_ALPHAS = (1, 2, 3, 4, 5, 6), (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
_cache = {}


def _data(task, fold):
    if (task, fold) not in _cache:
        _cache.clear()
        _cache[(task, fold)] = formulas.split(task, fold)
    return _cache[(task, fold)]


def _task(t):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kind, task, fold, shape, hyper, seed = t
    data = _data(task, fold)
    row = {"task": task, "fold": fold, "shape": shape, "seed": seed}
    if kind == "nn":
        r, _ = training.train_nn(data, shape, hyper, seed)
        return [{**row, "hyper": hyper, **r}]
    if kind == "gbdt":
        r, _ = training.fit_gbdt(data, *sweep.gbdt_shape(shape), hyper)
        return [{**row, "hyper": hyper, **r}]
    return [{**row, "hyper": alpha, **r} for alpha, (r, _) in training.fit_poly(data, int(shape[4:]), POLY_ALPHAS).items()]


def grid():
    out = []
    for task, fold in itertools.product(formulas.TASKS, range(formulas.K)):
        out += [("nn", task, fold, s, lr, seed) for s in MLP_SHAPES + KAN_SHAPES for lr in NN_LRS for seed in SEEDS]
        out += [("gbdt", task, fold, s, lr, 0) for s in GBDT_SHAPES for lr in GBDT_LRS]
        out += [("poly", task, fold, f"poly{d}", None, 0) for d in POLY_DEGREES]
    return out


def run(out, workers):
    sweep.run_pool(out, workers, grid(), {
        "tasks": formulas.TASKS, "folds": formulas.K, "mlp": MLP_SHAPES, "kan": KAN_SHAPES, "nn_lrs": NN_LRS,
        "seeds": SEEDS, "steps": training.STEPS, "batch": training.BATCH, "eval_every": training.EVAL_EVERY,
        "lut_variants": training.LUT_VARIANTS, "gbdt": GBDT_SHAPES, "gbdt_lrs": GBDT_LRS,
        "poly_degrees": POLY_DEGREES, "poly_alphas": POLY_ALPHAS}, _task)


def forests(out, dest):
    from kantab.export import forest_bytes, x_bytes
    sel = json.loads((Path(out) / "selection.json").read_text())
    for task in formulas.TASKS:
        (dest / task).mkdir(parents=True, exist_ok=True)
        for fold in range(formulas.K):
            data = formulas.split(task, fold)
            xte = data[2][0][:512]
            (dest / task / f"x_f{fold}.bin").write_bytes(x_bytes(xte))
            for shape in GBDT_SHAPES:
                lr = sel[task][f"{shape}/gbdt"][str(fold)]["hyper"]
                _, m = training.fit_gbdt(data, *sweep.gbdt_shape(shape), lr)
                (dest / task / f"{shape}_f{fold}.forest").write_bytes(forest_bytes(m))
                (dest / task / f"{shape}_f{fold}.expect").write_bytes(m.predict(xte).astype("<f4").tobytes())
        print(task, flush=True)


def checkset(dest):
    from kantab.export import forest_bytes, lut_bytes, poly_bytes, x_bytes
    from kantab.tables import LutKAN
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dest.mkdir(parents=True, exist_ok=True)
    data = formulas.split("kan_toy4", 0)
    (xtr, _), _, (xte, _), _ = data
    xte = xte[:512]
    (dest / "x.bin").write_bytes(x_bytes(xte))
    _, model = training.train_nn(data, "kan2_w16_g5", 3e-3, 0, steps=500)
    for name, bits, interp in (("lut16", 8, False), ("lin16", 6, True)):
        lut = LutKAN(model, bits, torch.from_numpy(xtr), table_bits=16, interp=interp)
        (dest / f"{name}.model").write_bytes(lut_bytes(model, lut))
        (dest / f"{name}.expect").write_bytes(lut(torch.from_numpy(xte)).numpy().reshape(-1).astype("<f4").tobytes())
    (_, w), = [v for a, v in training.fit_poly(data, 4, (1e-6,)).items()]
    (dest / "poly.model").write_bytes(poly_bytes(4, 4, w))
    feats = training.poly_features(xte.astype(np.float64), training.monomials(4, 4))
    (dest / "poly.expect").write_bytes((feats @ w).astype("<f4").tobytes())
    _, m = training.fit_gbdt(data, 200, 31, 0.1)
    (dest / "forest.model").write_bytes(forest_bytes(m))
    (dest / "forest.expect").write_bytes(m.predict(xte).astype("<f4").tobytes())


def checkcmp(dest):
    res = {}
    for kind in ("lut16", "lin16", "poly", "forest"):
        e = np.frombuffer((dest / f"{kind}.expect").read_bytes(), "<f4")
        o = np.frombuffer((dest / f"{kind}.out").read_bytes(), "<f4")
        parts = {"ref": o[:len(e)], "fast": o[len(e):]} if kind in ("lut16", "lin16") else {"fast": o}
        res[kind] = {p: float(np.abs(v - e).max()) for p, v in parts.items()} | {"expect_std": float(e.std())}
    print(json.dumps(res, indent=1))
    return res


def frontier(out, board):
    pts = sweep.points(out, board)
    verdicts = {task: sweep.judge(pts, task) for task in formulas.TASKS}
    on_val = [{**p, "rmse": p["val_rmse"]} for p in pts]
    verdicts_val = {task: sweep.judge(on_val, task) for task in formulas.TASKS}
    (Path(out) / f"frontier_{board}.json").write_text(json.dumps({"points": pts, "verdicts": verdicts,
                                                                 "verdicts_on_val": verdicts_val}, indent=1, default=str))
    for name in BASELINES:
        wins = [t for t, v in verdicts_val.items() if v[name]["win"]]
        print(f"on validation RMSE, {name}: {len(wins)}/{len(verdicts_val)} {wins}")
    short = lambda p: f"{p['shape']}/{p['variant']} {p['rmse']:.4g} {p['lat_ns'] / 1000:.2f}us {p['bytes']}B" if p else "—"
    for name in BASELINES:
        print(f"== {name} ({', '.join(BASELINES[name])})")
        wins = {}
        for task, v in verdicts.items():
            r = v[name]
            wins[task] = r["win"]
            print(f"{task:12s} T={r['target_rmse']:.4g} best={short(r['best_baseline'])} | fastest base {short(r['base_fastest'])}"
                  f" | kan fastest {short(r['kan_fastest'])} lat×{r['lat_ratio']:.3g} size×{r['size_ratio']:.3g}"
                  f" | kan best {r['kan_best_rmse']:.4g} (b: {short(r['b'])}) → {'WIN' if r['win'] else 'no'}")
        n_all = sum(wins.values())
        n_real = sum(wins[t] for t in formulas.UCI)
        print(f"{name}: {n_all}/{len(wins)} tasks (needs ≥ {len(wins) // 2}); real {n_real}/{len(formulas.UCI)}")
    return verdicts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=12)
    fo = sub.add_parser("forests")
    fo.add_argument("out", type=Path)
    fo.add_argument("dest", type=Path)
    for c in ("checkset", "checkcmp"):
        sub.add_parser(c).add_argument("dest", type=Path)
    fr = sub.add_parser("frontier")
    fr.add_argument("out", type=Path)
    fr.add_argument("--board", default="orangepi")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.out, a.workers)
    elif a.cmd == "forests":
        forests(a.out, a.dest)
    elif a.cmd in ("checkset", "checkcmp"):
        {"checkset": checkset, "checkcmp": checkcmp}[a.cmd](a.dest)
    else:
        frontier(a.out, a.board)

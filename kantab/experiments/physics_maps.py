import argparse
import json
import math
from pathlib import Path

import numpy as np

from kantab import sweep, training
from kantab.data import physics
from kantab.experiments import regression, structure_rule, battery_maps, pv_maps

GRID_N = {"tyre4": (3, 4, 5, 6, 8, 10, 12, 15), "msis7": (2, 3, 4), "flame5": (3, 4, 5, 6, 7, 8, 9)}
BUDGETS, MARGIN = pv_maps.BUDGETS, pv_maps.MARGIN
FILL = (1.0, 0.5)
PV_TASKS = physics.PV_TASKS
PV_MAPS, PHYSICS_MAPS = pv_maps.PV_MAPS, Path("results/raw/physics_maps")
CURV_POINTS, CURV_STEP = 2000, 0.05
_cache = {}


def curvature(task, fold=0):
    mod = physics
    x, y, train, _, _ = mod._parts(task, fold)
    lo, hi, sd = x[train].min(0), x[train].max(0), y[train].std()
    d, h = x.shape[1], CURV_STEP
    s = np.random.default_rng(42_042).uniform(-1 + h, 1 - h, (CURV_POINTS, d))
    phys = lambda u: lo + (u + 1) / 2 * (hi - lo)
    f0 = mod.model(task, phys(s))
    out = []
    for a in range(d):
        e = np.zeros(d)
        e[a] = h
        f2 = (mod.model(task, phys(s + e)) - 2 * f0 + mod.model(task, phys(s - e))) / h ** 2 / sd
        out.append(float(np.sqrt(np.mean(f2 ** 2))))
    return out


def aniso_counts(task, curv):
    from kantab.maps import grid_counts
    d = len(curv)
    out = {}
    for b in BUDGETS:
        for fill in FILL:
            counts = grid_counts(curv, int((b * fill - 4 - 4 * d) // 2))
            out["x".join(map(str, counts))] = counts
    return out


def _data(task):
    if task not in _cache:
        _cache.clear()
        _cache[task] = physics.split(task)
    return _cache[task]


def _task(t):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kind, task, shape, hyper, seed = t
    data = _data(task)
    row = {"task": task, "fold": 0, "shape": shape, "seed": seed}
    if kind in ("nn", "gbdt", "poly"):
        return _fit(kind, data, shape, hyper, seed, row)
    from kantab.maps import eval_grid, fit_grid
    n = sweep.grid_counts(shape)
    if kind == "gridnode":
        r = eval_grid(data, n, physics.node_values(task, n))
        return [{**row, "hyper": None, "val": r["val"], "test": r["test"], "n_bytes": r["n_bytes"]}]
    return [{**row, "hyper": lam, "val": r["val"], "test": r["test"], "n_bytes": r["n_bytes"]}
            for lam, (r, _) in fit_grid(data, n, battery_maps.GRID_LAMS).items()]


def _fit(kind, data, shape, hyper, seed, row):
    if kind == "nn":
        r, _ = training.train_nn(data, shape, hyper, seed, steps=battery_maps.STEPS, lut_variants=structure_rule.VARIANTS)
        return [{**row, "hyper": hyper, "steps": battery_maps.STEPS, **r}]
    if kind == "gbdt":
        r, _ = training.fit_gbdt(data, *sweep.gbdt_shape(shape), hyper)
        return [{**row, "hyper": hyper, **r}]
    return [{**row, "hyper": a, **r} for a, (r, _) in
            training.fit_poly(data, int(shape[4:]), regression.POLY_ALPHAS).items()]


def grid(curvs, tasks=PV_TASKS + physics.MAP_TASKS, grid_n=GRID_N, aniso_only=PV_TASKS):
    out = []
    for task in tasks:
        cs = sorted(aniso_counts(task, curvs[task]), key=lambda k: -math.prod(map(int, k.split("x"))))
        out += [("grid", task, f"gridv_n{c}", None, 0) for c in cs]
        out += [("gridnode", task, f"gridvnode_n{c}", None, 0) for c in cs]
        if task in aniso_only:
            continue
        out += [("grid", task, f"grid_n{n}", None, 0) for n in sorted(grid_n[task], reverse=True)]
        out += [("gridnode", task, f"gridnode_n{n}", None, 0) for n in grid_n[task]]
        out += [("nn", task, s, lr, seed) for s in battery_maps.KAN + battery_maps.MLP + battery_maps.CP + battery_maps.GA2M
                for lr in regression.NN_LRS for seed in (0, 1)]
        out += [("gbdt", task, s, lr, 0) for s in regression.GBDT_SHAPES for lr in regression.GBDT_LRS]
        out += [("poly", task, f"poly{d}", None, 0) for d in regression.POLY_DEGREES]
    return out


def forests(dest):
    from kantab.export import forest_bytes, x_bytes
    sel = json.loads((PHYSICS_MAPS / "selection.json").read_text())
    for task in physics.MAP_TASKS:
        data = physics.split(task)
        (dest / task).mkdir(parents=True, exist_ok=True)
        xte = data[2][0][:512]
        (dest / task / "x_f0.bin").write_bytes(x_bytes(xte))
        for shape in regression.GBDT_SHAPES:
            lr = sel[task][f"{shape}/gbdt"]["0"]["hyper"]
            _, m = training.fit_gbdt(data, *sweep.gbdt_shape(shape), lr)
            (dest / task / f"{shape}_f0.forest").write_bytes(forest_bytes(m))
            (dest / task / f"{shape}_f0.expect").write_bytes(m.predict(xte).astype("<f4").tobytes())


def frontier(board):
    new = sweep.deployed(sweep.points(PHYSICS_MAPS, board))
    old = sweep.deployed(sweep.points(PV_MAPS, board))
    result = {}
    f = lambda p: (f"{p['shape']}/{p['variant']} {p['rmse']:.4g} {p['bytes']}B"
                   + (f" {p['lat_ns'] / 1000:.2f}us" if p.get("lat_ns") else "")) if p else "—"
    def cells(pts, tasks, fams, pick, name):
        out = []
        for task in tasks:
            P = [p for p in pts if p["task"] == task]
            for b in BUDGETS:
                k = sweep.best_under(P, battery_maps.FAMILIES["kan"], b, pick)
                c = sweep.best_under(P, fams, b, pick)
                win = bool(k and c and k["rmse"] <= MARGIN * c["rmse"])
                out.append(win)
                print(f"  [{pick}] {name} {task} ≤{b}B: KAN {f(k)} | {'/'.join(fams)} {f(c)} → {'WIN' if win else 'no'}")
        return out
    for pick in ("val_rmse", "rmse"):
        rows = {}
        for name, fams in (("C1", battery_maps.FAMILIES["grid"]), ("C2", battery_maps.FAMILIES["maps"])):
            c = cells([p for p in new if p["task"] in physics.MAP_TASKS], physics.MAP_TASKS, fams, pick, name)
            rows[name] = {"cells": c, "holds": sum(c) >= 6}
            print(f"== [{pick}] {name}: {sum(c)}/9 cells → {'CONFIRMED' if sum(c) >= 6 else 'FALSIFIED'}")
        c = cells(old + [p for p in new if p["task"] in PV_TASKS], PV_TASKS, battery_maps.FAMILIES["grid"], pick, "C3")
        rows["C3"] = {"cells": c, "holds": sum(c) >= 4}
        print(f"== [{pick}] C3 (PV maps C1 with anisotropic grids): {sum(c)}/6 cells → {'CONFIRMED' if sum(c) >= 4 else 'FALSIFIED'}")
        result[pick] = rows
    for task in physics.MAP_TASKS:
        timed = [p for p in new if p["task"] == task and p.get("lat_ns")]
        r = sweep.judge(timed, task, ("kan",), {"all": battery_maps.FAMILIES["all"]})["all"]
        print(f"== C4 (descriptive) {task}: best alternative {f(r['best_baseline'])}; best KAN {r['kan_best_rmse']:.4g}"
              f" → {'WIN' if r['win'] else 'no'}")
        result[f"C4_{task}"] = r
    (PHYSICS_MAPS / f"frontier_{board}.json").write_text(json.dumps({"points": new, "criteria": result}, indent=1, default=str))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=10)
    sub.add_parser("counts")
    sub.add_parser("forests").add_argument("dest", type=Path)
    sub.add_parser("frontier").add_argument("--board", default="orangepi")
    a = p.parse_args()
    if a.cmd in ("run", "counts"):
        curvs = {task: curvature(task) for task in PV_TASKS + physics.MAP_TASKS}
        counts = {task: list(aniso_counts(task, curvs[task])) for task in curvs}
        for task in curvs:
            print(task, "curvature", [f"{c:.3g}" for c in curvs[task]], "→", counts[task])
        for task in physics.MAP_TASKS:
            physics.load(task)
    if a.cmd == "run":
        sweep.run_pool(a.out, a.workers, grid(curvs), {
            "tasks": physics.MAP_TASKS, "pv_tasks": PV_TASKS, "n": physics.N, "grid_n": GRID_N,
            "curvature": curvs, "aniso_counts": counts, "fill": FILL, "curv_points": CURV_POINTS, "curv_step": CURV_STEP,
            "grid_lams": battery_maps.GRID_LAMS, "mlp": battery_maps.MLP, "kan": battery_maps.KAN, "cp": battery_maps.CP,
            "ga2m": battery_maps.GA2M, "steps": battery_maps.STEPS, "lrs": regression.NN_LRS, "seeds": (0, 1),
            "lut_variants": structure_rule.VARIANTS, "gbdt": regression.GBDT_SHAPES, "poly": regression.POLY_DEGREES,
            "budgets": BUDGETS, "margin": MARGIN}, fn=_task)
    elif a.cmd == "forests":
        forests(a.dest)
    elif a.cmd == "frontier":
        frontier(a.board)

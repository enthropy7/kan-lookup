"""The worst test error of the models chosen on validation RMSE (split 0): each chosen KAN table, grid, CP or GA2M
table and sparse grid is rebuilt from its recorded settings, checked against its recorded test RMSE, and its
absolute errors on the test set are summarized by their maximum and 99.9th percentile, in the output's units."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from kantab import sweep, training
from kantab.data import physics
from kantab.experiments import battery_maps, data_splits, sparse_budgets, structure_rule

OUT = Path("results/raw/worst_case")
TASKS = data_splits.TASKS
MANY = [t for t in TASKS if sweep.n_inputs(t) >= 5]
BUDGETS = (4096, 16384, 32768, 131072)
REGISTERED = (32768, 131072)
GROUPS = {"kan": ("kan",), "maps": ("grid", "cp", "ga2m"), "sgrid": ("sgrid",)}
SEEDS = (0, 1)
TOLERANCE = 1e-6
_cache = {}


def _data(task):
    if task not in _cache:
        _cache.clear()
        _cache[task] = physics.split(task)
    return _cache[task]


def chosen():
    """The models chosen on validation RMSE within each budget, per task and group, with their recorded settings."""
    pts = sparse_budgets.points(0)
    fold0 = data_splits.selections_fold0()
    new = sparse_budgets.selections()
    out = {}
    for task in TASKS:
        P = [p for p in pts if p["task"] == task]
        for b in BUDGETS:
            for group, fams in GROUPS.items():
                p = sweep.best_under(P, fams, b, "val_rmse")
                if not p:
                    continue
                key = (task, p["shape"], p["variant"], 0)
                hyper = new[key]["hyper"] if key in new else fold0[task][f"{p['shape']}/{p['variant']}"]["0"]["hyper"]
                out[(task, p["shape"], p["variant"])] = {**p, "hyper": hyper}
    return out


def jobs():
    out = []
    for (task, shape, variant), p in chosen().items():
        if p["family"] in ("kan", "cp", "ga2m"):
            out += [("nn", task, shape, variant, p["hyper"], seed) for seed in SEEDS]
        else:
            out.append((p["family"], task, shape, variant, p["hyper"], 0))
    # networks and sparse-grid fits first: they take longest
    return sorted(out, key=lambda j: (j[0] == "grid", j[1]))


def _nn(task, shape, variant, lr, seed):
    import torch
    data = _data(task)
    (xtr, _), _, (xte, yte), (_, std) = data
    _, model = training.train_nn(data, shape, lr, seed, steps=battery_maps.STEPS, lut_variants=structure_rule.VARIANTS)
    f = training.deployments(model, shape, torch.from_numpy(xtr), structure_rule.VARIANTS)[variant]
    pred = training._predict(f, torch.from_numpy(xte))
    rmse = training._rmse(pred, torch.from_numpy(yte), std)
    return rmse, pred.numpy().astype(np.float64), yte, std


def _grid(task, shape, lam):
    import scipy.sparse as sp
    from kantab.maps import fit_grid, grid_size, grid_weights
    data = _data(task)
    _, _, (xte, yte), (_, std) = data
    n = sweep.grid_counts(shape)
    if shape.startswith(("gridnode", "gridvnode")):
        values = physics.node_values(task, n)
    else:
        values = fit_grid(data, n, battery_maps.GRID_LAMS)[lam][1]
    s = np.abs(values).max() / 32767
    vq = np.round(values / s) * s
    r, c, v = grid_weights(xte, n)
    pred = sp.csr_matrix((v, (r, c)), shape=(len(xte), grid_size(n, xte.shape[1])[0])) @ vq
    return float(std * np.sqrt(np.mean((pred - yte) ** 2))), pred, yte, std


def _sgrid(task, shape, lam):
    from scipy.sparse.linalg import lsqr
    (xtr, ytr), _, (xte, yte), (_, std) = _data(task)
    base = shape.removesuffix("_fit")
    g = next(g for s, g, _ in sparse_budgets._cuts(task, 0, base.split("_")[1]) if s == base)
    alpha = g.alpha
    if shape.endswith("_fit"):
        N = len(xtr)
        alpha = lsqr(g.design(xtr, 1 / math.sqrt(N)), ytr / math.sqrt(N), damp=math.sqrt(lam), x0=g.alpha,
                     iter_lim=sparse_budgets.ITERS, atol=1e-10, btol=1e-10)[0]
    pred = g(xte, sparse_budgets.g_q(g, alpha))
    return float(std * np.sqrt(np.mean((pred - yte) ** 2))), pred, yte, std


def _task(t):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kind, task, shape, variant, hyper, seed = t
    if kind == "nn":
        rmse, pred, y, std = _nn(task, shape, variant, hyper, seed)
    elif kind == "grid":
        rmse, pred, y, std = _grid(task, shape, hyper)
    else:
        rmse, pred, y, std = _sgrid(task, shape, hyper)
    err = std * np.abs(pred - y.astype(np.float64))
    return [{"task": task, "shape": shape, "variant": variant, "seed": seed, "hyper": hyper, "rmse": rmse,
             "max": float(err.max()), "p999": float(np.quantile(err, 0.999)), "std": std}]


def frontier():
    rows = defaultdict(list)
    for line in (OUT / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[(r["task"], r["shape"], r["variant"])].append(r)
    ch = chosen()
    metrics, failed = {}, []
    for key, p in ch.items():
        rs = rows.get(key, [])
        m = {k: statistics.mean(r[k] for r in rs) for k in ("rmse", "max", "p999")} if rs else None
        if not m or abs(m["rmse"] - p["rmse"]) > TOLERANCE * p["rmse"]:
            failed.append((key, p["rmse"], m and m["rmse"]))
            continue
        metrics[key] = {**m, "std": rs[0]["std"], "family": p["family"], "bytes": p["bytes"]}
    print(f"rebuilt {len(metrics)} of {len(ch)} chosen models to their recorded test RMSE"
          + (f"; not reproduced: {failed}" if failed else ""))
    pts = sparse_budgets.points(0)
    cells, counts = {}, {}
    for b in BUDGETS:
        for task in TASKS:
            P = [p for p in pts if p["task"] == task]
            cell = {}
            for group, fams in GROUPS.items():
                p = sweep.best_under(P, fams, b, "val_rmse")
                cell[group] = p and metrics.get((task, p["shape"], p["variant"])) and \
                    {"shape": p["shape"], "variant": p["variant"], **metrics[(task, p["shape"], p["variant"])]}
            cells[f"{task} {b}"] = cell
            k, m, s = cell["kan"], cell["maps"], cell["sgrid"]
            show = lambda x: f"{x['shape']:28} rmse {x['rmse']:.3g} max {x['max']:.3g} p99.9 {x['p999']:.3g}" if x else "—"
            print(f"{task:10} {b // 1024:>3}K KAN {show(k)}\n{'':15}map {show(m)}\n{'':15}sgr {show(s)}")
        for metric in ("max", "p999"):
            won = [t for t in MANY if cells[f"{t} {b}"]["kan"] and cells[f"{t} {b}"]["maps"]
                   and cells[f"{t} {b}"]["kan"][metric] < cells[f"{t} {b}"]["maps"][metric]]
            vs_s = [t for t in MANY if cells[f"{t} {b}"]["kan"] and cells[f"{t} {b}"]["sgrid"]
                    and cells[f"{t} {b}"]["kan"][metric] < cells[f"{t} {b}"]["sgrid"][metric]]
            counts[f"{metric} {b}"] = {"vs_maps": won, "vs_sgrid": vs_s}
            print(f"== {b // 1024} KB, {metric}: KAN table smaller than the best grid/CP/GA2M table on {len(won)}/8"
                  f" (not on: {', '.join(t for t in MANY if t not in won) or 'none'}); than the best sparse grid on"
                  f" {len(vs_s)}/8")
    w1 = all(len(counts[f"max {b}"]["vs_maps"]) >= 6 for b in REGISTERED)
    w2 = all(len(counts[f"p999 {b}"]["vs_maps"]) >= 6 for b in REGISTERED)
    tails = defaultdict(list)
    for key, m in metrics.items():
        tails["maps" if m["family"] in GROUPS["maps"] else m["family"]].append(m["max"] / m["rmse"])
    print("   largest error over RMSE per family: " + ", ".join(
        f"{g} {statistics.median(v):.1f} [{min(v):.1f}, {max(v):.1f}]" for g, v in sorted(tails.items())))
    print(f"== W1 (largest error, 32 and 128 KB) {'CONFIRMED' if w1 else 'FALSIFIED'} | "
          f"W2 (99.9th percentile) {'CONFIRMED' if w2 else 'FALSIFIED'}")
    (OUT / "frontier.json").write_text(json.dumps({"W1": w1, "W2": w2, "counts": counts, "cells": cells,
                                                   "not_reproduced": failed}, indent=1, default=str))
    return w1, w2


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").add_argument("--workers", type=int, default=10)
    sub.add_parser("jobs")
    sub.add_parser("frontier")
    a = p.parse_args()
    if a.cmd == "run":
        for task in TASKS:   # simulated once here, shared with the workers by fork
            physics.load(task)
        sweep.run_pool(OUT, a.workers, jobs(), {"tasks": TASKS, "budgets": BUDGETS, "groups": GROUPS, "seeds": SEEDS,
                                                "tolerance": TOLERANCE}, fn=_task)
    elif a.cmd == "jobs":
        for j in jobs():
            print(*j)
    else:
        frontier()

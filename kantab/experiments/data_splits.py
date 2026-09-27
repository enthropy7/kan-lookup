import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from kantab import sweep, training
from kantab.data import physics
from kantab.experiments import regression, structure_rule, battery_maps, pv_maps, physics_maps, input_rule

FOLDS = (1, 2)
DATA_SPLITS = Path("results/raw/data_splits")
SOURCE = {**dict.fromkeys(physics.PV_TASKS, pv_maps.PV_MAPS), **dict.fromkeys(physics.MAP_TASKS, physics_maps.PHYSICS_MAPS),
          **dict.fromkeys(physics.RULE_TASKS, input_rule.INPUT_RULE)}
GRID_N = {**pv_maps.GRID_N, **physics_maps.GRID_N, **input_rule.GRID_N}
TASKS = physics.PV_TASKS + physics.MAP_TASKS + physics.RULE_TASKS
TOP_MLP = 4
_cache = {}


def _data(task, fold):
    if (task, fold) not in _cache:
        _cache.clear()
        _cache[(task, fold)] = physics.split(task, fold)
    return _cache[(task, fold)]


def _task(t):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kind, task, shape, hyper, seed, fold = t
    data = _data(task, fold)
    row = {"task": task, "fold": fold, "shape": shape, "seed": seed}
    if kind == "nn":
        r, _ = training.train_nn(data, shape, hyper, seed, steps=battery_maps.STEPS, lut_variants=structure_rule.VARIANTS)
        return [{**row, "hyper": hyper, "steps": battery_maps.STEPS, **r}]
    if kind == "poly":
        return [{**row, "hyper": a, **r} for a, (r, _) in
                training.fit_poly(data, int(shape[4:]), regression.POLY_ALPHAS).items()]
    from kantab.maps import eval_grid, fit_grid
    n = sweep.grid_counts(shape)
    if kind == "gridnode":
        r = eval_grid(data, n, physics.node_values(task, n, fold))
        return [{**row, "hyper": None, "val": r["val"], "test": r["test"], "n_bytes": r["n_bytes"]}]
    return [{**row, "hyper": lam, "val": r["val"], "test": r["test"], "n_bytes": r["n_bytes"]}
            for lam, (r, _) in fit_grid(data, n, battery_maps.GRID_LAMS).items()]


def top_mlps(task):
    sel = json.loads((SOURCE[task] / "selection.json").read_text())[task]
    scored = sorted((v["0"]["val"], k.split("/")[0]) for k, v in sel.items() if k.startswith("mlp") and k.endswith("/fp32"))
    return [s for _, s in scored[:TOP_MLP]]


def grid(curvs):
    out = []
    for fold in FOLDS:
        for task in TASKS:
            cs = sorted(physics_maps.aniso_counts(task, curvs[(task, fold)]), key=lambda k: -math.prod(map(int, k.split("x"))))
            out += [("grid", task, f"gridv_n{c}", None, 0, fold) for c in cs]
            out += [("gridnode", task, f"gridvnode_n{c}", None, 0, fold) for c in cs]
            out += [("grid", task, f"grid_n{n}", None, 0, fold) for n in sorted(GRID_N[task], reverse=True)]
            out += [("gridnode", task, f"gridnode_n{n}", None, 0, fold) for n in GRID_N[task]]
            out += [("nn", task, s, lr, seed, fold) for s in battery_maps.KAN + battery_maps.CP + battery_maps.GA2M + tuple(top_mlps(task))
                    for lr in regression.NN_LRS for seed in (0, 1)]
            out += [("poly", task, f"poly{d}", None, 0, fold) for d in regression.POLY_DEGREES]
    return out


def selections():
    groups = defaultdict(list)
    for line in (DATA_SPLITS / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        for variant in r["val"]:
            groups[(r["task"], r["shape"], variant, r["fold"], r["hyper"])].append(r)
    best = {}
    for (task, shape, variant, fold, hyper), rs in groups.items():
        v = sum(r["val"][variant] for r in rs) / len(rs)
        key = (task, shape, variant, fold)
        if key not in best or v < best[key]["val"]:
            best[key] = {"val": v, "test": sum(r["test"][variant] for r in rs) / len(rs), "hyper": hyper}
    out = {f: defaultdict(dict) for f in FOLDS}
    for (task, shape, variant, fold), b in best.items():
        out[fold][task][f"{shape}/{variant}"] = {str(fold): b}
    out[0] = selections_fold0()
    return out


def selections_fold0():
    sel = defaultdict(dict)
    for d in (pv_maps.PV_MAPS, physics_maps.PHYSICS_MAPS, input_rule.INPUT_RULE):
        for task, entries in json.loads((d / "selection.json").read_text()).items():
            sel[task].update(entries)
    return sel


def fold_points(sel):
    sweep.latencies = lambda out, board: ({}, defaultdict(list))
    return sweep.deployed(sweep.points(DATA_SPLITS, "none", sel))


UNIFORM = ("grid_", "gridnode_")


def verdicts(pts, pick="val_rmse"):
    from scipy.stats import spearmanr
    kan, maps, B, M = battery_maps.FAMILIES["kan"], battery_maps.FAMILIES["maps"], physics_maps.BUDGETS, physics_maps.MARGIN
    by = defaultdict(list)
    for p in pts:
        by[p["task"]].append(p)
    def win(task, b, fams, keep=lambda p: True):
        P = [p for p in by[task] if p["family"] in kan or (p["family"] in fams and keep(p))]
        k = sweep.best_under(P, kan, b, pick)
        c = sweep.best_under([p for p in P if p["family"] in fams], fams, b, pick)
        return bool(k and c and k["rmse"] <= M * c["rmse"]), (c["rmse"] / k["rmse"] if k and c else None)
    uniform = lambda p: p["family"] != "grid" or p["shape"].startswith(UNIFORM)
    count = lambda tasks, fams, keep=lambda p: True: sum(win(t, b, fams, keep)[0] for t in tasks for b in B)
    v = {"pv C1": count(physics.PV_TASKS, ("grid",), uniform) >= 4,
         "pv C2": count(physics.PV_TASKS, maps, uniform) >= 4,
         "physics C1": count(physics.MAP_TASKS, ("grid",)) >= 6,
         "physics C2": count(physics.MAP_TASKS, maps) >= 6,
         "physics C3": count(physics.PV_TASKS, ("grid",)) >= 4}
    won = {t: all(win(t, b, maps)[0] for b in B[1:]) for t in TASKS}
    v["rule H1"] = sum(won[t] == input_rule.PREDICT[t] for t in physics.RULE_TASKS) >= 6
    v["rule H2"] = sum(not win(t, B[0], ("cp",))[0] for t in physics.RULE_TASKS) >= 6
    adv = {d: win(f"msis_d{d}" if d < 7 else "msis7", B[2], maps)[1] for d in (3, 4, 5, 6, 7)}
    rho = round(float(spearmanr(list(adv), list(adv.values())).statistic), 12)
    v["rule H3"] = rho >= 0.9
    cells = {(t, b): win(t, b, maps) for t in TASKS for b in B}
    return v, cells, won, adv, rho


def frontier(folds_run=True):
    sels = selections() if folds_run else {0: selections_fold0()}
    res = {}
    for fold in (0,) + FOLDS:
        for pick in ("val_rmse", "rmse"):
            v, cells, won, adv, rho = verdicts(fold_points(sels[fold]), pick)
            res[(fold, pick)] = (v, cells, won, adv, rho)
            print(f"== fold {fold} [{pick}]: " + ", ".join(f"{k} {'✓' if x else '✗'}" for k, x in v.items())
                  + " | MSIS advantage " + ", ".join(f"d{d} {a:.2f}" for d, a in adv.items()) + f" rho {rho:.2f}")
    v0, c0, w0, _, _ = res[(0, "val_rmse")]
    many = [t for t in TASKS if sweep.n_inputs(t) >= 5]
    few = [t for t in TASKS if t not in many]
    out = {}
    for fold in FOLDS:
        v, cells, won, _, _ = res[(fold, "val_rmse")]
        f1 = sum(v[k] == v0[k] for k in v)
        f2 = sum(cells[c][0] == c0[c][0] for c in cells)
        f3a, f3b = sum(won[t] for t in many), sum(won[t] for t in few)
        out[fold] = {"F1": f1, "F2": f2, "F3_many": f3a, "F3_few": f3b,
                     "holds": {"F1": f1 >= 7, "F2": f2 >= 30, "F3": f3a >= 6 and f3b <= 1}}
        print(f"== fold {fold}: F1 {f1}/8 verdicts as fold 0 → {'CONFIRMED' if f1 >= 7 else 'FALSIFIED'} | "
              f"F2 {f2}/36 cells as fold 0 → {'CONFIRMED' if f2 >= 30 else 'FALSIFIED'} | "
              f"F3 ≥5 inputs {f3a}/{len(many)}, ≤4 inputs {f3b}/{len(few)} → {'CONFIRMED' if f3a >= 6 and f3b <= 1 else 'FALSIFIED'}")
    ratios = defaultdict(list)
    for fold in (0,) + FOLDS:
        for (t, b), (_, r) in res[(fold, "val_rmse")][1].items():
            if r:
                ratios[(t, b)].append(r)
    print("== best table map / KAN per cell over folds 0-2 (mean, min, max):")
    for t in TASKS:
        print(f"  {t:10s} " + "  ".join(f"{b // 1024}KB {sum(r) / len(r):.2f} [{min(r):.2f}, {max(r):.2f}]"
                                         for b in physics_maps.BUDGETS for r in [ratios[(t, b)]] if r))
    (DATA_SPLITS / "frontier.json").write_text(json.dumps(
        {"folds": {f"{f} {p}": {"verdicts": x[0], "won": x[2], "msis_adv": x[3], "rho": x[4],
                                "cells": {f"{t} {b}": list(c) for (t, b), c in x[1].items()}} for (f, p), x in res.items()},
         "criteria": out}, indent=1, default=str))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=10)
    sub.add_parser("frontier")
    a = p.parse_args()
    if a.cmd == "run":
        for task in TASKS:
            physics.load(task)
        curvs = {(t, f): physics_maps.curvature(t, f) for f in FOLDS for t in TASKS}
        tasks = grid(curvs)
        print(len(tasks), "tasks", flush=True)
        sweep.run_pool(a.out, a.workers, tasks, {
            "folds": FOLDS, "tasks": TASKS, "grid_n": GRID_N, "curvature": {f"{t} {f}": c for (t, f), c in curvs.items()},
            "kan": battery_maps.KAN, "cp": battery_maps.CP, "ga2m": battery_maps.GA2M, "top_mlp": {t: top_mlps(t) for t in TASKS},
            "steps": battery_maps.STEPS, "lrs": regression.NN_LRS, "seeds": (0, 1), "lut_variants": structure_rule.VARIANTS,
            "poly": regression.POLY_DEGREES, "grid_lams": battery_maps.GRID_LAMS, "budgets": physics_maps.BUDGETS,
            "margin": physics_maps.MARGIN}, fn=_task)
    else:
        frontier()

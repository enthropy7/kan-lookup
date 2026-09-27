import argparse
import json
import math
import multiprocessing
import statistics
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

from kantab import env, sparse, sweep
from kantab.data import physics
from kantab.experiments import battery_maps, data_splits, physics_maps

OUT = Path("results/raw/sparse_budgets")
GRIDS = OUT / "grids"   # surpluses, rebuilt by build; only the level vectors are kept in levels.json
TASKS = data_splits.TASKS
FOLDS = (0, 1, 2)
BASES = ("mod", "bound")
W = (0.5, 0.9, 0.99)
CUTS = (4096, 8192, 16384, 32768, 65536, 131072)
LAMS = (1e-6, 1e-4, 1e-2, 1.0)
ITERS = 1000
MAX_BYTES = CUTS[-1]
BUDGETS = (8192, 16384, 32768, 65536, 131072)
NEW_GRID = 8192
MANY = [t for t in TASKS if sweep.n_inputs(t) >= 5]
FEW = [t for t in TASKS if t not in MANY]


def _chunk(task, x):
    return physics.model(task, x)


class Simulator:
    """The physics model on the scaled box, standardized with the training targets, cached by node."""
    def __init__(self, task, fold, pool):
        x, y, train, _, _ = physics._parts(task, fold)
        self.task, self.pool = task, pool
        self.lo, self.hi = x[train].min(0), x[train].max(0)
        self.mean, self.std = y[train].mean(), y[train].std()
        self.cache = {}

    def __call__(self, u):
        keys = [k.tobytes() for k in np.round(u * 2 ** 24).astype(np.int64)]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            phys = self.lo + (u[todo] + 1) / 2 * (self.hi - self.lo)
            if len(todo) >= 64:
                parts = self.pool.map(partial(_chunk, self.task), np.array_split(phys, min(len(todo) // 16, 160)))
                values = np.concatenate(parts)
            else:
                values = physics.model(self.task, phys)
            for i, v in zip(todo, (values - self.mean) / self.std):
                self.cache[keys[i]] = v
        return np.array([self.cache[k] for k in keys])


def _name(task, fold, basis, kind):
    return f"{task}_f{fold}_{basis}_{kind}"


def build():
    """Every sparse grid, interpolated from the simulator: the largest regular one and the adaptive ones."""
    snapshot = env.snapshot()
    env.require_clean(snapshot)
    GRIDS.mkdir(parents=True, exist_ok=True)
    levels = {}
    with multiprocessing.Pool(physics.WORKERS) as pool:
        for task in TASKS:
            d = sweep.n_inputs(task)
            for fold in FOLDS:
                f = Simulator(task, fold, pool)
                for basis in BASES:
                    n = 1
                    while sparse.regular(basis, d, n + 1).n_bytes(False) <= MAX_BYTES:
                        n += 1
                    g = sparse.regular(basis, d, n)
                    g.interpolate(f)
                    np.savez(GRIDS / (_name(task, fold, basis, "reg") + ".npz"), levels=g.L, alpha=g.alpha, n=n)
                    for w in W:
                        g, sizes = sparse.adaptive(basis, d, f, MAX_BYTES, w)
                        name = _name(task, fold, basis, f"w{w}")
                        np.savez(GRIDS / (name + ".npz"), levels=g.L, alpha=g.alpha, sizes=np.array(sizes))
                        levels[name] = g.levels
                    print(task, fold, basis, "nodes", len(f.cache), flush=True)
    (OUT / "levels.json").write_text(json.dumps(levels))
    (OUT / "build.json").write_text(json.dumps({"env": snapshot, "cuts": CUTS, "w": W, "max_bytes": MAX_BYTES}, indent=1,
                                               default=str))


def _cuts(task, fold, basis):
    """(shape, grid, adaptive) for every model cut from the built grids."""
    z = np.load(GRIDS / (_name(task, fold, basis, "reg") + ".npz"))
    full = sparse.Grid(basis, [tuple(l) for l in z["levels"]])
    full.alpha = z["alpha"]
    b = sparse.BASE[basis]
    for n in range(1, int(z["n"]) + 1):
        keep = [k for k, l in enumerate(full.levels) if sum(a - b for a in l) <= n - 1]
        g = sparse.Grid(basis, [full.levels[k] for k in keep])
        g.alpha = np.concatenate([full.alpha[full.offsets[k]:full.offsets[k + 1]] for k in keep])
        yield f"sgrid_{basis}_reg_n{n}", g, False
    for w in W:
        z = np.load(GRIDS / (_name(task, fold, basis, f"w{w}") + ".npz"))
        full = sparse.Grid(basis, [tuple(l) for l in z["levels"]])
        full.alpha = z["alpha"]
        sizes = list(z["sizes"])
        seen = set()
        for cut in CUTS:
            k = max(i + 1 for i, s in enumerate(sizes) if s <= cut)
            if k not in seen:
                seen.add(k)
                yield f"sgrid_{basis}_w{w}_b{cut // 1024}", sparse.prefix(full, k), True


_data_cache = {}


def _data(task, fold):
    if (task, fold) not in _data_cache:
        _data_cache.clear()
        _data_cache[(task, fold)] = physics.split(task, fold)
    return _data_cache[(task, fold)]


def _rmse(g, alpha, x, y, std):
    return float(std * np.sqrt(np.mean((g(x, alpha) - y) ** 2)))


def _task(t):
    import torch
    torch.set_num_threads(1)
    kind, task, fold, arg = t
    data = _data(task, fold)
    row = {"task": task, "fold": fold, "seed": 0}
    if kind in ("grid", "gridnode"):
        return [{**r, "kind": kind} for r in physics_maps_task(kind, task, fold, arg, data, row)]
    (xtr, ytr), (xva, yva), (xte, yte), (_, std) = data
    shape, fit = arg
    g, adaptive = next((g, a) for s, g, a in _cuts(task, fold, shape.split("_")[1]) if s == shape)
    nb = g.n_bytes(adaptive)
    evals = lambda a: {"val": {"fp64": _rmse(g, a, xva, yva, std), "i16": _rmse(g, g_q(g, a), xva, yva, std)},
                       "test": {"fp64": _rmse(g, a, xte, yte, std), "i16": _rmse(g, g_q(g, a), xte, yte, std)}}
    if not fit:
        return [{**row, "shape": shape, "hyper": None, **evals(g.alpha), "n_bytes": nb}]
    from scipy.sparse.linalg import lsqr
    N = len(xtr)
    A = g.design(xtr, 1 / math.sqrt(N))
    out = []
    for lam in LAMS:
        a = lsqr(A, ytr / math.sqrt(N), damp=math.sqrt(lam), x0=g.alpha, iter_lim=ITERS,
                 atol=1e-10, btol=1e-10)[0]
        out.append({**row, "shape": shape + "_fit", "hyper": lam, **evals(a), "n_bytes": nb})
    return out


def g_q(g, alpha):
    q = sparse.Grid(g.basis, g.levels)
    q.alpha = alpha
    return q.quantized()


def physics_maps_task(kind, task, fold, counts, data, row):
    from kantab.maps import eval_grid, fit_grid
    if kind == "gridnode":
        r = eval_grid(data, counts, physics.node_values(task, counts, fold))
        return [{**row, "shape": f"gridvnode_n{'x'.join(map(str, counts))}", "hyper": None, "val": r["val"],
                 "test": r["test"], "n_bytes": r["n_bytes"]}]
    return [{**row, "shape": f"gridv_n{'x'.join(map(str, counts))}", "hyper": lam, "val": r["val"], "test": r["test"],
             "n_bytes": r["n_bytes"]} for lam, (r, _) in fit_grid(data, counts, battery_maps.GRID_LAMS).items()]


def grid8(task, fold):
    from kantab.maps import grid_counts
    curv = physics_maps.curvature(task, fold)
    return grid_counts(curv, int((NEW_GRID - 4 - 4 * len(curv)) // 2))


def jobs():
    out = []
    for task in TASKS:
        for fold in FOLDS:
            counts = grid8(task, fold)
            out += [("grid", task, fold, counts), ("gridnode", task, fold, counts)]
            for basis in BASES:
                for shape, g, _ in _cuts(task, fold, basis):
                    out.append(("sgrid", task, fold, (shape, False)))
                    if basis == "mod":
                        out.append(("sgrid", task, fold, (shape, True)))
    # the long least-squares fits first
    return sorted(out, key=lambda t: -(t[0] == "sgrid" and t[3][1]))


def selections():
    """Per fold, the new models chosen on validation RMSE, with their bytes."""
    groups = defaultdict(list)
    for line in (OUT / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        for variant in r["val"]:
            groups[(r["task"], r["shape"], variant, r["fold"], r["hyper"])].append(r)
    best = {}
    for (task, shape, variant, fold, hyper), rs in groups.items():
        v = statistics.mean(r["val"][variant] for r in rs)
        key = (task, shape, variant, fold)
        if key not in best or v < best[key]["val"]:
            best[key] = {"val": v, "test": statistics.mean(r["test"][variant] for r in rs), "hyper": hyper,
                         "bytes": rs[0]["n_bytes"]}
    return best


def points(fold):
    original = sweep.latencies
    sweep.latencies = lambda out, board: ({}, defaultdict(list))
    try:
        pts = sweep.deployed(sweep.points(data_splits.DATA_SPLITS, "none", data_splits.selections()[fold]))
    finally:
        sweep.latencies = original
    for (task, shape, variant, f), b in selections().items():
        if f != fold or variant == "fp64":
            continue
        pts.append({"task": task, "shape": shape, "variant": variant, "rmse": b["test"], "val_rmse": b["val"],
                    "bytes": b["bytes"], "family": "sgrid" if shape.startswith("sgrid") else "grid"})
    return pts


def frontier():
    kan, maps, margin = battery_maps.FAMILIES["kan"], battery_maps.FAMILIES["maps"] + ("sgrid",), physics_maps.MARGIN
    tested = (8192, 16384, 32768, 65536, 131072)
    result = {"folds": {}}
    for fold in FOLDS:
        pts = points(fold)
        by = defaultdict(list)
        for p in pts:
            by[p["task"]].append(p)
        def best(task, fams, b):
            return sweep.best_under([p for p in by[task] if p["family"] in fams], fams, b, "val_rmse")
        def win(task, b):
            k, c = best(task, kan, b), best(task, maps, b)
            return bool(k and c and k["rmse"] <= margin * c["rmse"])
        cells = {f"{t} {b}": {"win": win(t, b), "kan": best(t, kan, b), "map": best(t, maps, b),
                              "sgrid": best(t, ("sgrid",), b)} for t in TASKS for b in (4096,) + tested}
        s1 = sum(win(t, 32768) and win(t, 131072) for t in MANY)
        bound = {b: sum(all(win(t, c) for c in tested if c >= b) for t in MANY) for b in (8192, 16384, 32768)}
        result["folds"][fold] = {"S1": s1, "bound": bound, "cells": cells}
        print(f"== split {fold}: S1 {s1}/8 models win 32 and 128 KB with sparse grids among the maps | from b on: "
              + ", ".join(f"{b // 1024} KB {n}/8" for b, n in bound.items()))
        for t in TASKS:
            row = []
            for b in (4096,) + tested:
                c = cells[f"{t} {b}"]
                k, m, s = c["kan"], c["map"], c["sgrid"]
                row.append(f"{b // 1024}K {m['rmse'] / k['rmse']:.2f}{'*' if c['win'] else ''}"
                           + (f" sg {s['rmse'] / k['rmse']:.2f}" if s and k else "")
                           + (" [sg best]" if m and m["family"] == "sgrid" else "") if k and m else f"{b // 1024}K —")
            print(f"   {t:10} " + " | ".join(row))
    s1 = all(result["folds"][f]["S1"] >= 7 for f in FOLDS)
    b1 = next((b for b in (8192, 16384, 32768) if all(result["folds"][f]["bound"][b] >= 7 for f in FOLDS)), None)
    result["S1"], result["B1"] = s1, b1
    print(f"== S1 {'CONFIRMED' if s1 else 'FALSIFIED'} | B1: bound {b1 // 1024 if b1 else None} KB"
          f" ({'as expected' if b1 == 16384 else 'expected 16 KB'})")
    (OUT / "frontier.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def specs():
    """bench_sgrid specs of the sparse grids chosen at 32 and 128 KB on split 0: NAME:d:basis:levels in hex."""
    pts = [p for p in points(0) if p["family"] == "sgrid"]
    out = []
    for task in TASKS:
        for b in (32768, 131072):
            p = sweep.best_under([q for q in pts if q["task"] == task], ("sgrid",), b, "val_rmse")
            if not p:
                continue
            shape = p["shape"].removesuffix("_fit")
            basis = shape.split("_")[1]
            g = next(g for s, g, _ in _cuts(task, 0, basis) if s == shape)
            assert g.L.max() < 16
            hexes = "".join("0123456789abcdef"[a] for l in g.levels for a in l)
            out.append(f"{task}.{shape}:{g.d}:{basis}:{hexes}")
    (OUT / "sgrid_specs.txt").write_text("\n".join(out) + "\n")
    return out


def all_specs():
    """bench_sgrid specs of every sparse grid of split 0, one file per task (a command line has a length limit)."""
    (OUT / "specs").mkdir(exist_ok=True)
    for task in TASKS:
        lines = []
        for basis in BASES:
            for shape, g, _ in _cuts(task, 0, basis):
                assert g.L.max() < 16
                lines.append(f"{task}.{shape}:{g.d}:{basis}:" + "".join("0123456789abcdef"[a] for l in g.levels for a in l))
        (OUT / "specs" / f"{task}.txt").write_text("\n".join(lines) + "\n")


def tradeoff():
    """L1 and the descriptive trade between error and latency on the Cortex-A53, split 0."""
    from kantab.experiments import vendor_kernels
    lat = {}
    for line in (OUT / "latency_orangepi_sgrid_all.txt").read_text().splitlines():
        f = line.split()
        lat[f[1].split(":", 1)[1]] = int(f[3])
    kpts = vendor_kernels.load_points(boards=("orangepi",))[(0, "orangepi")]
    spts = [{**p, "lat_ns": lat[f"{p['task']}.{p['shape'].removesuffix('_fit')}"]}
            for p in points(0) if p["family"] == "sgrid"]
    out, l1 = {}, 0
    for t in MANY:
        kan = [p for p in kpts[t] if p["fam"] == "kan" and p["lat_ns"]]
        sg = [p for p in spts if p["task"] == t]
        row = {}
        for b in (32768, 131072):
            k = min((p for p in kan if p["bytes"] <= b), key=lambda p: p["val_rmse"])
            within = [p for p in sg if p["lat_ns"] <= k["lat_ns"]]
            best = min(within, key=lambda p: p["rmse"]) if within else None
            reach = [p for p in sg if p["rmse"] <= k["rmse"]]
            fast = min(reach, key=lambda p: p["lat_ns"]) if reach else None
            row[b] = {"kan": k, "sgrid_at_kan_latency": best, "fastest_sgrid_at_kan_error": fast}
            print(f"{t:10} {b // 1024:>3}K: KAN {k['rmse']:.3g} in {k['lat_ns'] / 1000:.2f} µs | best sparse grid as fast: "
                  + (f"{best['rmse']:.3g} ({best['rmse'] / k['rmse']:.1f}× the error)" if best else "none")
                  + " | fastest sparse grid as accurate: "
                  + (f"{fast['lat_ns'] / 1000:.1f} µs ({fast['lat_ns'] / k['lat_ns']:.0f}× the time)" if fast else "none"))
        won = row[32768]["sgrid_at_kan_latency"] is None or row[32768]["sgrid_at_kan_latency"]["rmse"] > row[32768]["kan"]["rmse"]
        l1 += won
        out[t] = row
    print(f"== L1: {l1}/8 models where no sparse grid as fast as the 32 KB KAN table is as accurate → "
          f"{'CONFIRMED' if l1 >= 6 else 'FALSIFIED'}")
    (OUT / "tradeoff.json").write_text(json.dumps({"L1": l1 >= 6, "count": l1, "models": out}, indent=1, default=str))
    return out


HZ = {"esp32": 240e6, "stm32": 84e6}
BOARD_FILES = {"orangepi": ["latency_orangepi_sgrid_all.txt", "latency_orangepi_sgrid_tuned.txt"],
               "luckfox": ["latency_luckfox_sgrid.txt"], "esp32": ["esp32_sgrid_output.txt"],
               "stm32": ["stm32_sgrid_output.txt"]}


def sgrid_latency(board, ram=False):
    """ns per call of each sparse grid on a board, the faster of the kernels and runs."""
    out = {}
    for name in BOARD_FILES[board.removesuffix("ram")]:
        path = OUT / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            f = line.split()
            if len(f) < 4 or f[0] != "spec" or "n/a" in f or "error" in f:
                continue
            key = f[1].split(":", 1)[1]
            if board.removesuffix("ram") in HZ:
                if (f[2] == "ram") != ram:
                    continue
                ns = int(f[4 if ram else 3]) * 1e9 / HZ[board.removesuffix("ram")]
            else:
                ns = int(f[3])
            out[key] = min(out.get(key, ns), ns)
    return out


def _kan_points(board):
    from kantab.experiments import cortex_m4, vendor_kernels
    if board == "orangepi_scalar":
        lat = {l.split()[1]: int(l.split()[3]) for l in (OUT / "latency_orangepi_kan_scalar.txt").read_text().splitlines()}
        pts = vendor_kernels.load_points(boards=(board,), lat={board: (lat, defaultdict(list))}, vendor=False)
    elif board.startswith("stm32"):
        pts = vendor_kernels.load_points(boards=(board,), lat={board: sweep.latencies(cortex_m4.OUT, board)})
    else:
        pts = vendor_kernels.load_points(boards=(board,))
    return pts[(0, board)]


def _structure(p, d):
    """Table entries a KAN table reads per call."""
    w = int(p["shape"].split("_w")[1].split("_")[0])
    edges = d * w + w + (w * w if p["shape"].startswith("kan2") else 0)
    return edges * (2 if p["variant"].startswith("lin") else 1)


def boards():
    """T1 and T2 on every board: the fastest sparse grid at the KAN table's error, and the most accurate one at its
    latency, against the KAN tables chosen at 16 and 32 KB on split 0 (128 KB descriptive)."""
    sg_pts = [p for p in points(0) if p["family"] == "sgrid"]
    levels = {}
    for line in "\n".join((OUT / "specs" / f"{t}.txt").read_text() for t in TASKS).split():
        name, d, _, hexes = line.split(":")
        levels[name] = (int(d), hexes)
    result = {}
    for board, sg_board in (("orangepi", "orangepi"), ("orangepi_scalar", "orangepi"), ("luckfox", "luckfox"),
                            ("esp32", "esp32"), ("stm32", "stm32"), ("esp32ram", "esp32ram"), ("stm32ram", "stm32ram")):
        lat = sgrid_latency(sg_board.removesuffix("ram"), ram=sg_board.endswith("ram"))
        if not lat:
            continue
        kpts = _kan_points(board)
        rows = {}
        for b in (16384, 32768, 131072):
            t1 = t2 = t1v = t2v = 0
            for t in MANY:
                kan = [p for p in kpts[t] if p["fam"] == "kan" and p["lat_ns"]]
                k = min((p for p in kan if p["bytes"] <= b), key=lambda p: p["val_rmse"])
                sg = [{**p, "lat_ns": lat[f"{t}.{p['shape'].removesuffix('_fit')}"]} for p in sg_pts
                      if p["task"] == t and f"{t}.{p['shape'].removesuffix('_fit')}" in lat]
                reach = [p for p in sg if p["rmse"] <= k["rmse"]]
                fast = min(reach, key=lambda p: p["lat_ns"]) if reach else None
                within = [p for p in sg if p["lat_ns"] <= k["lat_ns"]]
                best = min(within, key=lambda p: p["rmse"]) if within else None
                time = fast["lat_ns"] / k["lat_ns"] if fast else None
                t1 += time is None or time >= 3
                t2 += best is None or best["rmse"] > k["rmse"]
                # the same with the sparse grid chosen on validation error, as a deployment would, instead of test
                reach_v = [p for p in sg if p["val_rmse"] <= k["val_rmse"]]
                fast_v = min(reach_v, key=lambda p: p["lat_ns"]) if reach_v else None
                best_v = min(within, key=lambda p: p["val_rmse"]) if within else None
                time_v = fast_v["lat_ns"] / k["lat_ns"] if fast_v else None
                t1v += time_v is None or time_v >= 3
                t2v += best_v is None or best_v["rmse"] > k["rmse"]
                d, hexes = levels[f"{t}.{fast['shape'].removesuffix('_fit')}"] if fast else (sweep.n_inputs(t), "")
                lv = len(hexes) // d
                rows.setdefault(t, {})[b] = {
                    "kan": {"shape": k["shape"], "variant": k["variant"], "rmse": k["rmse"], "lat_ns": k["lat_ns"],
                            "reads": _structure(k, sweep.n_inputs(t))},
                    "fastest_at_kan_error": fast and {"shape": fast["shape"], "rmse": fast["rmse"], "lat_ns": fast["lat_ns"],
                                                      "level_vectors": lv, "active": sum(c > "1" for c in hexes)},
                    "time": time, "error_at_kan_latency": best and best["rmse"] / k["rmse"],
                    "time_val": time_v, "error_at_kan_latency_val": best_v and best_v["rmse"] / k["rmse"]}
            rows[f"T1 {b}"], rows[f"T2 {b}"] = t1, t2
            rows[f"T1 val {b}"], rows[f"T2 val {b}"] = t1v, t2v
            times = [rows[t][b]["time"] for t in MANY]
            shown = [f"{x:.0f}" if x else "—" for x in times]
            shown_v = [f"{x:.0f}" if x else "—" for x in (rows[t][b]["time_val"] for t in MANY)]
            print(f"{board:15} {b // 1024:>3}K: T1 {t1}/8 T2 {t2}/8 | time at KAN error: {' '.join(shown)} | "
                  f"chosen on validation: T1 {t1v}/8 T2 {t2v}/8, {' '.join(shown_v)}")
        result[board] = rows
    registered = [b for b in ("orangepi", "orangepi_scalar", "luckfox", "esp32", "stm32") if b in result]
    t1 = all(result[b][f"T1 {s}"] >= 6 for b in registered for s in (16384, 32768))
    t2 = all(result[b][f"T2 {s}"] >= 6 for b in registered for s in (16384, 32768))
    print(f"== T1 {'CONFIRMED' if t1 else 'FALSIFIED'} | T2 {'CONFIRMED' if t2 else 'FALSIFIED'} (boards: {', '.join(registered)})")
    held_v = all(result[b][f"{k} val {s}"] >= 6 for b in registered for s in (16384, 32768) for k in ("T1", "T2"))
    print(f"   with the sparse grids chosen on validation error (descriptive): {'both hold' if held_v else 'not both'}")
    (OUT / "boards.json").write_text(json.dumps({"T1": t1, "T2": t2, "boards": result}, indent=1, default=str))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("run").add_argument("--workers", type=int, default=10)
    sub.add_parser("frontier")
    sub.add_parser("specs")
    sub.add_parser("all_specs")
    sub.add_parser("tradeoff")
    sub.add_parser("boards")
    a = p.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "run":
        sweep.run_pool(OUT, a.workers, jobs(), {"tasks": TASKS, "folds": FOLDS, "bases": BASES, "w": W, "cuts": CUTS,
                                                 "lams": LAMS, "iters": ITERS, "new_grid": NEW_GRID,
                                                 "grid_lams": battery_maps.GRID_LAMS}, fn=_task)
    elif a.cmd == "specs":
        print(len(specs()), "specs")
    elif a.cmd == "all_specs":
        all_specs()
    elif a.cmd == "tradeoff":
        tradeoff()
    elif a.cmd == "boards":
        boards()
    else:
        frontier()

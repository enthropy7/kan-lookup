import argparse
import json
from pathlib import Path


from kantab import sweep, training
from kantab.data import physics
from kantab.experiments import regression, structure_rule, battery_maps

GRID_N = {"pv_mp5": (3, 4, 5, 6, 7, 8, 9), "pv_chain6": (3, 4, 5, 6)}
BUDGETS = (4096, 32768, 131072)
MARGIN = 0.8
PV_MAPS = Path("results/raw/pv_maps")
_cache = {}


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
    if kind == "nn":
        r, _ = training.train_nn(data, shape, hyper, seed, steps=battery_maps.STEPS, lut_variants=structure_rule.VARIANTS)
        return [{**row, "hyper": hyper, "steps": battery_maps.STEPS, **r}]
    if kind == "gbdt":
        r, _ = training.fit_gbdt(data, *sweep.gbdt_shape(shape), hyper)
        return [{**row, "hyper": hyper, **r}]
    if kind == "poly":
        return [{**row, "hyper": a, **r} for a, (r, _) in
                training.fit_poly(data, int(shape[4:]), regression.POLY_ALPHAS).items()]
    from kantab.maps import eval_grid, fit_grid
    n = int(shape.split("_n")[1])
    if kind == "gridnode":
        r = eval_grid(data, n, physics.node_values(task, n))
        return [{**row, "hyper": None, "val": r["val"], "test": r["test"], "n_bytes": r["n_bytes"]}]
    return [{**row, "hyper": lam, "val": r["val"], "test": r["test"], "n_bytes": r["n_bytes"]}
            for lam, (r, _) in fit_grid(data, n, battery_maps.GRID_LAMS).items()]


def grid():
    out = []
    for task in physics.PV_TASKS:
        out += [("grid", task, f"grid_n{n}", None, 0) for n in sorted(GRID_N[task], reverse=True)]
        out += [("gridnode", task, f"gridnode_n{n}", None, 0) for n in GRID_N[task]]
        out += [("nn", task, s, lr, seed) for s in battery_maps.KAN + battery_maps.MLP + battery_maps.CP + battery_maps.GA2M
                for lr in regression.NN_LRS for seed in (0, 1)]
        out += [("gbdt", task, s, lr, 0) for s in regression.GBDT_SHAPES for lr in regression.GBDT_LRS]
        out += [("poly", task, f"poly{d}", None, 0) for d in regression.POLY_DEGREES]
    return out


def forests(dest):
    from kantab.export import forest_bytes, x_bytes
    sel = json.loads((PV_MAPS / "selection.json").read_text())
    for task in physics.PV_TASKS:
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
    pts = [p for p in sweep.points(PV_MAPS, board)
           if not p["family"].endswith("_fp32") and not (p["shape"].startswith("grid") and p["variant"] == "fp64")]
    result = {}
    f = lambda p: (f"{p['shape']}/{p['variant']} {p['rmse']:.4g} {p['bytes']}B"
                   + (f" {p['lat_ns'] / 1000:.2f}us" if p.get("lat_ns") else "")) if p else "—"
    for pick in ("val_rmse", "rmse"):
        rows = {}
        for name, fams in (("C1", battery_maps.FAMILIES["grid"]), ("C2", battery_maps.FAMILIES["maps"])):
            cells = []
            for task in physics.PV_TASKS:
                P = [p for p in pts if p["task"] == task]
                for b in BUDGETS:
                    k = sweep.best_under(P, battery_maps.FAMILIES["kan"], b, pick)
                    c = sweep.best_under(P, fams, b, pick)
                    win = bool(k and c and k["rmse"] <= MARGIN * c["rmse"])
                    cells.append(win)
                    print(f"  [{pick}] {name} {task} ≤{b}B: KAN {f(k)} | {'/'.join(fams)} {f(c)} → {'WIN' if win else 'no'}")
            rows[name] = {"cells": cells, "holds": sum(cells) >= 4}
            print(f"== [{pick}] {name}: {sum(cells)}/6 cells → {'CONFIRMED' if sum(cells) >= 4 else 'FALSIFIED'}")
        result[pick] = rows
    for task in physics.PV_TASKS:
        timed = [p for p in pts if p["task"] == task and p.get("lat_ns")]
        r = sweep.judge(timed, task, ("kan",), {"all": battery_maps.FAMILIES["all"]})["all"]
        print(f"== C3 (descriptive) {task}: best alternative {f(r['best_baseline'])}; best KAN {r['kan_best_rmse']:.4g}"
              f" → {'WIN' if r['win'] else 'no'}")
        result[f"C3_{task}"] = r
    (PV_MAPS / f"frontier_{board}.json").write_text(json.dumps({"points": pts, "criteria": result}, indent=1, default=str))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=10)
    sub.add_parser("forests").add_argument("dest", type=Path)
    sub.add_parser("frontier").add_argument("--board", default="orangepi")
    a = p.parse_args()
    if a.cmd == "run":
        sweep.run_pool(a.out, a.workers, grid(), {
            "tasks": physics.PV_TASKS, "n": physics.N, "grid_n": GRID_N, "grid_lams": battery_maps.GRID_LAMS,
            "mlp": battery_maps.MLP, "kan": battery_maps.KAN, "cp": battery_maps.CP, "ga2m": battery_maps.GA2M,
            "steps": battery_maps.STEPS, "lrs": regression.NN_LRS, "seeds": (0, 1), "lut_variants": structure_rule.VARIANTS,
            "gbdt": regression.GBDT_SHAPES, "poly": regression.POLY_DEGREES, "budgets": BUDGETS, "margin": MARGIN},
            fn=_task)
    elif a.cmd == "forests":
        forests(a.dest)
    else:
        frontier(a.board)

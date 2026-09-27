import argparse
import json
from pathlib import Path

import numpy as np

from kantab import sweep, training
from kantab.data import battery
from kantab.experiments import regression, structure_rule

TASK = "lg_soc"
MLP = tuple(f"mlp{L}_w{W}{a}" for a in ("", "_silu", "_tanh") for L in (1, 2, 3) for W in (16, 32, 64, 128))
KAN = tuple(f"kan1_w{W}_g{G}" for W in (4, 8, 16, 32) for G in (10, 20)) + ("kan2_w8_g5", "kan2_w16_g5")
CP = tuple(f"cp{R}_q{Q}" for R in (1, 2, 4, 8, 16) for Q in (16, 32, 64))
GA2M = tuple(f"ga2m_q{Q1}_{Q2}" for Q1 in (16, 32, 64) for Q2 in (8, 16, 32))
GRID_N, GRID_LAMS = (3, 4, 5, 6, 7, 8, 10), (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
STEPS, POLY_ROWS = 30_000, 100_000
BUDGETS = (2560, 16384, 71680)
MARGIN = 0.8
BATTERY_MAPS = Path("results/raw/battery_maps")
_cache = {}


def _data():
    if "d" not in _cache:
        _cache["d"] = battery.split(TASK)
    return _cache["d"]


def _task(t):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kind, shape, hyper, seed = t
    data = _data()
    row = {"task": TASK, "fold": 0, "shape": shape, "seed": seed}
    if kind == "nn":
        r, _ = training.train_nn(data, shape, hyper, seed, steps=STEPS, lut_variants=structure_rule.VARIANTS)
        return [{**row, "hyper": hyper, "steps": STEPS, **r}]
    if kind == "gbdt":
        r, _ = training.fit_gbdt(data, *sweep.gbdt_shape(shape), hyper)
        return [{**row, "hyper": hyper, **r}]
    if kind == "poly":
        (xtr, ytr), *rest = data
        idx = np.random.default_rng(40).choice(len(xtr), POLY_ROWS, replace=False)
        sub = ((xtr[idx], ytr[idx]), *rest)
        return [{**row, "hyper": a, **r} for a, (r, _) in
                training.fit_poly(sub, int(shape[4:]), regression.POLY_ALPHAS).items()]
    from kantab.maps import fit_grid
    n = int(shape.split("_n")[1])
    rows = []
    for lam, (r, _) in fit_grid(data, n, GRID_LAMS).items():
        rows.append({**row, "hyper": lam, "val": {"i16": r["val"]["i16"], "fp64": r["val"]["fp64"]},
                     "test": {"i16": r["test"]["i16"], "fp64": r["test"]["fp64"]}, "n_bytes": r["n_bytes"]})
    return rows


def grid():
    out = [("grid", f"grid_n{n}", None, 0) for n in sorted(GRID_N, reverse=True)]
    out += [("nn", s, lr, seed) for s in KAN + MLP + CP + GA2M for lr in regression.NN_LRS for seed in (0, 1)]
    out += [("gbdt", s, lr, 0) for s in regression.GBDT_SHAPES for lr in regression.GBDT_LRS]
    out += [("poly", f"poly{d}", None, 0) for d in regression.POLY_DEGREES]
    return out


def forests(dest):
    from kantab.export import forest_bytes, x_bytes
    sel = json.loads((BATTERY_MAPS / "selection.json").read_text())
    data = _data()
    (dest / TASK).mkdir(parents=True, exist_ok=True)
    xte = data[2][0][:512]
    (dest / TASK / "x_f0.bin").write_bytes(x_bytes(xte))
    for shape in regression.GBDT_SHAPES:
        lr = sel[TASK][f"{shape}/gbdt"]["0"]["hyper"]
        _, m = training.fit_gbdt(data, *sweep.gbdt_shape(shape), lr)
        (dest / TASK / f"{shape}_f0.forest").write_bytes(forest_bytes(m))
        (dest / TASK / f"{shape}_f0.expect").write_bytes(m.predict(xte).astype("<f4").tobytes())


FAMILIES = {"kan": ("kan",), "grid": ("grid",), "maps": ("grid", "cp", "ga2m"),
            "all": ("mlp_int8", "mlp_fp32", "gbdt", "poly", "grid", "cp", "ga2m")}


def frontier(board):
    pts = [p for p in sweep.points(BATTERY_MAPS, board) if p["family"] not in ("kan_fp32", "cp_fp32", "ga2m_fp32")]
    for p in pts:
        if p["shape"].startswith("grid") and p["variant"] == "fp64":
            p["family"] = "grid_fp64"
    result = {}
    for pick in ("val_rmse", "rmse"):
        rows = {}
        for name, fams in (("C1", FAMILIES["grid"]), ("C2", FAMILIES["maps"])):
            wins = []
            for b in BUDGETS:
                k, c = sweep.best_under(pts, FAMILIES["kan"], b, pick), sweep.best_under(pts, fams, b, pick)
                win = bool(k and c and k["rmse"] <= MARGIN * c["rmse"])
                wins.append(win)
                f = lambda p: f"{p['shape']}/{p['variant']} {100 * p['rmse']:.3f}% {p['bytes']}B" + (
                    f" {p['lat_ns'] / 1000:.2f}us" if p.get("lat_ns") else "") if p else "—"
                print(f"  [{pick}] {name} budget {b}B: KAN {f(k)} | {'/'.join(fams)} {f(c)} → {'WIN' if win else 'no'}")
            rows[name] = {"wins": wins, "holds": sum(wins) >= 2}
            print(f"== [{pick}] {name}: {sum(wins)}/3 budgets → {'CONFIRMED' if sum(wins) >= 2 else 'FALSIFIED'}")
        result[pick] = rows
    timed = [p for p in pts if p.get("lat_ns")]
    base = [p for p in timed if p["family"] in FAMILIES["all"]]
    kan = [p for p in timed if p["family"] == "kan"]
    target = min(base, key=lambda p: p["rmse"])
    print(f"== C3 (descriptive): best alternative {target['shape']}/{target['variant']} {100 * target['rmse']:.3f}%"
          f" {target['bytes']}B {target['lat_ns'] / 1000:.2f}us; best KAN "
          + (lambda p: f"{p['shape']}/{p['variant']} {100 * p['rmse']:.3f}% {p['bytes']}B {p['lat_ns'] / 1000:.2f}us")(
              min(kan, key=lambda p: p["rmse"])))
    result["C3"] = sweep.judge(timed, TASK, ("kan",), {"all": FAMILIES["all"]})["all"]
    print(f"== C3 (descriptive, regression test against every alternative): {'WIN' if result['C3']['win'] else 'no'}")
    (BATTERY_MAPS / f"frontier_{board}.json").write_text(json.dumps({"points": pts, "criteria": result}, indent=1, default=str))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=12)
    sub.add_parser("forests").add_argument("dest", type=Path)
    sub.add_parser("frontier").add_argument("--board", default="orangepi")
    a = p.parse_args()
    if a.cmd == "run":
        sweep.run_pool(a.out, a.workers, grid(), {
            "task": TASK, "mlp": MLP, "kan": KAN, "cp": CP, "ga2m": GA2M, "grid_n": GRID_N, "grid_lams": GRID_LAMS,
            "steps": STEPS, "lrs": regression.NN_LRS, "seeds": (0, 1), "lut_variants": structure_rule.VARIANTS,
            "gbdt": regression.GBDT_SHAPES, "poly": regression.POLY_DEGREES, "poly_rows": POLY_ROWS,
            "budgets": BUDGETS, "margin": MARGIN}, fn=_task)
    elif a.cmd == "forests":
        forests(a.dest)
    else:
        frontier(a.board)

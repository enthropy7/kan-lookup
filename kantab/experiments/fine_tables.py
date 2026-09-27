import argparse
import json
import math
import statistics
from pathlib import Path

from kantab import sweep, training
from kantab.data import formulas
from kantab.experiments import regression, strong_mlps

FINE_VARIANTS = training.LUT_VARIANTS + tuple((f"lin{b}_i16", b, 16, True) for b in (7, 8, 10))
DESCRIPTIVE = ("lin10_i16",)
SEEDS_C, STEPS_C = (2, 3), 30_000
FINE_TABLES = Path("results/raw/fine_tables")


def _task(t):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    task, fold, shape, lr, seed = t
    data = regression._data(task, fold)
    r, _ = training.train_nn(data, shape, lr, seed, steps=STEPS_C, lut_variants=FINE_VARIANTS)
    return [{"task": task, "fold": fold, "shape": f"{shape}@30k-r2", "seed": seed, "hyper": lr, "steps": STEPS_C, **r}]


def grid(top):
    return [(task, fold, s, lr, seed) for task in formulas.TASKS for fold in range(formulas.K)
            for s in top[task] for lr in regression.NN_LRS for seed in SEEDS_C]


def specs():
    sel = json.loads((FINE_TABLES / "selection.json").read_text())
    s = {sweep.spec(task, *key.split("/")) for task in sel for key in sel[task]
         if key.split("/")[1] in ("lin7_i16", "lin8_i16", "lin10_i16")}
    (FINE_TABLES / "specs.txt").write_text("\n".join(sorted(s)) + "\n")
    print(len(s), "specs")


def pool_points(board, paths, drop=DESCRIPTIVE):
    pts = sweep.points(strong_mlps.REGRESSION, board, strong_mlps.merged(*paths), extra_latency=(FINE_TABLES,))
    return [p for p in pts if p["variant"] not in drop]


def frontier(board):
    base = (strong_mlps.REGRESSION, strong_mlps.SMOOTH_ACTIVATIONS)
    pools = {"C (primary): regression + A + C": pool_points(board, base + (FINE_TABLES,)),
             "R2 (strong MLPs): regression + A + B": pool_points(board, base + (strong_mlps.LONG_TRAINING,)),
             "C with lin10": pool_points(board, base + (FINE_TABLES,), drop=()),
             "all: regression + A + B + C": pool_points(board, base + (strong_mlps.LONG_TRAINING, FINE_TABLES))}
    result = {}
    for label, pts in pools.items():
        on_val = [{**p, "rmse": p["val_rmse"]} for p in pts]
        v = {t: sweep.judge(pts, t) for t in formulas.TASKS}
        vv = {t: sweep.judge(on_val, t) for t in formulas.TASKS}
        result[label] = {"verdicts": v, "verdicts_on_val": vv}
        print(f"== {label}")
        for name in regression.BASELINES:
            wins = [t for t in v if v[t][name]["win"]]
            wv = [t for t in vv if vv[t][name]["win"]]
            syn = sum(t in formulas.SYNTHETIC for t in wins)
            print(f"  {name}: {len(wins)}/10, synthetic {syn}/5, real {len(wins) - syn}/5 {wins}"
                  f" | on validation: synthetic {sum(t in formulas.SYNTHETIC for t in wv)}/5, {len(wv)}/10")
        for task in formulas.TASKS:
            r = v[task]["N2"]
            b, k = r["best_baseline"], r["kan_fastest"]
            kb = f"{k['shape']}/{k['variant']} {k['rmse']:.4g} {k['lat_ns'] / 1000:.2f}us" if k else "—"
            print(f"    {task:12s} T={r['target_rmse']:.4g} ({b['shape']}/{b['variant']} {b['lat_ns'] / 1000:.2f}us)"
                  f" kan fastest ≤T: {kb} | kan best {r['kan_best_rmse']:.4g} → {'WIN' if r['win'] else 'no'}")
    sel = json.loads((FINE_TABLES / "selection.json").read_text())
    t2 = {}
    for task in formulas.SYNTHETIC:
        best = {}
        for key, folds in sel[task].items():
            shape, variant = key.split("/")
            if shape.startswith("kan"):
                m = statistics.mean(v["test"] for v in folds.values())
                best[variant] = min(best.get(variant, math.inf), m)
        t2[task] = {v: best[v] for v in ("fp32", "lin6_i16", "lin7_i16", "lin8_i16", "lin10_i16")}
        t2[task]["ratio_lin8"] = best["lin8_i16"] / best["fp32"]
    ok = [t for t in t2 if t2[t]["ratio_lin8"] <= 1.5]
    print("== T2: best lin8_i16 / best fp32 KAN (C shapes):",
          {t: round(t2[t]["ratio_lin8"], 3) for t in t2}, f"→ {len(ok)}/5 ≤ 1.5")
    for t, r in t2.items():
        print(f"    {t:12s} " + " ".join(f"{v} {r[v]:.4g}" for v in ("fp32", "lin6_i16", "lin7_i16", "lin8_i16", "lin10_i16")))
    result["T2"] = t2
    (FINE_TABLES / f"frontier_{board}.json").write_text(json.dumps(result, indent=1, default=str))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=12)
    sub.add_parser("specs")
    sub.add_parser("frontier").add_argument("--board", default="orangepi")
    a = p.parse_args()
    if a.cmd == "run":
        top = strong_mlps.top_shapes([strong_mlps.load_selection(strong_mlps.REGRESSION),
                                         strong_mlps.load_selection(strong_mlps.SMOOTH_ACTIVATIONS)])
        sweep.run_pool(a.out, a.workers, grid(top), {
            "top": top, "steps": STEPS_C, "seeds": SEEDS_C, "lrs": regression.NN_LRS,
            "lut_variants": FINE_VARIANTS, "descriptive": DESCRIPTIVE}, fn=_task)
    elif a.cmd == "specs":
        specs()
    else:
        frontier(a.board)

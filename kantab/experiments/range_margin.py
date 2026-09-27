import argparse
import json
import statistics
from pathlib import Path

from kantab import sweep, training
from kantab.data import formulas
from kantab.experiments import regression, strong_mlps, fine_tables

MARGIN = 0.05
MARGIN_VARIANTS = tuple((f"lin{b}_i16_m5", b, 16, True, MARGIN) for b in (6, 7, 8))
VARIANTS = tuple(v for v in fine_tables.FINE_VARIANTS if v[0] != "lin10_i16") + MARGIN_VARIANTS
SEEDS_D = (4, 5)
RANGE_MARGIN = Path("results/raw/range_margin")


def _task(t):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    task, fold, shape, lr, seed = t
    data = regression._data(task, fold)
    r, _ = training.train_nn(data, shape, lr, seed, steps=fine_tables.STEPS_C, lut_variants=VARIANTS)
    return [{"task": task, "fold": fold, "shape": f"{shape}@30k-r3", "seed": seed, "hyper": lr,
             "steps": fine_tables.STEPS_C, **r}]


def kan_top():
    top = strong_mlps.top_shapes([strong_mlps.load_selection(strong_mlps.REGRESSION),
                                     strong_mlps.load_selection(strong_mlps.SMOOTH_ACTIVATIONS)])
    return {task: [s for s in shapes if s.startswith("kan")] for task, shapes in top.items()}


def grid(top):
    return [(task, fold, s, lr, seed) for task in formulas.TASKS for fold in range(formulas.K)
            for s in top[task] for lr in regression.NN_LRS for seed in SEEDS_D]


def _best(sel, task, variant):
    return min(statistics.mean(v["test"] for v in folds.values())
               for key, folds in sel[task].items() if key.split("/")[1] == variant)


def frontier(board):
    sel = json.loads((RANGE_MARGIN / "selection.json").read_text())
    print("== M1: best lin8_i16_m5 / best fp32 KAN (D runs), formulas; M2: lin8_i16_m5 / lin8_i16, all tasks")
    m1, m2 = {}, {}
    for task in formulas.TASKS:
        fp, l8, l8m = (_best(sel, task, v) for v in ("fp32", "lin8_i16", "lin8_i16_m5"))
        m2[task] = l8m / l8
        if task in formulas.SYNTHETIC:
            m1[task] = l8m / fp
        print(f"    {task:12s} fp32 {fp:.4g} lin8 {l8:.4g} lin8_m5 {l8m:.4g} | m5/fp32 {l8m / fp:.3f} | m5/no-margin {l8m / l8:.3f}")
    n1 = sum(r <= 1.2 for r in m1.values())
    n2 = sum(r <= 1.05 for r in m2.values())
    print(f"M1: {n1}/5 formulas with lin8_m5 ≤ 1.2 × fp32 | M2: {n2}/10 tasks with lin8_m5 ≤ 1.05 × lin8")
    base = (strong_mlps.REGRESSION, strong_mlps.SMOOTH_ACTIVATIONS, fine_tables.FINE_TABLES)
    pools = {"fine tables primary": base, "+ D (margin)": base + (RANGE_MARGIN,)}
    verdicts = {}
    for label, paths in pools.items():
        pts = fine_tables.pool_points(board, paths)
        v = {t: sweep.judge(pts, t) for t in formulas.TASKS}
        vv = {t: sweep.judge([{**p, "rmse": p["val_rmse"]} for p in pts], t) for t in formulas.TASKS}
        w, wv = [t for t in v if v[t]["N2"]["win"]], [t for t in vv if vv[t]["N2"]["win"]]
        print(f"== {label}: N2 formulas {sum(t in formulas.SYNTHETIC for t in w)}/5, real {sum(t in formulas.UCI for t in w)}/5"
              f" {w} | on validation formulas {sum(t in formulas.SYNTHETIC for t in wv)}/5")
        for task in formulas.SYNTHETIC:
            r = v[task]["N2"]
            k = r["kan_fastest"]
            print(f"    {task:12s} T={r['target_rmse']:.4g} kan fastest ≤T "
                  + (f"{k['shape']}/{k['variant']} {k['rmse']:.4g} {k['lat_ns'] / 1000:.2f}us" if k else "—")
                  + f" | kan best {r['kan_best_rmse']:.4g}")
        verdicts[label] = v
    (RANGE_MARGIN / f"frontier_{board}.json").write_text(json.dumps({"M1": m1, "M2": m2, "verdicts": verdicts}, indent=1, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=12)
    sub.add_parser("frontier").add_argument("--board", default="orangepi")
    a = p.parse_args()
    if a.cmd == "run":
        top = kan_top()
        sweep.run_pool(a.out, a.workers, grid(top), {"top": top, "steps": fine_tables.STEPS_C, "seeds": SEEDS_D,
                                                           "lrs": regression.NN_LRS, "lut_variants": VARIANTS}, fn=_task)
    else:
        frontier(a.board)

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch

from kantab import sweep, training
from kantab.data import formulas
from kantab.experiments import regression

SMOOTH_SHAPES = tuple(f"{s}_{act}" for act in ("silu", "tanh") for s in regression.MLP_SHAPES)
TOP_B, STEPS_B = 2, 30_000
REGRESSION, SMOOTH_ACTIVATIONS, LONG_TRAINING = Path("results/raw/regression"), Path("results/raw/strong_mlps/activations"), Path("results/raw/strong_mlps/long_training")


def _task(t):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    task, fold, shape, lr, seed, steps = t
    data = regression._data(task, fold)
    r, _ = training.train_nn(data, shape, lr, seed, steps=steps)
    name = shape if steps == training.STEPS else f"{shape}@{steps // 1000}k"
    return [{"task": task, "fold": fold, "shape": name, "seed": seed, "hyper": lr, "steps": steps, **r}]


def grid_a():
    return [(task, fold, s, lr, seed, training.STEPS)
            for task, fold in itertools.product(formulas.TASKS, range(formulas.K))
            for s in SMOOTH_SHAPES for lr in regression.NN_LRS for seed in regression.SEEDS]


def nn_family(shape):
    if shape.startswith("kan"):
        return "kan"
    return "mlp_" + training.parse(shape)[4]


def top_shapes(selections, k=TOP_B):
    out = {}
    for task in formulas.TASKS:
        scored = {}
        for sel in selections:
            for key, folds in sel[task].items():
                shape, variant = key.split("/")
                if variant == "fp32" and shape[:3] in ("mlp", "kan"):
                    scored[shape] = statistics.mean(v["val"] for v in folds.values())
        out[task] = []
        for fam in ("mlp_relu", "mlp_silu", "mlp_tanh", "kan"):
            ranked = sorted((v, s) for s, v in scored.items() if nn_family(s) == fam)
            out[task] += [s for _, s in ranked[:k]]
    return out


def grid_b(top):
    return [(task, fold, s, lr, seed, STEPS_B)
            for task in formulas.TASKS for fold in range(formulas.K)
            for s in top[task] for lr in regression.NN_LRS for seed in regression.SEEDS]


def load_selection(path):
    return json.loads((path / "selection.json").read_text())


def merged(*paths):
    out = {}
    for p in paths:
        for task, entries in load_selection(p).items():
            out.setdefault(task, {}).update(entries)
    return out


def frontier(board):
    pools = {"regression": merged(REGRESSION), "R1: + SiLU/tanh MLP": merged(REGRESSION, SMOOTH_ACTIVATIONS),
             "R2: + 5x budget": merged(REGRESSION, SMOOTH_ACTIVATIONS, LONG_TRAINING)}
    result = {}
    for label, sel in pools.items():
        pts = sweep.points(REGRESSION, board, sel)
        on_val = [{**p, "rmse": p["val_rmse"]} for p in pts]
        v = {t: sweep.judge(pts, t) for t in formulas.TASKS}
        vv = {t: sweep.judge(on_val, t) for t in formulas.TASKS}
        result[label] = {"verdicts": v, "verdicts_on_val": vv}
        print(f"== {label}")
        for name in regression.BASELINES:
            wins = [t for t in v if v[t][name]["win"]]
            wins_val = [t for t in vv if vv[t][name]["win"]]
            syn = sum(t in formulas.SYNTHETIC for t in wins)
            print(f"  {name}: {len(wins)}/10, synthetic {syn}/5, real {len(wins) - syn}/5 {wins}"
                  f" | on validation {len(wins_val)}/10")
        for task in formulas.TASKS:
            r = v[task]["N2"]
            b = r["best_baseline"]
            print(f"    {task:12s} T={r['target_rmse']:.4g} ({b['shape']}/{b['variant']}, {b['lat_ns'] / 1000:.2f}us)"
                  f" kan best {r['kan_best_rmse']:.4g} lat×{r['lat_ratio']:.3g} size×{r['size_ratio']:.3g}"
                  f" → {'WIN' if r['win'] else 'no'}")
    (LONG_TRAINING / f"frontier_{board}.json").write_text(json.dumps(result, indent=1, default=str))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("run-a", "run-b"):
        r = sub.add_parser(c)
        r.add_argument("--out", type=Path, required=True)
        r.add_argument("--workers", type=int, default=12)
    sub.add_parser("frontier").add_argument("--board", default="orangepi")
    a = p.parse_args()
    if a.cmd == "run-a":
        sweep.run_pool(a.out, a.workers, grid_a(), {"shapes": SMOOTH_SHAPES, "lrs": regression.NN_LRS,
                                          "seeds": regression.SEEDS, "steps": training.STEPS}, _task)
    elif a.cmd == "run-b":
        top = top_shapes([load_selection(REGRESSION), load_selection(SMOOTH_ACTIVATIONS)])
        sweep.run_pool(a.out, a.workers, grid_b(top), {"top": top, "top_b": TOP_B, "steps": STEPS_B,
                                             "lrs": regression.NN_LRS, "seeds": regression.SEEDS}, _task)
    else:
        frontier(a.board)

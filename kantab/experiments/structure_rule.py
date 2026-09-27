import argparse
import itertools
import json
from pathlib import Path

from kantab import sweep, training
from kantab.data import formulas
from kantab.experiments import regression, range_margin

TASKS = tuple(formulas.NEW_FORMULAS)
FOLDS = (0, 1, 2)
MLP = tuple(f"mlp{L}_w{W}_{act}" for act in ("silu", "tanh") for L in (2, 3) for W in (32, 64, 128))
KAN = ("kan1_w8_g10", "kan1_w16_g10", "kan1_w32_g10", "kan1_w32_g20", "kan2_w8_g5", "kan2_w16_g5")
ROT = ("rot2_kan1_w8_g10", "rot2_kan1_w16_g10", "rot2_kan1_w32_g10", "rot2_kan2_w8_g5")
STEPS, SEED = 30_000, 0
VARIANTS = tuple(v for v in range_margin.VARIANTS if v[0] in
                 ("lut6_i8", "lut6_i16", "lut8_i8", "lut8_i16", "lin4_i16", "lin6_i16_m5", "lin7_i16_m5", "lin8_i16_m5"))
STRUCTURE_RULE = Path("results/raw/structure_rule")


def _task(t):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kind, task, fold, shape, hyper = t
    data = regression._data(task, fold)
    row = {"task": task, "fold": fold, "shape": shape, "seed": SEED}
    if kind == "nn":
        r, _ = training.train_nn(data, shape, hyper, SEED, steps=STEPS, lut_variants=VARIANTS)
        return [{**row, "hyper": hyper, "steps": STEPS, **r}]
    if kind == "gbdt":
        r, _ = training.fit_gbdt(data, *sweep.gbdt_shape(shape), hyper)
        return [{**row, "hyper": hyper, **r}]
    return [{**row, "hyper": a, **r} for a, (r, _) in
            training.fit_poly(data, int(shape[4:]), regression.POLY_ALPHAS).items()]


def grid():
    out = []
    for task, fold in itertools.product(TASKS, FOLDS):
        out += [("nn", task, fold, s, lr) for s in MLP + KAN + ROT for lr in regression.NN_LRS]
        out += [("gbdt", task, fold, s, lr) for s in regression.GBDT_SHAPES for lr in regression.GBDT_LRS]
        out += [("poly", task, fold, f"poly{d}", None) for d in regression.POLY_DEGREES]
    return out


def forests(dest):
    from kantab.export import forest_bytes, x_bytes
    sel = json.loads((STRUCTURE_RULE / "selection.json").read_text())
    for task in TASKS:
        (dest / task).mkdir(parents=True, exist_ok=True)
        for fold in FOLDS:
            data = formulas.split(task, fold)
            xte = data[2][0][:512]
            (dest / task / f"x_f{fold}.bin").write_bytes(x_bytes(xte))
            for shape in regression.GBDT_SHAPES:
                lr = sel[task][f"{shape}/gbdt"][str(fold)]["hyper"]
                _, m = training.fit_gbdt(data, *sweep.gbdt_shape(shape), lr)
                (dest / task / f"{shape}_f{fold}.forest").write_bytes(forest_bytes(m))
                (dest / task / f"{shape}_f{fold}.expect").write_bytes(m.predict(xte).astype("<f4").tobytes())
        print(task, flush=True)


def frontier(board):
    pts = sweep.points(STRUCTURE_RULE, board)
    result = {}
    for mode, P in (("test", pts), ("val", [{**p, "rmse": p["val_rmse"]} for p in pts])):
        rows = {}
        for task in TASKS:
            cls = formulas.NEW_FORMULAS[task][1]
            kan = sweep.judge(P, task, ("kan",))["N2"]
            rot = sweep.judge(P, task, ("rotkan",))["N2"]
            both = sweep.judge(P, task, ("kan", "rotkan"))["N2"]
            rows[task] = {"class": cls, "kan_win": kan["win"], "rot_win": rot["win"], "either_win": both["win"],
                          "target": kan["target_rmse"], "best": kan["best_baseline"], "kan_best": kan["kan_best_rmse"],
                          "rot_best": rot["kan_best_rmse"], "kan_fastest": kan["kan_fastest"], "rot_fastest": rot["kan_fastest"]}
        hit = sum((r["class"] == "A") == r["kan_win"] for r in rows.values())
        rot_b = sum(r["rot_win"] for r in rows.values() if r["class"] == "B")
        either_a = sum(r["either_win"] for r in rows.values() if r["class"] == "A")
        kan_a = sum(r["kan_win"] for r in rows.values() if r["class"] == "A")
        kan_b = sum(r["kan_win"] for r in rows.values() if r["class"] == "B")
        print(f"== {mode}: H1 class predicts plain-KAN outcome {hit}/16 (A wins {kan_a}/8, B wins {kan_b}/8)"
              f" | H2 RotKAN wins on B {rot_b}/8 | KAN or RotKAN on A {either_a}/8")
        if mode == "test":
            for task, r in rows.items():
                b = r["best"]
                f = lambda p: f"{p['variant']} {p['rmse']:.3g} {p['lat_ns'] / 1000:.2f}us" if p else "—"
                print(f"    {task:10s} {r['class']} T={r['target']:.3g} ({b['shape']}/{b['variant']} {b['lat_ns'] / 1000:.1f}us)"
                      f" | KAN best {r['kan_best']:.3g}, ≤T: {f(r['kan_fastest'])} → {'WIN' if r['kan_win'] else 'no'}"
                      f" | Rot best {r['rot_best']:.3g}, ≤T: {f(r['rot_fastest'])} → {'WIN' if r['rot_win'] else 'no'}")
        result[mode] = {"rows": rows, "H1_hits": hit, "H2_rot_B": rot_b, "either_A": either_a}
    (STRUCTURE_RULE / f"frontier_{board}.json").write_text(json.dumps(result, indent=1, default=str))
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
            "tasks": TASKS, "classes": {t: formulas.NEW_FORMULAS[t][1] for t in TASKS}, "folds": FOLDS,
            "mlp": MLP, "kan": KAN, "rot": ROT, "steps": STEPS, "seed": SEED, "lrs": regression.NN_LRS,
            "lut_variants": VARIANTS, "gbdt": regression.GBDT_SHAPES, "poly": regression.POLY_DEGREES}, fn=_task)
    elif a.cmd == "forests":
        forests(a.dest)
    else:
        frontier(a.board)

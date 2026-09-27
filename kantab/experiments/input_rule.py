import argparse
import json
from pathlib import Path

from kantab import sweep, training
from kantab.data import physics
from kantab.experiments import regression, structure_rule, battery_maps, physics_maps

GRID_N = {"gasz7": (2, 3, 4), "ign5": (3, 4, 5, 6, 7, 8, 9), "igrf4": (3, 4, 5, 6, 8, 10, 12, 15),
          "msis_d3": (4, 6, 8, 12, 16, 20, 25, 32, 40), "msis_d4": (3, 4, 5, 6, 8, 10, 12, 15),
          "msis_d5": (3, 4, 5, 6, 7, 8, 9), "msis_d6": (3, 4, 5, 6)}
BUDGETS, MARGIN = physics_maps.BUDGETS, physics_maps.MARGIN
PREDICT = {task: len(physics.BOX[task]) >= 5 for task in physics.RULE_TASKS}
PHYSICS_MAPS, INPUT_RULE = physics_maps.PHYSICS_MAPS, Path("results/raw/input_rule")
MAPS = battery_maps.FAMILIES["maps"]


def forests(dest):
    from kantab.export import forest_bytes, x_bytes
    sel = json.loads((INPUT_RULE / "selection.json").read_text())
    for task in physics.RULE_TASKS:
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
    from scipy.stats import spearmanr
    pts = sweep.deployed(sweep.points(INPUT_RULE, board))
    msis7 = [p for p in sweep.deployed(sweep.points(PHYSICS_MAPS, board)) if p["task"] == "msis7"]
    f = lambda p: (f"{p['shape']}/{p['variant']} {p['rmse']:.4g} {p['bytes']}B"
                   + (f" {p['lat_ns'] / 1000:.2f}us" if p.get("lat_ns") else "")) if p else "—"
    kan = battery_maps.FAMILIES["kan"]
    result = {}
    def cell(P, b, fams, pick):
        k = sweep.best_under(P, kan, b, pick)
        c = sweep.best_under(P, fams, b, pick)
        return k, c, bool(k and c and k["rmse"] <= MARGIN * c["rmse"])
    for pick in ("val_rmse", "rmse"):
        rows = {"H1": {}, "H2": {}}
        for task in physics.RULE_TASKS:
            P = [p for p in pts if p["task"] == task]
            wins = []
            for b in BUDGETS[1:]:
                k, c, w = cell(P, b, MAPS, pick)
                wins.append(w)
                print(f"  [{pick}] H1 {task} ≤{b}B: KAN {f(k)} | maps {f(c)} → {'WIN' if w else 'no'}")
            won = all(wins)
            rows["H1"][task] = {"won": won, "predicted": PREDICT[task], "correct": won == PREDICT[task]}
            k, c, w = cell(P, BUDGETS[0], ("cp",), pick)
            rows["H2"][task] = {"kan_beats_cp_at_4k": w, "correct": not w}
            print(f"  [{pick}] {task}: KAN wins 32+128 KB {won}, predicted {PREDICT[task]}; 4 KB vs CP: KAN {f(k)} | CP {f(c)}")
        h1 = sum(v["correct"] for v in rows["H1"].values())
        h2 = sum(v["correct"] for v in rows["H2"].values())
        rows["H1"]["holds"], rows["H2"]["holds"] = h1 >= 6, h2 >= 6
        print(f"== [{pick}] H1 (rule predicts the 32/128 KB outcome): {h1}/7 → {'CONFIRMED' if h1 >= 6 else 'FALSIFIED'}")
        print(f"== [{pick}] H2 (no KAN win over CP at 4 KB): {h2}/7 → {'CONFIRMED' if h2 >= 6 else 'FALSIFIED'}")
        adv = {}
        for d, P in [(d, [p for p in pts if p["task"] == f"msis_d{d}"]) for d in (3, 4, 5, 6)] + [(7, msis7)]:
            k, c, _ = cell(P, BUDGETS[2], MAPS, pick)
            adv[d] = c["rmse"] / k["rmse"]
        rho = round(float(spearmanr(list(adv), list(adv.values())).statistic), 12)
        rows["H3"] = {"advantage": adv, "spearman": rho, "holds": rho >= 0.9}
        print(f"== [{pick}] H3 (MSIS: best map / KAN at 128 KB rises with d): "
              + ", ".join(f"d{d} {v:.2f}" for d, v in adv.items()) + f"; Spearman {rho:.2f} → {'CONFIRMED' if rho >= 0.9 else 'FALSIFIED'}")
        result[pick] = rows
    for task in physics.RULE_TASKS:
        timed = [p for p in pts if p["task"] == task and p.get("lat_ns")]
        r = sweep.judge(timed, task, ("kan",), {"all": battery_maps.FAMILIES["all"]})["all"]
        print(f"== C4 (descriptive) {task}: best alternative {f(r['best_baseline'])}; best KAN {r['kan_best_rmse']:.4g}"
              f" → {'WIN' if r['win'] else 'no'}")
        result[f"C4_{task}"] = r
    (INPUT_RULE / f"frontier_{board}.json").write_text(json.dumps({"points": pts, "criteria": result}, indent=1, default=str))
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
        for task in physics.RULE_TASKS:
            physics.load(task)
        curvs = {task: physics_maps.curvature(task) for task in physics.RULE_TASKS}
        counts = {task: list(physics_maps.aniso_counts(task, curvs[task])) for task in curvs}
        for task in curvs:
            print(task, "curvature", [f"{c:.3g}" for c in curvs[task]], "→", counts[task], flush=True)
    if a.cmd == "run":
        sweep.run_pool(a.out, a.workers, physics_maps.grid(curvs, physics.RULE_TASKS, GRID_N, ()), {
            "tasks": physics.RULE_TASKS, "n": physics.N, "grid_n": GRID_N, "predict": PREDICT,
            "curvature": curvs, "aniso_counts": counts, "fill": physics_maps.FILL, "curv_points": physics_maps.CURV_POINTS,
            "curv_step": physics_maps.CURV_STEP, "grid_lams": battery_maps.GRID_LAMS, "mlp": battery_maps.MLP,
            "kan": battery_maps.KAN, "cp": battery_maps.CP, "ga2m": battery_maps.GA2M, "steps": battery_maps.STEPS,
            "lrs": regression.NN_LRS, "seeds": (0, 1), "lut_variants": structure_rule.VARIANTS,
            "gbdt": regression.GBDT_SHAPES, "poly": regression.POLY_DEGREES, "budgets": BUDGETS, "margin": MARGIN},
            fn=physics_maps._task)
    elif a.cmd == "forests":
        forests(a.dest)
    elif a.cmd == "frontier":
        frontier(a.board)

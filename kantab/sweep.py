import argparse
import json
import math
import statistics
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from kantab import env, training
from kantab.data import battery, formulas, physics

BASELINES = {"N1": ("mlp_int8",), "N2": ("mlp_int8", "mlp_fp32", "gbdt", "poly")}


def run_pool(out, workers, tasks, config, fn):
    snapshot = env.snapshot()
    env.require_clean(snapshot)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps({**config, "env": snapshot}, indent=1, default=str))
    with Pool(workers) as pool, open(out / "runs.jsonl", "w") as f:
        for i, rows in enumerate(pool.imap(fn, tasks, chunksize=2)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            if i % 200 == 0:
                f.flush()
                print(i, "/", len(tasks), flush=True)


def select(out):
    groups = defaultdict(list)
    for line in (Path(out) / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        for variant in r["val"]:
            groups[(r["task"], r["shape"], variant, r["fold"], r["hyper"])].append(r)
    best = {}
    for (task, shape, variant, fold, hyper), rs in groups.items():
        v = statistics.mean(r["val"][variant] for r in rs)
        key = (task, shape, variant, fold)
        if key not in best or v < best[key]["val"]:
            best[key] = {"val": v, "test": statistics.mean(r["test"][variant] for r in rs), "hyper": hyper,
                         **({"nodes": rs[0]["nodes"]} if "nodes" in rs[0] else {})}
    summary = defaultdict(dict)
    for (task, shape, variant, fold), b in sorted(best.items()):
        summary[task].setdefault(f"{shape}/{variant}", {})[str(fold)] = b
    (Path(out) / "selection.json").write_text(json.dumps(summary, indent=1))
    return summary


def gbdt_shape(shape):
    n, leaves = shape[len("gbdt_n"):].split("_l")
    return int(n), int(leaves)


def family(shape, variant):
    if shape.startswith("mlp"):
        return "mlp_" + variant
    if shape.startswith("kan"):
        return "kan_fp32" if variant == "fp32" else "kan"
    if shape.startswith("rot"):
        return "rotkan_fp32" if variant == "fp32" else "rotkan"
    if training.is_map(shape):
        kind = training.map_params(shape)[0]
        return kind + "_fp32" if variant == "fp32" else kind
    if shape.startswith("grid"):
        return "grid"
    return shape[:4]


_inputs = {}


def n_inputs(task):
    if task not in _inputs:
        if task in battery.TASKS:
            _inputs[task] = battery.split(task)[0][0].shape[1]
        elif task in physics.TASKS:
            _inputs[task] = len(physics.BOX[task])
        else:
            _inputs[task] = formulas.load(task)[0].shape[1]
    return _inputs[task]


def grid_counts(shape):
    n = shape.split("_n")[1]
    return tuple(int(v) for v in n.split("x")) if shape.startswith("gridv") else int(n)


def grid_bytes(shape, d):
    from kantab.maps import grid_size
    return grid_size(grid_counts(shape), d)[1]


def spec(task, shape, variant):
    d = n_inputs(task)
    if shape.startswith("poly"):
        return f"poly:{d}:{shape[4:]}"
    if shape.startswith("gbdt") or family(shape, variant).endswith("_fp32") and not shape.startswith("mlp"):
        return None
    if shape.startswith("gridv"):
        return f"gridv:{d}:{shape.split('_n')[1]}"
    if shape.startswith("grid"):
        return f"grid:{d}:{shape.split('_n')[1]}"
    if training.is_map(shape):
        kind, a, b = training.map_params(shape)
        return f"{kind}:{d}:{a}:{b}"
    dims = "-".join(map(str, training.dims(shape, d)))
    if shape.startswith("mlp"):
        act = training.parse(shape)[4]
        return ("mlp:" if variant == "int8" else "f32:") + dims + ("" if act == "relu" else f":{act}")
    b, t = training.table_bits(variant)
    kind = "lin16" if variant.startswith("lin") else "kan" if t == 8 else "kan16"
    if shape.startswith("rot"):
        kind = "rot16"
    return f"{kind}:{dims}:{2 ** b}"


def specs(out):
    sel = json.loads((Path(out) / "selection.json").read_text())
    s = sorted({spec(task, *k.split("/")) for task in sel for k in sel[task]} - {None})
    (Path(out) / "specs.txt").write_text("\n".join(s) + "\n")
    print(len(s), "specs")


def latencies(out, board):
    lat = {}
    for line in (Path(out) / f"latency_{board}_shapes.txt").read_text().splitlines():
        f = line.split()
        lat[f[1]] = int(f[3])
    forest = defaultdict(list)
    forests = Path(out) / f"latency_{board}_forests.txt"
    for line in forests.read_text().splitlines() if forests.exists() else ():
        f = line.split()
        forest[(f[0], f[1])].append(int(f[4]))
    return lat, forest


def points(out, board, sel=None, extra_latency=()):
    from kantab.export import forest_size
    sel = sel or json.loads((Path(out) / "selection.json").read_text())
    lat, forest = latencies(out, board)
    for d in extra_latency:
        for line in (Path(d) / f"latency_{board}_shapes.txt").read_text().splitlines():
            f = line.split()
            lat[f[1]] = int(f[3])
    pts = []
    for task, entries in sel.items():
        d = n_inputs(task)
        for key, folds in entries.items():
            shape, variant = key.split("/")
            fam = family(shape, variant)
            p = {"task": task, "shape": shape, "variant": variant, "family": fam,
                 "rmse": statistics.mean(v["test"] for v in folds.values()),
                 "val_rmse": statistics.mean(v["val"] for v in folds.values())}
            if fam == "gbdt":
                p["bytes"] = forest_size(gbdt_shape(shape)[0], statistics.mean(v["nodes"] for v in folds.values()))
                ts = forest.get((task, shape))
                p["lat_ns"] = statistics.mean(ts) if ts and len(ts) == len(folds) else None
            elif fam == "grid":
                p["bytes"] = grid_bytes(shape, d)
                p["lat_ns"] = lat.get(spec(task, shape, variant))
            elif fam == "poly":
                p["bytes"] = 7 * len(training.monomials(d, int(shape[4:])))
                p["lat_ns"] = lat.get(spec(task, shape, variant))
            else:
                p["bytes"] = training.nn_bytes(shape, d, variant)
                s = spec(task, shape, variant)
                p["lat_ns"] = lat.get(s) if s else None
            pts.append(p)
    return pts


def deployed(pts):
    # the fp32 KAN / map models and the fp64 grid fits are references, not deployments
    return [p for p in pts if not p["family"].endswith("_fp32") and not (p["shape"].startswith("grid") and p["variant"] == "fp64")]


def best_under(pts, fams, budget, key):
    ok = [p for p in pts if p["family"] in fams and p["bytes"] <= budget]
    return min(ok, key=lambda p: p[key]) if ok else None


def judge(pts, task, candidate=("kan",), baselines=None):
    P = [p for p in pts if p["task"] == task and p["lat_ns"] is not None]
    kan = [p for p in P if p["family"] in candidate]
    res = {}
    for name, fams in (baselines or BASELINES).items():
        base = [p for p in P if p["family"] in fams]
        target = min(p["rmse"] for p in base)
        best_base = min(base, key=lambda p: p["rmse"])

        def cheapest(ps, key):
            ok = [p for p in ps if p["rmse"] <= target]
            return min(ok, key=lambda p: p[key]) if ok else None

        lb, sb = cheapest(base, "lat_ns"), cheapest(base, "bytes")
        lk, sk = cheapest(kan, "lat_ns"), cheapest(kan, "bytes")
        lat_ratio = lk["lat_ns"] / lb["lat_ns"] if lk else math.inf
        size_ratio = sk["bytes"] / sb["bytes"] if sk else math.inf
        half = [p for p in kan if p["rmse"] <= target / 2 and p["lat_ns"] <= lb["lat_ns"]]
        res[name] = {"target_rmse": target, "best_baseline": best_base, "base_fastest": lb, "base_smallest": sb,
                     "kan_fastest": lk, "kan_smallest": sk, "lat_ratio": lat_ratio, "size_ratio": size_ratio,
                     "a": lat_ratio <= 0.25 or size_ratio <= 0.25,
                     "b": min(half, key=lambda p: p["lat_ns"]) if half else None,
                     "kan_best_rmse": min((p["rmse"] for p in kan), default=math.inf)}
        res[name]["win"] = res[name]["a"] or res[name]["b"] is not None
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("select", "specs"))
    p.add_argument("out", type=Path)
    a = p.parse_args()
    {"select": select, "specs": specs}[a.cmd](a.out)

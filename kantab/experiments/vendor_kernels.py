import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from kantab import sweep
from kantab.experiments import data_splits, input_rule, physics_maps, pv_maps

VENDOR = Path("results/raw/vendor_kernels")
SOURCES = (pv_maps.PV_MAPS, physics_maps.PHYSICS_MAPS, input_rule.INPUT_RULE)
BOARDS = ("orangepi", "luckfox", "esp32ram", "esp32")
SPLITS = (0, 1, 2)
BASE = {"f32v": "f32", "f32m": "f32", "mlpv": "mlp"}


def convert(board_output):
    """ESP32 cycles at 240 MHz to latency files, keyed by our spec names: the faster of the vendor variants."""
    best = {"esp32": {}, "esp32ram": {}}
    ours = {"esp32": {}, "esp32ram": {}}
    for line in Path(board_output).read_text().splitlines():
        f = line.split()
        if len(f) < 4 or f[0] != "spec" or "n/a" in f:
            continue
        ram = f[2] == "ram"
        board = "esp32ram" if ram else "esp32"
        kind, rest = f[1].split(":", 1)
        ns = round(int(f[4 if ram else 3]) * 1000 / 240)
        if kind in BASE:
            spec = BASE[kind] + ":" + rest
            best[board][spec] = min(best[board].get(spec, ns), ns)
        else:
            ours[board][f[1]] = ns
    VENDOR.mkdir(parents=True, exist_ok=True)
    for board, lat in best.items():
        (VENDOR / f"latency_{board}_shapes.txt").write_text(
            "".join(f"spec {s} median_ns {v}\n" for s, v in sorted(lat.items())))
        (VENDOR / f"latency_{board}_ours.txt").write_text(
            "".join(f"spec {s} median_ns {v}\n" for s, v in sorted(ours[board].items())))


def _read(path):
    return {line.split()[1]: int(line.split()[3]) for line in Path(path).read_text().splitlines()} if path.exists() else {}


def vendor_latency(board):
    """The vendor MLP times; XNNPACK timed later with setup on every call and once replaces its first run."""
    lat = {}
    for f in sorted(VENDOR.glob(f"latency_{board}_xnnpack_*.txt")) or [VENDOR / f"latency_{board}_shapes.txt"]:
        for spec, ns in _read(f).items():
            lat[spec] = min(lat.get(spec, ns), ns)
    return lat


def tables(board, vendor=True):
    lat, forest = {}, defaultdict(list)
    for d in SOURCES:
        l, f = sweep.latencies(d, board)
        lat.update(l)
        for k, v in f.items():
            forest[k] += v
    if vendor:
        for spec, ns in vendor_latency(board).items():
            if spec in lat:
                lat[spec] = min(lat[spec], ns)
    return lat, forest


def family(p):
    if p["family"] == "grid":
        return "gridv" if p["shape"].startswith("gridv") else "grid"
    return {"kan": "kan", "cp": "cp", "ga2m": "ga2m", "mlp_fp32": "mlp", "mlp_int8": "mlp8", "poly": "poly",
            "gbdt": "gbdt"}.get(p["family"])


def deployable(p):
    # sweep.deployed also drops the fp32 MLP, which is a deployment here
    return p["family"] == "mlp_fp32" or p in sweep.deployed([p])


def load_points(vendor=True, boards=BOARDS, lat=None):
    lat = lat or {b: tables(b, vendor) for b in boards}
    sels = data_splits.selections()
    original = sweep.latencies
    sweep.latencies = lambda out, board: lat[board]
    pts = {}
    for split in SPLITS:
        for board in boards:
            pts[(split, board)] = defaultdict(list)
            for p in sweep.points(".", board, sels[split]):
                if deployable(p) and family(p):
                    pts[(split, board)][p["task"]].append({**p, "fam": family(p)})
    sweep.latencies = original
    return pts


def best_under(P, fam, budget):
    Q = [p for p in P if p["fam"] == fam and p["bytes"] <= budget]
    return min(Q, key=lambda p: p["val_rmse"]) if Q else None


def grid_target(P):
    return best_under(P, "gridv", physics_maps.BUDGETS[-1])


def fastest(P, fam, rmse):
    Q = [p for p in P if p["fam"] == fam and p["rmse"] <= rmse and p["lat_ns"]]
    return min(Q, key=lambda p: p["lat_ns"]) if Q else None


def speedups(pts, board):
    at_mlp, at_grid = defaultdict(list), {}
    for s in SPLITS:
        for t, P in pts[(s, board)].items():
            timed = [p for p in P if p["fam"] == "mlp" and p["lat_ns"]]
            if not timed:
                continue
            mlp = min(timed, key=lambda p: p["val_rmse"])
            k = fastest(P, "kan", mlp["rmse"])
            if k:
                at_mlp[t].append(mlp["lat_ns"] / k["lat_ns"])
            if s == 0:
                g = grid_target(P)
                m, k = fastest(P, "mlp", g["rmse"]), fastest(P, "kan", g["rmse"])
                if m and k:
                    at_grid[t] = m["lat_ns"] / k["lat_ns"]
    return at_mlp, at_grid


def frontier():
    pts = load_points(boards=("orangepi", "esp32ram"))
    out = {}
    for board in ("orangepi", "esp32ram"):
        at_mlp, at_grid = speedups(pts, board)
        full = {t: statistics.mean(v) for t, v in at_mlp.items() if len(v) == len(SPLITS)}
        v1 = sum(r >= 10 for r in full.values())
        v2 = sum(r > 1 for r in at_grid.values())
        out[board] = {"at_mlp": full, "at_grid": at_grid, "V1": v1 >= 8, "V2": v2 >= 6}
        print(f"== {board}: V1 {v1}/{len(full)} models with KAN ≥ 10× faster at the best MLP's error → "
              f"{'CONFIRMED' if v1 >= 8 else 'FALSIFIED'} | V2 {v2}/{len(at_grid)} models with KAN faster at the grid "
              f"target → {'CONFIRMED' if v2 >= 6 else 'FALSIFIED'}")
        print("   at the best MLP's error: " + ", ".join(f"{t} {r:.1f}×" for t, r in sorted(full.items())))
        print("   at the grid target:      " + ", ".join(f"{t} {r:.2f}×" for t, r in sorted(at_grid.items())))
    (VENDOR / "frontier.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("convert").add_argument("board_output", type=Path)
    sub.add_parser("frontier")
    a = p.parse_args()
    if a.cmd == "convert":
        convert(a.board_output)
    else:
        frontier()

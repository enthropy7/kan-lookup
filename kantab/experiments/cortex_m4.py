import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from kantab import sweep
from kantab.experiments import vendor_kernels

OUT = Path("results/raw/cortex_m4")
BOARDS = ("stm32", "stm32ram")
HZ = 84_000_000


def _specs(board_output):
    """(spec, ram) -> (cycles, checksum) from the benchmark's output."""
    out = {}
    for line in Path(board_output).read_text().splitlines():
        f = line.split()
        if len(f) < 4 or f[0] != "spec" or "n/a" in f or "error" in f:
            continue
        ram = f[2] == "ram"
        rest = f[3:] if ram else f[2:]
        out[(f[1], ram)] = (int(rest[1]), float(rest[rest.index("sum") + 1]))
    return out


def convert(board_output, host_output, vendor_outputs=()):
    """Cycles at 84 MHz to latency files keyed by our spec names; each MLP takes the faster of ours and CMSIS, whose
    times may also come from earlier runs with other CMSIS builds. A time counts only where the checksum equals the
    host build's."""
    host = {spec: v[1] for (spec, _), v in _specs(host_output).items()}
    ours, vendor = defaultdict(dict), defaultdict(dict)
    for n, output in enumerate([board_output, *vendor_outputs]):
        for (spec, ram), (cycles, got) in _specs(output).items():
            kind, rest = spec.split(":", 1)
            if n and kind not in vendor_kernels.BASE:
                continue
            if got != host[spec]:
                print(f"dropped {spec}{' ram' if ram else ''} of {output}: checksum {got} against {host[spec]} on the host")
                continue
            board = "stm32ram" if ram else "stm32"
            ns = round(cycles * 1e9 / HZ)
            if kind in vendor_kernels.BASE:
                s = vendor_kernels.BASE[kind] + ":" + rest
                vendor[board][s] = min(vendor[board].get(s, ns), ns)
            else:
                ours[board][spec] = ns
    for board in BOARDS:
        best = dict(ours[board])
        for s, ns in vendor[board].items():
            best[s] = min(best.get(s, ns), ns)
        for name, lat in (("shapes", best), ("ours", ours[board]), ("vendor", vendor[board])):
            (OUT / f"latency_{board}_{name}.txt").write_text(
                "".join(f"spec {s} median_ns {v}\n" for s, v in sorted(lat.items())))


def check(board_output, host_output):
    """Checksums against a host build of the same C code, by kernel kind."""
    host = {spec: v[1] for (spec, _), v in _specs(host_output).items()}
    worst = defaultdict(lambda: [0, 0.0, 0])
    for (spec, ram), (_, got) in _specs(board_output).items():
        want = host[spec]
        rel = abs(got - want) / max(abs(want), 1e-12)
        w = worst[spec.split(":")[0]]
        w[0] += 1
        w[1] = max(w[1], rel)
        w[2] += got != want
    for kind, (n, rel, diff) in sorted(worst.items()):
        print(f"{kind:6} {n:5} checks, {diff:4} differ in print, max relative difference {rel:.1e}")


def frontier():
    lat = {b: sweep.latencies(OUT, b) for b in BOARDS}
    pts = vendor_kernels.load_points(boards=BOARDS, lat=lat)
    out = {}
    for board in BOARDS:
        at_mlp, at_grid = vendor_kernels.speedups(pts, board)
        full = {t: statistics.mean(v) for t, v in at_mlp.items() if len(v) == len(vendor_kernels.SPLITS)}
        m1 = sum(r >= 10 for r in full.values())
        m2 = sum(r > 1 for r in at_grid.values())
        out[board] = {"at_mlp": full, "at_grid": at_grid, "M1": m1 >= 8, "M2": m2 >= 6}
        print(f"== {board}: M1 {m1}/{len(full)} models with KAN ≥ 10× faster at the best MLP's error → "
              f"{'CONFIRMED' if m1 >= 8 else 'FALSIFIED'} | M2 {m2}/{len(at_grid)} models with KAN faster at the "
              f"grid target → {'CONFIRMED' if m2 >= 6 else 'FALSIFIED'}")
        print("   at the best MLP's error: " + ", ".join(f"{t} {r:.1f}×" for t, r in sorted(full.items())))
        print("   at the grid target:      " + ", ".join(f"{t} {r:.2f}×" for t, r in sorted(at_grid.items())))
    (OUT / "frontier.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("convert", "check"):
        c = sub.add_parser(name)
        c.add_argument("board_output", type=Path)
        c.add_argument("host_output", type=Path)
    sub.choices["convert"].add_argument("--vendor-from", type=Path, nargs="*", default=[])
    sub.add_parser("frontier")
    a = p.parse_args()
    if a.cmd == "convert":
        convert(a.board_output, a.host_output, a.vendor_from)
    elif a.cmd == "check":
        check(a.board_output, a.host_output)
    else:
        frontier()

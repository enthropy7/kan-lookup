import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from kantab import sweep
from kantab.data import physics
from kantab.experiments import cortex_m4, data_splits, input_rule, physics_maps, pv_maps, sparse_budgets
from kantab.experiments.vendor_kernels import (best_under, deployable, family, fastest, grid_target, load_points,
                                               vendor_latency)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
SPLITS = (0, 1, 2)
BOARDS = ("orangepi", "luckfox", "esp32ram", "esp32")
M4 = ROOT / cortex_m4.OUT
HERE_RAW = ROOT / "results" / "raw"
BUDGETS = physics_maps.BUDGETS
B128 = BUDGETS[-1]
MANY = [t for t in physics.TASKS if sweep.n_inputs(t) >= 5]

SPARSE = ("sgrid", "sparse grid", "#6f4fc9", "X")
FAM = [("kan", "KAN table", "#2a78d6", "o"), ("gridv", "anisotropic grid", "#eb6834", "s"),
       ("grid", "uniform grid", "#1baf7a", "^"), ("cp", "CP decomposition", "#eda100", "D"),
       ("mlp", "fp32 MLP", "#e87ba4", "v")]
INK, MUTED, GRID = "#1b1b1b", "#5f5e5a", "#e3e2de"
NAMES = {"pv_mp5": "PV module power", "pv_chain6": "PV chain", "tyre4": "tyre Fy/Fz", "msis7": "air density (MSIS)",
         "flame5": "flame temperature", "gasz7": "gas Z (GERG-2008)", "ign5": "ignition delay",
         "igrf4": "geomagnetic |B|", "msis_d3": "MSIS, 3 inputs", "msis_d4": "MSIS, 4 inputs",
         "msis_d5": "MSIS, 5 inputs", "msis_d6": "MSIS, 6 inputs"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})


def smallest(P, fam, rmse):
    Q = [p for p in P if p["fam"] == fam and p["rmse"] <= rmse]
    return min(Q, key=lambda p: p["bytes"]) if Q else None


def spread(xs):
    return {"mean": statistics.mean(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def fmt(x):
    return f"{x:.2g}" if x < 10 else f"{x:.0f}"


SG = {}


def sparse(num):
    """Sparse grids: error over the KAN table's per cell, and the trade between error and latency on the A53."""
    raw = HERE_RAW / "sparse_budgets"
    fr = json.loads((raw / "frontier.json").read_text())["folds"]
    tr = json.loads((raw / "tradeoff.json").read_text())
    bd = json.loads((raw / "boards.json").read_text())
    ratio = lambda c: c["sgrid"]["rmse"] / c["kan"]["rmse"]
    span = lambda xs: f"{fmt(min(xs))}–{fmt(max(xs))}"
    out = {}
    for t in physics.TASKS:
        row = {f"e{b // 1024}": spread([ratio(fr[str(s)]["cells"][f"{t} {b}"]) for s in SPLITS]) for b in (32768, 131072)}
        if t in MANY:
            k = bd["boards"]["orangepi"][t]["32768"]
            row["err_at_lat"], row["time_at_err"] = k["error_at_kan_latency"], k["time"]
        out[t] = row
    many = [t for t in MANY]
    e = [out[t][f"e{b}"]["mean"] for t in many for b in (32, 128)]
    few = [out[t][f"e{b}"]["mean"] for t in physics.TASKS if t not in MANY for b in (32, 128)]
    # models with five or more inputs on which the sparse grid is the more accurate, per budget: range over splits
    better = {b: [sum(ratio(fr[str(s)]["cells"][f"{t} {b}"]) < 1 for t in many) for s in SPLITS] for b in (32768, 131072)}
    lat = sparse_budgets.sgrid_latency("orangepi")
    chosen = [lat[x.split(":")[0]] / 1000 for x in (raw / "sgrid_specs.txt").read_text().split()]
    kept = sorted({k.split()[0] for s in SPLITS for k, c in fr[str(s)]["cells"].items()
                   if k.split()[0] in MANY and k.split()[1] in ("32768", "131072") and c["win"]})
    saved = lambda t, fam: num["memory"][t][fam]["mean"] if num["memory"][t][fam] else 0.0
    # the latency addendum: every board, at the KAN tables chosen at 16 and 32 KB on split 0
    rows = lambda board: [bd["boards"][board][t][b] for t in many for b in ("16384", "32768")]
    boards = {}
    for board in ("orangepi", "orangepi_scalar", "luckfox", "esp32", "stm32"):
        r = rows(board)
        boards[board] = {"kan_us": span([x["kan"]["lat_ns"] / 1000 for x in r]),
                         "time": span([x["time"] for x in r if x["time"]]),
                         "error": span([x["error_at_kan_latency"] for x in r if x["error_at_kan_latency"]]),
                         "t1": min(bd["boards"][board][f"T1 {b}"] for b in ("16384", "32768")),
                         "t2": min(bd["boards"][board][f"T2 {b}"] for b in ("16384", "32768"))}
    reached = [x for x in rows("orangepi") if x["fastest_at_kan_error"]]
    others = [x["time"] for b in ("luckfox", "esp32", "stm32") for x in rows(b) if x["time"]]
    m4 = [x["fastest_at_kan_error"]["lat_ns"] / 1000 for x in rows("stm32") if x["fastest_at_kan_error"]]
    m4_kan = [x["kan"]["lat_ns"] / 1000 for x in rows("stm32")]
    num["sparse"] = out
    num["sparse_boards"] = boards
    num["text"].update({
        "sparse_many": span(e), "sparse_few": span(few),
        "sparse_err_lat": boards["orangepi"]["error"], "sparse_time": boards["orangepi"]["time"],
        "sparse_time_others": span(others), "sparse_time_scalar": boards["orangepi_scalar"]["time"],
        "sparse_l1_err": span([m[b]["sgrid_at_kan_latency"]["rmse"] / m[b]["kan"]["rmse"]
                               for m in tr["models"].values() for b in ("32768",)]),
        "sparse_better32": "–".join(sorted({str(min(better[32768])), str(max(better[32768]))})),
        "sparse_better128": "–".join(sorted({str(min(better[131072])), str(max(better[131072]))})),
        "sparse_lat": span(chosen),
        "sparse_s1": " / ".join(str(fr[str(s)]["S1"]) for s in SPLITS),
        "sparse_kept": ", ".join(kept),
        "sparse_memory": str(sum(saved(t, "kan") > saved(t, "sgrid") for t in many)),
        "sparse_kan_reads": span([x["kan"]["reads"] for x in rows("orangepi")]),
        "sparse_level_vectors": span([x["fastest_at_kan_error"]["level_vectors"] for x in reached]),
        "sparse_factors": span([x["fastest_at_kan_error"]["active"] for x in reached]),
        # absolute times on the Cortex-M4, and as a share of the 1 ms period of a 1 kHz loop
        "sparse_m4_us": span(m4), "sparse_m4_pct": span([t / 10 for t in m4]),
        "kan_m4_us": span(m4_kan), "kan_m4_pct": span([t / 10 for t in m4_kan]),
    })


def worst(num):
    """The worst-case study: largest test errors of the models chosen on split 0, in the output's units."""
    fr = json.loads((HERE_RAW / "worst_case" / "frontier.json").read_text())
    cells, counts = fr["cells"], fr["counts"]
    scale = {"gasz7": 1e5}
    short = lambda x: f"{x:.0f}" if x >= 100 else f"{x:#.2g}".rstrip(".")
    num["worst"] = {t: {g: {k: short(c[k] * scale.get(t, 1)) for k in ("rmse", "max")} if (c := cells[f"{t} 32768"][g])
                        else None for g in ("kan", "maps", "sgrid")} for t in physics.TASKS}
    n = lambda key, vs: len(counts[key][vs])
    tails = {}
    for key, cell in cells.items():
        for g, c in cell.items():
            if c:
                tails[(key.split()[0], g, c["shape"], c["variant"])] = c["max"] / c["rmse"]
    tail = lambda g: fmt(statistics.median(v for (_, h, _, _), v in tails.items() if h == g))
    at = lambda t, b, g: cells[f"{t} {b}"][g]
    num["text"].update({
        **{f"worst_{m}_{b // 1024}": str(n(f"{m} {b}", "vs_maps")) for m in ("max", "p999") for b in (16384, 32768, 131072)},
        "worst_sgrid_smaller_128": str(len(MANY) - n("max 131072", "vs_sgrid")),
        "worst_sgrid_rmse_128": str(sum(at(t, 131072, "sgrid")["rmse"] < at(t, 131072, "kan")["rmse"] for t in MANY)),
        "worst_tail_kan": tail("kan"), "worst_tail_maps": tail("maps"), "worst_tail_sgrid": tail("sgrid"),
        "worst_pv16_kan": short(at("pv_mp5", 16384, "kan")["max"]), "worst_pv16_map": short(at("pv_mp5", 16384, "maps")["max"]),
    })


def numbers(pts):
    frontier = json.loads((data_splits.DATA_SPLITS / "frontier.json").read_text())["folds"]
    out = {"cells": {}, "msis": {}, "wins": {}, "vs_mlp": {}, "vs_mlp_grid_target": {}, "memory": {}}
    for t in physics.TASKS:
        for b in BUDGETS:
            out["cells"][f"{t} {b}"] = spread([frontier[f"{s} val_rmse"]["cells"][f"{t} {b}"][1] for s in SPLITS])
    for s in SPLITS:
        f = frontier[f"{s} val_rmse"]
        out["msis"][s] = f["msis_adv"]
        won = f["won"]
        out["wins"][s] = {"many": sum(won[t] for t in MANY), "few": sum(won[t] for t in physics.TASKS if t not in MANY)}
    for board in BOARDS + cortex_m4.BOARDS:
        ratio, grid_ratio = defaultdict(list), {}
        for s in SPLITS:
            for t, P in pts[(s, board)].items():
                timed = [p for p in P if p["fam"] == "mlp" and p["lat_ns"]]
                if not timed:
                    continue
                mlp = min(timed, key=lambda p: p["val_rmse"])
                k = fastest(P, "kan", mlp["rmse"])
                if k:
                    ratio[t].append(mlp["lat_ns"] / k["lat_ns"])
                if s == 0:
                    g = grid_target(P)
                    m, k = fastest(P, "mlp", g["rmse"]), fastest(P, "kan", g["rmse"])
                    if m and k:
                        grid_ratio[t] = m["lat_ns"] / k["lat_ns"]
        out["vs_mlp"][board] = {t: spread(v) for t, v in ratio.items()}
        out["vs_mlp_grid_target"][board] = grid_ratio
    for t in physics.TASKS:
        row = defaultdict(list)
        for s in SPLITS:
            P = pts[(s, "orangepi")][t]
            g = grid_target(P)
            for fam in ("kan", "cp", "poly") + (("mlp",) if s == 0 else ()):
                p = smallest(P, fam, g["rmse"])
                row[fam].append(g["bytes"] / p["bytes"] if p else None)
            p = smallest([q for q in SG[s] if q["task"] == t], "sgrid", g["rmse"])
            row["sgrid"].append(g["bytes"] / p["bytes"] if p else None)
        out["memory"][t] = {fam: spread(v) if None not in v else None for fam, v in row.items()}
    won_cells = [out["cells"][f"{t} {b}"]["mean"] for t in MANY for b in BUDGETS[1:]
                 if all(frontier[f"{s} val_rmse"]["won"][t] for s in SPLITS)]
    kan_mem = [out["memory"][t]["kan"]["mean"] for t in MANY if out["memory"][t]["kan"]]
    full = lambda board: [v["mean"] for v in out["vs_mlp"][board].values() if v["n"] == len(SPLITS)]
    a53, esp = full("orangepi"), full("esp32ram")
    short = lambda board: {t: v["n"] for t, v in out["vs_mlp"][board].items() if v["n"] < len(SPLITS)}
    grid53 = list(out["vs_mlp_grid_target"]["orangepi"].values())
    vendor = {}
    for board in ("orangepi", "luckfox", "esp32ram"):
        ours = {}
        for d in (pv_maps.PV_MAPS, physics_maps.PHYSICS_MAPS, input_rule.INPUT_RULE):
            ours.update(sweep.latencies(d, board)[0])
        vend = vendor_latency(board)
        r = sorted(ours[k] / v for k, v in vend.items() if k in ours)
        vendor[board] = {"median": statistics.median(r), "max": max(r), "faster": sum(x > 1 for x in r), "n": len(r)}
    read = lambda name: {l.split()[1]: int(l.split()[3]) for l in (M4 / name).read_text().splitlines()}
    ours, vend = read("latency_stm32_ours.txt"), read("latency_stm32_vendor.txt")
    r = sorted(ours[k] / v for k, v in vend.items() if k in ours)
    vendor["stm32"] = {"median": statistics.median(r), "max": max(r), "faster": sum(x > 1 for x in r), "n": len(r)}
    out["vendor"] = vendor
    margin = json.loads((ROOT / "results/raw/range_margin/frontier_orangepi.json").read_text())["M1"]
    out["text"] = {
        "table_vs_fp32": f"{max(margin.values()):.2f}",
        "wins_many": " / ".join(str(out["wins"][s]["many"]) for s in SPLITS),
        "wins_few": " / ".join(str(out["wins"][s]["few"]) for s in SPLITS),
        "map_ratio": f"{fmt(min(won_cells))}–{fmt(max(won_cells))}",
        "memory_ratio": f"{fmt(min(kan_mem))}–{fmt(max(kan_mem))}",
        "mlp_a53": f"{fmt(min(a53))}–{fmt(max(a53))}",
        "mlp_esp32": f"{fmt(min(esp))}–{fmt(max(esp))}",
        "mlp_a7": f"{fmt(min(full('luckfox')))}–{fmt(max(full('luckfox')))}",
        "mlp_esp32_flash": f"{fmt(min(full('esp32')))}–{fmt(max(full('esp32')))}",
        "mlp_not_reached": {board: short(board) for board in ("orangepi", "esp32ram")},
        "mlp_grid_a53": f"{fmt(min(grid53))}–{fmt(max(grid53))}",
        "mlp_m4": f"{fmt(min(full('stm32')))}–{fmt(max(full('stm32')))}",
    }
    return out


FORMULAS = ("kan_toy2", "kan_toy4", "feyn_i6_2", "feyn_i29_16", "feyn_i40_1")


def studies():
    load = lambda p: json.loads((ROOT / "results/raw" / p).read_text())
    reg = load("regression/frontier_orangepi.json")["verdicts"]
    strong = load("strong_mlps/long_training/frontier_orangepi.json")
    fine = load("fine_tables/frontier_orangepi.json")
    margin = load("range_margin/frontier_orangepi.json")
    struct = load("structure_rule/frontier_orangepi.json")["test"]
    battery = load("battery_maps/frontier_orangepi.json")
    wins = lambda v, name, tasks: sum(v[t][name]["win"] for t in tasks)
    primary = next(k for k in fine if k.startswith("C (primary)"))
    out = {
        "regression_n1": f"{wins(reg, 'N1', reg)}/{len(reg)}",
        "regression_n2": f"{wins(reg, 'N2', reg)}/{len(reg)}",
        "regression_n2_formulas": f"{wins(reg, 'N2', FORMULAS)}/5",
        "strong_r1": f"{wins(strong['R1: + SiLU/tanh MLP']['verdicts'], 'N2', FORMULAS)}/5",
        "strong_r2": f"{wins(strong['R2: + 5x budget']['verdicts'], 'N2', FORMULAS)}/5",
        "fine_t1": f"{wins(fine[primary]['verdicts'], 'N2', FORMULAS)}/5",
        "fine_t2": f"{sum(fine['T2'][t]['ratio_lin8'] <= 1.5 for t in FORMULAS)}/5",
        "margin_m1": f"{sum(v <= 1.2 for v in margin['M1'].values())}/5",
        "margin_m2": f"{sum(v <= 1.05 for v in margin['M2'].values())}/{len(margin['M2'])}",
        "structure_h1": f"{struct['H1_hits']}/16",
        "structure_h2": f"{struct['H2_rot_B']}/8",
        "battery_c1": f"{sum(battery['criteria']['val_rmse']['C1']['wins'])}/3",
        "battery_c2": f"{sum(battery['criteria']['val_rmse']['C2']['wins'])}/3",
    }
    P = [{**p, "fam": family(p)} for p in battery["points"] if deployable(p)]
    out["battery"] = {b: {fam: (best_under(P, fam, b) or {}).get("rmse") for fam in ("kan", "grid", "cp", "ga2m", "mlp8", "poly")}
                      for b in (2560, 16384, 71680)}
    return out


def extra(pts, frontier):
    out = {"models": [{"id": t, "name": NAMES[t], "inputs": sweep.n_inputs(t)} for t in physics.TASKS]}
    out["rule"] = {t: {"predicted": input_rule.PREDICT.get(t), "won": [frontier[f"{s} val_rmse"]["won"][t] for s in SPLITS]}
                   for t in physics.TASKS}
    kinds = {}
    for t in physics.TASKS:
        for b in BUDGETS:
            row = defaultdict(list)
            for s in SPLITS:
                P = pts[(s, "orangepi")][t]
                k = best_under(P, "kan", b)
                for fam in ("grid", "gridv", "cp"):
                    c = best_under(P, fam, b)
                    if k and c:
                        row[fam].append(c["rmse"] / k["rmse"])
            kinds[f"{t} {b}"] = {fam: spread(v) for fam, v in row.items() if len(v) == len(SPLITS)}
    out["map_kinds"] = kinds
    P = pts[(0, "orangepi")]["gasz7"]
    poly = min((p for p in P if p["fam"] == "poly"), key=lambda p: p["val_rmse"])
    kan = min((p for p in P if p["fam"] == "kan"), key=lambda p: p["val_rmse"])
    out["gasz7"] = {"poly": {k: poly[k] for k in ("shape", "rmse", "lat_ns", "bytes")},
                    "kan": {k: kan[k] for k in ("shape", "variant", "rmse", "lat_ns", "bytes")}}
    return out


def tidy_log(ax, y=True):
    ax.xaxis.set_minor_formatter(NullFormatter())
    if y:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=160)
    plt.close(fig)


def fig_tables(name):
    import numpy as np
    from scipy.interpolate import BSpline
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    # an edge function of the trained form, w_b SiLU(x) + w_s sum_k c_k B_k(x), with G = 5 and fixed coefficients
    G, k = 5, 3
    knots = np.linspace(-1 - k * 2 / G, 1 + k * 2 / G, G + 2 * k + 1)
    coef = np.array([0.1, 0.6, -0.4, 0.3, 0.9, -0.2, 0.2, 0.5])
    phi = lambda x: 0.4 * x / (1 + np.exp(-x)) + BSpline(knots, coef, k)(x)
    x = np.linspace(-1, 1, 400)
    q = np.linspace(-1, 1, 16)
    a.plot(x, phi(x), color=MUTED, lw=1.2, label="edge function")
    a.plot(q, phi(q), ls="none", marker="o", ms=3.5, color=FAM[0][2], label="table entries")
    i = 6
    xs = q[i] + 0.4 * (q[i + 1] - q[i])
    a.plot(q[i:i + 2], phi(q[i:i + 2]), color=FAM[0][2], lw=1.5)
    ys = phi(q[i]) + 0.4 * (phi(q[i + 1]) - phi(q[i]))
    a.plot([xs, xs], [a.get_ylim()[0], ys], color=INK, lw=0.7, ls=":")
    a.plot([xs], [ys], marker="o", ms=4, color=INK)
    a.annotate("$T_k$", (q[i], phi(q[i])), xytext=(6, -10), textcoords="offset points", color=INK)
    a.annotate("$T_{k+1}$", (q[i + 1], phi(q[i + 1])), xytext=(-24, 6), textcoords="offset points", color=INK)
    a.set_xlabel("input $x$ (scaled)")
    a.set_ylabel(r"$\varphi(x)$")
    a.set_title("(a) one edge, 16 entries shown", fontsize=8.5, color=INK, loc="left")
    a.legend(loc="upper left", fontsize=6.5)
    d = np.arange(2, 9)
    for n, ls in ((8, "--"), (16, "-")):
        b.plot(d, 2.0 * n ** d / 1024, color=FAM[2][2], ls=ls, lw=1.5, label=f"grid, $n = {n}$")
    for w, ls in ((8, "--"), (16, "-")):
        b.plot(d, 2.0 * 128 * (d + 1) * w / 1024, color=FAM[0][2], ls=ls, lw=1.5, label=f"KAN table, $W = {w}$")
    for kb in (4, 32, 128):
        b.axhline(kb, color=MUTED, lw=0.6, ls=":")
        b.annotate(f"{kb} KB", (8.25, kb), va="center", fontsize=6.5, color=MUTED, annotation_clip=False)
    b.set_yscale("log")
    b.set_ylim(1, 1e6)
    b.set_xlim(1.8, 8.2)
    b.set_xlabel("number of inputs $d$")
    b.set_ylabel("memory (KB)")
    b.set_title("(b) int16 storage", fontsize=8.5, color=INK, loc="left")
    b.legend(loc="upper left", fontsize=6.5)
    save(fig, name)


def fig_frontier(pts, tasks, name):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    budgets = [2 ** (9 + i / 4) for i in range(37)]
    for ax, t in zip(axes.flat, tasks):
        ref = {s: grid_target(pts[(s, "orangepi")][t])["rmse"] for s in SPLITS}
        for fam, label, color, marker in FAM:
            xs, mean, lo, hi = [], [], [], []
            for b in budgets:
                ys = [p["rmse"] / ref[s] for s in SPLITS for p in [best_under(pts[(s, "orangepi")][t], fam, b)] if p]
                if len(ys) == len(SPLITS):
                    xs.append(b / 1024)
                    mean.append(statistics.mean(ys))
                    lo.append(min(ys))
                    hi.append(max(ys))
            if not xs:
                continue
            ax.fill_between(xs, lo, hi, step="post", color=color, alpha=0.18, lw=0)
            ax.step(xs, mean, where="post", color=color, lw=1.5, label=label if t == tasks[0] else None)
            ax.plot(xs[::6], mean[::6], ls="none", marker=marker, ms=4, color=color, mec="white", mew=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        tidy_log(ax)
        for b in BUDGETS:
            ax.axvline(b / 1024, color=MUTED, lw=0.6, ls=":")
        ax.set_title(f"{NAMES[t]}, {sweep.n_inputs(t)} inputs", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("memory (KB)")
        ax.set_ylabel("RMSE / best 128 KB grid")
        ax.set_xlim(0.5, 300)
    fig.legend(loc="outside upper center", ncol=5, fontsize=7)
    save(fig, name)


def fig_msis(num, name):
    fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)
    ds = [3, 4, 5, 6, 7]
    for s in SPLITS:
        ax.plot(ds, [num["msis"][s][str(d)] for d in ds], color=GRID, lw=1, marker="o", ms=3, mfc=MUTED, mec=MUTED)
    ax.plot(ds, [statistics.mean(num["msis"][s][str(d)] for s in SPLITS) for d in ds], color=FAM[0][2], lw=1.8,
            marker="o", ms=4, label="mean of 3 splits")
    ax.axhline(1.25, color=MUTED, lw=0.7, ls="--")
    ax.text(3, 1.28, "KAN 20 % better", fontsize=6.5, color=MUTED)
    ax.axhline(1.0, color=MUTED, lw=0.5)
    ax.set_xticks(ds)
    ax.set_xlabel("free inputs of MSIS")
    ax.set_ylabel("best table map / KAN, 128 KB")
    ax.legend(loc="upper left", fontsize=6.5)
    save(fig, name)


def pareto(P):
    out, best = [], float("inf")
    for p in sorted(P, key=lambda p: (p["lat_ns"], p["rmse"])):
        if p["rmse"] < best:
            out.append(p)
            best = p["rmse"]
    return out


def fig_latency(pts, task, name):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True, sharey=True)
    titles = {"orangepi": "Cortex-A53 (1.4 GHz, NEON)", "esp32ram": "ESP32 (240 MHz, RAM)",
              "stm32": "Cortex-M4 (84 MHz, flash)"}
    for ax, board in zip(axes, titles):
        P = pts[(0, board)][task]
        ref = grid_target(pts[(0, "orangepi")][task])["rmse"]
        for fam, label, color, marker in FAM:
            F = pareto([p for p in P if p["fam"] == fam and p["lat_ns"]])
            if F:
                xs, ys = [p["lat_ns"] / 1000 for p in F], [p["rmse"] / ref for p in F]
                ax.step(xs, ys, where="post", color=color, lw=1.5, label=label)
                ax.plot(xs, ys, ls="none", marker=marker, ms=4, color=color, mec="white", mew=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        tidy_log(ax)
        ax.set_title(titles[board], fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("latency per call (µs)")
    axes[0].set_ylabel("RMSE / best 128 KB grid")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), fontsize=7)
    save(fig, name)


def fig_sparse(pts, tasks, name):
    """Error against latency on the Cortex-A53, split 0, with the sparse grids on the faster of the two kernels."""
    lat = sparse_budgets.sgrid_latency("orangepi")
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.0, 2.6), constrained_layout=True)
    fams = [FAM[0], SPARSE, FAM[1], FAM[3], FAM[4]]
    for ax, task in zip(axes, tasks):
        P = pts[(0, "orangepi")][task]
        S = [{**p, "fam": "sgrid", "lat_ns": lat[f"{task}.{p['shape'].removesuffix('_fit')}"]}
             for p in SG[0] if p["task"] == task]
        ref = grid_target(P)["rmse"]
        for fam, label, color, marker in fams:
            F = pareto([p for p in P + S if p["fam"] == fam and p["lat_ns"]])
            if F:
                xs, ys = [p["lat_ns"] / 1000 for p in F], [p["rmse"] / ref for p in F]
                ax.step(xs, ys, where="post", color=color, lw=1.5, label=label)
                ax.plot(xs, ys, ls="none", marker=marker, ms=4, color=color, mec="white", mew=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        tidy_log(ax)
        ax.set_title(f"{NAMES[task]} ({sweep.n_inputs(task)} inputs)", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("latency per call on the Cortex-A53 (µs)")
    axes[0].set_ylabel("RMSE / best 128 KB grid")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), fontsize=7)
    save(fig, name)


def fig_memory(num, name):
    rows = [t for t in MANY + ["tyre4"] if num["memory"][t]["kan"]]
    marks = [("kan", "KAN table", FAM[0][2], "o"), ("sgrid", "sparse grid", SPARSE[2], SPARSE[3]),
             ("cp", "CP decomposition", FAM[3][2], "D"), ("poly", "polynomial", "#008300", "P"),
             ("mlp", "fp32 MLP (split 0)", FAM[4][2], "v")]
    fig, ax = plt.subplots(figsize=(3.4, 3.6), constrained_layout=True)
    for i, t in enumerate(rows):
        for k, (fam, label, color, marker) in enumerate(marks):
            r = num["memory"][t].get(fam)
            if r:
                y = i + (k - 2) * 0.13
                ax.plot([r["min"], r["max"]], [y, y], color=color, lw=1)
                ax.plot([r["mean"]], [y], marker=marker, ms=5, ls="none", color=color, mec="white", mew=0.5,
                        label=label if fam not in ax.get_legend_handles_labels()[1] else None)
    ax.axvline(1, color=MUTED, lw=0.7)
    ax.set_yticks(range(len(rows)), [f"{NAMES[t]} ({sweep.n_inputs(t)})" for t in rows])
    ax.set_xscale("log")
    tidy_log(ax, y=False)
    ax.set_xlabel("memory saved (×)")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7,
              handletextpad=0.2, columnspacing=1.0)
    ax.grid(axis="y", visible=False)
    save(fig, name)


def main(out):
    """The paper's numbers (numbers.json) and figures (figures/), recomputed from results/raw."""
    global FIGURES
    FIGURES = Path(out) / "figures"
    pts = load_points()
    pts.update(load_points(boards=cortex_m4.BOARDS, lat={b: sweep.latencies(M4, b) for b in cortex_m4.BOARDS}))
    SG.update({s: [{**p, "fam": "sgrid"} for p in sparse_budgets.points(s) if p["family"] == "sgrid"] for s in SPLITS})
    num = numbers(pts)
    sparse(num)
    worst(num)
    num["studies"] = studies()
    num.update(extra(pts, json.loads((data_splits.DATA_SPLITS / "frontier.json").read_text())["folds"]))
    (Path(out) / "numbers.json").write_text(json.dumps(num, indent=1, default=str))
    fig_tables("fig_tables")
    fig_frontier(pts, ["pv_mp5", "msis7", "ign5", "igrf4"], "fig_frontier")
    fig_msis(num, "fig_msis")
    fig_latency(pts, "msis7", "fig_latency")
    fig_sparse(pts, ["pv_mp5", "msis7"], "fig_sparse")
    fig_memory(num, "fig_memory")
    return num


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "paper")
    print(json.dumps(main(p.parse_args().out)["text"], indent=1))

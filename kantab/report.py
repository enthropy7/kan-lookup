"""Every study's verdicts and the paper's figures, recomputed from results/raw into one directory with an index.html:
    python -m kantab.report --out report"""
import argparse
import html
import os
import subprocess
import sys
from pathlib import Path

from kantab import figures

M4 = "results/raw/cortex_m4/"
FRONTIERS = [("Regression", "regression", ["results/raw/regression"]), ("Strong MLPs", "strong_mlps", []),
             ("Fine tables", "fine_tables", []), ("Range margin", "range_margin", []),
             ("Structure rule", "structure_rule", []), ("Battery maps", "battery_maps", []),
             ("PV maps", "pv_maps", []), ("Physics maps", "physics_maps", []), ("Input rule", "input_rule", [])]
# the documented command of each study, each in its own process because the studies share module state
STUDIES = [(f"{name} ({board})", [module, "frontier", *extra, "--board", board])
           for name, module, extra in FRONTIERS for board in ("orangepi", "luckfox")] + [
    ("Data splits", ["data_splits", "frontier"]),
    ("Vendor kernels", ["vendor_kernels", "frontier"]),
    ("Cortex-M4: board output to latencies", ["cortex_m4", "convert", M4 + "stm32_output_rerun.txt",
                                              M4 + "host_checksums.txt", "--vendor-from", M4 + "stm32_output.txt"]),
    ("Cortex-M4", ["cortex_m4", "frontier"]),
    ("Sparse grids and intermediate budgets", ["sparse_budgets", "frontier"]),
    ("Sparse grids: accuracy against latency", ["sparse_budgets", "tradeoff"]),
    ("Sparse grids: every board, tuned kernel, scalar control", ["sparse_budgets", "boards"]),
    ("Worst-case error of the chosen models", ["worst_case", "frontier"]),
]
FIGURES = [
    ("fig_tables", "One KAN edge and its table; memory of a grid and of KAN tables against the number of inputs."),
    ("fig_frontier", "Test RMSE against memory on four models, relative to the best 128 KB anisotropic grid."),
    ("fig_sparse", "Test RMSE against latency on the Cortex-A53 with sparse grids, on PV module power and MSIS."),
    ("fig_msis", "Best table map over KAN table at 128 KB as MSIS gets more free inputs."),
    ("fig_latency", "Test RMSE against latency on msis7, on the Cortex-A53, the ESP32 and the Cortex-M4."),
    ("fig_memory", "Memory saved against the best 128 KB anisotropic grid at its accuracy."),
]


def verdicts():
    for title, (module, *args) in STUDIES:
        print(f"== {title}", flush=True)
        r = subprocess.run([sys.executable, "-m", f"kantab.experiments.{module}", *args], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{title} failed:\n{r.stderr}")
        yield title, r.stdout


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    results = list(verdicts())
    (out / "verdicts.txt").write_text("".join(f"== {t}\n{text}\n" for t, text in results))
    print("== figures", flush=True)
    figures.main(out)
    page = ["<!doctype html><meta charset=utf-8><title>kan-lookup results</title>",
            "<style>body{font:15px/1.5 sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}"
            "pre{white-space:pre-wrap;font-size:13px}img{max-width:100%}</style>",
            "<h1>kan-lookup: results recomputed from results/raw</h1>"]
    page += [f"<figure><img src='figures/{name}.png'><figcaption>{html.escape(cap)}</figcaption></figure>"
             for name, cap in FIGURES]
    page += [f"<h2>{html.escape(t)}</h2><pre>{html.escape(text)}</pre>" for t, text in results]
    (out / "index.html").write_text("\n".join(page))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("report"))
    out = p.parse_args().out.resolve()
    # the studies read results/raw relative to the repository root
    os.chdir(figures.ROOT)
    main(out)

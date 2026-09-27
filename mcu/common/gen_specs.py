"""specs.c for the MCU benchmarks: every timing spec of the physics-map studies (results/raw/{pv_maps,physics_maps,
input_rule}/specs.txt). --vendor adds each MLP spec on the vendor kernels; --only-mlp keeps the MLPs alone;
--sgrid gives the sparse grids of results/raw/sparse_budgets/specs instead. An --out ending in .txt gets one spec
per line, for the STM32's run.py --specs."""
import argparse
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results" / "raw"
KINDS = ("lin16", "kan16", "mlp", "f32", "poly", "grid", "gridv", "cp", "ga2m")
VENDOR = {"f32": ("f32v", "f32m"), "mlp": ("mlpv",)}

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--vendor", action="store_true")
p.add_argument("--only-mlp", action="store_true")
p.add_argument("--sgrid", action="store_true", help="only the modified-basis sparse grids of split 0")
a = p.parse_args()
kind = lambda s: s.split(":")[0]
specs = sorted({s for e in ("pv_maps", "physics_maps", "input_rule") for s in (R / e / "specs.txt").read_text().split()
                if kind(s) in KINDS})
if a.only_mlp:
    specs = [s for s in specs if kind(s) in VENDOR]
if a.vendor:
    specs += [v + s[s.index(":"):] for s in specs if kind(s) in VENDOR for v in VENDOR[kind(s)]]
if a.sgrid:
    specs = ["sgrid:" + line for f in sorted((R / "sparse_budgets" / "specs").glob("*.txt"))
             for line in f.read_text().split() if ":mod:" in line]
if a.out.suffix == ".txt":
    a.out.write_text("".join(s + "\n" for s in specs))
else:
    a.out.write_text("const char *const SPECS[] = {\n" + ",\n".join(f'    "{s}"' for s in specs) +
                     f"\n}};\nconst int NSPECS = {len(specs)};\n")
print(len(specs), "specs")

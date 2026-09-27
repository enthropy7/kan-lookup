# kan-lookup

Kolmogorov–Arnold networks stored as lookup tables, for firmware.

Firmware often evaluates physics models through multidimensional lookup tables: module power in solar inverters,
tyre forces in vehicle control, gas compressibility in flow meters, air density on small satellites. A table over
d inputs with n points per axis holds n^d values, so it either stays coarse or stays at three or four inputs.

A Kolmogorov–Arnold network (KAN) is a sum of learned functions of one variable. Stored as tables, each function is
read with linear interpolation, so the network runs on table reads and additions. It stores one table per edge, so
at a fixed hidden width its memory grows linearly with the number of inputs. This repository trains such networks, exports them as tables, and compares them at equal
memory with the tables firmware uses, as well as with MLPs, gradient-boosted trees and polynomials.

## When a KAN table helps

For a smooth, noise-free physics model:

1. try a low-degree polynomial first;
2. if memory is the constraint and a call may take several times longer: a fitted, dimension-adaptive sparse grid;
3. if latency is the constraint: up to 4 inputs an anisotropic grid, 5 or more inputs and 16 KB or more a KAN
   table, about 4 KB a CP-decomposed table.

With 5 or more inputs, at 32 and 128 KB, a KAN table has 1.6–4.1× lower RMSE than grids, CP and GA2M tables of the
same size on at least 7 of 8 models, and reaches the error of the most accurate MLP 21–56× faster on a Cortex-A53
and 38–123× faster on a Cortex-M4, with the MLP on vendor kernels (XNNPACK, CMSIS) or ours, whichever is faster. Its
largest observed test error (over 12 000 test points) is also the smaller against grids, CP and GA2M tables, on all
8 such models at 32 and 128 KB. A fitted sparse grid is more accurate than the KAN table at equal memory on 4–6 of 8
such models at 128 KB and on 3 at 32 KB. Where it reaches the error of the KAN tables chosen at 16 and 32 KB, it
takes 11–24× as long on a Cortex-A53, 4.6–26× on a Cortex-A7, an ESP32 and a Cortex-M4, and 3.3–6.6× against KAN
tables on a scalar Cortex-A53 kernel. On noisy measured data a KAN table gives no advantage.

## How it works

1. A KAN is trained in PyTorch (B-spline edges, `kantab/models.py`).
2. Each edge function is sampled at 64–256 points over its input range, with the ranges of hidden layers widened by
   5 %, and stored as int16 with one scale per output node (`kantab/tables.py`, `kantab/export.py`).
3. A C kernel evaluates the tables with linear interpolation and an 8-bit fraction (`c/kernels.c`), with NEON
   paths on ARM.

The comparison tables are built the way engineers build them: grids fitted to data or filled from the simulator,
anisotropic grids with points placed by curvature, CP decompositions, GA2M (`kantab/maps.py`, `c/maps.c`) and
sparse grids (`kantab/sparse.py`).

## Layout

- `kantab/`: models, training, table export, baseline maps, model selection (`sweep.py`)
- `kantab/data/`: symbolic formulas and UCI regression sets, LG 18650HG2 battery data, physics simulators
  (pvlib, CommonRoad tyre model, NRL MSIS 2.1, Cantera, CoolProp GERG-2008, IGRF)
- `kantab/experiments/`: one module per study, each with `run` and `frontier` commands
- `kantab/figures.py`, `kantab/report.py`: the figures and the report built by `nix run`
- `c/`: inference kernels, the shape benchmark `bench.c`, its XNNPACK counterpart `bench_xnnpack.c`, `bench_sgrid.c`
  for sparse grids, the model evaluator `run.c`
- `mcu/`: the kernels in firmware for an ESP32 and an STM32F411 (Cortex-M4)
- `studies/`: one README per study, with its hypotheses and results
- `results/raw/`: raw runs, model selections, board latencies and verdicts
- `tests/`

## Studies

Each study was registered before it ran: `studies/<study>/README.md` holds the hypotheses, the criteria, the
deviations and the results. Every verdict and figure can be recomputed from `results/raw/` without retraining:

```
nix run      # prints the verdicts and opens the report
nix build    # the report in result/: index.html, verdicts.txt, figures/
```

## Running

Everything, including the datasets and simulators, is pinned in the Nix flake.

```
nix develop
pytest tests
python -m kantab.experiments.physics_maps run --out results/raw/physics_maps
python -m kantab.sweep select results/raw/physics_maps
python -m kantab.sweep specs results/raw/physics_maps
```

Latency is measured on the board with a static binary; save its output as
`results/raw/physics_maps/latency_<board>_shapes.txt`, then compare:

```
zig cc -O2 -ffp-contract=off -target aarch64-linux-musl -mcpu=cortex_a53 -o bench c/bench.c c/kernels.c c/maps.c c/baselines.c -lm
./bench $(cat specs.txt)
python -m kantab.experiments.physics_maps frontier --board <board>
```

Sparse grids are timed with `bench_sgrid` (`c/bench_sgrid.c c/maps.c`, `-O3`), one spec file of
`results/raw/sparse_budgets/specs/` per model; `mcu/README.md` covers the microcontrollers.

A run records the commit and refuses to start on uncommitted changes. The commits recorded in `results/raw` are
those of the development history, which is not published.

## License

Apache 2.0. The datasets are downloaded by Nix under their own licenses.

# Sparse grids and intermediate budgets

Registered on 2026-09-28, before any sparse grid or 8 KB grid was built on the study tasks. Raw data:
`results/raw/sparse_budgets/`.

Two open questions of the paper. Sparse grids are the standard answer of numerical approximation to the n^d cost of
a grid and were not among the table maps. And the budgets were 4, 32 and 128 KB, so the paper cannot say where
between 4 and 32 KB KAN tables start to win.

## Registered hypotheses

All 12 physics models (2 PV, 3 physics-map, 7 input-rule tasks), all three data splits.

- **S1**: with sparse grids added to the table maps, the KAN table still wins both the 32 and the 128 KB cell
  (test RMSE ≤ 0.8 × the best table map, every model chosen on validation RMSE) on at least 7 of the 8 models with
  five or more inputs, on each split.
- **B1**: the rule's budget bound is the smallest b of 8, 16 and 32 KB such that the KAN table wins at b and at
  every larger tested budget (8, 16, 32, 64, 128 KB), against every table map including sparse grids, on at least 7
  of the 8 models with five or more inputs, on each split. We expect 16 KB.

If S1 fails, the paper names the models where a sparse grid takes a cell and narrows its claim to the rest. The
paper's rule states the bound B1 finds; if no b qualifies, it states the budgets at which KAN tables won.

Descriptive: the best sparse grid over the KAN table per cell; how often a sparse grid is the best table map; the
KAN table's cells on the four models with four or fewer inputs at the new budgets; the latency on the Cortex-A53 of
the sparse grids chosen at 32 and 128 KB on split 0.

## Setup

**Sparse grids** (`kantab/sparse.py`), piecewise linear on the scaled input box [-1, 1]^d, in two bases:

- `mod`: the modified linear basis without boundary nodes (constant at level 1, outermost hats extrapolating to the
  boundary);
- `bound`: hats with the boundary nodes at level 0.

Level sets:

- regular: every level vector with |l − base|₁ ≤ n − 1, every n up to 128 KB;
- dimension-adaptive (Gerstner and Griebel, 2003): a level vector's indicator is max(w · its largest absolute
  surplus / the first level vector's, (1 − w) · the first level vector's points / its points), w ∈ {0.5, 0.9, 0.99};
  refined up to 128 KB and cut at 4, 8, 16, 32, 64 and 128 KB.

Values: interpolation of the simulator at the nodes, standardized with the training targets like the node-filled
grids; for `mod` also a least-squares fit to the training set, min ‖Aα − y‖²/N + λ‖α − α_interp‖², by LSQR from the
interpolated surpluses (1000 iterations), λ ∈ {1e-6, 1e-4, 1e-2, 1} chosen on validation. The boundary basis is not
fitted: its boundary levels make the fit too costly in seven dimensions.

Storage: int16 surpluses with one float32 scale per level vector, 4 header bytes, and for adaptive grids one byte
per input for each level vector. Each variant and cut is one model; the table-map selection picks the best under a
budget by validation RMSE, as for the other maps.

**Intermediate budgets.** The anisotropic grids of the earlier studies exist at 2, 4, 16, 32, 64 and 128 KB (fill
1 and 0.5 of each budget); new ones at 8 KB, fitted and node-filled, as before. KAN tables, CP, GA2M and uniform
grids are the models already trained; the selection at 8, 16 and 64 KB uses them.

**Latency.** A C kernel for sparse grids in `c/maps.c`, timed with `bench` on the Orange Pi as before.

## Addendum: accuracy against latency

Registered on 2026-09-28, after S1 and B1 were computed and before the timing below. S1 failed: at 32 and 128 KB a
fitted, dimension-adaptive sparse grid is more accurate than the KAN table on six of the eight models with five or
more inputs. On the Cortex-A53 the sparse grids chosen at 32 and 128 KB took 12–125 µs, the KAN tables 1–2 µs.
`bench_sgrid` now builds, once at start-up, the offset of each level vector and the inputs whose level is above 1
(`sgrid_plan` in `c/maps.c`), which about halves those times; the first times stay in
`latency_orangepi_sgrid_first.txt`. For firmware the question becomes the trade between error and latency.

Every sparse grid of split 0, both bases and every cut, is timed on the Cortex-A53 with the planned kernel (a
fitted grid has the latency of its structure). For the eight models with five or more inputs, on split 0:

- **L1**: at the latency of the KAN table chosen at 32 KB, the most accurate sparse grid that is no slower has a
  larger test error than that KAN table, on at least 6 of the 8 models.
- Descriptive: the fastest sparse grid that reaches the error of the KAN table chosen at 32 KB and at 128 KB, and its
  latency over the KAN table's; the Pareto fronts of error against latency.

If L1 holds, the paper recommends KAN tables where a call must take about a microsecond and sparse grids where
memory is the constraint. If L1 fails, it recommends sparse grids as the table map and KAN tables only on the models
where they are more accurate at equal memory.

## Deviations

- `bench_sgrid`, a separate program, times the sparse grids instead of `bench`: its specs carry the level vectors,
  which do not fit `bench`'s spec buffer. Timing as in `bench`: median of single calls after 1000 warm-up calls.
- Every board: the Cortex-M4 receives the sparse-grid specs over USB, one line at a time, and the ESP32 has them
  compiled into its firmware (`gen_specs.py --sgrid`); both time each grid as the other specs, with the surpluses
  in flash and copied to RAM. The sparse-grid code (`c/maps.c`) is built with `-O3` on both.
- The first Cortex-M4 run stopped on a bus fault: with a large grid copied to RAM, the kernel's plan did not fit in
  the heap, and `sgrid_plan` did not check its allocations. It now does, the bench reports such a grid as not
  fitting, and the run was repeated from the start. 87 of the 308 grids have no time in RAM on the Cortex-M4.
- Two other variants of the tuned kernel were tried on the Cortex-A53 and dropped: level vectors sorted to share
  prefixes (slower, 35.5 against 29.5 µs on one grid) and four accumulators (no faster).
- The scalar KAN kernel on the Cortex-A53 is `bench` built with Zig for the Cortex-A53 with `__ARM_NEON` undefined
  and auto-vectorization off (`-U__ARM_NEON -U__ARM_NEON__ -fno-vectorize -fno-slp-vectorize`); the q8 NEON block
  in `c/kernels.c` now checks `__ARM_NEON` as well as `__aarch64__`.

## Results

Run on 2026-09-28. Recompute:

```
python -m kantab.experiments.sparse_budgets frontier
python -m kantab.experiments.sparse_budgets tradeoff
```

Both read the committed runs and latencies. The grids themselves (`build`) are not in the repository; `run` and
`specs` need them, and `results/raw/sparse_budgets/levels.json` holds their adaptive level vectors.

- **S1: falsified.** With sparse grids among the table maps the KAN table wins both the 32 and the 128 KB cell on
  2 of the 8 models with five or more inputs, pv_chain6 and msis7, on each split. A fitted dimension-adaptive
  sparse grid is the best table map in 48 of the 60 cells from 8 to 128 KB on each split. At 32 and 128 KB its test
  error over the KAN table's is 0.05–3.1 on the models with five or more inputs and 0.04–0.73 on those with four or
  fewer. The KAN table still wins single cells on ign5 and msis_d6 (and msis_d5 on split 2).
- **B1: no bound.** No budget of 8, 16 or 32 KB is one from which the KAN table wins every larger budget on 7 of 8
  models: at 8 KB 0 of 8, at 16 and 32 KB 2 of 8, on each split.
- **L1: confirmed, 8 of 8.** At the latency of the KAN table chosen at 32 KB (1–1.6 µs on the Cortex-A53), the most
  accurate sparse grid that is as fast has 6.5–615 times the KAN table's error (at 32 and 128 KB). The fastest sparse
  grid that reaches the KAN table's error takes 6.8–28 times as long; on pv_chain6 and msis7 no sparse grid reaches it.
- The sparse grids chosen at 32 and 128 KB take 7.5–63 µs on the Cortex-A53 with the planned kernel (12–125 µs with
  the first one), against 1–2 µs for the KAN tables.

The earlier table maps could not use the memory in five to seven dimensions: on pv_mp5 the best of them at 128 KB
was a 10 KB CP table. A sparse grid grows with the budget as a KAN table does, and on smooth models it is the more
accurate of the two per byte. A KAN table is the faster: it reads two table entries per edge, a sparse grid one
surplus per level vector, and a sparse grid needs hundreds of level vectors to reach a KAN table's error.

## Addendum: every board, a tuned kernel and a scalar control

Registered on 2026-09-28, before the timings below. The sparse grids were timed on the Cortex-A53 only, with a
scalar kernel, while the KAN tables run NEON code there, so the latency gap could come from vectorization. This
addendum:

- adds a second sparse-grid kernel, `sgrid_forward_fast` in `c/maps.c`: the one-dimensional hats from multiplications
  by powers of two instead of divisions, one table lookup per active input, `-O3`; every sparse grid takes the faster
  of the two kernels;
- times every sparse grid of split 0 on the Cortex-A7 and, in the modified basis only (no grid with boundary nodes
  was on the Cortex-A53's error–latency front), on the ESP32 and the Cortex-M4 with the surpluses in flash and,
  where they fit, in RAM; the microcontrollers run the KAN tables in plain C too;
- times the KAN tables on the Cortex-A53 with NEON disabled;
- counts, per call, the table entries a KAN table reads and the level vectors a sparse grid visits.

For the eight models with five or more inputs, on split 0, at the KAN tables chosen at 16 and at 32 KB:

- **T1**: the fastest sparse grid that reaches the KAN table's error takes at least 3 times as long on at least 6 of
  the 8 models, on each board (Cortex-A53, Cortex-A7, ESP32 and Cortex-M4 with the model in flash), and on the
  Cortex-A53 also against the KAN table on its scalar kernel.
- **T2**: on the same boards and budgets, the most accurate sparse grid that is as fast as the KAN table has a
  larger error on at least 6 of the 8 models.

If T1 and T2 hold everywhere, the paper states the latency advantage as a property of the representation; where
they fail, it states on which boards and kernels the advantage holds.

## Results of the addendum

Run on 2026-09-28. Recompute: `python -m kantab.experiments.sparse_budgets boards` (writes `boards.json`). Every
checksum of the microcontrollers matches the host's (1232 times on the ESP32, 1053 on the Cortex-M4).

- **T1: confirmed, 8 of 8 on every board at 16 and at 32 KB.** Where a sparse grid reaches the KAN table's error it
  takes 11–24 times as long on the Cortex-A53 (the KAN tables take 0.62–1.6 µs), 7.7–21 on the Cortex-A7, 5.7–26 on
  the ESP32 and 4.6–10 on the Cortex-M4, with the models in flash, and 3.3–6.6 times as long as the KAN tables on
  the scalar Cortex-A53 kernel. On pv_chain6 and msis7 no sparse grid reaches it.
- **T2: confirmed, 8 of 8 on every board at 16 and at 32 KB.** The most accurate sparse grid as fast as the KAN
  table has 5.5–576 times its error on the Cortex-A53, 1.9–103 against the scalar kernel, 5.3–386 on the Cortex-A7,
  3.0–289 on the ESP32 and 3.0–205 on the Cortex-M4.
- At 128 KB (descriptive) T2 holds 8 of 8 everywhere and T1 7 or 8 of 8: on pv_mp5 the sparse grid takes 1.9 times
  as long against the scalar kernel, 2.7 on the Cortex-A7 and 2.3 on the ESP32.
- With the models in RAM (descriptive), T1 and T2 hold 8 of 8 on the ESP32 and on the Cortex-M4, there over the
  grids that fit.
- The scalar KAN tables are 2.5–3.8 times slower than the NEON ones, so vectorization accounts for part of the gap
  on the Cortex-A53.
- The tuned kernel is a median 1.25 times faster than the planned one on the Cortex-A53, 1.11 on the Cortex-A7,
  1.17 on the ESP32 and 1.05 on the Cortex-M4 (grids above 2 µs on the Cortex-A cores, above 200 cycles on the
  microcontrollers).
- Structure, at the KAN tables chosen at 16 and 32 KB: a KAN table reads 96–768 table entries per call, two per
  edge; the fastest sparse grid at its error reads one surplus in each of 153–744 level vectors, whose addresses
  and weights take 332–2265 one-dimensional basis values.

- Post hoc (descriptive): the sparse grids above are picked by test error, which favours them. Picked by
  validation error, as the KAN tables are, T1 and T2 hold 8 of 8 on every board at 16 and 32 KB; every error ratio and 77 of the 80
  time ratios are unchanged, and on flame5 at 32 KB the grid chosen is slightly faster (10.7 instead of
  10.9 times the KAN table's time on the Cortex-A53).

T1 and T2 hold on every board, so the paper states the latency advantage as a property of the representation, with
the scalar control as its lower end.

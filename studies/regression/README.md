# KAN tables on smooth low-dimensional regression

Registered and run on 2026-09-25. Raw data: `results/raw/regression/`.

## Registered hypotheses

Tasks: five noise-free synthetic functions (two from the KAN paper; AI Feynman I.6.2, I.29.16, I.40.1) and five
UCI sets (CCPP, Airfoil, Concrete, Energy, Air Quality), 2–8 inputs. Five outer folds: test fold f, validation
fold f + 1. Latency: median single call on a Cortex-A53 (Orange Pi Zero 3, 1.416 GHz); the Cortex-A7 is
descriptive.

A point is a model and its deployment variant, with its mean test RMSE over folds, size and latency. For a set
of baseline families B, T is the lowest RMSE in B. A task counts for KAN if a table-deployed KAN

- (a) reaches T at ≤ ¼ of the latency, or ≤ ¼ of the size, of the fastest (smallest) point of B that reaches T; or
- (b) reaches T/2 no slower than the fastest point of B that reaches T.

- **N1**: B = {int8 MLP}; KAN wins ≥ 5 of 10 tasks.
- **N2**: B = {int8 MLP, fp32 MLP, GBDT, polynomial}; ≥ 5 of 10.

## Setup

- MLP: 1–3 hidden layers of 4–128, ReLU. KAN: [d, W, 1] with W 1–32, G ∈ {5, 10, 20}, and [d, W, W, 1].
  Adam, lr ∈ {1e-3, 3e-3, 1e-2}, 6000 steps, seeds 0 and 1, best state on validation.
- GBDT: 25–800 trees, 7/15/31 leaves, lr ∈ {0.05, 0.1, 0.3}. Polynomial: degree 1–6, ridge.
- KAN tables: nearest entry of 2^6 or 2^8 (int8, int16) and linear interpolation between 2^4 or 2^6 entries
  (int16, 8-bit fraction). The fp32 spline KAN is an accuracy reference only.
- Every C kernel is checked against the Python model on both boards (`checkset`, `checkcmp`): max |Δ| ≤ 4.3e-5.

## Deviations

- Before the criteria were fixed, trial runs on fold 0 of kan_toy4, feyn_i29_16 and Concrete showed that
  nearest-entry tables lose most of the KAN's accuracy. The interpolated variants were added after that. Input
  scaling changed from quantiles to min/max.
- Kernel changes after the criteria, before any timing: a NEON path for MLP layers with few inputs (int8 4-16-1
  959 → 667 ns, in the MLP's favour); NEON for int8 tables with more than 32 outputs; the `BENCH_INNER` mode.

## Results

| | |
|---|---|
| N1 | **confirmed, 9/10** (loses Air Quality) |
| N2 | **confirmed, 7/10**: all five formulas, Airfoil, Energy |

The same tasks win on the A53 and the A7, for single calls and batches of 16, and with selection on validation
RMSE.

Best test RMSE per family, mean over 5 folds:

| task | int8 MLP | fp32 MLP | KAN fp32 | KAN nearest | KAN interp. | GBDT | poly |
|---|---|---|---|---|---|---|---|
| kan_toy2 | 0.0180 | 0.0058 | 0.00077 | 0.0128 | **0.0017** | 0.036 | 0.091 |
| kan_toy4 | 0.0300 | 0.0265 | 0.0017 | 0.0095 | **0.0023** | 0.075 | 0.116 |
| feyn_i6_2 | 2.9e-4 | 8.6e-5 | 1.5e-5 | 2.2e-4 | **2.5e-5** | 6.0e-4 | 2.4e-4 |
| feyn_i29_16 | 0.0275 | 0.0173 | 0.0134 | 0.0244 | **0.0160** | 0.218 | 0.238 |
| feyn_i40_1 | 0.0115 | 0.0103 | 0.0012 | 0.0049 | **0.0021** | 0.090 | 0.025 |
| CCPP | 3.84 | 3.83 | 3.78 | 3.79 | 3.78 | **3.21** | 4.01 |
| Airfoil | 1.546 | 1.484 | 1.439 | 1.487 | **1.432** | 1.653 | 2.83 |
| Concrete | 5.12 | 5.14 | 4.80 | 4.78 | 4.79 | **4.34** | 5.67 |
| Energy | 0.498 | 0.475 | 0.415 | 0.415 | **0.413** | 0.450 | 0.500 |
| Air Quality | 21.74 | 21.74 | 22.42 | 22.41 | 22.23 | 22.20 | **21.26** |

On the formulas the interpolated table reaches the best MLP's error in 0.3–1.6 µs, against 13–44 µs. On CCPP and
Concrete GBDT is 9–15 % more accurate at 47–83 µs; on Air Quality the polynomial wins.

The [strong-MLP study](../strong_mlps) narrows the formula result.

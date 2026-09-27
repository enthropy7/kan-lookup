# The physics-map studies on two more data splits

Registered and run on 2026-09-26. Raw data: `results/raw/data_splits/`.

## Registered hypotheses

The PV-map, physics-map and input-rule studies used one split: test group 0, validation group 1, training the other three of five. Here everything is
repeated on splits 1 and 2 (test group f, validation f + 1).

- **F1**: on each of splits 1 and 2, at least 7 of the 8 verdicts equal split 0. The verdicts are C1, C2 of the PV-map study;
  C1, C2, C3 of the physics-map study; H1, H2, H3 of the input-rule study. On split 0 they are ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓.
- **F2**: of the 36 cells (12 models × 3 budgets) "KAN ≥ 20 % more accurate than the best table map", at least 30
  equal split 0 on each split.
- **F3**: on each split KAN wins both the 32 and the 128 KB cell on ≥ 6 of the 8 models with ≥ 5 inputs and on at
  most 1 of the 4 with ≤ 4 inputs. On split 0: 7 of 8 and 0 of 4.

If all three hold, the paper reports the mean and spread over the three splits. Otherwise it reports each split,
and only what holds on all three counts as a result.

## Setup

- The 12 models of those three studies on splits 1 and 2, with everything as before: KAN, CP and GA2M in all shapes
  (30 000 steps, three learning rates, seeds 0 and 1), four grid kinds, and polynomials. The anisotropic grids use
  the curvature of their own split's training part.
- MLP: the 4 fp32 shapes with the lowest split-0 validation RMSE; descriptive, not in any criterion.
- GBDT is not trained, because no criterion uses it.
- Criteria are computed by the same selection code on every split: selection on validation, test RMSE compared,
  threshold 0.8, budgets 4 / 32 / 128 KB.
- PV-map C1 and C2 are against the uniform grids, as registered; physics-map C3 against all four grid kinds.

```
python -m kantab.experiments.data_splits run --out results/raw/data_splits --workers 10
python -m kantab.experiments.data_splits frontier
```

## Deviations

- Before the run, the evaluation code was checked to reproduce all 8 verdicts and the per-model wins of split 0.
- Run with `CUDA_VISIBLE_DEVICES=`.

## Results

6204 tasks, 7956 rows (`runs.jsonl`); criteria: `frontier.json`.

| | split 1 | split 2 |
|---|---|---|
| F1: verdicts equal to split 0 (≥ 7/8) | **8/8** | **8/8** |
| F2: cells equal to split 0 (≥ 30/36) | **35/36** | **34/36** |
| F3: models won, ≥ 5 inputs (≥ 6/8) and ≤ 4 inputs (≤ 1/4) | **7/8, 0/4** | **8/8, 0/4** |

All three hold, also with selection on test RMSE. The three cells that differ from split 0 all sit at the threshold
(best map / KAN = 1.25):

- tyre4 at 128 KB: 1.17 on split 0, 1.30 on split 1;
- tyre4 at 32 KB: 1.23 on split 0, 1.250 on split 2;
- msis_d5 at 128 KB: 1.20 on split 0, 1.29 on split 2. On split 2 msis_d5 therefore wins both cells and the
  input-count rule predicts all 7 input-rule models.

MSIS dimension sweep, best map / KAN at 128 KB for d = 3…7, ρ = 0.90 on every split:

| split | d3 | d4 | d5 | d6 | d7 |
|---|---|---|---|---|---|
| 0 | 0.83 | 0.78 | 1.20 | 1.59 | 1.95 |
| 1 | 0.74 | 0.78 | 1.19 | 1.83 | 1.79 |
| 2 | 0.67 | 0.75 | 1.29 | 1.87 | 1.83 |

Best table map / KAN per cell, mean [min, max] over splits 0–2:

| model | 4 KB | 32 KB | 128 KB |
|---|---|---|---|
| pv_mp5 | 0.59 [0.48, 0.68] | 2.85 [2.66, 3.08] | 3.36 [2.33, 4.11] |
| pv_chain6 | 0.96 [0.94, 1.02] | 1.87 [1.67, 2.06] | 2.42 [2.08, 2.63] |
| tyre4 | 0.48 [0.45, 0.52] | 1.21 [1.16, 1.25] | 1.24 [1.17, 1.30] |
| msis7 | 1.03 [0.96, 1.07] | 1.62 [1.55, 1.70] | 1.86 [1.79, 1.95] |
| flame5 | 0.80 [0.76, 0.86] | 3.16 [3.08, 3.27] | 3.57 [3.50, 3.63] |
| gasz7 | 0.37 [0.35, 0.40] | 2.20 [1.74, 2.47] | 3.36 [2.72, 3.94] |
| ign5 | 0.93 [0.83, 1.02] | 3.67 [3.42, 3.85] | 4.13 [3.91, 4.28] |
| igrf4 | 0.30 [0.30, 0.31] | 0.81 [0.80, 0.82] | 0.90 [0.89, 0.91] |
| msis_d3 | 0.36 [0.35, 0.36] | 1.11 [1.04, 1.18] | 0.74 [0.67, 0.83] |
| msis_d4 | 0.72 [0.71, 0.72] | 1.30 [1.27, 1.32] | 0.77 [0.75, 0.78] |
| msis_d5 | 0.91 [0.90, 0.94] | 1.63 [1.59, 1.69] | 1.23 [1.19, 1.29] |
| msis_d6 | 0.95 [0.94, 0.95] | 1.61 [1.58, 1.63] | 1.76 [1.59, 1.87] |

By the registered decision, the paper reports the mean and spread over the three splits.

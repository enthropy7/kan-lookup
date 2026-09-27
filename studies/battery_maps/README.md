# KAN tables vs calibration maps on measured battery data

Registered on 2026-09-26, run after the fix below. Raw data: `results/raw/battery_maps/`.

## Registered hypotheses

Firmware stores multidimensional relations as grid maps with interpolation, whose memory grows as n^d. A KAN
table grows linearly with the number of inputs.

- **C1**: at memory budgets of 2.5, 16 and 70 KB, the best KAN table has test RMSE ≤ 0.8 × the best grid map of
  the same budget on at least 2 of 3 budgets.
- **C2**: the same against the best of grid, CP decomposition and GA2M (1-D plus pairwise 2-D tables).

Per budget and family, the point with the lowest validation RMSE within the budget is chosen; test RMSEs are
compared.

## Setup

- Task: state of charge of an LG 18650HG2 cell (Kollmeyer et al., Mendeley cp3473x7xv v3, CC BY 4.0).
- Inputs: the five of the authors' prepared set, namely voltage, current, cell temperature, and moving averages
  of voltage and current.
- Training: the authors' mixed drive cycles at −10, 0, 10 and 25 °C, 669 956 rows. Validation is 4 contiguous
  blocks of 20 (20 %). Test: the standard cycles (UDDS, HWFET, LA92, US06), 173 624 rows.
- Grid map: n ∈ {3, 4, 5, 6, 7, 8, 10} per axis, least squares with a second-difference penalty, as calibration
  tools fit them.
- CP: rank 1–16, 16–64-entry tables. GA2M: 16–64-entry 1-D tables plus all 10 pairwise tables.
- KAN, MLP (ReLU, SiLU, tanh), GBDT and polynomials as before. 30 000 steps, seeds 0 and 1.

## Deviations

- Before the criteria were fixed there were trial runs on this data: KAN 1.77 %, SiLU MLP 1.88 %, GBDT 1.68 %,
  grid map 1.32–1.85 %. The grid was already known to be a strong baseline. Budgets were then set by grid sizes of
  4, 6 and 8 points per axis, and the 20 % threshold in advance.
- The first run was killed by the OOM killer after 48 rows, which were not looked at. The fix made
  table calibration process the data in chunks, with bitwise identical tables. The run was restarted from zero.

## Results

| | |
|---|---|
| C1 | **refuted, 0/3**. On validation: KAN 1.74 / 1.57 / 1.57 % vs a 3^5 grid (490 B) at 1.59 % |
| C2 | **refuted, 0/3**. The best of the three maps is that grid in every budget |

Best test RMSE of state of charge (%), within the budget:

| budget | KAN table | grid | CP | GA2M | int8 MLP | polynomial |
|---|---|---|---|---|---|---|
| ≤ 2.5 KB | 1.738 | 1.459 | 1.862 | 1.516 | 1.412 | 1.412 |
| ≤ 16 KB | 1.453 | 1.417 | 1.532 | 1.516 | 1.412 | 1.394 |
| ≤ 70 KB | 1.453 | 1.311 | 1.532 | 1.516 | 1.412 | 1.394 |

Within 70 KB every family lands between 1.31 and 1.86 %. Without a memory limit the best is a 10^5 grid at
1.22 % (200 KB). A static map of instantaneous values does not see the cell's history, and that limit, not the
model, sets the error. The KAN table is faster (0.6–0.9 µs vs 1.2 µs for the grid on the A53) but
not more accurate.

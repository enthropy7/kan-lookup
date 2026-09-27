# Finer KAN tables on new seeds

Registered and run on 2026-09-25. Raw data: `results/raw/fine_tables/`.

## Registered hypotheses

In the [strong-MLP study](../strong_mlps) the fp32 KAN was more accurate than the best MLP, but its 64-point table was not. The error of linear
interpolation falls with the square of the step, so 256 points should recover most of the model's accuracy.

- **T1**: in the pool of the regression study, grid A of the strong-MLP study and C, N2 of the regression study holds on
  ≥ 4 of 5 formulas. C repeats grid B of the [strong-MLP study](../strong_mlps)
  (same shapes, 30 000 steps for every family) on new seeds 2 and 3, adding tables of 128 and 256 points
  (`lin7_i16`, `lin8_i16`). 1024 points (`lin10_i16`) are descriptive only.
- **T2**: on ≥ 4 of 5 formulas the best `lin8_i16` RMSE is ≤ 1.5 × the best fp32 KAN RMSE of the same shapes.

## Deviations

The 128-, 256- and 1024-point variants were chosen after the [strong-MLP study](../strong_mlps) result, which is why every model is retrained
on new seeds.

## Results

| | |
|---|---|
| T1 | **confirmed, 4/5** (loses feyn_i29_16). Same on both boards, both timing modes and on validation. Real data 1/5 (Energy) |
| T2 | **refuted, 3/5** (lin8 / fp32: kan_toy2 1.55, feyn_i6_2 2.60) |

N2 in the primary pool (A53):

| task | best alternative (T) | fastest KAN with RMSE ≤ T | best KAN |
|---|---|---|---|
| kan_toy2 | SiLU MLP 0.00137, 52.4 µs | lin7 0.00058, 1.58 µs | 0.00047 |
| kan_toy4 | SiLU MLP 0.0059, 52.8 µs | lin6 0.0024, 0.67 µs | 0.00072 |
| feyn_i6_2 | tanh MLP 1.69e-5, 52.8 µs | lin7 1.32e-5, 1.58 µs | 1.26e-5 |
| feyn_i29_16 | SiLU MLP 0.0044, 52.8 µs | none | 0.0071 |
| feyn_i40_1 | SiLU MLP 0.0023, 15.2 µs | lin6 0.0018, 0.92 µs | 0.0017 |

Going from 64 to 256 points costs 15–25 % latency and 4× memory.

Diagnosis after the result, not a criterion: the lin8 / fp32 gap sits in single folds, and lin10 has the same gap.
In 0.3–0.8 % of test rows a hidden value leaves the training range that sets the second layer's table range, and
the table clips it. A 5 % margin on that range brings lin8 to 1.01–1.04× of fp32 on the three models checked.
The [range-margin study](../range_margin) tests this.

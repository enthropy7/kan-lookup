# A 5 % margin on the hidden-layer table range

Registered and run on 2026-09-25. Raw data: `results/raw/range_margin/`.

## Registered hypotheses

- **M1**: with the hidden-layer table ranges widened by 5 % on each side, the best `lin8_i16_m5` RMSE is
  ≤ 1.2 × the best fp32 KAN RMSE of the same runs on ≥ 4 of 5 formulas.
- **M2**: the margin does not hurt: `lin8_i16_m5` ≤ 1.05 × `lin8_i16` on the same models on ≥ 9 of 10 tasks.

## Setup

The KAN shapes of grid B of the strong-MLP study and grid C of the fine-table study, 30 000 steps, new seeds 4 and 5, 5 folds, 10 tasks: 600 runs. Each
model is evaluated with and without the margin; the input layer is unchanged, since inputs are already clipped
to the training range. The margin changes only the stored range, not the kernel, size or latency.

## Deviations

The idea and the 5 % value come from the post-hoc diagnosis in the [fine-table study](../fine_tables) on three models, where 5 % and 20 % gave
the same result. Hence the new seeds.

## Results

| | |
|---|---|
| M1 | **confirmed, 5/5** (0.99–1.16) |
| M2 | **confirmed, 10/10** (0.45–1.001) |

Best test RMSE, mean over folds:

| task | fp32 KAN | lin8 | lin8 + 5 % | + 5 % / fp32 | + 5 % / lin8 |
|---|---|---|---|---|---|
| kan_toy2 | 2.19e-4 | 3.93e-4 | 2.55e-4 | 1.16 | 0.65 |
| kan_toy4 | 4.88e-4 | 7.22e-4 | 4.82e-4 | 0.99 | 0.67 |
| feyn_i6_2 | 5.23e-6 | 1.23e-5 | 5.51e-6 | 1.05 | 0.45 |
| feyn_i29_16 | 4.60e-3 | 4.70e-3 | 4.62e-3 | 1.01 | 0.98 |
| feyn_i40_1 | 8.58e-4 | 1.48e-3 | 9.96e-4 | 1.16 | 0.67 |
| CCPP | 3.804 | 3.803 | 3.803 | 1.00 | 1.00 |
| Airfoil | 1.377 | 1.385 | 1.375 | 1.00 | 0.99 |
| Concrete | 4.727 | 4.717 | 4.722 | 1.00 | 1.00 |
| Energy | 0.391 | 0.402 | 0.401 | 1.03 | 1.00 |
| Air Quality | 22.43 | 22.44 | 22.43 | 1.00 | 1.00 |

The margin is part of every later export (`lin{6,7,8}_i16_m5`). With it, on 4 of 5 formulas the table is 2.7–12×
more accurate than the best fp32 SiLU/tanh MLP at 1–2 µs against 15–53 µs on the A53; on feyn_i29_16 it is 6 %
less accurate.

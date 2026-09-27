# KAN tables vs calibration maps on noise-free PV models

Registered and run on 2026-09-26. Raw data: `results/raw/pv_maps/`.

## Registered hypotheses

In the [battery-map study](../battery_maps) every method stopped at the measurement noise. Here the map is built from an industry physics model, as
a table for firmware is built from a simulator.

- **C1**: in ≥ 4 of 6 cells (2 tasks × budgets of 4, 32 and 128 KB) the best KAN table has test RMSE ≤ 0.8 ×
  the best grid map of the same budget, fitted to points or filled with model values at the nodes.
- **C2**: the same against the best of grid, CP and GA2M.

Selection as in the [battery-map study](../battery_maps): the lowest validation RMSE within the budget, then test RMSE.

## Setup

- pvlib 0.14, CEC module Canadian Solar CS5P-220M, maximum power of the single-diode model (W).
- **pv_mp5**: plane-of-array irradiance, cell temperature, angle of incidence, absolute air mass, precipitable
  water → IAM, First Solar spectral correction, single diode.
- **pv_chain6**: direct and diffuse irradiance, zenith, sun azimuth relative to the module, air temperature, wind
  → Hay–Davies transposition (tilt 30°), SAPM cell temperature, IAM, spectrum, single diode.
- 60 000 uniform points per task: 36 000 train, 12 000 validation, 12 000 test (fold 0).
- The box includes physically extreme combinations: pv_chain6 reaches 439 W for a 220 W module.
- Models as in the [battery-map study](../battery_maps). Grids: `grid_n{n}` (least squares plus smoothness) and `gridnode_n{n}` (model values at the
  nodes); n 3–9 for 5 inputs, 3–6 for 6.

## Deviations

- pvlib's own test suite is disabled in the build: its 28 failures are in `test_spa.py`, the solar-position code,
  which is not used here.
- At standard test conditions the model gives 219.96 W, against a nameplate of 219.961 W.
- A 300-step pipeline check ran before the criteria. The budgets are the battery-map study's, scaled to 4/32/128 KB, so that
  each holds a sensible grid for 5 and 6 inputs.

## Results

| | |
|---|---|
| C1 | **confirmed, 6/6**. KAN is 38 / 108 / 83× more accurate than the grid on pv_mp5 and 12 / 26 / 27× on pv_chain6. Same with test selection |
| C2 | **confirmed, 4/6**. At 32 and 128 KB KAN is 2.8–3.7× more accurate than CP. At 4 KB CP is as good (pv_mp5: 0.086 vs 0.141 W) |

Test RMSE (W), the best point of each family within the budget:

| task | budget | KAN table | grid (LSQ) | grid (nodes) | CP | GA2M | int8 MLP | fp32 MLP | poly | GBDT |
|---|---|---|---|---|---|---|---|---|---|---|
| pv_mp5 | 4 KB | 0.141 | 5.33 | 12.1 | **0.086** | 1.13 | 0.79 | — | 0.27 | 5.05 |
| | 32 KB | **0.024** | 2.54 | 5.35 | 0.066 | 1.03 | 0.54 | — | 0.27 | 1.84 |
| | 128 KB | **0.018** | 1.50 | 2.32 | 0.066 | 1.03 | 0.54 | — | 0.27 | 1.21 |
| pv_chain6 | 4 KB | 0.853 | 10.3 | 20.0 | 0.867 | 11.3 | 1.66 | **0.816** | 7.18 | 12.9 |
| | 32 KB | **0.312** | 8.00 | 10.4 | 0.642 | 11.3 | 1.09 | 0.573 | 4.84 | 6.87 |
| | 128 KB | **0.184** | 4.87 | 5.93 | 0.642 | 11.3 | 0.95 | 0.410 | 4.84 | 4.26 |

The best fp32 SiLU MLP (132 KB, 53 µs on the A53) is 2–4× less accurate than the KAN table (2 µs). The [physics-map study](../physics_maps) shows
that much of the gap to the grid came from its uniform spacing.

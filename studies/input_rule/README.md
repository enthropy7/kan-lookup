# The input-count rule, tested by prediction

Registered and run on 2026-09-26. Raw data: `results/raw/input_rule/`.

## Registered hypotheses

Rule R, derived from the battery-, PV- and physics-map studies: for a smooth, noise-free physics model with **≥ 5 inputs** a KAN table is at least
20 % more accurate than every table map (four grids, CP, GA2M) at both 32 and 128 KB; with **≤ 4 inputs** it is
not; **at 4 KB** it is not 20 % more accurate than CP. Predictions for seven new tasks were written before
training.

- **H1**: R predicts "KAN wins both the 32 and the 128 KB cell" correctly on ≥ 6 of 7 tasks.
- **H2**: at 4 KB KAN is not ≤ 0.8 × CP on ≥ 6 of 7.
- **H3**: on the MSIS dimension sweep (d = 3…7), A(d) = best table map / KAN at 128 KB rises with d: Spearman
  ρ ≥ 0.9.

## Setup

| task | model | inputs | predicted |
|---|---|---|---|
| gasz7 | compressibility factor Z of natural gas with hydrogen, GERG-2008 (CoolProp 8.0.0) | T, p, ethane, propane, N2, CO2, H2 | win |
| ign5 | ignition delay, log10 s (Cantera 3.2.0, GRI-Mech 3.0, constant-pressure reactor, +400 K) | T0, p, φ, H2 fraction, CO2 fraction | win |
| igrf4 | total geomagnetic field, nT (IGRF-14 via ppigrf 2.1.0) | latitude, longitude, altitude 300–800 km, year 2020–2030 | no win |
| msis_d3…d6 | msis7 of the [physics-map study](../physics_maps) with only the first d inputs free (altitude, latitude, local time, day, 81-day F10.7, F10.7); the rest fixed at the box centre | 3–6 | d3, d4: no win; d5, d6: win |

Protocol, models and budgets as in the [physics-map study](../physics_maps).

## Deviations

- IGRF Gauss coefficients are linear in time within a 5-year epoch, so the field is computed for 2020, 2025 and
  2030 and interpolated. This agrees with a direct ppigrf call to 0.07 nT at 17–51 thousand nT.
- Spearman ρ here is exactly 0.9 = 1 − 6·2/(5·24). The first version of `frontier` compared the float
  0.8999… with the threshold and reported "refuted"; ρ is now rounded to 12 digits. The criterion is met at the
  boundary.
- Run with `CUDA_VISIBLE_DEVICES=` (see the [physics-map study](../physics_maps)).

## Results

| | |
|---|---|
| H1 | **confirmed, 6/7**. Wins as predicted: gasz7 (2.5 / 3.9× at 32 / 128 KB), ign5 (3.7 / 4.2×), msis_d6 (1.6 / 1.6×). No win as predicted: igrf4, msis_d3, msis_d4. Miss: msis_d5 wins at 32 KB (1.59×), at 128 KB only 1.20× over the anisotropic grid |
| H2 | **confirmed, 7/7** |
| H3 | **confirmed at the boundary**: A(d) = 0.83, 0.78, 1.20, 1.59, 1.95 for d = 3…7, ρ = 0.90 |

Same with test selection and on both boards.

Test RMSE, the best point of each family within the budget (gasz7 in 10⁻⁵ Z, ign5 in log10 s, igrf4 in nT,
msis in 10⁻³ log10 ρ):

| task | budget | KAN | grid LSQ | grid nodes | aniso. LSQ | aniso. nodes | CP | GA2M | int8 MLP | fp32 MLP | poly | GBDT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gasz7 | 4 KB | 31.0 | 488 | 598 | 32.3 | 131 | **11.6** | 208 | 32.5 | 7.54 | 13.9 | 644 |
| | 32 KB | **3.66** | 139 | 143 | 15.6 | 50.3 | 9.04 | 208 | 29.9 | 6.56 | 1.69 | 289 |
| | 128 KB | **2.30** | 65.3 | 63.0 | 10.2 | 31.2 | 9.04 | 208 | 29.9 | 4.80 | 1.69 | 195 |
| ign5 | 4 KB | **0.0091** | 0.043 | 0.139 | 0.016 | 0.049 | 0.0093 | 0.043 | 0.013 | 0.0065 | 0.013 | 0.068 |
| | 32 KB | **0.0019** | 0.027 | 0.073 | 0.0089 | 0.021 | 0.0070 | 0.043 | 0.0097 | 0.0041 | 0.013 | 0.032 |
| | 128 KB | **0.0016** | 0.020 | 0.041 | 0.0069 | 0.012 | 0.0070 | 0.043 | 0.0088 | 0.0033 | 0.013 | 0.018 |
| igrf4 | 4 KB | 199 | 991 | 1 784 | 82.5 | 189 | **60.8** | 362 | 194 | 76.1 | 1 077 | 1 651 |
| | 32 KB | 20.7 | 352 | 653 | **16.9** | 37.1 | 24.0 | 362 | 144 | 44.8 | 1 077 | 572 |
| | 128 KB | 11.6 | 167 | 278 | **10.3** | 16.6 | 24.0 | 362 | 126 | 39.1 | 1 077 | 334 |
| msis_d3 | 4 KB | 7.22 | 4.81 | 10.6 | 4.22 | 9.44 | **2.61** | 34.5 | 14.1 | 11.5 | 28.6 | 60.4 |
| | 32 KB | **0.934** | 1.10 | 2.35 | 1.10 | 2.28 | 2.61 | 34.5 | 9.27 | 7.16 | 28.6 | 24.3 |
| | 128 KB | 0.784 | **0.649** | 0.903 | 0.680 | 0.875 | 2.61 | 34.5 | 9.08 | 4.86 | 28.6 | 13.4 |
| msis_d4 | 4 KB | 13.2 | 31.5 | 48.8 | 23.0 | 40.2 | **9.57** | 44.8 | 19.4 | 19.9 | 32.7 | 73.7 |
| | 32 KB | **5.64** | 9.37 | 17.6 | 7.44 | 13.9 | 9.57 | 44.8 | 13.5 | 12.7 | 32.7 | 33.3 |
| | 128 KB | 5.61 | 4.60 | 7.63 | **4.37** | 6.62 | 9.57 | 44.8 | 12.5 | 10.1 | 32.7 | 21.0 |
| msis_d5 | 4 KB | 17.2 | 80.1 | 110 | 42.1 | 64.9 | **16.1** | 51.8 | 26.3 | 23.8 | 36.3 | 98.2 |
| | 32 KB | **10.2** | 34.3 | 50.7 | 18.2 | 29.0 | 16.1 | 51.8 | 16.2 | 15.5 | 36.3 | 49.4 |
| | 128 KB | **9.60** | 15.8 | 22.8 | 11.5 | 15.8 | 16.1 | 51.8 | 15.5 | 12.6 | 36.3 | 32.6 |
| msis_d6 | 4 KB | 21.2 | 112 | 153 | 65.0 | 104 | **19.8** | 57.7 | 28.6 | 24.7 | 57.1 | 114 |
| | 32 KB | **12.5** | 45.3 | 65.8 | 32.5 | 52.9 | 19.8 | 57.7 | 19.1 | 16.6 | 38.6 | 59.7 |
| | 128 KB | **12.5** | 40.0 | 50.7 | 22.3 | 31.0 | 19.8 | 57.7 | 16.9 | 14.4 | 38.6 | 40.5 |

Bold: the best table map.

Over the 12 physics maps of the three map studies, KAN wins both the 32 and the 128 KB cell on 7 of 8 models with ≥ 5
inputs and on 0 of 4 with ≤ 4. At 4 KB it is not 20 % more accurate than CP on any of them. On gasz7 the most
accurate model is a degree-6 polynomial (1.69 vs 2.27 × 10⁻⁵ at 16.5 vs 2.1 µs).

Against the best fp32 MLP, the fastest KAN table with no larger error is 33–63× faster on the A53 and 54–114×
faster on the A7. At the accuracy of the best 128 KB anisotropic grid the gap narrows: the fastest MLP that gets
there is 2.3–56× slower than the fastest KAN table (A53), and on gasz7 and ign5 it is 2–5× smaller.

These MLP timings use our kernels; the [vendor-kernel study](../vendor_kernels) re-times every MLP with XNNPACK,
esp-dsp and ESP-NN.

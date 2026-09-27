# Three more physics models; anisotropic grids

Registered and run on 2026-09-26. Raw data: `results/raw/physics_maps/`.

## Registered hypotheses

Two questions: is the [PV-map study](../pv_maps) specific to one model, and did it rest on a weak, uniform grid? An anisotropic grid is
added whose points per axis follow the model's curvature along that axis, as an engineer with a simulator would
place them. It also competes on the [PV-map study](../pv_maps) tasks.

- **C1**: in ≥ 6 of 9 cells (3 tasks × 4/32/128 KB) the best KAN table has test RMSE ≤ 0.8 × the best of four
  grids (uniform or anisotropic, least squares or node values).
- **C2**: the same against the best of grid, CP and GA2M.
- **C3**: C1 of the [PV-map study](../pv_maps) again on pv_mp5 and pv_chain6, with anisotropic grids among the rivals: ≥ 4 of 6 cells.

A cell where the rival has no point within the budget does not count for KAN. Before the run I expected tyre4
(4 inputs) to be the borderline case.

## Setup

- **tyre4**: Pacejka Magic Formula tyre of CommonRoad vehicle models 3.0.2, lateral force under combined slip
  over load, F_y / F_z. Inputs: longitudinal slip, slip angle, camber, road friction scale. F_y is proportional to
  F_z in this model, so F_z is divided out.
- **msis7**: NRL MSIS 2.1 (pymsis 0.13.0), log10 of total mass density. Inputs: altitude 200–800 km, latitude,
  local solar time, day of year, F10.7, its 81-day mean, Ap.
- **flame5**: adiabatic flame temperature, Cantera 3.2.0 with GRI-Mech 3.0, equilibrium at constant enthalpy and
  pressure. Methane–hydrogen and air diluted with CO2; inputs are initial temperature, pressure, equivalence ratio,
  H2 fraction and CO2 fraction.
- 60 000 uniform points, split as in the [PV-map study](../pv_maps). Models as in the [PV-map study](../pv_maps).
- Anisotropic grid, `kantab.maps.grid_counts`: the interpolation error is ≈ Σ_a C_a h_a² / 8, with C_a the RMS
  second derivative along axis a. Starting from 2 points per axis, a point goes to the axis with the largest error
  reduction per unit of log size while the table fits the budget. Sizes are each budget and its half.

## Deviations

- The first msis7 version called `pymsis.calculate` with a fractional date. It passes the whole day of year and
  the fraction as UT, so the map stepped once a day. It was replaced, before registration, by a call to the MSIS
  kernel with a continuous day and UT 12:00.
- The first run crashed after 156 rows: `env.snapshot` initialised CUDA in the parent process, and the forked
  workers failed. It was rerun with `CUDA_VISIBLE_DEVICES=`, on the CPU as before; the crashed rows are unused.
- The pipeline check before the criteria showed the anisotropic grid 2.9–28× more accurate than the uniform one.
  The registration text said 4–28×.

## Results

| | |
|---|---|
| C1 | **confirmed, 7/9**. msis7 and flame5 all six cells (2.0–4.9×). tyre4 1/3: at 4 KB the anisotropic grid is better (4.05 vs 6.63), at 128 KB KAN is only 1.17× better |
| C2 | **refuted, 4/9**. KAN wins msis7 and flame5 at 32 and 128 KB (1.7–3.6× over CP); not at 4 KB on any task, not on tyre4 |
| C3 | **confirmed, 6/6**. KAN over the anisotropic grid: 3.7 / 8.5 / 7.9× (pv_mp5), 2.3 / 2.4 / 2.6× (pv_chain6). The anisotropic grid is 5–13× better than the uniform one |

Same with test selection and on both boards.

Test RMSE, the best point of each family within the budget (tyre4 in 10⁻³ of F_y / F_z, msis7 in log10 ρ, flame5
in K):

| task | budget | KAN | grid LSQ | grid nodes | aniso. LSQ | aniso. nodes | CP | GA2M | int8 MLP | fp32 MLP | poly | GBDT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tyre4 | 4 KB | 6.63 | 31.5 | 44.4 | 4.05 | 8.71 | **3.00** | 37.7 | 6.48 | 1.84 | 78.1 | 59.2 |
| | 32 KB | **0.850** | 14.7 | 22.1 | 1.16 | 2.63 | 1.04 | 37.7 | 5.25 | 1.17 | 78.1 | 18.5 |
| | 128 KB | **0.587** | 12.5 | 17.8 | 0.688 | 1.22 | 1.04 | 37.7 | 4.54 | 0.894 | 78.1 | 11.0 |
| msis7 | 4 KB | **0.0374** | 0.216 | 0.376 | 0.108 | 0.161 | 0.0398 | 0.0620 | 0.0320 | 0.0307 | 0.0781 | 0.129 |
| | 32 KB | **0.0160** | 0.116 | 0.156 | 0.0508 | 0.0827 | 0.0272 | 0.0620 | 0.0216 | 0.0200 | 0.0406 | 0.0695 |
| | 128 KB | **0.0140** | 0.0857 | 0.109 | 0.0381 | 0.0605 | 0.0272 | 0.0620 | 0.0198 | 0.0162 | 0.0406 | 0.0479 |
| flame5 | 4 KB | 1.52 | 11.3 | 24.5 | 3.10 | 8.74 | **1.31** | 5.21 | 2.59 | 1.30 | 7.72 | 32.9 |
| | 32 KB | **0.272** | 4.92 | 10.8 | 1.33 | 3.24 | 0.889 | 4.95 | 2.27 | 0.674 | 7.72 | 14.4 |
| | 128 KB | **0.245** | 3.24 | 4.38 | 0.975 | 1.74 | 0.889 | 4.95 | 2.27 | 0.565 | 7.72 | 8.05 |

Bold: the best table map. The fp32 MLP is not a criterion family.

Memory at which each method reaches the accuracy of the best 128 KB anisotropic grid:

| task | KAN | CP | fp32 MLP |
|---|---|---|---|
| tyre4 | 40 KB | never | never |
| msis7 | 2.2 KB | 4.0 KB | 2.7 KB |
| flame5 | 6.1 KB | 5.4 KB | 9.1 KB |
| pv_mp5 | 3.1 KB | 2.7 KB | 68 KB |
| pv_chain6 | 15 KB | never | 35 KB |

Against the best fp32 SiLU MLP (132 KB, 53 µs on the A53), the fastest KAN table with no larger error takes
1.0–1.8 µs, 30–55× faster, and 51–97× faster on the A7.

These MLP timings use our kernels; the [vendor-kernel study](../vendor_kernels) re-times every MLP with XNNPACK,
esp-dsp and ESP-NN.

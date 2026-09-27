# The regression study against smooth MLPs and a 5× training budget

Registered and run on 2026-09-25. Raw data: `results/raw/strong_mlps/activations/`, `results/raw/strong_mlps/long_training/`.

## Registered hypotheses

The [regression study](../regression) used only ReLU MLPs, and in 80–99 % of its runs on the formulas the best state was in the last 500 of
6000 steps.

- **R1**: with SiLU and tanh MLPs added to the baselines (grid A), N2 of the [regression study](../regression) holds on ≥ 4 of 5 formulas.
- **R2**: the same after the 2 most accurate shapes of every family (ReLU, SiLU, tanh MLP, KAN) per task are
  retrained for 30 000 steps (grid B).

## Setup

- A: SiLU and tanh MLPs, the 18 shapes, tasks, folds, learning rates and seeds of the [regression study](../regression); fp32 and int8, the
  int8 activation through a 255-entry table as in TFLite Micro. 10 800 runs.
- B: 2 shapes per family and task, 30 000 steps, cosine schedule. 2400 runs. Both versions of a B shape (6000
  and 30 000 steps) are separate points.

## Deviations

- All results of the [regression study](../regression) were known.
- Only the most accurate shapes got the long budget; the small, fast KANs that won on latency in the [regression study](../regression) did not.
- SiLU and tanh MLPs were timed with the ReLU kernel, as registered. This carried over to every later study without a
  note. Every shape was later re-timed with real activations, which changed 2 of 1166 verdicts; the latencies
  below are the re-timed ones.

## Results

| | |
|---|---|
| R1 | **confirmed, 4/5**. Loses feyn_i29_16: SiLU MLP 0.0155, KAN 0.0160 |
| R2 | **refuted, 3/5** (2/5 on validation). Loses kan_toy2 (fp32 SiLU MLP 0.00137 vs table 0.00145) and feyn_i29_16 (0.0044 vs 0.0092) |

Same on both boards and in both timing modes. Real data: 1 of 5 under R2 (Energy).

Best test RMSE after 30 000 steps, formulas:

| task | SiLU MLP fp32 | SiLU MLP int8 | KAN fp32 | KAN table, 64 points |
|---|---|---|---|---|
| kan_toy2 | 0.0014 | 0.043 | 0.00025 | 0.0014 |
| kan_toy4 | 0.0055 | 0.034 | 0.00057 | 0.0015 |
| feyn_i6_2 | 1.74e-5 | 6.9e-4 | 5.2e-6 | 1.74e-5 |
| feyn_i29_16 | 0.0044 | 0.14 | 0.0051 | 0.0092 |
| feyn_i40_1 | 0.0021 | 0.015 | 0.00090 | 0.0019 |

The fp32 KAN is still 2.3–9.7× more accurate than the best MLP on 4 of 5 formulas, but the 64-point table caps it.
With the int8 activation table the SiLU MLP loses almost all of its accuracy, so the strong baseline is the fp32
SiLU MLP at 15–53 µs. The [fine-table study](../fine_tables) tests finer tables.

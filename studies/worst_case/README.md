# Worst-case error of the chosen models

Registered on 2026-09-28, before any model below was retrained or any error other than the RMSE was computed.
Raw data: `results/raw/worst_case/`.

Every comparison of the paper uses the test RMSE. A firmware specification often bounds the worst error instead,
and the test boxes include extreme combinations of inputs. The RMSE does not say whether the KAN tables, the other
table maps or the sparse grids have heavier tails.

## Registered hypotheses

Split 0; the models chosen on validation RMSE within each budget, as in the paper: the KAN table, the best grid, CP
or GA2M table, and the best sparse grid. Metrics on the 12 000 test points of each model, in the output's units: the
largest absolute error and its 99.9th percentile. For models trained with seeds 0 and 1, each metric is the mean
over the two seeds, as the RMSE is.

- **W1**: on the 8 models with five or more inputs, the KAN table has a smaller largest absolute error than the best
  grid, CP or GA2M table, on at least 6 of the 8 at 32 KB and on at least 6 of the 8 at 128 KB.
- **W2**: the same for the 99.9th percentile of the absolute error.

If W1 and W2 hold, the paper states that the KAN table's advantage over grids, CP and GA2M tables holds for the
worst test error as well. If either fails, the paper states the advantage for the RMSE only and names the models
where the KAN table's worst error is the larger.

Descriptive: the same against the best sparse grid; at 4 and 16 KB; on the four models with four or fewer inputs;
the ratio of the largest error to the RMSE per family; the RMSE and the largest error in the output's units of every
chosen model.

## Setup

- The chosen models are rebuilt from their recorded settings: the networks retrained with the same shape, learning
  rate, seed and code (training is deterministic on the CPU), the grids and sparse grids refitted with their
  recorded penalty. A rebuilt model must reproduce its recorded test RMSE to a relative 1e-6; one that does not is
  left out and reported.
- The largest error over 12 000 test points is a statistic of the sample, not a bound over the box.

## Deviations

- The run did not set `CUDA_VISIBLE_DEVICES=` as the earlier runs did. The models run on the CPU either way, and
  every one reproduced its recorded test RMSE exactly.

## Results

Run on 2026-09-28. Recompute: `python -m kantab.experiments.worst_case frontier` (reads `runs.jsonl`; `run`
rebuilds the models). All 125 chosen models were rebuilt to their recorded test RMSE exactly.

- **W1: confirmed.** The KAN table has the smaller largest absolute test error than the best grid, CP or GA2M table
  on 8 of the 8 models with five or more inputs at 32 KB and on 8 at 128 KB.
- **W2: confirmed.** The same for the 99.9th percentile: 8 of 8 at 32 and at 128 KB.
- At 16 KB the largest error is the smaller on 7 of 8 (the 99.9th percentile on 8): on pv_mp5 the KAN table
  (kan1_w8_g20) reaches 3.7 W against 0.85 W for the CP table, although its RMSE is the lower (0.047 against
  0.066 W). At 4 KB the KAN table's largest error is the smaller on 5 of 8.
- Against the best sparse grid the KAN table has the smaller largest error on 5 of 8 at 32 KB and at 128 KB, and
  the smaller 99.9th percentile on 5 and 3. At 128 KB the sparse grid has the lower RMSE on 6 of the 8 but the
  smaller largest error on 3 (flame5, gasz7, pv_mp5).
- Largest error over RMSE, median over the chosen models: KAN tables 9.2 (4.1–78), grids, CP and GA2M tables 11
  (5.2–32), sparse grids 15 (7.4–70). The 78 is the pv_mp5 KAN table at 16 KB.

The paper states the KAN table's advantage over grids, CP and GA2M tables for the worst test error as well, with
the 16 KB exception on pv_mp5.

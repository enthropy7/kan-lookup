# Does formula structure predict where KAN tables win?

Registered and run on 2026-09-25. Raw data: `results/raw/structure_rule/`.

## Registered hypotheses

A KAN is a sum of one-variable functions. A product of strictly positive factors decomposes into one
(xy = exp(log x + log y)); a product with a factor that changes sign or reaches zero on the domain does not.
Class **B**: the formula contains a product or quotient of two factors, one of which changes sign or reaches zero,
and the factors do not depend on the same single variable. Otherwise class **A**. Classes were assigned before
training (table below).

- **H1**: the class predicts N2 of the [regression study](../regression) for the plain KAN table (A → win, B → no) on ≥ 13 of 16 new formulas
  (P ≈ 0.011 by chance).
- **H2**: RotKAN (a learned linear mixing of the inputs before the tables, so that
  xy = ((x + y)² − (x − y)²)/4) passes N2 on ≥ 6 of the 8 class-B formulas.

## Setup

- 16 AI Feynman formulas not used before, 10 000 noise-free points each, folds 0–2 of 5.
- MLP (SiLU and tanh), KAN, RotKAN: 30 000 steps, lr ∈ {1e-3, 3e-3, 1e-2}, seed 0.
- GBDT and polynomials as in the [regression study](../regression).
- Tables with the 5 % margin of the [range-margin study](../range_margin).
- RotKAN is deployed as an fp32 mixing layer plus interpolated tables.

## Deviations

The rule was derived post hoc on the five formulas of the regression, strong-MLP and fine-table studies, where it fits 5 of 5; hence the test on new
formulas only. Before the criteria were fixed, RotKAN was tried on feyn_i29_16 (0.0040, like the best SiLU MLP;
plain KAN 0.0047), CCPP and Concrete (no gain). Nothing was trained on the new formulas except a 300-step code
check on f_i37_4 and f_i27_6.

## Results

| | |
|---|---|
| H1 | **refuted, 9/16** (8/16 on validation). KAN wins 4/8 class-A and 3/8 class-B formulas |
| H2 | **refuted, 2/8** (3/8 on validation) |

The validation count was 7/16 before the re-timing of 2026-09-26. The re-timed SiLU MLP flipped f_ii6_15a.

| formula | class | best alternative, RMSE, A53 | best KAN table | fastest KAN table ≤ T |
|---|---|---|---|---|
| f_i27_6: 1 / (1/d1 + n/d2) | A | SiLU MLP 2.0e-4, 52.5 µs | 7.7e-5 | 0.54 µs |
| f_i34_1: ω0 / (1 − v/c) | A | SiLU MLP 1.5e-3, 27.4 µs | 9.5e-4 | 1.79 µs |
| f_ii6_15a: 3pz√(x² + y²) / (4πεr⁵) | A | SiLU MLP 3.3e-3, 53.0 µs | 2.8e-3 | 0.92 µs |
| f_i24_6: m(ω² + ω0²)x² / 4 | A | poly⁵ 1.8e-3, 1.3 µs | 0.096 | — |
| f_ii35_18: n0 / (e^(μB/kT) + e^(−μB/kT)) | A | SiLU MLP 9.9e-4, 52.9 µs | 3.6e-4 | 0.62 µs |
| f_i41_16: hω³ / (π²c²(e^(hω/kT) − 1)) | A | SiLU MLP 0.0166, 52.9 µs | 0.0175 | — |
| f_i8_14: √((x2 − x1)² + (y2 − y1)²) | A | SiLU MLP 1.4e-3, 52.4 µs | 1.7e-3 | — |
| f_i11_19: x1y1 + x2y2 + x3y3 | A | poly⁴ 7.3e-5, 2.1 µs | 3.1e-3 | — |
| f_i12_11: q(Ef + Bv sin θ) | B | SiLU MLP 0.050, 52.9 µs | 0.027 | 0.58 µs |
| f_i13_12: Gm1m2(1/r2 − 1/r1) | B | SiLU MLP 0.040, 52.9 µs | 0.070 | — |
| f_i26_2: arcsin(n sin θ2) | B | tanh MLP 8.3e-4, 52.7 µs | 3.9e-4 | 0.50 µs |
| f_i37_4: I1 + I2 + 2√(I1I2) cos δ | B | SiLU MLP 1.9e-3, 52.5 µs | 8.7e-4 | 0.88 µs |
| f_i44_4: nkT ln(V2/V1) | B | poly⁶ 0.036, 4.5 µs | 0.146 | — |
| f_i50_26: x1(cos ωt + α cos² ωt) | B | SiLU MLP 6.2e-3, 52.4 µs | 7.9e-3 | — |
| f_iii9_52: (pEf t / ħ) sinc²((ω − ω0)t/2) | B | SiLU MLP 0.044, 53.0 µs | 0.057 | — |
| f_ii2_42: κ(T2 − T1)A / d | B | SiLU MLP 0.037, 52.9 µs | 0.049 | — |

Plain or rotated, a KAN table is more accurate than the best alternative on 8 of 16 formulas. The three large
losses (4–53×) are to polynomials, and two of those formulas are polynomials themselves. "Low-degree polynomial →
polynomial regression, otherwise KAN table" is a post-hoc hypothesis, not a result.

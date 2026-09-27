# Timing on an ESP32

Measured on 2026-09-26, after the re-timing. Raw data: `results/raw/esp32/` (board output) and
`results/raw/{pv_maps,physics_maps,input_rule}/latency_esp32*_shapes.txt` (ns at 240 MHz, in the format of `bench`).

The MLPs here run on our own kernels; the [vendor-kernel study](../vendor_kernels) re-times them with esp-dsp and
ESP-NN. This is a descriptive measurement without criteria, the MCU supplement announced in the [PV-map study](../pv_maps). No verdict of
the map studies depends on it.

## Setup

- Boards: LilyGO T8 (ESP32-WROVER-E, ESP32-D0WD-V3 rev 3.1, 240 MHz, 4 MB flash) and, as a repeat, a WROOM-32
  with flash from another vendor. PSRAM is unused.
- ESP-IDF 5.5.2 from `nixpkgs-esp-dev`, pinned in `mcu/esp32/flake.nix`; `-O2`, flash QIO 80 MHz.
- `mcu/common/mcubench.c` holds the fast paths of `c/kernels.c`; `c/maps.c` and `c/baselines.c` are compiled in
  unchanged. fp32 SiLU/tanh use the Cephes exp polynomial, int8 SiLU/tanh the activation table.
- Weights and tables are shared pseudo-random const arrays: latency depends on the shape, not the values.
- Cycle counter, 31 samples of 16 calls, median.
- Placement: `esp32` reads the model from flash through the flash cache, as const data in firmware. `esp32ram`
  first copies it to internal RAM when it fits in the largest free block (139 264 B); otherwise the flash time is
  kept (ten two-layer KANs of 190+ KB, above the 128 KB budget).
- Shapes: all 793 shapes of the three physics-map studies except nearest-entry int8 KAN tables, RotKAN and GBDT.

## Checks

- The checksum of the outputs on 64 inputs matches the host for all 793 shapes, on both boards and in both
  placements.
- The two boards agree to the cycle: ratio 1.000 at the 5th and 95th percentiles.
- A second run on the T8 with a larger, different data blob: median ratio 1.000 (0.988–1.018).
- Flash over RAM, median (max): fp32 MLP 1.07 (5.4), interpolated KAN tables 1.06 (2.2), grids 1.0 (1.5), CP
  1.04, int8 MLP 1.04.

## Results

"Best MLP" is the fp32 MLP with the lowest validation RMSE, almost always 3×128 SiLU (133 KB). "Grid target" is
the accuracy of the best 128 KB anisotropic grid.

| task | best MLP, RAM | fastest KAN ≤ its error, RAM | ratio (flash) | 128 KB grid, RAM | KAN to grid target | MLP to grid target | ratio (flash) | CP to grid target |
|---|---|---|---|---|---|---|---|---|
| pv_mp5 | 1 506 µs | 9.2 µs | 163 (705) | 14.2 µs | 6.3 µs | 822 µs | 131 (549) | 6.3 µs |
| pv_chain6 | 1 510 µs | 17.5 µs | 86 (350) | 29.2 µs | 17.5 µs | 483 µs | 28 (49) | never |
| tyre4 | 1 502 µs | 24.7 µs | 61 (229) | 7.2 µs | 24.7 µs | never | — | never |
| msis7 | 1 514 µs | 39.5 µs | 38 (125) | 61.2 µs | 7.6 µs | 52.3 µs | 6.9 (7.1) | 15.0 µs |
| flame5 | 1 506 µs | 15.1 µs | 100 (406) | 14.2 µs | 9.2 µs | 173 µs | 19 (19) | 11.4 µs |
| gasz7 | 1 514 µs | 17.6 µs | 86 (365) | 61.2 µs | 10.8 µs | 45.3 µs | 4.2 (4.1) | 15.0 µs |
| ign5 | 1 506 µs | 15.1 µs | 100 (426) | 14.2 µs | 9.2 µs | 43.7 µs | 4.7 (4.9) | never |
| igrf4 | 1 502 µs | 24.7 µs | 61 (257) | 7.2 µs | never | never | — | never |
| msis_d3 | 1 497 µs | 12.6 µs | 119 (513) | 3.9 µs | never | never | — | never |
| msis_d4 | 1 167 µs | 16.7 µs | 70 (277) | 7.2 µs | never | never | — | never |
| msis_d5 | 1 506 µs | 26.9 µs | 56 (229) | 14.2 µs | 26.9 µs | 1 506 µs | 56 (229) | never |
| msis_d6 | 1 510 µs | never | — | 29.2 µs | 11.1 µs | 170 µs | 15 (13) | 13.2 µs |

Table representations run in 4–90 µs; the best MLP takes 1.5 ms from RAM and 6.8 ms from flash. A KAN table with
no larger error is 38–163× faster (RAM) and 125–705× faster (flash); on the A53 with NEON the same gap is 30–91×.
At the grid target the KAN table is 4.2–131× faster than the fastest MLP that reaches it. Grids on 7 inputs are
slow (128 corner reads, 61–91 µs); on 3–4 inputs they are the fastest and the most accurate. ESP-NN was not
tried here, and energy was not measured; the [vendor-kernel study](../vendor_kernels) later timed every MLP with
esp-dsp and ESP-NN, and the paper uses those times.

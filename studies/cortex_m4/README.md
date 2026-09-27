# Timing on a Cortex-M4

## Registered hypotheses

The ESP32 has no SIMD; the Cortex-M4F has a single-precision FPU and DSP instructions, and CMSIS-NN and CMSIS-DSP
are written for it. Here every shape of the PV-map, physics-map and input-rule studies is timed on an STM32F4 Black
Pill: KAN tables and table maps with our kernels, MLPs with our kernels and with CMSIS (CMSIS-DSP for fp32, CMSIS-NN
for int8). Each MLP takes the faster of the two times. The model stays in flash, as firmware keeps it; RAM copies are
timed where they fit and are descriptive.

- **M1**: at the error of the most accurate MLP, the fastest KAN table with no larger error is at least 10× faster
  than that MLP, on at least 8 of the 10 models where the KAN table reaches that error on every split.
- **M2**: at the error of the best 128 KB anisotropic grid (split 0), the fastest KAN table is faster than the
  fastest MLP on at least 6 of the 8 models where both reach it.

If M1 or M2 fails, the paper reports the Cortex-M4 comparison as measured and drops the claim that fails.

## Setup

- STM32F401CE or STM32F411CE (read from the device ID), 84 MHz from the 25 MHz crystal, flash with prefetch and
  both caches, DWT cycle counter, median of 31 samples of 16 calls.
- `mcu/common/mcubench.c` with the same weights and inputs as on the ESP32; CMSIS-DSP `arm_dot_prod_f32` per output
  and `arm_mat_vec_mult_f32`, CMSIS-NN fully connected s8 with the int8 activation table.
- Each kernel's checksum is compared with a host build of the same C code.
- Flashing through the USB bootloader; results come back over USB serial.

## Deviations

- The board is an STM32F411CE (device ID 0x431, 512 KB flash, 128 KB RAM). The firmware links for 96 KB of RAM,
  as for an F401, which leaves a largest free block of 57 KB for the RAM copies.
- The system bootloader corrupted flash when it programmed it over USB on this board: command bytes landed in flash
  as data. It writes RAM correctly, so it only loads a flash writer into RAM (`mcu/stm32/loader.c`), which erases
  and programs flash and checks every 4 KB chunk and then the whole image by CRC-32 read back from flash.
- The first builds hung at the first output line. libopencm3 (2da12dc) NAKs an IN endpoint when it handles a
  completed packet, and a packet written before that stays NAKed; `_write` now waits for the completion callback.
  GCC had also removed the unused `malloc` in the search for the largest free block. A 30 s watchdog, refreshed on
  output, restarts the board after a hang, and `b` over USB reboots it into the bootloader. None of this runs
  inside a timed region, and no timing from the hung builds was kept.

## Results

Run on 2026-09-28: every one of the 1333 shapes from flash, and from RAM where the model fits in 57 KB (152 do
not). Recompute from the raw output:

```
python -m kantab.experiments.cortex_m4 convert results/raw/cortex_m4/stm32_output_rerun.txt \
    results/raw/cortex_m4/host_checksums.txt --vendor-from results/raw/cortex_m4/stm32_output.txt
python -m kantab.experiments.cortex_m4 frontier
```

Checks: every checksum, CMSIS-DSP and CMSIS-NN included, equals the host build of the same C code to all printed
digits, except one RAM copy. For `poly:7:6` in RAM an unchecked `malloc` in `poly_init` failed on the full heap and
the polynomial read garbage; that time is dropped, and a time counts only where its checksum matches.

CMSIS was faster than our kernels on 357 of 360 MLP shapes in flash: by a median of 1.25× for fp32 (CMSIS-DSP) and
2.09× for int8 (CMSIS-NN). Every MLP time below is the faster of the two.

- **M1: confirmed.** With the model in flash, at the error of the most accurate MLP the KAN table is 38–123× faster
  on 10 of 10 models: flame5 73×, gasz7 45×, ign5 62×, igrf4 67×, msis_d3 86×, msis_d4 45×, msis_d5 43×,
  pv_chain6 38×, pv_mp5 123×, tyre4 43×. As on the other boards, on msis7 and msis_d6 no KAN table reaches that
  error on every split.
- **M2: confirmed.** At the error of the best 128 KB anisotropic grid the KAN table is faster than the fastest MLP
  on 8 of 8 models, from 2.8× (gasz7) to 103× (pv_mp5).
- In RAM (descriptive) the most accurate MLPs, 135 KB each, do not fit, so the reference is the most accurate MLP
  that does (3×64, SiLU): the KAN table is 9.6–36× faster on 11 models, and 2.7–19× faster at the grid target on
  6 of 6.

## Addendum: CMSIS built as shipped

Registered on 2026-09-28, before the rerun. The run above compiled CMSIS-DSP without `ARM_MATH_LOOPUNROLL`, which
CMSIS-DSP's own build enables, and all of CMSIS at `-O2`. The whole benchmark runs again with CMSIS-DSP and
CMSIS-NN at `-O3` with loop unrolling; our kernels keep `-O2`. Each MLP takes the fastest of our kernel and the
CMSIS variants of either build; all other times come from the rerun.

- Expected: M1 and M2 keep their verdicts, and the M1 range moves by less than 20 %. Our kernels repeat the first
  run within 2 % at the median.

If a verdict flips, the paper reports the rerun and drops the claim that fails.

Results, 2026-09-28: our kernels repeated the first run to the cycle on all 1494 timings. CMSIS-DSP's matrix-vector
product, the fastest fp32 kernel on 177 of 180 shapes, did not change by a cycle; its dot product per output got
3.4 % faster at the median but up to 15 % slower on some shapes, and CMSIS-NN 0.6 % faster. With the faster CMSIS
build per shape, CMSIS beats our kernels by a median of 1.25× (fp32) and 2.11× (int8), and M1 (10/10, 38–123×)
and M2 (8/8, 2.8–103×) are unchanged. The same `poly:7:6` RAM copy failed its checksum and is dropped.

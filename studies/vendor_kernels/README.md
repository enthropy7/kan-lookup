# MLPs with vendor kernels

## Registered hypotheses

The latency comparisons of the PV-map, physics-map and input-rule studies timed every MLP with our own kernels:
NEON loops on the Cortex-A53, plain C on the ESP32. Vendor libraries may be faster. Here every MLP shape of those
studies (360 shapes, fp32 and int8, ReLU, SiLU and tanh) is timed again with vendor kernels, and each MLP point
takes the faster of the two timings.

- **V1**: at the error of the most accurate MLP, the fastest KAN table with no larger error is at least 10× faster
  than that MLP, on at least 8 of the 10 models where the KAN table reaches that error on every split, on the
  Cortex-A53 and on the ESP32 with the model in RAM.
- **V2**: at the error of the best 128 KB anisotropic grid (split 0), the fastest KAN table is faster than the
  fastest MLP on at least 6 of the 8 models where both reach it, on each of the two boards.

If V1 or V2 fails, the paper reports the latency comparison with vendor kernels and drops the claim that fails.
Either way, all latency figures of the paper are recomputed with the faster MLP timing.

## Setup

- Cortex-A53 (Orange Pi Zero 3): XNNPACK fully connected operators, fp32 (`f32`) and int8 with per-channel weights
  (`qs8_qc8w`), batch 1, one thread. ReLU is fused into the operator; SiLU and tanh run as XNNPACK unary operators
  in fp32 and as our 255-entry table in int8.
- ESP32: esp-dsp for fp32 (the assembly dot product for this core, one call per output) and ESP-NN for int8
  (fully connected layer). Activations are the same code as in our kernels. Data in flash and copied to RAM, as
  before.
- Timing as before: median of single calls. Each vendor path is checked against our kernel on the same weights.
- The KAN tables and table maps keep their measured latencies.

## Deviations

- XNNPACK (commit 98252f2) was built on the Orange Pi with gcc 13: the zig cross build rejects the per-file
  `-march` flags of XNNPACK. esp-dsp 1.8.2 and ESP-NN 1.4.1 come from the ESP-IDF component registry.
- On the original ESP32, ESP-NN's fully connected layer is its generic C version; its optimized kernels target
  later chips.
- Besides the registered dot product per output, esp-dsp's matrix-vector product was timed too; each MLP takes the
  faster of the two, which can only favour the MLP.
- The int8 SiLU and tanh on both boards use a 256-entry int8-to-int8 table, as TensorFlow Lite Micro does, instead
  of our 255-entry table with float output.
- The first ESP32 run was stopped early, before any analysis: the int8 check gave an all-zero checksum
  because the requantization shift was too large. The shift now scales with the layer width, and the run was
  restarted.

## Results

| | Cortex-A53 | ESP32, RAM |
|---|---|---|
| V1: KAN ≥ 10× faster at the best MLP's error | **confirmed, 10/10** (20.8–55.8×) | **confirmed, 10/10** (33.5–104.7×) |
| V2: KAN faster at the grid target | **confirmed, 8/8** (2.2–36.8×) | **confirmed, 8/8** (3.7–87.2×) |

Our kernel time over the vendor kernel time, 360 MLP shapes:

| board | vendor library | median | max | vendor faster on |
|---|---|---|---|---|
| Cortex-A53 | XNNPACK | 1.27 | 1.86 | 290 |
| Cortex-A7 | XNNPACK | 1.73 | 2.94 | 290 |
| ESP32, RAM | esp-dsp, ESP-NN | 1.21 | 2.00 | 330 |

The Cortex-A7 was not registered and is descriptive.

With the faster MLP timing, the KAN table reaches the error of the most accurate MLP 21–56× faster on the
Cortex-A53 (before: 32–84×), 24–78× on the Cortex-A7 (54–159×) and 33–105× on the ESP32
with the model in RAM (52–163×).

Checks:
- XNNPACK fp32 outputs match a plain C reference to 3e-8 on every shape.
- On the ESP32, the checksums of our kernels and of ESP-NN equal a host build of the same code bit for bit; esp-dsp is
  within 1.3e-6.
- Our kernels in this run reproduce the earlier ESP32 timings: median ratio 0.99, range 0.98–1.01.

## Addendum: XNNPACK set up once

Registered on 2026-09-28, before the run. The runs above call `xnn_setup_*` for every operator on every call. With
fixed buffers a deployment binds them once and then only runs the operators. Every MLP shape is timed again on the
Cortex-A53 and the Cortex-A7 with the operators set up once, outside the timed call, which copies the input into the
bound buffer and runs the operators; setup on every call is timed again in the same session. Each MLP takes the
fastest of our kernel and both XNNPACK modes.

- Expected: setting up once is faster by less than 15 % at the median, and V1 and V2 keep their verdicts on the
  Cortex-A53.

If a verdict flips, the paper reports the comparison with setup once and drops the claim that fails.

Results, 2026-09-28: setting up once was faster at the median by 5.1 % on the Cortex-A53 (at most 16 %, faster on
337 of 360 shapes) and by 1.3 % on the Cortex-A7; setup on every call repeated the first run within 1.6 % at the
median. Both verdicts hold on the Cortex-A53: V1 10/10 (20.7–55.6×), V2 8/8 (2.1–36.7×). The ESP32 is unchanged.
Every fp32 output still matches the reference within 5e-9.

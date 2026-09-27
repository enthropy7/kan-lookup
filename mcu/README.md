# Microcontroller benchmarks

The table kernels of `c/` built into firmware for an ESP32 and an STM32F411 (Cortex-M4F). `common/mcubench.c` holds
the kernels with weights in const arrays; `c/maps.c` and `c/baselines.c` are compiled in unchanged. Each shape is
timed with its data in flash and, when it fits, copied into internal RAM. Both builds also run the MLPs on vendor
kernels: `f32v` (dot product per output), `f32m` (matrix-vector product) and `mlpv` (int8 fully connected), on
esp-dsp and ESP-NN for the ESP32 and on CMSIS-DSP and CMSIS-NN for the STM32; `common/gen_specs.py --vendor` lists
every MLP shape with both.

```
cd mcu/common && nix develop ../.. --command python gen_blob.py
cc -O2 -ffp-contract=off -I../../c -o host_test host_test.c mcubench.c blob.c ../../c/{kernels,maps,baselines}.c -lm && ./host_test

cd ../esp32
nix develop --command python ../common/gen_specs.py --vendor --out main/specs.c
nix develop --command idf.py set-target esp32 build
nix develop --command python run.py --port /dev/ttyACM0 --out results.txt

cd ../stm32
nix develop --command make flash
nix develop --command python run.py --out results.txt
```

The sparse grids of `studies/sparse_budgets` (modified basis, split 0) are compiled into the ESP32 firmware
and sent to the STM32 over USB, one spec per line:

```
cd mcu/esp32
nix develop --command python ../common/gen_specs.py --sgrid --out main/specs.c
nix develop --command idf.py build
nix develop --command python run.py --port /dev/ttyACM0 --out ../../results/raw/sparse_budgets/esp32_sgrid_output.txt

cd ../stm32
nix develop --command python ../common/gen_specs.py --sgrid --out sgrid_specs.txt
nix develop --command python run.py --specs sgrid_specs.txt --out ../../results/raw/sparse_budgets/stm32_sgrid_output.txt
```

`host_test` checks the MCU kernels against `c/` on the host: equal bit for bit, except fp32 SiLU and tanh
(within 4e-8).

The STM32 needs only its USB port. Hold BOOT0 and press NRST once to enter the ROM bootloader; `make flash` loads a
flash writer into RAM through it and programs flash with a CRC check, because the ROM bootloader's own flash writes
were corrupt on our board. The firmware then reboots into the bootloader on `b`, so later flashes need no buttons,
and a watchdog restarts it after 30 s without output.

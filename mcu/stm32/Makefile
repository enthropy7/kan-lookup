CC = arm-none-eabi-gcc
ARCH = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
CFLAGS = $(ARCH) -O2 -std=gnu11 -ffp-contract=off -ffunction-sections -fdata-sections -Wall -DSTM32F4 \
         -DMCU_VENDOR_CMSIS -I. -I../common -I../../c -I$(OPENCM3)/include -I$(CMSIS_NN)/Include \
         -I$(CMSIS_DSP)/Include -I$(CMSIS_DSP)/PrivateInclude -I$(CMSIS_CORE)/Include
LDFLAGS = $(ARCH) -T stm32f4.ld -L$(OPENCM3)/lib -nostartfiles -Wl,--gc-sections --specs=nano.specs --specs=nosys.specs -u _printf_float
LIBS = -lopencm3_stm32f4 -lc -lm
SRC = main.c specs.c ../common/mcubench.c ../common/blob.c ../../c/baselines.c
CMSIS_SRC = $(CMSIS_NN)/Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s8.c \
            $(CMSIS_NN)/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_per_ch_s8.c \
            $(CMSIS_DSP)/Source/BasicMathFunctions/arm_dot_prod_f32.c \
            $(CMSIS_DSP)/Source/MatrixFunctions/arm_mat_vec_mult_f32.c
# CMSIS as its own build ships it: -O3 and loop unrolling
CMSIS_CFLAGS = $(subst -O2,-O3,$(CFLAGS)) -DARM_MATH_LOOPUNROLL

all: build/fw.bin build/loader.bin

specs.c: ../common/gen_specs.py
	python ../common/gen_specs.py --vendor --out specs.c

build/cmsis.a: $(CMSIS_SRC)
	mkdir -p build/cmsis
	for f in $(CMSIS_SRC); do $(CC) $(CMSIS_CFLAGS) -c $$f -o build/cmsis/$$(basename $$f .c).o || exit 1; done
	arm-none-eabi-ar rcs $@ build/cmsis/*.o

# the map kernels, sparse grids among them, at -O3 as registered for the sparse-grid timing
build/maps.o: ../../c/maps.c ../../c/maps.h
	mkdir -p build
	$(CC) $(subst -O2,-O3,$(CFLAGS)) -c ../../c/maps.c -o build/maps.o

build/fw.bin: $(SRC) build/maps.o build/cmsis.a ../common/mcubench.h boot.h stm32f4.ld
	mkdir -p build
	$(CC) $(CFLAGS) -o build/fw.elf $(SRC) build/maps.o build/cmsis.a $(LDFLAGS) $(LIBS)
	arm-none-eabi-size build/fw.elf
	arm-none-eabi-objcopy -O binary build/fw.elf build/fw.bin

build/loader.bin: loader.c loader.ld boot.h
	mkdir -p build
	$(CC) $(ARCH) -Os -std=gnu11 -ffunction-sections -Wall -DSTM32F4 -I. -I$(OPENCM3)/include -o build/loader.elf loader.c \
	    -T loader.ld -L$(OPENCM3)/lib -nostartfiles -Wl,--gc-sections --specs=nano.specs --specs=nosys.specs -lopencm3_stm32f4
	arm-none-eabi-size build/loader.elf
	arm-none-eabi-objcopy -O binary build/loader.elf build/loader.bin

flash: build/fw.bin build/loader.bin
	python flash.py build/loader.bin build/fw.bin

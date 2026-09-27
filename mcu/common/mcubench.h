#ifndef MCUBENCH_H
#define MCUBENCH_H
#include <stdint.h>
#include <stddef.h>

extern const int16_t BLOB16[];
extern const float BLOBF[];
extern const int8_t BLOB8[];
extern const size_t BLOB16_N, BLOBF_N, BLOB8_N;

uint32_t mcu_cycles(void);

int mcu_bench(const char *spec, int samples, int inner);
int mcu_bench_ram(const char *spec, int samples, int inner, size_t max_bytes);
int mcu_eval(const char *spec, const float *x, int n, float *y);
#endif

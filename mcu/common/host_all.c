#include <stdio.h>
#include <time.h>
#include "mcubench.h"
extern const char *const SPECS[];
extern const int NSPECS;
uint32_t mcu_cycles(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint32_t)(t.tv_sec * 1000000000ull + t.tv_nsec); }
int main(void) { for (int i = 0; i < NSPECS; i++) mcu_bench(SPECS[i], 3, 1); return 0; }

/* Times sparse grids (maps.c): each argument is NAME:d:basis:levels, levels being d hex digits per level vector;
   surpluses are random. mod grids are timed with the planned kernel (sgrid:) and the fast one (sgridfast:). Prints
   the median of single calls, as bench.c does. */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "maps.h"

static uint32_t rng = 12345;
static uint32_t next(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }
static float unif(void) { return (float)next() / (float)(1u << 24) * 2.0f - 1.0f; }

static int64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}

static int cmp64(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *rep = getenv("BENCH_REPEAT");
    int repeat = rep ? atoi(rep) : 20000;
    int64_t *t = malloc(sizeof(int64_t) * repeat);
    for (int a = 1; a < argc; a++) {
        char name[256], basis[8];
        int d = 0, used = 0;
        if (sscanf(argv[a], "%255[^:]:%d:%7[^:]:%n", name, &d, basis, &used) < 3 || d < 1 || d > 16) {
            fprintf(stderr, "bad spec %s\n", argv[a]);
            return 2;
        }
        const char *hex = argv[a] + used;
        const int s = (int)strlen(hex) / d, bound = !strcmp(basis, "bound");
        uint8_t *levels = malloc((size_t)s * d);
        size_t points = 0;
        for (int k = 0; k < s; k++) {
            size_t size = 1;
            for (int j = 0; j < d; j++) {
                const char c = hex[k * d + j];
                levels[k * d + j] = (uint8_t)(c <= '9' ? c - '0' : c - 'a' + 10);
                size *= levels[k * d + j] == 0 ? 2 : (size_t)1 << (levels[k * d + j] - 1);
            }
            points += size;
        }
        int16_t *v = malloc(sizeof(int16_t) * points);
        float *scale = malloc(sizeof(float) * s);
        for (size_t i = 0; i < points; i++) v[i] = (int16_t)((int)(next() % 65535) - 32767);
        for (int k = 0; k < s; k++) scale[k] = 1e-5f;
        SGridMap m = {d, s, bound, levels, scale, v};
        SGridPlan plan;
        const int planned = sgrid_plan(&m, &plan) == 0;
        float *xs = malloc(sizeof(float) * 256 * d);
        for (int i = 0; i < 256 * d; i++) xs[i] = unif();
        SGridFast fast;
        const int quick = planned && sgrid_fast(&m, &fast) == 0;
        for (int kernel = 0; kernel < 1 + quick; kernel++) {
            volatile float sink = 0.0f;
            for (int r = -1000; r < repeat; r++) {
                const float *x = xs + (size_t)((r + 1000) % 256) * d;
                int64_t t0 = now_ns();
                sink += kernel ? sgrid_forward_fast(&fast, x) : planned ? sgrid_forward_plan(&plan, x) : sgrid_forward(&m, x);
                if (r >= 0) t[r] = now_ns() - t0;
            }
            qsort(t, repeat, sizeof(int64_t), cmp64);
            printf("spec %s:%s median_ns %lld p10_ns %lld p90_ns %lld\n", kernel ? "sgridfast" : "sgrid", name,
                   (long long)t[repeat / 2], (long long)t[repeat / 10], (long long)t[repeat * 9 / 10]);
            (void)sink;
        }
        free(levels); free(v); free(scale); free(xs);
    }
    return 0;
}

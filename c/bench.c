#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernels.h"
#include "maps.h"
#include "baselines.h"

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
    const char *rep = getenv("BENCH_REPEAT"), *inn = getenv("BENCH_INNER");
    int repeat = rep ? atoi(rep) : 20000, inner = inn ? atoi(inn) : 1;
    int64_t *t = malloc(sizeof(int64_t) * repeat);
    for (int a = 1; a < argc; a++) {
        char spec[512];
        snprintf(spec, sizeof spec, "%s", argv[a]);
        char *colon = strchr(spec, ':');
        if (!colon) { fprintf(stderr, "bad spec %s\n", argv[a]); return 2; }
        *colon = 0;

        const char *kinds[] = {"kan", "mlp", "kan16", "lin16", "f32", "poly", "rot16", "grid", "cp", "ga2m", "gridv"};
        int kind = -1, q = 16, dims[16], n = 0;
        for (int k = 0; k < 11; k++) if (!strcmp(spec, kinds[k])) kind = k;
        GridMap gm = {0};
        GridVMap gvm = {0};
        CPMap cpm = {0};
        GA2MMap gam = {0};
        if (kind >= 7) {
            int a1 = 0, a2 = 0, a3 = 0;
            sscanf(colon + 1, "%d:%d:%d", &a1, &a2, &a3);
            size_t cells = 1;
            if (kind == 7) for (int k = 0; k < a1; k++) cells *= (size_t)a2;
            if (kind == 10) {
                const char *c = strchr(colon + 1, ':') + 1;
                gvm.d = a1;
                for (int k = 0; k < a1; k++) { gvm.n[k] = (int)strtol(c, (char **)&c, 10); c++; cells *= (size_t)gvm.n[k]; }
            }
            if (kind == 8) cells = (size_t)a1 * a2 * a3;
            if (kind == 9) cells = (size_t)a1 * a2 + (size_t)a1 * (a1 - 1) / 2 * a3 * a3;
            int16_t *t = malloc(sizeof(int16_t) * cells);
            for (size_t i = 0; i < cells; i++) t[i] = (int16_t)((int)(next() % 65535) - 32767);
            float *ones = malloc(sizeof(float) * 256);
            for (int i = 0; i < 256; i++) ones[i] = 1e-4f;
            if (kind == 7) gm = (GridMap){a1, a2, 1e-4f, t};
            if (kind == 10) { gvm.scale = 1e-4f; gvm.v = t; }
            if (kind == 8) cpm = (CPMap){a1, a2, a3, 0.0f, ones, ones, t};
            if (kind == 9) gam = (GA2MMap){a1, a2, a3, 0.0f, ones, ones, t, t + (size_t)a1 * a2};
            dims[0] = a1; dims[1] = 1; n = 2;
        }
        if (kind < 0) { fprintf(stderr, "bad spec %s\n", argv[a]); return 2; }
        char *body = colon + 1;
        colon = strchr(body, ':');
        if (colon) { *colon = 0; q = atoi(colon + 1); }
        int act = ACT_RELU;
        if (colon && (kind == 1 || kind == 4))
            act = !strcmp(colon + 1, "silu") ? ACT_SILU : !strcmp(colon + 1, "tanh") ? ACT_TANH : ACT_RELU;
        float table[255];
        for (int t = 0; t < 255; t++) {
            float v = (float)(t - 127) * 0.05f;
            table[t] = act == ACT_SILU ? v / (1.0f + expf(-v)) : tanhf(v);
        }
        if (kind == 5) { dims[0] = atoi(body); dims[1] = 1; n = 2; }
        else if (kind >= 7) {}
        else for (char *tok = strtok(body, "-"); tok && n < 16; tok = strtok(NULL, "-")) dims[n++] = atoi(tok);
        if (n < 2) { fprintf(stderr, "bad spec %s\n", argv[a]); return 2; }
        int lutk = kind == 0 || kind == 2 || kind == 3 || kind == 6;
        int layers = n - 1, widest = 0;
        for (int k = 0; k < n; k++) widest = dims[k] > widest ? dims[k] : widest;
        LutLayer *lut = calloc(layers, sizeof(LutLayer));
        Q8Layer *q8 = calloc(layers, sizeof(Q8Layer));
        F32Layer *f32 = calloc(layers, sizeof(F32Layer));
        Poly poly = {0};
        if (kind == 5) {
            int d = dims[0], cap = 1 << 16, terms = 1, start = 0;
            uint8_t (*mono)[16] = calloc(cap, 16);
            uint8_t *var = calloc(cap, 1);
            uint16_t *parent = calloc(cap, sizeof(uint16_t));
            float *coef = malloc(sizeof(float) * cap);
            for (int g = 1; g <= q; g++) {
                int end = terms;
                for (int t = start; t < end; t++)
                    for (int v = g == 1 ? 0 : mono[t][g - 2]; v < d && terms < cap; v++) {
                        memcpy(mono[terms], mono[t], 16);
                        mono[terms][g - 1] = (uint8_t)v;
                        var[terms] = (uint8_t)v; parent[terms] = (uint16_t)t;
                        terms++;
                    }
                start = end;
            }
            for (int t = 0; t < terms; t++) coef[t] = unif() * 1e-2f;
            poly_init(&poly, terms, parent, var, coef);
        }
        for (int k = 0; k < layers && kind != 5 && kind < 7; k++) {
            int in = dims[k], out = dims[k + 1];
            float *lo = malloc(sizeof(float) * in), *hi = malloc(sizeof(float) * in);
            float *scale = malloc(sizeof(float) * out), *bias = malloc(sizeof(float) * out);
            for (int i = 0; i < in; i++) { lo[i] = -1.0f; hi[i] = 1.0f; }
            for (int o = 0; o < out; o++) { scale[o] = 1e-3f; bias[o] = 0.0f; }
            const int mix = kind == 6 && k == 0;
            size_t nt = lutk && !mix ? (size_t)out * in * q : (size_t)out * in;
            if (mix) {
                float *w = malloc(sizeof(float) * nt);
                for (size_t i = 0; i < nt; i++) w[i] = unif() * 0.3f;
                f32_init(&f32[k], in, out, 0, w, bias);
            } else if (kind <= 1) {
                int8_t *w = malloc(nt);
                for (size_t i = 0; i < nt; i++) w[i] = (int8_t)((int)(next() % 255) - 127);
                if (kind == 0) lut_init(&lut[k], in, out, q, lo, hi, scale, w);
                else {
                    q8_init(&q8[k], in, out, k + 1 < layers ? act : ACT_NONE, 1.0f / 127.0f, scale, w, bias);
                    if (k + 1 < layers && act >= ACT_SILU) q8_set_table(&q8[k], 0.05f, table);
                }
            } else if (lutk) {
                int16_t *w = malloc(sizeof(int16_t) * nt);
                for (size_t i = 0; i < nt; i++) w[i] = (int16_t)((int)(next() % 65535) - 32767);
                lut16_init(&lut[k], in, out, q, lo, hi, scale, w);
            } else {
                float *w = malloc(sizeof(float) * nt);
                for (size_t i = 0; i < nt; i++) w[i] = unif() * 0.1f;
                f32_init(&f32[k], in, out, k + 1 < layers ? act : ACT_NONE, w, bias);
            }
        }
        float *xs = malloc(sizeof(float) * 256 * dims[0]);
        for (int i = 0; i < 256 * dims[0]; i++) xs[i] = unif();
        float *h0 = malloc(sizeof(float) * widest), *h1 = malloc(sizeof(float) * widest);
        volatile float sink = 0.0f;
        for (int r = -1000; r < repeat; r++) {
            int64_t t0 = now_ns();
            for (int j = 0; j < inner; j++) {
                const float *x = xs + (size_t)(((r + 1000) * inner + j) % 256) * dims[0];
                const float *h = x;
                if (kind == 5) { h0[0] = poly_forward(&poly, x); h = h0; }
                else if (kind == 7) { h0[0] = grid_forward(&gm, x); h = h0; }
                else if (kind == 10) { h0[0] = gridv_forward(&gvm, x); h = h0; }
                else if (kind == 8) { h0[0] = cp_forward(&cpm, x); h = h0; }
                else if (kind == 9) { h0[0] = ga2m_forward(&gam, x); h = h0; }
                else for (int k = 0; k < layers; k++) {
                    float *y = k % 2 ? h1 : h0;
                    if (kind == 0) lut_forward_fast(&lut[k], h, y);
                    else if (kind == 1) q8_forward_fast(&q8[k], h, y);
                    else if (kind == 2) lut16_forward_fast(&lut[k], h, y);
                    else if (kind == 3 || (kind == 6 && k > 0)) lin16_forward_fast(&lut[k], h, y);
                    else f32_forward_fast(&f32[k], h, y);
                    h = y;
                }
                sink += h[0];
            }
            if (r >= 0) t[r] = now_ns() - t0;
        }
        qsort(t, repeat, sizeof(int64_t), cmp64);
        printf("spec %s median_ns %lld p10_ns %lld p90_ns %lld\n", argv[a], (long long)(t[repeat / 2] / inner),
               (long long)(t[repeat / 10] / inner), (long long)(t[repeat * 9 / 10] / inner));
        (void)sink;
    }
    return 0;
}

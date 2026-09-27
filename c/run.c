#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernels.h"
#include "maps.h"
#include "baselines.h"

static unsigned char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(len ? len : 1);
    if (fread(b, 1, len, f) != len) exit(1);
    fclose(f);
    return b;
}

static void *take(const unsigned char **p, size_t bytes) {
    void *v = malloc(bytes ? bytes : 1);
    memcpy(v, *p, bytes);
    *p += bytes;
    return v;
}
#define TAKE(ptr, type, n) ((type *)take(&(ptr), sizeof(type) * (size_t)(n)))

enum { LUT16, LIN16, POLY, FOREST, LUT8, Q8, F32, ROT16, GRID, CP, GA2M, GRIDV, KINDS };
static const char *kind_names[KINDS] = {"lut16", "lin16", "poly", "forest", "lut8", "q8", "f32", "rot16",
                                        "grid", "cp", "ga2m", "gridv"};

typedef struct {
    int kind;
    int layers, widest;
    LutLayer *lut;
    Q8Layer *q8;
    F32Layer *f32;
    F32Layer mix;
    GridMap grid;
    GridVMap gridv;
    CPMap cp;
    GA2MMap ga2m;
    Poly poly;
    Forest forest;
    float *h0, *h1, *hm;
} Model;

static void load(Model *m, const char *kind, const char *path) {
    const unsigned char *p = slurp(path);
    m->kind = -1;
    for (int k = 0; k < KINDS; k++) if (!strcmp(kind, kind_names[k])) m->kind = k;
    if (m->kind < 0) { fprintf(stderr, "unknown kind %s\n", kind); exit(2); }
    if (m->kind == GRID) {
        uint32_t *d = TAKE(p, uint32_t, 2);
        size_t cells = 1;
        for (uint32_t a = 0; a < d[0]; a++) cells *= d[1];
        float scale = *TAKE(p, float, 1);
        m->grid = (GridMap){(int)d[0], (int)d[1], scale, TAKE(p, int16_t, cells)};
        return;
    }
    if (m->kind == GRIDV) {
        uint32_t d = *TAKE(p, uint32_t, 1);
        uint32_t *n = TAKE(p, uint32_t, d);
        size_t cells = 1;
        m->gridv.d = (int)d;
        for (uint32_t a = 0; a < d; a++) { m->gridv.n[a] = (int)n[a]; cells *= n[a]; }
        m->gridv.scale = *TAKE(p, float, 1);
        m->gridv.v = TAKE(p, int16_t, cells);
        return;
    }
    if (m->kind == CP) {
        uint32_t *d = TAKE(p, uint32_t, 3);
        float b = *TAKE(p, float, 1);
        float *w = TAKE(p, float, d[1]), *s = TAKE(p, float, (size_t)d[1] * d[0]);
        m->cp = (CPMap){(int)d[0], (int)d[1], (int)d[2], b, w, s, TAKE(p, int16_t, (size_t)d[1] * d[0] * d[2])};
        return;
    }
    if (m->kind == GA2M) {
        uint32_t *d = TAKE(p, uint32_t, 3);
        const int dd = (int)d[0], pairs = dd * (dd - 1) / 2;
        float b = *TAKE(p, float, 1);
        float *gs = TAKE(p, float, dd);
        int16_t *g = TAKE(p, int16_t, (size_t)dd * d[1]);
        float *hs = TAKE(p, float, pairs);
        int16_t *h = TAKE(p, int16_t, (size_t)pairs * d[2] * d[2]);
        m->ga2m = (GA2MMap){dd, (int)d[1], (int)d[2], b, gs, hs, g, h};
        return;
    }
    if (m->kind == ROT16) {
        uint32_t *d = TAKE(p, uint32_t, 2);
        float *w = TAKE(p, float, (size_t)d[0] * d[1]);
        f32_init(&m->mix, (int)d[0], (int)d[1], 0, w, TAKE(p, float, d[1]));
    }
    if (m->kind != POLY && m->kind != FOREST) {
        m->layers = (int)*TAKE(p, uint32_t, 1);
        m->lut = calloc(m->layers, sizeof(LutLayer));
        m->q8 = calloc(m->layers, sizeof(Q8Layer));
        m->f32 = calloc(m->layers, sizeof(F32Layer));
        m->widest = 1;
        for (int k = 0; k < m->layers; k++) {
            uint32_t *d = TAKE(p, uint32_t, 3);
            int in = (int)d[0], out = (int)d[1];
            if (m->kind == Q8) {
                float *act = TAKE(p, float, 1), *ws = TAKE(p, float, out);
                int8_t *w = TAKE(p, int8_t, (size_t)out * in);
                q8_init(&m->q8[k], in, out, (int)d[2], *act, ws, w, TAKE(p, float, out));
                if (d[2] >= ACT_SILU) {
                    float pre = *TAKE(p, float, 1);
                    q8_set_table(&m->q8[k], pre, TAKE(p, float, 255));
                }
            } else if (m->kind == F32) {
                float *w = TAKE(p, float, (size_t)out * in);
                f32_init(&m->f32[k], in, out, (int)d[2], w, TAKE(p, float, out));
            } else {
                int q = (int)d[2];
                float *lo = TAKE(p, float, in), *hi = TAKE(p, float, in), *scale = TAKE(p, float, out);
                if (m->kind == LUT8) lut_init(&m->lut[k], in, out, q, lo, hi, scale, TAKE(p, int8_t, (size_t)out * in * q));
                else lut16_init(&m->lut[k], in, out, q, lo, hi, scale, TAKE(p, int16_t, (size_t)out * in * q));
            }
            m->widest = out > m->widest ? out : m->widest;
        }
        if (m->kind == ROT16) m->widest = m->mix.out > m->widest ? m->mix.out : m->widest;
        m->h0 = malloc(sizeof(float) * m->widest);
        m->h1 = malloc(sizeof(float) * m->widest);
        m->hm = malloc(sizeof(float) * m->widest);
    } else if (m->kind == POLY) {
        int terms = (int)*TAKE(p, uint32_t, 1);
        uint16_t *parent = TAKE(p, uint16_t, terms);
        uint8_t *var = TAKE(p, uint8_t, terms);
        float *coef = TAKE(p, float, terms);
        poly_init(&m->poly, terms, parent, var, coef);
    } else {
        uint32_t *d = TAKE(p, uint32_t, 2);
        m->forest.trees = (int)d[0];
        m->forest.base = *TAKE(p, float, 1);
        m->forest.root = TAKE(p, int32_t, d[0]);
        m->forest.nodes = TAKE(p, TreeNode, d[1]);
    }
}

static float forward(Model *m, const float *x, int fast) {
    if (m->kind == POLY) return poly_forward(&m->poly, x);
    if (m->kind == GRID) return grid_forward(&m->grid, x);
    if (m->kind == GRIDV) return gridv_forward(&m->gridv, x);
    if (m->kind == CP) return cp_forward(&m->cp, x);
    if (m->kind == GA2M) return ga2m_forward(&m->ga2m, x);
    if (m->kind == FOREST) return forest_forward(&m->forest, x);
    const float *h = x;
    if (m->kind == ROT16) {
        (fast ? f32_forward_fast : f32_forward_ref)(&m->mix, x, m->hm);
        h = m->hm;
    }
    for (int k = 0; k < m->layers; k++) {
        float *y = k % 2 ? m->h1 : m->h0;
        switch (m->kind) {
        case LUT16: (fast ? lut16_forward_fast : lut16_forward_ref)(&m->lut[k], h, y); break;
        case LIN16: case ROT16: (fast ? lin16_forward_fast : lin16_forward_ref)(&m->lut[k], h, y); break;
        case LUT8: (fast ? lut_forward_fast : lut_forward_ref)(&m->lut[k], h, y); break;
        case Q8: (fast ? q8_forward_fast : q8_forward_ref)(&m->q8[k], h, y); break;
        default: (fast ? f32_forward_fast : f32_forward_ref)(&m->f32[k], h, y);
        }
        h = y;
    }
    return h[0];
}

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
    if (argc < 5) { fprintf(stderr, "usage: run eval|time KIND MODEL X\n"); return 2; }
    Model m = {0};
    load(&m, argv[2], argv[3]);
    const unsigned char *p = slurp(argv[4]);
    uint32_t *nd = TAKE(p, uint32_t, 2);
    int n = (int)nd[0], d = (int)nd[1];
    float *xs = TAKE(p, float, (size_t)n * d);
    if (!strcmp(argv[1], "eval")) {
        const int single = m.kind == POLY || m.kind == FOREST || m.kind == GRID || m.kind == GRIDV || m.kind == CP || m.kind == GA2M;
        for (int pass = 0; pass < (single ? 1 : 2); pass++)
            for (int s = 0; s < n; s++) {
                float y = forward(&m, xs + (size_t)s * d, pass);
                fwrite(&y, sizeof y, 1, stdout);
            }
        return 0;
    }
    const char *rep = getenv("BENCH_REPEAT"), *inn = getenv("BENCH_INNER");
    int repeat = rep ? atoi(rep) : 20, inner = inn ? atoi(inn) : 1, total = repeat * n / inner;
    int64_t *t = malloc(sizeof(int64_t) * total);
    volatile float sink = 0.0f;
    for (int s = 0; s < n && s < 1000; s++) sink += forward(&m, xs + (size_t)s * d, 1);
    for (int j = 0, s = 0; j < total; j++) {
        int64_t t0 = now_ns();
        for (int k = 0; k < inner; k++, s = (s + 1) % n) sink += forward(&m, xs + (size_t)s * d, 1);
        t[j] = now_ns() - t0;
    }
    qsort(t, total, sizeof(int64_t), cmp64);
    printf("median_ns %lld p10_ns %lld p90_ns %lld\n", (long long)(t[total / 2] / inner),
           (long long)(t[total / 10] / inner), (long long)(t[total * 9 / 10] / inner));
    (void)sink;
    return 0;
}

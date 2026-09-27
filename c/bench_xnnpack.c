#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "xnnpack.h"

#define MAXL 8

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

static void check(enum xnn_status s, const char *what) {
    if (s != xnn_status_success) { fprintf(stderr, "%s failed: %d\n", what, s); exit(1); }
}

static float act_ref(int act, float s) {
    return act == 1 ? (s < 0.0f ? 0.0f : s) : act == 2 ? s / (1.0f + expf(-s)) : act == 3 ? tanhf(s) : s;
}

typedef struct {
    int int8, layers, dims[MAXL + 1], act, once;
    float *in;
    xnn_operator_t fc[MAXL], unary[MAXL], mul[MAXL];
    float *wf[MAXL], *bf[MAXL], *hf[MAXL], *sig[MAXL];
    int8_t *w8[MAXL], *h8[MAXL], *xq, table[256];
    int32_t *b8[MAXL];
    float *ks[MAXL];
} Net;

static void build(Net *n) {
    for (int k = 0; k < n->layers; k++) {
        const int in = n->dims[k], out = n->dims[k + 1], hidden = k + 1 < n->layers;
        const int relu = hidden && n->act == 1;
        if (!n->int8) {
            n->wf[k] = malloc(sizeof(float) * in * out);
            n->bf[k] = calloc(out, sizeof(float));
            n->hf[k] = malloc(sizeof(float) * out);
            for (int i = 0; i < in * out; i++) n->wf[k][i] = unif() * 0.1f;
            check(xnn_create_fully_connected_nc_f32(in, out, in, out, n->wf[k], n->bf[k], relu ? 0.0f : -INFINITY, INFINITY,
                                                    0, NULL, &n->fc[k]), "fc f32");
            check(xnn_reshape_fully_connected_nc_f32(n->fc[k], 1, NULL), "reshape fc");
            if (hidden && n->act >= 2) {
                enum xnn_unary_operator op = n->act == 2 ? xnn_unary_sigmoid : xnn_unary_tanh;
                check(xnn_create_unary_elementwise_nc(op, xnn_datatype_fp32, xnn_datatype_fp32, NULL, NULL, NULL, NULL, 0,
                                                      &n->unary[k]), "unary");
                check(xnn_reshape_unary_elementwise_nc(n->unary[k], 1, out, out, out, NULL), "reshape unary");
                n->sig[k] = malloc(sizeof(float) * out);
                if (n->act == 2) {
                    size_t shape[2] = {1, (size_t)out};
                    check(xnn_create_binary_elementwise_nd(xnn_binary_multiply, xnn_datatype_fp32, NULL, NULL, NULL, 0,
                                                           &n->mul[k]), "multiply");
                    check(xnn_reshape_binary_elementwise_nd(n->mul[k], 2, shape, 2, shape, NULL), "reshape multiply");
                }
            }
        } else {
            n->w8[k] = malloc((size_t)in * out);
            n->b8[k] = calloc(out, sizeof(int32_t));
            n->ks[k] = malloc(sizeof(float) * out);
            n->h8[k] = malloc(out);
            for (int i = 0; i < in * out; i++) n->w8[k][i] = (int8_t)((int)(next() % 255) - 127);
            for (int o = 0; o < out; o++) n->ks[k][o] = 1e-2f;
            check(xnn_create_fully_connected_nc_qs8_qc8w(in, out, in, out, 0, 1.0f / 127.0f, n->ks[k], n->w8[k], n->b8[k], 0,
                                                         0.05f, relu ? 0 : -128, 127, 0, NULL, &n->fc[k]), "fc qs8");
            check(xnn_reshape_fully_connected_nc_qs8_qc8w(n->fc[k], 1, NULL), "reshape fc");
        }
    }
    if (n->int8) {
        n->xq = malloc(n->dims[0]);
        for (int t = 0; t < 256; t++) {
            float v = (float)(t - 128) * 0.05f, f = act_ref(n->act, v) / 0.05f;
            n->table[t] = (int8_t)(f > 127.0f ? 127.0f : f < -128.0f ? -128.0f : nearbyintf(f));
        }
    }
}

// Binds the buffers; with fixed buffers a deployment does this once (BENCH_SETUP=once), else it runs on every call
static void setup(Net *n, const float *x) {
    const float *h = x;
    const int8_t *h8 = n->xq;
    for (int k = 0; k < n->layers; k++) {
        if (n->int8) {
            check(xnn_setup_fully_connected_nc_qs8_qc8w(n->fc[k], h8, n->h8[k]), "setup fc");
            h8 = n->h8[k];
            continue;
        }
        check(xnn_setup_fully_connected_nc_f32(n->fc[k], h, n->hf[k]), "setup fc");
        if (n->unary[k]) {
            check(xnn_setup_unary_elementwise_nc(n->unary[k], n->hf[k], n->mul[k] ? n->sig[k] : n->hf[k]), "setup unary");
            if (n->mul[k])
                check(xnn_setup_binary_elementwise_nd(n->mul[k], n->hf[k], n->sig[k], n->hf[k]), "setup multiply");
        }
        h = n->hf[k];
    }
}

static float forward(Net *n, const float *x) {
    if (n->once && !n->int8) memcpy(n->in, x, sizeof(float) * n->dims[0]);
    else if (!n->once) setup(n, x);
    if (!n->int8) {
        for (int k = 0; k < n->layers; k++) {
            check(xnn_run_operator(n->fc[k], NULL), "run fc");
            if (n->unary[k]) {
                check(xnn_run_operator(n->unary[k], NULL), "run unary");
                if (n->mul[k]) check(xnn_run_operator(n->mul[k], NULL), "run multiply");
            }
        }
        return n->hf[n->layers - 1][0];
    }
    for (int i = 0; i < n->dims[0]; i++) {
        float v = nearbyintf(x[i] * 127.0f);
        n->xq[i] = (int8_t)(v < -127.0f ? -127.0f : v > 127.0f ? 127.0f : v);
    }
    for (int k = 0; k < n->layers; k++) {
        check(xnn_run_operator(n->fc[k], NULL), "run fc");
        if (k + 1 < n->layers && n->act >= 2)
            for (int o = 0; o < n->dims[k + 1]; o++) n->h8[k][o] = n->table[n->h8[k][o] + 128];
    }
    return (float)n->h8[n->layers - 1][0] * 0.05f;
}

static float reference(const Net *n, const float *x) {
    float a[256], b[256];
    memcpy(a, x, sizeof(float) * n->dims[0]);
    for (int k = 0; k < n->layers; k++) {
        const int in = n->dims[k], out = n->dims[k + 1];
        for (int o = 0; o < out; o++) {
            double s = n->bf[k][o];
            for (int i = 0; i < in; i++) s += (double)n->wf[k][(size_t)o * in + i] * a[i];
            b[o] = k + 1 < n->layers ? act_ref(n->act, (float)s) : (float)s;
        }
        memcpy(a, b, sizeof(float) * out);
    }
    return a[0];
}

int main(int argc, char **argv) {
    const char *rep = getenv("BENCH_REPEAT");
    int repeat = rep ? atoi(rep) : 20000;
    int64_t *t = malloc(sizeof(int64_t) * repeat);
    check(xnn_initialize(NULL), "initialize");
    for (int a = 1; a < argc; a++) {
        Net n = {0};
        char spec[256];
        snprintf(spec, sizeof spec, "%s", argv[a]);
        char *c = strchr(spec, ':');
        if (!c) { fprintf(stderr, "bad spec %s\n", argv[a]); return 2; }
        *c++ = 0;
        if (!strcmp(spec, "mlp")) n.int8 = 1;
        else if (strcmp(spec, "f32")) { fprintf(stderr, "bad spec %s\n", argv[a]); return 2; }
        char *actp = strchr(c, ':');
        n.act = 1;
        if (actp) { *actp++ = 0; n.act = !strcmp(actp, "silu") ? 2 : !strcmp(actp, "tanh") ? 3 : 1; }
        int d = 0;
        for (char *tok = strtok(c, "-"); tok && d <= MAXL; tok = strtok(NULL, "-")) n.dims[d++] = atoi(tok);
        n.layers = d - 1;
        build(&n);
        const char *mode = getenv("BENCH_SETUP");
        n.once = mode && !strcmp(mode, "once");
        n.in = malloc(sizeof(float) * n.dims[0]);
        if (n.once) setup(&n, n.in);
        float *xs = malloc(sizeof(float) * 256 * n.dims[0]);
        for (int i = 0; i < 256 * n.dims[0]; i++) xs[i] = unif();
        if (!n.int8) {
            double err = 0.0, mag = 0.0;
            for (int i = 0; i < 16; i++) {
                float y = forward(&n, xs + (size_t)i * n.dims[0]), r = reference(&n, xs + (size_t)i * n.dims[0]);
                err = fmax(err, fabs(y - r));
                mag = fmax(mag, fabs(r));
            }
            fprintf(stderr, "check %s max|xnn - ref| %.3g of %.3g\n", argv[a], err, mag);
        }
        volatile float sink = 0.0f;
        for (int r = -1000; r < repeat; r++) {
            int64_t t0 = now_ns();
            sink += forward(&n, xs + (size_t)((r + 1000) % 256) * n.dims[0]);
            if (r >= 0) t[r] = now_ns() - t0;
        }
        (void)sink;
        qsort(t, repeat, sizeof(int64_t), cmp64);
        printf("spec %s median_ns %lld p10_ns %lld p90_ns %lld\n", argv[a], (long long)t[repeat / 2],
               (long long)t[repeat / 10], (long long)t[repeat * 9 / 10]);
        for (int k = 0; k < n.layers; k++) {
            xnn_delete_operator(n.fc[k]);
            if (n.unary[k]) xnn_delete_operator(n.unary[k]);
            if (n.mul[k]) xnn_delete_operator(n.mul[k]);
            free(n.wf[k]); free(n.bf[k]); free(n.hf[k]); free(n.sig[k]); free(n.w8[k]); free(n.h8[k]); free(n.b8[k]);
            free(n.ks[k]);
        }
        free(n.xq);
        free(n.in);
        free(xs);
    }
    return 0;
}

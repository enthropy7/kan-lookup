#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernels.h"
#include "mcubench.h"

uint32_t mcu_cycles(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)(t.tv_sec * 1000000000ull + t.tv_nsec);
}

static float xs[64 * 16];
static float got[64], want[64];

static double maxdiff(int n) {
    double m = 0;
    for (int i = 0; i < n; i++) m = fmax(m, fabs(got[i] - want[i]));
    return m;
}

static int parse_dims(const char *spec, int *dims, const char **rest) {
    int n = 0;
    const char *p = strchr(spec, ':') + 1;
    while (*p && *p != ':') { dims[n++] = (int)strtol(p, (char **)&p, 10); if (*p == '-') p++; }
    *rest = p;
    return n;
}

int main(void) {
    for (int i = 0; i < 64 * 16; i++) xs[i] = (float)((i * 2654435761u) % 2000) / 1000.0f - 1.0f;
    int bad = 0;

    const char *lut_specs[] = {"lin16:5-8-1:64", "lin16:7-32-1:256", "kan16:5-16-1:128", "lin16:3-16-16-1:32"};
    for (int s = 0; s < 4; s++) {
        int dims[4];
        const char *p;
        int n = parse_dims(lut_specs[s], dims, &p), q = atoi(p + 1), interp = lut_specs[s][0] == 'l';
        size_t off = 0;
        LutLayer L[3];
        for (int k = 0; k + 1 < n; k++) {
            int in = dims[k], out = dims[k + 1];
            int16_t *t = malloc(sizeof(int16_t) * in * out * q);
            for (int o = 0; o < out; o++)
                for (int i = 0; i < in; i++)
                    for (int j = 0; j < q; j++) t[((size_t)o * in + i) * q + j] = BLOB16[off + ((size_t)i * q + j) * out + o];
            off += (size_t)in * q * out;
            float lo[64], hi[64], sc[64];
            for (int i = 0; i < in; i++) { lo[i] = -1.0f; hi[i] = 1.0f; }
            for (int o = 0; o < out; o++) sc[o] = 1e-3f;
            lut16_init(&L[k], in, out, q, lo, hi, sc, t);
        }
        for (int r = 0; r < 64; r++) {
            float h0[64], h1[64];
            const float *h = xs + r * dims[0];
            for (int k = 0; k + 1 < n; k++) {
                float *y = k % 2 ? h1 : h0;
                (interp ? lin16_forward_ref : lut16_forward_ref)(&L[k], h, y);
                h = y;
            }
            want[r] = h[0];
        }
        mcu_eval(lut_specs[s], xs, 64, got);
        double d = maxdiff(64);
        printf("%-22s max |mcu - c| %.3g\n", lut_specs[s], d);
        bad += d > 1e-5;
    }

    const char *mlp_specs[] = {"f32:5-32-32-1", "f32:5-32-32-1:silu", "f32:7-16-1:tanh",
                               "mlp:5-32-32-1", "mlp:5-32-32-1:silu", "mlp:7-16-1:tanh"};
    for (int s = 0; s < 6; s++) {
        int dims[4], act = ACT_RELU;
        const char *p;
        int n = parse_dims(mlp_specs[s], dims, &p);
        if (*p == ':') act = !strcmp(p + 1, "silu") ? ACT_SILU : ACT_TANH;
        float table[255];
        for (int t = 0; t < 255; t++) {
            float v = (float)(t - 127) * 0.05f;
            table[t] = act == ACT_SILU ? v / (1.0f + expf(-v)) : tanhf(v);
        }
        F32Layer F[3];
        Q8Layer Q[3];
        size_t of = 0, o8 = 0;
        for (int k = 0; k + 1 < n; k++) {
            int in = dims[k], out = dims[k + 1], a = k + 2 < n ? act : ACT_NONE;
            if (mlp_specs[s][0] == 'f') {
                f32_init(&F[k], in, out, a, BLOBF + of, BLOBF + of + (size_t)in * out);
                of += (size_t)in * out + out;
            } else {
                float ws[64], b[64];
                for (int o = 0; o < out; o++) { ws[o] = 1e-2f; b[o] = 0.0f; }
                q8_init(&Q[k], in, out, a, 1.0f / 127.0f, ws, BLOB8 + o8, b);
                o8 += (size_t)in * out;
                if (a >= ACT_SILU) q8_set_table(&Q[k], 0.05f, table);
            }
        }
        for (int r = 0; r < 64; r++) {
            float h0[64], h1[64];
            const float *h = xs + r * dims[0];
            for (int k = 0; k + 1 < n; k++) {
                float *y = k % 2 ? h1 : h0;
                if (mlp_specs[s][0] == 'f') f32_forward_ref(&F[k], h, y); else q8_forward_ref(&Q[k], h, y);
                h = y;
            }
            want[r] = h[0];
        }
        mcu_eval(mlp_specs[s], xs, 64, got);
        double d = maxdiff(64);
        printf("%-22s max |mcu - c| %.3g\n", mlp_specs[s], d);
        bad += d > 1e-5;
    }
    const char *timed[] = {"lin16:7-32-1:256", "gridv:7:6x6x7x7x3x3x4", "cp:7:16:16", "ga2m:5:32:16",
                           "f32:7-128-128-128-1:silu", "mlp:7-64-64-1:tanh", "poly:5:4"};
    for (int s = 0; s < 7; s++) mcu_bench(timed[s], 31, 16);
    printf(bad ? "FAIL\n" : "OK\n");
    return bad != 0;
}

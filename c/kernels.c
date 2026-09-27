#include "kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

static void lut_common(LutLayer *l, int in, int out, int q, const float *lo, const float *hi, const float *scale) {
    l->in = in; l->out = out; l->q = q; l->outp = (out + 7) & ~7;
    l->lo = malloc(sizeof(float) * in); l->hi = malloc(sizeof(float) * in);
    l->inv_step = malloc(sizeof(float) * in); l->scale = malloc(sizeof(float) * out);
    memcpy(l->lo, lo, sizeof(float) * in); memcpy(l->hi, hi, sizeof(float) * in);
    memcpy(l->scale, scale, sizeof(float) * out);
    for (int i = 0; i < in; i++) l->inv_step[i] = (float)(q - 1) / (hi[i] - lo[i]);
    l->table = NULL; l->t16 = NULL;
    l->acc = calloc(l->outp, sizeof(int32_t));
    l->rows = malloc(sizeof(int8_t *) * in);
}

void lut_init(LutLayer *l, int in, int out, int q, const float *lo, const float *hi, const float *scale,
              const int8_t *table_oiq) {
    lut_common(l, in, out, q, lo, hi, scale);
    l->table = calloc((size_t)in * q * l->outp, 1);
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in; i++)
            for (int k = 0; k < q; k++)
                l->table[((size_t)i * q + k) * l->outp + o] = table_oiq[((size_t)o * in + i) * q + k];
}

static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

void lut_forward_ref(LutLayer *l, const float *x, float *y) {
    memset(l->acc, 0, sizeof(int32_t) * l->outp);
    for (int i = 0; i < l->in; i++) {
        int k = clampi((int)nearbyintf((x[i] - l->lo[i]) / (l->hi[i] - l->lo[i]) * (float)(l->q - 1)), 0, l->q - 1);
        const int8_t *row = l->table + ((size_t)i * l->q + k) * l->outp;
        for (int o = 0; o < l->outp; o++) l->acc[o] += row[o];
    }
    for (int o = 0; o < l->out; o++) y[o] = l->scale[o] * (float)l->acc[o];
}

static inline int lut_index(const LutLayer *l, int i, float x) {
    return clampi((int)rintf((x - l->lo[i]) * l->inv_step[i]), 0, l->q - 1);   /* rintf: nearbyintf's rounding in one instruction */
}

static inline int lut_index_rt(const LutLayer *l, int i, float x) {
    float t = (x - l->lo[i]) * l->inv_step[i];
    t = t < 0.0f ? 0.0f : t > (float)(l->q - 1) ? (float)(l->q - 1) : t;
    return (int)(t + 0.5f);
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#define LUT_BODY(R, INDEX)                                                                       \
    {                                                                                            \
        int32x4_t a[2 * R];                                                                      \
        for (int r = 0; r < 2 * R; r++) a[r] = vdupq_n_s32(0);                                    \
        for (int i = 0; i < l->in; i++) {                                                        \
            const int8_t *row = l->table + ((size_t)i * l->q + INDEX(l, i, x[i])) * l->outp;     \
            for (int r = 0; r < R; r++) {                                                        \
                int16x8_t v = vmovl_s8(vld1_s8(row + 8 * r));                                    \
                a[2 * r] = vaddw_s16(a[2 * r], vget_low_s16(v));                                 \
                a[2 * r + 1] = vaddw_s16(a[2 * r + 1], vget_high_s16(v));                        \
            }                                                                                    \
        }                                                                                        \
        for (int r = 0; r < 2 * R; r++) vst1q_s32(l->acc + 4 * r, a[r]);                         \
    }
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#define LUT_CHUNK(R)                                                                             \
    {                                                                                            \
        int32x4_t a[2 * R];                                                                      \
        for (int r = 0; r < 2 * R; r++) a[r] = vdupq_n_s32(0);                                    \
        for (int i = 0; i < l->in; i++)                                                          \
            for (int r = 0; r < R; r++) {                                                        \
                int16x8_t v = vmovl_s8(vld1_s8(l->rows[i] + c + 8 * r));                         \
                a[2 * r] = vaddw_s16(a[2 * r], vget_low_s16(v));                                 \
                a[2 * r + 1] = vaddw_s16(a[2 * r + 1], vget_high_s16(v));                        \
            }                                                                                    \
        for (int r = 0; r < 2 * R; r++) vst1q_s32(l->acc + c + 4 * r, a[r]);                     \
    }
static void lut_wide(LutLayer *l) {
    int c = 0;
    for (; c + 32 <= l->outp; c += 32) LUT_CHUNK(4)
    for (; c < l->outp; c += 8) LUT_CHUNK(1)
}

#define LUT_FORWARD(NAME, INDEX)                                                                  \
    void NAME(LutLayer *l, const float *x, float *y) {                                            \
        switch (l->outp / 8) {                                                                    \
        case 1: LUT_BODY(1, INDEX) break;                                                         \
        case 2: LUT_BODY(2, INDEX) break;                                                         \
        case 3: LUT_BODY(3, INDEX) break;                                                         \
        case 4: LUT_BODY(4, INDEX) break;                                                         \
        default:                                                                                  \
            for (int i = 0; i < l->in; i++)                                                       \
                l->rows[i] = l->table + ((size_t)i * l->q + INDEX(l, i, x[i])) * l->outp;         \
            lut_wide(l);                                                                          \
        }                                                                                         \
        for (int o = 0; o < l->out; o++) y[o] = l->scale[o] * (float)l->acc[o];                   \
    }
#else
#define LUT_FORWARD(NAME, INDEX)                                                                  \
    void NAME(LutLayer *l, const float *x, float *y) {                                            \
        memset(l->acc, 0, sizeof(int32_t) * l->outp);                                             \
        for (int i = 0; i < l->in; i++) {                                                         \
            const int8_t *row = l->table + ((size_t)i * l->q + INDEX(l, i, x[i])) * l->outp;      \
            for (int o = 0; o < l->outp; o++) l->acc[o] += row[o];                                \
        }                                                                                         \
        for (int o = 0; o < l->out; o++) y[o] = l->scale[o] * (float)l->acc[o];                   \
    }
#endif

LUT_FORWARD(lut_forward_fast, lut_index)
LUT_FORWARD(lut_forward_fast_rt, lut_index_rt)

void lut16_init(LutLayer *l, int in, int out, int q, const float *lo, const float *hi, const float *scale,
                const int16_t *table_oiq) {
    lut_common(l, in, out, q, lo, hi, scale);
    l->t16 = calloc((size_t)in * q * l->outp, sizeof(int16_t));
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in; i++)
            for (int k = 0; k < q; k++)
                l->t16[((size_t)i * q + k) * l->outp + o] = table_oiq[((size_t)o * in + i) * q + k];
}

void lut16_forward_ref(LutLayer *l, const float *x, float *y) {
    memset(l->acc, 0, sizeof(int32_t) * l->outp);
    for (int i = 0; i < l->in; i++) {
        int k = clampi((int)nearbyintf((x[i] - l->lo[i]) / (l->hi[i] - l->lo[i]) * (float)(l->q - 1)), 0, l->q - 1);
        const int16_t *row = l->t16 + ((size_t)i * l->q + k) * l->outp;
        for (int o = 0; o < l->outp; o++) l->acc[o] += row[o];
    }
    for (int o = 0; o < l->out; o++) y[o] = l->scale[o] * (float)l->acc[o];
}

void lin16_forward_ref(LutLayer *l, const float *x, float *y) {
    memset(l->acc, 0, sizeof(int32_t) * l->outp);
    for (int i = 0; i < l->in; i++) {
        float t = (x[i] - l->lo[i]) / (l->hi[i] - l->lo[i]) * (float)(l->q - 1);
        t = t < 0.0f ? 0.0f : t > (float)(l->q - 1) ? (float)(l->q - 1) : t;
        int k = (int)floorf(t);
        k = k > l->q - 2 ? l->q - 2 : k;
        int f = (int)nearbyintf((t - (float)k) * 256.0f);
        const int16_t *r0 = l->t16 + ((size_t)i * l->q + k) * l->outp, *r1 = r0 + l->outp;
        for (int o = 0; o < l->outp; o++) l->acc[o] += r0[o] * (256 - f) + r1[o] * f;
    }
    for (int o = 0; o < l->out; o++) y[o] = l->scale[o] * (1.0f / 256.0f) * (float)l->acc[o];
}

static inline const int16_t *lin_rows(const LutLayer *l, int i, float x, int *f) {
    float t = (x - l->lo[i]) * l->inv_step[i];
    t = t < 0.0f ? 0.0f : t > (float)(l->q - 1) ? (float)(l->q - 1) : t;
    int k = (int)t;
    k = k > l->q - 2 ? l->q - 2 : k;
    *f = (int)((t - (float)k) * 256.0f + 0.5f);
    return l->t16 + ((size_t)i * l->q + k) * l->outp;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define LUT16_BODY(R)                                                                            \
    {                                                                                            \
        int32x4_t a[2 * R];                                                                      \
        for (int r = 0; r < 2 * R; r++) a[r] = vdupq_n_s32(0);                                    \
        for (int i = 0; i < l->in; i++) {                                                        \
            const int16_t *row = l->t16 + ((size_t)i * l->q + lut_index(l, i, x[i])) * l->outp;  \
            for (int r = 0; r < R; r++) {                                                        \
                int16x8_t v = vld1q_s16(row + 8 * r);                                            \
                a[2 * r] = vaddw_s16(a[2 * r], vget_low_s16(v));                                 \
                a[2 * r + 1] = vaddw_s16(a[2 * r + 1], vget_high_s16(v));                        \
            }                                                                                    \
        }                                                                                        \
        for (int r = 0; r < 2 * R; r++) vst1q_s32(l->acc + 4 * r, a[r]);                         \
    }
#define LIN16_BODY(R)                                                                            \
    {                                                                                            \
        int32x4_t a[2 * R];                                                                      \
        for (int r = 0; r < 2 * R; r++) a[r] = vdupq_n_s32(0);                                    \
        for (int i = 0; i < l->in; i++) {                                                        \
            int f;                                                                               \
            const int16_t *r0 = lin_rows(l, i, x[i], &f), *r1 = r0 + l->outp;                    \
            const int16_t f0 = (int16_t)(256 - f), f1 = (int16_t)f;                              \
            for (int r = 0; r < R; r++) {                                                        \
                int16x8_t v0 = vld1q_s16(r0 + 8 * r), v1 = vld1q_s16(r1 + 8 * r);                \
                a[2 * r] = vmlal_n_s16(a[2 * r], vget_low_s16(v0), f0);                          \
                a[2 * r] = vmlal_n_s16(a[2 * r], vget_low_s16(v1), f1);                          \
                a[2 * r + 1] = vmlal_n_s16(a[2 * r + 1], vget_high_s16(v0), f0);                 \
                a[2 * r + 1] = vmlal_n_s16(a[2 * r + 1], vget_high_s16(v1), f1);                 \
            }                                                                                    \
        }                                                                                        \
        for (int r = 0; r < 2 * R; r++) vst1q_s32(l->acc + 4 * r, a[r]);                         \
    }
#define DISPATCH(BODY)                                                                           \
    switch (l->outp / 8) {                                                                       \
    case 1: BODY(1) return 1;                                                                    \
    case 2: BODY(2) return 1;                                                                    \
    case 3: BODY(3) return 1;                                                                    \
    case 4: BODY(4) return 1;                                                                    \
    default: return 0;                                                                           \
    }
static int lut16_neon(LutLayer *l, const float *x) { DISPATCH(LUT16_BODY) }
static int lin16_neon(LutLayer *l, const float *x) { DISPATCH(LIN16_BODY) }
#else
static int lut16_neon(LutLayer *l, const float *x) { (void)l; (void)x; return 0; }
static int lin16_neon(LutLayer *l, const float *x) { (void)l; (void)x; return 0; }
#endif

void lut16_forward_fast(LutLayer *l, const float *x, float *y) {
    if (!lut16_neon(l, x)) {
        memset(l->acc, 0, sizeof(int32_t) * l->outp);
        for (int i = 0; i < l->in; i++) {
            const int16_t *row = l->t16 + ((size_t)i * l->q + lut_index(l, i, x[i])) * l->outp;
            for (int o = 0; o < l->outp; o++) l->acc[o] += row[o];
        }
    }
    for (int o = 0; o < l->out; o++) y[o] = l->scale[o] * (float)l->acc[o];
}

void lin16_forward_fast(LutLayer *l, const float *x, float *y) {
    if (!lin16_neon(l, x)) {
        memset(l->acc, 0, sizeof(int32_t) * l->outp);
        for (int i = 0; i < l->in; i++) {
            int f;
            const int16_t *r0 = lin_rows(l, i, x[i], &f), *r1 = r0 + l->outp;
            for (int o = 0; o < l->outp; o++) l->acc[o] += r0[o] * (256 - f) + r1[o] * f;
        }
    }
    for (int o = 0; o < l->out; o++) y[o] = l->scale[o] * (1.0f / 256.0f) * (float)l->acc[o];
}

/* SiLU and tanh of the fast paths use this polynomial exp (Cephes, as neon_mathfun's exp_ps): libm expf would
 * cost more than the layer's multiply-adds; the reference paths keep libm */
static inline float fexp(float x) {
    x = x > 88.3762626647949f ? 88.3762626647949f : x < -88.3762626647949f ? -88.3762626647949f : x;
    const float fx = x * 1.44269504088896341f + 0.5f;
    float t = (float)(int)fx;
    if (t > fx) t -= 1.0f;
    x = x - t * 0.693359375f - t * -2.12194440e-4f;
    const float z = x * x;
    float y = 1.9875691500e-4f;
    y = y * x + 1.3981999507e-3f;
    y = y * x + 8.3334519073e-3f;
    y = y * x + 4.1665795894e-2f;
    y = y * x + 1.6666665459e-1f;
    y = y * x + 5.0000001201e-1f;
    y = y * z + x + 1.0f;
    union { int32_t i; float f; } u = {((int32_t)t + 127) << 23};
    return y * u.f;
}

static inline float smooth_fast(int act, float v) {
    return act == ACT_SILU ? v / (1.0f + fexp(-v)) : 1.0f - 2.0f / (fexp(2.0f * v) + 1.0f);
}

static void smooth_ref(int act, float *y, int n) {
    if (act < ACT_SILU) return;
    for (int o = 0; o < n; o++) y[o] = act == ACT_SILU ? y[o] / (1.0f + expf(-y[o])) : tanhf(y[o]);
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static inline float32x4_t fexp4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f)), vdupq_n_f32(88.3762626647949f));
    const float32x4_t fx = vaddq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)), vdupq_n_f32(0.5f));
    float32x4_t t = vcvtq_f32_s32(vcvtq_s32_f32(fx));
    t = vbslq_f32(vcgtq_f32(t, fx), vsubq_f32(t, vdupq_n_f32(1.0f)), t);
    x = vsubq_f32(vsubq_f32(x, vmulq_f32(t, vdupq_n_f32(0.693359375f))), vmulq_f32(t, vdupq_n_f32(-2.12194440e-4f)));
    const float32x4_t z = vmulq_f32(x, x);
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vaddq_f32(vmulq_f32(y, x), vdupq_n_f32(1.3981999507e-3f));
    y = vaddq_f32(vmulq_f32(y, x), vdupq_n_f32(8.3334519073e-3f));
    y = vaddq_f32(vmulq_f32(y, x), vdupq_n_f32(4.1665795894e-2f));
    y = vaddq_f32(vmulq_f32(y, x), vdupq_n_f32(1.6666665459e-1f));
    y = vaddq_f32(vmulq_f32(y, x), vdupq_n_f32(5.0000001201e-1f));
    y = vaddq_f32(vaddq_f32(vmulq_f32(y, z), x), vdupq_n_f32(1.0f));
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(t), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

static inline float32x4_t div4(float32x4_t a, float32x4_t b) {
#if defined(__aarch64__)
    return vdivq_f32(a, b);
#else
    float32x4_t r = vrecpeq_f32(b);
    r = vmulq_f32(vrecpsq_f32(b, r), r);
    r = vmulq_f32(vrecpsq_f32(b, r), r);
    return vmulq_f32(a, r);
#endif
}

static void smooth_fast_n(int act, float *y, int n) {
    if (act < ACT_SILU) return;
    const float32x4_t one = vdupq_n_f32(1.0f), two = vdupq_n_f32(2.0f);
    int o = 0;
    for (; o + 4 <= n; o += 4) {
        const float32x4_t v = vld1q_f32(y + o);
        vst1q_f32(y + o, act == ACT_SILU ? div4(v, vaddq_f32(one, fexp4(vnegq_f32(v))))
                                         : vsubq_f32(one, div4(two, vaddq_f32(fexp4(vmulq_f32(two, v)), one))));
    }
    for (; o < n; o++) y[o] = smooth_fast(act, y[o]);
}
#else
static void smooth_fast_n(int act, float *y, int n) {
    if (act < ACT_SILU) return;
    for (int o = 0; o < n; o++) y[o] = smooth_fast(act, y[o]);
}
#endif

void q8_init(Q8Layer *l, int in, int out, int act, float act_scale, const float *w_scale, const int8_t *w,
             const float *bias) {
    l->in = in; l->out = out; l->act = act; l->act_scale = act_scale; l->inv_act_scale = 1.0f / act_scale;
    l->pre = 0.0f; l->table = NULL;
    l->w_scale = malloc(sizeof(float) * out); memcpy(l->w_scale, w_scale, sizeof(float) * out);
    l->bias = malloc(sizeof(float) * out); memcpy(l->bias, bias, sizeof(float) * out);
    l->w = malloc((size_t)in * out); memcpy(l->w, w, (size_t)in * out);
    const int outp = (out + 15) & ~15;
    l->wt = calloc((size_t)in * outp, 1);
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in; i++) l->wt[(size_t)i * outp + o] = w[(size_t)o * in + i];
    l->xq = malloc(in + 16);
    l->acc = malloc(sizeof(int32_t) * outp);
}

static void q8_quantize(Q8Layer *l, const float *x) {
    for (int i = 0; i < l->in; i++) {
        float v = nearbyintf(x[i] / l->act_scale);
        l->xq[i] = (int8_t)(v < -127.0f ? -127.0f : v > 127.0f ? 127.0f : v);
    }
}

void q8_set_table(Q8Layer *l, float pre_scale, const float *table) {
    l->pre = pre_scale;
    l->table = malloc(sizeof(float) * 255);
    memcpy(l->table, table, sizeof(float) * 255);
}

/* ReLU stays in the accumulation loops; smooth activations are a separate pass so those loops keep their code.
 * The int8 table index divides and rounds half to even, as kantab.tables.Int8MLP does, so C equals the simulator. */
static void q8_finish(Q8Layer *l, float *y) {
    if (!l->table) {
        for (int o = 0; o < l->out; o++) {
            float v = (float)l->acc[o] * (l->w_scale[o] * l->act_scale) + l->bias[o];
            y[o] = l->act == ACT_RELU && v < 0.0f ? 0.0f : v;
        }
        return;
    }
    int o = 0;
#if defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))

    const float32x4_t as = vdupq_n_f32(l->act_scale), pre = vdupq_n_f32(l->pre);
    for (; o + 4 <= l->out; o += 4) {
        float32x4_t v = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(l->acc + o)), vmulq_f32(vld1q_f32(l->w_scale + o), as)),
                                  vld1q_f32(l->bias + o));
        int32x4_t q = vcvtnq_s32_f32(vdivq_f32(v, pre));
        q = vaddq_s32(vminq_s32(vmaxq_s32(q, vdupq_n_s32(-127)), vdupq_n_s32(127)), vdupq_n_s32(127));
        y[o] = l->table[vgetq_lane_s32(q, 0)]; y[o + 1] = l->table[vgetq_lane_s32(q, 1)];
        y[o + 2] = l->table[vgetq_lane_s32(q, 2)]; y[o + 3] = l->table[vgetq_lane_s32(q, 3)];
    }
#endif
    for (; o < l->out; o++) {
        float q = rintf(((float)l->acc[o] * (l->w_scale[o] * l->act_scale) + l->bias[o]) / l->pre);
        y[o] = l->table[(int)(q < -127.0f ? -127.0f : q > 127.0f ? 127.0f : q) + 127];
    }
}

void q8_forward_ref(Q8Layer *l, const float *x, float *y) {
    q8_quantize(l, x);
    for (int o = 0; o < l->out; o++) {
        int32_t s = 0;
        const int8_t *w = l->w + (size_t)o * l->in;
        for (int i = 0; i < l->in; i++) s += (int32_t)w[i] * l->xq[i];
        l->acc[o] = s;
    }
    q8_finish(l, y);
}

static void q8_quantize_fast(Q8Layer *l, const float *x) {
    for (int i = 0; i < l->in; i++) {
        float v = rintf(x[i] * l->inv_act_scale);
        l->xq[i] = (int8_t)(v < -127.0f ? -127.0f : v > 127.0f ? 127.0f : v);
    }
}

void q8_forward_fast(Q8Layer *l, const float *x, float *y) {
    q8_quantize_fast(l, x);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (l->in < 16) {
        const int outp = (l->out + 15) & ~15;
        for (int c = 0; c < outp; c += 16) {
            int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
            for (int i = 0; i < l->in; i++) {
                int8x16_t w = vld1q_s8(l->wt + (size_t)i * outp + c);
                int16x8_t lo = vmovl_s8(vget_low_s8(w)), hi = vmovl_s8(vget_high_s8(w));
                const int16_t xv = l->xq[i];
                a0 = vmlal_n_s16(a0, vget_low_s16(lo), xv); a1 = vmlal_n_s16(a1, vget_high_s16(lo), xv);
                a2 = vmlal_n_s16(a2, vget_low_s16(hi), xv); a3 = vmlal_n_s16(a3, vget_high_s16(hi), xv);
            }
            vst1q_s32(l->acc + c, a0); vst1q_s32(l->acc + c + 4, a1);
            vst1q_s32(l->acc + c + 8, a2); vst1q_s32(l->acc + c + 12, a3);
        }
        q8_finish(l, y);
        return;
    }
    const int n16 = l->in & ~15;
    int o = 0;
    for (; o + 4 <= l->out; o += 4) {
        const int8_t *w0 = l->w + (size_t)o * l->in, *w1 = w0 + l->in, *w2 = w1 + l->in, *w3 = w2 + l->in;
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int i = 0; i < n16; i += 16) {
            int8x16_t xv = vld1q_s8(l->xq + i);
            int8x8_t xl = vget_low_s8(xv), xh = vget_high_s8(xv);
            int8x16_t v0 = vld1q_s8(w0 + i), v1 = vld1q_s8(w1 + i), v2 = vld1q_s8(w2 + i), v3 = vld1q_s8(w3 + i);
            a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(v0), xl)); a0 = vpadalq_s16(a0, vmull_s8(vget_high_s8(v0), xh));
            a1 = vpadalq_s16(a1, vmull_s8(vget_low_s8(v1), xl)); a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(v1), xh));
            a2 = vpadalq_s16(a2, vmull_s8(vget_low_s8(v2), xl)); a2 = vpadalq_s16(a2, vmull_s8(vget_high_s8(v2), xh));
            a3 = vpadalq_s16(a3, vmull_s8(vget_low_s8(v3), xl)); a3 = vpadalq_s16(a3, vmull_s8(vget_high_s8(v3), xh));
        }
        int32_t s[4][4];
        vst1q_s32(s[0], a0); vst1q_s32(s[1], a1); vst1q_s32(s[2], a2); vst1q_s32(s[3], a3);
        const int8_t *ws[4] = {w0, w1, w2, w3};
        for (int r = 0; r < 4; r++) {
            int32_t t = s[r][0] + s[r][1] + s[r][2] + s[r][3];
            for (int i = n16; i < l->in; i++) t += (int32_t)ws[r][i] * l->xq[i];
            l->acc[o + r] = t;
        }
    }
    for (; o < l->out; o++) {
        const int8_t *w = l->w + (size_t)o * l->in;
        int32x4_t a = vdupq_n_s32(0);
        for (int i = 0; i < n16; i += 16) {
            int8x16_t xv = vld1q_s8(l->xq + i), v = vld1q_s8(w + i);
            a = vpadalq_s16(a, vmull_s8(vget_low_s8(v), vget_low_s8(xv)));
            a = vpadalq_s16(a, vmull_s8(vget_high_s8(v), vget_high_s8(xv)));
        }
        int32_t s4[4];
        vst1q_s32(s4, a);
        int32_t t = s4[0] + s4[1] + s4[2] + s4[3];
        for (int i = n16; i < l->in; i++) t += (int32_t)w[i] * l->xq[i];
        l->acc[o] = t;
    }
#else
    for (int o = 0; o < l->out; o++) {
        int32_t s = 0;
        const int8_t *w = l->w + (size_t)o * l->in;
        for (int i = 0; i < l->in; i++) s += (int32_t)w[i] * l->xq[i];
        l->acc[o] = s;
    }
#endif
    q8_finish(l, y);
}

void f32_init(F32Layer *l, int in, int out, int act, const float *w, const float *bias) {
    l->in = in; l->out = out; l->act = act; l->outp = (out + 15) & ~15;
    l->w = malloc(sizeof(float) * in * out); memcpy(l->w, w, sizeof(float) * in * out);
    l->bias = malloc(sizeof(float) * out); memcpy(l->bias, bias, sizeof(float) * out);
    l->wt = calloc((size_t)in * l->outp, sizeof(float));
    for (int o = 0; o < out; o++)
        for (int i = 0; i < in; i++) l->wt[(size_t)i * l->outp + o] = w[(size_t)o * in + i];
}

void f32_forward_ref(const F32Layer *l, const float *x, float *y) {
    for (int o = 0; o < l->out; o++) {
        float s = l->bias[o];
        for (int i = 0; i < l->in; i++) s += l->w[(size_t)o * l->in + i] * x[i];
        y[o] = l->act == ACT_RELU && s < 0.0f ? 0.0f : s;
    }
    smooth_ref(l->act, y, l->out);
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static float hsum_f32(float32x4_t v) {
    float32x2_t p = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(p, p), 0);
}

void f32_forward_fast(const F32Layer *l, const float *x, float *y) {
    if (l->in < 8) {
        for (int c = 0; c < l->out; c += 16) {
            float32x4_t a[4];
            for (int r = 0; r < 4; r++) a[r] = vdupq_n_f32(0.0f);
            for (int i = 0; i < l->in; i++) {
                const float *w = l->wt + (size_t)i * l->outp + c;
                for (int r = 0; r < 4; r++) a[r] = vmlaq_n_f32(a[r], vld1q_f32(w + 4 * r), x[i]);
            }
            float t[16];
            for (int r = 0; r < 4; r++) vst1q_f32(t + 4 * r, a[r]);
            for (int o = c; o < l->out && o < c + 16; o++) {
                float s = l->bias[o] + t[o - c];
                y[o] = l->act == ACT_RELU && s < 0.0f ? 0.0f : s;
            }
        }
        smooth_fast_n(l->act, y, l->out);
        return;
    }
    int o = 0;
    for (; o + 4 <= l->out; o += 4) {
        const float *w0 = l->w + (size_t)o * l->in, *w1 = w0 + l->in, *w2 = w1 + l->in, *w3 = w2 + l->in;
        float32x4_t a[8];
        for (int q = 0; q < 8; q++) a[q] = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= l->in; i += 8) {
            float32x4_t x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
            a[0] = vmlaq_f32(a[0], vld1q_f32(w0 + i), x0); a[1] = vmlaq_f32(a[1], vld1q_f32(w0 + i + 4), x1);
            a[2] = vmlaq_f32(a[2], vld1q_f32(w1 + i), x0); a[3] = vmlaq_f32(a[3], vld1q_f32(w1 + i + 4), x1);
            a[4] = vmlaq_f32(a[4], vld1q_f32(w2 + i), x0); a[5] = vmlaq_f32(a[5], vld1q_f32(w2 + i + 4), x1);
            a[6] = vmlaq_f32(a[6], vld1q_f32(w3 + i), x0); a[7] = vmlaq_f32(a[7], vld1q_f32(w3 + i + 4), x1);
        }
        float s[4] = {hsum_f32(vaddq_f32(a[0], a[1])), hsum_f32(vaddq_f32(a[2], a[3])),
                      hsum_f32(vaddq_f32(a[4], a[5])), hsum_f32(vaddq_f32(a[6], a[7]))};
        const float *ws[4] = {w0, w1, w2, w3};
        for (int r = 0; r < 4; r++) {
            for (int j = i; j < l->in; j++) s[r] += ws[r][j] * x[j];
            float v = l->bias[o + r] + s[r];
            y[o + r] = l->act == ACT_RELU && v < 0.0f ? 0.0f : v;
        }
    }
    for (; o < l->out; o++) {
        const float *w = l->w + (size_t)o * l->in;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0;
        int i = 0;
        for (; i + 8 <= l->in; i += 8) {
            a0 = vmlaq_f32(a0, vld1q_f32(w + i), vld1q_f32(x + i));
            a1 = vmlaq_f32(a1, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
        }
        float s = hsum_f32(vaddq_f32(a0, a1));
        for (; i < l->in; i++) s += w[i] * x[i];
        float v = l->bias[o] + s;
        y[o] = l->act == ACT_RELU && v < 0.0f ? 0.0f : v;
    }
    smooth_fast_n(l->act, y, l->out);
}
#else
void f32_forward_fast(const F32Layer *l, const float *x, float *y) {
    for (int o = 0; o < l->out; o++) {
        float s = l->bias[o];
        for (int i = 0; i < l->in; i++) s += l->w[(size_t)o * l->in + i] * x[i];
        y[o] = l->act == ACT_RELU && s < 0.0f ? 0.0f : s;
    }
    smooth_fast_n(l->act, y, l->out);
}
#endif

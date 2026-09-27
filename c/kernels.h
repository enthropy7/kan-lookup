#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

typedef struct {
    int in, out, outp, q;
    float *lo, *hi, *inv_step, *scale;
    int8_t *table;
    int16_t *t16;
    int32_t *acc;
    const int8_t **rows;
} LutLayer;

enum { ACT_NONE = 0, ACT_RELU = 1, ACT_SILU = 2, ACT_TANH = 3 };

typedef struct {
    int in, out, act;
    float act_scale, inv_act_scale, *w_scale, *bias;
    float pre, *table;
    int8_t *w;
    int8_t *wt;
    int8_t *xq;
    int32_t *acc;
} Q8Layer;

typedef struct {
    int in, out, outp, act;
    float *w;
    float *wt;
    float *bias;
} F32Layer;

void lut_init(LutLayer *l, int in, int out, int q, const float *lo, const float *hi, const float *scale,
              const int8_t *table_oiq);
void lut_forward_ref(LutLayer *l, const float *x, float *y);
void lut_forward_fast(LutLayer *l, const float *x, float *y);

void lut_forward_fast_rt(LutLayer *l, const float *x, float *y);

void lut16_init(LutLayer *l, int in, int out, int q, const float *lo, const float *hi, const float *scale,
                const int16_t *table_oiq);
void lut16_forward_ref(LutLayer *l, const float *x, float *y);
void lut16_forward_fast(LutLayer *l, const float *x, float *y);
void lin16_forward_ref(LutLayer *l, const float *x, float *y);
void lin16_forward_fast(LutLayer *l, const float *x, float *y);

void q8_init(Q8Layer *l, int in, int out, int act, float act_scale, const float *w_scale, const int8_t *w,
             const float *bias);

void q8_set_table(Q8Layer *l, float pre_scale, const float *table);
void q8_forward_ref(Q8Layer *l, const float *x, float *y);
void q8_forward_fast(Q8Layer *l, const float *x, float *y);

void f32_init(F32Layer *l, int in, int out, int act, const float *w, const float *bias);
void f32_forward_ref(const F32Layer *l, const float *x, float *y);
void f32_forward_fast(const F32Layer *l, const float *x, float *y);

#endif

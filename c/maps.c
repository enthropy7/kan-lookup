#include "maps.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static inline int cell(float x, int n, float *f) {
    x = x < -1.0f ? -1.0f : x > 1.0f ? 1.0f : x;
    const float u = (x + 1.0f) * 0.5f * (float)(n - 1);
    int k = (int)u;
    k = k > n - 2 ? n - 2 : k;
    *f = u - (float)k;
    return k;
}

float grid_forward(const GridMap *m, const float *x) {
    float f[16], c[1 << 10];
    int base = 0;
    for (int a = 0; a < m->d; a++) base = base * m->n + cell(x[a], m->n, &f[a]);
    const int corners = 1 << m->d;
    for (int j = 0; j < corners; j++) {
        int off = 0;
        for (int a = 0; a < m->d; a++) off = off * m->n + ((j >> (m->d - 1 - a)) & 1);
        c[j] = (float)m->v[base + off];
    }
    for (int a = m->d - 1, len = corners; a >= 0; a--) {
        len >>= 1;
        for (int j = 0; j < len; j++) c[j] = c[2 * j] + f[a] * (c[2 * j + 1] - c[2 * j]);
    }
    return m->scale * c[0];
}

float gridv_forward(const GridVMap *m, const float *x) {
    float f[16], c[1 << 10];
    int base = 0, stride[16];
    for (int a = 0; a < m->d; a++) base = base * m->n[a] + cell(x[a], m->n[a], &f[a]);
    stride[m->d - 1] = 1;
    for (int a = m->d - 2; a >= 0; a--) stride[a] = stride[a + 1] * m->n[a + 1];
    const int corners = 1 << m->d;
    for (int j = 0; j < corners; j++) {
        int off = 0;
        for (int a = 0; a < m->d; a++) off += ((j >> (m->d - 1 - a)) & 1) * stride[a];
        c[j] = (float)m->v[base + off];
    }
    for (int a = m->d - 1, len = corners; a >= 0; a--) {
        len >>= 1;
        for (int j = 0; j < len; j++) c[j] = c[2 * j] + f[a] * (c[2 * j + 1] - c[2 * j]);
    }
    return m->scale * c[0];
}

float cp_forward(const CPMap *m, const float *x) {
    int k[16];
    float f[16], y = m->b;
    for (int i = 0; i < m->d; i++) k[i] = cell(x[i], m->q, &f[i]);
    for (int r = 0; r < m->r; r++) {
        float p = m->w[r];
        for (int i = 0; i < m->d; i++) {
            const int16_t *t = m->t + ((size_t)r * m->d + i) * m->q + k[i];
            p *= m->s[r * m->d + i] * ((float)t[0] + f[i] * (float)(t[1] - t[0]));
        }
        y += p;
    }
    return y;
}

float ga2m_forward(const GA2MMap *m, const float *x) {
    int k1[16], k2[16];
    float f1[16], f2[16], y = m->b;
    for (int i = 0; i < m->d; i++) {
        k1[i] = cell(x[i], m->q1, &f1[i]);
        k2[i] = cell(x[i], m->q2, &f2[i]);
        const int16_t *g = m->g + (size_t)i * m->q1 + k1[i];
        y += m->gs[i] * ((float)g[0] + f1[i] * (float)(g[1] - g[0]));
    }
    int p = 0;
    for (int a = 0; a < m->d; a++)
        for (int c = a + 1; c < m->d; c++, p++) {
            const int16_t *h = m->h + ((size_t)p * m->q2 + k2[a]) * m->q2 + k2[c];
            const float fa = f2[a], fc = f2[c];
            const float v0 = (float)h[0] + fc * (float)(h[1] - h[0]);
            const float v1 = (float)h[m->q2] + fc * (float)(h[m->q2 + 1] - h[m->q2]);
            y += m->hs[p] * (v0 + fa * (v1 - v0));
        }
    return y;
}

/* Piecewise-linear sparse grid (kantab/sparse.py). mod: level 1 is the constant, from level 2 the outermost hats
   extrapolate linearly to the boundary. bound: level 0 holds the two boundary functions, both nonzero everywhere. */
float sgrid_forward(const SGridMap *m, const float *x) {
    enum { L = 24 };
    int idx[16][L];
    float w[16][L], wb[16][2];
    int top = 0;
    for (size_t j = 0; j < (size_t)m->s * m->d; j++) top = m->levels[j] > top ? m->levels[j] : top;
    for (int a = 0; a < m->d; a++) {
        const float xa = x[a] < -1.0f ? -1.0f : x[a] > 1.0f ? 1.0f : x[a];
        wb[a][0] = 0.5f * (1.0f - xa);
        wb[a][1] = 0.5f * (1.0f + xa);
        for (int l = 1; l <= top; l++) {
            const int n = 1 << (l - 1);
            const float h = 2.0f / (float)(1 << l);
            int i = (int)((xa + 1.0f) / (2.0f * h));
            i = i > n - 1 ? n - 1 : i;
            float v;
            if (!m->bound && l == 1) v = 1.0f;
            else if (!m->bound && i == 0) v = 2.0f - (xa + 1.0f) / h;
            else if (!m->bound && i == n - 1) v = 2.0f - (1.0f - xa) / h;
            else v = 1.0f - fabsf(xa - (-1.0f + (float)(2 * i + 1) * h)) / h;
            idx[a][l] = i;
            w[a][l] = v > 0.0f ? v : 0.0f;
        }
    }
    float y = 0.0f;
    size_t off = 0;
    for (int k = 0; k < m->s; k++) {
        const uint8_t *lv = m->levels + (size_t)k * m->d;
        int zero[16], z = 0, size = 1, base = 0, stride[16];
        float c = 1.0f;
        for (int a = m->d - 1; a >= 0; a--) {
            const int cnt = lv[a] == 0 ? 2 : 1 << (lv[a] - 1);
            stride[a] = size;
            size *= cnt;
            if (lv[a] == 0) zero[z++] = a;
            else { base += idx[a][lv[a]] * stride[a]; c *= w[a][lv[a]]; }
        }
        float s = 0.0f;
        for (int j = 0; j < 1 << z; j++) {
            float cj = c;
            int o = base;
            for (int b = 0; b < z; b++) {
                const int bit = (j >> b) & 1;
                cj *= wb[zero[b]][bit];
                o += bit * stride[zero[b]];
            }
            s += cj * (float)m->v[off + o];
        }
        y += m->scale[k] * s;
        off += (size_t)size;
    }
    return y;
}

int sgrid_plan(const SGridMap *m, SGridPlan *p) {
    if (m->bound) return -1;
    size_t active = 0;
    int top = 1;
    for (size_t j = 0; j < (size_t)m->s * m->d; j++) {
        active += m->levels[j] > 1;
        top = m->levels[j] > top ? m->levels[j] : top;
    }
    p->m = m;
    p->top = top;
    p->off = malloc(sizeof(int32_t) * m->s);
    p->start = malloc(sizeof(int32_t) * (m->s + 1));
    p->dim = malloc(active + 1);
    p->lev = malloc(active + 1);
    p->stride = malloc(sizeof(int32_t) * (active + 1));
    if (!p->off || !p->start || !p->dim || !p->lev || !p->stride) {
        free(p->off); free(p->start); free(p->dim); free(p->lev); free(p->stride);
        return -1;
    }
    int32_t off = 0;
    size_t n = 0;
    for (int k = 0; k < m->s; k++) {
        const uint8_t *lv = m->levels + (size_t)k * m->d;
        int32_t size = 1;
        p->off[k] = off;
        p->start[k] = (int32_t)n;
        for (int a = m->d - 1; a >= 0; a--) {
            if (lv[a] > 1) { p->dim[n] = (uint8_t)a; p->lev[n] = lv[a]; p->stride[n] = size; n++; }
            size *= 1 << (lv[a] - 1);
        }
        off += size;
    }
    p->start[m->s] = (int32_t)n;
    return 0;
}

float sgrid_forward_plan(const SGridPlan *p, const float *x) {
    enum { L = 24 };
    const SGridMap *m = p->m;
    int idx[16][L];
    float w[16][L];
    for (int a = 0; a < m->d; a++) {
        const float xa = x[a] < -1.0f ? -1.0f : x[a] > 1.0f ? 1.0f : x[a];
        for (int l = 2; l <= p->top; l++) {
            const int n = 1 << (l - 1);
            const float h = 2.0f / (float)(1 << l);
            int i = (int)((xa + 1.0f) / (2.0f * h));
            i = i > n - 1 ? n - 1 : i;
            const float v = i == 0 ? 2.0f - (xa + 1.0f) / h : i == n - 1 ? 2.0f - (1.0f - xa) / h
                                   : 1.0f - fabsf(xa - (-1.0f + (float)(2 * i + 1) * h)) / h;
            idx[a][l] = i;
            w[a][l] = v > 0.0f ? v : 0.0f;
        }
    }
    float y = 0.0f;
    for (int k = 0; k < m->s; k++) {
        int32_t o = p->off[k];
        float c = m->scale[k];
        for (int j = p->start[k]; j < p->start[k + 1]; j++) {
            o += idx[p->dim[j]][p->lev[j]] * p->stride[j];
            c *= w[p->dim[j]][p->lev[j]];
        }
        y += c * (float)m->v[o];
    }
    return y;
}

int sgrid_fast(const SGridMap *m, SGridFast *p) {
    SGridPlan q;
    if (sgrid_plan(m, &q)) return -1;
    const int32_t n = q.start[m->s];
    p->m = m;
    p->top = q.top;
    p->off = q.off;
    p->start = q.start;
    p->stride = q.stride;
    p->key = malloc(sizeof(uint16_t) * (size_t)(n + 1));
    if (p->key)
        for (int32_t j = 0; j < n; j++) p->key[j] = (uint16_t)(q.dim[j] * 24 + q.lev[j]);
    free(q.dim);
    free(q.lev);
    if (!p->key) {
        free(q.off); free(q.start); free(q.stride);
        return -1;
    }
    return 0;
}

float sgrid_forward_fast(const SGridFast *p, const float *x) {
    const SGridMap *m = p->m;
    int32_t idx[16 * 24];
    float w[16 * 24];
    for (int a = 0; a < m->d; a++) {
        const float xa = x[a] < -1.0f ? -1.0f : x[a] > 1.0f ? 1.0f : x[a];
        float s = (xa + 1.0f) * 0.5f;   /* doubled at each level: the position in units of the level's cells */
        for (int l = 2; l <= p->top; l++) {
            s *= 2.0f;
            const int n = 1 << (l - 1);
            int i = (int)s;
            i = i > n - 1 ? n - 1 : i;
            float v = 1.0f - fabsf(2.0f * (s - (float)i) - 1.0f);
            if (i == 0) v = 2.0f - 2.0f * s;
            if (i == n - 1) v = 2.0f - 2.0f * ((float)n - s);
            idx[a * 24 + l] = i;
            w[a * 24 + l] = v > 0.0f ? v : 0.0f;
        }
    }
    /* in blocks: first every offset and weight, then the surplus reads, which then do not wait on each other */
    enum { B = 64 };
    int32_t o[B];
    float c[B], y = 0.0f;
    for (int k0 = 0; k0 < m->s; k0 += B) {
        const int nb = m->s - k0 < B ? m->s - k0 : B;
        for (int b = 0; b < nb; b++) {
            const int k = k0 + b;
            int32_t ok = p->off[k];
            float ck = m->scale[k];
            for (int32_t j = p->start[k]; j < p->start[k + 1]; j++) {
                const int key = p->key[j];
                ok += idx[key] * p->stride[j];
                ck *= w[key];
            }
            o[b] = ok;
            c[b] = ck;
        }
        for (int b = 0; b < nb; b++) y += c[b] * (float)m->v[o[b]];
    }
    return y;
}

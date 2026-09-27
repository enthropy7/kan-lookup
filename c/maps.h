#ifndef MAPS_H
#define MAPS_H
#include <stdint.h>

typedef struct { int d, n; float scale; const int16_t *v; } GridMap;
typedef struct { int d, n[16]; float scale; const int16_t *v; } GridVMap;
typedef struct { int d, r, q; float b; const float *w, *s; const int16_t *t; } CPMap;
typedef struct { int d, q1, q2; float b; const float *gs, *hs; const int16_t *g, *h; } GA2MMap;
/* s level vectors of d levels each, surpluses in row-major order per level vector, one scale per level vector */
typedef struct { int d, s, bound; const uint8_t *levels; const float *scale; const int16_t *v; } SGridMap;

float grid_forward(const GridMap *m, const float *x);
float gridv_forward(const GridVMap *m, const float *x);
float cp_forward(const CPMap *m, const float *x);
float ga2m_forward(const GA2MMap *m, const float *x);
float sgrid_forward(const SGridMap *m, const float *x);
/* Built once at start-up from a mod-basis SGridMap: per level vector the offset of its surpluses and the inputs
   whose level is above 1, with their levels and strides (level 1 is the constant, index 0). */
typedef struct { const SGridMap *m; int top; int32_t *off, *start; uint8_t *dim, *lev; int32_t *stride; } SGridPlan;
int sgrid_plan(const SGridMap *m, SGridPlan *p);
float sgrid_forward_plan(const SGridPlan *p, const float *x);
/* The same function with the hats from multiplications by powers of two and one lookup per active input, in a table
   keyed by input * 24 + level; the surpluses are read in blocks after their offsets are known. */
typedef struct { const SGridMap *m; int top; int32_t *off, *start, *stride; uint16_t *key; } SGridFast;
int sgrid_fast(const SGridMap *m, SGridFast *p);
float sgrid_forward_fast(const SGridFast *p, const float *x);
#endif

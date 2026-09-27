#include "baselines.h"

#include <stdlib.h>
#include <string.h>

void poly_init(Poly *p, int terms, const uint16_t *parent, const uint8_t *var, const float *coef) {
    p->terms = terms;
    p->parent = malloc(sizeof(uint16_t) * terms); memcpy(p->parent, parent, sizeof(uint16_t) * terms);
    p->var = malloc(terms); memcpy(p->var, var, terms);
    p->coef = malloc(sizeof(float) * terms); memcpy(p->coef, coef, sizeof(float) * terms);
    p->m = malloc(sizeof(float) * terms);
}

float poly_forward(Poly *p, const float *x) {
    float s = p->coef[0];
    p->m[0] = 1.0f;
    for (int t = 1; t < p->terms; t++) {
        p->m[t] = p->m[p->parent[t]] * x[p->var[t]];
        s += p->coef[t] * p->m[t];
    }
    return s;
}

float forest_forward(const Forest *f, const float *x) {
    float s = f->base;
    for (int t = 0; t < f->trees; t++) {
        const TreeNode *n = f->nodes + f->root[t];
        unsigned i = 0;
        while (n[i].feat != TREE_LEAF) i = x[n[i].feat] <= n[i].v ? i + 1 : n[i].right;
        s += n[i].v;
    }
    return s;
}

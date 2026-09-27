#ifndef BASELINES_H
#define BASELINES_H

#include <stdint.h>

typedef struct {
    int terms;
    uint16_t *parent;
    uint8_t *var;
    float *coef;
    float *m;
} Poly;

void poly_init(Poly *p, int terms, const uint16_t *parent, const uint8_t *var, const float *coef);
float poly_forward(Poly *p, const float *x);

#define TREE_LEAF 0xFFFFu

typedef struct {
    float v;
    uint16_t feat, right;
} TreeNode;

typedef struct {
    int trees;
    float base;
    int32_t *root;
    TreeNode *nodes;
} Forest;

float forest_forward(const Forest *f, const float *x);

#endif

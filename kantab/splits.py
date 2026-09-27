import numpy as np


def folds(groups, k):
    ids = np.unique(groups)
    return [ids[i::k] for i in range(k)]


def scale_inputs(x_train, *others):
    lo, hi = x_train.min(0), x_train.max(0)
    hi = np.where(hi > lo, hi, lo + 1.0)
    f = lambda x: np.clip(2 * (x - lo) / (hi - lo) - 1, -1, 1).astype(np.float32)
    return (f(x_train), *(f(o) for o in others))

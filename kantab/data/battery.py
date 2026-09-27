import os

import numpy as np

from kantab.splits import scale_inputs

TASKS = ("lg_soc",)
TEST_TEMPS = ("n10degC", "0degC", "10degC", "25degC")
VAL_BLOCKS, N_BLOCKS = (2, 7, 12, 17), 20


def _mat(path):
    import scipy.io
    m = scipy.io.loadmat(path)
    return m["X"].T.astype(np.float64), m["Y"].ravel().astype(np.float64)


def load(task):
    assert task == "lg_soc", task
    base = os.environ["KANTAB_LG_HG2"] + "/prepared"
    xtr, ytr = _mat(base + "/Train/TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat")
    parts = [_mat(base + f"/Test/0{i + 1}_TEST_LGHG2@{t}_Norm_(05_Inputs).mat") for i, t in enumerate(TEST_TEMPS)]
    xte = np.concatenate([p[0] for p in parts])
    yte = np.concatenate([p[1] for p in parts])
    temp = np.concatenate([np.full(len(p[1]), i) for i, p in enumerate(parts)])
    return xtr, ytr, xte, yte, temp


def split(task):
    x, y, xte, yte, _ = load(task)
    # validation by time blocks, not random rows: neighbouring samples of a drive cycle are correlated
    block = np.minimum(np.arange(len(x)) * N_BLOCKS // len(x), N_BLOCKS - 1)
    val = np.isin(block, VAL_BLOCKS)
    xtr, xva, xte = scale_inputs(x[~val], x[val], xte)
    mean, std = y[~val].mean(), y[~val].std()
    z = lambda v: ((v - mean) / std).astype(np.float32)
    return (xtr, z(y[~val])), (xva, z(y[val])), (xte, z(yte)), (float(mean), float(std))

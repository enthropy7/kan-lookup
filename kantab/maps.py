import itertools
import math

import numpy as np
import torch
import torch.nn as nn


def _cell(x, n):
    u = (x.clamp(-1, 1) + 1) / 2 * (n - 1)
    k = u.floor().clamp(max=n - 2)
    return k.long(), u - k


def q16(t, dims):
    s = t.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12) / 32767
    return (t / s).round().clamp(-32767, 32767) * s


class CPMap(nn.Module):
    def __init__(self, d, rank, q):
        super().__init__()
        self.d, self.rank, self.q = d, rank, q
        self.tables = nn.Parameter(1 + 0.1 * torch.randn(rank, d, q))
        self.w = nn.Parameter(0.1 * torch.randn(rank))
        self.b = nn.Parameter(torch.zeros(1))
    def factors(self, x, tables):
        k, f = _cell(x, self.q)
        i = torch.arange(self.d).unsqueeze(0)
        t0, t1 = tables[:, i, k], tables[:, i, k + 1]
        return t0 + f.unsqueeze(0) * (t1 - t0)
    def forward(self, x, tables=None):
        g = self.factors(x, self.tables if tables is None else tables)
        return (self.b + (self.w.unsqueeze(1) * g.prod(2)).sum(0)).unsqueeze(1)
    def deployed(self):
        tables = q16(self.tables.detach(), (2,))
        return lambda x: self.forward(x, tables)
    def n_bytes(self):
        return 2 * self.tables.numel() + 4 * self.rank * self.d + 4 * self.rank + 4


class GA2M(nn.Module):
    def __init__(self, d, q1, q2):
        super().__init__()
        self.d, self.q1, self.q2 = d, q1, q2
        self.pairs = list(itertools.combinations(range(d), 2))
        self.g = nn.Parameter(torch.zeros(d, q1))
        self.h = nn.Parameter(torch.zeros(len(self.pairs), q2, q2))
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, x, g=None, h=None):
        g = self.g if g is None else g
        h = self.h if h is None else h
        k, f = _cell(x, self.q1)
        i = torch.arange(self.d).unsqueeze(0)
        g0, g1 = g[i, k], g[i, k + 1]
        out = self.b + (g0 + f * (g1 - g0)).sum(1)
        k2, f2 = _cell(x, self.q2)
        for p, (a, c) in enumerate(self.pairs):
            ka, kc, fa, fc = k2[:, a], k2[:, c], f2[:, a], f2[:, c]
            v00, v01 = h[p, ka, kc], h[p, ka, kc + 1]
            v10, v11 = h[p, ka + 1, kc], h[p, ka + 1, kc + 1]
            out = out + (1 - fa) * ((1 - fc) * v00 + fc * v01) + fa * ((1 - fc) * v10 + fc * v11)
        return out.unsqueeze(1)
    def deployed(self):
        g, h = q16(self.g.detach(), (1,)), q16(self.h.detach(), (1, 2))
        return lambda x: self.forward(x, g, h)
    def n_bytes(self):
        return 2 * (self.g.numel() + self.h.numel()) + 4 * (self.d + len(self.pairs)) + 4


def _counts(n, d):
    return (n,) * d if isinstance(n, int) else tuple(int(v) for v in n)


def grid_size(n, d):
    cells = math.prod(_counts(n, d))
    return cells, 2 * cells + 4 + (0 if isinstance(n, int) else 4 * d)


def grid_counts(curv, max_nodes):
    n = [2] * len(curv)
    while True:
        best, gain = None, 0.0
        for a, c in enumerate(curv):
            m = n.copy()
            m[a] += 1
            if math.prod(m) > max_nodes:
                continue
            g = c * ((2 / (n[a] - 1)) ** 2 - (2 / n[a]) ** 2) / math.log((n[a] + 1) / n[a])
            if best is None or g > gain:
                best, gain = a, g
        if best is None:
            return tuple(n)
        n[best] += 1


def grid_weights(x, n):
    x = torch.as_tensor(x, dtype=torch.float64)
    N, d = x.shape
    if not isinstance(n, int):
        ns = _counts(n, d)
        k, f = _cell(x, torch.tensor(ns))
        strides = torch.tensor([math.prod(ns[a + 1:]) for a in range(d)])
        rows, cols, vals = [], [], []
        for corner in itertools.product((0, 1), repeat=d):
            c = torch.tensor(corner)
            w = torch.where(c.bool(), f, 1 - f).prod(1)
            rows.append(torch.arange(N))
            cols.append(((k + c) * strides).sum(1))
            vals.append(w)
        return torch.cat(rows).numpy(), torch.cat(cols).numpy(), torch.cat(vals).numpy()
    k, f = _cell(x, n)
    strides = torch.tensor([n ** (d - 1 - a) for a in range(d)])
    rows, cols, vals = [], [], []
    for corner in itertools.product((0, 1), repeat=d):
        c = torch.tensor(corner)
        w = torch.where(c.bool(), f, 1 - f).prod(1)
        idx = ((k + c) * strides).sum(1)
        rows.append(torch.arange(N))
        cols.append(idx)
        vals.append(w)
    return torch.cat(rows).numpy(), torch.cat(cols).numpy(), torch.cat(vals).numpy()


def _smoothness(n, d):
    import scipy.sparse as sp
    ns = _counts(n, d)
    rows, cols, vals, r = [], [], [], 0
    strides = [math.prod(ns[a + 1:]) for a in range(d)]
    for p in itertools.product(*(range(v) for v in ns)):
        base = sum(pi * s for pi, s in zip(p, strides))
        for a in range(d):
            if 0 < p[a] < ns[a] - 1:
                rows += [r, r, r]
                cols += [base - strides[a], base, base + strides[a]]
                vals += [1.0, -2.0, 1.0]
                r += 1
    return sp.csr_matrix((vals, (rows, cols)), shape=(r, math.prod(ns)))


def eval_grid(data, n, values):
    import scipy.sparse as sp
    _, (xva, yva), (xte, yte), (_, std) = data
    d = xva.shape[1]
    cells, n_bytes = grid_size(n, d)
    s = np.abs(values).max() / 32767
    vq = np.round(values / s) * s
    res = {"val": {}, "test": {}, "n_bytes": n_bytes}
    for split, xx, yy in (("val", xva, yva), ("test", xte, yte)):
        r, c, v = grid_weights(xx, n)
        A = sp.csr_matrix((v, (r, c)), shape=(len(xx), cells))
        for name, vv in (("fp64", values), ("i16", vq)):
            res[split][name] = float(std * np.sqrt(np.mean((A @ vv - yy) ** 2)))
    return res


def fit_grid(data, n, lams, iter_lim=3000):
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr
    (xtr, ytr), (xva, yva), (xte, yte), (_, std) = data
    d = xtr.shape[1]
    cells, n_bytes = grid_size(n, d)
    r, c, v = grid_weights(xtr, n)
    A = sp.csr_matrix((v, (r, c)), shape=(len(xtr), cells))
    L = _smoothness(n, d)
    rhs = np.concatenate([ytr.astype(np.float64) / math.sqrt(len(xtr)), np.zeros(L.shape[0])])
    evals = {name: sp.csr_matrix((vv, (rr, cc)), shape=(len(xx), cells))
             for name, xx in (("val", xva), ("test", xte)) for rr, cc, vv in [grid_weights(xx, n)]}
    ys = {"val": yva, "test": yte}
    out, x0 = {}, None
    for lam in sorted(lams, reverse=True):
        M = sp.vstack([A / math.sqrt(len(xtr)), L * math.sqrt(lam / max(L.shape[0], 1))]).tocsr()
        values = lsqr(M, rhs, iter_lim=iter_lim, atol=1e-10, btol=1e-10, x0=x0)[0]
        x0 = values
        s = np.abs(values).max() / 32767
        vq = np.round(values / s) * s
        res = {"val": {}, "test": {}, "n_bytes": n_bytes}
        for split in ("val", "test"):
            for name, vv in (("fp64", values), ("i16", vq)):
                res[split][name] = float(std * np.sqrt(np.mean((evals[split] @ vv - ys[split]) ** 2)))
        out[lam] = (res, values)
    return out

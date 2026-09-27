"""Sparse grids of piecewise-linear hierarchical functions on [-1, 1]^d with int16 surpluses.

Two one-dimensional bases. "bound" puts the two boundary nodes at level 0 and hats from level 1 on. "mod" has no
boundary nodes: level 1 is the constant, and from level 2 on the outermost hats extrapolate linearly to the
boundary (the modified linear basis of Pflüger, 2010). A grid is a downward-closed set of level vectors; each level
vector holds the product of its one-dimensional levels, and its surpluses are stored in row-major order. The
surpluses come from interpolating a function at the nodes or from least squares on data."""
import math

import numpy as np

BASE = {"bound": 0, "mod": 1}


def count(basis, level):
    return 2 if basis == "bound" and level == 0 else 2 ** (level - 1)


def nodes_1d(basis, level):
    if basis == "bound" and level == 0:
        return np.array([-1.0, 1.0])
    h = 2.0 ** (1 - level)
    return -1 + (2 * np.arange(2 ** (level - 1)) + 1) * h


def eval_1d(basis, level, x):
    """The functions of one level that are nonzero at x, as (index, value) pairs: one pair, two at level 0 of bound."""
    if basis == "bound" and level == 0:
        return [(np.zeros(len(x), np.int64), (1 - x) / 2), (np.ones(len(x), np.int64), (1 + x) / 2)]
    if basis == "mod" and level == 1:
        return [(np.zeros(len(x), np.int64), np.ones_like(x))]
    h, n = 2.0 ** (1 - level), 2 ** (level - 1)
    i = np.clip(np.floor((x + 1) / (2 * h)).astype(np.int64), 0, n - 1)
    v = np.maximum(0, 1 - np.abs(x - (-1 + (2 * i + 1) * h)) / h)
    if basis == "mod":
        v = np.where(i == 0, np.maximum(0, 2 - (x + 1) / h), v)
        v = np.where(i == n - 1, np.maximum(0, 2 - (1 - x) / h), v)
    return [(i, v)]


class Grid:
    def __init__(self, basis, levels):
        self.basis, self.levels = basis, [tuple(int(a) for a in l) for l in levels]
        self.d = len(self.levels[0])
        self.L = np.array(self.levels, dtype=np.int64)
        sizes = [math.prod(count(basis, a) for a in l) for l in self.levels]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.alpha = np.zeros(self.n_points)

    @property
    def n_points(self):
        return int(self.offsets[-1])

    def n_bytes(self, adaptive):
        # int16 surpluses, a float32 scale per level vector, and for an adaptive grid its level vectors (a byte each)
        return 2 * self.n_points + 4 * len(self.levels) + (self.d * len(self.levels) if adaptive else 0) + 4

    def nodes(self, k):
        axes = [nodes_1d(self.basis, a) for a in self.levels[k]]
        return np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)

    def all_nodes(self):
        return np.concatenate([self.nodes(k) for k in range(len(self.levels))])

    def _terms(self, k, table):
        """Column indices and values of level vector k at the points of a per-dimension table of eval_1d."""
        l = self.levels[k]
        strides = [math.prod(count(self.basis, b) for b in l[a + 1:]) for a in range(self.d)]
        n = len(table[0][l[0]][0][0])
        terms = [(np.full(n, self.offsets[k]), np.ones(n))]
        for a in range(self.d):
            terms = [(c + i * strides[a], w * v) for c, w in terms for i, v in table[a][l[a]]]
        return terms

    def _table(self, x):
        top = max(max(l) for l in self.levels)
        return [{lev: eval_1d(self.basis, lev, x[:, a]) for lev in range(BASE[self.basis], top + 1)} for a in range(self.d)]

    def __call__(self, x, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        table = self._table(np.clip(np.asarray(x, dtype=np.float64), -1, 1))
        y = np.zeros(len(x))
        for k in range(len(self.levels)):
            for c, w in self._terms(k, table):
                y += alpha[c] * w
        return y

    def add_interpolated(self, k, values):
        """Surpluses of level vector k from function values at its nodes, the others being set: the coarser level
        vectors below it are the only ones nonzero at its nodes."""
        x = self.nodes(k)
        table = self._table(x)
        below = [j for j in np.flatnonzero(np.all(self.L <= self.L[k], axis=1)) if j != k]
        r = np.array(values, dtype=np.float64)
        for j in below:
            for c, w in self._terms(j, table):
                r -= self.alpha[c] * w
        self.alpha[self.offsets[k]:self.offsets[k + 1]] = r

    def interpolate(self, f):
        """Surpluses that interpolate f (a function of an array of nodes) at every node."""
        for k in sorted(range(len(self.levels)), key=lambda k: sum(self.levels[k])):
            self.add_interpolated(k, f(self.nodes(k)))

    def design(self, x, scale=1.0):
        """The matrix of basis values at x times scale. Every row has the same number of terms, so it is built in
        CSR form directly, with 32-bit column indices."""
        import scipy.sparse as sp
        table = self._table(np.clip(np.asarray(x, dtype=np.float64), -1, 1))
        terms = [t for k in range(len(self.levels)) for t in self._terms(k, table)]
        cols = np.empty((len(x), len(terms)), dtype=np.int32)
        vals = np.empty((len(x), len(terms)))
        for j, (c, w) in enumerate(terms):
            cols[:, j], vals[:, j] = c, w * scale
        indptr = np.arange(0, cols.size + 1, len(terms), dtype=np.int64)
        return sp.csr_matrix((vals.ravel(), cols.ravel(), indptr), shape=(len(x), self.n_points))

    def quantized(self):
        """Surpluses rounded to int16 with one scale per level vector."""
        q = self.alpha.copy()
        for k in range(len(self.levels)):
            a = q[self.offsets[k]:self.offsets[k + 1]]
            s = np.abs(a).max() / 32767 if np.abs(a).max() > 0 else 1.0
            q[self.offsets[k]:self.offsets[k + 1]] = np.round(a / s) * s
        return q


def regular(basis, d, n):
    """All level vectors with |l - base|_1 <= n - 1 over the base level."""
    b = BASE[basis]
    def spread(d, left):
        if d == 0:
            yield ()
            return
        for a in range(left + 1):
            for rest in spread(d - 1, left - a):
                yield (a,) + rest
    return Grid(basis, [tuple(b + a for a in l) for l in spread(d, n - 1)])


def _extended(grid, level):
    bigger = Grid(grid.basis, grid.levels + [level])
    bigger.alpha[:grid.n_points] = grid.alpha
    return bigger


def adaptive(basis, d, f, max_bytes, w):
    """Dimension-adaptive refinement (Gerstner and Griebel, 2003). A level vector's indicator mixes its benefit, the
    largest absolute surplus relative to the first level vector's, with its cheapness, the first level vector's
    point count relative to its own: max(w benefit, (1 - w) cheapness). w = 1 refines by surplus alone, w = 0 grows a
    regular grid. A candidate that would pass max_bytes is skipped. Adding a level vector leaves the other surpluses
    unchanged, so the grid after any number of additions is a prefix of the final one: returns the final grid and
    the byte size after each addition."""
    b = BASE[basis]
    grid = Grid(basis, [(b,) * d])
    grid.interpolate(f)
    top0, n0 = max(float(np.abs(grid.alpha).max()), 1e-300), grid.n_points
    score = lambda k: max(w * float(np.abs(grid.alpha[grid.offsets[k]:grid.offsets[k + 1]]).max()) / top0,
                          (1 - w) * n0 / int(grid.offsets[k + 1] - grid.offsets[k]))
    indicator = {grid.levels[0]: score(0)}
    old, active, sizes = set(), {grid.levels[0]}, [grid.n_bytes(True)]
    while active:
        top = max(active, key=indicator.get)
        active.remove(top)
        old.add(top)
        added = []
        for a in range(d):
            cand = top[:a] + (top[a] + 1,) + top[a + 1:]
            if cand in old or cand in active:
                continue
            if any(cand[j] > b and cand[:j] + (cand[j] - 1,) + cand[j + 1:] not in old for j in range(d)):
                continue
            bigger = _extended(grid, cand)
            if bigger.n_bytes(True) > max_bytes:
                continue
            grid = bigger
            added.append(len(grid.levels) - 1)
            active.add(cand)
            sizes.append(grid.n_bytes(True))
        if added:
            # the candidates of one step differ from each other in two inputs, so no one is below another:
            # their nodes go to f in one call
            nodes = [grid.nodes(k) for k in added]
            values = np.split(f(np.concatenate(nodes)), np.cumsum([len(x) for x in nodes])[:-1])
            for k, v in zip(added, values):
                grid.add_interpolated(k, v)
                indicator[grid.levels[k]] = score(k)
    return grid, sizes


def prefix(grid, k):
    """The first k level vectors of an adaptive grid, with their surpluses."""
    g = Grid(grid.basis, grid.levels[:k])
    g.alpha = grid.alpha[:g.n_points].copy()
    return g

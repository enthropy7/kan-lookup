import struct

import numpy as np
import torch

from kantab.tables import Int8MLP, LutKAN, edge_tables
from kantab.training import monomials


def x_bytes(x):
    x = np.ascontiguousarray(x, dtype="<f4")
    return struct.pack("<II", *x.shape) + x.tobytes()


def int_parts(t, bits):
    top = 2 ** (bits - 1) - 1
    scale = t.abs().flatten(1).amax(1).clamp_min(1e-12) / top
    return (t / scale.view((-1,) + (1,) * (t.dim() - 1))).round().clamp(-top, top), scale


def lut_bytes(model, lut: LutKAN, bits=16):
    blob = [struct.pack("<I", len(lut.layers))]
    for (lo, hi, _), layer in zip(lut.layers, model.layers):
        grid = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * torch.linspace(0, 1, lut.levels).unsqueeze(0)
        with torch.no_grad():
            q, scale = int_parts(edge_tables(layer, grid), bits)
        blob += [struct.pack("<III", layer.in_features, layer.out_features, lut.levels),
                 lo.numpy().astype("<f4").tobytes(), hi.numpy().astype("<f4").tobytes(),
                 scale.numpy().astype("<f4").tobytes(), q.numpy().astype("<i1" if bits == 8 else "<i2").tobytes()]
    return b"".join(blob)


def rot_bytes(model, rlut):
    w, b = model.mix.weight.detach(), model.mix.bias.detach()
    return (struct.pack("<II", model.mix.in_features, model.mix.out_features) + w.numpy().astype("<f4").tobytes()
            + b.numpy().astype("<f4").tobytes() + lut_bytes(model.kan, rlut.lut))


ACT_CODE = {"relu": 1, "silu": 2, "tanh": 3}


def q8_bytes(mq: Int8MLP):
    from kantab.tables import _ACT
    blob = [struct.pack("<I", len(mq.linears))]
    for k, lin in enumerate(mq.linears):
        with torch.no_grad():
            q, scale = int_parts(lin.weight, 8)
        act = ACT_CODE[mq.act] if k + 1 < len(mq.linears) else 0
        blob += [struct.pack("<IIIf", lin.in_features, lin.out_features, act, mq.act_scale[k]),
                 scale.numpy().astype("<f4").tobytes(), q.numpy().astype("<i1").tobytes(),
                 lin.bias.detach().numpy().astype("<f4").tobytes()]
        if act >= 2:
            p = mq.pre_scale[k]
            table = _ACT[mq.act](torch.arange(-127, 128, dtype=torch.float32) * p)
            blob += [struct.pack("<f", p), table.numpy().astype("<f4").tobytes()]
    return b"".join(blob)


def f32_bytes(mlp):
    lins = [m for m in mlp.net if isinstance(m, torch.nn.Linear)]
    code = ACT_CODE[getattr(mlp, "act", "relu")]
    blob = [struct.pack("<I", len(lins))]
    for k, lin in enumerate(lins):
        blob += [struct.pack("<III", lin.in_features, lin.out_features, code if k + 1 < len(lins) else 0),
                 lin.weight.detach().numpy().astype("<f4").tobytes(), lin.bias.detach().numpy().astype("<f4").tobytes()]
    return b"".join(blob)


def poly_bytes(d, degree, w):
    terms = monomials(d, degree)
    index = {t: i for i, t in enumerate(terms)}
    parent = [0] + [index[t[:-1]] for t in terms[1:]]
    var = [0] + [t[-1] for t in terms[1:]]
    return (struct.pack("<I", len(terms)) + np.array(parent, "<u2").tobytes() + np.array(var, "u1").tobytes()
            + np.asarray(w, "<f4").tobytes())


def _below(thr):
    t = np.float32(thr)
    return np.nextafter(t, np.float32(-np.inf)) if t > thr else t


def forest_bytes(model):
    roots, nodes = [], []
    for (pred,) in model._predictors:
        n = pred.nodes
        roots.append(len(nodes))
        order, stack = [], [0]
        while stack:
            i = stack.pop()
            order.append(i)
            if not n["is_leaf"][i]:
                stack += [int(n["right"][i]), int(n["left"][i])]
        pos = {i: k for k, i in enumerate(order)}
        for i in order:
            if n["is_leaf"][i]:
                nodes.append((np.float32(n["value"][i]), 0xFFFF, 0))
            else:
                nodes.append((_below(n["num_threshold"][i]), int(n["feature_idx"][i]), pos[int(n["right"][i])]))
    base = float(np.asarray(model._baseline_prediction).ravel()[0])
    rec = np.array(nodes, dtype=[("v", "<f4"), ("feat", "<u2"), ("right", "<u2")])
    return struct.pack("<IIf", len(roots), len(rec), base) + np.array(roots, "<i4").tobytes() + rec.tobytes()


def forest_size(n_trees, n_nodes):
    return 8 * n_nodes + 4 * n_trees + 4


def grid_bytes(values, d, n):
    s = float(np.abs(values).max()) / 32767
    q = np.round(np.asarray(values) / s).astype("<i2")
    return struct.pack("<IIf", d, n, s) + q.tobytes()


def gridv_bytes(values, counts):
    s = float(np.abs(values).max()) / 32767
    q = np.round(np.asarray(values) / s).astype("<i2")
    return struct.pack(f"<I{len(counts)}If", len(counts), *counts, s) + q.tobytes()


def cp_bytes(model):
    t = model.tables.detach()
    s = t.abs().amax(2).clamp_min(1e-12) / 32767
    q = (t / s.unsqueeze(2)).round().clamp(-32767, 32767)
    return (struct.pack("<IIIf", model.d, model.rank, model.q, float(model.b.detach())) +
            model.w.detach().numpy().astype("<f4").tobytes() + s.numpy().astype("<f4").tobytes() +
            q.numpy().astype("<i2").tobytes())


def ga2m_bytes(model):
    g, h = model.g.detach(), model.h.detach()
    gs = g.abs().amax(1).clamp_min(1e-12) / 32767
    hs = h.abs().amax((1, 2)).clamp_min(1e-12) / 32767
    gq = (g / gs.unsqueeze(1)).round().clamp(-32767, 32767)
    hq = (h / hs.view(-1, 1, 1)).round().clamp(-32767, 32767)
    return (struct.pack("<IIIf", model.d, model.q1, model.q2, float(model.b.detach())) + gs.numpy().astype("<f4").tobytes() +
            gq.numpy().astype("<i2").tobytes() + hs.numpy().astype("<f4").tobytes() + hq.numpy().astype("<i2").tobytes())

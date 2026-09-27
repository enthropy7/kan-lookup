import torch
import torch.nn.functional as F

from kantab.models import KAN, KANLinear
from kantab.models import MLP


def edge_tables(layer: KANLinear, grid: torch.Tensor) -> torch.Tensor:
    n_in, q = grid.shape
    silu = F.silu(grid)
    bases = layer.b_splines(grid.T.reshape(q, n_in)).permute(1, 0, 2)
    spline = torch.einsum("oij,iqj->oiq", layer.spline_weight * layer.spline_scaler.unsqueeze(-1), bases)
    return layer.base_weight.unsqueeze(-1) * silu.unsqueeze(0) + spline


def quantize_index(x, lo, hi, levels):
    t = (x - lo) / (hi - lo) * (levels - 1)
    return t.round().clamp(0, levels - 1).long()


def int_rows(t, bits):
    top = 2 ** (bits - 1) - 1
    scale = t.abs().flatten(1).amax(1).clamp_min(1e-12) / top
    shape = (-1,) + (1,) * (t.dim() - 1)
    return (t / scale.view(shape)).round().clamp(-top, top) * scale.view(shape)


def int8_rows(t):
    return int_rows(t, 8)


CHUNK = 32_768


def layer_ranges(model: KAN, x: torch.Tensor):
    lo, hi = [None] * len(model.layers), [None] * len(model.layers)
    with torch.no_grad():
        for s in range(0, len(x), CHUNK):
            h = x[s:s + CHUNK]
            for i, layer in enumerate(model.layers):
                mn, mx = h.amin(0), h.amax(0)
                lo[i] = mn if lo[i] is None else torch.minimum(lo[i], mn)
                hi[i] = mx if hi[i] is None else torch.maximum(hi[i], mx)
                if i + 1 < len(model.layers):
                    h = layer(h)
    return lo, hi


class LutKAN:
    def __init__(self, model: KAN, bits: int, x_train: torch.Tensor, int8_tables: bool = False,
                 table_bits: int | None = None, interp: bool = False, margin: float = 0.0,
                 data_input_range: bool = False, ranges=None):
        table_bits = table_bits or (8 if int8_tables else None)
        self.levels, self.interp = 2 ** bits, interp
        self.layers = []
        lows, highs = ranges or layer_ranges(model, x_train)
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                if i == 0 and not data_input_range:
                    lo = torch.full((layer.in_features,), -1.0)
                    hi = torch.full((layer.in_features,), 1.0)
                else:
                    lo, hi = lows[i], highs[i]
                    hi = torch.where(hi > lo, hi, lo + 1e-6)
                    lo, hi = lo - margin * (hi - lo), hi + margin * (hi - lo)
                grid = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * torch.linspace(0, 1, self.levels).unsqueeze(0)
                tables = edge_tables(layer, grid)
                self.layers.append((lo, hi, int_rows(tables, table_bits) if table_bits else tables))

    @torch.no_grad()
    def __call__(self, x):
        for lo, hi, tables in self.layers:
            n_out, n_in, _ = tables.shape
            gather = lambda q: tables.unsqueeze(0).expand(len(x), -1, -1, -1).gather(
                3, q.view(len(x), 1, n_in, 1).expand(-1, n_out, -1, 1)).squeeze(3)
            if self.interp:
                t = ((x - lo) / (hi - lo) * (self.levels - 1)).clamp(0, self.levels - 1)
                k = t.floor().clamp(max=self.levels - 2)
                f = ((t - k) * 256).round().unsqueeze(1)
                k = k.long()
                x = ((gather(k) * (256 - f) + gather(k + 1) * f) / 256).sum(2)
            else:
                x = gather(quantize_index(x, lo, hi, self.levels)).sum(2)
        return x
    def table_bytes(self, bytes_per_entry):
        return sum(t.numel() for _, _, t in self.layers) * bytes_per_entry
    def edges(self):
        return sum(t.shape[0] * t.shape[1] for _, _, t in self.layers)


_ACT = {"relu": F.relu, "silu": F.silu, "tanh": torch.tanh}


class RotLutKAN:
    def __init__(self, model, bits, x_train, ranges=None, **kw):
        self.mix = model.mix
        with torch.no_grad():
            z = model.mix(x_train) if ranges is None else None
            self.lut = LutKAN(model.kan, bits, z, data_input_range=True, ranges=ranges, **kw)
    @torch.no_grad()
    def __call__(self, x):
        return self.lut(self.mix(x))


class Int8MLP:
    def __init__(self, model: MLP, x_train: torch.Tensor):
        self.linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        self.act = getattr(model, "act", "relu")
        self.act_scale = [1.0 / 127]
        self.pre_scale = []
        pre, act = [0.0] * (len(self.linears) - 1), [0.0] * (len(self.linears) - 1)
        with torch.no_grad():
            for s in range(0, len(x_train), CHUNK):
                h = x_train[s:s + CHUNK]
                for k, lin in enumerate(self.linears[:-1]):
                    z = lin(h)
                    pre[k] = max(pre[k], z.abs().max().item())
                    h = _ACT[self.act](z)
                    act[k] = max(act[k], h.abs().max().item())
        self.pre_scale = [max(v, 1e-12) / 127 for v in pre]
        self.act_scale += [max(v, 1e-12) / 127 for v in act]
        self.weights = [int8_rows(l.weight.detach()) for l in self.linears]
    @torch.no_grad()
    def __call__(self, x):
        for k, (lin, w) in enumerate(zip(self.linears, self.weights)):
            s = self.act_scale[k]
            x = (x / s).round().clamp(-127, 127) * s
            x = F.linear(x, w, lin.bias)
            if k + 1 < len(self.linears):
                if self.act != "relu":
                    p = self.pre_scale[k]
                    x = (x / p).round().clamp(-127, 127) * p
                x = _ACT[self.act](x)
        return x
    def weight_bytes(self):
        return sum(l.weight.numel() + 4 * l.bias.numel() for l in self.linears)
    def macs(self):
        return sum(l.weight.numel() for l in self.linears)

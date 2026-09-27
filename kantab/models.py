import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    def __init__(self, sizes, act="relu"):
        super().__init__()
        self.act = act
        layers = []
        for i, (d_in, d_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i < len(sizes) - 2:
                layers.append(ACTIVATIONS[act]())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0),
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order, self.grid_range = grid_size, spline_order, tuple(grid_range)
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.register_buffer("knots", torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters(scale_noise, scale_base, scale_spline)

    @torch.no_grad()
    def reset_parameters(self, scale_noise, scale_base, scale_spline):
        # the random draws of efficient-kan in the same order, so equal seeds give equal parameters
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5) * scale_noise / self.grid_size
        points = self.knots[self.spline_order:-self.spline_order].unsqueeze(1).expand(-1, self.in_features)
        A = self.b_splines(points).transpose(0, 1)
        coeffs = torch.linalg.lstsq(A, noise.transpose(0, 1)).solution
        self.spline_weight.copy_(coeffs.permute(2, 0, 1))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * scale_spline)

    def b_splines(self, x):
        t = self.knots
        x = x.unsqueeze(-1)
        bases = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
        for p in range(1, self.spline_order + 1):
            bases = ((x - t[:-(p + 1)]) / (t[p:-1] - t[:-(p + 1)]) * bases[..., :-1]
                     + (t[p + 1:] - x) / (t[p + 1:] - t[1:-p]) * bases[..., 1:])
        return bases

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        coeffs = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = F.linear(self.b_splines(x).flatten(1), coeffs.flatten(1))
        return base + spline


class KAN(nn.Module):
    def __init__(self, sizes, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(KANLinear(a, b, **layer_kwargs) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RotKAN(nn.Module):
    # xy = ((x + y)² - (x - y)²) / 4: a learned mixing lets univariate tables represent products
    def __init__(self, d, m, hidden, **layer_kwargs):
        super().__init__()
        self.mix = nn.Linear(d, m)
        self.kan = KAN([m, *hidden, 1], grid_range=(-2.0, 2.0), **layer_kwargs)

    def forward(self, x):
        return self.kan(self.mix(x))

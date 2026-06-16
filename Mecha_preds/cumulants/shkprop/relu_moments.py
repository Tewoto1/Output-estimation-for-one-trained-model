"""relu_moments.py -- conditional ReLU-Gaussian moments (the pointwise formulas).

For ``Z | h ~ N(mu(h), Cov(h))`` (the residual-Gaussian closure used at the only
approximate step), the raw activation moments are, per coordinate / pair:

    relu_m1(mu, var)      = E[ReLU(Z)]          = sig phi(a) + mu Phi(a),  a = mu/sig
    relu_m2_diag(mu, var) = E[ReLU(Z)^2]        = (mu^2+var) Phi(a) + mu sig phi(a)
    relu_pair_matrix      = E[ReLU(Z_i)ReLU(Z_j)]  (all i,j)

The pair matrix integrates one variable in closed form (the inner ReLU mean
``relu_m1``) and the other by half-line Gauss-Legendre past the ReLU kink, where
the integrand is smooth -- so it is near-exact (validated to the MC floor incl.
correlation ~0.9), unlike Gauss-Hermite which the kink defeats. All ops are
elementwise/batched torch (GPU-friendly); mirrors the numpy oracle.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from numpy.polynomial.legendre import leggauss

SQRT2PI = math.sqrt(2.0 * math.pi)


def npdf(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x * x) / SQRT2PI


def ncdf(x: Tensor) -> Tensor:
    return torch.special.ndtr(x)


def relu_m1(mu: Tensor, var: Tensor, eps: float = 1e-12) -> Tensor:
    """E[ReLU(Z)], Z ~ N(mu, var). Elementwise."""
    sig = torch.sqrt(torch.clamp(var, min=eps))
    a = mu / sig
    return sig * npdf(a) + mu * ncdf(a)


def relu_m2_diag(mu: Tensor, var: Tensor, eps: float = 1e-12) -> Tensor:
    """E[ReLU(Z)^2], Z ~ N(mu, var). Elementwise."""
    sig = torch.sqrt(torch.clamp(var, min=eps))
    a = mu / sig
    return (mu * mu + var) * ncdf(a) + mu * sig * npdf(a)


def relu_pair_matrix(mu: Tensor, C: Tensor, n_gl: int = 64, span: float = 12.0,
                     eps: float = 1e-12) -> Tensor:
    """E[ReLU(Z_i) ReLU(Z_j)] for all i,j with Z ~ N(mu, C). Returns (d,d), symmetric.

    Elementwise 2x2 Cholesky of the pair covariances:
        s_i = sqrt(C_ii),  L10_ij = C_ij / s_i,  L11_ij^2 = C_jj - L10_ij^2,
        Z_i = mu_i + s_i u,  Z_j = mu_j + L10_ij u + L11_ij v   (u, v iid N(0,1)).
    Integrate v exactly -> E_v[ReLU(Z_j)|u] = relu_m1(mu_j + L10_ij u, L11_ij^2);
    the remaining u-integral over the smooth half-line u > u0_i = -mu_i/s_i is done
    by Gauss-Legendre. Diagonal is overwritten with the exact closed form.
    """
    device, dtype = mu.device, mu.dtype
    d = mu.shape[0]
    diagC = torch.diagonal(C)
    vi = torch.clamp(diagC, min=eps)
    si = torch.sqrt(vi)                                    # (d,) = s_i
    L10 = C / si[:, None]                                  # (d,d) uses s_i on row i
    L11sq = torch.clamp(vi[None, :] - L10 * L10, min=0.0)  # (d,d) = L11_ij^2
    u0 = -mu / si                                          # (d,) per-row ReLU kink
    g, gw = leggauss(n_gl)                                 # on [-1, 1]
    g = torch.as_tensor(g, device=device, dtype=dtype)
    gw = torch.as_tensor(gw, device=device, dtype=dtype)
    nodes = 0.5 * (g + 1.0) * span                         # half-line [0, span]
    wts = 0.5 * span * gw
    out = torch.zeros((d, d), device=device, dtype=dtype)
    mu_row = mu[:, None]
    mu_col = mu[None, :]
    for k in range(n_gl):
        u = u0[:, None] + nodes[k]                         # (d,1) u >= u0_i (smooth)
        lin = mu_row + si[:, None] * u                     # (d,1) = ReLU(Z_i) value (>=0)
        m1j = relu_m1(mu_col + L10 * u, L11sq, eps=eps)    # (d,d) inner E_v[ReLU(Z_j)]
        out = out + wts[k] * lin * npdf(u) * m1j
    out = 0.5 * (out + out.T)                              # symmetric by construction
    idx = torch.arange(d, device=device)
    out[idx, idx] = relu_m2_diag(mu, diagC, eps=eps)       # exact diagonal
    return out

"""layers.py -- the three layer operations of clippedProp.

Each operation maps a ``ClippedState`` (the structured law of the layer input) to
the structured law of the layer output.

    linear_layer            propagate (mean, cov) through W x + b, re-split, refit scalar
    mean_subtraction_layer  x <- P x : scalar channel collapses to 0; perp law unchanged
    relu_layer              condition on the scalar latent g, apply the Gaussian-ReLU
                            formulas per Gauss-Hermite node, mix, re-split, refit scalar

ReLU is the only nonlinear step. Conditioned on the all-ones latent ``g`` the input
is exactly Gaussian ``X | g ~ N(s_k u + mu_z^{(k)}, Sigma_cond)`` (one shared
``Sigma_cond = Sigma_z - c_g c_g^T / v``), so each node reduces to a Gaussian-ReLU
moment computation; we integrate over ``g`` by Gauss-Hermite and combine with the
law of total covariance. Two per-node kernels:

    relu_cov="exact"  EXACT bivariate-Gaussian ReLU covariance (Owen's T; reuses the
                      project's verified ``exact_relu_covariance_np``). scipy/CPU, O(d^2).
    relu_cov="gain"   exact ReLU mean+variance, off-diagonal via the leading-order gain
                      Sigma_ij <- Sigma_ij * Phi(alpha_i) Phi(alpha_j) (no scipy bivariate).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from scipy.special import ndtr as _ndtr

from ..cumulants.kprop.exact_relu_covariance import (
    exact_relu_covariance_np,
    relu_moments_1d_np,
)
from .scalar import clipped_cross_beta
from .state import ClippedState, NEG_INF, proj_vec, proj_mat

_TINY = 1e-14


# ---------------------------------------------------------------------------
# Linear layer:  X' = W X + b
# ---------------------------------------------------------------------------
def _apply_linear(state: ClippedState, W: Tensor, b: Tensor | None) -> tuple[Tensor, Tensor]:
    """Push the state's full moments through the affine map. Returns (mu', Sigma')."""
    mu_X, Sigma_X = state.mean_cov()
    W = W.to(mu_X.dtype)
    mu = W @ mu_X
    if b is not None:
        mu = mu + b.to(mu_X.dtype)
    Sigma = W @ Sigma_X @ W.T
    return mu, 0.5 * (Sigma + Sigma.T)


def linear_layer(state: ClippedState, W: Tensor, b: Tensor | None = None, *,
                 lo: float = NEG_INF) -> ClippedState:
    """Affine map then re-split + refit. The output scalar channel is fit with
    clamp ``lo`` (default ``-inf`` = plain Gaussian, since a pre-activation
    ``u'^T(W X + b)`` is a general real scalar)."""
    mu, Sigma = _apply_linear(state, W, b)
    return ClippedState.from_full_moments(mu, Sigma, lo=lo)


def linear_output_moments(state: ClippedState, W: Tensor,
                          b: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Final (readout) linear map: return the raw output ``(mu, Sigma)`` without
    re-splitting. clippedProp is EXACT here -- ``E[out] = W E[a] + b`` -- so no
    closure approximation is incurred at the readout."""
    return _apply_linear(state, W, b)


# ---------------------------------------------------------------------------
# Mean-subtraction layer:  X <- P X
# ---------------------------------------------------------------------------
def mean_subtraction_layer(state: ClippedState) -> ClippedState:
    """Remove the all-ones component: ``y = x - mean_j(x_j) = P x``.

    The scalar channel becomes exactly 0 (``u^T P x = 0``); ``P (s u + z) = z`` so
    the perpendicular mean/cov are unchanged and the cross-cov vanishes.
    """
    return ClippedState(
        d=state.d, m=0.0, v=0.0, lo=NEG_INF,
        mu_z=state.mu_z.clone(),
        Sigma_z=state.Sigma_z.clone(),
        c_s=torch.zeros_like(state.c_s),
    )


# ---------------------------------------------------------------------------
# ReLU layer:  Y = relu(X),  conditioning on the scalar latent g
# ---------------------------------------------------------------------------
def _gauss_hermite(m: float, v: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for ``E_{g~N(m,v)}[.]`` (weights sum to 1)."""
    t, om = np.polynomial.hermite.hermgauss(n_nodes)
    g = m + math.sqrt(2.0 * v) * t
    w = om / math.sqrt(math.pi)
    return g, w


def _relu_node_exact(a_np: np.ndarray, Sigma_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return exact_relu_covariance_np(a_np, Sigma_np)


def _relu_node_gain(a_np: np.ndarray, Sigma_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact ReLU mean+variance; off-diagonal via the leading-order gain factor."""
    var = np.clip(np.diag(Sigma_np).copy(), 0.0, None)
    mu, _second, dvar = relu_moments_1d_np(a_np, var)
    sd = np.sqrt(var)
    safe = np.where(var > _TINY, sd, 1.0)
    gain = _ndtr(a_np / safe)                      # Phi(alpha): E[relu'(Z)]
    C = Sigma_np * np.outer(gain, gain)
    np.fill_diagonal(C, dvar)
    return mu, 0.5 * (C + C.T)


def relu_layer(state: ClippedState, *, n_nodes: int = 21, relu_cov: str = "exact",
               cross_guard: bool = True) -> ClippedState:
    """Propagate ``Y = relu(X)`` and re-split into a fresh clamped state (clamp 0).

    Conditioned on the scalar latent ``g`` (integrated by ``n_nodes``-point
    Gauss-Hermite), ``X | g`` is Gaussian; each node is reduced by ``relu_cov`` and
    the nodes are combined by the law of total covariance.
    """
    if relu_cov not in ("exact", "gain"):
        raise ValueError(f"relu_cov must be 'exact' or 'gain' (got {relu_cov!r})")
    kernel = _relu_node_exact if relu_cov == "exact" else _relu_node_gain

    dtype, device = state.dtype, state.device
    u = state.u
    d = state.d
    v = float(state.v)

    # Conditional perpendicular law given g:  z | g ~ N(mu_z + (c_g/v)(g - m), Sigma_cond)
    if v > _TINY:
        beta = clipped_cross_beta(state.m, v, state.lo)
        c_g = state.c_s / beta if abs(beta) > _TINY else torch.zeros_like(state.c_s)
        if cross_guard:                              # keep conditional diagonal >= 0
            diag = torch.clamp(torch.diag(state.Sigma_z), min=0.0)
            ratio = (c_g * c_g) / (v * torch.clamp(diag, min=_TINY))
            r = float(ratio.max()) if d else 0.0
            if r > 1.0:
                c_g = c_g / math.sqrt(r * (1.0 + 1e-9))
        Sigma_cond = state.Sigma_z - torch.outer(c_g, c_g) / v
        Sigma_cond = proj_mat(0.5 * (Sigma_cond + Sigma_cond.T), u)
        g_nodes, w_nodes = _gauss_hermite(state.m, v, n_nodes)
    else:                                            # deterministic scalar -> single node
        c_g = torch.zeros_like(state.c_s)
        Sigma_cond = state.Sigma_z
        g_nodes = np.array([state.m], dtype=np.float64)
        w_nodes = np.array([1.0], dtype=np.float64)

    Sigma_cond_np = Sigma_cond.detach().cpu().double().numpy()
    u_np = u.detach().cpu().double().numpy()
    mu_z_np = state.mu_z.detach().cpu().double().numpy()
    c_g_np = c_g.detach().cpu().double().numpy()
    lo = state.lo
    m = state.m

    mu_Y = np.zeros(d, dtype=np.float64)
    M2 = np.zeros((d, d), dtype=np.float64)
    for g_k, w_k in zip(g_nodes, w_nodes):
        s_k = g_k if not math.isfinite(lo) else max(lo, g_k)
        shift = 0.0 if v <= _TINY else (g_k - m) / v
        a_k = s_k * u_np + mu_z_np + shift * c_g_np
        mu_k, C_k = kernel(a_k, Sigma_cond_np)
        mu_Y += w_k * mu_k
        M2 += w_k * (C_k + np.outer(mu_k, mu_k))

    Sigma_Y = M2 - np.outer(mu_Y, mu_Y)
    Sigma_Y = 0.5 * (Sigma_Y + Sigma_Y.T)
    mu_Y_t = torch.as_tensor(mu_Y, device=device, dtype=dtype)
    Sigma_Y_t = torch.as_tensor(Sigma_Y, device=device, dtype=dtype)
    return ClippedState.from_full_moments(mu_Y_t, Sigma_Y_t, lo=0.0)

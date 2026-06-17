"""exact_meanprop.py -- EXACT-ReLU mean propagation ("exact mean-prop").

A deterministic predictor of ``E[model(X)]`` for ``X ~ N(0, input_std^2 I)`` that
tracks, for each coordinate and each layer, a marginal Gaussian summarised by its
MEAN and VARIANCE -- and crosses every ReLU with the **exact** rectified-Gaussian
moment integral (no Hermite truncation, no gain approximation).

What "exact" means here
-----------------------
At a ReLU, given the marginal ``Z ~ N(mu, sigma^2)`` of a coordinate, the exact
post-activation moments are (``alpha = mu/sigma``, ``phi``/``Phi`` the standard
normal pdf/cdf)::

    E[ReLU(Z)]   = mu*Phi(alpha) + sigma*phi(alpha)
    E[ReLU(Z)^2] = (mu^2 + sigma^2)*Phi(alpha) + mu*sigma*phi(alpha)
    Var[ReLU(Z)] = E[ReLU(Z)^2] - E[ReLU(Z)]^2

These are the same closed forms implemented (and validated to ~1e-15) in
``kprop.exact_relu_covariance.relu_moments_1d_np``; this module reproduces them
self-contained (numpy + an optional scipy ``ndtr``, falling back to ``math.erf``)
so it imports with no torch/scipy hard dependency and runs in float64.

How it differs from the default k=1 "mean-prop"
-----------------------------------------------
The harmonic k_max=1 path collapses the degree-2 piece to a FIXED metric
``diag(W W^T)`` -- i.e. it effectively assumes UNIT-variance input at every layer
and does not carry the actual post-ReLU variance forward. Exact mean-prop instead
*propagates* the variance: at a linear layer it maps ``(mu, v)`` by

    mu  <- W mu + b
    v   <- (W .* W) v            # diagonal of W diag(v) W^T (mean-field: drops cross-cov)

then applies the exact ReLU integral and STORES the new ``(mu, v)`` for the next
layer. The ONLY approximation left is the mean-field/diagonal assumption at the
linear mixing (cross-covariances between coordinates are dropped); the ReLU
crossing itself is exact. Consequences:
  * depth 1 (one ReLU, linear readout): the output MEAN is EXACT (no cross-cov is
    ever needed for the mean), so it matches Monte-Carlo to sampling noise;
  * depth >= 2: the output mean is approximate only through the dropped cross-cov
    in the propagated variances -- the residual is the price of the diagonal closure,
    which is exactly what the trained / weight-shifted tests below quantify.

Usage
-----
    from Mecha_preds.cumulants import run_exact_meanprop, estimate_empirical_mean, compare_means
    pred = run_exact_meanprop(model)["mean"]            # (output_dim,)
    mc, st = estimate_empirical_mean(model=model, input_dim=model.cfg.input_dim)
    print(compare_means(pred, mc, st)["relative_error_mean"])
"""
from __future__ import annotations
from math import erf, sqrt
from typing import Optional
import numpy as np

_SQRT2 = sqrt(2.0)
_INV_SQRT_2PI = 1.0 / sqrt(2.0 * np.pi)

try:                                            # exact, vectorised standard-normal CDF
    from scipy.special import ndtr as _Phi
except Exception:                               # dependency-light fallback (no scipy)
    _erf_vec = np.vectorize(erf)

    def _Phi(x):                                # 0.5*(1+erf(x/sqrt2)) == Phi(x)
        return 0.5 * (1.0 + _erf_vec(np.asarray(x, dtype=np.float64) / _SQRT2))


def _phi(x):
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) * _INV_SQRT_2PI


# --------------------------------------------------------------------------- #
def relu_gaussian_moments(mu, var, *, var_eps: float = 1e-12):
    """Exact (mean, second moment, variance) of ``ReLU(Z)`` for ``Z ~ N(mu, var)``,
    elementwise. Coordinates with ``var <= var_eps`` collapse to the point mass
    ``ReLU(mu)`` (mean = max(mu,0), variance = 0). Reproduces, elementwise, the
    ``relu_moments_1d_np`` closed form."""
    mu = np.asarray(mu, dtype=np.float64)
    var = np.clip(np.asarray(var, dtype=np.float64), 0.0, None)
    det = var <= var_eps
    sigma = np.sqrt(np.where(det, 1.0, var))          # safe; det entries overwritten below
    alpha = mu / sigma
    Phi = _Phi(alpha)
    phi = _phi(alpha)
    mean_stoch = mu * Phi + sigma * phi
    second_stoch = (mu * mu + var) * Phi + mu * sigma * phi
    mean_det = np.maximum(mu, 0.0)
    mean = np.where(det, mean_det, mean_stoch)
    second = np.where(det, mean_det * mean_det, second_stoch)
    var_out = np.where(det, 0.0, np.clip(second - mean * mean, 0.0, None))
    return mean, second, var_out


def _weight_bias(linear):
    """Read an nn.Linear's (W, b) as float64 numpy (W shape (out, in); b or None)."""
    W = linear.weight.detach().cpu().double().numpy()
    b = None if linear.bias is None else linear.bias.detach().cpu().double().numpy()
    return W, b


def run_exact_meanprop(model, input_dim: Optional[int] = None, *, input_std: float = 1.0,
                       return_layers: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, input_std^2 I)`` by exact-ReLU mean propagation.

    Tracks per-coordinate (mean, variance), propagates the variance through each
    linear layer as the diagonal ``(W .* W) v`` (mean-field), and crosses each ReLU
    with the exact rectified-Gaussian integral. ReLU is applied to every hidden
    block; the readout is linear (matching ``model.MLP.forward``).

    Returns ``{"mean": (output_dim,), "out_var": (output_dim,), [layer_means, layer_vars]}``.
    """
    cfg = model.cfg
    if cfg.activation != "relu":
        raise ValueError(f"exact mean-prop currently supports ReLU only; got activation "
                         f"{cfg.activation!r}. (The exact integral is the rectified-Gaussian one.)")
    if input_dim is None:
        input_dim = cfg.input_dim

    hidden = list(model.hidden_layers)
    layers = hidden + [model.readout]                 # depth hidden blocks + linear readout

    m = np.zeros(input_dim, dtype=np.float64)
    v = np.full(input_dim, float(input_std) ** 2, dtype=np.float64)
    layer_means, layer_vars = [], []
    for li, lin in enumerate(layers):
        W, b = _weight_bias(lin)
        if W.shape[1] != m.shape[0]:
            raise ValueError(f"layer {li} expects in_features={W.shape[1]} but state has {m.shape[0]}")
        m = W @ m + (b if b is not None else 0.0)     # mean: linear map (+ bias)
        v = (W * W) @ v                                # variance: diagonal of W diag(v) Wᵀ
        if li < len(layers) - 1:                       # ReLU on hidden blocks; readout is linear
            m, _, v = relu_gaussian_moments(m, v)      # EXACT rectified-Gaussian moments
            if return_layers:
                layer_means.append(m.copy()); layer_vars.append(v.copy())

    res = {"mean": np.asarray(m, dtype=np.float64).reshape(-1),
           "out_var": np.asarray(v, dtype=np.float64).reshape(-1)}
    if return_layers:
        res["layer_means"] = layer_means
        res["layer_vars"] = layer_vars
    return res

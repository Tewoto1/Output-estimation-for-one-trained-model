"""scalar.py -- the clamped ("rectified") Gaussian scalar law and its moment maps.

The clippedProp state models the all-ones component ``s = u^T x`` of a hidden
vector as a CLAMPED Gaussian::

    s = max(lo, g),     g ~ N(m, v)            (lo = 0 by default; lo = -inf => plain Gaussian)

This is the natural law for the all-ones channel of a ReLU network: post-ReLU
activations are non-negative, so their average ``s = (1/sqrt(d)) sum_i a_i`` is
non-negative and piles up against 0 -- a plain Gaussian fits it badly, a
rectified Gaussian fits it with the right point mass ``p0 = P(g <= lo)`` plus a
truncated tail. A pre-activation (post-linear) channel can be negative, so there
we use ``lo = -inf`` (the rectified family contains the plain Gaussian as the
``lo -> -inf`` limit, so the state representation is uniform).

This module provides the three maps the propagation needs:

    rect_gauss_moments(m, v, lo)  -- forward:  (m, v)        -> (E[s], Var[s], p0)
    fit_rect_gauss(mean, var, lo) -- inverse:  (E[s], Var[s]) -> (m, v)     (the "refit")
    clipped_cross_beta(m, v, lo)  -- beta = Cov(g, s) / v  (converts Cov(z,s) <-> Cov(z,g))

All quantities are plain Python floats (the scalar channel is one-dimensional);
the perpendicular Gaussian is carried as tensors in ``state.py``.
"""
from __future__ import annotations

import math

import numpy as np

try:  # scipy gives the accurate standard-normal CDF used everywhere here.
    from scipy.special import ndtr as _ndtr
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "clippedProp.scalar requires scipy (scipy.special.ndtr for the normal CDF)."
    ) from exc

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_TINY = 1e-300


def _phi(x: float) -> float:
    """Standard normal PDF (scalar)."""
    return math.exp(-0.5 * x * x) * _INV_SQRT_2PI


def _Phi(x: float) -> float:
    """Standard normal CDF (scalar)."""
    return float(_ndtr(x))


# ---------------------------------------------------------------------------
# Forward moment map:  (m, v, lo) -> (E[s], Var[s], p0)
# ---------------------------------------------------------------------------
def rect_gauss_moments(m: float, v: float, lo: float = 0.0) -> tuple[float, float, float]:
    """Mean, variance and clamp mass of ``s = max(lo, g)``, ``g ~ N(m, v)``.

    Returns ``(E[s], Var[s], p0)`` with ``p0 = P(g <= lo)``. For ``lo = -inf``
    (plain Gaussian) this is just ``(m, v, 0.0)``.
    """
    m = float(m)
    v = max(float(v), 0.0)
    if not math.isfinite(lo):                 # plain Gaussian (no clamp)
        return m, v, 0.0
    if v <= _TINY:                            # point mass at m -> s = max(lo, m)
        s = max(lo, m)
        return s, 0.0, (1.0 if m <= lo else 0.0)
    sd = math.sqrt(v)
    a = (m - lo) / sd                         # alpha
    Phi = _Phi(a)
    phi = _phi(a)
    e_max = (m - lo) * Phi + sd * phi         # E[max(0, g - lo)]
    e2_max = ((m - lo) ** 2 + v) * Phi + (m - lo) * sd * phi
    mean = lo + e_max
    var = max(e2_max - e_max * e_max, 0.0)
    p0 = _Phi(-a)
    return mean, var, p0


# ---------------------------------------------------------------------------
# Inverse moment map (the "refit"):  (E[s], Var[s], lo) -> (m, v)
# ---------------------------------------------------------------------------
def _ratio_of_alpha(a: float) -> float:
    """``Var[s]/E[s]^2`` for ``s = max(0, g)``, ``g = a*sd + sd*Z`` -- a function of
    ``alpha = a`` only (the scale ``sd`` cancels). Strictly decreasing in ``a``;
    ``-> +inf`` as ``a -> -inf`` (mostly clamped), ``-> 0`` as ``a -> +inf`` (Gaussian).
    """
    Phi = _Phi(a)
    phi = _phi(a)
    den = a * Phi + phi                        # = E[s]/sd  > 0
    if den <= _TINY:
        return math.inf
    num = (a * a + 1.0) * Phi + a * phi         # = E[s^2]/sd^2
    return num / (den * den) - 1.0


def fit_rect_gauss(
    mean_s: float, var_s: float, lo: float = 0.0, *,
    a_lo: float = -6.0, a_hi: float = 8.0, iters: int = 120,
) -> tuple[float, float]:
    """Invert the rectified-Gaussian moments: choose ``(m, v)`` so that
    ``max(lo, N(m, v))`` has mean ``mean_s`` and variance ``var_s``.

    Strategy: shift to ``t = g - lo`` so ``s - lo = max(0, t)``; the ratio
    ``R = Var/mean^2`` of a half-clamped Gaussian depends on ``alpha = (m-lo)/sd``
    ALONE and is monotone decreasing, so we bisect ``alpha`` on ``R`` over a
    well-conditioned bracket then recover ``sd`` from the mean. Outside the bracket
    the two limits are handled in closed form:

      * ``alpha >= a_hi`` (``R`` small): the clamp is negligible (``p0 ~ 1e-15``),
        so the rectified law IS a plain Gaussian -> return ``(mean_s, var_s)``;
      * ``alpha <= a_lo`` (``R`` large): the law is a near-point-mass at ``lo``
        (vanishing mean), saturated at ``alpha = a_lo``.

    ``lo = -inf`` returns the plain Gaussian ``(mean_s, var_s)`` directly.
    """
    mean_s = float(mean_s)
    var_s = max(float(var_s), 0.0)
    if not math.isfinite(lo):                  # plain Gaussian fit (no clamp)
        return mean_s, var_s

    mu_t = mean_s - lo                         # E[max(0, t)] >= 0
    if mu_t <= 1e-15:                          # essentially a point mass at the clamp
        v = max(var_s, 1e-30)
        return lo - 8.0 * math.sqrt(v), v      # alpha very negative => p0 ~ 1
    if var_s <= 1e-30:                         # zero-variance positive value -> ~Gaussian
        return lo + mu_t, 1e-30

    R = var_s / (mu_t * mu_t)
    r_lo = _ratio_of_alpha(a_hi)               # smallest R in bracket (alpha = a_hi)
    r_hi = _ratio_of_alpha(a_lo)               # largest  R in bracket (alpha = a_lo)
    if R <= r_lo:                              # nearly unclamped -> plain Gaussian
        return mean_s, var_s
    if R >= r_hi:                              # deeply clamped (rare) -> saturate
        a = a_lo
    else:
        lo_a, hi_a = a_lo, a_hi
        for _ in range(iters):                 # R decreasing => larger alpha lowers R
            mid = 0.5 * (lo_a + hi_a)
            if _ratio_of_alpha(mid) > R:
                lo_a = mid
            else:
                hi_a = mid
        a = 0.5 * (lo_a + hi_a)

    Phi = _Phi(a)
    phi = _phi(a)
    sd = mu_t / (a * Phi + phi)                # E[max(0,t)] = sd*(a*Phi + phi)
    v = sd * sd
    m = lo + a * sd
    return m, v


# ---------------------------------------------------------------------------
# Cross-covariance conversion factor:  beta = Cov(g, s) / v
# ---------------------------------------------------------------------------
def clipped_cross_beta(m: float, v: float, lo: float = 0.0, *, n_gh: int = 33) -> float:
    """``beta = Cov(g, s) / Var[g]`` for ``s = max(lo, g)``, ``g ~ N(m, v)``.

    For a jointly Gaussian latent ``(g, z)`` with ``Cov(z, g) = c_g`` we have
    ``Cov(z, s) = beta * c_g`` (regression of ``s`` on ``g``), so ``beta`` converts
    between the stored clamped cross-cov ``c_s = Cov(z, s)`` and the conditioning
    cross-cov ``c_g = Cov(z, g) = c_s / beta`` used in the ReLU step.

    Closed form for the two clamps the propagation actually uses; a 1-D
    Gauss-Hermite fallback for any other ``lo``.
        lo = -inf : s = g            => beta = 1
        lo =  0   : s = relu(g)      => beta = (E[s^2] - m E[s]) / v
    """
    v = float(v)
    if not math.isfinite(lo):
        return 1.0
    if v <= _TINY:
        return 0.0
    if lo == 0.0:
        mean, var, _ = rect_gauss_moments(m, v, 0.0)
        e2 = var + mean * mean                  # E[g*s] = E[s^2] for lo = 0
        return (e2 - m * mean) / v
    t, w = np.polynomial.hermite.hermgauss(n_gh)
    g = m + math.sqrt(2.0 * v) * t
    wn = w / math.sqrt(math.pi)
    s = np.maximum(lo, g)
    e_gs = float(np.sum(wn * g * s))
    e_s = float(np.sum(wn * s))
    return (e_gs - m * e_s) / v

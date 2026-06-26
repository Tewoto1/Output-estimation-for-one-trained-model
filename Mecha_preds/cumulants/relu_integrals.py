"""relu_integrals.py -- THE canonical Gaussian-ReLU integrals (numpy + scipy, torch-free).

Single source of truth for the closed-form rectified-Gaussian moments used by every
torch-free predictor in this package (``swkprop``, ``spikekprop``, ``exact_meanprop``,
``..binned_kprop``). Two closed forms (paper eqs 22-24):

    rank 1 (one neuron, ``Y ~ N(mu, var)``):                         -- ``relu_moments_1d``
        E[ReLU(Y)]   = mu*Phi(a) + sigma*phi(a)              a = mu/sigma
        E[ReLU(Y)^2] = (mu^2 + var)*Phi(a) + mu*sigma*phi(a)

    rank 2 (a pair ``(Z_i, Z_j) ~ N(mu, Sigma)``):                   -- ``exact_relu_covariance``
        E[ReLU(Z_i) ReLU(Z_j)]  via the exact bivariate-normal moments
        (Owen's T for Phi_2), then Cov = E[..] - E[ReLU_i]E[ReLU_j].

Relationship to the vendored kprop. These are the SAME formulas as the numpy core of
``kprop/exact_relu_covariance.py`` (``relu_moments_1d_np`` / ``exact_relu_covariance_np``,
validated to ~1e-15 vs quadrature + 20M-sample MC). That vendored module is kept as its own
copy ON PURPOSE: it is third-party/pinned and it ``import torch`` at module load (it ships the
torch wrappers + the ``exact_relu_covariance_kprop`` HTower bridge used by the harmonic-kprop
path). Importing it would therefore pull torch into the otherwise torch-free predictor cores,
so this module is the deliberate torch-free twin -- the one allowed duplication, across the
torch / no-torch boundary. Everything torch-free imports the integrals from HERE.

``swkprop/relu.py`` re-exports this module for backward compatibility.
"""
from __future__ import annotations

import math

import numpy as np

from scipy.special import ndtr as _ndtr        # standard normal CDF (vectorized, exact)
from scipy.special import owens_t as _owens_t   # Owen's T -> exact bivariate normal CDF

_DEFAULT_VAR_EPS = 1e-12
_NEG_RTOL = 1e-8
_RHO_TOL = 1e-7
_RHO_VALID_TOL = 1e-6
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)


def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _Phi(x: np.ndarray) -> np.ndarray:
    return _ndtr(x)


def _bvn_pdf(h: np.ndarray, k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Standard bivariate normal PDF phi2(h, k; rho), |rho| < 1."""
    s2 = 1.0 - rho * rho
    return np.exp(-(h * h - 2.0 * rho * h * k + k * k) / (2.0 * s2)) / (2.0 * math.pi * np.sqrt(s2))


def bvn_cdf(h: np.ndarray, k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Standard bivariate normal CDF P(X<=h, Y<=k) with correlation ``rho``, via Owen's T.

    Exact, vectorized, validated to ~1e-16 against a correlation-integral reference.
    Expects ``|rho| <= 1``; the singular |rho|==1 case is handled by the 1-D limit
    in the covariance routine, not here.
    """
    h = np.asarray(h, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    h, k, rho = np.broadcast_arrays(h, k, rho)
    s = np.sqrt(np.clip(1.0 - rho * rho, 0.0, None))

    def _a(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        safe_den = np.where(den != 0.0, den, 1.0)
        return np.where(den != 0.0, num / safe_den, np.where(num >= 0.0, np.inf, -np.inf))

    Th = _owens_t(h, _a(k - rho * h, h * s))
    Tk = _owens_t(k, _a(h - rho * k, k * s))
    hk = h * k
    c = np.where((hk > 0.0) | ((hk == 0.0) & (h + k >= 0.0)), 0.0, 0.5)
    P = 0.5 * _Phi(h) + 0.5 * _Phi(k) - Th - Tk - c

    both_zero = (h == 0.0) & (k == 0.0)
    if both_zero.any():
        P = np.where(both_zero, 0.25 + np.arcsin(np.clip(rho, -1.0, 1.0)) * _INV_2PI, P)
    return np.clip(P, 0.0, 1.0)


def relu_moments_1d(mu: np.ndarray, var: np.ndarray, *, var_eps: float = _DEFAULT_VAR_EPS,
                    neg_rtol: float = _NEG_RTOL):
    """Exact univariate Gaussian-ReLU moments ``(mean, second, variance)`` (paper eqs 22-23).

    Deterministic coordinates (``var <= var_eps``) collapse to the point mass ``mu``:
    ``mean = max(mu, 0)``, ``second = mean^2``, ``variance = 0``.
    """
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)

    vmin = float(var.min()) if var.size else 0.0
    if vmin < 0.0:
        scale = max(1.0, float(np.abs(var).max()))
        if vmin < -neg_rtol * scale:
            raise ValueError(f"relu_moments_1d got a meaningfully negative variance: min={vmin:.3e}")
    var = np.clip(var, 0.0, None)

    det = var <= var_eps
    sigma = np.sqrt(var)
    safe_sigma = np.where(det, 1.0, sigma)
    alpha = mu / safe_sigma

    Phi = _Phi(alpha)
    phi = _phi(alpha)
    mean_stoch = mu * Phi + sigma * phi
    second_stoch = (mu * mu + var) * Phi + mu * sigma * phi

    mean_det = np.maximum(mu, 0.0)
    mean = np.where(det, mean_det, mean_stoch)
    second = np.where(det, mean_det * mean_det, second_stoch)
    variance = np.where(det, 0.0, np.clip(second - mean * mean, 0.0, None))
    return mean, second, variance


def _relu_cross_moment_perfect_corr(mu, sigma, alpha, sign):
    """E[ReLU(Z_i)ReLU(Z_j)] in the perfectly (anti)correlated limit rho = sign."""
    MUi, MUj = mu[:, None], mu[None, :]
    SI, SJ = sigma[:, None], sigma[None, :]
    Ai, Aj = alpha[:, None], alpha[None, :]
    if sign > 0:
        t = -np.minimum(Ai, Aj)
        I0 = _Phi(-t)
        I1 = _phi(t)
        I2 = t * _phi(t) + _Phi(-t)
        return SI * SJ * I2 + (SI * MUj + SJ * MUi) * I1 + MUi * MUj * I0
    lo, hi = -Ai, Aj
    valid = hi > lo
    Phi_hi, Phi_lo = _Phi(hi), _Phi(lo)
    phi_hi, phi_lo = _phi(hi), _phi(lo)
    J0 = Phi_hi - Phi_lo
    J1 = phi_lo - phi_hi
    J2 = (lo * phi_lo - hi * phi_hi) + (Phi_hi - Phi_lo)
    out = -SI * SJ * J2 + (SI * MUj - SJ * MUi) * J1 + MUi * MUj * J0
    return np.where(valid, out, 0.0)


def exact_relu_covariance(mu, Sigma, *, var_eps: float = _DEFAULT_VAR_EPS, neg_rtol: float = _NEG_RTOL,
                          rho_tol: float = _RHO_TOL, rho_valid_tol: float = _RHO_VALID_TOL):
    """Exact rank-2 ReLU propagation of ``(mu, Sigma)`` (paper eq 24); numpy/scipy core.

    Returns ``(new_mu, new_Sigma)`` with ``new_mu_i = E[ReLU(Z_i)]`` and
    ``new_Sigma_ij = Cov(ReLU(Z_i), ReLU(Z_j))`` under ``Z ~ N(mu, Sigma)``, using the
    exact bivariate-Gaussian formula (no gain approximation). Diagonal is overwritten
    with the exact univariate ReLU variances.
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma must be ({n},{n}); got {Sigma.shape}")
    Sigma = 0.5 * (Sigma + Sigma.T)

    var = np.diag(Sigma).copy()
    new_mu, _second, diag_var = relu_moments_1d(mu, var, var_eps=var_eps, neg_rtol=neg_rtol)
    det = var <= var_eps

    sigma = np.sqrt(np.clip(var, 0.0, None))
    safe_sigma = np.where(det, 1.0, sigma)
    alpha = mu / safe_sigma

    denom = np.outer(safe_sigma, safe_sigma)
    rho = Sigma / denom
    pair_stoch = (~det)[:, None] & (~det)[None, :]
    rho = np.where(pair_stoch, np.clip(rho, -1.0, 1.0), 0.0)

    rho_g = np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12)
    s = np.sqrt(1.0 - rho_g * rho_g)
    Ai, Aj = alpha[:, None], alpha[None, :]
    SI, SJ = sigma[:, None], sigma[None, :]
    MUi, MUj = mu[:, None], mu[None, :]

    A = _phi(Ai) * _Phi((Aj - rho_g * Ai) / s)
    B = _phi(Aj) * _Phi((Ai - rho_g * Aj) / s)
    P = bvn_cdf(Ai, Aj, rho_g)
    D = _bvn_pdf(Ai, Aj, rho_g)
    M_i = A + rho_g * B
    M_j = B + rho_g * A
    M_ij = rho_g * P + (1.0 - rho_g * rho_g) * D - rho_g * Ai * A - rho_g * Aj * B
    Ecross = MUi * MUj * P + MUi * SJ * M_j + MUj * SI * M_i + SI * SJ * M_ij

    near_pos = rho >= 1.0 - rho_tol
    near_neg = rho <= -1.0 + rho_tol
    if near_pos.any():
        Ecross = np.where(near_pos, _relu_cross_moment_perfect_corr(mu, sigma, alpha, +1.0), Ecross)
    if near_neg.any():
        Ecross = np.where(near_neg, _relu_cross_moment_perfect_corr(mu, sigma, alpha, -1.0), Ecross)

    new_Sigma = Ecross - np.outer(new_mu, new_mu)
    det_involved = det[:, None] | det[None, :]
    new_Sigma = np.where(det_involved, 0.0, new_Sigma)
    new_Sigma = 0.5 * (new_Sigma + new_Sigma.T)
    np.fill_diagonal(new_Sigma, diag_var)
    return new_mu, new_Sigma

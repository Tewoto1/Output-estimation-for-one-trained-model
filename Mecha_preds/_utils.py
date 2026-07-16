"""_utils.py -- shared torch-free numerical kernel for the Mecha_preds predictors.

Single source of truth (numpy + scipy, NO torch) for the two families of helpers every
torch-free predictor in this package reuses -- the binned K=2 core (``..binned_kprop``)
and the cumulant predictors (``swkprop``, ``spikekprop``, ``exact_meanprop``):

  * Gaussian-ReLU integrals -- the closed-form rectified-Gaussian moments (paper eqs 22-24):
        rank 1 (``Y ~ N(mu, var)``):                                  -- ``relu_moments_1d``
            E[ReLU(Y)]   = mu*Phi(a) + sigma*phi(a)              a = mu/sigma
            E[ReLU(Y)^2] = (mu^2 + var)*Phi(a) + mu*sigma*phi(a)
        rank 2 (a pair ``(Z_i, Z_j) ~ N(mu, Sigma)``):               -- ``exact_relu_covariance``
            E[ReLU(Z_i) ReLU(Z_j)]  via the exact bivariate-normal moments
            (Owen's T for Phi_2), then Cov = E[..] - E[ReLU_i]E[ReLU_j].

  * Classic matrix utilities -- ``symmetrize`` and ``project_to_psd`` (eigenvalue clip).

This module lives at the TOP of ``Mecha_preds`` (not inside one predictor sub-package, and not
inside ``cumulants`` whose ``__init__`` eagerly imports torch) precisely so the torch-free cores
can import it with only numpy + scipy installed -- ``Mecha_preds/__init__.py`` is torch-free, so
``from Mecha_preds._utils import ...`` pulls in no torch. That removes the need for the old
per-package import shims (``binned_kprop/_relu.py`` and ``swkprop/relu.py``).

Relationship to the vendored kprop. The ReLU integrals are the SAME formulas as the numpy core
of ``cumulants/kprop/exact_relu_covariance.py`` (``relu_moments_1d_np`` / ``exact_relu_covariance_np``,
validated to ~1e-15 vs quadrature + 20M-sample MC). That vendored module is kept as its own copy ON
PURPOSE: it is third-party/pinned and ``import torch`` at module load (it ships the torch wrappers +
the HTower bridge for the harmonic-kprop path). This module is the deliberate torch-free twin -- the
one allowed duplication, across the torch / no-torch boundary. Everything torch-free imports from HERE.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from scipy.special import ndtr as _ndtr        # standard normal CDF (vectorized, exact)
from scipy.special import owens_t as _owens_t   # Owen's T -> exact bivariate normal CDF

_DEFAULT_VAR_EPS = 1e-12
_NEG_RTOL = 1e-8
_RHO_TOL = 1e-7
_RHO_VALID_TOL = 1e-6
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Gaussian-ReLU integrals
# --------------------------------------------------------------------------- #
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


def _perfect_corr_entries(mu, sigma, alpha, sign, I, J):
    """``_relu_cross_moment_perfect_corr`` evaluated ONLY at the index pairs
    ``(I[k], J[k])`` -- identical closed forms, O(#pairs) instead of O(d^2).
    The near-singular set is normally just the diagonal plus a handful of
    (anti)parallel coordinate pairs, so this is the cheap path the covariance
    routine uses; the full-matrix variant below is kept as the reference."""
    MUi, MUj = mu[I], mu[J]
    SI, SJ = sigma[I], sigma[J]
    Ai, Aj = alpha[I], alpha[J]
    if sign > 0:
        t = -np.minimum(Ai, Aj)
        I0 = _Phi(-t)
        I1 = _phi(t)
        I2 = t * I1 + I0
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


def _relu_cross_moment_perfect_corr(mu, sigma, alpha, sign):
    """E[ReLU(Z_i)ReLU(Z_j)] in the perfectly (anti)correlated limit rho = sign
    (full-matrix reference; see ``_perfect_corr_entries`` for the masked fast path)."""
    d = mu.shape[0]
    I, J = np.indices((d, d))
    return _perfect_corr_entries(mu, sigma, alpha, sign, I.ravel(), J.ravel()).reshape(d, d)


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

    # Perfect-|rho| limits are needed only AT the near-singular entries. The diagonal
    # (rho == 1) trips this on every call, and the old full-matrix evaluation
    # recomputed all d^2 entries only to keep a handful -- evaluate the identical
    # closed forms at the masked index pairs instead (~25-35% of the whole routine
    # saved; the diagonal is overwritten with the exact univariate variances below).
    for mask, sign in ((rho >= 1.0 - rho_tol, +1.0), (rho <= -1.0 + rho_tol, -1.0)):
        if mask.any():
            Ipc, Jpc = np.nonzero(mask)
            Ecross[Ipc, Jpc] = _perfect_corr_entries(mu, sigma, alpha, sign, Ipc, Jpc)

    new_Sigma = Ecross - np.outer(new_mu, new_mu)
    det_involved = det[:, None] | det[None, :]
    new_Sigma = np.where(det_involved, 0.0, new_Sigma)
    new_Sigma = 0.5 * (new_Sigma + new_Sigma.T)
    np.fill_diagonal(new_Sigma, diag_var)
    return new_mu, new_Sigma


def exact_relu_covariance_pairs(mu_i, mu_j, sig_i, sig_j, rho):
    """Exact ``Cov(ReLU(Z_i), ReLU(Z_j))`` for SELECTED pairs, vectorized over any
    broadcastable shapes (same closed form as :func:`exact_relu_covariance`, eq 24,
    without materializing an ``n x n`` matrix -- use when you need a sparse subset
    of pairs, e.g. sampled off-diagonal entries across a grid of conditioning values).

    Stochastic, non-degenerate entries only: callers must guarantee ``sig > 0`` and
    ``|rho|`` bounded away from 1 (bulk pairs have ``|rho| = O(1/sqrt(n))``); there is
    no perfect-correlation / point-mass handling here. Validated against
    ``exact_relu_covariance`` to ~1e-16 (see experiments/affine_conditional_layer1).
    """
    ai, aj = mu_i / sig_i, mu_j / sig_j
    r = np.clip(rho, -1 + 1e-12, 1 - 1e-12)
    s = np.sqrt(1.0 - r * r)
    A = _phi(ai) * _Phi((aj - r * ai) / s)
    B = _phi(aj) * _Phi((ai - r * aj) / s)
    P = bvn_cdf(ai, aj, r)
    D = _bvn_pdf(ai, aj, r)
    M_i = A + r * B
    M_j = B + r * A
    M_ij = r * P + (1.0 - r * r) * D - r * ai * A - r * aj * B
    Ecross = (mu_i * mu_j * P + mu_i * sig_j * M_j + mu_j * sig_i * M_i
              + sig_i * sig_j * M_ij)
    m_i = mu_i * _Phi(ai) + sig_i * _phi(ai)
    m_j = mu_j * _Phi(aj) + sig_j * _phi(aj)
    return Ecross - m_i * m_j


# --------------------------------------------------------------------------- #
# classic matrix utilities (symmetrize / PSD projection by eigenvalue clip)
# --------------------------------------------------------------------------- #
def symmetrize(A: np.ndarray) -> np.ndarray:
    """Symmetric part ``0.5 (A + A^T)`` -- kills the antisymmetric roundoff in a covariance."""
    return 0.5 * (A + A.T)


def project_to_psd(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """Clip negative eigenvalues to 0. Returns ``(A_psd, clipped_mass)`` where
    ``clipped_mass`` is the total magnitude of the removed negative eigenvalues
    (0 if already PSD) -- log it; it should be numerical roundoff only."""
    A = symmetrize(A)
    vals, vecs = np.linalg.eigh(A)
    vmin = float(vals.min()) if vals.size else 0.0
    if vmin >= 0.0:
        return A, 0.0
    clipped = float(-vals[vals < 0.0].sum())
    A = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
    return symmetrize(A), clipped


__all__ = [
    # Gaussian-ReLU integrals
    "_phi", "_Phi", "_bvn_pdf", "bvn_cdf", "relu_moments_1d",
    "_relu_cross_moment_perfect_corr", "_perfect_corr_entries", "exact_relu_covariance",
    "_DEFAULT_VAR_EPS", "_NEG_RTOL", "_RHO_TOL", "_RHO_VALID_TOL",
    # matrix utilities
    "symmetrize", "project_to_psd",
]

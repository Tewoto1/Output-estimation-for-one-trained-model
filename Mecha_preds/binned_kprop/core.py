"""core.py -- Coordinate-spike BINNED kprop, K=2 (numpy/scipy, torch-free).

Mechanistic predictor for a ReLU MLP whose hidden matrices carry a *coordinate*
spike on a single axis ``e = e_1`` (coordinate 0 in Python):

    M = W + e e^T,      W_{ij} ~ N(0, 1/n).

For a FLAT spike (``1/sqrt(n) 1 1^T``) contractions against the spike direction get
a flat-loop discount, so ordinary total-order kprop is already accurate (this is the
``..cumulants.spikekprop`` "ones" case). A COORDINATE spike has no such discount: the
linear step preserves a large residue of coordinate 0,

    A^+ = gamma A + r^T B,   B^+ = u A + V B,   (X = A e + B,  B in e^perp)

so the cumulants involving coordinate 0 are O(1) at every order and must NOT be
propagated as ordinary bulk cumulant-tensor entries. This module instead represents
the spike coordinate ``A`` EXPLICITLY by a discrete distribution over ``num_bins``
bins (a hidden-Markov-model-style scalar transition kernel), and propagates the
conditional law of the bulk ``B | A in bin`` by ordinary K=2 cumulant propagation
inside each bin.

THIS IS THE K = 2 IMPLEMENTATION.  Each spike bin stores a conditional bulk mean
vector and covariance matrix (``BinnedK2State``). ``num_bins`` is the adjustable
hyperparameter (the number of spike bins kept). The general ``K > 2`` extension --
conditional cumulant *tensors* per bin, with the per-bin ReLU delegated to the
ordinary harmonic kprop -- is provided as a hook in ``.kprop_hook`` (it imports and
calls ``Mecha_preds.cumulants.kprop``); see ``BinnedKState`` there.

State (per bin ``alpha``, K=2)::

    p[alpha]        P(A in bin alpha)                  (num_bins,)
    a[alpha]        representative spike value E[A|bin] (num_bins,)
    mu[alpha]       E[B | A in bin alpha]              (num_bins, d)
    Sigma[alpha]    Cov[B | A in bin alpha]            (num_bins, d, d)

with bulk dimension ``d = n - 1``.  Coordinate 0 is NEVER stored inside the bulk
mean/covariance arrays -- it lives only in ``p`` and ``a``.

The ReLU integrals reuse the repo's canonical numpy/scipy kernel
(``..cumulants.relu_integrals``); see ``_relu.py`` for the import shim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ._relu import _phi, _Phi, exact_relu_covariance, relu_moments_1d

# Coordinate of the spike (e = e_1 -> index 0 in Python). The bulk is coords 1..n-1.
SPIKE_COORD = 0

_TINY = 1e-30
_VAR_FLOOR = 1e-12
_DEFAULT_MIN_PROB = 1e-15


# --------------------------------------------------------------------------- #
# standard-normal helpers
# --------------------------------------------------------------------------- #
def normal_pdf(z: np.ndarray) -> np.ndarray:
    return _phi(np.asarray(z, dtype=np.float64))


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return _Phi(np.asarray(z, dtype=np.float64))


def _xphi(a: np.ndarray, phi_a: np.ndarray) -> np.ndarray:
    """``a * phi(a)`` with the convention ``(+/-inf) * phi(+/-inf) = 0`` (spec 6.2).

    Zero out the infinite entries of ``a`` BEFORE multiplying so ``inf * 0`` never
    forms a NaN (``phi(+/-inf) == 0`` already, so the finite product is unchanged)."""
    a = np.asarray(a, dtype=np.float64)
    a_fin = np.where(np.isfinite(a), a, 0.0)
    return a_fin * phi_a


def normal_interval_stats(mean: float, var: float, low, high, *,
                          min_prob: float = _DEFAULT_MIN_PROB):
    """Truncated-normal stats for ``Y ~ N(mean, var)`` restricted to ``[low, high)``.

    Returns ``(Q, y_mean, y_var, tau1, tau2)`` where ``Q = P(Y in [low,high))``,
    ``tau1 = E[Y-mean | in]``, ``tau2 = E[(Y-mean)^2 | in]``, ``y_mean = mean+tau1``,
    ``y_var = tau2 - tau1^2``. ``low``/``high`` may be scalars or arrays (vectorized
    over bins) and may be ``-inf`` / ``+inf``. Where ``Q <= min_prob`` the moments are
    returned as 0 (the caller skips that contribution).

    Sanity (full interval ``[-inf, inf]``): ``Q=1, tau1=0, tau2=var, y_var=var``.
    """
    var = float(var)
    sig = math.sqrt(max(var, _VAR_FLOOR))
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    a = (low - mean) / sig
    b = (high - mean) / sig

    phi_a, phi_b = _phi(a), _phi(b)
    Phi_a, Phi_b = _Phi(a), _Phi(b)
    Q = Phi_b - Phi_a

    ok = Q > min_prob
    Qsafe = np.where(ok, Q, 1.0)
    tau1 = np.where(ok, sig * (phi_a - phi_b) / Qsafe, 0.0)
    # tau2 = var * [1 + (a phi(a) - b phi(b)) / Q],  with x*phi(x)->0 at +/-inf
    tau2 = np.where(ok, var * (1.0 + (_xphi(a, phi_a) - _xphi(b, phi_b)) / Qsafe), 0.0)
    y_mean = np.where(ok, mean + tau1, mean)
    y_var = np.clip(tau2 - tau1 * tau1, 0.0, None)
    return Q, y_mean, y_var, tau1, tau2


# --------------------------------------------------------------------------- #
# bin / matrix utilities
# --------------------------------------------------------------------------- #
def find_bin(edges: np.ndarray, x: float) -> int:
    """Index ``beta`` with ``edges[beta] <= x < edges[beta+1]`` (clamped to a valid bin).

    ``edges`` is sorted length ``num_bins+1`` (may start at ``-inf`` / end at ``+inf``).
    """
    edges = np.asarray(edges, dtype=np.float64)
    m = edges.shape[0] - 1
    beta = int(np.searchsorted(edges, x, side="right") - 1)
    return min(max(beta, 0), m - 1)


def safe_bin_representative(edges: np.ndarray, beta: int) -> float:
    """A finite representative for (possibly empty) bin ``beta``: midpoint, or the
    finite boundary offset into an infinite tail bin."""
    lo, hi = float(edges[beta]), float(edges[beta + 1])
    if math.isinf(lo) and math.isinf(hi):
        return 0.0
    if math.isinf(lo):
        return hi - 1.0
    if math.isinf(hi):
        return lo + 1.0
    return 0.5 * (lo + hi)


def symmetrize(A: np.ndarray) -> np.ndarray:
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


def make_gaussian_edges(num_bins: int, *, std: float = 1.0,
                        tail_inf: bool = True) -> np.ndarray:
    """Pre-activation bin edges (length ``num_bins+1``) for a reference ``N(0, std^2)``.

    Interior edges are equal-probability Gaussian quantiles, so each bin carries ~the
    same reference mass. The outermost edges are ``-inf`` / ``+inf`` (``tail_inf``) so
    no transition mass can escape the grid. INCLUDES NEGATIVE BINS -- required for the
    pre-activation state because ``A^+ = gamma A + r^T B`` can be negative even when the
    incoming ``A`` is post-ReLU nonnegative (spec 12.1).
    """
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if num_bins == 1:
        return np.array([-np.inf, np.inf])
    from scipy.special import ndtri  # inverse normal CDF
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.empty(num_bins + 1, dtype=np.float64)
    edges[1:-1] = std * ndtri(qs[1:-1])
    edges[0] = -np.inf if tail_inf else std * ndtri(0.5 / num_bins)
    edges[-1] = np.inf if tail_inf else std * ndtri(1.0 - 0.5 / num_bins)
    return edges


def make_relu_post_edges(num_bins: int, *, std: float = 1.0,
                         tail_inf: bool = True) -> np.ndarray:
    """Post-ReLU bin edges (length ``num_bins+1``), NONNEGATIVE with a zero bin.

    ReLU output is ``>= 0`` with an atom at 0, so bin 0 ``= [0, e_1)`` captures the
    dead mass and the positive bins are half-normal quantiles (spec 12.1)."""
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if num_bins == 1:
        return np.array([0.0, np.inf])
    from scipy.special import ndtri
    # Quantiles of |N(0,std^2)| on (0,1): map to upper-half normal quantiles.
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.empty(num_bins + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:-1] = std * ndtri(0.5 + 0.5 * qs[1:-1])
    edges[-1] = np.inf if tail_inf else std * ndtri(0.5 + 0.5 * (1.0 - 0.5 / num_bins))
    return edges


# --------------------------------------------------------------------------- #
# WASSERSTEIN-OPTIMAL bin placement (Lloyd-Max scalar quantizer)
# --------------------------------------------------------------------------- #
# The binning collapses the spike coordinate A within each cell to one representative,
# so the squared error it introduces is exactly  sum_a p_a Var(A | cell_a) = W_2^2( law(A),
# binned ).  Minimizing that W_2 distance over (edges, reps) is optimal scalar quantization,
# whose stationarity conditions are: reps = cell CENTROIDS (conditional means) and interior
# edges = MIDPOINTS between adjacent reps -- the Lloyd-Max algorithm. This beats equal-mass
# quantiles (``make_gaussian_edges``), which over-resolve the dense centre; the W_2 optimum
# puts more points in the tails (point density ~ f^{1/3} vs ~ f). See ALGORITHM.md sec 7.
def _truncnorm_cell(a: float, b: float) -> Tuple[float, float, float]:
    """``(mass, centroid, within-cell variance)`` of ``N(0,1)`` truncated to ``[a, b)``."""
    Z = float(_Phi(np.array(b)) - _Phi(np.array(a)))
    if Z <= _TINY:
        mid = 0.0 if (math.isinf(a) or math.isinf(b)) else 0.5 * (a + b)
        return 0.0, mid, 0.0
    pa = 0.0 if math.isinf(a) else float(_phi(np.array(a)))
    pb = 0.0 if math.isinf(b) else float(_phi(np.array(b)))
    ap = 0.0 if math.isinf(a) else a * pa
    bp = 0.0 if math.isinf(b) else b * pb
    v = (pa - pb) / Z
    var = max(1.0 + (ap - bp) / Z - v * v, 0.0)
    return Z, v, var


def _lloyd_max_interval(lo: float, hi: float, num_pts: int, *, iters: int = 200,
                        tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max W2 quantizer of ``N(0,1)`` restricted to ``[lo, hi)`` with ``num_pts`` points.
    Returns ``(edges[num_pts+1], reps[num_pts])`` (edges[0]=lo, edges[-1]=hi)."""
    from scipy.special import ndtri
    if num_pts <= 1:
        return np.array([lo, hi], float), np.array([_truncnorm_cell(lo, hi)[1]])
    Plo, Phi_hi = float(_Phi(np.array(lo))), float(_Phi(np.array(hi)))
    qs = np.clip(np.linspace(Plo, Phi_hi, num_pts + 1), 1e-15, 1 - 1e-15)
    e = ndtri(qs); e[0], e[-1] = lo, hi                       # init equal-mass within [lo,hi)
    for _ in range(iters):
        v = np.array([_truncnorm_cell(e[i], e[i + 1])[1] for i in range(num_pts)])
        ne = e.copy(); ne[1:-1] = 0.5 * (v[:-1] + v[1:])       # edges <- midpoints of centroids
        if np.max(np.abs(ne[1:-1] - e[1:-1])) < tol:
            e = ne
            break
        e = ne
    v = np.array([_truncnorm_cell(e[i], e[i + 1])[1] for i in range(num_pts)])
    return e, v


def lloyd_max_edges(mean: float, std: float, num_bins: int, *, rectified: bool = False,
                    iters: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """W2-OPTIMAL bin ``(edges, representatives)`` for the expected continuous spike law.

    Minimizes the Wasserstein-2 distance to the layer's expected continuous law of ``A``:
    ``N(mean, std^2)`` for a pre-activation grid (``rectified=False``), or the rectified
    Gaussian ``max(N(mean, std^2), 0)`` for a post-ReLU grid (``rectified=True`` -- the
    0-atom of mass ``Phi(-mean/std)`` is given its own representative at 0, and the positive
    tail is Lloyd-Max-quantized with the remaining ``num_bins-1`` points). Drop-in alternative
    to ``make_gaussian_edges`` / ``make_relu_post_edges``; also returns the optimal reps
    (cell centroids) -- the linear/ReLU steps recompute reps dynamically, so the edges are
    what matters at grid-build time.
    """
    std = max(float(std), math.sqrt(_VAR_FLOOR))
    if not rectified:
        e_std, v_std = _lloyd_max_interval(-np.inf, np.inf, num_bins, iters=iters)
        return mean + std * e_std, mean + std * v_std
    if num_bins == 1:
        return np.array([0.0, np.inf]), np.array([max(mean, 0.0)])
    a0 = -mean / std                                          # standardized location of 0
    _e_pos, v_pos = _lloyd_max_interval(a0, np.inf, num_bins - 1, iters=iters)  # positive tail reps
    reps = np.empty(num_bins)
    reps[0] = 0.0                                             # the dead-ReLU 0-atom representative
    reps[1:] = np.maximum(mean + std * v_pos, 0.0)
    edges = np.empty(num_bins + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (reps[:-1] + reps[1:])               # Lloyd-Max nearest-rep boundaries
    edges[-1] = np.inf
    return np.maximum.accumulate(edges), reps


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
@dataclass
class BinnedK2State:
    """Binned K=2 law of one layer variable ``X = A e + B`` (see module docstring).

    p:      (num_bins,)      bin probabilities, >= 0, sum to 1
    a:      (num_bins,)      representative spike value E[A | bin]
    mu:     (num_bins, d)    conditional bulk mean  E[B | bin]
    Sigma:  (num_bins, d, d) conditional bulk covariance  Cov[B | bin]
    """
    p: np.ndarray
    a: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray

    @property
    def num_bins(self) -> int:
        return self.p.shape[0]

    @property
    def d(self) -> int:
        return self.mu.shape[1]

    def check(self, tol: float = 1e-8) -> None:
        m, d = self.num_bins, self.d
        assert self.p.shape == (m,), self.p.shape
        assert self.a.shape == (m,), self.a.shape
        assert self.mu.shape == (m, d), self.mu.shape
        assert self.Sigma.shape == (m, d, d), self.Sigma.shape
        assert np.all(self.p >= -tol), f"negative probability: min {self.p.min():.2e}"
        assert abs(self.p.sum() - 1.0) < tol, f"p sums to {self.p.sum():.6f}"
        for alpha in range(m):
            S = self.Sigma[alpha]
            assert np.allclose(S, S.T, atol=tol), f"Sigma[{alpha}] not symmetric"


def gaussian_initial_state(d: int, edges: np.ndarray, *, input_std: float = 1.0,
                           spike_std: Optional[float] = None) -> BinnedK2State:
    """Input law ``X ~ N(0, I)``: spike coord ``A ~ N(0, spike_std^2)`` (default = input_std),
    bulk ``B ~ N(0, input_std^2 I_d)``, ``A _|_ B``. Bins ``A`` by ``edges``.

    Per bin ``[l, u)``: ``p = Phi(u/s)-Phi(l/s)``, ``v = E[A|bin] = s (phi(l')-phi(u'))/p``
    (truncated-normal mean), ``mu = 0``, ``Sigma = input_std^2 I_d`` (spec 4.1).
    """
    edges = np.asarray(edges, dtype=np.float64)
    m = edges.shape[0] - 1
    s = float(input_std if spike_std is None else spike_std)
    Q, y_mean, _yv, _t1, _t2 = normal_interval_stats(0.0, s * s, edges[:-1], edges[1:])
    p = np.clip(Q, 0.0, None)
    total = p.sum()
    p = p / total if total > 0 else p
    a = np.where(p > 0, y_mean, [safe_bin_representative(edges, b) for b in range(m)])
    mu = np.zeros((m, d), dtype=np.float64)
    Sigma = np.broadcast_to((input_std ** 2) * np.eye(d), (m, d, d)).copy()
    Sigma[p <= 0] = 0.0
    return BinnedK2State(p=p, a=a, mu=mu, Sigma=Sigma)


# --------------------------------------------------------------------------- #
# bulk ReLU backends (the K=2 covariance ReLU applied to the bulk coords only)
# --------------------------------------------------------------------------- #
def _bulk_relu_update(mu: np.ndarray, Sigma: np.ndarray, method: str
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """One bin's bulk ReLU: ``(mu, Sigma) -> (E[ReLU(B)], Cov[ReLU(B)])``.

    method:
      "exact"  exact bivariate-Gaussian ReLU covariance (repo's validated kernel)
      "gain"   the leading-order spec-8.2 formula: Sigma_ij <- c_i c_j Sigma_ij
               (i!=j), diagonal exact, c_i = Phi(mu_i/sigma_i)
      "kprop"  delegate to the ordinary harmonic kprop ReLU step at k_max=2
               (imports Mecha_preds.cumulants.kprop; see .kprop_hook)
    """
    if method == "exact":
        return exact_relu_covariance(mu, Sigma)
    if method == "kprop":
        from .kprop_hook import bulk_relu_kprop
        return bulk_relu_kprop(mu, Sigma, k_max=2)
    if method == "gain":
        d = mu.shape[0]
        var = np.clip(np.diag(Sigma).copy(), _VAR_FLOOR, None)
        sigma = np.sqrt(var)
        z = mu / sigma
        Phi, phi = _Phi(z), _phi(z)
        mean_relu = mu * Phi + sigma * phi
        second_relu = (mu * mu + var) * Phi + mu * sigma * phi
        deriv = Phi  # c_i = E[ReLU'(Y_i)] = Phi(mu_i/sigma_i)
        Sig_out = Sigma * np.outer(deriv, deriv)
        np.fill_diagonal(Sig_out, second_relu - mean_relu ** 2)
        return mean_relu, symmetrize(Sig_out)
    raise ValueError(f"unknown bulk_relu method {method!r} (use 'exact', 'gain', or 'kprop')")


# --------------------------------------------------------------------------- #
# linear step  (spec sections 5-7)
# --------------------------------------------------------------------------- #
def linear_step_k2(state: BinnedK2State, M: np.ndarray, edges: np.ndarray, *,
                   min_prob: float = _DEFAULT_MIN_PROB, psd_clip: bool = True,
                   stats: Optional[dict] = None) -> BinnedK2State:
    """Propagate a binned K=2 state through the linear map ``M`` (= W + e e^T).

    ``edges`` are the NEW pre-activation bin edges (length ``num_bins_new+1``; must
    include negative bins). Uses the conditional-Gaussian closure inside each old bin
    (spec section 6) and the transition-kernel mixture (section 5). Memory-efficient:
    never allocates the ``(m, m, d, d)`` tensor from the spec pseudocode -- the
    per-old-bin covariance ``Sigma_C - g g^T / s_Y^2`` is beta-independent, so only the
    rank-1 ``g g^T`` piece is reweighted per new bin.
    """
    p, avals, mu, Sigma = state.p, state.a, state.mu, state.Sigma
    m_old = p.shape[0]
    d = mu.shape[1]
    edges = np.asarray(edges, dtype=np.float64)
    m_new = edges.shape[0] - 1
    low, high = edges[:-1], edges[1:]

    gamma = float(M[0, 0])
    r = np.asarray(M[0, 1:], dtype=np.float64)      # (d,)
    u = np.asarray(M[1:, 0], dtype=np.float64)      # (d,)
    V = np.asarray(M[1:, 1:], dtype=np.float64)     # (d, d)

    # transition scalars Q[beta, alpha] etc., and per-old-bin bulk pieces
    Q = np.zeros((m_new, m_old))
    ybar = np.zeros((m_new, m_old))
    tau1 = np.zeros((m_new, m_old))
    yvar = np.zeros((m_new, m_old))
    mC = np.zeros((m_old, d))
    g = np.zeros((m_old, d))
    ag = np.zeros((m_old, d))
    Sig0 = np.zeros((m_old, d, d))      # Sigma_C - g g^T / s_Y^2  (beta-independent)
    sY2 = np.zeros(m_old)

    for alpha in range(m_old):
        if p[alpha] <= 0:
            continue
        v = float(avals[alpha])
        mu_a = mu[alpha]
        Sig_a = Sigma[alpha]

        mY = gamma * v + float(r @ mu_a)
        mC[alpha] = u * v + V @ mu_a
        Sig_r = Sig_a @ r
        sY2[alpha] = float(r @ Sig_r)
        g[alpha] = V @ Sig_r
        SigC = V @ Sig_a @ V.T

        if sY2[alpha] < min_prob:                    # Y_alpha deterministic = mY (spec 6.1)
            beta = find_bin(edges, mY)
            Q[beta, alpha] = 1.0
            ybar[beta, alpha] = mY
            Sig0[alpha] = symmetrize(SigC)           # g ~ 0 here, so Sig0 = SigC
            continue

        ag[alpha] = g[alpha] / sY2[alpha]
        Sig0[alpha] = symmetrize(SigC - np.outer(g[alpha], g[alpha]) / sY2[alpha])
        Qa, y_mean, y_var, t1, _t2 = normal_interval_stats(mY, sY2[alpha], low, high,
                                                           min_prob=min_prob)
        keep = Qa > min_prob
        Q[:, alpha] = np.where(keep, Qa, 0.0)
        ybar[:, alpha] = np.where(keep, y_mean, 0.0)
        tau1[:, alpha] = np.where(keep, t1, 0.0)
        yvar[:, alpha] = np.where(keep, y_var, 0.0)

    p_new_raw = Q @ p                                # (m_new,)
    a_new = np.zeros(m_new)
    mu_new = np.zeros((m_new, d))
    Sig_new = np.zeros((m_new, d, d))
    psd_clipped = 0.0

    for beta in range(m_new):
        if p_new_raw[beta] <= min_prob:
            a_new[beta] = safe_bin_representative(edges, beta)
            continue
        eta = p * Q[beta, :] / p_new_raw[beta]       # posterior weights (m_old,)
        active = np.nonzero(eta > 0)[0]
        a_new[beta] = float(np.sum(eta[active] * ybar[beta, active]))

        # mixture mean: mu_{alpha->beta} = mC + ag * tau1
        mu_ab = mC[active] + ag[active] * tau1[beta, active][:, None]   # (k, d)
        mu_new[beta] = eta[active] @ mu_ab

        # mixture covariance: Sigma_{alpha->beta} = Sig0 + (yvar/sY2^2) g g^T
        S = np.zeros((d, d))
        for j, alpha in enumerate(active):
            Sig_ab = Sig0[alpha]
            if sY2[alpha] >= min_prob:
                Sig_ab = Sig_ab + (yvar[beta, alpha] / (sY2[alpha] ** 2)) * np.outer(g[alpha], g[alpha])
            delta = mu_ab[j] - mu_new[beta]
            S += eta[alpha] * (Sig_ab + np.outer(delta, delta))
        S = symmetrize(S)
        if psd_clip:
            S, c = project_to_psd(S)
            psd_clipped += c
        Sig_new[beta] = S

    total = p_new_raw.sum()
    mass_lost = float(max(0.0, 1.0 - total))
    if total <= 0:
        raise RuntimeError("all spike-bin mass vanished in linear_step_k2")
    p_new = p_new_raw / total

    if stats is not None:
        stats.setdefault("linear_mass_lost", []).append(mass_lost)
        stats.setdefault("linear_psd_clipped", []).append(psd_clipped)
    return BinnedK2State(p=p_new, a=a_new, mu=mu_new, Sigma=Sig_new)


# --------------------------------------------------------------------------- #
# ReLU step  (spec sections 8-9)
# --------------------------------------------------------------------------- #
def relu_step_k2(state: BinnedK2State, post_edges: np.ndarray, *,
                 min_prob: float = _DEFAULT_MIN_PROB, bulk_relu: str = "exact",
                 stats: Optional[dict] = None) -> BinnedK2State:
    """Propagate a binned K=2 state through coordinatewise ReLU.

    Bulk: apply the K=2 ReLU covariance update inside every old bin (``bulk_relu``
    backend). Spike: map each representative ``v -> max(v, 0)``, assign to ``post_edges``
    (nonnegative grid with a zero bin), and MERGE old bins landing in the same new bin
    by mixture moments (spec 8.3 / 9).
    """
    p, avals, mu, Sigma = state.p, state.a, state.mu, state.Sigma
    m_old = p.shape[0]
    d = mu.shape[1]
    post_edges = np.asarray(post_edges, dtype=np.float64)
    m_post = post_edges.shape[0] - 1

    # 1) bulk ReLU inside each old bin
    a_bulk = np.zeros((m_old, d))
    Sig_bulk = np.zeros((m_old, d, d))
    for alpha in range(m_old):
        if p[alpha] <= 0:
            continue
        a_bulk[alpha], Sig_bulk[alpha] = _bulk_relu_update(mu[alpha], Sigma[alpha], bulk_relu)

    # 2) map spike representatives through ReLU and merge bins
    old_to_new = [find_bin(post_edges, max(float(avals[alpha]), 0.0)) for alpha in range(m_old)]
    p_new_raw = np.zeros(m_post)
    for alpha in range(m_old):
        if p[alpha] > 0:
            p_new_raw[old_to_new[alpha]] += p[alpha]

    a_new = np.zeros(m_post)
    mu_new = np.zeros((m_post, d))
    Sig_new = np.zeros((m_post, d, d))
    for beta in range(m_post):
        if p_new_raw[beta] <= min_prob:
            a_new[beta] = safe_bin_representative(post_edges, beta)
            continue
        alphas = [a for a in range(m_old) if old_to_new[a] == beta and p[a] > 0]
        eta = np.array([p[a] / p_new_raw[beta] for a in alphas])
        a_new[beta] = float(sum(e * max(float(avals[a]), 0.0) for e, a in zip(eta, alphas)))
        for e, a in zip(eta, alphas):
            mu_new[beta] += e * a_bulk[a]
        S = np.zeros((d, d))
        for e, a in zip(eta, alphas):
            delta = a_bulk[a] - mu_new[beta]
            S += e * (Sig_bulk[a] + np.outer(delta, delta))
        Sig_new[beta] = symmetrize(S)

    total = p_new_raw.sum()
    mass_lost = float(max(0.0, 1.0 - total))
    if total <= 0:
        raise RuntimeError("all spike-bin mass vanished in relu_step_k2")
    p_new = p_new_raw / total
    if stats is not None:
        stats.setdefault("relu_mass_lost", []).append(mass_lost)
    return BinnedK2State(p=p_new, a=a_new, mu=mu_new, Sigma=Sig_new)


# --------------------------------------------------------------------------- #
# recover unconditional moments  (spec section 11)
# --------------------------------------------------------------------------- #
def unconditional_mean(state: BinnedK2State) -> np.ndarray:
    """Full mean ``E[X]`` (length ``n = d+1``): coord 0 = ``sum_a p_a v_a``,
    coords 1.. = ``sum_a p_a mu_a``."""
    p = state.p
    Abar = float(p @ state.a)
    Bbar = p @ state.mu
    out = np.empty(state.d + 1, dtype=np.float64)
    out[SPIKE_COORD] = Abar
    out[1:] = Bbar
    return out


def unconditional_mean_cov(state: BinnedK2State) -> Tuple[np.ndarray, np.ndarray]:
    """Full mean and block covariance ``Cov[X]`` (n x n) by the law of total
    (co)variance over the finite spike distribution (spec section 11)."""
    p, a, mu, Sigma = state.p, state.a, state.mu, state.Sigma
    d = state.d
    Abar = float(p @ a)
    Bbar = p @ mu
    da = a - Abar
    dB = mu - Bbar                                   # (m, d)
    var_A = float(p @ (da * da))
    cov_AB = (p * da) @ dB                           # (d,)
    cov_B = np.zeros((d, d))
    for alpha in range(state.num_bins):
        cov_B += p[alpha] * (Sigma[alpha] + np.outer(dB[alpha], dB[alpha]))
    cov_B = symmetrize(cov_B)

    n = d + 1
    mean = np.empty(n)
    mean[SPIKE_COORD] = Abar
    mean[1:] = Bbar
    cov = np.zeros((n, n))
    cov[0, 0] = var_A
    cov[0, 1:] = cov_AB
    cov[1:, 0] = cov_AB
    cov[1:, 1:] = cov_B
    return mean, symmetrize(cov)


# --------------------------------------------------------------------------- #
# full forward  (spec section 13)
# --------------------------------------------------------------------------- #
def run_binned_kprop_k2(weights: List[Tuple[np.ndarray, Optional[np.ndarray]]],
                        input_dim: int, num_bins: int, *,
                        pre_edges: Optional[np.ndarray] = None,
                        post_edges: Optional[np.ndarray] = None,
                        num_bins_post: Optional[int] = None,
                        input_std: float = 1.0, bulk_relu: str = "exact",
                        min_prob: float = _DEFAULT_MIN_PROB,
                        collect: bool = False) -> dict:
    """Predict ``E[f(X)]`` for ``X ~ N(0, input_std^2 I)`` by COORDINATE-SPIKE BINNED
    kprop (K=2) along ``e = e_1``.

    ``weights`` are ``(W, b)`` float64 pairs in forward order: the square ``n x n``
    hidden matrices ``M = W + e e^T`` (the spike is assumed already baked in -- pass the
    actual matrices), each followed by ReLU, then the linear readout (no ReLU).
    ``num_bins`` is THE hyperparameter: the number of spike bins. ``num_bins_post``
    (default = ``num_bins``) sizes the post-ReLU grid. Default grids are equal-mass
    Gaussian quantiles (``pre_edges`` includes negative bins; ``post_edges`` is
    nonnegative). Returns ``{"mean", "metadata", ...}``; with ``collect`` also the
    per-layer spike distribution and the final state.
    """
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    n_hidden = len(weights) - 1
    if n_hidden < 1:
        raise ValueError("need at least one hidden layer + a readout")
    d = input_dim - 1
    if num_bins_post is None:
        num_bins_post = num_bins
    if pre_edges is None:
        pre_edges = make_gaussian_edges(num_bins, std=input_std)
    if post_edges is None:
        post_edges = make_relu_post_edges(num_bins_post, std=input_std)
    pre_edges = np.asarray(pre_edges, dtype=np.float64)
    post_edges = np.asarray(post_edges, dtype=np.float64)

    stats: dict = {}
    state = gaussian_initial_state(d, pre_edges, input_std=input_std)
    spike_by_layer = []
    for li in range(n_hidden):
        W, _b = weights[li]
        M = np.asarray(W, dtype=np.float64)
        if M.shape != (input_dim, input_dim):
            raise ValueError(f"hidden layer {li} must be square ({input_dim},{input_dim}); "
                             f"got {M.shape}")
        state = linear_step_k2(state, M, pre_edges, min_prob=min_prob, stats=stats)
        state = relu_step_k2(state, post_edges, min_prob=min_prob, bulk_relu=bulk_relu, stats=stats)
        if collect:
            spike_by_layer.append(dict(layer=li, p=state.p.copy(), a=state.a.copy()))

    W_ro, b_ro = weights[-1]
    W_ro = np.asarray(W_ro, dtype=np.float64)
    full_mean = unconditional_mean(state)            # length n = d+1
    mean = W_ro @ full_mean + (0.0 if b_ro is None else np.asarray(b_ro, dtype=np.float64))

    out = {
        "mean": np.asarray(mean, dtype=np.float64).reshape(-1),
        "metadata": {
            "predictor": "binned_kprop_k2", "K": 2, "num_bins": int(num_bins),
            "num_bins_post": int(num_bins_post), "n_hidden": n_hidden,
            "input_dim": int(input_dim), "bulk_relu": bulk_relu,
            "output_dim": int(np.asarray(mean).reshape(-1).shape[0]),
            "max_linear_mass_lost": float(max(stats.get("linear_mass_lost", [0.0]))),
            "max_relu_mass_lost": float(max(stats.get("relu_mass_lost", [0.0]))),
            "total_psd_clipped": float(sum(stats.get("linear_psd_clipped", [0.0]))),
        },
    }
    if collect:
        out["spike_by_layer"] = spike_by_layer
        out["final_state"] = state
        out["stats"] = stats
    return out

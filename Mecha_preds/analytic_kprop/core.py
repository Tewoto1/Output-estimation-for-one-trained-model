"""core.py -- ANALYTIC AFFINE-CONDITIONED K=2 propagation for coordinate-spiked
ReLU MLPs (numpy/scipy, torch-free).

Implements the finite-grid algorithm of the write-up *"Analytic Affine-Conditioned
K=2 Propagation for Coordinate-Spiked ReLU Networks"* (analytic_affine_kprop.pdf,
Algorithm 7.2, with the exact-cell scalar backend of section 5.3). Same model class
as the companion ``..binned_kprop``:

    Z = M X + b,   X^+ = ReLU(Z),   M = W + e_1 e_1^T,   W_ij ~ N(0, 1/n),

with the spike coordinate ``A = e_1 . X`` explicit and the bulk ``B`` (dimension
``d = n - 1``) conditional on it. THE DIFFERENCE from the binned companion: instead
of carrying one bulk Gaussian PER BIN through the whole propagation (an HMM with a
bin->bin transition kernel and per-bin d^3 congruences), the bulk conditional law
between layers is compressed to a single AFFINE family in the new spike
pre-activation ``y``,

    C | Y = y  ~~>  N(mu0 + mu1 y,  Sigma0 + Sigma1 y),          (paper eq 11)

so the per-layer heavy algebra is O(1) congruences ``V . V^T`` (two aggregated
ones -- see ``_covariance_sums``) instead of O(num_bins) of them. The scalar law of
``Y`` is a KNOWN Gaussian mixture (one component per retained post-ReLU node, paper
eq 51), and the ``num_nodes`` quadrature cells only have to "bash" closed-form
probabilities/moments out of that mixture (truncated-normal identities, paper eqs
63-68) -- no per-node bulk state is tracked before the ReLU.

One layer (Algorithm 7.2):
  1. component params of the old atomic post-ReLU state under the block map
     (paper eqs 60-61);
  2. per-(cell, component) truncated-normal stats + exact cell-conditioned
     moments (eqs 63-68), merged within each cell (eqs 69-72);
  3. weighted least-squares AFFINE RE-PROJECTION of the conditional mean and
     covariance (eqs 86-87), with the optional moment-conservative intercept
     ``Sigma0 += R_m`` (eq 90, default here -- kprop tracks global 2nd moments);
  4. exact Gaussian-ReLU moments at every retained node (eqs 98-99, backend =
     the repo's shared exact bivariate kernel), positive nodes kept as atoms
     (eq 39), all nonpositive nodes merged EXACTLY into the zero atom by total
     expectation / total covariance (eqs 40-42).

Only two conceptual approximations remain (paper thm 10.1): the conditional K=2
Gaussian closure and the affine re-projection; the finite grid adds a controlled
1-D quantization term (eq 134, logged as ``scalar_distortion``).

The input layer is handled EXACTLY (no input discretization, unlike the binned
companion): ``X0 ~ N(0, I)`` is one Gaussian component with within-component spike
variance ``t2 = 1`` (the state's optional ``t2`` field), under which (Y, C) are
jointly Gaussian and the same cell formulas apply unchanged (paper section 9).

Shared kernels: ReLU integrals / PSD utilities from ``.._utils``; truncated-normal
cell stats and W2-optimal (Lloyd-Max) mixture grids from ``..binned_kprop.binning``;
the per-node bulk-ReLU backend dispatch from ``..binned_kprop.core``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .._utils import _Phi, symmetrize, project_to_psd
from ..binned_kprop.binning import (
    _DEFAULT_MIN_PROB, _VAR_FLOOR,
    find_bin, normal_interval_stats, lloyd_max_edges_mixture_split,
)
from ..binned_kprop.core import _bulk_relu_update

# Coordinate of the spike (e = e_1 -> index 0 in Python). The bulk is coords 1..n-1.
SPIKE_COORD = 0
_TINY_VAR_Y = 1e-14          # v_Y below this -> slope unidentifiable, use mu1 = 0 (paper 6.1)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
@dataclass
class AnalyticState:
    """Atomic (post-ReLU) slice state ``{p_i, a_i, m_i, S_i}`` (paper eq 97).

    p:      (m,)      node probabilities, >= 0, sum to 1
    a:      (m,)      spike node values (post-ReLU: >= 0, node 0 = the zero atom)
    mu:     (m, d)    conditional bulk mean   E[B | A = a_i]
    Sigma:  (m, d, d) conditional bulk cov    Cov[B | A = a_i]
    t2:     (m,)      OPTIONAL within-component spike variance Var(A | component).
                      0 for genuine atoms; the exact Gaussian INPUT state is the
                      single component (a=0, t2=1), which makes layer 1 exact
                      (paper section 9) with the same cell formulas.
    """
    p: np.ndarray
    a: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray
    t2: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.t2 is None:
            self.t2 = np.zeros_like(self.a)

    @property
    def num_nodes(self) -> int:
        return self.p.shape[0]

    @property
    def d(self) -> int:
        return self.mu.shape[1]

    def check(self, tol: float = 1e-8) -> None:
        m, d = self.num_nodes, self.d
        assert self.p.shape == (m,) and self.a.shape == (m,) and self.t2.shape == (m,)
        assert self.mu.shape == (m, d) and self.Sigma.shape == (m, d, d)
        assert np.all(self.p >= -tol), f"negative probability: min {self.p.min():.2e}"
        assert abs(self.p.sum() - 1.0) < tol, f"p sums to {self.p.sum():.6f}"
        for i in range(m):
            S = self.Sigma[i]
            assert np.allclose(S, S.T, atol=tol), f"Sigma[{i}] not symmetric"


def gaussian_input_state(d: int, *, input_std: float = 1.0) -> AnalyticState:
    """The EXACT input state for ``X ~ N(0, input_std^2 I_n)`` (paper eq 123): one
    Gaussian component, ``A ~ N(0, s^2)`` carried by ``t2`` (no discretization),
    ``B ~ N(0, s^2 I_d)`` independent."""
    s2 = float(input_std) ** 2
    return AnalyticState(
        p=np.array([1.0]), a=np.array([0.0]),
        mu=np.zeros((1, d)), Sigma=(s2 * np.eye(d))[None, :, :].copy(),
        t2=np.array([s2]))


@dataclass
class AffineState:
    """The affine pre-activation state (paper def 2.2) at one layer: scalar mixture
    law of ``Y`` (weights/means/vars, one component per old node) plus the fitted
    affine bulk model ``C | Y=y ~ N(mu0 + mu1 y, Sigma0 + Sigma1 y)`` and the grid
    ``(edges, w, y)`` of retained cells. Returned for diagnostics/collect."""
    mix_w: np.ndarray
    mix_mean: np.ndarray
    mix_var: np.ndarray
    edges: np.ndarray
    w: np.ndarray
    y: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    Sigma0: np.ndarray
    Sigma1: np.ndarray


# --------------------------------------------------------------------------- #
# layer blocks
# --------------------------------------------------------------------------- #
def _layer_block(M: np.ndarray, b: Optional[np.ndarray], d: int):
    """Split a square layer ``(M, b)`` into the spike/bulk blocks (paper eq 4):
    ``gamma = M[0,0]``, ``r = M[0,1:]``, ``u = M[1:,0]``, ``V = M[1:,1:]``,
    ``beta = b[0]``, ``eta = b[1:]``."""
    M = np.asarray(M, dtype=np.float64)
    gamma = float(M[0, 0])
    r = M[0, 1:].copy()
    u = M[1:, 0].copy()
    V = M[1:, 1:]
    if b is None:
        beta, eta = 0.0, np.zeros(d)
    else:
        b = np.asarray(b, dtype=np.float64)
        beta, eta = float(b[0]), b[1:].copy()
    return gamma, r, u, V, beta, eta


def _component_params(state: AnalyticState, gamma, r, u, V, beta, eta):
    """Per-component scalar/bulk parameters under the block map (paper eqs 60-61),
    generalized by the within-component spike variance ``t2`` (exact for the
    Gaussian input component; ``t2 = 0`` reduces to eqs 60-61 verbatim):

        m_Y,i  = gamma a_i + r . m_i + beta
        s2_Y,i = gamma^2 t2_i + r . S_i r
        m_C,i  = u a_i + V m_i + eta
        g_i    = gamma t2_i u + V S_i r          (= Cov(C, Y | component i))

    ``S_C,i = t2_i u u^T + V S_i V^T`` is NOT formed per component -- every use of
    it downstream is linear with scalar weights, so ``_covariance_sums`` aggregates
    ``S_i`` / ``t2_i`` first and applies ONE congruence per weighted sum."""
    a, mu, t2 = state.a, state.mu, state.t2
    mY = gamma * a + mu @ r + beta                              # (m,)
    Sr = np.einsum("iab,b->ia", state.Sigma, r, optimize=True)                 # (m, d)  S_i r
    sY2 = gamma * gamma * t2 + np.einsum("ia,a->i", Sr, r, optimize=True)      # (m,)
    mC = a[:, None] * u[None, :] + mu @ V.T + eta[None, :]      # (m, d)
    g = gamma * t2[:, None] * u[None, :] + Sr @ V.T             # (m, d)
    return mY, np.clip(sY2, 0.0, None), mC, g


# --------------------------------------------------------------------------- #
# scalar grids (edge exactly at zero -- paper checklist item 2)
# --------------------------------------------------------------------------- #
def negative_mass(p: np.ndarray, mY: np.ndarray, sY2: np.ndarray) -> float:
    """Mixture mass of ``{Y <= 0}`` (closed form)."""
    s = np.sqrt(np.maximum(sY2, _VAR_FLOOR))
    return float(np.sum(p * _Phi(-mY / s)))


def split_node_budget(num_nodes: int, neg_mass: float,
                      num_nodes_neg: Optional[int], num_nodes_pos: Optional[int]
                      ) -> Tuple[int, int]:
    """Allocate the ``num_nodes`` cell budget across the sign split, proportional to
    the mixture mass on each side (>= 1 cell each side; overrides win)."""
    if num_nodes_neg is not None and num_nodes_pos is not None:
        return max(1, int(num_nodes_neg)), max(1, int(num_nodes_pos))
    if num_nodes_neg is not None:
        return max(1, int(num_nodes_neg)), max(1, num_nodes - int(num_nodes_neg))
    if num_nodes_pos is not None:
        return max(1, num_nodes - int(num_nodes_pos)), max(1, int(num_nodes_pos))
    n_neg = int(np.clip(round(num_nodes * neg_mass), 1, num_nodes - 1))
    return n_neg, num_nodes - n_neg


def make_cells(p, mY, sY2, n_neg: int, n_pos: int, *, grid: str = "w2",
               uniform_pad: float = 6.0) -> np.ndarray:
    """Signed pre-activation cell edges (length ``n_neg + n_pos + 1``) with an edge
    EXACTLY at 0 and infinite outer tails, from the KNOWN scalar mixture.

    grid="w2"      Lloyd-Max (Wasserstein-2-optimal) quantization of the exact
                   Gaussian mixture, split at 0 (reuses the binned companion's
                   closed-form quantizer; this directly minimizes the scalar
                   distortion term, paper eq 134).
    grid="uniform" equal-width cells over the +-``uniform_pad`` sigma envelope of
                   the mixture, split at 0.
    """
    s = np.sqrt(np.maximum(np.asarray(sY2, float), _VAR_FLOOR))
    if grid == "w2":
        edges, _reps = lloyd_max_edges_mixture_split(p, mY, s, n_neg, n_pos)
        return edges
    if grid == "uniform":
        lo = float(np.min(mY - uniform_pad * s))
        hi = float(np.max(mY + uniform_pad * s))
        lo, hi = min(lo, -1e-3), max(hi, 1e-3)      # keep both sides nonempty
        e_neg = np.linspace(lo, 0.0, n_neg + 1) if n_neg > 1 else np.array([lo, 0.0])
        e_pos = np.linspace(0.0, hi, n_pos + 1) if n_pos > 1 else np.array([0.0, hi])
        edges = np.concatenate([[-np.inf], e_neg[1:-1], [0.0], e_pos[1:-1], [np.inf]])
        return edges
    raise ValueError(f"grid must be 'w2' or 'uniform', got {grid!r}")


# --------------------------------------------------------------------------- #
# one layer: linear + recondition + affine fit + ReLU  (Algorithm 7.2)
# --------------------------------------------------------------------------- #
def _pair_stats(p, mY, sY2, edges, *, min_prob):
    """Per-(component i, cell j) truncated-normal stats (paper eqs 63-66):
    ``Q[i,j] = P(Y_i in cell j)``, ``ymean[i,j] = E[Y | i, cell j]``,
    ``delta[i,j] = ymean - m_Y,i``, ``vv[i,j] = Var(Y | i, cell j)``.
    Deterministic components (``s2_Y,i < min_prob``) go wholly to their containing
    cell (paper eq 73)."""
    m = p.shape[0]
    low, high = edges[:-1], edges[1:]
    J = low.shape[0]
    Q = np.zeros((m, J)); ymean = np.zeros((m, J))
    delta = np.zeros((m, J)); vv = np.zeros((m, J))
    stoch = np.zeros(m, dtype=bool)
    for i in range(m):
        if p[i] <= 0.0:
            continue
        if sY2[i] < min_prob:                                   # deterministic branch
            j = find_bin(edges, float(mY[i]))
            Q[i, j] = 1.0; ymean[i, j] = mY[i]
            continue
        stoch[i] = True
        Qi, ym, yv, t1, _t2 = normal_interval_stats(float(mY[i]), float(sY2[i]),
                                                    low, high, min_prob=min_prob)
        keep = Qi > min_prob
        Q[i] = np.where(keep, Qi, 0.0)
        ymean[i] = np.where(keep, ym, 0.0)
        delta[i] = np.where(keep, t1, 0.0)
        vv[i] = np.where(keep, yv, 0.0)
    return Q, ymean, delta, vv, stoch


def _covariance_sums(p, Q, delta, vv, sY2, stoch, y, w, mhat,
                     Sigma, t2, mC, g, u, V):
    """The two grid-weighted covariance sums the affine fit needs (paper eq 87),

        T0 = sum_j w_j Shat_j,      T1 = sum_j w_j y_j Shat_j,

    WITHOUT forming any per-cell (or per-pair) d x d matrix. Expansion of
    ``Shat_j = sum_i eta_{i|j} [S_{i->j} + (m_{i->j} - mhat_j)(...)^T]`` (eqs 68/72)
    over components:

      * ``S_C,i``-part: scalar weights ``sum_j Q_ij c_j`` -> aggregate ``S_i``/``t2_i``
        first, then ONE congruence ``V (sum_i . S_i) V^T`` per sum  (the complexity
        win over the binned companion's per-bin congruences);
      * rank-1 ``g_i g_i^T`` part: scalar weights ``sum_j Q_ij c_j (v_ij - s2_i)/s2_i^2``;
      * between-mean part: ``m_{i->j} = m_C,i + u_reg,i delta_ij`` is affine in the
        scalar ``delta``, so second moments reduce to the scalar sums
        ``sum_j Q_ij c_j delta^k`` (k = 0,1,2), minus ``sum_j w_j c_j mhat_j mhat_j^T``.

    Returns raw (unnormalized-by-total-mass) ``T0, T1``.
    """
    s2 = np.where(stoch, sY2, 1.0)
    u_reg = np.where(stoch[:, None], g / s2[:, None], 0.0)              # (m, d)
    gcoef = np.where(stoch[:, None], 1.0, 0.0) * g                      # g zeroed if det

    out = []
    for cell_scale in (np.ones_like(y), y):
        Qc = Q * cell_scale[None, :]                                    # (m, J)
        cq = Qc.sum(axis=1)                                             # sum_j Q c_j
        # S_C part: V (sum_i p_i cq_i S_i) V^T + (sum_i p_i cq_i t2_i) u u^T
        Sagg = np.einsum("i,iab->ab", p * cq, Sigma, optimize=True)
        SC = V @ Sagg @ V.T + float(np.sum(p * cq * t2)) * np.outer(u, u)
        # rank-1 g g^T part: coefficient sum_j Q c_j (v - s2)/s2^2  (stochastic only)
        hv = (Qc * (vv - s2[:, None])).sum(axis=1) / (s2 * s2)
        hv = np.where(stoch, hv, 0.0)
        Gterm = np.einsum("i,ia,ib->ab", p * hv, gcoef, gcoef, optimize=True)
        # between-mean second moments: A0/A1/A2 = sum_j Q c_j delta^{0,1,2}
        A0 = cq
        A1 = (Qc * delta).sum(axis=1)
        A2 = (Qc * delta * delta).sum(axis=1)
        M2 = (np.einsum("i,ia,ib->ab", p * A0, mC, mC, optimize=True)
              + np.einsum("i,ia,ib->ab", p * A2, u_reg, u_reg, optimize=True))
        X = np.einsum("i,ia,ib->ab", p * A1, mC, u_reg, optimize=True)
        M2 += X + X.T
        Mhat = np.einsum("j,ja,jb->ab", w * cell_scale, mhat, mhat, optimize=True)
        out.append(symmetrize(SC + Gterm + M2 - Mhat))
    return out[0], out[1]


def percell_bulk_moments(p, Q, delta, vv, sY2, stoch, w, Sigma, t2, mC, g, u, V):
    """REFERENCE / diagnostics path: the per-cell merged moments ``(mhat_j, Shat_j)``
    (paper eqs 71-72), forming J d x d matrices (J congruences via the aggregated
    ``A_j = sum_i eta_{i|j} S_i``). Used by the selftest to validate
    ``_covariance_sums`` and by ``diagnostics=True`` for the E_S residual."""
    m, J = Q.shape
    PQ = p[:, None] * Q                                                 # (m, J)
    eta = PQ / np.where(w > 0, w, 1.0)[None, :]
    s2 = np.where(stoch, sY2, 1.0)
    u_reg = np.where(stoch[:, None], g / s2[:, None], 0.0)
    m_ij_0 = mC                                                          # (m, d)
    # mhat_j = sum_i eta_ij (mC_i + u_reg_i delta_ij)
    mhat = eta.T @ m_ij_0 + (eta * delta).T @ u_reg                      # (J, d)
    # Shat_j = sum_i eta [S_C,i + g g^T (v-s2)/s2^2 + (m_ij - mhat_j)(...)^T]
    Aj = np.einsum("ij,iab->jab", eta, Sigma, optimize=True)             # (J, d, d)
    t2j = eta.T @ t2                                                     # (J,)
    Shat = V @ Aj @ V.T                                                  # batched congruence (BLAS)
    Shat += t2j[:, None, None] * np.outer(u, u)[None, :, :]
    hv = np.where(stoch[:, None], (vv - s2[:, None]) / (s2 * s2)[:, None], 0.0)  # (m, J)
    gz = np.where(stoch[:, None], g, 0.0)
    Shat += np.einsum("ij,ia,ib->jab", eta * hv, gz, gz, optimize=True)
    # between-mean outer products
    dm = m_ij_0[:, None, :] + u_reg[:, None, :] * delta[:, :, None] - mhat[None, :, :]  # (m, J, d)
    Shat += np.einsum("ij,ija,ijb->jab", eta, dm, dm, optimize=True)
    return mhat, 0.5 * (Shat + np.swapaxes(Shat, 1, 2))


def analytic_layer_update(state: AnalyticState, M: np.ndarray, b: Optional[np.ndarray],
                          *, num_nodes: int, num_nodes_neg: Optional[int] = None,
                          num_nodes_pos: Optional[int] = None, grid: str = "w2",
                          bulk_relu: str = "exact", cov_intercept: str = "mc",
                          min_prob: float = _DEFAULT_MIN_PROB,
                          diagnostics: bool = False,
                          stats: Optional[dict] = None
                          ) -> Tuple[AnalyticState, AffineState]:
    """One hidden layer of Algorithm 7.2: linear + Bayesian reconditioning on the new
    spike coordinate + affine re-projection + slice-wise exact Gaussian-ReLU + exact
    zero-atom merge. Returns ``(new_post_relu_state, affine_state)``.

    ``cov_intercept``: "mc" (default) adds the moment-conservative correction
    ``Sigma0 += R_m`` (paper eq 90), which preserves the unconditional bulk
    covariance; "ls" keeps the literal least-squares intercept (eq 87)."""
    if cov_intercept not in ("mc", "ls"):
        raise ValueError(f"cov_intercept must be 'mc' or 'ls', got {cov_intercept!r}")
    d = state.d
    p_in = state.p
    gamma, r, u, V, beta, eta_b = _layer_block(M, b, d)

    # ---- components of the scalar mixture + conditional bulk (eqs 60-61, 51) ----
    mY, sY2, mC, g = _component_params(state, gamma, r, u, V, beta, eta_b)

    # ---- signed cell grid with an edge at 0 (checklist 2) ----
    n_neg, n_pos = split_node_budget(num_nodes, negative_mass(p_in, mY, sY2),
                                     num_nodes_neg, num_nodes_pos)
    edges = make_cells(p_in, mY, sY2, n_neg, n_pos, grid=grid)

    # ---- per-(component, cell) closed-form stats (eqs 63-66) ----
    Q, ymean, delta, vv, stoch = _pair_stats(p_in, mY, sY2, edges, min_prob=min_prob)

    w_raw = p_in @ Q                                                   # (J,)
    W_tot = float(w_raw.sum())
    if W_tot <= 0.0:
        raise RuntimeError("all scalar mass vanished in analytic_layer_update")
    retained = w_raw > min_prob
    Q = Q[:, retained]; ymean = ymean[:, retained]
    delta = delta[:, retained]; vv = vv[:, retained]
    w_raw = w_raw[retained]

    # exact mixture centroids as cell representatives (eq 70)
    PQ = p_in[:, None] * Q
    y = (PQ * ymean).sum(axis=0) / w_raw                               # (J,)

    # cell-merged conditional means (eqs 67, 71), O(m J d)
    s2 = np.where(stoch, sY2, 1.0)
    u_reg = np.where(stoch[:, None], g / s2[:, None], 0.0)
    mhat = (PQ.T @ mC + (PQ * delta).T @ u_reg) / w_raw[:, None]       # (J, d)

    # ---- affine re-projection (eqs 86-87) ----
    w = w_raw / W_tot
    ybar = float(w @ y)
    vY = float(w @ (y - ybar) ** 2)
    if vY > _TINY_VAR_Y:
        mu1 = ((w * (y - ybar))[:, None] * mhat).sum(axis=0) / vY
    else:
        mu1 = np.zeros(d)                                              # slope unidentifiable
    mu0 = (w[:, None] * mhat).sum(axis=0) - mu1 * ybar
    e_m = mhat - mu0[None, :] - y[:, None] * mu1[None, :]              # mean residual (eq 78)
    E_m = float(w @ (e_m * e_m).sum(axis=1))
    R_m = np.einsum("j,ja,jb->ab", w, e_m, e_m, optimize=True)                        # eq 88

    T0, T1 = _covariance_sums(p_in, Q, delta, vv, sY2, stoch, y, w_raw, mhat,
                              state.Sigma, state.t2, mC, g, u, V)
    T0 /= W_tot; T1 /= W_tot
    if vY > _TINY_VAR_Y:
        Sigma1 = symmetrize((T1 - ybar * T0) / vY)
    else:
        Sigma1 = np.zeros((d, d))
    Sigma0 = symmetrize(T0 - ybar * Sigma1)                            # LS intercept (eq 87)
    if cov_intercept == "mc":
        Sigma0 = symmetrize(Sigma0 + R_m)                              # eq 90

    E_S = float("nan")
    if diagnostics:
        _mh_ref, Shat = percell_bulk_moments(p_in, Q, delta, vv, sY2, stoch, w_raw,
                                             state.Sigma, state.t2, mC, g, u, V)
        resid = Shat - Sigma0[None] - y[:, None, None] * Sigma1[None] \
            + (R_m[None] if cov_intercept == "mc" else 0.0)            # E_S is vs the LS fit (eq 83)
        E_S = float(np.einsum("j,jab,jab->", w, resid, resid, optimize=True))

    # ---- slice-wise exact Gaussian-ReLU at every retained node (eqs 98-99) ----
    J = y.shape[0]
    r_nodes = np.zeros((J, d)); R_nodes = np.zeros((J, d, d))
    psd_clipped = 0.0
    for j in range(J):
        Sj = symmetrize(Sigma0 + y[j] * Sigma1)
        Sj, c = project_to_psd(Sj)                                     # constraint eq 93
        psd_clipped += c
        r_nodes[j], R_nodes[j] = _bulk_relu_update(mu0 + y[j] * mu1, Sj, bulk_relu)

    # ---- spike ReLU: keep positive nodes (eq 39), merge the rest (eqs 40-42) ----
    pos = y > 0.0
    p0 = float(w[~pos].sum())
    parts_p = [w[pos]]; parts_a = [y[pos]]
    parts_m = [r_nodes[pos]]; parts_S = [R_nodes[pos]]
    if p0 > 0.0:
        wz = w[~pos] / p0
        m0 = wz @ r_nodes[~pos]
        dm = r_nodes[~pos] - m0[None, :]
        S0 = (np.einsum("j,jab->ab", wz, R_nodes[~pos], optimize=True)
              + np.einsum("j,ja,jb->ab", wz, dm, dm, optimize=True))
        parts_p.insert(0, np.array([p0])); parts_a.insert(0, np.array([0.0]))
        parts_m.insert(0, m0[None, :]); parts_S.insert(0, symmetrize(S0)[None])
    p_new = np.concatenate(parts_p)
    new_state = AnalyticState(
        p=p_new / p_new.sum(),
        a=np.concatenate(parts_a),
        mu=np.concatenate(parts_m, axis=0),
        Sigma=np.concatenate(parts_S, axis=0))

    # ---- layer logs (checklist 9) ----
    if stats is not None:
        distortion = float((PQ * ((ymean - y[None, :]) ** 2 + vv)).sum() / W_tot)  # eq 134
        stats.setdefault("mass_lost", []).append(float(max(0.0, 1.0 - W_tot)))
        stats.setdefault("E_m", []).append(E_m)
        stats.setdefault("E_S", []).append(E_S)
        stats.setdefault("tr_R_m", []).append(float(np.trace(R_m)))
        stats.setdefault("scalar_distortion", []).append(distortion)
        stats.setdefault("psd_clipped", []).append(psd_clipped)
        stats.setdefault("num_cells", []).append(int(J))
        stats.setdefault("num_pos_nodes", []).append(int(pos.sum()))
        stats.setdefault("zero_atom_mass", []).append(p0)
    affine = AffineState(mix_w=p_in.copy(), mix_mean=mY, mix_var=sY2, edges=edges,
                         w=w, y=y, mu0=mu0, mu1=mu1, Sigma0=Sigma0, Sigma1=Sigma1)
    return new_state, affine


# --------------------------------------------------------------------------- #
# unconditional moments + readout  (paper eqs 126-127)
# --------------------------------------------------------------------------- #
def unconditional_mean(state: AnalyticState) -> np.ndarray:
    """Full mean ``E[X]`` (length ``n = d + 1``): coord 0 = ``sum_i p_i a_i``,
    coords 1.. = ``sum_i p_i m_i``."""
    out = np.empty(state.d + 1)
    out[SPIKE_COORD] = float(state.p @ state.a)
    out[1:] = state.p @ state.mu
    return out


def unconditional_mean_cov(state: AnalyticState) -> Tuple[np.ndarray, np.ndarray]:
    """Full mean and covariance of ``X`` by total (co)variance over the node law
    (paper eq 127; the input state's ``t2`` enters ``Var(A)``)."""
    p, a, mu, Sigma, t2 = state.p, state.a, state.mu, state.Sigma, state.t2
    d = state.d
    Abar = float(p @ a)
    Bbar = p @ mu
    da = a - Abar
    dB = mu - Bbar
    var_A = float(p @ (da * da + t2))
    cov_AB = (p * da) @ dB
    cov_B = symmetrize(np.einsum("i,iab->ab", p, Sigma, optimize=True)
                       + np.einsum("i,ia,ib->ab", p, dB, dB, optimize=True))
    n = d + 1
    mean = np.empty(n); mean[SPIKE_COORD] = Abar; mean[1:] = Bbar
    cov = np.zeros((n, n))
    cov[0, 0] = var_A; cov[0, 1:] = cov_AB; cov[1:, 0] = cov_AB; cov[1:, 1:] = cov_B
    return mean, symmetrize(cov)


# --------------------------------------------------------------------------- #
# full forward driver
# --------------------------------------------------------------------------- #
def run_analytic_kprop_k2(weights: List[Tuple[np.ndarray, Optional[np.ndarray]]],
                          input_dim: int, num_nodes: int = 40, *,
                          num_nodes_neg: Optional[int] = None,
                          num_nodes_pos: Optional[int] = None,
                          grid: str = "w2", bulk_relu: str = "exact",
                          cov_intercept: str = "mc", input_std: float = 1.0,
                          min_prob: float = _DEFAULT_MIN_PROB,
                          diagnostics: bool = False, collect: bool = False) -> dict:
    """Predict ``E[f(X)]`` for ``X ~ N(0, input_std^2 I)`` by ANALYTIC AFFINE K=2
    propagation along the coordinate spike ``e = e_1`` (Algorithm 7.2).

    ``weights`` are ``(W, b)`` float64 pairs in forward order (square ``n x n``
    hidden matrices with the spike baked in, each followed by ReLU, then the linear
    readout). ``num_nodes`` is THE hyperparameter: the total number of signed scalar
    quadrature cells per layer (allocated across the sign split proportionally to
    mixture mass unless ``num_nodes_neg``/``num_nodes_pos`` override). The bulk
    between layers is ONE affine family (2 vectors + 2 matrices), so the per-layer
    congruence cost is O(1), not O(num_nodes) -- see module docstring.

    Returns ``{"mean", "metadata", ...}``; ``collect`` adds per-layer affine states,
    the spike law by layer, the final state, and the raw stats lists.
    """
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2 (need >= 1 cell on each side of 0)")
    n_hidden = len(weights) - 1
    if n_hidden < 1:
        raise ValueError("need at least one hidden layer + a readout")
    d = input_dim - 1

    stats: dict = {}
    state = gaussian_input_state(d, input_std=input_std)
    affines = []; spike_by_layer = []
    for li in range(n_hidden):
        W, b = weights[li]
        M = np.asarray(W, dtype=np.float64)
        if M.shape != (input_dim, input_dim):
            raise ValueError(f"hidden layer {li} must be square ({input_dim},{input_dim}); "
                             f"got {M.shape}")
        state, aff = analytic_layer_update(
            state, M, b, num_nodes=num_nodes, num_nodes_neg=num_nodes_neg,
            num_nodes_pos=num_nodes_pos, grid=grid, bulk_relu=bulk_relu,
            cov_intercept=cov_intercept, min_prob=min_prob,
            diagnostics=diagnostics, stats=stats)
        if collect:
            affines.append(aff)
            spike_by_layer.append(dict(layer=li, p=state.p.copy(), a=state.a.copy()))

    W_ro, b_ro = weights[-1]
    W_ro = np.asarray(W_ro, dtype=np.float64)
    mean = W_ro @ unconditional_mean(state)                            # eq 126
    if b_ro is not None:
        mean = mean + np.asarray(b_ro, dtype=np.float64)

    def _mx(key):
        return float(max(stats.get(key, [0.0]) or [0.0]))

    out = {
        "mean": np.asarray(mean, dtype=np.float64).reshape(-1),
        "metadata": {
            "predictor": "analytic_kprop_k2", "K": 2,
            "num_nodes": int(num_nodes), "grid": grid,
            "bulk_relu": bulk_relu, "cov_intercept": cov_intercept,
            "n_hidden": n_hidden, "input_dim": int(input_dim),
            "output_dim": int(np.asarray(mean).reshape(-1).shape[0]),
            "max_mass_lost": _mx("mass_lost"),
            "max_E_m": _mx("E_m"),
            "max_tr_R_m": _mx("tr_R_m"),
            "max_scalar_distortion": _mx("scalar_distortion"),
            "total_psd_clipped": float(sum(stats.get("psd_clipped", [0.0]))),
            "num_pos_nodes_by_layer": list(stats.get("num_pos_nodes", [])),
        },
    }
    if diagnostics:
        out["metadata"]["E_S_by_layer"] = list(stats.get("E_S", []))
    if collect:
        out["affine_by_layer"] = affines
        out["spike_by_layer"] = spike_by_layer
        out["final_state"] = state
        out["stats"] = stats
    return out

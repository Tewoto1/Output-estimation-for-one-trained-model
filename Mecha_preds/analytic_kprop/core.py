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

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .._utils import _Phi, _phi, symmetrize, project_to_psd
from ..binned_kprop.binning import (
    _DEFAULT_MIN_PROB, _VAR_FLOOR, _xphi,
    Workers, resolve_workers, _run_bins,
    find_bin, lloyd_max_edges_mixture_split,
)
from ..binned_kprop.core import _bulk_relu_update

# Coordinate of the spike (e = e_1 -> index 0 in Python). The bulk is coords 1..n-1.
SPIKE_COORD = 0
_TINY_VAR_Y = 1e-14          # v_Y below this -> slope unidentifiable, use mu1 = 0 (paper 6.1)


def _torch_device(device):
    """Resolve the optional torch device knob. ``None``/``"numpy"`` -> pure numpy.
    A requested-but-unusable device (torch missing / CUDA unavailable) falls back to
    numpy with a one-line warning rather than failing -- the numpy path is always
    valid; torch only ACCELERATES the (m, d, d)-weighted congruences and batched
    matmuls (see ``_weighted_sigma_congruence`` / ``percell_bulk_moments``)."""
    if device in (None, "", "numpy"):
        return None
    try:
        import torch
    except ModuleNotFoundError:
        print(f"analytic_kprop: device={device!r} requested but torch is not installed "
              "-> numpy fallback")
        return None
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("analytic_kprop: CUDA requested but unavailable -> numpy fallback")
        return None
    return dev


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
                      num_nodes_neg: Optional[int], num_nodes_pos: Optional[int],
                      *, max_nodes_neg: Optional[int] = None) -> Tuple[int, int]:
    """Allocate the sign-split cell counts. The POSITIVE side always gets the FULL
    ``num_nodes`` budget; the negative side is MASS-ADAPTIVE.

    ReLU keeps only the positive cells as atoms (every nonpositive cell merges into
    the zero atom), so the positive count is the only resolution that survives a
    layer. The old rule split a fixed total ``num_nodes`` proportionally to the sign
    masses, which silently DILUTED the surviving budget (50% negative mass -> only
    ``num_nodes/2`` positive cells). Now:

        n_pos = num_nodes                                (or ``num_nodes_pos``)
        n_neg = ceil(n_pos * neg_mass / pos_mass)        (or ``num_nodes_neg``)

    -- the negative side gets the same mass-per-cell as the positive side (e.g.
    budget 20 at 50% negative -> 20 negative cells, 40 total; at 25% negative -> 7,
    27 total), it collapses into the zero atom at the ReLU, and the surviving count
    never dilutes. ``max_nodes_neg`` (default ``8 * n_pos``) caps the adaptive count
    in the near-dead regime (``neg_mass -> 1``). Both sides are always >= 1."""
    n_pos = max(1, int(num_nodes if num_nodes_pos is None else num_nodes_pos))
    if num_nodes_neg is not None:
        return max(1, int(num_nodes_neg)), n_pos
    pos_mass = max(1.0 - float(neg_mass), 1e-12)
    cap = (8 * n_pos) if max_nodes_neg is None else int(max_nodes_neg)
    n_neg = int(np.clip(np.ceil(n_pos * float(neg_mass) / pos_mass), 1, max(1, cap)))
    return n_neg, n_pos


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
    cell (paper eq 73).

    Fully vectorized over the (component, cell) grid -- the same formulas as
    ``normal_interval_stats`` (which is vectorized over cells only), broadcast as
    (m_stoch, J) arrays; identical numbers, no per-component python loop."""
    m = p.shape[0]
    low, high = edges[:-1], edges[1:]
    J = low.shape[0]
    Q = np.zeros((m, J)); ymean = np.zeros((m, J))
    delta = np.zeros((m, J)); vv = np.zeros((m, J))
    stoch = (p > 0.0) & (sY2 >= min_prob)
    det = (p > 0.0) & ~stoch

    if stoch.any():
        mean = mY[stoch, None]                                  # (ms, 1)
        var = sY2[stoch, None]
        sig = np.sqrt(np.maximum(var, _VAR_FLOOR))
        a = (low[None, :] - mean) / sig                         # (ms, J)
        b = (high[None, :] - mean) / sig
        phi_a, phi_b = _phi(a), _phi(b)
        Qs = _Phi(b) - _Phi(a)
        ok = Qs > min_prob
        Qsafe = np.where(ok, Qs, 1.0)
        tau1 = np.where(ok, sig * (phi_a - phi_b) / Qsafe, 0.0)
        tau2 = np.where(ok, var * (1.0 + (_xphi(a, phi_a) - _xphi(b, phi_b)) / Qsafe), 0.0)
        yv = np.clip(tau2 - tau1 * tau1, 0.0, None)
        Q[stoch] = np.where(ok, Qs, 0.0)
        ymean[stoch] = np.where(ok, mean + tau1, 0.0)
        delta[stoch] = tau1
        vv[stoch] = np.where(ok, yv, 0.0)

    for i in np.nonzero(det)[0]:                                # deterministic branch (rare)
        j = find_bin(edges, float(mY[i]))
        Q[i, j] = 1.0; ymean[i, j] = mY[i]
    return Q, ymean, delta, vv, stoch


def _weighted_sigma_congruence(V, Sigma, coeff_list, dev=None):
    """``[V (sum_i c_i S_i) V^T  for c in coeff_list]`` -- the only O(d^3)+O(m d^2)
    work in the affine fit. numpy by default; with a torch ``dev`` the Sigma stack is
    uploaded ONCE and the weighted reduction + congruence run on the device (the win
    at large d, especially on GPU)."""
    if dev is None:
        return [V @ np.einsum("i,iab->ab", c, Sigma, optimize=True) @ V.T
                for c in coeff_list]
    import torch
    Sig_t = torch.as_tensor(Sigma, device=dev)
    V_t = torch.as_tensor(V, device=dev)
    out = []
    for c in coeff_list:
        c_t = torch.as_tensor(np.ascontiguousarray(c), device=dev)
        Sagg = torch.tensordot(c_t, Sig_t, dims=1)                       # (d, d)
        out.append((V_t @ Sagg @ V_t.T).cpu().numpy())
    return out


def _covariance_sums(p, Q, delta, vv, sY2, stoch, y, w, mhat,
                     Sigma, t2, mC, g, u, V, dev=None):
    """The two grid-weighted covariance sums the affine fit needs (paper eq 87),

        T0 = sum_j w_j Shat_j,      T1 = sum_j w_j y_j Shat_j,

    WITHOUT forming any per-cell (or per-pair) d x d matrix. Expansion of
    ``Shat_j = sum_i eta_{i|j} [S_{i->j} + (m_{i->j} - mhat_j)(...)^T]`` (eqs 68/72)
    over components:

      * ``S_C,i``-part: scalar weights ``sum_j Q_ij c_j`` -> aggregate ``S_i``/``t2_i``
        first, then ONE congruence ``V (sum_i . S_i) V^T`` per sum  (the complexity
        win over the binned companion's per-bin congruences; torch-offloadable via
        ``dev``);
      * rank-1 ``g_i g_i^T`` part: scalar weights ``sum_j Q_ij c_j (v_ij - s2_i)/s2_i^2``;
      * between-mean part: ``m_{i->j} = m_C,i + u_reg,i delta_ij`` is affine in the
        scalar ``delta``, so second moments reduce to the scalar sums
        ``sum_j Q_ij c_j delta^k`` (k = 0,1,2), minus ``sum_j w_j c_j mhat_j mhat_j^T``.

    Returns raw (unnormalized-by-total-mass) ``T0, T1``.
    """
    s2 = np.where(stoch, sY2, 1.0)
    u_reg = np.where(stoch[:, None], g / s2[:, None], 0.0)              # (m, d)
    gcoef = np.where(stoch[:, None], 1.0, 0.0) * g                      # g zeroed if det

    scales = (np.ones_like(y), y)
    cqs = [(Q * c[None, :]).sum(axis=1) for c in scales]                # sum_j Q c_j
    congr = _weighted_sigma_congruence(V, Sigma, [p * cq for cq in cqs], dev)

    out = []
    for cell_scale, cq, VSV in zip(scales, cqs, congr):
        Qc = Q * cell_scale[None, :]                                    # (m, J)
        # S_C part: V (sum_i p_i cq_i S_i) V^T + (sum_i p_i cq_i t2_i) u u^T
        SC = VSV + float(np.sum(p * cq * t2)) * np.outer(u, u)
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


def percell_bulk_moments(p, Q, delta, vv, sY2, stoch, w, Sigma, t2, mC, g, u, V,
                         dev=None):
    """REFERENCE / diagnostics path: the per-cell merged moments ``(mhat_j, Shat_j)``
    (paper eqs 71-72), forming J d x d matrices (J congruences via the aggregated
    ``A_j = sum_i eta_{i|j} S_i``; batched on torch ``dev`` when given -- this is the
    costliest diagnostics step at large d). Used by the selftest to validate
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
    if dev is not None:
        import torch
        Sig_t = torch.as_tensor(Sigma, device=dev)
        V_t = torch.as_tensor(V, device=dev)
        eta_t = torch.as_tensor(eta, device=dev)
        Aj_t = torch.tensordot(eta_t.T, Sig_t.reshape(m, -1), dims=1).reshape(J, *Sigma.shape[1:])
        Shat = (V_t @ Aj_t @ V_t.T).cpu().numpy()                        # (J, d, d) on device
    else:
        Aj = np.einsum("ij,iab->jab", eta, Sigma, optimize=True)         # (J, d, d)
        Shat = V @ Aj @ V.T                                              # batched congruence (BLAS)
    t2j = eta.T @ t2                                                     # (J,)
    Shat += t2j[:, None, None] * np.outer(u, u)[None, :, :]
    hv = np.where(stoch[:, None], (vv - s2[:, None]) / (s2 * s2)[:, None], 0.0)  # (m, J)
    gz = np.where(stoch[:, None], g, 0.0)
    Shat += np.einsum("ij,ia,ib->jab", eta * hv, gz, gz, optimize=True)
    # between-mean outer products
    dm = m_ij_0[:, None, :] + u_reg[:, None, :] * delta[:, :, None] - mhat[None, :, :]  # (m, J, d)
    Shat += np.einsum("ij,ija,ijb->jab", eta, dm, dm, optimize=True)
    return mhat, 0.5 * (Shat + np.swapaxes(Shat, 1, 2))


def _chol_ok(S: np.ndarray) -> bool:
    """Cheap PSD test: Cholesky succeeds (strictly PD up to roundoff). ~3-6x faster
    than ``eigh``; a PSD-singular matrix may fail and simply falls through to the
    exact ``project_to_psd`` path, which then clips nothing."""
    try:
        np.linalg.cholesky(S)
        return True
    except np.linalg.LinAlgError:
        return False


def analytic_layer_update(state: AnalyticState, M: np.ndarray, b: Optional[np.ndarray],
                          *, num_nodes: int, num_nodes_neg: Optional[int] = None,
                          num_nodes_pos: Optional[int] = None,
                          max_nodes_neg: Optional[int] = None, grid: str = "w2",
                          bulk_relu: str = "exact", cov_intercept: str = "mc",
                          min_prob: float = _DEFAULT_MIN_PROB,
                          diagnostics: bool = False,
                          workers: Workers = None, dev=None,
                          stats: Optional[dict] = None
                          ) -> Tuple[AnalyticState, AffineState]:
    """One hidden layer of Algorithm 7.2: linear + Bayesian reconditioning on the new
    spike coordinate + affine re-projection + slice-wise exact Gaussian-ReLU + exact
    zero-atom merge. Returns ``(new_post_relu_state, affine_state)``.

    ``cov_intercept``: "mc" (default) adds the moment-conservative correction
    ``Sigma0 += R_m`` (paper eq 90), which preserves the unconditional bulk
    covariance; "ls" keeps the literal least-squares intercept (eq 87).

    ``workers``: thread count for the independent per-node Gaussian-ReLU updates
    (the dominant cost with the exact bivariate backend -- scipy's special-function
    ufuncs and LAPACK release the GIL, exactly like the binned companion's per-bin
    loops). ``None``/``"auto"`` auto-resolves per machine (``resolve_workers``,
    shared with binned; env override ``$BINNED_KPROP_WORKERS``); pass ``1`` for
    serial. Results are identical to serial regardless of the count.

    ``dev``: resolved torch device (see ``_torch_device``) or ``None``; offloads the
    Sigma-stack reductions + congruences (and the diagnostics batch congruence)."""
    if cov_intercept not in ("mc", "ls"):
        raise ValueError(f"cov_intercept must be 'mc' or 'ls', got {cov_intercept!r}")
    d = state.d
    p_in = state.p
    _tick = time.perf_counter
    _t = _tick()

    def _phase(name, t0):
        if stats is not None:
            stats.setdefault(f"t_{name}", []).append(_tick() - t0)
        return _tick()

    gamma, r, u, V, beta, eta_b = _layer_block(M, b, d)

    # ---- components of the scalar mixture + conditional bulk (eqs 60-61, 51) ----
    mY, sY2, mC, g = _component_params(state, gamma, r, u, V, beta, eta_b)
    _t = _phase("params", _t)

    # ---- signed cell grid with an edge at 0 (checklist 2) ----
    n_neg, n_pos = split_node_budget(num_nodes, negative_mass(p_in, mY, sY2),
                                     num_nodes_neg, num_nodes_pos,
                                     max_nodes_neg=max_nodes_neg)
    edges = make_cells(p_in, mY, sY2, n_neg, n_pos, grid=grid)
    _t = _phase("grid", _t)

    # ---- per-(component, cell) closed-form stats (eqs 63-66) ----
    Q, ymean, delta, vv, stoch = _pair_stats(p_in, mY, sY2, edges, min_prob=min_prob)
    _t = _phase("pairs", _t)

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

    _t = _phase("cells", _t)
    T0, T1 = _covariance_sums(p_in, Q, delta, vv, sY2, stoch, y, w_raw, mhat,
                              state.Sigma, state.t2, mC, g, u, V, dev)
    T0 /= W_tot; T1 /= W_tot
    if vY > _TINY_VAR_Y:
        Sigma1 = symmetrize((T1 - ybar * T0) / vY)
    else:
        Sigma1 = np.zeros((d, d))
    Sigma0 = symmetrize(T0 - ybar * Sigma1)                            # LS intercept (eq 87)
    if cov_intercept == "mc":
        Sigma0 = symmetrize(Sigma0 + R_m)                              # eq 90

    _t = _phase("fit", _t)
    E_S = float("nan")
    if diagnostics:
        _mh_ref, Shat = percell_bulk_moments(p_in, Q, delta, vv, sY2, stoch, w_raw,
                                             state.Sigma, state.t2, mC, g, u, V, dev)
        resid = Shat - Sigma0[None] - y[:, None, None] * Sigma1[None] \
            + (R_m[None] if cov_intercept == "mc" else 0.0)            # E_S is vs the LS fit (eq 83)
        E_S = float(np.einsum("j,jab,jab->", w, resid, resid, optimize=True))
    _t = _phase("diag", _t)

    # ---- slice-wise exact Gaussian-ReLU at every retained node (eqs 98-99) ----
    # PSD screen first: Sigma(y) = Sigma0 + y Sigma1 is AFFINE in y, so any interior
    # node is a convex combination of the two EXTREME nodes -- if those are PSD, all
    # are (constraint eq 93 holds grid-wide). Two Cholesky factorizations thus replace
    # J eigendecompositions in the common case; only Cholesky-failing nodes eigen-clip.
    J = y.shape[0]
    all_psd = (_chol_ok(symmetrize(Sigma0 + float(y.min()) * Sigma1))
               and _chol_ok(symmetrize(Sigma0 + float(y.max()) * Sigma1)))
    r_nodes = np.zeros((J, d)); R_nodes = np.zeros((J, d, d))

    def _node(j):
        """Independent per-node work (threads write disjoint rows j; the clip mass is
        RETURNED and summed by the caller -- a shared ``+=`` would race)."""
        Sj = symmetrize(Sigma0 + y[j] * Sigma1)
        c = 0.0
        if not (all_psd or _chol_ok(Sj)):
            Sj, c = project_to_psd(Sj)                                 # eigen-clip + log (eq 93)
        r_nodes[j], R_nodes[j] = _bulk_relu_update(mu0 + y[j] * mu1, Sj, bulk_relu)
        return c

    psd_clipped = float(sum(_run_bins(J, _node, workers)))
    _t = _phase("relu", _t)

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

    _t = _phase("merge", _t)
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
                          max_nodes_neg: Optional[int] = None,
                          grid: str = "w2", bulk_relu: str = "exact",
                          cov_intercept: str = "mc", input_std: float = 1.0,
                          min_prob: float = _DEFAULT_MIN_PROB,
                          fit: str = "pre", atom: str = "exact",
                          diagnostics: bool = False,
                          workers: Workers = None, device: Optional[str] = None,
                          collect: bool = False) -> dict:
    """Predict ``E[f(X)]`` for ``X ~ N(0, input_std^2 I)`` by ANALYTIC AFFINE K=2
    propagation along the coordinate spike ``e = e_1`` (Algorithm 7.2).

    ``weights`` are ``(W, b)`` float64 pairs in forward order (square ``n x n``
    hidden matrices with the spike baked in, each followed by ReLU, then the linear
    readout). ``num_nodes`` is THE hyperparameter: the POSITIVE-side scalar cell
    budget per layer. ReLU keeps only the positive cells as atoms (nonpositive cells
    merge exactly into the zero atom), so the positive side always gets the FULL
    budget; the negative side is MASS-ADAPTIVE -- ``ceil(num_nodes * neg/pos mass)``
    cells, same mass-per-cell as the positive side, capped at ``max_nodes_neg``
    (default ``8 * num_nodes``). E.g. budget 20 at 50% negative -> 40 cells total,
    20 surviving; at 25% negative -> 27 total, 20 surviving: the resolution that
    survives a layer never dilutes. (The old rule split a FIXED total across the
    signs, silently halving the surviving budget at 50% negative mass.) Override
    with ``num_nodes_neg``/``num_nodes_pos``. The bulk between layers is ONE affine
    family (2 vectors + 2 matrices), so the per-layer congruence cost is O(1), not
    O(num_nodes) -- see module docstring.

    ``workers`` threads the per-node Gaussian-ReLU loop exactly like the binned
    companion (``None``/``"auto"`` = auto-parallel per machine, ``1`` = serial;
    results identical either way; env override ``$BINNED_KPROP_WORKERS``).
    ``device`` (e.g. ``"cuda"``) offloads the Sigma-stack reductions/congruences
    (and the ``diagnostics`` batch congruence) to torch; falls back to numpy with a
    warning if torch/CUDA is unavailable. The exact bivariate ReLU kernel stays on
    CPU (scipy special functions) -- threading is what accelerates it.

    ``fit`` selects WHERE the affine projection is applied: "pre" (paper Algorithm
    7.2 -- fit the reconditioned pre-activation, keep exact nonlinear post-ReLU
    node moments) or "post" (fit the POST-ReLU slice functions; the linear step
    then just transforms (m0, m1, W0, W1) -- no per-node matrix state, memory
    O(d^2); see the PostAffineState section). ``atom`` ("post" only): "exact"
    keeps the zero atom (the merge of ALL negative cells) as a separate exact
    component; "fit" folds it into the affine family -- a toggle to test whether
    the linearity assumption may include the atom.

    Returns ``{"mean", "metadata", ...}``; ``collect`` adds per-layer affine states,
    the spike law by layer, the final state, and the raw stats lists.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1 (the positive-side cell budget; the "
                         "negative side gets its own mass-adaptive cells)")
    if fit not in ("pre", "post"):
        raise ValueError(f"fit must be 'pre' or 'post', got {fit!r}")
    n_hidden = len(weights) - 1
    if n_hidden < 1:
        raise ValueError("need at least one hidden layer + a readout")
    d = input_dim - 1
    dev = _torch_device(device)

    stats: dict = {}
    affines = []; spike_by_layer = []
    state = (gaussian_input_state_post(d, input_std=input_std) if fit == "post"
             else gaussian_input_state(d, input_std=input_std))
    for li in range(n_hidden):
        W, b = weights[li]
        M = np.asarray(W, dtype=np.float64)
        if M.shape != (input_dim, input_dim):
            raise ValueError(f"hidden layer {li} must be square ({input_dim},{input_dim}); "
                             f"got {M.shape}")
        if fit == "post":
            state = analytic_layer_update_post(
                state, M, b, num_nodes=num_nodes, num_nodes_neg=num_nodes_neg,
                num_nodes_pos=num_nodes_pos, max_nodes_neg=max_nodes_neg,
                grid=grid, bulk_relu=bulk_relu,
                cov_intercept=cov_intercept, atom=atom, min_prob=min_prob,
                workers=workers, dev=dev, stats=stats)
            aff = state                                     # the state IS the family
        else:
            state, aff = analytic_layer_update(
                state, M, b, num_nodes=num_nodes, num_nodes_neg=num_nodes_neg,
                num_nodes_pos=num_nodes_pos, max_nodes_neg=max_nodes_neg,
                grid=grid, bulk_relu=bulk_relu,
                cov_intercept=cov_intercept, min_prob=min_prob,
                diagnostics=diagnostics, workers=workers, dev=dev, stats=stats)
        if collect:
            affines.append(aff)
            spike_by_layer.append(dict(layer=li, p=state.p.copy(), a=state.a.copy()))

    W_ro, b_ro = weights[-1]
    W_ro = np.asarray(W_ro, dtype=np.float64)
    full_mean = (unconditional_mean_post(state) if fit == "post"
                 else unconditional_mean(state))
    mean = W_ro @ full_mean                                            # eq 126
    if b_ro is not None:
        mean = mean + np.asarray(b_ro, dtype=np.float64)

    def _mx(key):
        return float(max(stats.get(key, [0.0]) or [0.0]))

    out = {
        "mean": np.asarray(mean, dtype=np.float64).reshape(-1),
        "metadata": {
            "predictor": "analytic_kprop_k2", "K": 2,
            "num_nodes": int(num_nodes), "grid": grid,
            "fit": fit, "atom": (atom if fit == "post" else None),
            "bulk_relu": bulk_relu, "cov_intercept": cov_intercept,
            "workers": resolve_workers(workers),
            "device": ("numpy" if dev is None else str(dev)),
            "n_hidden": n_hidden, "input_dim": int(input_dim),
            "output_dim": int(np.asarray(mean).reshape(-1).shape[0]),
            "max_mass_lost": _mx("mass_lost"),
            "max_E_m": _mx("E_m"),
            "max_tr_R_m": _mx("tr_R_m"),
            "max_scalar_distortion": _mx("scalar_distortion"),
            "total_psd_clipped": float(sum(stats.get("psd_clipped", [0.0]))),
            "num_pos_nodes_by_layer": list(stats.get("num_pos_nodes", [])),
            "num_cells_by_layer": list(stats.get("num_cells", [])),
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


# =========================================================================== #
# POST-ACTIVATION AFFINE VARIANT (fit="post")
#
# The affine family is fitted to the POST-ReLU slice functions instead of the
# reconditioned pre-activation:
#
#     B | A = a  ~~>  N(m0 + m1 a,  W0 + W1 a)          (positive branch)
#
# with the ZERO ATOM either kept as a separate EXACT component (atom="exact",
# default -- the atom is the merge of ALL negative cells, so assuming it obeys
# the same linearity is a genuinely different hypothesis) or folded into the
# fit (atom="fit"; toggle to test which assumption is better).
#
# Why: linear maps preserve affinity EXACTLY, so the linear step reduces to
# transforming the four family objects (m0, m1, W0, W1) -- two congruences
# V W V^T + four matvecs -- instead of transforming per-node moments. The state
# carries NO (num_nodes, d, d) stack (memory O(d^2), not O(num_nodes d^2)); the
# exact-cell reconditioning survives because every within-component quantity is
# affine in the node value a_i, so all cell-merged moments live in a <=7-vector
# basis with closed-form scalar coefficients. The ReLU step is unchanged (exact
# Gaussian integrals per retained node, O(num_nodes d^2) special functions) and
# its inputs Shat_j are mixture covariances -- PSD by construction, so the
# affine-family PSD obstruction never reaches the kernel (a cheap Cholesky
# guard remains for roundoff).
#
# Approximation ordering vs fit="pre": the paper (checklist 7) deliberately
# avoids fitting the positive post-ReLU functions because r(a) = m_rho(...) is
# nonlinear in a; this variant makes exactly that extra projection and is
# therefore a (cheaper, lighter) DIFFERENT closure -- compare empirically.
# =========================================================================== #
@dataclass
class PostAffineState:
    """Post-ReLU state for fit="post": node law (p, a) + affine bulk family
    (m0, m1, W0, W1) on the positive branch + optional EXACT zero atom.

    p, a:    (K,) node probabilities / spike values (node 0 = atom when exact)
    m0, m1:  (d,)  affine conditional mean  E[B|A=a] ~ m0 + m1 a   (a > 0)
    W0, W1:  (d,d) affine conditional cov   Cov[B|A=a] ~ W0 + W1 a (a > 0)
    atom_m/atom_S: exact zero-atom bulk moments (None -> atom uses the family)
    t2:      (K,) within-component spike variance (input layer only)
    """
    p: np.ndarray
    a: np.ndarray
    m0: np.ndarray
    m1: np.ndarray
    W0: np.ndarray
    W1: np.ndarray
    atom_m: Optional[np.ndarray] = None
    atom_S: Optional[np.ndarray] = None
    t2: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.t2 is None:
            self.t2 = np.zeros_like(self.a)

    @property
    def num_nodes(self) -> int:
        return self.p.shape[0]

    @property
    def d(self) -> int:
        return self.m0.shape[0]


def gaussian_input_state_post(d: int, *, input_std: float = 1.0) -> PostAffineState:
    """Exact input state ``X ~ N(0, s^2 I)`` in post-affine form: one component
    (a=0, t2=s^2), constant family m(a)=0, S(a)=s^2 I, no atom."""
    s2 = float(input_std) ** 2
    return PostAffineState(p=np.array([1.0]), a=np.array([0.0]),
                           m0=np.zeros(d), m1=np.zeros(d),
                           W0=s2 * np.eye(d), W1=np.zeros((d, d)),
                           t2=np.array([s2]))


def _congr(V: np.ndarray, S: np.ndarray, dev=None) -> np.ndarray:
    """V S V^T, torch-offloadable."""
    if dev is None:
        return V @ S @ V.T
    import torch
    V_t = torch.as_tensor(V, device=dev)
    return (V_t @ torch.as_tensor(S, device=dev) @ V_t.T).cpu().numpy()


def analytic_layer_update_post(state: PostAffineState, M: np.ndarray,
                               b: Optional[np.ndarray], *, num_nodes: int,
                               num_nodes_neg: Optional[int] = None,
                               num_nodes_pos: Optional[int] = None,
                               max_nodes_neg: Optional[int] = None,
                               grid: str = "w2", bulk_relu: str = "exact",
                               cov_intercept: str = "mc", atom: str = "exact",
                               min_prob: float = _DEFAULT_MIN_PROB,
                               workers: Workers = None, dev=None,
                               stats: Optional[dict] = None
                               ) -> PostAffineState:
    """One hidden layer of the POST-fit variant (see the section banner):
    transform the affine family through the linear map (2-3 congruences), exact
    cell reconditioning in a 7-vector basis, exact Gaussian-ReLU per node with
    streaming covariance accumulators (no (J, d, d) storage), zero-atom merge,
    then the post-ReLU weighted-LS fit of (m0, m1, W0, W1)."""
    if cov_intercept not in ("mc", "ls"):
        raise ValueError(f"cov_intercept must be 'mc' or 'ls', got {cov_intercept!r}")
    if atom not in ("exact", "fit"):
        raise ValueError(f"atom must be 'exact' or 'fit', got {atom!r}")
    d = state.d
    p_in, a_in, t2 = state.p, state.a, state.t2
    K = p_in.shape[0]
    has_atom = state.atom_m is not None
    atom_idx = 0 if has_atom else -1                     # exact-atom component index
    _tick = time.perf_counter
    _t = _tick()

    def _phase(name, t0):
        if stats is not None:
            stats.setdefault(f"t_{name}", []).append(_tick() - t0)
        return _tick()

    gamma, r, u, V, beta, eta_b = _layer_block(M, b, d)

    # ---- transform the family: the whole "linear step" (2-3 congruences) ----
    c0 = V @ state.m0 + eta_b
    c1 = u + V @ state.m1
    g0 = V @ (state.W0 @ r)
    g1 = V @ (state.W1 @ r)
    s0 = float(r @ (state.W0 @ r))
    s1 = float(r @ (state.W1 @ r))
    G0 = _congr(V, state.W0, dev)
    G1 = _congr(V, state.W1, dev)
    if has_atom:
        mC_at = V @ state.atom_m + eta_b
        g_at = V @ (state.atom_S @ r)
        s_at = float(r @ (state.atom_S @ r))
        G_at = _congr(V, state.atom_S, dev)
    else:
        mC_at = np.zeros(d); g_at = np.zeros(d); s_at = 0.0; G_at = None

    nonat = np.ones(K, dtype=bool)
    if has_atom:
        nonat[atom_idx] = False
    mY = np.where(nonat, beta + float(r @ state.m0) + (gamma + float(r @ state.m1)) * a_in,
                  beta + float(r @ (state.atom_m if has_atom else state.m0)))
    sY2 = np.where(nonat, s0 + s1 * a_in + gamma * gamma * t2, s_at)
    sY2 = np.clip(sY2, 0.0, None)
    _t = _phase("params", _t)

    # ---- grid + closed-form pair stats (same machinery as fit="pre") ----
    n_neg, n_pos = split_node_budget(num_nodes, negative_mass(p_in, mY, sY2),
                                     num_nodes_neg, num_nodes_pos,
                                     max_nodes_neg=max_nodes_neg)
    edges = make_cells(p_in, mY, sY2, n_neg, n_pos, grid=grid)
    _t = _phase("grid", _t)
    Q, ymean, delta, vv, stoch = _pair_stats(p_in, mY, sY2, edges, min_prob=min_prob)
    _t = _phase("pairs", _t)

    w_raw = p_in @ Q
    W_tot = float(w_raw.sum())
    if W_tot <= 0.0:
        raise RuntimeError("all scalar mass vanished in analytic_layer_update_post")
    retained = w_raw > min_prob
    Q = Q[:, retained]; ymean = ymean[:, retained]
    delta = delta[:, retained]; vv = vv[:, retained]
    w_raw = w_raw[retained]
    J = w_raw.shape[0]
    PQ = p_in[:, None] * Q
    eta = PQ / w_raw[None, :]
    y = (PQ * ymean).sum(axis=0) / w_raw
    w = w_raw / W_tot

    # ---- everything in the 7-vector basis (all component moments are affine) ----
    # columns: 0 c0 | 1 c1 | 2 mC_atom | 3 g0/s^2 dir | 4 g1 dir | 5 g_atom dir | 6 u
    Bmat = np.stack([c0, c1, mC_at, g0, g1, g_at, u], axis=1)          # (d, 7)
    s2 = np.where(stoch, sY2, 1.0)
    inv_s2 = np.where(stoch, 1.0 / s2, 0.0)
    wc = np.zeros((K, J, 7))
    dl = delta * inv_s2[:, None]                                       # delta / s^2
    na = nonat
    wc[na, :, 0] = 1.0
    wc[na, :, 1] = a_in[na, None]
    wc[na, :, 3] = dl[na]
    wc[na, :, 4] = a_in[na, None] * dl[na]
    wc[na, :, 6] = (gamma * t2)[na, None] * dl[na]
    if has_atom:
        wc[atom_idx, :, 2] = 1.0
        wc[atom_idx, :, 5] = dl[atom_idx]
    # g-direction coords (for the rank-1 v-correction g_i g_i^T (v - s^2)/s^4)
    gc = np.zeros((K, 7))
    gc[na, 3] = 1.0; gc[na, 4] = a_in[na]; gc[na, 6] = (gamma * t2)[na]
    if has_atom:
        gc[atom_idx, 5] = 1.0

    wbar = np.einsum("ij,ijk->jk", eta, wc, optimize=True)             # (J, 7)
    dw = wc - wbar[None, :, :]
    Cm = np.einsum("ij,ijk,ijl->jkl", eta, dw, dw, optimize=True)      # between-mean part
    hv = np.where(stoch[:, None], (vv - s2[:, None]) * (inv_s2 * inv_s2)[:, None], 0.0)
    Cg = np.einsum("ij,ik,il->jkl", eta * hv, gc, gc, optimize=True)   # v-correction part
    C = Cm + Cg
    # S_C scalar coefficients: G0 + a G1 (+ t2 u u^T) per non-atom comp, G_at for atom
    alpha = eta[na].sum(axis=0)
    beta_c = (eta[na] * a_in[na, None]).sum(axis=0)
    omega = eta[atom_idx] if has_atom else np.zeros(J)
    C[:, 6, 6] += (eta * t2[:, None]).sum(axis=0)                      # t2 u u^T
    mhat = wbar @ Bmat.T                                               # (J, d)
    _t = _phase("cells", _t)

    # ---- exact Gaussian-ReLU per retained node, STREAMING accumulators ----
    # Slot-threaded: worker slot k owns nodes j = k (mod n_slots) and accumulates
    # into ITS OWN (d, d) partials -- no per-chunk synchronization bubbles, no
    # (J, d, d) storage (memory O(n_slots d^2)). The final slot-sum runs in fixed
    # order, so results are deterministic for a given worker count (fp-identical
    # to serial only at workers=1 -- the addition grouping differs otherwise).
    pos = y > 0.0
    r_nodes = np.zeros((J, d))
    n_slots = max(1, min(resolve_workers(workers), J))
    AR_pos_s = np.zeros((n_slots, d, d)); AyR_pos_s = np.zeros((n_slots, d, d))
    AR_neg_s = np.zeros((n_slots, d, d))
    R2_s = np.zeros(n_slots); clip_s = np.zeros(n_slots)

    def _slot(k):
        for j in range(k, J, n_slots):
            Sj = alpha[j] * G0 + beta_c[j] * G1 + Bmat @ C[j] @ Bmat.T
            if G_at is not None:
                Sj += omega[j] * G_at
            Sj = symmetrize(Sj)
            # Shat_j is a MIXTURE covariance -> PSD by construction; only roundoff
            # can violate it, and the exact kernel is defensive (rho/variance
            # clipping), so a diagonal floor replaces a per-node factorization.
            dg = np.einsum("aa->a", Sj)
            neg = float(dg.min())
            if neg < 0.0:
                clip_s[k] += -neg
                np.clip(dg, 0.0, None, out=dg)
            rj, Rj = _bulk_relu_update(mhat[j], Sj, bulk_relu)
            r_nodes[j] = rj
            if pos[j]:
                AR_pos_s[k] += w[j] * Rj
                AyR_pos_s[k] += (w[j] * y[j]) * Rj
                R2_s[k] += w[j] * float(np.einsum("ab,ab->", Rj, Rj))
            else:
                AR_neg_s[k] += w[j] * Rj

    _run_bins(n_slots, _slot, workers)
    AR_pos = AR_pos_s.sum(axis=0); AyR_pos = AyR_pos_s.sum(axis=0)
    AR_neg = AR_neg_s.sum(axis=0)
    R2_pos = float(R2_s.sum()); psd_clipped = float(clip_s.sum())
    _t = _phase("relu", _t)

    # ---- zero atom: exact merge of all nonpositive nodes (eqs 40-42) ----
    p0 = float(w[~pos].sum())
    if p0 > 0.0:
        wz = w[~pos] / p0
        m_at_new = wz @ r_nodes[~pos]
        dm = r_nodes[~pos] - m_at_new[None, :]
        S_at_new = symmetrize(AR_neg / p0 + np.einsum("j,ja,jb->ab", wz, dm, dm,
                                                      optimize=True))
    else:
        m_at_new = None; S_at_new = None
    _t = _phase("merge", _t)

    # ---- POST-ReLU affine fit (this variant's defining projection) ----
    # data: positive nodes (y_j, r_j, R_j-accumulators); atom="fit" adds the merged
    # atom as a data point at a=0 (folding the all-negative-cells merge into the
    # linearity hypothesis); atom="exact" keeps it out as an exact component.
    fit_w = list(w[pos]); fit_a = list(y[pos])
    fit_r = [r_nodes[pos]]
    AR_f = AR_pos.copy(); AyR_f = AyR_pos.copy()
    if atom == "fit" and p0 > 0.0:
        fit_w.append(p0); fit_a.append(0.0)
        fit_r.append(m_at_new[None, :])
        AR_f += p0 * S_at_new
    fit_w = np.asarray(fit_w); fit_a = np.asarray(fit_a)
    fit_r = np.concatenate(fit_r, axis=0)
    mass_f = float(fit_w.sum())
    E_m_post = float("nan"); E_S_post = float("nan")
    if mass_f > 0.0 and fit_w.size > 0:
        fw = fit_w / mass_f
        AR_f /= mass_f; AyR_f /= mass_f
        abar = float(fw @ fit_a)
        va = float(fw @ (fit_a - abar) ** 2)
        if va > _TINY_VAR_Y:
            m1_new = ((fw * (fit_a - abar))[:, None] * fit_r).sum(axis=0) / va
            W1_new = symmetrize((AyR_f - abar * AR_f) / va)
        else:
            m1_new = np.zeros(d); W1_new = np.zeros((d, d))
        m0_new = (fw[:, None] * fit_r).sum(axis=0) - m1_new * abar
        W0_new = symmetrize(AR_f - abar * W1_new)
        e_m = fit_r - m0_new[None, :] - fit_a[:, None] * m1_new[None, :]
        E_m_post = float(fw @ (e_m * e_m).sum(axis=1))
        R_m = np.einsum("j,ja,jb->ab", fw, e_m, e_m, optimize=True)
        if cov_intercept == "mc":
            W0_new = symmetrize(W0_new + R_m)              # conserve total covariance
        # streaming E_S vs the LS fit (diagnostic; excludes the atom-in-fit R term)
        W0_ls = W0_new - (R_m if cov_intercept == "mc" else 0.0)
        R2 = R2_pos / mass_f
        if atom == "fit" and p0 > 0.0:
            R2 += (p0 / mass_f) * float(np.einsum("ab,ab->", S_at_new, S_at_new))
        ay2 = float(fw @ (fit_a ** 2))
        E_S_post = (R2
                    - 2.0 * float(np.einsum("ab,ab->", AR_f, W0_ls))
                    - 2.0 * float(np.einsum("ab,ab->", AyR_f, W1_new))
                    + float(np.einsum("ab,ab->", W0_ls, W0_ls))
                    + 2.0 * abar * float(np.einsum("ab,ab->", W0_ls, W1_new))
                    + ay2 * float(np.einsum("ab,ab->", W1_new, W1_new)))
    else:                                                  # everything died at ReLU
        m0_new = np.zeros(d); m1_new = np.zeros(d)
        W0_new = np.zeros((d, d)); W1_new = np.zeros((d, d))
        R_m = np.zeros((d, d))

    keep_atom_exact = (atom == "exact" and p0 > 0.0)
    parts_p = [w[pos]]; parts_a = [y[pos]]
    if p0 > 0.0:
        parts_p.insert(0, np.array([p0])); parts_a.insert(0, np.array([0.0]))
    p_new = np.concatenate(parts_p)
    new_state = PostAffineState(
        p=p_new / p_new.sum(), a=np.concatenate(parts_a),
        m0=m0_new, m1=m1_new, W0=W0_new, W1=W1_new,
        atom_m=(m_at_new if keep_atom_exact else None),
        atom_S=(S_at_new if keep_atom_exact else None))
    _t = _phase("fit", _t)

    if stats is not None:
        distortion = float((PQ * ((ymean - y[None, :]) ** 2 + vv)).sum() / W_tot)
        stats.setdefault("mass_lost", []).append(float(max(0.0, 1.0 - W_tot)))
        stats.setdefault("E_m", []).append(E_m_post)       # POST-fit mean residual
        stats.setdefault("E_S", []).append(E_S_post)       # POST-fit cov residual
        stats.setdefault("tr_R_m", []).append(float(np.trace(R_m)))
        stats.setdefault("scalar_distortion", []).append(distortion)
        stats.setdefault("psd_clipped", []).append(psd_clipped)
        stats.setdefault("num_cells", []).append(int(J))
        stats.setdefault("num_pos_nodes", []).append(int(pos.sum()))
        stats.setdefault("zero_atom_mass", []).append(p0)
    return new_state


def unconditional_mean_post(state: PostAffineState) -> np.ndarray:
    """Full mean ``E[X]`` for a post-affine state (readout, eq 126)."""
    out = np.empty(state.d + 1)
    out[SPIKE_COORD] = float(state.p @ state.a)
    bulk = np.zeros(state.d)
    for k in range(state.num_nodes):
        if state.atom_m is not None and k == 0:
            bulk += state.p[k] * state.atom_m
        else:
            bulk += state.p[k] * (state.m0 + state.m1 * state.a[k])
    out[1:] = bulk
    return out

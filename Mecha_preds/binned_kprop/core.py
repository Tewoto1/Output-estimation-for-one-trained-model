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

The bin grids, truncated-normal stats, and W2-optimal (Lloyd-Max) quantizers live in
``.binning`` (imported below); the ReLU integrals and matrix helpers
(``symmetrize`` / ``project_to_psd``) reuse the repo's shared torch-free kernel
``Mecha_preds._utils``. This module keeps the state, the linear/ReLU update rules,
moment recovery, and the top-level ``run_binned_kprop_k2`` driver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .._utils import (
    _phi, _Phi, exact_relu_covariance, relu_moments_1d, symmetrize, project_to_psd,
)
from .binning import (
    _VAR_FLOOR, _DEFAULT_MIN_PROB,
    Workers, resolve_workers, _run_bins,
    normal_interval_stats, find_bin, safe_bin_representative,
    ensure_zero_edge, make_gaussian_edges, make_relu_post_edges,
    lloyd_max_edges, lloyd_max_edges_mixture, lloyd_max_edges_mixture_split,
)

# Coordinate of the spike (e = e_1 -> index 0 in Python). The bulk is coords 1..n-1.
SPIKE_COORD = 0


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
                   workers: Workers = None,
                   stats: Optional[dict] = None) -> BinnedK2State:
    """Propagate a binned K=2 state through the linear map ``M`` (= W + e e^T).

    ``edges`` are the NEW pre-activation bin edges (length ``num_bins_new+1``; must
    include negative bins). Uses the conditional-Gaussian closure inside each old bin
    (spec section 6) and the transition-kernel mixture (section 5). Memory-efficient:
    never allocates the ``(m, m, d, d)`` tensor from the spec pseudocode -- the
    per-old-bin covariance ``Sigma_C - g g^T / s_Y^2`` is beta-independent, so only the
    rank-1 ``g g^T`` piece is reweighted per new bin.

    ``workers``: thread count for the two independent per-bin loops (over old bins, then
    new bins). ``None``/``"auto"`` auto-resolves per machine (``resolve_workers``); pass
    ``1`` for serial. Results are identical to serial regardless of the count.
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

    def _old_bin(alpha):
        if p[alpha] <= 0:
            return
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
            return

        ag[alpha] = g[alpha] / sY2[alpha]
        Sig0[alpha] = symmetrize(SigC - np.outer(g[alpha], g[alpha]) / sY2[alpha])
        Qa, y_mean, y_var, t1, _t2 = normal_interval_stats(mY, sY2[alpha], low, high,
                                                           min_prob=min_prob)
        keep = Qa > min_prob
        Q[:, alpha] = np.where(keep, Qa, 0.0)
        ybar[:, alpha] = np.where(keep, y_mean, 0.0)
        tau1[:, alpha] = np.where(keep, t1, 0.0)
        yvar[:, alpha] = np.where(keep, y_var, 0.0)

    _run_bins(m_old, _old_bin, workers)

    p_new_raw = Q @ p                                # (m_new,)
    a_new = np.zeros(m_new)
    mu_new = np.zeros((m_new, d))
    Sig_new = np.zeros((m_new, d, d))

    def _new_bin(beta):
        """Returns the PSD-clipped eigen-mass for this bin (summed by the caller -- a
        shared ``+=`` accumulator would race under threads)."""
        if p_new_raw[beta] <= min_prob:
            a_new[beta] = safe_bin_representative(edges, beta)
            return 0.0
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
        if not psd_clip:
            Sig_new[beta] = S
            return 0.0
        S, c = project_to_psd(S)
        Sig_new[beta] = S
        return c

    psd_clipped = float(sum(_run_bins(m_new, _new_bin, workers)))

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
_POST_EDGES_NOTICE = [False]


def _post_edges_removed_notice(where: str) -> None:
    if not _POST_EDGES_NOTICE[0]:
        _POST_EDGES_NOTICE[0] = True
        print(f"binned_kprop.{where}: post-ReLU re-binning was REMOVED (2026-07). The "
              "passed post grid is ignored: positive bins now pass through ReLU exactly "
              "and all nonpositive bins merge into the zero atom.")


def relu_step_k2(state: BinnedK2State, post_edges: Optional[np.ndarray] = None, *,
                 min_prob: float = _DEFAULT_MIN_PROB, bulk_relu: str = "exact",
                 relu_merge: str = "post", workers: Workers = None,
                 stats: Optional[dict] = None) -> BinnedK2State:
    """Propagate a binned K=2 state through coordinatewise ReLU. NO RE-BINNING.

    ReLU is the identity on every positive spike value, so a positive bin passes
    through UNCHANGED in ``(p, a)`` (only its bulk gets the K=2 ReLU update via the
    ``bulk_relu`` backend). All nonpositive bins have ``max(a, 0) = 0``: they are
    coincident after ReLU and merge EXACTLY into a single zero atom (spec 8.3 / 9,
    mixture moments for the bulk). Output: ``[zero atom] + [positive bins, untouched]``
    -- the positive-bin count is fully preserved.

    The historical third step -- re-binning the ReLU output on a fresh nonnegative
    ``post_edges`` grid -- was removed: it could only MERGE distinct positive bins
    (a strict information loss compounding per layer) and never added resolution.
    ``post_edges`` is accepted for backward compatibility and IGNORED (one printed
    notice). The pre-activation grid must have an edge at 0 so no bin straddles the
    kink (``ensure_zero_edge`` / the split grid builders guarantee this).

    ``relu_merge`` sets the order of merge and ReLU for the zero atom only:
      ``"post"`` (default, Strategy 1) -- ReLU each nonpositive bin's bulk, THEN merge
        the post-ReLU moments. One exact-ReLU per nonpositive bin.
      ``"pre"``  (Strategy 2) -- MERGE the nonpositive bins' pre-ReLU bulk into one
        Gaussian, THEN apply ONE exact-ReLU. Far fewer bivariate-ReLU calls;
        accuracy-neutral when the merged bins are ~bulk-exchangeable.
    Positive bins are singletons, so the strategies coincide there.

    ``workers``: thread count for the independent per-bin bulk-ReLU calls.
    ``None``/``"auto"`` auto-resolves per machine; ``1`` = serial (identical results).
    """
    if relu_merge not in ("post", "pre"):
        raise ValueError(f"relu_merge must be 'post' or 'pre', got {relu_merge!r}")
    if post_edges is not None:
        _post_edges_removed_notice("relu_step_k2")
    p, avals, mu, Sigma = state.p, state.a, state.mu, state.Sigma
    m_old = p.shape[0]
    d = mu.shape[1]

    live = p > 0.0                                   # empty bins are dropped, not carried
    pos_idx = np.nonzero(live & (avals > 0.0))[0]    # ReLU-invariant bins (kept verbatim)
    neg_idx = np.nonzero(live & (avals <= 0.0))[0]   # coincident at 0 after ReLU (merged)
    if pos_idx.size == 0 and neg_idx.size == 0:
        raise RuntimeError("all spike-bin mass vanished in relu_step_k2")

    # positive bins: spike untouched, bulk through the K=2 ReLU (one call per bin)
    k_pos = pos_idx.size
    mu_pos = np.zeros((k_pos, d)); Sig_pos = np.zeros((k_pos, d, d))

    def _pos_bin(j):
        alpha = pos_idx[j]
        mu_pos[j], Sig_pos[j] = _bulk_relu_update(mu[alpha], Sigma[alpha], bulk_relu)

    _run_bins(k_pos, _pos_bin, workers)

    # zero atom: exact merge of ALL nonpositive bins (mixture moments)
    parts_p = [p[pos_idx]]; parts_a = [avals[pos_idx]]
    parts_m = [mu_pos]; parts_S = [Sig_pos]
    p0 = float(p[neg_idx].sum())
    if p0 > 0.0:
        eta = p[neg_idx] / p0
        if relu_merge == "post":                     # Strategy 1: ReLU each, then merge
            k_neg = neg_idx.size
            mu_neg = np.zeros((k_neg, d)); Sig_neg = np.zeros((k_neg, d, d))

            def _neg_bin(j):
                alpha = neg_idx[j]
                mu_neg[j], Sig_neg[j] = _bulk_relu_update(mu[alpha], Sigma[alpha], bulk_relu)

            _run_bins(k_neg, _neg_bin, workers)
            m0 = eta @ mu_neg
            dm = mu_neg - m0[None, :]
            S0 = (np.einsum("j,jab->ab", eta, Sig_neg, optimize=True)
                  + np.einsum("j,ja,jb->ab", eta, dm, dm, optimize=True))
        else:                                        # Strategy 2: merge pre-ReLU, ONE ReLU
            mu_pre = eta @ mu[neg_idx]
            dm = mu[neg_idx] - mu_pre[None, :]
            S_pre = (np.einsum("j,jab->ab", eta, Sigma[neg_idx], optimize=True)
                     + np.einsum("j,ja,jb->ab", eta, dm, dm, optimize=True))
            m0, S0 = _bulk_relu_update(mu_pre, symmetrize(S_pre), bulk_relu)
        parts_p.insert(0, np.array([p0])); parts_a.insert(0, np.array([0.0]))
        parts_m.insert(0, m0[None, :]); parts_S.insert(0, symmetrize(S0)[None])

    p_new_raw = np.concatenate(parts_p)
    total = p_new_raw.sum()
    if total <= 0:
        raise RuntimeError("all spike-bin mass vanished in relu_step_k2")
    if stats is not None:                            # nothing can escape now; kept for logs
        stats.setdefault("relu_mass_lost", []).append(float(max(0.0, 1.0 - total)))
    return BinnedK2State(p=p_new_raw / total,
                         a=np.concatenate(parts_a),
                         mu=np.concatenate(parts_m, axis=0),
                         Sigma=np.concatenate(parts_S, axis=0))


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
def _spike_mixture(state: BinnedK2State, M: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pre-activation spike law ``A^+ = gamma A + r.B`` as a Gaussian mixture over the current
    bins: component ``alpha`` is ``N(m_Y,alpha, s_Y,alpha^2)`` with weight ``p_alpha`` (the
    linear-step algebra). Returns ``(weights, means, stds)`` for ``lloyd_max_edges_mixture``."""
    gamma = float(M[0, 0]); r = np.asarray(M[0, 1:], dtype=np.float64)
    m_Y = gamma * state.a + state.mu @ r
    s_Y = np.sqrt(np.clip((state.Sigma @ r * r).sum(axis=1), _VAR_FLOOR, None))
    return state.p, m_Y, s_Y


def adaptive_neg_bins(num_pos: int, neg_mass: float, *,
                      max_bins_neg: Optional[int] = None) -> int:
    """MASS-ADAPTIVE negative-bin count: same mass-per-bin as the positive side.

    The positive side always gets its FULL ``num_pos`` budget (ReLU keeps only
    positive bins, so that is the resolution that survives); the negative side gets
    ``ceil(num_pos * neg_mass / pos_mass)`` bins -- e.g. 50% negative mass with a
    20-bin budget -> 20 negative bins (40 total), 25% negative -> 7 (27 total). After
    ReLU the negatives collapse into the single zero atom, leaving the ``num_pos``
    positive bins + the atom: the bin count never dilutes across layers.

    ``max_bins_neg`` (default ``8 * num_pos``) caps the count in the near-dead regime
    (``neg_mass -> 1`` would otherwise blow up the grid; the merged zero atom gains
    little from extra negative resolution there)."""
    pos_mass = max(1.0 - float(neg_mass), 1e-12)
    cap = (8 * num_pos) if max_bins_neg is None else int(max_bins_neg)
    return int(np.clip(np.ceil(num_pos * float(neg_mass) / pos_mass), 1, max(1, cap)))


def run_binned_kprop_k2(weights: List[Tuple[np.ndarray, Optional[np.ndarray]]],
                        input_dim: int, num_bins: int, *, grid: str = "wasserstein",
                        pre_edges: Optional[np.ndarray] = None,
                        num_bins_pre_pos: Optional[int] = None,
                        num_bins_pre_neg: Optional[int] = None,
                        max_bins_neg: Optional[int] = None,
                        input_std: float = 1.0, bulk_relu: str = "exact",
                        relu_merge: str = "post",
                        min_prob: float = _DEFAULT_MIN_PROB,
                        workers: Workers = None,
                        collect: bool = False,
                        post_edges: Optional[np.ndarray] = None,      # DEPRECATED, ignored
                        num_bins_post: Optional[int] = None) -> dict:  # DEPRECATED, ignored
    """Predict ``E[f(X)]`` for ``X ~ N(0, input_std^2 I)`` by COORDINATE-SPIKE BINNED
    kprop (K=2) along ``e = e_1``.

    ``weights`` are ``(W, b)`` float64 pairs in forward order: the square ``n x n``
    hidden matrices ``M = W + e e^T`` (the spike is assumed already baked in -- pass the
    actual matrices), each followed by ReLU, then the linear readout (no ReLU).
    ``num_bins`` is THE hyperparameter: the POSITIVE-side spike-bin budget per layer.

    Each layer is: LINEAR (re-bin the exact Gaussian-mixture law of ``A^+`` on a grid
    with an edge at 0) then ReLU (positive bins pass through UNTOUCHED; all nonpositive
    bins merge exactly into the zero atom). There is NO post-ReLU re-binning -- the old
    third step ("re-bin on a nonnegative post grid") could only merge distinct positive
    bins and lost resolution every layer; it was removed. ``post_edges``/``num_bins_post``
    are accepted for backward compatibility and IGNORED (one printed notice).

    ``grid`` selects the pre-activation bin placement:
      ``"wasserstein"`` (default) -- per-layer DYNAMIC Lloyd-Max bins minimizing the W2 distance
        to the layer's exact Gaussian-mixture spike law ``A^+ = sum_a p_a N(m_Y,a, s_Y,a^2)``,
        SPLIT AT 0: the positive side always gets the full ``num_bins`` budget
        (override ``num_bins_pre_pos``); the negative side is MASS-ADAPTIVE --
        ``ceil(num_bins * neg_mass / pos_mass)`` bins, same mass-per-bin as the positive side
        (override ``num_bins_pre_neg``; capped at ``max_bins_neg``, default ``8 * num_bins``).
        E.g. budget 20 at 50% positive -> 40 bins pre-ReLU, 20 + zero-atom after; at 75%
        positive -> 27 pre, 20 + atom after. The surviving resolution never dilutes.
      ``"fixed"`` (legacy) -- equal-mass Gaussian quantiles built once and reused every layer
        (``pre_edges`` override; a 0 edge is inserted if absent). Positive resolution is
        whatever the fixed grid puts right of 0 (~``num_bins/2``); kept for ablations.

    ``workers`` sets the per-bin thread count inside each linear/ReLU step. ``None``/``"auto"``
    (the default) auto-resolves per machine -- ``8`` on a CUDA box, else ``min(8, cpu_count)``,
    overridable via ``$BINNED_KPROP_WORKERS`` -- so the default is PARALLEL; pass ``workers=1``
    for serial. Results are identical to serial regardless of the thread count. The resolved
    count is echoed in ``metadata["workers"]``. (If a threaded numpy build (MKL/OpenBLAS) is in
    use, cap its inner threads, e.g. ``OMP_NUM_THREADS``, to avoid oversubscription.)

    Returns ``{"mean", "metadata", ...}``; with ``collect`` also the per-layer spike distribution
    and the final state.
    """
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if grid not in ("fixed", "wasserstein"):
        raise ValueError(f"grid must be 'fixed' or 'wasserstein', got {grid!r}")
    if post_edges is not None or num_bins_post is not None:
        _post_edges_removed_notice("run_binned_kprop_k2")
    n_hidden = len(weights) - 1
    if n_hidden < 1:
        raise ValueError("need at least one hidden layer + a readout")
    d = input_dim - 1
    if num_bins_pre_pos is None:                      # positive pre-activation bins (ReLU keeps these)
        num_bins_pre_pos = num_bins                   #   the full budget, every layer
    dynamic = grid == "wasserstein"
    if dynamic:
        init_edges, _ = lloyd_max_edges(0.0, input_std, num_bins)   # W2-optimal N(0,std) over all reals
    else:
        if pre_edges is None:
            pre_edges = make_gaussian_edges(num_bins, std=input_std)
        pre_edges = ensure_zero_edge(np.asarray(pre_edges, dtype=np.float64))
        init_edges = pre_edges

    stats: dict = {}
    state = gaussian_initial_state(d, init_edges, input_std=input_std)
    spike_by_layer = []
    neg_bins_by_layer: List[int] = []
    for li in range(n_hidden):
        W, _b = weights[li]
        M = np.asarray(W, dtype=np.float64)
        if M.shape != (input_dim, input_dim):
            raise ValueError(f"hidden layer {li} must be square ({input_dim},{input_dim}); "
                             f"got {M.shape}")
        if dynamic:                                  # re-grid to the W2-optimal Lloyd-Max bins
            p_mix, m_Y, s_Y = _spike_mixture(state, M)
            if num_bins_pre_neg is None:             # mass-adaptive negative side (see docstring)
                neg_mass = float(np.sum(p_mix * _Phi(-m_Y / s_Y)))
                n_neg = adaptive_neg_bins(num_bins_pre_pos, neg_mass,
                                          max_bins_neg=max_bins_neg)
            else:
                n_neg = max(1, int(num_bins_pre_neg))
            neg_bins_by_layer.append(n_neg)
            # pre-activation grid split at 0: pin the positive side (the only side ReLU keeps)
            layer_pre, _ = lloyd_max_edges_mixture_split(p_mix, m_Y, s_Y,
                                                         n_neg, num_bins_pre_pos)
        else:
            layer_pre = pre_edges
        state = linear_step_k2(state, M, layer_pre, min_prob=min_prob, workers=workers, stats=stats)
        state = relu_step_k2(state, min_prob=min_prob, bulk_relu=bulk_relu,
                             relu_merge=relu_merge,
                             workers=workers, stats=stats)
        if collect:
            spike_by_layer.append(dict(layer=li, p=state.p.copy(), a=state.a.copy()))

    W_ro, b_ro = weights[-1]
    W_ro = np.asarray(W_ro, dtype=np.float64)
    full_mean = unconditional_mean(state)            # length n = d+1
    mean = W_ro @ full_mean + (0.0 if b_ro is None else np.asarray(b_ro, dtype=np.float64))

    out = {
        "mean": np.asarray(mean, dtype=np.float64).reshape(-1),
        "metadata": {
            "predictor": "binned_kprop_k2", "K": 2, "num_bins": int(num_bins), "grid": grid,
            "n_hidden": n_hidden,
            "num_bins_pre_pos": int(num_bins_pre_pos),
            "num_bins_pre_neg": ("adaptive" if num_bins_pre_neg is None
                                 else int(num_bins_pre_neg)),
            "neg_bins_by_layer": list(neg_bins_by_layer),
            "workers": resolve_workers(workers),
            "input_dim": int(input_dim), "bulk_relu": bulk_relu, "relu_merge": relu_merge,
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

"""core.py -- SIMPLE SPIKE-PROP: constant (unconditioned) bulk K=2 + exact 1-d
spike-channel recursion for coordinate-spiked ReLU MLPs (numpy/scipy, torch-free).

Same model class as ``..binned_kprop`` / ``..analytic_kprop``:

    Z = M X + b,   X^+ = ReLU(Z),   M = W + e_1 e_1^T,   W_ij ~ N(0, 1/n),

with the spike coordinate ``A = e_1 . X`` explicit and the bulk ``B`` (dimension
``d = n - 1``) beside it. THE DIFFERENCE from both companions: the bulk law is kept
completely UNCONDITIONAL -- one Gaussian ``N(m, Sigma)`` with NO dependence on the
spike value at all (the "CONST" surrogate: binned = one bulk Gaussian per spike bin,
analytic = one affine-in-the-spike family, simple = one constant Gaussian). The spike
coordinate is tracked as a full nonparametric 1-d law on a grid, driven by the exact
scalar channel recursion of writeups/affine_knee:

    A' = c ReLU-or-id(S) + xi,   xi ~ N(mu, omega^2) independent of S,
    mu = w^T m (+ bias),         omega^2 = w^T Sigma w,

i.e. ``p_{l+1} = phi_omega * (c ReLU + mu)_# p_l`` with the channel scalars read off
the TRACKED bulk moments and the actual weight rows (per-network, no MC).

One layer (M split as ``c = M[0,0]``, ``w = M[0,1:]``, ``u = M[1:,0]``, ``V = M[1:,1:]``):

  1. channel scalars ``mu = w^T m + b_0``, ``omega^2 = w^T Sigma w`` from the tracked
     bulk; spike moments ``E[S], Var(S)`` from the tracked 1-d law;
  2. spike pre-activation law by the grid convolution
     ``p'(a) = atom * phi(a; mu, omega) + int p(t) phi(a; c t + mu, omega) dt``;
  3. bulk linear step with BOTH re-aggregation terms:
     ``m' = u E[S] + V m + b_bulk``,  ``Sigma' = V Sigma V^T + Var(S) u u^T``
     (bulk variance flows into the spike through ``omega^2``; spike variance flows
     into the bulk through the rank-one ``u u^T`` -- the linear layer swaps variance
     between the two blocks and it must be re-collected on both sides);
  4. ReLU: spike law folds exactly on the grid (negative mass -> zero atom, positive
     part kept verbatim); bulk moments through the exact rank-2 Gaussian-ReLU map
     (shared ``.._utils.exact_relu_covariance``).

What is dropped, and the error budget (the CONST accounting): the bulk-spike
cross-covariance (the rho / m_1 route) and every conditional-on-the-spike structure.
Both dropped amplitudes are O(n^{-1/2}) per coordinate; they are mass-centered, so
they enter mass-integrated propagated MEANS only at second order O(1/n) per
coordinate, and O(n^{-1/2}) readout rows then give an output-mean error O(1/n) --
output MSE ~ n^{-2}, the same ORDER as the affine/binned predictors, with a larger
constant (the entire linear-response component is left in the residual). The spike
coordinate itself must carry its full non-Gaussian law (atom + branch is an O(1)
structure): replacing the grid law by two moments would cost O(1/sqrt(n)) at the
output, which is why the 1-d recursion is the one place the model is nonparametric.

Cost per layer: one ``V Sigma V^T`` congruence + one exact d x d ReLU covariance
(what binned pays PER BIN) + an O(G^2) scalar grid convolution (negligible).
Grid caveat: a fixed G-point grid resolves the O(1)-spike regime (theta ~ 1); for
strongly super/subcritical spikes (|c| >> 1 stretching, theta ~ sqrt(n)) see the
two-scale representation notes in the README before trusting the grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .._utils import _phi, exact_relu_covariance, symmetrize

# Coordinate of the spike (e = e_1 -> index 0 in Python). The bulk is coords 1..n-1.
SPIKE_COORD = 0

_OMEGA2_MIN = 1e-14          # below this the channel noise is degenerate -> refuse
_DEFAULT_NUM_GRID = 2001
_DEFAULT_SPAN = 8.0          # Gaussian tail span (sigmas) beyond the component range


# --------------------------------------------------------------------------- #
# the 1-d spike law: optional atom at 0 + a density on a finite grid
# --------------------------------------------------------------------------- #
@dataclass
class SpikeLaw:
    """1-d law of the spike variable: ``atom`` mass at exactly 0 plus a density
    sampled at strictly increasing nodes ``t`` (trapezoid quadrature). Pre-activation
    laws have ``atom == 0``; post-ReLU laws have ``t >= 0`` and the folded mass in
    ``atom``. A dead spike is ``atom == 1`` with empty arrays."""
    atom: float
    t: np.ndarray
    pdf: np.ndarray

    def check(self, tol: float = 1e-8) -> None:
        assert self.t.shape == self.pdf.shape, (self.t.shape, self.pdf.shape)
        assert -tol <= self.atom <= 1.0 + tol, self.atom
        if self.t.size:
            assert np.all(np.diff(self.t) > 0), "nodes not strictly increasing"
            assert self.pdf.min() >= -tol, f"negative density {self.pdf.min():.2e}"
        assert abs(law_mass(self) - 1.0) < 1e-6, f"law mass {law_mass(self):.8f}"


def _trapz_weights(t: np.ndarray) -> np.ndarray:
    """Trapezoid quadrature weights for sorted nodes ``t`` (non-uniform ok)."""
    if t.size < 2:
        return np.zeros(t.shape, dtype=np.float64)
    dt = np.diff(t)
    w = np.empty_like(t)
    w[0] = 0.5 * dt[0]
    w[-1] = 0.5 * dt[-1]
    w[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    return w


def law_mass(law: SpikeLaw) -> float:
    return float(law.atom + _trapz_weights(law.t) @ law.pdf) if law.t.size \
        else float(law.atom)


def law_moments(law: SpikeLaw) -> Tuple[float, float, float]:
    """``(E[S], E[S^2], Var(S))`` of the law (the atom sits at 0, so it only
    contributes mass). Exact within the trapezoid representation."""
    if not law.t.size:
        return 0.0, 0.0, 0.0
    wp = _trapz_weights(law.t) * law.pdf
    m1 = float(wp @ law.t)
    m2 = float(wp @ (law.t * law.t))
    return m1, m2, max(m2 - m1 * m1, 0.0)


def _normalize(law: SpikeLaw, stats: Optional[dict], key: str) -> SpikeLaw:
    total = law_mass(law)
    if stats is not None:
        stats.setdefault(key, []).append(abs(1.0 - total))
    if total <= 0.0:
        raise RuntimeError("spike-law mass vanished (grid span too small?)")
    return SpikeLaw(atom=law.atom / total, t=law.t, pdf=law.pdf / total)


def _insert_zero(a: np.ndarray) -> np.ndarray:
    """Insert an exact 0 node if the grid straddles 0 without containing it."""
    if a.size == 0 or a[0] >= 0.0 or a[-1] <= 0.0 or np.any(a == 0.0):
        return a
    k = int(np.searchsorted(a, 0.0))
    return np.insert(a, k, 0.0)


def gaussian_spike_law(mean: float, var: float, num_grid: int = _DEFAULT_NUM_GRID,
                       span: float = _DEFAULT_SPAN) -> SpikeLaw:
    """Pure Gaussian law on a grid (the input layer: ``S = X_0 ~ N(0, std^2)``)."""
    sd = float(np.sqrt(var))
    if sd <= 0.0:
        raise ValueError("gaussian_spike_law needs var > 0")
    a = _insert_zero(np.linspace(mean - span * sd, mean + span * sd, int(num_grid)))
    pdf = _phi((a - mean) / sd) / sd
    return _normalize(SpikeLaw(0.0, a, pdf), None, "")


def channel_push(law: SpikeLaw, c: float, mu: float, omega2: float, *,
                 num_grid: int = _DEFAULT_NUM_GRID, span: float = _DEFAULT_SPAN,
                 stats: Optional[dict] = None) -> SpikeLaw:
    """Push the post law ``S`` through the scalar channel ``A' = c S + xi``,
    ``xi ~ N(mu, omega^2)`` independent of ``S`` (eq: ``p' = phi_omega * (c . + mu)_# p``).

    Exact mixture evaluation on a fresh uniform grid (0 node inserted): the atom
    contributes ``atom * N(mu, omega^2)``, each density node ``t_j`` contributes its
    trapezoid mass at ``N(c t_j + mu, omega^2)``. Returns a continuous pre-activation
    law (``atom == 0``), renormalized (drift logged to ``stats['push_mass_drift']``).
    """
    omega2 = float(omega2)
    if omega2 < _OMEGA2_MIN:
        raise ValueError(
            f"channel noise degenerate (omega^2 = {omega2:.3e}): the bulk variance "
            "seen by the spike row vanished -- not supported (the recursion assumes "
            "an O(1) Gaussian channel).")
    om = float(np.sqrt(omega2))
    locs = c * law.t + mu if law.t.size else np.zeros(0)
    lo_core = min(locs.min(), mu) if locs.size else mu
    hi_core = max(locs.max(), mu) if locs.size else mu
    a = _insert_zero(np.linspace(lo_core - span * om, hi_core + span * om, int(num_grid)))
    dens = np.zeros(a.shape, dtype=np.float64)
    if law.atom > 0.0:
        dens += law.atom * _phi((a - mu) / om) / om
    if law.t.size:
        wts = _trapz_weights(law.t) * law.pdf                    # component masses
        dens += (_phi((a[:, None] - locs[None, :]) / om) / om) @ wts
    return _normalize(SpikeLaw(0.0, a, dens), stats, "push_mass_drift")


def relu_law(law: SpikeLaw, stats: Optional[dict] = None) -> SpikeLaw:
    """Fold the law through ReLU exactly (within the grid): every node ``t >= 0``
    passes verbatim, the negative mass joins the zero atom. The pre-activation grid
    always carries an exact 0 node when it straddles 0, so the split is clean."""
    if not law.t.size or law.t[-1] <= 0.0:                       # fully dead
        return SpikeLaw(1.0, np.zeros(0), np.zeros(0))
    keep = law.t >= 0.0
    t_pos, pdf_pos = law.t[keep], law.pdf[keep]
    pos_mass = float(_trapz_weights(t_pos) @ pdf_pos)
    atom = float(np.clip(1.0 - pos_mass, 0.0, 1.0))
    if stats is not None:
        stats.setdefault("relu_folded_mass", []).append(atom - law.atom)
    return _normalize(SpikeLaw(atom, t_pos, pdf_pos), stats, "relu_mass_drift")


# --------------------------------------------------------------------------- #
# moment recovery (model-implied full-vector moments; cross block is 0 by design)
# --------------------------------------------------------------------------- #
def unconditional_mean(law: SpikeLaw, m: np.ndarray) -> np.ndarray:
    """Full post-activation mean (length ``n = d + 1``): coord 0 from the spike law,
    coords 1.. from the tracked bulk mean."""
    out = np.empty(m.shape[0] + 1, dtype=np.float64)
    out[SPIKE_COORD] = law_moments(law)[0]
    out[1:] = m
    return out


def unconditional_mean_cov(law: SpikeLaw, m: np.ndarray, Sigma: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Full mean and covariance under the model. NOTE the spike-bulk cross block is
    identically 0 -- that is the CONST approximation itself, not an oversight."""
    d = m.shape[0]
    ES, _, varS = law_moments(law)
    mean = np.empty(d + 1)
    mean[SPIKE_COORD] = ES
    mean[1:] = m
    cov = np.zeros((d + 1, d + 1))
    cov[0, 0] = varS
    cov[1:, 1:] = Sigma
    return mean, symmetrize(cov)


# --------------------------------------------------------------------------- #
# full forward
# --------------------------------------------------------------------------- #
def run_simple_spikeprop_core(weights: List[Tuple[np.ndarray, Optional[np.ndarray]]],
                              input_dim: int, *,
                              num_grid: int = _DEFAULT_NUM_GRID,
                              span: float = _DEFAULT_SPAN,
                              input_std: float = 1.0,
                              collect: bool = False) -> dict:
    """Predict ``E[f(X)]`` for ``X ~ N(0, input_std^2 I)`` by SIMPLE SPIKE-PROP.

    ``weights``: ``(W, b)`` float64 pairs in forward order -- square ``n x n`` hidden
    matrices (spike baked in, coordinate 0), each followed by ReLU, then the linear
    readout (no ReLU). ``num_grid`` / ``span`` control the 1-d spike grid (the only
    discretization in the algorithm; everything else is closed form).

    Returns ``{"mean", "cov", "metadata", ...}``; ``cov`` is the model-implied output
    covariance (cross spike-bulk block dropped by construction -- treat as indicative).
    With ``collect`` also ``spike_by_layer`` (the tracked law per layer) and
    ``final_state = (law, m, Sigma)``.
    """
    n_hidden = len(weights) - 1
    if n_hidden < 1:
        raise ValueError("need at least one hidden layer + a readout")
    d = input_dim - 1
    stats: dict = {}
    # layer-0 "post" state is the input itself (identity activation, no ReLU, no atom)
    law = gaussian_spike_law(0.0, input_std ** 2, num_grid, span)
    m = np.zeros(d, dtype=np.float64)
    Sigma = (input_std ** 2) * np.eye(d, dtype=np.float64)
    spike_by_layer: List[dict] = []

    for li in range(n_hidden):
        W, b = weights[li]
        M = np.asarray(W, dtype=np.float64)
        if M.shape != (input_dim, input_dim):
            raise ValueError(f"hidden layer {li} must be square ({input_dim},{input_dim}); "
                             f"got {M.shape}")
        c = float(M[SPIKE_COORD, SPIKE_COORD])
        w = M[SPIKE_COORD, 1:]
        u = M[1:, SPIKE_COORD]
        V = M[1:, 1:]
        b0 = 0.0 if b is None else float(np.asarray(b, dtype=np.float64)[SPIKE_COORD])
        bb = None if b is None else np.asarray(b, dtype=np.float64)[1:]

        ES, _ES2, varS = law_moments(law)
        mu_chan = float(w @ m) + b0                     # channel scalars from the
        omega2 = float(w @ (Sigma @ w))                 #   TRACKED bulk moments
        # spike: exact scalar channel on the grid;  bulk: linear with re-aggregation
        law_pre = channel_push(law, c, mu_chan, omega2,
                               num_grid=num_grid, span=span, stats=stats)
        m_pre = u * ES + V @ m + (0.0 if bb is None else bb)
        Sigma_pre = symmetrize(V @ (Sigma @ V.T)) + varS * np.outer(u, u)
        # ReLU: exact fold on the grid; exact rank-2 Gaussian-ReLU map on the bulk
        law = relu_law(law_pre, stats)
        m, Sigma = exact_relu_covariance(m_pre, Sigma_pre)
        if collect:
            ESp, _, varSp = law_moments(law)
            spike_by_layer.append(dict(
                layer=li, c=c, mu=mu_chan, omega2=omega2,
                atom=law.atom, t=law.t.copy(), pdf=law.pdf.copy(),
                post_mean=ESp, post_var=varSp))

    W_ro, b_ro = weights[-1]
    W_ro = np.asarray(W_ro, dtype=np.float64)
    full_mean = unconditional_mean(law, m)
    mean = W_ro @ full_mean + (0.0 if b_ro is None else np.asarray(b_ro, dtype=np.float64))
    _fm, full_cov = unconditional_mean_cov(law, m, Sigma)
    cov = symmetrize(W_ro @ full_cov @ W_ro.T)

    out = {
        "mean": np.asarray(mean, dtype=np.float64).reshape(-1),
        "cov": cov,
        "metadata": {
            "predictor": "simple_spikeprop", "K": 2, "bulk": "const",
            "num_grid": int(num_grid), "span": float(span),
            "n_hidden": int(n_hidden), "input_dim": int(input_dim),
            "output_dim": int(np.asarray(mean).reshape(-1).shape[0]),
            "max_push_mass_drift": float(max(stats.get("push_mass_drift", [0.0]))),
            "max_relu_mass_drift": float(max(stats.get("relu_mass_drift", [0.0]))),
        },
    }
    if collect:
        out["spike_by_layer"] = spike_by_layer
        out["final_state"] = (law, m, Sigma)
        out["stats"] = stats
    return out

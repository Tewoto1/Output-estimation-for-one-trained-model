"""kprop_hook.py -- bridge from the binned predictor to ORDINARY (harmonic) kprop.

The binned algorithm represents the spike COORDINATE explicitly (bins) and propagates
the conditional BULK law inside each bin with ordinary cumulant propagation. This
module is that bridge: it imports and calls ``Mecha_preds.cumulants.kprop`` (the
harmonic kprop -- ``relu_kprop`` / ``mlp_kprop`` machinery) so the per-bin bulk ReLU
is the *same* validated routine used everywhere else in the repo.

Two entry points:

  * ``bulk_relu_kprop(mu, Sigma, k_max=2)`` -- a drop-in K=2 bulk-ReLU backend
    (``relu_step_k2(..., bulk_relu="kprop")`` routes here). At ``k_max=2`` the harmonic
    kprop reproduces the leading-order gain off-diagonal (= our ``"gain"`` backend);
    with ``exact_relu_cov=True`` it uses the exact bivariate covariance (= ``"exact"``).

  * ``bulk_relu_kprop_tower(cumulants, k_max)`` -- the general ``K > 2`` hook
    (spec 10.4): run the ordinary kprop activation routine on a bin's conditional bulk
    cumulant tower of ANY order. ``BinnedKState`` + ``relu_step_k_general`` use it.

Everything here imports kprop LAZILY (inside the functions) so the torch-free K=2
numpy core (``core.py``) stays importable without torch; this module's calls require
torch + the kprop deps (Python >= 3.12 or the repo's ``_kprop_compat`` shim, both
handled by ``Mecha_preds.cumulants.__init__``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .binning import find_bin, safe_bin_representative, _DEFAULT_MIN_PROB
from .core import BinnedK2State, symmetrize, SPIKE_COORD


# --------------------------------------------------------------------------- #
# lazy kprop import (torch-based)
# --------------------------------------------------------------------------- #
def _load_kprop():
    """Import the harmonic-kprop symbols we need (lazily; requires torch).

    Importing ``..cumulants.kprop.kprop_harmonic`` runs ``Mecha_preds.cumulants.__init__``,
    which installs the kprop Python-compat shim on interpreters < 3.12 and otherwise is a
    no-op; kprop itself targets Python >= 3.12 (the repo env)."""
    import torch  # noqa: F401  (surface a clear error if torch is absent)
    from ..cumulants.kprop.kprop_harmonic import relu_kprop, coerce_input, Kind
    return torch, relu_kprop, coerce_input, Kind


def _to_kind(kind, Kind):
    """Map a string / ``Kind`` to the kprop ``Kind`` enum (default SIMPLE)."""
    if isinstance(kind, str):
        return {"SIMPLE": Kind.SIMPLE, "AUGMENT": Kind.AUGMENT,
                "OLD": Kind.OLD, "BASE": Kind.BASE}[kind.upper()]
    return kind


# --------------------------------------------------------------------------- #
# general-K bulk ReLU via ordinary kprop  (spec 10.4)
# --------------------------------------------------------------------------- #
def bulk_relu_kprop_tower(cumulants: Dict[int, np.ndarray], k_max: int,
                          kind="SIMPLE", *, exact_relu_cov: bool = False
                          ) -> Dict[int, np.ndarray]:
    """Run the ordinary kprop ReLU activation on ONE bin's bulk cumulant tower.

    ``cumulants`` maps order ``d -> dense numpy tensor`` of shape ``(bulk_d,)*d``
    (``cumulants[1]`` = conditional bulk mean, ``[2]`` = covariance, ``[3], [4]`` =
    higher conditional bulk cumulants). Returns the post-ReLU tower in the same dense
    format. This is literally ``ordinary_kprop_activation_step`` from spec 10.4 -- it
    delegates to ``Mecha_preds.cumulants.kprop.relu_kprop`` so the per-bin bulk slice
    gets the full harmonic / power-cumulant treatment (NOT ablated).
    """
    torch, relu_kprop, coerce_input, Kind = _load_kprop()
    kind_e = _to_kind(kind, Kind)
    K_raw = {d: torch.as_tensor(np.asarray(t, dtype=np.float64), dtype=torch.float64)
             for d, t in cumulants.items()}
    K_in = coerce_input(K_raw, k_max=k_max, kind=kind_e)
    K_out = relu_kprop(K_in, k_max=k_max, kind=kind_e, exact_relu_cov=exact_relu_cov)
    out: Dict[int, np.ndarray] = {}
    for d, H in K_out.items():
        out[d] = H.to_tensor().detach().cpu().numpy().astype(np.float64)
    return out


def bulk_relu_kprop(mu: np.ndarray, Sigma: np.ndarray, k_max: int = 2,
                    kind="SIMPLE", *, exact_relu_cov: bool = False):
    """K=2 bulk-ReLU backend: ``(mu, Sigma) -> (E[ReLU(B)], Cov[ReLU(B)])`` via kprop.

    Routed to by ``relu_step_k2(..., bulk_relu="kprop")``. Returns the order-1 and
    order-2 cumulants from ``bulk_relu_kprop_tower``.
    """
    out = bulk_relu_kprop_tower({1: np.asarray(mu, np.float64), 2: np.asarray(Sigma, np.float64)},
                                k_max=k_max, kind=kind, exact_relu_cov=exact_relu_cov)
    return out[1], symmetrize(out[2])


# --------------------------------------------------------------------------- #
# general-K binned state + ReLU step (spec sections 3 & 10)
# --------------------------------------------------------------------------- #
@dataclass
class BinnedKState:
    """General-K binned law of ``X = A e + B``: conditional bulk CUMULANT TENSORS per bin.

    p:          (num_bins,)            bin probabilities
    a:          (num_bins,)            representative spike value E[A | bin]
    cumulants:  dict order d -> array (num_bins, bulk_d, ..., bulk_d)  [d bulk axes]
                cumulants[1] = conditional bulk mean, [2] = covariance, [3..] = higher.

    The spike coordinate is carried ONLY by ``p`` and ``a`` (spec section 3): never store
    cumulants involving coordinate 0 here. ``K`` is the max bulk-cumulant order tracked.
    """
    p: np.ndarray
    a: np.ndarray
    cumulants: Dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def num_bins(self) -> int:
        return self.p.shape[0]

    @property
    def bulk_d(self) -> int:
        return self.cumulants[1].shape[1]

    @property
    def K(self) -> int:
        return max(self.cumulants.keys())

    @classmethod
    def from_k2(cls, st: BinnedK2State) -> "BinnedKState":
        return cls(p=st.p.copy(), a=st.a.copy(),
                   cumulants={1: st.mu.copy(), 2: st.Sigma.copy()})

    def to_k2(self) -> BinnedK2State:
        return BinnedK2State(p=self.p.copy(), a=self.a.copy(),
                             mu=self.cumulants[1].copy(), Sigma=self.cumulants[2].copy())


def relu_step_k_general(state: BinnedKState, post_edges: np.ndarray, k_max: int,
                        kind="SIMPLE", *, min_prob: float = _DEFAULT_MIN_PROB,
                        exact_relu_cov: bool = False,
                        merge_order: int = 2) -> BinnedKState:
    """ReLU step for the general-K binned state (spec section 10.4 + the merge of 8.3/9).

    Per bin, the bulk cumulant tower goes through ORDINARY kprop
    (``bulk_relu_kprop_tower``); the spike representatives are mapped through ReLU and
    bins are merged. The cross-bin MIXTURE is computed up to ``merge_order``
    (default 2 = mean + covariance, the explicit formulas of spec 5-7/10.3). Merging the
    within-bin cumulants of order >= 3 across bins needs the full ``mix_cumulants``
    moment<->cumulant machinery (spec 10.3); that is intentionally left as the documented
    extension boundary (spec 17), so ``merge_order > 2`` raises ``NotImplementedError``.
    The per-bin high-K propagation itself is fully exercised regardless of ``merge_order``.
    """
    if merge_order != 2:
        raise NotImplementedError(
            "cross-bin merge is implemented at order 2 (mean+cov); higher-order mixture "
            "needs mix_cumulants (spec 10.3) -- per-bin kprop runs at any k_max, but the "
            "discrete-mixture merge of order>=3 cumulants is the documented K>2 boundary.")
    p, avals = state.p, state.a
    m_old = p.shape[0]
    d = state.bulk_d
    post_edges = np.asarray(post_edges, dtype=np.float64)
    m_post = post_edges.shape[0] - 1

    # 1) per-bin bulk ReLU through ordinary kprop (any k_max)
    a_bulk = np.zeros((m_old, d))
    Sig_bulk = np.zeros((m_old, d, d))
    for alpha in range(m_old):
        if p[alpha] <= 0:
            continue
        tower = {dd: state.cumulants[dd][alpha] for dd in state.cumulants}
        out = bulk_relu_kprop_tower(tower, k_max=k_max, kind=kind, exact_relu_cov=exact_relu_cov)
        a_bulk[alpha] = out[1]
        Sig_bulk[alpha] = symmetrize(out.get(2, np.zeros((d, d))))

    # 2) map spike reps through ReLU, merge bins (order-2 mixture)
    old_to_new = [find_bin(post_edges, max(float(avals[a]), 0.0)) for a in range(m_old)]
    p_new = np.zeros(m_post)
    for alpha in range(m_old):
        if p[alpha] > 0:
            p_new[old_to_new[alpha]] += p[alpha]
    a_new = np.zeros(m_post)
    mu_new = np.zeros((m_post, d))
    Sig_new = np.zeros((m_post, d, d))
    for beta in range(m_post):
        if p_new[beta] <= min_prob:
            a_new[beta] = safe_bin_representative(post_edges, beta)
            continue
        alphas = [a for a in range(m_old) if old_to_new[a] == beta and p[a] > 0]
        eta = np.array([p[a] / p_new[beta] for a in alphas])
        a_new[beta] = float(sum(e * max(float(avals[a]), 0.0) for e, a in zip(eta, alphas)))
        for e, a in zip(eta, alphas):
            mu_new[beta] += e * a_bulk[a]
        S = np.zeros((d, d))
        for e, a in zip(eta, alphas):
            delta = a_bulk[a] - mu_new[beta]
            S += e * (Sig_bulk[a] + np.outer(delta, delta))
        Sig_new[beta] = symmetrize(S)

    total = p_new.sum()
    if total <= 0:
        raise RuntimeError("all spike-bin mass vanished in relu_step_k_general")
    p_new = p_new / total
    return BinnedKState(p=p_new, a=a_new, cumulants={1: mu_new, 2: Sig_new})

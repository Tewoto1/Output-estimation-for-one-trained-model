"""core.py -- Shifted-Weight K-Propagation (SW-KPROP), numpy core (torch-free).

Implements the algorithm of "Shifted-Weight K-Propagation with Special-Direction
Cumulants" for a ReLU MLP whose hidden matrices are the shifted random matrices
``M = W - (1/sqrt(n)) 11^T`` (the all-ones / "special" direction is amplified by
~sqrt(n) each layer, so truncating by total cumulant order is wrong; the special
direction must be kept to high order).

State (split / "fiber" basis)
-----------------------------
For a length-n layer variable we carry, with the special direction ``u = 1/sqrt(n) 1``:

    mu        full mean vector            (n,)            -- special mean d1 = u . mu
    vS        variance of S = u . X        scalar          -- pure-special kappa_2  (eq: C_{2,0})
    g         cross covariance  P Sigma u  (n,)  (g _|_ u) -- special<->transverse  (C_{1,1})
    Sig_perp  transverse covariance P Sigma P (n,n)        -- the dense transverse block
    d[p]      pure-special cumulants kappa_p(S), p=3..R     -- the AMPLIFIED legs (C_{p,0})

so the full covariance is ``Sigma = vS u u^T + u g^T + g u^T + Sig_perp``. The huge
sqrt(n)-amplified scale lives ONLY in the scalar ``vS`` and in the special component of
``mu`` -- never inside the dense O(1) block ``Sig_perp`` -- which is exactly why this
representation stays accurate in float where a naive dense covariance would not.

Linear step  (paper eq 1, 13 -- exact by cumulant multilinearity)
    mu <- M mu;   Sigma <- M Sigma M^T;   d[p] <- a^p d[p],  a = u_out^T M u_in
done block-wise so the dense work is one congruence ``M Sig_perp M^T`` (GPU-friendly;
pass ``mm`` to route it to a device) and the special blocks are matvecs.

ReLU step  (paper sec 6 -- exact at rank<=2; special cumulants via the closure)
We condition on the scalar special mode S (Gaussian for R=2; reweighted to the tracked
cumulants d3..dR by a Gram-Charlier / Edgeworth series for R>=3 -- paper eq 21), run the
EXACT rank-2 Gaussian-ReLU step (relu.exact_relu_covariance, paper eqs 22-24) on the
O(1) conditional law at each Gauss-Hermite node, and mix. Conditioning removes the
amplified rank-1 spike, so every per-node covariance is well-conditioned; the
across-node variation of the conditional means rebuilds the output's special structure
and propagates d3..dR forward. For R=2 this equals the exact joint bivariate ReLU
(quadrature is spectrally exact for the smooth conditional moments); tau=0 in the paper.

The core is numpy-only (scipy for the ReLU integrals). ``run_sw_kprop`` in ``adapter.py``
wraps a torch ``model.MLP`` and can route the congruence to CUDA float64.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss

from ..relu_integrals import relu_moments_1d, exact_relu_covariance
# Hermite He_p and the central-moment -> cumulant conversion are shared (one copy) in
# ``.._cumulant_math``; imported here under their original private names.
from .._cumulant_math import He as _He, central_moments_to_cumulants as _central_moments_to_cumulants

_TINY = 1e-30


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class State:
    """Split-basis law of one layer variable (see module docstring)."""

    __slots__ = ("n", "u", "mu", "vS", "g", "Sig_perp", "d")

    def __init__(self, n: int, mu: np.ndarray, vS: float, g: np.ndarray,
                 Sig_perp: np.ndarray, d: Dict[int, float]):
        self.n = n
        self.u = np.full(n, 1.0 / np.sqrt(n), dtype=np.float64)
        self.mu = mu
        self.vS = float(vS)
        self.g = g
        self.Sig_perp = Sig_perp
        self.d = d

    @property
    def d1(self) -> float:
        return float(self.u @ self.mu)


def initial_state(n: int, input_std: float = 1.0) -> State:
    """Input law ``X ~ N(0, input_std^2 I_n)`` in split form: vS=std^2, g=0,
    Sig_perp = std^2 (I - u u^T), no higher special cumulants."""
    s2 = float(input_std) ** 2
    u = np.full(n, 1.0 / np.sqrt(n))
    Sig_perp = s2 * (np.eye(n) - np.outer(u, u))
    return State(n, mu=np.zeros(n), vS=s2, g=np.zeros(n), Sig_perp=Sig_perp, d={})


# --------------------------------------------------------------------------- #
# linear step  (exact, paper eq 1/13)
# --------------------------------------------------------------------------- #
def linear_step(st: State, W: np.ndarray, b: Optional[np.ndarray],
                mm: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> State:
    """Propagate the split state through ``Z = W X (+ b)`` exactly.

    ``mm(A, B) -> A @ B`` is the matmul backend for the only dense O(n^3) work
    (the transverse congruence ``W Sig_perp W^T``); default numpy, or a CUDA-backed
    callable for GPU.
    """
    n_out = W.shape[0]
    u_in = st.u
    u_out = np.full(n_out, 1.0 / np.sqrt(n_out), dtype=np.float64)

    mu_out = W @ st.mu + (b if b is not None else 0.0)
    p_vec = W @ u_in                              # image of the special direction
    q_vec = W @ st.g
    C = mm(W, mm(st.Sig_perp, W.T))               # dense transverse congruence (heavy)
    C = 0.5 * (C + C.T)

    a = float(u_out @ p_vec)                       # = u_out^T W u_in  (~ -sqrt(n) when shifted)
    uq = float(u_out @ q_vec)
    Cu = C @ u_out
    uCu = float(u_out @ Cu)

    vS_out = st.vS * a * a + 2.0 * a * uq + uCu
    Sig_u = st.vS * p_vec * a + p_vec * uq + q_vec * a + Cu   # Sigma_Z @ u_out
    g_out = Sig_u - vS_out * u_out
    g_out = g_out - float(u_out @ g_out) * u_out   # re-orthogonalize against roundoff

    p_perp = p_vec - a * u_out
    q_perp = q_vec - uq * u_out
    PCP = C - np.outer(u_out, Cu) - np.outer(Cu, u_out) + uCu * np.outer(u_out, u_out)
    Sig_perp_out = (PCP + st.vS * np.outer(p_perp, p_perp)
                    + np.outer(p_perp, q_perp) + np.outer(q_perp, p_perp))
    Sig_perp_out = 0.5 * (Sig_perp_out + Sig_perp_out.T)

    d_out = {p: (a ** p) * val for p, val in st.d.items()}   # a^p d_p  (eq 15 leading term)
    return State(n_out, mu=mu_out, vS=vS_out, g=g_out, Sig_perp=Sig_perp_out, d=d_out)


# --------------------------------------------------------------------------- #
# special-mode Edgeworth quadrature  (the summation of the tracked cumulants)
# --------------------------------------------------------------------------- #
def special_mode_quadrature(d1: float, vS: float, d: Dict[int, float], R: int,
                            n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Edgeworth / Gram-Charlier quadrature for the special mode S (paper eq 21).

    The nonlinear step is NOT the exact Gaussian-ReLU integral -- that would assume the
    special mode is Gaussian. Instead it is the finite Edgeworth/Gram-Charlier closure:
    every tracked higher cumulant lives along the special direction u
    (``kappa_{i_1..i_p} = d_p u_{i_1}..u_{i_p}``), so the order-R operator
    ``exp_{<=R}( sum_{p>=3} d_p/p! (u.d/dx)^p )`` acting on the Gaussian-ReLU moment is
    realized EXACTLY as Gauss-Hermite nodes for the Gaussian part N(d1, vS) reweighted by
    the truncated Edgeworth series

        w_k = w_GH_k * [ 1 + (g1/6) He_3(xi_k) + (g2/24) He_4(xi_k) ],
        g1 = d3 / vS^{3/2}  (skewness),   g2 = d4 / vS^2  (excess kurtosis),

    with xi_k the standardized node. R=2 -> no correction (Gaussian special mode); R=3
    adds the skew term; R=4 adds the kurtosis term. The weights are SIGNED (the Edgeworth
    measure is not a probability measure) -- that is the literal cumulant summation. The
    weights still sum to 1 exactly (Gauss-Hermite integrates He_3, He_4 to zero), so no
    renormalization. The exact Gaussian-ReLU integral is then used only on the conditional
    transverse law, which IS Gaussian in this closure.
    """
    t, om = hermgauss(n_nodes)
    sig = np.sqrt(max(vS, 0.0))
    s = d1 + np.sqrt(2.0) * sig * t
    w = om / np.sqrt(np.pi)
    w = w / w.sum()                                 # base Gaussian weights (sum 1)
    if R >= 3 and sig > 0 and d:
        xi = np.sqrt(2.0) * t                       # standardized nodes (s-d1)/sig
        fac = np.ones_like(xi) + (d.get(3, 0.0) / sig ** 3 / 6.0) * _He(3, xi)
        if R >= 4:
            fac = fac + (d.get(4, 0.0) / sig ** 4 / 24.0) * _He(4, xi)
        w = w * fac                                 # Edgeworth weights (signed; still sum to 1)
    return s, w


# --------------------------------------------------------------------------- #
# ReLU step  (exact rank-2 per node + mixing; paper sec 6)
# --------------------------------------------------------------------------- #
def relu_step(st: State, R: int, n_nodes: int) -> State:
    """Propagate the split state through coordinatewise ReLU (paper eq 21).

    Condition on the special mode S; the conditional law given S=s is Gaussian with
    mean shifted by ``c_vec*(s-d1)`` and an s-independent covariance ``Sig_cond``. The
    exact rank-2 Gaussian-ReLU integral is applied ONLY to that conditional transverse
    law (which is Gaussian in this closure); the special mode's non-Gaussianity is
    carried by the SIGNED Edgeworth/Gram-Charlier weights of
    ``special_mode_quadrature`` -- i.e. the moments are the Edgeworth SUMMATION of the
    tracked special cumulants (d3..dR), not an assumed-Gaussian integral.

    Because the Edgeworth weights are signed, the mixed covariance can lose positive-
    definiteness for R>=3; we project the transverse block back to PSD (its u-row/col is
    re-zeroed) so it remains a valid covariance to propagate. R=2 has positive weights
    and skips the projection.
    """
    n, u = st.n, st.u
    d1, vS = st.d1, st.vS

    if vS > _TINY:
        c_vec = u + st.g / vS
        Sig_cond = st.Sig_perp - np.outer(st.g, st.g) / vS
        s_nodes, w = special_mode_quadrature(d1, vS, st.d, R, n_nodes)
    else:                                          # degenerate special mode -> point mass
        c_vec = u
        Sig_cond = st.Sig_perp.copy()
        s_nodes = np.array([d1]); w = np.array([1.0])
    Sig_cond = 0.5 * (Sig_cond + Sig_cond.T)
    if R >= 3:                                      # signed Edgeworth weights upstream can make the
        vals, vecs = np.linalg.eigh(Sig_cond)       # joint covariance invalid -> project the ReLU input
        Sig_cond = (vecs * np.clip(vals, 0.0, None)) @ vecs.T  # back to PSD (it is well-conditioned: no spike)
        Sig_cond = 0.5 * (Sig_cond + Sig_cond.T)

    part_cond = np.zeros((n, n))                   # sum_k w_k Cov(X | s_k)
    rm_list: List[np.ndarray] = []                 # conditional means E[X | s_k]
    for k, s_k in enumerate(s_nodes):
        mu_k = st.mu + c_vec * (s_k - d1)
        rm_k, Spost_k = exact_relu_covariance(mu_k, Sig_cond)
        rm_list.append(rm_k)
        part_cond = part_cond + w[k] * Spost_k
    rm = np.array(rm_list)                          # (K, n)
    mu_X = np.clip(w @ rm, 0.0, None)              # E[ReLU] >= 0

    # re-split the post-ReLU law (law of total covariance) WITHOUT forming the spike
    Sig_u = part_cond @ u
    base_uSu = float(u @ Sig_u)
    Sig_perp_X = part_cond - np.outer(u, Sig_u) - np.outer(Sig_u, u) + base_uSu * np.outer(u, u)
    vS_X = base_uSu
    for k in range(len(s_nodes)):
        e = rm[k] - mu_X                            # across-node deviation
        ue = float(u @ e)
        vS_X += w[k] * ue * ue
        Sig_u = Sig_u + w[k] * e * ue
        e_perp = e - ue * u
        Sig_perp_X = Sig_perp_X + w[k] * np.outer(e_perp, e_perp)
    g_X = Sig_u - vS_X * u
    g_X = g_X - float(u @ g_X) * u
    Sig_perp_X = 0.5 * (Sig_perp_X + Sig_perp_X.T)
    vS_X = max(vS_X, 0.0)

    # forward the higher special cumulants: cumulants of {r_k = u.E[X|s_k], w_k}
    d_X: Dict[int, float] = {}
    if R >= 3:
        r = rm @ u
        rbar = float(w @ r)
        mc = {c: float(np.sum(w * (r - rbar) ** c)) for c in range(2, R + 1)}
        kap = _central_moments_to_cumulants(mc, R)
        for p in range(3, R + 1):
            d_X[p] = kap.get(p, 0.0)
    return State(n, mu=mu_X, vS=vS_X, g=g_X, Sig_perp=Sig_perp_X, d=d_X)


# --------------------------------------------------------------------------- #
# full forward
# --------------------------------------------------------------------------- #
def sw_kprop_predict(weights: List[Tuple[np.ndarray, Optional[np.ndarray]]],
                     input_dim: int, *, R: int = 2, n_nodes: int = 9,
                     input_std: float = 1.0,
                     mm: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                     collect: bool = False) -> dict:
    """Predict ``E[f(X)]`` for ``X ~ N(0, input_std^2 I)`` by SW-KPROP.

    ``weights`` are ``(W, b)`` float64 numpy pairs in forward order: the hidden
    matrices (ReLU after each) then the linear readout (no ReLU). Returns
    ``{"mean": (out_dim,), "metadata": {...}, [diagnostics]}``.
    """
    if mm is None:
        mm = lambda A, B: A @ B
    R = int(R)
    if R < 2:
        raise ValueError("SW-KPROP tracks at least the covariance (R>=2)")
    n_hidden = len(weights) - 1
    if n_hidden < 1:
        raise ValueError("need at least one hidden layer + a readout")

    st = initial_state(input_dim, input_std=input_std)
    special_by_layer = []
    for li in range(n_hidden):
        W, b = weights[li]
        st = linear_step(st, np.asarray(W, np.float64),
                         None if b is None else np.asarray(b, np.float64), mm)
        st = relu_step(st, R=R, n_nodes=n_nodes)
        if collect:
            special_by_layer.append(dict(layer=li, d1=st.d1, vS=st.vS,
                                         d={p: st.d.get(p, 0.0) for p in range(3, R + 1)}))

    W_ro, b_ro = weights[-1]                        # readout: linear, exact for the mean
    mean = np.asarray(W_ro, np.float64) @ st.mu + (0.0 if b_ro is None else np.asarray(b_ro, np.float64))

    out = {"mean": np.asarray(mean, np.float64).reshape(-1),
           "metadata": {"R": R, "n_nodes": n_nodes, "n_hidden": n_hidden,
                        "input_dim": input_dim, "output_dim": int(mean.reshape(-1).shape[0])}}
    if collect:
        out["special_by_layer"] = special_by_layer
        out["last_hidden"] = dict(mu=st.mu.copy(), vS=st.vS, g=st.g.copy(),
                                  Sig_perp=st.Sig_perp.copy(), d=dict(st.d))
    return out

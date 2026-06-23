"""core.py -- Spike K-Propagation (SPIKE-KPROP), numpy core (torch-free).

Direction-general split-basis cumulant propagation for a ReLU MLP whose hidden matrices
carry a rank-one spike of *O(1) eigenvalue* in a fixed unit direction ``v``:

    M = W' + theta * v v^T,     W'_{ij} ~ N(0, 1/fan_in),  ||v||_2 = 1,  |theta| <= n^{o(1)}.

This GENERALIZES SW-KPROP (``Mecha_preds.cumulants.swkprop``) from the hardwired all-ones
direction ``u = 1/sqrt(n) 1`` to an ARBITRARY spike direction ``v`` -- e.g. the *localized*
``e1`` (a single coordinate) or the *flat* ``1/sqrt(n) 1`` (all-ones). The spike is already
baked into the model weights; SPIKE-KPROP only needs the DIRECTION ``v`` to set up the split
basis (``theta`` is read implicitly from the weights). Setting ``v = 1/sqrt(n) 1`` reproduces
SW-KPROP bit-for-bit.

Why a configurable direction matters (the trace-projection theorem)
-------------------------------------------------------------------
Write the degree-``r`` cumulant tensor ``C`` and its contraction against the spike direction
``C(v^{otimes s}, .)``. After a shifted linear map ``M = W' + theta v v^T`` the diagram
splits into ordinary ``W'``-pairings on the open slots PLUS *explicit contractions of the old
cumulant against ``v``*. Power counting (W-edges give loops, v-edges give none):

    generic connected coefficient with q open slots :   squared size  n^{1-q}
    fully paired TRACE component, even q             :   squared size  n^{2-q}

For budget ``K`` the first omitted order is ``q = K+1``. If ``K`` is odd this is even, so its
*trace* part has squared size ``n^{1-K}`` -- too large to drop -- while the traceless residual
is the safe ``n^{-K}``. Hence one must keep the ``q = K+1`` **trace projection**. In the SPIKED
case the trace boundary is *exposed through the spike direction*: e.g. for ``C = theta G_4``
(``G_4`` = sum of the three pairings ``delta delta``),

    C(v,v,v,v) = 3 theta,        C(v,v,i,j) = theta(delta_{ij} + 2 v_i v_j),

and ``C(v,v,v,v)`` enters with ZERO random W-edges (no averaging), so its size is the raw
trace mass ``theta``. For a FLAT ``v = 1/sqrt(n) 1`` every directional cumulant decays,
``c_{r,n}(vv^T) = O(n^{2-r})``, so only the covariance survives and ordinary total-order kprop
is already exact. For a LOCALIZED ``v = e1`` we have ``sum_i |v_i|^r = 1`` for all ``r``, so
the spike-direction cumulants are ``O(1)`` at every order and MUST be retained.

What SPIKE-KPROP retains (and what it approximates)
---------------------------------------------------
It carries, in the split ``(v, v^perp)`` basis (see ``State``): the full covariance exactly,
and the **pure spike-direction cumulants** ``d_p = C(v,...,v) = kappa_p(S)`` of the special
mode ``S = v . X`` for ``p = 3..R`` -- precisely the ``q = 0`` trace projections that dominate
in the localized-spike case (``C(v,v,v,v)`` is exactly ``d_4``). These are injected at the ReLU
by the Edgeworth / Gram-Charlier summation over ``S`` (eq below). The mixed ``q >= 1`` trace
contractions (e.g. ``C(v,v,i,j) ~ theta delta_{ij}`` on the transverse block) are handled by the
exact rank-2 conditional-Gaussian closure (treated Gaussian given ``S``) rather than propagated
as a separate degree-4 trace tensor -- this is the one documented approximation; see the
notebook for where it shows up as a residual.

State (split / "fiber" basis)
-----------------------------
With the spike direction ``v`` (``||v||=1``), for a length-n layer variable we carry:

    mu        full mean vector            (n,)            -- special mean d1 = v . mu
    vS        variance of S = v . X        scalar          -- pure-special kappa_2
    g         cross covariance  P Sigma v  (n,)  (g _|_ v) -- special<->transverse
    Sig_perp  transverse covariance P Sigma P (n,n)        -- the dense transverse block
    d[p]      pure-special cumulants kappa_p(S), p=3..R     -- C(v,...,v), the trace-projection legs

so the full covariance is ``Sigma = vS v v^T + v g^T + g v^T + Sig_perp``.

Linear step  (exact by cumulant multilinearity)
    mu <- M mu;   Sigma <- M Sigma M^T;   d[p] <- a^p d[p],   a = v_out^T M v_in
done block-wise so the dense work is one congruence ``M Sig_perp M^T`` (GPU-friendly via ``mm``).

ReLU step
    condition on the scalar special mode S (Gaussian for R=2; reweighted to the tracked
    d3..dR by the Edgeworth/Gram-Charlier series for R>=3), run the EXACT rank-2 Gaussian-ReLU
    step on the O(1) conditional law at each Gauss-Hermite node, and mix.

The core is numpy-only (scipy for the ReLU integrals, reused from ``swkprop.relu``).
``run_spike_kprop`` in ``adapter.py`` wraps a torch ``model.MLP``.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss

# Reuse SW-KPROP's validated, direction-independent Gaussian-ReLU kernel (numpy/scipy).
# ``_phi``/``_Phi`` are the scipy-backed standard-normal pdf/cdf used everywhere here, so
# the analytic Wick coefficients below match ``kprop.wick.relu_wick_coef`` numerically.
from ..swkprop.relu import relu_moments_1d, exact_relu_covariance, _phi, _Phi

_TINY = 1e-30
_VAR_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# probabilists' Hermite He_p (for the Gram-Charlier special-mode reweight)
# (standard; mirrors swkprop.core._He so this module stays self-contained)
# --------------------------------------------------------------------------- #
def _He(p: int, z: np.ndarray) -> np.ndarray:
    if p == 0:
        return np.ones_like(z)
    if p == 1:
        return z
    if p == 2:
        return z * z - 1.0
    if p == 3:
        return z ** 3 - 3.0 * z
    if p == 4:
        return z ** 4 - 6.0 * z ** 2 + 3.0
    if p == 5:
        return z ** 5 - 10.0 * z ** 3 + 15.0 * z
    if p == 6:
        return z ** 6 - 15.0 * z ** 4 + 45.0 * z ** 2 - 15.0
    raise NotImplementedError(f"He_{p} not implemented (R<=6 supported)")


def _central_moments_to_cumulants(mu_c: Dict[int, float], max_p: int) -> Dict[int, float]:
    """Central moments {2:mu2, 3:mu3, ...} -> cumulants {p: kappa_p} for p=2..max_p."""
    k: Dict[int, float] = {}
    m2 = mu_c.get(2, 0.0)
    k[2] = m2
    if max_p >= 3:
        k[3] = mu_c.get(3, 0.0)
    if max_p >= 4:
        k[4] = mu_c.get(4, 0.0) - 3.0 * m2 * m2
    if max_p >= 5:
        k[5] = mu_c.get(5, 0.0) - 10.0 * mu_c.get(3, 0.0) * m2
    if max_p >= 6:
        k[6] = (mu_c.get(6, 0.0) - 15.0 * mu_c.get(4, 0.0) * m2
                - 10.0 * mu_c.get(3, 0.0) ** 2 + 30.0 * m2 ** 3)
    return k


def unit_vector(spec, n: int) -> np.ndarray:
    """Build a unit spike direction of length ``n`` from a spec.

    ``"e1"`` -> the first coordinate axis (localized spike);
    ``"ones"``/``"flat"`` -> ``1/sqrt(n) 1`` (flat / all-ones spike);
    an array-like -> normalized to unit L2 norm.
    """
    if isinstance(spec, str):
        s = spec.lower()
        if s in ("e1", "localized", "coord", "coordinate"):
            v = np.zeros(n, dtype=np.float64)
            v[0] = 1.0
            return v
        if s in ("ones", "flat", "allones", "all-ones", "mean"):
            return np.full(n, 1.0 / np.sqrt(n), dtype=np.float64)
        raise ValueError(f"unknown spike-direction spec {spec!r} (use 'e1', 'ones', or a vector)")
    v = np.asarray(spec, dtype=np.float64).reshape(-1)
    if v.shape[0] != n:
        raise ValueError(f"spike direction has length {v.shape[0]}, expected {n}")
    nrm = float(np.linalg.norm(v))
    if nrm < _TINY:
        raise ValueError("spike direction is the zero vector")
    return v / nrm


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class State:
    """Split-basis law of one layer variable along spike direction ``v`` (see module docstring)."""

    __slots__ = ("n", "u", "mu", "vS", "g", "Sig_perp", "d")

    def __init__(self, n: int, v: np.ndarray, mu: np.ndarray, vS: float, g: np.ndarray,
                 Sig_perp: np.ndarray, d: Dict[int, float]):
        self.n = n
        self.u = np.asarray(v, dtype=np.float64).reshape(-1)   # the (general) spike direction
        self.mu = mu
        self.vS = float(vS)
        self.g = g
        self.Sig_perp = Sig_perp
        self.d = d

    @property
    def d1(self) -> float:
        return float(self.u @ self.mu)


def initial_state(n: int, v: np.ndarray, input_std: float = 1.0) -> State:
    """Input law ``X ~ N(0, input_std^2 I_n)`` in split form for spike direction ``v``:
    vS = std^2 (variance along any unit direction), g = 0, Sig_perp = std^2 (I - v v^T)."""
    s2 = float(input_std) ** 2
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    Sig_perp = s2 * (np.eye(n) - np.outer(v, v))
    return State(n, v, mu=np.zeros(n), vS=s2, g=np.zeros(n), Sig_perp=Sig_perp, d={})


# --------------------------------------------------------------------------- #
# linear step  (exact, by cumulant multilinearity)
# --------------------------------------------------------------------------- #
def linear_step(st: State, W: np.ndarray, b: Optional[np.ndarray],
                mm: Callable[[np.ndarray, np.ndarray], np.ndarray],
                v_out: Optional[np.ndarray] = None) -> State:
    """Propagate the split state through ``Z = W X (+ b)`` exactly.

    ``v_out`` is the spike direction in the OUTPUT space (defaults to ``st.u`` -- correct for
    the square hidden layers of this study, where the same ``v v^T`` shift acts on every
    layer). ``mm(A, B) -> A @ B`` is the matmul backend for the only dense O(n^3) work (the
    transverse congruence ``W Sig_perp W^T``); default numpy, or a CUDA-backed callable.
    """
    n_out = W.shape[0]
    u_in = st.u
    u_out = st.u if v_out is None else np.asarray(v_out, dtype=np.float64).reshape(-1)

    mu_out = W @ st.mu + (b if b is not None else 0.0)
    p_vec = W @ u_in                              # image of the special direction
    q_vec = W @ st.g
    C = mm(W, mm(st.Sig_perp, W.T))               # dense transverse congruence (heavy)
    C = 0.5 * (C + C.T)

    a = float(u_out @ p_vec)                       # = v_out^T W v_in  (~ theta when shifted)
    uq = float(u_out @ q_vec)
    Cu = C @ u_out
    uCu = float(u_out @ Cu)

    vS_out = st.vS * a * a + 2.0 * a * uq + uCu
    Sig_u = st.vS * p_vec * a + p_vec * uq + q_vec * a + Cu   # Sigma_Z @ v_out
    g_out = Sig_u - vS_out * u_out
    g_out = g_out - float(u_out @ g_out) * u_out   # re-orthogonalize against roundoff

    p_perp = p_vec - a * u_out
    q_perp = q_vec - uq * u_out
    PCP = C - np.outer(u_out, Cu) - np.outer(Cu, u_out) + uCu * np.outer(u_out, u_out)
    Sig_perp_out = (PCP + st.vS * np.outer(p_perp, p_perp)
                    + np.outer(p_perp, q_perp) + np.outer(q_perp, p_perp))
    Sig_perp_out = 0.5 * (Sig_perp_out + Sig_perp_out.T)

    d_out = {p: (a ** p) * val for p, val in st.d.items()}   # a^p d_p  (leading term)
    return State(n_out, u_out, mu=mu_out, vS=vS_out, g=g_out, Sig_perp=Sig_perp_out, d=d_out)


# --------------------------------------------------------------------------- #
# LEGACY special-mode Gauss-Hermite quadrature.
# Retained ONLY as a regression reference (the analytic ``relu_step_edgeworth``
# below is the active ReLU step; see handoff Test 2, which compares the analytic
# mean against this GH path at a high node count). NOT used by spike_kprop_predict.
# --------------------------------------------------------------------------- #
def special_mode_quadrature(d1: float, vS: float, d: Dict[int, float], R: int,
                            n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Edgeworth / Gram-Charlier quadrature for the special mode S.

    Gauss-Hermite nodes for the Gaussian part ``N(d1, vS)`` reweighted by the truncated
    Edgeworth series ``w_k = w_GH_k [1 + (g1/6)He_3 + (g2/24)He_4]`` with skewness
    ``g1 = d3/vS^{3/2}`` and excess kurtosis ``g2 = d4/vS^2``. R=2 -> no correction; R=3 adds
    the skew term; R=4 adds the kurtosis term. The weights are SIGNED (the Edgeworth measure is
    not a probability measure) and still sum to 1 exactly (GH integrates He_3, He_4 to zero).
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
# LEGACY ReLU step  (exact rank-2 per Gauss-Hermite node + mixing).
# Kept for regression tests only; ``relu_step_edgeworth`` is the active step.
# --------------------------------------------------------------------------- #
def relu_step(st: State, R: int, n_nodes: int) -> State:
    """Propagate the split state through coordinatewise ReLU.

    Condition on the special mode S; the conditional law given S=s is Gaussian with mean
    shifted by ``c_vec*(s-d1)`` and an s-independent covariance ``Sig_cond``. The exact rank-2
    Gaussian-ReLU integral is applied to that conditional transverse law; the special mode's
    non-Gaussianity is carried by the SIGNED Edgeworth/Gram-Charlier weights. Because those
    weights are signed, the mixed covariance can lose positive-definiteness for R>=3; we
    project the transverse block back to PSD. R=2 has positive weights and skips the projection.
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
        Sig_cond = (vecs * np.clip(vals, 0.0, None)) @ vecs.T  # back to PSD (well-conditioned: no spike)
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

    # forward the higher spike-direction cumulants: cumulants of {r_k = v.E[X|s_k], w_k}
    d_X: Dict[int, float] = {}
    if R >= 3:
        r = rm @ u
        rbar = float(w @ r)
        mc = {c: float(np.sum(w * (r - rbar) ** c)) for c in range(2, R + 1)}
        kap = _central_moments_to_cumulants(mc, R)
        for p in range(3, R + 1):
            d_X[p] = kap.get(p, 0.0)
    return State(n, u, mu=mu_X, vS=vS_X, g=g_X, Sig_perp=Sig_perp_X, d=d_X)


# =========================================================================== #
# ANALYTIC EDGEWORTH / WICK ReLU STEP  (GH-FREE -- the active ReLU step)
# =========================================================================== #
# The special scalar mode  S = u . X  carries cumulants  d_p = kappa_p(S).  For a
# ReLU observable  G,  the truncated Edgeworth/Wick summation replaces the old
# Gauss-Hermite quadrature with a FINITE analytic sum:
#
#     E[G(X)] = E_G[G(X_G)] + sum_{p=3..R} d_p/p! * E_G[(c.grad)^p G(X_G)]
#
# with loading  c = Sigma u / vS = u + g/vS  and  X_G ~ N(mu, Sigma).  This is the
# Edgeworth density form turned into the Wick derivative form by Gaussian
# integration by parts (Stein/Hermite identity); no special_mode_quadrature, no
# Gauss-Hermite nodes, and no signed-weight PSD pathology -- the only error is the
# dropped d5+ truncation.  G = rho(X_i) needs the univariate coefficients
# E_G[rho^(p)(Z_i)] (same closed forms as kprop.wick.relu_wick_coef); the raw
# second moment G = rho(X_i)rho(X_j) needs the bivariate coefficients
# B[a,b] = E_G[rho^(a)(Z_i) rho^(b)(Z_j)], derived by conditioning (handoff 5-7).


def _relu_wick_uni(mu: np.ndarray, var: np.ndarray, k: int) -> np.ndarray:
    """``E_G[ rho^(k)(Z) ]`` for ``Z ~ N(mu, var)``, coordinatewise (numpy, k=0..4).

    Closed forms (probabilists' Hermite), identical to ``kprop.wick.relu_wick_coef``::

        k=0: sig phi(a) + mu Phi(a)     k=1: Phi(a)            k=2: phi(a)/sig
        k=3: -a phi(a)/sig^2            k=4: (a^2-1) phi(a)/sig^3      a = mu/sig

    A tiny variance floor keeps the distributional derivatives finite (handoff 3, opt. A).
    """
    mu = np.asarray(mu, dtype=np.float64)
    var = np.clip(np.asarray(var, dtype=np.float64), _VAR_FLOOR, None)
    sig = np.sqrt(var)
    a = mu / sig
    ph, Ph = _phi(a), _Phi(a)
    if k == 0:
        return sig * ph + mu * Ph
    if k == 1:
        return Ph
    if k == 2:
        return ph / sig
    if k == 3:
        return -a * ph / sig ** 2
    if k == 4:
        return (a * a - 1.0) * ph / sig ** 3
    raise NotImplementedError(f"univariate ReLU Wick coefficient k={k} not implemented (k<=4)")


def _pair_cond_derivs(mU, sU2, mV, sV2, q):
    """Conditional ReLU/Heaviside delta-moments, conditioning on ``V`` (handoff 6).

    Inputs are broadcastable grids for the ordered pair ``(U=Z_i, V=Z_j)``. Returns
    six arrays for ``E[ f(U) delta^{(j)}(V) ]`` via ``D_j`` of ``f_V(v) E[f(U)|V=v]``:

        rho * {delta, delta', delta''}  and  H * {delta, delta', delta''}.
    """
    sV2 = np.clip(sV2, _VAR_FLOOR, None)
    beta = q / sV2
    # Conditional variance tau^2 = sU2 (1 - rho^2). Clip |rho| away from 1 so a
    # near-degenerate pair (|rho| -> 1) cannot blow up the 1/tau, 1/tau^2 terms; such
    # pairs do not arise for well-conditioned network covariances, and the diagonal
    # (rho == 1) is overwritten by the scalar formula regardless.
    rho2 = np.clip(q * q / (sU2 * sV2), 0.0, 1.0 - 1e-6)
    tau2 = np.clip(sU2 * (1.0 - rho2), _VAR_FLOOR, None)
    tau = np.sqrt(tau2)
    m0 = mU - beta * mV
    eta = m0 / tau
    ph_eta, Ph_eta = _phi(eta), _Phi(eta)
    sigV = np.sqrt(sV2)
    f0 = _phi(mV / sigV) / sigV                 # marginal pdf of V at 0 (phi is even)
    f1 = (mV / sV2) * f0                         # f_V'(0)
    f2 = ((mV * mV) / (sV2 * sV2) - 1.0 / sV2) * f0   # f_V''(0)
    R0 = tau * ph_eta + m0 * Ph_eta             # E[U_+ | V=0]
    R1 = beta * Ph_eta                          # d/dt at 0
    R2 = beta * beta * ph_eta / tau
    H0 = Ph_eta                                 # P(U>0 | V=0)
    H1 = beta * ph_eta / tau
    H2 = -beta * beta * eta * ph_eta / (tau * tau)
    rho_delta = f0 * R0                          # E[rho(U) delta(V)]   = D0
    rho_delta_p = -(f1 * R0 + f0 * R1)          # E[rho(U) delta'(V)]  = -D1
    rho_delta_pp = f2 * R0 + 2.0 * f1 * R1 + f0 * R2   # E[rho(U) delta''(V)] = D2
    H_delta = f0 * H0
    H_delta_p = -(f1 * H0 + f0 * H1)
    H_delta_pp = f2 * H0 + 2.0 * f1 * H1 + f0 * H2
    return rho_delta, rho_delta_p, rho_delta_pp, H_delta, H_delta_p, H_delta_pp


def bivariate_relu_wick(mu: np.ndarray, Sigma: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    """All bivariate ReLU Wick coefficients ``B[a,b][i,j] = E_G[rho^(a)(Z_i) rho^(b)(Z_j)]``
    needed for ``R<=4`` (a+b in {3,4}), as a dict of ``(n,n)`` arrays.

    Uses ``rho^0=x_+, rho^1=H, rho^2=delta, rho^3=delta', rho^4=delta''`` so every term
    reduces to a conditional Gaussian pdf/cdf expression. Diagonal entries (``i==j``)
    are invalid (singular pair) and MUST be replaced by the scalar diagonal formula by
    the caller. Validated by finite differences of ``E[rho(Z_i)rho(Z_j)]`` (handoff Test 5).
    """
    mu = np.asarray(mu, np.float64)
    Sigma = np.asarray(Sigma, np.float64)
    s2 = np.clip(np.diag(Sigma).copy(), 0.0, None)
    mU, sU2 = mu[:, None], s2[:, None]
    mV, sV2 = mu[None, :], s2[None, :]
    q = Sigma
    rd, rdp, rdpp, Hd, Hdp, Hdpp = _pair_cond_derivs(mU, sU2, mV, sV2, q)   # condition on V=Z_j
    B = {
        (0, 3): rdp,  (3, 0): rdp.T,        # rho_i * delta'_j      /  delta'_i * rho_j
        (0, 4): rdpp, (4, 0): rdpp.T,       # rho_i * delta''_j     /  delta''_i * rho_j
        (1, 2): Hd,   (2, 1): Hd.T,         # H_i   * delta_j       /  delta_i  * H_j
        (1, 3): Hdp,  (3, 1): Hdp.T,        # H_i   * delta'_j      /  delta'_i * H_j
    }
    # B[2,2] = E[delta(Z_i) delta(Z_j)] = bivariate Gaussian density at (0,0).
    det = np.clip(sU2 * sV2 - q * q, 1e-300, None)
    quad = (sV2 * mU * mU - 2.0 * q * mU * mV + sU2 * mV * mV) / det
    B[(2, 2)] = np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det))
    return B


def _tiny_psd_cleanup(S: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Project a covariance back to PSD by clipping negative eigenvalues to 0 (handoff 8).

    Only does the (cheap) eigen-clip when a negative eigenvalue is actually present, and
    emits a warning when a *large* negative appears: that signals real Edgeworth
    divergence (e.g. the death/sub regime), which should surface rather than be hidden.
    """
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    if vals.size == 0 or vals.min() >= 0.0:
        return S
    vmax = max(float(vals.max()), 1.0)
    if vals.min() < -rtol * vmax:
        import warnings
        warnings.warn(f"spikekprop: non-roundoff negative eigenvalue {vals.min():.2e} "
                      f"(rel {vals.min()/vmax:.2e}) -- possible Edgeworth divergence.",
                      RuntimeWarning, stacklevel=2)
    S = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
    return 0.5 * (S + S.T)


def _raw_moments_to_cumulants_34(m: Dict[int, float], R: int) -> Dict[int, float]:
    """Cumulants kappa_3, kappa_4 from RAW moments m[1..R] (handoff 9)."""
    out: Dict[int, float] = {}
    if R >= 3:
        out[3] = m[3] - 3.0 * m[2] * m[1] + 2.0 * m[1] ** 3
    if R >= 4:
        out[4] = (m[4] - 4.0 * m[3] * m[1] - 3.0 * m[2] ** 2
                  + 12.0 * m[2] * m[1] ** 2 - 6.0 * m[1] ** 4)
    return out


def _one_sided_gauss_moments(m: float, var: float, qmax: int) -> List[float]:
    """One-sided Gaussian moments ``M_q = E[ S^q 1{S>0} ] = E[ReLU(S)^q]`` for ``S~N(m,var)``.

    ``M_0 = Phi(alpha)`` (= E[1{S>0}]), ``M_1 = m Phi + sig phi``, then the standard recursion
    ``M_q = m M_{q-1} + (q-1) var M_{q-2}`` (the boundary term vanishes for q>=2). Exact and
    kink-safe -- this is how the localized special mode passes ReLU without any quadrature.
    """
    sig = math.sqrt(max(var, _VAR_FLOOR))
    al = m / sig
    Phi, phi = float(_Phi(al)), float(_phi(al))
    M = [0.0] * (qmax + 1)
    M[0] = Phi
    if qmax >= 1:
        M[1] = m * Phi + sig * phi
    for q in range(2, qmax + 1):
        M[q] = m * M[q - 1] + (q - 1) * var * M[q - 2]
    return M


def _relu_power_wick_scalar(M: List[float], m: float, var: float, k: int, p: int) -> float:
    """Scalar ``E[ d^k ReLU(Z)^p ]`` for ``Z~N(m,var)`` (same identity as kprop.wick).

    ``rho^p`` is C^{p-1}: for ``k<=p`` the derivative is smooth, ``= p!/(p-k)! E[ReLU^{p-k}]``
    (with ``E[ReLU^0]=Phi``); for ``k>p`` it is ``p!`` times a distributional ReLU derivative,
    ``= p! E[ d^{k-p+1} ReLU(Z) ]`` (the univariate Wick coefficient).
    """
    if k <= p:
        return math.perm(p, k) * M[p - k]
    return math.factorial(p) * float(_relu_wick_uni(np.array([m]), np.array([var]), k - p + 1)[0])


def _spike_cumulants_scalar_closure(mu, Sigma, u, c, vS, d1, d, R) -> Dict[int, float]:
    """Update the higher special-mode cumulants ``d_new[p] = kappa_p(u . rho(X))`` (handoff 9B).

    TEMPORARY: scalar-closure cumulant update (not the full diagrammatic 9A); mean and
    covariance ARE analytic Edgeworth/Wick. Two regimes, both Gauss-Hermite-FREE:

      * LOCALIZED (u ~ a coordinate axis, e.g. ``e1``): the special mode itself is a ReLU
        input, so ``u . rho(X) = ReLU(S)`` and its cumulants are the EXACT rectified-Gaussian
        power moments plus the Edgeworth d3/d4 corrections -- closed form, KINK-SAFE (this is
        the case the old Gauss-Hermite step integrated through the kink with quadrature error).
      * FLAT / smooth (e.g. ``ones``): the special mode is an average of many ReLUs, so the
        conditional-mean curve ``r(s) = u . E[rho(X)|S=s]`` is smooth; a 4th-order delta method
        around ``s=d1`` (analytic derivatives via the univariate Wick coefficients) gives its
        cumulants. Directional cumulants are O(n^{2-p}) here, so this term is negligible anyway.
    """
    if vS <= _TINY:
        return {}
    d3_in, d4_in = d.get(3, 0.0), d.get(4, 0.0)
    i_star = int(np.argmax(np.abs(u)))

    if abs(u[i_star]) > 0.9:                  # LOCALIZED: u . rho(X) = ReLU(special mode)
        M = _one_sided_gauss_moments(d1, vS, R)
        coef = {0: 1.0, 3: d3_in / 6.0, 4: d4_in / 24.0}
        m_raw = {q: sum(cj * _relu_power_wick_scalar(M, d1, vS, j, q) for j, cj in coef.items())
                 for q in range(1, R + 1)}
        return _raw_moments_to_cumulants_34(m_raw, R)

    # FLAT / smooth: cumulants of the smooth conditional-mean curve r(S) via a delta method.
    s2 = np.clip(np.diag(Sigma).copy(), 0.0, None)
    tau2 = np.maximum(s2 - vS * c * c, 1e-6 * np.maximum(s2, _VAR_FLOOR))   # relative floor: stay smooth
    rp = [float(np.sum(u * (c ** p) * _relu_wick_uni(mu, tau2, p))) for p in range(R + 1)]
    a = np.zeros(R + 1)                        # Y = r(S) - r(d1) ~ sum_p a_p X^p,  X = S - d1
    for p in range(1, R + 1):
        a[p] = rp[p] / math.factorial(p)
    kappa = {2: vS}
    if R >= 3:
        kappa[3] = d3_in
    if R >= 4:
        kappa[4] = d4_in
    qmax = R * R
    muX = [0.0] * (qmax + 1)                   # raw moments of X from its cumulants
    muX[0] = 1.0
    for nn in range(1, qmax + 1):
        muX[nn] = sum(math.comb(nn - 1, k - 1) * kappa.get(k, 0.0) * muX[nn - k]
                      for k in range(1, nn + 1))
    EY = {1: float(np.dot(a, muX[:len(a)]))}   # raw moments of Y about r(d1)
    poly = a.copy()
    for q in range(2, R + 1):
        poly = np.convolve(poly, a)
        EY[q] = float(np.dot(poly, muX[:len(poly)]))
    return _raw_moments_to_cumulants_34(EY, R)


def relu_step_edgeworth(st: State, R: int) -> State:
    """Active ReLU step: analytic Edgeworth/Wick summation, NO Gauss-Hermite (handoff 10).

    Mean and covariance are computed to the tracked order in closed form; the higher
    special-mode cumulants use the scalar closure above. R=2 reproduces the one-shot
    exact bivariate Gaussian ReLU (``exact_relu_covariance``) on the full covariance.
    """
    if R < 2:
        raise ValueError("R must be >= 2")
    if R > 4:
        raise NotImplementedError("Analytic spike Edgeworth ReLU supports R<=4 for now")
    n, u, mu = st.n, st.u, st.mu
    Sigma = (st.vS * np.outer(u, u) + np.outer(u, st.g) + np.outer(st.g, u) + st.Sig_perp)
    Sigma = 0.5 * (Sigma + Sigma.T)
    c = (u + st.g / st.vS) if st.vS > _TINY else u.copy()    # loading: Sigma u / vS
    s2 = np.clip(np.diag(Sigma).copy(), 0.0, None)
    d3, d4 = st.d.get(3, 0.0), st.d.get(4, 0.0)

    # ---- mean: univariate ReLU Wick coefficients ----
    mu_new = _relu_wick_uni(mu, s2, 0)
    if R >= 3:
        mu_new = mu_new + (d3 / 6.0) * c ** 3 * _relu_wick_uni(mu, s2, 3)
    if R >= 4:
        mu_new = mu_new + (d4 / 24.0) * c ** 4 * _relu_wick_uni(mu, s2, 4)

    # ---- raw second moment: exact Gaussian base + bivariate Edgeworth corrections ----
    base_mu, base_cov = exact_relu_covariance(mu, Sigma)
    raw = base_cov + np.outer(base_mu, base_mu)
    if R >= 3:
        B = bivariate_relu_wick(mu, Sigma)
        ci, cj = c[:, None], c[None, :]
        corr3 = (cj ** 3 * B[(0, 3)] + 3.0 * ci * cj ** 2 * B[(1, 2)]
                 + 3.0 * ci ** 2 * cj * B[(2, 1)] + ci ** 3 * B[(3, 0)])
        np.fill_diagonal(corr3, 0.0)                          # diagonal set by scalar formula
        raw = raw + (d3 / 6.0) * corr3
        if R >= 4:
            corr4 = (cj ** 4 * B[(0, 4)] + 4.0 * ci * cj ** 3 * B[(1, 3)]
                     + 6.0 * ci ** 2 * cj ** 2 * B[(2, 2)] + 4.0 * ci ** 3 * cj * B[(3, 1)]
                     + ci ** 4 * B[(4, 0)])
            np.fill_diagonal(corr4, 0.0)
            raw = raw + (d4 / 24.0) * corr4

    # ---- diagonal raw second moment: scalar E[rho(Z)^2] + d3/d4 corrections (handoff 7) ----
    sig = np.sqrt(np.clip(s2, _VAR_FLOOR, None))
    a = mu / sig
    ph, Ph = _phi(a), _Phi(a)
    diag = (mu * mu + s2) * Ph + mu * sig * ph                # C0 = E[rho(Z)^2]
    if R >= 3:
        diag = diag + (d3 / 6.0) * c ** 3 * (2.0 * ph / sig)              # C3 = 2 phi/sig
    if R >= 4:
        diag = diag + (d4 / 24.0) * c ** 4 * (-2.0 * a / s2 * ph)         # C4 = -2 a/sig^2 phi
    np.fill_diagonal(raw, diag)
    raw = 0.5 * (raw + raw.T)

    Sigma_new = raw - np.outer(mu_new, mu_new)
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    if R >= 3:                                                # signed corrections -> roundoff PSD only
        Sigma_new = _tiny_psd_cleanup(Sigma_new)

    # ---- re-split into spike state (handoff 8) ----
    vS_new = float(u @ Sigma_new @ u)
    Sig_u = Sigma_new @ u
    g_new = Sig_u - vS_new * u
    g_new = g_new - float(u @ g_new) * u
    Sig_perp_new = (Sigma_new - vS_new * np.outer(u, u)
                    - np.outer(u, g_new) - np.outer(g_new, u))
    Sig_perp_new = 0.5 * (Sig_perp_new + Sig_perp_new.T)
    vS_new = max(vS_new, 0.0)

    d_new = (_spike_cumulants_scalar_closure(mu, Sigma, u, c, st.vS, st.d1, st.d, R)
             if R >= 3 else {})
    return State(n, u, mu=mu_new, vS=vS_new, g=g_new, Sig_perp=Sig_perp_new, d=d_new)


# --------------------------------------------------------------------------- #
# full forward
# --------------------------------------------------------------------------- #
def spike_kprop_predict(weights: List[Tuple[np.ndarray, Optional[np.ndarray]]],
                        input_dim: int, spike_dir="e1", *, R: int = 2, n_nodes: int = 9,
                        input_std: float = 1.0, relu_method: str = "edgeworth",
                        mm: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                        collect: bool = False) -> dict:
    """Predict ``E[f(X)]`` for ``X ~ N(0, input_std^2 I)`` by SPIKE-KPROP along ``spike_dir``.

    ``weights`` are ``(W, b)`` float64 numpy pairs in forward order: the hidden matrices (ReLU
    after each, square n x n) then the linear readout (no ReLU). ``spike_dir`` is ``"e1"``,
    ``"ones"``, or a length-``input_dim`` vector. The spike (theta v v^T) is assumed already
    present in ``weights``; only the DIRECTION is needed here.

    ``relu_method`` selects the ReLU step: ``"edgeworth"`` (default) is the analytic,
    Gauss-Hermite-FREE Edgeworth/Wick summation (``relu_step_edgeworth``); ``"gh"`` is the
    legacy Gauss-Hermite quadrature (``relu_step``, kept only for regression). ``n_nodes`` is
    ignored unless ``relu_method == "gh"``. Returns ``{"mean": (out_dim,), "metadata", ...}``.
    """
    if mm is None:
        mm = lambda A, B: A @ B
    R = int(R)
    if R < 2:
        raise ValueError("SPIKE-KPROP tracks at least the covariance (R>=2)")
    if relu_method not in ("edgeworth", "gh"):
        raise ValueError(f"relu_method must be 'edgeworth' or 'gh', got {relu_method!r}")
    n_hidden = len(weights) - 1
    if n_hidden < 1:
        raise ValueError("need at least one hidden layer + a readout")

    v = unit_vector(spike_dir, input_dim)
    st = initial_state(input_dim, v, input_std=input_std)
    special_by_layer = []
    for li in range(n_hidden):
        W, b = weights[li]
        st = linear_step(st, np.asarray(W, np.float64),
                         None if b is None else np.asarray(b, np.float64), mm, v_out=v)
        st = (relu_step_edgeworth(st, R=R) if relu_method == "edgeworth"
              else relu_step(st, R=R, n_nodes=n_nodes))
        if collect:
            special_by_layer.append(dict(layer=li, d1=st.d1, vS=st.vS,
                                         d={p: st.d.get(p, 0.0) for p in range(3, R + 1)}))

    W_ro, b_ro = weights[-1]                        # readout: linear, exact for the mean
    mean = np.asarray(W_ro, np.float64) @ st.mu + (0.0 if b_ro is None else np.asarray(b_ro, np.float64))

    out = {"mean": np.asarray(mean, np.float64).reshape(-1),
           "metadata": {"R": R, "n_nodes": n_nodes, "n_hidden": n_hidden, "relu_method": relu_method,
                        "input_dim": input_dim, "output_dim": int(mean.reshape(-1).shape[0]),
                        "spike_dir": spike_dir if isinstance(spike_dir, str) else "custom"}}
    if collect:
        out["special_by_layer"] = special_by_layer
        out["last_hidden"] = dict(mu=st.mu.copy(), vS=st.vS, g=st.g.copy(),
                                  Sig_perp=st.Sig_perp.copy(), d=dict(st.d))
    return out

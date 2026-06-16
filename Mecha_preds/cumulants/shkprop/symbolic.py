"""symbolic.py -- symbolic hidden-mode kprop (scalar h, k_max = 2).

Carries the conditional cumulants of each layer's activations ``A | h`` as
polynomial jets in the centered scalar hidden mode ``dh`` (``PolyJet``), instead
of re-running kprop at quadrature nodes of ``h``. Per the spec:

  * linear layers push cumulants forward EXACTLY (einsum per degree);
  * the ReLU layer is the only approximation -- the conditional-Gaussian residual
    closure, evaluated as a jet by collocation over ``dh`` (pseudo-spectral, the
    scalar realization of the spec's Taylor-jet composition);
  * the hidden mode is marginalized at the end from its cumulants (Gauss-Hermite
    quadrature for Gaussian ``h``; law of total covariance for K2);
  * an adaptive tail diagnostic decides / reports whether the hidden degree ``p``
    is sufficient and never silently assumes Gaussianity or a fixed ``p``.

Scope (intentionally the smallest correct core): scalar hidden mode (q = 1),
``k_max = 2`` (mean + covariance), ReLU activation. Vector h, Edgeworth, dynamic
mode-splitting and projected K3/K4 are out of scope and reported as unsupported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .polyjet import PolyJet, jet_outer, fit_jet
from .hidden import HiddenCumulants
from .relu_moments import relu_m1, relu_m2_diag, relu_pair_matrix

# A network is a list of layers, each a tuple:
#   ("linear", W (out,in), b (out,) or None)      ("relu",)
Layer = tuple


@dataclass
class SymbolicConfig:
    """Knobs for symbolic hidden-mode kprop (defaults follow the spec)."""
    k_max: int = 2                       # mean + covariance (only value supported here)
    hidden_degree_initial: int = 6       # starting dh-polynomial degree p
    hidden_degree_max: int = 12          # adaptive ceiling
    hidden_tail_tol: float = 1e-4        # accept p when the tail score is below this
    hidden_tail_band: int = 2            # #top degrees counted in the tail
    auto_refine: bool = True             # bump p and re-propagate if the tail is fat
    full_covariance: bool = True         # track conditional off-diagonal covariance
    activation_method: str = "relu_gaussian_exact"
    n_gl: int = 64                       # Gauss-Legendre nodes for the pair moment
    collocation_span: float = 4.0        # dh collocation half-width (~4 sigma)
    collocation_oversample: int = 3      # M = oversample*p + 1 collocation nodes
    n_quad_margin: int = 6               # GH marginalization nodes = p + this
    var_floor: float = 1e-12
    device: str = "cpu"
    dtype: torch.dtype = torch.float64


@dataclass
class SymbolicState:
    """Conditional cumulants of one layer's activations, as jets in dh."""
    hidden: HiddenCumulants
    K: Dict[int, PolyJet]                # K[1]: [p+1, d]; K[2]: [p+1, d, d]
    dim: int
    k_max: int
    hidden_degree: int

    @property
    def K1(self) -> PolyJet:
        return self.K[1]

    @property
    def K2(self) -> PolyJet:
        return self.K[2]


# ---------------------------------------------------------------------------
# Input state for X ~ N(0, I) with a scalar latent h = V^T X
# ---------------------------------------------------------------------------
def make_input_state(input_dim: int, cfg: SymbolicConfig,
                     direction: Optional[Tensor] = None) -> SymbolicState:
    """Build the layer-0 conditional cumulants of ``X | h`` for ``X ~ N(0, I)``.

    direction V (unit, shape (d,)): scalar latent ``h = V^T X`` (q = 1). Then
    ``X | h ~ N(V h, I - V V^T)`` so K1(h) = V dh (degree 1), K2 = I - V V^T
    (degree 0), and ``h ~ N(0, 1)``. ``direction=None`` -> q = 0 (no hidden mode):
    K1 = 0, K2 = I, which reduces the algorithm to ordinary k=2 ReLU kprop.
    """
    p = cfg.hidden_degree_initial
    dev, dt = cfg.device, cfg.dtype
    K1 = PolyJet.zeros(p, (input_dim,), device=dev, dtype=dt)
    K2 = PolyJet.zeros(p, (input_dim, input_dim), device=dev, dtype=dt)
    eye = torch.eye(input_dim, device=dev, dtype=dt)
    if direction is None:
        K2.coeffs[0] = eye
        hidden = HiddenCumulants.none(device=dev, dtype=dt)
    else:
        V = torch.as_tensor(direction, device=dev, dtype=dt).reshape(-1)
        V = V / V.norm()
        K1.coeffs[1] = V
        K2.coeffs[0] = eye - torch.outer(V, V)
        hidden = HiddenCumulants.gaussian(var=1.0, device=dev, dtype=dt)
    return SymbolicState(hidden=hidden, K={1: K1, 2: K2}, dim=input_dim,
                         k_max=cfg.k_max, hidden_degree=p)


# ---------------------------------------------------------------------------
# Linear layer: exact cumulant pushforward
# ---------------------------------------------------------------------------
def linear_pushforward_symbolic(state: SymbolicState, W: Tensor,
                                b: Optional[Tensor]) -> SymbolicState:
    """Z = W A + b.  K1_out[a,o]=W K1[a]; K2_out[a]=W K2[a] W^T; bias to degree 0."""
    K1, K2 = state.K[1], state.K[2]
    K1o = torch.einsum("oi,ai->ao", W, K1.coeffs)
    if b is not None:
        K1o = K1o.clone()
        K1o[0] = K1o[0] + b
    K2o = torch.einsum("oi,aij,pj->aop", W, K2.coeffs, W)
    out_dim = W.shape[0]
    return SymbolicState(hidden=state.hidden, K={1: PolyJet(K1o), 2: PolyJet(K2o)},
                         dim=out_dim, k_max=state.k_max, hidden_degree=state.hidden_degree)


# ---------------------------------------------------------------------------
# ReLU activation: conditional-Gaussian closure as a jet (collocation)
# ---------------------------------------------------------------------------
def _collocation_nodes(p: int, cfg: SymbolicConfig) -> Tensor:
    M = cfg.collocation_oversample * p + 1
    if M == 1:
        return torch.zeros(1, device=cfg.device, dtype=cfg.dtype)
    k = torch.arange(M, device=cfg.device, dtype=cfg.dtype)
    return cfg.collocation_span * torch.cos(math.pi * k / (M - 1))   # Chebyshev extrema


def activation_relu_symbolic(state: SymbolicState, cfg: SymbolicConfig) -> SymbolicState:
    """K_A^(r)(h) from K_Z^(r)(h) under Z|h ~ N(mu(h), Cov(h)) (ReLU, k_max=2)."""
    if cfg.activation_method != "relu_gaussian_exact":
        raise NotImplementedError(
            f"activation_method={cfg.activation_method!r} not implemented "
            "(scalar-h k_max=2 scope supports 'relu_gaussian_exact' only)")
    p = state.hidden_degree
    d = state.dim
    K1, K2 = state.K[1], state.K[2]
    nodes = _collocation_nodes(p, cfg)
    M = nodes.shape[0]
    raw1 = torch.empty((M, d), device=cfg.device, dtype=cfg.dtype)
    raw2 = torch.empty((M, d, d), device=cfg.device, dtype=cfg.dtype)
    for k in range(M):
        t = float(nodes[k])
        mu = K1.eval(t)                                  # (d,)
        C = K2.eval(t)                                   # (d,d)
        C = 0.5 * (C + C.T)
        var = torch.clamp(torch.diagonal(C), min=cfg.var_floor)
        raw1[k] = relu_m1(mu, var, eps=cfg.var_floor)
        if cfg.full_covariance:
            raw2[k] = relu_pair_matrix(mu, C, n_gl=cfg.n_gl, span=12.0, eps=cfg.var_floor)
        else:
            # diagonal-only closure (spec-permitted first-impl simplification):
            # treat post-activation coords as conditionally uncorrelated.
            r1 = raw1[k]
            raw2[k] = torch.outer(r1, r1)
            raw2[k].diagonal().copy_(relu_m2_diag(mu, var, eps=cfg.var_floor))
    raw1_jet = fit_jet(nodes, raw1, p, cfg.collocation_span)
    raw2_jet = fit_jet(nodes, raw2, p, cfg.collocation_span)
    # raw moments -> cumulants (k_max = 2):  K2 = raw2 - raw1 (outer) raw1
    K1o = raw1_jet
    K2o = raw2_jet.sub(jet_outer(raw1_jet, raw1_jet, p))
    return SymbolicState(hidden=state.hidden, K={1: K1o, 2: K2o}, dim=d,
                         k_max=state.k_max, hidden_degree=p)


# ---------------------------------------------------------------------------
# Adaptive hidden-degree truncation diagnostics
# ---------------------------------------------------------------------------
def tail_diagnostics(state: SymbolicState, cfg: SymbolicConfig) -> dict:
    """Contribution-by-degree tail score of the mean jet (spec diagnostic)."""
    contrib = state.hidden.contribution_by_degree(state.K[1])      # (p+1,)
    total = float(contrib.sum()) + 1e-30
    p = state.hidden_degree
    band = min(cfg.hidden_tail_band, p)
    tail = float(contrib[p - band + 1:].sum()) if band >= 1 else 0.0
    score = tail / total
    return {"tail_score": score,
            "contribution_by_degree": contrib.detach().cpu().tolist(),
            "resolved": score <= cfg.hidden_tail_tol}


# ---------------------------------------------------------------------------
# Marginalize the hidden mode at the output
# ---------------------------------------------------------------------------
def marginalize_hidden_modes(state: SymbolicState, cfg: SymbolicConfig
                             ) -> Tuple[Tensor, Tensor]:
    """Unconditional (mean, cov) via the law of total covariance.

    Gaussian h: Gauss-Hermite quadrature over h (evaluate the jets at |h_k| <~ 4 sigma)
    -- stable, avoids summing coeff * E[dh^a] with rapidly growing high moments.
    No hidden mode: evaluate at dh = 0. Non-Gaussian: moment-weighted fallback.
    """
    K1, K2 = state.K[1], state.K[2]
    hidden = state.hidden
    if not hidden.has_mode():
        return K1.eval(0.0), 0.5 * (K2.eval(0.0) + K2.eval(0.0).T)
    if hidden.is_gaussian():
        p = state.hidden_degree
        n_quad = max(p + cfg.n_quad_margin, 8)
        nodes, w = hidden.gauss_hermite(n_quad)
        d = state.dim
        mean = torch.zeros(d, device=cfg.device, dtype=cfg.dtype)
        E_cov = torch.zeros(d, d, device=cfg.device, dtype=cfg.dtype)
        E_mmT = torch.zeros(d, d, device=cfg.device, dtype=cfg.dtype)
        for hk, wk in zip(nodes.tolist(), w.tolist()):
            m_h = K1.eval(hk)
            mean = mean + wk * m_h
            E_cov = E_cov + wk * K2.eval(hk)
            E_mmT = E_mmT + wk * torch.outer(m_h, m_h)
        cov = E_cov + (E_mmT - torch.outer(mean, mean))
        return mean, 0.5 * (cov + cov.T)
    # non-Gaussian fallback (moment-weighted; less stable at high degree)
    mv = hidden.moment_vector(2 * state.hidden_degree)
    mean = K1.expectation(mv)
    E_cov = K2.expectation(mv)
    E_mmT = jet_outer(K1, K1, 2 * state.hidden_degree).expectation(mv)
    cov = E_cov + (E_mmT - torch.outer(mean, mean))
    return mean, 0.5 * (cov + cov.T)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_once(layers: Sequence[Layer], input_dim: int, cfg: SymbolicConfig,
              direction: Optional[Tensor]) -> dict:
    state = make_input_state(input_dim, cfg, direction=direction)
    diags: List[dict] = []
    for layer in layers:
        if layer[0] == "linear":
            _, W, b = layer
            state = linear_pushforward_symbolic(state, W, b)
        elif layer[0] == "relu":
            state = activation_relu_symbolic(state, cfg)
            state.K[1] = state.K[1].truncate(state.hidden_degree)
            state.K[2] = state.K[2].truncate(state.hidden_degree)
            diags.append(tail_diagnostics(state, cfg))
        else:
            raise ValueError(f"unknown layer kind {layer[0]!r}")
    mean, cov = marginalize_hidden_modes(state, cfg)
    unresolved = [i for i, dg in enumerate(diags) if not dg["resolved"]]
    return {"mean": mean, "cov": cov, "layer_diagnostics": diags,
            "hidden_degree": cfg.hidden_degree_initial,
            "unresolved_layers": unresolved, "final_state": state}


def symbolic_hidden_mode_kprop(layers: Sequence[Layer], input_dim: int,
                               cfg: Optional[SymbolicConfig] = None, *,
                               direction: Optional[Tensor] = None) -> dict:
    """Run symbolic hidden-mode kprop through ``layers`` for ``X ~ N(0, I)``.

    Returns a dict: ``mean`` (out_dim,), ``cov`` (out_dim, out_dim),
    ``layer_diagnostics`` (per-ReLU tail scores), ``unresolved_layers``,
    ``hidden_degree`` (the p actually used), and ``approximations`` (reported, per
    the spec's design rule -- the method never silently hides where it approximates).

    If ``cfg.auto_refine`` and any layer's tail is above ``hidden_tail_tol``, the
    whole propagation is re-run at a larger hidden degree (up to ``hidden_degree_max``).
    """
    if cfg is None:
        cfg = SymbolicConfig()
    if cfg.k_max != 2:
        raise NotImplementedError("scalar-h scope supports k_max=2 only")
    p = cfg.hidden_degree_initial
    res = _run_once(layers, input_dim, cfg, direction)
    if cfg.auto_refine and direction is not None:
        while res["unresolved_layers"] and p < cfg.hidden_degree_max:
            p = min(p + 2, cfg.hidden_degree_max)
            cfg = _with_degree(cfg, p)
            res = _run_once(layers, input_dim, cfg, direction)
        res["hidden_degree"] = p
    res["approximations"] = _approximation_report(cfg, res)
    return res


def _with_degree(cfg: SymbolicConfig, p: int) -> SymbolicConfig:
    from dataclasses import replace
    return replace(cfg, hidden_degree_initial=p)


def _approximation_report(cfg: SymbolicConfig, res: dict) -> List[str]:
    notes = ["ReLU conditional-Gaussian residual closure (k_max=2): the only "
             "approximate step; exact for linear layers."]
    if not cfg.full_covariance:
        notes.append("full_covariance=False: conditional off-diagonal covariance "
                     "dropped at activations (depth>1 mean is approximate).")
    if res["unresolved_layers"]:
        notes.append(f"hidden-degree tail NOT resolved at layers "
                     f"{res['unresolved_layers']} (p={res['hidden_degree']} hit "
                     f"hidden_degree_max or auto_refine off): retained hidden "
                     f"cumulants may be insufficient -- raise hidden_degree_max.")
    notes.append("scalar hidden mode (q=1); vector h, Edgeworth, dynamic "
                 "mode-splitting, projected K3/K4 are out of scope.")
    return notes

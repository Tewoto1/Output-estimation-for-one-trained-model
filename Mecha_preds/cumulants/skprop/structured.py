"""structured.py -- structured power KPROP: spike-aware cumulant propagation.

The failure mode this fixes: a meaned/spiked weight matrix ``A = W + U S V^T``
turns a SHARED low-dimensional latent into coherent O(1) off-diagonal cumulant
structure (Cov(X_i, X_j) ~ rank-one, O(1)), which vanilla kprop's harmonic
truncation assumes is O(n^{-1/2}) -- so its error stops shrinking with width.
Power cumulants alone do not fix this: they handle repeated-index (diagonal)
terms, not the coherent latent.

The fix implemented here (the "right division of labor"):

    track the spike-selected latent EXPLICITLY            (this module)
    + power cumulants for the residual noise around it     (vendored kprop)

Algorithm (input latents -- exact conditioning):
  1. Find the latent directions ``V`` (q <= q_max) of the first weight matrix
     (``spikes.detect_spikes``), or take them from ``directions=``.
  2. Condition the Gaussian input on ``H = V^T X = h``: X|h ~ N(mu_h, Sigma_h)
     with an EXACTLY Gaussian conditional law. Conditioned on the latent, the
     coherent structure moves into the conditional MEAN, which kprop carries
     exactly through every linear layer; the conditional residual is incoherent
     again, so the vendored power-cumulant machinery (use_pK) controls it to
     the usual O(n^{-k_max/2}) amplitude.
  3. Run vanilla ``mlp_kprop`` once per Gauss--Hermite node h_k and mix:
     E[f(out)] = sum_k w_k E[f(out) | h_k]. Mixing means is exact; the
     quadrature error is spectrally small in n_nodes for smooth latents.

Optional deep mode (``deep=True``, k_max==2 only): spikes in HIDDEN layers
amplify residual noise into a new O(1) latent that input conditioning cannot
see. There we condition the (approximately Gaussian) preactivation on the
spike's left direction u -- Gaussian conditioning of the tracked (mean, cov)
state -- and branch over nodes again. This is the conditional-CLT approximation
of the writeup: the residual error budget is

    amplitude error <= C * ( n^{-k_max/2} + quadrature + E_condCLT ).

If no spike is detected (q=0) the algorithm degenerates to vanilla kprop
EXACTLY (one node, weight 1).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from numpy.polynomial.hermite import hermgauss

from ..kprop import Kind, MLP as KpropMLP, WICK_COEF_D, mlp_kprop, linear_kprop, nonlin_kprop, coerce_input
from ..kprop.harmonic import HTensor
from .spikes import SpikeInfo, detect_spikes, detect_spikes_all_layers, orthonormalize

logger = logging.getLogger(__name__)

_TINY = 1e-14


# ---------------------------------------------------------------------------
# Quadrature over the latent
# ---------------------------------------------------------------------------
def gauss_hermite_grid(
    n_nodes: int, q: int, *, weight_tol: float = 1e-12,
    device=None, dtype=torch.float64,
) -> Tuple[Tensor, Tensor]:
    """Tensor-product Gauss--Hermite grid for ``H ~ N(0, I_q)``.

    Returns (nodes (K, q), weights (K,)), weights summing to ~1; nodes with
    product weight < weight_tol are pruned (matters for q >= 2).
    """
    t, om = hermgauss(n_nodes)
    h1 = torch.as_tensor(t, device=device, dtype=dtype) * math.sqrt(2.0)
    w1 = torch.as_tensor(om, device=device, dtype=dtype) / math.sqrt(math.pi)
    nodes, weights = h1[:, None], w1
    for _ in range(q - 1):
        nodes = torch.cat(
            [nodes.repeat_interleave(n_nodes, 0), h1.repeat(nodes.shape[0])[:, None]], dim=1
        )
        weights = (weights[:, None] * w1[None, :]).reshape(-1)
    keep = weights > weight_tol
    return nodes[keep], weights[keep] / weights[keep].sum()


# ---------------------------------------------------------------------------
# Gaussian conditioning
# ---------------------------------------------------------------------------
def condition_gaussian_on_subspace(
    mu: Tensor, Sigma: Tensor, V: Tensor, h: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Law of ``X ~ N(mu, Sigma)`` given that the standardized latent of
    ``T = V^T X`` equals ``h`` (i.e. T = E[T] + chol(Cov T) h).

    For mu=0, Sigma=I, orthonormal V this is N(V h, I - V V^T).
    """
    C = V.T @ Sigma @ V                              # (q, q) latent covariance
    L = torch.linalg.cholesky(C + _TINY * torch.eye(C.shape[0], dtype=C.dtype, device=C.device))
    G = Sigma @ V                                    # (n, q)
    # mu_c = mu + G C^{-1} (tau - V^T mu), tau - V^T mu = L h  =>  G C^{-1} L h = G L^{-T} h
    mu_c = mu + G @ torch.linalg.solve_triangular(L.T, h[:, None], upper=True)[:, 0]
    Sigma_c = Sigma - G @ torch.cholesky_solve(G.T, L)
    return mu_c, 0.5 * (Sigma_c + Sigma_c.T)


def _condition_gaussian_on_direction(
    mu: Tensor, Sigma: Tensor, u: Tensor, h: float,
) -> Tuple[Tensor, Tensor, float]:
    """Condition N(mu, Sigma) on ``u . x = u . mu + sqrt(u' Sigma u) * h``.

    Returns (mu_c, Sigma_c, s2) with s2 = u' Sigma u (s2 ~ 0 => no-op).
    """
    g = Sigma @ u
    s2 = float(u @ g)
    if s2 <= _TINY:
        return mu, Sigma, s2
    mu_c = mu + g * (h / math.sqrt(s2))
    Sigma_c = Sigma - torch.outer(g, g) / s2
    return mu_c, 0.5 * (Sigma_c + Sigma_c.T), s2


# ---------------------------------------------------------------------------
# The structured algorithm
# ---------------------------------------------------------------------------
def structured_mlp_kprop(
    mlp: KpropMLP,
    K_in: Dict[int, Tensor],
    k_max: int,
    *,
    directions: Optional[Tensor] = None,   # (input_dim, q) explicit latent directions
    q_max: int = 1,
    margin: float = 1.15,
    n_nodes: int = 15,
    weight_tol: float = 1e-12,
    deep: bool = False,
    deep_layers: Optional[Sequence[int]] = None,
    deep_directions: Optional[Dict[int, Tensor]] = None,
    deep_n_nodes: int = 9,
    deep_q_max: int = 1,
    deep_margin: float = 1.3,
    output_d_max: int = 1,
    kind: Kind = Kind.SIMPLE,
    use_avg_metric: bool = False,
    factor: bool = False,
    use_pK: bool = True,
    exact_relu_cov: bool = False,
    progress: bool = False,
) -> dict:
    """Structured power KPROP through a kprop ``MLP``.

    Args:
        K_in: GAUSSIAN input cumulants {1: mean (d,), 2: cov (d, d)} (orders >= 3
            are not supported in structured mode -- the conditioning step is
            exact only for a Gaussian input).
        directions: explicit latent directions in INPUT space (skips detection).
        deep: also condition on spike noise channels of HIDDEN layers
            (k_max == 2 only; conditional-CLT approximation, see module docstring).
        Remaining kwargs mirror ``mlp_kprop``.

    Returns dict with:
        mean (out_dim,), K2 (out_dim, out_dim) if output_d_max >= 2 and tracked,
        spikes (input-layer SpikeInfo), deep_spikes, n_branches,
        nodes/weights/per_node_means (input quadrature; not in deep mode).
    """
    if any(d > 2 for d in K_in):
        raise ValueError("structured_mlp_kprop requires a Gaussian input (cumulant orders 1, 2)")
    if deep and k_max != 2:
        raise NotImplementedError("deep=True conditions the tracked (mean, cov) state; only k_max=2")
    if deep and factor:
        raise NotImplementedError("deep=True with factor=True is not supported")

    W0 = mlp.Ws[0].weight
    device, dtype = W0.device, W0.dtype
    input_dim = W0.shape[1]
    Sigma0 = torch.as_tensor(K_in[2], device=device, dtype=dtype)
    mu0 = torch.as_tensor(
        K_in.get(1, torch.zeros(input_dim)), device=device, dtype=dtype
    )

    # -- 1. latent directions -------------------------------------------------
    if directions is not None:
        V = orthonormalize(torch.as_tensor(directions, device=device, dtype=dtype))
        spike0 = None
    else:
        spike0 = detect_spikes(W0, q_max=q_max, margin=margin, layer=0)
        V = spike0.V.to(device=device, dtype=dtype)
    q = V.shape[1] if V.numel() else 0

    kprop_kwargs = dict(kind=kind, use_avg_metric=use_avg_metric, factor=factor,
                        use_pK=use_pK, exact_relu_cov=exact_relu_cov)

    # -- 2. input quadrature branches -----------------------------------------
    if q == 0:
        nodes = torch.zeros(1, 0, device=device, dtype=dtype)
        weights = torch.ones(1, device=device, dtype=dtype)
        cond_laws = [(mu0, Sigma0)]
    else:
        nodes, weights = gauss_hermite_grid(n_nodes, q, weight_tol=weight_tol,
                                            device=device, dtype=dtype)
        cond_laws = [condition_gaussian_on_subspace(mu0, Sigma0, V, h) for h in nodes]

    deep_spikes: List[SpikeInfo] = []
    if deep:
        if deep_directions is not None:
            # Explicit conditioning channels per LINEAR-layer index (e.g. the left
            # singular vectors of DeltaW_l = W_l - W_l^init, which raw-W detection
            # misses because the random bulk masks them). Directions live in the
            # OUTPUT space of layer l (the preactivation conditioned before phi).
            for l, U in deep_directions.items():
                if not (1 <= l < len(mlp.nonlins) + 1) or l >= len(mlp.Ws):
                    raise ValueError(f"deep_directions layer {l} out of range")
                Uq = orthonormalize(torch.as_tensor(U, device=device, dtype=dtype))
                deep_spikes.append(SpikeInfo(layer=l, q=Uq.shape[1], U=Uq,
                                             s=torch.zeros(Uq.shape[1]), V=Uq,
                                             sigma_hat=0.0, bulk_edge=0.0))
        elif deep_layers is None:
            cands = detect_spikes_all_layers(
                [W.weight for W in mlp.Ws[1:len(mlp.nonlins)]],
                q_max=deep_q_max, margin=deep_margin)
            deep_spikes = [s for s in cands if s.q > 0]
            for s in deep_spikes:
                s.layer += 1  # offset: detection ran on Ws[1:]
        else:
            deep_spikes = [detect_spikes(mlp.Ws[l].weight, q_max=deep_q_max,
                                         margin=deep_margin, layer=l) for l in deep_layers]
            deep_spikes = [s for s in deep_spikes if s.q > 0]
    deep_by_layer = {s.layer: s for s in deep_spikes}

    out_dim = mlp.Ws[-1].weight.shape[0]
    acc = {
        "mean": torch.zeros(out_dim, device=device, dtype=dtype),
        "m2": torch.zeros(out_dim, out_dim, device=device, dtype=dtype) if output_d_max >= 2 else None,
        "weight": 0.0, "branches": 0, "per_node_means": [],
    }

    def _accumulate(K_out: dict, bw: float, record_node_mean: bool = False) -> None:
        mu_b = K_out[1].to_tensor() if hasattr(K_out[1], "to_tensor") else torch.as_tensor(K_out[1])
        acc["mean"] += bw * mu_b
        if record_node_mean:
            acc["per_node_means"].append(mu_b.detach().clone())
        if acc["m2"] is not None and 2 in K_out:
            S_b = K_out[2].to_tensor() if hasattr(K_out[2], "to_tensor") else torch.as_tensor(K_out[2])
            acc["m2"] += bw * (S_b + torch.outer(mu_b, mu_b))
        acc["weight"] += bw
        acc["branches"] += 1

    # -- 3a. flat path: one vanilla kprop run per input node ------------------
    if not deep_by_layer:
        it = zip(weights, cond_laws)
        for w_k, (mu_c, Sig_c) in it:
            K_out = mlp_kprop(mlp, {1: mu_c, 2: Sig_c}, k_max=k_max,
                              output_d_max=output_d_max, **kprop_kwargs)
            _accumulate(K_out, float(w_k), record_node_mean=True)
    # -- 3b. deep path: per-layer loop with mid-network conditioning ----------
    else:
        nonlin_coefs = [WICK_COEF_D[name] for name in mlp.nonlin_names]
        n_linear = len(mlp.Ws)

        def propagate(K, l: int, bw: float) -> None:
            while l < n_linear:
                W_mod = mlp.Ws[l]
                if l == len(mlp.nonlins):           # readout: linear only, then done
                    K = linear_kprop(K, W_mod.weight, k_max=k_max,
                                     d_max=output_d_max, bias=W_mod.bias)
                    _accumulate(K, bw)
                    return
                metric = mlp.init_scale[l] if use_avg_metric else None
                WK = linear_kprop(K, W_mod.weight, k_max=k_max,
                                  set_metric=metric, bias=W_mod.bias)
                spike = deep_by_layer.get(l)
                if spike is not None and spike.q > 0:
                    mu_z, Sig_z = WK[1].core, WK[2].core
                    # keep only channels with non-degenerate noise on THIS branch
                    chans = [spike.U[:, j].to(device=device, dtype=dtype)
                             for j in range(spike.q)]
                    chans = [u for u in chans if float(u @ (Sig_z @ u)) > _TINY]
                    if chans:
                        grid, gw = gauss_hermite_grid(deep_n_nodes, len(chans),
                                                      device=device, dtype=dtype)
                        for h_vec, w_j in zip(grid, gw):
                            mu_c, Sig_c = mu_z, Sig_z
                            for u, h_ju in zip(chans, h_vec):
                                mu_c, Sig_c, _ = _condition_gaussian_on_direction(
                                    mu_c, Sig_c, u, float(h_ju))
                            K_c = {1: HTensor(mu_c, r=0), 2: HTensor(Sig_c, r=0)}
                            K_a = nonlin_kprop(K_c, nonlin_coefs[l], k_max=k_max,
                                               kind=kind, use_pK=use_pK,
                                               exact_relu_cov=exact_relu_cov)
                            propagate(K_a, l + 1, bw * float(w_j))
                        return                      # sub-branches handled recursively
                K = nonlin_kprop(WK, nonlin_coefs[l], k_max=k_max, kind=kind,
                                 use_pK=use_pK, exact_relu_cov=exact_relu_cov)
                l += 1
            _accumulate(K, bw)                      # nonlin-terminated MLP (no readout)

        for w_k, (mu_c, Sig_c) in zip(weights, cond_laws):
            K0 = coerce_input({1: mu_c, 2: Sig_c}, k_max=k_max, kind=kind)
            propagate(K0, 0, float(w_k))

    if abs(acc["weight"] - 1.0) > 1e-6:
        logger.warning("branch weights sum to %.6f (pruning/degenerate channels); renormalizing",
                       acc["weight"])
        acc["mean"] /= acc["weight"]
        if acc["m2"] is not None:
            acc["m2"] /= acc["weight"]

    out = {
        "mean": acc["mean"],
        "spikes": spike0,
        "directions": V,
        "q": q,
        "deep_spikes": deep_spikes,
        "n_branches": acc["branches"],
        "nodes": nodes,
        "weights": weights,
    }
    if acc["per_node_means"]:
        out["per_node_means"] = torch.stack(acc["per_node_means"])
    if acc["m2"] is not None:
        out["K2"] = acc["m2"] - torch.outer(acc["mean"], acc["mean"])
    return out

"""empirical_structure.py -- empirical test of the "binless" structure hypothesis for
the coordinate-spike case ``M = W + e_1 e_1^T`` (spike on hidden coordinate 0).

THE SUPPOSITION (what this module measures)
-------------------------------------------
binned_kprop stores, per spike bin ``alpha``, a conditional BULK law ``(mu_alpha, Sigma_alpha)``
and pays ``O(num_bins * d^3)`` to congruence each ``Sigma_alpha`` through the next linear map.
The supposition is that binning may be UNNECESSARY because, ACROSS bins, that conditional law
is a smooth low-rank family in the spike value ``a`` (= E[spike | bin]):

    mean:        mu(a)    ~=  mu0 + a * c                (LINEAR in a; c a fixed direction)
    covariance:  Sigma(a) ~=  Sigma0 + f(a) * v v^T      (RANK-1 correction; v a fixed direction)

Where it comes from: writing the pre-activation bulk as ``z_bulk = u * A + V * B`` (``u = M[1:,0]``
is the "w_i e_1" leak of the spike ``A`` into the other coordinates, ``V = M[1:,1:]``), conditioning
on the spike shifts the bulk mean linearly and leaves a within-bin residual spike-variance
``sigma_a^2`` that bumps the covariance by ``sigma_a^2 u u^T`` -- rank one. Exact for the linear
step under joint Gaussianity; the ReLU + depth composition is what this module probes empirically.

If the family holds, the per-bin ``(mu_alpha, Sigma_alpha)`` collapse to ``(mu0, c, Sigma0, v, f)``
and the next-layer congruence becomes ``V Sigma0 V^T`` (once) ``+ f(a) (V v)(V v)^T`` (O(d^2)/bin):
a binless, analytic e_1-parametrised predictor.

NOISE MATTERS. Each ``Sigma_alpha`` is a ``d x d`` empirical covariance; the bin-to-bin DIFFERENCE
is a small signal under ``O(1/sqrt(N))`` covariance noise, which inflates every eigenvalue and would
make any covariance look full-rank. So the covariance test uses SPLIT-HALF unbiased estimators
(cross-products of two independent sample halves -> the noise cross-terms vanish in expectation),
the same debiasing this repo uses in the S/V cumulant-scaling work. It also reports a signal-SIZE
metric first: does the covariance vary across bins AT ALL (beyond noise), before asking if it is rank-1.

WHAT THIS MODULE PROVIDES
-------------------------
* ``build_spiked_net``        -- random ``N(0,1/n)`` hidden matrices + ``theta e_c e_c^T`` spike.
* ``empirical_binned_states`` -- stream ``X ~ N(0,I_n)``; at each hidden layer bin the PRE-activation
                                 spike coord into equal-mass bins; accumulate conditional bulk
                                 ``(p, a, mu, Sigma)`` for the pre-ReLU AND post-ReLU representation,
                                 in two independent sample halves (``mu_A/mu_B, Sigma_A/Sigma_B``).
* ``mean_linearity``          -- (split-half debiased) R^2 of ``mu(a) ~ affine(a)`` + slope dir c.
* ``cov_rank1_structure``     -- (split-half debiased) signal size, family coherence, common dir v,
                                 direction alignment, magnitude ``f(a)`` vs a.
* ``spike_coupling_column``   -- theory direction ``u = M[1:,0]`` to compare c and v against.
* ``structure_report`` / ``summarize_report`` -- run everything per layer -> dict / table.

The ``(p, a, mu, Sigma)`` layout is exactly ``binned_kprop.core.BinnedK2State`` with ``d = n - 1``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "build_spiked_net",
    "empirical_binned_states",
    "mean_linearity",
    "cov_rank1_structure",
    "spike_coupling_column",
    "structure_report",
    "summarize_report",
]

_EPS = 1e-30


# --------------------------------------------------------------------------- #
# model: e_1 e_1^T shift of a random matrix (per the repo's convention)
# --------------------------------------------------------------------------- #
def build_spiked_net(n: int, depth: int, seed: int = 0, *, theta: float = 1.0,
                     out_dim: int = 1, spike_coord: int = 0
                     ) -> List[Tuple[np.ndarray, None]]:
    """``depth`` square hidden matrices ``W ~ N(0, 1/n)`` each with a coordinate spike
    ``W[c, c] += theta`` (so ``M = W + theta e_c e_c^T``), plus a random ``(out_dim, n)``
    readout. Returns ``[(W, None), ...]`` in forward order (W shape (out, in), bias None).
    ``theta = 1`` is the plain ``e_1 e_1^T`` case."""
    rng = np.random.default_rng(seed)
    Ws: List[Tuple[np.ndarray, None]] = []
    for _ in range(depth):
        W = rng.standard_normal((n, n)) / np.sqrt(n)
        W[spike_coord, spike_coord] += theta
        Ws.append((W, None))
    Ws.append((rng.standard_normal((out_dim, n)) / np.sqrt(n), None))
    return Ws


def spike_coupling_column(M: np.ndarray, spike_coord: int = 0) -> np.ndarray:
    """Theory-predicted coupling direction ``u = M[bulk, spike_coord]`` -- the column that leaks
    the spike ``A`` into the other (bulk) coordinates (the "w_i e_1" term). Length d = n-1."""
    M = np.asarray(M, dtype=np.float64)
    bulk = [i for i in range(M.shape[0]) if i != spike_coord]
    return M[np.ix_(bulk, [spike_coord])].reshape(-1)


# --------------------------------------------------------------------------- #
# empirical per-bin conditional bulk moments (streaming, split-half)
# --------------------------------------------------------------------------- #
def _equal_mass_edges(samples: np.ndarray, num_bins: int) -> np.ndarray:
    """Equal-probability-mass bin edges (quantiles of the observed spike samples), with +/- inf
    outer edges so the tails always land in a bin; interior ties nudged apart."""
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(samples, qs)
    edges = np.maximum.accumulate(edges)
    edges[0], edges[-1] = -np.inf, np.inf
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


class _Accum:
    """Streaming per-bin (count, sum spike, sum bulk, sum bulk-outer-bulk) for one representation
    and one sample HALF."""

    def __init__(self, num_bins: int, d: int):
        self.cnt = np.zeros(num_bins, dtype=np.float64)
        self.ssp = np.zeros(num_bins, dtype=np.float64)
        self.sb = np.zeros((num_bins, d), dtype=np.float64)
        self.sbb = np.zeros((num_bins, d, d), dtype=np.float64)

    def add(self, bin_idx: np.ndarray, spike: np.ndarray, bulk: np.ndarray) -> None:
        # sort once, then accumulate each bin from a CONTIGUOUS slice (one BLAS GEMM per bin) --
        # avoids re-scanning the whole batch with a boolean mask num_bins times.
        nb = self.cnt.shape[0]
        order = np.argsort(bin_idx, kind="stable")
        bs = bin_idx[order]; Bs = bulk[order]; sps = spike[order]
        bounds = np.searchsorted(bs, np.arange(nb + 1))
        for b in range(nb):
            s, e = int(bounds[b]), int(bounds[b + 1])
            if e > s:
                seg = Bs[s:e]
                self.cnt[b] += (e - s)
                self.ssp[b] += float(sps[s:e].sum())
                self.sb[b] += seg.sum(axis=0)
                self.sbb[b] += seg.T @ seg

    def raw(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.cnt, self.ssp, self.sb, self.sbb


def _finalize_raw(cnt, ssp, sb, sbb) -> Dict[str, np.ndarray]:
    total = cnt.sum()
    p = cnt / total if total > 0 else cnt.copy()
    safe = np.where(cnt > 0, cnt, 1.0)
    mu = sb / safe[:, None]
    a = np.where(cnt > 0, ssp / safe, 0.0)
    Sigma = np.empty_like(sbb)
    for b in range(cnt.shape[0]):
        if cnt[b] > 0:
            S = sbb[b] / cnt[b] - np.outer(mu[b], mu[b])
            Sigma[b] = 0.5 * (S + S.T)
        else:
            Sigma[b] = 0.0
    return {"p": p, "a": a, "mu": mu, "Sigma": Sigma}


def _finalize_split(accA: _Accum, accB: _Accum) -> Dict[str, np.ndarray]:
    """Finalize a representation into full + per-half states (A/B from disjoint sample halves)."""
    rawA, rawB = accA.raw(), accB.raw()
    A = _finalize_raw(*rawA)
    B = _finalize_raw(*rawB)
    full = _finalize_raw(*[a + b for a, b in zip(rawA, rawB)])
    full["mu_A"], full["mu_B"] = A["mu"], B["mu"]
    full["Sigma_A"], full["Sigma_B"] = A["Sigma"], B["Sigma"]
    full["p_A"], full["p_B"] = A["p"], B["p"]
    return full


def empirical_binned_states(Ws: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]], n: int, *,
                            num_bins: int = 21, spike_coord: int = 0,
                            n_samples: int = 2_000_000, n_edge_samples: int = 200_000,
                            batch: int = 100_000, seed: int = 0,
                            backend: str = "numpy") -> Dict[str, object]:
    """Stream ``X ~ N(0, I_n)`` through the coordinate-spiked ReLU MLP ``Ws``. At each hidden layer,
    bin the PRE-activation spike coordinate into ``num_bins`` equal-mass bins, and accumulate the
    conditional bulk mean & covariance of BOTH:
        * the pre-ReLU bulk   ``z[:, bulk]``            (the linear-step law), and
        * the post-ReLU bulk  ``relu(z)[:, bulk]``      (what relu_step_k2 produces per bin),
    conditioned on the SAME pre-activation spike bin, in TWO independent sample halves (for
    split-half noise debiasing downstream).

    Returns ``{"edges": [L arrays], "pre": [L states], "post": [L states], ...}`` where each state
    is ``{p, a, mu, Sigma, mu_A, mu_B, Sigma_A, Sigma_B, p_A, p_B}`` with ``d = n - 1``.
    ``a`` = E[pre-activation spike | bin] (the binning variable, shared by pre and post).

    ``backend="numpy"`` (default, torch-free) or ``"torch"`` (GPU; numpy fallback). Memory
    ~ O(L * num_bins * d^2). Split into halves by row parity within each batch."""
    Ws = [(np.asarray(W, dtype=np.float64), b) for (W, b) in Ws]
    L = len(Ws) - 1
    d = n - 1
    bulk_idx = np.array([i for i in range(n) if i != spike_coord])

    if backend == "torch":
        try:
            return _empirical_binned_states_torch(
                Ws, n, num_bins=num_bins, spike_coord=spike_coord, bulk_idx=bulk_idx,
                n_samples=n_samples, n_edge_samples=n_edge_samples, batch=batch, seed=seed)
        except Exception as e:  # pragma: no cover
            print(f"  (torch backend unavailable -> numpy): {e}")

    Wh = [Ws[li][0] for li in range(L)]

    # ---- pass 1: equal-mass edges on the pre-activation spike coord, per hidden layer ----
    rng = np.random.default_rng(seed)
    cols: List[List[np.ndarray]] = [[] for _ in range(L)]
    got = 0
    while got < n_edge_samples:
        b = min(batch, n_edge_samples - got)
        h = rng.standard_normal((b, n))
        for li in range(L):
            z = h @ Wh[li].T
            cols[li].append(z[:, spike_coord].copy())
            h = np.maximum(z, 0.0)
        got += b
    edges = [_equal_mass_edges(np.concatenate(cols[li]), num_bins) for li in range(L)]

    # ---- pass 2: accumulate conditional bulk moments (pre & post) in two halves ----
    preA = [_Accum(num_bins, d) for _ in range(L)]
    preB = [_Accum(num_bins, d) for _ in range(L)]
    postA = [_Accum(num_bins, d) for _ in range(L)]
    postB = [_Accum(num_bins, d) for _ in range(L)]
    rng = np.random.default_rng(seed + 1)
    got = 0
    while got < n_samples:
        b = min(batch, n_samples - got)
        h = rng.standard_normal((b, n))
        half = (np.arange(b) % 2) == 0                    # A = even rows, B = odd rows
        for li in range(L):
            z = h @ Wh[li].T
            spike = z[:, spike_coord]
            bin_idx = np.searchsorted(edges[li], spike, side="right") - 1
            np.clip(bin_idx, 0, num_bins - 1, out=bin_idx)
            zr = np.maximum(z, 0.0)
            zb, zrb = z[:, bulk_idx], zr[:, bulk_idx]
            preA[li].add(bin_idx[half], spike[half], zb[half])
            preB[li].add(bin_idx[~half], spike[~half], zb[~half])
            postA[li].add(bin_idx[half], spike[half], zrb[half])
            postB[li].add(bin_idx[~half], spike[~half], zrb[~half])
            h = zr
        got += b

    return {
        "edges": edges,
        "pre": [_finalize_split(preA[li], preB[li]) for li in range(L)],
        "post": [_finalize_split(postA[li], postB[li]) for li in range(L)],
        "spike_coord": spike_coord, "n": n, "num_bins": num_bins, "depth": L,
    }


def _empirical_binned_states_torch(Ws, n, *, num_bins, spike_coord, bulk_idx,
                                   n_samples, n_edge_samples, batch, seed):  # pragma: no cover
    """torch backend (GPU). Two things matter for speed on a T4:
      (1) fp32 compute -- the T4 runs fp64 GEMMs ~32x slower; we accumulate in fp64 for accuracy.
      (2) per-bin covariance via ONE argsort + contiguous-segment GEMMs (shared by pre & post,
          one CPU sync per layer-half) instead of a num_bins-way boolean-mask loop of tiny kernels.
    Numerically matches the numpy path up to fp32 rounding + MC noise."""
    import numpy as _np
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32                                   # compute dtype (was float64: the T4 killer)
    L = len(Ws) - 1
    d = n - 1
    Wh = [torch.as_tensor(Ws[li][0], dtype=dt, device=dev) for li in range(L)]
    bulk_t = torch.as_tensor(bulk_idx, device=dev)

    # ---- pass 1: equal-mass edges on the pre-activation spike coord, per hidden layer ----
    g = torch.Generator(device=dev).manual_seed(seed)
    cols = [[] for _ in range(L)]
    got = 0
    while got < n_edge_samples:
        b = min(batch, n_edge_samples - got)
        h = torch.randn(b, n, generator=g, dtype=dt, device=dev)
        for li in range(L):
            z = h @ Wh[li].T
            cols[li].append(z[:, spike_coord].detach().clone())
            h = torch.relu(z)
        got += b
    qs = torch.linspace(0, 1, num_bins + 1, dtype=dt, device=dev)
    edges = []
    for li in range(L):
        e = torch.cummax(torch.quantile(torch.cat(cols[li]), qs), 0).values
        e[0], e[-1] = float("-inf"), float("inf")
        edges.append(e.contiguous())

    # ---- fp64 accumulators, one dict per (rep, half) ----
    def new():
        return dict(cnt=torch.zeros(num_bins, dtype=torch.float64, device=dev),
                    ssp=torch.zeros(num_bins, dtype=torch.float64, device=dev),
                    sb=torch.zeros(num_bins, d, dtype=torch.float64, device=dev),
                    sbb=torch.zeros(num_bins, d, d, dtype=torch.float64, device=dev))
    accs = {(rep, hf): [new() for _ in range(L)] for rep in ("pre", "post") for hf in ("A", "B")}

    g = torch.Generator(device=dev).manual_seed(seed + 1)
    got = 0
    while got < n_samples:
        b = min(batch, n_samples - got)
        h = torch.randn(b, n, generator=g, dtype=dt, device=dev)
        half = (torch.arange(b, device=dev) % 2) == 0
        for li in range(L):
            z = h @ Wh[li].T
            spike = z[:, spike_coord]
            bi = torch.clamp(torch.searchsorted(edges[li], spike.contiguous(), right=True) - 1,
                             0, num_bins - 1)                      # same binning as the numpy path
            zr = torch.relu(z)
            Bpre = z.index_select(1, bulk_t)
            Bpost = zr.index_select(1, bulk_t)
            for hf, msk in (("A", half), ("B", ~half)):
                idx = msk.nonzero(as_tuple=True)[0]               # rows of this half
                bim = bi.index_select(0, idx)
                order = torch.argsort(bim)
                sidx = idx.index_select(0, order)                 # global rows, sorted by bin
                counts = torch.bincount(bim, minlength=num_bins).cpu().numpy()   # 1 sync / layer-half
                off = _np.concatenate([[0], counts.cumsum()]).astype(int)
                Bpre_s = Bpre.index_select(0, sidx)
                Bpost_s = Bpost.index_select(0, sidx)
                sps_s = spike.index_select(0, sidx)
                accP = accs[("pre", hf)][li]; accQ = accs[("post", hf)][li]
                for bb in range(num_bins):
                    s, e = int(off[bb]), int(off[bb + 1])
                    if e <= s:
                        continue
                    cnt_be = float(e - s)
                    ssp_be = sps_s[s:e].sum().double()
                    segP, segQ = Bpre_s[s:e], Bpost_s[s:e]
                    accP["cnt"][bb] += cnt_be; accP["ssp"][bb] += ssp_be
                    accP["sb"][bb] += segP.sum(0).double(); accP["sbb"][bb] += (segP.t() @ segP).double()
                    accQ["cnt"][bb] += cnt_be; accQ["ssp"][bb] += ssp_be
                    accQ["sb"][bb] += segQ.sum(0).double(); accQ["sbb"][bb] += (segQ.t() @ segQ).double()
            h = zr
        got += b

    def to_np(acc):
        return (acc["cnt"].cpu().numpy(), acc["ssp"].cpu().numpy(),
                acc["sb"].cpu().numpy(), acc["sbb"].cpu().numpy())

    def fin_split(li, rep):
        rawA, rawB = to_np(accs[(rep, "A")][li]), to_np(accs[(rep, "B")][li])
        A, B = _finalize_raw(*rawA), _finalize_raw(*rawB)
        full = _finalize_raw(*[x + y for x, y in zip(rawA, rawB)])
        full["mu_A"], full["mu_B"] = A["mu"], B["mu"]
        full["Sigma_A"], full["Sigma_B"] = A["Sigma"], B["Sigma"]
        full["p_A"], full["p_B"] = A["p"], B["p"]
        return full

    return {
        "edges": [e.double().cpu().numpy() for e in edges],
        "pre": [fin_split(li, "pre") for li in range(L)],
        "post": [fin_split(li, "post") for li in range(L)],
        "spike_coord": spike_coord, "n": n, "num_bins": num_bins, "depth": L,
    }


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def mean_linearity(a: np.ndarray, mu: np.ndarray, p: Optional[np.ndarray] = None, *,
                   mu_A: Optional[np.ndarray] = None, mu_B: Optional[np.ndarray] = None
                   ) -> Dict[str, object]:
    """Test ``mu(a) ~= alpha + a * c`` (weighted LS, weights = bin probs ``p``).

    If per-half means ``mu_A, mu_B`` are given, R2 is SPLIT-HALF DEBIASED: the residual and total
    sums of squares use cross-half products ``<x_A, x_B>`` (unbiased for the noise-free squared
    magnitude), so MC noise does not deflate R2.

    Returns: ``R2`` (debiased if halves given), ``R2_raw``, ``residual_frac``, ``slope`` (=c),
    ``slope_unit``, ``intercept``, ``R2_per_coord``, ``mean_var_rel`` (size of the across-bin mean
    variation relative to the typical bulk-mean magnitude; small => mean barely varies with a)."""
    a = np.asarray(a, float); mu = np.asarray(mu, float)
    m, d = mu.shape
    w = np.ones(m) if p is None else np.asarray(p, float)
    w = np.where(w > 0, w, 0.0); W = w.sum()
    if W <= 0:
        return {"R2": float("nan"), "R2_raw": float("nan"), "residual_frac": float("nan"),
                "slope": np.zeros(d), "slope_unit": np.zeros(d), "intercept": np.zeros(d),
                "R2_per_coord": np.full(d, np.nan), "mean_var_rel": float("nan")}
    abar = float((w * a).sum() / W); da = a - abar
    Saa = float((w * da * da).sum())
    mubar = (w[:, None] * mu).sum(0) / W
    dmu = mu - mubar
    beta = (w[:, None] * da[:, None] * dmu).sum(0) / (Saa + _EPS)
    intercept = mubar - beta * abar
    pred = intercept[None, :] + a[:, None] * beta[None, :]
    resid = mu - pred
    ss_tot_j = (w[:, None] * dmu ** 2).sum(0)
    ss_res_j = (w[:, None] * resid ** 2).sum(0)
    R2_per = 1.0 - ss_res_j / (ss_tot_j + _EPS)
    R2_raw = 1.0 - float(ss_res_j.sum()) / (float(ss_tot_j.sum()) + _EPS)

    R2 = R2_raw
    mean_var_rel = float(np.sqrt(max(0.0, float(ss_tot_j.sum()))) /
                         (np.sqrt((w[:, None] * mu ** 2).sum()) + _EPS))
    if mu_A is not None and mu_B is not None:
        mu_A = np.asarray(mu_A, float); mu_B = np.asarray(mu_B, float)
        dmuA, dmuB = mu_A - mubar, mu_B - mubar
        resA, resB = mu_A - pred, mu_B - pred
        tot = float((w[:, None] * dmuA * dmuB).sum())
        res = float((w[:, None] * resA * resB).sum())
        R2 = 1.0 - res / (tot + _EPS) if tot > 0 else float("nan")
        mean_var_rel = float(np.sqrt(max(0.0, tot)) /
                             (np.sqrt(abs((w[:, None] * mu_A * mu_B).sum())) + _EPS))
    return {"R2": float(R2), "R2_raw": float(R2_raw),
            "residual_frac": float(np.sqrt(max(0.0, 1.0 - R2))) if np.isfinite(R2) else float("nan"),
            "slope": beta, "slope_unit": beta / (np.linalg.norm(beta) + _EPS),
            "intercept": intercept, "R2_per_coord": R2_per, "mean_var_rel": mean_var_rel}


def _top_eig_sym(S: np.ndarray) -> Tuple[float, np.ndarray]:
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    k = int(np.argmax(np.abs(w)))
    return float(w[k]), V[:, k]


def cov_rank1_structure(a: np.ndarray, Sigma: np.ndarray, p: Optional[np.ndarray] = None, *,
                        ref: str = "weighted", predicted_dir: Optional[np.ndarray] = None,
                        Sigma_A: Optional[np.ndarray] = None, Sigma_B: Optional[np.ndarray] = None
                        ) -> Dict[str, object]:
    """Test whether the family ``{D_alpha = Sigma_alpha - Sigma_ref}`` is a single rank-1 family
    ``D_alpha ~= f(a) v v^T`` (one shared direction ``v``, scalar magnitude ``f`` varying with a).

    SPLIT-HALF DEBIASING: if per-half covariances ``Sigma_A, Sigma_B`` are given, all quadratic
    quantities use cross-half products (energy matrix ``sum_a p * sym(D_a^A D_a^B)``, coherence
    numerator/denominator, signal size) so covariance MC noise does not inflate the rank or the
    apparent variation. STRONGLY RECOMMENDED (a single-half covariance difference is ~all noise).

    Returns:
      ``cov_var_rel``       size of the across-bin covariance variation relative to ``||Sigma_ref||``
                            (debiased). ~0 => covariance is bin-INDEPENDENT (use one Sigma0; even
                            cheaper than rank-1). Only if this is non-trivial does rank matter.
      ``family_coherence``  ``sum_a p f_a^2 / sum_a p ||D_a||_F^2`` in [0,1]; 1 iff every D_a is
                            rank-1 along the SAME v (the whole supposition). Debiased via halves.
      ``common_dir``        v (d,), the shared direction (top eigvec of the energy matrix).
      ``dir_alignment``     structure-weighted mean ``|cos(v_a, v)|`` over bins.
      ``rank1_sq_frac``     debiased per-bin rank-1 fraction in squared-eigenvalue scale.
      ``f``                 (num_bins,) magnitude ``v^T D_a v``; ``f_R2_linear`` / ``f_R2_quadratic``
                            = how well f(a) is fit by a line / parabola.
      ``cos_predicted``     ``|cos(v, predicted_dir)|`` if a direction (e.g. u) is given.
    """
    a = np.asarray(a, float); Sigma = np.asarray(Sigma, float)
    m, d, _ = Sigma.shape
    w = np.ones(m) if p is None else np.asarray(p, float)
    w = np.where(w > 0, w, 0.0); W = w.sum()

    def _ref(S):
        if isinstance(ref, str) and ref == "weighted":
            return (w[:, None, None] * S).sum(0) / (W + _EPS)
        if isinstance(ref, str) and ref == "mid":
            return S[m // 2]
        return S[int(ref)]

    Sref = _ref(Sigma)
    ref_fro = float(np.linalg.norm(Sref))
    Dfull = Sigma - Sref[None]
    split = Sigma_A is not None and Sigma_B is not None
    if split:
        SA, SB = np.asarray(Sigma_A, float), np.asarray(Sigma_B, float)
        DA = SA - _ref(SA)[None]
        DB = SB - _ref(SB)[None]
    else:
        DA = DB = Dfull

    # Common direction v from the FULL-sample energy matrix G = sum_a p D_a^2 (low variance; the
    # isotropic noise bias barely rotates its top eigenvector). Cross-half products are reserved
    # for the bias-sensitive SCALAR magnitudes below.
    G = np.zeros((d, d))
    for b in range(m):
        if w[b] > 0:
            G += w[b] * (Dfull[b] @ Dfull[b])
    _, v = _top_eig_sym(G)
    v = v / (np.linalg.norm(v) + _EPS)

    fA = np.einsum("i,mij,j->m", v, DA, v)
    fB = np.einsum("i,mij,j->m", v, DB, v)
    cross_fro = np.einsum("mij,mij->m", DA, DB)           # <D_a^A, D_a^B> (unbiased ||D_a||^2)
    num = float((w * fA * fB).sum())
    den = float((w * cross_fro).sum())
    coherence = float(np.clip(num / (den + _EPS), 0.0, 1.0)) if den > 0 else float("nan")
    cov_var_rel = float(np.sqrt(max(0.0, den)) / (ref_fro + _EPS))

    # how much of the (debiased) covariance variation is DIAGONAL (per-coordinate variance shifts)
    # vs the single rank-1 direction -- a cheaper alternative parametrisation if diag_frac is high.
    diagA = np.einsum("mii->mi", DA); diagB = np.einsum("mii->mi", DB)
    diag_num = float((w * (diagA * diagB).sum(1)).sum())
    diag_frac = float(np.clip(diag_num / (den + _EPS), 0.0, 1.0)) if den > 0 else float("nan")

    # ABSOLUTE across-bin cov variation (debiased Frobenius) and the DIAGONAL-only relative variation
    # (variances are what the next ReLU is most sensitive to) -- for width-scaling power-law fits.
    cov_abs = float(np.sqrt(max(0.0, den)))
    diag_ref_norm = float(np.linalg.norm(np.diag(Sref)))
    diag_var_rel = float(np.sqrt(max(0.0, diag_num)) / (diag_ref_norm + _EPS))

    # per-bin rank-1 (squared scale) + direction alignment, from cross-symmetrized M_a
    r1 = np.zeros(m); align = np.zeros(m); aw = np.zeros(m)
    f = 0.5 * (fA + fB)
    for b in range(m):
        if w[b] <= 0:
            continue
        Mb = 0.5 * (DA[b] @ DB[b] + DB[b] @ DA[b])
        ew = np.linalg.eigvalsh(0.5 * (Mb + Mb.T))
        s = float(np.abs(ew).sum())
        r1[b] = float(np.abs(ew).max() / (s + _EPS)) if s > 0 else 0.0
        _, vb = _top_eig_sym(Dfull[b])          # per-bin direction from full sample (low variance)
        nb = float(np.sqrt(max(0.0, cross_fro[b])))
        if nb > 1e-9:
            align[b] = abs(float(vb @ v)); aw[b] = w[b] * nb
    rank1_sq = float((w * r1).sum() / (W + _EPS))
    dir_alignment = float((aw * align).sum() / (aw.sum() + _EPS)) if aw.sum() > 0 else float("nan")

    def _wR2(cols):
        sw = np.sqrt(w); coef, *_ = np.linalg.lstsq(cols * sw[:, None], f * sw, rcond=None)
        pred = cols @ coef
        fbar = float((w * f).sum() / (W + _EPS))
        return 1.0 - float((w * (f - pred) ** 2).sum()) / (float((w * (f - fbar) ** 2).sum()) + _EPS)

    f_lin = _wR2(np.stack([np.ones(m), a], 1))
    f_quad = _wR2(np.stack([np.ones(m), a, a ** 2], 1))

    cos_pred = None
    if predicted_dir is not None:
        u = np.asarray(predicted_dir, float).reshape(-1); nu = np.linalg.norm(u)
        if nu > 0:
            cos_pred = abs(float(v @ (u / nu)))

    return {"cov_var_rel": cov_var_rel, "cov_abs": cov_abs, "diag_var_rel": diag_var_rel,
            "family_coherence": coherence, "diag_frac": diag_frac,
            "common_dir": v, "dir_alignment": dir_alignment, "rank1_sq_frac": rank1_sq,
            "f": f, "f_R2_linear": float(f_lin), "f_R2_quadratic": float(f_quad),
            "cos_predicted": cos_pred, "ref_frobenius": ref_fro}


def structure_report(states: Dict[str, object], Ws: Sequence[Tuple[np.ndarray, Optional[np.ndarray]]],
                     *, which: str = "post", ref: str = "weighted") -> List[Dict[str, object]]:
    """Run ``mean_linearity`` + ``cov_rank1_structure`` (split-half debiased when the states carry
    A/B halves) on every hidden layer of a states dict. ``which`` = ``"pre"`` or ``"post"``.
    Compares c and v against the per-layer coupling column ``u = M[1:,0]``."""
    spike_coord = int(states["spike_coord"])
    layers = states[which]                                # type: ignore[index]
    out = []
    for li, st in enumerate(layers):
        u = spike_coupling_column(Ws[li][0], spike_coord)
        ml = mean_linearity(st["a"], st["mu"], st["p"],
                            mu_A=st.get("mu_A"), mu_B=st.get("mu_B"))
        cs = cov_rank1_structure(st["a"], st["Sigma"], st["p"], ref=ref, predicted_dir=u,
                                 Sigma_A=st.get("Sigma_A"), Sigma_B=st.get("Sigma_B"))
        cos_c_u = abs(float(ml["slope_unit"] @ (u / (np.linalg.norm(u) + _EPS))))
        out.append({
            "layer": li, "which": which,
            "mean_R2": ml["R2"], "mean_var_rel": ml["mean_var_rel"], "mean_slope_vs_u": cos_c_u,
            "cov_var_rel": cs["cov_var_rel"], "cov_abs": cs["cov_abs"],
            "cov_diag_var_rel": cs["diag_var_rel"], "cov_ref_fro": cs["ref_frobenius"],
            "cov_family_coherence": cs["family_coherence"], "cov_diag_frac": cs["diag_frac"],
            "cov_dir_alignment": cs["dir_alignment"], "cov_rank1_sq": cs["rank1_sq_frac"],
            "cov_dir_vs_u": cs["cos_predicted"],
            "cov_f_R2_linear": cs["f_R2_linear"], "cov_f_R2_quadratic": cs["f_R2_quadratic"],
            "_mean": ml, "_cov": cs,
        })
    return out


def summarize_report(rows: List[Dict[str, object]]) -> str:
    """One-line-per-layer table of the headline structure metrics."""
    hdr = (f"{'layer':>5} {'rep':>4} | {'meanR2':>7} {'m_var':>6} {'c·u':>5} | "
           f"{'c_var':>6} {'coher':>6} {'diag':>5} {'algn':>5} {'v·u':>5} | {'f~a²':>5}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        cvu = r["cov_dir_vs_u"]; cvu_s = f"{cvu:5.2f}" if cvu is not None else "  n/a"
        def g(x): return f"{x:5.2f}" if (x is not None and np.isfinite(x)) else "  n/a"
        lines.append(
            f"{r['layer']:>5} {r['which']:>4} | {r['mean_R2']:7.3f} {g(r['mean_var_rel'])} "
            f"{g(r['mean_slope_vs_u'])} | {g(r['cov_var_rel'])} {g(r['cov_family_coherence'])} "
            f"{g(r['cov_diag_frac'])} {g(r['cov_dir_alignment'])} {cvu_s} | {g(r['cov_f_R2_quadratic'])}")
    return "\n".join(lines)

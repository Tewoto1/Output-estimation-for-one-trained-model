"""analysis/Tools/cumulants_sv.py -- joint-cumulant scaling tools for the S/V
collective coordinates of a trained-to-zero MLP.

For an input x ~ N(0, I_n) and a hidden POST-ReLU latent X in R^n, define two
scalar collective coordinates (LITERAL typed definitions used by this study):

    S = sum_i X_i                  (all-ones / coherent projection, unnormalised)
    V = (1/sqrt(n)) sum_i X_i^2     (energy)

This module estimates the joint cumulants  kappa_{a,b} = cum(S..S [a], V..V [b])
up to total order a+b <= 4 and answers two questions:

  1. SCALING:  how does kappa_{a,b} scale with width n?  Under INDEPENDENCE of the
     n coordinates (the iid null), with these literal definitions,
         kappa_{a,b}(S,V) = (1/sqrt n)^b * sum_i cum_{a,b}(X_i, X_i^2)   ~  n^{1 - b/2}
     (the explicit n-power is 1 - b/2, INDEPENDENT of a, because S is an
     unnormalised sum). The user's heuristic n^{1 - a/2 - b} instead corresponds
     to the rescaled coordinates S=(1/sqrt n)sum X_i, V=(1/n)sum X_i^2; both
     predicted slopes are reported for comparison.

  2. CORRELATION:  the assumption-free null is the DIAGONAL reference
     kappa_{a,b}^diag = the value the cumulant WOULD take if the n coordinates were
     independent with the SAME per-coordinate marginals. The gap between the
     measured cumulant and this diagonal reference is precisely the contribution of
     inter-coordinate correlation -- which trained-to-0 nets develop as a rank-1
     all-ones (-mu) mode, inflating the all-ones cumulants kappa_{a,0} above the null.

A matched random-direction control R = sqrt(n) * (u^T X), |u|=1 (same vector norm
as the all-ones direction behind S) isolates whether the coherence is specific to
the all-ones mode: for iid coordinates kappa_a(S) ~ kappa_a(R); a rank-1 all-ones
spike makes kappa_a(S) >> kappa_a(R).

Sampling error on every cumulant is a delete-one-block JACKKNIFE standard
deviation, and a cumulant is reported as RESOLVED only when |estimate| > z * sd
(default z = 2) -- so nothing is plotted inside the sampling noise.

The cumulant algebra is pure numpy (validated against analytic exponential/Poisson
cumulants, the Gaussian null, independence, and the iid-sum diagonal identity).
`stream_collective_coordinates` is the only torch-dependent entry point and imports
torch lazily, so the estimators import on any Python.
"""
from __future__ import annotations
from math import comb, sqrt
from typing import Dict, List, Tuple, Sequence
import numpy as np

# (p,q) index set for total order <= 4
PQ4: List[Tuple[int, int]] = [(p, q) for o in range(5) for p in range(o + 1) for q in [o - p]]
# the cumulant orders we report for (S,V) -- the two means are returned separately
AB_REPORT: List[Tuple[int, int]] = [(2, 0), (3, 0), (4, 0),      # pure S  (the all-ones coherent mode)
                                    (0, 2), (0, 3), (0, 4),      # pure V  (energy)
                                    (1, 1), (2, 1), (1, 2),      # mixed, low order
                                    (2, 2), (3, 1), (1, 3)]      # mixed, order 4


# --------------------------------------------------------------------------- #
#  moment <-> cumulant algebra (bivariate, total order <= 4)                  #
# --------------------------------------------------------------------------- #
def central_from_raw_2d(M: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    """Central moments mu_{pq}=E[(Y-EY)^p (Z-EZ)^q] from RAW moments M_{pq}=E[Y^p Z^q].
    `M` must contain every (p,q) with p+q<=4 and M[(0,0)]=1."""
    my, mz = M[(1, 0)], M[(0, 1)]
    mu: Dict[Tuple[int, int], float] = {}
    for (p, q) in PQ4:
        s = 0.0
        for j in range(p + 1):
            for k in range(q + 1):
                s += comb(p, j) * comb(q, k) * ((-my) ** (p - j)) * ((-mz) ** (q - k)) * M[(j, k)]
        mu[(p, q)] = s
    return mu


def cumulants_from_central_2d(mu: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    """Joint cumulants kappa_{a,b} (2 <= a+b <= 4) from central moments mu_{pq}.
    Standard bivariate relations (mixed analogues of k4 = mu4 - 3 mu2^2)."""
    k: Dict[Tuple[int, int], float] = {}
    k[(2, 0)] = mu[(2, 0)]
    k[(1, 1)] = mu[(1, 1)]
    k[(0, 2)] = mu[(0, 2)]
    for ab in [(3, 0), (2, 1), (1, 2), (0, 3)]:        # order 3: cumulant == central moment
        k[ab] = mu[ab]
    k[(4, 0)] = mu[(4, 0)] - 3 * mu[(2, 0)] ** 2
    k[(3, 1)] = mu[(3, 1)] - 3 * mu[(2, 0)] * mu[(1, 1)]
    k[(2, 2)] = mu[(2, 2)] - mu[(2, 0)] * mu[(0, 2)] - 2 * mu[(1, 1)] ** 2
    k[(1, 3)] = mu[(1, 3)] - 3 * mu[(0, 2)] * mu[(1, 1)]
    k[(0, 4)] = mu[(0, 4)] - 3 * mu[(0, 2)] ** 2
    return k


def cumulants_from_raw_2d(M: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    return cumulants_from_central_2d(central_from_raw_2d(M))


def cumulants_from_samples_2d(Y: np.ndarray, Z: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Cumulants of (Y,Z) from samples, centred first (numerically safe)."""
    Y = np.asarray(Y, np.float64); Z = np.asarray(Z, np.float64)
    yc = Y - Y.mean(); zc = Z - Z.mean()
    mu = {(p, q): float(np.mean((yc ** p) * (zc ** q))) for (p, q) in PQ4}
    return cumulants_from_central_2d(mu)


def cumulants_1d(x: np.ndarray) -> Dict[int, float]:
    """Univariate cumulants k2,k3,k4 (centred)."""
    x = np.asarray(x, np.float64); xc = x - x.mean()
    m2 = float(np.mean(xc ** 2)); m3 = float(np.mean(xc ** 3)); m4 = float(np.mean(xc ** 4))
    return {2: m2, 3: m3, 4: m4 - 3 * m2 ** 2}


# --------------------------------------------------------------------------- #
#  diagonal / iid reference from per-coordinate moments                       #
# --------------------------------------------------------------------------- #
def diagonal_reference(coord_moments: np.ndarray, n: int) -> Dict[Tuple[int, int], float]:
    """IID reference kappa_{a,b}^diag for S=sum X_i, V=(1/sqrt n) sum X_i^2.

    `coord_moments[k-1, i] = E[X_i^k]`, k=1..8, shape (8, n). Under independence
    across coordinates,
        kappa_{a,b}(S,V) = (1/sqrt n)^b * sum_i cum_{a,b}(X_i, X_i^2),
    where cum_{a,b}(X_i, X_i^2) uses raw moments E[X_i^p (X_i^2)^q] = m_{p+2q}.
    """
    m = np.asarray(coord_moments, np.float64)          # (8, n)
    ncoord = m.shape[1]
    out = {ab: 0.0 for ab in AB_REPORT}
    for i in range(ncoord):
        mk = [1.0] + [float(m[k - 1, i]) for k in range(1, 9)]   # mk[k] = E[X_i^k], k=0..8
        Mi = {(p, q): mk[p + 2 * q] for (p, q) in PQ4}
        ki = cumulants_from_raw_2d(Mi)
        for ab in AB_REPORT:
            out[ab] += ki[ab]
    return {ab: (n ** (-ab[1] / 2.0)) * out[ab] for ab in AB_REPORT}


# --------------------------------------------------------------------------- #
#  delete-one-block jackknife                                                 #
# --------------------------------------------------------------------------- #
def _block_bounds(N: int, B: int) -> List[Tuple[int, int]]:
    edges = np.linspace(0, N, B + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(B) if edges[i + 1] > edges[i]]


def jackknife_cumulants_2d(Y: np.ndarray, Z: np.ndarray, B: int = 40
                           ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Delete-one-block jackknife. Returns (full_sample_point_estimate, jackknife_sd)."""
    Y = np.asarray(Y, np.float64); Z = np.asarray(Z, np.float64)
    bounds = _block_bounds(Y.size, B); Bn = len(bounds)
    psum = {pq: np.empty(Bn) for pq in PQ4}; cnt = np.empty(Bn)
    for bi, (lo, hi) in enumerate(bounds):
        ys, zs = Y[lo:hi], Z[lo:hi]; cnt[bi] = hi - lo
        for (p, q) in PQ4:
            psum[(p, q)][bi] = np.sum((ys ** p) * (zs ** q))
    tot = {pq: psum[pq].sum() for pq in PQ4}; Ntot = cnt.sum()
    point = cumulants_from_raw_2d({pq: tot[pq] / Ntot for pq in PQ4})
    jk = {ab: np.empty(Bn) for ab in AB_REPORT}
    for bi in range(Bn):
        Nk = Ntot - cnt[bi]
        kk = cumulants_from_raw_2d({pq: (tot[pq] - psum[pq][bi]) / Nk for pq in PQ4})
        for ab in AB_REPORT:
            jk[ab][bi] = kk[ab]
    sd = {ab: float(sqrt((Bn - 1) / Bn * np.sum((jk[ab] - jk[ab].mean()) ** 2))) for ab in AB_REPORT}
    return point, sd


def jackknife_cumulants_1d(x: np.ndarray, B: int = 40
                           ) -> Tuple[Dict[int, float], Dict[int, float]]:
    x = np.asarray(x, np.float64)
    bounds = _block_bounds(x.size, B); Bn = len(bounds)
    pw = {p: np.empty(Bn) for p in (1, 2, 3, 4)}; cnt = np.empty(Bn)
    for bi, (lo, hi) in enumerate(bounds):
        xs = x[lo:hi]; cnt[bi] = hi - lo
        for p in (1, 2, 3, 4):
            pw[p][bi] = np.sum(xs ** p)
    tot = {p: pw[p].sum() for p in (1, 2, 3, 4)}; Ntot = cnt.sum()

    def cum_from_pow(tp, NN):
        m1, m2, m3, m4 = tp[1] / NN, tp[2] / NN, tp[3] / NN, tp[4] / NN
        c2 = m2 - m1 ** 2
        c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
        mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        return {2: c2, 3: c3, 4: mu4 - 3 * c2 ** 2}
    point = cum_from_pow(tot, Ntot)
    jk = {p: np.empty(Bn) for p in (2, 3, 4)}
    for bi in range(Bn):
        Nk = Ntot - cnt[bi]
        kk = cum_from_pow({p: tot[p] - pw[p][bi] for p in (1, 2, 3, 4)}, Nk)
        for p in (2, 3, 4):
            jk[p][bi] = kk[p]
    sd = {p: float(sqrt((Bn - 1) / Bn * np.sum((jk[p] - jk[p].mean()) ** 2))) for p in (2, 3, 4)}
    return point, sd


# --------------------------------------------------------------------------- #
#  predicted slopes & log-log fits                                            #
# --------------------------------------------------------------------------- #
def predicted_slope_literal(a: int, b: int) -> float:
    """IID n-power for the LITERAL defs S=sum X_i, V=(1/sqrt n)sum X_i^2: 1 - b/2."""
    return 1.0 - b / 2.0


def predicted_slope_heuristic(a: int, b: int) -> float:
    """The user's heuristic n^{1 - a/2 - b} (rescaled coordinates)."""
    return 1.0 - a / 2.0 - b


def fit_loglog_slope(widths: Sequence[float], values: Sequence[float],
                     mask: Sequence[bool] = None) -> float:
    """Slope of log|value| vs log(width) over points where value!=0 (and mask True)."""
    w = np.asarray(widths, float); v = np.asarray(values, float)
    m = np.isfinite(v) & (v != 0)
    if mask is not None:
        m &= np.asarray(mask, bool)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(w[m]), np.log(np.abs(v[m])), 1)[0])


# --------------------------------------------------------------------------- #
#  reduce collected scalars -> the full per-(layer,width,seed) result dict    #
# --------------------------------------------------------------------------- #
def reduce_sv(S: np.ndarray, V: np.ndarray, proj: np.ndarray,
              coord_moments: np.ndarray, n: int, *, n_blocks: int = 40,
              z_gate: float = 2.0) -> Dict:
    """Turn the collected per-sample scalars into measured cumulants, jackknife
    SDs, the diagonal/iid reference, the random-direction control, the 2-point
    decomposition, marginal non-Gaussianity, and resolution flags.

    S, V : (Nmc,) per-sample collective coordinates.
    proj : (Nmc, n_rand) random-unit-direction projections u_d^T X.
    coord_moments : (8, n) per-coordinate moments E[X_i^k], k=1..8.
    """
    S = np.asarray(S, np.float64); V = np.asarray(V, np.float64)
    meas, sd = jackknife_cumulants_2d(S, V, B=n_blocks)
    diag = diagonal_reference(coord_moments, n)

    # random-direction control: R_d = sqrt(n) * proj_d  (matched |.|=sqrt n to the all-ones dir)
    proj = np.asarray(proj, np.float64)
    Rk = {2: [], 3: [], 4: []}; Rsd = {2: [], 3: [], 4: []}
    for d in range(proj.shape[1]):
        Rd = sqrt(n) * proj[:, d]
        pt_d, sd_d = jackknife_cumulants_1d(Rd, B=n_blocks)
        for a in (2, 3, 4):
            Rk[a].append(pt_d[a]); Rsd[a].append(sd_d[a])
    R_mean = {a: float(np.mean(Rk[a])) for a in (2, 3, 4)}
    R_sd = {a: float(sqrt(np.mean(np.square(Rsd[a])) + np.var(Rk[a]))) for a in (2, 3, 4)}

    # per-coordinate marginals (non-Gaussianity)
    m = np.asarray(coord_moments, np.float64)
    mu1 = m[0]; var_i = m[1] - m[0] ** 2
    k3_i = m[2] - 3 * m[0] * m[1] + 2 * m[0] ** 3
    mu4_i = m[3] - 4 * m[0] * m[2] + 6 * m[0] ** 2 * m[1] - 3 * m[0] ** 4
    k4_i = mu4_i - 3 * var_i ** 2
    sv = np.clip(var_i, 1e-30, None)
    skew_i = k3_i / sv ** 1.5
    exkurt_i = k4_i / sv ** 2
    dead_frac = float(np.mean(var_i < 1e-12))

    # 2-point decomposition of Var(S): diagonal + off-diagonal
    var_S = meas[(2, 0)]; diag_var = diag[(2, 0)]
    offdiag = var_S - diag_var
    twopoint = dict(var_S=float(var_S), diag_var=float(diag_var),
                    offdiag_total=float(offdiag),
                    offdiag_frac=float(offdiag / var_S) if var_S != 0 else float("nan"),
                    mean_offdiag_cov=float(offdiag / (n * (n - 1))) if n > 1 else float("nan"),
                    n_times_mean_cov=float(offdiag / (n - 1)) if n > 1 else float("nan"))

    # coherent-enhancement ratio measured/diagonal for the pure-S cumulants
    enh = {a: (float(meas[(a, 0)] / diag[(a, 0)]) if diag[(a, 0)] != 0 else float("nan"))
           for a in (2, 3, 4)}
    # all-ones vs random direction
    svr = {a: (float(meas[(a, 0)] / R_mean[a]) if R_mean[a] != 0 else float("nan"))
           for a in (2, 3, 4)}

    out = dict(n=int(n), n_samples=int(S.size), n_blocks=int(n_blocks), z_gate=float(z_gate),
               mean_S=float(S.mean()), mean_V=float(V.mean()))
    for ab in AB_REPORT:
        key = f"{ab[0]}{ab[1]}"
        out[f"k_{key}"] = float(meas[ab])
        out[f"sd_{key}"] = float(sd[ab])
        out[f"diag_{key}"] = float(diag[ab])
        out[f"resolved_{key}"] = bool(abs(meas[ab]) > z_gate * sd[ab])
    for a in (2, 3, 4):
        out[f"R_k{a}"] = R_mean[a]; out[f"R_sd{a}"] = R_sd[a]
        out[f"enh_k{a}"] = enh[a]; out[f"SvsR_k{a}"] = svr[a]
    out.update({f"twopoint_{k}": v for k, v in twopoint.items()})
    out.update(dead_frac=dead_frac,
               marg_mean=float(np.mean(mu1)), marg_var_med=float(np.median(var_i)),
               marg_skew_med=float(np.median(skew_i)), marg_exkurt_med=float(np.median(exkurt_i)),
               marg_skew_mean=float(np.mean(skew_i)), marg_exkurt_mean=float(np.mean(exkurt_i)))
    return out


# --------------------------------------------------------------------------- #
#  torch streamer (lazy import): forward passes -> per-sample scalars         #
# --------------------------------------------------------------------------- #
def stream_collective_coordinates(model, input_dim: int, num_samples: int, *,
                                  layers: Sequence[int] = (0, 1, 2),
                                  whichs: Sequence[str] = ("post",), which: str = None,
                                  n_rand: int = 3, batch: int = 8192,
                                  device=None, dtype=None, input_std: float = 1.0,
                                  data_seed: int = 0, dir_seed: int = 777,
                                  accum_dtype="float64") -> Dict[str, Dict[int, Dict]]:
    """Stream x ~ N(0, input_std^2 I) through `model`, collecting per-sample S, V and
    random-direction projections, plus per-coordinate moments, for each requested
    activation type ("pre" = pre-ReLU W h, "post" = post-ReLU) and each hidden layer,
    all from a SINGLE forward pass.

    `whichs` is an iterable of {"pre","post"} (scalar `which=` also accepted for
    backward compat). Returns a nested dict
        {which: {layer: {"S":(N,), "V":(N,), "proj":(N,n_rand), "coord_moments":(8,n),
                         "n":n, "which":which}}}.
    Only per-sample scalars are kept, so cost is O(N*(2+n_rand)) floats per (which,layer).
    """
    import torch
    if which is not None:                       # backward-compat scalar
        whichs = (which,)
    whichs = tuple(whichs)
    for w in whichs:
        if w not in ("pre", "post"):
            raise ValueError(f"which must be 'pre' or 'post'; got {w!r}")
    model.eval()
    dev = device or next(model.parameters()).device
    layers = list(layers)
    n = model.cfg.hidden_dim

    # fixed random unit directions per layer (shared across whichs/seeds/widths via dir_seed)
    U = {}
    for ell in layers:
        g = torch.Generator(device="cpu").manual_seed(dir_seed + 1009 * ell + n)
        Um = torch.randn(n, n_rand, generator=g, dtype=torch.float64)
        Um = Um / Um.norm(dim=0, keepdim=True).clamp_min(1e-30)
        U[ell] = Um.to(dev)

    keys = [(w, ell) for w in whichs for ell in layers]
    S = {k: [] for k in keys}; Vv = {k: [] for k in keys}; PR = {k: [] for k in keys}
    msum = {k: np.zeros((8, n), dtype=np.float64) for k in keys}
    sqn = float(np.sqrt(n))
    gdata = torch.Generator(device="cpu").manual_seed(data_seed)
    done = 0
    while done < num_samples:
        b = int(min(batch, num_samples - done)); done += b
        x = (torch.randn(b, input_dim, generator=gdata, dtype=torch.float32) * input_std).to(dev)
        with torch.no_grad():
            _, acts = model(x, return_activations=True)
        for w in whichs:
            for ell in layers:
                X = acts[w][ell].to(torch.float64)              # (b, n)
                S[(w, ell)].append((X.sum(1)).cpu().numpy())
                Vv[(w, ell)].append((X.pow(2).sum(1) / sqn).cpu().numpy())
                PR[(w, ell)].append((X @ U[ell]).cpu().numpy())  # (b, n_rand)
                xk = torch.ones_like(X)
                for k in range(1, 9):                            # per-coordinate moment sums E[X^k]
                    xk = xk * X
                    msum[(w, ell)][k - 1] += xk.sum(0).cpu().numpy()
    out: Dict[str, Dict[int, Dict]] = {w: {} for w in whichs}
    for (w, ell) in keys:
        out[w][ell] = dict(S=np.concatenate(S[(w, ell)]), V=np.concatenate(Vv[(w, ell)]),
                           proj=np.concatenate(PR[(w, ell)], axis=0),
                           coord_moments=msum[(w, ell)] / float(done), n=n, which=w)
    return out

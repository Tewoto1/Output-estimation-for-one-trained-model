"""binning.py -- bin grids, truncated-normal stats, and W2-optimal quantizers.

Everything the coordinate-spike BINNED kprop needs to place and score spike bins,
independent of the K=2 propagation state (which lives in ``core.py``):

* the per-bin parallel map (``resolve_workers`` / ``_run_bins``) -- the linear and
  ReLU steps run one thread per bin, and the auto thread-count logic is shared infra;
* standard-normal helpers and the truncated-normal interval stats
  (``normal_interval_stats``) used to score a Gaussian against a bin ``[low, high)``;
* the bin utilities ``find_bin`` / ``safe_bin_representative``;
* the FIXED bin grids ``make_gaussian_edges`` (pre-activation, signed) and
  ``make_relu_post_edges`` (post-ReLU, nonnegative + zero bin);
* the WASSERSTEIN-optimal (Lloyd-Max) quantizers ``lloyd_max_edges`` /
  ``lloyd_max_edges_mixture`` / ``lloyd_max_edges_mixture_split`` that place bins to
  minimize the W2 distance to the layer's expected continuous spike law.

The ReLU integrals and matrix helpers reuse the repo's shared torch-free kernel
``Mecha_preds._utils`` (``_phi`` / ``_Phi`` imported below). ``core.py`` imports the
bin utilities it needs from here; the two modules together are the K=2 predictor.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .._utils import _phi, _Phi

_TINY = 1e-30
_VAR_FLOOR = 1e-12
_DEFAULT_MIN_PROB = 1e-15


# --------------------------------------------------------------------------- #
# per-bin parallel map (auto thread count; resolves "auto"/None per machine)
# --------------------------------------------------------------------------- #
#: ``Workers`` is an explicit thread count, ``1`` for serial, or ``None`` / ``"auto"`` to
#: auto-resolve (see ``resolve_workers``). The env var ``BINNED_KPROP_WORKERS`` overrides auto.
Workers = Union[int, str, None]
_WORKERS_ENV = "BINNED_KPROP_WORKERS"
_CUDA_CACHED: Optional[bool] = None


def _cuda_available() -> bool:
    """``True`` if a CUDA torch build reports a device (cached). GUARDED: the numpy core
    still imports and runs with no torch installed -- a missing/cpu torch yields ``False``,
    so this never makes torch a hard dependency of the torch-free core."""
    global _CUDA_CACHED
    if _CUDA_CACHED is None:
        try:
            import torch  # noqa: PLC0415
            _CUDA_CACHED = bool(torch.cuda.is_available())
        except Exception:
            _CUDA_CACHED = False
    return _CUDA_CACHED


def resolve_workers(workers: Workers = None) -> int:
    """Resolve the per-bin thread count to a concrete ``>= 1`` (``1`` == serial).

    Precedence: an explicit integer ``workers`` (``>= 1``) wins; then the env var
    ``$BINNED_KPROP_WORKERS``; then AUTO. Auto = ``8`` on a CUDA box (the common
    big-machine signal), else ``min(8, os.cpu_count())``. ``None`` and ``"auto"`` both mean
    auto -- so the DEFAULT is parallel, not serial; pass ``workers=1`` to force serial.
    """
    if isinstance(workers, str):
        if workers.lower() != "auto":
            raise ValueError(f"workers must be an int, None, or 'auto'; got {workers!r}")
        workers = None
    if workers is not None:
        return max(1, int(workers))
    env = os.environ.get(_WORKERS_ENV)
    if env:
        return max(1, int(env))
    if _cuda_available():
        return 8
    return max(1, min(8, os.cpu_count() or 1))


def _run_bins(n: int, body: Callable[[int], object], workers: Workers) -> list:
    """Run ``body(i)`` for ``i in range(n)`` and return the results in index order.

    Serial -- bit-for-bit a plain ``[body(i) for i in range(n)]`` -- when the resolved
    thread count (``resolve_workers(workers)``) is ``1`` or ``n <= 1``; otherwise the per-bin
    bodies run on a ``ThreadPoolExecutor`` with ``min(workers, n)`` threads. THREADS, not
    processes: the bin work is numpy/LAPACK-heavy (``@``, ``eigh``) and releases the GIL, no
    pickling is needed (the output arrays are shared), and each ``body(i)`` writes ONLY to
    disjoint slices (row/column ``i``) of those arrays -- so concurrent writes never alias.
    Cross-bin reductions are RETURNED by ``body`` and combined by the caller, never
    accumulated in place (which would race). Exceptions propagate as in the serial path.
    """
    w = resolve_workers(workers)
    if w <= 1 or n <= 1:
        return [body(i) for i in range(n)]
    with ThreadPoolExecutor(max_workers=min(w, n)) as ex:
        return list(ex.map(body, range(n)))


# --------------------------------------------------------------------------- #
# standard-normal helpers
# --------------------------------------------------------------------------- #
def normal_pdf(z: np.ndarray) -> np.ndarray:
    return _phi(np.asarray(z, dtype=np.float64))


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return _Phi(np.asarray(z, dtype=np.float64))


def _xphi(a: np.ndarray, phi_a: np.ndarray) -> np.ndarray:
    """``a * phi(a)`` with the convention ``(+/-inf) * phi(+/-inf) = 0`` (spec 6.2).

    Zero out the infinite entries of ``a`` BEFORE multiplying so ``inf * 0`` never
    forms a NaN (``phi(+/-inf) == 0`` already, so the finite product is unchanged)."""
    a = np.asarray(a, dtype=np.float64)
    a_fin = np.where(np.isfinite(a), a, 0.0)
    return a_fin * phi_a


def normal_interval_stats(mean: float, var: float, low, high, *,
                          min_prob: float = _DEFAULT_MIN_PROB):
    """Truncated-normal stats for ``Y ~ N(mean, var)`` restricted to ``[low, high)``.

    Returns ``(Q, y_mean, y_var, tau1, tau2)`` where ``Q = P(Y in [low,high))``,
    ``tau1 = E[Y-mean | in]``, ``tau2 = E[(Y-mean)^2 | in]``, ``y_mean = mean+tau1``,
    ``y_var = tau2 - tau1^2``. ``low``/``high`` may be scalars or arrays (vectorized
    over bins) and may be ``-inf`` / ``+inf``. Where ``Q <= min_prob`` the moments are
    returned as 0 (the caller skips that contribution).

    Sanity (full interval ``[-inf, inf]``): ``Q=1, tau1=0, tau2=var, y_var=var``.
    """
    var = float(var)
    sig = math.sqrt(max(var, _VAR_FLOOR))
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    a = (low - mean) / sig
    b = (high - mean) / sig

    phi_a, phi_b = _phi(a), _phi(b)
    Phi_a, Phi_b = _Phi(a), _Phi(b)
    Q = Phi_b - Phi_a

    ok = Q > min_prob
    Qsafe = np.where(ok, Q, 1.0)
    tau1 = np.where(ok, sig * (phi_a - phi_b) / Qsafe, 0.0)
    # tau2 = var * [1 + (a phi(a) - b phi(b)) / Q],  with x*phi(x)->0 at +/-inf
    tau2 = np.where(ok, var * (1.0 + (_xphi(a, phi_a) - _xphi(b, phi_b)) / Qsafe), 0.0)
    y_mean = np.where(ok, mean + tau1, mean)
    y_var = np.clip(tau2 - tau1 * tau1, 0.0, None)
    return Q, y_mean, y_var, tau1, tau2


# --------------------------------------------------------------------------- #
# bin utilities
# --------------------------------------------------------------------------- #
def find_bin(edges: np.ndarray, x: float) -> int:
    """Index ``beta`` with ``edges[beta] <= x < edges[beta+1]`` (clamped to a valid bin).

    ``edges`` is sorted length ``num_bins+1`` (may start at ``-inf`` / end at ``+inf``).
    """
    edges = np.asarray(edges, dtype=np.float64)
    m = edges.shape[0] - 1
    beta = int(np.searchsorted(edges, x, side="right") - 1)
    return min(max(beta, 0), m - 1)


def safe_bin_representative(edges: np.ndarray, beta: int) -> float:
    """A finite representative for (possibly empty) bin ``beta``: midpoint, or the
    finite boundary offset into an infinite tail bin."""
    lo, hi = float(edges[beta]), float(edges[beta + 1])
    if math.isinf(lo) and math.isinf(hi):
        return 0.0
    if math.isinf(lo):
        return hi - 1.0
    if math.isinf(hi):
        return lo + 1.0
    return 0.5 * (lo + hi)


def make_gaussian_edges(num_bins: int, *, std: float = 1.0,
                        tail_inf: bool = True) -> np.ndarray:
    """Pre-activation bin edges (length ``num_bins+1``) for a reference ``N(0, std^2)``.

    Interior edges are equal-probability Gaussian quantiles, so each bin carries ~the
    same reference mass. The outermost edges are ``-inf`` / ``+inf`` (``tail_inf``) so
    no transition mass can escape the grid. INCLUDES NEGATIVE BINS -- required for the
    pre-activation state because ``A^+ = gamma A + r^T B`` can be negative even when the
    incoming ``A`` is post-ReLU nonnegative (spec 12.1).
    """
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if num_bins == 1:
        return np.array([-np.inf, np.inf])
    from scipy.special import ndtri  # inverse normal CDF
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.empty(num_bins + 1, dtype=np.float64)
    edges[1:-1] = std * ndtri(qs[1:-1])
    edges[0] = -np.inf if tail_inf else std * ndtri(0.5 / num_bins)
    edges[-1] = np.inf if tail_inf else std * ndtri(1.0 - 0.5 / num_bins)
    return edges


def make_relu_post_edges(num_bins: int, *, std: float = 1.0,
                         tail_inf: bool = True) -> np.ndarray:
    """Post-ReLU bin edges (length ``num_bins+1``), NONNEGATIVE with a zero bin.

    ReLU output is ``>= 0`` with an atom at 0, so bin 0 ``= [0, e_1)`` captures the
    dead mass and the positive bins are half-normal quantiles (spec 12.1)."""
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if num_bins == 1:
        return np.array([0.0, np.inf])
    from scipy.special import ndtri
    # Quantiles of |N(0,std^2)| on (0,1): map to upper-half normal quantiles.
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.empty(num_bins + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:-1] = std * ndtri(0.5 + 0.5 * qs[1:-1])
    edges[-1] = np.inf if tail_inf else std * ndtri(0.5 + 0.5 * (1.0 - 0.5 / num_bins))
    return edges


# --------------------------------------------------------------------------- #
# WASSERSTEIN-OPTIMAL bin placement (Lloyd-Max scalar quantizer)
# --------------------------------------------------------------------------- #
# The binning collapses the spike coordinate A within each cell to one representative,
# so the squared error it introduces is exactly  sum_a p_a Var(A | cell_a) = W_2^2( law(A),
# binned ).  Minimizing that W_2 distance over (edges, reps) is optimal scalar quantization,
# whose stationarity conditions are: reps = cell CENTROIDS (conditional means) and interior
# edges = MIDPOINTS between adjacent reps -- the Lloyd-Max algorithm. This beats equal-mass
# quantiles (``make_gaussian_edges``), which over-resolve the dense centre; the W_2 optimum
# puts more points in the tails (point density ~ f^{1/3} vs ~ f). See ALGORITHM.md sec 7.
def _truncnorm_cell(a: float, b: float) -> Tuple[float, float, float]:
    """``(mass, centroid, within-cell variance)`` of ``N(0,1)`` truncated to ``[a, b)``."""
    Z = float(_Phi(np.array(b)) - _Phi(np.array(a)))
    if Z <= _TINY:
        mid = 0.0 if (math.isinf(a) or math.isinf(b)) else 0.5 * (a + b)
        return 0.0, mid, 0.0
    pa = 0.0 if math.isinf(a) else float(_phi(np.array(a)))
    pb = 0.0 if math.isinf(b) else float(_phi(np.array(b)))
    ap = 0.0 if math.isinf(a) else a * pa
    bp = 0.0 if math.isinf(b) else b * pb
    v = (pa - pb) / Z
    var = max(1.0 + (ap - bp) / Z - v * v, 0.0)
    return Z, v, var


def _lloyd_max_interval(lo: float, hi: float, num_pts: int, *, iters: int = 200,
                        tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max W2 quantizer of ``N(0,1)`` restricted to ``[lo, hi)`` with ``num_pts`` points.
    Returns ``(edges[num_pts+1], reps[num_pts])`` (edges[0]=lo, edges[-1]=hi)."""
    from scipy.special import ndtri
    if num_pts <= 1:
        return np.array([lo, hi], float), np.array([_truncnorm_cell(lo, hi)[1]])
    Plo, Phi_hi = float(_Phi(np.array(lo))), float(_Phi(np.array(hi)))
    qs = np.clip(np.linspace(Plo, Phi_hi, num_pts + 1), 1e-15, 1 - 1e-15)
    e = ndtri(qs); e[0], e[-1] = lo, hi                       # init equal-mass within [lo,hi)
    for _ in range(iters):
        v = np.array([_truncnorm_cell(e[i], e[i + 1])[1] for i in range(num_pts)])
        ne = e.copy(); ne[1:-1] = 0.5 * (v[:-1] + v[1:])       # edges <- midpoints of centroids
        if np.max(np.abs(ne[1:-1] - e[1:-1])) < tol:
            e = ne
            break
        e = ne
    v = np.array([_truncnorm_cell(e[i], e[i + 1])[1] for i in range(num_pts)])
    return e, v


def lloyd_max_edges(mean: float, std: float, num_bins: int, *, rectified: bool = False,
                    iters: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """W2-OPTIMAL bin ``(edges, representatives)`` for the expected continuous spike law.

    Minimizes the Wasserstein-2 distance to the layer's expected continuous law of ``A``:
    ``N(mean, std^2)`` for a pre-activation grid (``rectified=False``), or the rectified
    Gaussian ``max(N(mean, std^2), 0)`` for a post-ReLU grid (``rectified=True`` -- the
    0-atom of mass ``Phi(-mean/std)`` is given its own representative at 0, and the positive
    tail is Lloyd-Max-quantized with the remaining ``num_bins-1`` points). Drop-in alternative
    to ``make_gaussian_edges`` / ``make_relu_post_edges``; also returns the optimal reps
    (cell centroids) -- the linear/ReLU steps recompute reps dynamically, so the edges are
    what matters at grid-build time.
    """
    std = max(float(std), math.sqrt(_VAR_FLOOR))
    if not rectified:
        e_std, v_std = _lloyd_max_interval(-np.inf, np.inf, num_bins, iters=iters)
        return mean + std * e_std, mean + std * v_std
    if num_bins == 1:
        return np.array([0.0, np.inf]), np.array([max(mean, 0.0)])
    a0 = -mean / std                                          # standardized location of 0
    _e_pos, v_pos = _lloyd_max_interval(a0, np.inf, num_bins - 1, iters=iters)  # positive tail reps
    reps = np.empty(num_bins)
    reps[0] = 0.0                                             # the dead-ReLU 0-atom representative
    reps[1:] = np.maximum(mean + std * v_pos, 0.0)
    edges = np.empty(num_bins + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (reps[:-1] + reps[1:])               # Lloyd-Max nearest-rep boundaries
    edges[-1] = np.inf
    return np.maximum.accumulate(edges), reps


# The expected continuous law of A at a layer is, under the K=2 closure, a GAUSSIAN MIXTURE
# (one component per old bin: A^+ = sum_alpha p_alpha N(m_Y,alpha, s_Y,alpha^2)). Its per-cell
# mean has a CLOSED FORM -- a mass-weighted blend of each component's truncated-Gaussian first
# moment -- so Lloyd-Max can quantize the *true* mixture, not a single moment-matched Gaussian,
# with every iteration still closed form (no quadrature). (This is exactly the centroid the
# linear step already computes as sum_alpha eta_{alpha|beta} E[Y_alpha | I_beta].)
def _mixture_cell(w: np.ndarray, m: np.ndarray, s: np.ndarray, a: float, b: float
                  ) -> Tuple[float, float]:
    """``(mass, centroid)`` of cell ``[a,b)`` under the Gaussian mixture ``sum_k w_k N(m_k,s_k^2)``."""
    al = (a - m) / s
    be = (b - m) / s
    Zc = _Phi(be) - _Phi(al)                                  # per-component mass in the cell
    Z = float(np.sum(w * Zc))
    if Z <= _TINY:
        mid = 0.0 if (math.isinf(a) or math.isinf(b)) else 0.5 * (a + b)
        return 0.0, mid
    fm = float(np.sum(w * (m * Zc + s * (_phi(al) - _phi(be)))))   # unnormalized 1st moment
    return Z, fm / Z


def _lloyd_max_mixture_interval(w, m, s, lo, hi, num_pts, *, iters=1000, tol=1e-10):
    """Lloyd-Max W2 quantizer of the Gaussian mixture restricted to ``[lo, hi)`` (closed form).

    ``iters`` defaults to 1000: a multimodal mixture converges to the Lloyd fixed point more
    slowly than a single Gaussian, and each iteration is only O(num_pts * num_components)."""
    from scipy.special import ndtri
    if num_pts <= 1:
        return np.array([lo, hi], float), np.array([_mixture_cell(w, m, s, lo, hi)[1]])
    mu = float(w @ m)
    var = max(float(w @ (s ** 2 + m ** 2) - mu * mu), _VAR_FLOOR)
    sd = math.sqrt(var)
    Plo = float(_Phi(np.array((lo - mu) / sd)))
    Phi_hi = float(_Phi(np.array((hi - mu) / sd)))
    qs = np.clip(np.linspace(Plo, Phi_hi, num_pts + 1), 1e-15, 1 - 1e-15)
    e = mu + sd * ndtri(qs); e[0], e[-1] = lo, hi             # init: moment-matched-Gaussian quantiles
    for _ in range(iters):
        v = np.array([_mixture_cell(w, m, s, e[i], e[i + 1])[1] for i in range(num_pts)])
        ne = e.copy(); ne[1:-1] = 0.5 * (v[:-1] + v[1:])      # edges <- midpoints of mixture centroids
        if np.max(np.abs(ne[1:-1] - e[1:-1])) < tol:
            e = ne
            break
        e = ne
    v = np.array([_mixture_cell(w, m, s, e[i], e[i + 1])[1] for i in range(num_pts)])
    return e, v


def lloyd_max_edges_mixture(weights, means, stds, num_bins, *, rectified: bool = False,
                            iters: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """W2-optimal bin ``(edges, reps)`` for the EXACT Gaussian-mixture spike law (no quadrature).

    ``weights, means, stds`` describe the layer's expected continuous law of ``A`` as the mixture
    ``sum_k w_k N(means_k, stds_k^2)`` -- e.g. one component per current bin
    (``w=p``, ``means=gamma*v + r.mu``, ``stds=sqrt(r.Sigma r)`` for the pre-activation). Returns the
    Lloyd-Max ``(edges, representatives)`` where each representative is the closed-form mixture
    centroid (``_mixture_cell``) and the interior edges are midpoints. ``rectified=True`` quantizes
    ``max(mixture,0)`` (post-ReLU): one representative pinned at the 0-atom, the rest Lloyd-Max on the
    positive part. Unlike ``lloyd_max_edges`` (which uses a single moment-matched Gaussian) this is
    exact for the true mixture."""
    w = np.asarray(weights, float); m = np.asarray(means, float)
    s = np.maximum(np.asarray(stds, float), math.sqrt(_VAR_FLOOR))
    w = w / w.sum()
    if not rectified:
        return _lloyd_max_mixture_interval(w, m, s, -np.inf, np.inf, num_bins, iters=iters)
    if num_bins == 1:
        return np.array([0.0, np.inf]), np.array([max(float(w @ m), 0.0)])
    _e, v_pos = _lloyd_max_mixture_interval(w, m, s, 0.0, np.inf, num_bins - 1, iters=iters)
    reps = np.empty(num_bins); reps[0] = 0.0; reps[1:] = np.maximum(v_pos, 0.0)
    edges = np.empty(num_bins + 1); edges[0] = 0.0
    edges[1:-1] = 0.5 * (reps[:-1] + reps[1:]); edges[-1] = np.inf
    return np.maximum.accumulate(edges), reps


def lloyd_max_edges_mixture_split(weights, means, stds, num_neg: int, num_pos: int,
                                  *, iters: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """W2-optimal PRE-ACTIVATION grid with a GUARANTEED positive-bin budget, split at 0.

    Lloyd-Max-quantizes the Gaussian-mixture spike law ``sum_k w_k N(means_k, stds_k^2)``
    SEPARATELY on ``(-inf, 0)`` with ``num_neg`` bins and on ``[0, inf)`` with ``num_pos``
    bins, then concatenates them (the shared ``0`` edge appears once). Returns
    ``(edges[num_neg+num_pos+1], reps[num_neg+num_pos])``.

    Contrast with ``lloyd_max_edges_mixture`` (non-rectified), which quantizes over ALL the
    reals with a single ``num_bins`` budget and lets the optimizer decide the sign split --
    typically ~half the bins land negative. The following ReLU keeps ONLY the positive
    pre-activation bins (every negative bin collapses into the single zero bin), so that
    ~half is the only resolution that survives, and it HALVES the positive-bin count at every
    linear step. Pinning the positive side to ``num_pos`` bins (= the post-grid's positive
    budget) stops that collapse; the ``num_neg`` negative bins still resolve the negative mass
    whose bulk law feeds the merged zero bin. Total ``num_neg + num_pos`` grows past a single
    all-reals ``num_bins`` grid, concentrated where it survives ReLU (spec sec 7.3 / 8).
    """
    w = np.asarray(weights, float); m = np.asarray(means, float)
    s = np.maximum(np.asarray(stds, float), math.sqrt(_VAR_FLOOR))
    w = w / w.sum()
    num_neg = max(1, int(num_neg))   # >= 1 negative bin (a single [-inf,0) catch-bin; spec sec 8)
    num_pos = max(1, int(num_pos))
    e_neg, v_neg = _lloyd_max_mixture_interval(w, m, s, -np.inf, 0.0, num_neg, iters=iters)
    e_pos, v_pos = _lloyd_max_mixture_interval(w, m, s, 0.0, np.inf, num_pos, iters=iters)
    edges = np.concatenate([e_neg[:-1], e_pos])        # drop the duplicated 0 edge
    reps = np.concatenate([v_neg, v_pos])
    return edges, reps

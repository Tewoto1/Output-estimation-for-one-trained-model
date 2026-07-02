"""selftest.py -- torch-free numerical verification of the coordinate-spike binned K=2 core.

Builds coordinate-spiked ReLU MLPs (``M = W + e_1 e_1^T``) in numpy and checks the
algorithm against a numpy Monte-Carlo reference (no torch needed for the K=2 core). Run:

    python -m Mecha_preds.binned_kprop.selftest

Checks (spec section 14):
  1. ``normal_interval_stats``: full interval, symmetric interval, narrow interval.
  2. shape / probability invariants after a linear and a ReLU step.
  3. linear-step conditional closure vs Monte-Carlo (one old bin): Q, E[C|bin], Cov[C|bin].
  4. end-to-end small net vs MC: more bins beats one bin, and many-bin error is small.
  5. degenerate cases: r = 0 (deterministic A^+), one bin, no spike.
  6. kprop-harmonic hook (``..cumulants.kprop``): equivalence at k_max=2 + runs as a
     bulk-ReLU backend. SKIPS (does not fail) if torch is unavailable.
"""
from __future__ import annotations

import numpy as np

from .binning import (
    normal_interval_stats, make_gaussian_edges, make_relu_post_edges,
)
from .core import (
    run_binned_kprop_k2, linear_step_k2, relu_step_k2, gaussian_initial_state,
    unconditional_mean, _phi, _Phi, exact_relu_covariance,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def coordinate_spike_net(n, depth, seed, *, theta=1.0, out_dim=8):
    """Random ReLU net with a coordinate spike ``theta e_1 e_1^T`` on every hidden layer."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n)); P[0, 0] = theta
    Ws = [(rng.standard_normal((n, n)) / np.sqrt(n) + P, None) for _ in range(depth)]
    Ws.append((rng.standard_normal((out_dim, n)) / np.sqrt(n), None))
    return Ws


def mc_output_mean(Ws, n, samples, batch, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros(Ws[-1][0].shape[0]); accsq = np.zeros_like(acc); c = 0
    while c < samples:
        b = min(batch, samples - c); h = rng.standard_normal((b, n))
        for li, (W, _b) in enumerate(Ws):
            z = h @ W.T; h = np.maximum(z, 0.0) if li < len(Ws) - 1 else z
        acc += h.sum(0); accsq += (h ** 2).sum(0); c += b
    mu = acc / c
    return mu, np.sqrt(np.clip(accsq / c - mu ** 2, 0.0, None) / c)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --------------------------------------------------------------------------- #
# the suite
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> bool:
    rng = np.random.default_rng(0); ok = True

    # 1. normal_interval_stats -------------------------------------------------
    Q, ym, yv, t1, t2 = normal_interval_stats(0.7, 2.0, -np.inf, np.inf)
    full_ok = (abs(Q - 1) < 1e-12 and abs(ym - 0.7) < 1e-12 and abs(yv - 2.0) < 1e-9
               and abs(t1) < 1e-12 and abs(t2 - 2.0) < 1e-9)
    Qs, yms, *_ = normal_interval_stats(0.0, 1.0, -1.3, 1.3)
    sym_ok = abs(yms - 0.0) < 1e-12
    Qn, ymn, yvn, *_ = normal_interval_stats(0.0, 1.0, 0.499, 0.501)
    narrow_ok = abs(ymn - 0.5) < 1e-3 and yvn < 1e-4
    t1_ok = full_ok and sym_ok and narrow_ok
    ok &= t1_ok
    if verbose:
        print(f"[1] normal_interval_stats (full / symmetric / narrow):   "
              f"{'OK' if t1_ok else 'FAIL'}")

    # 2. invariants after a linear + ReLU step --------------------------------
    n, d = 16, 15
    nb = 11
    edges = make_gaussian_edges(nb); post = make_relu_post_edges(nb)
    st = gaussian_initial_state(d, edges)
    M = rng.standard_normal((n, n)) / np.sqrt(n); M[0, 0] += 1.0
    st1 = linear_step_k2(st, M, edges); st1.check()
    st2 = relu_step_k2(st1, post); st2.check()
    inv_ok = (st1.p.min() >= -1e-12 and abs(st1.p.sum() - 1) < 1e-9
              and st2.p.min() >= -1e-12 and abs(st2.p.sum() - 1) < 1e-9
              and st2.mu.shape == (nb, d) and st2.Sigma.shape == (nb, d, d))
    ok &= inv_ok
    if verbose:
        print(f"[2] shape / probability invariants after linear & ReLU:  "
              f"{'OK' if inv_ok else 'FAIL'}")

    # 3. linear-step conditional closure vs MC (one old bin) ------------------
    rng3 = np.random.default_rng(12345)            # dedicated RNG (independent of test order)
    d = 4; n = d + 1
    gamma = 1.7
    r = rng3.standard_normal(d) * 0.6
    u = rng3.standard_normal(d) * 0.6
    V = rng3.standard_normal((d, d)) / np.sqrt(d)
    M = np.zeros((n, n)); M[0, 0] = gamma; M[0, 1:] = r; M[1:, 0] = u; M[1:, 1:] = V
    v = 0.3
    mu_a = rng3.standard_normal(d) * 0.4
    A = rng3.standard_normal((d, d)); Sig_a = A @ A.T / d + 0.5 * np.eye(d)

    mY = gamma * v + r @ mu_a
    mC = u * v + V @ mu_a
    SrA = Sig_a @ r; sY2 = float(r @ SrA); g = V @ SrA; SigC = V @ Sig_a @ V.T

    # (i) deterministic, noise-free: on the FULL interval the closure must reduce to
    #     (mC, SigC) exactly (tau1=0, y_var=sY2 => the g g^T corrections cancel).
    Qf, _ymf, yvf, t1f, _t2f = normal_interval_stats(mY, sY2, -np.inf, np.inf)
    mu_full = mC + (g / sY2) * t1f
    Sig_full = SigC - np.outer(g, g) / sY2 + np.outer(g, g) * (yvf / sY2 ** 2)
    exact_ok = (abs(Qf - 1) < 1e-12 and _rel(mu_full, mC) < 1e-12 and _rel(Sig_full, SigC) < 1e-10)

    # (ii) Monte-Carlo: bins with >=150k samples; cov estimates are inherently noisier
    #      (their relative error scales like 1/sqrt(N_bin)), so split the tolerances.
    rng_mc = np.random.default_rng(2024)
    NS = 4_000_000
    L = np.linalg.cholesky(Sig_a)
    B = mu_a + rng_mc.standard_normal((NS, d)) @ L.T
    Yv = gamma * v + B @ r
    Cv = v * u + B @ V.T
    edges3 = make_gaussian_edges(9)
    scaleC = float(np.linalg.norm(mC) + np.sqrt(np.mean(np.diag(SigC))))  # stable mean scale
    wQ = wM = wC = 0.0
    for beta in range(1, 8):                       # central bins (tails are MC-noisy)
        lo, hi = edges3[beta], edges3[beta + 1]
        mask = (Yv >= lo) & (Yv < hi)
        if mask.sum() < 150_000:
            continue
        Qhat = mask.mean(); Chat = Cv[mask].mean(0); covhat = np.cov(Cv[mask].T)
        Qc, _ym, yv2, tau1, _t2 = normal_interval_stats(mY, sY2, lo, hi)
        mu_ab = mC + (g / sY2) * tau1
        Sig_ab = SigC - np.outer(g, g) / sY2 + np.outer(g, g) * (yv2 / sY2 ** 2)
        wQ = max(wQ, abs(Qhat - Qc) / (Qc + 1e-9))
        # normalize the mean error by a fixed scale (not the per-bin norm, which can
        # cross zero), so the metric reflects absolute closure accuracy.
        wM = max(wM, float(np.linalg.norm(Chat - mu_ab) / scaleC))
        wC = max(wC, _rel(covhat, Sig_ab))
    mc_ok = wQ < 8e-3 and wM < 8e-3 and wC < 1.5e-2
    t3_ok = exact_ok and mc_ok
    ok &= t3_ok
    if verbose:
        print(f"[3] linear closure: exact full-interval {'OK' if exact_ok else 'FAIL'}; "
              f"MC Q {wQ:.1e} mean {wM:.1e} cov {wC:.1e}  {'OK' if t3_ok else 'FAIL'}")

    # 4. end-to-end small net vs MC -------------------------------------------
    n, depth = 48, 2
    Ws = coordinate_spike_net(n, depth, seed=7)
    mc, se = mc_output_mean(Ws, n, 3_000_000, 300_000, seed=123)
    err = {nb: _rel(run_binned_kprop_k2(Ws, n, num_bins=nb)["mean"], mc)
           for nb in (1, 5, 21)}
    err_w2 = _rel(run_binned_kprop_k2(Ws, n, num_bins=21, grid="wasserstein")["mean"], mc)
    refine_ok = err[21] < err[5] < err[1]              # more bins -> less error
    helps_ok = err[21] < 0.25 * err[1]                 # binning is a big win
    small_ok = err[21] < 0.05                          # many-bin error is small
    w2_ok = np.isfinite(err_w2) and err_w2 < 0.05       # wasserstein grid runs & is comparable
    t4_ok = refine_ok and helps_ok and small_ok and w2_ok
    ok &= t4_ok
    if verbose:
        print(f"[4] end-to-end vs MC (n={n}, depth={depth}): "
              f"err 1bin {err[1]:.2e} -> 5 {err[5]:.2e} -> 21 {err[21]:.2e} "
              f"(W2-grid {err_w2:.2e})   {'OK' if t4_ok else 'FAIL'}")

    # 5. degenerate cases ------------------------------------------------------
    # (a) r = 0  -> A^+ = gamma A deterministic inside each old bin.
    n, depth = 32, 2
    Ws0 = coordinate_spike_net(n, depth, seed=5)
    Ws0 = [(W.copy(), b) for (W, b) in Ws0]
    for li in range(depth):
        Ws0[li][0][0, 1:] = 0.0                        # r = 0 on hidden layers
    deg_r = run_binned_kprop_k2(Ws0, n, num_bins=11)["mean"]
    # (b) one bin (crude mixture-free Gaussian bulk) and (c) no spike -- just stable & finite.
    one_bin = run_binned_kprop_k2(coordinate_spike_net(n, depth, seed=5), n, num_bins=1)["mean"]
    Wsns = coordinate_spike_net(n, depth, seed=5, theta=0.0)   # no spike
    no_spike = run_binned_kprop_k2(Wsns, n, num_bins=11)["mean"]
    t5_ok = all(np.all(np.isfinite(x)) for x in (deg_r, one_bin, no_spike))
    ok &= t5_ok
    if verbose:
        print(f"[5] degenerate (r=0 / one-bin / no-spike) stable & finite: "
              f"{'OK' if t5_ok else 'FAIL'}")

    # 6. kprop-harmonic hook (skips if torch unavailable) ---------------------
    try:
        import torch  # noqa: F401
        from .kprop_hook import bulk_relu_kprop
        d = 12
        mu = rng.standard_normal(d) * 0.5
        Asd = rng.standard_normal((d, d)); Sig = Asd @ Asd.T / d + 0.3 * np.eye(d)
        # exact_relu_cov=True must reproduce the exact bivariate kernel
        mk, Sk = bulk_relu_kprop(mu, Sig, k_max=2, exact_relu_cov=True)
        me, Se = exact_relu_covariance(mu, Sig)
        eq_exact = max(_rel(mk, me), _rel(Sk, Se))
        # and the K=2 binned predictor runs end-to-end with the kprop backend
        n2 = 24
        Ws2 = coordinate_spike_net(n2, 2, seed=3)
        pred_kprop = run_binned_kprop_k2(Ws2, n2, num_bins=11, bulk_relu="kprop")["mean"]
        t6_ok = eq_exact < 1e-9 and np.all(np.isfinite(pred_kprop))
        ok &= t6_ok
        if verbose:
            print(f"[6] kprop hook: exact_relu_cov match {eq_exact:.1e}, backend runs   "
                  f"{'OK' if t6_ok else 'FAIL'}")
    except ModuleNotFoundError:
        if verbose:
            print("[6] kprop hook: SKIP (torch not installed; runs on the repo env)")

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)

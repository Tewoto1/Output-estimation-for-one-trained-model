"""selftest.py -- torch-free numerical verification of the analytic affine K=2 core.

Builds coordinate-spiked ReLU MLPs (``M = W + e_1 e_1^T``) in numpy and checks the
algorithm against closed forms and a numpy Monte-Carlo reference. Run:

    python -m Mecha_preds.analytic_kprop.selftest

Checks (paper sections 6-10 + implementation checklist 12):
  1. exact scalar identities after one layer: cell masses sum to 1 and the
     centroid representatives reproduce E[Y] exactly (eq 70 is exact).
  2. fast aggregated covariance sums ``(T0, T1)`` == the per-cell reference
     (``percell_bulk_moments``), i.e. the O(1)-congruence path is exact algebra.
  3. affine-fit identities: LS orthogonality (eq 79) and, with the
     moment-conservative intercept, TOTAL bulk covariance conservation (eq 91).
  4. layer-1 exactness: the fitted affine state after the first linear layer
     equals the closed-form joint-Gaussian regression (paper section 9), and the
     mean residual E_m is numerically zero.
  5. depth-1 closed form: E[out] = W_ro . E[ReLU(Z)], Z ~ N(0, M M^T) exactly
     known; error is pure scalar quadrature and must shrink with ``num_nodes``.
  6. end-to-end small net vs MC (depth 2): small error at 40 nodes, more nodes
     no worse than few nodes; parity with the binned companion printed.
  7. degenerate cases: r = 0 (deterministic scalar), no spike, biases, uniform
     grid, LS intercept, forced 1-negative-cell budget -- all stable & finite.
  8. optimization equivalences: threaded per-node ReLU (workers=4) is bit-identical
     to serial (workers=1); the vectorized Lloyd-Max mixture cells match the scalar
     ``_mixture_cell`` reference; PSD endpoint screening changes nothing.
  9. torch device path parity (``device="cpu"``): identical to the numpy path to
     BLAS reordering tolerance. SKIPS (does not fail) if torch is unavailable.
 10. fit="post" (affine family fitted on the POST-activation, linear step =
     transform of (m0, m1, W0, W1)): depth-1 closed form (the weighted-LS fit
     conserves the readout mean exactly), end-to-end vs MC at parity with
     fit="pre", both atom toggles ("exact" | "fit") finite and close, and
     threaded == serial to fp-regrouping tolerance.
"""
from __future__ import annotations

import numpy as np

from ..binned_kprop.selftest import coordinate_spike_net, mc_output_mean
from ..binned_kprop.core import run_binned_kprop_k2
from .core import (
    AnalyticState, gaussian_input_state, analytic_layer_update,
    _component_params, _layer_block, _pair_stats, _covariance_sums,
    percell_bulk_moments, make_cells, split_node_budget, negative_mass,
    unconditional_mean_cov, run_analytic_kprop_k2,
)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _rand_state(rng, m, d, *, with_input_t2=False):
    """A random valid atomic state (positive weights, PSD Sigmas)."""
    p = rng.random(m) + 0.2; p /= p.sum()
    a = np.abs(rng.standard_normal(m)) * 1.5
    mu = rng.standard_normal((m, d)) * 0.5
    Sig = np.zeros((m, d, d))
    for i in range(m):
        A = rng.standard_normal((d, d))
        Sig[i] = A @ A.T / d + 0.3 * np.eye(d)
    t2 = np.zeros(m)
    if with_input_t2:
        t2 = np.abs(rng.standard_normal(m)) * 0.5
    return AnalyticState(p=p, a=a, mu=mu, Sigma=Sig, t2=t2)


def _rand_block(rng, d, *, bias=False):
    n = d + 1
    M = rng.standard_normal((n, n)) / np.sqrt(n)
    M[0, 0] += 1.0
    b = (rng.standard_normal(n) * 0.3) if bias else None
    return M, b


def run(verbose: bool = True) -> bool:
    rng = np.random.default_rng(0); ok = True

    # ---- shared small problem for tests 1-3 --------------------------------
    m, d = 7, 6
    st = _rand_state(rng, m, d, with_input_t2=True)   # nonzero t2 exercises the input path
    M, b = _rand_block(rng, d, bias=True)
    gamma, r, u, V, beta, eta_b = _layer_block(M, b, d)
    mY, sY2, mC, g = _component_params(st, gamma, r, u, V, beta, eta_b)
    n_neg, n_pos = split_node_budget(24, negative_mass(st.p, mY, sY2), None, None)
    edges = make_cells(st.p, mY, sY2, n_neg, n_pos, grid="w2")
    Q, ymean, delta, vv, stoch = _pair_stats(st.p, mY, sY2, edges, min_prob=1e-15)
    w_raw = st.p @ Q; W_tot = float(w_raw.sum())
    keep = w_raw > 1e-15
    Qk, ymk, dk, vk, wk = Q[:, keep], ymean[:, keep], delta[:, keep], vv[:, keep], w_raw[keep]
    PQ = st.p[:, None] * Qk
    y = (PQ * ymk).sum(0) / wk

    # [1] exact scalar identities ------------------------------------------------
    mass_ok = abs(W_tot - 1.0) < 1e-12
    EY_mix = float(st.p @ mY)                              # mixture mean, closed form
    EY_grid = float((wk / W_tot) @ y)                      # grid mean via centroids
    cent_ok = abs(EY_grid - EY_mix) < 1e-10
    t1_ok = mass_ok and cent_ok
    ok &= t1_ok
    if verbose:
        print(f"[1] scalar identities: mass 1-{abs(W_tot-1):.1e}, centroid E[Y] err "
              f"{abs(EY_grid-EY_mix):.1e}   {'OK' if t1_ok else 'FAIL'}")

    # [2] fast covariance sums == per-cell reference ------------------------------
    s2 = np.where(stoch, sY2, 1.0)
    u_reg = np.where(stoch[:, None], g / s2[:, None], 0.0)
    mhat = (PQ.T @ mC + (PQ * dk).T @ u_reg) / wk[:, None]
    T0, T1 = _covariance_sums(st.p, Qk, dk, vk, sY2, stoch, y, wk, mhat,
                              st.Sigma, st.t2, mC, g, u, V)
    mh_ref, Shat = percell_bulk_moments(st.p, Qk, dk, vk, sY2, stoch, wk,
                                        st.Sigma, st.t2, mC, g, u, V)
    T0_ref = np.einsum("j,jab->ab", wk, Shat)
    T1_ref = np.einsum("j,jab->ab", wk * y, Shat)
    t2a = max(_rel(T0, T0_ref), _rel(T1, T1_ref))
    t2b = _rel(mhat, mh_ref)
    t2_ok = t2a < 1e-10 and t2b < 1e-12
    ok &= t2_ok
    if verbose:
        print(f"[2] fast (T0,T1) vs per-cell reference: {t2a:.1e}; mhat {t2b:.1e}   "
              f"{'OK' if t2_ok else 'FAIL'}")

    # [3] affine-fit identities (orthogonality + total covariance conservation) --
    _st_new, aff = analytic_layer_update(st, M, b, num_nodes=24, cov_intercept="mc")
    w, yv = aff.w, aff.y
    # recompute mhat/Shat on the SAME retained grid the layer used
    Q2, ym2, d2, v2, stoch2 = _pair_stats(st.p, mY, sY2, aff.edges, min_prob=1e-15)
    w2r = st.p @ Q2; keep2 = w2r > 1e-15
    mh2, Sh2 = percell_bulk_moments(st.p, Q2[:, keep2], d2[:, keep2], v2[:, keep2],
                                    sY2, stoch2, w2r[keep2],
                                    st.Sigma, st.t2, mC, g, u, V)
    ybar = float(w @ yv)
    e_m = mh2 - aff.mu0[None] - yv[:, None] * aff.mu1[None]
    orth = max(float(np.linalg.norm((w[:, None] * e_m).sum(0))),
               float(np.linalg.norm(((w * (yv - ybar))[:, None] * e_m).sum(0))))
    vY = float(w @ (yv - ybar) ** 2)
    lhs = aff.Sigma0 + ybar * aff.Sigma1 + vY * np.outer(aff.mu1, aff.mu1)   # E[Sig(Y)] + Cov(aff mean)
    mbar = (w[:, None] * mh2).sum(0)
    rhs = (np.einsum("j,jab->ab", w, Sh2)
           + np.einsum("j,ja,jb->ab", w, mh2 - mbar, mh2 - mbar))            # true Cov(C) under surrogate
    cons = _rel(lhs, rhs)
    t3_ok = orth < 1e-10 and cons < 1e-10
    ok &= t3_ok
    if verbose:
        print(f"[3] LS orthogonality {orth:.1e}; total-covariance conservation (eq 91) "
              f"{cons:.1e}   {'OK' if t3_ok else 'FAIL'}")

    # [4] layer-1 exactness (paper section 9) -------------------------------------
    d4 = 10
    M4, b4 = _rand_block(np.random.default_rng(42), d4, bias=True)
    g4, r4, u4, V4, be4, et4 = _layer_block(M4, b4, d4)
    sY2_exact = g4 * g4 + float(r4 @ r4)
    mu1_exact = (g4 * u4 + V4 @ r4) / sY2_exact
    mu0_exact = et4 - mu1_exact * be4
    st0 = gaussian_input_state(d4)
    _stn, aff4 = analytic_layer_update(st0, M4, b4, num_nodes=32, cov_intercept="mc")
    e_mu1 = _rel(aff4.mu1, mu1_exact); e_mu0 = _rel(aff4.mu0, mu0_exact)
    stats4: dict = {}
    _ = analytic_layer_update(st0, M4, b4, num_nodes=32, stats=stats4)
    t4_ok = e_mu1 < 1e-9 and e_mu0 < 1e-9 and stats4["E_m"][0] < 1e-18
    ok &= t4_ok
    if verbose:
        print(f"[4] layer-1 exact affine: mu1 {e_mu1:.1e}, mu0 {e_mu0:.1e}, "
              f"E_m {stats4['E_m'][0]:.1e}   {'OK' if t4_ok else 'FAIL'}")

    # [5] depth-1 closed form + quadrature convergence -----------------------------
    n5 = 40
    Ws5 = coordinate_spike_net(n5, 1, seed=3)
    M5 = Ws5[0][0]; Wro = Ws5[1][0]
    sig = np.sqrt(np.einsum("ij,ij->i", M5, M5))          # Z ~ N(0, M M^T)
    exact5 = Wro @ (sig / np.sqrt(2.0 * np.pi))           # E[ReLU(N(0,s^2))] = s/sqrt(2pi)
    errs5 = {nn: _rel(run_analytic_kprop_k2(Ws5, n5, num_nodes=nn)["mean"], exact5)
             for nn in (8, 40, 160)}
    t5_ok = errs5[40] < 2e-3 and errs5[160] < errs5[8] and errs5[160] < 5e-4
    ok &= t5_ok
    if verbose:
        print(f"[5] depth-1 closed form: err 8 {errs5[8]:.1e} -> 40 {errs5[40]:.1e} "
              f"-> 160 {errs5[160]:.1e}   {'OK' if t5_ok else 'FAIL'}")

    # [6] end-to-end vs MC (depth 2): parity with the binned companion at the
    # closure floor, and the error must IMPROVE with width (the K=2 rate), since at
    # these widths the floor -- not scalar quadrature -- dominates (more nodes are
    # NOT expected to help; see the notebook for the num_nodes knee).
    n6, depth6 = 48, 2
    Ws6 = coordinate_spike_net(n6, depth6, seed=7)
    mc, _se = mc_output_mean(Ws6, n6, 3_000_000, 300_000, seed=123)
    err6 = {nn: _rel(run_analytic_kprop_k2(Ws6, n6, num_nodes=nn)["mean"], mc)
            for nn in (4, 12, 40)}
    res6 = run_analytic_kprop_k2(Ws6, n6, num_nodes=40, collect=True, diagnostics=True)
    res6["final_state"].check()
    err_binned = _rel(run_binned_kprop_k2(Ws6, n6, num_bins=40)["mean"], mc)
    n6b = 96
    Ws6b = coordinate_spike_net(n6b, depth6, seed=7)
    mc_b, _ = mc_output_mean(Ws6b, n6b, 3_000_000, 300_000, seed=321)
    err6_wide = _rel(run_analytic_kprop_k2(Ws6b, n6b, num_nodes=40)["mean"], mc_b)
    meta6 = res6["metadata"]
    clean = (meta6["max_mass_lost"] < 1e-9 and np.all(np.isfinite(res6["mean"])))
    t6_ok = (err6[40] < 0.05                       # small at 40 nodes
             and err6[40] <= 1.3 * err6[4]         # more nodes never much worse
             and err6[40] < 2.0 * err_binned       # parity with binned @ equal budget
             and err6_wide < 0.8 * err6[40]        # width law: error falls with n
             and clean)
    ok &= t6_ok
    if verbose:
        print(f"[6] end-to-end vs MC (depth={depth6}): n=48 err 4 {err6[4]:.2e} -> "
              f"12 {err6[12]:.2e} -> 40 {err6[40]:.2e} (binned@40 {err_binned:.2e}); "
              f"n=96 err@40 {err6_wide:.2e}; mass-lost {meta6['max_mass_lost']:.0e}, "
              f"psd-clip {meta6['total_psd_clipped']:.0e}, E_m max {meta6['max_E_m']:.1e}   "
              f"{'OK' if t6_ok else 'FAIL'}")

    # [7] degenerate / option coverage ---------------------------------------------
    n7, depth7 = 32, 2
    Ws7r0 = [(W.copy(), bb) for (W, bb) in coordinate_spike_net(n7, depth7, seed=5)]
    for li in range(depth7):
        Ws7r0[li][0][0, 1:] = 0.0                          # r = 0: deterministic scalar
    preds = {
        "r=0": run_analytic_kprop_k2(Ws7r0, n7, num_nodes=16)["mean"],
        "no-spike": run_analytic_kprop_k2(coordinate_spike_net(n7, depth7, seed=5, theta=0.0),
                                          n7, num_nodes=16)["mean"],
        "uniform-grid": run_analytic_kprop_k2(coordinate_spike_net(n7, depth7, seed=5),
                                              n7, num_nodes=16, grid="uniform")["mean"],
        "ls-intercept": run_analytic_kprop_k2(coordinate_spike_net(n7, depth7, seed=5),
                                              n7, num_nodes=16, cov_intercept="ls")["mean"],
        "neg-budget-1": run_analytic_kprop_k2(coordinate_spike_net(n7, depth7, seed=5),
                                              n7, num_nodes=16, num_nodes_neg=1)["mean"],
        "gain-relu": run_analytic_kprop_k2(coordinate_spike_net(n7, depth7, seed=5),
                                           n7, num_nodes=16, bulk_relu="gain")["mean"],
    }
    # biased net
    rng7 = np.random.default_rng(9)
    Wsb = [(W, rng7.standard_normal(W.shape[0]) * 0.2) for (W, _b) in
           coordinate_spike_net(n7, depth7, seed=8)]
    preds["biases"] = run_analytic_kprop_k2(Wsb, n7, num_nodes=16)["mean"]
    mean7, cov7 = unconditional_mean_cov(
        run_analytic_kprop_k2(coordinate_spike_net(n7, depth7, seed=5), n7,
                              num_nodes=16, collect=True)["final_state"])
    t7_ok = (all(np.all(np.isfinite(v)) for v in preds.values())
             and np.all(np.isfinite(mean7)) and np.all(np.isfinite(cov7))
             and float(np.linalg.eigvalsh(cov7).min()) > -1e-8)
    ok &= t7_ok
    if verbose:
        bad = [k for k, v in preds.items() if not np.all(np.isfinite(v))]
        print(f"[7] degenerate/options (r=0, no-spike, uniform, ls, neg1, gain, biases) "
              f"stable & finite; total cov PSD   {'OK' if t7_ok else 'FAIL'}"
              + (f"  bad={bad}" if bad else ""))

    # [8] optimization equivalences ------------------------------------------------
    n8 = 48
    Ws8 = coordinate_spike_net(n8, 2, seed=11)
    r_serial = run_analytic_kprop_k2(Ws8, n8, num_nodes=24, workers=1)
    r_thread = run_analytic_kprop_k2(Ws8, n8, num_nodes=24, workers=4)
    thread_eq = (np.array_equal(r_serial["mean"], r_thread["mean"])
                 and r_serial["metadata"]["total_psd_clipped"]
                 == r_thread["metadata"]["total_psd_clipped"])
    from ..binned_kprop.binning import _mixture_cell, _mixture_cells_vec
    rng8 = np.random.default_rng(3)
    wmix = rng8.random(6); wmix /= wmix.sum()
    mmix = rng8.standard_normal(6) * 2.0
    smix = rng8.random(6) + 0.3
    edges8 = np.concatenate([[-np.inf], np.sort(rng8.standard_normal(9) * 2), [np.inf]])
    Zv, Cv = _mixture_cells_vec(wmix, mmix, smix, edges8)
    Zr = np.array([_mixture_cell(wmix, mmix, smix, edges8[i], edges8[i + 1])[0]
                   for i in range(10)])
    Cr = np.array([_mixture_cell(wmix, mmix, smix, edges8[i], edges8[i + 1])[1]
                   for i in range(10)])
    vec_eq = max(float(np.abs(Zv - Zr).max()), float(np.abs(Cv - Cr).max())) < 1e-13
    t8_ok = thread_eq and vec_eq
    ok &= t8_ok
    if verbose:
        print(f"[8] workers 4 == serial (bit-identical): {thread_eq}; vectorized "
              f"Lloyd cells == scalar reference: {vec_eq}   {'OK' if t8_ok else 'FAIL'}")

    # [9] torch device path parity (skips without torch) ---------------------------
    try:
        import torch  # noqa: F401
        r_np = run_analytic_kprop_k2(Ws8, n8, num_nodes=24, diagnostics=True)
        r_t = run_analytic_kprop_k2(Ws8, n8, num_nodes=24, diagnostics=True, device="cpu")
        dev_eq = _rel(r_t["mean"], r_np["mean"])
        es_eq = max(abs(a - b) / (abs(b) + 1e-30) for a, b in
                    zip(r_t["metadata"]["E_S_by_layer"], r_np["metadata"]["E_S_by_layer"]))
        t9_ok = dev_eq < 1e-10 and es_eq < 1e-6 and r_t["metadata"]["device"] == "cpu"
        ok &= t9_ok
        if verbose:
            print(f"[9] torch device parity: mean {dev_eq:.1e}, E_S {es_eq:.1e}   "
                  f"{'OK' if t9_ok else 'FAIL'}")
    except ModuleNotFoundError:
        if verbose:
            print("[9] torch device parity: SKIP (torch not installed)")

    # [10] fit="post" variant --------------------------------------------------------
    # (a) depth-1 closed form: post-fit readout mean is LS-conserved -> pure quadrature
    err10 = {nn: _rel(run_analytic_kprop_k2(Ws5, n5, num_nodes=nn, fit="post")["mean"],
                      exact5) for nn in (8, 40)}
    # (b) end-to-end vs MC, parity with fit="pre" + atom toggle
    e_post = _rel(run_analytic_kprop_k2(Ws6, n6, num_nodes=40, fit="post")["mean"], mc)
    e_postf = _rel(run_analytic_kprop_k2(Ws6, n6, num_nodes=40, fit="post",
                                         atom="fit")["mean"], mc)
    # (c) threading: slot-grouped accumulation -> allclose (not bit-identical)
    r_s = run_analytic_kprop_k2(Ws8, n8, num_nodes=24, fit="post", workers=1)["mean"]
    r_t = run_analytic_kprop_k2(Ws8, n8, num_nodes=24, fit="post", workers=4)["mean"]
    thr10 = _rel(r_t, r_s)
    # (d) degenerate: no spike, r = 0
    fin10 = all(np.all(np.isfinite(run_analytic_kprop_k2(W_, n7, num_nodes=16,
                                                         fit="post")["mean"]))
                for W_ in (coordinate_spike_net(n7, 2, seed=5, theta=0.0), Ws7r0))
    t10_ok = (err10[40] < 2e-3 and err10[40] <= err10[8]
              and e_post < 1.5 * err6[40] and e_postf < 1.5 * err6[40]
              and thr10 < 1e-12 and fin10)
    ok &= t10_ok
    if verbose:
        print(f"[10] fit=post: depth-1 {err10[8]:.1e}->{err10[40]:.1e}; MC err "
              f"{e_post:.2e} (atom-fit {e_postf:.2e}, pre {err6[40]:.2e}); "
              f"threads-vs-serial {thr10:.1e}; degenerate finite {fin10}   "
              f"{'OK' if t10_ok else 'FAIL'}")

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)

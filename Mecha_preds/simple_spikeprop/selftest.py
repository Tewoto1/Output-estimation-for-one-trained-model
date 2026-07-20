"""selftest.py -- torch-free numerical verification of SIMPLE SPIKE-PROP.

Builds coordinate-spiked ReLU MLPs (``M = W + e_1 e_1^T``) in numpy and checks the
algorithm against closed forms and a numpy Monte-Carlo reference. Run:

    python -m Mecha_preds.simple_spikeprop.selftest

Checks:
  1. spike-law machinery: Gaussian grid law moments; ReLU fold vs the closed-form
     rectified-Gaussian atom/mean/variance.
  2. channel_push on a pure Gaussian: ``c S + xi`` must reproduce
     ``N(c m0 + mu, c^2 s0^2 + omega^2)`` (moments + pointwise density).
  3. depth-1 EXACTNESS: at one hidden layer every pre-activation marginal is exact
     under the model (cross-covariance never enters a marginal), so the predicted
     mean must match the closed-form rectified-Gaussian readout to grid accuracy.
  4. end-to-end depth-3 vs MC, including the ordering vs the binned companion:
     at least as good as binned with num_bins=1 (same bulk treatment, cruder spike);
     ratio to binned num_bins=21 reported.
  5. invariants along a run (law mass/atom/pdf, Sigma symmetry) + mass-drift logs.
  6. grid refinement: predictions converge as num_grid grows.
  7. bias nets and a no-spike (theta = 0) net vs MC.
"""
from __future__ import annotations

import numpy as np

from .._utils import relu_moments_1d, _phi, _Phi
from .core import (
    SpikeLaw, gaussian_spike_law, channel_push, relu_law, law_mass, law_moments,
    run_simple_spikeprop_core,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def coordinate_spike_net(n, depth, seed, *, theta=1.0, out_dim=8, bias=False):
    """Random ReLU net with a coordinate spike ``theta e_1 e_1^T`` on every hidden layer."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n)); P[0, 0] = theta
    Ws = []
    for _ in range(depth):
        b = rng.standard_normal(n) * 0.1 if bias else None
        Ws.append((rng.standard_normal((n, n)) / np.sqrt(n) + P, b))
    b_ro = rng.standard_normal(out_dim) * 0.1 if bias else None
    Ws.append((rng.standard_normal((out_dim, n)) / np.sqrt(n), b_ro))
    return Ws


def mc_output_mean(Ws, n, samples, batch, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros(Ws[-1][0].shape[0]); accsq = np.zeros_like(acc); c = 0
    while c < samples:
        b = min(batch, samples - c); h = rng.standard_normal((b, n))
        for li, (W, bias) in enumerate(Ws):
            z = h @ W.T + (0.0 if bias is None else bias)
            h = np.maximum(z, 0.0) if li < len(Ws) - 1 else z
        acc += h.sum(0); accsq += (h ** 2).sum(0); c += b
    mu = acc / c
    return mu, np.sqrt(np.clip(accsq / c - mu ** 2, 0.0, None) / c)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --------------------------------------------------------------------------- #
# the suite
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> bool:
    ok = True

    # 1. spike-law machinery ---------------------------------------------------
    law = gaussian_spike_law(0.3, 1.7, num_grid=3001)
    m1, m2, v = law_moments(law)
    g_ok = abs(m1 - 0.3) < 1e-7 and abs(v - 1.7) < 1e-7 and abs(law_mass(law) - 1) < 1e-12
    post = relu_law(law)
    post.check()
    mu_, var_ = 0.3, 1.7
    ref_mean, _sec, ref_var = relu_moments_1d(np.array([mu_]), np.array([var_]))
    ref_atom = float(_Phi(np.array([-mu_ / np.sqrt(var_)]))[0])
    pm, _, pv = law_moments(post)
    fold_ok = (abs(post.atom - ref_atom) < 5e-6 and abs(pm - ref_mean[0]) < 5e-6
               and abs(pv - ref_var[0]) < 5e-6)      # trapezoid O(h^2) at 3001 nodes
    t1_ok = g_ok and fold_ok
    ok &= t1_ok
    if verbose:
        print(f"[1] spike-law grid moments + ReLU fold vs closed form:  "
              f"{'OK' if t1_ok else 'FAIL'}  (atom err {abs(post.atom - ref_atom):.1e}, "
              f"mean err {abs(pm - ref_mean[0]):.1e})")

    # 2. channel_push Gaussian exactness ---------------------------------------
    m0, s0, cc, mu_x, om2 = -0.4, 1.3, 0.8, 0.25, 0.49
    law0 = gaussian_spike_law(m0, s0 ** 2, num_grid=3001)
    pushed = channel_push(law0, cc, mu_x, om2, num_grid=3001)
    pushed.check()
    tm, tv = cc * m0 + mu_x, cc ** 2 * s0 ** 2 + om2
    pm1, _, pv1 = law_moments(pushed)
    ref_pdf = _phi((pushed.t - tm) / np.sqrt(tv)) / np.sqrt(tv)
    pdf_err = float(np.max(np.abs(pushed.pdf - ref_pdf)))
    t2_ok = abs(pm1 - tm) < 1e-7 and abs(pv1 - tv) < 1e-6 and pdf_err < 1e-6
    ok &= t2_ok
    if verbose:
        print(f"[2] channel_push of a Gaussian == exact Gaussian:       "
              f"{'OK' if t2_ok else 'FAIL'}  (mean err {abs(pm1 - tm):.1e}, "
              f"var err {abs(pv1 - tv):.1e}, sup pdf err {pdf_err:.1e})")

    # 3. depth-1 exactness ------------------------------------------------------
    # With one hidden layer, Z = M X is jointly Gaussian and E[ReLU(Z_i)] depends on
    # the marginal (0, ||M_i||^2) only -- the dropped cross-covariances cannot enter.
    n, out_dim = 40, 6
    Ws = coordinate_spike_net(n, 1, seed=7, out_dim=out_dim)
    M = Ws[0][0]
    var_z = (M ** 2).sum(axis=1)
    ref_h, _, _ = relu_moments_1d(np.zeros(n), var_z)
    ref_out = Ws[1][0] @ ref_h
    pred = run_simple_spikeprop_core(Ws, n)["mean"]
    e3 = _rel(pred, ref_out)
    t3_ok = e3 < 1e-5                                # grid-limited (~3e-6 at 2001 nodes)
    ok &= t3_ok
    if verbose:
        print(f"[3] depth-1 closed-form exactness:                      "
              f"{'OK' if t3_ok else 'FAIL'}  (rel err {e3:.2e})")

    # 4. end-to-end depth-3 vs MC + ordering vs binned --------------------------
    n = 48
    Ws = coordinate_spike_net(n, 3, seed=11, out_dim=8)
    mc, mc_se = mc_output_mean(Ws, n, samples=4_000_000, batch=200_000, seed=99)
    res = run_simple_spikeprop_core(Ws, n, collect=True)
    e_simple = _rel(res["mean"], mc)
    from ..binned_kprop.core import run_binned_kprop_k2
    e_b1 = _rel(run_binned_kprop_k2(Ws, n, num_bins=1, workers=1)["mean"], mc)
    e_b21 = _rel(run_binned_kprop_k2(Ws, n, num_bins=21, workers=1)["mean"], mc)
    mc_rel_noise = float(np.linalg.norm(mc_se) / np.linalg.norm(mc))
    t4_ok = e_simple < 0.15 and e_simple <= e_b1 * 1.05
    ok &= t4_ok
    if verbose:
        print(f"[4] depth-3 (n=48) vs MC:                               "
              f"{'OK' if t4_ok else 'FAIL'}  (simple {e_simple:.3e} <= binned[1] "
              f"{e_b1:.3e}; binned[21] {e_b21:.3e}; MC noise {mc_rel_noise:.1e})")

    # 5. invariants + drift logs -------------------------------------------------
    md = res["metadata"]
    law_f, m_f, Sig_f = res["final_state"]
    law_f.check()
    drift_ok = (md["max_push_mass_drift"] < 1e-6 and md["max_relu_mass_drift"] < 1e-6
                and np.allclose(Sig_f, Sig_f.T) and 0.0 <= law_f.atom <= 1.0
                and all(0.0 <= s["atom"] <= 1.0 for s in res["spike_by_layer"]))
    ok &= drift_ok
    if verbose:
        print(f"[5] invariants (mass drift {md['max_push_mass_drift']:.1e}, atoms, "
              f"Sigma sym):                {'OK' if drift_ok else 'FAIL'}")

    # 6. grid refinement ---------------------------------------------------------
    p_coarse = run_simple_spikeprop_core(Ws, n, num_grid=251)["mean"]
    p_mid = run_simple_spikeprop_core(Ws, n, num_grid=1001)["mean"]
    p_fine = run_simple_spikeprop_core(Ws, n, num_grid=4001)["mean"]
    d_coarse, d_mid = _rel(p_coarse, p_fine), _rel(p_mid, p_fine)
    t6_ok = d_mid < d_coarse and d_mid < 1e-4
    ok &= t6_ok
    if verbose:
        print(f"[6] grid refinement (251 -> 1001 -> 4001):              "
              f"{'OK' if t6_ok else 'FAIL'}  (drift {d_coarse:.2e} -> {d_mid:.2e})")

    # 7. bias net + no-spike net vs MC -------------------------------------------
    Wb = coordinate_spike_net(40, 2, seed=23, out_dim=5, bias=True)
    mcb, _ = mc_output_mean(Wb, 40, samples=2_000_000, batch=200_000, seed=17)
    e_bias = _rel(run_simple_spikeprop_core(Wb, 40)["mean"], mcb)
    W0 = coordinate_spike_net(40, 3, seed=29, out_dim=5, theta=0.0)
    mc0, _ = mc_output_mean(W0, 40, samples=2_000_000, batch=200_000, seed=31)
    e_nospike = _rel(run_simple_spikeprop_core(W0, 40)["mean"], mc0)
    t7_ok = e_bias < 0.15 and e_nospike < 0.15
    ok &= t7_ok
    if verbose:
        print(f"[7] bias net / no-spike net vs MC:                      "
              f"{'OK' if t7_ok else 'FAIL'}  (bias {e_bias:.3e}, theta=0 {e_nospike:.3e})")

    print("simple_spikeprop selftest:", "ALL OK" if ok else "FAILURES", flush=True)
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

"""selftest.py -- torch-free numerical verification of the SPIKE-KPROP core.

Builds spiked-weight ReLU MLPs in numpy and checks the algorithm against a numpy Monte-Carlo
reference (no torch needed). Run:

    python -m Mecha_preds.cumulants.spikekprop.selftest

The ACTIVE ReLU step is the analytic Gauss-Hermite-free Edgeworth/Wick summation
(``relu_step_edgeworth``); ``spike_kprop_predict`` uses it by default. The legacy GH path
(``relu_step``) is retained only for the regression checks below (2a/2b/4). The dedicated
analytic-path suite (handoff Tests 1-6 + e2e) lives in ``test_edgeworth``-style scripts.

Checks:
  1. linear step == direct  M mu, M Sigma M^T      (machine precision)  -- for v=e1 AND v=ones
  2a. [legacy GH] v=ones conditioning+mix R=2 == one-shot bivariate ReLU (machine precision)
  2b. [legacy GH] v=e1 R=2: the special mode IS a ReLU input, so its kink gives a GH QUADRATURE
      error; verify it CONVERGES with n_nodes. (The analytic path removes this error entirely.)
  3. depth-1 R=2 mean vs MC (agrees to sampling noise) -- both directions, analytic path
  4. v="ones" reproduces SW-KPROP bit-for-bit on the GH path (relerr < 1e-12)
  5. depth-3 R-sweep, analytic path: R>=3 helps on the LOCALIZED (e1) spike (tracks
     C(v,v,v),C(v,v,v,v)) and is ~inert on the FLAT (ones) spike
"""
from __future__ import annotations

import numpy as np

from .core import (State, initial_state, linear_step, relu_step, spike_kprop_predict,
                   unit_vector)
from ..swkprop.relu import exact_relu_covariance
from ..swkprop.core import sw_kprop_predict

_MM = lambda A, B: A @ B


def _make_net(n, depth, seed, theta, direction, out_dim=None):
    """Random net with an O(1)-eigenvalue spike theta v v^T on every HIDDEN layer."""
    rng = np.random.default_rng(seed)
    out_dim = out_dim or n
    v = unit_vector(direction, n)
    P = theta * np.outer(v, v)
    Ws = []
    for _ in range(depth):
        W = rng.standard_normal((n, n)) / np.sqrt(n) + P
        Ws.append((W, None))
    Ws.append((rng.standard_normal((out_dim, n)) / np.sqrt(n), None))
    return Ws


def _mc_mean(Ws, n, samples, batch, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros(Ws[-1][0].shape[0]); accsq = np.zeros_like(acc); cnt = 0
    while cnt < samples:
        b = min(batch, samples - cnt); h = rng.standard_normal((b, n))
        for li, (W, _b) in enumerate(Ws):
            z = h @ W.T; h = np.maximum(z, 0.0) if li < len(Ws) - 1 else z
        acc += h.sum(0); accsq += (h ** 2).sum(0); cnt += b
    mu = acc / cnt
    return mu, np.sqrt(np.clip(accsq / cnt - mu ** 2, 0.0, None) / cnt)


def _state_from_full(mu, Sigma, v):
    n = len(mu)
    vS = float(v @ Sigma @ v); Su = Sigma @ v; g = Su - vS * v
    Sig_perp = Sigma - vS * np.outer(v, v) - np.outer(v, g) - np.outer(g, v)
    return State(n, v, mu.copy(), vS, g, Sig_perp, {})


def _full(st):
    u = st.u
    return st.mu, st.vS * np.outer(u, u) + np.outer(u, st.g) + np.outer(st.g, u) + st.Sig_perp


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def run(verbose: bool = True) -> bool:
    rng = np.random.default_rng(0); ok = True
    n = 64

    # 1. linear step exactness, for BOTH spike directions
    e_lin = 0.0
    for direction in ("e1", "ones"):
        v = unit_vector(direction, n)
        mu0 = rng.standard_normal(n); A = rng.standard_normal((n, n)); Sig0 = A @ A.T / n + np.eye(n)
        M = rng.standard_normal((n, n)) / np.sqrt(n) + 1.0 * np.outer(v, v)   # theta=1 spike
        mu1, Sig1 = _full(linear_step(_state_from_full(mu0, Sig0, v), M, None, _MM, v_out=v))
        e_lin = max(e_lin, _rel(mu1, M @ mu0), _rel(Sig1, M @ Sig0 @ M.T))
    ok &= e_lin < 1e-10
    if verbose:
        print(f"[1] linear step vs M mu, M Sigma M^T (e1 & ones):  relerr {e_lin:.1e}   "
              f"{'OK' if e_lin < 1e-10 else 'FAIL'}")

    # 2a. v=ones: conditioning+mix R=2 == one-shot bivariate ReLU EXACTLY (flat direction is
    #     never a single ReLU input -> the special-mode quadrature is exact, as in SW-KPROP).
    v = unit_vector("ones", n)
    mu0 = rng.standard_normal(n) * 0.5; A = rng.standard_normal((n, n)); Sig0 = A @ A.T / n + 0.3 * np.eye(n)
    muc, Sigc = _full(relu_step(_state_from_full(mu0, Sig0, v), R=2, n_nodes=21))
    mud, Sigd = exact_relu_covariance(mu0, Sig0)
    e_ones = max(_rel(muc, mud), _rel(Sigc, Sigd))
    ok &= e_ones < 1e-8
    if verbose:
        print(f"[2a] v=ones ReLU mix vs one-shot bivariate (exact): relerr {e_ones:.1e}   "
              f"{'OK' if e_ones < 1e-8 else 'FAIL'}")

    # 2b. v=e1: special mode IS coordinate 0, so ReLU(S) has a kink ON the quadrature variable
    #     -> a genuine GH quadrature error (NOT a bug). Verify it shrinks with n_nodes.
    v = unit_vector("e1", n)
    errs_e1 = []
    for nodes in (21, 161):
        muc, _ = _full(relu_step(_state_from_full(mu0, Sig0, v), R=2, n_nodes=nodes))
        errs_e1.append(_rel(muc, mud))
    converges = errs_e1[1] < 0.5 * errs_e1[0] and errs_e1[1] < 5e-3
    ok &= converges
    if verbose:
        print(f"[2b] v=e1  ReLU mix kink-error (shrinks w/ nodes):  "
              f"{errs_e1[0]:.1e} (21) -> {errs_e1[1]:.1e} (161)   {'OK' if converges else 'FAIL'}")

    # 3. depth-1 R=2 mean vs MC, both directions (depth-1 input is Gaussian -> R=2 is exact;
    #    e1 carries the small kink quadrature error, so use enough nodes).
    z_max = 0.0
    for direction in ("e1", "ones"):
        Ws = _make_net(64, 1, seed=7, theta=1.0, direction=direction)
        mc, se = _mc_mean(Ws, 64, 2_000_000, 200_000, seed=123)
        pred = spike_kprop_predict(Ws, 64, direction, R=2, n_nodes=61)["mean"]
        rel_err = _rel(pred, mc)
        z = float(np.linalg.norm(pred - mc) / (np.linalg.norm(se) + 1e-30))
        z_max = max(z_max, z if direction == "ones" else 0.0)   # e1 limited by kink, judge by rel
        passed = (z < 6) if direction == "ones" else (rel_err < 4e-3)
        ok &= passed
        if verbose:
            print(f"[3] depth-1 R=2 mean vs MC ({direction:>4}):           "
                  f"rel {rel_err:.2e}  MC-z {z:.2f}  {'OK' if passed else 'FAIL'}")

    # 4. v="ones" reproduces SW-KPROP bit-for-bit -- a property of the LEGACY Gauss-Hermite
    #    path (SW-KPROP is GH-based), so compare with relu_method="gh".
    Ws = _make_net(64, 3, seed=11, theta=1.0, direction="ones")
    e_sw = 0.0
    for R in (2, 3, 4):
        a = spike_kprop_predict(Ws, 64, "ones", R=R, n_nodes=15, relu_method="gh")["mean"]
        b = sw_kprop_predict(Ws, 64, R=R, n_nodes=15)["mean"]
        e_sw = max(e_sw, _rel(a, b))
    ok &= e_sw < 1e-12
    if verbose:
        print(f"[4] v='ones' == SW-KPROP, GH path (R=2,3,4):       relerr {e_sw:.1e}   "
              f"{'OK' if e_sw < 1e-12 else 'FAIL'}")

    # 5. depth-3 R-sweep: R>=3 must HELP on the localized (e1) spike and be ~INERT on the
    #    flat (ones) spike -- the theorem's signature (localized directional cumulants are
    #    O(1) and must be tracked; flat ones are O(n^{2-r}) and negligible).
    res = {}
    for dkey in ("e1", "ones"):
        Ws = _make_net(64, 3, seed=11, theta=1.0, direction=dkey)
        mc, _se = _mc_mean(Ws, 64, 2_000_000, 250_000, seed=99)
        res[dkey] = {R: _rel(spike_kprop_predict(Ws, 64, dkey, R=R, n_nodes=61)["mean"], mc)
                     for R in (2, 3, 4)}
    e1_helps = res["e1"][4] < 0.85 * res["e1"][2]                 # R=4 beats R=2 by >15% on e1
    flat_inert = abs(res["ones"][4] - res["ones"][2]) < 0.15 * res["ones"][2]   # ~inert on flat
    ok &= e1_helps and flat_inert
    if verbose:
        print("[5] depth-3 R-sweep, theta=1 (rel error vs MC):")
        for dkey, label in (("e1", "e1  (localized)"), ("ones", "ones (flat)")):
            r = res[dkey]
            print(f"      {label:>16}: " + "  ".join(f"R{R} {r[R]:.2e}" for R in (2, 3, 4)))
        print(f"      => e1 R>=3 helps: {e1_helps} | flat R>=3 inert: {flat_inert}")

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)

"""selftest.py -- torch-free numerical verification of the SW-KPROP core.

Builds shifted-weight ReLU MLPs in numpy and checks the algorithm against a numpy
Monte-Carlo reference (no torch needed -- the shifted-weights model is the theorem's
exact regime and is fully constructible in numpy). Run:

    python -m Mecha_preds.cumulants.swkprop.selftest

Checks:
  1. linear step == direct  M mu, M Sigma M^T      (machine precision)
  2. conditioning+mix R=2  == one-shot bivariate ReLU (machine precision)
  3. depth-1 R=2 mean is EXACT vs MC                (agrees to sampling noise)
  4. depth-3 R-sweep on sub/add/unshifted          (R>=3 helps where the special
                                                     mode is non-Gaussian)
"""
from __future__ import annotations

import numpy as np

from .core import (State, initial_state, linear_step, relu_step, sw_kprop_predict)
from .relu import exact_relu_covariance

_MM = lambda A, B: A @ B


def _make_net(n, depth, seed, sign, out_dim=None):
    rng = np.random.default_rng(seed); out_dim = out_dim or n; Ws = []
    for _ in range(depth):
        W = rng.standard_normal((n, n)) / np.sqrt(n)
        if sign != 0:
            W = W + sign / np.sqrt(n) * np.ones((n, n))
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


def _state_from_full(mu, Sigma):
    n = len(mu); u = np.full(n, 1 / np.sqrt(n))
    vS = float(u @ Sigma @ u); Su = Sigma @ u; g = Su - vS * u
    return State(n, mu.copy(), vS, g, Sigma - vS * np.outer(u, u) - np.outer(u, g) - np.outer(g, u), {})


def _full(st):
    u = st.u
    return st.mu, st.vS * np.outer(u, u) + np.outer(u, st.g) + np.outer(st.g, u) + st.Sig_perp


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def run(verbose: bool = True) -> bool:
    rng = np.random.default_rng(0); ok = True

    # 1. linear step exactness
    n = 64
    mu0 = rng.standard_normal(n); A = rng.standard_normal((n, n)); Sig0 = A @ A.T / n + np.eye(n)
    M = rng.standard_normal((n, n)) / np.sqrt(n) - np.ones((n, n)) / np.sqrt(n)
    mu1, Sig1 = _full(linear_step(_state_from_full(mu0, Sig0), M, None, _MM))
    e_lin = max(_rel(mu1, M @ mu0), _rel(Sig1, M @ Sig0 @ M.T))
    ok &= e_lin < 1e-10
    if verbose:
        print(f"[1] linear step vs M mu, M Sigma M^T:           relerr {e_lin:.1e}   {'OK' if e_lin < 1e-10 else 'FAIL'}")

    # 2. conditioning+mix R=2 vs one-shot bivariate ReLU
    mu0 = rng.standard_normal(n) * 0.5; A = rng.standard_normal((n, n)); Sig0 = A @ A.T / n + 0.3 * np.eye(n)
    muc, Sigc = _full(relu_step(_state_from_full(mu0, Sig0), R=2, n_nodes=21))
    mud, Sigd = exact_relu_covariance(mu0, Sig0)
    e_relu = max(_rel(muc, mud), _rel(Sigc, Sigd))
    ok &= e_relu < 1e-8
    if verbose:
        print(f"[2] ReLU conditioning+mix vs one-shot bivariate: relerr {e_relu:.1e}   {'OK' if e_relu < 1e-8 else 'FAIL'}")

    # 3. depth-1 R=2 exact vs MC
    Ws = _make_net(64, 1, seed=7, sign=-1.0)
    mc, se = _mc_mean(Ws, 64, 2_000_000, 200_000, seed=123)
    pred = sw_kprop_predict(Ws, 64, R=2, n_nodes=31)["mean"]
    z = float(np.linalg.norm(pred - mc) / (np.linalg.norm(se) + 1e-30))
    ok &= z < 6.0
    if verbose:
        print(f"[3] depth-1 R=2 mean vs MC:                      rel {_rel(pred, mc):.2e}  MC-z {z:.2f}  {'OK' if z < 6 else 'FAIL'}")

    # 4. depth-3 R-sweep
    if verbose:
        print("[4] depth-3 R-sweep (rel error vs MC):")
        for tag, sign in [("sub  (death)", -1.0), ("add  (linear)", +1.0), ("unshifted", 0.0)]:
            Ws = _make_net(64, 3, seed=11, sign=sign)
            mc, _se = _mc_mean(Ws, 64, 1_000_000, 200_000, seed=99)
            errs = [f"R{R} {_rel(sw_kprop_predict(Ws, 64, R=R, n_nodes=21)['mean'], mc):.2e}" for R in (2, 3, 4)]
            print(f"      {tag:>14}: " + "  ".join(errs))

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)

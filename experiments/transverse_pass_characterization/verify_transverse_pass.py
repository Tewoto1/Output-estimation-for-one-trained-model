"""Verify the transverse-pass characterization (writeups/transverse_cumulant_characterization.md).

Claims tested
-------------
A. Exact Gaussian-contraction identities on dense symmetric tensors T_r, w ~ N(0, I/n):
     mean:  E T_r[w^r] = (r-1)!! pt_r / n^{r/2}   (r even; 0 odd)
     var :  Var T_r[w^r] = n^{-r} * sum_p a_{r,p} ||T^{(p)}||_F^2,
            a_{2,0}=2 ; a_{3,0}=6, a_{3,1}=9 ; a_{4,0}=24, a_{4,1}=72.
B. Annealed identity (the ||a||^2 criterion, exact): kappa_4(w.X) over joint (w,X)
     = 3 * kappa_2(||X||^2) / n^2 ; kappa_3 annealed = 0.
C. Odd-order quenched/annealed gap: per fixed probe w, kappa_3(w.X) scatters with
     sd = sqrt((6||T3||_F^2 + 9||T3^{(1)}||^2)/n^3) (computed EXACTLY for X = B xi),
     while the pooled (annealed) kappa_3 -> 0. The ||X||^2 criterion sees none of this.
D. Model level (e1 spike, theta=-1, depth 3, W ~ N(0,1/n), no bias, X~N(0,I)):
     input-side functionals per post-ReLU layer (transverse bulk = coords 2..n):
       t2 = tr Cov, pt4 = sum_ij k4(c_i,c_i,c_j,c_j)  [probe pairs],
       ||T3||_F^2 [probe triples, split-half cross],  ||T3^{(1)}||^2 [probe singles, split-half]
     -- extensivity: all ~ n (slope ~ 1 across widths);
     downstream closure at the next layer's transverse coordinates h_b:
       kappa_4(h_b) ~= 3*pt4/n^2 ,  RMS_b kappa_3(h_b) ~= sqrt((6||T3||^2+9||T3^{(1)}||^2)/n^3).

MC budgets: A: 2e5 probe draws (n=12). B: 1.5e6 joint draws (n=48). C: 24 probe seeds x 4e5.
D: widths 64/128 (3 seeds) and 256 (2 seeds), 2e6 samples, float32 matmuls / float64 stats.
Results cached in stats_cache.json next to this file (delete or --force to recompute).

Run: python verify_transverse_pass.py [--parts abcd] [--quick] [--force]
"""

import argparse
import itertools
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "stats_cache.json")


# ----------------------------------------------------------------- utilities

def central_k34(raw, N):
    """raw = [S1,S2,S3,S4] power sums of a scalar sample -> (k2,k3,k4) central."""
    m1 = raw[0] / N
    m2 = raw[1] / N
    m3 = raw[2] / N
    m4 = raw[3] / N
    c2 = m2 - m1 ** 2
    c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return c2, c3, c4 - 3 * c2 ** 2


def sym3(T):
    out = np.zeros_like(T)
    for p in itertools.permutations(range(3)):
        out += np.transpose(T, p)
    return out / 6.0


def sym4(T):
    out = np.zeros_like(T)
    for p in itertools.permutations(range(4)):
        out += np.transpose(T, p)
    return out / 24.0


def block_se(vals, nblocks=25):
    """SE of the mean and of the variance of `vals` via block splitting."""
    blocks = np.array_split(vals, nblocks)
    bm = np.array([b.mean() for b in blocks])
    bv = np.array([b.var() for b in blocks])
    return bm.std(ddof=1) / math.sqrt(nblocks), bv.std(ddof=1) / math.sqrt(nblocks)


def report(name, measured, predicted, se, results, tol_z=4.0, tol_rel=0.05):
    z = abs(measured - predicted) / se if se > 0 else float("inf")
    rel = abs(measured - predicted) / abs(predicted) if predicted != 0 else float("nan")
    ok = (z < tol_z) or (rel < tol_rel)
    rtxt = f"{rel:6.1%}" if predicted != 0 else "   n/a"
    print(f"  {name:44s} meas {measured:+.5g}  pred {predicted:+.5g}  z={z:5.2f}  "
          f"rel={rtxt}  {'PASS' if ok else 'FAIL'}")
    results[name] = dict(measured=measured, predicted=predicted, z=z, rel=rel,
                         ok=bool(ok))
    return ok


# ------------------------------------------------------- A: exact identities

def part_A(quick, results):
    print("\n[A] Gaussian-contraction identities (dense tensors, n=12)")
    rng = np.random.default_rng(0)
    n = 12
    NW = 60_000 if quick else 240_000
    A2 = rng.standard_normal((n, n)); A2 = (A2 + A2.T) / 2
    T3 = sym3(rng.standard_normal((n,) * 3))
    T4 = sym4(rng.standard_normal((n,) * 4))

    # exact functionals
    fro2_2 = float((A2 ** 2).sum());            tr2 = float(np.trace(A2))
    fro2_3 = float((T3 ** 2).sum());            m3 = np.einsum("ijj->i", T3)
    fro2_4 = float((T4 ** 2).sum())
    T41 = np.einsum("kkij->ij", T4);            pt4 = float(np.einsum("iijj->", T4))

    v2 = np.empty(NW); v3 = np.empty(NW); v4 = np.empty(NW)
    cs = 4000
    for s in range(0, NW, cs):
        w = rng.standard_normal((min(cs, NW - s), n)) / math.sqrt(n)
        v2[s:s + len(w)] = ((w @ A2) * w).sum(1)
        t1 = np.tensordot(w, T3, axes=(1, 0))
        v3[s:s + len(w)] = (np.einsum("sjk,sj->sk", t1, w, optimize=True) * w).sum(1)
        t1 = np.tensordot(w, T4, axes=(1, 0))
        t2 = np.einsum("sjkl,sj->skl", t1, w, optimize=True)
        v4[s:s + len(w)] = (np.einsum("skl,sk->sl", t2, w, optimize=True) * w).sum(1)

    for r, vals, mpred, vpred in [
        (2, v2, tr2 / n,            2 * fro2_2 / n ** 2),
        (3, v3, 0.0,                (6 * fro2_3 + 9 * float(m3 @ m3)) / n ** 3),
        (4, v4, 3 * pt4 / n ** 2,   (24 * fro2_4 + 72 * float((T41 ** 2).sum())) / n ** 4),
    ]:
        se_m, se_v = block_se(vals)
        report(f"A.mean r={r}", vals.mean(), mpred, se_m, results)
        report(f"A.var  r={r}", vals.var(), vpred, se_v, results)


# ---------------------------------------------------- B: annealed = ||X||^2

def part_B(quick, results):
    print("\n[B] Annealed identity: kappa_4(w.X) = 3 kappa_2(||X||^2)/n^2")
    rng = np.random.default_rng(1)
    n = 48
    N = 400_000 if quick else 1_500_000
    k = 0.6                                   # gamma shape: skewed, kurtotic
    Bm = np.eye(n) + 0.25 * rng.standard_normal((n, n)) / math.sqrt(n)
    raw_y = np.zeros(4); raw_q = np.zeros(4)
    ys = []; qs = []
    cs = 100_000
    for s in range(0, N, cs):
        b = min(cs, N - s)
        xi = (rng.gamma(k, 1.0, size=(b, n)) - k) / math.sqrt(k)
        rs = np.sqrt(rng.choice([0.8, 1.2], size=b))     # global radius: big kappa_2(||X||^2)
        X = rs[:, None] * (xi @ Bm.T)
        w = rng.standard_normal((b, n)) / math.sqrt(n)   # FRESH w each sample = annealed
        y = (w * X).sum(1); q = (X * X).sum(1)
        for p in range(4):
            raw_y[p] += (y ** (p + 1)).sum(); raw_q[p] += (q ** (p + 1)).sum()
        ys.append(y); qs.append(q)
    y = np.concatenate(ys); q = np.concatenate(qs)
    k2y, k3y, k4y = central_k34(raw_y, N)
    k2q, _, _ = central_k34(raw_q, N)
    # block SEs
    se_k4 = np.array([central_k34([ (b**1).sum(), (b**2).sum(), (b**3).sum(), (b**4).sum() ], len(b))[2]
                      for b in np.array_split(y, 25)]).std(ddof=1) / 5
    se_k3 = np.array([central_k34([ (b**1).sum(), (b**2).sum(), (b**3).sum(), (b**4).sum() ], len(b))[1]
                      for b in np.array_split(y, 25)]).std(ddof=1) / 5
    report("B.k4_annealed = 3*k2(normsq)/n^2", k4y, 3 * k2q / n ** 2, se_k4, results)
    report("B.k3_annealed = 0 (mixture symmetric)", k3y, 0.0, max(se_k3, 1e-12), results)
    print(f"    (kappa_2(||X||^2) = {k2q:.4g}; the radius mixture makes the annealed "
          f"kappa_4 O(1)-visible: criterion binds)")


# --------------------------------------------- C: odd-order quenched scatter

def part_C(quick, results):
    print("\n[C] Quenched kappa_3 scatter vs exact T3 functionals (annealed sees 0)")
    rng = np.random.default_rng(2)
    n = 48
    k = 0.6; k3xi = 2.0 / math.sqrt(k)
    NSEED = 24; N = 150_000 if quick else 400_000
    Bm = np.eye(n) + 0.25 * rng.standard_normal((n, n)) / math.sqrt(n)
    # exact functionals of T3_{abc} = k3xi * sum_i B_ai B_bi B_ci
    Gm = Bm.T @ Bm
    fro2 = k3xi ** 2 * float((Gm ** 3).sum())
    d = (Bm ** 2).sum(0)
    mvec = k3xi * (Bm @ d)
    sd_pred = math.sqrt((6 * fro2 + 9 * float(mvec @ mvec)) / n ** 3)

    k3s = []
    for seed in range(NSEED):
        w = np.random.default_rng(100 + seed).standard_normal(n) / math.sqrt(n)
        p = Bm.T @ w
        raw = np.zeros(4); ncum = 0
        for s in range(0, N, 200_000):
            b = min(200_000, N - s)
            xi = (rng.gamma(k, 1.0, size=(b, n)) - k) / math.sqrt(k)
            y = xi @ p
            for j in range(4):
                raw[j] += (y ** (j + 1)).sum()
            ncum += b
        k3s.append(central_k34(raw, ncum)[1])
    k3s = np.array(k3s)
    sd_emp = k3s.std(ddof=1)
    ratio = sd_emp / sd_pred
    ok = 0.66 < ratio < 1.5           # chi^2 band for 23 dof
    print(f"  quenched sd(kappa_3): emp {sd_emp:.4g}  pred {sd_pred:.4g}  "
          f"ratio {ratio:.3f}  {'PASS' if ok else 'FAIL'}")
    print(f"  per-seed values span [{k3s.min():+.3g}, {k3s.max():+.3g}]; "
          f"mean {k3s.mean():+.3g} (annealed limit 0)")
    results["C.sd_ratio"] = dict(measured=sd_emp, predicted=sd_pred, ratio=ratio, ok=bool(ok))
    results["C.seed_mean"] = dict(measured=float(k3s.mean()), predicted=0.0,
                                  ok=bool(abs(k3s.mean()) < 3 * sd_emp / math.sqrt(NSEED)))


# --------------------------------------------------------- D: model level

def relu(x):
    return np.maximum(x, 0.0)


def make_net(n, seed, theta=-1.0, depth=3):
    rng = np.random.default_rng(seed)
    Ms = []
    for _ in range(depth):
        W = (rng.standard_normal((n, n)) / math.sqrt(n)).astype(np.float32)
        W[0, 0] += theta                      # e1 e1^T spike, hidden layers only
        Ms.append(W)
    return Ms


def part_D(quick, results, cache=None, deadline=None):
    print("\n[D] e1-spiked ReLU net: input-side functionals + downstream closure")
    widths_seeds = [(64, (1, 2, 3)), (128, (1, 2, 3)), (256, (1, 2))]
    N = 600_000 if quick else 2_000_000
    P4, P3, P1, NCOORD = 48, 64, 48, 12
    tab = dict(cache.get("D_configs", {})) if cache is not None else {}
    for n, seeds in widths_seeds:
        prng = np.random.default_rng(777 + n)
        Gp = prng.standard_normal((n - 1, 2 * P4)).astype(np.float32)       # pairs
        Gt = prng.standard_normal((n - 1, 3 * P3)).astype(np.float32)       # triples
        Gs = prng.standard_normal((n - 1, P1)).astype(np.float32)           # singles
        sel = prng.choice(np.arange(1, n), size=NCOORD, replace=False)      # transverse coords
        for seed in seeds:
            if any(k.startswith(f"n{n}_s{seed}_") for k in tab):
                continue                                   # cached config
            if deadline is not None and time.time() > deadline:
                print("  (time budget reached -- rerun to continue)")
                if cache is not None:
                    cache["D_configs"] = tab
                return False
            t0 = time.time()
            Ms = make_net(n, seed)
            # ---- pass 1: means
            rng = np.random.default_rng(10_000 + seed)
            N1 = 300_000
            mu = [np.zeros(n) for _ in range(4)]     # a1, a2, h2, h3
            for s in range(0, N1, 100_000):
                b = min(100_000, N1 - s)
                x = rng.standard_normal((b, n), dtype=np.float32)
                a1 = relu(x @ Ms[0].T); h2 = a1 @ Ms[1].T
                a2 = relu(h2);          h3 = a2 @ Ms[2].T
                for arr, m in zip((a1, a2, h2, h3), mu):
                    m += arr.sum(0, dtype=np.float64)
            mu = [m / N1 for m in mu]
            # ---- pass 2: stats
            L = {}
            for li in (0, 1):
                L[li] = dict(
                    c2=0.0, cresid=np.zeros(n - 1),
                    pair=np.zeros((8, P4)),                # x,y,x2,y2,xy,x2y,xy2,x2y2
                    tri=[np.zeros((7, P3)), np.zeros((7, P3))],   # halves: 111,011,101,110,100,010,001
                    sing=[np.zeros((3, P1)), np.zeros((3, P1))],  # halves: x, q, xq
                    nh=[0, 0],
                    dsum=np.zeros((4, NCOORD)),
                )
            rng = np.random.default_rng(20_000 + seed)
            done = 0
            ci = 0
            while done < N:
                b = min(100_000, N - done)
                x = rng.standard_normal((b, n), dtype=np.float32)
                a1 = relu(x @ Ms[0].T); h2 = a1 @ Ms[1].T
                a2 = relu(h2);          h3 = a2 @ Ms[2].T
                for li, (a, h, mua, muh) in enumerate(((a1, h2, mu[0], mu[2]),
                                                       (a2, h3, mu[1], mu[3]))):
                    st = L[li]
                    c = a[:, 1:] - mua[1:].astype(np.float32)
                    st["c2"] += float((c.astype(np.float64) ** 2).sum())
                    st["cresid"] += c.sum(0, dtype=np.float64)
                    q = (c * c).sum(1, dtype=np.float64)
                    half = ci % 2
                    # pairs -> pt4
                    Y = (c @ Gp).astype(np.float64)
                    X_, Y_ = Y[:, 0::2], Y[:, 1::2]
                    for j, arr in enumerate((X_, Y_, X_**2, Y_**2, X_*Y_,
                                             X_**2*Y_, X_*Y_**2, X_**2*Y_**2)):
                        st["pair"][j] += arr.sum(0)
                    # triples -> ||T3||_F^2 (split-half)
                    Z = (c @ Gt).astype(np.float64)
                    z1, z2, z3 = Z[:, 0::3], Z[:, 1::3], Z[:, 2::3]
                    T = st["tri"][half]
                    for j, arr in enumerate((z1*z2*z3, z2*z3, z1*z3, z1*z2, z1, z2, z3)):
                        T[j] += arr.sum(0)
                    # singles -> ||T3^{(1)}||^2 (split-half)
                    S_ = (c @ Gs).astype(np.float64)
                    Sg = st["sing"][half]
                    Sg[0] += S_.sum(0); Sg[1] += q.sum(); Sg[2] += (S_ * q[:, None]).sum(0)
                    st["nh"][half] += b
                    # downstream coords
                    dcs = h[:, sel].astype(np.float64) - muh[sel]
                    for p in range(4):
                        st["dsum"][p] += (dcs ** (p + 1)).sum(0)
                done += b
                ci += 1

            for li in (0, 1):
                st = L[li]
                nb = st["nh"][0] + st["nh"][1]
                # t2
                resid = st["cresid"] / nb
                t2 = st["c2"] / nb - float(resid @ resid)
                # pt4 from pairs (central conversion)
                P_ = st["pair"] / nb
                x1, y1, x2, y2, xy, x2y, xy2, x2y2 = P_
                cx2 = x2 - x1**2; cy2 = y2 - y1**2; cxy = xy - x1*y1
                cx2y2 = (x2y2 - 2*x1*xy2 - 2*y1*x2y + x1**2*y2 + y1**2*x2
                         + 4*x1*y1*xy - 3*x1**2*y1**2)
                pt4_each = cx2y2 - cx2*cy2 - 2*cxy**2
                pt4_m = float(pt4_each.mean())
                pt4_se = float(pt4_each.std(ddof=1) / math.sqrt(P4))
                # ||T3||^2 from triples, split-half cross
                th = []
                for hh in (0, 1):
                    T = st["tri"][hh] / st["nh"][hh]
                    m111, m011, m101, m110, m100, m010, m001 = T
                    th.append(m111 - m100*m011 - m010*m101 - m001*m110
                              + 2*m100*m010*m001)
                cross3 = th[0] * th[1]
                fro3_m = float(cross3.mean()); fro3_se = float(cross3.std(ddof=1)/math.sqrt(P3))
                # ||T3^{(1)}||^2 from singles, split-half cross
                sh = []
                for hh in (0, 1):
                    Sg = st["sing"][hh] / st["nh"][hh]
                    sh.append(Sg[2] - Sg[0] * Sg[1])
                cross1 = sh[0] * sh[1]
                m31_m = float(cross1.mean()); m31_se = float(cross1.std(ddof=1)/math.sqrt(P1))
                # downstream cumulants
                D_ = st["dsum"]
                k2d = np.empty(NCOORD); k3d = np.empty(NCOORD); k4d = np.empty(NCOORD)
                for j in range(NCOORD):
                    k2d[j], k3d[j], k4d[j] = central_k34(D_[:, j], nb)
                pred_k4 = 3 * pt4_m / n ** 2
                pred_k3rms = math.sqrt(max(6 * fro3_m + 9 * m31_m, 0.0) / n ** 3)
                key = f"n{n}_s{seed}_L{li+1}"
                tab[key] = dict(n=int(n), seed=int(seed), layer=li + 1, t2=t2,
                                pt4=pt4_m, pt4_se=pt4_se, fro3=fro3_m, fro3_se=fro3_se,
                                m31=m31_m, m31_se=m31_se,
                                k4_down=float(k4d.mean()),
                                k4_down_se=float(k4d.std(ddof=1)/math.sqrt(NCOORD)),
                                k4_pred=pred_k4,
                                k3_down_rms=float(np.sqrt((k3d**2).mean())),
                                k3_pred_rms=pred_k3rms,
                                k2_down=float(k2d.mean()), k2_pred=t2 / n)
            print(f"  n={n} seed={seed} done in {time.time()-t0:.0f}s")
            if cache is not None:
                cache["D_configs"] = tab
                with open(CACHE, "w") as f:
                    json.dump(cache, f, indent=1, default=float)

    # ---- report
    print(f"\n  {'cfg':14s} {'t2/n':>7s} {'pt4/n':>7s} {'fro3/n':>7s} {'m31/n':>7s} "
          f"{'k4_dn':>9s} {'k4_pred':>9s} {'k3rms_dn':>9s} {'k3rms_pr':>9s}")
    for key, r in tab.items():
        n = r["n"]
        print(f"  {key:14s} {r['t2']/n:7.3f} {r['pt4']/n:7.3f} {r['fro3']/n:7.3f} "
              f"{r['m31']/n:7.3f} {r['k4_down']:9.5f} {r['k4_pred']:9.5f} "
              f"{r['k3_down_rms']:9.5f} {r['k3_pred_rms']:9.5f}")
    # slopes across widths (seeds pooled, per layer)
    for li in (1, 2):
        for fn in ("t2", "pt4", "fro3", "m31"):
            xs, ys = [], []
            for r in tab.values():
                if r["layer"] == li and r[fn] > 0:
                    xs.append(math.log(r["n"])); ys.append(math.log(r[fn]))
            if len(set(xs)) > 1:
                slope = np.polyfit(xs, ys, 1)[0]
                ok = abs(slope - 1.0) < 0.35
                print(f"  slope[{fn} ~ n^s]  layer {li}:  s = {slope:5.2f}  "
                      f"(class: 1)  {'PASS' if ok else 'FAIL'}")
                results[f"D.slope_{fn}_L{li}"] = dict(slope=slope, ok=bool(ok))
    # downstream closure ratios
    rk4 = [r["k4_down"] / r["k4_pred"] for r in tab.values() if abs(r["k4_pred"]) > 0]
    rk3 = [r["k3_down_rms"] / r["k3_pred_rms"] for r in tab.values() if r["k3_pred_rms"] > 0]
    rk2 = [r["k2_down"] / r["k2_pred"] for r in tab.values()]
    for nm, rr, band in (("k4_down/3pt4*n^-2", rk4, (0.6, 1.5)),
                         ("k3rms_down/pred", rk3, (0.5, 1.7)),
                         ("k2_down/(t2/n)", rk2, (0.9, 1.12))):
        gm = float(np.exp(np.mean(np.log(np.abs(rr)))))
        ok = band[0] < gm < band[1]
        print(f"  closure ratio {nm:24s} geomean {gm:.3f}  "
              f"spread [{min(rr):.2f},{max(rr):.2f}]  {'PASS' if ok else 'FAIL'}")
        results[f"D.closure_{nm}"] = dict(geomean=gm, lo=float(min(rr)),
                                          hi=float(max(rr)), ok=bool(ok))
    results["D.table"] = tab
    return True


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="abcd")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--maxsec", type=float, default=None,
                    help="soft time budget; part D resumes on next run")
    args = ap.parse_args()

    cache = {}
    if os.path.exists(CACHE) and not args.force:
        with open(CACHE) as f:
            cache = json.load(f)

    results = cache.get("results", {})
    t0 = time.time()
    deadline = (t0 + args.maxsec) if args.maxsec else None
    for p, fn in (("a", part_A), ("b", part_B), ("c", part_C), ("d", part_D)):
        tag = f"part_{p}_done"
        if p in args.parts:
            if cache.get(tag) and not args.force:
                print(f"\n[{p.upper()}] cached -- skipping (use --force to redo)")
                continue
            if p == "d":
                done = part_D(args.quick, results, cache=cache, deadline=deadline)
                if not done:
                    cache["results"] = results
                    with open(CACHE, "w") as f:
                        json.dump(cache, f, indent=1, default=float)
                    print("== part D partial; rerun to continue")
                    return
            else:
                fn(args.quick, results)
            cache[tag] = True
            cache["results"] = results
            with open(CACHE, "w") as f:
                json.dump(cache, f, indent=1, default=float)
    n_ok = sum(1 for v in results.values() if isinstance(v, dict) and "ok" in v and v["ok"])
    n_all = sum(1 for v in results.values() if isinstance(v, dict) and "ok" in v)
    print(f"\n== {n_ok}/{n_all} checks passed ({time.time()-t0:.0f}s) -> {CACHE}")


if __name__ == "__main__":
    main()

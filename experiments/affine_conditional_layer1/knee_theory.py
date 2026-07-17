"""Layer-2 knee: exact-in-surrogate predicted curves at scale (n=2048 seed 0, + g at 4096).

Latent form of the layer-2 conditioning. Layer-1 exact: s ~ N(0,tau2); A = ReLU(s);
B|s ~ N(m0 + m1 s, Sigma0) up to O(1/n). Layer 2: A' = c A + w^T B, Z'_i = u_i A + v_i^T B.
So with xi ~ N(0, om2), om2 = w^T Sigma0 w, kap = w^T m1, mu = w^T m0:

    A'   = c ReLU(s) + kap s + mu + xi
    E[Z'_i | a'] = rho_i a' + (u_i - rho_i c) g1(a') + (v_i^T m1 - rho_i kap) g2(a') + const
    g1(a') = E[ReLU(s)|a'],  g2(a') = E[s|a'],  rho_i = v_i^T Sigma0 w / om2.

Both coefficient vectors are O(1/sqrt(n)); the affine-residual across coordinates is
rank-<=2 (empirically rank-1: g1,g2 residuals nearly proportional). Also computes the
lambda-direction conditional-variance profile (the rank-one cov knee):
    Var(lam^T Z' | a') = Var_post[(lam^T u - b c) ReLU(s) + (lam^T V m1 - b kap) s] + const.
"""
import os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
CKPT = os.path.join(REPO, "checkpoints", "affine_conditional_layer1")
from Mecha_preds._utils import relu_moments_1d, exact_relu_covariance, _phi
from Mecha_preds.binned_kprop.empirical_structure import build_spiked_net

def wproj_out(a, w, Y):
    """Project columns-of-Y... here Y (P, S): remove weighted affine part in a per row."""
    w = w / w.sum()
    X = np.stack([np.ones_like(a), a], 1)
    Xw = X * w[:, None]
    co = np.linalg.solve(X.T @ Xw, Xw.T @ Y.T)
    return Y - (X @ co).T

def knee_theory(n, seed, s_pts=1601, a_pts=241, smax=6.0, amax=4.0, full_sigma0=True):
    t0 = time.time()
    Ws = build_spiked_net(n, depth=2, seed=seed)
    M, Mp = Ws[0][0], Ws[1][0]
    G = M @ M.T
    tau2 = G[0, 0]; tau = np.sqrt(tau2)
    beta = G[1:, 0] / tau2
    C = G[1:, 1:] - np.outer(beta, beta) * tau2          # cond bulk cov given s (const)
    sig2 = np.diag(C)

    # layer-1 post-act affine surrogate B|s: exact curves -> weighted affine fit
    sg = tau * np.linspace(-4, 4, 41); wg = _phi(sg / tau); wg /= wg.sum()
    MU = beta[:, None] * sg[None, :]
    mean_c, _, _ = relu_moments_1d(MU, np.broadcast_to(sig2[:, None], MU.shape))
    X = np.stack([np.ones_like(sg), sg], 1); Xw = X * wg[:, None]
    co = np.linalg.solve(X.T @ Xw, Xw.T @ mean_c.T)      # (2, d)
    m0, m1 = co[0], co[1]

    c = Mp[0, 0]; w = Mp[0, 1:]; u = Mp[1:, 0]; V = Mp[1:, 1:]
    if full_sigma0:
        _, Sigma0 = exact_relu_covariance(np.zeros(n - 1), C)   # bulk post cov at s=0
        S0w = Sigma0 @ w
    else:                                                # diag approx (for big n)
        _, _, var0 = relu_moments_1d(np.zeros(n - 1), sig2)
        S0w = var0 * w
    om2 = float(w @ S0w); kap = float(w @ m1); mu_ = float(w @ m0)
    rho = (V @ S0w) / om2
    Vm1 = V @ m1
    coef1 = u - rho * c                                  # multiplies g1 residual
    coef2 = Vm1 - rho * kap                              # multiplies g2 residual
    lam_naive = coef1 + coef2                            # ~ knee amplitude per coord

    # posterior quadrature over latent s given a'
    EA = tau * _phi(np.zeros(1))[0]
    varA = tau2 / 2 - EA ** 2
    sAp2 = c * c * varA + om2 + kap ** 2 * tau2 + 2 * c * kap * (tau2 / 2)
    sAp = np.sqrt(sAp2)
    mAp = c * EA + mu_
    s = np.linspace(-smax * tau, smax * tau, s_pts)
    ap = mAp + sAp * np.linspace(-amax, amax, a_pts)
    prior = _phi(s / tau) / tau
    mean_ch = c * np.maximum(s, 0) + kap * s + mu_       # channel mean given s
    Lk = _phi((ap[None, :] - mean_ch[:, None]) / np.sqrt(om2))   # (s_pts, a_pts)
    post = prior[:, None] * Lk
    Zc = np.trapezoid(post, s, axis=0)                   # \propto p_{A'}(a')
    g1 = np.trapezoid(post * np.maximum(s, 0)[:, None], s, axis=0) / Zc
    g2 = np.trapezoid(post * s[:, None], s, axis=0) / Zc
    p_ap = Zc / np.trapezoid(Zc, ap)

    # per-coordinate predicted mean curves + affine residuals (mass-weighted)
    wq = p_ap.copy()
    G1r = wproj_out(ap, wq, g1[None, :])[0]
    G2r = wproj_out(ap, wq, g2[None, :])[0]
    # rank-1 check between the two profiles
    ip = lambda x, y: float((wq / wq.sum() * x * y).sum())
    cos12 = ip(G1r, G2r) / np.sqrt(ip(G1r, G1r) * ip(G2r, G2r))
    R = coef1[:, None] * G1r[None, :] + coef2[:, None] * G2r[None, :]   # (d, a_pts) residual pred
    # top component
    sw = np.sqrt(wq / wq.sum())
    Uu, Ss, Vt = np.linalg.svd(R * sw[None, :], full_matrices=False)
    lam_hat = Uu[:, 0] * Ss[0]                           # signed amplitudes
    prof = Vt[0] / sw                                    # shared profile (unit in weighted norm)
    r1share = Ss[0] ** 2 / (Ss ** 2).sum()

    # pooled 1-R2 of affine fit on predicted curves (vs their own variation)
    full_curve = rho[:, None] * ap[None, :] + coef1[:, None] * g1[None, :] + coef2[:, None] * g2[None, :]
    wn = wq / wq.sum()
    ybar = (wn[None, :] * full_curve).sum(1, keepdims=True)
    ss_tot = (wn[None, :] * (full_curve - ybar) ** 2).sum(1)
    ss_res = (wn[None, :] * R ** 2).sum(1)
    pooled = float(ss_res.sum() / ss_tot.sum())
    # windowed (safe region): windows in units of (a'-mAp)/sAp
    t = (ap - mAp) / sAp
    win = {}
    for name, mask in [("full", np.ones_like(t, bool)), ("branch a'>mAp+1.5s", t > 1.5),
                       ("plateau a'<mAp-1.5s", t < -1.5), ("strip |t|<=1.5", np.abs(t) <= 1.5)]:
        ww = wn * mask
        if ww.sum() < 1e-12: continue
        Yw = full_curve[:, mask]; aw = ap[mask]; www = wn[mask] / wn[mask].sum()
        Xw_ = np.stack([np.ones_like(aw), aw], 1) * www[:, None]
        X_ = np.stack([np.ones_like(aw), aw], 1)
        co_ = np.linalg.solve(X_.T @ Xw_, Xw_.T @ Yw.T)
        res = Yw - (X_ @ co_).T
        yb = (www[None, :] * Yw).sum(1, keepdims=True)
        win[name] = float((www[None, :] * res ** 2).sum() /
                          (www[None, :] * (Yw - yb) ** 2).sum())

    # rank-one cov knee: variance profile along lam_hat direction
    ln = lam_hat / np.linalg.norm(lam_hat)
    a1 = float(ln @ u) - float(ln @ (V @ S0w)) / om2 * c
    a2 = float(ln @ Vm1) - float(ln @ (V @ S0w)) / om2 * kap
    fmix = a1 * np.maximum(s, 0) + a2 * s
    m_f = np.trapezoid(post * fmix[:, None], s, axis=0) / Zc
    m_f2 = np.trapezoid(post * (fmix ** 2)[:, None], s, axis=0) / Zc
    var_q = m_f2 - m_f ** 2                              # + const (dropped)

    out = dict(n=n, seed=seed, tau2=tau2, c=c, om2=om2, kap=kap, mu=mu_, mAp=mAp, sAp=sAp,
               ap=ap, p_ap=p_ap, g1=g1, g2=g2, prof=prof, lam_hat=lam_hat, rho=rho,
               coef1=coef1, coef2=coef2, r1share=r1share, cos12=cos12, pooled=pooled,
               var_q=var_q, lam_rms_sqrtn=float(np.sqrt((lam_hat ** 2).mean()) * np.sqrt(n)),
               curves_full=full_curve[np.argsort(-np.abs(lam_hat))[:4]],
               resid_pred=R[np.argsort(-np.abs(lam_hat))[:4]],
               top_idx=np.argsort(-np.abs(lam_hat))[:4],
               windows=str(win))
    np.savez_compressed(os.path.join(CKPT, f"knee2_theory_n{n}_seed{seed}.npz"), **out)
    print(f"n={n} seed={seed} ({time.time()-t0:.0f}s): om2={om2:.3f} kap={kap:+.3f} c={c:.3f} "
          f"sAp={sAp:.3f}")
    print(f"  pooled PRE 1-R2 (predicted) = {pooled:.4f}   rank1 share = {r1share:.4f} "
          f"cos(g1r,g2r) = {cos12:.4f}   rms(lam)*sqrt(n) = {out['lam_rms_sqrtn']:.3f}")
    print(f"  windows: {win}")
    return out

if __name__ == "__main__":
    r2048 = knee_theory(2048, 0, full_sigma0=True)
    r4096 = knee_theory(4096, 0, full_sigma0=False)   # diag-Sigma0 om2 (off-diag = O(1/sqrt n) rel)

"""Analyze the n=2048 layer-2 knee MC vs theory; make the verification figure."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(REPO, "checkpoints", "affine_conditional_layer1")
FIGD = os.path.join(CKPT, "figs")
n, seed = 2048, 0
th = np.load(os.path.join(CKPT, f"knee2_theory_n{n}_seed{seed}.npz"))
mc = np.load(os.path.join(CKPT, f"knee2_mc_n{n}_seed{seed}_N960000_B24.npz"))

cnt, sa, SZ = mc["cnt"], mc["sa"], mc["SZ"]
cnt_t = cnt.sum(0); ab = sa.sum(0) / cnt_t
p = cnt_t / cnt_t.sum()
mu = SZ.sum(0) / cnt_t[:, None]                    # (bins, d) pooled
muA = SZ[0] / cnt[0][:, None]; muB = SZ[1] / cnt[1][:, None]

def affine_resid(a, Y, w):
    w = w / w.sum(); X = np.stack([np.ones_like(a), a], 1); Xw = X * w[:, None]
    co = np.linalg.solve(X.T @ Xw, Xw.T @ Y)
    return Y - X @ co, co

RA, _ = affine_resid(ab, muA, p); RB, _ = affine_resid(ab, muB, p); Rp, co_p = affine_resid(ab, mu, p)

# debiased pooled 1-R2 (cross-half products)
wn = p / p.sum()
dA = muA - (wn[:, None] * muA).sum(0); dB = muB - (wn[:, None] * muB).sum(0)
ss_res = float((wn[:, None] * RA * RB).sum())
ss_tot = float((wn[:, None] * dA * dB).sum())
print(f"debiased pooled PRE 1-R2 (MC, n=2048) = {ss_res/ss_tot:.4f}   theory predicted = {float(th['pooled']):.4f}")

# shared profile: weighted SVD of pooled residual, split-half cosine
sw = np.sqrt(wn)
U_, S_, Vt_ = np.linalg.svd((Rp * sw[:, None]).T, full_matrices=False)   # coords x bins
lamE = U_[:, 0] * S_[0]; profE = Vt_[0] / sw
r1 = S_[0] ** 2 / (S_ ** 2).sum()
UA, SA_, VtA = np.linalg.svd((RA * sw[:, None]).T, full_matrices=False)
UB, SB_, VtB = np.linalg.svd((RB * sw[:, None]).T, full_matrices=False)
cosAB = abs(float((VtA[0] * VtB[0]).sum()))
# theory profile at bin abscissas (residualized on the same bins)
g_th_curves = th["coef1"][:, None] * np.interp(ab, th["ap"], th["g1"])[None, :] \
            + th["coef2"][:, None] * np.interp(ab, th["ap"], th["g2"])[None, :] \
            + th["rho"][:, None] * ab[None, :]
R_th, _ = affine_resid(ab, g_th_curves.T, p)
Ut, St, Vtt = np.linalg.svd((R_th * sw[:, None]).T, full_matrices=False)
profT = Vtt[0] / sw; lamT = Ut[:, 0] * St[0]
ipw = lambda x, y: float((wn * x * y).sum())
cosET = abs(ipw(profE, profT) / np.sqrt(ipw(profE, profE) * ipw(profT, profT)))
print(f"rank-1 share (MC pooled) = {r1:.3f}   split-half profile cos = {cosAB:.4f}   cos(MC prof, theory prof) = {cosET:.4f}")

# lambda amplitudes: project halves on theory profile, debias via cross product
pT = profT / np.sqrt(ipw(profT, profT))
lamA = (wn[None, :] * RA.T * pT[None, :]).sum(1); lamB = (wn[None, :] * RB.T * pT[None, :]).sum(1)
lam_rms = np.sqrt(max(0.0, float((lamA * lamB).mean())))
lamTn = (wn[None, :] * R_th.T * pT[None, :]).sum(1)
print(f"rms knee amplitude x sqrt(n): MC = {lam_rms*np.sqrt(n):.3f}   theory = {np.sqrt((lamTn**2).mean())*np.sqrt(n):.3f}")
print(f"cos(lambda_MC_debiased?, lambda_theory) = {float((lamA+lamB) @ lamTn / (np.linalg.norm(lamA+lamB)*np.linalg.norm(lamTn))):.4f}")

# variance knee along lambda vs control
Vq = mc["Sq2"] / cnt_t - (mc["Sq"] / cnt_t) ** 2
Vc = mc["Sc2"] / cnt_t - (mc["Sc"] / cnt_t) ** 2
vq_th = np.interp(ab, th["ap"], th["var_q"])
off = float((wn * (Vq - vq_th)).sum())
resid_v = Vq - (vq_th + off)
print(f"Var(lam^T Z'|bin): range MC = {Vq.max()-Vq.min():.3f}, theory = {vq_th.max()-vq_th.min():.3f}, "
      f"rms mismatch = {np.sqrt((wn*resid_v**2).sum()):.4f}")
print(f"control direction: range = {Vc.max()-Vc.min():.3f} (should be ~flat)")

# ---------------------------------------------------------------- figure
i0 = int(np.argsort(-np.abs(lamTn))[0])
se = np.sqrt(0.55 / cnt_t)
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
sAp = float(th["sAp"]); mAp = float(th["mAp"]); t = (ab - mAp) / sAp
tf = (th["ap"] - mAp) / sAp
fine = th["rho"][i0] * th["ap"] + th["coef1"][i0] * th["g1"] + th["coef2"][i0] * th["g2"]
fine = fine - np.interp(ab, th["ap"], fine) @ wn + mu[:, i0] @ wn      # align free intercept
ax[0,0].errorbar(t, mu[:, i0], yerr=2*se, fmt="o", ms=4, color="C0", label="MC bin means ±2SE (960k)")
ax[0,0].plot(tf, fine, "C3", lw=1.5, label="surrogate-exact prediction")
Xf = np.stack([np.ones_like(ab), ab], 1)
ax[0,0].plot(t, Xf @ co_p[:, i0], "k--", lw=1, label="affine fit")
ax[0,0].set_title(f"n=2048: E[Z'_i | a'] — WORST-case coord (knee amp {abs(lamTn[i0]):.3f}, ~3.5x rms;\n"
                  f"here the linear part happens weak, so the knee dominates visibly)")
ax[0,0].set_xlabel(r"$(a'-m_{A'})/\sigma_{A'}$"); ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3)

ax[0,1].errorbar(t, Rp[:, i0], yerr=2*se, fmt="o", ms=4, color="C0", label="MC residual (mean − affine)")
ax[0,1].plot(t, R_th[:, i0], "C3", lw=1.5, label="theory: $\\lambda_i \\hat g(a')$")
ax[0,1].axhline(0, color="gray", lw=.5)
ax[0,1].set_title("same coord, affine fit subtracted: the knee appears")
ax[0,1].set_xlabel(r"$(a'-m_{A'})/\sigma_{A'}$"); ax[0,1].legend(fontsize=8); ax[0,1].grid(alpha=.3)

sgn = np.sign(ipw(profE, profT))
ax[1,0].plot(t, sgn*profE/np.sqrt(ipw(profE,profE)), "o-", ms=4, color="C0", label="MC shared profile (SVD across 2047 coords)")
ax[1,0].plot(t, pT, "C3", lw=1.5, label="theory knee profile $\\hat g$")
ax[1,0].set_title(f"shared residual profile: cos(MC, theory) = {cosET:.3f}\nrank-1 share = {r1:.2f}, split-half cos = {cosAB:.3f}")
ax[1,0].set_xlabel(r"$(a'-m_{A'})/\sigma_{A'}$"); ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=.3)

ax[1,1].plot(t, Vq, "o", ms=4, color="C1", label=r"MC Var$(\hat\lambda^\top Z'\,|$ bin$)$")
ax[1,1].plot(tf, th["var_q"] + off, "C3", lw=1.5, label="theory rank-1 cov knee + const")
ax[1,1].plot(t, Vc, "s", ms=3, color="gray", alpha=.6, label="control direction (⊥ λ)")
ax[1,1].set_title("covariance version: variance along the knee direction\nis an S-curve; a random direction is flat")
ax[1,1].set_xlabel(r"$(a'-m_{A'})/\sigma_{A'}$"); ax[1,1].legend(fontsize=8); ax[1,1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIGD, "F5_knee2_n2048.png"), dpi=150)
print("fig saved")

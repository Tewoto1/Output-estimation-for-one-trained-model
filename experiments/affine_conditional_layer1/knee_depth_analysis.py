"""Depth propagation of the knee -- analysis of knee3_mc_* runs + 1-d marginal recursion theory.

Per (n, seed), per pre-act layer l in {2,3}:
  * split-half DEBIASED pooled affine 1-R^2 of the PRE bulk-given-spike means
    (repo convention: empirical_structure.mean_linearity), and the POST version
    on positive bins only (the fold);
  * shared affine-residual profile across coordinates via the CROSS-HALF bin-space
    operator K = sum_i R_A[:,i] R_B[:,i]^T (unbiased top component + rank-1 share);
  * knee amplitude rms(lambda)*sqrt(n).

Theory (self-consistent 1-d spike-marginal recursion, channel scalars MEASURED from
the same run):   p_{l+1} = phi_omega_l * (c_l ReLU + mu_l)_# p_l .
  * marginal overlays + TV distance vs the measured spike histograms;
  * knee = Tweedie:  nonaffine part of E[c ReLU(y)|A=a]  (= omega^2 * nonaffine(score));
    parameter-free SHAPE prediction for the measured profiles (cosine test);
  * fixed point: iterate to depth 7 -> 1-R^2(g_l) and shape converge;
  * damping factor q of a propagated knee through one more channel:
        q = ||(I-Pi_aff) S g_2||_{p3} / ||(I-Pi_aff) g_2||_{p2},  S g(a) = E[g(y)|A'=a].

Run:  python experiments/affine_conditional_layer1/knee_depth_analysis.py
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
CKPT = os.path.join(REPO, "checkpoints", "affine_conditional_layer1")
FIGD = os.path.join(CKPT, "figs")
from Mecha_preds.binned_kprop.empirical_structure import mean_linearity
from Mecha_preds._utils import _phi

# --------------------------------------------------------------------------- #
def wcos(x, y, p):
    p = p / p.sum()
    ip = lambda u, v: float((p * u * v).sum())
    return ip(x, y) / np.sqrt(ip(x, x) * ip(y, y))

def aff_residual(a, Y, p):
    """Remove weighted affine part of each column of Y (S, d)."""
    p = p / p.sum()
    X = np.stack([np.ones_like(a), a], 1)
    Xw = X * p[:, None]
    co = np.linalg.solve(X.T @ Xw, Xw.T @ Y)
    return Y - X @ co

def layer_stats(d, l, n):
    """PRE/POST debiased fits + cross-half shared profile for 0-indexed layer l."""
    cnt = d[f"cnt{l}"]; N = cnt.sum()
    p = cnt.sum(0) / N
    abar = d[f"sa{l}"].sum(0) / cnt.sum(0)
    muA = d[f"SZpre{l}"][0] / cnt[0][:, None]
    muB = d[f"SZpre{l}"][1] / cnt[1][:, None]
    ml = mean_linearity(abar, 0.5 * (muA + muB), p=p, mu_A=muA, mu_B=muB)
    # POST on positive bins (the fold: atom + negative side excluded)
    edges = d[f"edges{l}"]
    pos = edges[:-1] >= 0.0
    if pos.sum() >= 4:
        hA = d[f"SZpost{l}"][0] / cnt[0][:, None]
        hB = d[f"SZpost{l}"][1] / cnt[1][:, None]
        mlp = mean_linearity(abar[pos], 0.5 * (hA + hB)[pos], p=p[pos],
                             mu_A=hA[pos], mu_B=hB[pos])
        post = 1 - mlp["R2"]
        post_massfrac = float(p[pos].sum())
    else:
        post, post_massfrac = np.nan, 0.0
    # cross-half shared residual profile (bin space, weighted)
    RA = aff_residual(abar, muA, p)       # (bins, d) minus per-coord affine
    RB = aff_residual(abar, muB, p)
    sw = np.sqrt(p / p.sum())
    RAw = RA * sw[:, None]; RBw = RB * sw[:, None]
    K = RAw @ RBw.T                        # (bins, bins) cross-half operator
    Ksym = 0.5 * (K + K.T)
    ev, V = np.linalg.eigh(Ksym)
    prof_w = V[:, -1]
    prof = prof_w / sw
    prof2 = V[:, -2] / sw
    r1share = float(ev[-1] / np.trace(Ksym))
    r2share = float(ev[-2] / np.trace(Ksym))
    lamA = RAw.T @ prof_w; lamB = RBw.T @ prof_w
    lam2 = float(lamA @ lamB)              # cross-half unbiased |lambda|^2
    lam_rms_sqrtn = np.sqrt(max(lam2, 0) / (n - 1)) * np.sqrt(n)
    # sign convention: profile increasing at right end
    if prof[-1] < prof[0]:
        prof = -prof
    return dict(pre=1 - ml["R2"], pre_raw=1 - ml["R2_raw"], post=post,
                post_massfrac=post_massfrac, abar=abar, p=p, prof=prof, prof2=prof2,
                r1share=r1share, r2share=r2share, lam_rms_sqrtn=lam_rms_sqrtn)

# --------------------------------------------------------------------------- #
# 1-d marginal recursion + knee + damping, from measured channel scalars
# --------------------------------------------------------------------------- #
def channel_params(d, l):
    """Channel into 0-indexed layer l (=1,2): A^l = c ReLU(A^{l-1}) + xi."""
    c = float(d["cs"][l - 1])
    N = d[f"cnt{l}"].sum()
    S = d[f"chan{l-1}"]
    mxi, mxi2, mAp, mAp2, mxAp = S / N
    var_xi = mxi2 - mxi ** 2
    var_Ap = mAp2 - mAp ** 2
    cov = mxAp - mxi * mAp
    om2 = var_xi - cov ** 2 / var_Ap          # fresh-noise part (regress out A+)
    return dict(c=c, mu=mxi, om2=om2, corr=cov / np.sqrt(var_xi * var_Ap))

def recursion(tau1, chans, n_grid=1401, span=9.0):
    """Iterate p_{l+1} = phi_om * (c ReLU + mu)_# p_l starting from N(0, tau1^2).
    chans: list of dicts (c, mu, om2). Returns per-layer dict with grid a, p, knee g,
    its mass-weighted affine 1-R^2, and the transition kernels for damping."""
    lim = span * max(tau1, 1.0) + sum(abs(ch["mu"]) + 3 * np.sqrt(ch["om2"]) for ch in chans)
    a = np.linspace(-lim, lim, n_grid)
    p = _phi(a / tau1) / tau1
    out = [dict(a=a, p=p, g=None, one_m_r2=0.0)]
    for ch in chans:
        c, mu, om = ch["c"], ch["mu"], np.sqrt(ch["om2"])
        Kn = _phi((a[None, :] - (c * np.maximum(a, 0) + mu)[:, None]) / om) / om  # (y, a')
        joint = p[:, None] * Kn
        pz = np.trapezoid(joint, a, axis=0)
        g = np.trapezoid(joint * (c * np.maximum(a, 0))[:, None], a, axis=0) / np.maximum(pz, 1e-300)
        pn = pz / np.trapezoid(pz, a)
        gr = aff_residual(a, g[:, None], np.maximum(pn, 0))[:, 0]
        tot = (pn / pn.sum() * (g - (pn / pn.sum() * g).sum()) ** 2).sum()
        res = (pn / pn.sum() * gr ** 2).sum()
        out.append(dict(a=a, p=pn, g=g, g_res=gr, one_m_r2=float(res / tot),
                        kernel=(joint, pz)))
        p = pn
    return out

def damping_q(rec, l_from, return_profile=False):
    """q for propagating layer-l_from knee through the next channel:
    S g(a) = E[g(y) | A_{l+1} = a], then remove the affine part."""
    g_res = rec[l_from]["g_res"]; p_from = rec[l_from]["p"]
    joint, pz = rec[l_from + 1]["kernel"]
    a = rec[l_from]["a"]
    Sg = np.trapezoid(joint * g_res[:, None], a, axis=0) / np.maximum(pz, 1e-300)
    p_to = rec[l_from + 1]["p"]
    Sg_res = aff_residual(a, Sg[:, None], p_to)[:, 0]
    nrm = lambda x, p: np.sqrt(float((p / p.sum() * x ** 2).sum()))
    q = nrm(Sg_res, p_to) / nrm(g_res, p_from)
    return (q, Sg_res) if return_profile else q

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(CKPT, "knee3_mc_n*_N1000000_B24.npz")))
    print("runs:", [os.path.basename(f) for f in files])
    results = {}
    for f in files:
        d = dict(np.load(f))
        n = int(os.path.basename(f).split("_n")[1].split("_")[0])
        seed = int(os.path.basename(f).split("seed")[1].split("_")[0])
        # tau1 exact from the net
        from Mecha_preds.binned_kprop.empirical_structure import build_spiked_net
        M1 = build_spiked_net(n, 3, seed=seed)[0][0]
        tau1 = float(np.linalg.norm(M1[0]))
        chans = [channel_params(d, 1), channel_params(d, 2)]
        rec = recursion(tau1, chans)
        row = dict(n=n, seed=seed, tau1=tau1, chans=chans, rec=rec)
        for l in (1, 2):
            st = layer_stats(d, l, n)
            # score/knee shape prediction at the bin centers (eigenvector sign arbitrary -> |cos|)
            gpred = np.interp(st["abar"], rec[l]["a"], rec[l]["g_res"])
            st["cos_theory"] = abs(wcos(st["prof"], gpred, st["p"]))
            if l == 2:   # is the 2nd component the DAMPED PROPAGATED layer-2 knee?
                _, Sg = damping_q(rec, 1, return_profile=True)
                st["cos2_propagated"] = abs(wcos(st["prof2"],
                                                 np.interp(st["abar"], rec[2]["a"], Sg),
                                                 st["p"]))
            # marginal TV vs measured histogram
            hg, hh = d[f"hgrid{l}"], d[f"hist{l}"]
            ctr = 0.5 * (hg[1:] + hg[:-1])
            ph = hh / hh.sum() / np.diff(hg)
            pth = np.interp(ctr, rec[l]["a"], rec[l]["p"])
            st["tv"] = 0.5 * float(np.sum(np.abs(ph - pth) * np.diff(hg)))
            row[f"L{l+1}"] = st
        row["q23"] = damping_q(rec, 1)
        results[(n, seed)] = row
        print(f"n={n} seed={seed}: c={chans[0]['c']:.3f}/{chans[1]['c']:.3f} "
              f"mu={chans[0]['mu']:+.3f}/{chans[1]['mu']:+.3f} "
              f"om2={chans[0]['om2']:.3f}/{chans[1]['om2']:.3f} "
              f"corr(xi,A+)={chans[0]['corr']:+.3f}/{chans[1]['corr']:+.3f}")
        for l in (1, 2):
            st = row[f"L{l+1}"]
            extra = (f"  rank2 = {st['r2share']:.3f} cos(prof2, damped-L2-knee) = "
                     f"{st['cos2_propagated']:.3f}") if "cos2_propagated" in st else ""
            print(f"   L{l+1}: PRE 1-R2 = {st['pre']:.4f} (raw {st['pre_raw']:.4f})  "
                  f"POST(pos bins, {st['post_massfrac']:.0%} mass) = {st['post']:.4f}  "
                  f"rank1 = {st['r1share']:.3f}  cos(prof, theory-knee) = {st['cos_theory']:.4f}  "
                  f"rms(lam)*sqrt(n) = {st['lam_rms_sqrtn']:.3f}  TV(marg) = {st['tv']:.3f}{extra}")
        print(f"   theory knee 1-R2: L2 {rec[1]['one_m_r2']:.4f}  L3 {rec[2]['one_m_r2']:.4f}   "
              f"damping q(2->3) = {row['q23']:.3f}")

    # ---- fixed-point iteration --------------------------------------------
    # Generic variance-stationary channel (the natural normalization): c=1, mu=0,
    # om2 = the measured layer-2 value. In the actual UNNORMALIZED net, om2 contracts
    # ~x1/3 per layer and mu = w^T m0 drifts O(1) at random per seed -- that drift,
    # not depth itself, is what moves the per-seed L3 numbers (0.06-0.17).
    (n0, s0) = sorted(results)[-1]
    base = results[(n0, s0)]
    chan_rep = [dict(c=1.0, mu=0.0, om2=base["chans"][0]["om2"])] * 6
    rec_fp = recursion(base["tau1"], chan_rep)
    fp_r2 = [r["one_m_r2"] for r in rec_fp[1:]]
    fp_cos = [wcos(np.interp(rec_fp[i]["a"], rec_fp[i + 1]["a"], rec_fp[i + 1]["g_res"]),
                   rec_fp[i]["g_res"], rec_fp[i]["p"]) for i in range(1, len(rec_fp) - 1)]
    qs = [damping_q(rec_fp, i) for i in range(1, len(rec_fp) - 1)]
    print(f"\nfixed point (channel repeated): knee 1-R2 by layer = "
          + " ".join(f"{x:.4f}" for x in fp_r2))
    print("shape cos(g_l, g_l+1) =", " ".join(f"{x:.4f}" for x in fp_cos))
    print("damping q by layer    =", " ".join(f"{x:.3f}" for x in qs))

    np.savez_compressed(os.path.join(CKPT, "knee3_analysis.npz"),
                        summary=str({k: {kk: (vv if not isinstance(vv, dict) else "...")
                                         for kk, vv in v.items() if kk.startswith("L")}
                                     for k, v in results.items()}),
                        fp_r2=fp_r2, fp_cos=fp_cos, qs=qs)

    # ================= figures =================
    big = results[(n0, s0)]
    d = dict(np.load(os.path.join(CKPT, f"knee3_mc_n{n0}_seed{s0}_N1000000_B24.npz")))
    # F6 marginals
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, l in enumerate((0, 1, 2)):
        hg, hh = d[f"hgrid{l}"], d[f"hist{l}"]
        ctr = 0.5 * (hg[1:] + hg[:-1])
        ax[i].plot(ctr, hh / hh.sum() / np.diff(hg), "C0", lw=1.5,
                   label="MC marginal" if i == 0 else None)
        ax[i].plot(big["rec"][l]["a"], big["rec"][l]["p"], "C3--", lw=1.2,
                   label="1-d recursion" if i == 0 else None)
        ax[i].set_xlim(ctr[0], ctr[-1])
        ax[i].set_title(f"spike marginal, pre-act layer {l+1}"
                        + ("" if l == 0 else f"   TV={big[f'L{l+1}']['tv']:.3f}" if l else ""))
        ax[i].set_xlabel(f"$a_{l+1}$")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"n={n0} seed={s0}: the spike marginal converges to a smeared "
                 "(atom+branch)*Gaussian fixed shape", y=1.02, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "F6_marginals_depth.png"), dpi=150,
                                    bbox_inches="tight")

    # F7 profiles vs theory knee
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, l in enumerate((1, 2)):
        st = big[f"L{l+1}"]
        gpred = np.interp(st["abar"], big["rec"][l]["a"], big["rec"][l]["g_res"])
        nrm = lambda x: x / np.sqrt((st["p"] / st["p"].sum() * x ** 2).sum())
        sgn = np.sign((st["p"] * nrm(gpred) * nrm(st["prof"])).sum())
        ax[i].plot(st["abar"], sgn * nrm(st["prof"]), "o-", color="C0", ms=4,
                   label="measured shared residual profile")
        ax[i].plot(st["abar"], nrm(gpred), "C3--", lw=1.5,
                   label="recursion knee (zero-param shape)")
        ax[i].set_title(f"layer {l+1} PRE: cos = {st['cos_theory']:.4f}, "
                        f"rank-1 share = {st['r1share']:.3f}")
        ax[i].set_xlabel(f"$a_{l+1}$"); ax[i].legend(fontsize=8); ax[i].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "F7_profiles_depth.png"), dpi=150)

    # F8 degradation + fixed point
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (nn, ss), row in sorted(results.items()):
        pre = [row["L2"]["pre"], row["L3"]["pre"]]
        post = [row["L2"]["post"], row["L3"]["post"]]
        ax[0].plot([2, 3], pre, "o-", label=f"PRE n={nn} s{ss}")
        ax[0].plot([2, 3], post, "s--", alpha=.7, label=f"POST n={nn} s{ss}")
    ax[0].set_xticks([2, 3]); ax[0].set_xlabel("pre-act layer")
    ax[0].set_ylabel("pooled debiased 1-R2"); ax[0].set_yscale("log")
    ax[0].set_title("affine quality by layer: PRE saturates O(1),\nPOST stays ~an order better")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].plot(range(2, 2 + len(fp_r2)), fp_r2, "o-", color="C3",
               label=r"recursion knee $1-R^2(g_\ell)$")
    ax[1].set_xlabel("layer"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("1-d recursion iterated: the knee's non-affineness\nconverges to a fixed point"
                    f" (shape cos {min(fp_cos):.3f}..{max(fp_cos):.3f}, q≈{np.mean(qs):.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "F8_depth_degradation.png"), dpi=150)
    print("figures written to", FIGD)

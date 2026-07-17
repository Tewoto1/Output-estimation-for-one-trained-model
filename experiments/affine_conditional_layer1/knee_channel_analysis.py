"""Channel-structure + recursion-accuracy analysis of chanmc_* runs.

(1) The dropped structure, measured (split-half debiased powers, expected ~ 1/n):
      E[xi|a] deviation from const   (kappa*a tilt + old-knee leak, amplitude n^{-1/2})
      Var[xi|a]/om2 - 1              (tilt, n^{-1/2})
      skew(xi|a)                     (n^{-1/2})
(2) Recursion accuracy: cross-half-debiased L2 density error
      D^2 = int (p_true - p_rec)^2 da     (unbiased via half-histogram cross products)
    at layers 2 and 3, for the 3-scalar recursion (c, mu, om2) and the 4-scalar
    variant (+ measured kappa tilt).  Expected D^2 ~ n^{-1}.
(3) Layer-2 CLOSED FORM check: with s ~ N(0, tau^2), A2 = c ReLU(s) + kappa s + mu + xi:
      p2(a) = phi_{s-}(u) Phi(-kappa tau u / (omega s-))            [atom side]
            + phi_{s+}(u) Phi( b tau u / (omega s+) ),              [branch side]
      u = a - mu,  b = c + kappa,  s-^2 = kappa^2 tau^2 + om2,  s+^2 = b^2 tau^2 + om2.
    Verified against quadrature below; kappa=0 gives  1/2-mass smeared atom
    phi_om(u)*Phi(0) + skew-normal branch.

Run:  python experiments/affine_conditional_layer1/knee_channel_analysis.py
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
CKPT = os.path.join(REPO, "checkpoints", "affine_conditional_layer1")
FIGD = os.path.join(CKPT, "figs")
from Mecha_preds._utils import _phi, _Phi
from Mecha_preds.binned_kprop.empirical_structure import build_spiked_net

def wmean(x, p): return float((p * x).sum() / p.sum())

def p2_closed_form(a, tau, c, kap, mu, om):
    u = a - mu
    sm = np.sqrt(kap ** 2 * tau ** 2 + om ** 2)
    b = c + kap
    sp = np.sqrt(b ** 2 * tau ** 2 + om ** 2)
    return (_phi(u / sm) / sm * _Phi(-kap * tau * u / (om * sm))
            + _phi(u / sp) / sp * _Phi(b * tau * u / (om * sp)))

def recursion_chain(tau1, chans, n_grid=1601, span=8.0):
    lim = span * max(tau1, 1.5) + sum(abs(c["mu"]) + 3 * np.sqrt(c["om2"]) for c in chans)
    y = np.linspace(-lim, lim, n_grid)
    p = _phi(y / tau1) / tau1
    ps = [p]
    for ch in chans:
        om = np.sqrt(ch["om2"])
        mch = ch["c"] * np.maximum(y, 0) + ch.get("kap", 0.0) * (y - ch.get("abar", 0.0)) + ch["mu"]
        Kn = _phi((y[None, :] - mch[:, None]) / om) / om
        pz = np.trapezoid(p[:, None] * Kn, y, axis=0)
        p = pz / np.trapezoid(pz, y)
        ps.append(p)
    return y, ps

def channel_struct(d, l):
    """Split-half debiased deviation powers of E[xi|a], Var[xi|a], skew per bin."""
    cnt = d[f"cnt{l}"]; p = cnt.sum(0); p = p / p.sum()
    abar = d[f"sa{l}"].sum(0) / cnt.sum(0)
    m = d[f"sxi{l}"] / cnt
    v = d[f"sxi2{l}"] / cnt - m ** 2
    m3 = d[f"sxi3{l}"] / cnt - 3 * m * v - m ** 3
    skew = m3 / np.maximum(v, 1e-12) ** 1.5
    dm = m - np.array([[wmean(m[h], p)] for h in (0, 1)])
    P_mean = float((p * dm[0] * dm[1]).sum())
    om2 = wmean(0.5 * (v[0] + v[1]), p)
    dv = v - np.array([[wmean(v[h], p)] for h in (0, 1)])
    P_var = float((p * dv[0] * dv[1]).sum())
    # kappa = weighted LS slope of the mean curve
    aw = abar - wmean(abar, p)
    kap = float((p * aw * 0.5 * (dm[0] + dm[1])).sum() / (p * aw * aw).sum())
    sk = wmean(0.5 * (skew[0] + skew[1]), p)
    return dict(abar=abar, p=p, mcurve=0.5 * (m[0] + m[1]), vcurve=0.5 * (v[0] + v[1]),
                P_mean=P_mean, P_var=P_var, om2=om2, kap=kap, skew=sk,
                mu=wmean(0.5 * (m[0] + m[1]), p), var_a=float((p * aw * aw).sum()))

def density_D2(d, l, y, p_rec):
    g = d[f"hgrid{l}"]; ctr = 0.5 * (g[1:] + g[:-1]); dx = np.diff(g)
    pr = np.interp(ctr, y, p_rec)
    e, sig2 = [], []
    for h in (0, 1):
        Nh = d[f"hist{l}"][h].sum()
        ph = d[f"hist{l}"][h] / Nh / dx
        e.append(ph - pr)
        sig2.append(ph / (Nh * dx))          # Poisson density variance per bin
    D2 = float((dx * e[0] * e[1]).sum())
    ebar = 0.5 * (e[0] + e[1])
    varD2 = float((dx ** 2 * (sig2[0] * ebar ** 2 + sig2[1] * ebar ** 2
                              + sig2[0] * sig2[1])).sum())
    tv = 0.5 * float((np.abs(ebar) * dx).sum())
    return D2, np.sqrt(varD2), tv

if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(os.path.join(CKPT, "chanmc_n*_N1000000.npz"))):
        d = dict(np.load(f))
        n = int(re.search(r"_n(\d+)_", f).group(1))
        seed = int(re.search(r"seed(\d+)", f).group(1))
        M1 = build_spiked_net(n, 3, seed=seed)[0][0]
        tau1 = float(np.linalg.norm(M1[0]))
        ch = [channel_struct(d, l) for l in (0, 1)]
        cs = d["cs"]
        mk = lambda l, kap: dict(c=float(cs[l]), mu=ch[l]["mu"], om2=ch[l]["om2"],
                                 kap=(ch[l]["kap"] if kap else 0.0),
                                 abar=wmean(ch[l]["abar"], ch[l]["p"]))
        y, ps3 = recursion_chain(tau1, [mk(0, False), mk(1, False)])
        _, ps4 = recursion_chain(tau1, [mk(0, True), mk(1, True)])
        # closed form vs quadrature check at layer 2 (kappa version)
        cf = p2_closed_form(y, tau1, float(cs[0]), ch[0]["kap"], ch[0]["mu"] -
                            ch[0]["kap"] * wmean(ch[0]["abar"], ch[0]["p"]),
                            np.sqrt(ch[0]["om2"]))
        cf_err = float(np.trapezoid(np.abs(cf - ps4[1]), y)) / 2
        row = dict(n=n, seed=seed)
        for l, name in ((1, "L2"), (2, "L3")):
            D3, se3, tv3 = density_D2(d, l, y, ps3[l])
            D4, se4, tv4 = density_D2(d, l, y, ps4[l])
            row[f"D2_{name}_v3"], row[f"D2_{name}_v4"] = D3, D4
            row[f"se_{name}"] = max(se3, se4)
            row[f"tv_{name}"] = tv3
        for l, name in ((0, "12"), (1, "23")):
            row[f"Pmean_{name}"] = ch[l]["P_mean"]
            row[f"Pvar_rel_{name}"] = ch[l]["P_var"] / ch[l]["om2"] ** 2
            row[f"kap_{name}"] = ch[l]["kap"]
            row[f"skew_{name}"] = ch[l]["skew"]
        row["cf_err"] = cf_err
        row["ch"] = ch; row["y"] = y; row["ps3"] = ps3
        rows.append(row)
        print(f"n={n:5d} s{seed}: kap12={ch[0]['kap']:+.4f} (x sqrt n = {ch[0]['kap']*np.sqrt(n):+.2f})  "
              f"skew12={ch[0]['skew']:+.4f}  sqrt(Pmean)12={np.sqrt(max(ch[0]['P_mean'],0)):.4f}  "
              f"relVar-dev12={np.sqrt(max(row['Pvar_rel_12'],0)):.4f}")
        print(f"          D2 L2: v3 {row['D2_L2_v3']:.2e}  v4 {row['D2_L2_v4']:.2e} (SE {row['se_L2']:.1e})   "
              f"D2 L3: v3 {row['D2_L3_v3']:.2e}  v4 {row['D2_L3_v4']:.2e}   TV L2 {row['tv_L2']:.4f}  "
              f"closed-form|quad L2 err {cf_err:.1e}")

    # ---------------- scaling fits ----------------
    ns = sorted({r["n"] for r in rows})
    def agg(key):
        return np.array([np.mean([r[key] for r in rows if r["n"] == n_]) for n_ in ns])
    print("\nscaling exponents (log-log fit over widths):")
    for key in ["Pmean_12", "Pvar_rel_12", "D2_L2_v3", "D2_L2_v4", "D2_L3_v3", "D2_L3_v4"]:
        yv = agg(key)
        if (yv <= 0).any():
            print(f"  {key}: has non-positive entries (noise floor) -> "
                  + " ".join(f"{v:.1e}" for v in yv))
            continue
        sl = np.polyfit(np.log(ns), np.log(yv), 1)[0]
        print(f"  {key}: {sl:+.2f}   (values: " + " ".join(f"{v:.1e}" for v in yv) + ")")
    k2 = np.array([np.mean([abs(r["kap_12"]) * np.sqrt(r["n"]) for r in rows if r["n"] == n_]) for n_ in ns])
    sk = np.array([np.mean([abs(r["skew_12"]) * np.sqrt(r["n"]) for r in rows if r["n"] == n_]) for n_ in ns])
    print("  |kappa|*sqrt(n) by n:", " ".join(f"{v:.2f}" for v in k2))
    print("  |skew|*sqrt(n)  by n:", " ".join(f"{v:.2f}" for v in sk))

    # ---------------- figures ----------------
    # F9: the dropped channel structure, sqrt(n)-collapsed
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in rows:
        if r["seed"] != 0 or r["n"] not in (128, 512, 2048): continue
        c0 = r["ch"][0]; n = r["n"]
        t = (c0["abar"] - wmean(c0["abar"], c0["p"])) / np.sqrt(c0["var_a"])
        ax[0].plot(t, np.sqrt(n) * (c0["mcurve"] - wmean(c0["mcurve"], c0["p"])), "o-",
                   ms=3, label=f"n={n}")
        ax[1].plot(t, np.sqrt(n) * (c0["vcurve"] / c0["om2"] - 1), "o-", ms=3, label=f"n={n}")
    ax[0].set_title(r"$\sqrt{n}\,\big(E[\xi|a]-\mu\big)$: the dropped mean structure"
                    "\n(tilt $\\kappa a$ + old-knee leak, both $O(n^{-1/2})$ -> collapse)")
    ax[1].set_title(r"$\sqrt{n}\,\big(\mathrm{Var}(\xi|a)/\omega^2-1\big)$: dropped variance tilt")
    for a_ in ax:
        a_.set_xlabel(r"standardized source spike $a$"); a_.legend(fontsize=8); a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "F9_channel_structure.png"), dpi=150)

    # F10: recursion density error scaling
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, lab, c, mk_ in [("D2_L2_v3", "L2, 3-scalar (c, mu, om2)", "C0", "o"),
                             ("D2_L2_v4", "L2, + kappa tilt", "C0", "s"),
                             ("D2_L3_v3", "L3, 3-scalar", "C3", "o"),
                             ("D2_L3_v4", "L3, + kappa tilt", "C3", "s")]:
        yv = agg(key)
        ls = "-" if key.endswith("v3") else "--"
        ax[0].loglog(ns, np.abs(yv), mk_ + ls, color=c, label=lab, alpha=.9)
    ax[0].loglog(ns, 0.2 / np.array(ns, float), ":", color="gray", label=r"$\propto 1/n$")
    ax[0].set_xlabel("width n"); ax[0].set_ylabel(r"debiased $D^2=\int(p-\hat p_{\rm rec})^2$")
    ax[0].set_title("recursion marginal error: $D^2 \\propto 1/n$\n(amplitude $O(n^{-1/2})$)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which="both")
    for key, lab, c in [("Pmean_12", r"$\|E[\xi|a]-\mu\|_p^2$ (mean structure)", "C0"),
                        ("Pvar_rel_12", r"$\|\mathrm{Var}(\xi|a)/\omega^2-1\|_p^2$", "C1")]:
        ax[1].loglog(ns, np.abs(agg(key)), "o-", color=c, label=lab)
    ax[1].loglog(ns, np.array([np.mean([r["skew_12"] ** 2 for r in rows if r["n"] == n_])
                               for n_ in ns]), "o-", color="C2", label=r"skew$(\xi)^2$")
    ax[1].loglog(ns, 1.0 / np.array(ns, float), ":", color="gray", label=r"$\propto 1/n$")
    ax[1].set_xlabel("width n"); ax[1].set_title("dropped-term powers, channel 1->2: all $\\propto 1/n$")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "F10_recursion_accuracy.png"), dpi=150)

    # F11: layer-2 closed form decomposition (n=1024 seed 0)
    r = [x for x in rows if x["n"] == 1024 and x["seed"] == 0][0]
    d = dict(np.load(os.path.join(CKPT, "chanmc_n1024_seed0_N1000000.npz")))
    g = d["hgrid1"]; ctr = 0.5 * (g[1:] + g[:-1]); dx = np.diff(g)
    ph = d["hist1"].sum(0) / d["hist1"].sum() / dx
    c0 = r["ch"][0]
    M1 = build_spiked_net(1024, 3, seed=0)[0][0]; tau = float(np.linalg.norm(M1[0]))
    cc, mu, om = float(d["cs"][0]), c0["mu"], np.sqrt(c0["om2"])
    u = ctr - mu
    atom = _phi(u / om) / om * 0.5
    sp = np.sqrt(cc ** 2 * tau ** 2 + om ** 2)
    branch = _phi(u / sp) / sp * _Phi(cc * tau * u / (om * sp))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ctr, ph, "C0", lw=1.6, label="MC marginal of $A_2$ (1M)")
    ax.plot(ctr, atom + branch, "k--", lw=1.2, label="closed form (3 scalars)")
    ax.fill_between(ctr, 0, atom, color="C3", alpha=.25,
                    label=r"smeared atom $\frac12\,\varphi_\omega(a-\mu)$")
    ax.fill_between(ctr, 0, branch, color="C2", alpha=.2,
                    label=r"branch $\varphi_{s_+}(u)\,\Phi\!\big(\frac{c\tau u}{\omega s_+}\big)$ (skew-normal)")
    ax.set_title(f"layer-2 spike marginal, n=1024: closed form\n"
                 f"$\\tau$={tau:.2f}, c={cc:.2f}, $\\mu$={mu:+.2f}, $\\omega$={om:.2f}"
                 f"   (TV to MC = {r['tv_L2']:.3f})")
    ax.set_xlabel("$a_2$"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGD, "F11_layer2_closed_form.png"), dpi=150)
    print("figures written to", FIGD)

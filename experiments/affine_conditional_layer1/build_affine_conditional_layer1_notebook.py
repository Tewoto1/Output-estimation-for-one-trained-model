"""Generates affine_conditional_layer1_colab.ipynb (valid nbformat-4 JSON).

WHY is the bulk-given-spike law so AFFINE after the ReLU? The affine_r2 notebook
(experiments/analytic_kprop) measured, per layer on the deep surrogate, that the
POST-activation conditional mean/cov are increasingly-in-width well fit by an
affine family in the spike coordinate, while deep PRE-activation is not. This
notebook isolates the MECHANISM in the smallest possible model:

    one layer,  M = W + e1 e1^T  (W ~ N(0,1/n), no training),  X ~ N(0, I_n),
    h = ReLU(M X),  condition on the spike coordinate  s = (M X)_0.

Key trick -- NO Monte-Carlo, NO binning: (s, Z_bulk) is jointly Gaussian, so
    Z_bulk | s  ~  N(beta s, C)   EXACTLY,   beta_i = (MM^T)_{i0}/tau^2,
with C constant in s. The post-ReLU conditional mean / variance / off-diagonal
covariance at each s are then EXACT rectified-Gaussian integrals (the repo's
verified kernels in Mecha_preds/_utils.py). We fit affine-in-s per coordinate /
pair, weighted by the Gaussian mass of s, and get machine-precision R^2 -- the
1-R^2 ~ 1e-4 residuals at n=4096 are invisible to binned MC (600k samples reads
7.5e-3 where the truth is 3.5e-3; that inflation is exactly what the split-half
debiasing in empirical_structure exists for).

RESULT (the "why", verified to ~1% by closed form):
  * pre-activation: mean exactly linear in s, cov exactly constant (Gaussianity).
    ALL curvature must come from the ReLU step.
  * ReLU enters only through analytic scalar maps (rectified-Gaussian integrals)
    with smoothing scale sigma_i ~ 1: the kink is convolved away by the O(1)
    conditional bulk noise.
  * the spike moves each bulk coordinate's conditional mean by only
    beta_i s = O(sqrt(1/n)) across the whole observable s-range, so the affine
    model is the first-order Taylor of a smooth map on a shrinking interval:
        1-R^2(mean_i)  ~=  2 phi(0)^2 beta_i^2 tau^2 / sigma_i^2      ~ 1/n
        1-R^2(var_i)   ~=  ((1/2-1/pi)/(sigma_i phi(0)))^2 beta_i^2 tau^2 / 2
    (measured pooled values sit on these to <1% at n=4096; per-coordinate
    scatter collapses on y=x over 9 decades).
  * conditional variance is NOT constant: it tilts with slope
    phi(0) sigma_i beta_i = 2 phi(0) sigma_i x (mean slope) ~ 0.80 sigma_i --
    same O(1/sqrt(n)) order as the mean's tilt. A constant-Sigma model errs at
    O(1/sqrt(n)); the affine family captures everything down to O(1/n).
  * NOT a near-0 accident: the residual is the quadratic Taylor parabola, worst
    at edge cells (x3.3 the pooled value; central |s|<1.5tau is x3.2 better)
    but the SAME 1/n law everywhere observable. Real failure needs
    |beta_i s| ~ sigma_i, i.e. a >~ sqrt(2n)-sigma spike excursion (~7 tau for
    the most-coupled coordinate at n=512, ~22 tau for typical ones).
  * fitted slope direction == Phi(0) beta EXACTLY (cos = 1 - 1e-16); beta is
    column-leak W[1:,0] PLUS row-overlap W_bulk w_0, equal-variance halves, so
    cos(slope, naked column) ~= 1/sqrt(2) ~ 0.71 -- refines the earlier
    "aligns with coupling column" reading in binned_structure_test.

Sections:
  §1 config + cache   §2 exact kernels + selftest   §3 exact conditional analysis
  §4 width sweep + table + scaling exponents        §5 anatomy figure
  §6 theory-collapse figure                          §7 MC validation
  §8 conclusions + what this does NOT explain (deeper layers)

CACHE: per-(width,seed) exact-fit rows -> checkpoints/affine_conditional_layer1/
aff_l1_*.npz; MC validation -> mc_l1_*.npz; figures -> figs/. Nothing recomputes
on re-run.

Run:  python "experiments/affine_conditional_layer1/build_affine_conditional_layer1_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# Why is the bulk-given-spike law **affine** *after* the ReLU? (one layer, exact)

**Question** (from the affine-$R^2$ measurements in `experiments/analytic_kprop`): pretending the
conditional bulk mean and covariance vary **linearly along the spike coordinate $s$** is an
increasingly accurate assumption *post-activation*. Why?

**Smallest model that shows it**: one layer, $M = W + e_1e_1^\top$ ($W \sim N(0,1/n)$, no training),
$X \sim N(0, I_n)$, $h = \mathrm{ReLU}(MX)$, conditioned on the spike coordinate $s = (MX)_0$.
Pre-activation is *jointly Gaussian*, so given $s$ the bulk is **exactly**

$$Z_{\mathrm{bulk}} \mid s \;\sim\; N(\beta s,\; C), \qquad
\beta_i = (MM^\top)_{i0}/\tau^2, \quad \tau^2 = (MM^\top)_{00} \approx 2,$$

with $C$ **constant in $s$** — the pre-activation conditional law is trivially affine (don't track it);
all curvature must be created by the ReLU. The post-ReLU conditional stats at each $s$ are **exact
rectified-Gaussian integrals** (repo kernels, no MC, no binning, no noise floor), so we can measure
$1-R^2 \sim 10^{-4}$ residuals that binned MC cannot see (600k samples read $7.5\times10^{-3}$ where
the truth is $3.5\times10^{-3}$).

**Headline** (widths 64→4096, 5 seeds, mass-weighted pooled $R^2$ over all bulk coords / 400 pairs):

| curve | pooled $1-R^2$ @ n=512 | @ n=4096 | width exponent | Taylor prediction |
|---|---|---|---|---|
| cond. mean $E[h_i\|s]$ | $1.87\times10^{-3}$ | $2.38\times10^{-4}$ | $-0.97$ | $2\phi(0)^2\beta_i^2\tau^2/\sigma_i^2$ (matches to 0.8%) |
| cond. variance $\mathrm{Var}(h_i\|s)$ | $6.05\times10^{-4}$ | $7.74\times10^{-5}$ | $-0.96$ | $\big(\tfrac{1/2-1/\pi}{\sigma_i\phi(0)}\big)^2\beta_i^2\tau^2/2$ (3.07× more linear than mean) |
| cond. off-diag $\mathrm{Cov}(h_i,h_j\|s)$ | $1.07\times10^{-3}$ | $1.35\times10^{-4}$ | $-1.00$ | same $\propto\beta^2\tau^2 \sim 1/n$ mechanism |

**The mechanism in one sentence**: the ReLU's conditional-moment maps are *analytic* with $O(1)$
smoothing scale $\sigma_i$ (the kink is convolved away by the bulk's own conditional noise), and the
spike can only push each bulk coordinate $|\beta_i s| = O(\sqrt{1/n})$ deep into that smooth map —
so affine-in-$s$ is the tangent line of a smooth function on a shrinking interval, with quadratic
(Taylor) residual $\Rightarrow 1-R^2 \propto \beta_i^2 \propto 1/n$.""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 Config + cache

Everything cached per (width, seed) in `checkpoints/affine_conditional_layer1/` — re-runs and
seed/width extensions are incremental.""")
code(r"""
import os, json, time, collections
import numpy as np
import matplotlib.pyplot as plt

try:
    import experiments as E
    QUICK = E.QUICK
except Exception:            # torch-free environment: default to full CPU sweep (it is cheap)
    QUICK = False

WIDTHS   = [64, 128, 256, 512] if QUICK else [64, 128, 256, 512, 1024, 2048, 4096]
SEEDS    = [0, 1, 2, 3, 4]
THETA    = 1.0        # M = W + THETA * e1 e1^T
S_GRID   = 41         # conditioning-grid points in s
S_MAX    = 4.0        # grid spans |s| <= S_MAX * tau  (99.994% of the s-mass)
N_PAIRS  = 400        # sampled off-diagonal (i,j) pairs per net
RUN_MC   = not QUICK  # MC cross-check cell (n=512, ~10-30 s)
MC_N, MC_BINS = 600_000, 24

CKPT_DIR = os.path.join("checkpoints", "affine_conditional_layer1")
FIG_DIR  = os.path.join(CKPT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)
PHI0 = 1.0 / np.sqrt(2.0 * np.pi)
print(f"QUICK={QUICK}  widths={WIDTHS}  cache={CKPT_DIR}")
""")

# =============================================================================
md(r"""## §2 Exact kernels (+ selftest)

Univariate rectified-Gaussian moments and the **pairwise** exact bivariate ReLU covariance
(`Mecha_preds._utils.exact_relu_covariance_pairs`, same closed form as the verified
`exact_relu_covariance`, vectorized over sampled pairs × grid without an $n\times n$ matrix).
Weighted-polynomial fit helper; the mean-curve fit is cross-checked against the repo convention
`empirical_structure.mean_linearity` in §3.""")
code(r"""
from Mecha_preds._utils import (relu_moments_1d, exact_relu_covariance,
                                exact_relu_covariance_pairs, _phi)
from Mecha_preds.binned_kprop.empirical_structure import (
    build_spiked_net, spike_coupling_column, mean_linearity)

def _selftest_pairwise(seed=0, d=7):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) / np.sqrt(d)
    Sig = A @ A.T + 0.5 * np.eye(d)
    mu = 0.3 * rng.standard_normal(d)
    _, S_ref = exact_relu_covariance(mu, Sig)
    sig = np.sqrt(np.diag(Sig))
    ii, jj = np.triu_indices(d, k=1)
    got = exact_relu_covariance_pairs(mu[ii], mu[jj], sig[ii], sig[jj],
                                      Sig[ii, jj] / (sig[ii] * sig[jj]))
    err = np.max(np.abs(got - S_ref[ii, jj]))
    assert err < 1e-12, err
    return err

def wls_poly_r2(a, Y, w, deg=1):
    '''Weighted LS fit of each column of Y (S, d) on a polynomial in a (S,).'''
    a = np.asarray(a, float); w = np.asarray(w, float); w = w / w.sum()
    X = np.stack([a ** k for k in range(deg + 1)], axis=1)
    Xw = X * w[:, None]
    coef = np.linalg.solve(X.T @ Xw, Xw.T @ Y)
    resid = Y - X @ coef
    ybar = (w[:, None] * Y).sum(0)
    ss_tot = (w[:, None] * (Y - ybar) ** 2).sum(0)
    ss_res = (w[:, None] * resid ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    return dict(ss_tot=ss_tot, ss_res=ss_res, r2=r2, coef=coef, resid=resid)

def pooled_1mR2(fit):
    return float(fit["ss_res"].sum() / fit["ss_tot"].sum())

print(f"pairwise exact-cov selftest vs exact_relu_covariance: max err {_selftest_pairwise():.2e}")
""")

# =============================================================================
md(r"""## §3 Exact conditional analysis for one net

Per (width, seed): build $M$, get the exact conditional family $(\beta_i s,\ \sigma_i^2,\ C_{ij})$,
push it through the exact ReLU integrals on the $s$-grid, fit affine (and quadratic) per
coordinate/pair with Gaussian-mass weights, and store measured + Taylor-predicted $1-R^2$, slopes,
windowed (central/edge) variants, and example curves. The Taylor constants come from
$f(\mu,\sigma)=\mu\Phi(\mu/\sigma)+\sigma\phi(\mu/\sigma)$ (mean) and
$g=\mathrm{Var}$: $f'(0)=\tfrac12$, $f''(0)=\phi(0)/\sigma$, $g'(0)=\sigma\phi(0)$,
$g''(0)=\tfrac12-\tfrac1\pi$; best-affine residual under Gaussian weight is the centered quadratic
$\tfrac{c_2}{2}\beta_i^2(s^2-\langle s^2\rangle)$, giving

$$1-R^2_{\mathrm{mean},i} \approx 2\phi(0)^2\,\beta_i^2\tau^2/\sigma_i^2, \qquad
1-R^2_{\mathrm{var},i} \approx \Big(\tfrac{1/2-1/\pi}{\sigma_i\phi(0)}\Big)^2\beta_i^2\tau^2/2 .$$""")
code(r"""
def analyze(n, seed, S=S_GRID, smax=S_MAX, n_pairs=N_PAIRS, theta=THETA, cache=True):
    tag = f"aff_l1_n{n}_seed{seed}_S{S}_sm{smax:g}_P{n_pairs}_th{theta:g}"
    path = os.path.join(CKPT_DIR, tag + ".npz")
    if cache and os.path.exists(path):
        return dict(np.load(path, allow_pickle=False))

    M = build_spiked_net(n, depth=1, seed=seed, theta=theta)[0][0]
    G = M @ M.T
    tau2 = G[0, 0]; tau = np.sqrt(tau2)
    beta = G[1:, 0] / tau2                      # conditional-mean slope (pre-act, exact)
    sig2 = np.diag(G)[1:] - beta ** 2 * tau2    # conditional var (constant in s, exact)
    sig = np.sqrt(sig2)
    u = spike_coupling_column(M)                # naked column leak W[1:,0]

    sgrid = tau * np.linspace(-smax, smax, S)
    w = _phi(sgrid / tau); w = w / w.sum()

    MU = beta[:, None] * sgrid[None, :]                            # (d, S)
    mean_c, _, var_c = relu_moments_1d(MU, np.broadcast_to(sig2[:, None], MU.shape))

    rng = np.random.default_rng(seed + 987)
    d = n - 1
    ii = rng.integers(0, d, size=n_pairs); jj = rng.integers(0, d, size=n_pairs)
    ii, jj = ii[ii != jj], jj[ii != jj]
    Cij = G[1 + ii, 1 + jj] - beta[ii] * beta[jj] * tau2
    rho = Cij / (sig[ii] * sig[jj])
    cov_pairs = exact_relu_covariance_pairs(MU[ii], MU[jj], sig[ii, None], sig[jj, None],
                                            rho[:, None])          # (P, S)

    fit_mean = wls_poly_r2(sgrid, mean_c.T, w, deg=1)
    fit_var  = wls_poly_r2(sgrid, var_c.T, w, deg=1)
    fit_off  = wls_poly_r2(sgrid, cov_pairs.T, w, deg=1)
    q_mean   = wls_poly_r2(sgrid, mean_c.T, w, deg=2)
    q_var    = wls_poly_r2(sgrid, var_c.T, w, deg=2)
    cen = np.abs(sgrid) <= 1.5 * tau
    fit_mean_c = wls_poly_r2(sgrid[cen], mean_c.T[cen], w[cen], deg=1)
    fit_var_c  = wls_poly_r2(sgrid[cen], var_c.T[cen], w[cen], deg=1)
    edge = np.abs(sgrid) >= 2.5 * tau                      # eval of the FULL-range fit at edges
    we = w[edge] / w[edge].sum()
    ybar_e = (we[:, None] * mean_c.T[edge]).sum(0)
    ss_tot_e = (we[:, None] * (mean_c.T[edge] - ybar_e) ** 2).sum(0)
    edge_1mr2_mean = float((we[:, None] * fit_mean["resid"][edge] ** 2).sum(0).sum()
                           / ss_tot_e.sum())

    ml = mean_linearity(sgrid, mean_c.T, p=w)              # repo-convention cross-check
    assert abs(ml["R2_raw"] - (1 - pooled_1mR2(fit_mean))) < 1e-9

    pred_mean = 2.0 * PHI0 ** 2 * beta ** 2 * tau2 / sig2
    pred_var = ((0.5 - 1.0 / np.pi) / (sig * PHI0)) ** 2 * beta ** 2 * tau2 / 2.0

    sl_m, sl_v = fit_mean["coef"][1], fit_var["coef"][1]
    nrm = np.linalg.norm
    top = np.argsort(-np.abs(beta))[:6]
    out = dict(
        n=n, seed=seed, tau2=tau2,
        pooled_mean=pooled_1mR2(fit_mean), pooled_var=pooled_1mR2(fit_var),
        pooled_off=pooled_1mR2(fit_off),
        pooled_mean_central=pooled_1mR2(fit_mean_c), pooled_var_central=pooled_1mR2(fit_var_c),
        edge_1mr2_mean=edge_1mr2_mean,
        pred_pooled_mean=float((pred_mean * fit_mean["ss_tot"]).sum() / fit_mean["ss_tot"].sum()),
        pred_pooled_var=float((pred_var * fit_var["ss_tot"]).sum() / fit_var["ss_tot"].sum()),
        cos_slope_beta=float(sl_m @ beta / (nrm(sl_m) * nrm(beta))),
        cos_slope_u=float(sl_m @ u / (nrm(sl_m) * nrm(u))),
        slope_ratio_med=float(np.median(sl_v / sl_m)),
        beta=beta, sig=sig,
        r2_mean_per=fit_mean["r2"], r2_var_per=fit_var["r2"],
        pred_mean_per=pred_mean, pred_var_per=pred_var,
        quad_mean=q_mean["coef"][2], quad_var=q_var["coef"][2],
        slope_mean=sl_m, slope_var=sl_v,
        sgrid=sgrid, wgrid=w, top_idx=top,
        mean_curves_top=mean_c[top], var_curves_top=var_c[top],
        off_curves_top=cov_pairs[np.argsort(-np.abs(rho))[:4]],
        rho_top=rho[np.argsort(-np.abs(rho))[:4]],
    )
    if cache:
        np.savez_compressed(path, **out)
    return out
""")

# =============================================================================
md(r"""## §4 The sweep: pooled $1-R^2$ vs width (+ Taylor prediction, windows, scaling exponents)""")
code(r"""
rows = []
for n in WIDTHS:
    t0 = time.time()
    for sd in SEEDS:
        r = analyze(n, sd)
        rows.append({k: float(r[k]) for k in
                     ["n", "seed", "tau2", "pooled_mean", "pooled_var", "pooled_off",
                      "pooled_mean_central", "pooled_var_central", "edge_1mr2_mean",
                      "pred_pooled_mean", "pred_pooled_var",
                      "cos_slope_beta", "cos_slope_u", "slope_ratio_med"]})
    print(f"n={n:5d}  ({time.time()-t0:5.1f}s)")
with open(os.path.join(CKPT_DIR, "summary_rows.json"), "w") as f:
    json.dump(rows, f, indent=1)

agg = collections.defaultdict(list)
for x in rows: agg[int(x["n"])].append(x)
gm = lambda n, k: np.mean([x[k] for x in agg[n]])
print("\n  n   | 1-R2 mean (Taylor)    | 1-R2 var (Taylor)     | 1-R2 offdiag | "
      "central mean | edge mean | cos(b,beta) cos(b,col) | slopeVar/slopeMean")
for n in WIDTHS:
    print(f"{n:5d} | {gm(n,'pooled_mean'):.3e} ({gm(n,'pred_pooled_mean'):.3e}) | "
          f"{gm(n,'pooled_var'):.3e} ({gm(n,'pred_pooled_var'):.3e}) | "
          f"{gm(n,'pooled_off'):.3e}    | {gm(n,'pooled_mean_central'):.3e}   | "
          f"{gm(n,'edge_1mr2_mean'):.3e} | {gm(n,'cos_slope_beta'):.4f}  {gm(n,'cos_slope_u'):.4f}    | "
          f"{gm(n,'slope_ratio_med'):.4f}")
ns = np.array(WIDTHS, float)
for key in ["pooled_mean", "pooled_var", "pooled_off"]:
    y = np.array([gm(int(n), key) for n in ns])
    print(f"width-scaling exponent {key}: {np.polyfit(np.log(ns), np.log(y), 1)[0]:+.3f}  (theory: -1)")
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for key, lab, c in [("pooled_mean", "cond. mean E[h_i|s]", "C0"),
                    ("pooled_var", "cond. variance Var(h_i|s)", "C1"),
                    ("pooled_off", "cond. off-diag Cov(h_i,h_j|s)", "C2")]:
    for n in WIDTHS:
        ax[0].plot([n]*len(agg[n]), [x[key] for x in agg[n]], ".", color=c, alpha=.35, ms=4)
    ax[0].loglog(ns, [gm(int(n), key) for n in ns], "o-", color=c, label=lab)
for key, c in [("pred_pooled_mean", "C0"), ("pred_pooled_var", "C1")]:
    ax[0].loglog(ns, [gm(int(n), key) for n in ns], "--", color=c, lw=1, alpha=.8)
ax[0].loglog(ns, 1.0/ns, ":", color="gray", label=r"$\propto 1/n$")
ax[0].set_xlabel("width n"); ax[0].set_ylabel(r"pooled $1-R^2$ of affine fit in $s$")
ax[0].set_title("post-ReLU conditional stats vs spike coord s\n(dashed = Taylor prediction)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")
for key, lab, c in [("pooled_mean", r"full range $|s|\leq 4\tau$", "C0"),
                    ("pooled_mean_central", r"central $|s|\leq 1.5\tau$", "C4"),
                    ("edge_1mr2_mean", r"edge $|s|\geq 2.5\tau$ (full-range fit)", "C3")]:
    ax[1].loglog(ns, [gm(int(n), key) for n in ns], "o-", color=c, label=lab)
ax[1].loglog(ns, 10/ns, ":", color="gray", label=r"$\propto 1/n$")
ax[1].set_xlabel("width n"); ax[1].set_ylabel(r"$1-R^2$ (mean curves)")
ax[1].set_title("where does linearity fail? edges are worse\nby a constant factor, same 1/n law")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "F1_scaling.png"), dpi=150); plt.show()
""")

# =============================================================================
md(r"""## §5 Anatomy of one net (n=512): the residual **is** the Taylor parabola

Strongest-coupled coordinate (worst case, $|\beta_i| \approx 3\times$ typical): the exact conditional
mean curve is visibly curved only at $|s|\gtrsim 2.5\tau$; subtracting the affine fit leaves
$\frac{\phi(0)\beta_i^2}{2\sigma_i}(s^2 - \langle s^2\rangle)$ on the nose. The conditional variance
is **not** constant — it tilts with slope $2\phi(0)\sigma_i \times$ the mean slope $\approx 0.8\,
\sigma_i$ — but affine captures that too. Off-diagonal pair covariances are the straightest of all.""")
code(r"""
r = analyze(512 if 512 in WIDTHS else WIDTHS[-1], 0)
sg, w = r["sgrid"], r["wgrid"]; tau = np.sqrt(float(r["tau2"])); x = sg / tau
ti = r["top_idx"].astype(int); beta, sig = r["beta"], r["sig"]
mc_, vc_ = r["mean_curves_top"], r["var_curves_top"]
b0, s0 = beta[ti[0]], sig[ti[0]]

fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
fit = wls_poly_r2(sg, mc_[0][:, None], w, deg=1); a_, b_ = fit["coef"][:, 0]
ax[0,0].plot(x, mc_[0], "C0", lw=2, label=r"exact $E[h_i\,|\,s]$")
ax[0,0].plot(x, a_ + b_*sg, "k--", lw=1, label="affine fit")
ax[0,0].set_title(f"strongest-coupled coord, $\\beta_i$={b0:+.3f}\n$1-R^2$ = {1-fit['r2'][0]:.1e}")
ax[0,0].set_xlabel(r"$s/\tau$"); ax[0,0].legend(fontsize=8)

resid = mc_[0] - (a_ + b_*sg); s2w = float((w*sg**2).sum())
ax[0,1].plot(x, 1e3*resid, "C0", lw=2, label="residual (exact − affine)")
ax[0,1].plot(x, 1e3*PHI0*b0**2/(2*s0)*(sg**2 - s2w), "k--", lw=1,
             label=r"Taylor $\frac{\phi(0)\beta_i^2}{2\sigma_i}(s^2-\langle s^2\rangle)$")
ax2 = ax[0,1].twinx(); ax2.plot(x, w/w.max(), color="gray", alpha=.25); ax2.set_yticks([])
ax[0,1].set_title("the residual IS the quadratic Taylor term\n(gray: Gaussian mass of s)")
ax[0,1].set_xlabel(r"$s/\tau$"); ax[0,1].set_ylabel(r"residual $\times 10^3$"); ax[0,1].legend(fontsize=8)

fitv = wls_poly_r2(sg, vc_[0][:, None], w, deg=1); av, bv = fitv["coef"][:, 0]
ax[1,0].plot(x, vc_[0], "C1", lw=2, label=r"exact Var$(h_i|s)$")
ax[1,0].plot(x, av + bv*sg, "k--", lw=1, label="affine fit")
ax[1,0].set_title(f"conditional variance is NOT constant: slope ratio var/mean = {bv/b_:.3f} "
                  f"(theory $2\\phi(0)\\sigma_i$ = {2*PHI0*s0:.3f})\n$1-R^2$ = {1-fitv['r2'][0]:.1e}")
ax[1,0].set_xlabel(r"$s/\tau$"); ax[1,0].legend(fontsize=8)

oc, rho_t = r["off_curves_top"], r["rho_top"]
fito = wls_poly_r2(sg, oc[0][:, None], w, deg=1)
ax[1,1].plot(x, oc[0], "C2", lw=2, label=rf"exact Cov$(h_i,h_j|s)$, $\rho$={rho_t[0]:+.3f}")
ax[1,1].plot(x, fito["coef"][0,0] + fito["coef"][1,0]*sg, "k--", lw=1, label="affine fit")
ax[1,1].set_title(f"off-diagonal pair cov: $1-R^2$ = {1-fito['r2'][0]:.1e}")
ax[1,1].set_xlabel(r"$s/\tau$"); ax[1,1].legend(fontsize=8)
for a in ax.ravel(): a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "F2_anatomy_n512.png"), dpi=150); plt.show()
""")

# =============================================================================
md(r"""## §6 Quantitative explanation: per-coordinate collapse on the Taylor prediction

Every bulk coordinate's measured $1-R^2$ (mean AND variance curves) sits on the closed-form
prediction over ~9 decades — the affine accuracy is *fully explained* by
"first-order Taylor of a smooth rectified-Gaussian map on an $O(\beta_i s)$ interval".
The fitted slope vector equals $\Phi(0)\,\beta$ to machine precision, and only
$\cos \approx 1/\sqrt 2$ of it lives in the naked coupling column $W_{[1:,0]}$ (the other half is
the row-overlap $W_{\mathrm{bulk}} w_0^\top$).""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for n, c in [(WIDTHS[len(WIDTHS)//2], "C0"), (WIDTHS[-1], "C3")]:
    rr = analyze(n, 0)
    ax[0].loglog(rr["pred_mean_per"], 1 - rr["r2_mean_per"], ".", color=c, ms=3, alpha=.35, label=f"n={n}")
    ax[1].loglog(rr["pred_var_per"], 1 - rr["r2_var_per"], ".", color=c, ms=3, alpha=.35, label=f"n={n}")
for a, t in [(ax[0], r"mean curves: $1\!-\!R^2_i$ vs $2\phi(0)^2\beta_i^2\tau^2/\sigma_i^2$"),
             (ax[1], r"var curves: $1\!-\!R^2_i$ vs $\left(\frac{1/2-1/\pi}{\sigma_i\phi(0)}\right)^2\!\beta_i^2\tau^2/2$")]:
    lim = np.array([1e-12, 1e-1]); a.loglog(lim, lim, "k--", lw=1)
    a.set_xlabel("Taylor prediction"); a.set_ylabel("measured $1-R^2$ per coord")
    a.set_title(t); a.legend(); a.grid(alpha=.3, which="both")
rmid = analyze(WIDTHS[len(WIDTHS)//2], 0)
ax[0].text(.05, .95, f"slope dir: cos(b, $\\beta$) = {float(rmid['cos_slope_beta']):.6f}\n"
           f"cos(b, col $W_{{[:,0]}}$) = {float(rmid['cos_slope_u']):.3f}",
           transform=ax[0].transAxes, va="top", fontsize=8, bbox=dict(fc="white", alpha=.8, ec="gray"))
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "F3_collapse.png"), dpi=150); plt.show()
""")

# =============================================================================
md(r"""## §7 MC validation (n=512): the exact curves are what binned MC converges to

600k samples, 24 equal-mass bins of $s$. Bin means/variances agree with the exact conditional
curves within sampling error (max $|z| \approx 2.9$ over 288 bin-stats). Note the cautionary tale:
the naive binned-MC $1-R^2$ on the same coords/bins reads **~2× the true residual** — MC noise
inflates $1-R^2$ long before the true $O(1/n)$ curvature is visible (that is what the split-half
debiasing in `empirical_structure` is for, and why this notebook conditions exactly instead).""")
code(r"""
if RUN_MC:
    n, seed = 512, 0
    from scipy.stats import norm as _norm
    M = build_spiked_net(n, depth=1, seed=seed)[0][0]
    G = M @ M.T; tau2 = G[0,0]; tau = np.sqrt(tau2)
    beta = G[1:,0]/tau2; sig2 = np.diag(G)[1:] - beta**2*tau2; sig = np.sqrt(sig2)
    coords = np.argsort(-np.abs(beta))[:12]
    mc_path = os.path.join(CKPT_DIR, f"mc_l1_n{n}_seed{seed}_N{MC_N}_B{MC_BINS}.npz")
    if not os.path.exists(mc_path):
        edges = _norm.ppf(np.linspace(0, 1, MC_BINS+1)) * tau; edges[0], edges[-1] = -np.inf, np.inf
        rng = np.random.default_rng(12345)
        cnt = np.zeros(MC_BINS); s_sum = np.zeros(MC_BINS)
        h_sum = np.zeros((MC_BINS, len(coords))); h2_sum = np.zeros((MC_BINS, len(coords)))
        Mf = M.astype(np.float32); done = 0
        while done < MC_N:
            b = min(100_000, MC_N - done)
            Z = rng.standard_normal((b, n), dtype=np.float32) @ Mf.T
            s = Z[:,0].astype(np.float64)
            Hb = np.maximum(Z[:,1:][:, coords], 0).astype(np.float64)
            bi = np.clip(np.searchsorted(edges, s)-1, 0, MC_BINS-1)
            cnt += np.bincount(bi, minlength=MC_BINS)
            s_sum += np.bincount(bi, weights=s, minlength=MC_BINS)
            for c in range(len(coords)):
                h_sum[:,c] += np.bincount(bi, weights=Hb[:,c], minlength=MC_BINS)
                h2_sum[:,c] += np.bincount(bi, weights=Hb[:,c]**2, minlength=MC_BINS)
            done += b
        s_bar = s_sum/cnt; mc_mean = h_sum/cnt[:,None]; mc_var = h2_sum/cnt[:,None] - mc_mean**2
        np.savez_compressed(mc_path, cnt=cnt, s_bar=s_bar, mc_mean=mc_mean, mc_var=mc_var, coords=coords)
    d = np.load(mc_path)
    cnt, s_bar, mc_mean, mc_var, coords = d["cnt"], d["s_bar"], d["mc_mean"], d["mc_var"], d["coords"]
    MUc = beta[coords][None,:]*s_bar[:,None]
    ex_mean, _, ex_var = relu_moments_1d(MUc, np.broadcast_to(sig2[coords][None,:], MUc.shape))
    z_mean = (mc_mean-ex_mean)/np.sqrt(np.maximum(mc_var,1e-12)/cnt[:,None])
    z_var  = (mc_var-ex_var)/(mc_var*np.sqrt(2.0/cnt[:,None]))
    print(f"mean curves: max|z| = {np.abs(z_mean).max():.2f}  rms z = {np.sqrt((z_mean**2).mean()):.2f}")
    print(f"var  curves: max|z| = {np.abs(z_var).max():.2f}  rms z = {np.sqrt((z_var**2).mean()):.2f} "
          f"(SE formula ignores ReLU kurtosis -> slightly conservative)")
    w_mc = cnt/cnt.sum()
    print(f"naive binned-MC pooled 1-R2 (top-12 coords) = {pooled_1mR2(wls_poly_r2(s_bar, mc_mean, w_mc)):.3e}")
    print(f"exact curves, same bins/coords              = {pooled_1mR2(wls_poly_r2(s_bar, ex_mean, w_mc)):.3e}")

    c0 = 0; sfine = tau*np.linspace(-3.2, 3.2, 200)
    exm, _, exv = relu_moments_1d(beta[coords[c0]]*sfine, np.full_like(sfine, sig2[coords[c0]]))
    wf = _phi(sfine/tau)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(s_bar/tau, mc_mean[:,c0], yerr=2*np.sqrt(mc_var[:,c0]/cnt), fmt="o", ms=4,
                   color="C0", label="MC bin means ±2SE")
    ax[0].plot(sfine/tau, exm, "C3", lw=1.5, label="exact conditional curve")
    f_ = wls_poly_r2(sfine, exm[:,None], wf); ax[0].plot(sfine/tau, f_["coef"][0,0]+f_["coef"][1,0]*sfine,
                                                         "k--", lw=1, label="affine fit")
    ax[0].set_title(f"MC validation, n=512: E[h_i | s]  (coord {coords[c0]})")
    ax[1].errorbar(s_bar/tau, mc_var[:,c0], yerr=2*mc_var[:,c0]*np.sqrt(2/cnt), fmt="o", ms=4,
                   color="C1", label="MC bin variances ±2SE")
    ax[1].plot(sfine/tau, exv, "C3", lw=1.5, label="exact conditional curve")
    fv_ = wls_poly_r2(sfine, exv[:,None], wf); ax[1].plot(sfine/tau, fv_["coef"][0,0]+fv_["coef"][1,0]*sfine,
                                                          "k--", lw=1, label="affine fit")
    ax[1].set_title("Var(h_i | s): visibly tilted, affine tracks it")
    for a in ax: a.set_xlabel(r"$s/\tau$"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "F4_mc_overlay.png"), dpi=150); plt.show()
else:
    print("RUN_MC=False (QUICK) -- skipping")
""")

# =============================================================================
md(r"""## §8 Conclusions

**Why affine-in-$s$ works post-activation (layer 1, exact):**

1. **Pre-activation is exactly affine** — $(s, Z_{\mathrm{bulk}})$ jointly Gaussian $\Rightarrow$
   conditional mean exactly linear in $s$, conditional covariance exactly constant. The ReLU is the
   only possible source of curvature.
2. **The ReLU is smooth here** — post-activation conditional stats are rectified-Gaussian integrals
   of $(\beta_i s, \sigma_i^2)$; the bulk's own conditional noise $\sigma_i \approx 1$ convolves the
   kink away, leaving analytic maps with $O(1)$ curvature scale.
3. **The spike coupling is weak** — $|\beta_i s| \lesssim 4|\beta_i|\tau = O(\sqrt{1/n})$ over the
   entire observable $s$-range. Affine-in-$s$ is the tangent line of a smooth map on a shrinking
   interval: $1-R^2 = 2\phi(0)^2\beta_i^2\tau^2/\sigma_i^2 \propto 1/n$ (mean), with the variance
   curve $3.07\times$ more linear and off-diagonals more linear still. Verified to <1% pooled and
   per-coordinate over 9 decades.
4. **Not a near-zero accident** — the residual is the Taylor parabola: edge cells are ~3.3× worse
   than pooled, central $|s|\le1.5\tau$ ~3.2× better, all falling as $1/n$. Genuine failure of
   affine requires $|\beta_i s| \sim \sigma_i$, i.e. a $\sim\!\sqrt{2n}\,$-sd spike excursion
   (≈$7\tau$ for the most-coupled coordinate at $n=512$, ≈$22\tau$ typical) — outside any bin that
   carries mass.
5. **The covariance is NOT constant in $s$** (user expectation check): the conditional variance
   tilts with slope $\phi(0)\sigma_i\beta_i = 2\phi(0)\sigma_i\times$(mean slope) $\approx 0.80\,
   \sigma_i\times$; in *relative* terms mean and variance move almost identically
   ($2\phi(0)^2/(1/2-1/(2\pi)) \approx 0.93$). A constant-$\Sigma$ model errs at $O(1/\sqrt n)$;
   the affine family is what removes the whole $O(1/\sqrt n)$ term, leaving $O(1/n)$ — this is why
   affine-conditioned K=2 (analytic_kprop) can match binned with $O(1)$ parameters per layer.
6. **Slope direction** $= \Phi(0)\,\beta$ with $\beta = (MM^\top)_{\mathrm{bulk},0}/\tau^2$ — half
   column-leak $W_{[1:,0]}$, half row-overlap $W_{\mathrm{bulk}}w_0^\top$ ($\cos$ with the naked
   column $\approx 1/\sqrt2$), refining the "aligns with coupling column" reading in
   `binned_structure_test`.

**What this does *not* explain**: deeper layers. There the bulk-given-spike *pre-activation* law is
no longer exactly Gaussian/affine (the affine_r2 notebook sees PRE $1-R^2 \sim 0.15$–$0.25$ at
L1/L2), so the deep-layer question becomes: how much conditional non-Gaussianity/curvature does the
*propagated* law accumulate, given that (per this notebook) each ReLU step adds only $O(1/n)$
curvature in $s$? Natural next step: feed a *non*-Gaussian but affine conditional family through one
exact layer and measure the curvature it creates — i.e. the layer-2 version of this experiment.""")

nb.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "affine_conditional_layer1_colab.ipynb"))

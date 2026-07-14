"""Generates widthlaw_significance_colab.ipynb (valid nbformat-4 JSON).

IS THE -1.8 REAL? Statistical verification of the binned-kprop (K=2) width law on the
coordinate spike M = W + e1 e1^T. A previous sweep fitted MSE ~ n^{-1.8}; theory says
n^{-2}. Three artifacts can fake -1.8 and this notebook kills each one:

  1. MC-NOISE INFLATION. The old metric ||pred - mc||^2 carries a +E||se||^2 bias that
     FLOORS at large n and flattens the slope. Fix: SPLIT-HALF CROSS-MSE
         <pred - mc_A, pred - mc_B> / ||mc||^2
     over two independent MC halves -- exactly unbiased for the true squared error at any
     sample count. (We also plot the naive MSE next to it: if naive shows -1.8 while cross
     shows -2.0, the old slope was the noise floor.)
  2. BIN-DISCRETIZATION FLOOR. Fixed num_bins => width-independent spike-resolution error
     that flattens the slope at large n. Fix: generous NB_MAIN=63 wasserstein bins + an
     explicit convergence check (127 bins at selected widths must not move the cross-MSE).
  3. WEAK STATISTICS. 3-4 widths x 2 seeds, unweighted polyfit, no SE. Fix: ~12 log-dense
     widths x 8 seeds, WEIGHTED OLS on log(MSE) with seed-scatter SEs -> slope +/- SE, 95% CI,
     z-test of H0: slope = -2; PLUS the sharper mixture fit MSE = A n^{-2} + B n^{-1}
     (an apparent -1.8 over a finite window is usually a -2 term plus a small -1 tail --
     bootstrap CI on B tells you if the shallow component is real); PLUS local pairwise
     slopes (curvature: crossover vs genuine anomalous exponent).

MC budget: per-seed SD(cross-MSE)/MSE ~ sqrt(2) * noise/err, and err ~ E0/n, so
N(n) = clip(MC_COEF n^2, ., .) keeps the top width resolved. MC runs on GPU (float32
forward, float64 accumulators -- f32 rounding ~1e-7 << err ~1e-4) and every artifact
(MC halves, predictions) is cached per (width, seed) -> disconnects resume for free.

Run:  python "experiments/binned_kprop/build_widthlaw_significance_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# Is the $-1.8$ real? — significance test of the binned-kprop $\mathrm{MSE}\sim n^{-2}$ width law

**Question.** The coordinate-spike ($M=W+e_1e_1^\top$) binned-$K2$ sweep fitted
$\mathrm{MSE}\sim n^{-1.8}$; the budget law says $n^{-2}$. Is $-1.8$ statistically distinguishable
from $-2$, or an artifact?

**Three artifacts that fake $-1.8$, and the fix for each:**

| artifact | why it flattens the slope | fix here |
|---|---|---|
| MC-noise inflation | $\|\hat e\|^2 = \|e\|^2 + \mathbb E\|se\|^2$ — the noise term floors at large $n$ | **split-half cross-MSE** $\langle \mathrm{pred}-\mathrm{mc}_A,\ \mathrm{pred}-\mathrm{mc}_B\rangle/\|\mathrm{mc}\|^2$, exactly unbiased; naive MSE plotted alongside for the autopsy |
| bin-discretization floor | fixed `num_bins` ⇒ width-independent error term | generous 63 wasserstein bins + **127-bin convergence check** at selected widths |
| weak statistics | 3–4 widths × 2 seeds, unweighted fit, no SE | **~12 log-dense widths × 8 seeds**, weighted OLS with seed-scatter SEs → slope ± SE, 95% CI, $z$-test vs $-2$; **mixture fit** $A n^{-2}+B n^{-1}$ with bootstrap CI on $B$; local pairwise slopes |

**Reading the outcome.** If cross-MSE fits slope $\approx-2$ while naive MSE fits $\approx-1.8$:
the old slope was the MC noise floor. If cross-MSE itself sits at $-1.8$ with tight SE **and** the
mixture fit needs $B>0$: there is a real subleading $n^{-1}$ error channel (e.g. residual spike
structure the binning doesn't capture). If local slopes drift $-1.6\to-2$ with $n$: finite-width
crossover, asymptotically $-2$.

**Budget.** GPU MC (float32 forward, float64 accumulators), $N(n)=\mathrm{clip}(36n^2,\ 2\mathrm M,\ 40\mathrm M)$;
everything cached per (width, seed) → resumable. Full sweep ≈ **25–40 min on a T4**
(`RUN_XL=True` adds $n{=}1536$, ~+20 min). `QUICK` (auto on CPU): widths ≤ 192, ~5–10 min.""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 — Config, spiked net, split-half MC (cached), cross-MSE

Same net builder as `Mecha_preds.binned_kprop` (`theta=1`, `out_dim=8` readout). The recyclable
artifacts are the **MC half-means** and the **prediction vectors**, cached per (width, seed) in
`CKPT_DIR` — re-runs and Colab disconnects recompute nothing.""")
code(r"""
import os, time, math, json
import numpy as np
import matplotlib.pyplot as plt

from Mecha_preds.binned_kprop import run_binned_kprop_k2

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    torch, HAS_CUDA = None, False

# ---------------- knobs ----------------
QUICK    = not HAS_CUDA          # CPU -> small sweep; set False manually to force the full one
DEPTH    = 2                     # the depth the -1.8 was measured at (knob: 3)
THETA    = 1.0
OUT_DIM  = 8
NB_MAIN  = 63                    # main bin count (wasserstein grid)
NB_CHECK = 127                   # bin-convergence check value
GRID     = "wasserstein"
BULK_RELU = "exact"

WIDTHS   = [24, 32, 48, 64, 96, 128, 192] if QUICK else \
           [24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
RUN_XL   = False                 # append n=1536 (T4: ~+20 min)
if RUN_XL and not QUICK: WIDTHS = WIDTHS + [1536]
SEEDS    = [10, 11, 12, 13] if QUICK else [10, 11, 12, 13, 14, 15, 16, 17]
CHECK_WIDTHS = [64, 192] if QUICK else [256, 1024]   # where the 127-bin check runs

# MC budget: noise ~ a/sqrt(N), err ~ E0/n  =>  N ~ n^2 keeps noise/err bounded.
MC_COEF, MC_MIN, MC_CAP = 36, 2_000_000, 40_000_000
def mc_for(n): return int(min(MC_CAP, max(MC_MIN, MC_COEF * n * n)))
MC_BATCH = 262_144 if HAS_CUDA else 100_000

CKPT_DIR = "checkpoints/binned_kprop/widthlaw"
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"QUICK={QUICK} cuda={HAS_CUDA} | depth={DEPTH} nb={NB_MAIN}({GRID}) check={NB_CHECK}@{CHECK_WIDTHS}")
print(f"widths={WIDTHS}\nseeds={SEEDS} | MC N(n): " + ", ".join(f"{n}:{mc_for(n):.0e}" for n in WIDTHS))
""")

code(r"""
# ---- net (identical to Mecha_preds.binned_kprop.selftest.coordinate_spike_net) ----
def coordinate_spike_net(n, depth, seed, *, theta=THETA, out_dim=OUT_DIM):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n)); P[0, 0] = theta
    Ws = [(rng.standard_normal((n, n)) / np.sqrt(n) + P, None) for _ in range(depth)]
    Ws.append((rng.standard_normal((out_dim, n)) / np.sqrt(n), None))
    return Ws

# ---- one MC half: E[output] + per-coordinate se. GPU: f32 forward, f64 accumulators ----
def _mc_half_numpy(Ws, n, samples, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros(Ws[-1][0].shape[0]); accsq = np.zeros_like(acc); c = 0
    while c < samples:
        b = min(MC_BATCH, samples - c); h = rng.standard_normal((b, n))
        for li, (W, _b) in enumerate(Ws):
            z = h @ W.T; h = np.maximum(z, 0.0) if li < len(Ws) - 1 else z
        acc += h.sum(0); accsq += (h ** 2).sum(0); c += b
    mu = acc / c
    return mu, np.sqrt(np.clip(accsq / c - mu ** 2, 0, None) / c)

def _mc_half_torch(Ws, n, samples, seed):
    dev = torch.device("cuda")
    Wt = [torch.as_tensor(W, dtype=torch.float32, device=dev) for W, _ in Ws]
    g = torch.Generator(device=dev).manual_seed(seed)
    acc = torch.zeros(Ws[-1][0].shape[0], dtype=torch.float64, device=dev); accsq = acc.clone(); c = 0
    while c < samples:
        b = min(MC_BATCH, samples - c)
        h = torch.randn(b, n, generator=g, dtype=torch.float32, device=dev)
        for li, W in enumerate(Wt):
            z = h @ W.T; h = torch.relu(z) if li < len(Wt) - 1 else z
        acc += h.sum(0).double(); accsq += (h.double() ** 2).sum(0); c += b
    mu = acc / c; se = torch.sqrt(torch.clamp(accsq / c - mu ** 2, min=0) / c)
    return mu.cpu().numpy(), se.cpu().numpy()

# ---- cached split-half MC: two INDEPENDENT halves per (width, seed) ----
def mc_halves(n, seed):
    key = os.path.join(CKPT_DIR, f"mc_d{DEPTH}_w{n}_s{seed}_th{THETA:g}_N{mc_for(n)}.npz")
    if os.path.exists(key):
        z = np.load(key); return z["muA"], z["muB"], z["seA"], z["seB"]
    Ws = coordinate_spike_net(n, DEPTH, seed)
    half = mc_for(n) // 2
    f = _mc_half_torch if HAS_CUDA else _mc_half_numpy
    muA, seA = f(Ws, n, half, 10_000 + seed)
    muB, seB = f(Ws, n, half, 20_000 + seed)
    np.savez(key, muA=muA, muB=muB, seA=seA, seB=seB)
    return muA, muB, seA, seB

# ---- cached predictions (tiny vectors) ----
def prediction(n, seed, nb):
    key = os.path.join(CKPT_DIR, f"pred_d{DEPTH}_w{n}_s{seed}_nb{nb}_{GRID}.npz")
    if os.path.exists(key):
        return np.load(key)["mean"]
    p = run_binned_kprop_k2(coordinate_spike_net(n, DEPTH, seed), n,
                            num_bins=nb, grid=GRID, bulk_relu=BULK_RELU)["mean"]
    np.savez(key, mean=np.asarray(p, dtype=np.float64)); return p

# ---- the estimators ----
def cross_mse(pred, muA, muB, mu):        # UNBIASED for ||pred-mu_true||^2/||mu||^2
    return float((pred - muA) @ (pred - muB) / (mu @ mu))
def naive_mse(pred, mu):                  # what the old pipeline measured
    return float((pred - mu) @ (pred - mu) / (mu @ mu))
print("helpers ready: mc_halves[cached], prediction[cached], cross_mse, naive_mse")
""")

# =============================================================================
md(r"""## §2 — Sweep: split-half MC + predictions over (width, seed)

Per run: two independent MC halves, the 63-bin prediction, and the 1-bin baseline (its $\sim n^{-1}$
slope is the contrast). Prints per-width cross-MSE mean ± SEM next to the naive MSE and the MC noise
level — you can already see whether naive and cross diverge at large $n$.""")
code(r"""
R = {}          # (n, seed) -> dict
t0 = time.time()
for n in WIDTHS:
    for s in SEEDS:
        t1 = time.time()
        muA, muB, seA, seB = mc_halves(n, s)
        mu = 0.5 * (muA + muB)
        p_main = prediction(n, s, NB_MAIN)
        p_1bin = prediction(n, s, 1)
        noise2 = float((0.25 * (seA**2 + seB**2)).sum() / (mu @ mu))   # E||se_full||^2 / ||mu||^2
        R[(n, s)] = dict(
            cross=cross_mse(p_main, muA, muB, mu), naive=naive_mse(p_main, mu),
            cross1=cross_mse(p_1bin, muA, muB, mu), noise2=noise2, secs=time.time() - t1)
    cs = np.array([R[(n, s)]["cross"] for s in SEEDS]); ns_ = np.array([R[(n, s)]["naive"] for s in SEEDS])
    print(f"n={n:5d} | cross-MSE {cs.mean():.3e} ± {cs.std(ddof=1)/math.sqrt(len(SEEDS)):.1e} | "
          f"naive {ns_.mean():.3e} | noise^2 {np.mean([R[(n,s)]['noise2'] for s in SEEDS]):.1e} | "
          f"1bin {np.mean([R[(n,s)]['cross1'] for s in SEEDS]):.2e} | "
          f"{sum(R[(n,s)]['secs'] for s in SEEDS):5.1f}s", flush=True)
print(f"\nsweep done in {(time.time()-t0)/60:.1f} min (cached points are instant)")
""")

# =============================================================================
md(r"""## §3 — Bin-convergence check: is 63 bins past the knee at every width used?

At `CHECK_WIDTHS`, recompute with 127 bins. If the cross-MSE moves by more than ~20 %, the
bin-discretization floor is contaminating those widths and `NB_MAIN` must rise before the slope
means anything.""")
code(r"""
for n in CHECK_WIDTHS:
    r63  = np.mean([cross_mse(prediction(n, s, NB_MAIN),  *mc_halves(n, s)[:2],
                    0.5*(mc_halves(n, s)[0]+mc_halves(n, s)[1])) for s in SEEDS])
    r127 = np.mean([cross_mse(prediction(n, s, NB_CHECK), *mc_halves(n, s)[:2],
                    0.5*(mc_halves(n, s)[0]+mc_halves(n, s)[1])) for s in SEEDS])
    ratio = r127 / r63 if r63 > 0 else float("nan")
    print(f"n={n:5d}: cross-MSE 63 bins {r63:.3e}  vs 127 bins {r127:.3e}  ratio {ratio:.2f}  "
          f"-> {'bin-CONVERGED' if 0.8 <= ratio <= 1.25 else 'NOT converged: raise NB_MAIN'}")
""")

# =============================================================================
md(r"""## §4 — The statistics: slope ± SE, $z$-test vs $-2$, mixture fit, local slopes

1. **Weighted OLS** of $\log(\text{cross-MSE})$ on $\log n$, weights from seed-scatter SEMs
   (SE inflated by $\sqrt{\chi^2/\mathrm{dof}}$ if the fit is over-dispersed) → slope ± SE, 95 % CI,
   and the $z$-test of $H_0\!:\ p=-2$. Same fit on the **naive** MSE for the autopsy.
2. **Mixture fit** $\mathrm{MSE}(n) = A\,n^{-2} + B\,n^{-1}$ (weighted, $A,B\ge0$) with a
   seed-resampling **bootstrap CI on $B$** — the sharp question "is there a real shallow
   component?", which a single power-law exponent can't answer over a finite window.
3. **Local pairwise slopes** between adjacent widths — curvature reveals crossover vs a genuine
   anomalous exponent.""")
code(r"""
WID = np.array(WIDTHS, float)
def seed_stats(key):
    m, se = [], []
    for n in WIDTHS:
        v = np.array([R[(n, s)][key] for s in SEEDS])
        m.append(v.mean()); se.append(v.std(ddof=1) / math.sqrt(len(v)))
    return np.array(m), np.array(se)

def wls_loglog(m, sem):
    ok = (m > 2 * sem)                       # resolved: mean positive at 2 sigma
    x, y = np.log(WID[ok]), np.log(m[ok]); sy = (sem / m)[ok]   # delta method
    W_ = 1.0 / sy**2
    X = np.vstack([x, np.ones_like(x)]).T
    C = np.linalg.inv(X.T @ (W_[:, None] * X))
    beta = C @ X.T @ (W_ * y)
    resid = y - X @ beta
    chi2 = float((W_ * resid**2).sum()); dof = max(len(x) - 2, 1)
    infl = max(1.0, math.sqrt(chi2 / dof))   # over-dispersion inflation
    se_slope = math.sqrt(C[0, 0]) * infl
    return dict(slope=float(beta[0]), se=se_slope, chi2=chi2, dof=dof, npts=int(ok.sum()), used=ok)

from scipy.optimize import nnls
from scipy.stats import norm
def mixture_fit(m, sem, ok):
    X = np.vstack([WID[ok]**-2.0, WID[ok]**-1.0]).T / sem[ok, None]
    y = m[ok] / sem[ok]
    coef, _ = nnls(X, y); return coef       # A, B >= 0

m_c, se_c = seed_stats("cross")
m_n, se_n = seed_stats("naive")
m_1, se_1 = seed_stats("cross1")

f_c = wls_loglog(m_c, se_c); f_n = wls_loglog(m_n, se_n); f_1 = wls_loglog(m_1, se_1)
z = (f_c["slope"] - (-2.0)) / f_c["se"]; pval = 2 * (1 - norm.cdf(abs(z)))
print(f"binned {NB_MAIN}-bin CROSS-MSE : slope {f_c['slope']:+.3f} ± {f_c['se']:.3f} "
      f"(95% CI [{f_c['slope']-1.96*f_c['se']:+.2f}, {f_c['slope']+1.96*f_c['se']:+.2f}], "
      f"chi2/dof {f_c['chi2']/f_c['dof']:.1f}, {f_c['npts']} widths)")
print(f"  H0 slope = -2 :  z = {z:+.2f},  p = {pval:.3f}   -> "
      + ("CONSISTENT with -2" if pval > 0.05 else "REJECTS -2"))
print(f"binned NAIVE MSE (old metric): slope {f_n['slope']:+.3f} ± {f_n['se']:.3f}"
      f"   <- if this is ~-1.8 while cross is ~-2, the old -1.8 was the MC noise floor")
print(f"1-bin baseline cross-MSE     : slope {f_1['slope']:+.3f} ± {f_1['se']:.3f}   (theory ~ -1)")

# ---- mixture fit + bootstrap CI on the shallow component B ----
ok = f_c["used"]
A, B = mixture_fit(m_c, se_c, ok)
rng = np.random.default_rng(0); Bs = []
for _ in range(2000):
    mb = np.array([np.mean(rng.choice([R[(n, s)]["cross"] for s in SEEDS], len(SEEDS), replace=True))
                   for n in WIDTHS])
    Bs.append(mixture_fit(mb, se_c, ok)[1])
Bs = np.array(Bs); lo, hi = np.percentile(Bs, [2.5, 97.5])
n_top = WID[ok][-1]
share = B * n_top**-1 / (A * n_top**-2 + B * n_top**-1 + 1e-300)
print(f"\nmixture MSE = A n^-2 + B n^-1 :  A = {A:.3g}, B = {B:.3g}, "
      f"bootstrap 95% CI on B = [{lo:.2g}, {hi:.2g}]")
print(f"  -> shallow n^-1 component {'REAL' if lo > 0 else 'NOT significant'}; "
      f"at n={int(n_top)} it carries {100*share:.0f}% of the error")

# ---- local pairwise slopes ----
print("\nlocal slopes (adjacent widths, cross-MSE):")
for i in range(len(WIDTHS) - 1):
    if m_c[i] > 0 and m_c[i+1] > 0:
        sl = (math.log(m_c[i+1]) - math.log(m_c[i])) / (math.log(WID[i+1]) - math.log(WID[i]))
        er = math.sqrt((se_c[i]/m_c[i])**2 + (se_c[i+1]/m_c[i+1])**2) / (math.log(WID[i+1]) - math.log(WID[i]))
        print(f"  {WIDTHS[i]:>5d} -> {WIDTHS[i+1]:<5d}: {sl:+.2f} ± {er:.2f}")
""")

# =============================================================================
md(r"""## §5 — Plots

Left: cross-MSE (the unbiased truth) vs the naive MSE (old metric) and the MC noise floor, with
$n^{-2}$, $n^{-1.8}$, $n^{-1}$ references and the fitted mixture curve. Right: local slopes with
error bars — flat at the fitted value = clean power law; drifting toward $-2$ = crossover.""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
nz2, _ = seed_stats("noise2")
ax.errorbar(WID, m_c, yerr=se_c, fmt="o-", color="tab:blue", label=f"cross-MSE (slope {f_c['slope']:+.2f}±{f_c['se']:.2f})")
ax.loglog(WID, m_n, "s--", color="tab:orange", alpha=.8, label=f"naive MSE, old metric ({f_n['slope']:+.2f})")
ax.loglog(WID, m_1, "^-", color="tab:gray", alpha=.7, label=f"1-bin baseline ({f_1['slope']:+.2f})")
ax.loglog(WID, nz2, ":", color="k", alpha=.6, label="MC noise$^2$ floor")
a0 = m_c[np.argmax(WID)]
for p, c, lab in ((-2.0, "k", "$n^{-2}$"), (-1.8, "tab:red", "$n^{-1.8}$"), (-1.0, "0.6", "$n^{-1}$")):
    ax.loglog(WID, a0 * (WID / WID[-1])**p, "--", color=c, lw=1, alpha=.6, label=lab)
ax.loglog(WID, A * WID**-2.0 + B * WID**-1.0, "-", color="tab:green", lw=1.2, alpha=.8,
          label="mixture $An^{-2}+Bn^{-1}$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("width n"); ax.set_ylabel("relative MSE")
ax.set_title(f"binned-K2 width law (depth {DEPTH}, {NB_MAIN} bins, {len(SEEDS)} seeds)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)

ax = axes[1]
xs_, ys_, es_ = [], [], []
for i in range(len(WIDTHS) - 1):
    if m_c[i] > 0 and m_c[i+1] > 0:
        dx = math.log(WID[i+1]) - math.log(WID[i])
        xs_.append(math.sqrt(WID[i] * WID[i+1]))
        ys_.append((math.log(m_c[i+1]) - math.log(m_c[i])) / dx)
        es_.append(math.sqrt((se_c[i]/m_c[i])**2 + (se_c[i+1]/m_c[i+1])**2) / dx)
ax.errorbar(xs_, ys_, yerr=es_, fmt="o-", color="tab:blue")
ax.axhline(-2.0, color="k", ls="--", lw=1, label="-2 (theory)")
ax.axhline(-1.8, color="tab:red", ls="--", lw=1, alpha=.7, label="-1.8 (old fit)")
ax.axhline(f_c["slope"], color="tab:blue", ls=":", lw=1, label=f"global fit {f_c['slope']:+.2f}")
ax.set_xscale("log"); ax.set_xlabel("width n (geometric midpoint)"); ax.set_ylabel("local slope")
ax.set_title("local pairwise slopes (curvature check)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §6 — Checkpoints (recycle across sessions)""")
code(r"""
import shutil
tot = 0
for f in sorted(os.listdir(CKPT_DIR)):
    tot += os.path.getsize(os.path.join(CKPT_DIR, f))
print(f"checkpoint dir: {os.path.abspath(CKPT_DIR)}  ({len(os.listdir(CKPT_DIR))} files, {tot/1e6:.1f} MB)")
if IN_COLAB:
    from google.colab import files
    z = shutil.make_archive("/content/widthlaw_ckpts", "zip", CKPT_DIR)
    print("zipped ->", z, "-- downloading..."); files.download(z)
""")

# =============================================================================
md(r"""## §7 — Summary

- **What ran:** binned-kprop $K2$ ({NB} wasserstein bins) on $M=W+e_1e_1^\top$, depth-{D}, ~12
  log-dense widths × {S} seeds, split-half GPU Monte-Carlo with $N(n)\propto n^2$.
- **The unbiased metric:** cross-MSE $\langle \mathrm{pred}-\mathrm{mc}_A,\mathrm{pred}-\mathrm{mc}_B\rangle$
  — immune to the MC-noise floor that inflates the naive $\|\mathrm{pred}-\mathrm{mc}\|^2$ (plotted
  side by side; their divergence at large $n$ is the autopsy of the old $-1.8$).
- **The verdicts (§4):** weighted slope ± SE with a $z$-test against $-2$; the mixture fit
  $An^{-2}+Bn^{-1}$ with a bootstrap CI on $B$ (is a shallow error channel real?); local slopes
  (crossover vs constant exponent). §3 certifies the bin floor isn't binding.
- **Recycling:** MC halves and prediction vectors cached per (width, seed) in
  `checkpoints/binned_kprop/widthlaw` — disconnects resume, extra seeds/widths are incremental.""".replace(
    "{NB}", "63").replace("{D}", "2").replace("{S}", "8"))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "widthlaw_significance_colab.ipynb")
nb.save(out)

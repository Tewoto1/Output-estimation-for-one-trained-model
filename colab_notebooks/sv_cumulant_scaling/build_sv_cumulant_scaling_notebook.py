"""build_sv_cumulant_scaling_notebook.py -- GENERATE the S/V cumulant-scaling notebook.

The notebook is GENERATED: edit THIS builder and re-run it to refresh the .ipynb
(see colab_notebooks/_nb.py). It tests, on the existing trained-to-0 checkpoints
(checkpoints/kprop_checkpoints/kprop-zero_d3_w*_tol5_seed{3..6}, LOAD-ONLY), how the
joint cumulants of the collective coordinates

    S = sum_i X_i                  (all-ones / coherent projection, unnormalised)
    V = (1/sqrt n) sum_i X_i^2       (energy)

of a hidden POST-ReLU latent X in R^n scale with width n, and whether trained-to-0
correlation breaks the iid scaling. All heavy MC is cached to results/sv_cumulant_scaling/
and reloaded on re-run (the repo recycling rule).

Run:  python colab_notebooks/sv_cumulant_scaling/build_sv_cumulant_scaling_notebook.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =========================================================================== #
md(r"""# Cumulant scaling of the collective coordinates $S$ and $V$ on trained-to-0 MLPs

For a hidden **post-ReLU** latent $X\in\mathbb R^n$ (one per input $x\sim\mathcal N(0,I_n)$)
of a depth-3 square ReLU MLP trained to output $0$, define two scalar collective coordinates
(the **literal** definitions chosen for this study):

$$S=\sum_{i=1}^n X_i \qquad\qquad V=\frac{1}{\sqrt n}\sum_{i=1}^n X_i^2 .$$

We estimate the **joint cumulants** $\kappa_{a,b}=\operatorname{cum}(\underbrace{S,\dots,S}_{a},\underbrace{V,\dots,V}_{b})$
up to total order $a+b\le 4$ and ask two things.

**1. Scaling with width.** Under the **iid null** (the $n$ coordinates independent with the same
marginals), with these literal definitions

$$\kappa_{a,b}(S,V)=\Big(\tfrac{1}{\sqrt n}\Big)^{b}\sum_{i=1}^n \operatorname{cum}_{a,b}(X_i,X_i^2)\;\sim\; n^{\,1-b/2},$$

so the explicit $n$-power is $\boxed{1-b/2}$ — it depends only on the number of $V$'s, **not** on the
number of $S$'s, because $S$ is an *unnormalised* sum. (Your heuristic $n^{\,1-\#S/2-\#V}$ instead
corresponds to the *rescaled* coordinates $S=\tfrac1{\sqrt n}\sum X_i,\;V=\tfrac1n\sum X_i^2$; we
print **both** predicted slopes next to the measured one.)

**2. Correlation breaks the null.** The assumption-free reference is the **diagonal cumulant**
$\kappa_{a,b}^{\mathrm{diag}}$ — exactly the value $\kappa_{a,b}$ would take if the coordinates were
independent with the measured per-coordinate marginals. The **gap** $\kappa_{a,b}-\kappa_{a,b}^{\mathrm{diag}}$
is the contribution of inter-coordinate correlation. Trained-to-0 nets are expected to grow a rank-1
**all-ones / $-\mu$** mode (see the repo's Q2/Q4 findings), which inflates the *all-ones* cumulants
$\kappa_{a,0}$ (functions of $S$) far above the diagonal null while barely touching a random direction.

**Controls & honesty.**
* A matched **random-direction** projection $R=\sqrt n\,(u^\top X)$, $\lvert u\rvert=1$ (same vector
  norm as the all-ones direction behind $S$): iid $\Rightarrow \kappa_a(S)\approx\kappa_a(R)$; a rank-1
  all-ones spike $\Rightarrow \kappa_a(S)\gg\kappa_a(R)$.
* Sampling error on **every** cumulant is a delete-one-block **jackknife** SD, combined across the 4
  seeds. A cumulant is plotted **only if** $\lvert\kappa\rvert > z\cdot\mathrm{sd}$ (default $z=2$) — nothing
  inside the sampling noise is shown.

Estimators live in `analysis/Tools/cumulants_sv.py` (validated against analytic exponential/Poisson
cumulants, the Gaussian null, independence, and the iid-sum diagonal identity).
""")

# --------------------------------------------------------------------------- #
code(BOOTSTRAP_CELL)

# --------------------------------------------------------------------------- #
md(r"""## 1. Config — this notebook owns its knobs

Checkpoints are **loaded, never trained** (they already exist from the kprop scaling study).
Per-`(width, seed)` MC statistics are cached to `results/sv_cumulant_scaling/` and reloaded on
re-run, so the expensive forward passes happen once. Set `N_MC` smaller for a quick look.
""")

code(r"""
import os, json, math, time
import numpy as np
import matplotlib.pyplot as plt
import torch

import experiments as E
from model import MLP
from analysis.Tools.cumulants_sv import (
    reduce_sv, stream_collective_coordinates, AB_REPORT,
    predicted_slope_literal, predicted_slope_heuristic, fit_loglog_slope)

DEVICE      = E.DEVICE
MODEL_DTYPE = torch.float32

# --- the trained-to-0 checkpoint set (LOAD ONLY) ----------------------------
CKPT_DIR = "checkpoints/kprop_checkpoints"          # kprop-zero_d3_w*_tol5_seed{3..6}
PREFIX, DEPTH, TOL_TAG = "kprop-zero", 3, 5

# --- sweep -------------------------------------------------------------------
QUICK  = E.QUICK                                     # True on CPU-only -> small smoke sweep
WIDTHS = [16, 32, 64, 128] if QUICK else [16, 32, 64, 128, 256, 512, 1024, 1536, 2048]
SEEDS  = [3, 4, 5, 6]
LAYERS = [0, 1, 2]          # ALL three hidden blocks (h1, h2, h3); h3 feeds the readout
WHICH  = "post"             # POST-ReLU latents (the layer carrying the -mu / all-ones structure)
N_RAND = 3                  # matched random-direction controls per layer

# --- Monte-Carlo & statistics ------------------------------------------------
N_MC     = 300_000 if QUICK else 4_000_000          # MC inputs per (width, seed)
MC_BATCH = 8192                                      # forward batch (auto-shrunk at large width)
N_BLOCKS = 40                                        # jackknife blocks
Z_GATE   = 2.0                                       # plot kappa only if |kappa| > Z_GATE * sd

CACHE_DIR = "results/sv_cumulant_scaling/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"device={DEVICE} | widths={WIDTHS} | seeds={SEEDS} | layers(post-ReLU)={LAYERS}")
print(f"N_MC={N_MC:,} | jackknife blocks={N_BLOCKS} | resolution gate z={Z_GATE}")
print(f"checkpoints (load-only): {CKPT_DIR}/{PREFIX}_d{DEPTH}_w*_tol{TOL_TAG}_seed*  |  cache: {CACHE_DIR}")
""")

# --------------------------------------------------------------------------- #
md(r"""## 2. Estimator sanity check (no checkpoints needed)

Confirm on synthetic data that (a) **independent** coordinates land on the diagonal reference
(enhancement $\approx 1$, off-diagonal $\approx 0$, $S$-vs-random $\approx 1$), and (b) an injected
rank-1 **all-ones** mode is detected (enhancement $\gg 1$, $S$-vs-random $\gg 1$). This is the same
algebra `cumulants_sv.py` ships with its unit tests.
""")

code(r"""
rng = np.random.default_rng(0)
n0, Ns0 = 24, 400_000
U0 = rng.normal(size=(n0, 3)); U0 /= np.linalg.norm(U0, axis=0)

def _stats(Xc):
    S = Xc.sum(1); V = (Xc**2).sum(1)/np.sqrt(n0); proj = Xc @ U0
    cm = np.stack([(Xc**k).mean(0) for k in range(1, 9)])
    return reduce_sv(S, V, proj, cm, n0, n_blocks=40, z_gate=2.0)

Xi = np.abs(rng.normal(0.3, 1.0, size=(Ns0, n0)))                       # independent coords
c0 = rng.normal(0, 1.0, size=(Ns0, 1))
Xc = np.maximum(rng.normal(0, 1.0, size=(Ns0, n0)) + 1.0*c0, 0.0)       # rank-1 all-ones mode
ri, rc = _stats(Xi), _stats(Xc)
print("independent coords : enh_k2=%.3f  offdiag_frac=%+.3f  S/R_k2=%.3f   (expect ~1, ~0, ~1)"
      % (ri['enh_k2'], ri['twopoint_offdiag_frac'], ri['SvsR_k2']))
print("rank-1 all-ones    : enh_k2=%.2f  offdiag_frac=%+.3f  S/R_k2=%.2f   (expect >>1, ->1, >>1)"
      % (rc['enh_k2'], rc['twopoint_offdiag_frac'], rc['SvsR_k2']))
""")

# --------------------------------------------------------------------------- #
md(r"""## 3. Load checkpoints, stream the Monte-Carlo, reduce — with caching

For each `(width, seed)`: load the trained-to-0 model, stream `N_MC` Gaussian inputs through it,
collect the per-sample scalars $S,V$ and the random-direction projections for **every** hidden layer
in a single forward pass, accumulate per-coordinate moments $E[X_i^k]$ ($k=1..8$), and reduce to
cumulants + jackknife SDs + the diagonal reference. Results are cached per `(width, seed)`; a re-run
loads them instantly.
""")

code(r"""
def ckpt_for(w, s):
    return E.ckpt_path(CKPT_DIR, E.run_name(PREFIX, depth=DEPTH, width=w, tol=TOL_TAG, seed=s))

def cache_for(w, s):
    return os.path.join(CACHE_DIR, f"svcum_d{DEPTH}_w{w}_seed{s}_{WHICH}_N{N_MC}_nr{N_RAND}.json")

def get_or_compute(w, s):
    cp = cache_for(w, s)
    if os.path.exists(cp):                                  # recycle
        with open(cp) as f:
            return json.load(f), True
    path = ckpt_for(w, s)
    if not os.path.exists(path):
        print(f"   [skip] missing checkpoint {os.path.basename(path)}")
        return None, False
    model, payload = MLP.load(path, map_location=DEVICE)
    model = model.to(device=DEVICE, dtype=MODEL_DTYPE).eval()
    in_dim   = model.cfg.input_dim
    mc_batch = min(MC_BATCH, max(4096, (1 << 26) // max(in_dim, 1)))   # bound batch*n memory
    streamed = stream_collective_coordinates(
        model, in_dim, N_MC, layers=LAYERS, which=WHICH, n_rand=N_RAND,
        batch=mc_batch, device=DEVICE, dtype=MODEL_DTYPE, data_seed=1000 + s)
    per_layer = {}
    for ell in LAYERS:
        d = streamed[ell]
        per_layer[str(ell)] = reduce_sv(d["S"], d["V"], d["proj"], d["coord_moments"], d["n"],
                                        n_blocks=N_BLOCKS, z_gate=Z_GATE)
    per_layer["_meta"] = dict(width=w, seed=s, final_loss=float(E.final_loss(payload)), N_MC=N_MC)
    with open(cp, "w") as f:
        json.dump(per_layer, f)
    return per_layer, False

RESULTS = {}
t0 = time.time()
for w in WIDTHS:
    for s in SEEDS:
        r, loaded = get_or_compute(w, s)
        if r is not None:
            RESULTS[(w, s)] = r
            print(f"   w={w:5d} s={s} {'[cache]' if loaded else '[computed]'}"
                  f"  final_loss={r['_meta']['final_loss']:.1e}")
print(f"done in {time.time()-t0:.1f}s | {len(RESULTS)} (width,seed) cells")
""")

# --------------------------------------------------------------------------- #
md(r"""## 4. Aggregate over seeds

Per `(layer, width)`: the point estimate is the seed mean; the error combines the within-seed
jackknife SD with the seed-to-seed spread. A cumulant is **resolved** when $\lvert\text{mean}\rvert>z\cdot\text{sd}$.
""")

code(r"""
def aggregate(layer):
    out = {}
    for w in WIDTHS:
        seeds = [RESULTS[(w, s)][str(layer)] for s in SEEDS if (w, s) in RESULTS]
        if not seeds:
            continue
        Nseed = len(seeds)
        row = {"n": w, "Nseed": Nseed}
        for ab in AB_REPORT:
            key = f"{ab[0]}{ab[1]}"
            ks  = np.array([d[f"k_{key}"]  for d in seeds], float)
            sds = np.array([d[f"sd_{key}"] for d in seeds], float)
            mean    = float(ks.mean())
            within  = float(np.sqrt(np.sum(sds**2)) / Nseed)          # sampling error of the mean
            between = float(ks.std(ddof=1) / np.sqrt(Nseed)) if Nseed > 1 else 0.0
            tot     = float(np.sqrt(within**2 + between**2))
            row[f"k_{key}"]   = mean
            row[f"sd_{key}"]  = tot
            row[f"diag_{key}"] = float(np.mean([d[f"diag_{key}"] for d in seeds]))
            row[f"res_{key}"]  = bool(abs(mean) > Z_GATE * tot)
        for a in (2, 3, 4):
            for f in (f"enh_k{a}", f"SvsR_k{a}", f"R_k{a}"):
                row[f] = float(np.mean([d[f] for d in seeds]))
        for kk in ("twopoint_offdiag_frac", "twopoint_n_times_mean_cov", "twopoint_var_S",
                   "twopoint_diag_var", "marg_skew_med", "marg_exkurt_med", "dead_frac"):
            row[kk] = float(np.mean([d[kk] for d in seeds]))
        out[w] = row
    return out

AGG = {ell: aggregate(ell) for ell in LAYERS}
LCOL = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}                 # per-layer colour
LNAME = {ell: f"h{ell+1}{' (readout-feeding)' if ell == max(LAYERS) else ''}" for ell in LAYERS}
print("aggregated layers:", {ell: list(AGG[ell].keys()) for ell in LAYERS})
""")

# --------------------------------------------------------------------------- #
md(r"""## 5. The headline — width scaling of $\lvert\kappa_{a,b}\rvert$

One panel per cumulant order $(a,b)$. **Markers** = measured (filled = resolved at $z{=}2$, hollow =
gated out by the sampling SD and *not* trusted); $\triangle$ if $\kappa>0$, $\triangledown$ if $\kappa<0$.
**Dashed** = the diagonal / iid reference (the assumption-free null). The fitted measured slope and the
two predicted slopes (literal $1-b/2$; heuristic $1-a/2-b$) are printed per layer. **A measured curve
peeling *above* its dashed null = inter-coordinate correlation** (expected for the all-ones $\kappa_{a,0}$).
""")

code(r"""
def _xy(layer, key, want_resolved):
    ws = sorted(AGG[layer].keys()); xs, ys, sg = [], [], []
    for w in ws:
        r = AGG[layer][w]
        if r[f"res_{key}"] != want_resolved:
            continue
        v = r[f"k_{key}"]
        if v == 0:
            continue
        xs.append(w); ys.append(abs(v)); sg.append(np.sign(v))
    return np.array(xs, float), np.array(ys, float), np.array(sg, float)

def _diag_xy(layer, key):
    ws = sorted(AGG[layer].keys()); xs, ys = [], []
    for w in ws:
        v = AGG[layer][w][f"diag_{key}"]
        if v != 0:
            xs.append(w); ys.append(abs(v))
    return np.array(xs, float), np.array(ys, float)

fig, axes = plt.subplots(4, 3, figsize=(16, 18)); axes = axes.ravel()
for idx, ab in enumerate(AB_REPORT):
    a, b = ab; key = f"{a}{b}"; ax = axes[idx]
    txt = []
    for ell in LAYERS:
        col = LCOL[ell]
        # measured, split by sign and resolution
        for resolved, ms, fill in [(True, 90, True), (False, 55, False)]:
            xs, ys, sg = _xy(ell, key, resolved)
            for mk, sel in [("^", sg > 0), ("v", sg < 0)]:
                if sel.any():
                    ax.scatter(xs[sel], ys[sel], s=ms, marker=mk,
                               facecolors=(col if fill else "none"), edgecolors=col,
                               linewidths=1.4, zorder=3)
        # measured connecting line over RESOLVED points only
        xs, ys, sg = _xy(ell, key, True)
        if xs.size:
            o = np.argsort(xs); ax.plot(xs[o], ys[o], "-", color=col, lw=1.2, alpha=0.7,
                                        label=f"{LNAME[ell]} (meas)")
            sl = fit_loglog_slope(xs, ys)
        else:
            sl = float("nan")
        # diagonal / iid reference
        dx, dy = _diag_xy(ell, key)
        if dx.size:
            o = np.argsort(dx); ax.plot(dx[o], dy[o], "--", color=col, lw=1.3, alpha=0.9)
        txt.append(f"{LNAME[ell].split()[0]}: meas {sl:+.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"$\\kappa_{{{a},{b}}}$  (#S={a}, #V={b})", fontsize=12)
    ax.set_xlabel("width $n$"); ax.set_ylabel(f"$|\\kappa_{{{a},{b}}}|$")
    pl, ph = predicted_slope_literal(a, b), predicted_slope_heuristic(a, b)
    ax.text(0.03, 0.03, "predicted slope\n  literal $1-b/2$ = %+.2f\n  heuristic = %+.2f\n%s"
            % (pl, ph, "\n".join(txt)), transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    if idx == 0:
        ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Width scaling of joint cumulants $|\\kappa_{a,b}(S,V)|$  "
             "(solid+markers = measured, filled = resolved; dashed = iid/diagonal null)",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985]); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""## 6. Correlation enhancement — measured $/$ diagonal for the all-ones cumulants $\kappa_{a,0}$

Ratio $=1$ (grey) means iid; a ratio **growing with $n$** is the rank-1 all-ones mode building up.
A democratic correlation $\operatorname{Cov}(X_i,X_j)=O(1/n)$ that is *coherent* on the all-ones
direction makes $\kappa_{2,0}$ enhancement scale $\propto n$; higher orders grow even faster.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for j, a in enumerate((2, 3, 4)):
    ax = axes[j]
    for ell in LAYERS:
        ws = sorted(AGG[ell].keys())
        y  = [AGG[ell][w][f"enh_k{a}"] for w in ws]
        ax.plot(ws, np.abs(y), "o-", color=LCOL[ell], label=LNAME[ell])
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width $n$"); ax.set_ylabel(f"$|\\kappa_{{{a},0}}|$ measured / diagonal")
    ax.set_title(f"enhancement of $\\kappa_{{{a},0}}$ (S only)")
    if j == 0:
        ax.legend(fontsize=9)
fig.suptitle("Inter-coordinate correlation: how far the all-ones cumulants sit above the iid null", y=1.02)
fig.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""## 7. Is the coherence specifically the all-ones mode? $\kappa_a(S)$ vs a random direction

$S$ and $R=\sqrt n\,(u^\top X)$ are projections onto **equal-norm** directions. For iid coordinates
the ratio is $\approx 1$; a ratio $\gg 1$ that grows with $n$ says the excess cumulant lives in the
all-ones direction specifically — the signature of the trained $-\mu$ spike, not generic heavy tails.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for j, a in enumerate((2, 3, 4)):
    ax = axes[j]
    for ell in LAYERS:
        ws = sorted(AGG[ell].keys())
        y  = [AGG[ell][w][f"SvsR_k{a}"] for w in ws]
        ax.plot(ws, np.abs(y), "s-", color=LCOL[ell], label=LNAME[ell])
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("width $n$"); ax.set_ylabel(f"$|\\kappa_{a}(S)| / |\\kappa_{a}(R_{{rand}})|$")
    ax.set_title(f"all-ones vs random direction, order {a}")
    if j == 0:
        ax.legend(fontsize=9)
fig.suptitle("All-ones direction vs a matched random direction (ratio $\\gg1$ ⇒ coherence is the all-ones mode)", y=1.02)
fig.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""## 8. Higher cumulants *between* the $X_i$ — the 2-point structure

$\operatorname{Var}(S)=\sum_i\operatorname{Var}(X_i)+\sum_{i\ne j}\operatorname{Cov}(X_i,X_j)$. The
**off-diagonal fraction** is the share of $\operatorname{Var}(S)$ coming from cross-coordinate
correlation (0 = iid, $\to 1$ = fully coherent). $(n-1)\,\overline{\operatorname{Cov}}$ tests whether
the correlation is **democratic** $O(1/n)$: if it is, this quantity is roughly **flat** in $n$ while
$\operatorname{Var}(S)/\sum\operatorname{Var}(X_i)$ grows $\propto n$.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ell in LAYERS:
    ws = sorted(AGG[ell].keys()); col = LCOL[ell]
    axes[0].plot(ws, [AGG[ell][w]["twopoint_offdiag_frac"] for w in ws], "o-", color=col, label=LNAME[ell])
    axes[1].plot(ws, [abs(AGG[ell][w]["twopoint_n_times_mean_cov"]) for w in ws], "o-", color=col)
    ratio = [AGG[ell][w]["twopoint_var_S"]/max(AGG[ell][w]["twopoint_diag_var"], 1e-30) for w in ws]
    axes[2].plot(ws, np.abs(ratio), "o-", color=col)
axes[0].axhline(0, color="0.5", ls=":"); axes[0].set_ylabel("off-diagonal fraction of Var(S)")
axes[0].set_title("share of Var(S) from correlation"); axes[0].set_xscale("log"); axes[0].legend(fontsize=9)
axes[1].set_ylabel(r"$|(n-1)\,\overline{Cov}|$"); axes[1].set_title("democratic-correlation test (flat ⇒ $O(1/n)$)")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[2].set_ylabel("Var(S) / sum Var(X_i)"); axes[2].set_title("coherent enhancement (∝ n ⇒ rank-1)")
axes[2].set_xscale("log"); axes[2].set_yscale("log")
for ax in axes:
    ax.set_xlabel("width $n$")
fig.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""## 9. Per-coordinate (marginal) non-Gaussianity

The individual $X_i$ are post-ReLU, so they are one-sided and non-Gaussian by construction. Median
skewness, excess kurtosis, and the dead-ReLU fraction vs width characterise how the *marginals* drift —
the diagonal reference already folds these in, so the §5–§8 gaps are correlation **beyond** marginal
non-Gaussianity.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ell in LAYERS:
    ws = sorted(AGG[ell].keys()); col = LCOL[ell]
    axes[0].plot(ws, [AGG[ell][w]["marg_skew_med"]   for w in ws], "o-", color=col, label=LNAME[ell])
    axes[1].plot(ws, [AGG[ell][w]["marg_exkurt_med"] for w in ws], "o-", color=col)
    axes[2].plot(ws, [AGG[ell][w]["dead_frac"]       for w in ws], "o-", color=col)
axes[0].set_title("median skewness of $X_i$"); axes[0].legend(fontsize=9)
axes[1].set_title("median excess kurtosis of $X_i$")
axes[2].set_title("dead-ReLU fraction (Var$(X_i)\\approx0$)")
for ax, yl in zip(axes, ("skewness", "excess kurtosis", "dead fraction")):
    ax.set_xlabel("width $n$"); ax.set_ylabel(yl); ax.set_xscale("log")
fig.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""## 10. Slope table & resolution summary

For each $(a,b)$ and layer: the fitted measured log-log slope (over **resolved** widths only), the
diagonal-reference slope, and the two predicted slopes. `nres` is how many widths were resolved at
$z={}$`Z_GATE` (unresolved widths are excluded from the fit and not plotted as filled markers).
""")

code(r"""
def slope_over_resolved(layer, key, which="k"):
    ws = sorted(AGG[layer].keys()); xs, ys = [], []
    for w in ws:
        r = AGG[layer][w]
        if which == "k" and not r[f"res_{key}"]:
            continue
        v = r[f"{which}_{key}"]
        if v != 0:
            xs.append(w); ys.append(abs(v))
    return fit_loglog_slope(xs, ys), len(xs)

print(f"{'(a,b)':7s} {'layer':6s} {'meas slope':>10s} {'diag slope':>10s} "
      f"{'lit 1-b/2':>9s} {'heur':>6s} {'nres':>5s}")
print("-" * 60)
for ab in AB_REPORT:
    a, b = ab; key = f"{a}{b}"
    for ell in LAYERS:
        ms, nres = slope_over_resolved(ell, key, "k")
        ds, _    = slope_over_resolved(ell, key, "diag")
        print(f"({a},{b})   {LNAME[ell].split()[0]:5s} {ms:>10.2f} {ds:>10.2f} "
              f"{predicted_slope_literal(a,b):>9.2f} {predicted_slope_heuristic(a,b):>6.2f} {nres:>5d}")
    print()

# which (a,b, n) cells were gated out (not resolved)
print("Gated-out (unresolved at z=%.1f) cells [layer: (a,b)@width ...]:" % Z_GATE)
for ell in LAYERS:
    gated = [f"({a},{b})@{w}" for w in sorted(AGG[ell].keys())
             for (a, b) in AB_REPORT if not AGG[ell][w][f"res_{a}{b}"]]
    print(f"  {LNAME[ell].split()[0]}: {', '.join(gated) if gated else 'none — all resolved'}")
""")

# --------------------------------------------------------------------------- #
md(r"""## 11. How to read this

* **Pure-$V$ cumulants $\kappa_{0,b}$** should track the dashed iid null with slope $\approx 1-b/2$ —
  $V$ is a sum of squares with only weak cross-coupling, so the diagonal reference is a good model.
* **Pure-$S$ cumulants $\kappa_{a,0}$** are the test. If trained-to-0 has built the rank-1 all-ones
  $-\mu$ mode, the measured $\kappa_{a,0}$ peel **above** the dashed null (§5), the enhancement ratio
  **grows with $n$** (§6), the all-ones$/$random ratio is $\gg 1$ (§7), and the off-diagonal fraction
  of $\operatorname{Var}(S)\to 1$ with $(n-1)\overline{\operatorname{Cov}}$ roughly flat (§8, democratic
  $O(1/n)$ coherence). The measured slope then **exceeds** $1-b/2$ — the iid scaling is broken exactly
  as expected. The deepest (readout-feeding) layer $h3$ should show this most strongly.
* **Mixed $\kappa_{a,b}$** interpolate; a non-zero, resolved, growing $\kappa_{2,1},\kappa_{1,2}$ beyond
  the diagonal null means $S$ and $V$ fluctuations are coupled through the same coherent mode.
* Anything plotted **hollow** is inside the sampling noise ($\lvert\kappa\rvert<2\,\mathrm{sd}$) — don't
  read a slope into it. Raise `N_MC` to resolve the 4th-order tails, or lower the widths considered.

To probe a single layer / pre-activations / a different gate, edit the §1 knobs (`WHICH="pre"`,
`LAYERS`, `Z_GATE`, `N_MC`) and re-run — cached cells reload, only new settings recompute.
""")

# --------------------------------------------------------------------------- #
out_path = os.path.join(os.path.dirname(__file__), "sv_cumulant_scaling_colab.ipynb")
nb.save(out_path)

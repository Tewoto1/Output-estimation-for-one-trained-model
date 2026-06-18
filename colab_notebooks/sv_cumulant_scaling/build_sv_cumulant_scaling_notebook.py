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

For a hidden latent $X\in\mathbb R^n$ — **both pre-ReLU ($Wh$) and post-ReLU are analysed**, one per
input $x\sim\mathcal N(0,I_n)$ — of a depth-3 square ReLU MLP trained to output $0$, define two scalar collective coordinates
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

# Robust import of the cumulant tools. Preferred path is the package
# (analysis/Tools/__init__.py must use a RELATIVE `from . import cumulants_sv`,
# never a bare `import cumulants_sv`, or the whole analysis package fails to
# import). Fall back to loading the module directly by file path so a stale
# clone / partial checkout still runs.
try:
    from analysis.Tools.cumulants_sv import (
        reduce_sv, stream_collective_coordinates, AB_REPORT,
        predicted_slope_literal, predicted_slope_heuristic, fit_loglog_slope)
except Exception as _e:
    import importlib.util as _ilu
    _csv_path = os.path.join(REPO, "analysis", "Tools", "cumulants_sv.py")
    _spec = _ilu.spec_from_file_location("cumulants_sv", _csv_path)
    _csv = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_csv)
    reduce_sv = _csv.reduce_sv
    stream_collective_coordinates = _csv.stream_collective_coordinates
    AB_REPORT = _csv.AB_REPORT
    predicted_slope_literal = _csv.predicted_slope_literal
    predicted_slope_heuristic = _csv.predicted_slope_heuristic
    fit_loglog_slope = _csv.fit_loglog_slope
    print("cumulants_sv: loaded directly by file path (package import failed:", repr(_e), ")")

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
WHICHS = ["post", "pre"]    # latents to analyse: POST-ReLU and PRE-ReLU (Wh), both from ONE forward
                            # pass. "post" carries the -mu/all-ones structure; "pre" is the
                            # near-Gaussian input to each ReLU and the layer the shift acts on.
N_RAND = 3                  # matched random-direction controls per layer

# --- Monte-Carlo & statistics ------------------------------------------------
N_MC     = 300_000 if QUICK else 4_000_000          # MC inputs per (width, seed)
MC_BATCH = 8192                                      # forward batch (auto-shrunk at large width)
N_BLOCKS = 40                                        # jackknife blocks
Z_GATE   = 2.0                                       # plot kappa only if |kappa| > Z_GATE * sd

CACHE_DIR = "results/sv_cumulant_scaling/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"device={DEVICE} | widths={WIDTHS} | seeds={SEEDS} | layers={LAYERS} | activations={WHICHS}")
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

WHICH_TAG = "-".join(WHICHS)

def cache_for(w, s):
    return os.path.join(CACHE_DIR, f"svcum_d{DEPTH}_w{w}_seed{s}_{WHICH_TAG}_N{N_MC}_nr{N_RAND}.json")

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
    # ONE forward pass yields BOTH pre- and post-activation collective coordinates per layer
    streamed = stream_collective_coordinates(
        model, in_dim, N_MC, layers=LAYERS, whichs=WHICHS, n_rand=N_RAND,
        batch=mc_batch, device=DEVICE, dtype=MODEL_DTYPE, data_seed=1000 + s)
    rec = {}
    for which in WHICHS:
        for ell in LAYERS:
            d = streamed[which][ell]
            rec[f"{which}|{ell}"] = reduce_sv(d["S"], d["V"], d["proj"], d["coord_moments"], d["n"],
                                              n_blocks=N_BLOCKS, z_gate=Z_GATE)
    rec["_meta"] = dict(width=w, seed=s, final_loss=float(E.final_loss(payload)), N_MC=N_MC)
    with open(cp, "w") as f:
        json.dump(rec, f)
    return rec, False

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
def aggregate(which, layer):
    out = {}
    for w in WIDTHS:
        seeds = [RESULTS[(w, s)][f"{which}|{layer}"] for s in SEEDS
                 if (w, s) in RESULTS and f"{which}|{layer}" in RESULTS[(w, s)]]
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

AGG = {(which, ell): aggregate(which, ell) for which in WHICHS for ell in LAYERS}
LCOL = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}                 # per-layer colour
LNAME = {ell: f"h{ell+1}{' (readout-feeding)' if ell == max(LAYERS) else ''}" for ell in LAYERS}
WHLABEL = {"post": "post-ReLU", "pre": "pre-ReLU (Wh)"}
print("aggregated (activation, layer) -> #widths:",
      {f"{which}|h{ell+1}": len(AGG[(which, ell)]) for which in WHICHS for ell in LAYERS})
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
def _xy(which, layer, key, want_resolved):
    agg = AGG[(which, layer)]; ws = sorted(agg.keys()); xs, ys, sg = [], [], []
    for w in ws:
        r = agg[w]
        if r[f"res_{key}"] != want_resolved:
            continue
        v = r[f"k_{key}"]
        if v == 0:
            continue
        xs.append(w); ys.append(abs(v)); sg.append(np.sign(v))
    return np.array(xs, float), np.array(ys, float), np.array(sg, float)

def _diag_xy(which, layer, key):
    agg = AGG[(which, layer)]; ws = sorted(agg.keys()); xs, ys = [], []
    for w in ws:
        v = agg[w][f"diag_{key}"]
        if v != 0:
            xs.append(w); ys.append(abs(v))
    return np.array(xs, float), np.array(ys, float)

def plot_scaling(which):
    fig, axes = plt.subplots(4, 3, figsize=(16, 18)); axes = axes.ravel()
    for idx, ab in enumerate(AB_REPORT):
        a, b = ab; key = f"{a}{b}"; ax = axes[idx]
        txt = []
        for ell in LAYERS:
            col = LCOL[ell]
            for resolved, ms, fill in [(True, 90, True), (False, 55, False)]:
                xs, ys, sg = _xy(which, ell, key, resolved)
                for mk, sel in [("^", sg > 0), ("v", sg < 0)]:
                    if sel.any():
                        ax.scatter(xs[sel], ys[sel], s=ms, marker=mk,
                                   facecolors=(col if fill else "none"), edgecolors=col,
                                   linewidths=1.4, zorder=3)
            xs, ys, sg = _xy(which, ell, key, True)
            if xs.size:
                o = np.argsort(xs); ax.plot(xs[o], ys[o], "-", color=col, lw=1.2, alpha=0.7,
                                            label=f"{LNAME[ell]} (meas)")
                sl = fit_loglog_slope(xs, ys)
            else:
                sl = float("nan")
            dx, dy = _diag_xy(which, ell, key)
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
    fig.suptitle(f"[{WHLABEL[which]}]  width scaling of $|\\kappa_{{a,b}}(S,V)|$  "
                 "(markers = measured, filled = resolved; dashed = iid/diagonal null)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985]); plt.show()

for _w in WHICHS:
    plot_scaling(_w)
""")

# --------------------------------------------------------------------------- #
md(r"""## 5b. Pre- vs post-activation scale, side by side

The direct comparison you want: $|\kappa_{a,b}|$ vs width for **pre-ReLU** ($Wh$, dashed/squares) and
**post-ReLU** (solid/circles), coloured by layer, for the pure-$S$ ($\kappa_{2,0},\kappa_{3,0},\kappa_{4,0}$)
and pure-$V$ ($\kappa_{0,2},\kappa_{0,3}$) cumulants (resolved points only). Pre-activations are the
near-Gaussian sums feeding each ReLU and are exactly where the all-ones shift acts; post-activations are
the rectified outputs. Reading the two together shows how much of the cumulant scale is created by the
ReLU vs already present in the pre-activation.
""")

code(r"""
from matplotlib.lines import Line2D

def _meas_xy(which, ell, key):
    agg = AGG[(which, ell)]; ws = sorted(agg.keys()); xs, ys = [], []
    for w in ws:
        r = agg[w]
        if not r[f"res_{key}"]:
            continue
        v = r[f"k_{key}"]
        if v != 0:
            xs.append(w); ys.append(abs(v))
    return np.array(xs, float), np.array(ys, float)

WHSTYLE = {"post": "-", "pre": "--"}; WHMK = {"post": "o", "pre": "s"}
panels = [(2, 0), (3, 0), (4, 0), (0, 2), (0, 3)]
fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.ravel()
for idx, (a, b) in enumerate(panels):
    ax = axes[idx]; key = f"{a}{b}"
    for ell in LAYERS:
        for which in WHICHS:
            xs, ys = _meas_xy(which, ell, key)
            if xs.size:
                o = np.argsort(xs)
                ax.plot(xs[o], ys[o], WHSTYLE.get(which, "-"), marker=WHMK.get(which, "o"),
                        color=LCOL[ell], lw=1.3, ms=5, alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"$|\\kappa_{{{a},{b}}}|$  (#S={a}, #V={b})")
    ax.set_xlabel("width $n$"); ax.set_ylabel(f"$|\\kappa_{{{a},{b}}}|$")
axes[-1].axis("off")
handles = [Line2D([0], [0], color=LCOL[ell], lw=2, label=LNAME[ell]) for ell in LAYERS] + \
          [Line2D([0], [0], color="0.3", ls=WHSTYLE[w], marker=WHMK[w], label=WHLABEL[w]) for w in WHICHS]
axes[-1].legend(handles=handles, loc="center", fontsize=11, title="colour = layer, style = activation")
fig.suptitle("Pre- vs post-activation cumulant scale (resolved points only)", y=1.0)
fig.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""## 6. Correlation enhancement — measured $/$ diagonal for the all-ones cumulants $\kappa_{a,0}$

Ratio $=1$ (grey) means iid; a ratio **growing with $n$** is the rank-1 all-ones mode building up.
A democratic correlation $\operatorname{Cov}(X_i,X_j)=O(1/n)$ that is *coherent* on the all-ones
direction makes $\kappa_{2,0}$ enhancement scale $\propto n$; higher orders grow even faster.
""")

code(r"""
def plot_enh(which):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for j, a in enumerate((2, 3, 4)):
        ax = axes[j]
        for ell in LAYERS:
            ws = sorted(AGG[(which, ell)].keys())
            y  = [AGG[(which, ell)][w][f"enh_k{a}"] for w in ws]
            ax.plot(ws, np.abs(y), "o-", color=LCOL[ell], label=LNAME[ell])
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("width $n$"); ax.set_ylabel(f"$|\\kappa_{{{a},0}}|$ measured / diagonal")
        ax.set_title(f"enhancement of $\\kappa_{{{a},0}}$ (S only)")
        if j == 0:
            ax.legend(fontsize=9)
    fig.suptitle(f"[{WHLABEL[which]}] inter-coordinate correlation: all-ones cumulants above the iid null", y=1.02)
    fig.tight_layout(); plt.show()

for _w in WHICHS:
    plot_enh(_w)
""")

# --------------------------------------------------------------------------- #
md(r"""## 7. Is the coherence specifically the all-ones mode? $\kappa_a(S)$ vs a random direction

$S$ and $R=\sqrt n\,(u^\top X)$ are projections onto **equal-norm** directions. For iid coordinates
the ratio is $\approx 1$; a ratio $\gg 1$ that grows with $n$ says the excess cumulant lives in the
all-ones direction specifically — the signature of the trained $-\mu$ spike, not generic heavy tails.
""")

code(r"""
def plot_svr(which):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for j, a in enumerate((2, 3, 4)):
        ax = axes[j]
        for ell in LAYERS:
            ws = sorted(AGG[(which, ell)].keys())
            y  = [AGG[(which, ell)][w][f"SvsR_k{a}"] for w in ws]
            ax.plot(ws, np.abs(y), "s-", color=LCOL[ell], label=LNAME[ell])
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("width $n$"); ax.set_ylabel(f"$|\\kappa_{a}(S)| / |\\kappa_{a}(R_{{rand}})|$")
        ax.set_title(f"all-ones vs random direction, order {a}")
        if j == 0:
            ax.legend(fontsize=9)
    fig.suptitle(f"[{WHLABEL[which]}] all-ones vs matched random direction (ratio $\\gg1$ ⇒ all-ones mode)", y=1.02)
    fig.tight_layout(); plt.show()

for _w in WHICHS:
    plot_svr(_w)
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
def plot_twopoint(which):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ell in LAYERS:
        ws = sorted(AGG[(which, ell)].keys()); col = LCOL[ell]; agg = AGG[(which, ell)]
        axes[0].plot(ws, [agg[w]["twopoint_offdiag_frac"] for w in ws], "o-", color=col, label=LNAME[ell])
        axes[1].plot(ws, [abs(agg[w]["twopoint_n_times_mean_cov"]) for w in ws], "o-", color=col)
        ratio = [agg[w]["twopoint_var_S"]/max(agg[w]["twopoint_diag_var"], 1e-30) for w in ws]
        axes[2].plot(ws, np.abs(ratio), "o-", color=col)
    axes[0].axhline(0, color="0.5", ls=":"); axes[0].set_ylabel("off-diagonal fraction of Var(S)")
    axes[0].set_title("share of Var(S) from correlation"); axes[0].set_xscale("log"); axes[0].legend(fontsize=9)
    axes[1].set_ylabel(r"$|(n-1)\,\overline{Cov}|$"); axes[1].set_title("democratic-correlation test (flat ⇒ $O(1/n)$)")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[2].set_ylabel("Var(S) / sum Var(X_i)"); axes[2].set_title("coherent enhancement (∝ n ⇒ rank-1)")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("width $n$")
    fig.suptitle(f"[{WHLABEL[which]}] 2-point structure of Var(S)", y=1.03)
    fig.tight_layout(); plt.show()

for _w in WHICHS:
    plot_twopoint(_w)
""")

# --------------------------------------------------------------------------- #
md(r"""## 9. Per-coordinate (marginal) non-Gaussianity

Median skewness, excess kurtosis, and the near-degenerate ($\mathrm{Var}(X_i)\approx0$) fraction vs
width. **Post-ReLU** coordinates are one-sided → positive skew, non-zero excess kurtosis; **pre-ReLU**
coordinates are sums ($Wh$) and should be much closer to Gaussian (skew, excess kurtosis $\approx0$) —
a useful contrast. The diagonal reference already folds these marginals in, so the §5–§8 gaps are
correlation **beyond** marginal non-Gaussianity.
""")

code(r"""
def plot_marg(which):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ell in LAYERS:
        ws = sorted(AGG[(which, ell)].keys()); col = LCOL[ell]; agg = AGG[(which, ell)]
        axes[0].plot(ws, [agg[w]["marg_skew_med"]   for w in ws], "o-", color=col, label=LNAME[ell])
        axes[1].plot(ws, [agg[w]["marg_exkurt_med"] for w in ws], "o-", color=col)
        axes[2].plot(ws, [agg[w]["dead_frac"]       for w in ws], "o-", color=col)
    axes[0].set_title("median skewness of $X_i$"); axes[0].legend(fontsize=9)
    axes[1].set_title("median excess kurtosis of $X_i$")
    axes[2].set_title(r"near-degenerate fraction (Var$(X_i)\approx0$)")
    for ax, yl in zip(axes, ("skewness", "excess kurtosis", "degenerate fraction")):
        ax.set_xlabel("width $n$"); ax.set_ylabel(yl); ax.set_xscale("log")
    fig.suptitle(f"[{WHLABEL[which]}] per-coordinate marginal shape", y=1.03)
    fig.tight_layout(); plt.show()

for _w in WHICHS:
    plot_marg(_w)
""")

# --------------------------------------------------------------------------- #
md(r"""## 10. Slope table & resolution summary

For each $(a,b)$ and layer: the fitted measured log-log slope (over **resolved** widths only), the
diagonal-reference slope, and the two predicted slopes. `nres` is how many widths were resolved at
$z={}$`Z_GATE` (unresolved widths are excluded from the fit and not plotted as filled markers).
""")

code(r"""
def slope_over_resolved(which, layer, key, field="k"):
    agg = AGG[(which, layer)]; ws = sorted(agg.keys()); xs, ys = [], []
    for w in ws:
        r = agg[w]
        if field == "k" and not r[f"res_{key}"]:
            continue
        v = r[f"{field}_{key}"]
        if v != 0:
            xs.append(w); ys.append(abs(v))
    return fit_loglog_slope(xs, ys), len(xs)

def slope_table(which):
    print(f"================  {WHLABEL[which]}  ================")
    print(f"{'(a,b)':7s} {'layer':6s} {'meas slope':>10s} {'diag slope':>10s} "
          f"{'lit 1-b/2':>9s} {'heur':>6s} {'nres':>5s}")
    print("-" * 60)
    for ab in AB_REPORT:
        a, b = ab; key = f"{a}{b}"
        for ell in LAYERS:
            ms, nres = slope_over_resolved(which, ell, key, "k")
            ds, _    = slope_over_resolved(which, ell, key, "diag")
            print(f"({a},{b})   {LNAME[ell].split()[0]:5s} {ms:>10.2f} {ds:>10.2f} "
                  f"{predicted_slope_literal(a,b):>9.2f} {predicted_slope_heuristic(a,b):>6.2f} {nres:>5d}")
        print()
    print("Gated-out (unresolved at z=%.1f) [layer: (a,b)@width ...]:" % Z_GATE)
    for ell in LAYERS:
        agg = AGG[(which, ell)]
        gated = [f"({a},{b})@{w}" for w in sorted(agg.keys())
                 for (a, b) in AB_REPORT if not agg[w][f"res_{a}{b}"]]
        print(f"  {LNAME[ell].split()[0]}: {', '.join(gated) if gated else 'none — all resolved'}")
    print()

for _w in WHICHS:
    slope_table(_w)
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
* **Pre- vs post-activation (§5b and the per-activation panels).** Pre-activations $Wh$ are the
  near-Gaussian sums feeding each ReLU and are exactly where the all-ones shift is injected, so the
  coherent $\kappa_{a,0}$ enhancement is typically *already present* (often cleaner) in the pre-acts;
  the ReLU then rectifies it into the post-acts. Pre-act marginals are near-Gaussian (§9 skew/kurt
  $\approx0$) while post-acts are one-sided — yet both share the same all-ones correlation, which is the
  point: the coherence is a weight-geometry effect, not a ReLU artifact.
* Anything plotted **hollow** is inside the sampling noise ($\lvert\kappa\rvert<2\,\mathrm{sd}$) — don't
  read a slope into it. Raise `N_MC` to resolve the 4th-order tails, or lower the widths considered.

To analyse only one activation / a different gate, edit the §1 knobs (`WHICHS=["post"]` or `["pre"]`,
`LAYERS`, `Z_GATE`, `N_MC`) and re-run — cached cells reload, only new settings recompute.
""")

# --------------------------------------------------------------------------- #
out_path = os.path.join(os.path.dirname(__file__), "sv_cumulant_scaling_colab.ipynb")
nb.save(out_path)

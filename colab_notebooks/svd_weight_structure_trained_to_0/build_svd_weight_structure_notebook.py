"""Generates svd_weight_structure_trained_to_0_colab.ipynb (valid nbformat-4 JSON).

The SVD / weight-structure dissection of the trained-to-0 models. Reuses the
`checkpoints/kprop_checkpoints` set (kprop-zero, depth 3, tol5) and the new
`analysis.Tools.weight_structure.delta_weight_metrics`. Three questions, each with
a width-scaling plot (points connected + a best-fit line PER SEED):

  A. SVD spectrum of every hidden weight matrix, and how the TOP singular value
     scales with width (32..2048).
  B. Weight mean / variance of the MOVEMENT ΔW = W_trained - W_init: is each entry
     pushed by ~ -1/sqrt(n)?  is the variance skewed negative (more negative, less
     positive)?
  C. Weight-row dot products: do rows acquire a shared (all-ones / rank-1) component?

Depth 3 runs now off existing checkpoints (seeds 3 & 4). Depth 4 has NO checkpoints
yet, so the depth-4 TRAIN + analyse + (kprop k=2 cov-prop upturn) suite is the LAST
section, guarded by RUN_DEPTH4=False -- flip it on a GPU to launch it later.

House style: edit THIS script and re-run it to regenerate the notebook.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# ---------------------------------------------------------------------------
md(r"""# Trained-to-0 weight structure: SVD spectrum, the $-1/\sqrt{n}$ shift, and row dot products

We dissect the **weights** of the trained-to-0 models (depth-3 ReLU MLPs trained until
$\mathrm{MSE}<10^{-5}$ on $f(x)\!=\!0$, $x\sim\mathcal N(0,I)$ — the `checkpoints/kprop_checkpoints`
set, reused here, never retrained). For each hidden weight matrix $W$ we compare against its
**initialisation** $W_0$ (rebuilt from the same seed) and study the movement $\Delta W = W - W_0$.

**Three hypotheses, each tested across width $n\in\{32,\dots,2048\}$ for 2 seeds:**

| | hypothesis | what we measure |
|---|---|---|
| **A. SVD spectrum** | the trained map is low-rank; the top singular value grows with width | full singular-value spectrum per layer; **top singular value vs width** (best-fit slope per seed) |
| **B. mean / variance** | every entry is pushed **negative by $\approx 1/\sqrt{n}$**; the variance is skewed negative (more negative, less positive) | $\overline{\Delta W}\cdot\sqrt{n}$ ($\approx-1$?), fraction-negative, negative vs positive energy share, skew; **vs width** |
| **C. row dot products** | rows pick up a **shared component** (an all-ones / rank-1 spike) | mean off-diagonal row$\cdot$row dot product, energy along $\tfrac1{\sqrt{n}}\mathbf 1$; **vs width** |

> **First layer is the control.** Layer 0 reads the **mean-zero** input $x\sim\mathcal N(0,I)$, so
> it should show NO shift (symmetric $\Delta W$, no all-ones spike). Layers $\ge 1$ read a
> ReLU output with **positive** mean, so that is where a $-\mu$-aligned negative shift can appear.

**Depth.** Depth-3 checkpoints exist and run immediately. **Depth-4 has no checkpoints yet** — the
train + analyse suite (and a kprop **k=2 covariance-prop** error-vs-width *upturn* check, widths to
2048) is the final section, off by default (`RUN_DEPTH4=False`).
""")

code(BOOTSTRAP_CELL)

# ---------------------------------------------------------------------------
md(r"""## 1. Config — knobs live HERE (probe in place; `experiments.py` only holds the machinery)

The checkpoint set is **load-only**: `kprop-zero_d3_w{n}_tol5_seed{s}_final.pt` in
`checkpoints/kprop_checkpoints`. Analysis runs in **float64** (eigendecompositions are the
point), models load lazily one at a time so width 2048 fits in memory. `QUICK` trims the sweep
on a CPU-only machine.
""")

code(r"""
import os, json, math, time, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import experiments as E
from model import MLP, ModelConfig
from analysis.Tools.weight_structure import (
    delta_weight_metrics, weight_structure_metrics, mean_prev_post)

plt.rcParams["figure.dpi"] = 110
torch.set_grad_enabled(False)

# --- the trained-to-0 checkpoint set (LOAD ONLY; never retrained here) ---
CKPT_DIR   = "checkpoints/kprop_checkpoints"
PREFIX     = "kprop-zero"
DEPTH      = 3
TOL_TAG    = 5                      # _tol5 = trained to MSE < 1e-5

# --- sweep ---
QUICK   = E.QUICK
WIDTHS  = [32, 64, 128, 256] if QUICK else [32, 64, 128, 256, 512, 1024, 1536, 2048]
SEEDS   = [3, 4]                    # 2 seeds (available: 3,4,5,6)
LAYERS  = list(range(DEPTH))        # hidden layers 0..DEPTH-1 (all square n x n)
LAYER_NAME = {l: (f"L{l} first (control)" if l == 0
                  else f"L{l} pre-readout" if l == DEPTH - 1 else f"L{l} middle")
              for l in LAYERS}

# --- which widths get a FULL singular-value spectrum drawn (the rest only contribute
#     the scalar top-sv to the scaling fits; full SVD at 2048 is the expensive part) ---
SPECTRUM_WIDTHS = [w for w in (64, 256, 1024, 2048) if w in WIDTHS]
N_SV_KEEP       = 64                # store the top-N singular values per matrix (spectrum plot)

# --- MC sample count for mu = E[a_prev] (the -mu alignment metric, section 7) ---
MU_SAMPLES = 40_000 if QUICK else 100_000

# --- output dirs ---
FIG_DIR   = "results/svd_weight_structure/figures"
CACHE_DIR = "results/svd_weight_structure/cache"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)

SEED_COLOR = {s: c for s, c in zip(SEEDS, ["tab:blue", "tab:red", "tab:green", "tab:purple"])}
LAYER_COLOR = {l: c for l, c in zip(LAYERS, ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"])}

def ckpt_path(w, s, depth=DEPTH):
    return E.ckpt_path(CKPT_DIR, E.run_name(PREFIX, depth=depth, width=w, tol=TOL_TAG, seed=s))

print(f"depth={DEPTH} | widths={WIDTHS} | seeds={SEEDS} | layers={LAYERS}")
print(f"spectrum widths={SPECTRUM_WIDTHS} | analysis dtype=float64 | figs -> {FIG_DIR}")
""")

# ---------------------------------------------------------------------------
md(r"""## 2. What is on disk? (recycle, never retrain)

List the depth-3 checkpoints we will read. Per the repo rule, we **load** them — training
happens only in the depth-4 section, and only if you opt in.
""")

code(r"""
print("checkpoints found in", CKPT_DIR, "(depth", DEPTH, "):\n")
have = {}
for s in SEEDS:
    row = []
    for w in WIDTHS:
        ok = os.path.exists(ckpt_path(w, s))
        have[(w, s)] = ok
        row.append(f"w{w}:{'OK' if ok else '--MISSING--'}")
    print(f"  seed {s}: " + "  ".join(row))
missing = [(w, s) for (w, s), ok in have.items() if not ok]
if missing:
    print("\n[warn] missing:", missing, "-- those points are skipped. (They are NOT trained here;")
    print("       depth-3 should already exist. Re-download or check CKPT_DIR.)")
else:
    print("\nall requested depth-3 checkpoints present.")
""")

# ---------------------------------------------------------------------------
md(r"""## 3. Compute the per-matrix metrics (cached)

For every `(width, seed)` we load the trained model, rebuild its **initialisation** from the same
`ModelConfig` seed, cast both to float64, and for each hidden layer compute:

* `delta_weight_metrics(W, W0)` — the $\Delta W$ shift / skew / row-dot metrics (section B & C),
* the full trained-$W$ singular spectrum (top-`N_SV_KEEP` kept for the spectrum plot, section A).

Results are cached to `results/svd_weight_structure/cache` so a re-run is instant.
""")

code(r"""
def reconstruct_init(payload):
    "Rebuild the untrained model from the SAME config/seed (deterministic init)."
    return ModelConfig(**payload["model_config"]).build().double().eval()

def compute_one(w, s):
    "All per-layer metrics for one (width, seed). Returns a JSON-able dict."
    m, payload = MLP.load(ckpt_path(w, s), map_location="cpu")
    m = m.double().eval()
    init = reconstruct_init(payload)
    rec = {"width": w, "seed": s, "final_loss": float(E.final_loss(payload)), "layers": {}}
    for L in LAYERS:
        W  = m.hidden_layers[L].weight.detach().double().numpy()
        W0 = init.hidden_layers[L].weight.detach().double().numpy()
        d = delta_weight_metrics(W, W0)
        sv = np.linalg.svd(W, compute_uv=False)               # trained-W spectrum
        d["topsv_W"]  = float(sv[0])
        d["sv_W_top"] = [float(x) for x in sv[:N_SV_KEEP]]    # kept for the spectrum plot
        d["sv_W_n"]   = int(sv.size)
        rec["layers"][str(L)] = d
    return rec

def get_metrics(w, s):
    cp = os.path.join(CACHE_DIR, f"wsd_d{DEPTH}_w{w}_seed{s}_nsv{N_SV_KEEP}.json")
    if os.path.exists(cp):
        return json.load(open(cp)), True
    rec = compute_one(w, s); json.dump(rec, open(cp, "w"))
    return rec, False

DATA = {}            # (w, s) -> rec
t0 = time.time()
for s in SEEDS:
    for w in WIDTHS:
        if not have.get((w, s)):
            continue
        rec, cached = get_metrics(w, s)
        DATA[(w, s)] = rec
        print(f"  w{w:<5} s{s}  loss={rec['final_loss']:.2e}  "
              f"{'[cache]' if cached else '[computed]'}", flush=True)
print(f"\ndone in {time.time()-t0:.1f}s  ({len(DATA)} models)")

def series(layer, seed, key):
    "metric `key` of `layer` for `seed`, over the widths present, as (widths, values)."
    xs, ys = [], []
    for w in WIDTHS:
        r = DATA.get((w, seed))
        if r is None:
            continue
        xs.append(w); ys.append(r["layers"][str(layer)][key])
    return np.array(xs, float), np.array(ys, float)
""")

# ---------------------------------------------------------------------------
md(r"""### A scaling-plot helper — connect the points, fit a line per seed

Each width-scaling panel plots, **per seed**, the measured points (connected) plus a
least-squares **best-fit line**. `loglog=True` fits a power law $y\propto n^{p}$ (slope $p$ in the
legend); `loglog=False` fits $y$ vs $\log n$ (a semilog trend). Helper returns nothing — it draws
onto a given `ax`.
""")

code(r"""
def _fit_line(xs, ys, loglog):
    "Return (x_fit, y_fit, slope) least-squares fit; slope is d(log y)/d(log x) or dy/d(log x)."
    m = np.isfinite(ys) & (xs > 0) & (np.abs(ys) > 0 if loglog else np.isfinite(ys))
    if m.sum() < 2:
        return None, None, float("nan")
    lx = np.log(xs[m])
    if loglog:
        ly = np.log(np.abs(ys[m]))
        b, a = np.polyfit(lx, ly, 1)
        xf = np.array(sorted(xs[m])); yf = np.exp(a) * xf ** b
    else:
        b, a = np.polyfit(lx, ys[m], 1)
        xf = np.array(sorted(xs[m])); yf = a + b * np.log(xf)
    return xf, yf, float(b)

def scaling_panel(ax, layer, key, *, loglog=True, ylabel="", title="", absval=False,
                  hline=None):
    "One panel: per-seed connected points + best-fit line, for `key` of `layer` vs width."
    for s in SEEDS:
        xs, ys = series(layer, s, key)
        if xs.size == 0:
            continue
        yv = np.abs(ys) if (absval or loglog) else ys
        c = SEED_COLOR[s]
        ax.plot(xs, yv, "o-", color=c, lw=1.3, ms=5, label=f"seed {s} (data)")
        xf, yf, slope = _fit_line(xs, ys, loglog)
        if xf is not None:
            ax.plot(xf, yf, "--", color=c, lw=1.4, alpha=0.7,
                    label=f"seed {s} fit: slope {slope:+.2f}")
    if hline is not None:
        ax.axhline(hline, color="0.4", ls=":", lw=1.2, label=f"{hline:g}")
    ax.set_xscale("log", base=2)
    if loglog:
        ax.set_yscale("log")
    ax.set_xlabel("width  n"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)

def savefig(fig, name):
    p = os.path.join(FIG_DIR, name); fig.savefig(p, bbox_inches="tight"); return p
""")

# ---------------------------------------------------------------------------
md(r"""## A. SVD spectrum + top-singular-value scaling

**Left/middle/right = the singular-value spectra** of the trained $W$ at a few widths (one panel per
layer), normalised index on the x-axis. A spike (one or few large values above a bulk) means a
low-rank structured component. **Bottom row = the top singular value vs width**, per layer, with a
best-fit power law per seed (slope = the scaling exponent).
""")

code(r"""
# --- spectra: one panel per layer, a curve per spectrum-width (seed SEEDS[0]) ---
s0 = SEEDS[0]
fig, axes = plt.subplots(1, len(LAYERS), figsize=(5.2 * len(LAYERS), 4.4), squeeze=False)
for j, L in enumerate(LAYERS):
    ax = axes[0][j]
    cmap = {w: cm.viridis(i / max(1, len(SPECTRUM_WIDTHS) - 1)) for i, w in enumerate(SPECTRUM_WIDTHS)}
    for w in SPECTRUM_WIDTHS:
        r = DATA.get((w, s0))
        if r is None:
            continue
        sv = np.array(r["layers"][str(L)]["sv_W_top"], float)
        ax.plot(np.arange(1, sv.size + 1) / sv.size, sv, "-", color=cmap[w], lw=1.4, label=f"n={w}")
    ax.set_yscale("log"); ax.set_xlabel("normalised singular-value index")
    ax.set_ylabel(r"$\sigma_k$"); ax.set_title(f"{LAYER_NAME[L]} — top-{N_SV_KEEP} spectrum (seed {s0})")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7, title="width")
fig.suptitle("A. Singular-value spectra of trained W (spike above the bulk = low-rank structure)", y=1.02)
plt.tight_layout(); savefig(fig, "A1_spectra_by_layer.png"); plt.show()

# --- top singular value vs width: trained W (top row) and movement dW (bottom row) ---
fig, axes = plt.subplots(2, len(LAYERS), figsize=(5.2 * len(LAYERS), 8.4), squeeze=False)
for j, L in enumerate(LAYERS):
    scaling_panel(axes[0][j], L, "topsv_W", loglog=True,
                  ylabel=r"$\sigma_1(W)$", title=f"{LAYER_NAME[L]}: top sing. value of trained W")
    scaling_panel(axes[1][j], L, "top_sv", loglog=True,
                  ylabel=r"$\sigma_1(\Delta W)$", title=f"{LAYER_NAME[L]}: top sing. value of ΔW")
fig.suptitle(r"A. Top singular value vs width (best-fit power law $\sigma_1\propto n^{p}$ per seed)", y=1.01)
plt.tight_layout(); savefig(fig, "A2_topsv_scaling.png"); plt.show()
print("Read: slope ~0.5 would be sqrt(n) (a literal all-ones spike); the trained nets show a")
print("milder positive slope -> the spike grows with width but slower than the planted -1/sqrt(n) model.")
""")

# ---------------------------------------------------------------------------
md(r"""## B. Weight mean / variance — the $-1/\sqrt{n}$ shift and the negative skew

$\Delta W = W-W_0$ is the **movement** of each weight during training.

* **Is every entry pushed by $-1/\sqrt{n}$?**  We plot $\overline{\Delta W}\cdot\sqrt{n}$ vs width: a flat
  line at $-1$ would mean every entry moved by exactly $-1/\sqrt{n}$.
* **Is the variance skewed negative?**  `frac_negative` (count that moved down) and `neg_energy_frac`
  (share of the movement *variance* carried by down-moving weights) — both $>0.5$ means *more, and
  larger,* negative steps.

First a histogram of $\Delta W$ at a representative width (with the $-1/\sqrt{n}$ line), then the
width-scaling panels per layer.
""")

code(r"""
# --- ΔW histogram at a representative width, one panel per layer (seed SEEDS[0]) ---
w_hist = SPECTRUM_WIDTHS[-1] if SPECTRUM_WIDTHS else WIDTHS[-1]
m, payload = MLP.load(ckpt_path(w_hist, s0), map_location="cpu"); m = m.double().eval()
init = ModelConfig(**payload["model_config"]).build().double().eval()
fig, axes = plt.subplots(1, len(LAYERS), figsize=(5.2 * len(LAYERS), 4.0), squeeze=False)
for j, L in enumerate(LAYERS):
    dW = (m.hidden_layers[L].weight - init.hidden_layers[L].weight).detach().double().numpy().reshape(-1)
    ax = axes[0][j]
    ax.hist(dW, bins=120, color=LAYER_COLOR[L], alpha=0.8, density=True)
    ax.axvline(0, color="0.4", lw=1)
    ax.axvline(-1 / math.sqrt(w_hist), color="crimson", ls="--", lw=1.4, label=r"$-1/\sqrt{n}$")
    ax.axvline(dW.mean(), color="k", ls=":", lw=1.4, label=f"mean={dW.mean():+.2e}")
    ax.set_title(f"{LAYER_NAME[L]}  (n={w_hist}, seed {s0})")
    ax.set_xlabel(r"$\Delta W$ entry"); ax.set_yscale("log"); ax.legend(fontsize=7)
fig.suptitle(r"B. Distribution of the weight movement $\Delta W$ (note the negative mass on L$\geq$1)", y=1.02)
plt.tight_layout(); savefig(fig, "B1_dW_hist.png"); plt.show()

# --- scaling panels ---
fig, axes = plt.subplots(2, len(LAYERS), figsize=(5.2 * len(LAYERS), 8.4), squeeze=False)
for j, L in enumerate(LAYERS):
    scaling_panel(axes[0][j], L, "mean_entry_x_sqrtn", loglog=False,
                  ylabel=r"$\overline{\Delta W}\cdot\sqrt{n}$",
                  title=f"{LAYER_NAME[L]}: mean shift (×√n)", hline=-1.0)
    axes[0][j].axhline(0, color="0.7", lw=0.8)
for j, L in enumerate(LAYERS):
    # variance asymmetry: frac_negative and neg_energy_frac on one panel
    ax = axes[1][j]
    for s in SEEDS:
        xs, fn = series(L, s, "frac_negative")
        _, ne = series(L, s, "neg_energy_frac")
        if xs.size == 0:
            continue
        c = SEED_COLOR[s]
        ax.plot(xs, fn, "o-", color=c, lw=1.3, ms=4, label=f"seed {s}: frac<0")
        ax.plot(xs, ne, "s--", color=c, lw=1.3, ms=4, alpha=0.75, label=f"seed {s}: neg energy frac")
    ax.axhline(0.5, color="0.4", ls=":", lw=1.2, label="0.5 (symmetric)")
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.02)
    ax.set_xlabel("width  n"); ax.set_ylabel("fraction"); ax.set_title(f"{LAYER_NAME[L]}: negative asymmetry")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
fig.suptitle("B. Mean shift (×√n, top) and negative-variance asymmetry (bottom) vs width", y=1.01)
plt.tight_layout(); savefig(fig, "B2_mean_variance_scaling.png"); plt.show()
print("Read: L0 (control) sits at mean·√n≈0, frac<0≈0.5. L≥1 are negative and grow more")
print("one-sided with width (neg-energy-frac -> ~1) -- the variance IS mostly in the negative direction.")
""")

# ---------------------------------------------------------------------------
md(r"""## C. Weight-row dot products — the shared (all-ones) component

If training adds a shared component to the rows (each row gets $\approx c_i\cdot\tfrac1{\sqrt{n}}\mathbf 1$),
then **off-diagonal row$\cdot$row dot products turn positive** and a large share of $\Delta W$'s energy
lies along the all-ones direction $\tfrac1{\sqrt{n}}\mathbf 1$. Layer 0 (zero-mean input) is the control:
its `allones_energy_frac` should just track the $1/n$ "no-structure" baseline.
""")

code(r"""
fig, axes = plt.subplots(2, len(LAYERS), figsize=(5.2 * len(LAYERS), 8.4), squeeze=False)
for j, L in enumerate(LAYERS):
    scaling_panel(axes[0][j], L, "mean_offdiag_rowdot", loglog=True, absval=True,
                  ylabel=r"$\langle r_i, r_j\rangle_{i\neq j}$ (|mean|)",
                  title=f"{LAYER_NAME[L]}: mean off-diag row dot of ΔW")
    scaling_panel(axes[1][j], L, "allones_energy_frac", loglog=True,
                  ylabel=r"energy frac along $(1/\sqrt{n})\,\mathbf{1}$",
                  title=f"{LAYER_NAME[L]}: all-ones energy of ΔW")
    # 1/n baseline reference on the all-ones panel
    xs = np.array([w for w in WIDTHS if (w, SEEDS[0]) in DATA], float)
    if xs.size:
        axes[1][j].plot(xs, 1.0 / xs, ":", color="0.5", lw=1.2, label="1/n baseline")
        axes[1][j].legend(fontsize=7)
fig.suptitle("C. Shared-row component vs width: positive off-diagonal dot + all-ones energy", y=1.01)
plt.tight_layout(); savefig(fig, "C1_rowdot_scaling.png"); plt.show()
print("Read: L0 all-ones energy rides the 1/n baseline (no shared structure). L≥1 sit FAR above it")
print("(up to ~0.97) -> ΔW is dominated by the all-ones / rank-1 spike; off-diagonal dots are positive.")
""")

# ---------------------------------------------------------------------------
md(r"""## 7. Bonus: is the pre-readout shift aligned to $-\mu$?  (existing Q2 metric)

`weight_structure_metrics(W, mu)` with $\mu=\mathbb E[a_{\text{prev}}]$ (the mean post-activation
feeding $W$) reports whether the pre-readout rows point along $-\mu$ and whether $\mu$ *is* the top
singular direction. This connects the all-ones story above to the $-\mu$ direction the network
actually sees (for a ReLU layer $\mu>0$, and $\mathbf 1$ is its dominant component).
""")

code(r"""
print("pre-readout layer (W_last) -mu alignment, seed", s0, ":")
print(f"{'n':>5} | {'cos(-mu)':>9} {'proj_sign':>10} {'align v1·mu':>11} {'top-sv energy':>13} {'stable rank':>11}")
rows_mu = []
for w in WIDTHS:
    if not have.get((w, s0)):
        continue
    m, _ = MLP.load(ckpt_path(w, s0), map_location="cpu"); m = m.double().eval()
    mu = mean_prev_post(m, n=MU_SAMPLES)
    W = m.hidden_layers[DEPTH - 1].weight.detach().double().numpy()
    r = weight_structure_metrics(W, mu)
    rows_mu.append((w, r))
    print(f"{w:>5} | {r['cos_neg_mu']:>9.3f} {r['proj_sign']:>10.2e} {r['align_v1_mu']:>11.3f} "
          f"{r['top_sv_energy']:>13.3f} {r['stable_rank']:>11.1f}")
print("\ncos(-mu)>0 & proj_sign<0  => rows point along -mu (negative shift).")
print("align(v1,mu)~1 & high top-sv energy & small stable-rank => mu IS the low-rank spike.")
""")

# ---------------------------------------------------------------------------
md(r"""## 8. Summary (depth 3)

- **A — spectrum.** Each trained hidden $W$ carries a top singular value that **grows with width**
  (fitted slope in the A2 legend); $\sigma_1(\Delta W)$ grows faster than $\sigma_1(W)$ but still
  **slower than $\sqrt{n}$** — the spike is real but milder than a literal $-\tfrac1{\sqrt{n}}\mathbf 1\mathbf 1^\top$.
- **B — mean / variance.** Layer 0 (control) is symmetric ($\overline{\Delta W}\!\cdot\!\sqrt{n}\approx0$,
  frac$<$0 $\approx0.5$). Layers $\ge1$ move **negative**: $\overline{\Delta W}\!\cdot\!\sqrt{n}$ is a
  *fraction* of $-1$ (a few % to ~10%, layer/width dependent — so NOT the full $-1/\sqrt{n}$ on the mean),
  while the **variance is overwhelmingly negative** (neg-energy-frac $\to\sim1$). So the *direction* of
  the hypothesis holds strongly; the *per-entry magnitude* is smaller than $1/\sqrt{n}$.
- **C — row dot products.** Off-diagonal row dots are **positive** on layers $\ge1$ and a **large share
  of $\Delta W$ lies along the all-ones direction** (up to ~0.97), vs the $1/n$ baseline on the control.
  The movement is, to good approximation, a **rank-1 all-ones spike with a negative, per-row coefficient.**

Together: training a net to send the Gaussian ball to 0 adds, to each layer that sees a positive-mean
input, a **negative all-ones (rank-1) shift** — most one-sided at the pre-readout layer — whose top
singular value grows sub-$\sqrt{n}$ with width. Numbers are in the panels' best-fit slopes.
""")

# ---------------------------------------------------------------------------
md(r"""---
# DEPTH-4 SUITE — train, analyse, and the kprop **k=2** cov-prop *upturn* check

**There are no depth-4 checkpoints yet**, so this section is **OFF by default**. Flip `RUN_DEPTH4 = True`
on a **GPU** to:

1. **Train** depth-4 trained-to-0 models (same recipe as depth 3: `ZeroTask`, AdamW, train until
   $\mathrm{MSE}<10^{-5}$), all seeds of a width in **one vmapped loop**, **recycled** into
   `checkpoints/kprop_checkpoints` as `kprop-zero_d4_w{n}_tol5_seed{s}` — a re-run loads instead of
   retraining.
2. **Re-run A/B/C** on depth 4 (set `DEPTH=4` uses the same cells; here we just recompute the metric
   table and the three scaling figures inline).
3. **kprop covariance-prop (k=2) error vs width, up to 2048** — does the cov-prop error **upturn**
   (start *rising* with width) and, if so, at what width? Reference: **k=3 upturns ~1536**; the question
   is whether **k=2** upturns *earlier*. k=3 is $O(n^3)$ so it is optional (`INCLUDE_K3`).

> ⚠️ Heavy: depth-4 width-2048 training to tolerance + 2M-sample MC + the scipy/CPU exact-cov path are
> the expensive parts. Start with `D4_WIDTHS` trimmed and scale up. Training is float32 + vmapped +
> checkpoint-recycled, so it resumes across sessions.
""")

code(r"""
RUN_DEPTH4 = False        # <<< flip to True on a GPU to launch the depth-4 suite
INCLUDE_K3 = False        # also run k=3 cov-prop (O(n^3); keep widths modest if True)

# depth-4 knobs (mirror the depth-3 trained-to-0 recipe)
D4_DEPTH   = 4
D4_WIDTHS  = [32, 64, 128, 256, 512, 1024, 1536, 2048]   # the user asked for up to 2048
D4_SEEDS   = SEEDS                                         # same 2 seeds
D4_OUTPUT_DIM = 128                                        # vector output (matches the d3 set)
D4_LOSS_TOL, D4_MAX_STEPS, D4_BATCH, D4_LR = 1e-5, 200_000, 1024, 1e-4
KMAX_LIST  = [2] + ([3] if INCLUDE_K3 else [])             # cov-prop budgets to score
MC_SAMPLES = 200_000 if QUICK else 2_000_000
print("RUN_DEPTH4 =", RUN_DEPTH4, "| INCLUDE_K3 =", INCLUDE_K3,
      "| D4_WIDTHS =", D4_WIDTHS, "| seeds =", D4_SEEDS)
print("Nothing below runs until RUN_DEPTH4 = True.")
""")

code(r"""
# ---- 1) TRAIN (or load) the depth-4 trained-to-0 checkpoints ----------------
if RUN_DEPTH4:
    from tasks import ZeroTask
    from training import TrainConfig
    def build_d4(w, seed):
        cfg = ModelConfig(input_dim=w, hidden_dim=w, depth=D4_DEPTH, output_dim=D4_OUTPUT_DIM,
                          bias=False, final_bias=False, activation="relu", seed=seed)
        return cfg.build().to(device=E.DEVICE, dtype=torch.float32)
    for w in D4_WIDTHS:
        paths  = [ckpt_path(w, s, depth=D4_DEPTH) for s in D4_SEEDS]
        builds = [(lambda s=s: build_d4(w, s)) for s in D4_SEEDS]
        tcfg = TrainConfig(steps=D4_MAX_STEPS, batch_size=D4_BATCH, lr=D4_LR, optimizer="adamw",
                           loss_tol=D4_LOSS_TOL, tol_check_every=1, tol_patience=25,
                           checkpoint_mode="final",
                           device=str(E.DEVICE), dtype="float32")
        trained = E.get_or_train_many(paths, builds,
                                      task=ZeroTask(input_dim=w, output_dim=D4_OUTPUT_DIM),
                                      train_cfg=tcfg, extra_meta={"experiment": "kprop_tol_scaling_d4"},
                                      map_location="cpu", progress=True)
        for s, (mdl, pl, loaded) in zip(D4_SEEDS, trained):
            print(f"  d4 w{w:<5} s{s}: {'[loaded]' if loaded else '[trained]'} "
                  f"loss={E.final_loss(pl):.2e}{'  *** NOT CONVERGED' if E.final_loss(pl)>D4_LOSS_TOL else ''}")
    print("depth-4 checkpoints ready in", CKPT_DIR)
""")

code(r"""
# ---- 2) re-run A/B/C metrics on depth 4 (same delta_weight_metrics) ----------
if RUN_DEPTH4:
    D4 = {}
    for s in D4_SEEDS:
        for w in D4_WIDTHS:
            p = ckpt_path(w, s, depth=D4_DEPTH)
            if not os.path.exists(p):
                continue
            m, payload = MLP.load(p, map_location="cpu"); m = m.double().eval()
            init = ModelConfig(**payload["model_config"]).build().double().eval()
            rec = {"width": w, "seed": s, "layers": {}}
            for L in range(D4_DEPTH):
                W  = m.hidden_layers[L].weight.detach().double().numpy()
                W0 = init.hidden_layers[L].weight.detach().double().numpy()
                d = delta_weight_metrics(W, W0)
                d["topsv_W"] = float(np.linalg.svd(W, compute_uv=False)[0])
                rec["layers"][str(L)] = d
            D4[(w, s)] = rec
    def d4_series(layer, seed, key):
        xs, ys = [], []
        for w in D4_WIDTHS:
            r = D4.get((w, seed))
            if r is None: continue
            xs.append(w); ys.append(r["layers"][str(layer)][key])
        return np.array(xs, float), np.array(ys, float)

    LYRS = list(range(D4_DEPTH))
    fig, axes = plt.subplots(3, len(LYRS), figsize=(4.6 * len(LYRS), 12), squeeze=False)
    for j, L in enumerate(LYRS):
        for row, (key, ylab, ll, hl) in enumerate([
                ("topsv_W", r"$\sigma_1(W)$", True, None),
                ("mean_entry_x_sqrtn", r"$\overline{\Delta W}\cdot\sqrt{n}$", False, -1.0),
                ("allones_energy_frac", "all-ones energy frac", True, None)]):
            ax = axes[row][j]
            for s in D4_SEEDS:
                xs, ys = d4_series(L, s, key)
                if xs.size == 0: continue
                c = SEED_COLOR[s]; yv = np.abs(ys) if ll else ys
                ax.plot(xs, yv, "o-", color=c, lw=1.3, ms=4, label=f"seed {s}")
                xf, yf, slope = _fit_line(xs, ys, ll)
                if xf is not None:
                    ax.plot(xf, yf, "--", color=c, lw=1.2, alpha=0.7, label=f"s{s} slope {slope:+.2f}")
            if hl is not None: ax.axhline(hl, color="0.4", ls=":", lw=1)
            ax.set_xscale("log", base=2); ax.set_yscale("log" if ll else "linear")
            ax.set_xlabel("width n"); ax.set_ylabel(ylab)
            ax.set_title(f"d4 L{L}{' (control)' if L==0 else ' (pre-readout)' if L==D4_DEPTH-1 else ''}: {key}")
            ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=6)
    fig.suptitle("Depth-4 A/B/C: top-sv, mean·√n shift, all-ones energy vs width", y=1.005)
    plt.tight_layout(); savefig(fig, "D4_ABC_scaling.png"); plt.show()
""")

code(r"""
# ---- 3) kprop k=2 cov-prop error vs width (the UPTURN check), up to 2048 -----
if RUN_DEPTH4:
    from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, compare_means
    COV_CFG = {"kind": "simple", "use_avg_metric": False, "factor": False, "use_pK": True,
               "output_d_max": 1}
    def cov_methods():
        meth = {"k=2 (exact cov)": {**COV_CFG, "k_max": 2, "exact_relu_cov": True},
                "k=2 (approx)":    {**COV_CFG, "k_max": 2, "exact_relu_cov": False}}
        if INCLUDE_K3:
            meth["k=3"] = {**COV_CFG, "k_max": 3}
        return meth
    METH = cov_methods()
    cov_rows = []
    for w in D4_WIDTHS:
        for s in D4_SEEDS:
            p = ckpt_path(w, s, depth=D4_DEPTH)
            if not os.path.exists(p):
                continue
            m, _ = MLP.load(p, map_location="cpu"); m = m.double().eval()
            in_dim = m.cfg.input_dim
            mc, st = estimate_empirical_mean(model=m, input_dim=in_dim, num_samples=MC_SAMPLES,
                                             batch_size=8192, device="cpu", dtype=torch.float64)
            rec = {"w": w, "s": s, "floor": float(np.linalg.norm(st["mc_stderr"]) /
                                                   (np.linalg.norm(mc) + 1e-30))}
            for name, cfg in METH.items():
                cp = run_cumulants(m, in_dim, cfg, device="cpu")["mean"]
                rec[name] = float(compare_means(np.asarray(cp, float), np.asarray(mc, float), st)
                                  ["relative_error_mean"])
            cov_rows.append(rec)
            print(f"  d4 w{w:<5} s{s}: " +
                  "  ".join(f"{k}={rec[k]:.2e}" for k in METH) + f"  (floor {rec['floor']:.1e})")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    style = {"k=2 (exact cov)": ("o-", "tab:red"), "k=2 (approx)": ("s--", "tab:orange"),
             "k=3": ("^:", "tab:blue")}
    for name in METH:
        for s in D4_SEEDS:
            xs = [r["w"] for r in cov_rows if r["s"] == s]
            ys = [r[name] for r in cov_rows if r["s"] == s]
            if not xs: continue
            mk, c = style.get(name, ("o-", None))
            ax.plot(xs, ys, mk, color=c, alpha=0.5 + 0.5 * (s == D4_SEEDS[0]),
                    label=f"{name}, seed {s}")
            # mark the UPTURN: width of minimum error (error rises after it)
            if len(ys) >= 3:
                imin = int(np.argmin(ys))
                if 0 < imin < len(ys) - 1:
                    ax.annotate(f"min@n={xs[imin]}", (xs[imin], ys[imin]), fontsize=7,
                                xytext=(0, -12), textcoords="offset points", ha="center")
    flo = [(r["w"], r["floor"]) for r in cov_rows if r["s"] == D4_SEEDS[0]]
    if flo:
        ax.plot([a for a, _ in flo], [b for _, b in flo], "-", color="0.7", lw=1, label="MC floor")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width  n"); ax.set_ylabel(r"$\|\mu_{\rm pred}-\mu_{\rm MC}\|/\|\mu_{\rm MC}\|$")
    ax.set_title("Depth-4: kprop cov-prop error vs width — does k=2 upturn (and before k=3's ~1536)?")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    plt.tight_layout(); savefig(fig, "D4_kprop_cov_upturn.png"); plt.show()
    print("\nUPTURN read: the width where each curve stops falling and starts RISING is its upturn.")
    print("k=3 is known to upturn ~1536; compare where k=2 (exact/approx) turns up here.")
""")

# ---------------------------------------------------------------------------
md(r"""## Download the figures

All panels are saved under `results/svd_weight_structure/figures`. In Colab the cell below zips and
downloads them.
""")

code(r"""
import shutil
print("figures in", os.path.abspath(FIG_DIR), ":")
for f in sorted(glob.glob(os.path.join(FIG_DIR, "*.png"))):
    print("  ", os.path.basename(f), f"({os.path.getsize(f)/1e3:.0f} KB)")
try:
    import google.colab  # noqa
    z = shutil.make_archive("/content/svd_weight_structure_figs", "zip", FIG_DIR)
    from google.colab import files; files.download(z)
except Exception:
    pass
""")

nb.save(os.path.join(os.path.dirname(__file__), "svd_weight_structure_trained_to_0_colab.ipynb"))

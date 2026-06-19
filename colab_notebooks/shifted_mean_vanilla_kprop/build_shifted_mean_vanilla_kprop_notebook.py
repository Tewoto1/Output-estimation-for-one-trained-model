"""Generates shifted_mean_vanilla_kprop_colab.ipynb (valid nbformat-4 JSON).

The SIMPLE companion of the structured-kprop study. Random mean-shifted MLPs, NO
TRAINING, and the question: how well does PURE MEAN PROPAGATION (k_max=1) estimate the
output mean -- and how much does adding the covariance (k_max=2) buy?

    W = W' + B,   W'_{ij} ~ N(0, 1/fan_in)  (i.i.d. Gaussian)
                  B = s * (1/sqrt(n)) 11^T,   s = -1 ("sub", DEFAULT) or +1 ("add")

The shift is on the HIDDEN matrices only. Same |spike| = sqrt(n); the SIGN sets the regime
(see below). Default here is SUB (s=-1): the regime where the output collapses and a Gaussian
closure struggles -- the interesting stress test for a mean-only predictor.

Two predictors, both "traditional" cumulant propagation, differing only in budget k_max:
  * MEAN-PROP  (k_max=1): tracks ONLY the degree-1 cumulant (the mean). The degree-2 piece
    is kept as a DIAGONAL metric (get_r_x(2,1)=1; linear_kprop sets metric = diag(W W^T)),
    so each ReLU uses an exact per-neuron marginal Gaussian moment (mean + marginal variance)
    but NO cross-neuron covariance. The cheapest "mean-field" kprop.
  * KPROP k=2  (k_max=2): adds the full off-diagonal covariance as a tracked cumulant.
Both vs a Monte-Carlo reference. (k_max=1 is explicitly supported in kprop_harmonic.py.)

Why the SIGN sets the regime. Layer 1 sees x ~ N(0,I) (mean 0), so z^1 = W^1 x is mean-zero
either way. But for layers ell >= 2 the input a^{ell-1} is POST-ReLU with mean mu > 0, so
1^T a ~ n*mu and the shared shift is s*(1/sqrt n)*(1^T a) ~ s*sqrt(n)*mu (an O(sqrt n) mean shift):
  * s=-1 ("sub"): pre-acts strongly NEGATIVE -> ReLUs DIE -> output collapses to ~0 (a
    point-mass-at-0 mixture). The TRUE mean is a small fluctuation effect; whether a mean-only
    (k=1) state can estimate it at all -- vs k=2 -- is exactly what this notebook measures.
  * s=+1 ("add"): pre-acts strongly POSITIVE -> ReLUs LINEAR -> activations ~Gaussian -> both
    closures should be accurate (a useful sanity control; flip SHIFT="add").

Why hidden layers only / why these k. kprop is EXACT on the final linear readout, so shifting
it adds no prediction error (it only rescales ||E[out]||). Both k=1 and k=2 are feasible at every
width up to 3072 (k=2 is the n x n covariance; k>=3 would be an n^3 tensor, infeasible).

REPO POLICIES: recycling (MC + kprop cached by config under checkpoints/shifted_mean_vanilla_kprop;
models saved too; nothing recomputed on a re-run); GPU (MC + kprop on E.DEVICE; float64 falls back
to CPU on Apple MPS).

Needs Python >= 3.12 OR the kprop-compat shim (auto-active on import); + torch.
Run:  python "colab_notebooks/shifted_mean_vanilla_kprop/build_shifted_mean_vanilla_kprop_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# How good is **pure mean propagation** ($k{=}1$) at estimating the output mean? (negative shift, **no training**)

**Setup — random models, then shift the weights.** Depth $d\in\{3,4,5\}$ ReLU MLPs, square layers,
no bias, `input_dim = output_dim = width` (all widths equal). Each weight is drawn
$W'_{ij}\sim\mathcal N(0,1/\text{fan\_in})$ and the **hidden** matrices are mean-shifted by $s/\sqrt n$
(`SHIFT="sub"` $\Rightarrow s=-1$, the default; `"add"` $\Rightarrow s=+1$):

$$W = W' + B,\qquad B = s\,\tfrac{1}{\sqrt n}\,\mathbf 1\mathbf 1^\top .$$

**No training** — generate, shift, run kprop.

**The predictors — both cumulant propagation, differing only in the budget $k_{\max}$:**

1. **mean-prop ($k{=}1$)** — tracks **only the mean** (degree-1 cumulant). The degree-2 piece is
   collapsed to a **diagonal** metric ($\mathrm{diag}(WW^\top)$), so each ReLU uses an exact *per-neuron
   marginal* Gaussian moment (mean + marginal variance) but **no cross-neuron covariance**. The cheapest
   "mean-field" kprop — *the predictor under test.*
2. **kprop ($k{=}2$)** — adds the full off-diagonal **covariance** as a tracked cumulant (the reference).

Both compared against a Monte-Carlo mean. *Question:* in the collapse regime, can a mean-only state
estimate the (small, fluctuation-driven) output mean at all — and how much does the $k{=}2$ covariance help?

**Why `sub` is the stress test.** Layer 1 sees $x\sim\mathcal N(0,I)$, so $z^1=W^1x$ is mean-zero. But
for layers $\ell\ge2$ the input is post-ReLU with mean $\mu>0$, so the shared shift is
$s\sqrt n\,\mu$. With $s=-1$ this is a large **negative** push → ReLUs **die** → the output collapses to
$\approx0$ (a point-mass-at-0 mixture). The true mean is then a delicate fluctuation effect; a single
Gaussian (let alone a mean-only) state may miss it. *(`add` is the easy linear-regime control.)*

**Design notes.** *Hidden layers only* — kprop is exact on the linear readout, so shifting it adds zero
prediction error (only rescales $\lVert E[\text{out}]\rVert$). *$k\in\{1,2\}$* — both feasible at every
width to 3072 ($k{=}2$ is the $n\times n$ covariance; $k{\ge}3$ is an $n^3$ tensor, infeasible).

> **Recycling + GPU (repo policy).** MC references + kprop predictions are **cached by config** in
> `checkpoints/shifted_mean_vanilla_kprop` (models saved too); a re-run recomputes nothing. MC + kprop
> run on **`E.DEVICE`** (CUDA; float64 falls back to CPU on Apple MPS).

| | view | what to look for (`sub`, $s=-1$) |
|---|---|---|
| **§2** | actual unscaled $\lVert E[\text{out}]\rVert$ (MC vs $k{=}1$ vs $k{=}2$) **beside** the scaled rel-$L_2$ error | does $k{=}1$ even track the collapsed magnitude? how far is each from MC? |
| **§3** | per-coordinate parity ($k{=}1$ & $k{=}2$ vs MC) + magnitudes table | are the points on $y=x$, or does mean-prop sit off-axis? $k{=}2$ vs $k{=}1$ gap |
| **§4** | **cumulant fidelity, layer by layer** — propagated $\kappa_1$ (mean) & $\kappa_2$ (covariance) vs MC, **exact** vs vanilla $k{=}2$ | is exact $k{=}2$ accurate on the cumulants? *where* do they drift; does exact track the off-diagonal? |
| **§5** | fit $\text{error}\propto n^{\,p}$, per predictor | does either error shrink with width, or stay $O(1)$? how much does covariance buy? |

Needs Python ≥ 3.12 *or* the kprop-compat shim (auto-active on import), plus torch.""")

code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## 1. Config — knobs, device & recycling (probe here, not in `experiments.py`)

`WIDTHS` runs up to **3072**; `DEPTHS = [3,4,5]`; `SEEDS = [1,2]`. The predictor under test is
`KMAX_MEANPROP = 1` (pure mean propagation); `KMAX_REF = 2` is the covariance-aware reference.""")
code(r"""
import math, time, os, copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP
from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, compare_means

QUICK  = E.QUICK
DEVICE = E.DEVICE                     # cuda -> mps -> cpu (auto); TF32 matmuls enabled in experiments.py
torch.set_num_threads(max(torch.get_num_threads(), 2))

# ---- the sweep (NO TRAINING anywhere in this notebook) ----
DEPTHS      = [3] if QUICK else [3, 4, 5]
WIDTHS      = [32, 64, 128] if QUICK else [64, 128, 256, 512, 1024, 2048, 3072]
SEEDS       = [1, 2]
ACTIVATION  = "relu"
SHIFT       = "sub"                    # "sub": W = W' - (1/sqrt n)11^T  -> pre-acts NEGATIVE -> DEAD ReLUs (the stress test)
                                       # "add": W = W' + (1/sqrt n)11^T  -> pre-acts POSITIVE -> ReLU ~linear (easy control)
SIGN        = -1.0 if SHIFT == "sub" else +1.0
MC_SAMPLES  = 100_000 if QUICK else 1_000_000

# ---- the two predictors: budget k_max only ----
KMAX_MEANPROP = 1                      # PREDICTOR UNDER TEST: pure mean propagation (degree-1 cumulant only)
KMAX_REF      = 2                      # reference: covariance-aware kprop (full k=2 off-diagonal)

# ---- GPU policy: float32 compute on GPU, float64 for measurement (repo policy) ----
# MC accumulators + kprop need float64. CUDA has float64 -> run on GPU. Apple MPS has
# NO float64 -> the float64 paths fall back to CPU there (sampling stays correct).
if DEVICE.type == "cuda":
    MC_DEVICE, MC_DTYPE, MC_BATCH = DEVICE, torch.float32, 65_536
    KPROP_DEVICE = str(DEVICE)         # kprop on the GPU too (CUDA supports float64)
else:
    MC_DEVICE, MC_DTYPE, MC_BATCH = torch.device("cpu"), torch.float64, 8_192
    KPROP_DEVICE = "cpu"               # MPS lacks float64; CPU otherwise

# ---- result/model recycling (this notebook's OWN family under checkpoints/) ----
CKPT_DIR = "checkpoints/shifted_mean_vanilla_kprop"
RECYCLE  = True                        # load cached MC/kprop results instead of recomputing
os.makedirs(CKPT_DIR, exist_ok=True)

print("DEVICE:", DEVICE, "| MC:", MC_DEVICE.type, MC_DTYPE, "batch", MC_BATCH, "| kprop dev:", KPROP_DEVICE)
print("QUICK:", QUICK, "| depths:", DEPTHS, "| widths:", WIDTHS, "| seeds:", SEEDS)
print(f"shift: {SHIFT} (s={int(SIGN):+d}) | predictors: mean-prop k={KMAX_MEANPROP} vs kprop k={KMAX_REF}"
      f" | MC_SAMPLES: {MC_SAMPLES:,} | CKPT_DIR: {CKPT_DIR}")
""")

code(r"""
# ---- builder: the mean-shifted random MLP (float64 master). NO TRAINING. ----
def shift_const(n):
    "scalar c with B = s*c*11^T;  c = 1/sqrt(n)  =>  every hidden weight entry shifted by s*(1/sqrt n)."
    return 1.0 / math.sqrt(n)

def shifted_mean_mlp(width, seed, depth):
    "model.MLP with W = W' + s*c*11^T: W'~N(0,1/fan_in) on every layer; shift on HIDDEN layers only (s=SIGN)."
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation=ACTIVATION).double().eval()
    g = torch.Generator().manual_seed(1_000_000 * depth + 10_000 * seed + 7 * width)
    c = shift_const(width)
    with torch.no_grad():
        layers = list(m.hidden_layers) + [m.readout]
        for li, layer in enumerate(layers):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64) / math.sqrt(in_f)
            if li < len(m.hidden_layers):                       # shift HIDDEN weight matrices only
                W = W + SIGN * c * torch.ones(out_f, in_f, dtype=torch.float64)
            layer.weight.copy_(W)
    return m
""")

# =============================================================================
md(r"""## 1b. Recycling + GPU Monte-Carlo (the repo rule: never recompute what's on disk)

`get_model` saves/loads the random model so it is reproducible; `cache_get/cache_put` persist
the expensive MC + kprop results (keyed by config) to the same folder. `mc_reference` runs MC on
`DEVICE` (GPU on CUDA) without mutating the float64 master that kprop consumes.""")
code(r"""
def model_path(w, seed, depth):
    return E.ckpt_path(CKPT_DIR, E.run_name(f"shifted-{SHIFT}", depth=depth, width=w, seed=seed))

def get_model(w, seed, depth):
    "RECYCLE: load the checkpoint if present, else build the random W=W'+s*c*11^T model and SAVE it."
    path = model_path(w, seed, depth)
    if RECYCLE and os.path.exists(path):
        m, _ = MLP.load(path, map_location="cpu")
        return m.double().eval()
    m = shifted_mean_mlp(w, seed, depth)
    m.save(path, extra={"family": "shifted_mean_vanilla_kprop", "shift": SHIFT,
                        "depth": depth, "width": w, "seed": seed})
    return m

# results cache: one .pt per CONFIG signature (changing k_max's / MC_SAMPLES / shift -> fresh file)
CFG_SIG = f"mp{KMAX_MEANPROP}_ref{KMAX_REF}_mc{MC_SAMPLES}_{SHIFT}_{ACTIVATION}"
RESULTS_PATH = os.path.join(CKPT_DIR, f"results_{CFG_SIG}.pt")
_results = torch.load(RESULTS_PATH) if (RECYCLE and os.path.exists(RESULTS_PATH)) else {}
def cache_get(key):       return _results.get(key) if RECYCLE else None
def cache_put(key, val):  _results[key] = val; torch.save(_results, RESULTS_PATH)

def mc_reference(m, w):
    "Monte-Carlo E[out] on DEVICE (GPU on CUDA), float64 accumulators; does NOT mutate m."
    mdev = copy.deepcopy(m).to(device=MC_DEVICE, dtype=MC_DTYPE)
    mc, stats = estimate_empirical_mean(model=mdev, input_dim=w, num_samples=MC_SAMPLES,
                                        device=str(MC_DEVICE), dtype=MC_DTYPE, batch_size=MC_BATCH)
    del mdev
    if MC_DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return mc, stats

def rel(cp, mc, stats):
    return compare_means(cp, mc, stats)["relative_error_mean"]      # = ||cp - mc|| / ||mc||

print(f"results cache {os.path.basename(RESULTS_PATH)}: {len(_results)} runs "
      f"({'HIT -> recycling' if _results else 'empty -> will compute + save'})")
""")

# =============================================================================
md(r"""## §2 — Width sweep: actual means **and** scaled error (mean-prop $k{=}1$ vs kprop $k{=}2$)

For each `(depth, width, seed)` we keep both the **actual unscaled** output means
$\mu_{\text{MC}}$, $\mu_{k1}$, $\mu_{k2}$ (full vectors) and the scaled relative $L_2$ errors. Each run
is recycled from the cache when present; otherwise the model is built/loaded, MC runs on the GPU, both
kprop budgets run, and the result is saved. `floor` is the MC sampling noise (`||stderr|| / ||mc||`).

The plot puts them **side by side**: left = the actual mean *magnitudes* $\lVert\mu\rVert$ (MC vs both
predictors, unscaled), right = the scaled relative error + MC floor.""")
code(r"""
rows, t0 = [], time.time()
for depth in DEPTHS:
    for w in WIDTHS:
        for seed in SEEDS:
            key = f"d{depth}|w{w}|s{seed}"
            r = cache_get(key); src = "recycled"
            if r is None:
                src = "computed"
                m = get_model(w, seed, depth)
                mc, stats = mc_reference(m, w)
                mp = run_cumulants(m, config={"k_max": KMAX_MEANPROP, "factor": False},
                                   device=KPROP_DEVICE)["mean"]      # MEAN PROPAGATION (k=1)
                k2 = run_cumulants(m, config={"k_max": KMAX_REF, "factor": False},
                                   device=KPROP_DEVICE)["mean"]      # covariance kprop (k=2)
                mc = np.asarray(mc, float); mp = np.asarray(mp, float); k2 = np.asarray(k2, float)
                nm = float(np.linalg.norm(mc)) + 1e-30
                r = dict(depth=depth, w=w, seed=seed,
                         mp=rel(mp, mc, stats), k2=rel(k2, mc, stats),
                         floor=float(np.linalg.norm(stats["mc_stderr"])) / nm,
                         mc_norm=float(np.linalg.norm(mc)), mp_norm=float(np.linalg.norm(mp)),
                         k2_norm=float(np.linalg.norm(k2)),
                         mc_mean=torch.tensor(mc, dtype=torch.float32),   # actual UNSCALED mean vectors,
                         mp_mean=torch.tensor(mp, dtype=torch.float32),   # kept (~12 KB each) for the
                         k2_mean=torch.tensor(k2, dtype=torch.float32))   # per-coordinate parity view
                cache_put(key, r)
            rows.append(r)
            print(f"d{depth} w={w:>4} s{seed} [{src:>8}] | rel-err: mean-prop(k1) {r['mp']:.3e}  "
                  f"kprop(k2) {r['k2']:.3e} (floor {r['floor']:.1e}) | ||mu||: MC {r['mc_norm']:.3e}  "
                  f"k1 {r['mp_norm']:.3e}  k2 {r['k2_norm']:.3e}", flush=True)
print(f"\nsweep done in {time.time() - t0:.1f}s ({len(rows)} runs; recycled ones are instant)")
""")
code(r"""
from matplotlib.lines import Line2D
def series(depth, key):
    "mean over seeds of `key` at each width, for one depth"
    return [float(np.mean([r[key] for r in rows if r["depth"] == depth and r["w"] == w]))
            for w in WIDTHS]

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(DEPTHS)))
cmap = dict(zip(DEPTHS, colors))           # colour encodes DEPTH; marker/linestyle encodes PREDICTOR
op = "+" if SIGN > 0 else "-"              # sign of the shift, for the titles
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.4))

# LEFT -- actual UNSCALED magnitude ||E[out]||: MC (o-) vs mean-prop k=1 (x--) vs kprop k=2 (^:)
for d in DEPTHS:
    c = cmap[d]
    axL.loglog(WIDTHS, series(d, "mc_norm"), "o-",  color=c)
    axL.loglog(WIDTHS, series(d, "mp_norm"), "x--", color=c)
    axL.loglog(WIDTHS, series(d, "k2_norm"), "^:",  color=c)
axL.set_xlabel("width  n"); axL.set_ylabel(r"$\|E[\mathrm{out}]\|_2$   (unscaled)")
axL.set_title(r"actual mean magnitude on $W=W'" + op + r"\frac{1}{\sqrt n}11^\top$ (MC vs kprop)")
axL.grid(alpha=0.3, which="both")

# RIGHT -- SCALED relative L2 error: mean-prop k=1 (x--) vs kprop k=2 (^:), + one MC-noise-floor line
for d in DEPTHS:
    c = cmap[d]
    axR.loglog(WIDTHS, series(d, "mp"), "x--", color=c)
    axR.loglog(WIDTHS, series(d, "k2"), "^:",  color=c)
floor_hi = [max(series(d, "floor")[i] for d in DEPTHS) for i in range(len(WIDTHS))]
axR.loglog(WIDTHS, floor_hi, "-", color="0.6", lw=1.1, label="MC noise floor")
axR.set_xlabel("width  n"); axR.set_ylabel(r"$\|\mu_{\mathrm{pred}}-\mu_{\mathrm{MC}}\| / \|\mu_{\mathrm{MC}}\|$")
axR.set_title(r"scaled error on $W=W'" + op + r"\frac{1}{\sqrt n}11^\top$  (lower = better)")
axR.grid(alpha=0.3, which="both")

# dual key: colour = depth, marker/linestyle = predictor (shared across both panels)
depth_h = [Line2D([0], [0], color=cmap[d], lw=2.4, label=f"depth {d}") for d in DEPTHS]
pred_h  = [Line2D([0], [0], color="0.35", marker="o", ls="-",  label="MC (truth)"),
           Line2D([0], [0], color="0.35", marker="x", ls="--", label="mean-prop  k=1"),
           Line2D([0], [0], color="0.35", marker="^", ls=":",  label="kprop  k=2")]
_l1 = axL.legend(handles=depth_h, loc="lower right", fontsize=8, title="colour = depth"); axL.add_artist(_l1)
axL.legend(handles=pred_h, loc="upper left", fontsize=8, title="style = predictor")
axR.legend(loc="lower left", fontsize=8)   # MC-floor line; colour/style key is the left panel
plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §3 — The actual means, unscaled: magnitudes table + per-coordinate parity

The relative error hides *what the means actually are*. Here we read them raw. The table prints the
unscaled magnitudes $\lVert\mu_{\text{MC}}\rVert$, $\lVert\mu_{k1}\rVert$, $\lVert\mu_{k2}\rVert$ and the
ratios to MC. The scatter takes one representative model (largest width) and plots **both** predictors'
mean against MC's **per output coordinate** — a perfect predictor sits on the dashed $y=x$ line, so the
spread/tilt off it is the literal bias. Watch whether mean-prop ($k{=}1$, blue) is even on the same axis
as the covariance kprop ($k{=}2$, red), and how far either sits from $y=x$.""")
code(r"""
# --- numeric table: ACTUAL unscaled magnitudes, MC vs mean-prop(k1) vs kprop(k2) (mean over seeds) ---
print("actual UNSCALED output-mean magnitudes (averaged over seeds):\n")
print(f"{'depth':>5} {'width':>6} | {'||mu_MC||':>11} | {'||mu_k1||':>11} {'k1/MC':>7} "
      f"| {'||mu_k2||':>11} {'k2/MC':>7}")
print("-" * 70)
for depth in DEPTHS:
    for w in WIDTHS:
        rs = [r for r in rows if r["depth"] == depth and r["w"] == w]
        mcn = float(np.mean([r["mc_norm"] for r in rs]))
        k1n = float(np.mean([r["mp_norm"] for r in rs]))
        k2n = float(np.mean([r["k2_norm"] for r in rs]))
        print(f"{depth:>5} {w:>6} | {mcn:>11.4e} | {k1n:>11.4e} {k1n / mcn:>7.3f} "
              f"| {k2n:>11.4e} {k2n / mcn:>7.3f}")

# --- parity scatter at the largest width (both predictors run at every width) ---
W0, D0, S0 = WIDTHS[-1], DEPTHS[0], SEEDS[0]
rr = next(r for r in rows if r["depth"] == D0 and r["w"] == W0 and r["seed"] == S0)
mc_v = np.asarray(rr["mc_mean"], float)
k1_v = np.asarray(rr["mp_mean"], float)
k2_v = np.asarray(rr["k2_mean"], float)

def _fit(y):
    s = float(np.polyfit(mc_v, y, 1)[0]); return s, float(np.corrcoef(mc_v, y)[0, 1])
s1, c1 = _fit(k1_v); s2, c2 = _fit(k2_v)
lim = float(max(np.abs(mc_v).max(), np.abs(k1_v).max(), np.abs(k2_v).max())) * 1.1

fig, ax = plt.subplots(figsize=(6.4, 6.2))
ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="$y=x$ (perfect)")
ax.scatter(mc_v, k1_v, s=9, alpha=0.35, color="tab:blue", label=f"mean-prop k=1  (slope {s1:.2f}, corr {c1:.2f})")
ax.scatter(mc_v, k2_v, s=9, alpha=0.35, color="tab:red",  label=f"kprop k=2  (slope {s2:.2f}, corr {c2:.2f})")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel(r"MC mean  $\mu_{\mathrm{MC}}[i]$  (unscaled)")
ax.set_ylabel(r"kprop mean  $\mu_{\mathrm{pred}}[i]$  (unscaled)")
ax.set_title(f"actual per-coordinate means: kprop vs MC  (depth {D0}, width {W0}, seed {S0})")
ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
print(f"\nrep d{D0} w{W0} s{S0}:  ||mu_MC|| = {np.linalg.norm(mc_v):.4e}   "
      f"||mu_k1|| = {np.linalg.norm(k1_v):.4e}   ||mu_k2|| = {np.linalg.norm(k2_v):.4e}")
""")

# =============================================================================
md(r"""## §4 — Cumulant fidelity, layer by layer: do the propagated cumulants match the actual ones?

The sweep above only scores the **final output mean**. Here we open the box: kprop's `mlp_kprop` has a
debug path, `output_all=True`, that returns the **cumulant tower at every layer** (`pre{l}` after each
linear step, `act{l}` after each ReLU). We read off the predicted $\kappa_1$ (mean) and $\kappa_2$
(covariance) and compare them, **layer by layer**, to Monte-Carlo estimates of the *actual* activation
mean & covariance — so we see *where* the prediction drifts and whether **exact-ReLU-cov $k{=}2$** keeps
the cumulants accurate.

- **mean ($\kappa_1$):** compared for all three — mean-prop ($k{=}1$), vanilla ($k{=}2$), exact-cov ($k{=}2$).
- **covariance ($\kappa_2$):** compared for the two $k{=}2$ variants, split into the **diagonal** (per-neuron
  variance) and the **off-diagonal** — the off-diagonal is exactly what `exact_relu_cov` computes exactly,
  so it is the decisive test. *(k=1's degree-2 is only a fixed $\mathrm{diag}(WW^\top)$ metric, not a
  predicted covariance, so we don't score its $\kappa_2$.)*
- **"kept to a similar rate":** alongside the errors we print the **magnitude ratios** $\lVert\kappa^{\text{pred}}\rVert/\lVert\kappa^{\text{MC}}\rVert$;
  a ratio that stays $O(1)$ across layers means the cumulant is tracked at the right scale, a ratio that
  collapses/explodes flags where the single-Gaussian state stops matching the (non-Gaussian) truth.

Representative depth `CUMULANT_DEPTH` over a few **small** widths (a full $n\times n$ covariance needs a
modest $n$ + enough MC samples). Exact-cov is scipy/CPU here.""")
code(r"""
from collections import OrderedDict
from Mecha_preds.cumulants.kprop import mlp_kprop as _mlp_kprop, Kind as _Kind
from Mecha_preds.cumulants.adapter import model_to_kprop as _model_to_kprop

# ---- knobs for THIS section (full covariance -> keep widths modest) ----
CUMULANT_DEPTH  = max(DEPTHS)
CUMULANT_WIDTHS = [32, 64] if QUICK else [64, 128, 256]
CUMULANT_MC     = 200_000 if QUICK else 1_000_000
CUM_SEED        = SEEDS[0]
CUM_BATCH       = min(MC_BATCH, 8192)

def kprop_layer_cumulants(m, k_max, exact=False):
    "Per-layer kprop cumulants via output_all=True -> {layer_key: (mean (n,), cov (n,n) or None)}."
    n = m.cfg.input_dim
    kmlp = _model_to_kprop(m, device="cpu")                       # float64 kprop copy of the SAME weights
    K_in = {1: torch.zeros(n, dtype=torch.float64), 2: torch.eye(n, dtype=torch.float64)}
    K_by = _mlp_kprop(kmlp, K_in, k_max=k_max, kind=_Kind.SIMPLE, use_avg_metric=False, factor=False,
                      use_pK=True, output_all=True, output_d_max=2, exact_relu_cov=exact)
    out = OrderedDict()
    for key, K in K_by.items():
        mean = K[1].to_tensor().double().cpu().numpy().reshape(-1)
        cov = K[2].to_tensor().double().cpu().numpy() if (k_max >= 2 and 2 in K) else None
        out[key] = (mean, cov)
    return out

@torch.no_grad()
def mc_layer_cumulants(m, n, num_samples, batch, device, dtype):
    "Streaming-MC mean + covariance of EVERY pre/post activation and the output, keyed like kprop."
    depth = m.cfg.depth
    md = copy.deepcopy(m).to(device=device, dtype=dtype).eval()
    keys = [f"pre{l}" for l in range(depth)] + [f"act{l}" for l in range(depth)] + [f"pre{depth}"]
    s1 = {k: torch.zeros(n, dtype=torch.float64, device=device) for k in keys}
    s2 = {k: torch.zeros(n, n, dtype=torch.float64, device=device) for k in keys}
    N = 0
    while N < num_samples:
        b = min(batch, num_samples - N)
        acts = md.activations(torch.randn(b, n, device=device, dtype=dtype))
        tens = {f"pre{l}": acts["pre"][l] for l in range(depth)}
        tens.update({f"act{l}": acts["post"][l] for l in range(depth)})
        tens[f"pre{depth}"] = acts["output"]
        for k, t in tens.items():
            t = t.double(); s1[k] += t.sum(0); s2[k] += t.T @ t
        N += b
    out = OrderedDict()
    for k in keys:
        mu = s1[k] / N
        out[k] = (mu.cpu().numpy(), (s2[k] / N - torch.outer(mu, mu)).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out

def cumulant_errors(pred_mean, pred_cov, mc_mean, mc_cov):
    "rel errors + magnitude ratios for kappa_1 (mean) and (if predicted) kappa_2 (covariance)."
    eps = 1e-30
    d = dict(mean_relerr=float(np.linalg.norm(pred_mean - mc_mean) / (np.linalg.norm(mc_mean) + eps)),
             mean_ratio=float(np.linalg.norm(pred_mean) / (np.linalg.norm(mc_mean) + eps)))
    if pred_cov is not None:
        off = ~np.eye(mc_cov.shape[0], dtype=bool)
        dvp, dvm = np.diag(pred_cov), np.diag(mc_cov)
        d.update(var_relerr=float(np.linalg.norm(dvp - dvm) / (np.linalg.norm(dvm) + eps)),
                 off_relerr=float(np.linalg.norm((pred_cov - mc_cov)[off]) / (np.linalg.norm(mc_cov[off]) + eps)),
                 cov_relerr=float(np.linalg.norm(pred_cov - mc_cov) / (np.linalg.norm(mc_cov) + eps)),
                 var_ratio=float(np.linalg.norm(dvp) / (np.linalg.norm(dvm) + eps)))
    return d

def _errtable(pred, mc, keys):
    return {kk: cumulant_errors(pred[kk][0], pred[kk][1], mc[kk][0], mc[kk][1]) for kk in keys}
print("cumulant-fidelity knobs | depth:", CUMULANT_DEPTH, "widths:", CUMULANT_WIDTHS,
      "MC:", f"{CUMULANT_MC:,}", "seed:", CUM_SEED)
""")
code(r"""
# Propagate predicted cumulants (k=1, k=2 vanilla, k=2 exact) and compare to MC, per layer, per width.
cum, t0 = {}, time.time()
for w in CUMULANT_WIDTHS:
    ckey = f"cum|d{CUMULANT_DEPTH}|w{w}|s{CUM_SEED}|mc{CUMULANT_MC}"
    rec = cache_get(ckey); src = "recycled"
    if rec is None:
        src = "computed"
        m = get_model(w, CUM_SEED, CUMULANT_DEPTH)
        mc  = mc_layer_cumulants(m, w, CUMULANT_MC, CUM_BATCH, MC_DEVICE, MC_DTYPE)
        k1  = kprop_layer_cumulants(m, k_max=1, exact=False)
        k2v = kprop_layer_cumulants(m, k_max=2, exact=False)
        k2e = kprop_layer_cumulants(m, k_max=2, exact=True)
        keys = list(k2v.keys())
        rec = dict(keys=keys, e_k1=_errtable(k1, mc, keys),
                   e_k2v=_errtable(k2v, mc, keys), e_k2e=_errtable(k2e, mc, keys))
        cache_put(ckey, rec)
    cum[w] = rec
    out_k = rec["keys"][-1]
    print(f"w={w:>4} [{src:>8}] output mean rel-err: k1 {rec['e_k1'][out_k]['mean_relerr']:.3e}  "
          f"k2 {rec['e_k2v'][out_k]['mean_relerr']:.3e}  k2-exact {rec['e_k2e'][out_k]['mean_relerr']:.3e}",
          flush=True)
print(f"\ncumulant fidelity done in {time.time() - t0:.1f}s")

# ---- per-layer table at the largest width ----
wbig = CUMULANT_WIDTHS[-1]; rec = cum[wbig]; keys = rec["keys"]
print(f"\nper-layer rel-errors (depth {CUMULANT_DEPTH}, width {wbig}, seed {CUM_SEED}, MC {CUMULANT_MC:,}):")
print("MEAN cols = rel-err of kappa_1; 'k2ex rat' = ||mu_pred||/||mu_MC||; COV split into off-diagonal & diag(var)\n")
print(f"{'layer':>7} | {'mean k1':>8} {'mean k2':>8} {'mean k2ex':>9} {'k2ex rat':>8} "
      f"| {'covOFF k2':>9} {'covOFF k2ex':>11} | {'covDIAG k2':>10} {'covDIAG k2ex':>12}")
print("-" * 96)
for k in keys:
    e1, ev, ee = rec['e_k1'][k], rec['e_k2v'][k], rec['e_k2e'][k]
    print(f"{k:>7} | {e1['mean_relerr']:>8.2e} {ev['mean_relerr']:>8.2e} {ee['mean_relerr']:>9.2e} "
          f"{ee['mean_ratio']:>8.2f} | {ev.get('off_relerr', float('nan')):>9.2e} "
          f"{ee.get('off_relerr', float('nan')):>11.2e} | {ev.get('var_relerr', float('nan')):>10.2e} "
          f"{ee.get('var_relerr', float('nan')):>12.2e}")

off_ratios = [rec['e_k2v'][k]['off_relerr'] / max(rec['e_k2e'][k]['off_relerr'], 1e-30)
              for k in keys if 'off_relerr' in rec['e_k2v'][k]]
print(f"\nmedian (vanilla off-diag err / exact off-diag err) over layers = {np.median(off_ratios):.2f}x"
      "   (>1 => exact-cov tracks the off-diagonal covariance better; ~1 => no better)")
""")
code(r"""
# ---- plots: (left) per-layer mean & cov fidelity at the largest width; (right) cumulant error vs width ----
wbig = CUMULANT_WIDTHS[-1]; rec = cum[wbig]; keys = rec["keys"]; xi = np.arange(len(keys))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.8, 5.2))

a1.semilogy(xi, [rec['e_k1'][k]['mean_relerr']  for k in keys], "o-",  label=r"mean ($\kappa_1$) k=1")
a1.semilogy(xi, [rec['e_k2v'][k]['mean_relerr'] for k in keys], "x--", label=r"mean ($\kappa_1$) k=2")
a1.semilogy(xi, [rec['e_k2e'][k]['mean_relerr'] for k in keys], "^:",  label=r"mean ($\kappa_1$) k=2 exact")
a1.semilogy(xi, [rec['e_k2v'][k]['off_relerr']  for k in keys], "x--", color="0.5", label=r"cov off-diag ($\kappa_2$) k=2")
a1.semilogy(xi, [rec['e_k2e'][k]['off_relerr']  for k in keys], "^:",  color="tab:red", label=r"cov off-diag ($\kappa_2$) k=2 exact")
a1.set_xticks(xi); a1.set_xticklabels(keys, rotation=45, fontsize=7)
a1.set_ylabel("relative error vs MC"); a1.set_title(f"per-layer cumulant fidelity (d{CUMULANT_DEPTH}, w{wbig})")
a1.legend(fontsize=7); a1.grid(alpha=0.3, which="both")

outk = f"pre{CUMULANT_DEPTH}"; lasthid = f"act{CUMULANT_DEPTH - 1}"
a2.loglog(CUMULANT_WIDTHS, [cum[w]['e_k2v'][outk]['mean_relerr'] for w in CUMULANT_WIDTHS], "x--", label="output mean: k=2")
a2.loglog(CUMULANT_WIDTHS, [cum[w]['e_k2e'][outk]['mean_relerr'] for w in CUMULANT_WIDTHS], "^:",  label="output mean: k=2 exact")
a2.loglog(CUMULANT_WIDTHS, [cum[w]['e_k2v'][lasthid]['cov_relerr'] for w in CUMULANT_WIDTHS], "x-", color="0.5", label="last-hidden cov: k=2")
a2.loglog(CUMULANT_WIDTHS, [cum[w]['e_k2e'][lasthid]['cov_relerr'] for w in CUMULANT_WIDTHS], "^-", color="tab:red", label="last-hidden cov: k=2 exact")
a2.set_xlabel("width  n"); a2.set_ylabel("relative error vs MC")
a2.set_title(f"cumulant error vs width (depth {CUMULANT_DEPTH})"); a2.legend(fontsize=8); a2.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §5 — How does the (final-mean) error scale? Fit $\text{error}\propto n^{\,p}$, per predictor

A *working* predictor has $p<0$ (error shrinking with width) or already rides the MC floor; a *broken*
one stays flat at $p\approx0$. We fit the slope per depth for **both** mean-prop ($k{=}1$) and kprop
($k{=}2$), and report how much (if anything) the covariance buys.""")
code(r"""
def fit_slope(key):
    "log-log slope of (mean rel-err vs width) per depth, over the widths where the value is finite & >0"
    out = {}
    w = np.array(WIDTHS, float)
    for depth in DEPTHS:
        e = np.array(series(depth, key)); ok = np.isfinite(e) & (e > 0)
        out[depth] = float(np.polyfit(np.log(w[ok]), np.log(e[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
    return out

sl_k1, sl_k2 = fit_slope("mp"), fit_slope("k2")
print("log-log slope p of (rel-err vs width):  p~0 => O(1) constant error; p<0 => shrinking with width\n")
print(f"{'depth':>5} | {'k1 slope':>9} {'k1 err@maxN':>13} | {'k2 slope':>9} {'k2 err@maxN':>13} | {'floor@maxN':>11}")
print("-" * 74)
ratios = []
for depth in DEPTHS:
    e1 = np.array(series(depth, "mp")); e2 = np.array(series(depth, "k2")); fl = np.array(series(depth, "floor"))
    ratios.append(e1[-1] / max(e2[-1], 1e-30))
    print(f"{depth:>5} | {sl_k1[depth]:>9.3f} {e1[-1]:>13.3e} | {sl_k2[depth]:>9.3f} {e2[-1]:>13.3e} | {fl[-1]:>11.1e}")

k1_flat   = all(np.isfinite(sl_k1[d]) and abs(sl_k1[d]) < 0.15 for d in DEPTHS)
k2_better = float(np.median(ratios))      # >1 means k=1 error is that many x worse than k=2
near_floor_k2 = all(series(d, "k2")[-1] < 5 * series(d, "floor")[-1] for d in DEPTHS)
print()
print(f"=> mean-prop(k1) slopes ~flat: {k1_flat} | median (k1 err / k2 err) = {k2_better:.2f}x"
      f" | k=2 near MC floor: {near_floor_k2}")
if SIGN < 0:
    print("   SUB regime: deep ReLUs die -> output collapses to a point-mass-at-0 mixture. The true mean is")
    print("   a small fluctuation effect. Read above whether MEAN-PROP (k=1) can estimate it at all, and how")
    print("   much the k=2 covariance helps (ratio>1) -- vs whether BOTH stay O(1) off (single-Gaussian limit).")
else:
    print("   ADD regime (control): pre-acts positive -> ReLUs linear -> activations ~Gaussian -> both k=1 and")
    print("   k=2 should be accurate; here the covariance should add little (the layer is essentially linear).")
""")

# =============================================================================
md(r"""## §6 — Checkpoints: save / load / **download** (recycle across sessions)

The sweep wrote the random models + a results cache to `checkpoints/shifted_mean_vanilla_kprop`.
If the repo lives on Google Drive (set `LOCAL_REPO_DIR` in the bootstrap) it already persists —
next session re-runs and **recycles**. Otherwise download a zip and re-upload it later.""")
code(r"""
import shutil
print("checkpoint dir:", os.path.abspath(CKPT_DIR))
for f in sorted(os.listdir(CKPT_DIR)):
    print("  ", f, f"({os.path.getsize(os.path.join(CKPT_DIR, f)) / 1e6:.2f} MB)")

if IN_COLAB:
    from google.colab import files
    zpath = shutil.make_archive("/content/shifted_mean_vanilla_kprop_ckpts", "zip", CKPT_DIR)
    print("\nzipped ->", zpath, "-- downloading...")
    files.download(zpath)

# To RESTORE in a fresh Colab runtime (so nothing recomputes), upload the zip and unpack:
#   from google.colab import files; up = files.upload()
#   import io, zipfile, os
#   os.makedirs(CKPT_DIR, exist_ok=True)
#   zipfile.ZipFile(io.BytesIO(next(iter(up.values())))).extractall(CKPT_DIR)
""")

# =============================================================================
md(r"""## §7 — Summary

- **What ran:** random depth-{3,4,5} ReLU MLPs (square, no bias), every hidden weight matrix
  mean-shifted by $-1/\sqrt n$ — `SHIFT="sub"`, $W=W'-\tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top$ (flip to
  `"add"` for the linear-regime control), **no training**; **two budgets** — mean-prop ($k{=}1$, the
  predictor under test) and covariance kprop ($k{=}2$, reference) — vs Monte-Carlo, widths to 3072, seeds 1–2.
- **What $k{=}1$ is:** it propagates **only the mean** (degree-1 cumulant), with the degree-2 piece kept
  as a diagonal metric $\mathrm{diag}(WW^\top)$ — so each ReLU uses an exact *marginal* mean+variance but
  **no cross-neuron covariance**. $k{=}2$ adds that covariance.
- **Final-mean error (§2–§3, §5):** the **unscaled magnitudes** (does $\lVert\mu_{k1}\rVert$ track
  $\lVert\mu_{\text{MC}}\rVert$ or over/under-shoot the collapse), the **parity** (on $y=x$?), and the
  width **scaling** — with the printed `median (k1 err / k2 err)` saying how much the covariance buys.
- **Cumulant fidelity (§4):** kprop's `output_all=True` exposes every layer's tower, so we score the
  predicted $\kappa_1$ (mean) and $\kappa_2$ (covariance) against MC **layer by layer** — splitting $\kappa_2$
  into diagonal (variance) and off-diagonal, the latter being exactly what `exact_relu_cov` computes. The
  printed `median (vanilla off-diag err / exact off-diag err)` says whether exact $k{=}2$ tracks the
  covariance better; the magnitude *ratios* say whether each cumulant is kept at the right scale through depth.
- **Why `sub` is hard:** the $-\sqrt n\,\mu$ shift at depth kills the ReLUs → the output collapses to a
  point-mass-at-0 mixture whose residual mean is a fluctuation effect a single-Gaussian (let alone
  mean-only) state may not capture. The `add` control is the easy linear-regime opposite.

**Recycling:** models + MC/kprop results live in `checkpoints/shifted_mean_vanilla_kprop` (keyed by config,
so `sub`/`add` and different $k$ never mix); re-runs load instead of recomputing, and §6 downloads the dir.
**GPU:** MC + kprop run on `E.DEVICE` (CUDA), float64-on-CPU fallback on MPS.""")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shifted_mean_vanilla_kprop_colab.ipynb")
nb.save(out)

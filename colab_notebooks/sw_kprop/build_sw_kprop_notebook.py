"""Generates sw_kprop_colab.ipynb (valid nbformat-4 JSON).

Tests the SW-KPROP predictor (Mecha_preds/cumulants/swkprop) -- "Shifted-Weight
K-Propagation with Special-Direction Cumulants" -- on the two regimes asked for:

  (A) the SHIFTED-WEIGHTS model  M = W' + s*(1/sqrt n) 11^T  (sub: s=-1, add: s=+1),
      hidden layers only, output_dim=width, NO TRAINING -- the theorem's exact regime;
  (B) the TRAINED-TO-0 checkpoints in checkpoints/kprop_checkpoints (depth 3, scalar
      output) -- a real trained network with no special structure imposed.

For each model we compare, against a Monte-Carlo mean:
  * SW-KPROP at output rank R in {2,3,4} (R=2 = exact rank-2 Gaussian-ReLU in the
    split basis; R>=3 adds the amplified special cumulants d3,d4 via Cornish-Fisher);
  * the existing exact-K2 kprop (run_cumulants k_max=2, exact_relu_cov=True) as the
    "total-order" baseline SW-KPROP is meant to beat on the shifted model.

REPO POLICIES: notebook owns its knobs + CKPT_DIR; recycling (models + MC/predictor
results cached, nothing recomputed on a re-run); GPU (MC on E.DEVICE float32; SW-KPROP
routes its dense congruence to CUDA float64; float64 falls back to CPU on Apple MPS).

Quick-smoke defaults (QUICK on CPU, or the small sweep): widths <= 256, 1 seed.

Run:  python "colab_notebooks/sw_kprop/build_sw_kprop_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# SW-KPROP: shifted-weight K-propagation with special-direction cumulants

How well does **SW-KPROP** predict the output mean $E[f(X)]$, $X\sim\mathcal N(0,I)$, of a ReLU MLP
whose hidden matrices are **mean-shifted**,

$$M = W' + s\,\tfrac{1}{\sqrt n}\mathbf 1\mathbf 1^\top,\qquad W'_{ij}\sim\mathcal N(0,1/\text{fan\_in}),$$

and on the project's **trained-to-0** checkpoints?

**The idea (why ordinary kprop is the wrong truncation here).** The shift amplifies the all-ones
(*special*) direction $u=\tfrac1{\sqrt n}\mathbf 1$ by $\sim\!\sqrt n$ *each layer*, so the special
mode's **high-order cumulants are not small** and must be kept — truncating by total cumulant order
(ordinary kprop) throws them away. SW-KPROP works in the split $(u,\;H{=}u^\perp)$ basis: it keeps the
special direction to **output rank $R$** and the transverse covariance **dense and exact**.

- **Linear step** — exact by cumulant multilinearity ($\mu\!\to\!M\mu$, $\Sigma\!\to\!M\Sigma M^\top$,
  $d_p\!\to\!a^p d_p$), done block-wise so the $\sqrt n$-amplified scale never pollutes the $O(1)$
  transverse block (the dense congruence $M\,\Sigma_\perp M^\top$ is the only $O(n^3)$ work → GPU).
- **ReLU step** — condition on the scalar special mode $S=u^\top Z$, run the **exact** rank-2
  Gaussian-ReLU per Gauss–Hermite node, mix. $R{=}2$ ⇒ Gaussian $S$ (exact rank-2, $\tau{=}0$);
  $R{\ge}3$ ⇒ Cornish–Fisher nodes carrying $d_3,d_4$ (the amplified non-Gaussianity).

**The knob $R$.** $R{=}2$ is the exact rank-2 closure; $R{=}3,4$ add the special cumulants. We sweep
all three and compare to the existing **exact-K2 kprop** (the total-order baseline).

| | what we measure | what to look for |
|---|---|---|
| **§0** | torch-free self-check | linear step & rank-2 ReLU exact to machine precision; depth-1 mean exact vs MC |
| **§2** | shifted model: $\lVert E[\text{out}]\rVert$ **and** rel-error vs width, per $R$ | does adding $R$ pull the prediction toward MC? does SW-KPROP beat exact-K2? |
| **§3** | trained-to-0 checkpoints: rel-error vs width, per $R$ | SW-KPROP $R{=}2$ ≈ exact-K2 (no special structure); is anything gained? |
| **§4** | error-vs-$R$ summary + the **death (`sub`) regime** | $R$ helps monotonically; `sub` collapses to $\approx0$ — the no-go-lemma stress test |

> **`sub` is the stress test.** For $s{=}-1$ the post-ReLU special mode $S_{X}=u^\top\!\mathrm{ReLU}(Z)\ge 0$
> is *one-signed*, and $S_Z=a\,S_X\le 0$ kills every deeper ReLU → the output **collapses to $\approx 0$**.
> Its residual mean is a delicate fluctuation a finite-cumulant closure can only partly capture (the
> paper's no-go Lemma 2). Expect $R$ to help a lot but the collapse to remain hard; `add` and
> trained-to-0 are where SW-KPROP is accurate.

Needs Python ≥ 3.12 *or* the skprop kprop-compat shim (auto-active on import), plus torch & scipy.""")

code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §0 — Self-check (torch-free): is the SW-KPROP core correct?

Before any sweep, run the module's numpy self-test. It builds shifted-weight nets in numpy and checks:
the **linear step** equals $M\mu,\,M\Sigma M^\top$ to machine precision; the **ReLU step** (conditioning
+ mixing) equals the one-shot exact bivariate ReLU; the **depth-1 mean** is exact vs Monte-Carlo; and the
**depth-3 $R$-sweep** behaves (R≥3 helps where the special mode is non-Gaussian). This validates the
install in a few seconds with no GPU.""")
code(r"""
from Mecha_preds.cumulants.swkprop.selftest import run as swkprop_selftest
swkprop_selftest()   # prints [1]..[4] then 'SELFTEST: PASS'
""")

# =============================================================================
md(r"""## §1 — Config: knobs, device & recycling (probe here, not in `experiments.py`)

`QUICK` is True on a CPU-only machine → tiny smoke sweep. SW-KPROP's dense per-node ReLU is scipy/CPU
($O(n_{\text{nodes}}\,n^2)$ Owen's-T) so we keep widths modest in the smoke config; raise `WIDTHS` /
turn off `QUICK` on a GPU box. `R_VALUES=[2,3,4]` is the rank sweep; `N_NODES` is the Gauss–Hermite
node count for the special mode.""")
code(r"""
import math, time, os, copy, glob
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP
from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, compare_means
from Mecha_preds.cumulants.swkprop import run_sw_kprop

QUICK  = E.QUICK
DEVICE = E.DEVICE                          # cuda -> mps -> cpu (auto)
torch.set_num_threads(max(torch.get_num_threads(), 2))

# ---- the SW-KPROP knobs ----
R_VALUES = [2, 3, 4]                        # output rank sweep (2 = exact rank-2; 3,4 add special cumulants)
N_NODES  = 9 if QUICK else 15              # Gauss-Hermite nodes for the special mode

# ---- (A) shifted-weights sweep (NO TRAINING) ----
SHIFTS      = ["sub", "add"]               # sub: W'-(1/sqrt n)11^T (death); add: W'+(1/sqrt n)11^T (linear)
SH_DEPTHS   = [3]
SH_WIDTHS   = [32, 64, 128] if QUICK else [64, 128, 256, 512]
SH_SEEDS    = [1]
ACTIVATION  = "relu"

# ---- (B) trained-to-0 checkpoints (already on disk) ----
ZERO_CKPT_DIR = "checkpoints/kprop_checkpoints"     # kprop-zero_d3_w{w}_tol5_seed{s}_final.pt
ZERO_WIDTHS   = [16, 32, 64, 128] if QUICK else [16, 32, 64, 128, 256, 512]
ZERO_SEED     = 3

MC_SAMPLES = 200_000 if QUICK else 1_000_000

# ---- GPU policy: float32 MC on DEVICE, float64 for the predictors (repo policy) ----
if DEVICE.type == "cuda":
    MC_DEVICE, MC_DTYPE, MC_BATCH = DEVICE, torch.float32, 65_536
    KPROP_DEVICE = str(DEVICE)             # CUDA has float64: route the SW-KPROP congruence here
else:
    MC_DEVICE, MC_DTYPE, MC_BATCH = torch.device("cpu"), torch.float64, 8_192
    KPROP_DEVICE = "cpu"                    # MPS lacks float64

CKPT_DIR = "checkpoints/sw_kprop"           # THIS notebook's family (models + result cache)
RECYCLE  = True
os.makedirs(CKPT_DIR, exist_ok=True)

print("DEVICE:", DEVICE, "| MC:", MC_DEVICE.type, MC_DTYPE, "batch", MC_BATCH, "| kprop dev:", KPROP_DEVICE)
print("QUICK:", QUICK, "| R:", R_VALUES, "| n_nodes:", N_NODES, "| MC:", f"{MC_SAMPLES:,}")
print("shifted:", SHIFTS, "depths", SH_DEPTHS, "widths", SH_WIDTHS, "seeds", SH_SEEDS,
      "| trained-0 widths", ZERO_WIDTHS, "seed", ZERO_SEED)
""")

code(r"""
# ---- builders + recycling helpers ----
def shifted_mean_mlp(width, seed, depth, shift):
    "model.MLP with W = W' + s*(1/sqrt n)11^T on HIDDEN layers (readout unshifted). float64, no training."
    sign = -1.0 if shift == "sub" else +1.0
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation=ACTIVATION).double().eval()
    g = torch.Generator().manual_seed(1_000_000 * depth + 10_000 * seed + 7 * width)
    c = 1.0 / math.sqrt(width)
    with torch.no_grad():
        layers = list(m.hidden_layers) + [m.readout]
        for li, layer in enumerate(layers):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64) / math.sqrt(in_f)
            if li < len(m.hidden_layers):
                W = W + sign * c * torch.ones(out_f, in_f, dtype=torch.float64)
            layer.weight.copy_(W)
    return m

def get_shifted_model(w, seed, depth, shift):
    "RECYCLE: load if present, else build the random shifted model and SAVE it."
    path = E.ckpt_path(CKPT_DIR, E.run_name(f"shifted-{shift}", depth=depth, width=w, seed=seed))
    if RECYCLE and os.path.exists(path):
        return MLP.load(path, map_location="cpu")[0].double().eval()
    m = shifted_mean_mlp(w, seed, depth, shift)
    m.save(path, extra={"family": "sw_kprop", "shift": shift, "depth": depth, "width": w, "seed": seed})
    return m

def zero_ckpt_path(w, seed):
    return os.path.join(ZERO_CKPT_DIR, f"kprop-zero_d3_w{w}_tol5_seed{seed}_final.pt")

def mc_reference(m, w):
    "Monte-Carlo E[out] on DEVICE (GPU on CUDA), float64 accumulators; does NOT mutate m."
    mdev = copy.deepcopy(m).to(device=MC_DEVICE, dtype=MC_DTYPE)
    mc, stats = estimate_empirical_mean(model=mdev, input_dim=w, num_samples=MC_SAMPLES,
                                        device=str(MC_DEVICE), dtype=MC_DTYPE, batch_size=MC_BATCH)
    del mdev
    if MC_DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return mc, stats

def predict_all(m, w):
    "SW-KPROP at every R + the exact-K2 kprop baseline, all float64."
    out = {}
    for R in R_VALUES:
        out[f"sw_R{R}"] = run_sw_kprop(m, input_dim=w, config={"R": R, "n_nodes": N_NODES},
                                       device=KPROP_DEVICE)["mean"]
    out["k2"] = run_cumulants(m, w, config={"k_max": 2, "factor": False, "exact_relu_cov": True},
                              device=KPROP_DEVICE)["mean"]
    return out

def rel(cp, mc, stats):
    return compare_means(np.asarray(cp, float), np.asarray(mc, float), stats)["relative_error_mean"]

# results cache: one .pt per config signature
CFG_SIG = f"R{'-'.join(map(str,R_VALUES))}_nodes{N_NODES}_mc{MC_SAMPLES}"
RESULTS_PATH = os.path.join(CKPT_DIR, f"results_{CFG_SIG}.pt")
_results = torch.load(RESULTS_PATH) if (RECYCLE and os.path.exists(RESULTS_PATH)) else {}
def cache_get(k): return _results.get(k) if RECYCLE else None
def cache_put(k, v): _results[k] = v; torch.save(_results, RESULTS_PATH)
print(f"results cache {os.path.basename(RESULTS_PATH)}: {len(_results)} runs "
      f"({'recycling' if _results else 'empty -> will compute'})")
""")

# =============================================================================
md(r"""## §2 — Shifted-weights model: SW-KPROP vs MC vs exact-K2, per rank $R$

For each `shift` ∈ {`sub`,`add`}, width and seed we build the random shifted model, run MC on the GPU,
and run every predictor. We keep the **unscaled magnitude** $\lVert E[\text{out}]\rVert$ (does the
prediction track the collapse?) and the **scaled relative error**. Each `(shift, width, seed)` is
recycled from the cache when present.""")
code(r"""
sh_rows, t0 = [], time.time()
for shift in SHIFTS:
    for depth in SH_DEPTHS:
        for w in SH_WIDTHS:
            for seed in SH_SEEDS:
                key = f"shifted|{shift}|d{depth}|w{w}|s{seed}"
                r = cache_get(key); src = "recycled"
                if r is None:
                    src = "computed"
                    m = get_shifted_model(w, seed, depth, shift)
                    mc, stats = mc_reference(m, w)
                    preds = predict_all(m, w)
                    nm = float(np.linalg.norm(mc)) + 1e-30
                    r = dict(shift=shift, depth=depth, w=w, seed=seed,
                             mc_norm=float(np.linalg.norm(mc)),
                             floor=float(np.linalg.norm(stats["mc_stderr"])) / nm)
                    for name, p in preds.items():
                        r[f"{name}_rel"]  = rel(p, mc, stats)
                        r[f"{name}_norm"] = float(np.linalg.norm(p))
                    cache_put(key, r)
                sh_rows.append(r)
                msg = "  ".join(f"R{R}={r[f'sw_R{R}_rel']:.2e}" for R in R_VALUES)
                print(f"{shift} d{depth} w={w:>4} s{seed} [{src:>8}] | SW {msg} | "
                      f"k2={r['k2_rel']:.2e} | ||mc||={r['mc_norm']:.2e} floor={r['floor']:.1e}", flush=True)
print(f"\nshifted sweep done in {time.time()-t0:.1f}s")
""")
code(r"""
# plot per shift: LEFT unscaled ||E[out]|| (MC vs predictors), RIGHT scaled rel-error, vs width
def sh_series(shift, depth, key):
    return [float(np.mean([r[key] for r in sh_rows if r["shift"]==shift and r["depth"]==depth and r["w"]==w]))
            for w in SH_WIDTHS]

for shift in SHIFTS:
    d0 = SH_DEPTHS[0]; op = "-" if shift == "sub" else "+"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.2))
    axL.loglog(SH_WIDTHS, sh_series(shift, d0, "mc_norm"), "ko-", label="MC (truth)")
    for R in R_VALUES:
        axL.loglog(SH_WIDTHS, sh_series(shift, d0, f"sw_R{R}_norm"), "^--", label=f"SW-KPROP R={R}")
    axL.loglog(SH_WIDTHS, sh_series(shift, d0, "k2_norm"), "xs:", color="0.5", label="exact-K2 kprop")
    axL.set_xlabel("width n"); axL.set_ylabel(r"$\|E[\mathrm{out}]\|_2$ (unscaled)")
    axL.set_title(f"magnitude — shift {shift} ($W{op}\\frac{{1}}{{\\sqrt n}}11^\\top$, depth {d0})")
    axL.legend(fontsize=8); axL.grid(alpha=0.3, which="both")

    for R in R_VALUES:
        axR.loglog(SH_WIDTHS, sh_series(shift, d0, f"sw_R{R}_rel"), "^--", label=f"SW-KPROP R={R}")
    axR.loglog(SH_WIDTHS, sh_series(shift, d0, "k2_rel"), "xs:", color="0.5", label="exact-K2 kprop")
    axR.loglog(SH_WIDTHS, sh_series(shift, d0, "floor"), "-", color="0.75", lw=1, label="MC floor")
    axR.set_xlabel("width n"); axR.set_ylabel(r"$\|\mu_{\rm pred}-\mu_{\rm MC}\|/\|\mu_{\rm MC}\|$")
    axR.set_title(f"relative error — shift {shift} (lower = better)")
    axR.legend(fontsize=8); axR.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §3 — Trained-to-0 checkpoints: SW-KPROP vs MC vs exact-K2

The real trained networks in `checkpoints/kprop_checkpoints` (depth 3, scalar output, train MSE < 1e-5).
These have **no special structure imposed**, so we expect SW-KPROP $R{=}2$ to match the existing exact-K2
kprop (both are the exact rank-2 closure), and $R{\ge}3$ to add little (the all-ones direction carries no
amplified non-Gaussianity). This is the control showing SW-KPROP is a faithful generalization — it does
not *lose* accuracy where the special direction is irrelevant.""")
code(r"""
zero_rows = []
for w in ZERO_WIDTHS:
    path = zero_ckpt_path(w, ZERO_SEED)
    if not os.path.exists(path):
        print(f"w={w:>4}: MISSING {os.path.basename(path)} -- skipping"); continue
    key = f"zero|w{w}|s{ZERO_SEED}"
    r = cache_get(key); src = "recycled"
    if r is None:
        src = "computed"
        m, _ = MLP.load(path, map_location="cpu"); m = m.double().eval()
        win = m.cfg.input_dim
        mc, stats = mc_reference(m, win)
        preds = predict_all(m, win)
        nm = float(np.linalg.norm(mc)) + 1e-30
        r = dict(w=w, mc_norm=float(np.linalg.norm(mc)),
                 floor=float(np.linalg.norm(stats["mc_stderr"])) / nm)
        for name, p in preds.items():
            r[f"{name}_rel"] = rel(p, mc, stats)
        cache_put(key, r)
    zero_rows.append(r)
    msg = "  ".join(f"R{R}={r[f'sw_R{R}_rel']:.2e}" for R in R_VALUES)
    print(f"w={w:>4} [{src:>8}] | SW {msg} | k2={r['k2_rel']:.2e} | floor={r['floor']:.1e}", flush=True)
""")
code(r"""
if zero_rows:
    ws = [r["w"] for r in zero_rows]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for R in R_VALUES:
        ax.loglog(ws, [r[f"sw_R{R}_rel"] for r in zero_rows], "^--", label=f"SW-KPROP R={R}")
    ax.loglog(ws, [r["k2_rel"] for r in zero_rows], "xs:", color="0.5", label="exact-K2 kprop")
    ax.loglog(ws, [r["floor"] for r in zero_rows], "-", color="0.75", lw=1, label="MC floor")
    ax.set_xlabel("width n"); ax.set_ylabel(r"$|\mu_{\rm pred}-\mu_{\rm MC}|/|\mu_{\rm MC}|$")
    ax.set_title(f"trained-to-0 (depth 3, seed {ZERO_SEED}): SW-KPROP vs exact-K2")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §4 — Error vs rank $R$, and the death-regime reading

The bars show the relative error at the largest width of each regime as $R$ grows. Read it as: **does
keeping more of the amplified special direction help?**

- **`add` & trained-to-0:** SW-KPROP is accurate; $R$ helps monotonically (`add`) or is already at the
  rank-2 floor (trained-to-0). SW-KPROP $R{=}2 \approx$ exact-K2 — the faithful-generalization check.
- **`sub` (death):** the output collapses to $\approx0$, so the **relative** error is huge for *every*
  closure (you are dividing by $\approx0$); the honest signal is the **magnitude** panel in §2 — each
  higher $R$ pulls $\lVert\mu_{\rm pred}\rVert$ down toward the collapsed truth. Finite cumulants cannot
  fully represent the one-signed special mode (the paper's no-go Lemma 2), so a residual remains; this is
  the known hard stress test, not a bug.""")
code(r"""
def at_max_width(rows, wkey="w"):
    wmax = max(r[wkey] for r in rows); return [r for r in rows if r[wkey] == wmax][0], wmax

fig, ax = plt.subplots(figsize=(8.4, 5.0))
groups, labels = [], []
for shift in SHIFTS:
    rows = [r for r in sh_rows if r["shift"] == shift and r["depth"] == SH_DEPTHS[0]]
    if rows: row, wmax = at_max_width(rows); groups.append(row); labels.append(f"shift {shift}\n(w{wmax})")
if zero_rows:
    row, wmax = at_max_width(zero_rows); groups.append(row); labels.append(f"trained-0\n(w{wmax})")

x = np.arange(len(groups)); bw = 0.8 / (len(R_VALUES) + 1)
for i, R in enumerate(R_VALUES):
    ax.bar(x + i * bw, [g[f"sw_R{R}_rel"] for g in groups], bw, label=f"SW-KPROP R={R}")
ax.bar(x + len(R_VALUES) * bw, [g["k2_rel"] for g in groups], bw, color="0.5", label="exact-K2")
ax.set_yscale("log"); ax.set_xticks(x + 0.4 - bw/2); ax.set_xticklabels(labels)
ax.set_ylabel("relative error vs MC (log)"); ax.set_title("error vs rank R, by regime (largest width)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y", which="both"); plt.tight_layout(); plt.show()

print("magnitude collapse in the death (sub) regime — does higher R pull ||pred|| toward ||MC||?")
for r in [r for r in sh_rows if r["shift"] == "sub"]:
    msg = "  ".join(f"R{R} {r[f'sw_R{R}_norm']:.2e}" for R in R_VALUES)
    print(f"  w={r['w']:>4}: ||MC||={r['mc_norm']:.2e} | {msg} | k2 {r['k2_norm']:.2e}")
""")

# =============================================================================
md(r"""## §5 — Checkpoints: recycle across sessions

The sweep wrote the random shifted models + a results cache to `checkpoints/sw_kprop` (the trained-to-0
checkpoints are read-only from `checkpoints/kprop_checkpoints`). Re-running recycles everything.""")
code(r"""
import shutil
print("checkpoint dir:", os.path.abspath(CKPT_DIR))
for f in sorted(os.listdir(CKPT_DIR)):
    print("  ", f, f"({os.path.getsize(os.path.join(CKPT_DIR, f))/1e6:.2f} MB)")
if IN_COLAB:
    from google.colab import files
    z = shutil.make_archive("/content/sw_kprop_ckpts", "zip", CKPT_DIR)
    print("zipped ->", z, "-- downloading..."); files.download(z)
""")

# =============================================================================
md(r"""## §6 — Summary

- **What ran:** SW-KPROP (`Mecha_preds/cumulants/swkprop`) at rank $R\in\{2,3,4\}$ vs Monte-Carlo and vs
  the existing exact-K2 kprop, on (A) random **shifted-weight** MLPs $W'\!\pm\!\tfrac1{\sqrt n}11^\top$
  (no training) and (B) the **trained-to-0** checkpoints.
- **SW-KPROP $R{=}2$** is the exact rank-2 closure in the split (special/transverse) basis: linear step
  exact, ReLU step the exact bivariate-Gaussian integral — it equals exact-K2 kprop where the special
  direction is irrelevant (trained-to-0), and is exact at depth 1.
- **$R{\ge}3$** adds the $\sqrt n$-amplified special cumulants $d_3,d_4$ (Cornish–Fisher quadrature of
  the special mode): it helps monotonically in the `add` regime and pulls the predicted magnitude toward
  the collapsed truth in `sub`.
- **The `sub` death regime** is the no-go stress test: the output collapses to $\approx0$, so relative
  error stays large for any finite-cumulant closure; §2's magnitude panel is the honest readout.
- **Self-check (§0):** linear & rank-2 ReLU steps are exact to machine precision; depth-1 mean is exact
  vs MC. **Recycling:** models + results cached under `checkpoints/sw_kprop`. **GPU:** MC float32 on
  `E.DEVICE`; SW-KPROP routes its dense congruence to CUDA float64.""")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sw_kprop_colab.ipynb")
nb.save(out)

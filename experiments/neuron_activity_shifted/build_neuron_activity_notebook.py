"""Generates neuron_activity_shifted_colab.ipynb (valid nbformat-4 JSON).

SELF-CONTAINED notebook (analysis code is INLINE -- no repo tool): which hidden neurons are active
(ReLU pre-activation z>0) as inputs propagate through a mean-shifted MLP, how HOT each neuron runs,
how much its activation VARIES, and whether the SAME neurons fire every input.

Headline view: per hidden layer, a HEAT PLOT of the neurons --
   * firing rate   P_X[z_i > 0]      (how often the neuron is active = its "heat" of activating)
   * mean activation E[a_i]          (how hot it runs, a_i = ReLU(z_i))
   * variance       Var(a_i)         (how much its activation fluctuates across inputs)

Plus a scalar across-input consistency (do the same neurons fire?) to quantify the hypothesis:
   * RANDOM shifted model -> a SMALL active set that VARIES every input (low consistency);
   * TRAINED model (trained-to-0 checkpoints) -> the SAME neurons fire (high consistency).

Models (depth 3, square, no bias): random hidden-shifted M = W' + B (NO training) for
B in {-(1/sqrt n)11^T (big-sub: death/sparse), +(1/sqrt n)11^T (big-add: linear/dense),
e1 e1^T (small-e1), (1/n)11^T (small-ones), 0 (plain)}; vs trained-to-0 kprop_checkpoints (d3).

REPO POLICIES: notebook owns its knobs + CKPT_DIR; recycling (shifted models + per-neuron stats
cached; trained checkpoints read-only); GPU (activation forwards on E.DEVICE).
Run:  python "experiments/neuron_activity_shifted/build_neuron_activity_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# Hidden-neuron **heat & variance** maps — shifted random vs trained

As inputs $X\sim\mathcal N(0,I)$ propagate through a **mean-shifted** MLP, we look, per hidden layer
and per neuron, at three things (a neuron is **active** when its pre-activation $z>0$):

- **firing rate** $\;p_i = P_X[z_i>0]\;$ — how often it activates (its *heat* of activating);
- **mean activation** $\;\mathbb E[a_i]\;$ — how hot it runs ($a_i=\mathrm{ReLU}(z_i)$);
- **variance** $\;\mathrm{Var}(a_i)\;$ — how much its activation fluctuates input-to-input.

The headline is a **heat plot of the neurons of each hidden layer** for these quantities. We also
score the across-input **consistency** of the active set to test the hypothesis:

| model | expected | |
|---|---|---|
| **random shifted** (no training) | a **small** active set (esp. the $-\tfrac1{\sqrt n}11^\top$ death regime) that **varies** every input | low consistency, diffuse heat |
| **trained** (trained-to-0 ckpts) | the **same** neurons fire every input | high consistency, stable hot columns |

**Consistency** is $(O-f)/(1-f)$ with $f=\tfrac1n\sum_i p_i$ and $O=\frac{\sum p_i^2}{\sum p_i}$ (the
probability a neuron active for one input is active for an independent input): $0$ = active set as
random as chance allows, $1$ = one fixed always-on set.

**Models** (depth 3, square, no bias): random hidden-shifted $M=W'+B$ (no training) for
$B\in\{\mp\tfrac1{\sqrt n}11^\top,\ e_1e_1^\top,\ \tfrac1n11^\top,\ 0\}$, vs the **trained-to-0**
`kprop_checkpoints`. *The analysis is inline below — no repo import.*""")

code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 — Inline analysis code (active = $z>0$; per-neuron heat + variance; consistency)

`activity_stats` streams Gaussian inputs through a model and accumulates, per hidden layer, each
neuron's firing rate, mean activation and activation variance. `consistency_from_p` turns the
firing-rate vector into the across-input active-set consistency. A two-line sanity check confirms
the metric (a fixed always-on set → consistency 1; a sparse-but-random-each-input set → 0).""")
code(r"""
import math, time, os, copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP

@torch.no_grad()
def activity_stats(m, w, N, batch, device, dtype):
    "Stream N ~N(0,I) inputs; per hidden layer return per-neuron firing rate p, mean act, var act."
    md_ = copy.deepcopy(m).to(device=device, dtype=dtype).eval()
    depth = m.cfg.depth
    fire = {l: torch.zeros(w, dtype=torch.float64) for l in range(depth)}   # sum 1[z>0]
    asum = {l: torch.zeros(w, dtype=torch.float64) for l in range(depth)}   # sum a
    asq  = {l: torch.zeros(w, dtype=torch.float64) for l in range(depth)}   # sum a^2  (a = ReLU(z))
    seen = 0
    while seen < N:
        b = min(batch, N - seen)
        _, acts = md_(torch.randn(b, w, device=device, dtype=dtype), return_activations=True)
        for l in range(depth):
            z, a = acts["pre"][l], acts["post"][l]
            fire[l] += (z > 0).sum(0).double().cpu()
            asum[l] += a.double().sum(0).cpu()
            asq[l]  += (a.double() ** 2).sum(0).cpu()
        seen += b
    out = {}
    for l in range(depth):
        p = (fire[l] / seen).numpy()
        mean_act = (asum[l] / seen).numpy()
        var_act = np.clip((asq[l] / seen).numpy() - mean_act ** 2, 0.0, None)
        out[l] = {"p": p.astype("float32"), "mean_act": mean_act.astype("float32"),
                  "var_act": var_act.astype("float32")}
    return out

def consistency_from_p(p):
    "Across-input active-set summary from the firing-rate vector p (all numpy, no masks needed)."
    p = np.asarray(p, dtype=np.float64); n = p.size
    f = float(p.mean()); sp = float(p.sum())
    O = float((p * p).sum() / (sp + 1e-30))                 # P(active-now -> active-next)
    cons = float((O - f) / (1.0 - f)) if f < 1.0 - 1e-12 else 1.0
    return {"active_fraction": f, "overlap_ratio": O, "consistency": cons,
            "always_on": int((p > 0.95).sum()), "dead": int((p < 0.05).sum()),
            "variable": int(((p >= 0.05) & (p <= 0.95)).sum())}

# --- sanity check (no torch needed) ---
rng = np.random.default_rng(0); N, n = 4000, 128
p_fixed = np.array([1.0]*8 + [0.0]*(n-8))                    # 8 always-on -> consistency 1
M_var = np.zeros((N, n));                                    # each input a random 8-subset -> ~0
for r in range(N): M_var[r, rng.choice(n, 8, replace=False)] = 1.0
print("sanity: fixed always-on set    -> consistency", round(consistency_from_p(p_fixed)["consistency"], 3))
print("sanity: sparse-varying (rand 8) -> consistency", round(consistency_from_p(M_var.mean(0))["consistency"], 3),
      "(f =", round(float(M_var.mean()), 3), ")")
""")

# =============================================================================
md(r"""## §2 — Config: models, device & recycling (probe here, not in `experiments.py`)

Shifted models are built (and recycled) at depth 3, matched widths to the trained-to-0
`kprop_checkpoints`. `N_INPUTS` Gaussian inputs estimate the per-neuron statistics.""")
code(r"""
QUICK  = E.QUICK
DEVICE = E.DEVICE
torch.set_num_threads(max(torch.get_num_threads(), 2))

DEPTH       = 3
WIDTHS      = [32, 64] if QUICK else [64, 128, 256]
SHIFT_SEEDS = [1, 2]
TRAIN_SEEDS = [3, 4]                        # seeds present in kprop_checkpoints
N_INPUTS    = 20_000 if QUICK else 50_000
ACTIVATION  = "relu"

SHIFTS = ["plain", "big-sub", "big-add", "small-e1", "small-ones"]
ZERO_CKPT_DIR = "checkpoints/kprop_checkpoints"     # kprop-zero_d3_w{w}_tol5_seed{s}_final.pt (read-only)
MODEL_NAMES = SHIFTS + ["trained-0"]

IN_DEVICE = DEVICE
IN_DTYPE  = torch.float32 if DEVICE.type == "cuda" else torch.float64
IN_BATCH  = 16_384 if DEVICE.type == "cuda" else 4_096

CKPT_DIR = "checkpoints/neuron_activity_shifted"
RECYCLE  = True
os.makedirs(CKPT_DIR, exist_ok=True)

print("DEVICE:", DEVICE, "| inputs:", IN_DTYPE, "batch", IN_BATCH)
print("QUICK:", QUICK, "| depth:", DEPTH, "| widths:", WIDTHS, "| N_inputs:", f"{N_INPUTS:,}")
print("models:", MODEL_NAMES, "| shift seeds", SHIFT_SEEDS, "| trained seeds", TRAIN_SEEDS)
""")

code(r"""
# ---- builders + loaders (recycling) ----
def shift_matrix(kind, n):
    "B added to every HIDDEN weight matrix (W' ~ N(0,1/n))."
    if kind == "plain":      return torch.zeros(n, n, dtype=torch.float64)
    if kind == "big-sub":    return -(1.0 / math.sqrt(n)) * torch.ones(n, n, dtype=torch.float64)
    if kind == "big-add":    return +(1.0 / math.sqrt(n)) * torch.ones(n, n, dtype=torch.float64)
    if kind == "small-ones": return (1.0 / n) * torch.ones(n, n, dtype=torch.float64)
    if kind == "small-e1":
        B = torch.zeros(n, n, dtype=torch.float64); B[0, 0] = 1.0; return B
    raise ValueError(kind)

def shifted_mlp(kind, width, seed, depth):
    "model.MLP with M = W' + B(kind) on HIDDEN layers (readout unshifted). float64, NO training."
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation=ACTIVATION).double().eval()
    g = torch.Generator().manual_seed(1_000_000 * depth + 10_000 * seed + 7 * width + abs(hash(kind)) % 97)
    B = shift_matrix(kind, width)
    with torch.no_grad():
        layers = list(m.hidden_layers) + [m.readout]
        for li, layer in enumerate(layers):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64) / math.sqrt(in_f)
            if li < len(m.hidden_layers):
                W = W + B
            layer.weight.copy_(W)
    return m

def get_shifted(kind, w, seed, depth):
    path = E.ckpt_path(CKPT_DIR, E.run_name(f"shift-{kind}", depth=depth, width=w, seed=seed))
    if RECYCLE and os.path.exists(path):
        return MLP.load(path, map_location="cpu")[0].double().eval()
    m = shifted_mlp(kind, w, seed, depth)
    m.save(path, extra={"family": "neuron_activity_shifted", "kind": kind,
                        "depth": depth, "width": w, "seed": seed})
    return m

def get_trained(w, seed):
    "load a trained-to-0 checkpoint (read-only); None if absent."
    path = os.path.join(ZERO_CKPT_DIR, f"kprop-zero_d3_w{w}_tol5_seed{seed}_final.pt")
    return MLP.load(path, map_location="cpu")[0].double().eval() if os.path.exists(path) else None

def get_model(name, w, seed):
    return get_trained(w, seed) if name == "trained-0" else get_shifted(name, w, seed, DEPTH)

# per-neuron stats cache (one .pt per config)
RESULTS_PATH = os.path.join(CKPT_DIR, f"stats_N{N_INPUTS}_d{DEPTH}.pt")
_results = torch.load(RESULTS_PATH) if (RECYCLE and os.path.exists(RESULTS_PATH)) else {}
def cache_get(k): return _results.get(k) if RECYCLE else None
def cache_put(k, v): _results[k] = v; torch.save(_results, RESULTS_PATH)
print(f"stats cache {os.path.basename(RESULTS_PATH)}: {len(_results)} runs",
      "(recycling)" if _results else "(empty -> will compute)")
""")

# =============================================================================
md(r"""## §3 — Compute per-neuron statistics for every model (recycled)

For each model (shifted regimes + trained-to-0) at each width/seed we stream `N_INPUTS` Gaussian
inputs and record, per hidden layer, every neuron's firing rate / mean activation / variance, plus
the scalar consistency. Watch the *last hidden layer*, where the shift has compounded most.""")
code(r"""
def seeds_for(name): return TRAIN_SEEDS if name == "trained-0" else SHIFT_SEEDS

rows, t0 = [], time.time()
for name in MODEL_NAMES:
    for w in WIDTHS:
        for seed in seeds_for(name):
            key = f"{name}|w{w}|s{seed}"
            r = cache_get(key); src = "recycled"
            if r is None:
                m = get_model(name, w, seed)
                if m is None:
                    continue
                src = "computed"
                stats = activity_stats(m, w, N_INPUTS, IN_BATCH, IN_DEVICE, IN_DTYPE)
                cons = {l: consistency_from_p(stats[l]["p"]) for l in stats}
                r = {"name": name, "w": w, "seed": seed, "stats": stats, "cons": cons}
                cache_put(key, r)
            rows.append(r)
            L = DEPTH - 1; c = r["cons"][L]
            print(f"{name:>10} w={w:>4} s{seed} [{src:>8}] | last layer: f={c['active_fraction']:.3f} "
                  f"consistency={c['consistency']:.3f} always_on={c['always_on']} dead={c['dead']}", flush=True)
print(f"\ndone in {time.time()-t0:.1f}s ({len(rows)} model runs; recycled ones instant)")
""")

# =============================================================================
md(r"""## §4 — Heat plots of the hidden-layer neurons (the headline)

For one instance of each model (largest width, first seed) we draw, per hidden layer (rows) and
neuron (columns), three heat maps: **firing rate** $p_i$, **mean activation** $\mathbb E[a_i]$ (the
"heat"), and **activation variance** $\mathrm{Var}(a_i)$. Neurons are shown in index order. Read the
contrast: the **death** regime (`big-sub`) goes mostly cold in the deep layers with only scattered
warm neurons (a *different* few each input → diffuse, low-variance-but-sparse); **trained-0** is
expected to show **stable hot columns** (the same neurons carrying signal); `big-add` runs hot
everywhere (linear regime).""")
code(r"""
W0 = WIDTHS[-1]
def instance_stats(name):
    rs = [r for r in rows if r["name"] == name and r["w"] == W0]
    return rs[0]["stats"] if rs else None

def heat_plot(name):
    st = instance_stats(name)
    if st is None:
        print(f"{name}: no model at width {W0}"); return
    depth = len(st)
    P  = np.stack([st[l]["p"]        for l in range(depth)])   # (depth, w)
    Mn = np.stack([st[l]["mean_act"] for l in range(depth)])
    Vr = np.stack([st[l]["var_act"]  for l in range(depth)])
    panels = [("firing rate  P[z>0]", P, "hot", None),
              ("mean activation  E[a]", Mn, "inferno", None),
              ("variance  Var(a)", Vr, "viridis", None)]
    fig, axes = plt.subplots(3, 1, figsize=(min(13.5, W0 / 22 + 4), 5.4), constrained_layout=True)
    for ax, (lab, data, cmap, _) in zip(axes, panels):
        im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_yticks(range(depth)); ax.set_yticklabels([f"L{l}" for l in range(depth)])
        ax.set_ylabel("layer"); ax.set_title(lab, fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.026, pad=0.01)
    axes[-1].set_xlabel("neuron index")
    fig.suptitle(f"{name}   (width {W0}, depth {DEPTH})", fontsize=11, fontweight="bold")
    plt.show()

for name in MODEL_NAMES:
    heat_plot(name)
""")

# =============================================================================
md(r"""## §5 — Sparsity vs consistency: random shifted vs trained (quantified)

The heat maps, summarized: **active fraction** $f$ (how many fire) and **consistency** (do the *same*
ones fire), last hidden layer, mean over seeds. The hypothesis predicts random-shifted models at
**low consistency** (a different small subset each input) and **trained-0** at **high consistency**.""")
code(r"""
def agg(name, w, field):
    L = DEPTH - 1
    vals = [r["cons"][L][field] for r in rows if r["name"] == name and r["w"] == w]
    return float(np.mean(vals)) if vals else float("nan")

print(f"LAST hidden layer (depth {DEPTH}), mean over seeds:\n")
print(f"{'model':>11} | " + "  ".join(f"w{w}: {'f':>5} {'consist':>7}" for w in WIDTHS))
print("-" * (13 + 18 * len(WIDTHS)))
for nm in MODEL_NAMES:
    print(f"{nm:>11} |" + "  ".join(f"     {agg(nm,w,'active_fraction'):>5.3f} {agg(nm,w,'consistency'):>7.3f}"
                                     for w in WIDTHS))

colors = {"plain":"0.6","big-sub":"tab:red","big-add":"tab:orange",
          "small-e1":"tab:purple","small-ones":"tab:blue","trained-0":"tab:green"}
xs = np.arange(len(MODEL_NAMES))
fig, (axF, axC) = plt.subplots(1, 2, figsize=(13.5, 4.8))
axF.bar(xs, [agg(nm, W0, "active_fraction") for nm in MODEL_NAMES], color=[colors[n] for n in MODEL_NAMES])
axF.set_ylabel("active fraction  f"); axF.set_title(f"sparsity — how many fire (w{W0}, last layer)")
axC.bar(xs, [agg(nm, W0, "consistency") for nm in MODEL_NAMES], color=[colors[n] for n in MODEL_NAMES])
axC.set_ylabel("consistency  (O-f)/(1-f)"); axC.set_title("do the SAME neurons fire across inputs?")
for ax in (axF, axC):
    ax.set_xticks(xs); ax.set_xticklabels(MODEL_NAMES, rotation=30, ha="right"); ax.grid(alpha=0.3, axis="y")
axC.axhline(0, color="k", lw=0.6)
plt.tight_layout(); plt.show()
print("\nf = active fraction (sparsity) ; consistency in [0,1]: 0 = a different subset every input, 1 = fixed set.")
print("Note: consistency's (1-f) normalization degenerates in the DENSE regime (big-add, f~1, all on)")
print("-> there read 'always_on' (all neurons fire every input) instead.")
""")

# =============================================================================
md(r"""## §6 — Checkpoints: recycle across sessions

Shifted models + the per-neuron stats cache live in `checkpoints/neuron_activity_shifted`
(trained-to-0 checkpoints are read-only from `checkpoints/kprop_checkpoints`). Re-running recycles.""")
code(r"""
import shutil
print("checkpoint dir:", os.path.abspath(CKPT_DIR))
for f in sorted(os.listdir(CKPT_DIR)):
    print("  ", f, f"({os.path.getsize(os.path.join(CKPT_DIR, f))/1e6:.2f} MB)")
if IN_COLAB:
    from google.colab import files
    z = shutil.make_archive("/content/neuron_activity_ckpts", "zip", CKPT_DIR)
    print("zipped ->", z, "-- downloading..."); files.download(z)
""")

# =============================================================================
md(r"""## §7 — Summary

- **What ran (all inline — no repo tool):** per-neuron **firing rate / mean activation / variance**
  for random hidden-shifted MLPs $M=W'+B$ (no training) and the **trained-to-0** `kprop_checkpoints`
  (d3), via `activity_stats` over `N_INPUTS` Gaussian inputs; active $\equiv z>0$.
- **Heat plots (§4):** per hidden layer, the neurons' firing rate, mean activation (heat) and
  variance. The `big-sub` death regime goes cold/sparse in deep layers; `big-add` runs hot
  everywhere; **trained-0** is expected to show stable hot columns (the same neurons carry signal).
- **Consistency (§5):** active fraction $f$ and across-input consistency $(O-f)/(1-f)$. Hypothesis —
  random shifted = small but **varying** active set (low consistency); trained = **same** neurons
  (high consistency). `big-add` is the dense control (read `always_on`, not consistency).
- **Recycling:** shifted models + per-neuron stats in `checkpoints/neuron_activity_shifted`; re-runs
  load instead of recomputing. **GPU:** activation forwards on `E.DEVICE`.""")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuron_activity_shifted_colab.ipynb")
nb.save(out)

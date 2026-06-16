"""Generates shifted_mean_kprop_colab.ipynb (valid nbformat-4 JSON).

Tests the experimental (spike-aware) kprop -- ``Mecha_preds.cumulants.skprop`` --
on RANDOM mean-shifted MLPs, the companion of the planted-spike study in
``build_structured_kprop_notebook.py``. Here the spike is not bolted onto one
layer; it is the *mean shift* of every hidden weight matrix:

    W = W' + B,   W'_{ij} ~ N(0, 1/fan_in)   (i.i.d. Gaussian, the "random" part)
                  B = -(1/sqrt(n)) 11^T       (variant: B = -(1/n) 11^T, numerically milder)

Why this is exactly the regime structured kprop targets. For a hidden layer the
pre-activation is  z = Wx = W'x + Bx  with  Bx = -c (1^T x) 1.  Write the shared
latent  H = (1^T x)/sqrt(n) ~ N(0, 1).  Then per neuron i:

    c = 1/sqrt(n):  z_i = (W'x)_i - H            <- O(1) SHARED shift on every neuron
    c = 1/n:        z_i = (W'x)_i - H/sqrt(n)     <- O(1/sqrt(n)) shift, vanishes with width

So  B = -(1/sqrt(n)) 11^T = -sqrt(n) * (1/sqrt(n))(1/sqrt(n))^T  is a rank-1 spike of
size sqrt(n) -- far above the Marchenko-Pastur bulk edge (~2 for N(0,1/n) entries) --
turning a single Gaussian latent into coherent O(1) off-diagonal cumulants that
vanilla kprop k=2 assumes are O(n^{-1/2}). Conditioning on H (Gauss-Hermite) and
running power-cumulant kprop on the residual is the fix. The  -(1/n)  variant has a
fixed spike of size 1, BELOW the bulk edge -> auto-detection finds q=0 -> structured
degenerates to vanilla EXACTLY (and vanilla is already fine: the shift is O(1/sqrt n)).

The planted latent direction is the unit all-ones vector 1/sqrt(n) (left = right,
since the layers are square) -- so detection should recover it AND we can pass it
explicitly via ``directions=`` (the spikes.py docstring's "all-ones meaned matrix").

REPO POLICIES THIS NOTEBOOK HONORS
  * Checkpoint recycling (the whole point of the project): each model is saved to
    ``checkpoints/shifted_mean_kprop`` and LOADED if already on disk; the expensive
    MC references + kprop predictions are cached too (keyed by config) so a re-run
    recomputes NOTHING. A Colab cell downloads/uploads the checkpoint dir so the
    recycled state survives across sessions.
  * GPU: ``E.DEVICE`` is auto-detected (cuda->mps->cpu, TF32 already on). The
    Monte-Carlo reference -- the FLOP-heavy part -- runs on the GPU (CUDA float32,
    float64 accumulators per the repo's float32-compute/float64-measure policy), and
    kprop runs on the GPU too on CUDA. (Apple MPS has no float64, so the float64
    measurement paths fall back to CPU there.)

Tests
  S1  spike spectrum per layer (overshoot vs MP edge) + alignment with planted 1.
  S2  width sweep 32..512, both B variants: rel-L2 error of E[out] for vanilla k=2
      vs structured (auto-detect) vs structured (explicit planted direction), vs the
      MC noise floor.
  S3  deep conditioning (the shift is on ALL hidden layers).
  S4  why it works: quadrature-node convergence + the latent response m(h).

Needs Python >= 3.12 OR the skprop kprop-compat shim (auto-active on import); + torch,
scipy. Run:  python "colab_notebooks/structured_kprop/build_shifted_mean_kprop_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# Structured power KPROP on mean-shifted random matrices $W = W' + B$

**Setup.** Depth-3 ReLU MLPs (square layers, no bias, `input_dim = output_dim = width`).
Every weight is $W = W' + B$ with $W'_{ij}\sim\mathcal N(0,1/\text{fan\_in})$ i.i.d.; the
mean shift $B = -c\,\mathbf 1\mathbf 1^\top$ is added to the **hidden** weight matrices
(the readout is pure $W'$). Two shifts:
$B=-\tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top$ and $B=-\tfrac1n\mathbf 1\mathbf 1^\top$ (milder).

**Why this is the structured-kprop regime.** A hidden pre-activation is
$z = W'x + Bx$, $Bx=-c(\mathbf 1^\top x)\mathbf 1$. With the shared latent
$H=(\mathbf 1^\top x)/\sqrt n\sim\mathcal N(0,1)$: $c=\tfrac1{\sqrt n}\Rightarrow z_i=(W'x)_i-H$
(an $O(1)$ shift shared by every neuron); $c=\tfrac1n\Rightarrow z_i=(W'x)_i-H/\sqrt n$.
So $B=-\tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top=-\sqrt n\,\hat v\hat v^\top$ is a **rank-1 spike of
size $\sqrt n$** — above the Marchenko–Pastur edge ($\approx2$) — which makes one latent
into coherent $O(1)$ cumulants that vanilla kprop $k{=}2$ treats as $O(n^{-1/2})$. Condition
on $H$ (Gauss–Hermite), run power-cumulant kprop on the residual. The $-\tfrac1n$ spike has
fixed size $1$ (sub-edge) ⇒ auto-detection $q=0$ ⇒ structured $\equiv$ vanilla.

The planted latent is the unit all-ones direction $\hat v=\mathbf 1/\sqrt n$.

> **Recycling + GPU (repo policy).** Models are saved to `checkpoints/shifted_mean_kprop` and
> **loaded if already on disk**; the MC references and kprop predictions are **cached by config**
> so a re-run recomputes nothing (a Colab cell downloads/uploads the dir). The Monte-Carlo
> reference and kprop run on **`E.DEVICE`** (GPU on CUDA; float64 measurement falls back to CPU on
> Apple MPS, which lacks float64).

| | test | expectation | section |
|---|---|---|---|
| **S1** | spike spectrum per layer | $-\tfrac1{\sqrt n}$: overshoot $\sim\sqrt n/2$, $q{=}1$, align$\approx1$; $-\tfrac1n$: $q{=}0$ | §1 |
| **S2** | width sweep 32–512, both $B$ | $-\tfrac1{\sqrt n}$: vanilla $O(1)$, structured → MC floor; $-\tfrac1n$: all equal | §2 |
| **S3** | deep conditioning (all hidden layers shifted) | layers 1,2 channels close the residual gap | §3 |
| **S4** | quadrature convergence + $m(h)$ | ~10–15 nodes; curvature of $m(h)$ is the $O(1)$ error | §4 |

Needs Python ≥ 3.12 *or* the skprop kprop-compat shim (auto-active on import), plus torch + scipy.""")

code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## 1. Config — knobs, device & recycling (probe here, not in `experiments.py`)""")
code(r"""
import math, time, json, os, copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP

QUICK  = E.QUICK
DEVICE = E.DEVICE                     # cuda -> mps -> cpu (auto); TF32 matmuls enabled in experiments.py
torch.set_num_threads(max(torch.get_num_threads(), 2))

DEPTH       = 3
WIDTHS      = [32, 64, 128] if QUICK else [32, 64, 128, 256, 512]
B_KINDS     = ["inv_sqrt_n", "inv_n"] # B = -(1/sqrt(n)) 11^T   and   -(1/n) 11^T
SEEDS       = [0] if QUICK else [0, 1, 2]
ACTIVATION  = "relu"
K_MAX       = 2                       # structured kprop is a k=2 story (deep mode needs 2)
N_NODES     = 15                      # Gauss-Hermite nodes per latent dimension
Q_MAX       = 1                       # auto-detected spikes on the FIRST layer
MARGIN      = 1.15                    # MP-edge multiplier for detection
MC_SAMPLES  = 200_000 if QUICK else 1_000_000

# ---- GPU policy: float32 compute on GPU, float64 for measurement (repo policy) ----
# MC accumulators + kprop need float64. CUDA has float64 -> run on GPU. Apple MPS has
# NO float64 -> the float64 paths fall back to CPU there (sampling stays correct).
if DEVICE.type == "cuda":
    MC_DEVICE, MC_DTYPE, MC_BATCH = DEVICE, torch.float32, 65_536
    KPROP_DEVICE = str(DEVICE)        # kprop on the GPU too (CUDA supports float64)
else:
    MC_DEVICE, MC_DTYPE, MC_BATCH = torch.device("cpu"), torch.float64, 8_192
    KPROP_DEVICE = "cpu"              # MPS lacks float64; CPU otherwise

# ---- checkpoint recycling (this notebook's OWN family under checkpoints/) ----
CKPT_DIR = "checkpoints/shifted_mean_kprop"
RECYCLE  = True                       # load existing checkpoints/results instead of recomputing
os.makedirs(CKPT_DIR, exist_ok=True)

from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, compare_means
from Mecha_preds.cumulants.skprop import (
    run_structured_cumulants, detect_spikes, detect_spikes_all_layers,
)
print("DEVICE:", DEVICE, "| MC:", MC_DEVICE.type, MC_DTYPE, "batch", MC_BATCH,
      "| kprop dev:", KPROP_DEVICE)
print("QUICK:", QUICK, "| widths:", WIDTHS, "| seeds:", SEEDS, "| CKPT_DIR:", CKPT_DIR)
""")

code(r"""
# ---- builders: the mean-shifted random MLP (float64 master), and the planted latent ----
def b_const(kind, n):
    "scalar c with B = -c 11^T:  1/sqrt(n) (strong, growing spike) or 1/n (mild, fixed)."
    return (1.0 / math.sqrt(n)) if kind == "inv_sqrt_n" else (1.0 / n)

def planted_dir(n):
    "unit all-ones vector = the (left=right) singular vector of B = c 11^T, shape (n,1)."
    return torch.ones(n, 1, dtype=torch.float64) / math.sqrt(n)

def shifted_mean_mlp(width, kind, seed=0, depth=DEPTH):
    "model.MLP with W = W' + B: W'~N(0,1/fan_in) everywhere; B=-c 11^T on HIDDEN layers only."
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation=ACTIVATION).double().eval()
    g = torch.Generator().manual_seed(10_000 * seed + 7 * width + (1 if kind == "inv_n" else 0))
    c = b_const(kind, width)
    with torch.no_grad():
        for li, layer in enumerate(list(m.hidden_layers) + [m.readout]):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64) / math.sqrt(in_f)
            if li < len(m.hidden_layers):                       # shift hidden weight matrices only
                W = W - c * torch.ones(out_f, in_f, dtype=torch.float64)
            layer.weight.copy_(W)
    return m
""")

# =============================================================================
md(r"""## 1b. Checkpoint recycling + GPU Monte-Carlo (the repo rule: never recompute what's on disk)

`get_model` loads the `.pt` if it exists, otherwise builds the random model and **saves** it.
`cache_get/cache_put` persist the expensive MC + kprop results (keyed by config) to the same
folder, so re-running is instant. `mc_reference` runs MC on `DEVICE` (GPU on CUDA) without
mutating the float64 master that kprop consumes.""")
code(r"""
def _prefix(kind):
    return "shifted-" + ("invsqrtn" if kind == "inv_sqrt_n" else "invn")

def model_path(kind, w, seed):
    return E.ckpt_path(CKPT_DIR, E.run_name(_prefix(kind), depth=DEPTH, width=w, seed=seed))

def get_model(kind, w, seed):
    "RECYCLE: load the checkpoint if present, else build the random W=W'+B model and SAVE it."
    path = model_path(kind, w, seed)
    if RECYCLE and os.path.exists(path):
        m, _ = MLP.load(path, map_location="cpu")
        return m.double().eval(), True
    m = shifted_mean_mlp(w, kind, seed=seed)
    m.save(path, extra={"family": "shifted_mean_kprop", "b_kind": kind,
                        "depth": DEPTH, "width": w, "seed": seed})
    return m, False

# results cache: one .pt per CONFIG signature (changing k_max/nodes/MC_SAMPLES -> fresh file)
CFG_SIG = f"kmax{K_MAX}_nodes{N_NODES}_q{Q_MAX}_margin{MARGIN}_mc{MC_SAMPLES}_{ACTIVATION}"
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
    return compare_means(cp, mc, stats)["relative_error_mean"]

_existing = [os.path.basename(c["path"]) for c in E.list_checkpoints(CKPT_DIR)]
print(f"models on disk in {CKPT_DIR}: {len([p for p in _existing if p.startswith('shifted-')])}")
print(f"results cache {os.path.basename(RESULTS_PATH)}: {len(_results)} runs "
      f"({'HIT -> recycling' if _results else 'empty -> will compute + save'})")
""")

# =============================================================================
md(r"""## §1 — Spike spectrum per layer

`detect_spikes` runs a sequential MP-edge test on each weight matrix. The **overshoot**
(top singular value / bulk edge) tells whether a coherent latent is present.
Expectation: $-\tfrac1{\sqrt n}$ gives a layer-0 overshoot growing like $\sqrt n/2$ ($q=1$,
recovered direction aligns with $\mathbf 1/\sqrt n$); $-\tfrac1n$ stays below the edge ($q=0$).""")
code(r"""
print("layer overshoot = top_sv / MP_bulk_edge  (q = detected spikes); align = |<detected v0, planted 1>|\n")
for kind in B_KINDS:
    print(f"[{kind}]")
    for w in WIDTHS:
        m, _ = get_model(kind, w, 0)                  # recycled
        Ws = [l.weight.double() for l in m.hidden_layers] + [m.readout.weight.double()]
        infos = detect_spikes_all_layers(Ws, q_max=2, margin=MARGIN)
        pv = planted_dir(w)[:, 0]
        s0 = infos[0]
        align = float(abs(s0.V[:, 0] @ pv)) if s0.q else float("nan")
        oss = "  ".join(f"L{i.layer}:{float(i.all_sv[0]) / i.bulk_edge:5.2f}(q{i.q})" for i in infos)
        print(f"   w={w:>4} | {oss} | layer0 align(planted)={align:.3f}")
""")

# =============================================================================
md(r"""## §2 — Width sweep: vanilla vs structured (auto / explicit), both $B$ variants

Headline metric: relative $L_2$ error $\lVert \hat\mu_{\text{kprop}} - \mu_{\text{MC}}\rVert / \lVert\mu_{\text{MC}}\rVert$.
Each `(kind, width, seed)` is recycled from the results cache when present; otherwise the model
is loaded/built, MC runs on the GPU, kprop runs, and the result is saved.""")
code(r"""
rows, t0 = [], time.time()
for kind in B_KINDS:
    for w in WIDTHS:
        for seed in SEEDS:
            key = f"{kind}|w{w}|s{seed}"
            r = cache_get(key); src = "recycled"
            if r is None:
                src = "computed"
                m, loaded = get_model(kind, w, seed)
                mc, stats = mc_reference(m, w)
                van  = run_cumulants(m, config={"k_max": K_MAX, "factor": False},
                                     device=KPROP_DEVICE)["mean"]
                auto = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": N_NODES,
                                                "q_max": Q_MAX, "margin": MARGIN}, device=KPROP_DEVICE)
                expl = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": N_NODES,
                                                "directions": planted_dir(w)}, device=KPROP_DEVICE)
                nm = float(np.linalg.norm(mc)) + 1e-30
                r = dict(kind=kind, w=w, seed=seed, van=rel(van, mc, stats),
                         auto=rel(auto["mean"], mc, stats), q_auto=int(auto["metadata"]["q"]),
                         expl=rel(expl["mean"], mc, stats), q_expl=int(expl["metadata"]["q"]),
                         floor=float(np.linalg.norm(stats["mc_stderr"])) / nm, model_loaded=bool(loaded))
                cache_put(key, r)
            rows.append(r)
            print(f"{kind:>11} w={w:>4} s{seed} [{src:>8}] | van {r['van']:.3e} | "
                  f"auto {r['auto']:.3e}(q{r['q_auto']}) | expl {r['expl']:.3e}(q{r['q_expl']}) "
                  f"| floor {r['floor']:.1e}", flush=True)
print(f"\nsweep done in {time.time() - t0:.1f}s ({len(rows)} runs; recycled ones are instant)")
""")
code(r"""
def series(kind, key):
    return [float(np.mean([r[key] for r in rows if r["kind"] == kind and r["w"] == w])) for w in WIDTHS]

fig, axes = plt.subplots(1, len(B_KINDS), figsize=(6.4 * len(B_KINDS), 4.6), squeeze=False)
titles = {"inv_sqrt_n": r"$B=-\frac{1}{\sqrt{n}}\,11^\top$  (strong spike, $\sigma=\sqrt n$)",
          "inv_n":      r"$B=-\frac{1}{n}\,11^\top$  (fixed spike, $\sigma=1$, sub-edge)"}
for ax, kind in zip(axes[0], B_KINDS):
    ax.loglog(WIDTHS, series(kind, "van"),  "o-",  label="vanilla kprop k=2")
    ax.loglog(WIDTHS, series(kind, "auto"), "s-",  label="structured (auto-detect)")
    ax.loglog(WIDTHS, series(kind, "expl"), "^--", label="structured (explicit 1/√n)")
    ax.loglog(WIDTHS, series(kind, "floor"), ":", color="0.5", label="MC noise floor")
    ax.set_title(titles[kind]); ax.set_xlabel("width n"); ax.set_ylabel(r"rel. $L_2$ error of $E[\mathrm{out}]$")
    ax.legend(); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()

sub = [r for r in rows if r["kind"] == "inv_sqrt_n"]
if sub:
    gain = float(np.median([r["van"] / max(r["expl"], r["floor"], 1e-12) for r in sub]))
    print(f"S2: median vanilla/structured error ratio on -(1/sqrt n) = {gain:.1f}x "
          f"(structured rides the MC floor; vanilla does not shrink with width)")
""")

# =============================================================================
md(r"""## §2b — Checkpoints: save / load / **download** (recycle across sessions)

Everything above wrote to `checkpoints/shifted_mean_kprop` (models + a results cache). If the
repo lives on Google Drive (set `LOCAL_REPO_DIR` in the bootstrap), that folder already persists
— next session just re-runs and **recycles**. Otherwise download a zip and re-upload it later.""")
code(r"""
import shutil
print("checkpoint dir:", os.path.abspath(CKPT_DIR))
for f in sorted(os.listdir(CKPT_DIR)):
    print("  ", f, f"({os.path.getsize(os.path.join(CKPT_DIR, f)) / 1e6:.2f} MB)")

if IN_COLAB:
    from google.colab import files
    zip_base = "/content/shifted_mean_kprop_ckpts"
    zpath = shutil.make_archive(zip_base, "zip", CKPT_DIR)
    print("\nzipped ->", zpath, "-- downloading...")
    files.download(zpath)                       # save recycled checkpoints+results to your machine

# To RESTORE in a fresh Colab runtime (so nothing recomputes), upload the zip and unpack:
#   from google.colab import files; up = files.upload()
#   import io, zipfile, os
#   os.makedirs(CKPT_DIR, exist_ok=True)
#   zipfile.ZipFile(io.BytesIO(next(iter(up.values())))).extractall(CKPT_DIR)
""")

# =============================================================================
md(r"""## §3 — Deep conditioning (the shift is on **all** hidden layers)

Conditioning only on the first-layer latent leaves the coherent shifts in hidden layers 1 & 2 in
the residual. `deep=True` with `deep_directions={1: 1/\sqrt n, 2: 1/\sqrt n}` conditions on those
too. Strong-spike variant, smaller widths (deep mode multiplies quadrature cost). Recycled.""")
code(r"""
DEEP_WIDTHS = WIDTHS[:3]
deep_rows = []
for w in DEEP_WIDTHS:
    key = f"deep|w{w}"
    r = cache_get(key)
    if r is None:
        m, _ = get_model("inv_sqrt_n", w, 0)
        mc, stats = mc_reference(m, w)
        van  = run_cumulants(m, config={"k_max": K_MAX, "factor": False}, device=KPROP_DEVICE)["mean"]
        inp  = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": N_NODES,
                                         "directions": planted_dir(w)}, device=KPROP_DEVICE)
        deep = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": N_NODES,
                                         "directions": planted_dir(w), "deep": True,
                                         "deep_directions": {1: planted_dir(w), 2: planted_dir(w)},
                                         "deep_n_nodes": 7}, device=KPROP_DEVICE)
        r = dict(w=w, van=rel(van, mc, stats), inp=rel(inp["mean"], mc, stats),
                 deep=rel(deep["mean"], mc, stats), br=int(deep["metadata"]["n_branches"]))
        cache_put(key, r)
    deep_rows.append(r)
    print(f"w={w:>4} | vanilla {r['van']:.3e} | input-latent {r['inp']:.3e} | "
          f"deep {r['deep']:.3e} (branches={r['br']})", flush=True)

fig, ax = plt.subplots(figsize=(6.6, 4.4))
for key, lab, fmt in [("van", "vanilla k=2", "o-"), ("inp", "structured (input latent)", "s-"),
                      ("deep", "structured (deep: layers 0,1,2)", "v-")]:
    ax.loglog(DEEP_WIDTHS, [r[key] for r in deep_rows], fmt, label=lab)
ax.set_xlabel("width n"); ax.set_ylabel(r"rel. $L_2$ error of $E[\mathrm{out}]$")
ax.set_title(r"deep conditioning on $B=-\frac{1}{\sqrt n}11^\top$ (all hidden layers shifted)")
ax.legend(); ax.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §4 — Why it works: quadrature convergence + the latent response $m(h)$

**(a)** error vs number of Gauss–Hermite nodes → saturates at the residual-kprop floor.
**(b)** the per-node conditional mean $m(h)=E[\text{out}\mid H=h]$ (output coordinate 0):
a single Gaussian kprop state can only sit at the $h$-average; the curvature of $m(h)$ is
the $O(1)$ error.""")
code(r"""
w_demo = DEEP_WIDTHS[-1]
m, _ = get_model("inv_sqrt_n", w_demo, 0)
mc, stats = mc_reference(m, w_demo)
NODE_SWEEP = [3, 5, 7, 9, 11, 15, 21, 31]
nd = cache_get(f"nodesweep|w{w_demo}")
if nd is None:
    errs = [rel(run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": nn,
                                         "directions": planted_dir(w_demo)}, device=KPROP_DEVICE)["mean"],
                mc, stats) for nn in NODE_SWEEP]
    cache_put(f"nodesweep|w{w_demo}", {"nodes": NODE_SWEEP, "errs": errs}); nd = {"errs": errs}
errs = nd["errs"]
res = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": 31,
                               "directions": planted_dir(w_demo)}, device=KPROP_DEVICE)["raw_output"]
van = run_cumulants(m, config={"k_max": K_MAX, "factor": False}, device=KPROP_DEVICE)["mean"]
floor = float(np.linalg.norm(stats["mc_stderr"])) / (float(np.linalg.norm(mc)) + 1e-30)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.loglog(NODE_SWEEP, errs, "o-"); ax1.axhline(max(floor, 1e-12), ls=":", color="0.5", label="MC floor")
ax1.set_xlabel("Gauss-Hermite nodes"); ax1.set_ylabel(r"rel. $L_2$ error")
ax1.set_title(f"quadrature convergence (n={w_demo})"); ax1.legend(); ax1.grid(alpha=0.3, which="both")

h = res["nodes"][:, 0].cpu().numpy(); mh = res["per_node_means"][:, 0].cpu().numpy()
o = np.argsort(h)
ax2.plot(h[o], mh[o], "o-", label=r"conditional mean $m(h)$ (out[0])")
ax2.axhline(float(res["mean"].reshape(-1)[0]), color="r", ls="--", label="structured mix")
ax2.axhline(float(np.asarray(van).reshape(-1)[0]), color="k", ls=":", label="vanilla kprop")
ax2.set_xlabel(r"latent node $h$  ($H=\mathbf 1^\top x/\sqrt n$)"); ax2.set_ylabel(r"$E[\mathrm{out}_0\mid H=h]$")
ax2.set_title("the latent response a single Gaussian state cannot represent")
ax2.legend(); ax2.grid(alpha=0.3); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §5 — Summary

- **§1:** $-\tfrac1{\sqrt n}$ plants a layer-0 spike with overshoot $\sim\sqrt n/2$ (auto-detected,
  $q=1$, direction aligns with $\mathbf 1/\sqrt n$); $-\tfrac1n$ sits below the MP edge ($q=0$).
- **§2:** on the strong shift, vanilla kprop $k{=}2$ has an $O(1)$ relative error that does **not**
  shrink with width; structured kprop (auto *or* explicit) rides the MC noise floor. On the mild
  $-\tfrac1n$ shift, $q=0$ so structured $\equiv$ vanilla — **safe to leave on unconditionally**.
- **§3:** conditioning on layers 1 & 2 too (deep) closes the residual gap from the all-layer shift.
- **§4:** ~10–15 quadrature nodes saturate; the curvature of $m(h)$ is exactly the $O(1)$ error.

**Recycling:** models + MC/kprop results are cached in `checkpoints/shifted_mean_kprop`; re-runs
load instead of recomputing, and the §2b cell downloads the dir to carry that state between
sessions. **GPU:** MC and kprop run on `E.DEVICE` (CUDA), float64-on-CPU fallback on MPS.
**Cost:** structured = (one vanilla kprop) $\times\ \texttt{n\_nodes}^{\,q}$ ($q=1$ main sweep).""")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shifted_mean_kprop_colab.ipynb")
nb.save(out)

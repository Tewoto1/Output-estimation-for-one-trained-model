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

These are RANDOM, seed-reproducible models (no training) -> no checkpoints to
recycle; we rebuild them from the seed, exactly as the planted-spike notebook does.

Tests
  S1  spike spectrum per layer (overshoot vs MP edge) + alignment with planted 1.
  S2  width sweep 32..512, both B variants: rel-L2 error of E[out] for vanilla k=2
      vs structured (auto-detect) vs structured (explicit planted direction), vs the
      MC noise floor. Expectation: -(1/sqrt n) -> vanilla O(1), structured -> floor;
      -(1/n) -> all small and equal (q=0).
  S3  deep conditioning (the shift is on ALL hidden layers): condition on the planted
      channel in layers 1 & 2 too -- does it beat input-latent-only?
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
(the readout is pure $W'$, no shift). Two shifts:

$$B = -\tfrac{1}{\sqrt n}\,\mathbf 1\mathbf 1^\top \qquad\text{and}\qquad B = -\tfrac{1}{n}\,\mathbf 1\mathbf 1^\top\ \text{(numerically milder)}.$$

**Why this is the structured-kprop regime.** A hidden pre-activation is
$z = W'x + Bx$ with $Bx = -c(\mathbf 1^\top x)\mathbf 1$. With the shared latent
$H = (\mathbf 1^\top x)/\sqrt n \sim \mathcal N(0,1)$,

$$c=\tfrac1{\sqrt n}:\; z_i = (W'x)_i - H\ \ \text{(an }O(1)\text{ shift shared by every neuron)};\qquad
c=\tfrac1{n}:\; z_i = (W'x)_i - H/\sqrt n\ \ (O(n^{-1/2})).$$

So $B=-\tfrac{1}{\sqrt n}\mathbf 1\mathbf 1^\top = -\sqrt n\,\hat v\hat v^\top$ is a **rank-1 spike of
size $\sqrt n$** — far above the Marchenko–Pastur bulk edge ($\approx 2$ for $\mathcal N(0,1/n)$
entries) — which turns one Gaussian latent into coherent $O(1)$ off-diagonal cumulants that
vanilla kprop $k{=}2$ treats as $O(n^{-1/2})$. The fix: condition on $H$ (Gauss–Hermite),
run power-cumulant kprop on the residual. The $-\tfrac1n$ spike has fixed size $1$ (below the
edge) ⇒ auto-detection returns $q=0$ ⇒ structured $\equiv$ vanilla, and vanilla is already fine.

The planted latent is the **unit all-ones direction** $\hat v=\mathbf 1/\sqrt n$ (left $=$ right,
square layers) — so detection should recover it and we can also pass it explicitly via
`directions=`.

| | test | expectation | section |
|---|---|---|---|
| **S1** | spike spectrum per layer (overshoot vs MP edge) | $-\tfrac1{\sqrt n}$: overshoot $\sim\sqrt n/2$, $q{=}1$, align$\approx1$; $-\tfrac1n$: $q{=}0$ | §1 |
| **S2** | width sweep 32–512, both $B$ | $-\tfrac1{\sqrt n}$: vanilla $O(1)$, structured → MC floor; $-\tfrac1n$: all equal | §2 |
| **S3** | deep conditioning (shift is on *all* hidden layers) | layers 1,2 channels close the residual gap | §3 |
| **S4** | quadrature convergence + latent response $m(h)$ | ~10–15 nodes suffice; curvature of $m(h)$ is the $O(1)$ error | §4 |

These are random, seed-reproducible models (no training) → nothing to checkpoint;
rebuilt from the seed each run. Needs Python ≥ 3.12 *or* the skprop kprop-compat shim
(auto-active on import), plus torch + scipy.""")

code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## 1. Config — this notebook's knobs (probe here, not in `experiments.py`)""")
code(r"""
import math, time, json
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP

QUICK = E.QUICK                       # CPU-only -> smaller sweeps
torch.set_num_threads(max(torch.get_num_threads(), 2))

DEPTH       = 3                       # hidden Linear+ReLU blocks (4 weight matrices)
WIDTHS      = [32, 64, 128] if QUICK else [32, 64, 128, 256, 512]
B_KINDS     = ["inv_sqrt_n", "inv_n"] # B = -(1/sqrt(n)) 11^T   and   -(1/n) 11^T
SEEDS       = [0] if QUICK else [0, 1, 2]
ACTIVATION  = "relu"

K_MAX       = 2                       # structured kprop is a k=2 story (deep mode needs 2)
N_NODES     = 15                      # Gauss-Hermite nodes per latent dimension
Q_MAX       = 1                       # auto-detected spikes on the FIRST layer
MARGIN      = 1.15                    # MP-edge multiplier for detection
MC_SAMPLES  = 200_000 if QUICK else 1_000_000

from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, compare_means
from Mecha_preds.cumulants.skprop import (
    run_structured_cumulants, detect_spikes, detect_spikes_all_layers,
)
print("QUICK:", QUICK, "| widths:", WIDTHS, "| seeds:", SEEDS, "| k_max:", K_MAX)
""")

code(r"""
# ---- builders: the mean-shifted random MLP, and the planted latent direction ----
def b_const(kind, n):
    "scalar c with B = -c 11^T:  1/sqrt(n) (strong, growing spike) or 1/n (mild, fixed)."
    return (1.0 / math.sqrt(n)) if kind == "inv_sqrt_n" else (1.0 / n)

def planted_dir(n):
    "unit all-ones vector = the (left=right) singular vector of B = c 11^T, shape (n,1)."
    return torch.ones(n, 1, dtype=torch.float64) / math.sqrt(n)

def shifted_mean_mlp(width, kind, seed=0, depth=DEPTH):
    "Study model.MLP with W = W' + B: W'~N(0,1/fan_in) everywhere; B=-c 11^T on HIDDEN layers only."
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation=ACTIVATION).double().eval()
    g = torch.Generator().manual_seed(10_000 * seed + 7 * width + (1 if kind == "inv_n" else 0))
    c = b_const(kind, width)
    with torch.no_grad():
        layers = list(m.hidden_layers) + [m.readout]
        for li, layer in enumerate(layers):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64) / math.sqrt(in_f)
            if li < len(m.hidden_layers):                       # shift hidden weight matrices only
                W = W - c * torch.ones(out_f, in_f, dtype=torch.float64)
            layer.weight.copy_(W)
    return m

# sanity: the shared-latent identity z_i = (W'x)_i - (c*sqrt(n)) * H   (H = 1^T x / sqrt(n))
_m = shifted_mean_mlp(64, "inv_sqrt_n", seed=0)
print("built", type(_m).__name__, "depth", _m.cfg.depth, "io", _m.cfg.input_dim, "->", _m.cfg.output_dim,
      "| #hidden weight matrices shifted:", len(_m.hidden_layers))
""")

# =============================================================================
md(r"""## §1 — Spike spectrum per layer

`detect_spikes` runs a sequential MP-edge test on each weight matrix. The **overshoot**
(top singular value / bulk edge) tells whether a coherent latent is present.
Expectation: $B=-\tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top$ gives a layer-0 overshoot growing like
$\sqrt n/2$ with detected $q=1$ and a recovered direction that aligns with the planted
$\mathbf 1/\sqrt n$; $B=-\tfrac1n\mathbf 1\mathbf 1^\top$ stays below the edge ($q=0$).""")
code(r"""
print("layer overshoot = top_sv / MP_bulk_edge  (q = detected spikes); align = |<detected v0, planted 1>|\n")
spec = {}
for kind in B_KINDS:
    print(f"[{kind}]")
    for w in WIDTHS:
        m = shifted_mean_mlp(w, kind, seed=0)
        Ws = [l.weight.double() for l in m.hidden_layers] + [m.readout.weight.double()]
        infos = detect_spikes_all_layers(Ws, q_max=2, margin=MARGIN)
        pv = planted_dir(w)[:, 0]
        s0 = infos[0]
        align = float(abs(s0.V[:, 0] @ pv)) if s0.q else float("nan")
        spec[(kind, w)] = dict(overshoot=[float(i.all_sv[0]) / i.bulk_edge for i in infos],
                               q=[i.q for i in infos], align=align)
        oss = "  ".join(f"L{i.layer}:{float(i.all_sv[0]) / i.bulk_edge:5.2f}(q{i.q})" for i in infos)
        print(f"   w={w:>4} | {oss} | layer0 align(planted)={align:.3f}")
""")

# =============================================================================
md(r"""## §2 — Width sweep: vanilla vs structured (auto / explicit), both $B$ variants

Headline metric: relative $L_2$ error $\lVert \hat\mu_{\text{kprop}} - \mu_{\text{MC}}\rVert / \lVert\mu_{\text{MC}}\rVert$.
`auto` = `run_structured_cumulants` with MP detection (`q_max=1`); `explicit` passes the
known planted direction $\mathbf 1/\sqrt n$ via `directions=`.""")
code(r"""
def rel(cp, mc, stats):
    return compare_means(cp, mc, stats)["relative_error_mean"]

rows = []
t0 = time.time()
for kind in B_KINDS:
    for w in WIDTHS:
        for seed in SEEDS:
            m = shifted_mean_mlp(w, kind, seed=seed)
            mc, stats = estimate_empirical_mean(model=m, input_dim=w, num_samples=MC_SAMPLES)
            van  = run_cumulants(m, config={"k_max": K_MAX, "factor": False})["mean"]
            auto = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": N_NODES,
                                                       "q_max": Q_MAX, "margin": MARGIN})
            expl = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": N_NODES,
                                                       "directions": planted_dir(w)})
            mc_norm = float(np.linalg.norm(mc)) + 1e-30
            rows.append(dict(kind=kind, w=w, seed=seed,
                             van=rel(van, mc, stats),
                             auto=rel(auto["mean"], mc, stats), q_auto=auto["metadata"]["q"],
                             expl=rel(expl["mean"], mc, stats), q_expl=expl["metadata"]["q"],
                             floor=float(np.linalg.norm(stats["mc_stderr"])) / mc_norm))
            r = rows[-1]
            print(f"{kind:>11} w={w:>4} s{seed} | van {r['van']:.3e} | auto {r['auto']:.3e}(q{r['q_auto']}) "
                  f"| expl {r['expl']:.3e}(q{r['q_expl']}) | floor {r['floor']:.1e}", flush=True)
print(f"\nsweep done in {time.time() - t0:.1f}s, {len(rows)} runs")
""")
code(r"""
def series(kind, key):
    "mean over seeds of `key` at each width, for one B variant."
    return [float(np.mean([r[key] for r in rows if r["kind"] == kind and r["w"] == w])) for w in WIDTHS]

fig, axes = plt.subplots(1, len(B_KINDS), figsize=(6.4 * len(B_KINDS), 4.6), squeeze=False)
titles = {"inv_sqrt_n": r"$B=-\frac{1}{\sqrt{n}}\,11^\top$  (strong spike, $\sigma=\sqrt n$)",
          "inv_n":      r"$B=-\frac{1}{n}\,11^\top$  (fixed spike, $\sigma=1$, sub-edge)"}
for ax, kind in zip(axes[0], B_KINDS):
    ax.loglog(WIDTHS, series(kind, "van"),  "o-", label="vanilla kprop k=2")
    ax.loglog(WIDTHS, series(kind, "auto"), "s-", label="structured (auto-detect)")
    ax.loglog(WIDTHS, series(kind, "expl"), "^--", label="structured (explicit 1/√n)")
    ax.loglog(WIDTHS, series(kind, "floor"), ":", color="0.5", label="MC noise floor")
    ax.set_title(titles[kind]); ax.set_xlabel("width n"); ax.set_ylabel(r"rel. $L_2$ error of $E[\mathrm{out}]$")
    ax.legend(); ax.grid(alpha=0.3, which="both")
plt.tight_layout(); plt.show()

# headline number: worst-case improvement on the strong spike
sub = [r for r in rows if r["kind"] == "inv_sqrt_n"]
if sub:
    gain = float(np.median([r["van"] / max(r["expl"], r["floor"], 1e-12) for r in sub]))
    print(f"S2: median vanilla/structured error ratio on -(1/sqrt n) = {gain:.1f}x "
          f"(structured rides the MC floor; vanilla does not shrink with width)")
""")

# =============================================================================
md(r"""## §3 — Deep conditioning (the shift is on **all** hidden layers)

Conditioning only on the first-layer latent leaves the coherent shifts in hidden layers
1 & 2 in the residual. `deep=True` with `deep_directions={1: 1/\sqrt n, 2: 1/\sqrt n}`
(their output-side all-ones channel) conditions on those too. Run on the strong-spike
variant for the smaller widths (deep mode multiplies quadrature cost by `deep_n_nodes`
per extra channel).""")
code(r"""
DEEP_WIDTHS = WIDTHS[:3]
deep_rows = []
for w in DEEP_WIDTHS:
    m = shifted_mean_mlp(w, "inv_sqrt_n", seed=0)
    mc, stats = estimate_empirical_mean(model=m, input_dim=w, num_samples=MC_SAMPLES)
    van  = run_cumulants(m, config={"k_max": K_MAX, "factor": False})["mean"]
    inp  = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": N_NODES,
                                               "directions": planted_dir(w)})
    deep = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": N_NODES,
                                               "directions": planted_dir(w), "deep": True,
                                               "deep_directions": {1: planted_dir(w), 2: planted_dir(w)},
                                               "deep_n_nodes": 7})
    deep_rows.append(dict(w=w, van=rel(van, mc, stats), inp=rel(inp["mean"], mc, stats),
                          deep=rel(deep["mean"], mc, stats), br=deep["metadata"]["n_branches"]))
    r = deep_rows[-1]
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
exactly the information the spike converts into $O(1)$ error.""")
code(r"""
w_demo = DEEP_WIDTHS[-1]
m = shifted_mean_mlp(w_demo, "inv_sqrt_n", seed=0)
mc, stats = estimate_empirical_mean(model=m, input_dim=w_demo, num_samples=MC_SAMPLES)
NODE_SWEEP = [3, 5, 7, 9, 11, 15, 21, 31]
errs = [rel(run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": nn,
                                                "directions": planted_dir(w_demo)})["mean"], mc, stats)
        for nn in NODE_SWEEP]
res = run_structured_cumulants(m, config={"k_max": K_MAX, "n_nodes": 31,
                                          "directions": planted_dir(w_demo)})["raw_output"]
van = run_cumulants(m, config={"k_max": K_MAX, "factor": False})["mean"]
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

- **§1:** $B=-\tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top$ plants a layer-0 spike whose overshoot grows
  like $\sqrt n/2$ (auto-detected, $q=1$, recovered direction aligns with $\mathbf 1/\sqrt n$);
  $B=-\tfrac1n\mathbf 1\mathbf 1^\top$ sits below the MP edge ($q=0$).
- **§2:** on the strong shift, vanilla kprop $k{=}2$ has an $O(1)$ relative error that does
  **not** shrink with width; structured kprop (auto *or* explicit) rides the MC noise floor.
  On the mild $-\tfrac1n$ shift, detection returns $q=0$ so structured $\equiv$ vanilla, and
  both are already accurate — i.e. structured kprop is **safe to leave on unconditionally**.
- **§3:** because the shift is on every hidden layer, conditioning on layers 1 & 2 too (deep
  mode) closes the residual gap that input-latent-only conditioning leaves.
- **§4:** ~10–15 quadrature nodes saturate; the curvature of $m(h)$ is precisely the $O(1)$
  error a single-Gaussian kprop state omits.

**Cost:** structured = (one vanilla kprop) $\times\ \texttt{n\_nodes}^{\,q}$; here $q=1$ for the
main sweep, with deep mode adding $\texttt{deep\_n\_nodes}$ per conditioned hidden channel.""")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shifted_mean_kprop_colab.ipynb")
nb.save(out)

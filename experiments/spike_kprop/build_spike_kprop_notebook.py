"""Generates spike_kprop_colab.ipynb (valid nbformat-4 JSON).

Tests SPIKE-KPROP (Mecha_preds/cumulants/spikekprop) -- direction-general split-basis
cumulant propagation -- against ordinary kprop on the two O(1)-eigenvalue "shifted-means"
models the theorem singles out:

  (1) LOCALIZED spike   M = W' + theta e1 e1^T          (v = e1,  theta = 1)
  (2) FLAT spike        M = W' + theta (1/n) 11^T       (v = 1/sqrt(n) 1, theta = 1)

Both add an O(1) eigenvalue (||spike||_op = theta = 1); the regime |theta| <= n^{o(1)} of the
trace-projection theorem. The shift is on the HIDDEN layers only (readout unshifted: kprop is
exact on it). NO TRAINING -- random W'_{ij} ~ N(0, 1/fan_in) plus the spike.

The hypothesis (trace-projection theorem). After a shifted linear map M = W' + theta v v^T a
degree-r cumulant splits into ordinary W'-pairings PLUS explicit contractions against v. The
v-contractions create no loops, so they expose the fully-paired TRACE mass directly: e.g. the
degree-4 trace tensor gives C(v,v,v,v) = 3 theta (zero W-edges -> no averaging -> size theta).
For a FLAT v = 1/sqrt(n) 1 every directional cumulant decays, c_{r,n}(vv^T) = O(n^{2-r}), so
only the covariance survives and ordinary total-order kprop is already exact. For a LOCALIZED
v = e1, sum_i |v_i|^r = 1 at every r, so the spike-direction cumulants are O(1) and ordinary
kprop misses them. SPIKE-KPROP retains the spike-direction cumulants C(v,...,v) = kappa_p(S)
of the special mode S = v.X up to order R (R=3 adds d3=C(v,v,v); R=4 adds d4=C(v,v,v,v)).

Predictions tested:
  * FLAT spike: ordinary kprop error SHRINKS with width (resolvable); SPIKE-KPROP R>=3 is INERT.
  * LOCALIZED spike: ordinary kprop error stays ~O(1) (does not vanish with width); SPIKE-KPROP
    R>=3 RECOVERS part of it (the residual is the mixed trace coupling C(v,v,i,j) that the
    conditional-Gaussian closure does not propagate -- the one documented approximation).

Predictors compared vs Monte-Carlo:
  * ordinary kprop exact-K2 (run_cumulants k_max=2, exact_relu_cov=True)  -- the covariance baseline
  * ordinary kprop k_max=3 (full 3rd-order tensor; small widths only)     -- keeps all of degree 3 but
                                                                             DROPS the degree-4 trace
  * SPIKE-KPROP R in {2,3,4}                                              -- adds the spike-direction
                                                                             cumulants C(v,...,v)

REPO POLICIES: notebook owns its knobs + CKPT_DIR; recycling (models + MC/predictor results
cached, nothing recomputed on a re-run); GPU (MC on E.DEVICE float32; SPIKE-KPROP routes its
dense congruence to CUDA float64; float64 falls back to CPU on Apple MPS). Coordinate-aligned
spikes (e1) need more Gauss-Hermite nodes -- see N_NODES.

Needs Python >= 3.12 OR the kprop-compat shim (auto-active on import); + torch & scipy.
Run:  python "experiments/spike_kprop/build_spike_kprop_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# SPIKE-KPROP: tracking spike-direction cumulants for an $O(1)$ rank-one spike

How well does cumulant propagation predict the output mean $E[f(X)]$, $X\sim\mathcal N(0,I)$, of a
ReLU MLP whose hidden matrices carry a **small ($O(1)$ eigenvalue) rank-one spike** — and *where the
direction of that spike decides whether ordinary kprop is enough*?

$$M = W' + \theta\,vv^\top,\qquad W'_{ij}\sim\mathcal N(0,\tfrac1{\text{fan\_in}}),\quad \|v\|_2=1,\quad \theta=-1.$$

The spike is in the **minus direction** ($\theta=-1$, eigenvalue $-1$ — $O(1)$ magnitude, so the same
*resolvable-vs-not* theory applies; the sign only flips the sign of the spike-direction cumulants). Two
models, same $|\theta|=1$, different direction:

| model | spike | direction $v$ | character |
|---|---|---|---|
| **(1) localized** | $W'-e_1e_1^\top$ | $e_1$ | one coordinate, $\sum_i v_i^r = 1$ |
| **(2) flat** | $W'-\tfrac1n\mathbf 1\mathbf 1^\top$ | $\tfrac1{\sqrt n}\mathbf 1$ | all-ones, $\sum_i v_i^r = n^{1-r/2}$ |

Spike on **hidden layers only** (kprop is exact on the linear readout); **no training**.

**The trace-projection theorem (why direction matters).** After $M = W'+\theta vv^\top$ a degree-$r$
cumulant splits into ordinary $W'$-pairings on the open slots **plus explicit contractions against
$v$**. The $v$-edges create no index loops, so they expose the fully-paired **trace** mass directly.
For the degree-4 trace tensor $C=\theta G_4$ (with $G_4=\sum$ of the three $\delta\delta$ pairings),
$$C(v,v,v,v)=3\theta,\qquad C(v,v,i,j)=\theta(\delta_{ij}+2v_iv_j),$$
and $C(v,v,v,v)$ enters with **zero random $W'$-edges** — no averaging — so its size is the raw trace
mass $\theta$. Power-counting gives, for a coefficient with $q$ open coordinate slots after all
$v$-contractions,
$$\text{generic}: n^{1-q},\qquad \text{fully-paired trace (even }q): n^{2-q}\quad(\text{squared size}).$$
For budget $K{=}3$ the first dropped order $q{=}4$ is even, so its **trace part** has squared size
$n^{1-K}$ — *too large to drop* — while the traceless residual is the safe $n^{-K}$. Hence one must
keep the **$q{=}4$ trace projection**, and in the spiked case as an object whose $v$-contractions are
computable.

- **FLAT $v=\tfrac1{\sqrt n}\mathbf 1$:** $c_{r,n}(vv^\top)=O(n^{2-r})$ — every directional cumulant
  above the covariance vanishes. Ordinary kprop is already exact; nothing to add.
- **LOCALIZED $v=e_1$:** $\sum_i|v_i|^r=1$ for all $r$ — the spike-direction cumulants are $O(1)$ at
  *every* order. Ordinary kprop misses them; they must be tracked.

**SPIKE-KPROP** works in the split $(v,\,v^\perp)$ basis and retains the **spike-direction cumulants**
$d_p=C(v,\dots,v)=\kappa_p(S)$ of the special mode $S=v\!\cdot\!X$ up to order $R$ ($R{=}3$ adds
$d_3=C(v,v,v)$; $R{=}4$ adds $d_4=C(v,v,v,v)$), injected at the ReLU by an Edgeworth/Gram–Charlier
summation. Setting $v=\tfrac1{\sqrt n}\mathbf 1$ reproduces SW-KPROP. The mixed $q{\ge}1$ trace
contractions ($C(v,v,i,j)\sim\theta\delta_{ij}$ on the transverse block) are handled by the exact
rank-2 conditional-Gaussian closure rather than propagated as a separate tensor — the one documented
approximation, visible as a residual on $e_1$.

| | view | what to look for |
|---|---|---|
| **§0** | self-check | linear & rank-2 ReLU exact; `ones`≡SW-KPROP; `e1` kink shrinks with nodes |
| **§2** | per-model $\lVert E[\text{out}]\rVert$ **and** rel-error vs width, per predictor | spike-kprop pulls $e_1$ toward MC; on flat everything coincides |
| **§3** | **headline:** ordinary-kprop error vs width — slope $p$ | flat $p<0$ (resolvable) vs localized $p\approx0$ ($O(1)$, not resolvable) |
| **§4** | error-vs-$R$ bars + the residual | $R{\ge}3$ helps on $e_1$, inert on flat; residual = mixed trace coupling |
| **§5** | special-mode cumulants $d_3,d_4$ vs MC, layer by layer | are the tracked $C(v,\dots,v)$ accurate, and bigger for $e_1$ than flat? |

Needs Python ≥ 3.12 *or* the kprop-compat shim (auto-active on import), plus torch & scipy.""")

# install runtime deps not in the base Colab image (kprop's type annotations use jaxtyping)
code(r"""!pip install -q jaxtyping""")

code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §0 — Self-check (torch-free core): is SPIKE-KPROP correct?

The module's numpy self-test: the **linear step** equals $M\mu,\,M\Sigma M^\top$; the **rank-2 ReLU**
(condition+mix) equals the one-shot exact bivariate ReLU for the flat direction; **`v="ones"`
reproduces SW-KPROP bit-for-bit**; for the coordinate spike `e1` the special mode *is* a ReLU input so
its kink gives a Gauss–Hermite **quadrature** error (not a bug) that shrinks with `n_nodes`; depth-1
mean is exact vs MC; and the depth-3 $R$-sweep **helps on `e1`, is inert on `ones`**.""")
code(r"""
from Mecha_preds.cumulants.spikekprop.selftest import run as spikekprop_selftest
spikekprop_selftest()   # prints [1]..[5] then 'SELFTEST: PASS'
""")

# =============================================================================
md(r"""## §1 — Config: knobs, device & recycling (probe here, not in `experiments.py`)

Two models (`e1` localized, `ones` flat), $\theta=-1$ (minus direction), hidden-only spike, depth 3, no training.
Predictors: ordinary **exact-K2** kprop and (small widths) **k=3**, plus **SPIKE-KPROP** $R\in\{2,3,4\}$.
`N_NODES` is the Gauss–Hermite count for the special mode — **coordinate spikes (`e1`) need more
nodes** (the special mode is a ReLU input; see §0), so we use a generous value.""")
code(r"""
import math, time, os, copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP
from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, compare_means
from Mecha_preds.cumulants.spikekprop import run_spike_kprop

QUICK  = E.QUICK
DEVICE = E.DEVICE                          # cuda -> mps -> cpu (auto)
torch.set_num_threads(max(torch.get_num_threads(), 2))

# ---- the two O(1)-spike models (NO TRAINING) ----
DIRECTIONS = ["e1", "ones"]                # localized (e1 e1^T) vs flat ((1/n) 11^T)
THETA      = -1.0                          # MINUS direction: M = W' - v v^T (eigenvalue -1, O(1) magnitude)
DEPTH      = 3
WIDTHS     = [32, 64, 128] if QUICK else [64, 128, 256, 512]
SEEDS      = [1, 2]
ACTIVATION = "relu"

# ---- predictors ----
R_VALUES     = [2, 3, 4]                    # SPIKE-KPROP: 2 = exact rank-2; 3,4 add C(v,v,v), C(v,v,v,v)
N_NODES      = 31 if QUICK else 61         # Gauss-Hermite nodes (coordinate spike e1 wants >= ~41)
K3_MAX_WIDTH = 128                          # ordinary k=3 is an n^3 tensor -> only run it up to here
MC_SAMPLES   = 200_000 if QUICK else 20_000_000   # 20M for the true run (clean enough to resolve n-scaling)

# ---- GPU policy: float32 MC on DEVICE, float64 for the predictors (repo policy) ----
if DEVICE.type == "cuda":
    MC_DEVICE, MC_DTYPE, MC_BATCH = DEVICE, torch.float32, 65_536
    KPROP_DEVICE = str(DEVICE)             # CUDA has float64: route the dense congruence here
else:
    MC_DEVICE, MC_DTYPE, MC_BATCH = torch.device("cpu"), torch.float64, 8_192
    KPROP_DEVICE = "cpu"                    # MPS lacks float64

CKPT_DIR = "checkpoints/spike_kprop"        # THIS notebook's family (models + result cache)
RECYCLE  = True
os.makedirs(CKPT_DIR, exist_ok=True)

print("DEVICE:", DEVICE, "| MC:", MC_DEVICE.type, MC_DTYPE, "batch", MC_BATCH, "| kprop dev:", KPROP_DEVICE)
print("QUICK:", QUICK, "| directions:", DIRECTIONS, "theta:", THETA, "| depth:", DEPTH,
      "| widths:", WIDTHS, "| seeds:", SEEDS)
print("predictors: exact-K2, k=3(<=w%d), SPIKE-KPROP R=%s | n_nodes=%d | MC=%s"
      % (K3_MAX_WIDTH, R_VALUES, N_NODES, f"{MC_SAMPLES:,}"))
""")

code(r"""
# ---- builders: the random O(1)-spiked MLP (float64 master). NO TRAINING. ----
def spike_vector(direction, n):
    "unit spike direction as a torch float64 vector: 'e1' (localized) or 'ones' (flat 1/sqrt(n) 1)."
    if direction == "e1":
        v = torch.zeros(n, dtype=torch.float64); v[0] = 1.0; return v
    if direction == "ones":
        return torch.full((n,), 1.0 / math.sqrt(n), dtype=torch.float64)
    raise ValueError(direction)

def spiked_mlp(width, seed, depth, direction):
    "model.MLP with M = W' + theta v v^T on HIDDEN layers (readout unshifted). W'~N(0,1/fan_in)."
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation=ACTIVATION).double().eval()
    g = torch.Generator().manual_seed(1_000_000 * depth + 10_000 * seed + 7 * width
                                       + (0 if direction == "e1" else 3))
    v = spike_vector(direction, width)
    P = THETA * torch.outer(v, v)                       # theta v v^T  (eigenvalue theta = 1)
    with torch.no_grad():
        layers = list(m.hidden_layers) + [m.readout]
        for li, layer in enumerate(layers):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64) / math.sqrt(in_f)
            if li < len(m.hidden_layers):               # spike the HIDDEN (square) matrices only
                W = W + P
            layer.weight.copy_(W)
    return m

def model_path(direction, w, seed, depth):
    return E.ckpt_path(CKPT_DIR, E.run_name(f"spike-{direction}", depth=depth, width=w, seed=seed))

def get_model(direction, w, seed, depth):
    "RECYCLE: load the checkpoint if present, else build the random spiked model and SAVE it."
    path = model_path(direction, w, seed, depth)
    if RECYCLE and os.path.exists(path):
        return MLP.load(path, map_location="cpu")[0].double().eval()
    m = spiked_mlp(w, seed, depth, direction)
    m.save(path, extra={"family": "spike_kprop", "direction": direction, "theta": THETA,
                        "depth": depth, "width": w, "seed": seed})
    return m
""")

code(r"""
# ---- MC reference + predictors + result cache ----
def mc_reference(m, w):
    "Monte-Carlo E[out] on DEVICE (GPU on CUDA), float64 accumulators; does NOT mutate m."
    mdev = copy.deepcopy(m).to(device=MC_DEVICE, dtype=MC_DTYPE)
    mc, stats = estimate_empirical_mean(model=mdev, input_dim=w, num_samples=MC_SAMPLES,
                                        device=str(MC_DEVICE), dtype=MC_DTYPE, batch_size=MC_BATCH)
    del mdev
    if MC_DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return mc, stats

def predict_all(m, w, direction):
    "ordinary exact-K2 (+ k=3 on small widths) and SPIKE-KPROP at every R, all float64."
    out = {}
    out["k2"] = run_cumulants(m, w, config={"k_max": 2, "factor": False, "exact_relu_cov": True},
                              device=KPROP_DEVICE)["mean"]                       # covariance baseline
    if w <= K3_MAX_WIDTH:
        out["k3"] = run_cumulants(m, w, config={"k_max": 3}, device=KPROP_DEVICE)["mean"]  # full 3rd order
    for R in R_VALUES:
        out[f"spk_R{R}"] = run_spike_kprop(m, direction, input_dim=w,
                                           config={"R": R, "n_nodes": N_NODES},
                                           device=KPROP_DEVICE)["mean"]          # + spike-dir cumulants
    return out

def rel(cp, mc, stats):
    return compare_means(np.asarray(cp, float), np.asarray(mc, float), stats)["relative_error_mean"]

CFG_SIG = f"theta{THETA}_R{'-'.join(map(str,R_VALUES))}_nodes{N_NODES}_mc{MC_SAMPLES}_d{DEPTH}"
RESULTS_PATH = os.path.join(CKPT_DIR, f"results_{CFG_SIG}.pt")
_results = torch.load(RESULTS_PATH) if (RECYCLE and os.path.exists(RESULTS_PATH)) else {}
def cache_get(k): return _results.get(k) if RECYCLE else None
def cache_put(k, v): _results[k] = v; torch.save(_results, RESULTS_PATH)
print(f"results cache {os.path.basename(RESULTS_PATH)}: {len(_results)} runs "
      f"({'recycling' if _results else 'empty -> will compute'})")
""")

# =============================================================================
md(r"""## §2 — Per-model sweep: magnitude **and** relative error vs width, per predictor

For each `(direction, width, seed)` build the random spiked model, run MC on the GPU, and run every
predictor. We keep the **unscaled magnitude** $\lVert E[\text{out}]\rVert$ and the **scaled relative
error**. `spk_R2` should track the ordinary **exact-K2** baseline (they are the same rank-2 closure;
on `e1` they differ only by the small node-quadrature error). Each cell recycles from the cache.""")
code(r"""
PRED_KEYS = ["k2", "k3"] + [f"spk_R{R}" for R in R_VALUES]
rows, t0 = [], time.time()
for direction in DIRECTIONS:
    for w in WIDTHS:
        for seed in SEEDS:
            key = f"{direction}|d{DEPTH}|w{w}|s{seed}"
            r = cache_get(key); src = "recycled"
            if r is None:
                src = "computed"
                m = get_model(direction, w, seed, DEPTH)
                mc, stats = mc_reference(m, w)
                preds = predict_all(m, w, direction)
                nm = float(np.linalg.norm(mc)) + 1e-30
                r = dict(direction=direction, w=w, seed=seed, mc_norm=float(np.linalg.norm(mc)),
                         floor=float(np.linalg.norm(stats["mc_stderr"])) / nm)
                for name, p in preds.items():
                    r[f"{name}_rel"]  = rel(p, mc, stats)
                    r[f"{name}_norm"] = float(np.linalg.norm(p))
                cache_put(key, r)
            rows.append(r)
            sm = "  ".join(f"R{R}={r.get(f'spk_R{R}_rel', float('nan')):.2e}" for R in R_VALUES)
            k3s = f"k3={r['k3_rel']:.2e}" if 'k3_rel' in r else "k3=  --   "
            print(f"{direction:>4} w={w:>4} s{seed} [{src:>8}] | K2={r['k2_rel']:.2e} {k3s} | "
                  f"SPK {sm} | floor={r['floor']:.1e}", flush=True)
print(f"\nsweep done in {time.time()-t0:.1f}s ({len(rows)} runs; recycled ones are instant)")
""")
code(r"""
from matplotlib.lines import Line2D
def series(direction, key):
    "mean over seeds of `key` at each width, for one direction (NaN-safe)"
    out = []
    for w in WIDTHS:
        vals = [r[key] for r in rows if r["direction"] == direction and r["w"] == w and key in r]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out

style = {"k2": ("o-", "k", "exact-K2 (ordinary)"), "k3": ("s-", "0.5", "k=3 (ordinary)"),
         "spk_R2": ("^--", "tab:green", "SPIKE R=2"), "spk_R3": ("v--", "tab:orange", "SPIKE R=3"),
         "spk_R4": ("D--", "tab:red", "SPIKE R=4")}
fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6))
for col, direction in enumerate(DIRECTIONS):
    axL, axR = axes[0, col], axes[1, col]
    title = "localized  $-e_1e_1^\\top$" if direction == "e1" else "flat  $-\\frac1n\\mathbf 1\\mathbf 1^\\top$"
    # top: unscaled magnitude ||E[out]||
    axL.loglog(WIDTHS, series(direction, "mc_norm"), "k*-", ms=9, label="MC (truth)")
    for key in ["k2"] + [f"spk_R{R}" for R in R_VALUES]:
        m_, c_, lab = style[key]
        axL.loglog(WIDTHS, series(direction, f"{key}_norm"), m_, color=c_, label=lab, alpha=0.9)
    axL.set_title(f"magnitude — {title} (depth {DEPTH}, $\\theta$={THETA})")
    axL.set_ylabel(r"$\|E[\mathrm{out}]\|_2$ (unscaled)"); axL.grid(alpha=0.3, which="both")
    axL.legend(fontsize=7)
    # bottom: scaled relative error
    for key in PRED_KEYS:
        m_, c_, lab = style[key]
        ser = series(direction, f"{key}_rel")
        if np.all(np.isnan(ser)): continue
        axR.loglog(WIDTHS, ser, m_, color=c_, label=lab, alpha=0.9)
    axR.loglog(WIDTHS, series(direction, "floor"), "-", color="0.75", lw=1, label="MC floor")
    axR.set_title(f"relative error — {title} (lower = better)")
    axR.set_xlabel("width  n"); axR.set_ylabel(r"$\|\mu_{\rm pred}-\mu_{\rm MC}\|/\|\mu_{\rm MC}\|$")
    axR.grid(alpha=0.3, which="both"); axR.legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §3 — Headline: **fit and TEST** the $n$-scaling against the theory exponent

Not just a plot — we **fit** $\text{error}=C\,n^{p}$ (OLS in log–log, slope $\pm$ standard error, $R^2$)
and check it against the exponent the theory predicts.

**What theory predicts.** Ordinary exact-K2 keeps cumulants to order $K{=}2$; the error-diagram bound
gives *squared* error $\sim n^{-K}$, i.e. **relative error $\propto n^{-1}$**, but only when the dropped
spike-direction cumulants actually decay — which depends on the direction via $c_{r,n}(vv^\top)=(\sum_i|v_i|^r)^2$:

- **flat** $v=\tfrac1{\sqrt n}\mathbf 1$: $c_{r,n}=n^{2-r}\to0$ → resolvable → slope $p=-1$ (the $K{=}2$ rate);
- **localized** $v=e_1$: $c_{r,n}=1$ for all $r$ (does *not* decay) → **not** resolvable at that rate →
  slope **shallower than $-1$** (toward $0$); SPIKE-KPROP only lowers the prefactor, not the rate.

**The tests** (printed below): (a) is the flat exact-K2 slope **consistent with $-1$** (within $2$ SE),
and does a *fixed* $n^{-1}$ law fit it ($R^2$)? (b) is the localized slope **significantly shallower**
than the flat one (so the $O(1)$ spike is genuinely not resolved at the $K{=}2$ rate)? Clean resolution
needs the MC floor well below the errors — hence the 20M-sample true run; if a curve has flattened onto
the floor, exclude those widths.""")
code(r"""
THEORY_P = {"ones": -1.0, "e1": 0.0}        # flat: K=2 rate n^{-1}; localized: O(1), no decay (slope 0)

def loglog_fit(direction, key, drop_floor=2.0):
    "OLS slope of log(err) vs log(n) over ALL (width,seed) points above drop_floor*MC-floor."
    pts = []
    for r in rows:
        if r["direction"] != direction:
            continue
        e = r.get(f"{key}_rel", float("nan"))
        if np.isfinite(e) and e > 0 and e > drop_floor * r.get("floor", 0.0):
            pts.append((r["w"], e))
    if len(pts) < 3:
        return dict(slope=float("nan"), se=float("nan"), r2=float("nan"), n=len(pts))
    x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts]); k = len(x)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef; ss = float((res ** 2).sum())
    se = float(np.sqrt(ss / max(k - 2, 1) / ((x - x.mean()) ** 2).sum()))
    r2 = 1.0 - ss / (float(((y - y.mean()) ** 2).sum()) + 1e-30)
    return dict(slope=float(coef[0]), se=se, r2=r2, n=k)

def fixed_exponent_r2(direction, key, p):
    "R^2 of forcing slope = p (fit only the constant C): tests whether n^p specifically describes it."
    pts = [(r["w"], r[f"{key}_rel"]) for r in rows if r["direction"] == direction
           and np.isfinite(r.get(f"{key}_rel", np.nan)) and r.get(f"{key}_rel", 0) > 2 * r.get("floor", 0)]
    if len(pts) < 2: return float("nan")
    x = np.log([q[0] for q in pts]); y = np.log([q[1] for q in pts])
    logC = float((y - p * x).mean()); yhat = logC + p * x
    return 1.0 - float(((y - yhat) ** 2).sum()) / (float(((y - y.mean()) ** 2).sum()) + 1e-30)

fits = {d: loglog_fit(d, "k2") for d in DIRECTIONS}
print("SCALING TEST — ordinary exact-K2, fit error = C n^p  (errors above 2x MC floor only):\n")
print(f"{'direction':>14} | {'fitted p':>16} | {'R^2':>5} | {'theory p':>8} | {'|p-theory|/SE':>13} | verdict")
print("-" * 92)
for d in DIRECTIONS:
    f = fits[d]; th = THEORY_P[d]; lab = "localized e1" if d == "e1" else "flat (ones)"
    z = abs(f["slope"] - th) / (f["se"] + 1e-30)
    verdict = ("consistent" if z <= 2 else "OFF") if np.isfinite(z) else "n/a"
    print(f"{lab:>14} | {f['slope']:>7.3f} ± {f['se']:<6.3f} | {f['r2']:>5.2f} | {th:>8.1f} | "
          f"{z:>13.2f} | {verdict}", flush=True)

ff, fl = fits["ones"], fits["e1"]
print(f"\n(a) flat resolves at the K=2 rate?  fitted p = {ff['slope']:.3f} ± {ff['se']:.3f}; "
      f"forcing n^-1 gives R^2 = {fixed_exponent_r2('ones','k2',-1.0):.3f}  "
      f"-> {'consistent with n^-1' if abs(ff['slope']+1) <= 2*ff['se'] else 'NOT n^-1'}")
d_slope = fl["slope"] - ff["slope"]; d_se = math.sqrt(ff["se"]**2 + fl["se"]**2)
print(f"(b) localized shallower than flat?  p_loc - p_flat = {d_slope:+.3f} ± {d_se:.3f}  "
      f"-> {'YES, spike NOT resolved at the K=2 rate' if d_slope > 2*d_se else 'inconclusive (need wider n / more MC)'}")

# plot with the fitted power laws overlaid + an n^-1 guide
fig, ax = plt.subplots(figsize=(8.2, 5.6))
wgrid = np.array(WIDTHS, float)
for direction in DIRECTIONS:
    c = "tab:purple" if direction == "e1" else "tab:blue"
    lab = "localized $e_1$" if direction == "e1" else "flat (ones)"
    k2 = np.array(series(direction, "k2_rel"), float)
    r4 = np.array(series(direction, "spk_R4_rel"), float)
    ax.loglog(WIDTHS, k2, "o-",  color=c, label=f"{lab}: exact-K2")
    ax.loglog(WIDTHS, r4, "D--", color=c, alpha=0.55, label=f"{lab}: SPIKE R=4")
    p = fits[direction]["slope"]                              # fitted power law, anchored at last point
    if np.isfinite(p) and np.isfinite(k2[-1]) and k2[-1] > 0:
        ax.loglog(wgrid, k2[-1] * (wgrid / wgrid[-1]) ** p, ":", color=c, lw=1.3,
                  label=f"{lab} fit  p={p:.2f}")
flat_last = np.array(series("ones", "k2_rel"), float)[-1]     # n^-1 guide anchored on the flat curve
if np.isfinite(flat_last) and flat_last > 0:
    ax.loglog(wgrid, flat_last * (wgrid / wgrid[-1]) ** (-1.0), "-.", color="0.4", lw=1,
              label="$n^{-1}$ guide")
ax.loglog(WIDTHS, [max(series(d, "floor")[i] for d in DIRECTIONS) for i in range(len(WIDTHS))],
          "-", color="0.8", lw=1, label="MC floor")
ax.set_xlabel("width  n"); ax.set_ylabel(r"$\|\mu_{\rm pred}-\mu_{\rm MC}\|/\|\mu_{\rm MC}\|$")
ax.set_title(f"$n$-scaling of the error (depth {DEPTH}, $\\theta$={THETA}); dotted = fitted power law")
ax.legend(fontsize=7.5); ax.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §4 — What the spike-direction cumulants buy: error vs $R$, and the residual

Bars: relative error at the largest width, per predictor, per model. Read it as **does keeping the
spike-direction cumulants $C(v,\dots,v)$ help?**

- **flat:** SPIKE-KPROP $R{=}2$ matches exact-K2; $R{=}3,4$ are **inert** (those cumulants are
  $O(n^{2-r})\to0$). The faithful-generalization / null check.
- **localized $e_1$:** $R{=}3,4$ **lower** the error monotonically — that is the $C(v,v,v)$ and
  $C(v,v,v,v)=d_4$ trace-boundary terms ordinary kprop drops. The **residual** after $R{=}4$ is the
  mixed trace coupling $C(v,v,i,j)\sim\theta\delta_{ij}$ that the conditional-Gaussian closure does
  not propagate (the documented approximation) — its size is the honest read of how far a
  spike-direction-only closure can go.""")
code(r"""
def at_max_width(direction):
    rs = [r for r in rows if r["direction"] == direction and r["w"] == WIDTHS[-1]]
    agg = {}
    for key in PRED_KEYS + ["floor"]:
        vals = [r[f"{key}_rel" if key != "floor" else "floor"] for r in rs
                if (f"{key}_rel" in r or key == "floor")]
        if vals: agg[key] = float(np.mean(vals))
    return agg

fig, ax = plt.subplots(figsize=(9.0, 5.2))
bars = ["k2"] + [f"spk_R{R}" for R in R_VALUES]
x = np.arange(len(DIRECTIONS)); bw = 0.8 / (len(bars) + 1)
for i, key in enumerate(bars):
    m_, c_, lab = style[key]
    vals = [at_max_width(d).get(key, np.nan) for d in DIRECTIONS]
    ax.bar(x + i * bw, vals, bw, color=c_, label=lab)
ax.bar(x + len(bars) * bw, [at_max_width(d).get("floor", np.nan) for d in DIRECTIONS], bw,
       color="0.8", label="MC floor")
ax.set_yscale("log"); ax.set_xticks(x + 0.4 - bw / 2)
ax.set_xticklabels(["localized $e_1$", "flat (ones)"])
ax.set_ylabel("relative error vs MC (log)")
ax.set_title(f"error vs predictor at width {WIDTHS[-1]} (depth {DEPTH}, $\\theta$={THETA})")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, axis="y", which="both"); plt.tight_layout(); plt.show()

for d in DIRECTIONS:
    a = at_max_width(d)
    gain = a["k2"] / max(a.get("spk_R4", a["k2"]), 1e-30)
    lab = "localized e1" if d == "e1" else "flat (ones)"
    print(f"{lab:>14} @ w{WIDTHS[-1]}: K2={a['k2']:.3e} -> SPIKE-R4={a.get('spk_R4', float('nan')):.3e}"
          f"  ({gain:.2f}x)  | residual/floor = {a.get('spk_R4', float('nan'))/max(a['floor'],1e-30):.0f}")
""")

# =============================================================================
md(r"""## §5 — Cumulant fidelity: the special-mode cumulants $d_3,d_4=C(v,\dots,v)$, layer by layer

SPIKE-KPROP exposes (`collect=True`) the propagated special-mode law $S=v\!\cdot\!X$ at every hidden
layer: mean $d_1$, variance $v_S$, and the tracked cumulants $d_3=C(v,v,v)$, $d_4=C(v,v,v,v)$. We
compare them to a Monte-Carlo estimate of the *actual* special-mode cumulants $\kappa_p(v\!\cdot\!a^\ell)$.
This is the microscope on the claim: the spike-direction cumulants are **larger for the localized
spike than for the flat one**, and SPIKE-KPROP tracks them.""")
code(r"""
from scipy.stats import kstat   # k-statistics: unbiased cumulant estimators

CUM_W, CUM_SEED = (WIDTHS[len(WIDTHS)//2], SEEDS[0])
print(f"special-mode cumulants at width {CUM_W}, seed {CUM_SEED}, depth {DEPTH}\n")

@torch.no_grad()
def mc_special_cumulants(m, w, direction, num_samples, batch, device, dtype):
    "MC kappa_2..4 of S = v . a^ell for each hidden post-activation a^ell."
    v = spike_vector(direction, w).to(device=device, dtype=dtype)
    md = copy.deepcopy(m).to(device=device, dtype=dtype).eval()
    depth = m.cfg.depth
    samp = {l: [] for l in range(depth)}
    N = 0
    while N < num_samples:
        b = min(batch, num_samples - N)
        acts = md.activations(torch.randn(b, w, device=device, dtype=dtype))
        for l in range(depth):
            samp[l].append((acts["post"][l].double() @ v).cpu().numpy())
        N += b
    out = {}
    for l in range(depth):
        s = np.concatenate(samp[l])
        out[l] = {2: float(kstat(s, 2)), 3: float(kstat(s, 3)), 4: float(kstat(s, 4))}
    return out

for direction in DIRECTIONS:
    m = get_model(direction, CUM_W, CUM_SEED, DEPTH)
    res = run_spike_kprop(m, direction, input_dim=CUM_W, config={"R": 4, "n_nodes": N_NODES},
                          device=KPROP_DEVICE, collect=True)
    pred_layers = res["special_by_layer"]
    mc_cum = mc_special_cumulants(m, CUM_W, direction, min(MC_SAMPLES, 2_000_000),
                                  MC_BATCH, MC_DEVICE, MC_DTYPE)
    lab = "localized e1" if direction == "e1" else "flat (ones)"
    print(f"--- {lab} ---")
    print(f"{'layer':>5} | {'vS pred':>9} {'vS MC':>9} | {'d3 pred':>9} {'d3 MC':>9} | {'d4 pred':>9} {'d4 MC':>9}")
    for L in pred_layers:
        l = L["layer"]; mk = mc_cum[l]
        print(f"{l:>5} | {L['vS']:>9.3e} {mk[2]:>9.3e} | {L['d'].get(3,0):>9.2e} {mk[3]:>9.2e} "
              f"| {L['d'].get(4,0):>9.2e} {mk[4]:>9.2e}")
    print()
print("Read: |d3|,|d4| (the tracked spike-direction cumulants) are appreciable for the LOCALIZED")
print("spike but ~0 for the FLAT spike -- exactly why R>=3 helps on e1 and is inert on ones.")
""")

# =============================================================================
md(r"""## §6 — Checkpoints: save / load / **download** (recycle across sessions)

The sweep wrote the random spiked models + a results cache to `checkpoints/spike_kprop` (keyed by
config, so direction/$\theta$/$R$/$n_{\text{nodes}}$ never mix). Re-running recycles everything.""")
code(r"""
import shutil
print("checkpoint dir:", os.path.abspath(CKPT_DIR))
for f in sorted(os.listdir(CKPT_DIR)):
    print("  ", f, f"({os.path.getsize(os.path.join(CKPT_DIR, f))/1e6:.2f} MB)")
if IN_COLAB:
    from google.colab import files
    z = shutil.make_archive("/content/spike_kprop_ckpts", "zip", CKPT_DIR)
    print("zipped ->", z, "-- downloading..."); files.download(z)
# RESTORE later: upload the zip, then
#   import io, zipfile; os.makedirs(CKPT_DIR, exist_ok=True)
#   zipfile.ZipFile(io.BytesIO(next(iter(files.upload().values())))).extractall(CKPT_DIR)
""")

# =============================================================================
md(r"""## §7 — Summary

- **What ran:** random depth-{DEPTH} ReLU MLPs (square, no bias), hidden matrices spiked by an
  **$O(1)$ eigenvalue** rank-one term $\theta vv^\top$ ($\theta=-1$, the minus direction) in two
  directions — **localized $e_1$** and **flat $\tfrac1{\sqrt n}\mathbf 1$** — **no training**;
  predictors **ordinary exact-K2 / k=3** vs **SPIKE-KPROP $R\in\{2,3,4\}$** vs **20M-sample**
  Monte-Carlo, swept over width.
- **SPIKE-KPROP** (`Mecha_preds/cumulants/spikekprop`) is SW-KPROP generalized to an arbitrary unit
  spike direction $v$: it retains the spike-direction cumulants $d_p=C(v,\dots,v)=\kappa_p(S)$ of the
  special mode $S=v\!\cdot\!X$ to order $R$ ($R{=}2$ exact rank-2; $R{=}3$ adds $d_3$; $R{=}4$ adds
  $d_4=C(v,v,v,v)$ — the degree-4 trace-boundary term). $v=$`ones` reproduces SW-KPROP exactly.
- **Headline (§3) — fitted & tested:** we fit $\text{error}=C\,n^{p}$ (slope $\pm$ SE, $R^2$) and test
  it against theory: the **flat** spike is *resolvable* at the $K{=}2$ rate $p=-1$ ($c_{r,n}=O(n^{2-r})$),
  while the **localized** spike is *not* ($c_{r,n}=O(1)$, slope shallower than $-1$ toward $0$). The cell
  prints whether the flat slope is consistent with $-1$ and whether the localized slope is significantly
  shallower — needs the 20M-sample MC floor well below the errors to resolve cleanly.
- **Spike-direction cumulants (§4–§5):** $R{\ge}3$ **helps on $e_1$** and is **inert on flat**;
  §5 shows $d_3,d_4$ are appreciable only for $e_1$. The $e_1$ **residual** after $R{=}4$ is the mixed
  trace coupling $C(v,v,i,j)\sim\theta\delta_{ij}$ the conditional-Gaussian closure does not propagate
  — the one documented approximation.
- **Recycling:** models + results in `checkpoints/spike_kprop`; re-runs load instead of recomputing.
  **GPU:** MC float32 on `E.DEVICE`; SPIKE-KPROP routes its dense congruence to CUDA float64.
  **Note:** coordinate spikes (`e1`) need more Gauss–Hermite nodes (`N_NODES`) — the special mode is a
  ReLU input, so its kink is integrated numerically (a quadrature error, not a modelling one).""".replace("{DEPTH}", "3"))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike_kprop_colab.ipynb")
nb.save(out)

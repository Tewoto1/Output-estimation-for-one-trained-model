"""build_exact_meanprop_notebook.py -- GENERATE the exact-ReLU mean-prop test notebook.

GENERATED notebook (edit THIS builder, re-run to refresh the .ipynb; see _nb.py).

Tests the new EXACT mean-propagation predictor (Mecha_preds/cumulants/exact_meanprop.py,
`run_exact_meanprop`) -- tracks per-coordinate (mean, variance), propagates variance
through linear layers as the diagonal (W.*W)v, and crosses every ReLU with the EXACT
rectified-Gaussian integral (mu*Phi+sigma*phi etc.) -- against:
  * Monte-Carlo E[out]  (ground truth),
  * the existing APPROXIMATE k=1 mean-prop (run_cumulants k_max=1; fixed diag(WWᵀ) metric),
  * (optional) the covariance-aware k=2 kprop reference.

on two beds:
  A) shifted-weight random MLPs  W = W' + s(1/sqrt n)11ᵀ  (sub: dead ReLUs / add: ~linear),
  B) the trained-to-0 checkpoints checkpoints/kprop_checkpoints/kprop-zero_d3_w*_tol5_seed*.

Heavy MC + kprop results are cached under results/exact_meanprop/ (repo recycling rule).

Run:  python "colab_notebooks/exact_meanprop/build_exact_meanprop_notebook.py"
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =========================================================================== #
md(r"""# Exact-ReLU mean propagation — does doing the ReLU integral exactly help?

**Mean propagation** tracks, for every coordinate and layer, a marginal Gaussian summarised by a
**mean** $\mu_i$ and a **variance** $v_i$. The two operations are:

$$\text{linear: }\;\mu\leftarrow W\mu+b,\quad v\leftarrow (W\odot W)\,v\;\;(\text{diagonal of }W\,\mathrm{diag}(v)\,W^\top)
\qquad\qquad \text{ReLU: }\;(\mu,v)\leftarrow \text{moments of }\mathrm{ReLU}(\mathcal N(\mu,v)).$$

The **exact** ReLU moments (no Hermite truncation, no gain approximation) are, with $\alpha=\mu/\sigma$,
$\sigma=\sqrt v$:

$$\mathbb E[\mathrm{ReLU}(Z)]=\mu\,\Phi(\alpha)+\sigma\,\phi(\alpha),\qquad
\mathbb E[\mathrm{ReLU}(Z)^2]=(\mu^2+v)\,\Phi(\alpha)+\mu\sigma\,\phi(\alpha),\qquad
\mathrm{Var}=\mathbb E[\mathrm{ReLU}^2]-\mathbb E[\mathrm{ReLU}]^2 .$$

`run_exact_meanprop` (in `Mecha_preds/cumulants/exact_meanprop.py`) implements exactly this — the
same closed form as the validated `kprop.exact_relu_covariance.relu_moments_1d_np`, but used to
**propagate and store** the variance forward.

**What this changes vs the existing $k{=}1$ mean-prop.** The harmonic $k{=}1$ path collapses the
degree-2 piece to a *fixed* metric $\mathrm{diag}(WW^\top)$ — effectively assuming **unit-variance
input at every layer** rather than carrying the true post-ReLU variance forward. Exact mean-prop
propagates the variance properly. The **only** remaining approximation is the mean-field/diagonal
closure at the linear mixing (cross-covariances between coordinates are dropped). Therefore:

* **depth 1** (one ReLU + linear readout): the output mean is **EXACT** — it needs only the
  post-ReLU *means*, which are exact marginals — so it equals MC up to sampling noise (a hard check below);
* **depth $\ge 2$**: the mean is approximate **only** through the dropped cross-covariance in the
  propagated variances. The shifted ($s{=}-1$, dead-ReLU) and trained-to-0 beds stress exactly that.
""")

# --------------------------------------------------------------------------- #
code(BOOTSTRAP_CELL)

# --------------------------------------------------------------------------- #
md(r"""## 1. Config — this notebook owns its knobs

Predictors: **exact mean-prop** (new) vs **k=1 approx mean-prop** vs optional **k=2 covariance kprop**,
all against Monte-Carlo. Shifted models are built/cached; trained models are loaded (never trained).
Results cache under `results/exact_meanprop/`.
""")

code(r"""
import os, math, copy, time, json, logging
import numpy as np
import matplotlib.pyplot as plt
import torch

# The APPROXIMATE k=2 covariance (gain approximation) is not PSD in the dead-ReLU
# regime and spams "Snapping negative variance to zero"; we default k=2 to the EXACT
# bivariate covariance (K2_EXACT below). Quiet the warning either way.
logging.getLogger("Mecha_preds.cumulants.kprop.wick").setLevel(logging.ERROR)

import experiments as E
from model import MLP, ModelConfig
from tasks import ZeroTask
from Mecha_preds.cumulants import (run_cumulants, run_exact_meanprop,
                                   estimate_empirical_mean, compare_means)

DEVICE      = E.DEVICE
MC_DEVICE   = E.DEVICE                  # MC runs on GPU if available
MC_DTYPE    = torch.float64
KPROP_DEVICE = "cpu"                    # kprop / exact-mp run in float64 on CPU (exact, robust)
ACTIVATION  = "relu"
RECYCLE     = True

QUICK = E.QUICK

# ---- predictors to compare -------------------------------------------------
INCLUDE_K2    = True                    # covariance-aware reference (O(n^2); see K2_MAX_WIDTH)
K2_EXACT      = True                     # k=2 uses the EXACT bivariate ReLU covariance (PSD, stable).
                                         # The approximate gain-based k=2 goes NON-PSD in the dead-ReLU
                                         # "sub" regime -> negative variances snapped by wick -> the
                                         # prediction blows up (rel ~ 1e5-1e6). Keep True here.
K2_MAX_WIDTH  = 1024                     # skip k=2 above this width (exact cov is O(n^2) Owen's-T)

# ---- bed A: shifted-weight random MLPs ------------------------------------
RUN_SHIFTED   = True
SHIFT         = "sub"                    # "sub" (s=-1): dead ReLUs (stress); "add" (s=+1): ~linear (control)
SIGN          = -1.0 if SHIFT == "sub" else +1.0
DEPTHS_SH     = [3] if QUICK else [3, 4, 5]
WIDTHS_SH     = [64, 128, 256] if QUICK else [64, 128, 256, 512, 1024, 2048]
SEEDS_SH      = [1, 2]
SH_CKPT_DIR   = "checkpoints/shifted_mean_vanilla_kprop"   # reuse the shifted models if present

# ---- bed B: trained-to-0 checkpoints (LOAD ONLY) --------------------------
RUN_TRAINED   = True
TR_CKPT_DIR   = "checkpoints/kprop_checkpoints"
TR_PREFIX, TR_DEPTH, TR_TOL = "kprop-zero", 3, 5
WIDTHS_TR     = [16, 32, 64, 128] if QUICK else [16, 32, 64, 128, 256, 512, 1024, 1536, 2048]
SEEDS_TR      = [3, 4, 5, 6]

# ---- Monte-Carlo -----------------------------------------------------------
MC_SAMPLES = 300_000 if QUICK else 2_000_000
MC_BATCH   = 8192

CACHE_DIR = "results/exact_meanprop"
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"device={DEVICE} | predictors: exact-mp, k1(approx){' , k2' if INCLUDE_K2 else ''} | MC={MC_SAMPLES:,}")
print(f"bed A (shifted, {SHIFT}): depths={DEPTHS_SH} widths={WIDTHS_SH} seeds={SEEDS_SH} | run={RUN_SHIFTED}")
print(f"bed B (trained d{TR_DEPTH}): widths={WIDTHS_TR} seeds={SEEDS_TR} | run={RUN_TRAINED}")
""")

# --------------------------------------------------------------------------- #
md(r"""## 2. Sanity — the ReLU step is the exact integral, and depth-1 is exact vs MC

(a) `run_exact_meanprop`'s ReLU moments equal the validated `relu_moments_1d_np` to machine precision.
(b) On a depth-1 net the predicted output mean equals Monte-Carlo to sampling noise (exact, since the
mean never needs a cross-covariance). Run this once; it needs no checkpoints.
""")

code(r"""
from Mecha_preds.cumulants.exact_meanprop import relu_gaussian_moments
from Mecha_preds.cumulants.kprop.exact_relu_covariance import relu_moments_1d_np

rng = np.random.default_rng(0)
mm = rng.normal(0, 2, 3000); vv = rng.gamma(2.0, 1.0, 3000)
em, es, ev = relu_gaussian_moments(mm, vv)
om, os_, ov = relu_moments_1d_np(mm, vv)
print("ReLU moments vs relu_moments_1d_np:  max|dmean|=%.1e  max|dvar|=%.1e (machine precision)"
      % (np.max(np.abs(em-om)), np.max(np.abs(ev-ov))))

# depth-1 net: exact-mp output mean == MC
m1 = E.build_mlp(64, depth=1, output_dim=8, seed=0, activation=ACTIVATION).double().eval()
pred1 = run_exact_meanprop(m1, input_dim=64)["mean"]
mc1, st1 = estimate_empirical_mean(model=m1.to(MC_DEVICE, MC_DTYPE), input_dim=64,
                                   num_samples=2_000_000, device=str(MC_DEVICE), dtype=MC_DTYPE, batch_size=MC_BATCH)
rel1 = np.linalg.norm(pred1 - mc1) / (np.linalg.norm(mc1) + 1e-30)
floor1 = np.linalg.norm(st1["mc_stderr"]) / (np.linalg.norm(mc1) + 1e-30)
print(f"depth-1 exact-mp vs MC: rel-L2={rel1:.2e}  (MC floor={floor1:.2e})  -> exact within sampling:", rel1 < 5*floor1+2e-3)
""")

# --------------------------------------------------------------------------- #
md(r"""## 3. Shared machinery — predictors, MC reference, caching

`predict_all` runs the three predictors on one float64 model; `mc_reference` runs MC on the GPU
copy; `eval_one` assembles the comparison row (relative + absolute error, MC-noise $z$, magnitudes,
the actual mean vectors for parity, and the exact-mp output variance vs MC variance). All cached by
config signature, so a re-run recomputes nothing.
""")

code(r"""
def predict_all(m, w, include_k2):
    out = {}
    emp = run_exact_meanprop(m, input_dim=w)
    out["exact_mp"] = np.asarray(emp["mean"], float)
    out["exact_mp_var"] = np.asarray(emp["out_var"], float)
    out["k1"] = np.asarray(run_cumulants(m, w, {"k_max": 1, "factor": False}, device=KPROP_DEVICE)["mean"], float)
    if include_k2:
        out["k2"] = np.asarray(run_cumulants(m, w, {"k_max": 2, "factor": False,
                               "exact_relu_cov": K2_EXACT}, device=KPROP_DEVICE)["mean"], float)
    return out

def mc_reference(m, w):
    mdev = copy.deepcopy(m).to(device=MC_DEVICE, dtype=MC_DTYPE)
    mc, st = estimate_empirical_mean(model=mdev, input_dim=w, num_samples=MC_SAMPLES,
                                     device=str(MC_DEVICE), dtype=MC_DTYPE, batch_size=MC_BATCH)
    del mdev
    if MC_DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    # per-output MC variance (for the variance-accuracy panel)
    mc_var = np.clip(np.asarray(st["per_output_second_moment"], float) - np.asarray(mc, float) ** 2, 0.0, None)
    return np.asarray(mc, float), st, mc_var

def eval_one(m, w, depth, seed, include_k2):
    mc, st, mc_var = mc_reference(m, w)
    preds = predict_all(m, w, include_k2)
    nm = float(np.linalg.norm(mc)) + 1e-30
    floor = float(np.linalg.norm(st["mc_stderr"]))
    row = dict(depth=depth, w=w, seed=seed, mc_norm=float(np.linalg.norm(mc)),
               floor_abs=floor, floor_rel=floor / nm)
    for name in ("exact_mp", "k1", "k2"):
        if name not in preds:
            continue
        cm = compare_means(preds[name], mc, st)
        row[f"{name}_rel"] = cm["relative_error_mean"]
        row[f"{name}_abs"] = cm["mean_l2_error"]
        row[f"{name}_z"] = cm["mc_noise_z"]
        row[f"{name}_norm"] = float(np.linalg.norm(preds[name]))
        row[f"{name}_vec"] = preds[name].astype(np.float32)
    # exact-mp output-variance accuracy (diagonal): ||v_pred - v_mc|| / ||v_mc||
    vnm = float(np.linalg.norm(mc_var)) + 1e-30
    row["exact_mp_var_rel"] = float(np.linalg.norm(preds["exact_mp_var"] - mc_var) / vnm)
    row["mc_vec"] = mc.astype(np.float32)
    return row

def run_bed(tag, model_iter, cfg_sig):
    path = os.path.join(CACHE_DIR, f"results_{tag}_{cfg_sig}.pt")
    cache = torch.load(path) if (RECYCLE and os.path.exists(path)) else {}
    rows, t0 = [], time.time()
    for key, builder, (w, depth, seed) in model_iter:
        r = cache.get(key) if RECYCLE else None
        src = "recycled"
        if r is None:
            m = builder()
            if m is None:
                print(f"   [skip] {key} (missing)"); continue
            inc_k2 = INCLUDE_K2 and (w <= K2_MAX_WIDTH)
            r = eval_one(m.double().eval(), w, depth, seed, inc_k2)
            cache[key] = r; torch.save(cache, path); src = "computed"
        rows.append(r)
        emp_n = r.get("exact_mp_norm", float("nan"))
        collapsed = emp_n < 1e-9 * (r["mc_norm"] + 1e-30)        # predicted ~0 -> rel saturates at 1
        msg = f"   {key:>22} [{src:>8}] exact-mp rel={r.get('exact_mp_rel', float('nan')):.3e}  k1 rel={r.get('k1_rel', float('nan')):.3e}"
        if "k2_rel" in r:
            msg += f"  k2 rel={r['k2_rel']:.3e}"
        msg += f"  | ||emp||={emp_n:.1e} ||mc||={r['mc_norm']:.1e} (floor {r['floor_rel']:.1e})"
        if collapsed:
            msg += "  <- exact-mp COLLAPSED to ~0 (dead ReLUs; rel->1 by construction)"
        print(msg, flush=True)
    print(f"   {tag}: {len(rows)} runs in {time.time()-t0:.1f}s")
    return rows
""")

# --------------------------------------------------------------------------- #
md(r"""## 4. Bed A — shifted-weight models  $W=W'+s\,(1/\sqrt n)\mathbf 1\mathbf 1^\top$

`SHIFT="sub"` ($s=-1$) drives hidden pre-activations negative → ReLUs die → output collapses toward a
point mass: the regime where a mean-only state is most stressed. `SHIFT="add"` ($s=+1$) keeps ReLUs in
the ~linear regime (an easy control where mean-prop should be near-exact). The shift is on hidden
layers only (the readout is linear; shifting it would only inflate $\|E[\mathrm{out}]\|$).

**Expect `exact-mp rel ≈ 1.000` at larger $n$ in the `sub` regime — and that is the exact integral
being _faithful_, not failing.** The shared shift adds $s\,\tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top$, so
for $\ell\ge2$ the pre-activation mean is $\approx -\sqrt n\,\bar\mu$ (grows with $\sqrt n$) while
$\sigma=O(1)$. Then $\alpha=\mu/\sigma\to-\sqrt n$, and the exact moments
$\mathbb E[\mathrm{ReLU}],\mathrm{Var}[\mathrm{ReLU}]\sim e^{-\alpha^2/2}\to0$ **underflow**: the
coordinate becomes a dead point mass at $0$, the layer's variance hits $0$, and everything downstream
is exactly $0$. So exact mean-prop outputs $\mathbf 0$, and $\mathrm{rel}=\lVert\mathbf 0-\mu_{MC}\rVert/\lVert\mu_{MC}\rVert=1$
**by construction** ($\mathrm{rel}{=}1\Leftrightarrow$ "predicted zero"). The true $\mu_{MC}$ is a tiny
**nonzero** residual produced entirely by inter-coordinate **correlations**, which any mean-field
(diagonal) propagator discards — so a more exact _ReLU gate_ cannot recover it; the missing ingredient
is the **covariance** (the $k{=}2$ exact path below is what closes it). The cruder $k{=}1$ avoids the
collapse only because its _fixed_ $\mathrm{diag}(WW^\top)$ metric never lets the variance die — its
nonzero answer is an artifact of that wrong metric, not higher accuracy. Watch $\lVert\text{emp}\rVert$
vs $\lVert\text{mc}\rVert$ in the log to see the collapse directly.
""")

code(r"""
def shifted_mean_mlp(width, seed, depth):
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation=ACTIVATION).double().eval()
    g = torch.Generator().manual_seed(1_000_000 * depth + 10_000 * seed + 7 * width)
    c = 1.0 / math.sqrt(width)
    with torch.no_grad():
        layers = list(m.hidden_layers) + [m.readout]
        for li, layer in enumerate(layers):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64) / math.sqrt(in_f)
            if li < len(m.hidden_layers):                       # shift HIDDEN matrices only
                W = W + SIGN * c * torch.ones(out_f, in_f, dtype=torch.float64)
            layer.weight.copy_(W)
    return m

def shifted_get(w, seed, depth):
    path = E.ckpt_path(SH_CKPT_DIR, E.run_name(f"shifted-{SHIFT}", depth=depth, width=w, seed=seed))
    if RECYCLE and os.path.exists(path):
        m, _ = MLP.load(path, map_location="cpu"); return m.double().eval()
    m = shifted_mean_mlp(w, seed, depth)
    m.save(path, extra={"family": "shifted_mean_vanilla_kprop", "shift": SHIFT})
    return m

if RUN_SHIFTED:
    it = [(f"d{d}|w{w}|s{s}", (lambda w=w, s=s, d=d: shifted_get(w, s, d)), (w, d, s))
          for d in DEPTHS_SH for w in WIDTHS_SH for s in SEEDS_SH]
    CFG_SH = f"{SHIFT}_mc{MC_SAMPLES}_k2{int(INCLUDE_K2)}"
    rows_sh = run_bed(f"shifted_{SHIFT}", it, CFG_SH)
else:
    rows_sh = []
""")

code(r"""
def series(rows, depth, key):
    ws = sorted({r["w"] for r in rows if r["depth"] == depth})
    return ws, [float(np.nanmean([r[key] for r in rows if r["depth"] == depth and r["w"] == w and key in r]))
                for w in ws]

def plot_bed(rows, depths, title):
    if not rows:
        print("(bed not run)"); return
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(depths), 1)))
    cmap = dict(zip(depths, colors))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    # (1) magnitudes ||E[out]||: MC o-, exact-mp x--, k1 ^:, k2 s-.
    for d in depths:
        c = cmap[d]
        ws, y = series(rows, d, "mc_norm");        ax[0].loglog(ws, y, "o-",  color=c, label=f"d{d} MC")
        ws, y = series(rows, d, "exact_mp_norm");  ax[0].loglog(ws, y, "x--", color=c)
        ws, y = series(rows, d, "k1_norm");        ax[0].loglog(ws, y, "^:",  color=c)
        if any("k2_norm" in r for r in rows):
            ws2 = sorted({r["w"] for r in rows if r["depth"] == d and "k2_norm" in r})
            if ws2:
                y2 = [float(np.nanmean([r["k2_norm"] for r in rows if r["depth"] == d and r["w"] == w and "k2_norm" in r])) for w in ws2]
                ax[0].loglog(ws2, y2, "s-.", color=c, alpha=0.6)
    ax[0].set_title(f"{title}\n||E[out]||: MC(o) exact-mp(x) k1(^) k2(s)"); ax[0].set_xlabel("width n"); ax[0].set_ylabel("||E[out]||")
    ax[0].legend(fontsize=7)
    # (2) relative error vs width
    for d in depths:
        c = cmap[d]
        ws, y = series(rows, d, "exact_mp_rel"); ax[1].loglog(ws, y, "x--", color=c, label=f"d{d} exact-mp")
        ws, y = series(rows, d, "k1_rel");       ax[1].loglog(ws, y, "^:",  color=c, label=f"d{d} k1")
        ws, y = series(rows, d, "floor_rel");    ax[1].loglog(ws, y, ":",   color="0.6")
    ax[1].set_title("relative L2 error vs MC (dashed grey = MC floor)"); ax[1].set_xlabel("width n"); ax[1].set_ylabel("||pred-MC||/||MC||")
    ax[1].legend(fontsize=7)
    # (3) what exact buys: k1_err / exact_mp_err (>1 -> exact-mp better)
    for d in depths:
        c = cmap[d]
        ws = sorted({r["w"] for r in rows if r["depth"] == d})
        ratio = [float(np.nanmean([r["k1_abs"] / (r["exact_mp_abs"] + 1e-30)
                                   for r in rows if r["depth"] == d and r["w"] == w])) for w in ws]
        ax[2].loglog(ws, ratio, "o-", color=c, label=f"d{d}")
    ax[2].axhline(1.0, color="0.5", ls=":")
    ax[2].set_title("k1 error / exact-mp error  (>1 = exact-mp closer to MC)"); ax[2].set_xlabel("width n"); ax[2].set_ylabel("error ratio")
    ax[2].legend(fontsize=7)
    fig.tight_layout(); plt.show()

plot_bed(rows_sh, DEPTHS_SH, f"Shifted weights (s={int(SIGN):+d}, {SHIFT})")
""")

code(r"""
# Per-coordinate parity at one representative (depth, width, seed): predicted vs MC
def parity(rows, pick):
    r = next((x for x in rows if (x["depth"], x["w"], x["seed"]) == pick), None)
    if r is None:
        print("(no row for", pick, ")"); return
    mc = r["mc_vec"].astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for j, (name, col) in enumerate([("exact_mp", "#1f77b4"), ("k1", "#d62728")]):
        if f"{name}_vec" not in r:
            continue
        p = r[f"{name}_vec"].astype(float)
        ax[j].scatter(mc, p, s=8, alpha=0.5, color=col)
        lim = [min(mc.min(), p.min()), max(mc.max(), p.max())]
        ax[j].plot(lim, lim, "k--", lw=1)
        ax[j].set_title(f"{name}: per-coord predicted vs MC  (d{pick[0]} w{pick[1]} s{pick[2]})")
        ax[j].set_xlabel("MC E[out]_i"); ax[j].set_ylabel(f"{name} E[out]_i")
    fig.tight_layout(); plt.show()

if rows_sh:
    dpick = DEPTHS_SH[len(DEPTHS_SH)//2]; wpick = sorted({r["w"] for r in rows_sh if r["depth"]==dpick})[-1]
    parity(rows_sh, (dpick, wpick, SEEDS_SH[0]))
""")

# --------------------------------------------------------------------------- #
md(r"""## 5. Bed B — trained-to-0 checkpoints

Load `kprop-zero_d3_w*_tol5_seed{3..6}` (never retrain). These output $\approx 0$, so $\|E[\mathrm{out}]\|$
is tiny and the **MC-noise $z$** (error in units of MC's own standard error) is the most honest read:
$z\lesssim 1$ means the predictor agrees with MC to within sampling noise. We show $z$ and absolute
$L_2$ error alongside the relative error.
""")

code(r"""
def trained_get(w, seed):
    path = E.ckpt_path(TR_CKPT_DIR, E.run_name(TR_PREFIX, depth=TR_DEPTH, width=w, tol=TR_TOL, seed=seed))
    if not os.path.exists(path):
        return None
    m, _ = MLP.load(path, map_location="cpu")
    return m.double().eval()

if RUN_TRAINED:
    it = [(f"d{TR_DEPTH}|w{w}|s{s}", (lambda w=w, s=s: trained_get(w, s)), (w, TR_DEPTH, s))
          for w in WIDTHS_TR for s in SEEDS_TR]
    CFG_TR = f"trained_mc{MC_SAMPLES}_k2{int(INCLUDE_K2)}"
    rows_tr = run_bed("trained", it, CFG_TR)
else:
    rows_tr = []
""")

code(r"""
def plot_trained(rows):
    if not rows:
        print("(trained bed not run)"); return
    ws = sorted({r["w"] for r in rows})
    def mean_at(key):
        return [float(np.nanmean([r[key] for r in rows if r["w"] == w and key in r])) for w in ws]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    # (1) ||E[out]||: MC vs predictors (all tiny; trained to 0)
    ax[0].loglog(ws, mean_at("mc_norm"), "o-", color="k", label="MC")
    ax[0].loglog(ws, mean_at("exact_mp_norm"), "x--", color="#1f77b4", label="exact-mp")
    ax[0].loglog(ws, mean_at("k1_norm"), "^:", color="#d62728", label="k1 approx")
    if any("k2_norm" in r for r in rows):
        ws2 = sorted({r["w"] for r in rows if any("k2_norm" in rr for rr in rows if rr["w"]==r["w"])})
        ax[0].loglog(ws2, [float(np.nanmean([r["k2_norm"] for r in rows if r["w"]==w and "k2_norm" in r])) for w in ws2], "s-.", color="#2ca02c", alpha=0.7, label="k2")
    ax[0].set_title("trained-to-0: ||E[out]|| (tiny)"); ax[0].set_xlabel("width n"); ax[0].set_ylabel("||E[out]||"); ax[0].legend(fontsize=8)
    # (2) MC-noise z (resolved test): z<~1 = within MC noise
    ax[1].loglog(ws, mean_at("exact_mp_z"), "x--", color="#1f77b4", label="exact-mp")
    ax[1].loglog(ws, mean_at("k1_z"), "^:", color="#d62728", label="k1 approx")
    ax[1].axhline(1.0, color="0.5", ls=":")
    ax[1].set_title("error in MC-sigma units (z); z<~1 within MC noise"); ax[1].set_xlabel("width n"); ax[1].set_ylabel("||pred-MC||/MC_se"); ax[1].legend(fontsize=8)
    # (3) absolute L2 error + what exact buys
    ax[2].loglog(ws, mean_at("exact_mp_abs"), "x--", color="#1f77b4", label="exact-mp abs err")
    ax[2].loglog(ws, mean_at("k1_abs"), "^:", color="#d62728", label="k1 abs err")
    ax[2].loglog(ws, mean_at("floor_abs"), ":", color="0.6", label="MC floor")
    ax[2].set_title("absolute L2 error vs MC"); ax[2].set_xlabel("width n"); ax[2].set_ylabel("||pred-MC||"); ax[2].legend(fontsize=8)
    fig.tight_layout(); plt.show()

plot_trained(rows_tr)
if rows_tr:
    wtr = sorted({r["w"] for r in rows_tr})[len(set(r["w"] for r in rows_tr))//2]
    parity(rows_tr, (TR_DEPTH, wtr, SEEDS_TR[0]))
""")

# --------------------------------------------------------------------------- #
md(r"""## 6. Output-variance accuracy & summary

Exact mean-prop also returns a (diagonal) **output variance**; the panel shows its relative error vs
the MC per-output variance — a check that the propagated variances themselves are sensible (limited by
the same diagonal closure). The table prints, per bed, the median relative error of exact-mp vs the
approximate $k{=}1$, and the median "what exact buys" ratio.
""")

code(r"""
def var_panel(rows, label):
    if not rows:
        return
    ws = sorted({r["w"] for r in rows})
    y = [float(np.nanmean([r["exact_mp_var_rel"] for r in rows if r["w"] == w])) for w in ws]
    plt.figure(figsize=(6, 4))
    plt.loglog(ws, y, "o-", color="#1f77b4")
    plt.title(f"{label}: exact-mp output-variance rel. error vs MC")
    plt.xlabel("width n"); plt.ylabel("||v_pred - v_MC|| / ||v_MC||"); plt.tight_layout(); plt.show()

var_panel(rows_sh, f"shifted ({SHIFT})")
var_panel(rows_tr, "trained-to-0")

def summarize(rows, name):
    if not rows:
        print(f"{name}: (not run)"); return
    emp = np.array([r["exact_mp_rel"] for r in rows if "exact_mp_rel" in r])
    k1 = np.array([r["k1_rel"] for r in rows if "k1_rel" in r])
    ratio = np.array([r["k1_abs"] / (r["exact_mp_abs"] + 1e-30) for r in rows if "k1_abs" in r and "exact_mp_abs" in r])
    print(f"{name:18s}  median rel-err: exact-mp={np.median(emp):.3e}  k1={np.median(k1):.3e}  "
          f"| median (k1 err / exact-mp err)={np.median(ratio):.2f}  (>1 = exact-mp closer)")

print("=" * 92)
summarize(rows_sh, f"shifted ({SHIFT})")
summarize(rows_tr, "trained-to-0")
print("=" * 92)
""")

# --------------------------------------------------------------------------- #
md(r"""## 7. How to read this

* **Depth-1 sanity (§2):** exact-mp == MC to sampling noise — confirms the implementation and that the
  *mean* is exact wherever no cross-covariance is dropped.
* **Shifted `add` (control):** ReLUs are ~linear, the diagonal closure is nearly exact, so exact-mp
  should sit on the MC floor. Flip `SHIFT="add"` in §1 to see it.
* **Shifted `sub` (stress) & trained-to-0:** hidden ReLUs die / the output collapses to $\approx 0$.
  Here the residual error is the **mean-field/diagonal closure** (dropped cross-covariance), *not* the
  ReLU integral — so exact-mp and the approximate $k{=}1$ can both miss, and the §4/§5 "what exact buys"
  ratio shows whether propagating the variance exactly through the ReLU still helps. Where it doesn't,
  the covariance-aware $k{=}2$ is the predictor that closes the gap (cross-cov is the missing ingredient).
* **MC-noise $z$ (trained bed):** the honest metric when $\|E[\mathrm{out}]\|\to0$; $z\lesssim1$ means
  "indistinguishable from MC at this sample budget" — raise `MC_SAMPLES` to tighten it.

Knobs in §1: `SHIFT` (sub/add), `INCLUDE_K2`/`K2_MAX_WIDTH`, widths/depths/seeds, `MC_SAMPLES`. Cached
results reload; only new configs recompute.
""")

# --------------------------------------------------------------------------- #
out_path = os.path.join(os.path.dirname(__file__), "exact_meanprop_colab.ipynb")
nb.save(out_path)

"""Generates clipped_prop_colab.ipynb (valid nbformat-4 JSON).

The test suite + validation notebook for ``Mecha_preds.clippedProp`` -- the
structured "clamped-Gaussian all-ones channel" predictor:

    X = s * u + z,   u = 1_d/sqrt(d),   s = max(lo, N(m, v)),   z ~ N(mu_z, Sigma_z)

It carries the all-ones (mean-shift) component as an EXPLICIT clamped-Gaussian
scalar and the rest as a (cross-correlated) Gaussian in u^perp. ReLU conditions
on the scalar latent g (Gauss-Hermite) so z|g is incoherent and the standard
Gaussian-ReLU covariance is accurate per node.

What the notebook checks (NO TRAINING anywhere):
  §1  UNIT MATH (asserts): rectified-Gaussian moment round-trip, cross-cov beta,
      single-ReLU layer vs the project's EXACT bivariate kernel (machine precision
      as Gauss-Hermite nodes grow), mean-subtraction, linear-readout exactness.
  §2  VALIDATION -- shifted-Gaussian INPUT  X ~ N(c 1, I)  into a standard random
      He MLP: clippedProp vs Monte-Carlo across widths (its design regime).
  §3  VALIDATION -- clamped-Gaussian INPUT  X = max(0,N(mu_s,sd_s)) u + z  (the
      exact structured assumption): clippedProp vs Monte-Carlo across widths.
  §4  STRESS -- weight-shifted MLP  W = W' - (1/sqrt n) 11^T  with X ~ N(0,I):
      clippedProp vs vanilla / exact-cov k=2 cumulant propagation vs MC. This is
      the HARD case (the all-ones channel turns strongly skewed -- a 2-moment
      scalar closure is challenged; cf. skprop, which conditions on the input
      latent once and carries it exactly).

Repo policies honored: float64 for kprop/MC accuracy (GPU float32 compute where
available, float64 accumulators); per-config recycling of the expensive MC
references under checkpoints/clipped_prop; the notebook owns its own CKPT_DIR.

Run:  python "colab_notebooks/clipped_prop/build_clipped_prop_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# clippedProp — structured propagation with a clamped-Gaussian all-ones channel (test suite + validation)

**The predictor.** clippedProp carries the law of a hidden vector $X\in\mathbb R^d$ in the structured form

$$X = s\,u + z,\qquad u=\tfrac{1}{\sqrt d}\mathbf 1,\qquad P=I-uu^\top,$$

with the **all-ones / mean-shift component** tracked as an explicit *clamped* ("rectified") Gaussian scalar
$s=\max(\text{lo}, g),\ g\sim\mathcal N(m,v)$ (so $\text{lo}=0$ gives a point mass $p_0=\Pr(g\le0)$ plus a
positive tail; $\text{lo}=-\infty$ is a plain Gaussian), and the perpendicular part $z\sim\mathcal N(\mu_z,\Sigma_z)$
in $u^\perp$, allowed to be cross-correlated $c_s=\mathrm{Cov}(z,s)$.

**The three layer maps** (`Mecha_preds.clippedProp.layers`):

| layer | what it does |
|---|---|
| **linear** | push $(\mu,\Sigma)$ through $Wx+b$, re-split onto the *new* $u'/u'^\perp$, refit the scalar |
| **mean-subtraction** | $x\!\leftarrow\!Px$: the scalar channel collapses to $0$; $(\mu_z,\Sigma_z)$ unchanged |
| **ReLU** | condition on the latent $g$ (Gauss–Hermite), apply the Gaussian-ReLU moments per node ($X\mid g$ is Gaussian), mix by the law of total covariance, re-split, refit the scalar at clamp $0$ |

ReLU reuses the project's **verified exact bivariate-Gaussian ReLU covariance** (`exact_relu_covariance_np`, Owen's T)
as the per-node kernel (`relu_cov="exact"`), or a cheap leading-order gain off-diagonal (`relu_cov="gain"`).

**Why a clamped scalar.** The all-ones average of post-ReLU activations is non-negative and piles up against $0$; a
plain Gaussian fits it badly, the rectified Gaussian fits the point mass + tail. A coherent $O(1)$ mean shift — the
failure mode of vanilla $k{=}2$ cumulant propagation — lives entirely in this scalar channel, which the ReLU step
conditions on so the perpendicular residual is incoherent again.

**Sections.** §1 unit-math asserts (machine precision); §2–§3 validation on shifted / clamped-mean **inputs** to a
standard MLP (its design regime — expect $\sim\!10^{-3}$ rel-$L_2$); §4 the weight-shifted **stress** case vs vanilla
cumulant propagation.

> No training anywhere. Models are random; the expensive Monte-Carlo references are cached per config under
> `checkpoints/clipped_prop` (the last cell zips/downloads it on Colab).""")

code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 — Config, device & recycling

`WIDTHS`/`DEPTHS` drive the validation sweeps; `QUICK` (auto-True on a CPU-only box) trims them. clippedProp's
exact path materializes $d\times d$ matrices per ReLU node, so very wide nets are capped like the exact-cov kprop
notebook.""")
code(r"""
import math, time, os, copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP
from Mecha_preds.clippedProp import (
    run_clipped, clipped_mlp_forward, ClippedState,
    rect_gauss_moments, fit_rect_gauss, clipped_cross_beta,
    relu_layer, mean_subtraction_layer, linear_output_moments,
)
from Mecha_preds.clippedProp.state import NEG_INF
from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, compare_means
from Mecha_preds.cumulants.kprop.exact_relu_covariance import exact_relu_covariance_np

torch.set_default_dtype(torch.float64)
QUICK  = E.QUICK
DEVICE = E.DEVICE
torch.set_num_threads(max(torch.get_num_threads(), 2))

DEPTHS   = [3] if QUICK else [3, 4]
WIDTHS   = [32, 64, 128] if QUICK else [32, 64, 128, 256]
SEED     = 1
N_NODES  = 21                      # Gauss-Hermite nodes for the per-ReLU scalar integral
MC_SAMPLES = 200_000 if QUICK else 600_000
MC_BATCH   = 20_000

CKPT_DIR = "checkpoints/clipped_prop"
RECYCLE  = True
os.makedirs(CKPT_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(CKPT_DIR, f"mcref_seed{SEED}_mc{MC_SAMPLES}.pt")
_mc = torch.load(RESULTS_PATH) if (RECYCLE and os.path.exists(RESULTS_PATH)) else {}
def mc_get(k):      return _mc.get(k) if RECYCLE else None
def mc_put(k, v):   _mc[k] = v; torch.save(_mc, RESULTS_PATH)

def relL2(pred, ref): return float(np.linalg.norm(pred-ref) / (np.linalg.norm(ref) + 1e-30))

print("DEVICE:", DEVICE, "| QUICK:", QUICK, "| depths:", DEPTHS, "| widths:", WIDTHS)
print("N_NODES:", N_NODES, "| MC_SAMPLES:", f"{MC_SAMPLES:,}", "| CKPT_DIR:", CKPT_DIR,
      "| MC cache:", f"{len(_mc)} entries")
""")

# =============================================================================
md(r"""## §1a — UNIT MATH (asserts). The building blocks, checked to machine precision.

Each block prints `PASS` with the measured error. These are the correctness gates: if any fails, the propagation
math is wrong.""")
code(r"""
rng = np.random.default_rng(0)
fails = []
def check(name, err, tol):
    ok = err < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} err={err:.2e}  (tol {tol:.0e})")
    if not ok: fails.append(name)

# (1) rectified-Gaussian moment round-trip: fit_rect_gauss is the exact inverse of rect_gauss_moments
worst = 0.0
for _ in range(4000):
    m, v = rng.uniform(-4, 4), rng.uniform(1e-3, 9.0)
    mean, var, _ = rect_gauss_moments(m, v, 0.0)
    if mean < 1e-8 or var < 1e-12: continue          # degenerate point-mass (moments ~0)
    m2, v2 = fit_rect_gauss(mean, var, 0.0)
    mean2, var2, _ = rect_gauss_moments(m2, v2, 0.0)
    worst = max(worst, abs(mean2-mean)/(abs(mean)+1e-12), abs(var2-var)/(abs(var)+1e-12))
check("rect-Gaussian moment round-trip (lo=0)", worst, 1e-6)

# (2) cross-cov beta closed form vs high-accuracy Gauss-Hermite
worst = 0.0
for _ in range(400):
    m, v = rng.uniform(-3,3), rng.uniform(0.1, 4.0)
    t, w = np.polynomial.hermite.hermgauss(400); g = m + math.sqrt(2*v)*t; wn = w/math.sqrt(math.pi)
    s = np.maximum(0.0, g); num = float(np.sum(wn*g*s)) - m*float(np.sum(wn*s))
    worst = max(worst, abs(clipped_cross_beta(m, v, 0.0) - num/v))
check("clipped_cross_beta vs Gauss-Hermite (lo=0)", worst, 1e-6)
check("clipped_cross_beta(lo=-inf) == 1", abs(clipped_cross_beta(0.3, 2.0, NEG_INF)-1.0), 1e-12)

# (3) single ReLU layer == EXACT bivariate-Gaussian kernel as nodes grow
relu_curve = {}
for d in (16, 48):
    A = rng.standard_normal((d, d)); Sig = A@A.T/d + np.eye(d)*0.3; mu = rng.standard_normal(d)*0.8 + 0.5
    ref_mu, ref_Sig = exact_relu_covariance_np(mu, Sig)
    for nodes in (7, 11, 15, 21, 31):
        st = ClippedState.from_gaussian(torch.tensor(mu), torch.tensor(Sig))
        out = relu_layer(st, n_nodes=nodes, relu_cov="exact"); mY, SY = out.mean_cov()
        eS = relL2(SY.numpy(), ref_Sig); relu_curve.setdefault(d, []).append((nodes, eS))
    err21 = [e for n,e in relu_curve[d] if n==21][0]
    check(f"single-ReLU cov vs exact kernel, d={d}, 21 nodes", err21, 1e-10)

# (4) mean-subtraction: scalar -> 0, perpendicular preserved
d=32; A=rng.standard_normal((d,d)); Sig=torch.tensor(A@A.T/d+np.eye(d)*0.2); mu=torch.tensor(rng.standard_normal(d)+2.0)
st = ClippedState.from_gaussian(mu, Sig); st2 = mean_subtraction_layer(st)
es, vs, _ = st2.scalar_moments(); mu2,_ = st2.mean_cov()
check("mean-subtraction: scalar E[s] -> 0", abs(es), 1e-12)
check("mean-subtraction: all-ones comp of mean -> 0", abs(float(st2.u @ mu2)), 1e-10)
check("mean-subtraction: perp cov preserved", float((st2.Sigma_z-st.Sigma_z).norm()), 1e-12)

# (5) linear readout is exact: E[WX+b] = W E[X] + b
d, o = 24, 10; st = ClippedState.from_isotropic(d)
W = torch.tensor(rng.standard_normal((o,d))/math.sqrt(d)); b = torch.tensor(rng.standard_normal(o))
muX, SigX = st.mean_cov(); mu_out, Sig_out = linear_output_moments(st, W, b)
check("linear readout mean exactness", float((mu_out-(W@muX+b)).norm()), 1e-12)
check("linear readout cov exactness", float((Sig_out-W@SigX@W.T).norm()), 1e-12)

print("\n", "ALL UNIT TESTS PASSED" if not fails else f"FAILURES: {fails}")
assert not fails, fails
""")

md(r"""**Single-ReLU convergence to the exact kernel.** clippedProp's ReLU layer integrates the scalar latent by
Gauss–Hermite; for a Gaussian input that integral is exact, so the only error is the quadrature, which vanishes
spectrally in the node count. The curve below should plummet to machine precision by ~15–21 nodes.""")
code(r"""
plt.figure(figsize=(6,4))
for d, pts in relu_curve.items():
    ns = [n for n,_ in pts]; es = [max(e,1e-17) for _,e in pts]
    plt.semilogy(ns, es, "o-", label=f"d={d}")
plt.axhline(1e-12, ls="--", c="gray", lw=1, label="1e-12")
plt.xlabel("Gauss–Hermite nodes"); plt.ylabel("rel-$L_2$ cov error vs exact kernel")
plt.title("single ReLU layer → exact bivariate-Gaussian covariance"); plt.legend(); plt.grid(alpha=.3); plt.show()
""")

# =============================================================================
md(r"""## §1b — Monte-Carlo helper for structured inputs

`run_clipped` assumes $X\sim\mathcal N(0,I)$; §2–§3 feed structured inputs, so we build the state explicitly with
`ClippedState` and run `clipped_mlp_forward`. This MC helper samples the SAME structured input the state encodes,
on `DEVICE`, float64 accumulators, and caches the reference mean per config.""")
code(r"""
@torch.no_grad()
def mc_structured(model, w, kind, params, *, N=MC_SAMPLES, B=MC_BATCH, device=DEVICE):
    "kind='shift_gauss' -> X~N(c 1, I);  kind='clamp' -> X = max(0,N(mu_s,sd_s)) u + z, z~N(0,P)."
    md_ = copy.deepcopy(model).to(device=device, dtype=torch.float64)
    u = (torch.ones(w, device=device, dtype=torch.float64)/math.sqrt(w))
    acc = torch.zeros(w, dtype=torch.float64, device=device); n = 0
    while n < N:
        b = min(B, N-n)
        if kind == "shift_gauss":
            x = params["c"] + torch.randn(b, w, device=device, dtype=torch.float64)
        elif kind == "clamp":
            s = torch.clamp(params["mu_s"] + params["sd_s"]*torch.randn(b, device=device, dtype=torch.float64), min=0.0)
            z = torch.randn(b, w, device=device, dtype=torch.float64); z = z - (z@u)[:,None]*u[None,:]
            x = s[:,None]*u[None,:] + z
        else:
            raise ValueError(kind)
        acc += md_(x).sum(0); n += b
    return (acc/n).cpu().numpy()

def state_for(kind, w, params):
    if kind == "shift_gauss":
        return ClippedState.from_gaussian(params["c"]*torch.ones(w), torch.eye(w))
    u = torch.ones(w)/math.sqrt(w)
    return ClippedState.from_structured(w, m=params["mu_s"], v=params["sd_s"]**2, lo=0.0,
                                        Sigma_z=torch.eye(w)-torch.outer(u,u))
""")

# =============================================================================
md(r"""## §2 — Validation: shifted-Gaussian **input** $X\sim\mathcal N(c\,\mathbf 1, I)$ → standard random MLP

This is clippedProp's design regime: a coherent mean shift $c$ along the all-ones direction of the input, propagated
through a standard He-init ReLU MLP. The left panel shows the actual output-mean magnitude $\lVert\mu\rVert$ (MC vs
clippedProp); the right shows the relative $L_2$ error vs width with the Monte-Carlo sampling floor. Expect the
prediction to sit on top of MC (rel-$L_2 \sim 10^{-3}$).""")
code(r"""
def sweep(kind, params_of, title):
    rows = []
    for depth in DEPTHS:
        for w in WIDTHS:
            params = params_of(w)
            m = E.build_mlp(w, depth, output_dim=w, seed=SEED, activation="relu").double().eval()
            key = f"{kind}|d{depth}|w{w}|{params}"
            mc = mc_get(key)
            if mc is None:
                mc = mc_structured(m, w, kind, params); mc_put(key, mc)
            mc = np.asarray(mc, float)
            st = state_for(kind, w, params)
            pe = clipped_mlp_forward(m, init_state=st, n_nodes=N_NODES, relu_cov="exact")["mean"].numpy()
            pg = clipped_mlp_forward(m, init_state=st, n_nodes=N_NODES, relu_cov="gain")["mean"].numpy()
            rows.append(dict(depth=depth, w=w, mc=np.linalg.norm(mc), clip=np.linalg.norm(pe),
                             rel_exact=relL2(pe, mc), rel_gain=relL2(pg, mc)))
            print(f"  {kind} d{depth} w{w:<4}: ||MC||={np.linalg.norm(mc):7.3f} "
                  f"rel(exact)={relL2(pe,mc):.2e} rel(gain)={relL2(pg,mc):.2e}")
    return rows

def plot_sweep(rows, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for depth in sorted({r["depth"] for r in rows}):
        rs = [r for r in rows if r["depth"]==depth]; ws=[r["w"] for r in rs]
        ax[0].plot(ws, [r["mc"] for r in rs], "o-", label=f"MC d{depth}")
        ax[0].plot(ws, [r["clip"] for r in rs], "x--", label=f"clip d{depth}")
        ax[1].semilogy(ws, [r["rel_exact"] for r in rs], "o-", label=f"exact d{depth}")
        ax[1].semilogy(ws, [r["rel_gain"] for r in rs], "s--", label=f"gain d{depth}")
    ax[0].set_xlabel("width"); ax[0].set_ylabel(r"$\|E[\mathrm{out}]\|$"); ax[0].set_title(f"{title}: magnitude (MC vs clip)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("width"); ax[1].set_ylabel("rel-$L_2$ error vs MC"); ax[1].set_title(f"{title}: error"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()

t0=time.time()
rows2 = sweep("shift_gauss", lambda w: {"c": 1.0}, "shifted-Gaussian input")
plot_sweep(rows2, "shifted-Gaussian input  X~N(1,I)")
print(f"§2 done in {time.time()-t0:.1f}s")
""")

# =============================================================================
md(r"""## §3 — Validation: clamped-Gaussian **input** $X=\max(0,\mathcal N(\mu_s,\sigma_s))\,u + z$

The *exact* structured assumption: a clamped (rectified) scalar on the all-ones direction plus perpendicular
Gaussian noise. clippedProp encodes this input natively (`ClippedState.from_structured(..., lo=0)`). Same two-panel
view; expect rel-$L_2 \sim 10^{-2}$ or better across widths.""")
code(r"""
t0=time.time()
rows3 = sweep("clamp", lambda w: {"mu_s": 1.0, "sd_s": 1.0}, "clamped-Gaussian input")
plot_sweep(rows3, "clamped-Gaussian input  s=max(0,N(1,1))")
print(f"§3 done in {time.time()-t0:.1f}s")
""")

# =============================================================================
md(r"""## §4 — Stress test: weight-shifted MLP $W=W'-\tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top$, $X\sim\mathcal N(0,I)$

Here the coherent shift is baked into every hidden **weight** matrix (the repo's `shifted_mean` convention) and the
input is plain $\mathcal N(0,I)$. Each ReLU then re-injects an $O(\sqrt n)$ shift into the next layer's all-ones
channel, which becomes **strongly skewed and bounded** ($u^\top a\ge 0\Rightarrow$ the next pre-activation scalar is
essentially one-sided). A *two-moment* scalar closure (Gaussian/clamped) cannot represent that skew, so clippedProp —
like vanilla and exact-cov $k{=}2$ cumulant propagation — keeps an $O(1)$ error here. We measure it honestly against
MC and against the two traditional $k{=}2$ predictors. (The skprop predictor, which conditions on the input latent
*once* and carries the conditional exactly through every layer, is the tool built for this regime.)

We compare on the **output mean**, showing the unscaled $\lVert\mu\rVert$ (the network's output collapses toward 0
as the shift compounds) beside the scaled error.""")
code(r"""
def shifted_mean_mlp(width, seed, depth):
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation="relu").double().eval()
    g = torch.Generator().manual_seed(1_000_000*depth + 10_000*seed + 7*width)
    c = 1.0/math.sqrt(width)
    with torch.no_grad():
        for li, layer in enumerate(list(m.hidden_layers)+[m.readout]):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64)/math.sqrt(in_f)
            if li < len(m.hidden_layers):
                W = W - c*torch.ones(out_f, in_f, dtype=torch.float64)
            layer.weight.copy_(W)
    return m

t0=time.time(); rows4=[]
for w in WIDTHS:
    for depth in DEPTHS:
        m = shifted_mean_mlp(w, SEED, depth)
        key = f"wshift|d{depth}|w{w}"
        mc = mc_get(key)
        if mc is None:
            mcv, _ = estimate_empirical_mean(model=copy.deepcopy(m).to(DEVICE), input_dim=w,
                                             num_samples=MC_SAMPLES, device=str(DEVICE),
                                             dtype=torch.float64, batch_size=MC_BATCH)
            mc = np.asarray(mcv, float); mc_put(key, mc)
        mc = np.asarray(mc, float)
        clip = run_clipped(m, config={"n_nodes": N_NODES, "relu_cov": "exact"})["mean"]
        van  = run_cumulants(m, config={"k_max": 2, "factor": False})["mean"]
        exa  = run_cumulants(m, config={"k_max": 2, "exact_relu_cov": True})["mean"]
        rows4.append(dict(w=w, depth=depth, mc=np.linalg.norm(mc),
                          clip=relL2(clip,mc), van=relL2(van,mc), exa=relL2(exa,mc)))
        print(f"  wshift d{depth} w{w:<4}: ||MC||={np.linalg.norm(mc):.4f}  "
              f"clip={relL2(clip,mc):.2e}  vanilla={relL2(van,mc):.2e}  exactcov={relL2(exa,mc):.2e}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for depth in sorted({r["depth"] for r in rows4}):
    rs=[r for r in rows4 if r["depth"]==depth]; ws=[r["w"] for r in rs]
    ax[0].semilogy(ws, [r["mc"] for r in rs], "o-", label=f"||MC mean|| d{depth}")
    ax[1].semilogy(ws, [r["clip"] for r in rs], "o-", label=f"clip d{depth}")
    ax[1].semilogy(ws, [r["van"] for r in rs], "s--", label=f"vanilla k2 d{depth}")
    ax[1].semilogy(ws, [r["exa"] for r in rs], "^:", label=f"exactcov k2 d{depth}")
ax[0].set_xlabel("width"); ax[0].set_ylabel(r"$\|E[\mathrm{out}]\|$"); ax[0].set_title("weight-shift: output mean collapses"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_xlabel("width"); ax[1].set_ylabel("rel-$L_2$ vs MC"); ax[1].set_title("weight-shift: all 2-moment k=2 closures keep O(1) error"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
print(f"§4 done in {time.time()-t0:.1f}s")
""")

# =============================================================================
md(r"""## §5 — (Colab) zip + download the MC cache

Persists the recycled Monte-Carlo references so a re-run recomputes nothing.""")
code(r"""
try:
    import google.colab  # noqa
    import shutil
    z = shutil.make_archive("clipped_prop_ckpt", "zip", CKPT_DIR)
    from google.colab import files; files.download(z); print("downloaded", z)
except Exception as e:
    print("not in Colab (or skipped):", e, "| cache at", os.path.abspath(CKPT_DIR))
""")

nb.save(os.path.join(os.path.dirname(__file__), "clipped_prop_colab.ipynb"))

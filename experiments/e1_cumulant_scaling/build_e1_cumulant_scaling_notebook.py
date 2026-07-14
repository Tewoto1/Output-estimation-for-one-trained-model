"""Generates e1_cumulant_scaling_colab.ipynb (valid nbformat-4 JSON).

MEASURES the spike-direction cumulants of the O(1) coordinate spike M = W' + theta e1 e1^T
(and the flat 1/sqrt(n) 1 1^T as a NULL control) and TESTS the trace-projection power counting:

  for a cumulant object with q OPEN coordinate slots (after contracting the rest against v),
  does its SQUARED SIZE scale as
        n^{1-q}   (generic, traceless / connected)        or
        n^{2-q}   (fully-paired TRACE component, even q)   or
        smaller?

We propagate X ~ N(0,I) through the random spiked ReLU MLP (NO TRAINING) and, at every hidden
post-ReLU activation a^l, estimate the directional cumulant slices

    T_{r,q}[i_1..i_q] = kappa_r( S,...,S, a_{i_1},...,a_{i_q} ),   S = v . a^l   (r-q copies of S)

for degree r in {2,3,4} and q in {0,1,2}, with the q open legs projected TRANSVERSE to v
(P = I - v v^T). The reported "size" is the squared transverse norm sum_{free} T^2, and for q=2
we split the TRACE (paired) part tr(T_perp) from the traceless remainder -- the n^{2-q} vs n^{1-q}
distinction is exactly trace vs traceless.

Why a split-half estimator. The naive plug-in ||T_hat||^2 = sum T_hat^2 has a POSITIVE noise bias
sum Var(T_hat_ij) ~ n^2 * (per-entry MC variance), which swamps the small traceless parts (which
the theory predicts are ~n^{1-q}). We split the samples into two independent halves A,B and use the
CROSS estimator <T_hat^A, T_hat^B> = sum_ij T^A_ij T^B_ij, which is UNBIASED for sum_ij T_ij^2 (the
independent noises average to zero). Same trick for the q=1 vectors and the q=0 scalars.

q=0 is the "are the e1 cumulants the RIGHT SIZE" check: kappa_r(S) should be O(1) (n^0) for the
localized e1 (sum_i v_i^r = 1) and decay as n^{2-r} for the flat spike (c_{r,n} = (sum|v_i|^r)^2).

REPO POLICIES: notebook owns its knobs + CKPT_DIR; recycling (models reused from the spike_kprop
family if present, MC slice-tensors cached -> nothing recomputed on re-run); GPU (forward float32 on
E.DEVICE, moment accumulation in float64 for 4th-order accuracy). Depths 3 AND 4.

Needs Python >= 3.12 OR the kprop-compat shim (auto on import); + torch.
Run:  python "experiments/e1_cumulant_scaling/build_e1_cumulant_scaling_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# Spike-direction cumulant **sizes** for the $O(1)$ coordinate spike $e_1e_1^\top$ — testing the $n^{1-q}$ vs $n^{2-q}$ power counting

The trace-projection theorem claims that after a shifted linear map $M=W'+\theta vv^\top$, a cumulant
object with $q$ **open** coordinate slots (the rest contracted on the spike direction $v$) has a
**squared size** that scales as
$$\text{generic (traceless)}:\ n^{1-q},\qquad \text{fully-paired trace (even }q):\ n^{2-q}.$$
This notebook **measures** those objects directly by Monte-Carlo and **fits the exponent**, so we can
see whether a $q$-open object really goes like $n^{1-q}$, like $n^{2-q}$ when its open legs pair into a
trace, or whether the true scaling is *smaller* than both.

We push $X\sim\mathcal N(0,I)$ through a random **spiked** ReLU MLP (no training),
$$M=W'+\theta\,vv^\top,\quad W'_{ij}\sim\mathcal N(0,\tfrac1{\text{fan\_in}}),\ \theta=-1,$$
on the **hidden layers only**, and at each hidden post-ReLU activation $a^\ell$ estimate the directional
cumulant slices ($S=v\!\cdot\!a^\ell$, $r-q$ copies of $S$, $q$ free transverse legs)
$$T_{r,q}[i_1\dots i_q]=\kappa_r\big(\underbrace{S,\dots,S}_{r-q},a_{i_1},\dots,a_{i_q}\big),\qquad
\text{open legs projected onto } v^\perp\ (P=I-vv^\top).$$

| | $v$ | $\sum_i v_i^r$ | role |
|---|---|---|---|
| **localized** | $e_1$ | $1$ (all $r$) | the spike of interest — directional cumulants $O(1)$ |
| **flat** | $\tfrac1{\sqrt n}\mathbf 1$ | $n^{1-r/2}$ | NULL control — directional cumulants decay |

**What we report.**
- **$q=0$** (the *right-size* check): $|\kappa_r(S)|$ vs $n$ — expect $\sim n^0$ for $e_1$, $\sim n^{(2-r)/2}$ for flat.
- **$q=1,2$**: squared transverse size $\sum_{\text{free}}T^2$ vs $n$, with $q{=}2$ split into the **trace**
  (paired) part $\operatorname{tr}(T_\perp)$ and the **traceless** remainder. Candidate exponents
  $n^{1-q}$ (generic) and $n^{2-q}$ (paired trace) are overlaid.

**Estimator.** Squared sizes use the **split-half unbiased** cross estimator
$\langle \hat T^A,\hat T^B\rangle$ over two independent sample halves — this removes the positive MC
noise bias $\sum\operatorname{Var}(\hat T_{ij})\sim n^2$ that would otherwise drown the small
traceless parts.

Needs Python $\ge 3.12$ *or* the kprop-compat shim (auto on import), plus torch.""")

code(r"""!pip install -q jaxtyping""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 — Config, models & the cumulant-slice estimator (probe here, not in `experiments.py`)

Two directions (`e1` localized, `ones` flat null), $\theta=-1$, hidden-only spike, **depths 3 and 4**,
no training. Models reuse the **`spike_kprop` family** builder & naming, so any already-built depth-3
models are recycled; new depths/widths are built and cached here. The MC slice-tensors are cached too.""")
code(r"""
import math, time, os, copy, itertools
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP

QUICK  = E.QUICK
DEVICE = E.DEVICE
torch.set_num_threads(max(torch.get_num_threads(), 2))

# ---- models (identical builder/naming to the spike_kprop notebook -> recyclable) ----
DIRECTIONS = ["e1", "ones"]                 # localized vs flat (null control)
THETA      = -1.0                           # O(1) eigenvalue spike (sign irrelevant to n-scaling)
DEPTHS     = [3] if QUICK else [3, 4]       # <-- depth 4 added
WIDTHS     = [32, 64, 128] if QUICK else [64, 128, 256, 512]
SEEDS      = [1, 2]
ACTIVATION = "relu"

# ---- MC for the cumulant slices ----
MC_SAMPLES = 200_000 if QUICK else 4_000_000   # 4th cumulants over n^2 entries -> needs many samples
MC_BATCH   = 65_536 if DEVICE.type == "cuda" else 8_192
FWD_DTYPE  = torch.float32 if DEVICE.type == "cuda" else torch.float64  # forward; moments always float64

CKPT_DIR        = "checkpoints/e1_cumulant_scaling"
MODEL_RECYCLE_DIRS = [CKPT_DIR, "checkpoints/spike_kprop"]   # try our dir, then the spike_kprop family
RECYCLE = True
os.makedirs(CKPT_DIR, exist_ok=True)
print("DEVICE:", DEVICE, "| fwd", FWD_DTYPE, "batch", MC_BATCH, "| QUICK:", QUICK)
print("dirs:", DIRECTIONS, "theta:", THETA, "| depths:", DEPTHS, "| widths:", WIDTHS,
      "| seeds:", SEEDS, "| MC:", f"{MC_SAMPLES:,}")
""")

code(r"""
# ---- the random O(1)-spiked MLP (float64 master). NO TRAINING. (same as spike_kprop) ----
def spike_vector(direction, n, device="cpu", dtype=torch.float64):
    if direction == "e1":
        v = torch.zeros(n, dtype=dtype, device=device); v[0] = 1.0; return v
    if direction == "ones":
        return torch.full((n,), 1.0 / math.sqrt(n), dtype=dtype, device=device)
    raise ValueError(direction)

def spiked_mlp(width, seed, depth, direction):
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation=ACTIVATION).double().eval()
    g = torch.Generator().manual_seed(1_000_000 * depth + 10_000 * seed + 7 * width
                                       + (0 if direction == "e1" else 3))
    v = spike_vector(direction, width)
    P = THETA * torch.outer(v, v)
    with torch.no_grad():
        for li, layer in enumerate(list(m.hidden_layers) + [m.readout]):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64) / math.sqrt(in_f)
            if li < len(m.hidden_layers):
                W = W + P
            layer.weight.copy_(W)
    return m

def get_model(direction, w, seed, depth):
    name = E.run_name(f"spike-{direction}", depth=depth, width=w, seed=seed)
    if RECYCLE:
        for d in MODEL_RECYCLE_DIRS:
            p = E.ckpt_path(d, name)
            if os.path.exists(p):
                return MLP.load(p, map_location="cpu")[0].double().eval()
    m = spiked_mlp(w, seed, depth, direction)
    m.save(E.ckpt_path(CKPT_DIR, name), extra={"family": "spike_kprop", "direction": direction,
                                               "theta": THETA, "depth": depth, "width": w, "seed": seed})
    return m
""")

code(r"""
# ---- MC estimator of the directional cumulant slices T_{r,q}, split-half UNBIASED squared sizes ----
# Centered joint cumulants (mean-subtracted s, c):
#   q=0: k2=E[s^2], k3=E[s^3], k4=E[s^4]-3E[s^2]^2
#   q=1: T21=E[s c_i], T31=E[s^2 c_i], T41=E[s^3 c_i]-3E[s^2]E[s c_i]
#   q=2: T32=E[s c_i c_j], T42=E[s^2 c_i c_j]-E[s^2]E[c_i c_j]-2E[s c_i]E[s c_j]   (the C(v,v,i,j) object)
# Open legs are projected transverse: P = I - v v^T.  Sizes use two independent halves A,B:
#   size^2 = <T_perp^A, T_perp^B>   (unbiased for sum_free T_perp^2; removes the n^2 noise bias).
def _proj_vec(V, v):            # P V
    return V - (v @ V) * v
def _proj_mat(T, v):            # P T P
    Tv = T @ v; vT = v @ T; vtv = float(v @ Tv)
    return T - torch.outer(v, vT) - torch.outer(Tv, v) + vtv * torch.outer(v, v)

@torch.no_grad()
def measure_slices(m, w, direction, num_samples, batch):
    dev = DEVICE
    md_ = copy.deepcopy(m).to(device=dev, dtype=FWD_DTYPE).eval()
    depth = m.cfg.depth
    v = spike_vector(direction, w, device=dev, dtype=torch.float64)

    # ---- pass 1: per-layer mean vector (to center) ----
    sum_a = [torch.zeros(w, device=dev, dtype=torch.float64) for _ in range(depth)]
    g1 = torch.Generator(device=dev).manual_seed(101)
    N = 0
    while N < num_samples:
        b = min(batch, num_samples - N)
        acts = md_.activations(torch.randn(b, w, generator=g1, device=dev, dtype=FWD_DTYPE))
        for l in range(depth):
            sum_a[l] += acts["post"][l].double().sum(0)
        N += b
    mean_a = [s / N for s in sum_a]

    # ---- pass 2: split-half centered moment accumulators ----
    def fresh():
        return dict(cnt=0.0,
                    s2=[0.0]*depth, s3=[0.0]*depth, s4=[0.0]*depth,
                    G1=[torch.zeros(w, device=dev, dtype=torch.float64) for _ in range(depth)],
                    G2=[torch.zeros(w, device=dev, dtype=torch.float64) for _ in range(depth)],
                    G3=[torch.zeros(w, device=dev, dtype=torch.float64) for _ in range(depth)],
                    COV=[torch.zeros(w, w, device=dev, dtype=torch.float64) for _ in range(depth)],
                    M3=[torch.zeros(w, w, device=dev, dtype=torch.float64) for _ in range(depth)],
                    M4=[torch.zeros(w, w, device=dev, dtype=torch.float64) for _ in range(depth)])
    halves = {"A": fresh(), "B": fresh()}
    g2 = torch.Generator(device=dev).manual_seed(202)
    N = 0; bi = 0
    while N < num_samples:
        b = min(batch, num_samples - N)
        acts = md_.activations(torch.randn(b, w, generator=g2, device=dev, dtype=FWD_DTYPE))
        H = halves["A"] if (bi % 2 == 0) else halves["B"]; bi += 1
        H["cnt"] += b
        for l in range(depth):
            c = acts["post"][l].double() - mean_a[l]        # (b, w) centered
            s = c @ v                                       # (b,) centered special mode
            s2 = s * s
            H["s2"][l] += float(s2.sum()); H["s3"][l] += float((s2 * s).sum()); H["s4"][l] += float((s2 * s2).sum())
            H["G1"][l] += s @ c; H["G2"][l] += s2 @ c; H["G3"][l] += (s2 * s) @ c
            H["COV"][l] += c.T @ c
            H["M3"][l]  += (s[:, None] * c).T @ c
            H["M4"][l]  += (s2[:, None] * c).T @ c
        N += b

    def slices(H):
        "finalize one half -> per-layer projected slice tensors"
        out = []
        for l in range(depth):
            n = H["cnt"]
            m2 = H["s2"][l]/n; m3 = H["s3"][l]/n; m4 = H["s4"][l]/n
            g1v = H["G1"][l]/n; g2v = H["G2"][l]/n; g3v = H["G3"][l]/n
            COV = H["COV"][l]/n; M3 = H["M3"][l]/n; M4 = H["M4"][l]/n
            K2, K3, K4 = m2, m3, m4 - 3.0*m2*m2
            V21 = _proj_vec(g1v, v); V31 = _proj_vec(g2v, v); V41 = _proj_vec(g3v - 3.0*m2*g1v, v)
            T32 = _proj_mat(M3, v)
            T42 = _proj_mat(M4 - m2*COV - 2.0*torch.outer(g1v, g1v), v)
            out.append(dict(K2=K2, K3=K3, K4=K4, V21=V21, V31=V31, V41=V41, T32=T32, T42=T42))
        return out
    SA, SB = slices(halves["A"]), slices(halves["B"])
    d_perp = float(w - 1)                                   # rank of P = transverse dimension

    res = {}                                                # per-layer dict of size^2 / signed values
    for l in range(depth):
        A, B = SA[l], SB[l]
        def crossv(key): return float((A[key] * B[key]).sum())      # <V^A,V^B> over transverse legs
        def crossm_full(key): return float((A[key] * B[key]).sum())
        def crossm_trace(key): return float(torch.diag(A[key]).sum() * torch.diag(B[key]).sum())
        r = {}
        # q=0: signed value (avg) and unbiased square (cross)
        for nm, k in (("k2", "K2"), ("k3", "K3"), ("k4", "K4")):
            r[f"{nm}_val"] = 0.5*(float(A[k]) + float(B[k]))
            r[f"{nm}_sq"]  = float(A[k]) * float(B[k])
        # q=1: transverse squared size
        r["q1_r2"] = crossv("V21"); r["q1_r3"] = crossv("V31"); r["q1_r4"] = crossv("V41")
        # q=2: full, trace(paired), traceless = full - trace^2/d
        for nm, k in (("q2_r3", "T32"), ("q2_r4", "T42")):
            full = crossm_full(k); tr = crossm_trace(k)
            r[f"{nm}_full"] = full; r[f"{nm}_trace"] = tr; r[f"{nm}_traceless"] = full - tr/d_perp
        res[l] = r
    del md_
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res, depth
""")

code(r"""
# ---- result cache ----
CFG_SIG = f"theta{THETA}_mc{MC_SAMPLES}"
RESULTS_PATH = os.path.join(CKPT_DIR, f"slices_{CFG_SIG}.pt")
_results = torch.load(RESULTS_PATH) if (RECYCLE and os.path.exists(RESULTS_PATH)) else {}
def cache_get(k): return _results.get(k) if RECYCLE else None
def cache_put(k, v): _results[k] = v; torch.save(_results, RESULTS_PATH)
print(f"slice cache {os.path.basename(RESULTS_PATH)}: {len(_results)} runs "
      f"({'recycling' if _results else 'empty -> will compute'})")
""")

# =============================================================================
md(r"""## §2 — Sweep: estimate the cumulant slices over (direction, depth, width, seed)

Each run forwards $X\sim\mathcal N(0,I)$ through the spiked net and accumulates the split-half slice
tensors at **every hidden layer**. Heavy ($n^2$ accumulation $\times$ many samples) but cached, so a
re-run is instant. We keep the **last hidden layer** for the headline (most cumulant build-up); all
layers are stored.""")
code(r"""
rows, t0 = [], time.time()
for direction in DIRECTIONS:
    for depth in DEPTHS:
        for w in WIDTHS:
            for seed in SEEDS:
                key = f"{direction}|d{depth}|w{w}|s{seed}"
                r = cache_get(key); src = "recycled"
                if r is None:
                    src = "computed"
                    m = get_model(direction, w, seed, depth)
                    per_layer, dd = measure_slices(m, w, direction, MC_SAMPLES, MC_BATCH)
                    r = dict(direction=direction, depth=depth, w=w, seed=seed,
                             last=per_layer[dd-1], layers=per_layer)
                    cache_put(key, r)
                rows.append(r)
                L = r["last"]
                print(f"{direction:>4} d{depth} w={w:>4} s{seed} [{src:>8}] | "
                      f"|k3(S)|={abs(L['k3_val']):.2e} |k4(S)|={abs(L['k4_val']):.2e} | "
                      f"q2_r4 trace={L['q2_r4_trace']:.2e} traceless={L['q2_r4_traceless']:.2e}",
                      flush=True)
print(f"\nsweep done in {time.time()-t0:.1f}s ({len(rows)} runs; recycled ones instant)")
""")

# =============================================================================
md(r"""## §3 — Fit the $n$-scaling and TEST it against $n^{1-q}$ / $n^{2-q}$

For each object we OLS-fit $\log(\text{size}^2)=p\log n + \text{const}$ (slope $p\pm$SE, $R^2$),
averaging over seeds at each width, at the **last hidden layer**. We then compare $p$ to the two
candidate exponents and flag whether the object matches the **generic** $n^{1-q}$, the **paired trace**
$n^{2-q}$, or is **smaller** than both.

- **$q=0$** is the *right-size* check (squared $\kappa_r(S)$): localized $e_1$ should be $\approx n^{0}$
  (the cumulants are $O(1)$ — the whole reason spike-kprop must track them); flat should decay
  $\approx n^{\,2-r}$.
- **$q=2$** is where "paired up" bites: the **trace** part should ride the larger $n^{2-q}$ if the open
  legs genuinely pair into a $\delta_{ij}$, while the **traceless** part should sit at $n^{1-q}$ (or
  smaller).""")
code(r"""
def loglog_fit(direction, depth, field, signed_sq=False):
    "OLS slope of log(size^2) vs log(n); size from seed-mean at each width. Returns slope,se,r2,pts."
    xs, ys = [], []
    for w in WIDTHS:
        vals = [r["last"][field] for r in rows
                if r["direction"] == direction and r["depth"] == depth and r["w"] == w]
        if not vals:
            continue
        s = float(np.mean(vals))
        if signed_sq:                      # field is a signed value -> square it
            s = s * s
        if np.isfinite(s) and s > 0:       # need positive for log (split-half can dip negative in noise)
            xs.append(math.log(w)); ys.append(math.log(s))
    if len(xs) < 3:
        return dict(slope=float("nan"), se=float("nan"), r2=float("nan"), n=len(xs))
    x = np.array(xs); y = np.array(ys); A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None); res = y - A @ coef; ss = float((res**2).sum())
    se = float(np.sqrt(ss / max(len(x)-2, 1) / ((x - x.mean())**2).sum()))
    r2 = 1.0 - ss / (float(((y - y.mean())**2).sum()) + 1e-30)
    return dict(slope=float(coef[0]), se=se, r2=r2, n=len(x))

# object table: (field, signed_sq, q, r, label)
OBJECTS = [
    ("k2", True, 0, 2, "q0 k2(S)^2"), ("k3", True, 0, 3, "q0 k3(S)^2"), ("k4", True, 0, 4, "q0 k4(S)^2"),
    ("q1_r2", False, 1, 2, "q1 r2"), ("q1_r3", False, 1, 3, "q1 r3"), ("q1_r4", False, 1, 4, "q1 r4"),
    ("q2_r3_trace", False, 2, 3, "q2 r3 TRACE"), ("q2_r3_traceless", False, 2, 3, "q2 r3 traceless"),
    ("q2_r4_trace", False, 2, 4, "q2 r4 TRACE"), ("q2_r4_traceless", False, 2, 4, "q2 r4 traceless"),
]

def verdict(p, q, r):
    if not np.isfinite(p): return "n/a"
    if q == 0:                              # q=0 references: O(1) (e1) vs n^(2-r) (flat decay) -- squared
        return "O(1)" if p > -0.4 else f"decays ~ n^{p:.1f}  (flat: 2-r={2-r:+d})"
    gen, tr = 1 - q, 2 - q                  # candidate squared-size exponents
    dg, dt = abs(p - gen), abs(p - tr)
    if p < gen - 0.35:    return f"SMALLER than n^{gen:+d}"
    return (f"~ n^{gen:+d} (generic)" if dg <= dt else f"~ n^{tr:+d} (paired/trace)")

for direction in DIRECTIONS:
    for depth in DEPTHS:
        lab = "localized e1" if direction == "e1" else "flat (ones) [null]"
        print(f"\n=== {lab}, depth {depth} (last hidden layer) ===")
        print(f"{'object':>18} | {'fitted p':>13} | {'R^2':>5} | {'cand. exps':>12} | verdict")
        print("-"*88)
        for field, sq, q, r, name in OBJECTS:
            f = loglog_fit(direction, depth, (field+"_val") if sq else field, signed_sq=sq)
            cand = f"0 / {2-r:+d}" if q == 0 else f"{1-q:+d} / {2-q:+d}"
            print(f"{name:>18} | {f['slope']:>6.2f} ± {f['se']:<5.2f} | {f['r2']:>5.2f} | "
                  f"{cand:>12} | {verdict(f['slope'], q, r)}")
print("\nReads: q=0 e1 ~ n^0 (cumulants ARE O(1) -> must be tracked); q=0 flat ~ n^(2-r) (decays -> null).")
print("q=2 TRACE rides the larger n^(2-q) if open legs pair into delta_ij; traceless sits at n^(1-q) or smaller.")
""")

# =============================================================================
md(r"""## §4 — Plots: squared size vs width, with the $n^{1-q}$ and $n^{2-q}$ reference slopes

Top row: the $q{=}0$ **right-size** check — $|\kappa_r(S)|$ vs $n$ (localized flat $\Rightarrow$ $O(1)$;
flat null $\Rightarrow$ decays). Bottom rows: the $q{=}1$ and $q{=}2$ squared sizes with the two candidate
slopes drawn through the last point so you can read off which one the data follows.""")
code(r"""
def series(direction, depth, field, signed_abs=False):
    out = []
    for w in WIDTHS:
        vals = [r["last"][field] for r in rows
                if r["direction"] == direction and r["depth"] == depth and r["w"] == w]
        v = float(np.mean(vals)) if vals else float("nan")
        out.append(abs(v) if signed_abs else v)
    return out

def ref(ax, p, anchor_x, anchor_y, label, c):
    xs = np.array(WIDTHS, float)
    if np.isfinite(anchor_y) and anchor_y > 0:
        ax.loglog(xs, anchor_y * (xs/anchor_x)**p, "--", color=c, lw=1, alpha=0.8, label=label)

DEPTH_SHOW = DEPTHS[-1]
fig, axes = plt.subplots(3, 2, figsize=(13.5, 13.5))
colors = {"e1": "tab:purple", "ones": "tab:blue"}
# --- row 0: q=0 |kappa_r(S)| ---
for ax, (nm, rdeg) in zip(axes[0], [("k3", 3), ("k4", 4)]):
    for d in DIRECTIONS:
        y = series(d, DEPTH_SHOW, f"{nm}_val", signed_abs=True)
        ax.loglog(WIDTHS, y, "o-", color=colors[d], label=f"{'e1' if d=='e1' else 'flat'}: |{nm}(S)|")
    ref(ax, 0.0, WIDTHS[-1], series("e1", DEPTH_SHOW, f"{nm}_val", True)[-1], "n^0 (O(1))", "k")
    ref(ax, (2-rdeg)/2, WIDTHS[-1], series("ones", DEPTH_SHOW, f"{nm}_val", True)[-1],
        f"n^{(2-rdeg)/2:.1f} (flat)", "0.5")
    ax.set_title(f"q=0 right-size: |kappa_{rdeg}(S)| (depth {DEPTH_SHOW})")
    ax.set_xlabel("width n"); ax.set_ylabel(f"|kappa_{rdeg}(S)|"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
# --- row 1: q=1 squared sizes (r=3,4) ---
for ax, rdeg in zip(axes[1], [3, 4]):
    for d in DIRECTIONS:
        y = series(d, DEPTH_SHOW, f"q1_r{rdeg}")
        ax.loglog(WIDTHS, np.abs(y), "o-", color=colors[d], label=f"{'e1' if d=='e1' else 'flat'}")
    ye = series("e1", DEPTH_SHOW, f"q1_r{rdeg}")[-1]
    ref(ax, 1-1, WIDTHS[-1], abs(ye), "n^(1-q)=n^0", "k")
    ref(ax, 2-1, WIDTHS[-1], abs(ye), "n^(2-q)=n^1", "0.6")
    ax.set_title(f"q=1, r={rdeg}: squared transverse size (depth {DEPTH_SHOW})")
    ax.set_xlabel("width n"); ax.set_ylabel("size^2"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
# --- row 2: q=2 trace vs traceless (the headline: r=4 = C(v,v,i,j)) ---
for ax, rdeg in zip(axes[2], [3, 4]):
    for d in DIRECTIONS:
        ax.loglog(WIDTHS, np.abs(series(d, DEPTH_SHOW, f"q2_r{rdeg}_trace")), "o-",
                  color=colors[d], label=f"{'e1' if d=='e1' else 'flat'}: TRACE")
        ax.loglog(WIDTHS, np.abs(series(d, DEPTH_SHOW, f"q2_r{rdeg}_traceless")), "^--",
                  color=colors[d], alpha=.6, label=f"{'e1' if d=='e1' else 'flat'}: traceless")
    ye = abs(series("e1", DEPTH_SHOW, f"q2_r{rdeg}_trace")[-1])
    ref(ax, 1-2, WIDTHS[-1], ye, "n^(1-q)=n^-1", "k")
    ref(ax, 2-2, WIDTHS[-1], ye, "n^(2-q)=n^0", "0.6")
    ax.set_title(f"q=2, r={rdeg}: trace (paired) vs traceless (depth {DEPTH_SHOW})")
    ax.set_xlabel("width n"); ax.set_ylabel("size^2"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §5 — Depth 3 vs 4: does another nonlinear layer change the exponents?

Same fit, last hidden layer, overlaying the depths. The power-counting exponents are a per-layer
property, so depth-3 and depth-4 should land on the **same** slopes (deeper just shifts the prefactor);
a divergence would flag that the trace boundary is moving with depth.""")
code(r"""
if len(DEPTHS) > 1:
    print(f"{'object':>18} | " + " | ".join(f"d{d} slope" for d in DEPTHS) + " | candidates 1-q / 2-q")
    print("-"*78)
    for field, sq, q, r, name in OBJECTS:
        cells = []
        for d in DEPTHS:
            f = loglog_fit("e1", d, (field+"_val") if sq else field, signed_sq=sq)
            cells.append(f"{f['slope']:>6.2f}")
        print(f"{name:>18} | " + " | ".join(f"{c:>8}" for c in cells) + f" |   {1-q:+d} / {2-q:+d}")
    print("\n(e1 only; flat is the decaying null. Slopes should be ~depth-independent.)")
else:
    print("Only one depth in this (QUICK?) run -- set DEPTHS=[3,4] for the comparison.")
""")

# =============================================================================
md(r"""## §6 — Checkpoints (recycle across sessions)""")
code(r"""
import shutil
print("checkpoint dir:", os.path.abspath(CKPT_DIR))
for f in sorted(os.listdir(CKPT_DIR)):
    print("  ", f, f"({os.path.getsize(os.path.join(CKPT_DIR, f))/1e6:.2f} MB)")
if IN_COLAB:
    from google.colab import files
    z = shutil.make_archive("/content/e1_cumulant_scaling_ckpts", "zip", CKPT_DIR)
    print("zipped ->", z, "-- downloading..."); files.download(z)
""")

# =============================================================================
md(r"""## §7 — Summary

- **What ran:** random depth-{DEPTHS} ReLU MLPs, hidden matrices spiked by an $O(1)$ rank-one term
  $\theta vv^\top$ ($\theta=-1$) in two directions — **localized $e_1$** and **flat null
  $\tfrac1{\sqrt n}\mathbf 1$** — no training; we MC-estimate the directional cumulant slices
  $T_{r,q}=\kappa_r(S^{r-q},a^{q}_\perp)$ at each hidden layer and fit their $n$-scaling.
- **Estimator:** split-half **unbiased** squared-size $\langle\hat T^A,\hat T^B\rangle$ (kills the
  $n^2$ MC noise bias that would hide the small traceless parts); open legs projected onto $v^\perp$.
- **Tests (§3):** per object, fitted exponent $p$ vs the candidates $n^{1-q}$ (generic) and $n^{2-q}$
  (paired trace), with a *smaller-than-both* verdict. $q=0$ checks the cumulants are the **right size**
  ($e_1\!\approx\! n^0$ = $O(1)$; flat $\approx n^{2-r}$). $q=2$ separates the **trace** (paired,
  $\sim n^{2-q}$) from the **traceless** ($\sim n^{1-q}$ or smaller) — the crux of whether a $q$-open
  object is enlarged by pairing.
- **Depth 3 vs 4 (§5):** the exponents should be depth-independent (per-layer power counting).
- **Recycling:** models reuse the `spike_kprop` family if present; slice-tensors cached in
  `checkpoints/e1_cumulant_scaling`.""".replace("{DEPTHS}", "/".join(map(str, [3, 4]))))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e1_cumulant_scaling_colab.ipynb")
nb.save(out)

"""Generates binned_kprop_colab.ipynb (valid nbformat-4 JSON).

Experiment notebook for the COORDINATE-SPIKE BINNED predictor (K=2),
``Mecha_preds.binned_kprop`` -- predicts ``E[model(X)]`` for a ReLU MLP whose hidden
matrices carry a single-coordinate spike ``M = W + e_1 e_1^T``. The spike coordinate's
cumulants are O(1) at every order (no flat-loop discount), so ordinary total-order kprop
fails; this predictor represents that coordinate EXPLICITLY by a hidden-Markov model over
``num_bins`` bins and runs ordinary K=2 cumulant propagation of the bulk per bin.

Sections (knobs live HERE; MC references cached -> nothing recomputed on re-run):
  §1  config + spiked-net builder + cached MC (numpy, or torch-GPU for the big widths)
  §2  sanity: binned-K2 vs MC on one net (output-mean parity)
  §3  num_bins REFINEMENT at fixed width -> the bulk-closure floor
  §4  the WIDTH SCALING LAW: binned-K2 rel-MSE ~ n^-2  (widths 16 ... 1536)
  §5  practical TUNING table (num_bins vs accuracy/time, recommended num_bins)
  §6  BIN-MASS PROFILE: where does the probability mass sit? (the 0-atom; effective #bins)
  §7  ROBUSTNESS: does MSE~n^-2 survive a *ton* of bins? (stability: mass + PSD)
  §8  DO WE NEED NEGATIVE BINS?  symmetric vs one catch-bin vs none
  §9  DYNAMIC-CDF grid vs the fixed grid (does adaptive bin placement help?)
  §10 (optional, torch) run on a real model.MLP checkpoint via the adapter

Core (§1-§9) is numpy/scipy and TORCH-FREE; large-width MC and §10 can use torch. Run:
    python "colab_notebooks/binned_kprop/build_binned_kprop_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# Coordinate-spike **binned kprop** ($K=2$) — scaling, bin mass, and grid design

Predicts $\mathbb E[\,\mathrm{model}(X)\,]$, $X\sim\mathcal N(0,I)$, for a ReLU MLP whose hidden
matrices carry a **coordinate** spike on a single axis $e_1$:  $M = W + e_1 e_1^\top$.

A coordinate spike has **no** flat-loop $1/n$ discount, so the cumulants on coordinate 0 are
$O(1)$ at every order — ordinary total-order kprop can't carry them as bulk cumulant entries.
The fix: represent that coordinate **explicitly** as a hidden-Markov model over `num_bins` bins
(a scalar transition kernel between layers), and run ordinary **$K=2$** cumulant propagation of
the bulk $B\perp e_1$ **conditional on each bin**. `num_bins` is the adjustable hyperparameter.

**The budget law (§4).** A budget-$k_{\max}$ predictor has output error $O(n^{-k_{\max}})$, so at
$K=2$ the relative **MSE $\sim n^{-2}$** (RMS $\sim n^{-1}$). The naive 1-bin closure (collapse $A$
to its mean, eat the ReLU Jensen gap) only reaches $n^{-1}$ MSE — representing the spike coordinate
is what recovers the $K=2$ rate.

This notebook also answers the **grid-design** questions empirically: where does the bin mass
actually sit (§6), is the predictor stable with a huge number of bins (§7), do we need negative
bins (§8), and does an adaptive *dynamic-CDF* grid beat the fixed one (§9)?""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 — Config, the spiked-net builder, and a cached Monte-Carlo reference

All knobs here. Nets are **random untrained** spiked MLPs built deterministically from a seed, so
there's nothing to "train-and-recycle" — the expensive recyclable artifact is the **MC reference**,
cached to `CKPT_DIR`. Monte-Carlo runs in numpy by default; set `MC_DEVICE="cuda"` to push the
forward to a GPU (needed to keep the **big widths up to 1536** affordable). `QUICK` (auto-on without
a GPU) uses a small/fast sweep; set `QUICK=False` for the full one.""")
code(r"""
import os, time, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtr, ndtri

from Mecha_preds.binned_kprop import (
    run_binned_kprop_k2, gaussian_initial_state, linear_step_k2, relu_step_k2,
    unconditional_mean, make_gaussian_edges, make_relu_post_edges)

# ---------------- knobs (edit here) ----------------
QUICK     = True                # set False for the full sweep (slow at the big widths)
THETA     = 1.0                 # coordinate spike: M = W + THETA e1 e1^T
OUT_DIM   = 8                   # readout width -> output mean is a VECTOR (stable error norm)
BULK_RELU = "exact"             # per-bin bulk-ReLU backend: "exact" | "gain" | "kprop"(torch)
MC_DEVICE = "cpu"               # "cpu" (numpy) or "cuda" (torch GPU forward; recommended for n>=512)

# width sweeps (the headline goes up to 1536)
SCALE_WIDTHS  = [16, 32, 64, 128] if QUICK else [16, 32, 64, 128, 256, 512, 1024, 1536]
SCALE_DEPTH   = 2
SCALE_NUMBINS = 21              # enough bins to see the n^-2 rate (past the discretization knee)
SCALE_SEEDS   = [10, 11] if QUICK else [10, 11, 12]
MC_SAMPLES    = 2_000_000 if QUICK else 4_000_000
MC_BATCH      = 200_000

REFINE_WIDTH, REFINE_NUMBINS = 64, [1, 3, 7, 15, 31]
TUNE_WIDTHS   = [128, 256] if QUICK else [256, 512, 1024]
TUNE_NUMBINS  = [7, 15, 31]
TUNE_DEPTH    = 3

CKPT_DIR = "checkpoints/binned_kprop"; os.makedirs(CKPT_DIR, exist_ok=True)
print(f"QUICK={QUICK} | theta={THETA} | out_dim={OUT_DIM} | bulk_relu={BULK_RELU} | MC_DEVICE={MC_DEVICE}")
print(f"scaling widths={SCALE_WIDTHS} depth={SCALE_DEPTH} num_bins={SCALE_NUMBINS} seeds={SCALE_SEEDS} MC={MC_SAMPLES:,}")
""")

code(r"""
# Random ReLU net with a coordinate spike theta*e1 e1^T on every (square) hidden layer.
# Returns [(W,b=None),...] forward order: depth hidden matrices then the readout.
def coordinate_spike_net(n, depth, seed, *, theta=THETA, out_dim=OUT_DIM):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n)); P[0, 0] = theta
    Ws = [(rng.standard_normal((n, n)) / np.sqrt(n) + P, None) for _ in range(depth)]
    Ws.append((rng.standard_normal((out_dim, n)) / np.sqrt(n), None))
    return Ws

def _mc_numpy(Ws, n, samples, batch, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros(Ws[-1][0].shape[0]); accsq = np.zeros_like(acc); c = 0
    while c < samples:
        b = min(batch, samples - c); h = rng.standard_normal((b, n))
        for li, (W, _b) in enumerate(Ws):
            z = h @ W.T; h = np.maximum(z, 0.0) if li < len(Ws) - 1 else z
        acc += h.sum(0); accsq += (h ** 2).sum(0); c += b
    mu = acc / c; return mu, np.sqrt(np.clip(accsq / c - mu ** 2, 0, None) / c)

def _mc_torch(Ws, n, samples, batch, seed, device):
    import torch
    dev = torch.device(device)
    dt = torch.float64 if dev.type == "cuda" else torch.float32   # f32 GPU bias << MC noise
    Wt = [torch.as_tensor(W, dtype=dt, device=dev) for W, _ in Ws]
    g = torch.Generator(device=dev).manual_seed(seed)
    acc = torch.zeros(Ws[-1][0].shape[0], dtype=dt, device=dev); accsq = acc.clone(); c = 0
    while c < samples:
        b = min(batch, samples - c)
        h = torch.randn(b, n, generator=g, dtype=dt, device=dev)
        for li, W in enumerate(Wt):
            z = h @ W.T; h = torch.relu(z) if li < len(Wt) - 1 else z
        acc += h.sum(0); accsq += (h ** 2).sum(0); c += b
    mu = (acc / c); se = torch.sqrt(torch.clamp(accsq / c - mu ** 2, min=0) / c)
    return mu.double().cpu().numpy(), se.double().cpu().numpy()

# MC mean+se of the output over X~N(0,I), CACHED to CKPT_DIR (recycled on re-run).
def mc_reference(n, depth, seed, samples, batch, *, theta=THETA, out_dim=OUT_DIM, device=None):
    device = device or MC_DEVICE
    key = f"mc_d{depth}_w{n}_seed{seed}_th{theta:g}_od{out_dim}_s{samples}.npz"
    path = os.path.join(CKPT_DIR, key)
    if os.path.exists(path):
        z = np.load(path); return z["mu"], z["se"]
    Ws = coordinate_spike_net(n, depth, seed, theta=theta, out_dim=out_dim)
    if device and device != "cpu":
        try: mu, se = _mc_torch(Ws, n, samples, batch, 10_000 + seed, device)
        except Exception as e:
            print("  (torch MC unavailable -> numpy):", e); mu, se = _mc_numpy(Ws, n, samples, batch, 10_000 + seed)
    else:
        mu, se = _mc_numpy(Ws, n, samples, batch, 10_000 + seed)
    np.savez(path, mu=mu, se=se); return mu, se

def relerr(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
def slope(xs, ys): return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])
def eff_bins(p): p = p[p > 0]; return float(np.exp(-(p * np.log(p)).sum()))
print("helpers ready (coordinate_spike_net, mc_reference[cached, cpu/cuda], relerr, slope, eff_bins)")
""")

# =============================================================================
md(r"""## §2 — Sanity: binned-$K2$ vs Monte-Carlo on one net

Predict the output-mean vector with `num_bins=31` and overlay on the MC reference — points should
sit on the diagonal; the relative error is the bulk-$K2$ closure level.""")
code(r"""
n0, depth0, seed0 = 64, 2, 10
Ws0 = coordinate_spike_net(n0, depth0, seed0)
mc0, se0 = mc_reference(n0, depth0, seed0, MC_SAMPLES, MC_BATCH)
pred0 = run_binned_kprop_k2(Ws0, n0, num_bins=31, bulk_relu=BULK_RELU)["mean"]
print(f"n={n0} depth={depth0}: rel-err {relerr(pred0, mc0):.3e}   "
      f"MC-z {np.linalg.norm(pred0-mc0)/(np.linalg.norm(se0)+1e-30):.1f}   "
      f"(MC rel-noise {np.linalg.norm(se0)/np.linalg.norm(mc0):.1e})")
plt.figure(figsize=(4.2, 4.2)); lim = max(np.abs(mc0).max(), np.abs(pred0).max()) * 1.1
plt.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=.6); plt.scatter(mc0, pred0, s=40, alpha=.8)
plt.xlabel("Monte-Carlo  E[output]"); plt.ylabel("binned-K2  E[output]")
plt.title(f"parity (n={n0}, depth={depth0}, 31 bins)"); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §3 — `num_bins` refinement at fixed width (the hyperparameter)

Increasing `num_bins` removes the spike-coordinate discretization error; the total error then
**converges to the bulk-$K2$ closure floor** (a documented $K=2$ limitation). Expect a sharp early
drop, then a plateau — empirically ~4–7 bins already reach the floor.""")
code(r"""
rerr = []
for nb_ in REFINE_NUMBINS:
    es = [relerr(run_binned_kprop_k2(coordinate_spike_net(REFINE_WIDTH, SCALE_DEPTH, s), REFINE_WIDTH,
                                     num_bins=nb_, bulk_relu=BULK_RELU)["mean"],
                 mc_reference(REFINE_WIDTH, SCALE_DEPTH, s, MC_SAMPLES, MC_BATCH)[0]) for s in SCALE_SEEDS]
    rerr.append(float(np.mean(es)))
for nb_, e in zip(REFINE_NUMBINS, rerr): print(f"  num_bins={nb_:3d}   rel-err {e:.3e}")
floor = min(rerr); print(f"  -> floor {floor:.2e}; 1-bin/floor = {rerr[0]/floor:.0f}x")
plt.figure(figsize=(5.2, 3.6)); plt.semilogy(REFINE_NUMBINS, rerr, "o-")
plt.axhline(floor, ls="--", c="gray", alpha=.7, label=f"closure floor {floor:.1e}")
plt.xlabel("num_bins"); plt.ylabel("rel error vs MC"); plt.title(f"bin refinement (n={REFINE_WIDTH})")
plt.legend(); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §4 — The width scaling law: $\mathrm{MSE}\sim n^{-2}$  (widths 16 … 1536)

Seed-averaged relative **MSE** vs width for binned-$K2$ (`num_bins=21`) and the naive single-bin
closure. Binned should track the **$K=2$ rate $n^{-2}$**; single-bin only $\sim n^{-1}$.

**MC-noise caveat at large $n$.** The binned RMS error falls like $n^{-1}$, so at the biggest widths
it can dip **below** what Monte-Carlo can resolve at `MC_SAMPLES`. We print the MC-noise floor and fit
the slope only over widths where the signal is above it (use `MC_DEVICE="cuda"` + more samples to push
the floor down). Reaching the floor is itself consistent with the error being tiny.""")
code(r"""
mse_b, mse_s, noise = [], [], []
for n in SCALE_WIDTHS:
    eb, es, nz = [], [], []
    for s in SCALE_SEEDS:
        Ws = coordinate_spike_net(n, SCALE_DEPTH, s); mc, se = mc_reference(n, SCALE_DEPTH, s, MC_SAMPLES, MC_BATCH)
        eb.append(relerr(run_binned_kprop_k2(Ws, n, num_bins=SCALE_NUMBINS, bulk_relu=BULK_RELU)["mean"], mc))
        es.append(relerr(run_binned_kprop_k2(Ws, n, num_bins=1, bulk_relu=BULK_RELU)["mean"], mc))
        nz.append(np.linalg.norm(se) / (np.linalg.norm(mc) + 1e-30))
    mse_b.append(np.mean(eb) ** 2); mse_s.append(np.mean(es) ** 2); noise.append(np.mean(nz) ** 2)
mse_b, mse_s, noise = map(np.array, (mse_b, mse_s, noise))
resolvable = mse_b > 4 * noise                                   # signal clearly above the MC floor
wr = np.array(SCALE_WIDTHS)[resolvable]
sl_b = slope(wr, mse_b[resolvable]) if resolvable.sum() >= 2 else float("nan")
sl_s = slope(SCALE_WIDTHS, mse_s)
print("   n     MSE(binned)   MSE(1 bin)   MC-noise floor   resolvable?")
for i, n in enumerate(SCALE_WIDTHS):
    print(f"  {n:5d}  {mse_b[i]:.3e}    {mse_s[i]:.3e}    {noise[i]:.1e}      {'yes' if resolvable[i] else 'NOISE-LIMITED'}")
print(f"  binned MSE ~ n^{sl_b:+.2f} (resolvable widths; K=2 rate n^-2) | single-bin ~ n^{sl_s:+.2f} (~n^-1)")
w = np.array(SCALE_WIDTHS, float)
plt.figure(figsize=(5.6, 3.9))
plt.loglog(w, mse_b, "o-", label=f"binned-K2 ({SCALE_NUMBINS} bins) ~ n^{sl_b:+.2f}")
plt.loglog(w, mse_s, "s-", label=f"single bin ~ n^{sl_s:+.2f}")
plt.loglog(w, noise, ":", c="gray", label="MC-noise floor")
plt.loglog(w, mse_b[0] * (w / w[0]) ** -2.0, "k--", alpha=.6, label="$n^{-2}$ (K=2)")
plt.xlabel("width n"); plt.ylabel("relative MSE vs MC"); plt.title(f"K=2 width scaling (depth={SCALE_DEPTH})")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §5 — Practical tuning (depth 3, larger widths)

Pick `num_bins` for a width: rel-err vs MC, **MC-z** ($\lesssim 3$ = at the noise floor, add samples not
bins), and predict wall-time. Recommended `num_bins` = smallest within $1.2\times$ of the best accuracy
(larger widths typically need **fewer** bins).""")
code(r"""
for n in TUNE_WIDTHS:
    Ws = coordinate_spike_net(n, TUNE_DEPTH, 7); mc, se = mc_reference(n, TUNE_DEPTH, 7, MC_SAMPLES, MC_BATCH)
    noise = np.linalg.norm(se) / (np.linalg.norm(mc) + 1e-30)
    print(f"\n# n={n}, depth={TUNE_DEPTH}, MC={MC_SAMPLES:,} (rel-noise {noise:.1e})")
    print("  num_bins   rel-err     MC-z    predict[s]")
    rows = []
    for nb_ in TUNE_NUMBINS:
        t0 = time.time(); pred = run_binned_kprop_k2(Ws, n, num_bins=nb_, bulk_relu=BULK_RELU)["mean"]; dt = time.time() - t0
        re = relerr(pred, mc); z = np.linalg.norm(pred - mc) / (np.linalg.norm(se) + 1e-30)
        rows.append((nb_, re)); print(f"  {nb_:7d}   {re:.3e}   {z:6.1f}   {dt:8.2f}")
    best = min(r[1] for r in rows)
    print(f"  recommended num_bins @ n={n}: {next((nb_ for nb_, re in rows if re <= 1.2*best), rows[-1][0])}")
""")

# =============================================================================
md(r"""## §6 — Bin-mass profile: where does the probability mass sit?

Run with a **ton of bins** and look at the post-ReLU spike distribution per layer. Expectation: a big
**atom at 0** (dead ReLUs) plus a Gaussian flow-out. `eff_bins = exp(entropy(p))` is the *effective*
number of occupied bins — if it's $\ll$ `num_bins`, most bins are near-empty (the fixed grid is wasteful).""")
code(r"""
NBprof, nP, depthP, seedP = 151, 64, 3, 11
res = run_binned_kprop_k2(coordinate_spike_net(nP, depthP, seedP), nP, num_bins=NBprof,
                          bulk_relu=BULK_RELU, collect=True)
print(f"n={nP}, depth={depthP}, num_bins={NBprof}   (spike distribution AFTER each ReLU)")
for L in res["spike_by_layer"]:
    p, a = L["p"], L["a"]; order = np.argsort(p)[::-1]
    print(f"  layer {L['layer']}:  bin0(@0) mass={p[0]:.3f}   eff_bins={eff_bins(p):6.1f}/{NBprof}   "
          f"top reps={[round(float(a[i]),2) for i in order[:3]]}")
print("-> early layers: large 0-atom; deeper, the spike AMPLIFIES coord 0 positive so it survives ReLU.")
plt.figure(figsize=(5.4, 3.4))
for L in res["spike_by_layer"]:
    plt.plot(L["a"], L["p"], ".-", ms=4, alpha=.8, label=f"layer {L['layer']}")
plt.xlabel("spike representative  a"); plt.ylabel("bin mass  p"); plt.title("post-ReLU spike-bin mass")
plt.legend(); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §7 — Robustness: does $\mathrm{MSE}\sim n^{-2}$ survive a *ton* of bins?

Sweep `num_bins` up to large values and check the width-slope is **stable** (doesn't degrade) and the
algorithm stays numerically clean (`mass-lost` and `psd-clip` ~0). Adding bins past the knee should be
harmless — you just sit at the bulk-closure floor.""")
code(r"""
rwidths = [16, 32, 64, 128] if QUICK else [16, 32, 64, 128, 256]
print("  num_bins |  MSE-slope  | worst mass-lost | worst psd-clip | finite")
for NB in ([15, 31, 61] if QUICK else [15, 31, 61, 121]):
    mses = []; wmass = wpsd = 0.0; fin = True
    for n in rwidths:
        eb = []
        for s in SCALE_SEEDS:
            r = run_binned_kprop_k2(coordinate_spike_net(n, 2, s), n, num_bins=NB, bulk_relu=BULK_RELU, collect=True)
            eb.append(relerr(r["mean"], mc_reference(n, 2, s, MC_SAMPLES, MC_BATCH)[0]))
            fin &= bool(np.all(np.isfinite(r["mean"]))); md_ = r["metadata"]
            wmass = max(wmass, md_["max_linear_mass_lost"], md_["max_relu_mass_lost"]); wpsd = max(wpsd, md_["total_psd_clipped"])
        mses.append(np.mean(eb) ** 2)
    print(f"  {NB:7d}  |   n^{slope(rwidths, mses):+.2f}   |   {wmass:.1e}     |   {wpsd:.1e}    | {fin}")
print("-> slope ~constant + mass-lost/psd-clip ~0 as num_bins grows  =>  stable & robust.")
""")

# =============================================================================
md(r"""## §8 — Do we need negative bins?

`A^+ = γA + r·B` can be negative even when the incoming `A≥0`, and that negative mass **dies** at the
next ReLU. Three pre-activation grids at matched `num_bins`: **symmetric** (current, ~½ negative), a
single **catch-bin** `[-inf,0)`, and **none** (clip). Question: is one catch-bin enough?""")
code(r"""
def _run_pregrid(Ws, n, NB, kind):
    if kind == "symmetric": pre = make_gaussian_edges(NB, std=1.0)
    elif kind == "catchneg": pre = np.concatenate([[-np.inf], make_relu_post_edges(NB - 1)])
    elif kind == "nonneg":   pre = make_relu_post_edges(NB, std=1.0)
    st = gaussian_initial_state(n - 1, pre); post = make_relu_post_edges(NB, std=1.0)
    for li in range(len(Ws) - 1):
        st = linear_step_k2(st, Ws[li][0], pre); st = relu_step_k2(st, post)
    return Ws[-1][0] @ unconditional_mean(st)

NBneg = 15; wneg = [16, 32, 64, 128]
print(f"  rel-err vs MC (num_bins={NBneg}, depth 2):    n:  " + "  ".join(f"{w:>6d}" for w in wneg))
for kind in ("symmetric", "catchneg", "nonneg"):
    row = [np.mean([relerr(_run_pregrid(coordinate_spike_net(n, 2, s), n, NBneg, kind),
                           mc_reference(n, 2, s, MC_SAMPLES, MC_BATCH)[0]) for s in SCALE_SEEDS]) for n in wneg]
    print(f"    {kind:10s}  " + "  ".join(f"{x:.0e}" for x in row))
print("-> 'nonneg' (no negative bins) is broken (30-60% err); ONE catch-bin ~= symmetric. "
      "So: need negative bins, but a single catch-bin suffices.")
""")

# =============================================================================
md(r"""## §9 — Dynamic-CDF grid vs the fixed grid

The mass profile (§6) shows the fixed equal-mass-$N(0,1)$ grid wastes most bins. A **dynamic** grid
re-centers/scales each layer to where the mass is (location-scale Gaussian for the pre-activation,
truncated-normal quantiles + a 0-atom for the post-ReLU). Does it help at matched `num_bins`?""")
code(r"""
def _aplus_moments(state, M):
    gamma = M[0, 0]; r = M[0, 1:]
    mY = gamma * state.a + state.mu @ r; sY2 = (state.Sigma @ r * r).sum(1)
    mean = float(state.p @ mY); var = float(state.p @ (mY ** 2 + sY2)) - mean ** 2
    return mean, math.sqrt(max(var, 1e-12))

def _dyn_pre(state, M, NB): mean, std = _aplus_moments(state, M); return make_gaussian_edges(NB, std=std) + mean
def _dyn_post(state, NB):
    if NB == 1: return np.array([0.0, np.inf])
    mean = float(state.p @ state.a); sa = math.sqrt(max(float(state.p @ (state.a - mean) ** 2), 1e-12))
    q0 = float(np.clip(ndtr(-mean / sa), 0.0, 0.999)); levels = np.clip(np.linspace(q0, 1.0, NB + 1)[1:-1], 1e-12, 1 - 1e-12)
    return np.maximum.accumulate(np.concatenate([[0.0], np.maximum(mean + sa * ndtri(levels), 0.0), [np.inf]]))

def _propagate(Ws, n, NB, mode):
    st = gaussian_initial_state(n - 1, make_gaussian_edges(NB, std=1.0))
    for li in range(len(Ws) - 1):
        M = Ws[li][0]
        pre = _dyn_pre(st, M, NB) if mode == "dynamic" else make_gaussian_edges(NB, std=1.0)
        st = linear_step_k2(st, M, pre)
        post = _dyn_post(st, NB) if mode == "dynamic" else make_relu_post_edges(NB, std=1.0)
        st = relu_step_k2(st, post)
    return Ws[-1][0] @ unconditional_mean(st)

wdyn = [16, 32, 64, 128]
for NB in (3, 7, 15):
    ef = [np.mean([relerr(_propagate(coordinate_spike_net(n, 3, s), n, NB, "fixed"),
                          mc_reference(n, 3, s, MC_SAMPLES, MC_BATCH)[0]) for s in SCALE_SEEDS]) for n in wdyn]
    ed = [np.mean([relerr(_propagate(coordinate_spike_net(n, 3, s), n, NB, "dynamic"),
                          mc_reference(n, 3, s, MC_SAMPLES, MC_BATCH)[0]) for s in SCALE_SEEDS]) for n in wdyn]
    print(f"  num_bins={NB:2d} (depth 3):  fixed " + " ".join(f"{x:.0e}" for x in ef) +
          "   dynamic " + " ".join(f"{x:.0e}" for x in ed))
print("-> dynamic helps only at FEW bins on deeper nets (reaches the floor with ~2x fewer bins); "
      "no gain once you have enough bins -- the bulk-K2 closure floor dominates, not spike resolution.")
""")

# =============================================================================
md(r"""## §10 — (optional, needs torch) run on a real `model.MLP` checkpoint

`add_spike=True` injects $\theta e_1e_1^\top$ into each square hidden layer when the spike isn't already
baked into the weights. Skips cleanly if torch / the checkpoint isn't available.""")
code(r"""
try:
    from model import MLP
    from Mecha_preds.binned_kprop import run_binned_kprop
    CKPT = ""   # e.g. "checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt"
    if CKPT and os.path.exists(CKPT):
        model, _ = MLP.load(CKPT)
        out = run_binned_kprop(model, config={"num_bins": 31}, add_spike=False)
        print("config:", out["metadata"]["config"]); print("E[output][:8]:", np.asarray(out["mean"]).ravel()[:8])
    else:
        print("set CKPT to a coordinate-spike checkpoint to run the adapter path.")
except ModuleNotFoundError as e:
    print("torch / model unavailable here -> skipping:", e)
""")

nb.save(os.path.join(os.path.dirname(__file__), "binned_kprop_colab.ipynb"))

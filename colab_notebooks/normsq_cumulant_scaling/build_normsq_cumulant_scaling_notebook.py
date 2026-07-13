"""Generates normsq_cumulant_scaling_colab.ipynb (valid nbformat-4 JSON).

TESTS the extensivity assumption: at every hidden post-ReLU activation X in R^n,
are the cumulants kappa_r(||X||_2^2), r = 2,3,4, O(n)?

Why it matters (the mixing consequence): the next layer mixes X with fresh
transverse weights g ~ N(0, I/n). Conditional on X, z = g.X is a Gaussian scale
mixture, z | X ~ N(0, ||X||^2 / n), so EXACTLY

    kappa_{2r}(z) = (2r-1)!! * kappa_r(||X||^2) / n^r .

If kappa_r(||X||^2) ~ n (extensive, as for n weakly-dependent coordinates), the
mixed-unit cumulants decay as n^{1-r} -- mixing Gaussianizes at the CLT rate.
If instead the energy has an O(1) collective mode (||X||^2 ~ n * F(h)), then
kappa_r(||X||^2) ~ n^r and kappa_{2r}(z) ~ n^0 -- non-Gaussianity SURVIVES mixing.

Models (random, NO TRAINING; hidden layers only):
    e1    M = W' + e1 e1^T           (O(1) coordinate spike -- expect extensive)
    ones  M = W' + (1/sqrt n) 1 1^T  (sqrt-n flat spike / shifted-mean "add" --
                                      the collective S-mode candidate)
    none  M = W'                     (unshifted control)

We stream x ~ N(0, I_n), collect ||X||^2 per sample at EVERY hidden layer via the
validated analysis.Tools.cumulants_sv streamer (V = ||X||^2/sqrt(n)), estimate
kappa_{2,3,4} with delete-one-block jackknife SDs (2-sigma resolution gate), and
overlay the iid/diagonal reference sum_i kappa_r(X_i^2) built from per-coordinate
moments -- so any deviation from slope 1 is attributed to inter-coordinate
correlation. Log-log fit of |kappa_r| vs n per (direction, depth, layer).

MC budget: SD(kappa4_hat) ~ sqrt(96/N) * kappa2^2 while kappa4/kappa2^2 ~ 1/n
under the extensive H, so N must grow ~ n^2: full mode uses N(w)=clip(8 w^2, 1M, 8M).

REPO POLICIES: notebook owns knobs + CKPT_DIR; stats cached per run (re-run
recycles, nothing recomputed); float32 forward on E.DEVICE, float64 accumulation.
Depths 3 AND 4. No kprop import -> any Python works.

Run:  python "colab_notebooks/normsq_cumulant_scaling/build_normsq_cumulant_scaling_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# Width-scaling of $\kappa_r(\|X\|_2^2)$ — is the post-activation energy **extensive**?

**Assumption under test.** At every hidden post-ReLU activation $X\in\mathbb R^n$, the higher
cumulants of the squared norm are $O(n)$:
$$\kappa_r\!\big(\|X\|_2^2\big) \sim n, \qquad r=2,3,4 .$$
That is what $n$ weakly-dependent coordinates would give: $\kappa_r(\sum_i X_i^2)=\sum_i\kappa_r(X_i^2)\sim n$
under independence.

**Why it matters — the mixing consequence.** The next layer mixes $X$ with fresh transverse weights
$g\sim\mathcal N(0,\tfrac1n I)$. Conditional on $X$, $z=g\!\cdot\!X\sim\mathcal N(0,\|X\|^2/n)$ is a
Gaussian **scale mixture**, so exactly
$$\kappa_{2r}(z) \;=\; (2r-1)!!\;\frac{\kappa_r(\|X\|^2)}{n^r}.$$
- extensive ($\kappa_r\sim n$) $\Rightarrow$ $\kappa_{2r}(z)\sim n^{1-r}$ — mixing Gaussianizes at the CLT rate;
- a **collective mode** ($\|X\|^2\approx n\,F(h)$ driven by one shared fluctuation $h$)
  $\Rightarrow$ **self-similar** cumulants $\kappa_r\sim\kappa_2^{\,r/2}$, i.e. slope$_r = \tfrac r2\,$slope$_2$
  ($= n^r$ when $h=O(1)$; larger when the mode's amplitude itself grows with $n$) and
  $\kappa_{2r}(z)$ does **not** decay as $n^{1-r}$ — non-Gaussianity survives mixing.

**Models** (random, **no training**; spike on hidden layers only, $W'_{ij}\sim\mathcal N(0,\tfrac1{\text{fan\_in}})$):

| tag | $M$ | spike eigenvalue | expectation |
|---|---|---|---|
| `e1` | $W' + e_1e_1^\top$ | $O(1)$, localized | extensive (slope 1) |
| `ones` | $W' + \tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top$ | $\sqrt n$, flat (shifted-mean "add") | collective, self-similar (slope$_r=\tfrac r2$slope$_2\gg1$; the coherent amplitude compounds $\sim\!\sqrt n$ per layer) |
| `none` | $W'$ | — | control (slope 1) |

**Measurement.** Stream $x\sim\mathcal N(0,I_n)$; at every hidden layer collect per-sample
$\|X\|^2=\sqrt n\,V$ from the **validated** `analysis.Tools.cumulants_sv` streamer; estimate
$\kappa_{2,3,4}$ with delete-one-block **jackknife** SDs and a $2\sigma$ resolution gate; overlay the
**iid/diagonal reference** $\sum_i\kappa_r(X_i^2)$ (from per-coordinate moments $E[X_i^{2k}]$, $k\le4$)
so a deviation from slope 1 is attributed to inter-coordinate correlation. Then log-log fit
$\log|\kappa_r|$ vs $\log n$ per (direction, depth, layer).

**MC budget.** $\mathrm{SD}(\hat\kappa_4)\approx\sqrt{96/N}\,\kappa_2^2$ while
$\kappa_4/\kappa_2^2\sim1/n$ under the extensive H $\Rightarrow$ $N\propto n^2$;
full mode uses $N(w)=\mathrm{clip}(8w^2,\,10^6,\,8\!\times\!10^6)$.

No kprop import (any Python); needs torch. **QUICK** (auto on CPU): widths 32–128, depth 3,
1 seed, 200k samples (~2–5 min CPU). **Full**: widths 64–1024, depths 3–4, 2 seeds
(~10 min A100 / 30–60 min T4; trim `WIDTHS` to ≤512 to cut ~85% of the cost).""")

code(r"""!pip install -q jaxtyping""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 — Config, models & the measurement (knobs live here, not in `experiments.py`)

Three ensembles (`e1`, `ones`, `none`), `SIGN=+1` per the spec ($+e_1e_1^\top$, $+\tfrac1{\sqrt n}\mathbf{11}^\top$;
note `SIGN=-1` on `ones` is the ReLU-death "sub" regime — a different study). Models are deterministic
from generator seeds, so by default only the **stats** are checkpointed (a w=1024 float64 model is
~40 MB; set `SAVE_MODELS=True` to keep them).""")
code(r"""
import math, os, time, copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP
from analysis.Tools import cumulants_sv as CSV

QUICK  = E.QUICK
DEVICE = E.DEVICE
torch.set_num_threads(max(torch.get_num_threads(), 2))

DIRECTIONS = ["e1", "ones", "none"]      # e1 e1^T | (1/sqrt n) 1 1^T | unshifted control
SIGN       = +1.0                        # '+' spikes per spec (-1 on ones = ReLU-death regime)
DEPTHS     = [3] if QUICK else [3, 4]
WIDTHS     = [32, 64, 128] if QUICK else [64, 128, 256, 512, 1024]
SEEDS      = [1, 2] if QUICK else [1, 2, 3]   # deep-layer kappa_r have LARGE quenched (seed) scatter
                                              # at small n -- fits use seed-means; add seeds if noisy
ACTIVATION = "relu"

# MC budget: resolving kappa4 needs N ~ n^2 (SD ~ sqrt(96/N) kappa2^2, kappa4/kappa2^2 ~ 1/n under H)
MC_FLOOR, MC_COEF, MC_CAP = (200_000, 0, 200_000) if QUICK else (1_000_000, 8, 8_000_000)
def mc_for(w): return int(min(MC_CAP, max(MC_FLOOR, MC_COEF * w * w)))
MC_BATCH = 65_536 if DEVICE.type == "cuda" else 8_192
N_BLOCKS, Z_GATE = 40, 2.0               # jackknife blocks; resolved iff |k| > Z_GATE * sd

CKPT_DIR    = "checkpoints/normsq_cumulant_scaling"
RECYCLE     = True
SAVE_MODELS = False                      # deterministic rebuild is O(n^2); stats cache is what matters
os.makedirs(CKPT_DIR, exist_ok=True)
print("DEVICE:", DEVICE, "| QUICK:", QUICK, "| batch:", MC_BATCH)
print("dirs:", DIRECTIONS, "| sign:", SIGN, "| depths:", DEPTHS, "| widths:", WIDTHS,
      "| seeds:", SEEDS, "| MC:", {w: mc_for(w) for w in WIDTHS})
""")

code(r"""
# ---- the random spiked MLP (float64 master; NO TRAINING; spike on hidden layers only) ----
# Same builder/seed layout as the spike_kprop family (e1 offset 0), so e1 models are
# bit-identical to that family when SIGN=-1 and can be recycled from its checkpoints.
def spike_matrix(direction, n):
    if direction == "e1":
        P = torch.zeros(n, n, dtype=torch.float64); P[0, 0] = SIGN; return P     # SIGN * e1 e1^T
    if direction == "ones":
        return torch.full((n, n), SIGN / math.sqrt(n), dtype=torch.float64)      # SIGN/sqrt(n) * 1 1^T
    if direction == "none":
        return torch.zeros(n, n, dtype=torch.float64)
    raise ValueError(direction)

def spiked_mlp(width, seed, depth, direction):
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, activation=ACTIVATION).double().eval()
    g = torch.Generator().manual_seed(1_000_000 * depth + 10_000 * seed + 7 * width
                                       + {"e1": 0, "ones": 3, "none": 6}[direction])
    P = spike_matrix(direction, width)
    with torch.no_grad():
        for li, layer in enumerate(list(m.hidden_layers) + [m.readout]):
            out_f, in_f = layer.weight.shape
            W = torch.randn(out_f, in_f, generator=g, dtype=torch.float64) / math.sqrt(in_f)
            if li < len(m.hidden_layers):
                W = W + P
            layer.weight.copy_(W)
    return m

def get_model(direction, w, seed, depth):
    name = E.run_name(f"normsq-{direction}-{'p' if SIGN > 0 else 'm'}", depth=depth, width=w, seed=seed)
    p = E.ckpt_path(CKPT_DIR, name)
    if RECYCLE and os.path.exists(p):
        return MLP.load(p, map_location="cpu")[0].double().eval()
    if RECYCLE and direction == "e1" and SIGN == -1.0:           # bit-identical to spike_kprop family
        for d in ("checkpoints/spike_kprop", "checkpoints/e1_cumulant_scaling"):
            p2 = E.ckpt_path(d, E.run_name("spike-e1", depth=depth, width=w, seed=seed))
            if os.path.exists(p2):
                return MLP.load(p2, map_location="cpu")[0].double().eval()
    m = spiked_mlp(w, seed, depth, direction)
    if SAVE_MODELS:
        m.save(p, extra={"family": "normsq_cumulant_scaling", "direction": direction,
                         "sign": SIGN, "depth": depth, "width": w, "seed": seed})
    return m
""")

code(r"""
# ---- per-run measurement: kappa_{2,3,4}(||X||^2) at EVERY hidden post-activation ----
# Reuses the VALIDATED cumulants_sv streamer: per-sample V = ||X||^2/sqrt(n)  ->  ||X||^2 = sqrt(n) V,
# plus per-coordinate moments E[X_i^k], k=1..8 (Y=X_i^2 needs E[Y^4]=E[X^8]) for the iid reference.
def diag_iid_reference(coord_moments):
    "kappa_r^iid(||X||^2) = sum_i kappa_r(X_i^2) from per-coordinate moments (independence null)."
    m = np.asarray(coord_moments, np.float64)          # (8, n): row k-1 = E[X_i^k]
    m1, m2, m3, m4 = m[1], m[3], m[5], m[7]            # E[Y^k] = E[X^{2k}] for Y = X^2
    c2  = m2 - m1**2
    c3  = m3 - 3*m1*m2 + 2*m1**3
    mu4 = m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4
    return {2: float(c2.sum()), 3: float(c3.sum()), 4: float((mu4 - 3*c2**2).sum())}

@torch.no_grad()
def measure_normsq_cumulants(m, w, data_seed):
    md_ = copy.deepcopy(m).to(device=DEVICE, dtype=torch.float32).eval()   # repo float32 runtime
    depth = m.cfg.depth
    res = CSV.stream_collective_coordinates(
        md_, input_dim=w, num_samples=mc_for(w), layers=list(range(depth)),
        whichs=("post",), n_rand=1, batch=MC_BATCH, device=DEVICE, data_seed=data_seed)["post"]
    out, sqn = {}, math.sqrt(w)
    for l in range(depth):
        Vraw = sqn * res[l]["V"]                       # per-sample ||X||_2^2 at layer l
        Vc = Vraw - Vraw.mean()                        # center first: kappa_{r>=2} shift-invariant,
        point, sd = CSV.jackknife_cumulants_1d(Vc, B=N_BLOCKS)   # kills the (n/2)^4 cancellation
        diag = diag_iid_reference(res[l]["coord_moments"])
        out[l] = dict(meanE=float(Vraw.mean()))
        for r in (2, 3, 4):
            out[l][f"k{r}"], out[l][f"sd{r}"], out[l][f"diag{r}"] = point[r], sd[r], diag[r]
            out[l][f"res{r}"] = bool(abs(point[r]) > Z_GATE * sd[r])
    del md_, res
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out
""")

code(r"""
# ---- stats cache (the recycling artifact: re-runs recompute NOTHING) ----
RESULTS_PATH = os.path.join(CKPT_DIR, f"normsq_stats_sign{SIGN:+g}.pt")
_results = torch.load(RESULTS_PATH) if (RECYCLE and os.path.exists(RESULTS_PATH)) else {}
def cache_get(k): return _results.get(k) if RECYCLE else None
def cache_put(k, v): _results[k] = v; torch.save(_results, RESULTS_PATH)
print(f"stats cache {os.path.basename(RESULTS_PATH)}: {len(_results)} runs "
      f"({'recycling' if _results else 'empty -> will compute'})")
""")

# =============================================================================
md(r"""## §2 — Sweep: $\kappa_r(\|X\|^2)$ over (direction, depth, width, seed)

One streaming pass per run collects all layers at once. Cached per run, so an interrupted sweep
resumes where it stopped. `?` marks a cumulant that failed the $2\sigma$ jackknife gate.""")
code(r"""
rows, t0 = [], time.time()
for direction in DIRECTIONS:
    for depth in DEPTHS:
        for w in WIDTHS:
            for seed in SEEDS:
                key = f"{direction}|d{depth}|w{w}|s{seed}|mc{mc_for(w)}"
                r = cache_get(key); src = "recycled"
                if r is None:
                    src = "computed"; t1 = time.time()
                    per_layer = measure_normsq_cumulants(
                        get_model(direction, w, seed, depth), w,
                        data_seed=7919*depth + 104729*seed + w + 15485863*DIRECTIONS.index(direction))
                    r = dict(direction=direction, depth=depth, w=w, seed=seed,
                             layers=per_layer, mc=mc_for(w), secs=time.time() - t1)
                    cache_put(key, r)
                rows.append(r)
                L = r["layers"][depth - 1]
                print(f"{direction:>4} d{depth} w={w:>4} s{seed} [{src:>8}] mc={r['mc']:.0e}"
                      f" ({r['secs']:5.1f}s) | last layer: "
                      f"k2={L['k2']:.3e}{'' if L['res2'] else '?'} "
                      f"k3={L['k3']:.3e}{'' if L['res3'] else '?'} "
                      f"k4={L['k4']:.3e}{'' if L['res4'] else '?'}", flush=True)
print(f"\nsweep done in {(time.time()-t0)/60:.1f} min ({len(rows)} runs; recycled ones instant)")
""")

# =============================================================================
md(r"""## §3 — Log-log fits: slope of $\log|\kappa_r|$ vs $\log n$, per (direction, depth, layer)

OLS on the seed-mean, **only** widths where every seed passes the $2\sigma$ gate. Verdicts:
slope $\approx1$ = extensive (assumption **holds**); slope$_r\approx\tfrac r2$slope$_2$ with slope$_2\gg1$
= collective self-similar mode (assumption **fails**); in between = partial coherence.

Deep-layer $\kappa_r$ carry **large quenched (seed-to-seed) scatter at small $n$** — that's why the
fit uses seed-means over several seeds; add seeds if the SEs are big.""")
code(r"""
def fit_slope(direction, depth, layer, r):
    "OLS slope +/- SE of log|k_r| vs log n (seed-mean; widths where ALL seeds resolve k_r)."
    xs, ys = [], []
    for w in WIDTHS:
        rs = [q for q in rows if q["direction"] == direction and q["depth"] == depth and q["w"] == w]
        if not rs:
            continue
        v = float(np.mean([q["layers"][layer][f"k{r}"] for q in rs]))
        if all(q["layers"][layer][f"res{r}"] for q in rs) and np.isfinite(v) and v != 0:
            xs.append(math.log(w)); ys.append(math.log(abs(v)))
    if len(xs) < 3:
        return dict(slope=float("nan"), se=float("nan"), r2=float("nan"), npts=len(xs))
    x, y = np.array(xs), np.array(ys); A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None); resd = y - A @ coef; ss = float((resd**2).sum())
    se = float(np.sqrt(ss / max(len(x) - 2, 1) / ((x - x.mean())**2).sum()))
    r2 = 1.0 - ss / (float(((y - y.mean())**2).sum()) + 1e-30)
    return dict(slope=float(coef[0]), se=se, r2=r2, npts=len(x))

# verdict: compare slope p of k_r against H1 (extensive: p = 1) and H2 (collective self-similar:
# k_r ~ k2^{r/2}, i.e. p = r*p2/2 -- one shared mode drives ||X||^2, as for the 'ones' spike
# whose coherent amplitude itself grows with n).
def verdict(p, r, p2):
    if not np.isfinite(p):
        return "n/a (unresolved)"
    if abs(p - 1) <= 0.15:
        return "O(n) extensive -> H HOLDS"
    if np.isfinite(p2) and p2 > 1.5 and abs(p - r * p2 / 2) <= 0.35 * r:
        return f"collective self-similar (pred r*p2/2 = {r*p2/2:.1f}) -> H FAILS"
    return f"n^{p:.2f} ({'sub' if p < 1 else 'super'}-extensive)"

for direction in DIRECTIONS:
    for depth in DEPTHS:
        print(f"\n=== {direction:>4}  depth {depth} ===   (H: slope 1;  collective: slope_r = r*p2/2)")
        print(f"{'layer':>6} | " + " | ".join(f"k{r}: slope+/-se (R^2,pts)".rjust(26) for r in (2, 3, 4)))
        for l in range(depth):
            cells = []
            for r in (2, 3, 4):
                f = fit_slope(direction, depth, l, r)
                cells.append(f"{f['slope']:5.2f}+/-{f['se']:.2f} ({f['r2']:.2f},{f['npts']})")
            print(f"{l:>6} | " + " | ".join(c.rjust(26) for c in cells))
        last = depth - 1
        p2 = fit_slope(direction, depth, last, 2)["slope"]
        print("  last-layer verdicts: " + ";  ".join(
            f"k{r}: {verdict(fit_slope(direction, depth, last, r)['slope'], r, p2)}" for r in (2, 3, 4)))
""")

# =============================================================================
md(r"""## §4 — Plots: $|\kappa_r(\|X\|^2)|$ vs $n$ (log-log), all layers

One figure per depth; rows $r=2,3,4$, columns = ensembles. Per layer: solid = measured (error bars =
jackknife MC error $\oplus$ seed scatter, `x` = failed the $2\sigma$ gate, excluded from fits),
dotted = the **iid/diagonal reference** $\sum_i\kappa_r(X_i^2)$ (the gap to it *is* the correlation
contribution). Black dashed = slope 1 (extensive H); red dashed = the collective self-similar slope
$\tfrac r2\,\hat p_2$ (drawn only where slope$_2$ came out $\gg1$).""")
code(r"""
def seed_mean(direction, depth, layer, key):
    out = []
    for w in WIDTHS:
        vals = [q["layers"][layer][key] for q in rows
                if q["direction"] == direction and q["depth"] == depth and q["w"] == w]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return np.array(out)

def seed_err(direction, depth, layer, r):
    "SEM of the seed-mean of k_r: jackknife MC error (+) quenched seed-to-seed scatter."
    out = []
    for w in WIDTHS:
        ks  = [q["layers"][layer][f"k{r}"] for q in rows
               if q["direction"] == direction and q["depth"] == depth and q["w"] == w]
        sds = [q["layers"][layer][f"sd{r}"] for q in rows
               if q["direction"] == direction and q["depth"] == depth and q["w"] == w]
        if not ks:
            out.append(float("nan")); continue
        S = len(ks)
        out.append(math.sqrt(float(np.mean(np.square(sds))) / S + float(np.var(ks)) / S))
    return np.array(out)

LCOL = plt.cm.viridis(np.linspace(0.1, 0.8, max(DEPTHS)))
for depth in DEPTHS:
    fig, axes = plt.subplots(3, len(DIRECTIONS), figsize=(4.6 * len(DIRECTIONS), 11.5), squeeze=False)
    for ci, direction in enumerate(DIRECTIONS):
        for ri, r in enumerate((2, 3, 4)):
            ax, xs = axes[ri][ci], np.array(WIDTHS, float)
            for l in range(depth):
                y   = np.abs(seed_mean(direction, depth, l, f"k{r}"))
                err = seed_err(direction, depth, l, r)
                ok  = seed_mean(direction, depth, l, f"res{r}") == 1.0       # all seeds resolved
                ax.errorbar(xs[ok], y[ok], yerr=err[ok], fmt="o-", color=LCOL[l], ms=4, lw=1.4,
                            label=f"layer {l}")
                if (~ok).any():
                    ax.plot(xs[~ok], y[~ok], "x", color=LCOL[l], alpha=.45)
                ax.plot(xs, np.abs(seed_mean(direction, depth, l, f"diag{r}")), ":",
                        color=LCOL[l], alpha=.55)
            yl = np.abs(seed_mean(direction, depth, depth - 1, f"k{r}"))
            a = yl[np.isfinite(yl) & (yl > 0)]
            if a.size:
                ax.plot(xs, a[-1] * (xs / xs[-1]) ** 1.0, "k--", lw=1, alpha=.8, label="slope 1 (H)")
                p2 = fit_slope(direction, depth, depth - 1, 2)["slope"]      # collective self-similar
                if np.isfinite(p2) and p2 > 1.5:                             # guide: k_r ~ k2^{r/2}
                    ax.plot(xs, a[-1] * (xs / xs[-1]) ** (r * p2 / 2), "r--", lw=1, alpha=.6,
                            label=f"slope r*p2/2 = {r*p2/2:.1f} (collective)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"{direction}: |kappa_{r}(||X||^2)|  (depth {depth})", fontsize=10)
            ax.set_xlabel("width n"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=6)
    plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §5 — The mixing consequence, read off the same data

Exact for a Gaussian scale mixture: $\kappa_{2r}(z)=(2r-1)!!\,\kappa_r(\|X\|^2)/n^r$ for
$z=g\!\cdot\!X$, $g\sim\mathcal N(0,\tfrac1nI)$ fresh. Extensive H $\Rightarrow$ slope $1-r$
(grey dashed); a collective mode $\Rightarrow$ slope $0$ (non-Gaussianity survives the mixing).
Last hidden layer shown.""")
code(r"""
DFACT = {2: 3.0, 3: 15.0, 4: 105.0}                     # (2r-1)!!
for depth in DEPTHS:
    fig, axes = plt.subplots(1, len(DIRECTIONS), figsize=(4.6 * len(DIRECTIONS), 4.0), squeeze=False)
    for ci, direction in enumerate(DIRECTIONS):
        ax, xs = axes[0][ci], np.array(WIDTHS, float)
        for r in (2, 3, 4):
            y = np.abs(seed_mean(direction, depth, depth - 1, f"k{r}")) * DFACT[r] / xs ** r
            ax.loglog(xs, y, "o-", label=f"|kappa_{2*r}(z)|  (from k{r})")
            a = y[np.isfinite(y) & (y > 0)]
            if a.size:
                ax.loglog(xs, a[-1] * (xs / xs[-1]) ** (1.0 - r), "--", color="0.55", lw=1)
        ax.set_title(f"{direction}: implied mixed-unit cumulants (last hidden, d{depth})", fontsize=9)
        ax.set_xlabel("width n"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.show()
print("grey dashed = the extensive-H prediction n^(1-r); flat lines = collective mode (H fails).")
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
    z = shutil.make_archive("/content/normsq_cumulant_scaling_ckpts", "zip", CKPT_DIR)
    print("zipped ->", z, "-- downloading..."); files.download(z)
""")

# =============================================================================
md(r"""## §7 — Summary

- **What ran:** random depth-3/4 ReLU MLPs, hidden matrices shifted by $+e_1e_1^\top$ (O(1)
  coordinate spike), $+\tfrac1{\sqrt n}\mathbf{11}^\top$ ($\sqrt n$ flat spike), or nothing
  (control); no training. Per hidden post-activation we MC-estimated $\kappa_{2,3,4}(\|X\|_2^2)$
  and fitted the width-scaling exponent.
- **The test:** extensivity $\kappa_r\sim n$ (slope 1) — which via the exact scale-mixture identity
  $\kappa_{2r}(z)=(2r-1)!!\,\kappa_r(\|X\|^2)/n^r$ implies the transversely-mixed cumulants decay as
  $n^{1-r}$. The competing outcome is a collective mode: self-similar slope$_r=\tfrac r2$slope$_2$
  ($\ge n^r$), mixed cumulants that don't decay.
- **Reading the plots:** dotted iid/diagonal reference = what independence would give; the measured/
  dotted gap is pure inter-coordinate correlation. `x` points failed the $2\sigma$ jackknife gate and
  are excluded from fits.
- **Estimator provenance:** per-sample streaming + jackknife from the validated
  `analysis.Tools.cumulants_sv`; $\|X\|^2=\sqrt n\,V$; samples centered before the k-statistics to
  avoid the $(n/2)^4$ cancellation; $N(w)\propto n^2$ so the $\kappa_4$ gate doesn't empty out at
  large width.
- **Recycling:** every (direction, depth, width, seed) run is cached in
  `checkpoints/normsq_cumulant_scaling`; re-runs and interrupted sweeps recompute nothing.""")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normsq_cumulant_scaling_colab.ipynb")
nb.save(out)

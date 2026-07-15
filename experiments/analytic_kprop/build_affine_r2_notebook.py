"""Generates affine_r2_colab.ipynb (valid nbformat-4 JSON).

HOW ACCURATE IS THE AFFINE (linear-in-spike) HYPOTHESIS, and how does it scale
with width? At every layer the analytic predictor makes exactly one of two
projections (depending on ``fit``):

  * PRE-activation:  m̂(y), Ŝ(y)  -> affine in the spike pre-activation y
  * POST-activation: r(a), R(a)   -> affine in the post-ReLU spike a

This notebook measures BOTH, separately, at every layer: propagate the exact
paper path (fit="pre" carries exact nonlinear node moments), fit the weighted
linear model from the bins/nodes to the mean and covariance, and report the
weighted R^2 (plus the variation scale that says whether there is anything to
fit). NO Monte-Carlo is needed -- the fit quality is internal to the surrogate.

Sections:
  §1  config + net builder + R^2 helpers + cache
  §2  the per-layer measurement (inline; uses core internals)
  §3  sanity: one net, R^2 table per layer (pre/post x mean/cov, atom in/out)
  §4  SCALING: 1 - R^2 vs width, per layer, 8 seeds, four panels
  §5  reading the result

Torch-free; a few minutes for the full sweep (widths to 1024, 8 seeds). Run:
    python "experiments/analytic_kprop/build_affine_r2_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# How linear is the bulk-given-spike law? — per-layer $R^2$ of the affine hypothesis

Both analytic variants compress the bulk conditional to an **affine family** in the spike
coordinate; the only difference is *where* the projection happens:

* `fit="pre"`: $\hat m(y),\hat S(y)$ — the exactly-reconditioned **pre-activation** cell moments;
* `fit="post"`: $r(a), R(a)$ — the exact **post-ReLU** node moments (nonlinear in $a$ by
  construction: $\Phi$ factors from the ReLU).

Here we measure both directly: propagate the exact paper path, and at every layer fit the
**weighted linear model from the bins to the mean and covariance**, reporting weighted
$R^2 = 1 - \mathrm{SS}_{\rm res}/\mathrm{SS}_{\rm tot}$ (pooled over coordinates; Frobenius for
covariances) and its **width scaling**. If the affine-closure error is a finite-width effect,
$1 - R^2$ should fall with $n$ — that would explain the observed MSE $\sim n^{-2}$ law surviving
the projection. We also report `var_scale` $= \mathrm{SS}_{\rm tot}/\|\bar\cdot\|^2$: when it is
$\approx 0$ the function is constant and $R^2$ is meaningless (e.g. layer-0 covariance, where the
only variation across cells is quantization-induced).

No Monte-Carlo anywhere — this is a property of the surrogate itself, so the sweep is cheap.""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 — Config, net builder, weighted-$R^2$ helpers, cache

`POST_WITH_ATOM` adds a second post-activation measurement that *includes* the zero atom as a
data point at $a=0$ — directly testing the `atom="fit"` hypothesis (is the merge of all negative
cells on the same line?).""")
code(r"""
import os, json, time
import numpy as np
import matplotlib.pyplot as plt

# ---------------- knobs (edit here) ----------------
QUICK     = True                 # False -> full sweep (widths to 1024, 8 seeds)
WIDTHS    = [16, 32, 64, 128] if QUICK else [16, 32, 64, 128, 256, 512, 768, 1024]
SEEDS     = [10, 11, 12] if QUICK else list(range(10, 18))     # ~8 seeds
DEPTH     = 3
NUM_NODES = 40
GRID      = "w2"
THETA     = 1.0
WORKERS   = "auto"
POST_WITH_ATOM = True            # also fit post WITH the zero atom included

CKPT_DIR = "checkpoints/analytic_kprop"; os.makedirs(os.path.join(CKPT_DIR, "r2"), exist_ok=True)
print(f"widths={WIDTHS} seeds={SEEDS} depth={DEPTH} num_nodes={NUM_NODES} grid={GRID}")

def coordinate_spike_net(n, depth, seed, *, theta=THETA):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n)); P[0, 0] = theta
    Ws = [(rng.standard_normal((n, n)) / np.sqrt(n) + P, None) for _ in range(depth)]
    Ws.append((rng.standard_normal((8, n)) / np.sqrt(n), None))
    return Ws

def wls_affine_r2(w, x, data):
    '''Weighted affine fit data_j ~ b0 + b1 x_j and pooled R^2.
    data: (J, ...) -- vectors or matrices, Frobenius pooling. Returns
    (r2, var_scale): var_scale in [0, 1] = SS_tot / (weighted 2nd moment) -- the
    fraction of the target's magnitude that actually VARIES across bins (is
    there anything to fit?); r2 = nan when SS_tot ~ 0.'''
    w = np.asarray(w, float); w = w / w.sum()
    x = np.asarray(x, float)
    D = np.asarray(data, float).reshape(len(w), -1)
    xbar = float(w @ x); vx = float(w @ (x - xbar) ** 2)
    Dbar = w @ D
    ss_tot = float(w @ ((D - Dbar) ** 2).sum(axis=1))
    scale = float(w @ (D ** 2).sum(axis=1)) + 1e-300
    if vx <= 1e-14 or ss_tot <= 1e-14 * scale:
        return float("nan"), ss_tot / scale
    b1 = (w * (x - xbar)) @ D / vx
    b0 = Dbar - b1 * xbar
    resid = D - b0[None, :] - x[:, None] * b1[None, :]
    ss_res = float(w @ (resid ** 2).sum(axis=1))
    return 1.0 - ss_res / ss_tot, ss_tot / scale

print("helpers ready")
""")

# =============================================================================
md(r"""## §2 — The per-layer measurement

Propagate with the exact paper path (`fit="pre"` keeps exact nonlinear post-ReLU node moments,
so nothing we measure is contaminated by an earlier fit of the SAME kind we are testing — the
pre-activation fit *is* used to cross layers, as in the algorithm).

Per layer:
* **pre**: recompute the exact cell-merged $(\hat m_j, \hat S_j)$ on the same retained grid the
  layer used (`percell_bulk_moments`), fit affine in $y_j$ with weights $w_j$;
* **post**: take the advanced state's exact positive nodes $(p_k, a_k, m_k, S_k)$, fit affine in
  $a_k$ — with and without the zero atom as an $a{=}0$ data point.""")
code(r"""
from Mecha_preds.analytic_kprop import (gaussian_input_state, analytic_layer_update,
                                        percell_bulk_moments)
from Mecha_preds.analytic_kprop.core import _layer_block, _component_params, _pair_stats

def affine_r2_by_layer(Ws, n, *, num_nodes=NUM_NODES, grid=GRID, workers=WORKERS):
    d = n - 1
    state = gaussian_input_state(d)
    rows = []
    for li in range(len(Ws) - 1):
        M = np.asarray(Ws[li][0], dtype=np.float64)
        gamma, r, u, V, beta, eta_b = _layer_block(M, None, d)
        mY, sY2, mC, g = _component_params(state, gamma, r, u, V, beta, eta_b)
        new_state, aff = analytic_layer_update(state, M, None, num_nodes=num_nodes,
                                               grid=grid, workers=workers)
        # ---- PRE: exact cell moments on the SAME retained grid the layer used ----
        Q, ym, dl, vv, stoch = _pair_stats(state.p, mY, sY2, aff.edges, min_prob=1e-15)
        wr = state.p @ Q; keep = wr > 1e-15
        mh, Sh = percell_bulk_moments(state.p, Q[:, keep], dl[:, keep], vv[:, keep],
                                      sY2, stoch, wr[keep],
                                      state.Sigma, state.t2, mC, g, u, V)
        r2_pm, vs_pm = wls_affine_r2(aff.w, aff.y, mh)
        r2_pc, vs_pc = wls_affine_r2(aff.w, aff.y, Sh)
        # ---- POST: exact post-ReLU node moments of the advanced state ----
        pos = new_state.a > 0
        r2_qm, vs_qm = wls_affine_r2(new_state.p[pos], new_state.a[pos], new_state.mu[pos])
        r2_qc, vs_qc = wls_affine_r2(new_state.p[pos], new_state.a[pos], new_state.Sigma[pos])
        row = dict(layer=li,
                   pre_mean=r2_pm, pre_cov=r2_pc, post_mean=r2_qm, post_cov=r2_qc,
                   vs_pre_mean=vs_pm, vs_pre_cov=vs_pc, vs_post_mean=vs_qm, vs_post_cov=vs_qc)
        if POST_WITH_ATOM and (~pos).any():
            r2_am, _ = wls_affine_r2(new_state.p, np.maximum(new_state.a, 0.0), new_state.mu)
            r2_ac, _ = wls_affine_r2(new_state.p, np.maximum(new_state.a, 0.0), new_state.Sigma)
            row.update(post_mean_atom=r2_am, post_cov_atom=r2_ac)
        rows.append(row)
        state = new_state
    return rows

def r2_cached(n, seed):
    path = os.path.join(CKPT_DIR, "r2", f"r2_d{DEPTH}_w{n}_s{seed}_nn{NUM_NODES}_{GRID}.json")
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    rows = affine_r2_by_layer(coordinate_spike_net(n, DEPTH, seed), n)
    json.dump(rows, open(path, "w"))
    print(f"  [r2] n={n} seed={seed}: {time.time()-t0:.1f}s", flush=True)
    return rows

print("measurement ready (affine_r2_by_layer, r2_cached)")
""")

# =============================================================================
md(r"""## §3 — Sanity: one net, the full $R^2$ table

Expectations: **pre / layer 0** mean is exactly affine ($R^2=1$ to fp; the input layer is a
joint-Gaussian regression) and its covariance is ~constant (`var_scale`$\approx 0$ — $R^2$
meaningless, printed as nan). **Post** mean/cov are genuinely nonlinear ($\Phi$ factors), so
$R^2<1$ there is the real hypothesis test; the `+atom` columns show whether the zero atom sits
on the same line.""")
code(r"""
n0, seed0 = 128, 10
rows0 = affine_r2_by_layer(coordinate_spike_net(n0, DEPTH, seed0), n0)
cols = ["pre_mean", "pre_cov", "post_mean", "post_cov", "post_mean_atom", "post_cov_atom"]
print(f"n={n0} seed={seed0}   (R^2; '--' = var_scale<1e-14: nothing to fit)")
print("  layer | " + " | ".join(f"{c:>14s}" for c in cols))
for row in rows0:
    cells = " | ".join(f"{row.get(c, float('nan')):14.6f}"
                       if np.isfinite(row.get(c, float('nan'))) else " " * 12 + "--"
                       for c in cols)
    print(f"    {row['layer']}   | {cells}")
print("\n  variation scales (SS_tot / ||mean||^2 -- how nonconstant the target is):")
for row in rows0:
    print(f"    layer {row['layer']}: pre_mean {row['vs_pre_mean']:.2e}  pre_cov {row['vs_pre_cov']:.2e}  "
          f"post_mean {row['vs_post_mean']:.2e}  post_cov {row['vs_post_cov']:.2e}")
""")

# =============================================================================
md(r"""## §4 — Scaling: $1 - R^2$ vs width, per layer

Four panels (pre/post $\times$ mean/cov), one curve per layer, seed-mean of $1-R^2$ on log-log
axes with the fitted power. Falling curves = the affine hypothesis improves with width; the
fitted slope quantifies the rate. `post +atom` is overlaid dashed where enabled.""")
code(r"""
def collect(col):
    out = np.full((len(WIDTHS), DEPTH, len(SEEDS)), np.nan)
    for wi, n in enumerate(WIDTHS):
        for si, s in enumerate(SEEDS):
            for row in r2_cached(n, s):
                v = row.get(col, float('nan'))
                if v is not None and np.isfinite(v):
                    out[wi, row['layer'], si] = 1.0 - v
    return out

def slope(xs, ys):
    m = np.isfinite(ys) & (ys > 0)
    if m.sum() < 2: return float('nan')
    return float(np.polyfit(np.log(np.asarray(xs)[m]), np.log(ys[m]), 1)[0])

panels = [("pre_mean", "pre-activation mean  1-R^2"), ("pre_cov", "pre-activation cov  1-R^2"),
          ("post_mean", "post-activation mean  1-R^2"), ("post_cov", "post-activation cov  1-R^2")]
w = np.array(WIDTHS, float)
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
for ax, (col, title) in zip(axes.ravel(), panels):
    dat = collect(col)
    for L in range(DEPTH):
        ymean = np.nanmean(dat[:, L, :], axis=1)
        sl = slope(WIDTHS, ymean)
        ax.loglog(w, ymean, "o-", label=f"layer {L} ~ n^{sl:+.2f}")
        for si in range(len(SEEDS)):
            ax.loglog(w, dat[:, L, si], ".", ms=3, alpha=.25, color=ax.lines[-1].get_color())
    if col.startswith("post") and POST_WITH_ATOM:
        dat_a = collect(col + "_atom")
        for L in range(DEPTH):
            ax.loglog(w, np.nanmean(dat_a[:, L, :], axis=1), "--", alpha=.6,
                      label=(f"layer {L} +atom" if L == 0 else None))
    ax.set_title(title, fontsize=11); ax.set_xlabel("width n"); ax.legend(fontsize=7)
plt.suptitle(f"affine-hypothesis error 1-R^2 vs width (depth {DEPTH}, {len(SEEDS)} seeds, "
             f"{NUM_NODES} nodes)", y=1.0)
plt.tight_layout(); plt.show()

print("   width |  " + "  ".join(f"{c:>10s}" for c, _ in panels) + "   (layer-mean of 1-R^2)")
for wi, n in enumerate(WIDTHS):
    vals = [np.nanmean(collect(c)[wi]) for c, _ in panels]
    print(f"   {n:5d} |  " + "  ".join(f"{v:.3e}" for v in vals))
""")

# =============================================================================
md(r"""## §5 — Reading the result

* **pre vs post**: whichever family has the higher $R^2$ (smaller $1-R^2$) at matched width is
  the better projection surface — this is the direct test of the `fit="pre"` vs `fit="post"`
  design choice, layer by layer.
* **slope**: $1-R^2$ falling as a power of $n$ says the affine closure error is a finite-width
  effect, consistent with the $K{=}2$ budget law surviving either projection.
* **`+atom` gap**: if `post_*_atom` tracks `post_*`, the zero atom sits on the family line and
  `atom="fit"` is safe; a persistent gap says keep `atom="exact"`.
* Caveat: nan / tiny `var_scale` cells mean the target barely varies across bins — the affine
  question is moot there (layer-0 pre-cov is the canonical case: exactly constant in the
  continuous limit, only quantization jitter on the grid).""")

nb.save(os.path.join(os.path.dirname(__file__), "affine_r2_colab.ipynb"))

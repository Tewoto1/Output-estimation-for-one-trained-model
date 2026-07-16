"""Generates affine_r2_colab.ipynb (valid nbformat-4 JSON).

HOW LINEAR IS THE BULK-GIVEN-SPIKE LAW, per layer, and how does it scale with
width? Replaces the deleted ``affine_r2_experiment.py`` (the ARC-infra .py; this
notebook version is Colab-sized: widths up to 1024, 3-4 seeds, ~30-40 min on a T4,
QUICK subset in ~a minute).

Both analytic variants compress the bulk conditional to an AFFINE family in the
spike coordinate; they differ in where the projection happens. The notebook
measures the hypothesis DIRECTLY, at every layer, separately for:

  * PRE-activation:  exact reconditioned cell moments  (m_hat_j, S_hat_j) vs y_j
  * POST-activation: exact post-ReLU node moments      (r_k, R_k)         vs a_k
    (and optionally WITH the zero atom as an a=0 data point -- the atom="fit"
    hypothesis: is the merge of all negative cells on the same line?)

via the weighted pooled R^2 (Frobenius pooling for covariances), plus var_scale
to flag constant targets (layer-0 pre-cov is exactly constant -> R^2 meaningless).
NO Monte-Carlo anywhere -- the question is internal to the surrogate.

Sections (knobs live in the notebook; per-(width,seed) rows cached ->
nothing recomputed on re-run):
  §1  config + cache
  §2  measurement (wls_affine_r2 / affine_r2_by_layer / cached runner)
  §3  the sweep (widths x seeds; per-layer progress at big widths)
  §4  aggregation: 1-R^2 table + width slopes; results saved to results/
  §5  plot: four panels (pre/post x mean/cov), 1-R^2 vs n, one curve per layer

CACHE NOTE: points are cached as checkpoints/analytic_kprop/r2/r2v2_*.json.
The v2 prefix is deliberate: on 2026-07-15 ``num_nodes`` became the POSITIVE-side
cell budget (the negative side is now mass-adaptive), so grids differ from the
old ``r2_*.json`` rows -- those are left untouched but not reused.

Run:
    python "experiments/analytic_kprop/build_affine_r2_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# The **affine hypothesis**, measured: $R^2$ of the bulk-given-spike law vs width

Both analytic variants compress the bulk conditional law to an **affine family** in the spike
coordinate — `fit="pre"` projects the reconditioned pre-activation cells,

$$C \mid Y = y \;\rightsquigarrow\; \mathcal N\!\big(\mu_0 + \mu_1 y,\; \Sigma_0 + \Sigma_1 y\big),$$

`fit="post"` projects the post-ReLU nodes, $B \mid A{=}a \rightsquigarrow \mathcal N(m_0+m_1a,\,W_0+W_1a)$.
This notebook measures that hypothesis **directly**, at every layer, on the exact propagated
moments (no Monte-Carlo — the question is internal to the surrogate; `fit="pre"` carries exact
nonlinear node moments, so the measured data is never contaminated by the projection being tested,
except across layers where the algorithm itself projects):

* **PRE**: exact cell moments $(\hat m_j, \hat S_j)$ vs cell value $y_j$ — weighted affine fit,
  pooled $R^2$ (Frobenius for covariances);
* **POST**: exact post-ReLU node moments $(r_k, R_k)$ vs node value $a_k$ (positive nodes), and
  optionally **with the zero atom** as an $a{=}0$ data point (the `atom="fit"` hypothesis: is the
  merge of all negative cells on the same line?).

`var_scale` flags constant targets (canonical: layer-0 pre-cov is *exactly* bin-independent, so its
$R^2$ is meaningless and reported as NaN). Headline output: **$1-R^2$ vs width, per layer** — does
the affine error vanish with $n$, and at what rate?

**Runtime**: full sweep (widths to 1024, 3 seeds, depth 3) ≈ 30–40 min on a Colab T4 — the cost is
the exact bivariate-ReLU kernel per retained cell, not the fits. `QUICK=True` (widths ≤ 128) runs in
about a minute on CPU. Per-(width, seed) rows are **cached** in `checkpoints/analytic_kprop/r2/`
(`r2v2_*` keys — v2 because `num_nodes` is the *positive-side* budget since 2026-07-15, so grids
differ from old `r2_*` rows), so re-runs and seed extensions are incremental.

*Context*: the deleted `affine_r2_experiment.py` quick run (widths ≤ 128) found POST the much better
linear surface (1−R² ~ 1e-2..1e-3, falling with width) while PRE L1/L2 sat flat-or-rising at
~0.15–0.25; the 2026-07-15 empirical MC check on real nets (binned conditional covariances,
split-half debiased) matched: covariance affine-R² ≈ 1.0 at layer 0 post, 0.66–0.93 on positive bins
deeper, with the leftover mostly smooth curvature (quadratic → 0.88–1.0) — **not** a
flat-except-beginning-region structure. This sweep verifies the width trend properly.""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 — Config + cache

All knobs here. Nets are random untrained coordinate-spiked MLPs, deterministic per seed. The
recyclable artifact is the per-`(width, seed)` R² row set (JSON, one file each) — delete a file to
force recomputation. `DEVICE`/`GPU_MIN_WIDTH` torch-offload the per-cell congruences (the
diagnostics-path batch congruence inside `percell_bulk_moments`) at big widths; the exact ReLU
kernel itself is CPU (scipy special functions) and is what `WORKERS` parallelizes.""")
code(r"""
import os, json, time
import numpy as np
import matplotlib.pyplot as plt

from Mecha_preds.analytic_kprop import (
    gaussian_input_state, analytic_layer_update, percell_bulk_moments)
from Mecha_preds.analytic_kprop.core import (
    _layer_block, _component_params, _pair_stats, _torch_device)

# ---------------- knobs (edit here) ----------------
QUICK         = True             # False -> the full sweep (widths to 1024; GPU box recommended)
WIDTHS        = [16, 32, 64, 128] if QUICK else [16, 32, 64, 128, 256, 512, 1024]
SEEDS         = [10, 11, 12]     # 3 seeds; append 13 for a 4th
DEPTH         = 3
NUM_NODES     = 40               # POSITIVE-side cell budget (2026-07-15 semantics; neg side adaptive)
GRID          = "w2"             # "w2" (Lloyd-Max on the exact mixture) | "uniform"
THETA         = 1.0              # M = W + THETA e1 e1^T
ATOM_FIT      = True             # also fit POST with the zero atom included as an a=0 point
WORKERS       = "auto"           # per-node exact-ReLU threads (1 = serial; identical results)
GPU_MIN_WIDTH = 512              # torch-offload congruences from this width up (if CUDA available)

def _cuda():
    try:
        import torch; return bool(torch.cuda.is_available())
    except Exception:
        return False
HAVE_GPU = _cuda()

R2_DIR = os.path.join("checkpoints", "analytic_kprop", "r2"); os.makedirs(R2_DIR, exist_ok=True)
RESULTS_DIR = os.path.join("results", "analytic_kprop_affine_r2"); os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"QUICK={QUICK} | widths={WIDTHS} | seeds={SEEDS} | depth={DEPTH} | num_nodes={NUM_NODES} "
      f"| grid={GRID} | atom_fit={ATOM_FIT} | gpu={HAVE_GPU} (offload >= n={GPU_MIN_WIDTH})")
""")

# =============================================================================
md(r"""## §2 — Measurement

`wls_affine_r2` is the one estimator: weighted affine fit of per-cell/per-node data against the
scalar cell/node value, pooled $R^2$ + `var_scale` (≈0 ⇒ the target is constant across cells and
$R^2$ is meaningless → NaN). `affine_r2_by_layer` advances the exact `fit="pre"` state layer by
layer and measures PRE on the *same retained grid* the layer used and POST on the advanced state's
nodes. Rows are cached per `(width, seed)`.""")
code(r"""
def coordinate_spike_net(n, depth, seed, *, theta=THETA):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n)); P[0, 0] = theta
    Ws = [(rng.standard_normal((n, n)) / np.sqrt(n) + P, None) for _ in range(depth)]
    Ws.append((rng.standard_normal((8, n)) / np.sqrt(n), None))
    return Ws


def wls_affine_r2(w, x, data):
    '''Weighted affine fit data_j ~ b0 + b1 x_j; pooled R^2 (Frobenius for matrices)
    + var_scale in [0,1] = SS_tot / weighted second moment.'''
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


def affine_r2_by_layer(Ws, n, *, num_nodes=NUM_NODES, grid=GRID, workers=WORKERS,
                       dev=None, atom_fit=ATOM_FIT, tag=None):
    '''Per-layer weighted R^2 of the affine hypothesis, PRE and POST.'''
    d = n - 1
    state = gaussian_input_state(d)
    rows = []
    for li in range(len(Ws) - 1):
        t_layer = time.time()
        M = np.asarray(Ws[li][0], dtype=np.float64)
        gamma, r, u, V, beta, eta_b = _layer_block(M, None, d)
        mY, sY2, mC, g = _component_params(state, gamma, r, u, V, beta, eta_b)
        new_state, aff = analytic_layer_update(
            state, M, None, num_nodes=num_nodes, grid=grid, workers=workers, dev=dev)
        # PRE: exact cell moments on the SAME retained grid the layer used
        Q, ym, dl, vv, stoch = _pair_stats(state.p, mY, sY2, aff.edges, min_prob=1e-15)
        wr = state.p @ Q; keep = wr > 1e-15
        mh, Sh = percell_bulk_moments(state.p, Q[:, keep], dl[:, keep], vv[:, keep],
                                      sY2, stoch, wr[keep],
                                      state.Sigma, state.t2, mC, g, u, V, dev)
        r2_pm, vs_pm = wls_affine_r2(aff.w, aff.y, mh)
        r2_pc, vs_pc = wls_affine_r2(aff.w, aff.y, Sh)
        # POST: exact post-ReLU node moments of the advanced state (positive nodes)
        pos = new_state.a > 0
        r2_qm, vs_qm = wls_affine_r2(new_state.p[pos], new_state.a[pos], new_state.mu[pos])
        r2_qc, vs_qc = wls_affine_r2(new_state.p[pos], new_state.a[pos], new_state.Sigma[pos])
        row = dict(layer=li,
                   pre_mean=r2_pm, pre_cov=r2_pc, post_mean=r2_qm, post_cov=r2_qc,
                   vs_pre_mean=vs_pm, vs_pre_cov=vs_pc,
                   vs_post_mean=vs_qm, vs_post_cov=vs_qc)
        if atom_fit and (~pos).any():
            r2_am, _ = wls_affine_r2(new_state.p, np.maximum(new_state.a, 0.0), new_state.mu)
            r2_ac, _ = wls_affine_r2(new_state.p, np.maximum(new_state.a, 0.0), new_state.Sigma)
            row.update(post_mean_atom=r2_am, post_cov_atom=r2_ac)
        rows.append(row)
        if tag:
            print(f"    {tag} layer {li}: {time.time()-t_layer:.1f}s", flush=True)
        state = new_state
    return rows


def r2_cached(n, seed):
    # v2 keys: num_nodes = POSITIVE-side budget since 2026-07-15 (old r2_* rows used
    # the proportional split -> different grids; left untouched, not reused).
    path = os.path.join(R2_DIR, f"r2v2_d{DEPTH}_w{n}_s{seed}_nn{NUM_NODES}_{GRID}.json")
    if os.path.exists(path):
        return json.load(open(path)), True
    dev = _torch_device("cuda") if (HAVE_GPU and n >= GPU_MIN_WIDTH) else None
    t0 = time.time()
    Ws = coordinate_spike_net(n, DEPTH, seed)
    rows = affine_r2_by_layer(Ws, n, dev=dev,
                              tag=(f"n={n} seed={seed}" if n >= 256 else None))
    json.dump(rows, open(path, "w"))
    print(f"  [r2] n={n} seed={seed}: {time.time()-t0:.1f}s "
          f"({'cuda' if dev is not None else 'numpy'})", flush=True)
    return rows, False
""")

# =============================================================================
md(r"""## §3 — The sweep

Cached points load instantly; only missing `(width, seed)` pairs compute. Big widths print per-layer
progress.""")
code(r"""
points = []                                            # dict(width, seed, layer, metrics...)
for n in WIDTHS:
    t0 = time.time(); fresh = 0
    for s in SEEDS:
        rows, cached = r2_cached(n, s)
        fresh += (not cached)
        for row in rows:
            points.append(dict(width=n, seed=s, **row))
    print(f"[width {n:5d}] {len(SEEDS)} seeds in {time.time()-t0:6.1f}s "
          f"({fresh} computed, {len(SEEDS)-fresh} from cache)", flush=True)
print(f"total rows: {len(points)}")
""")

# =============================================================================
md(r"""## §4 — Aggregation: $1-R^2$ per (metric, layer) vs width, with fitted slopes

Seed-mean and SEM of $1-R^2$; log-log slope per (metric, layer). NaN cells = the target had no
across-cell variation to explain (`var_scale` ≈ 0 — layer-0 pre-cov). Results also saved to
`results/analytic_kprop_affine_r2/` (points.jsonl, results.csv/json, summary.log).""")
code(r"""
PANELS = [("pre_mean", "pre-activation mean"), ("pre_cov", "pre-activation cov"),
          ("post_mean", "post-activation mean"), ("post_cov", "post-activation cov")]
metrics = [c for c, _ in PANELS] + (["post_mean_atom", "post_cov_atom"] if ATOM_FIT else [])

def one_minus_r2(col, n, L):
    vals = [1.0 - p[col] for p in points
            if p["width"] == n and p["layer"] == L
            and p.get(col) is not None and np.isfinite(p.get(col, np.nan))]
    return (float(np.mean(vals)) if vals else float("nan"),
            float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            len(vals))

def slope(xs, ys):
    ys = np.asarray(ys, float)
    m = np.isfinite(ys) & (ys > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(np.asarray(xs, float)[m]), np.log(ys[m]), 1)[0])

agg = []
lines = [f"affine-hypothesis R^2 scaling -- depth {DEPTH}, seeds {SEEDS}, num_nodes {NUM_NODES} "
         f"(positive-side budget), grid {GRID}", "",
         "1 - R^2 (seed mean); slope fitted over width per (metric, layer)", "",
         "  metric          L  |  " + "  ".join(f"n={n:<7d}" for n in WIDTHS)]
for col in metrics:
    for L in range(DEPTH):
        ys = []
        for n in WIDTHS:
            m_, sem, k = one_minus_r2(col, n, L)
            agg.append(dict(metric=col, layer=L, width=n,
                            one_minus_r2_mean=m_, sem=sem, n_seeds=k))
            ys.append(m_)
        sl = slope(WIDTHS, ys)
        cells = "  ".join(f"{v:.3e}" if np.isfinite(v) else "   --    " for v in ys)
        lines.append(f"  {col:15s} L{L} |  {cells}   ~ n^{sl:+.2f}")
    lines.append("")
summary = "\n".join(lines)
print(summary)

with open(os.path.join(RESULTS_DIR, "points.jsonl"), "w") as f:
    for p in points:
        f.write(json.dumps(p) + "\n")
with open(os.path.join(RESULTS_DIR, "summary.log"), "w") as f:
    f.write(summary + "\n")
with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
    json.dump({"config": dict(widths=WIDTHS, seeds=SEEDS, depth=DEPTH, num_nodes=NUM_NODES,
                              grid=GRID, theta=THETA, atom_fit=ATOM_FIT),
               "aggregates": agg, "points": points}, f, indent=1)
import csv
with open(os.path.join(RESULTS_DIR, "results.csv"), "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=["metric", "layer", "width",
                                         "one_minus_r2_mean", "sem", "n_seeds"])
    wcsv.writeheader()
    for r in agg:
        wcsv.writerow(r)
print("saved ->", RESULTS_DIR)
""")

# =============================================================================
md(r"""## §5 — Plot: affine-hypothesis error vs width

Four panels (pre/post × mean/cov), $1-R^2$ vs $n$ log-log, one curve per layer; dashed = POST with
the zero atom folded in (`atom="fit"` hypothesis — expect it FLAT: the atom stays off the line,
which is why `atom="exact"` is the default). Quick-run reference: POST falls with width
(~1e-2 → 1e-3), PRE L1/L2 flat-or-rising ~0.15–0.25 (pre_mean L2 was ~$n^{+0.7}$).""")
code(r"""
w = np.array(WIDTHS, float)
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
for ax, (col, title) in zip(axes.ravel(), PANELS):
    for L in range(DEPTH):
        ys = np.array([next(a["one_minus_r2_mean"] for a in agg
                            if a["metric"] == col and a["layer"] == L and a["width"] == n)
                       for n in WIDTHS])
        sl = slope(WIDTHS, ys)
        ax.loglog(w, ys, "o-", label=f"layer {L} ~ n^{sl:+.2f}")
        if col.startswith("post") and ATOM_FIT:
            ya = np.array([next(a["one_minus_r2_mean"] for a in agg
                                if a["metric"] == col + "_atom" and a["layer"] == L
                                and a["width"] == n) for n in WIDTHS])
            if np.isfinite(ya).any():
                ax.loglog(w, ya, "--", alpha=.5, color=ax.lines[-1].get_color(),
                          label=("+atom" if L == 0 else None))
    ax.set_title(title + "  (1 - R^2)", fontsize=11)
    ax.set_xlabel("width n"); ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
plt.suptitle(f"affine-hypothesis error vs width (depth {DEPTH}, {len(SEEDS)} seeds, "
             f"{NUM_NODES} positive-side nodes)", y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "r2_scaling.png"), dpi=150)
plt.show()
print("plot ->", os.path.join(RESULTS_DIR, "r2_scaling.png"))
""")

nb.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "affine_r2_colab.ipynb"))

"""Generates error_by_layer_colab.ipynb (valid nbformat-4 JSON).

PER-BIN accuracy of the binned-K2 predictor, by layer and step -- NOT the congregated overall
(mean, cov). At each step we know the bin regions on the spike coordinate (the grid edges) and the
predictor's CONDITIONAL bulk moments per bin (mu_a, Sigma_a). We then run a lot of MC, bin the
samples by the SAME spike-coordinate edges, and measure the empirical conditional (mean, cov) of the
bulk inside each bin. The per-bin error isolates the WITHIN-bin Gaussian-closure quality (it drops the
between-bin spread that the congregated cov mixes in).

Steps: lin{l} = pre-activation of layer l (bins on the pre-edges, spike = A^+),
       relu{l} = post-activation of layer l (bins on the post-edges, spike = ReLU(A^+)).

Outputs: (1) per-bin cov/mean error vs bin (for a chosen step) -- which bin REGIONS are inaccurate;
         (2) the p-weighted AVERAGE per-bin error per step, and how it scales with width.

Run:  python "experiments/binned_kprop/build_error_by_layer_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

md(r"""# Per-**bin** accuracy by layer & step (not congregated)

The binned predictor's real approximation is, *inside each bin* `α` (a region of the spike coordinate),
`B | A∈bin_α ≈ N(μ_α, Σ_α)`. So the honest error to measure is **per bin**: compare `(μ_α, Σ_α)` to the
**MC-empirical** conditional mean/cov of the bulk over the samples whose spike falls in bin `α`. This
drops the between-bin spread that the congregated `unconditional_mean_cov` mixes in, and isolates the
**within-bin Gaussian-closure** quality.

Procedure: run the algorithm → read off the **bin edges** (where the bins are) and the per-bin
`(p_α, μ_α, Σ_α)` at every step → run a lot of MC → bin samples by the *same* edges → empirical
`(mean_α, cov_α)`. We report per-bin error for a few bins, and the **p-weighted average per-bin error**
per step and its width scaling.

Steps: `lin{l}` (pre-activation, spike `A⁺`), `relu{l}` (post-activation, spike `ReLU(A⁺)`).""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## Config

`MC_DEVICE` auto-selects the **T4 GPU** for the MC ground-truth when CUDA is present (the predictor
itself is torch-free numpy and stays on CPU — it's a fraction of the cost). On GPU the network matmuls
and the O(d²) per-bin covariance both run on-device, which is the whole ballgame for wide models.""")
code(r"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from Mecha_preds.binned_kprop.core import (gaussian_initial_state, linear_step_k2, relu_step_k2,
    lloyd_max_edges, lloyd_max_edges_mixture, make_gaussian_edges, make_relu_post_edges, _spike_mixture)

QUICK      = True
WIDTHS     = [32, 64, 128] if QUICK else [32, 64, 128, 256, 512, 1024]
DEPTH      = 3
NUMBINS    = 21
THETA      = 1.0
OUT_DIM    = 8
SEED       = 10
GRID       = "wasserstein"                # "wasserstein" | "fixed"
try:
    import torch; _HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    torch = None; _HAS_CUDA = False
MC_DEVICE  = "cuda" if _HAS_CUDA else "cpu"    # MC ground-truth runs on the T4 when available
MC_SAMPLES = 2_000_000 if QUICK else 20_000_000
MC_ELEMS   = 2.0e8 if MC_DEVICE == "cuda" else 4.0e7   # width-aware batch (~constant memory: batch*n)
MC_WORKERS = 1 if MC_DEVICE == "cuda" else max(1, os.cpu_count() or 1)  # CPU path: parallel sample shards
def mc_batch(n): return int(min(MC_SAMPLES, 8_000_000, max(100_000, MC_ELEMS // n)))  # fewer batches at small n
MIN_COUNT  = 2000                         # base gate: skip a bin (mean/diagonal error) below this MC count
COV_MIN_PER_DIM = 200                     # FULL-cov gate scales with width: an empirical dxd cov needs
                                          # ~d samples to beat its own noise (rel err ~ sqrt(d/N)); require
                                          # >= COV_MIN_PER_DIM * d samples before trusting a bin's full cov
PLOT_STEP  = 2 * DEPTH - 1                # which step's per-bin curve to plot (default: last relu)
CKPT_DIR   = "checkpoints/binned_kprop/perbin"; os.makedirs(CKPT_DIR, exist_ok=True)
STEP_LABELS = [f"{k}{l}" for l in range(DEPTH) for k in ("lin", "relu")]
print(f"QUICK={QUICK} widths={WIDTHS} depth={DEPTH} num_bins={NUMBINS} grid={GRID} MC={MC_SAMPLES:,}")
print(f"MC device: {MC_DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if MC_DEVICE=='cuda' else f"  [numpy CPU x{MC_WORKERS} workers]"))
print(f"width-aware MC batch: n={min(WIDTHS)}-> {mc_batch(min(WIDTHS)):,},  n={max(WIDTHS)}-> {mc_batch(max(WIDTHS)):,}")
print(f"steps: {STEP_LABELS}   (plotting per-bin curve for step {STEP_LABELS[PLOT_STEP]})")
""")

# =============================================================================
md(r"""## Helpers — capture bin regions + per-bin predictor moments, and the MC per-bin conditional moments""")
code(r"""
def net(n, depth, seed, theta=THETA, out_dim=OUT_DIM):
    rng = np.random.default_rng(seed); P = np.zeros((n, n)); P[0, 0] = theta
    Ws = [(rng.standard_normal((n, n)) / np.sqrt(n) + P, None) for _ in range(depth)]
    Ws.append((rng.standard_normal((out_dim, n)) / np.sqrt(n), None)); return Ws

# propagate; capture per step: (label, spike-edges, p, a, mu, Sigma). The state after linear_step is
# binned on pre_edges (spike = A^+); after relu_step on post_edges (spike = ReLU(A^+)).
def capture(Ws, n):
    d = n - 1
    init = lloyd_max_edges(0.0, 1.0, NUMBINS)[0] if GRID == "wasserstein" else make_gaussian_edges(NUMBINS)
    st = gaussian_initial_state(d, init); caps = []
    for li in range(DEPTH):
        M = Ws[li][0]
        if GRID == "wasserstein":
            p, mY, sY = _spike_mixture(st, M)
            pre = lloyd_max_edges_mixture(p, mY, sY, NUMBINS)[0]
            post = lloyd_max_edges_mixture(p, mY, sY, NUMBINS, rectified=True)[0]
        else:
            pre, post = make_gaussian_edges(NUMBINS), make_relu_post_edges(NUMBINS)
        st = linear_step_k2(st, M, pre);  caps.append(("lin", pre, st.p.copy(), st.a.copy(), st.mu.copy(), st.Sigma.copy()))
        st = relu_step_k2(st, post);      caps.append(("relu", post, st.p.copy(), st.a.copy(), st.mu.copy(), st.Sigma.copy()))
    return caps

# MC empirical conditional (mean, cov, count) of the bulk PER BIN, per step, streamed (numpy/CPU).
# Split into a picklable SHARD (one process's chunk of samples -> raw sums) + finalize, so the CPU path
# can run shards in parallel across cores (multiprocessing) and just add the per-bin accumulators.
def _mc_shard(args):
    Ws, n, edges, S, B, seed = args
    L = len(Ws) - 1; m = [len(e) - 1 for e in edges]; d = n - 1
    sm = [np.zeros((m[i], d)) for i in range(2 * L)]
    sq = [np.zeros((m[i], d, d)) for i in range(2 * L)]
    cnt = [np.zeros(m[i], dtype=np.int64) for i in range(2 * L)]
    def acc(i, spike, bulk):
        idx = np.clip(np.searchsorted(edges[i], spike, side="right") - 1, 0, m[i] - 1)
        for a in np.unique(idx):
            Bm = bulk[idx == a]
            sm[i][a] += Bm.sum(0); sq[i][a] += Bm.T @ Bm; cnt[i][a] += Bm.shape[0]
    rng = np.random.default_rng(seed); c = 0
    while c < S:
        b = min(B, S - c); h = rng.standard_normal((b, n))
        for li, (W, _b) in enumerate(Ws):
            z = h @ W.T
            if li < L:
                acc(2 * li, z[:, 0], z[:, 1:])
                hz = np.maximum(z, 0.0); acc(2 * li + 1, hz[:, 0], hz[:, 1:]); h = hz
        c += b
    return sm, sq, cnt

def _mc_finalize(sm, sq, cnt, m, d):
    out = []
    for i in range(len(m)):
        means = np.zeros((m[i], d)); covs = np.zeros((m[i], d, d))
        for a in range(m[i]):
            if cnt[i][a] > 0:
                mu = sm[i][a] / cnt[i][a]; means[a] = mu; covs[a] = sq[i][a] / cnt[i][a] - np.outer(mu, mu)
        out.append((means, covs, cnt[i]))
    return out

def mc_perbin_np(Ws, n, caps, S, B, seed, workers=1):
    L = len(Ws) - 1; edges = [caps[i][1] for i in range(2 * L)]; m = [len(e) - 1 for e in edges]; d = n - 1
    if workers and workers > 1 and S >= 2 * workers:
        import multiprocessing as mp
        per = -(-S // workers)   # ceil-divide samples across workers; distinct seed per shard
        args = [(Ws, n, edges, min(per, S - k * per), B, seed + 1000 * k)
                for k in range(workers) if k * per < S]
        with mp.Pool(len(args)) as pool:
            parts = pool.map(_mc_shard, args)
        sm = [sum(pt[0][i] for pt in parts) for i in range(2 * L)]
        sq = [sum(pt[1][i] for pt in parts) for i in range(2 * L)]
        cnt = [sum(pt[2][i] for pt in parts) for i in range(2 * L)]
    else:
        sm, sq, cnt = _mc_shard((Ws, n, edges, S, B, seed))
    return _mc_finalize(sm, sq, cnt, m, d)

# Same computation on the GPU (T4): network matmul + per-bin covariance accumulation, all on CUDA.
# Mirrors the numpy version (mask + matmul per present bin); the tiny sync from masking is negligible
# next to the O(d^2) matmuls -- unlike a one-hot form, this does only the essential flops (not m x more).
# fp32 matmul (T4 is slow at fp64) with fp64 accumulators; fp32 error is far below the MC noise floor.
def mc_perbin_torch(Ws, n, caps, S, B, seed, device="cuda"):
    import torch
    L = len(Ws) - 1; d = n - 1
    m = [caps[i][1].shape[0] - 1 for i in range(2 * L)]
    edges = [torch.as_tensor(caps[i][1], device=device, dtype=torch.float64) for i in range(2 * L)]
    Wt = [torch.as_tensor(W, device=device, dtype=torch.float32) for (W, _b) in Ws]
    sm = [torch.zeros((m[i], d), device=device, dtype=torch.float64) for i in range(2 * L)]
    sq = [torch.zeros((m[i], d, d), device=device, dtype=torch.float64) for i in range(2 * L)]
    cnt = [torch.zeros(m[i], device=device, dtype=torch.int64) for i in range(2 * L)]
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    def acc(i, spike, bulk):
        idx = torch.clamp(torch.searchsorted(edges[i], spike.double(), right=True) - 1, 0, m[i] - 1)
        for a in torch.unique(idx).tolist():
            Bm = bulk[idx == a]
            sm[i][a] += Bm.sum(0).double(); sq[i][a] += (Bm.t() @ Bm).double(); cnt[i][a] += Bm.shape[0]
    c = 0
    while c < S:
        b = min(B, S - c)
        h = torch.randn((b, n), device=device, dtype=torch.float32, generator=gen)
        for li, (W, _b) in enumerate(Ws):
            z = h @ Wt[li].t()
            if li < L:
                acc(2 * li, z[:, 0], z[:, 1:])
                hz = torch.clamp(z, min=0.0); acc(2 * li + 1, hz[:, 0], hz[:, 1:]); h = hz
        c += b
    out = []
    for i in range(2 * L):
        cn = cnt[i].clamp(min=1).double()
        mu = sm[i] / cn[:, None]
        cov = sq[i] / cn[:, None, None] - mu[:, :, None] * mu[:, None, :]
        empty = (cnt[i] == 0)
        mu = mu.clone(); mu[empty] = 0.0; cov = cov.clone(); cov[empty] = 0.0
        out.append((mu.cpu().numpy(), cov.cpu().numpy(), cnt[i].cpu().numpy()))
    return out

def mc_perbin(Ws, n, caps, S, B, seed):
    if MC_DEVICE == "cuda":
        return mc_perbin_torch(Ws, n, caps, S, B, seed, device="cuda")
    return mc_perbin_np(Ws, n, caps, S, B, seed, workers=MC_WORKERS)

def rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
def cov_diag_rel(pred, mc):                   # width-ROBUST cov error: diagonal variances only.
    dp = np.diag(pred); dm = np.diag(mc)      # MC noise on a variance is ~sqrt(2/N) -- independent of d,
    return float(np.linalg.norm(dp - dm) / (np.linalg.norm(dm) + 1e-30))   # unlike full Frobenius ~sqrt(d/N)
def cov_noise_floor(d, N):                    # rel-Frobenius error of an empirical dxd cov from N samples
    return float(np.sqrt(d / max(N, 1)))      # (= the error you'd measure even with a PERFECT predictor)
def mean_scaled(pm, mm, cov):                 # mean error normalized by the bin's activation scale
    sc = np.sqrt((float(mm @ mm) + float(np.trace(cov))) / len(mm)); return float(np.linalg.norm(pm - mm) / (sc + 1e-30))
print("helpers ready")
""")

# =============================================================================
md(r"""## Run — per-bin error at every step (MC cached); p-weighted average per step""")
code(r"""
per_bin = {}   # [width] -> list over steps of dict(mean[m], cov[m], p[m], a[m], cnt[m], ok[m])
avg = {}       # [width] -> list over steps of (avg_mean_err, avg_cov_err)  (p-weighted, resolvable bins)
for n in WIDTHS:
    import pickle
    Ws = net(n, DEPTH, SEED); caps = capture(Ws, n)
    key = os.path.join(CKPT_DIR, f"mcpb_w{n}_d{DEPTH}_nb{NUMBINS}_{GRID}_s{SEED}_S{MC_SAMPLES}.pkl")
    if os.path.exists(key):
        mc = pickle.load(open(key, "rb"))
    else:
        t0 = time.time(); mc = mc_perbin(Ws, n, caps, MC_SAMPLES, mc_batch(n), 10_000 + SEED)
        pickle.dump(mc, open(key, "wb")); print(f"  n={n}: MC {time.time()-t0:.1f}s  (batch {mc_batch(n):,}, {MC_DEVICE})")
    steps_pb = []; steps_avg = []
    minc_cov = max(MIN_COUNT, int(COV_MIN_PER_DIM * (n - 1)))   # width-scaled gate for the FULL cov
    for i in range(len(caps)):
        _lbl, _edg, p, a, mu, Sig = caps[i]; mmc, cmc, cnt = mc[i]; mm = len(p)
        me = np.full(mm, np.nan); ce = np.full(mm, np.nan); cd = np.full(mm, np.nan); nf = np.full(mm, np.nan)
        ok = cnt >= MIN_COUNT; okc = cnt >= minc_cov            # base gate (mean/diag) | strict gate (full cov)
        for al in range(mm):
            if ok[al]:
                me[al] = mean_scaled(mu[al], mmc[al], cmc[al])
                cd[al] = cov_diag_rel(Sig[al], cmc[al])          # width-robust
                ce[al] = rel(Sig[al], cmc[al])                   # full Frobenius (noise ~ sqrt(d/N))
                nf[al] = cov_noise_floor(n - 1, cnt[al])         # that noise floor, per bin
        wb = p * ok;  wb = wb / (wb.sum() + 1e-30)               # weights for mean/diag
        wf = p * okc; wf = wf / (wf.sum() + 1e-30)               # weights for full cov (strict gate)
        steps_pb.append(dict(mean=me, cov=ce, cov_diag=cd, nf=nf, p=p, a=a, cnt=cnt, ok=ok, okc=okc))
        steps_avg.append((float(np.nansum(wb * np.nan_to_num(me))),
                          float(np.nansum(wf * np.nan_to_num(ce))),
                          float(np.nansum(wb * np.nan_to_num(cd)))))
    per_bin[n] = steps_pb; avg[n] = steps_avg
    print(f"  n={n:5d} | ROBUST diag-cov err by step: " +
          "  ".join(f"{STEP_LABELS[i]}:{steps_avg[i][2]:.1e}" for i in range(len(caps))))
    print(f"        | full-cov err (>= {minc_cov:,} samp/bin gate): " +
          "  ".join(f"{STEP_LABELS[i]}:{steps_avg[i][1]:.1e}" for i in range(len(caps))))
print("(mean/diagonal use >= MIN_COUNT; full cov uses the width-scaled gate; averages are p-weighted)")
""")

# =============================================================================
md(r"""## Plot 1 — WHERE are the inaccurate bins? per-bin error vs bin (for one step)

Per-bin cov error against the bin representative `a`, bubble size ∝ mass `p`. **Two things to know before
reading this:**

- **Wasserstein/Lloyd–Max bins are NOT equal-mass.** They minimize squared quantization distortion, which
  places bin density ∝ f^(1/3), *not* ∝ f. So mass per bin is deliberately unequal (20×+ range: dense
  near the mode, sparse in the tails). Equal-mass would be a *different* objective (quantile binning) and
  is worse for the K=2 propagation. Also, the spike law is a Gaussian **mixture** (one component per prior
  bin) with a post-ReLU **atom at 0**, so it is multimodal — a bin whose representative lands in a valley
  gets ≈0 mass, which is why you see tiny-mass bins *between* big ones. Those are excluded from the average.
- **Low-mass bins show high FULL-cov error because of MC noise, not the predictor.** An empirical d×d
  covariance from N samples has rel error ≈ sqrt(d/N) regardless of the predictor. The **dashed line** is
  that noise floor per bin — low-mass bins sit right on it. The **right panel** uses only the diagonal
  variances, whose noise is ≈ sqrt(2/N) *independent of width*, so it reveals the real per-bin structure.""")
code(r"""
si = PLOT_STEP
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for n in WIDTHS:
    dd = per_bin[n][si]; ok = dd["ok"]; o = np.argsort(dd["a"][ok])
    sc = ax[0].scatter(dd["a"][ok], dd["cov"][ok], s=300 * dd["p"][ok] + 8, alpha=.55, label=f"n={n}")
    # dashed = MC noise floor sqrt(d/N_bin): the error you'd measure even if the predictor were PERFECT.
    ax[0].plot(dd["a"][ok][o], dd["nf"][ok][o], "--", lw=1.2, alpha=.8, color=sc.get_facecolor()[0])
    ax[1].scatter(dd["a"][ok], dd["cov_diag"][ok], s=300 * dd["p"][ok] + 8, alpha=.55, label=f"n={n}")
ax[0].set_title(f"FULL cov error @ {STEP_LABELS[si]}   (dashed = MC noise floor)")
ax[1].set_title(f"DIAGONAL cov error @ {STEP_LABELS[si]}   (width-robust)")
for a in ax:
    a.set_yscale("log"); a.set_xlabel(f"bin representative  a  (spike value; bubble ∝ mass p)")
    a.set_ylabel("rel error vs MC (within-bin)"); a.grid(True, which="both", alpha=.25); a.legend(fontsize=8)
fig.tight_layout(); plt.show()
print("Low-mass bins sit ON the dashed noise floor in the LEFT panel -> their 'error' is MC sampling noise,")
print("not predictor error. The RIGHT (diagonal) panel is width-robust and shows the real per-bin structure.")
""")

# =============================================================================
md(r"""## Plot 2 — p-weighted **average** per-bin error, per step, vs width

The average within-bin inaccuracy (not congregated). Left: trajectory across steps for each width
(where does the within-bin closure degrade?). Right: scaling with width per step, with fitted slopes —
does the within-bin closure error stay ~n⁻¹ (RMS) at early layers and flatten at later ones?""")
code(r"""
x = np.arange(len(STEP_LABELS)); W = np.array(WIDTHS, float)
avg_cov      = {n: np.array([avg[n][i][1] for i in range(len(STEP_LABELS))]) for n in WIDTHS}  # FULL (noise-limited at shallow layers)
avg_cov_diag = {n: np.array([avg[n][i][2] for i in range(len(STEP_LABELS))]) for n in WIDTHS}  # width-robust (diagonal)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for n in WIDTHS:
    ax[0].plot(x, avg_cov_diag[n], "o-", label=f"n={n}")
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(STEP_LABELS, rotation=45)
ax[0].set_ylabel("p-wtd avg diagonal-cov error"); ax[0].set_title("ROBUST within-bin error across steps"); ax[0].grid(True, which="both", alpha=.25); ax[0].legend(fontsize=8)
for i, lab in enumerate(STEP_LABELS):
    ys = np.array([avg_cov_diag[n][i] for n in WIDTHS])
    sl = float(np.polyfit(np.log(W), np.log(ys + 1e-30), 1)[0])
    ax[1].loglog(W, ys, "o-", label=f"{lab} ~ n^{sl:+.2f}")
ax[1].set_xlabel("width n"); ax[1].set_ylabel("p-wtd avg diagonal-cov error"); ax[1].set_title("width scaling (robust metric)"); ax[1].grid(True, which="both", alpha=.25); ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout(); plt.show()
print("Primary plot uses the width-robust DIAGONAL metric (avg_cov_diag). The full-Frobenius avg is kept")
print("as avg_cov for reference, but is noise-limited at shallow layers/low-mass bins (see Plot 1).")
""")

nb.save(os.path.join(os.path.dirname(__file__), "error_by_layer_colab.ipynb"))

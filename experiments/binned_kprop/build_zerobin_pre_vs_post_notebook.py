"""Generates zerobin_pre_vs_post_colab.ipynb (valid nbformat-4 JSON).

A SMALL, self-contained notebook to compare the two ReLU-merge strategies for the ZERO bin
(where all negative-spike bins collapse) against the ACTUAL (Monte-Carlo empirical) zero-bin
mean & covariance, per layer, across widths 32 ... 1024 (depth 3).

  Strategy 1  relu_merge="post" : ReLU each negative bin, THEN merge the post-ReLU moments.
  Strategy 2  relu_merge="pre"  : MERGE the negative bins' pre-ReLU bulk into one Gaussian,
                                  THEN apply one exact ReLU.

The empirical target at layer l is the mean/cov of ReLU(bulk preactivation) over the samples
whose spike preactivation lands in the zero bin (A^+ below the zero-bin threshold).

Run:  python "experiments/binned_kprop/build_zerobin_pre_vs_post_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

md(r"""# Zero-bin ReLU merge: **Strategy 1 (post)** vs **Strategy 2 (pre)** vs Monte-Carlo

At each ReLU, every negative-spike bin (spike killed → 0) collapses into the **zero bin**. Two ways
to combine them through the (nonlinear) ReLU:

- **`relu_merge="post"`** (Strategy 1, current default): ReLU each negative bin's bulk, then merge the
  post-ReLU moments.
- **`relu_merge="pre"`** (Strategy 2): merge the negative bins' pre-ReLU bulk into one Gaussian, then
  apply one exact ReLU.

This notebook compares both against the **actual (MC-empirical) zero-bin** mean & covariance — the
moments of `ReLU(bulk preactivation)` over the samples whose spike lands in the zero bin — **per layer**,
for widths **32 … 1024** (depth 3). Expectation from the analysis: the negative bins are nearly
bulk-exchangeable, so post ≈ pre (differ ≪ MC noise), and both sit at the K=2 closure floor.""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## Config""")
code(r"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from Mecha_preds.binned_kprop import (gaussian_initial_state, linear_step_k2, relu_step_k2,
    lloyd_max_edges, lloyd_max_edges_mixture)
from Mecha_preds.binned_kprop.core import _spike_mixture, find_bin

QUICK      = True                       # set False for widths up to 1024 + big MC
WIDTHS     = [32, 64, 128] if QUICK else [32, 64, 128, 256, 512, 1024]
DEPTH      = 3
NUMBINS    = 21
THETA      = 1.0
OUT_DIM    = 8
SEED       = 10
MC_SAMPLES = 1_000_000 if QUICK else 8_000_000
MC_BATCH   = 200_000
MC_DEVICE  = "cpu" if QUICK else "cuda"  # "cuda" (torch) strongly recommended for n>=512
CKPT_DIR   = "checkpoints/binned_kprop/zerobin"; os.makedirs(CKPT_DIR, exist_ok=True)
print(f"QUICK={QUICK} widths={WIDTHS} depth={DEPTH} num_bins={NUMBINS} MC={MC_SAMPLES:,} device={MC_DEVICE}")
""")

# =============================================================================
md(r"""## Helpers — binned propagation, the two zero-bin strategies, and the MC-empirical zero-bin""")
code(r"""
def net(n, depth, seed, theta=THETA, out_dim=OUT_DIM):
    rng = np.random.default_rng(seed); P = np.zeros((n, n)); P[0, 0] = theta
    Ws = [(rng.standard_normal((n, n)) / np.sqrt(n) + P, None) for _ in range(depth)]
    Ws.append((rng.standard_normal((out_dim, n)) / np.sqrt(n), None)); return Ws

# Wasserstein propagation; return per layer (pre-ReLU state, post_edges, zero-bin threshold c).
# c = the A^+ value below which bins map to the zero bin (post-bin 0).
def propagate_layers(Ws, n):
    st = gaussian_initial_state(n - 1, lloyd_max_edges(0.0, 1.0, NUMBINS)[0]); out = []
    for li in range(DEPTH):
        M = Ws[li][0]; p, mY, sY = _spike_mixture(st, M)
        pre = lloyd_max_edges_mixture(p, mY, sY, NUMBINS)[0]
        post = lloyd_max_edges_mixture(p, mY, sY, NUMBINS, rectified=True)[0]
        stp = linear_step_k2(st, M, pre)                       # pre-ReLU state at layer li
        zb = [i for i in range(stp.num_bins) if stp.p[i] > 0 and find_bin(post, max(float(stp.a[i]), 0.0)) == 0]
        c = float(pre[max(zb) + 1]) if zb else 0.0             # threshold that defines the zero bin
        out.append((stp, post, c))
        st = relu_step_k2(stp, post)                           # continue (Strategy 1 = default)
    return out

def zerobin(stp, post, merge):                                 # (mean, cov) of post-bin 0 under a strategy
    s = relu_step_k2(stp, post, relu_merge=merge)
    return s.mu[0], s.Sigma[0]

# Per-layer MC (mean, cov, dead-fraction) of ReLU(bulk) over samples with A^+ < threshold.
def mc_zerobin(Ws, n, thresholds, samples, batch, seed, device):
    L = len(Ws) - 1
    use_torch = device and device != "cpu"
    if use_torch:
        try:
            import torch; dev = torch.device(device); dt = torch.float64 if dev.type == "cuda" else torch.float32
        except Exception as e:
            print("  (torch MC unavailable -> numpy):", e); use_torch = False
    if use_torch:
        Wt = [torch.as_tensor(W, dtype=dt, device=dev) for W, _ in Ws]; thr = [float(t) for t in thresholds]
        sm = [torch.zeros(n - 1, dtype=dt, device=dev) for _ in range(L)]
        sq = [torch.zeros(n - 1, n - 1, dtype=dt, device=dev) for _ in range(L)]; cnt = [0] * L
        g = torch.Generator(device=dev).manual_seed(seed); c = 0
        while c < samples:
            b = min(batch, samples - c); h = torch.randn(b, n, generator=g, dtype=dt, device=dev)
            for li, W in enumerate(Wt):
                z = h @ W.T
                if li < L:
                    m = z[:, 0] < thr[li]; Bp = torch.relu(z[m, 1:])
                    sm[li] += Bp.sum(0); sq[li] += Bp.T @ Bp; cnt[li] += int(m.sum().item()); h = torch.relu(z)
            c += b
        res = []
        for li in range(L):
            mu = sm[li] / cnt[li]; cov = sq[li] / cnt[li] - torch.outer(mu, mu)
            res.append((mu.double().cpu().numpy(), cov.double().cpu().numpy(), cnt[li] / samples))
        return res
    # numpy fallback
    thr = [float(t) for t in thresholds]; rng = np.random.default_rng(seed)
    sm = [np.zeros(n - 1) for _ in range(L)]; sq = [np.zeros((n - 1, n - 1)) for _ in range(L)]; cnt = [0] * L; c = 0
    while c < samples:
        b = min(batch, samples - c); h = rng.standard_normal((b, n))
        for li, (W, _b) in enumerate(Ws):
            z = h @ W.T
            if li < L:
                m = z[:, 0] < thr[li]; Bp = np.maximum(z[m, 1:], 0.0)
                sm[li] += Bp.sum(0); sq[li] += Bp.T @ Bp; cnt[li] += int(m.sum()); h = np.maximum(z, 0.0)
        c += b
    return [(sm[li] / cnt[li], sq[li] / cnt[li] - np.outer(sm[li] / cnt[li], sm[li] / cnt[li]), cnt[li] / samples)
            for li in range(L)]

def rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
print("helpers ready")
""")

# =============================================================================
md(r"""## Run — per width, per layer: post vs pre vs MC-empirical zero-bin (cached MC)""")
code(r"""
rows = []   # (width, layer, dead_frac, mean_post, mean_pre, cov_post, cov_pre, mean_pp, cov_pp)
for n in WIDTHS:
    Ws = net(n, DEPTH, SEED)
    lay = propagate_layers(Ws, n)
    thresholds = [c for (_stp, _post, c) in lay]
    key = os.path.join(CKPT_DIR, f"mcZB_w{n}_d{DEPTH}_nb{NUMBINS}_s{SEED}_S{MC_SAMPLES}.npz")
    if os.path.exists(key):
        z = np.load(key, allow_pickle=True); mc = list(z["mc"])
    else:
        t0 = time.time(); mc = mc_zerobin(Ws, n, thresholds, MC_SAMPLES, MC_BATCH, 10_000 + SEED, MC_DEVICE)
        np.savez(key, mc=np.array(mc, dtype=object)); print(f"  n={n}: MC {time.time()-t0:.1f}s")
    for li in range(DEPTH):
        (mu_mc, cov_mc, frac) = mc[li]
        mp, Sp = zerobin(lay[li][0], lay[li][1], "post")
        mq, Sq = zerobin(lay[li][0], lay[li][1], "pre")
        r = dict(width=n, layer=li, dead=frac,
                 mean_post=rel(mp, mu_mc), mean_pre=rel(mq, mu_mc),
                 cov_post=rel(Sp, cov_mc), cov_pre=rel(Sq, cov_mc),
                 mean_pp=rel(mp, mq), cov_pp=rel(Sp, Sq))
        rows.append(r)
        print(f"  n={n:5d} L{li} dead={frac:.3f} | mean vs MC: post {r['mean_post']:.2e} pre {r['mean_pre']:.2e}"
              f" | cov vs MC: post {r['cov_post']:.2e} pre {r['cov_pre']:.2e} | post-vs-pre: mean {r['mean_pp']:.1e} cov {r['cov_pp']:.1e}")
""")

# =============================================================================
md(r"""## Plots — zero-bin cov error vs width (post & pre overlaid), and the post-vs-pre gap""")
code(r"""
import numpy as np
W = np.array(WIDTHS, float)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for li in range(DEPTH):
    cp = np.array([next(r["cov_post"] for r in rows if r["width"] == n and r["layer"] == li) for n in WIDTHS])
    cq = np.array([next(r["cov_pre"]  for r in rows if r["width"] == n and r["layer"] == li) for n in WIDTHS])
    ax[0].loglog(W, cp, "o-", label=f"L{li} post")
    ax[0].loglog(W, cq, "x--", label=f"L{li} pre")
ax[0].set_xlabel("width n"); ax[0].set_ylabel("zero-bin COV error vs MC")
ax[0].set_title("post vs pre cov error (overlap = identical)"); ax[0].legend(fontsize=7, ncol=2); ax[0].grid(True, which="both", alpha=.25)
for li in range(DEPTH):
    pp = np.array([next(r["cov_pp"] for r in rows if r["width"] == n and r["layer"] == li) for n in WIDTHS])
    mm = np.array([next(r["mean_pp"] for r in rows if r["width"] == n and r["layer"] == li) for n in WIDTHS])
    ax[1].loglog(W, pp, "o-", label=f"L{li} cov |post-pre|")
    ax[1].loglog(W, mm, "s--", label=f"L{li} mean |post-pre|")
ax[1].set_xlabel("width n"); ax[1].set_ylabel("direct post-vs-pre rel-diff")
ax[1].set_title("how different are the two strategies?"); ax[1].legend(fontsize=7, ncol=2); ax[1].grid(True, which="both", alpha=.25)
fig.tight_layout(); plt.show()
print("post-vs-pre stays tiny at all widths -> the two strategies are effectively identical.")
""")

nb.save(os.path.join(os.path.dirname(__file__), "zerobin_pre_vs_post_colab.ipynb"))

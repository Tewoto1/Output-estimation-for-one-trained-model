"""Generates analytic_kprop_colab.ipynb (valid nbformat-4 JSON).

Experiment notebook for the ANALYTIC AFFINE-CONDITIONED K=2 predictor,
``Mecha_preds.analytic_kprop`` (analytic_affine_kprop.pdf, Algorithm 7.2) -- the
discrete-e1-quadrature variant: the spike coordinate is discretized transiently into
``num_nodes`` cells per layer with closed-form probabilities from the KNOWN mixture
scalar law, while the bulk conditional is ONE affine family (mu0 + mu1 y, Sigma0 +
Sigma1 y) rather than one Gaussian per bin.

Sections (knobs live HERE; MC references cached -> nothing recomputed on re-run;
MC cache is SHARED with checkpoints/binned_kprop when the config matches):
  §1  config + spiked-net builder + cached MC (reuses binned_kprop's MC cache)
  §2  sanity: analytic-K2 vs MC on one net (output-mean parity + layer logs)
  §3  error vs num_nodes per width (the knee) + depth-1 pure-quadrature convergence
  §4  scaling: relative MSE vs width per num_nodes, vs binned @ matched budget
  §5  RUNTIME head-to-head vs binned at matched budget (the O(1)-congruence claim)
  §6  per-layer diagnostics: E_m / E_S / tr(R_m) / scalar distortion / zero-atom
      mass; is the affine hypothesis better at larger width?
  §7  node-budget split ablation: adaptive vs symmetric vs single negative cell
  §8  (optional, torch) run on a real model.MLP checkpoint via the adapter

Core (§1-§7) is numpy/scipy and TORCH-FREE; large-width MC and §8 can use torch. Run:
    python "colab_notebooks/analytic_kprop/build_analytic_kprop_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# **Analytic affine** $K=2$ propagation — accuracy, cost, and the affine hypothesis

Predicts $\mathbb E[\,\mathrm{model}(X)\,]$, $X\sim\mathcal N(0,I)$, for a ReLU MLP whose hidden
matrices carry a **coordinate** spike $M = W + e_1 e_1^\top$ — same model class as the binned
companion, but a different compression of the bulk-given-spike law
(*analytic_affine_kprop.pdf*, Algorithm 7.2):

$$C \mid Y = y \;\rightsquigarrow\; \mathcal N\!\big(\mu_0 + \mu_1 y,\; \Sigma_0 + \Sigma_1 y\big),$$

i.e. **one affine family** instead of one bulk Gaussian per spike bin. The spike direction is
discretized only *transiently*: `num_nodes` quadrature cells per layer whose masses and truncated
moments are **closed-form** from the known Gaussian-mixture scalar law (nothing is propagated per
node before ReLU). Consequences to test here:

* **cost** — the per-layer $d^3$ congruence count is $O(1)$ (two aggregated $V\cdot V^\top$), not
  $O(\text{num\_bins})$: §5 times it head-to-head;
* **accuracy** — the only approximations beyond the shared conditional-$K2$ closure are the affine
  re-projection (residuals $\mathcal E_m,\mathcal E_S$, logged per layer) and the 1-D quadrature
  (eq 134, shrinks with `num_nodes`): §3, §4, §6 separate the three;
* **exactness anchors** — layer 1 is exact (no input discretization), cell masses/centroids are
  exact, and the zero atom is merged exactly (see `analytic_kprop/selftest.py`, all machine-precision).""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §1 — Config, the spiked-net builder, and a cached Monte-Carlo reference

All knobs here. Nets are **random untrained** spiked MLPs built deterministically from a seed, so
the expensive recyclable artifact is the **MC reference** — cached to `CKPT_DIR`, and **shared with
the binned notebook's cache** (`checkpoints/binned_kprop`) when depth/width/seed/samples match, so
neither notebook recomputes the other's MC. `QUICK` keeps the sweep small; set `QUICK=False` for
the full one (GPU recommended for MC at $n\ge 512$: `MC_DEVICE="cuda"`).""")
code(r"""
import os, time, math, json
import numpy as np
import matplotlib.pyplot as plt

from Mecha_preds.analytic_kprop import run_analytic_kprop_k2
from Mecha_preds.binned_kprop import run_binned_kprop_k2

# ---------------- knobs (edit here) ----------------
QUICK         = True            # set False for the full sweep (slow at the big widths)
THETA         = 1.0             # coordinate spike: M = W + THETA e1 e1^T
OUT_DIM       = 8               # readout width -> output mean is a VECTOR (stable error norm)
GRID          = "w2"            # cell placement: "w2" (Lloyd-Max on the exact mixture) | "uniform"
BULK_RELU     = "exact"         # per-node Gaussian-ReLU backend (same kernel as binned: fair)
COV_INTERCEPT = "mc"            # Sigma0 intercept: "mc" (moment-conservative, eq 90) | "ls"
MC_DEVICE     = "cpu" if QUICK else "cuda"

SCALE_WIDTHS  = [16, 32, 64, 128] if QUICK else [16, 32, 64, 128, 256, 512, 1024]
SCALE_DEPTH   = 2
SCALE_SEEDS   = [10, 11]
MC_SAMPLES    = 2_000_000 if QUICK else 10_000_000
MC_BATCH      = 200_000

NUMNODES_GRID = [6, 12, 20, 40, 80] if QUICK else [6, 12, 20, 40, 80, 160]
BINNED_BUDGET = 40              # binned num_bins matched to the analytic num_nodes headline
NODES_HEAD    = 40              # headline num_nodes (the "~40 nodes" working point)

CKPT_DIR = "checkpoints/analytic_kprop"; os.makedirs(CKPT_DIR, exist_ok=True)
MC_CACHE_DIRS = [CKPT_DIR, "checkpoints/binned_kprop"]   # shared MC cache (recycle!)
print(f"QUICK={QUICK} | theta={THETA} | out_dim={OUT_DIM} | grid={GRID} | bulk_relu={BULK_RELU} "
      f"| cov_intercept={COV_INTERCEPT} | MC_DEVICE={MC_DEVICE}")
print(f"widths={SCALE_WIDTHS} depth={SCALE_DEPTH} num_nodes grid={NUMNODES_GRID} "
      f"seeds={SCALE_SEEDS} MC={MC_SAMPLES:,}")
""")

code(r"""
# Random ReLU net with a coordinate spike theta*e1 e1^T on every (square) hidden layer.
# IDENTICAL to the binned notebook's builder (same seeds -> same nets -> shared MC cache).
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
    dt = torch.float64 if dev.type == "cuda" else torch.float32
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

# MC mean+se of the output, CACHED -- checks BOTH this predictor's dir and the binned one
# (identical key format), so previously computed references are recycled, never recomputed.
def mc_reference(n, depth, seed, samples, batch, *, theta=THETA, out_dim=OUT_DIM, device=None):
    device = device or MC_DEVICE
    key = f"mc_d{depth}_w{n}_seed{seed}_th{theta:g}_od{out_dim}_s{samples}.npz"
    for cdir in MC_CACHE_DIRS:
        path = os.path.join(cdir, key)
        if os.path.exists(path):
            z = np.load(path); return z["mu"], z["se"]
    Ws = coordinate_spike_net(n, depth, seed, theta=theta, out_dim=out_dim)
    if device and device != "cpu":
        try: mu, se = _mc_torch(Ws, n, samples, batch, 10_000 + seed, device)
        except Exception as e:
            print("  (torch MC unavailable -> numpy):", e); mu, se = _mc_numpy(Ws, n, samples, batch, 10_000 + seed)
    else:
        mu, se = _mc_numpy(Ws, n, samples, batch, 10_000 + seed)
    np.savez(os.path.join(CKPT_DIR, key), mu=mu, se=se); return mu, se

def relerr(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
def slope(xs, ys): return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])

# Resumable scalar-result cache (one JSON per point under CKPT_DIR/pts): a disconnect
# RESUMES the sweep instead of recomputing it.
def cached_scalar(tag, fn):
    path = os.path.join(CKPT_DIR, "pts", tag + ".json"); os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return json.load(open(path))["v"]
    v = float(fn()); json.dump({"v": v}, open(path, "w")); return v

def analytic_err(n, s, mc, nodes, *, depth=None):
    depth = SCALE_DEPTH if depth is None else depth
    return cached_scalar(f"ana_w{n}_s{s}_nn{nodes}_{GRID}_{COV_INTERCEPT}_d{depth}",
        lambda: relerr(run_analytic_kprop_k2(coordinate_spike_net(n, depth, s), n,
                       num_nodes=nodes, grid=GRID, bulk_relu=BULK_RELU,
                       cov_intercept=COV_INTERCEPT)["mean"], mc))

def binned_err(n, s, mc, nbins, *, depth=None):
    depth = SCALE_DEPTH if depth is None else depth
    return cached_scalar(f"bin_w{n}_s{s}_nb{nbins}_d{depth}",
        lambda: relerr(run_binned_kprop_k2(coordinate_spike_net(n, depth, s), n,
                       num_bins=nbins, bulk_relu=BULK_RELU)["mean"], mc))

print("helpers ready (coordinate_spike_net, mc_reference[shared cache], cached_scalar[resumable], "
      "analytic_err, binned_err, relerr, slope)")
""")

# =============================================================================
md(r"""## §2 — Sanity: analytic-$K2$ vs Monte-Carlo on one net

Predict the output-mean vector at the headline `num_nodes=40` and overlay on the MC reference —
points on the diagonal; the printout shows the per-layer accounting the paper asks to log
(checklist 9): affine mean residual $\mathcal E_m$, covariance residual $\mathcal E_S$,
$\mathrm{tr}\,\mathcal R_m$, scalar distortion (eq 134), PSD clip, zero-atom mass.""")
code(r"""
n0, depth0, seed0 = 64, 2, 10
Ws0 = coordinate_spike_net(n0, depth0, seed0)
mc0, se0 = mc_reference(n0, depth0, seed0, MC_SAMPLES, MC_BATCH)
res0 = run_analytic_kprop_k2(Ws0, n0, num_nodes=NODES_HEAD, grid=GRID, bulk_relu=BULK_RELU,
                             cov_intercept=COV_INTERCEPT, diagnostics=True, collect=True)
pred0 = res0["mean"]
print(f"n={n0} depth={depth0}: rel-err {relerr(pred0, mc0):.3e}   "
      f"MC-z {np.linalg.norm(pred0-mc0)/(np.linalg.norm(se0)+1e-30):.1f}   "
      f"(MC rel-noise {np.linalg.norm(se0)/np.linalg.norm(mc0):.1e})")
st = res0["stats"]
print("  layer   E_m       E_S       tr(R_m)   distortion  psd-clip  0-atom  pos-nodes")
for L in range(len(st["E_m"])):
    print(f"    {L}   {st['E_m'][L]:.2e}  {st['E_S'][L]:.2e}  {st['tr_R_m'][L]:.2e}  "
          f"{st['scalar_distortion'][L]:.2e}   {st['psd_clipped'][L]:.1e}  "
          f"{st['zero_atom_mass'][L]:.3f}   {st['num_pos_nodes'][L]}")
plt.figure(figsize=(4.2, 4.2)); lim = max(np.abs(mc0).max(), np.abs(pred0).max()) * 1.1
plt.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=.6); plt.scatter(mc0, pred0, s=40, alpha=.8)
plt.xlabel("Monte-Carlo  E[output]"); plt.ylabel("analytic-K2  E[output]")
plt.title(f"parity (n={n0}, depth={depth0}, {NODES_HEAD} nodes)"); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §3 — Error vs `num_nodes` (the knee), and pure quadrature at depth 1

**Left:** rel-err vs `num_nodes` per width, depth 2. At these widths the error is dominated by the
shared closure + affine-projection floor, so the curve should go flat past a small knee — i.e. **a
handful of nodes buys the full accuracy** (that is the point of not tracking per-node bulk state).
**Right:** at depth 1 the affine state is *exact* and $\mathbb E[\mathrm{out}]$ has a closed form
($Z\sim\mathcal N(0,MM^\top)$, $\mathbb E[\mathrm{ReLU}(Z_i)]=\sigma_i/\sqrt{2\pi}$), so the error
there is **pure scalar quadrature** — it must fall as a power of `num_nodes` with no floor.""")
code(r"""
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
for n in SCALE_WIDTHS:
    errs = []
    for nn in NUMNODES_GRID:
        mcs = {s: mc_reference(n, SCALE_DEPTH, s, MC_SAMPLES, MC_BATCH)[0] for s in SCALE_SEEDS}
        e = float(np.mean([analytic_err(n, s, mcs[s], nn) for s in SCALE_SEEDS]))
        errs.append(e); print(f"[Progress] n={n:5d} nodes={nn:4d} -> {e:.3e}", flush=True)
    knee = next((nn for nn, e in zip(NUMNODES_GRID, errs) if e <= 1.2 * min(errs)), NUMNODES_GRID[-1])
    print(f"  n={n:5d}: knee ~ {knee} nodes (err {min(errs):.2e})")
    axL.plot(NUMNODES_GRID, errs, "o-", label=f"n={n}")
axL.set_xlabel("num_nodes"); axL.set_ylabel("rel error vs MC"); axL.set_yscale("log"); axL.set_xscale("log")
axL.set_title(f"depth {SCALE_DEPTH}: error vs node budget (floor = closure)"); axL.legend(fontsize=8)

n1 = 64
Ws1 = coordinate_spike_net(n1, 1, seed=3)
sig1 = np.sqrt(np.einsum("ij,ij->i", Ws1[0][0], Ws1[0][0]))
exact1 = Ws1[1][0] @ (sig1 / np.sqrt(2 * np.pi))
nodes1 = [4, 8, 16, 32, 64, 128, 256]
q1 = [relerr(run_analytic_kprop_k2(Ws1, n1, num_nodes=nn, grid=GRID)["mean"], exact1) for nn in nodes1]
axR.loglog(nodes1, q1, "o-", label="depth-1 err vs closed form")
axR.loglog(nodes1, q1[1] * (np.array(nodes1) / nodes1[1]) ** -2.0, "k:", alpha=.6, label=r"$\mathrm{nodes}^{-2}$ guide")
axR.set_xlabel("num_nodes"); axR.set_ylabel("rel error"); axR.set_title("depth 1: pure quadrature (no floor)")
axR.legend(fontsize=8); plt.tight_layout(); plt.show()
print(f"depth-1 quadrature slope: nodes^{slope(nodes1, q1):+.2f}")
""")

# =============================================================================
md(r"""## §4 — Scaling: relative MSE vs width, per `num_nodes`, vs binned @ matched budget

The $K=2$ budget law says relative **MSE $\sim n^{-2}$**. Each `num_nodes` gets its own scaling
curve; the **binned companion at `num_bins=40`** is the baseline (same bulk-ReLU kernel, same nets,
same MC). Points below $4\times$ the MC-noise floor are excluded from slope fits.""")
code(r"""
w = np.array(SCALE_WIDTHS, float)
mc_by = {}; noise = []
for n in SCALE_WIDTHS:
    nz = []
    for s in SCALE_SEEDS:
        mc, se = mc_reference(n, SCALE_DEPTH, s, MC_SAMPLES, MC_BATCH); mc_by[(n, s)] = mc
        nz.append(np.linalg.norm(se) / (np.linalg.norm(mc) + 1e-30))
    noise.append(np.mean(nz) ** 2)
noise = np.array(noise)
mse = {nn: np.array([np.mean([analytic_err(n, s, mc_by[(n, s)], nn) for s in SCALE_SEEDS]) ** 2
                     for n in SCALE_WIDTHS]) for nn in NUMNODES_GRID}
mse_b = np.array([np.mean([binned_err(n, s, mc_by[(n, s)], BINNED_BUDGET) for s in SCALE_SEEDS]) ** 2
                  for n in SCALE_WIDTHS])
def _sl(m):
    r = m > 4 * noise
    return slope(w[r], m[r]) if r.sum() >= 2 else float("nan")
print("   n     " + "  ".join(f"nn={nn:<5d}" for nn in NUMNODES_GRID) + f"  binned@{BINNED_BUDGET}  MC-noise")
for i, n in enumerate(SCALE_WIDTHS):
    row = "  ".join(f"{mse[nn][i]:.2e}" for nn in NUMNODES_GRID)
    print(f"  {int(n):5d}  {row}   {mse_b[i]:.2e}   {noise[i]:.1e}")
print("  slopes: " + "  ".join(f"nn{nn}:n^{_sl(mse[nn]):+.2f}" for nn in NUMNODES_GRID) +
      f"  | binned:n^{_sl(mse_b):+.2f}   (K=2 target n^-2)")
plt.figure(figsize=(6.6, 4.3))
for nn in NUMNODES_GRID:
    plt.loglog(w, mse[nn], "o-", label=f"{nn} nodes ~ n^{_sl(mse[nn]):+.2f}")
plt.loglog(w, mse_b, "s--", color="k", label=f"binned@{BINNED_BUDGET} ~ n^{_sl(mse_b):+.2f}")
plt.loglog(w, noise, ":", c="gray", label="MC-noise floor")
plt.loglog(w, mse[NUMNODES_GRID[-1]][0] * (w / w[0]) ** -2.0, "k:", alpha=.5, label="$n^{-2}$ (K=2)")
plt.xlabel("width n"); plt.ylabel("relative MSE vs MC")
plt.title(f"scaling per num_nodes vs binned (depth={SCALE_DEPTH})")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §5 — Runtime head-to-head at matched budget (the $O(1)$-congruence claim)

Per layer the analytic method does **two** aggregated $V\!\cdot\!V^\top$ congruences + $O(mJ)$
closed-form scalars, vs the binned method's **per-bin** congruences ($O(\text{num\_bins})\,d^3$) +
the $O(\text{bins}^2)$ transition kernel. Same exact bivariate ReLU kernel in both (that cost,
$O(\text{nodes}\,d^2)$ special functions, is shared).

Expectation for the wall-clock at matched budget 40: at **small widths binned wins** — the analytic
method pays a fixed scalar overhead per layer (the Lloyd-Max mixture grid + $m\times J$
truncated-normal sweeps) that dwarfs a $63^3$ congruence; the $O(1)$-congruence advantage takes over
once $d^3$ work dominates (CPU crossover $\approx n\gtrsim 200$; run `QUICK=False` to see it, and
§3 says you can also just drop to `num_nodes≈10` at no accuracy cost). Binned runs with its default
thread parallelism; analytic is single-thread numpy — the comparison is conservative.""")
code(r"""
tw = SCALE_WIDTHS if QUICK else [64, 128, 256, 512, 1024]
rows = []
for n in tw:
    Ws = coordinate_spike_net(n, SCALE_DEPTH, SCALE_SEEDS[0])
    t0 = time.time(); ra = run_analytic_kprop_k2(Ws, n, num_nodes=BINNED_BUDGET, grid=GRID,
                                                 bulk_relu=BULK_RELU); tA = time.time() - t0
    t0 = time.time(); rb = run_binned_kprop_k2(Ws, n, num_bins=BINNED_BUDGET,
                                               bulk_relu=BULK_RELU); tB = time.time() - t0
    mc = mc_reference(n, SCALE_DEPTH, SCALE_SEEDS[0], MC_SAMPLES, MC_BATCH)[0]
    rows.append((n, tA, tB, relerr(ra["mean"], mc), relerr(rb["mean"], mc)))
    print(f"[Progress] n={n:5d}  analytic {tA:7.2f}s  binned {tB:7.2f}s  "
          f"(err {rows[-1][3]:.2e} / {rows[-1][4]:.2e})", flush=True)
print("\n   n     analytic[s]  binned[s]  speedup   err(analytic)  err(binned)")
for n, tA, tB, eA, eB in rows:
    print(f"  {n:5d}   {tA:8.2f}   {tB:8.2f}   {tB/max(tA,1e-9):5.1f}x    {eA:.2e}     {eB:.2e}")
plt.figure(figsize=(5.6, 3.8))
plt.loglog([r[0] for r in rows], [r[1] for r in rows], "o-", label="analytic @40 nodes")
plt.loglog([r[0] for r in rows], [r[2] for r in rows], "s--", label="binned @40 bins")
plt.xlabel("width n"); plt.ylabel("predict wall-time [s]")
plt.title(f"runtime at matched budget (depth={SCALE_DEPTH})"); plt.legend(); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §6 — Per-layer diagnostics: how good is the affine hypothesis?

The paper's error budget (thm 10.1) says: beyond the shared $K2$ closure, this method adds ONLY the
affine re-projection ($\mathcal E_m,\mathcal E_S$) and scalar quadrature (distortion). The binless
structure test found the per-bin mean **is** ~linear in the spike early, less cleanly later — here
that is directly $\mathcal E_m$ by layer. We normalize $\mathcal E_m$ by the bulk second moment
$\mathrm{tr}(\Sigma_0+\bar y\Sigma_1)$ and $\mathcal E_S$ by its square so widths are comparable;
falling curves with width = the affine compression gets *more* valid as $n$ grows.""")
code(r"""
diag_w = SCALE_WIDTHS
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for n in diag_w:
    res = run_analytic_kprop_k2(coordinate_spike_net(n, 3, SCALE_SEEDS[0]), n,
                                num_nodes=NODES_HEAD, grid=GRID, bulk_relu=BULK_RELU,
                                cov_intercept=COV_INTERCEPT, diagnostics=True, collect=True)
    st = res["stats"]; L = np.arange(len(st["E_m"]))
    norm = [float(np.trace(a.Sigma0 + (a.w @ a.y) * a.Sigma1)) for a in res["affine_by_layer"]]
    axes[0].semilogy(L, np.array(st["E_m"]) / np.array(norm), "o-", label=f"n={n}")
    axes[1].semilogy(L, np.array(st["E_S"]) / np.array(norm) ** 2, "o-", label=f"n={n}")
    axes[2].semilogy(L, np.array(st["scalar_distortion"]), "o-", label=f"n={n}")
    print(f"n={n:5d}: 0-atom mass by layer {[round(x,3) for x in st['zero_atom_mass']]}   "
          f"pos-nodes {st['num_pos_nodes']}   psd-clip {sum(st['psd_clipped']):.1e}")
for ax, t in zip(axes, ["E_m / tr(Sigma(ybar))  [affine mean residual]",
                        "E_S / tr(Sigma(ybar))^2  [affine cov residual]",
                        "scalar distortion (eq 134)"]):
    ax.set_xlabel("layer"); ax.set_title(t); ax.legend(fontsize=7)
plt.suptitle("affine-residual + quadrature diagnostics (depth 3)", y=1.02); plt.tight_layout(); plt.show()
print("-> E_m/E_S falling with width = the affine (mean, cov) compression improves as n grows; "
      "distortion is the num_nodes knob, independent of the closure.")
""")

# =============================================================================
md(r"""## §7 — Node-budget split ablation: where should the cells go?

The default splits `num_nodes` across the sign **proportionally to mixture mass** (all negative
cells merge into the zero atom at ReLU, but they still quadrature the atom's bulk moments, eqs
40–42). Alternatives: symmetric half/half, and a single negative catch-cell (the binned notebook's
§8 found one catch-bin ≈ enough *there*). Matched total budget.""")
code(r"""
NN = 16
print(f"  rel-err vs MC (num_nodes={NN}, depth {SCALE_DEPTH}):    n:  " +
      "  ".join(f"{n:>6d}" for n in SCALE_WIDTHS))
for label, kw in [("adaptive (default)", {}),
                  ("symmetric", dict(num_nodes_neg=NN // 2)),
                  ("neg=1 catch-cell", dict(num_nodes_neg=1)),
                  ("neg=3/4", dict(num_nodes_neg=3 * NN // 4))]:
    row = []
    for n in SCALE_WIDTHS:
        errs = [relerr(run_analytic_kprop_k2(coordinate_spike_net(n, SCALE_DEPTH, s), n,
                                             num_nodes=NN, grid=GRID, bulk_relu=BULK_RELU, **kw)["mean"],
                       mc_reference(n, SCALE_DEPTH, s, MC_SAMPLES, MC_BATCH)[0])
                for s in SCALE_SEEDS]
        row.append(np.mean(errs))
    print(f"    {label:18s}  " + "  ".join(f"{x:.0e}" for x in row))
print("-> if all splits tie, the floor is the closure, not scalar resolution (expected past the §3 knee).")
""")

# =============================================================================
md(r"""## §8 — (optional, needs torch) run on a real `model.MLP` checkpoint

`add_spike=True` injects $\theta e_1e_1^\top$ into each square hidden layer when the spike isn't
already baked into the weights. Skips cleanly if torch / the checkpoint isn't available.""")
code(r"""
try:
    from model import MLP
    from Mecha_preds.analytic_kprop import run_analytic_kprop
    CKPT = ""   # e.g. "checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt"
    if CKPT and os.path.exists(CKPT):
        model, _ = MLP.load(CKPT)
        out = run_analytic_kprop(model, config={"num_nodes": 40}, add_spike=False)
        print("config:", out["metadata"]["config"]); print("E[output][:8]:", np.asarray(out["mean"]).ravel()[:8])
    else:
        print("set CKPT to a coordinate-spike checkpoint to run the adapter path.")
except ModuleNotFoundError as e:
    print("torch / model unavailable here -> skipping:", e)
""")

nb.save(os.path.join(os.path.dirname(__file__), "analytic_kprop_colab.ipynb"))

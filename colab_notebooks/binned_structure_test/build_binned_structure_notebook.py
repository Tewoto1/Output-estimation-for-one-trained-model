"""Generates binned_structure_test_colab.ipynb -- PER-STEP ERROR ATTRIBUTION.

Question: where does binned-kprop's error come from? Run the REAL predictor (linear_step_k2 +
relu_step_k2, wasserstein grid) on M = W + e1 e1^T (e1e1^T shift of a random N(0,1/n) matrix), and at
EVERY hidden layer compare its prediction to Monte-Carlo:

    mean error   ||E_pred[h^l] - E_MC[h^l]||^2  (per-coord MSE)   -- and the final output E[model(X)]
    prob  error  TV(p_pred, p_MC)  on the pre-activation spike bins  -- the REBINNING transition error

The hypothesis: the Gaussian rebinning corrupts the bin PROBABILITIES, and that is what breaks the
otherwise ~n^-2 mean scaling. So we plot both vs width n at every step and look for co-movement / lag
(prob error scaling badly at a layer -> mean error scaling badly at/after it). And we GRID-SEARCH the
bin count num_bins in {21,42,63} (not just 21) to see whether more bins fixes it.

Run:  python "colab_notebooks/binned_structure_test/build_binned_structure_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

md(r"""# Where does the error come from? **Per-step error attribution** (num_bins grid)

`M = W + e1 e1^T` (e1e1^T shift of a random `N(0,1/n)` matrix, no training), `X ~ N(0,I)`, ReLU. We run
the **actual** binned-kprop predictor (`linear_step_k2` + `relu_step_k2`, wasserstein grid) and at
**every hidden layer** compare to Monte-Carlo:

- **mean error** `‖E_pred[h^ℓ] − E_MC[h^ℓ]‖²` (per-coord MSE), and the **final output** `E[model(X)]` MSE;
- **probability error** `TV(p_pred, p_MC)` on the **pre-activation spike bins** — i.e. the error the
  Gaussian **rebinning** transition makes in the bin masses.

Idea: mean MSE "should" fall like `~n^-2`; if the rebinning corrupts the bin probabilities, that error
should track the mean error (same layer or one layer later) and spoil the scaling. We sweep width `n`
**and grid-search `num_bins ∈ {21,42,63}`** to see where `n^-2` breaks and whether more bins fixes it.
The MC noise floor (`tr Cov/(n·N)`) is drawn so real error is distinguishable from sampling noise.""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## Config""")
code(r"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
import experiments as E
from Mecha_preds.binned_kprop import (build_spiked_net, gaussian_initial_state, linear_step_k2,
                                      relu_step_k2, unconditional_mean, lloyd_max_edges,
                                      lloyd_max_edges_mixture)
from Mecha_preds.binned_kprop.core import _spike_mixture

QUICK = E.QUICK
def _envlist(name, default):
    v = os.environ.get(name, "")
    return [int(x) for x in v.split(",") if x.strip()] or default

WIDTHS      = _envlist("EA_WIDTHS", [24, 48, 96] if QUICK else [32, 64, 128, 256, 512])
NUMBINS_GRID = _envlist("EA_BINS", [7, 14, 21] if QUICK else [21, 42, 63])
DEPTH       = int(os.environ.get("EA_DEPTH", 3 if QUICK else 4))
SEEDS       = _envlist("EA_SEEDS", [1, 2])
THETA, OUT_DIM = 1.0, 8
N_SAMPLES   = int(os.environ.get("EA_SAMPLES", 400_000 if QUICK else 6_000_000))
BATCH       = int(os.environ.get("EA_BATCH", 100_000 if QUICK else 300_000))
CKPT_DIR    = "checkpoints/binned_kprop/error_attribution"; os.makedirs(CKPT_DIR, exist_ok=True)

try:
    import torch; MC_BACKEND = "torch" if torch.cuda.is_available() else "numpy"
except Exception:
    MC_BACKEND = "numpy"
print(f"QUICK={QUICK} widths={WIDTHS} num_bins_grid={NUMBINS_GRID} depth={DEPTH} seeds={SEEDS} "
      f"MC={N_SAMPLES:,} mc_backend={MC_BACKEND}")
""")

# =============================================================================
md(r"""## Helpers — run the predictor (collect per-layer states+edges), MC ground truth, compare""")
code(r"""
def propagate(Ws, n, num_bins, depth):
    # Real binned-kprop propagation; collect per hidden layer the pre-ReLU state (post-rebinning),
    # the post-ReLU state, and the bin edges used. Wasserstein (lloyd-max) grid, like run_binned_kprop_k2.
    st = gaussian_initial_state(n - 1, lloyd_max_edges(0.0, 1.0, num_bins)[0])
    layers = []
    for li in range(depth):
        M = Ws[li][0]
        p, mY, sY = _spike_mixture(st, M)
        pre_edges = lloyd_max_edges_mixture(p, mY, sY, num_bins)[0]
        post_edges = lloyd_max_edges_mixture(p, mY, sY, num_bins, rectified=True)[0]
        st_pre = linear_step_k2(st, M, pre_edges)          # rebinning happens here -> st_pre.p
        st_post = relu_step_k2(st_pre, post_edges)
        layers.append(dict(pre=st_pre, post=st_post, pre_edges=pre_edges))
        st = st_post
    return layers

def mc_stats(Ws, n, pre_edges_by_layer, depth, num_bins, N, batch, seed, backend):
    # MC over X~N(0,I): per hidden layer -> E[h^l] (post-act, (n,)), post-act 2nd moment (for noise
    # floor), and pre-activation spike bin counts (with the predictor's pre_edges). Plus E[output].
    Wh = [Ws[li][0] for li in range(depth)]; Wout = Ws[depth][0]
    use_torch = backend == "torch"
    if use_torch:
        import torch
        dev = torch.device("cuda"); dt = torch.float32
        Wt = [torch.as_tensor(W, dtype=dt, device=dev) for W in Wh]; Wo = torch.as_tensor(Wout, dtype=dt, device=dev)
        ed = [torch.as_tensor(e, dtype=dt, device=dev).contiguous() for e in pre_edges_by_layer]
        sm = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(depth)]
        sq = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(depth)]
        pc = [torch.zeros(num_bins, dtype=torch.float64, device=dev) for _ in range(depth)]
        so = torch.zeros(Wout.shape[0], dtype=torch.float64, device=dev); cnt = 0
        g = torch.Generator(device=dev).manual_seed(seed); got = 0
        while got < N:
            b = min(batch, N - got); h = torch.randn(b, n, generator=g, dtype=dt, device=dev)
            for li in range(depth):
                z = h @ Wt[li].T
                bi = torch.clamp(torch.searchsorted(ed[li], z[:, 0].contiguous(), right=True) - 1, 0, num_bins - 1)
                pc[li].index_add_(0, bi, torch.ones(b, dtype=torch.float64, device=dev))
                h = torch.relu(z); sm[li] += h.sum(0).double(); sq[li] += (h * h).sum(0).double()
            so += (h @ Wo.T).sum(0).double(); cnt += b; got += b
        return (dict(mean=[ (sm[l]/cnt).cpu().numpy() for l in range(depth)],
                     m2=[ (sq[l]/cnt).cpu().numpy() for l in range(depth)],
                     p=[ (pc[l]/cnt).cpu().numpy() for l in range(depth)],
                     out=(so/cnt).cpu().numpy()))
    # numpy
    ed = [e for e in pre_edges_by_layer]
    sm = [np.zeros(n) for _ in range(depth)]; sq = [np.zeros(n) for _ in range(depth)]
    pc = [np.zeros(num_bins) for _ in range(depth)]; so = np.zeros(Wout.shape[0]); cnt = 0
    rng = np.random.default_rng(seed); got = 0
    while got < N:
        b = min(batch, N - got); h = rng.standard_normal((b, n))
        for li in range(depth):
            z = h @ Wh[li].T
            bi = np.clip(np.searchsorted(ed[li], z[:, 0], side="right") - 1, 0, num_bins - 1)
            np.add.at(pc[li], bi, 1.0); h = np.maximum(z, 0.0); sm[li] += h.sum(0); sq[li] += (h * h).sum(0)
        so += (h @ Wout.T).sum(0); cnt += b; got += b
    return dict(mean=[sm[l]/cnt for l in range(depth)], m2=[sq[l]/cnt for l in range(depth)],
                p=[pc[l]/cnt for l in range(depth)], out=so/cnt)

def attribute(Ws, n, num_bins, depth, N, batch, seed, backend):
    layers = propagate(Ws, n, num_bins, depth)
    mc = mc_stats(Ws, n, [L["pre_edges"] for L in layers], depth, num_bins, N, batch, 10_000 + seed, backend)
    Wout = Ws[depth][0]
    mean_mse = np.zeros(depth); prob_err = np.zeros(depth); noise = np.zeros(depth)
    for li in range(depth):
        pred_mean = unconditional_mean(layers[li]["post"])          # E_pred[h^l]  (n,)
        mc_mean = mc["mean"][li]
        mean_mse[li] = float(np.mean((pred_mean - mc_mean) ** 2))
        var = np.clip(mc["m2"][li] - mc_mean ** 2, 0, None)
        noise[li] = float(var.sum() / (n * N))                      # per-coord MC noise floor
        pred_p = np.asarray(layers[li]["pre"].p, float); mcp = mc["p"][li]
        prob_err[li] = 0.5 * float(np.abs(pred_p - mcp).sum())      # TV on pre-activation spike bins
    pred_out = Wout @ unconditional_mean(layers[-1]["post"])
    out_mse = float(np.mean((pred_out - mc["out"]) ** 2))
    return dict(mean_mse=mean_mse, prob_err=prob_err, noise=noise, out_mse=out_mse)
print("helpers ready")
""")

# =============================================================================
md(r"""## Run — sweep width × num_bins × seed (cached)""")
code(r"""
res = {}   # (n, num_bins, seed) -> dict(mean_mse[L], prob_err[L], noise[L], out_mse)
for nb in NUMBINS_GRID:
    for n in WIDTHS:
        for sd in SEEDS:
            key = os.path.join(CKPT_DIR, f"ea_d{DEPTH}_w{n}_nb{nb}_s{sd}_S{N_SAMPLES}.npz")
            if os.path.exists(key):
                z = np.load(key); res[(n, nb, sd)] = {k: z[k] for k in ("mean_mse", "prob_err", "noise", "out_mse")}
                continue
            t0 = time.time()
            Ws = build_spiked_net(n, DEPTH, seed=sd, theta=THETA, out_dim=OUT_DIM)
            r = attribute(Ws, n, nb, DEPTH, N_SAMPLES, BATCH, sd, MC_BACKEND)
            np.savez(key, **r); res[(n, nb, sd)] = r
            print(f"  nb{nb} n{n} s{sd}: out_mse={r['out_mse']:.2e}  ({time.time()-t0:.1f}s)")
print("done")

def avg(n, nb, field):   # seed-averaged
    xs = [res[(n, nb, sd)][field] for sd in SEEDS]
    return np.mean(xs, axis=0)
def slope(xs, ys):
    m = np.array(ys) > 0
    return np.polyfit(np.log(np.array(xs)[m]), np.log(np.array(ys)[m]), 1)[0] if m.sum() >= 2 else np.nan
""")

# =============================================================================
md(r"""## Exponents — does output MSE ~ n^-2, and where does the per-layer mean error break it?""")
code(r"""
print("output MSE ~ n^alpha   (want alpha ~ -2):")
for nb in NUMBINS_GRID:
    om = [float(avg(n, nb, "out_mse")) for n in WIDTHS]
    print(f"  num_bins={nb:>3}: " + " ".join(f"{v:.2e}" for v in om) + f"   alpha={slope(WIDTHS, om):+.2f}")

print("\nper-layer MEAN-MSE exponent alpha (n^alpha) [rows=num_bins, cols=layer]:")
for nb in NUMBINS_GRID:
    al = [slope(WIDTHS, [float(avg(n, nb, "mean_mse")[l]) for n in WIDTHS]) for l in range(DEPTH)]
    print(f"  nb={nb:>3}: " + " ".join(f"L{l}:{al[l]:+.2f}" for l in range(DEPTH)))

print("\nper-layer PROB-ERROR exponent alpha (n^alpha) [rows=num_bins, cols=layer]:")
for nb in NUMBINS_GRID:
    al = [slope(WIDTHS, [float(avg(n, nb, "prob_err")[l]) for n in WIDTHS]) for l in range(DEPTH)]
    print(f"  nb={nb:>3}: " + " ".join(f"L{l}:{al[l]:+.2f}" for l in range(DEPTH)))
print("\n(mean-MSE flattening (alpha -> 0) at some layer = where the clean scaling breaks; compare to the")
print(" layer/width where prob-error stops improving.)")
""")

# =============================================================================
md(r"""## Plots""")
code(r"""
Wn = np.array(WIDTHS, float); nb_ref = NUMBINS_GRID[-1]
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
# (0,0) output MSE vs n, per num_bins, with n^-2 ref
for nb in NUMBINS_GRID:
    om = np.array([float(avg(n, nb, "out_mse")) for n in WIDTHS]); ax[0,0].loglog(Wn, om, "o-", label=f"nb={nb}")
ax[0,0].loglog(Wn, om[0]*(Wn/Wn[0])**-2, "k:", alpha=.6, label="n^-2")
ax[0,0].set_title("final output MSE vs n"); ax[0,0].set_xlabel("width n"); ax[0,0].legend(fontsize=7); ax[0,0].grid(True,which="both",alpha=.25)
# (0,1) per-layer mean MSE vs n (num_bins=nb_ref) + noise floor + n^-2
for l in range(DEPTH):
    mm = np.array([float(avg(n, nb_ref, "mean_mse")[l]) for n in WIDTHS])
    nf = np.array([float(avg(n, nb_ref, "noise")[l]) for n in WIDTHS])
    ax[0,1].loglog(Wn, mm, "o-", label=f"L{l}"); ax[0,1].loglog(Wn, nf, ":", alpha=.4)
ax[0,1].loglog(Wn, mm[0]*(Wn/Wn[0])**-2, "k--", alpha=.5, label="n^-2")
ax[0,1].set_title(f"per-layer mean MSE vs n (nb={nb_ref}); dotted=MC noise floor"); ax[0,1].set_xlabel("width n"); ax[0,1].legend(fontsize=7); ax[0,1].grid(True,which="both",alpha=.25)
# (0,2) per-layer prob error vs n (num_bins=nb_ref)
for l in range(DEPTH):
    pe = np.array([float(avg(n, nb_ref, "prob_err")[l]) for n in WIDTHS]); ax[0,2].loglog(Wn, pe, "o-", label=f"L{l}")
ax[0,2].set_title(f"per-layer probability error (TV) vs n (nb={nb_ref})"); ax[0,2].set_xlabel("width n"); ax[0,2].legend(fontsize=7); ax[0,2].grid(True,which="both",alpha=.25)
# (1,0) co-movement across layers at the largest n (nb_ref): mean MSE and prob error vs layer
nmax = WIDTHS[-1]; L = np.arange(DEPTH)
axb = ax[1,0].twinx()
ax[1,0].plot(L, avg(nmax, nb_ref, "mean_mse"), "o-", color="C0", label="mean MSE")
axb.plot(L, avg(nmax, nb_ref, "prob_err"), "s--", color="C3", label="prob error")
ax[1,0].set_yscale("log"); ax[1,0].set_xlabel("layer"); ax[1,0].set_ylabel("mean MSE", color="C0"); axb.set_ylabel("prob err (TV)", color="C3")
ax[1,0].set_title(f"co-movement across layers (n={nmax}, nb={nb_ref})")
# (1,1) num_bins effect: output MSE vs num_bins, per n
NB = np.array(NUMBINS_GRID, float)
for n in WIDTHS:
    om = [float(avg(n, nb, "out_mse")) for nb in NUMBINS_GRID]; ax[1,1].loglog(NB, om, "o-", label=f"n={n}")
ax[1,1].set_title("output MSE vs num_bins (grid search)"); ax[1,1].set_xlabel("num_bins"); ax[1,1].legend(fontsize=7); ax[1,1].grid(True,which="both",alpha=.25)
# (1,2) num_bins effect: last-layer mean MSE and prob error vs num_bins (n=largest)
mm = [float(avg(nmax, nb, "mean_mse")[-1]) for nb in NUMBINS_GRID]
pe = [float(avg(nmax, nb, "prob_err")[-1]) for nb in NUMBINS_GRID]
ax[1,2].loglog(NB, mm, "o-", label="mean MSE (last L)"); ax[1,2].loglog(NB, pe, "s-", label="prob err (last L)")
ax[1,2].set_title(f"last-layer error vs num_bins (n={nmax})"); ax[1,2].set_xlabel("num_bins"); ax[1,2].legend(fontsize=7); ax[1,2].grid(True,which="both",alpha=.25)
fig.tight_layout(); plt.show()
""")

nb.save(os.path.join(os.path.dirname(__file__), "binned_structure_test_colab.ipynb"))

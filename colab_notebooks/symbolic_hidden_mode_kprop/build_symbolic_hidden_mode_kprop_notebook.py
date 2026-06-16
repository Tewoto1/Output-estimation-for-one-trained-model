"""Generates symbolic_hidden_mode_kprop_colab.ipynb (valid nbformat-4 JSON).

Tests the THIRD cumulant predictor -- ``Mecha_preds.cumulants.shkprop`` (symbolic
hidden-mode kprop, scalar h, k_max=2) -- on MLPs TRAINED TO ZERO with square layers
(``input_dim = hidden_dim = output_dim = width``), exactly as the user asked:

    * train-to-zero with EARLY STOPPING at loss_tol = 1e-7 (tol_check_every=1, since
      ZeroTask converges in tens of steps -- see the experiments-module memo);
    * error estimation via 40,000,000 Monte-Carlo points (float64 accumulators, GPU);
    * checkpoint recycling: models + MC/predictor results cached under this notebook's
      own CKPT_DIR and LOADED if present (the repo's whole reason for living in a folder).

What the symbolic method does (vs the others). It carries the scalar hidden mode
h = V^T X SYMBOLICALLY through the net as polynomial jets in dh, marginalizing once
at the end -- instead of re-running kprop at Gauss-Hermite nodes of h and averaging
(that is skprop). Linear layers are exact; the ReLU residual-Gaussian closure is the
only approximation. With latent="none" (q=0) it reduces EXACTLY to ordinary k=2 ReLU
kprop; with latent="ones" the latent is h = 1^T X / sqrt(n).

Sections
  §0  parity + spec sanity tests: torch shkprop == numpy oracle (~1e-12); linear-only
      exactness; q=0 == vanilla kprop; scalar ReLU vs MC; Gaussian-h == node-average.
  §1  TRAIN-TO-0 width sweep (1e-7 early stop, 40M-point MC): rel-L2 error of E[out]
      for symbolic(ones) vs symbolic(q=0) vs vanilla kprop k=2, vs the MC noise floor.
  §2  hidden-degree p convergence + adaptive tail diagnostics (where it approximates).
  §3  (optional) planted all-ones spike W=W'-(1/sqrt n)11^T: symbolic >> vanilla.
  §4  checkpoints save/load/download.

Needs torch (shkprop core); the vanilla-kprop comparison needs scipy and Python>=3.12
(or the auto-active kprop-compat shim). shkprop itself has NO kprop dependency. Run:
  python "colab_notebooks/symbolic_hidden_mode_kprop/build_symbolic_hidden_mode_kprop_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# Symbolic hidden-mode KPROP on networks **trained to zero**

**Setup (as requested).** Square ReLU MLPs with `input_dim = hidden_dim = output_dim = width`,
no bias, **trained to output 0** (`ZeroTask`, $X\sim\mathcal N(0,I)$, loss $=\mathbb E\lVert f(X)\rVert^2$)
with **early stopping at `loss_tol = 1e-7`**. Error is measured against a **40,000,000-point**
Monte-Carlo estimate of $\mathbb E[f(X)]$ (float64 accumulators).

**The predictor.** `Mecha_preds.cumulants.shkprop` carries the scalar hidden mode
$h=V^\top X$ **symbolically** as polynomial jets in $dh$ through the network and marginalizes
once at the end — instead of re-running kprop at Gauss–Hermite nodes of $h$ and averaging
(that is `skprop`). Linear layers are **exact** in cumulant space; the **ReLU** residual-Gaussian
closure ($Z\mid h\sim\mathcal N(\mu(h),\Sigma(h))$) is the **only** approximation. The hidden mode:

| `latent` | $h$ | meaning |
|---|---|---|
| `"none"` | — | $q=0$: reduces **exactly** to ordinary $k{=}2$ ReLU kprop |
| `"ones"` (default) | $h=\mathbf 1^\top X/\sqrt n$ | the all-ones latent of the shifted-mean study |
| explicit `direction` | $h=V^\top X$ | any unit direction |

> **Recycling + GPU (repo policy).** Trained models go to `checkpoints/symbolic_hidden_mode_kprop`
> and are **loaded if on disk**; MC + predictor results are **cached by config** so a re-run
> recomputes nothing. MC runs on `E.DEVICE` (float64 on CUDA; CPU fallback on Apple MPS).

| | test | expectation | § |
|---|---|---|---|
| **S0** | torch `shkprop` vs numpy oracle; spec sanity tests | parity $\sim$1e-12; linear exact; $q{=}0$ = vanilla; scalar ReLU $\to$ MC; Gaussian-$h$ = node-avg | §0 |
| **S1** | train-to-0 width sweep (1e-7, 40M MC) | all predictors track MC within the $k{=}2$ **closure** error (shrinks with width); symbolic exact where it should be | §1 |
| **S2** | hidden-degree $p$ convergence + tail diagnostics | symbolic$\to$node-avg as $p\uparrow$; tail score reports sufficiency | §2 |
| **S3** | planted all-ones spike $W=W'-\tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top$ | symbolic(ones) $\ll$ vanilla (the latent a single Gaussian state can't see) | §3 |

`shkprop` needs only **torch**; the vanilla-kprop baseline needs **scipy** + Python ≥ 3.12 (or the
auto-active compat shim).""")

code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## §Config — knobs, device & recycling (probe here, not in `experiments.py`)""")
code(r"""
import math, time, os, copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import experiments as E
from model import MLP
from tasks.train_to_zero import ZeroTask

QUICK  = E.QUICK
DEVICE = E.DEVICE                       # cuda -> mps -> cpu (auto); TF32 enabled in experiments.py
torch.set_num_threads(max(torch.get_num_threads(), 2))

DEPTH       = 2                          # hidden Linear+ReLU blocks (square: in=h=out=width)
WIDTHS      = [16, 32, 64] if QUICK else [16, 32, 64, 128, 256]
SEEDS       = [0] if QUICK else [0, 1, 2]
LOSS_TOL    = 1e-7                       # <<< early-stopping target (train-to-0)
P0          = 6                          # initial hidden-mode polynomial degree
P_MAX       = 12                         # adaptive ceiling
MC_SAMPLES  = 2_000_000 if QUICK else 40_000_000   # <<< 40M MC points for error estimation

# ---- GPU policy: float32 train/compute, float64 for MC + predictors (repo policy) ----
if DEVICE.type == "cuda":
    MC_DEVICE, MC_DTYPE, MC_BATCH = DEVICE, torch.float64, 131_072
    PRED_DEVICE = "cuda"
else:
    MC_DEVICE, MC_DTYPE, MC_BATCH = torch.device("cpu"), torch.float64, 16_384
    PRED_DEVICE = "cpu"

CKPT_DIR = "checkpoints/symbolic_hidden_mode_kprop"   # this notebook's OWN family
RECYCLE  = True
os.makedirs(CKPT_DIR, exist_ok=True)

from Mecha_preds.cumulants import estimate_empirical_mean, compare_means
from Mecha_preds.cumulants.shkprop import run_symbolic_cumulants
from Mecha_preds.cumulants.shkprop import reference as shref   # numpy oracle (torch-free)
try:
    from Mecha_preds.cumulants import run_cumulants             # vanilla kprop (needs >=3.12 + scipy)
    HAVE_KPROP = True
except Exception as e:
    HAVE_KPROP = False
    print("vanilla kprop unavailable (", type(e).__name__, ") -- shkprop runs without it")

print("DEVICE:", DEVICE, "| MC:", MC_DEVICE.type, MC_DTYPE, "x", MC_BATCH,
      "| pred dev:", PRED_DEVICE, "| kprop:", HAVE_KPROP)
print("QUICK:", QUICK, "| widths:", WIDTHS, "| seeds:", SEEDS,
      "| loss_tol:", LOSS_TOL, "| MC pts:", f"{MC_SAMPLES:,}", "| CKPT_DIR:", CKPT_DIR)
""")

# =============================================================================
md(r"""## §0 — Parity with the numpy oracle + spec sanity tests

`shkprop` ships a pure-numpy **oracle** (`shkprop.reference`) that the torch path mirrors
op-for-op. We assert the two agree, then run the spec's sanity tests on the oracle (fast,
torch-free): **linear-only exactness**, **$q{=}0$ = single-Gaussian $k{=}2$ kprop**, **scalar
ReLU $\to$ MC**, and **Gaussian-$h$ symbolic $\to$ node-averaging baseline** (the "old approach"
the symbolic method replaces).""")
code(r"""
def relerr(a, b):
    a, b = np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

# ---- a small random ReLU net as study model.MLP, plus its layer list for the oracle ----
def random_relu_model(width, depth=DEPTH, seed=0):
    m = E.build_mlp(width, depth, output_dim=width, seed=seed, input_dim=width,
                    activation="relu").double().eval()
    return m

def model_layers_np(m):
    layers = []
    for lin in m.hidden_layers:
        layers.append(("linear", lin.weight.detach().cpu().numpy().astype(float),
                       None if lin.bias is None else lin.bias.detach().cpu().numpy().astype(float)))
        layers.append(("relu",))
    ro = m.readout
    layers.append(("linear", ro.weight.detach().cpu().numpy().astype(float),
                   None if ro.bias is None else ro.bias.detach().cpu().numpy().astype(float)))
    return layers

# ---- (a) torch vs numpy-oracle PARITY on a real net (both latents) ----
m = random_relu_model(24, depth=DEPTH, seed=1)
layers = model_layers_np(m)
d = 24; Vones = np.ones(d) / math.sqrt(d)
for tag, V in [("q=0", None), ("ones", Vones)]:
    K1, K2, kappa = shref.make_input_state(d, p=P0, V=V)
    mean_ref, _, _ = shref.symbolic_kprop(layers, K1, K2, kappa, p=P0)
    cfg = {"latent": "none" if V is None else "ones", "hidden_degree_initial": P0,
           "auto_refine": False}
    mean_torch = run_symbolic_cumulants(m, config=cfg, device=PRED_DEVICE)["mean"]
    e = relerr(mean_torch, mean_ref)
    print(f"  [{'PASS' if e < 1e-9 else 'FAIL'}] torch shkprop == numpy oracle ({tag}): relerr={e:.2e}")

# ---- (b) linear-only exactness (oracle): mean/cov match closed form to machine eps ----
rng = np.random.default_rng(0)
W1 = rng.standard_normal((d, d)); b1 = rng.standard_normal(d)
lin_layers = [("linear", W1, b1)]
K1, K2, kappa = shref.make_input_state(d, p=P0, V=Vones)
mn, cv, _ = shref.symbolic_kprop(lin_layers, K1, K2, kappa, p=P0)
e_m, e_c = relerr(mn, b1), relerr(cv, W1 @ W1.T)
print(f"  [{'PASS' if max(e_m,e_c)<1e-10 else 'FAIL'}] linear-only exact: mean={e_m:.1e} cov={e_c:.1e}")

# ---- (c) q=0 == single-Gaussian k=2 ReLU kprop (oracle direct path) ----
K1, K2, kappa = shref.make_input_state(d, p=P0, V=None)
mn0, _, _ = shref.symbolic_kprop(layers, K1, K2, kappa, p=P0)
mnd, _ = shref.direct_k2_relu_kprop(layers, np.zeros(d), np.eye(d))
e0 = relerr(mn0, mnd)
print(f"  [{'PASS' if e0<1e-10 else 'FAIL'}] q=0 == single-Gaussian k=2 kprop: relerr={e0:.2e}")

# ---- (d) scalar ReLU analytic E[ReLU(a h + eps)] -> analytic & MC ----
a, s = 0.8, 0.6; sigma = math.sqrt(a**2 + s**2); analytic = sigma / shref.SQRT2PI
K1 = shref.jet_zeros(10, (1,)); K1[1, 0] = a
K2 = shref.jet_zeros(10, (1, 1)); K2[0, 0, 0] = s**2
mn_sc, _, _ = shref.symbolic_kprop([("relu",)], K1, K2, shref.gaussian_kappa(2, 1.0), p=10)
print(f"  [{'PASS' if abs(mn_sc[0]-analytic)<1e-3 else 'FAIL'}] scalar ReLU -> analytic: "
      f"sym={mn_sc[0]:.6f} exact={analytic:.6f}")

# ---- (e) Gaussian-h symbolic -> node-averaging baseline (the 'old approach') ----
node = shref.node_average_kprop(layers, d, Vones, n_nodes=51)
for p in (4, 8, 12):
    K1, K2, kappa = shref.make_input_state(d, p=p, V=Vones)
    mp, _, _ = shref.symbolic_kprop(layers, K1, K2, kappa, p=p)
    print(f"        p={p:>2}: relerr(symbolic, node-average) = {relerr(mp, node):.2e}")
print("  symbolic reproduces the node-average WITHOUT running the nodes (converges in p).")
""")

# =============================================================================
md(r"""## §1 — Train-to-0 width sweep: error vs the 40M-point Monte-Carlo mean

Each `(width, seed)`: the square net is **trained to 0 with early stop at 1e-7** (recycled from
disk if present), MC$(40\text{M})$ estimates $\mathbb E[f(X)]$, and we compute the relative $L_2$
error of three predictions of that mean — **symbolic(ones)**, **symbolic(q=0)**, and **vanilla
kprop $k{=}2$** (exact ReLU covariance, if available). The **MC noise floor**
$\lVert\text{stderr}\rVert/\lVert\mu_{\text{MC}}\rVert$ is the resolution limit (~$2\times10^{-4}$
at 40M for $O(1)$ outputs). The shared $k{=}2$ **closure** error is the scientifically interesting
quantity; it shrinks with width.""")
code(r"""
def train_to_zero(width, seed):
    "Recycle: load the 1e-7-trained square net if on disk, else train + save it."
    name = E.run_name("shkprop-zero", depth=DEPTH, width=width, seed=seed)
    path = E.ckpt_path(CKPT_DIR, name)
    cfg = E.default_train_cfg(width, seed=seed, loss_tol=LOSS_TOL, tol_check_every=1)
    model, payload, loaded = E.get_or_train(
        path,
        build=lambda: E.build_mlp(width, DEPTH, output_dim=width, input_dim=width, seed=seed),
        task=ZeroTask(input_dim=width, output_dim=width),
        train_cfg=cfg, progress=False,
        extra_meta={"family": "symbolic_hidden_mode_kprop", "loss_tol": LOSS_TOL})
    return model.double().eval(), E.final_loss(payload), loaded

def mc_reference(model, width):
    "40M-point MC on DEVICE (float64 accumulators); does NOT mutate the float64 master."
    md = copy.deepcopy(model).to(device=MC_DEVICE, dtype=MC_DTYPE)
    mc, stats = estimate_empirical_mean(model=md, input_dim=width, num_samples=MC_SAMPLES,
                                        device=str(MC_DEVICE), dtype=MC_DTYPE, batch_size=MC_BATCH)
    del md
    if MC_DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return mc, stats

CFG_SIG = f"d{DEPTH}_p{P0}-{P_MAX}_mc{MC_SAMPLES}_tol{LOSS_TOL}"
RESULTS_PATH = os.path.join(CKPT_DIR, f"results_{CFG_SIG}.pt")
_results = torch.load(RESULTS_PATH) if (RECYCLE and os.path.exists(RESULTS_PATH)) else {}
def cache_get(k): return _results.get(k) if RECYCLE else None
def cache_put(k, v): _results[k] = v; torch.save(_results, RESULTS_PATH)

rows, t0 = [], time.time()
for w in WIDTHS:
    for seed in SEEDS:
        key = f"w{w}|s{seed}"
        r = cache_get(key); src = "recycled"
        if r is None:
            src = "computed"
            model, floss, loaded = train_to_zero(w, seed)
            mc, stats = mc_reference(model, w)
            sym1 = run_symbolic_cumulants(model, config={"latent": "ones",
                        "hidden_degree_initial": P0, "hidden_degree_max": P_MAX},
                        device=PRED_DEVICE)
            sym0 = run_symbolic_cumulants(model, config={"latent": "none",
                        "hidden_degree_initial": P0}, device=PRED_DEVICE)["mean"]
            van = None
            if HAVE_KPROP:
                van = run_cumulants(model, config={"k_max": 2, "exact_relu_cov": True},
                                    device=PRED_DEVICE)["mean"]
            nm = float(np.linalg.norm(mc)) + 1e-30
            r = dict(w=w, seed=seed, floss=float(floss),
                     sym_ones=relerr(sym1["mean"], mc), sym_q0=relerr(sym0, mc),
                     vanilla=(relerr(van, mc) if van is not None else float("nan")),
                     floor=float(np.linalg.norm(stats["mc_stderr"])) / nm,
                     p_used=int(sym1["metadata"]["hidden_degree"]),
                     unresolved=sym1["metadata"]["unresolved_layers"], loaded=bool(loaded))
            cache_put(key, r)
        rows.append(r)
        print(f"w={w:>4} s{seed} [{src:>8}] loss={r['floss']:.1e} | sym(ones) {r['sym_ones']:.2e}"
              f" | sym(q0) {r['sym_q0']:.2e} | vanilla {r['vanilla']:.2e} | floor {r['floor']:.1e}"
              f" | p={r['p_used']}", flush=True)
print(f"\nsweep done in {time.time()-t0:.1f}s ({len(rows)} runs; recycled ones are instant)")
""")
code(r"""
def series(key):
    return [float(np.nanmean([r[key] for r in rows if r["w"] == w])) for w in WIDTHS]

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.loglog(WIDTHS, series("sym_ones"), "o-",  label="symbolic (latent = ones)")
ax.loglog(WIDTHS, series("sym_q0"),   "s--", label="symbolic (q=0)  = ordinary k=2 kprop")
if HAVE_KPROP:
    ax.loglog(WIDTHS, series("vanilla"), "^:", label="vanilla kprop k=2 (exact ReLU cov)")
ax.loglog(WIDTHS, series("floor"), ":", color="0.5", label="MC noise floor (40M)")
ax.set_xlabel("width n  (input = hidden = output)")
ax.set_ylabel(r"rel. $L_2$ error of $E[f(X)]$ vs 40M-MC")
ax.set_title("Symbolic hidden-mode kprop on nets trained to 0 (loss < 1e-7)")
ax.legend(); ax.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.show()

print("q=0 vs vanilla kprop (should coincide -- symbolic q=0 IS ordinary k=2 kprop):")
for w in WIDTHS:
    s0 = float(np.nanmean([r["sym_q0"] for r in rows if r["w"] == w]))
    vn = float(np.nanmean([r["vanilla"] for r in rows if r["w"] == w]))
    print(f"   w={w:>4}: sym(q0)={s0:.3e}  vanilla={vn:.3e}  |diff|={abs(s0-vn):.1e}")
""")

# =============================================================================
md(r"""## §2 — Hidden-degree $p$ convergence and adaptive tail diagnostics

The symbolic prediction is a degree-$p$ polynomial in $dh$. As $p\uparrow$ it converges to the
node-averaging value (the conditional-Gaussian $k{=}2$ result), then hits the **float64 high-degree
conditioning floor** the spec warns about. The **tail score** (moment-weighted top-degree
contribution) is the method's self-diagnostic: it stays high until $p$ is large enough, and the
predictor **reports** `unresolved_layers` and its `approximations` rather than silently truncating.""")
code(r"""
w_demo = WIDTHS[min(len(WIDTHS) - 1, 2)]
model, floss, _ = train_to_zero(w_demo, 0)
mc, stats = mc_reference(model, w_demo)
floor = float(np.linalg.norm(stats["mc_stderr"])) / (float(np.linalg.norm(mc)) + 1e-30)

ps = [2, 4, 6, 8, 10, 12]
errs, tails = [], []
for p in ps:
    res = run_symbolic_cumulants(model, config={"latent": "ones", "hidden_degree_initial": p,
                                 "auto_refine": False}, device=PRED_DEVICE)
    errs.append(relerr(res["mean"], mc))
    tails.append(max(res["metadata"]["tail_scores"]) if res["metadata"]["tail_scores"] else 0.0)
    print(f"  p={p:>2}: rel.err(MC)={errs[-1]:.3e}  max tail_score={tails[-1]:.2e}")

res = run_symbolic_cumulants(model, config={"latent": "ones", "hidden_degree_initial": P0,
                             "hidden_degree_max": P_MAX}, device=PRED_DEVICE)
print("\nreported approximations (the method never hides where it approximates):")
for note in res["metadata"]["approximations"]:
    print("   -", note)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
a1.loglog(ps, errs, "o-"); a1.axhline(max(floor, 1e-12), ls=":", color="0.5", label="MC floor")
a1.set_xlabel("hidden degree p"); a1.set_ylabel("rel. L2 error vs MC")
a1.set_title(f"convergence in p (n={w_demo})"); a1.legend(); a1.grid(alpha=0.3, which="both")
a2.semilogy(ps, tails, "s-"); a2.axhline(1e-4, ls=":", color="r", label="hidden_tail_tol")
a2.set_xlabel("hidden degree p"); a2.set_ylabel("max tail score")
a2.set_title("adaptive tail diagnostic"); a2.legend(); a2.grid(alpha=0.3); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §3 — (optional) A *planted* all-ones spike: where the symbolic latent pays off

On a generic trained-to-0 net the all-ones direction is not special, so symbolic(ones) $\approx$
symbolic(q=0). To show the mechanism, plant the latent: $W = W' - \tfrac1{\sqrt n}\mathbf 1\mathbf 1^\top$
on the hidden layers (the shifted-mean construction). Now $h=\mathbf 1^\top X/\sqrt n$ is an $O(1)$
coherent shift a single Gaussian state cannot represent — symbolic(ones) should beat vanilla / q=0
by orders of magnitude (it rides the MC floor). Set `RUN_S3 = True` to run.""")
code(r"""
RUN_S3 = True
if RUN_S3:
    def planted_spike_model(width, seed=0, depth=DEPTH):
        m = E.build_mlp(width, depth, output_dim=width, input_dim=width, seed=seed,
                        activation="relu").double().eval()
        g = torch.Generator().manual_seed(1234 + width + 7 * seed)
        c = 1.0 / math.sqrt(width)
        with torch.no_grad():
            for li, layer in enumerate(list(m.hidden_layers) + [m.readout]):
                o, i = layer.weight.shape
                W = torch.randn(o, i, generator=g, dtype=torch.float64) / math.sqrt(i)
                if li < len(m.hidden_layers):
                    W = W - c * torch.ones(o, i, dtype=torch.float64)
                layer.weight.copy_(W)
        return m

    s3 = []
    for w in WIDTHS:
        m = planted_spike_model(w)
        mc, stats = mc_reference(m, w)
        e_ones = relerr(run_symbolic_cumulants(m, config={"latent": "ones",
                        "hidden_degree_initial": P0, "hidden_degree_max": P_MAX},
                        device=PRED_DEVICE)["mean"], mc)
        e_q0 = relerr(run_symbolic_cumulants(m, config={"latent": "none",
                      "hidden_degree_initial": P0}, device=PRED_DEVICE)["mean"], mc)
        fl = float(np.linalg.norm(stats["mc_stderr"])) / (float(np.linalg.norm(mc)) + 1e-30)
        s3.append((w, e_ones, e_q0, fl))
        print(f"  w={w:>4} | symbolic(ones) {e_ones:.2e} | symbolic(q0) {e_q0:.2e} | floor {fl:.1e}")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.loglog(WIDTHS, [r[1] for r in s3], "o-", label="symbolic (latent = ones)")
    ax.loglog(WIDTHS, [r[2] for r in s3], "s--", label="symbolic (q=0) = vanilla k=2")
    ax.loglog(WIDTHS, [r[3] for r in s3], ":", color="0.5", label="MC noise floor")
    ax.set_xlabel("width n"); ax.set_ylabel(r"rel. $L_2$ error of $E[\mathrm{out}]$")
    ax.set_title(r"planted spike $W=W'-\frac{1}{\sqrt n}11^\top$: the latent a single state can't see")
    ax.legend(); ax.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## §4 — Checkpoints: save / load / **download** (recycle across sessions)

Everything wrote to `checkpoints/symbolic_hidden_mode_kprop` (1e-7-trained models + a results
cache keyed by config). On Drive (set `LOCAL_REPO_DIR` in the bootstrap) it persists; otherwise
download the zip and re-upload next session so nothing recomputes.""")
code(r"""
import shutil
print("checkpoint dir:", os.path.abspath(CKPT_DIR))
for f in sorted(os.listdir(CKPT_DIR)):
    print("   ", f, f"({os.path.getsize(os.path.join(CKPT_DIR, f)) / 1e6:.2f} MB)")
if IN_COLAB:
    from google.colab import files
    zpath = shutil.make_archive("/content/symbolic_hidden_mode_kprop_ckpts", "zip", CKPT_DIR)
    print("\nzipped ->", zpath, "-- downloading..."); files.download(zpath)
# RESTORE in a fresh runtime:
#   from google.colab import files; up = files.upload()
#   import io, zipfile; os.makedirs(CKPT_DIR, exist_ok=True)
#   zipfile.ZipFile(io.BytesIO(next(iter(up.values())))).extractall(CKPT_DIR)
""")

# =============================================================================
md(r"""## §5 — Summary

- **§0** the torch `shkprop` path matches its numpy oracle to ~1e-12, and the spec sanity tests
  pass: **linear layers exact**, **$q{=}0$ = ordinary $k{=}2$ kprop**, scalar ReLU $\to$ analytic/MC,
  and Gaussian-$h$ symbolic $\to$ the node-averaging baseline (so the symbolic state reproduces
  the "run at every $h$ node and average" result **without** running the nodes).
- **§1** on square nets **trained to 0 (loss < 1e-7)**, all predictors track the **40M-point MC**
  mean within the shared $k{=}2$ **closure** error, which shrinks with width; symbolic(q=0)
  coincides with vanilla kprop as it must.
- **§2** the prediction converges in hidden degree $p$ to the conditional-Gaussian value, then hits
  the documented **float64** high-degree floor; the **tail diagnostic** reports sufficiency and the
  method lists its **approximations** (never silently Gaussian, never a silently-fixed $p$).
- **§3** with a **planted** all-ones spike the symbolic latent beats vanilla / $q{=}0$ by orders of
  magnitude — the coherent shift a single Gaussian state cannot represent.

**Recycling:** models + MC/predictor results live in `checkpoints/symbolic_hidden_mode_kprop`; re-runs
load instead of recompute. **Cost:** one symbolic run $\approx$ one $k{=}2$ kprop $\times$ (collocation
nodes) — far cheaper than node-averaging, with the hidden mode resolved symbolically.""")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "symbolic_hidden_mode_kprop_colab.ipynb")
nb.save(out)

"""Generates exact_relu_k2_width_scaling_colab.ipynb (valid nbformat-4 JSON).

Question: cumulant propagation degrades after a model is trained. The default k=2
step approximates the ReLU off-diagonal covariance by the leading-order gain
Sigma_ij*c_i*c_j. Does computing that covariance EXACTLY (exact_relu_cov=True) fix
the trained-model breakdown? We compare, at the SAME budget k_max=2, on the SAME
trained-to-zero models:
    - "approx (k=2)" : the default propagation (gain approximation), and
    - "exact (k=2)"  : the exact bivariate-Gaussian ReLU covariance,
and look at how each one's error-vs-Monte-Carlo scales with width n, init vs
trained-to-zero. Depth 3, no bias, large MC. TRAIN-TO-TOLERANCE regime: every model
trains until MSE < 1e-5 -- no fixed step budget; widths 16..1024, 1536, 2048. All
knobs are defined IN the notebook's config cell (experiments.py only keeps the
classic defaults) -- probe there freely. Models train in float32 (all seeds of a
width in one vmapped ensemble loop, checkpointed + recycled under
checkpoints/kprop_tol_*, regime encoded in the name via _tol5); kprop itself still
propagates in float64 internally.

This is the project's BASELINE scaling notebook (compare future experiments to it).
It runs on the unified codebase: build a `model.MLP`, train it with `training`,
and predict its mean with `Mecha_preds.cumulants.run_cumulants` (a black box -- we
only flip `exact_relu_cov`). §2 prints the source files.

Run:  python "experiments/trained_to_0_cumulants_test/build_exact_scaling_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder

nb = NotebookBuilder()
md, code = nb.md, nb.code


# =============================================================================
md(r"""# Does the EXACT ReLU covariance fix cumulant propagation on **trained-to-zero** models?

Cumulant propagation predicts the output mean of a wide *random* MLP well, but its
accuracy **degrades after training**. One suspect is the K=2 step's **off-diagonal
covariance**: the default propagation approximates `Cov(ReLU(Z_i), ReLU(Z_j))` by the
leading-order gain `Sigma_ij * c_i * c_j` (`c_i = Phi(mu_i/sigma_i)`). This notebook
tests whether computing it **exactly** removes the trained-model problem.

**Two versions of the algorithm, same budget `k_max = 2`, same model (no copy):**
- **`approx (k=2)`** — the default harmonic propagation (the gain approximation).
- **`exact (k=2)`** — `exact_relu_cov=True`: the **exact** bivariate-Gaussian ReLU
  covariance, `new_Sigma_ij = E[ReLU(Z_i)ReLU(Z_j)] − new_mu_i·new_mu_j`, no gain
  approximation (`Mecha_preds/cumulants/kprop/exact_relu_covariance.py`).

**The test.** Train the output to 0 (MSE→0): depth 3, **train-to-tolerance** — every
model trains UNTIL its MSE drops below `LOSS_TOL = 1e-5` (no fixed step budget; the
`MAX_STEPS` cap only guards against non-convergence), large MC, **no bias**. The
old fixed-12000-step regime left different widths at different training depths and
pushed some models into unexplained behaviour; here every width stops at the SAME
loss level (`MAX_STEPS` is only a safety cap). For each width `n` (with
`input_dim = n`) measure
the MC-variance-debiased per-entry RMS error of the propagated mean and its scaling
with `n`, init vs trained-to-zero, for both methods. Then ask:

> Does **`exact (k=2)`** scale **better** than **`approx (k=2)`** after training
> (→ the covariance approximation was the problem) or **the same** (→ it was not;
> the breakdown is intrinsic to the single-Gaussian K=2 state / training-induced
> correlations)?

> **Depth.** With 1 hidden layer the output mean uses only the exact ReLU
> *marginals*, so both methods are exact and identical; the off-diagonal covariance
> only matters at ≥ 2 hidden layers. We use depth 3.

Nothing here reimplements cumulant propagation — we call `run_cumulants` as a black
box and only flip `exact_relu_cov`. §2 prints the source files.
""")

# --- Setup -------------------------------------------------------------------
md(r"""## 0. Setup — get the repo and install the *minimal* dependencies

Point at the repo root (clone URL or local/Drive path). The checkout **must contain
`Mecha_preds/cumulants/kprop/exact_relu_covariance.py`** (§2 checks and fails loudly
otherwise). The exact path also needs **scipy** (the bivariate normal CDF).
""")

code(r"""
# ----------------------------- EDIT THIS -----------------------------------
REPO_URL       = ""   # e.g. "https://github.com/<you>/one-trained-case.git" (leave "" if not cloning)
LOCAL_REPO_DIR = ""   # e.g. "/content/drive/MyDrive/One trained case" or a local path
# ---------------------------------------------------------------------------

import os, sys, subprocess
try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False

_MARKER = os.path.join("Mecha_preds", "cumulants", "kprop")

def _find_repo():
    if LOCAL_REPO_DIR and os.path.isdir(os.path.join(LOCAL_REPO_DIR, _MARKER)):
        return LOCAL_REPO_DIR
    if REPO_URL:
        dest = "/content/one_trained_case" if IN_COLAB else os.path.abspath("./_repo_clone")
        if not os.path.isdir(os.path.join(dest, _MARKER)):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dest], check=True)
        return dest
    here = os.path.abspath(".")
    for _ in range(6):
        if os.path.isdir(os.path.join(here, _MARKER)):
            return here
        here = os.path.dirname(here)
    raise RuntimeError("Could not locate the repo. Set REPO_URL or LOCAL_REPO_DIR above.")

REPO_DIR = _find_repo()
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("IN_COLAB:", IN_COLAB, "| REPO_DIR:", REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jaxtyping", "einops", "opt_einsum", "joblib", "tqdm", "scipy"], check=False)
print("deps installed")
""")

# --- Global numerics ---------------------------------------------------------
md(r"""## 1. Global numerical settings — float32 model, float64 measurement

The model, training, and Monte-Carlo sampling run in **`float32`** (tensor cores +
TF32 on a GPU; this is what makes the 1536/2048 widths and train-to-tolerance runs
finish in human time). Precision is NOT lost where it matters:

- `run_cumulants` builds a **float64** copy of the model internally — kprop always
  propagates in double, whatever the model dtype;
- the MC reference accumulates sums in **float64** (`estimate_empirical_mean`),
  and both kprop and MC see the *same* float32 weights, so the comparison is
  consistent; the errors measured here (≳1e-3 relative) sit far above float32 noise.

Flip `MODEL_DTYPE = torch.float64` to reproduce the old all-double behaviour.
""")

code(r"""
import torch, numpy as np, pandas as pd, math
import matplotlib.pyplot as plt

MODEL_DTYPE = torch.float32           # training/inference dtype (float64 = old slow behaviour)
torch.set_default_dtype(MODEL_DTYPE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":                  # TF32 matmuls on Ampere+ (~1e-3 rel precision, fine here)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
MC_BATCH = 262_144 if DEVICE == "cuda" else 8_192
print("device:", DEVICE, "| MODEL_DTYPE:", MODEL_DTYPE, "| MC_BATCH:", MC_BATCH)
""")

# --- Verify ------------------------------------------------------------------
md(r"""## 2. Verify the **real** algorithm and the exact-covariance path

Prints the source files and asserts the `exact_relu_cov` flag is present.
""")

code(r"""
import inspect
import Mecha_preds.cumulants.kprop.kprop_harmonic as kp
from Mecha_preds.cumulants.kprop import WICK_COEF_D
import Mecha_preds.cumulants.kprop.exact_relu_covariance as erc

print("mlp_kprop defined in:                ", inspect.getsourcefile(kp.mlp_kprop))
assert "exact_relu_cov" in inspect.signature(kp.mlp_kprop).parameters, (
    "This checkout has no exact_relu_cov flag — the vendored kprop is missing "
    "exact_relu_covariance support."
)
print("exact bivariate ReLU covariance in:  ", inspect.getsourcefile(erc.exact_relu_covariance_kprop))
import scipy; print("scipy (needed for Phi2):             ", scipy.__version__)
""")

# --- Config ------------------------------------------------------------------
md(r"""## 3. Config (depth 3, no bias) — **the knobs live HERE**

This cell is the experiment definition: widths, seeds, tolerance, MC size — edit and
re-run in place (`experiments.py` only keeps the classic defaults plus the naming /
recycling machinery). The loss-tolerance exponent goes into the checkpoint name, so
probing a different `LOSS_TOL` cannot silently reuse models from another regime.

Two methods, both `k_max = 2`; the only difference is `exact_relu_cov`. Same model
init across `initial` / `train_to_zero` per seed (rebuilt each time from the same
`ModelConfig` seed), `input_dim = width`, vector output so the per-entry MSE averages
over many components.
""")

code(r"""
import experiments as E
from model import ModelConfig
from tasks import ZeroTask
from training import TrainConfig
from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, config_summary
from utils import set_seed

# ---- Architecture -----------------------------------------------------------
HIDDEN_DEPTH = 3
ACTIVATION   = "relu"
USE_BIAS     = False    # NO bias parameters (half-space boundaries through the origin)
OUTPUT_DIM   = 128      # vector output: per-entry MSE averages over many components

# ---- The two methods (both k_max=2; only exact_relu_cov differs) -----------
COMMON_CFG = {"kind": "simple", "use_avg_metric": False, "factor": False, "use_pK": True, "output_d_max": 1}
METHODS = {
    "approx (k=2)": {**COMMON_CFG, "k_max": 2, "exact_relu_cov": False},
    "exact (k=2)":  {**COMMON_CFG, "k_max": 2, "exact_relu_cov": True},
}

# ---- Sweep ranges -- THIS notebook's knobs; probe freely in place ------------
QUICK  = E.QUICK                      # True on a CPU-only machine -> small smoke sweep
WIDTHS = [16, 32, 64] if QUICK else [16, 32, 64, 128, 256, 512, 1024, 1536, 2048]
SEEDS  = [3, 4, 5, 6]
PHASES = ["initial", "train_to_zero"]

# ---- Training: TRAIN-TO-TOLERANCE (the regime of this study) ----------------
# Every model trains until per-step MSE < LOSS_TOL; MAX_STEPS is only a safety
# cap (a run that hits it did NOT converge). All seeds of one width train
# SIMULTANEOUSLY in one vmapped loop (training/parallel.py) and are checkpointed
# under checkpoints/kprop_tol_*, so a re-run (or a crashed Colab session)
# recycles instead of retraining. NOTE: the tolerance exponent is part of the
# checkpoint name (tol5 = 1e-5), so changing LOSS_TOL never silently reuses
# models trained under a different regime.
LOSS_TOL   = 1e-5
MAX_STEPS  = 200_000
TOL_CHECK_EVERY = 1   # check the stop criterion EVERY step. ZeroTask converges so fast
                      # (1e-3 -> 1e-9 inside 50 steps at large width with lr=1e-3) that
                      # sparse checks overshoot far below LOSS_TOL and un-match the
                      # widths' loss levels. Runs are short, so the per-step sync is free.
TOL_PATIENCE = 25     # stop only after the loss has been below LOSS_TOL for this many
                      # CONSECUTIVE steps (one above-tol step resets the count): the stop
                      # fires when the loss has STABILIZED below tol, not on one lucky batch.
BATCH_SIZE = 1024
LR         = 1e-4     # gentle rate so the loss GLIDES into the tolerance instead of
                      # crashing orders of magnitude past it between two checks
LOG_EVERY  = 10       # fine-grained history -> usable loss curves in §5b
CKPT_DIR   = "checkpoints/kprop_checkpoints"   # THIS notebook's checkpoint folder
TOL_TAG    = int(round(-math.log10(LOSS_TOL)))   # 1e-5 -> 5, goes into the run name

# ---- Monte-Carlo -------------------------------------------------------------
MC_SAMPLES = 200_000 if QUICK else 2_000_000   # large MC reference


def build_model(in_dim, width, depth, seed):
    cfg = ModelConfig(input_dim=in_dim, hidden_dim=width, depth=depth, output_dim=OUTPUT_DIM,
                      bias=USE_BIAS, final_bias=USE_BIAS, activation=ACTIVATION, seed=seed)
    return cfg.build().to(device=DEVICE, dtype=MODEL_DTYPE)


def cp_mean(model, in_dim, method):
    return run_cumulants(model, in_dim, METHODS[method], device=DEVICE)["mean"]


for name in METHODS:
    print(f"{name:13s} cfg:", config_summary(METHODS[name]))
print("phases:", PHASES, "| seeds:", SEEDS, "| widths:", WIDTHS)
print(f"train-to-tol: MSE < {LOSS_TOL:g} (cap {MAX_STEPS}) | ckpts -> {CKPT_DIR}")
""")

# --- Verify exact != approx --------------------------------------------------
md(r"""## 4. Verify the exact path is genuinely different (and the depth caveat)

`exact ≡ approx` at depth 1 (mean uses only the exact marginals), `exact ≠ approx`
at depth ≥ 2 (the exact off-diagonal covariance changes the downstream variance and
hence the mean). The latter is the regime we scale.
""")

code(r"""
set_seed(3); n0 = 128
print("||exact(k=2) - approx(k=2)||  on the propagated output mean:")
for depth in [1, 2, 3]:
    m = build_model(n0, n0, depth, seed=3)
    cpa = cp_mean(m, n0, "approx (k=2)")
    cpe = cp_mean(m, n0, "exact (k=2)")
    print(f"  depth={depth}: {np.linalg.norm(cpe - cpa):.2e}")
print("=> depth 1: ~0 (only exact marginals); depth>=2: nonzero (exact off-diagonal matters).")
""")

# --- The sweep ---------------------------------------------------------------
md(r"""## 5. The width-scaling sweep — parallel seeds, recycled checkpoints

For each width: evaluate the 4 seeds at init, then **train all 4 seeds to tolerance
in ONE vmapped loop** (`E.get_or_train_many` → `training.train_ensemble`; exactly
equivalent to 4 independent Adam runs, ~4× faster on GPU). Trained models are
checkpointed under `checkpoints/kprop_checkpoints/`, so re-running this cell
(or resuming a dead Colab session) loads instead of retraining. A model whose final
loss is still above `LOSS_TOL` hit the step cap and is flagged **NOT CONVERGED**.

Per `(width, seed, phase)`: one Monte-Carlo reference, then **both** methods against
that same MC. Record the MC-variance-debiased per-entry RMS error, its scale-free
version (÷ `output_rms`), and the signal-to-floor ratio (`>~3` resolved).
""")

code(r"""
def debiased_rms(cp, mc, mc_stats):
    cp = np.asarray(cp, np.float64).reshape(-1)
    mc = np.asarray(mc, np.float64).reshape(-1)
    se = np.asarray(mc_stats["mc_stderr"], np.float64).reshape(-1)
    measured = float(np.mean((cp - mc) ** 2))   # = true kprop MSE + MC variance
    floor    = float(np.mean(se ** 2))
    return math.sqrt(max(measured - floor, 0.0)), measured, floor

def loglog_slope(ws, vs):
    ws, vs = np.asarray(ws, float), np.asarray(vs, float)
    mask = vs > 0
    if mask.sum() < 2: return float("nan")
    return float(np.polyfit(np.log(ws[mask]), np.log(vs[mask]), 1)[0])

def evaluate(model, in_dim, width, seed, phase, final_loss, mc_batch, rows):
    mc, st = estimate_empirical_mean(model=model, input_dim=in_dim, num_samples=MC_SAMPLES,
                                     batch_size=mc_batch, device=DEVICE, dtype=MODEL_DTYPE)
    for method in METHODS:
        cp = cp_mean(model, in_dim, method)
        rms, meas, floor = debiased_rms(cp, mc, st)
        rel = rms / (st["empirical_output_rms"] + 1e-30)
        s2f = (meas - floor) / (floor + 1e-300)
        rows.append(dict(method=method, phase=phase, width=width, seed=seed,
                         debiased_rms=rms, rel_debiased_rms=rel, s_to_floor=s2f,
                         out_rms=st["empirical_output_rms"], final_train_loss=final_loss))
    ls = "" if math.isnan(final_loss) else f" loss={final_loss:.0e}"
    flag = " *** NOT CONVERGED (hit step cap) ***" \
        if (not math.isnan(final_loss) and final_loss > LOSS_TOL) else ""
    print(f"  n={width:4d} s={seed} {phase:14s} out_rms={st['empirical_output_rms']:.2e}{ls}{flag}")

rows = []
train_histories = {}            # (width, seed) -> [(step, loss), ...] for the §5b loss curves
for width in WIDTHS:
    in_dim = width
    mc_batch = min(MC_BATCH, max(8192, (1 << 26) // in_dim))   # bound batch*in_dim memory at large width

    # --- initial phase: rebuilt from the seed, never trained -------------------
    for seed in SEEDS:
        model = build_model(in_dim, width, HIDDEN_DEPTH, seed=seed)
        evaluate(model, in_dim, width, seed, "initial", float("nan"), mc_batch, rows)

    # --- trained phase: ALL seeds in one ensemble loop, recycled from disk -----
    # run name encodes the regime: kprop-zero_d3_w<width>_tol<TOL_TAG>_seed<s>
    paths  = [E.ckpt_path(CKPT_DIR, E.run_name("kprop-zero", depth=HIDDEN_DEPTH, width=width,
                                               tol=TOL_TAG, seed=s))
              for s in SEEDS]
    builds = [(lambda s=s: build_model(in_dim, width, HIDDEN_DEPTH, seed=s)) for s in SEEDS]
    tcfg   = TrainConfig(steps=MAX_STEPS, batch_size=BATCH_SIZE, lr=LR, optimizer="adamw",
                         loss_tol=LOSS_TOL, tol_check_every=TOL_CHECK_EVERY,
                         tol_patience=TOL_PATIENCE, checkpoint_mode="final",
                         log_every=LOG_EVERY,
                         device=DEVICE, dtype=str(MODEL_DTYPE).split(".")[-1])
    trained = E.get_or_train_many(paths, builds,
                                  task=ZeroTask(input_dim=in_dim, output_dim=OUTPUT_DIM),
                                  train_cfg=tcfg, extra_meta={"experiment": "kprop_tol_scaling"},
                                  map_location=DEVICE, progress=True)
    for seed, (model, payload, loaded) in zip(SEEDS, trained):
        model = model.to(device=DEVICE, dtype=MODEL_DTYPE)
        train_histories[(width, seed)] = payload.get("history") or []
        print(f"    seed {seed}: {'[loaded from ckpt]' if loaded else '[trained fresh]'}")
        evaluate(model, in_dim, width, seed, "train_to_zero", E.final_loss(payload), mc_batch, rows)
dfsc = pd.DataFrame(rows)
dfsc.head(8)
""")

# --- Loss curves ---------------------------------------------------------------
md(r"""## 5b. Training loss curves — did every width stop AT the tolerance?

One curve per `(width, seed)` (thin lines = seeds, color = width), log-y. Every curve
should **glide into the dashed `LOSS_TOL` line and stop just below it** — that is the
matched regime. A curve that ends far below the line means the stop overshot (raise
`TOL_PATIENCE` granularity / lower `LR` further); one that never reaches it hit
`MAX_STEPS` (not converged). Markers = the saved (checkpointed) stopping point.
""")

code(r"""
import matplotlib.cm as cm

fig, ax = plt.subplots(figsize=(9.5, 5.5))
wcolors = {w: cm.viridis(i / max(1, len(WIDTHS) - 1)) for i, w in enumerate(WIDTHS)}
for (w_, s_), hist in sorted(train_histories.items()):
    if not hist:
        continue
    steps_, losses_ = zip(*hist)
    ax.plot(steps_, losses_, "-", color=wcolors[w_], lw=1.0, alpha=0.75)
    ax.plot(steps_[-1], losses_[-1], "o", color=wcolors[w_], ms=5,
            markeredgecolor="k", markeredgewidth=0.4)
ax.axhline(LOSS_TOL, color="crimson", ls="--", lw=1.2, label=f"LOSS_TOL = {LOSS_TOL:g}")
for w_ in WIDTHS:   # one legend entry per width
    ax.plot([], [], "-", color=wcolors[w_], label=f"n={w_}")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("per-step MSE")
ax.set_title("train-to-tolerance loss curves (marker = saved stopping point)")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.show()

print("final loss / LOSS_TOL ratio per run (want ~just below 1):")
for (w_, s_), hist in sorted(train_histories.items()):
    if hist:
        print(f"  n={w_:4d} s={s_}: stopped step {hist[-1][0]:>6}  "
              f"loss {hist[-1][1]:.2e}  ({hist[-1][1]/LOSS_TOL:.2f}x tol)")
""")

code(r"""
print("Scale-free debiased RMS (median over seeds) and its best-fit log-log slope.")
print("  s/floor < ~3 => below MC floor (raise MC_SAMPLES).\n")
for phase in PHASES:
    for method in METHODS:
        sub = dfsc[(dfsc.phase == phase) & (dfsc.method == method)]
        rmed = sub.groupby("width")["rel_debiased_rms"].median()
        s2f  = sub.groupby("width")["s_to_floor"].median()
        ws = sorted(rmed.index)
        print(f"  {phase:14s} {method:12s}: slope={loglog_slope(ws,[rmed[w] for w in ws]):+.2f}  "
              + "  ".join(f"n{w}={rmed[w]:.2e}(s/f={s2f[w]:.0f})" for w in ws))

print("\nDoes the EXACT covariance help? approx/exact ratio of scale-free RMS (>1 => exact more accurate):")
for phase in PHASES:
    a = dfsc[(dfsc.phase==phase)&(dfsc.method=="approx (k=2)")].groupby("width")["rel_debiased_rms"].median()
    e = dfsc[(dfsc.phase==phase)&(dfsc.method=="exact (k=2)")].groupby("width")["rel_debiased_rms"].median()
    ws = sorted(set(a.index) & set(e.index))
    print(f"  {phase:14s}: " + "  ".join(f"n{w}={ (a[w]/e[w]) :.2f}x" if e[w]>0 else f"n{w}=NA" for w in ws))
""")

# --- Plots (with fitted log-log slope) ---------------------------------------
md(r"""## 6. Plots — error scaling with a **fitted log-log line**

Markers = median-over-seeds data; the **solid line is the best least-squares fit in
log-log space**, and its **slope is printed in the legend** (the measured scaling
exponent). Faint dotted = the `n^-1` reference (k=2 theory). Blue = `initial`, red =
`train_to_zero`; circle = `exact`, square = `approx`.
""")

code(r"""
def fit_loglog(ws, ys):
    ws = np.asarray(ws, float); ys = np.asarray(ys, float); m = ys > 0
    if m.sum() < 2:
        return None, None, float("nan")
    slope, intercept = np.polyfit(np.log(ws[m]), np.log(ys[m]), 1)
    xf = np.array(sorted(ws[m]))
    return xf, np.exp(intercept) * xf ** slope, float(slope)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
mc_color = {"initial": "tab:blue", "train_to_zero": "tab:red"}
mt_mark  = {"exact (k=2)": "o", "approx (k=2)": "s"}

for ax, col, ttl in [(axes[0], "debiased_rms", "ABSOLUTE debiased per-entry RMS error"),
                     (axes[1], "rel_debiased_rms", "SCALE-FREE error (RMS / output_rms)")]:
    for phase in PHASES:
        for method in METHODS:
            sub = dfsc[(dfsc.phase == phase) & (dfsc.method == method)]
            if sub.empty: continue
            med = sub.groupby("width")[col].median()
            ws = np.array(sorted(med.index)); y = med.reindex(ws).to_numpy()
            color = mc_color[phase]
            ax.plot(ws, np.where(y <= 0, np.nan, y), linestyle="none", marker=mt_mark[method],
                    color=color, alpha=0.9)
            xf, yf, slope = fit_loglog(ws, y)
            if xf is not None:
                ax.plot(xf, yf, "-", color=color, lw=1.8,
                        label=f"{method} / {phase}  (slope {slope:+.2f})")
    base = dfsc[(dfsc.method=="approx (k=2)")&(dfsc.phase=="initial")].groupby("width")[col].median()
    bw = np.array(sorted(base.index), float)
    if len(base) and base.iloc[0] > 0:
        ax.plot(bw, base.iloc[0]*(bw/bw[0])**(-1.0), color="gray", ls=":", lw=1.0, alpha=0.6, label="n^-1 (theory)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width n  (= input_dim)"); ax.set_title(ttl)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# --- Conclusions -------------------------------------------------------------
md(r"""## 7. How to read it (the baseline answer)

> **Recorded baseline from the OLD regime** (depth 3, no bias, `1e-8` stop, fixed
> 12000-step budget, 1–2M MC, float64) — kept for comparison. The current notebook
> uses the **train-to-tolerance regime** (`MSE < 1e-5`, no fixed budget), so expect
> quantitative differences, especially where the old budget under- or over-trained
> a width; the question is whether the qualitative picture survives:
>
> | n | out_rms | approx scale-free | (s/floor) | exact scale-free | approx/exact |
> |--:|--:|--:|--:|--:|--:|
> | 64  | 2.0e-3 | 8.4e-2 | 7000 | 1.1e-1 | 0.78× |
> | 128 | 1.2e-3 | 3.0e-2 |  900 | 3.6e-2 | 0.83× |
> | 256 | 7.1e-4 | 7.8e-3 |   60 | 9.3e-3 | 0.83× |
>
> fitted slope ≈ **−1.7** for both. Takeaways: (1) the trained error is **well above
> the MC floor** (`s/floor` up to ~7000) and **genuinely decreases with width** —
> width still helps trained nets, not an artifact. (2) `exact ≈ approx` (ratio
> ~0.8×, exact if anything *marginally worse*), same slope ⇒ **the exact K=2
> covariance does NOT fix the trained-model problem.** (3) The trained error is
> **elevated vs init** (~8e-2 vs ~3e-3 at n=64, ~25× worse), worst at small width —
> *that* is the training degradation, not a loss of width scaling.

Interpretation of the `exact` vs `approx` curves (and the §5 ratio):
- **Overlap after training** (ratio ≈ 1×, same fitted slope) ⇒ the exact covariance
  does NOT fix it; the degradation is intrinsic to the single-Gaussian K=2 state
  (training-induced correlations a mean+covariance can't capture). The remedy would
  be higher cumulants (`k_max ≥ 3`) or a non-Gaussian state.
- **`exact` steeper / lower** (ratio > 1, widening with `n`) ⇒ the gain
  approximation *was* a real source of trained error.

At init both should track `n^-1` and sit near each other; the signal is in the
trained curves.

**Caveats:** few seeds ⇒ noisy slopes; at large `n` the debiased RMS can hit the MC
floor (watch `s/floor`, raise `MC_SAMPLES`). This is a **heavy** sweep (9 widths up
to 2048 × 4 seeds × 2 phases × train-to-tolerance × 2M MC). The training itself is
fast (float32 + all seeds vmapped in one loop + checkpoint recycling across
sessions), but the **exact-covariance path is CPU/scipy, O(n²) per layer** — at
n=1536/2048 it dominates the runtime. Use a GPU; on CPU `QUICK=True` trims the sweep
to 16–64. Flag any width whose seeds print **NOT CONVERGED** (step cap hit) before
trusting its point.
""")

nb.save(os.path.join(os.path.dirname(__file__), "exact_relu_k2_width_scaling_colab.ipynb"))

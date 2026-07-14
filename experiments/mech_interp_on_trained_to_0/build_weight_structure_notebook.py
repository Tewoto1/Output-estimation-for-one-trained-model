"""Generates weight_structure_vs_randomness.ipynb (valid nbformat-4 JSON).

Tests the "structure + randomness" decomposition of trained ReLU-MLP hidden weights:
a *covariance-adjusted negative-mean drift* (the structured component) plus an
approximately-Gaussian residual. Plugs into the unified codebase:
`model.MLP` / `tasks` / `training` for models & checkpoints, `analysis.Tools` for the
shared linear-algebra primitives, and (optionally) `Mecha_preds.cumulants` to connect
the decomposition back to cumulant propagation.

Mirrors the conventions of the other notebook generators in this folder:
  * cell builders `md()` / `code()`,
  * a §0 setup cell that finds-or-clones the repo (paste your GitHub URL),
  * everything in float64, results written under results/weight_structure/.

Run:  python experiments/mech_interp_on_trained_to_0/build_weight_structure_notebook.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder

nb = NotebookBuilder()
md, code = nb.md, nb.code


# =============================================================================
md(r"""# Structure + randomness in trained ReLU-MLP weights
### Testing a *covariance-adjusted negative-mean drift* as the structured component

The cumulant-propagation paper estimates `E[f(X)]` for **wide random** MLPs by pushing
low-order cumulants through the layers. Extending it to **trained** networks likely
needs a **structure-vs-randomness** split: model a trained weight matrix as a
*structured component selected by training* **+** a *random residual*. This notebook
tests one candidate structured component for **zero-output-trained** ReLU MLPs.

**The conjecture (whitened coordinates).** For the layer `W⁽ˡ⁺¹⁾` that reads the
previous post-activation `X⁽ˡ⁾` (mean `μ_l`, covariance `Σ_l`), ridge-stabilise
`S_l = Σ_l + λI`, whiten the rows `W̃ = W S_l^{1/2}`, and let
`q̂_l = S_l^{-1/2}μ_l / ‖·‖`. Then

$$\widetilde W^{(l+1)} \;\approx\; \tfrac{\sigma_l}{\sqrt n}\,G \;-\; \tfrac{1}{\sqrt n}\,a^{(l+1)}\,\hat q_l^{\top},\qquad G_{ij}\sim\mathcal N(0,1),\ a_i\ge 0.$$

i.e. **hidden rows point *against* the previous layer's mean, in the geometry set by
its covariance**, and what's left over looks Gaussian. The headline statistic is

$$\hat a_i = -\sqrt n\,(\widetilde W^{(l+1)})_i\!\cdot\!\hat q_l,\qquad \text{predict } \overline{\hat a_i} > 0 \text{ for middle layers.}$$

**What we check** (see §11 for the full pass/fail table):
1. middle-layer rows have systematically negative covariance-adjusted mean alignment
   (`mean âᵢ > 0`), **absent at init / in the first layer / under random controls**;
2. the readout mostly **shrinks** instead of developing that gating drift;
3. covariance adjustment vs. raw `μ` alignment;
4. after removing the `q̂_l` direction the residual looks **more Gaussian**;
5. **task** models (random half-space, max-input) grow task-direction structure on top.

> **A note we keep honest throughout.** Algebraically `W̃_i·q̂_l = (w_iᵀμ_l)/‖q‖`, so the
> *sign* of the whitened projection equals the sign of the raw `w_iᵀμ_l`. Covariance
> adjustment therefore does **not** change the sign of the per-row mean-alignment — its
> teeth are in the *normalisation* (the Mahalanobis cosine `ρ`), the *drift-removal
> direction* `S_l^{-1}μ_l`, and the *residual geometry*. We report both raw and
> covariance-adjusted views so this is visible rather than hidden.""")

# =============================================================================
md(r"""## 0 · Setup — find or clone the repo

Paste your GitHub URL into `REPO_URL` (or point `LOCAL_REPO_DIR` at a local / Drive
checkout). The checkout must contain the `model/`, `tasks/`, `training/` and
`analysis/` packages and the tracked zero-task checkpoints under
`checkpoints/weight_analysis_checkpoints/`.""")

code(r"""
# ----------------------------- EDIT THIS -----------------------------------
REPO_URL       = ""   # e.g. "https://github.com/<you>/one-trained-case.git"  (paste yours)
LOCAL_REPO_DIR = ""   # e.g. "/content/drive/MyDrive/One trained case"  (or leave "")
# ---------------------------------------------------------------------------

import os, sys, subprocess
try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# A repo root is identified by these top-level packages.
def _is_root(c):
    return (os.path.isdir(os.path.join(c, "model"))
            and os.path.isdir(os.path.join(c, "tasks"))
            and os.path.isdir(os.path.join(c, "training")))

def _find_repo():
    if LOCAL_REPO_DIR and _is_root(LOCAL_REPO_DIR):
        return LOCAL_REPO_DIR
    if REPO_URL:
        dest = "/content/one_trained_case" if IN_COLAB else os.path.abspath("./_repo_clone")
        if not _is_root(dest):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dest], check=True)
        return dest
    # fall back to walking up from CWD (works when the notebook lives in the repo)
    here = os.path.abspath(".")
    for _ in range(6):
        if _is_root(here):
            return here
        here = os.path.dirname(here)
    raise RuntimeError("Could not locate the repo. Set REPO_URL or LOCAL_REPO_DIR above.")

REPO_DIR = _find_repo()
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("IN_COLAB:", IN_COLAB, "| REPO_DIR:", REPO_DIR)

# Minimal deps (torch / numpy / matplotlib usually preinstalled on Colab).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numpy", "scipy", "pandas", "matplotlib", "tqdm"], check=False)
print("deps ok")
""")

md(r"""### Run configuration

All knobs live here. Defaults are sized for a Colab session; every value can also be
overridden by an environment variable (handy for a fast headless smoke-run).""")

code(r"""
import os, numpy as np, torch

# everything numerical in double precision (covariances / eigendecompositions)
torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _env_int(k, d):   return int(os.environ.get(k, d))
def _env_float(k, d): return float(os.environ.get(k, d))
def _env_list(k, d):  return [x for x in os.environ.get(k, d).split(",") if x]

N_ACT_SAMPLES = _env_int("WS_N_SAMPLES", 65536)     # Gaussian inputs for activation stats
ACT_BATCH     = _env_int("WS_ACT_BATCH", 8192)
TRAIN_STEPS   = _env_int("WS_TRAIN_STEPS", 6000)    # steps for tasks we have to train here
TRAJ_STEPS    = _env_int("WS_TRAJ_STEPS", 6000)     # steps for the training-trajectory run
TRAJ_EVERY    = _env_int("WS_TRAJ_EVERY", 750)      # checkpoint cadence for the trajectory
NEW_WIDTHS    = [int(w) for w in _env_list("WS_WIDTHS", "64,128")]   # widths to train tasks at
NEW_DEPTHS    = [int(d) for d in _env_list("WS_DEPTHS", "3")]        # depths to train tasks at
RIDGE_ALPHAS  = [float(a) for a in _env_list("WS_RIDGE", "1e-6,1e-4,1e-2,1e-1")]
MAIN_ALPHA    = _env_float("WS_MAIN_ALPHA", 1e-4)   # ridge used for the main tables/plots
SEED          = _env_int("WS_SEED", 0)

# Output dirs (results/ is git-ignored = regenerable; checkpoints under a study subdir).
RESULTS_DIR = os.path.join(REPO_DIR, "results", "weight_structure")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
TAB_DIR     = os.path.join(RESULTS_DIR, "tables")
CKPT_DIR    = os.path.join(REPO_DIR, "checkpoints", "weight_analysis_checkpoints")
for d in (FIG_DIR, TAB_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 110

def savefig(fig, name):
    'Save a figure under results/weight_structure/figures and return the path.'
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, bbox_inches="tight")
    return p

print("device:", DEVICE, "| float64 |", "N_ACT_SAMPLES", N_ACT_SAMPLES,
      "| TRAIN_STEPS", TRAIN_STEPS, "| widths", NEW_WIDTHS, "depths", NEW_DEPTHS)
print("results ->", RESULTS_DIR)
print("checkpoints ->", CKPT_DIR)
""")

# =============================================================================
md(r"""## 1 · Diagnostics library

Small, self-contained functions (each returns a dict — no master "report" step, in
keeping with the repo's analysis style). PyTorch `Linear.weight` is `[d_out, d_in]`, so
**rows are output neurons**; the layer `W` that produces hidden block `i` reads the
post-activation `X⁽ˡ⁾` with `l = i-1` (and the readout reads the last hidden block).
The very first layer reads the input `X⁽⁰⁾ ~ N(0, I)`, for which `μ₀ = 0` exactly — so it
has *no* mean direction to point against (the predicted null case).""")

code(r"""
import math
from analysis.Tools.common import svd_spectrum   # shared SVD/effective-rank primitive
from scipy import stats as sstats

# ---- input sampler & layer bookkeeping -------------------------------------

def gaussian_inputs(n, num, seed=0, std=1.0):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(num, n, generator=g, dtype=torch.float64) * std

# One entry per weight matrix in forward order. prev_key indexes the previous
# post-activation: -1 = input X^(0); k = post-act of hidden block k. kind: first/middle/final.
def layer_roles(model):
    depth = model.cfg.depth
    roles = []
    biases = dict(model.named_biases())
    for name, W in model.named_weights():
        if name == "readout":
            prev, kind = depth - 1, "final"
        else:
            i = int(name.replace("hidden", ""))
            prev = -1 if i == 0 else i - 1
            kind = "first" if i == 0 else "middle"
        b = biases.get(name, None)
        roles.append({"name": name, "W": W.detach().double(),
                      "bias": None if b is None else b.detach().double(),
                      "kind": kind, "prev_key": prev})
    return roles
""")

code(r"""
# ---- activation statistics (one streaming pass, float64) -------------------

# Estimate per post-activation layer: mu, Sigma, second_moment(diag E[X^2]),
# active_prob P(z>0), cov_xt (Cov(X,T) if target_fn else None). Input added as key -1
# analytically (mu=0, Sigma=I). One streaming float64 pass.
@torch.no_grad()
def estimate_layer_stats(model, num=N_ACT_SAMPLES, batch=ACT_BATCH, seed=0,
                         std=1.0, target_fn=None):
    model = model.double().to(DEVICE).eval()
    n_in, depth = model.cfg.input_dim, model.cfg.depth
    g = torch.Generator().manual_seed(int(seed))
    keys = list(range(depth))
    s1 = {k: 0.0 for k in keys}        # sum X         (post)
    s2 = {k: 0.0 for k in keys}        # sum X X^T     (post)
    sq = {k: 0.0 for k in keys}        # sum X^2 diag  (post second moment)
    sa = {k: 0.0 for k in keys}        # sum 1[z>0]    (pre active count)
    sxt = {k: 0.0 for k in keys}       # sum X * T
    sT, sTT, cnt = 0.0, 0.0, 0
    nb = math.ceil(num / batch)
    for _ in range(nb):
        x = (torch.randn(batch, n_in, generator=g, dtype=torch.float64) * std).to(DEVICE)
        _, acts = model(x, return_activations=True)
        T = None
        if target_fn is not None:
            T = target_fn(x).reshape(-1).double()       # (batch,)
            sT += T.sum().item(); sTT += (T * T).sum().item()
        for i in keys:
            P = acts["post"][i].double(); Z = acts["pre"][i].double()
            s1[i] = s1[i] + P.sum(0)
            s2[i] = s2[i] + P.T @ P
            sq[i] = sq[i] + (P * P).sum(0)
            sa[i] = sa[i] + (Z > 0).double().sum(0)
            if T is not None:
                sxt[i] = sxt[i] + (P * T[:, None]).sum(0)
        cnt += batch
    out = {}
    ET = (sT / cnt) if target_fn is not None else None
    for i in keys:
        mu = (s1[i] / cnt).cpu()
        Sig = (s2[i] / cnt).cpu() - torch.outer(mu, mu)
        Sig = 0.5 * (Sig + Sig.T)
        cov_xt = None
        if target_fn is not None:
            cov_xt = (sxt[i] / cnt).cpu() - mu * ET
        out[i] = {"mu": mu, "Sigma": Sig,
                  "second_moment": (sq[i] / cnt).cpu(),
                  "active_prob": (sa[i] / cnt).cpu(),
                  "cov_xt": cov_xt, "num": cnt}
    out[-1] = {"mu": torch.zeros(n_in), "Sigma": torch.eye(n_in),
               "second_moment": torch.ones(n_in), "active_prob": 0.5 * torch.ones(n_in),
               "cov_xt": None, "num": cnt}
    out["_ET"] = ET
    return out
""")

code(r"""
# ---- whitening utilities ----------------------------------------------------

# Return (S_half, S_inv_half, S, lam) for S = Sigma + lam I, lam = ridge_alpha*tr(Sigma)/n.
# method: 'full' | 'diag' | 'none' (raw, S=I).
def make_whitener(Sigma, ridge_alpha=MAIN_ALPHA, method="full"):
    Sigma = Sigma.double()
    n = Sigma.shape[0]
    lam = float(ridge_alpha) * float(torch.trace(Sigma)) / n
    if method == "none":                         # no whitening (raw Euclidean geometry)
        I = torch.eye(n, dtype=torch.float64)
        return I, I, I, 0.0
    if method == "diag":                         # diagonal-covariance whitening
        d = torch.diag(Sigma).clamp_min(0) + lam
        return torch.diag(d.sqrt()), torch.diag(d.rsqrt()), torch.diag(d), lam
    S = Sigma + lam * torch.eye(n, dtype=torch.float64)
    w, U = torch.linalg.eigh(S)
    w = w.clamp_min(1e-30)
    S_half = (U * w.sqrt()) @ U.T
    S_inv_half = (U * w.rsqrt()) @ U.T
    return S_half, S_inv_half, S, lam

def _sd(t):
    'std that is 0 for a single-element tensor (e.g. the 1-row readout) instead of NaN.'
    return float(t.std()) if t.numel() > 1 else 0.0

def _moments(x):
    x = x.reshape(-1).double()
    m, s = x.mean(), x.std(unbiased=True).clamp_min(1e-30)
    z = (x - m) / s
    return dict(mean=float(m), std=float(s),
                skew=float((z ** 3).mean()),
                excess_kurtosis=float((z ** 4).mean() - 3.0))
""")

code(r"""
# ---- the core per-layer diagnostic -----------------------------------------

# Covariance-adjusted-drift + Gaussian-residual diagnostics for one weight matrix.
# W:(d_out,d_in); mu/Sigma describe the previous post-activation. task_dirs: optional list
# of raw (d_in,) task directions (e.g. Cov(X,T)) added to the structure basis. qhat_override:
# use a supplied whitened unit direction instead of S^{-1/2}mu (for controls). Returns a flat
# dict of scalars plus a few arrays under '_arrays'.
def analyze_weight_layer(W, bias, mu, Sigma, ridge_alpha=MAIN_ALPHA, method="full",
                         task_dirs=None, qhat_override=None):
    W = W.double(); mu = mu.double(); Sigma = Sigma.double()
    d_out, n = W.shape
    S_half, S_inv_half, S, lam = make_whitener(Sigma, ridge_alpha, method)

    q = S_inv_half @ mu
    q_norm = float(q.norm())
    degenerate = q_norm < 1e-9                       # first layer: mu=0 -> no direction
    qhat = (q / q_norm) if not degenerate else torch.zeros(n)
    if qhat_override is not None:
        qhat = qhat_override.double(); degenerate = False

    Wt = W @ S_half                                   # whitened rows
    Wt_fro2 = float((Wt ** 2).sum())
    p = Wt @ qhat                                     # signed whitened mean projection
    a_hat = -math.sqrt(n) * p
    Wmu = W @ mu                                      # raw weight->preact mean contribution

    # Mahalanobis cosine rho_i = (w_i.mu)/sqrt((w_i.S.w_i)(mu.S^{-1}.mu))
    wSw = ((W @ S) * W).sum(1).clamp_min(1e-30)
    tau = float((mu @ torch.linalg.solve(S, mu))) if not degenerate else 0.0
    rho = Wmu / torch.sqrt(wSw * max(tau, 1e-30))

    # raw Euclidean alignment with -mu (for the raw-vs-covadj comparison)
    raw_cos = Wmu / (W.norm(dim=1).clamp_min(1e-30) * float(mu.norm().clamp_min(1e-30)))

    # structure basis B (orthonormal whitened rows): [qhat] (+ whitened task dirs)
    basis = []
    if not degenerate:
        basis.append(qhat)
    if task_dirs:
        for c in task_dirs:
            t = S_inv_half @ c.double()
            if float(t.norm()) > 1e-12:
                basis.append(t / t.norm())
    if basis:
        B = torch.stack(basis, 0)                     # (k, n)
        Q, _ = torch.linalg.qr(B.T)                   # orthonormalise -> (n, k)
        B = Q.T
        C = Wt @ B.T                                  # (d_out, k)
        R_struct = (C @ B)
        Rt = Wt - R_struct
        r2_struct_full = float((R_struct ** 2).sum() / max(Wt_fro2, 1e-30))
        # R2 from qhat alone (first basis vector ~ qhat direction)
        if not degenerate:
            c0 = Wt @ qhat
            r2_qhat = float((c0 ** 2).sum() / max(Wt_fro2, 1e-30))
        else:
            r2_qhat = float("nan")
    else:
        Rt = Wt
        r2_struct_full = 0.0; r2_qhat = float("nan")

    # residual (after removing qhat only) for the Gaussianity tests
    if not degenerate:
        Rt_q = Wt - torch.outer(p, qhat)
    else:
        Rt_q = Wt
    resid_var = float(n * (Rt_q ** 2).sum() / (d_out * max(n - 1, 1)))
    sigma_hat = math.sqrt(max(resid_var, 1e-30)) / math.sqrt(n)   # per-entry std of W̃ residual
    rm = _moments(Rt_q)
    # KS of residual entries vs N(0, std) and QQ slope
    re = Rt_q.reshape(-1).double().numpy()
    ks = float(sstats.kstest((re - re.mean()) / (re.std() + 1e-30), "norm").statistic)
    osm, osr = sstats.probplot(re, dist="norm", fit=False)
    qq_slope = float(np.polyfit(osm, osr, 1)[0]) / (re.std() + 1e-30)

    # singular values of the whitened residual
    sv = np.sort(svd_spectrum(Rt_q)["S"])[::-1]
    row_norm = Rt_q.norm(dim=1).numpy()
    drift_frac = float((p ** 2).sum() / max(Wt_fro2, 1e-30)) if not degenerate else 0.0

    b = bias
    bvals = (b.double() if b is not None else torch.zeros(d_out))
    total_pre = Wmu + bvals

    return {
        "width_in": n, "width_out": d_out, "ridge_alpha": float(ridge_alpha),
        "cov_method": method, "degenerate": bool(degenerate),
        "q_norm": q_norm, "q_norm_over_sqrt_n": q_norm / math.sqrt(n),
        "mean_p": float(p.mean()), "std_p": _sd(p),
        "mean_a_hat": float(a_hat.mean()), "std_a_hat": _sd(a_hat),
        "frac_a_hat_positive": float((a_hat > 0).double().mean()),
        "mean_rho": float(rho.mean()), "std_rho": _sd(rho),
        "frac_rho_negative": float((rho < 0).double().mean()),
        "mean_raw_cos": float(raw_cos.mean()), "frac_raw_cos_negative": float((raw_cos < 0).double().mean()),
        "drift_frobenius_fraction": drift_frac,
        "r2_structure_qhat": r2_qhat, "r2_structure_full": r2_struct_full,
        "residual_variance": resid_var, "sigma_hat": sigma_hat,
        "residual_entry_mean": rm["mean"], "residual_entry_std": rm["std"],
        "residual_entry_skew": rm["skew"], "residual_entry_excess_kurtosis": rm["excess_kurtosis"],
        "residual_qq_slope": qq_slope, "residual_ks_stat": ks,
        "row_norm_mean": float(row_norm.mean()), "row_norm_std": float(row_norm.std()),
        "row_norm_cv": float(row_norm.std() / (row_norm.mean() + 1e-30)),
        "singular_top": float(sv[0]), "singular_median": float(np.median(sv)),
        "singular_top_over_median": float(sv[0] / (np.median(sv) + 1e-30)),
        "weight_fro_norm": float(W.norm()),
        "bias_mean": float(bvals.mean()), "bias_std": float(bvals.std()),
        "mean_weight_contribution_to_preactivation": float(Wmu.mean()),
        "mean_bias_contribution_to_preactivation": float(bvals.mean()),
        "mean_total_preactivation_mean": float(total_pre.mean()),
        "_arrays": {"a_hat": a_hat.numpy(), "p": p.numpy(), "rho": rho.numpy(),
                    "raw_cos": raw_cos.numpy(), "residual_entries": re,
                    "residual_singulars": sv, "row_norm": row_norm,
                    "qhat": (qhat.numpy() if not degenerate else None),
                    "Wt": Wt.numpy()},
    }
""")

code(r"""
# ---- controls --------------------------------------------------------------

def matched_gaussian(W, seed=0):
    'Fresh Gaussian matrix, same shape, matched Frobenius norm.'
    g = torch.Generator().manual_seed(int(seed))
    G = torch.randn(*W.shape, generator=g, dtype=torch.float64)
    return G * (W.norm() / G.norm().clamp_min(1e-30))

# mean âᵢ when projecting whitened rows onto random unit directions (the null band for
# the headline statistic). Returns an array of length n_dirs.
def random_direction_null(W, S_half, n_dirs=200, seed=0):
    Wt = (W.double() @ S_half.double())
    n = Wt.shape[1]
    g = torch.Generator().manual_seed(int(seed))
    out = []
    for _ in range(n_dirs):
        u = torch.randn(n, generator=g, dtype=torch.float64); u = u / u.norm()
        out.append(float((-math.sqrt(n) * (Wt @ u)).mean()))
    return np.array(out)
""")

# =============================================================================
md(r"""## 2 · Models & tasks

We study three targets:

* **zero** — `T(X)=0` (the main case; pretrained checkpoints ship with the repo);
* **halfspace** — `T(X)=1[uᵀX>b]` (a genuine task direction `u`);
* **max** — `T(X)=maxⱼ Xⱼ` (not summarised by one direction).

Zero checkpoints are loaded from the repo. Halfspace / max / a bias-on zero model / a
training-trajectory run are trained here **only if their checkpoints are missing**
(saved under `checkpoints/weight_analysis_checkpoints/`). Training is light — these are tiny MLPs.""")

code(r"""
import glob
from model import MLP, ModelConfig
from tasks import ZeroTask, HalfspaceTask
from training import Trainer, TrainConfig

# A max-input task (not in the infra) defined inline as a pure tasks.Task.
from tasks.base import Task
class MaxTask(Task):
    "Target T(X) = max_j X_j, x ~ N(0, std^2 I)."
    output_dim = 1
    def __init__(self, input_dim, input_std=1.0):
        self.input_dim = input_dim; self.input_std = input_std
    def sample_batch(self, batch_size, device):
        x = torch.randn(batch_size, self.input_dim, device=device) * self.input_std
        y = x.max(dim=1, keepdim=True).values
        return x, y

def _train_if_missing(run_name, task_builder, *, width, depth, bias=False,
                      steps=TRAIN_STEPS, mode="final", every=1000, ckpt_dir=CKPT_DIR):
    "Load {run_name}_final.pt if present else train and checkpoint it. Returns path glob root."
    final = os.path.join(ckpt_dir, f"{run_name}_final.pt")
    if os.path.exists(final):
        return final
    in_dim = width
    out_dim = 1
    task = task_builder(in_dim)
    mcfg = ModelConfig(input_dim=in_dim, hidden_dim=width, depth=depth, output_dim=out_dim,
                       bias=bias, final_bias=bias, activation="relu", seed=SEED)
    tcfg = TrainConfig(steps=steps, seed=SEED, checkpoint_mode=mode, checkpoint_every=every,
                       device=DEVICE)
    Trainer(mcfg.build(), task, tcfg, checkpoint_dir=ckpt_dir, run_name=run_name).train(progress=True)
    return final

# --- registry of models to analyse: list of {model_id, target_type, ckpt, depth, width, bias}
REGISTRY = []

# zero checkpoints already in the repo
for p in sorted(glob.glob(os.path.join(REPO_DIR, "checkpoints", "weight_analysis_checkpoints", "zero_*_final.pt"))):
    m, _ = MLP.load(p)
    REGISTRY.append({"model_id": os.path.basename(p).replace("_final.pt", ""),
                     "target_type": "zero", "ckpt": p, "depth": m.cfg.depth,
                     "width": m.cfg.hidden_dim, "bias": m.cfg.bias})

# halfspace + max at the requested (depth,width) grid
for depth in NEW_DEPTHS:
    for width in NEW_WIDTHS:
        hs = f"halfspace_d{depth}_w{width}_seed{SEED}"
        p = _train_if_missing(hs, lambda n: HalfspaceTask(input_dim=n, offset_std=1.0, seed=SEED),
                              width=width, depth=depth)
        REGISTRY.append({"model_id": hs, "target_type": "halfspace", "ckpt": p,
                         "depth": depth, "width": width, "bias": False})
        mx = f"max_d{depth}_w{width}_seed{SEED}"
        p = _train_if_missing(mx, lambda n: MaxTask(input_dim=n),
                              width=width, depth=depth)
        REGISTRY.append({"model_id": mx, "target_type": "max", "ckpt": p,
                         "depth": depth, "width": width, "bias": False})

# one bias-ON zero model (to see whether biases absorb the negative drift)
_bw = NEW_WIDTHS[0]; _bd = NEW_DEPTHS[-1]
bz = f"zerobias_d{_bd}_w{_bw}_seed{SEED}"
p = _train_if_missing(bz, lambda n: ZeroTask(input_dim=n), width=_bw, depth=_bd, bias=True)
REGISTRY.append({"model_id": bz, "target_type": "zero_bias", "ckpt": p,
                 "depth": _bd, "width": _bw, "bias": True})

print(f"{len(REGISTRY)} models registered:")
for r in REGISTRY:
    print(" ", r["model_id"], "|", r["target_type"], "| d", r["depth"], "w", r["width"], "bias", r["bias"])
""")

code(r"""
# target_fn factory (used for task-direction Cov(X,T)); None for the zero task
def target_fn_for(target_type, model, ckpt_path):
    if target_type in ("zero", "zero_bias"):
        return None
    if target_type == "halfspace":
        # rebuild the exact fixed half-space from its seed (HalfspaceTask is seeded)
        hs = HalfspaceTask(input_dim=model.cfg.input_dim, offset_std=1.0, seed=SEED)
        w, b = hs.w.double(), float(hs.b)
        return lambda x: ((x.double() @ w - b) > 0).double()
    if target_type == "max":
        return lambda x: x.double().max(dim=1).values
    return None
""")

# =============================================================================
md(r"""## 3 · Main result — covariance-adjusted negative-mean drift by layer

Build the master table (one row per model × layer at the main ridge), then plot the
headline statistic `mean âᵢ` and `frac(ρ<0)` by layer for the **zero** models, with the
**init**, **matched-Gaussian** and **random-direction** nulls overlaid.""")

code(r"""
import pandas as pd

# Run the per-layer diagnostics for one registry record across ridge settings and control
# variants. Returns (list_of_row_dicts, arrays_by_(layer,variant)).
def analyze_model(rec, ridge_alphas=(MAIN_ALPHA,), variants=("trained",), num=N_ACT_SAMPLES):
    model, payload = MLP.load(rec["ckpt"]); model = model.double()
    step = payload.get("step", None)
    tfn = target_fn_for(rec["target_type"], model, rec["ckpt"])
    stats_trained = estimate_layer_stats(model, num=num, seed=SEED, target_fn=tfn)

    # init model = rebuild from the SAME config/seed (the t=0 weights), its OWN stats
    init = ModelConfig(**payload["model_config"]).build().double()
    stats_init = estimate_layer_stats(init, num=num, seed=SEED, target_fn=tfn)

    rows, arrays = [], {}
    for ridge in ridge_alphas:
        for role_t, role_i in zip(layer_roles(model), layer_roles(init)):
            name, kind, pk = role_t["name"], role_t["kind"], role_t["prev_key"]
            mu_t, Sig_t = stats_trained[pk]["mu"], stats_trained[pk]["Sigma"]
            mu_i, Sig_i = stats_init[pk]["mu"], stats_init[pk]["Sigma"]
            cov_xt = stats_trained[pk]["cov_xt"]
            task_dirs = [cov_xt] if (cov_xt is not None) else None

            base = {"model_id": rec["model_id"], "target_type": rec["target_type"],
                    "checkpoint_step": step, "layer_name": name, "layer_kind": kind,
                    "num_activation_samples": num}

            for variant in variants:
                if variant == "trained":
                    d = analyze_weight_layer(role_t["W"], role_t["bias"], mu_t, Sig_t,
                                             ridge, task_dirs=task_dirs)
                elif variant == "init":
                    d = analyze_weight_layer(role_i["W"], role_i["bias"], mu_i, Sig_i, ridge)
                elif variant == "matchedG":
                    d = analyze_weight_layer(matched_gaussian(role_t["W"], seed=SEED),
                                             None, mu_t, Sig_t, ridge)
                elif variant == "permuted":
                    perm = torch.randperm(mu_t.numel(), generator=torch.Generator().manual_seed(SEED))
                    d = analyze_weight_layer(role_t["W"], role_t["bias"],
                                             mu_t[perm], Sig_t[perm][:, perm], ridge)
                else:
                    continue
                row = {**base, "variant": variant, **{k: v for k, v in d.items() if k != "_arrays"}}
                rows.append(row)
                if ridge == MAIN_ALPHA:
                    arrays[(name, variant)] = d["_arrays"]
    return rows, arrays

# main table at the main ridge, trained + 3 controls
all_rows, arrays_by_model = [], {}
for rec in REGISTRY:
    rows, arrays = analyze_model(rec, ridge_alphas=(MAIN_ALPHA,),
                                 variants=("trained", "init", "matchedG", "permuted"))
    all_rows += rows
    arrays_by_model[rec["model_id"]] = arrays
df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(TAB_DIR, "main_diagnostics.csv"), index=False)
print("master table:", df.shape, "-> tables/main_diagnostics.csv")
df[df.variant == "trained"][["model_id", "layer_name", "layer_kind", "mean_a_hat",
    "frac_a_hat_positive", "mean_rho", "frac_rho_negative", "drift_frobenius_fraction"]].round(3).head(24)
""")

code(r"""
# Plot 1 — mean â_hat by layer (zero models), trained vs init vs matched-Gaussian + random-dir band
zero_ids = [r["model_id"] for r in REGISTRY if r["target_type"] == "zero"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for mid in zero_ids:
    sub = df[(df.model_id == mid)]
    layers = list(sub[sub.variant == "trained"]["layer_name"])
    xi = range(len(layers))
    axes[0].plot(xi, sub[sub.variant == "trained"]["mean_a_hat"], "-o", label=mid, alpha=.8)
axes[0].axhline(0, color="gray", lw=1)
axes[0].set_title(r"trained: mean $\hat a_i$ by layer (zero models)")
axes[0].set_xlabel("layer (forward order)"); axes[0].set_ylabel(r"mean $\hat a_i$  (>0 = neg. drift)")
axes[0].set_xticks(list(xi)); axes[0].set_xticklabels(layers, rotation=30); axes[0].legend(fontsize=7)

# one representative model: trained vs controls
mid = zero_ids[-1] if zero_ids else REGISTRY[0]["model_id"]
sub = df[df.model_id == mid]
layers = list(sub[sub.variant == "trained"]["layer_name"]); xi = list(range(len(layers)))
for v, mk in [("trained", "-o"), ("init", "--s"), ("matchedG", ":^"), ("permuted", ":x")]:
    axes[1].plot(xi, sub[sub.variant == v]["mean_a_hat"], mk, label=v, alpha=.85)
# random-direction null band for the trained matrices
m, payload = MLP.load([r for r in REGISTRY if r["model_id"] == mid][0]["ckpt"]); m = m.double()
st = estimate_layer_stats(m, num=N_ACT_SAMPLES, seed=SEED)
band = []
for role in layer_roles(m):
    S_half, _, _, _ = make_whitener(st[role["prev_key"]]["Sigma"], MAIN_ALPHA)
    rd = random_direction_null(role["W"], S_half, n_dirs=150, seed=SEED)
    band.append((rd.mean(), rd.std()))
band = np.array(band)
axes[1].fill_between(xi, band[:, 0] - 2 * band[:, 1], band[:, 0] + 2 * band[:, 1],
                     color="gray", alpha=.18, label="random-dir ±2σ")
axes[1].axhline(0, color="gray", lw=1)
axes[1].set_title(f"{mid}: trained vs controls"); axes[1].set_xlabel("layer")
axes[1].set_xticks(xi); axes[1].set_xticklabels(layers, rotation=30); axes[1].legend(fontsize=7)
plt.tight_layout(); savefig(fig, "01_mean_ahat_by_layer.png"); plt.show()
""")

code(r"""
# Plot 2 — Mahalanobis-cosine ρ histograms by layer (a representative zero model)
mid = zero_ids[-1] if zero_ids else REGISTRY[0]["model_id"]
arr = arrays_by_model[mid]
layers = [n for (n, v) in arr.keys() if v == "trained"]
fig, ax = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 3.4), squeeze=False)
for j, name in enumerate(layers):
    rho_t = arr[(name, "trained")]["rho"]
    rho_g = arr[(name, "matchedG")]["rho"]
    ax[0][j].hist(rho_t, bins=30, alpha=.7, density=True, label="trained")
    ax[0][j].hist(rho_g, bins=30, alpha=.5, density=True, label="matched-G")
    ax[0][j].axvline(0, color="r", lw=1)
    ax[0][j].set_title(f"{name}\nmean ρ={rho_t.mean():.3f}", fontsize=9)
    ax[0][j].legend(fontsize=7)
fig.suptitle(f"ρ (Mahalanobis cosine with μ) by layer — {mid}", y=1.03)
plt.tight_layout(); savefig(fig, "02_rho_hist_by_layer.png"); plt.show()
""")

# =============================================================================
md(r"""## 4 · Raw vs covariance-adjusted alignment, and covariance ablations

Per the note up top, the *sign* of the per-row mean-alignment is the same raw or
whitened. Here we (a) compare the **raw** Euclidean cosine with `−μ` to the
**Mahalanobis** cosine `ρ` (does whitening *concentrate / clean up* the signal?), and
(b) sweep the covariance model — raw (no whitening) / diagonal / full — and the ridge
`α` grid, asking whether full/shrinkage covariance makes the negative alignment cleaner
(more consistently negative, lower variance) than raw `μ`.""")

code(r"""
# raw vs covariance-adjusted alignment, middle layers of zero models
mid = zero_ids[-1] if zero_ids else REGISTRY[0]["model_id"]
arr = arrays_by_model[mid]
mids_layers = [n for (n, v) in arr.keys() if v == "trained" and "hidden" in n]
fig, ax = plt.subplots(1, len(mids_layers), figsize=(3.6 * len(mids_layers), 3.4), squeeze=False)
for j, name in enumerate(mids_layers):
    raw = arr[(name, "trained")]["raw_cos"]; rho = arr[(name, "trained")]["rho"]
    ax[0][j].hist(raw, bins=30, alpha=.6, density=True, label=f"raw cos(w,μ) mean={raw.mean():.3f}")
    ax[0][j].hist(rho, bins=30, alpha=.6, density=True, label=f"Mahalanobis ρ mean={rho.mean():.3f}")
    ax[0][j].axvline(0, color="r", lw=1); ax[0][j].set_title(name, fontsize=9); ax[0][j].legend(fontsize=7)
fig.suptitle(f"raw vs covariance-adjusted alignment — {mid}", y=1.03)
plt.tight_layout(); savefig(fig, "03_raw_vs_covadj.png"); plt.show()

# covariance-method + ridge ablation on the headline statistic (one model, middle layers)
rec = [r for r in REGISTRY if r["model_id"] == mid][0]
m, payload = MLP.load(rec["ckpt"]); m = m.double()
st = estimate_layer_stats(m, num=N_ACT_SAMPLES, seed=SEED)
abl = []
for method in ["none", "diag", "full"]:
    for ridge in RIDGE_ALPHAS:
        for role in layer_roles(m):
            if role["kind"] != "middle":
                continue
            d = analyze_weight_layer(role["W"], role["bias"], st[role["prev_key"]]["mu"],
                                     st[role["prev_key"]]["Sigma"], ridge, method=method)
            abl.append({"method": method, "ridge": ridge, "layer": role["name"],
                        "mean_a_hat": d["mean_a_hat"], "frac_rho_negative": d["frac_rho_negative"],
                        "resid_excess_kurtosis": d["residual_entry_excess_kurtosis"],
                        "r2_qhat": d["r2_structure_qhat"]})
abl = pd.DataFrame(abl)
abl.to_csv(os.path.join(TAB_DIR, "covariance_ablation.csv"), index=False)
# The SIGN test (frac ρ<0, sign of mean alignment) is invariant to the whitening, since
# sign(ρ_i)=sign(w_iᵀμ) for any S. What the covariance model changes is the *cleanliness*
# of the residual after removing q̂ — so we ablate that. Lower |excess kurtosis| = cleaner.
print("frac(ρ<0) over middle layers (note: invariant to whitening by construction):",
      round(float(abl["frac_rho_negative"].mean()), 3))
print("\nresidual excess kurtosis (closer to 0 = cleaner Gaussian) by covariance model × ridge α:")
print(abl.groupby(["method", "ridge"])["resid_excess_kurtosis"].mean().unstack("ridge").round(3))
print("\nR² explained by the q̂ drift direction, by covariance model × ridge α:")
print(abl.groupby(["method", "ridge"])["r2_qhat"].mean().unstack("ridge").round(3))
""")

# =============================================================================
md(r"""## 5 · Gaussian residual diagnostics

Remove the `q̂_l` direction from the whitened matrix and ask whether the residual
`R̃ = W̃(I − q̂q̂ᵀ)` looks Gaussian: entry QQ-plot, moments, and singular-value spectrum
against a **matched Gaussian** matrix (same shape, same per-entry variance `σ̂²`). The
question is not whether every test passes perfectly, but whether removing the single
covariance-adjusted-mean direction makes the residual *substantially* more random.""")

code(r"""
mid = zero_ids[-1] if zero_ids else REGISTRY[0]["model_id"]
arr = arrays_by_model[mid]
mids_layers = [n for (n, v) in arr.keys() if v == "trained" and "hidden" in n]
name = mids_layers[-1] if mids_layers else [n for (n, v) in arr.keys() if v == "trained"][0]
A = arr[(name, "trained")]

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
# QQ of residual entries
re = A["residual_entries"]; re = (re - re.mean()) / (re.std() + 1e-30)
osm, osr = sstats.probplot(re, dist="norm", fit=False)
ax[0].plot(osm, np.sort(re), ".", ms=2); lim = [osm.min(), osm.max()]
ax[0].plot(lim, lim, "r-", lw=1); ax[0].set_title(f"residual QQ — {name}")
ax[0].set_xlabel("normal quantiles"); ax[0].set_ylabel("residual quantiles")
# histogram vs normal
ax[1].hist(re, bins=60, density=True, alpha=.7)
xs = np.linspace(re.min(), re.max(), 200)
ax[1].plot(xs, np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi), "r-", lw=1.5)
ax[1].set_title(f"residual entries (skew {_moments(torch.tensor(A['residual_entries']))['skew']:.2f}, "
                f"exkurt {_moments(torch.tensor(A['residual_entries']))['excess_kurtosis']:.2f})")
# singular values vs matched Gaussian
sv = A["residual_singulars"]
d_out = arrays_by_model[mid][(name, "trained")]["row_norm"].shape[0]
n_in = len(A["qhat"]) if A["qhat"] is not None else sv.shape[0]
g = torch.randn(d_out, n_in, generator=torch.Generator().manual_seed(SEED)) * df[
    (df.model_id == mid) & (df.layer_name == name) & (df.variant == "trained")]["sigma_hat"].values[0]
sv_g = np.sort(np.linalg.svd(g.numpy(), compute_uv=False))[::-1]
ax[2].plot(sv, "-o", ms=3, label="residual R̃")
ax[2].plot(sv_g, "--", label="matched Gaussian")
ax[2].set_title("singular values"); ax[2].set_xlabel("index"); ax[2].legend(fontsize=8)
plt.tight_layout(); savefig(fig, "04_residual_gaussianity.png"); plt.show()

print("residual moments by middle layer (trained zero model", mid, "):")
print(df[(df.model_id == mid) & (df.variant == "trained") & (df.layer_kind == "middle")][
    ["layer_name", "residual_entry_skew", "residual_entry_excess_kurtosis",
     "residual_ks_stat", "residual_qq_slope", "drift_frobenius_fraction"]].round(3).to_string(index=False))
""")

# =============================================================================
md(r"""## 6 · Structured-row regression `R²_structure`

How much of the whitened weight matrix is explained by a tiny structure basis `B`?
Zero: `B=[q̂]`. Task models: `B=orth([q̂, t̂])` with the whitened task direction
`t̂ = S^{-1/2} Cov(X,T)`. We report `R²` from `q̂` alone and from the full basis, so the
*incremental* task structure beyond the generic negative-mean drift is visible.""")

code(r"""
rows = []
for rec in REGISTRY:
    model, payload = MLP.load(rec["ckpt"]); model = model.double()
    tfn = target_fn_for(rec["target_type"], model, rec["ckpt"])
    st = estimate_layer_stats(model, num=N_ACT_SAMPLES, seed=SEED, target_fn=tfn)
    for role in layer_roles(model):
        pk = role["prev_key"]
        cov_xt = st[pk]["cov_xt"]
        d = analyze_weight_layer(role["W"], role["bias"], st[pk]["mu"], st[pk]["Sigma"],
                                 MAIN_ALPHA, task_dirs=[cov_xt] if cov_xt is not None else None)
        rows.append({"model_id": rec["model_id"], "target": rec["target_type"],
                     "layer": role["name"], "kind": role["kind"],
                     "R2_qhat": d["r2_structure_qhat"], "R2_full": d["r2_structure_full"]})
r2 = pd.DataFrame(rows); r2.to_csv(os.path.join(TAB_DIR, "structured_regression.csv"), index=False)
print("R²_structure (whitened) — qhat-only vs qhat+task, middle layers:")
print(r2[r2.kind == "middle"].round(3).to_string(index=False))
""")

# =============================================================================
md(r"""## 7 · Training trajectory

Train a zero model with periodic checkpoints and watch the drift grow as the loss
falls: `mean âᵢ` (middle layers), `frac(ρ<0)`, the readout Frobenius norm, and the loss.""")

code(r"""
traj_name = f"zero_traj_d{NEW_DEPTHS[-1]}_w{NEW_WIDTHS[0]}_seed{SEED}"
traj_final = os.path.join(CKPT_DIR, f"{traj_name}_final.pt")
if not glob.glob(os.path.join(CKPT_DIR, f"{traj_name}_step*.pt")):
    _train_if_missing(traj_name, lambda n: ZeroTask(input_dim=n),
                      width=NEW_WIDTHS[0], depth=NEW_DEPTHS[-1],
                      steps=TRAJ_STEPS, mode="periodic", every=TRAJ_EVERY)

import re as _re
def _snap_step(p):
    m = _re.search(r"step(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else 10**9          # the final checkpoint sorts last
snaps = sorted(set(glob.glob(os.path.join(CKPT_DIR, f"{traj_name}_step*.pt")) +
                   ([traj_final] if os.path.exists(traj_final) else [])), key=_snap_step)
traj = []
for p in snaps:
    m, pl = MLP.load(p); m = m.double()
    st = estimate_layer_stats(m, num=max(16384, N_ACT_SAMPLES // 2), seed=SEED)
    mids, rhon = [], []
    for role in layer_roles(m):
        d = analyze_weight_layer(role["W"], role["bias"], st[role["prev_key"]]["mu"],
                                 st[role["prev_key"]]["Sigma"], MAIN_ALPHA)
        if role["kind"] == "middle":
            mids.append(d["mean_a_hat"]); rhon.append(d["frac_rho_negative"])
        if role["kind"] == "final":
            rd_fro = d["weight_fro_norm"]
    step = pl.get("step", 0)
    loss = pl["history"][-1][1] if pl.get("history") else float("nan")
    traj.append({"step": step, "loss": loss, "mean_a_hat_mid": np.mean(mids),
                 "frac_rho_neg_mid": np.mean(rhon), "readout_fro": rd_fro})
traj = pd.DataFrame(traj).sort_values("step"); traj.to_csv(os.path.join(TAB_DIR, "trajectory.csv"), index=False)

fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
ax[0].semilogy(traj.step, traj.loss, "-o"); ax[0].set_title("training loss"); ax[0].set_xlabel("step")
ax[1].plot(traj.step, traj.mean_a_hat_mid, "-o"); ax[1].axhline(0, color="gray", lw=1)
ax[1].set_title(r"mean $\hat a_i$ (middle)"); ax[1].set_xlabel("step")
ax[2].plot(traj.step, traj.frac_rho_neg_mid, "-o"); ax[2].axhline(.5, color="gray", ls="--")
ax[2].set_title("frac(ρ<0) (middle)"); ax[2].set_xlabel("step")
ax[3].plot(traj.step, traj.readout_fro, "-o"); ax[3].set_title("readout Frobenius norm"); ax[3].set_xlabel("step")
plt.tight_layout(); savefig(fig, "05_trajectory.png"); plt.show()
print(traj.round(4).to_string(index=False))
""")

# =============================================================================
md(r"""## 8 · Layerwise activation-prediction check

Gaussian-moment formulas for a row's pre-activation `z ~ N(m, s²)`,
`m = wᵀμ + b`, `s² = wᵀΣw`, give predicted post-activation mean
`E[ReLU(z)] = s φ(r) + m Φ(r)` and active probability `Φ(r)`, `r = m/s`. We compare
these to the **empirical** activation mean / probability — and check that middle layers
of zero models have *more negative* `r` (lower activity) than at init.""")

code(r"""
from scipy.stats import norm as _norm
def predicted_activity(W, bias, mu, Sigma):
    W = W.double(); mu = mu.double(); Sigma = Sigma.double()
    m = W @ mu + (bias.double() if bias is not None else 0.0)
    s = torch.sqrt(((W @ Sigma) * W).sum(1).clamp_min(1e-30))
    r = (m / s).numpy()
    Ephi = _norm.pdf(r); Phi = _norm.cdf(r)
    pred_mean = s.numpy() * Ephi + m.numpy() * Phi
    return r, Phi, pred_mean

mid = zero_ids[-1] if zero_ids else REGISTRY[0]["model_id"]
rec = [r for r in REGISTRY if r["model_id"] == mid][0]
m, payload = MLP.load(rec["ckpt"]); m = m.double()
init = ModelConfig(**payload["model_config"]).build().double()
st_t = estimate_layer_stats(m, num=N_ACT_SAMPLES, seed=SEED)
st_i = estimate_layer_stats(init, num=N_ACT_SAMPLES, seed=SEED)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
# pick a middle hidden layer
roles = layer_roles(m)
mid_role = [r for r in roles if r["kind"] == "middle"][-1]
li = int(mid_role["name"].replace("hidden", ""))
r_t, Phi_t, pred_t = predicted_activity(mid_role["W"], mid_role["bias"],
                                        st_t[mid_role["prev_key"]]["mu"], st_t[mid_role["prev_key"]]["Sigma"])
emp_mean_t = st_t[li]["mu"].numpy(); emp_prob_t = st_t[li]["active_prob"].numpy()
ax[0].plot(pred_t, emp_mean_t, ".", ms=4); lim = [min(pred_t.min(), emp_mean_t.min()), max(pred_t.max(), emp_mean_t.max())]
ax[0].plot(lim, lim, "r-", lw=1); ax[0].set_title(f"{mid_role['name']}: predicted vs empirical post-mean")
ax[0].set_xlabel("predicted E[ReLU(z)]"); ax[0].set_ylabel("empirical mean X⁺")
ax[1].plot(Phi_t, emp_prob_t, ".", ms=4); ax[1].plot([0, 1], [0, 1], "r-", lw=1)
ax[1].set_title("predicted Φ(r) vs empirical P(active)"); ax[1].set_xlabel("Φ(r)"); ax[1].set_ylabel("empirical")
# trained vs init: distribution of r = m/s (more negative = more gated)
mid_role_i = [r for r in layer_roles(init) if r["kind"] == "middle"][-1]
r_i, _, _ = predicted_activity(mid_role_i["W"], mid_role_i["bias"],
                               st_i[mid_role_i["prev_key"]]["mu"], st_i[mid_role_i["prev_key"]]["Sigma"])
ax[2].hist(r_t, bins=30, alpha=.7, density=True, label=f"trained mean r={r_t.mean():.2f}")
ax[2].hist(r_i, bins=30, alpha=.6, density=True, label=f"init mean r={r_i.mean():.2f}")
ax[2].axvline(0, color="r", lw=1); ax[2].set_title("r = m/s (lower = less active)"); ax[2].legend(fontsize=8)
plt.tight_layout(); savefig(fig, "06_activation_prediction.png"); plt.show()
print(f"{mid_role['name']}: mean r trained={r_t.mean():.3f} vs init={r_i.mean():.3f} ; "
      f"mean P(active) trained={emp_prob_t.mean():.3f}")
""")

# =============================================================================
md(r"""## 9 · Are biases doing the same job? (bias-on zero model)

With biases OFF (the default), the whole pre-activation mean is `wᵀμ`. With biases ON,
training could instead push the **bias** negative and leave the weights alone. We
separate `wᵀμ`, `b`, and the total `m = wᵀμ + b` for the bias-on zero model.""")

code(r"""
brec = [r for r in REGISTRY if r["target_type"] == "zero_bias"]
if brec:
    rec = brec[0]; m, payload = MLP.load(rec["ckpt"]); m = m.double()
    st = estimate_layer_stats(m, num=N_ACT_SAMPLES, seed=SEED)
    roles = layer_roles(m)
    fig, ax = plt.subplots(1, len([r for r in roles if r["kind"] != "first"]) or 1,
                           figsize=(4 * max(1, len(roles) - 1), 3.6), squeeze=False)
    j = 0
    for role in roles:
        if role["kind"] == "first":
            continue
        mu = st[role["prev_key"]]["mu"]
        wmu = (role["W"] @ mu).numpy()
        b = (role["bias"].numpy() if role["bias"] is not None else np.zeros_like(wmu))
        ax[0][j].hist(wmu, bins=25, alpha=.6, label=f"wᵀμ ({wmu.mean():.3f})")
        ax[0][j].hist(b, bins=25, alpha=.6, label=f"b ({b.mean():.3f})")
        ax[0][j].hist(wmu + b, bins=25, alpha=.5, label=f"total ({(wmu+b).mean():.3f})")
        ax[0][j].axvline(0, color="r", lw=1); ax[0][j].set_title(role["name"], fontsize=9)
        ax[0][j].legend(fontsize=7); j += 1
    fig.suptitle(f"weight vs bias contribution to pre-activation mean — {rec['model_id']}", y=1.03)
    plt.tight_layout(); savefig(fig, "07_bias_contribution.png"); plt.show()
else:
    print("no bias-on model in registry")
""")

# =============================================================================
md(r"""## 10 · Connection to cumulant propagation (optional)

The whole point of the structure-vs-randomness split is to make trained networks look
more like the *random* networks cumulant propagation assumes. As a sanity probe (if
`Mecha_preds.cumulants` imports), compare the predicted output mean on a zero model from
cumulant propagation on (i) the **trained** weights, (ii) its **init**, and (iii) the
**drift-removed** weights `W_perp` (each row's covariance-adjusted mean component
removed), against a Monte-Carlo reference. The target output mean is ~0 for zero models,
so we report the absolute predicted mean (closer to 0 = better).""")

code(r"""
try:
    from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean
    HAVE_KPROP = True
except Exception as e:
    HAVE_KPROP = False; print("kprop unavailable:", e)

# Return a copy of `model` with each hidden layer's covariance-adjusted mean component
# removed: W_perp = W - (W μ / τ)(S^{-1} μ)ᵀ, so W_perp μ = 0.
def drift_removed_model(model):
    import copy
    m2 = copy.deepcopy(model).double()
    st = estimate_layer_stats(model, num=N_ACT_SAMPLES, seed=SEED)
    for role, layer in zip(layer_roles(model), list(m2.hidden_layers) + [m2.readout]):
        if role["kind"] == "first":
            continue
        mu = st[role["prev_key"]]["mu"]; Sig = st[role["prev_key"]]["Sigma"]
        _, _, S, _ = make_whitener(Sig, MAIN_ALPHA)
        Sinv_mu = torch.linalg.solve(S, mu)
        tau = float(mu @ Sinv_mu)
        if tau < 1e-12:
            continue
        W = role["W"]
        W_perp = W - torch.outer((W @ mu) / tau, Sinv_mu)
        with torch.no_grad():
            layer.weight.copy_(W_perp)
    return m2

if HAVE_KPROP:
    rows = []
    for rec in [r for r in REGISTRY if r["target_type"] == "zero"][:3]:
        model, _ = MLP.load(rec["ckpt"]); model = model.double()
        init = ModelConfig(**MLP.load(rec["ckpt"])[1]["model_config"]).build().double()
        mc, _stats = estimate_empirical_mean(model=model, input_dim=model.cfg.input_dim,
                                              num_samples=100_000)
        def absmean(mod):
            return float(np.abs(np.atleast_1d(run_cumulants(mod, config={"k_max": 3})["mean"])).mean())
        rows.append({"model_id": rec["model_id"],
                     "MC_|mean|": float(np.abs(np.atleast_1d(mc)).mean()),
                     "kprop_trained": absmean(model),
                     "kprop_init": absmean(init),
                     "kprop_drift_removed": absmean(drift_removed_model(model))})
    kp = pd.DataFrame(rows); kp.to_csv(os.path.join(TAB_DIR, "kprop_compare.csv"), index=False)
    print("|predicted output mean| (zero target -> smaller is better):")
    print(kp.round(6).to_string(index=False))
""")

# =============================================================================
md(r"""## 11 · Summary table & success criteria

A compact pass / fail / ambiguous read of the headline predictions, plus a save of all
tables and figures. The key positive result is `mean âᵢ > 0` in **middle** layers of
**zero** models, near-zero in the **first** layer and at **init**, with the readout
**shrinking** rather than gating.""")

code(r"""
def verdict(val, lo, hi):
    if val > hi:  return "PASS"
    if val < lo:  return "FAIL"
    return "ambig"

# extra middle-layer structure from the task direction (from §6's r2 table)
_r2mid = (r2[r2.kind == "middle"].assign(extra=lambda d: d.R2_full - d.R2_qhat)
          .groupby("model_id")["extra"].mean().to_dict())

summary = []
for rec in REGISTRY:
    mid_id = rec["model_id"]
    sub = df[(df.model_id == mid_id) & (df.variant == "trained")]
    init = df[(df.model_id == mid_id) & (df.variant == "init")]
    mids = sub[sub.layer_kind == "middle"]
    first = sub[sub.layer_kind == "first"]
    final = sub[sub.layer_kind == "final"]
    mid_a = float(mids["mean_a_hat"].mean()) if len(mids) else float("nan")
    init_mid_a = float(init[init.layer_kind == "middle"]["mean_a_hat"].mean()) if len(init) else float("nan")
    is_zero = rec["target_type"] in ("zero", "zero_bias")
    summary.append({
        "model_id": mid_id, "target": rec["target_type"],
        "mid_mean_a_hat": round(mid_a, 3),
        "mid_frac_rho_neg": round(float(mids["frac_rho_negative"].mean()), 3) if len(mids) else None,
        "init_mid_mean_a_hat": round(init_mid_a, 3),
        "first_mean_a_hat": round(float(first["mean_a_hat"].mean()), 3) if len(first) else None,
        "readout_fro": round(float(final["weight_fro_norm"].mean()), 4) if len(final) else None,
        "mid_resid_exkurt": round(float(mids["residual_entry_excess_kurtosis"].mean()), 3) if len(mids) else None,
        "mid_R2_task_extra": round(_r2mid.get(mid_id, float("nan")), 3),
        "drift_present(zero)": (verdict(mid_a, 0.0, 0.02) if is_zero else "n/a(task)"),
        "absent_at_init": (("PASS" if abs(init_mid_a) < 0.02 else "ambig")
                           if init_mid_a == init_mid_a else "n/a"),
    })
summary = pd.DataFrame(summary)
summary.to_csv(os.path.join(TAB_DIR, "summary.csv"), index=False)
print(summary.to_string(index=False))
print("\nLegend: drift_present checks middle-layer mean âᵢ>0 (covariance-adjusted zero-")
print("suppression drift) — the core prediction for ZERO models. Task models (halfspace/max)")
print("are not expected to show it; they carry task structure instead, read off mid_R2_task_extra")
print("(extra whitened-weight variance from the task direction beyond q̂).")
""")

code(r"""
# bundle results for easy hand-off (figures + tables + new checkpoints list)
import shutil
manifest = os.path.join(RESULTS_DIR, "MANIFEST.txt")
with open(manifest, "w") as f:
    f.write("weight_structure results\n")
    f.write("figures:\n");  [f.write(f"  {p}\n") for p in sorted(glob.glob(os.path.join(FIG_DIR, '*.png')))]
    f.write("tables:\n");   [f.write(f"  {p}\n") for p in sorted(glob.glob(os.path.join(TAB_DIR, '*.csv')))]
    f.write("new checkpoints:\n"); [f.write(f"  {p}\n") for p in sorted(glob.glob(os.path.join(CKPT_DIR, '*.pt')))]
zip_base = os.path.join(REPO_DIR, "results", "weight_structure_bundle")
shutil.make_archive(zip_base, "zip", RESULTS_DIR)
print("wrote", manifest)
print("bundle:", zip_base + ".zip")
try:
    import google.colab; from google.colab import files  # noqa
    # files.download(zip_base + ".zip")   # uncomment to auto-download in Colab
    print("(in Colab: files.download('%s.zip') to pull the bundle)" % zip_base)
except Exception:
    pass
""")

md(r"""### How to read the results

* **§3 main plot** — the load-bearing one. Middle-layer `mean âᵢ` should sit **above 0**
  for trained zero models, hug **0** for `init` / `matched-Gaussian` / `permuted` and stay
  inside the **random-direction band**; the first layer (`μ₀=0`) should be ~0 and the
  readout should not show the same gating drift.
* **§4** — if the Mahalanobis `ρ` is *more tightly* negative than raw `cos(w,μ)`, covariance
  adjustment is cleaning the signal; the ablation table says which covariance model / ridge
  is sharpest. (Recall the *sign* is identical raw vs whitened — this is about concentration.)
* **§5** — residual skew/kurtosis near 0, QQ near the diagonal, and residual singular values
  tracking the matched-Gaussian curve ⇒ what's left after removing `q̂` is plausibly random.
* **§6** — `R²_qhat` is the share explained by the single drift direction; for task models
  `R²_full − R²_qhat` is the *extra* structure from the task direction.
* **§7** — the drift should **grow as the loss falls**, and the readout norm should shrink.
* **§9** — if biases carry the negative pre-activation mean, `b`'s histogram sits left of 0
  and `wᵀμ` stays centered (the weight-mean mechanism is replaced by a bias mechanism).""")

# =============================================================================
nb.save(os.path.join(os.path.dirname(__file__), "weight_structure_vs_randomness.ipynb"))

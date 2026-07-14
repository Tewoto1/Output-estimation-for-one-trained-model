"""Generates kprop_failure_analysis_colab.ipynb (valid nbformat-4 JSON).

Question: as the REQUIRED training accuracy rises (lower loss tolerance), kprop's
prediction degrades for WIDE models specifically, while staying usable at low width.
Why? This notebook LOADS the saved train-to-tolerance checkpoints (no retraining --
the recycling rule) and dissects them, testing three suspicions:

  H1 (low-rank / -mu): training installs a low-rank update whose rows all point in
      the SAME direction, -mu (mu = mean post-activation of the previous layer) --
      the structure already proven for frozen readouts in the Q2 study.
  H2 (training has stopped): once trained slightly past the tolerance the weights
      barely move any more, because the converged net no longer props gradients back
      (dead ReLUs / zeroed outputs) -- so "more accuracy" = more of the SAME structure.
  H3 (width dependence): the structure (and the kprop failure) grows with width;
      at low width the net stays comparatively alive/Gaussian, so kprop keeps working.

Inputs: checkpoints/kprop_exact_checks/kprop-zero_d3_w{16..2048}_tol5_seed{3..6}_final.pt
(d3, no bias, output_dim 128, trained until MSE < 1e-5, lr 1e-4, per-step tol checks).
Optionally extends each model to tol 1e-6 / 1e-7 (recycled as _tol6/_tol7) to measure
the across-tolerance axis directly. Structure metrics come from
analysis.Tools.weight_structure (weight_structure_metrics, layer_stats, grad_flow);
kprop is called as a black box via Mecha_preds.cumulants.run_cumulants (needs Py>=3.12
+ scipy; that section is guarded and skippable -- all structure sections are torch-only).

Run:  python "experiments/kprop_failure_analysis/build_kprop_failure_analysis_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

# =============================================================================
md(r"""# Why does kprop fail on WIDE trained-to-tolerance models (but not narrow ones)?

**The observation.** Cumulant propagation (k=2) predicts the output mean of these nets
well at init, but as the required training accuracy rises the prediction degrades —
and the degradation is concentrated at **large width**. Low-width models stay
kprop-predictable.

**Three suspicions, each measured directly on the saved checkpoints:**

| | hypothesis | measurement | section |
|---|---|---|---|
| **H1** | training installs a **low-rank** update whose rows all point along **−μ** (μ = mean previous-layer post-activation) | stable rank of `W` and of `ΔW = W − W_init`; `cos(rows, −μ)`; `|cos(v₁, μ)|`; μ-energy | §4–5 |
| **H2** | slightly more training **doesn't change the weights** — the converged net stops propping gradients back | per-layer `‖W_tol6 − W_tol5‖/‖W_tol5‖`; per-layer gradient scale `‖∇W‖/‖W‖`; dead-ReLU fractions | §3, §6 |
| **H3** | the structure (and kprop's failure) **grows with width**; narrow nets stay alive/Gaussian | every metric above plotted vs width; kprop error vs width per tolerance; error-vs-structure scatter | §5–7 |

The mechanistic chain being tested: train-to-zero drives rows toward −μ ⇒ pre-activations
go negative ⇒ ReLUs die ⇒ (a) gradients stop flowing (weights freeze — H2), and
(b) the activation distribution becomes a point-mass-at-0 mixture which a single-Gaussian
k=2 state cannot represent ⇒ kprop fails. If wide nets reach this state faster/harder at
the same loss tolerance (H3), that explains the width pattern.

Background from the Q2 study (frozen-identity readout, d2): the pre-readout matrix
provably develops a **rank-1, −μ-aligned spike**. Here the readout is **trainable**
(d3, output_dim 128), so the net has more escape routes — *which* layer carries the
structure is exactly what §4–5 establishes.

Everything LOADS the existing checkpoints (`checkpoints/kprop_exact_checks/`); nothing
retrains unless you opt into the §3 tolerance extension (which is itself checkpointed +
recycled as `_tol6`/`_tol7`).
""")

# --- Setup -------------------------------------------------------------------
md(r"""## 0. Setup — locate the repo
Structure sections need only torch/numpy/pandas/matplotlib. §7 (kprop) additionally
needs **Python ≥ 3.12 + scipy** and is skipped automatically otherwise.
""")
code(BOOTSTRAP_CELL)

# --- Config ------------------------------------------------------------------
md(r"""## 1. Config — **the knobs live HERE** (probe in place)

Checkpoints are discovered from `CKPT_DIR` (whatever widths/seeds/tol tags exist).
Analysis runs in **float64** (models are cast after loading); any §3 extension
*training* runs in float32 via the Trainer, as everywhere in this repo.
""")
code(r"""
import copy, math, sys
import torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
torch.set_default_dtype(torch.float64)        # analysis dtype

import experiments as E
from model import MLP, ModelConfig
from tasks import ZeroTask
from training import TrainConfig
from analysis import weight_structure_metrics, layer_stats, grad_flow

DEVICE   = str(E.DEVICE)
QUICK    = E.QUICK                  # CPU-only machine -> trim the sweep
CKPT_DIR = "checkpoints/kprop_exact_checks"   # THIS notebook reads the kprop study ckpts
PREFIX   = "kprop-zero"
DEPTH    = 3
BASE_TOL = 5                        # the tol tag the sweep was trained to (1e-5)

# ---- optional across-tolerance axis (H2): extend tol5 -> tol6 -> tol7 --------
RUN_EXTEND  = True                  # False = pure loading, zero training
EXTEND_TOLS = [6, 7]                # trained as _tol6/_tol7 ckpts, recycled on rerun
EXT_LR, EXT_PATIENCE, EXT_MAX_STEPS = 1e-4, 25, 200_000

# ---- analysis budgets ---------------------------------------------------------
MU_SAMPLES   = 20_000 if QUICK else 200_000   # MC for per-layer mu / active fractions
GRAD_BATCHES = 4 if QUICK else 8              # batches for the gradient-scale probe
DEAD_THRESH  = 1e-3                           # unit is "dead" if P(pre>0) < this

# ---- discover what exists on disk --------------------------------------------
recs = [r for r in E.list_checkpoints(CKPT_DIR)
        if r.get("prefix") == PREFIX and r.get("depth") == DEPTH
        and r.get("tol") == BASE_TOL and r.get("tag") == "final"]
WIDTHS = sorted({r["width"] for r in recs})
SEEDS  = sorted({r["seed"] for r in recs})
if QUICK:
    WIDTHS = [w for w in WIDTHS if w in (16, 64, 256)] or WIDTHS[:3]
    SEEDS  = SEEDS[:2]
assert recs, f"no {PREFIX}_d{DEPTH}_*_tol{BASE_TOL}_* checkpoints in {CKPT_DIR}"
print(f"found tol{BASE_TOL} ckpts | widths {WIDTHS} | seeds {SEEDS} | device {DEVICE}")
""")

# --- Load --------------------------------------------------------------------
md(r"""## 2. Load the checkpoints (+ matched inits) and summarize convergence

`init` controls are rebuilt from each checkpoint's own `ModelConfig` (the seed is
stored in it), so every trained net is compared against ITS OWN starting point.
`MODELS[(tag, width, seed)]` is the master dict all later sections read;
tags are `"init"`, `5`, and (after §3) `6`, `7`.
""")
code(r"""
MODELS, CONV = {}, []
for w in WIDTHS:
    for s in SEEDS:
        path = E.ckpt_path(CKPT_DIR, E.run_name(PREFIX, depth=DEPTH, width=w,
                                                tol=BASE_TOL, seed=s))
        m, payload = MLP.load(path)
        MODELS[(BASE_TOL, w, s)] = m.double().to(DEVICE)
        MODELS[("init", w, s)] = ModelConfig(**payload["model_config"]).build() \
                                     .double().to(DEVICE)
        hist = payload.get("history") or []
        CONV.append(dict(width=w, seed=s, steps=payload.get("step"),
                         final_loss=payload.get("final_loss"),
                         start_loss=hist[0][1] if hist else float("nan")))
dfc = pd.DataFrame(CONV)
print(dfc.groupby("width")[["steps", "final_loss"]].median().rename(
    columns={"steps": "median steps to tol", "final_loss": "median final loss"}))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
gm = dfc.groupby("width")
ax[0].errorbar(gm["steps"].median().index, gm["steps"].median(),
               yerr=gm["steps"].std(), fmt="o-")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
ax[0].set_xlabel("width"); ax[0].set_ylabel("steps to reach 1e-5")
ax[0].set_title("wider = converges in fewer steps"); ax[0].grid(alpha=0.3)
ax[1].scatter(dfc.width, dfc.final_loss, s=18)
ax[1].axhline(10.0**-BASE_TOL, color="crimson", ls="--", lw=1)
ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
ax[1].set_xlabel("width"); ax[1].set_ylabel("loss at stop")
ax[1].set_title("all widths stopped AT the tolerance?"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

# --- Extend ------------------------------------------------------------------
md(r"""## 3. (Optional) the across-tolerance axis: train on to 1e-6, 1e-7 — H2 part 1

Each tol5 model continues training until the next tolerance (same regime: lr 1e-4,
per-step checks, patience). Saved/recycled as `_tol6`/`_tol7`, so this cell costs
nothing on re-run. **The H2 readout is the table below:** if the converged net no
longer props gradients back, `‖W_next − W_prev‖/‖W_prev‖` should be tiny — and
*shrink* with width.
""")
code(r"""
if RUN_EXTEND:
    prev_tag = BASE_TOL
    for tol in EXTEND_TOLS:
        cfg = TrainConfig(steps=EXT_MAX_STEPS, batch_size=1024, lr=EXT_LR,
                          optimizer="adamw", loss_tol=10.0**-tol, tol_check_every=1,
                          tol_patience=EXT_PATIENCE, checkpoint_mode="final",
                          log_every=50, device=DEVICE, dtype="float32")
        for w in WIDTHS:
            paths  = [E.ckpt_path(CKPT_DIR, E.run_name(PREFIX, depth=DEPTH, width=w,
                                                       tol=tol, seed=s)) for s in SEEDS]
            builds = [(lambda w=w, s=s, p=prev_tag: copy.deepcopy(MODELS[(p, w, s)]))
                      for s in SEEDS]
            outs = E.get_or_train_many(paths, builds,
                       task=ZeroTask(input_dim=w, output_dim=MODELS[(BASE_TOL, w, SEEDS[0])].cfg.output_dim),
                       train_cfg=cfg, extra_meta={"extended_from": f"tol{prev_tag}"},
                       map_location=DEVICE, progress=False)
            for s, (m2, payload, loaded) in zip(SEEDS, outs):
                MODELS[(tol, w, s)] = m2.double().to(DEVICE)
                print(f"  w{w:>5} s{s} tol1e-{tol}: {'[loaded]' if loaded else '[trained]'} "
                      f"loss={E.final_loss(payload):.2e}")
        prev_tag = tol

    print("\nH2: relative weight movement per tolerance step (median over seeds)")
    move = []
    tags = [BASE_TOL] + EXTEND_TOLS
    for a, b in zip(tags[:-1], tags[1:]):
        for w in WIDTHS:
            for s in SEEDS:
                ma, mb = MODELS[(a, w, s)], MODELS[(b, w, s)]
                for (name, Wa), (_, Wb) in zip(ma.named_weights(), mb.named_weights()):
                    Wa, Wb = Wa.detach().cpu(), Wb.detach().cpu()
                    move.append(dict(step=f"tol{a}->tol{b}", width=w, seed=s, layer=name,
                                     rel_move=float((Wb - Wa).norm() / (Wa.norm() + 1e-30))))
    dfm = pd.DataFrame(move)
    piv = dfm.groupby(["step", "layer", "width"])["rel_move"].median().unstack("width")
    print(piv.round(5))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for (stp, lay), sub in dfm.groupby(["step", "layer"]):
        mw = sub.groupby("width")["rel_move"].median()
        ax.plot(mw.index, mw.values, "o-" if "5->" in stp else "s--",
                label=f"{lay} {stp}", alpha=0.8)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width"); ax.set_ylabel("||W_next - W_prev|| / ||W_prev||")
    ax.set_title("H2: more accuracy barely moves the weights (and less so when wide?)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.show()
else:
    print("RUN_EXTEND=False: skipping (H2 weight-movement test needs it)")
""")

# --- Master metric loop --------------------------------------------------------
md(r"""## 4. The master measurement loop — every model, every layer, one tidy table

Per `(tag, width, seed)` and per weight matrix:

- **ΔW vs init**: `rel_change = ‖W − W₀‖/‖W₀‖`, stable rank of `ΔW`, top-σ energy of
  `ΔW` (low-rankness of what training *added*);
- **−μ alignment** (for matrices fed by a post-activation, i.e. hidden1, hidden2,
  readout): `weight_structure_metrics(W, μ_prev)` → `cos_neg_mu`, `align_v1_mu`,
  `rank1_energy_mu`, `stable_rank`; the same metrics on `ΔW` isolate the trained part;
- **aliveness**: per-layer dead-unit fraction and post-activation RMS (`layer_stats`);
- **gradient flow**: `‖∇W‖/‖W‖` per layer at the checkpoint (`grad_flow`).
""")
code(r"""
import os
rows = []
for (tag, w, s), m in sorted(MODELS.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
    m0 = MODELS[("init", w, s)]
    ls = layer_stats(m, n=MU_SAMPLES, dead_thresh=DEAD_THRESH)
    gf = grad_flow(m, ZeroTask(input_dim=w, output_dim=m.cfg.output_dim),
                   n_batches=GRAD_BATCHES)
    pairs = list(zip(m.named_weights(), m0.named_weights()))
    for li, ((name, W), (_, W0)) in enumerate(pairs):
        W, W0 = W.detach().cpu().numpy(), W0.detach().cpu().numpy()
        dW = W - W0
        trained = np.linalg.norm(dW) > 1e-12           # init rows: dW == 0, skip dW metrics
        row = dict(tag=str(tag), width=w, seed=s, layer=name,
                   rel_change=float(np.linalg.norm(dW) / (np.linalg.norm(W0) + 1e-30)),
                   grad_scale=gf[name])
        if trained:
            _, Sd, _ = np.linalg.svd(dW)
            row["dW_stable_rank"] = float((Sd**2).sum() / (Sd[0]**2 + 1e-300))
            row["dW_top_energy"]  = float(Sd[0]**2 / ((Sd**2).sum() + 1e-300))
        if 1 <= li <= m.cfg.depth:                     # fed by post-activation li-1
            mu = ls[li - 1]["mu"]
            row.update({f"W_{k}": v for k, v in weight_structure_metrics(W, mu).items()
                        if k != "width"})
            if trained:
                row.update({f"dW_{k}": v for k, v in weight_structure_metrics(dW, mu).items()
                            if k != "width"})
            row["prev_dead_frac"] = ls[li - 1]["dead_frac"]
            row["prev_post_rms"]  = ls[li - 1]["post_rms"]
        if li < m.cfg.depth:                           # this layer's own aliveness
            row["dead_frac"] = ls[li]["dead_frac"]
            row["post_rms"]  = ls[li]["post_rms"]
        rows.append(row)
    print(f"  measured tag={tag!s:>4} w={w:<5} s={s}")
df = pd.DataFrame(rows)
os.makedirs("results/kprop_failure_analysis", exist_ok=True)
df.to_csv("results/kprop_failure_analysis/structure_metrics.csv", index=False)
df.head(6)
""")

# --- H1 plots ------------------------------------------------------------------
md(r"""## 5. H1 — the low-rank −μ story, layer by layer, across width

Top row: what training ADDED (`ΔW`): its relative size, how low-rank it is, and how
much of it lies along μ. Bottom row: the −μ alignment of the full matrix and the
aliveness of the layer feeding it. `init` (gray) is the no-structure baseline.

Reading guide: H1 holds where `dW_top_energy → 1`, `dW_align_v1_mu → 1`, and
`W_cos_neg_mu > 0`. H3 holds if those curves RISE with width while init stays flat.
""")
code(r"""
LAYERS = [l for l in df.layer.unique() if l != "hidden0"]   # fed by post-activations
TAGS   = [t for t in df.tag.unique() if t != "init"]
tagcol = {t: cm.plasma(i / max(1, len(TAGS) - 1)) for i, t in enumerate(TAGS)}

def med(sub, col): return sub.groupby("width")[col].median()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
panels = [("rel_change",      "||dW||/||W0||  (how much training changed)", 0),
          ("dW_top_energy",   "top-sigma energy of dW  (1 = rank-1 update)", 1),
          ("dW_align_v1_mu",  "|cos(v1(dW), mu)|  (update points along mu?)", 2),
          ("W_cos_neg_mu",    "mean cos(rows W, -mu)  (rows -> -mu?)", 3),
          ("W_stable_rank",   "stable rank of W  (down = spike forming)", 4),
          ("prev_dead_frac",  "dead fraction of the FEEDING layer", 5)]
for colname, title, k in panels:
    ax = axes[k // 3, k % 3]
    if colname not in df.columns:
        ax.set_title(title + " (n/a)", fontsize=9); continue
    for lay in LAYERS:
        for t in TAGS:
            sub = df[(df.tag == t) & (df.layer == lay)].dropna(subset=[colname])
            if len(sub) == 0: continue
            mm = med(sub, colname)
            ax.plot(mm.index, mm.values, "o-", color=tagcol[t],
                    alpha={"hidden1": .45, "hidden2": .8, "readout": 1.0}[lay],
                    lw={"hidden1": 1, "hidden2": 1.6, "readout": 2.2}[lay],
                    label=f"{lay} tol{t}")
        subi = df[(df.tag == "init") & (df.layer == lay)].dropna(subset=[colname])
        if len(subi):
            mi = med(subi, colname)
            ax.plot(mi.index, mi.values, ":", color="gray", lw=1)
    ax.set_xscale("log", base=2); ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    if colname in ("rel_change", "W_stable_rank"): ax.set_yscale("log")
axes[0, 0].legend(fontsize=6, ncol=2)
for a in axes[1]: a.set_xlabel("width")
plt.suptitle("H1/H3: low-rank -mu structure vs width (gray dotted = init baseline)")
plt.tight_layout(); plt.show()
""")

# --- H2 plots ------------------------------------------------------------------
md(r"""## 6. H2 — "nothing props back": gradient scale and aliveness at the checkpoint

If the converged solution kills backprop, `‖∇W‖/‖W‖` collapses (more with width, more
with tolerance) and dead fractions / shrinking `post_rms` show *how*: rows along −μ
push pre-activations negative, ReLUs die, and the chain rule has nothing to carry.
The readout's `prev_post_rms` panel shows the alternative escape route — the net can
also just hand ~0 activations to the readout.
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for lay in df.layer.unique():
    for t in TAGS:
        sub = df[(df.tag == t) & (df.layer == lay)]
        if len(sub) == 0: continue
        mm = sub.groupby("width")["grad_scale"].median()
        axes[0].plot(mm.index, mm.values, "o-", color=tagcol[t], alpha=0.7,
                     label=f"{lay} tol{t}" if lay == "hidden2" else None)
si = df[df.tag == "init"].groupby("width")["grad_scale"].median()
axes[0].plot(si.index, si.values, ":", color="gray", lw=1.5, label="init")
axes[0].set_yscale("log"); axes[0].set_title("gradient scale ||grad W||/||W||")

for t in TAGS + ["init"]:
    sub = df[(df.tag == str(t)) & df.dead_frac.notna()]
    mm = sub.groupby("width")["dead_frac"].median()
    axes[1].plot(mm.index, mm.values, "o-" if t != "init" else ":",
                 color=tagcol.get(t, "gray"), label=f"tol{t}" if t != "init" else "init")
axes[1].set_title(f"dead-unit fraction (P(pre>0) < {DEAD_THRESH:g})")

for t in TAGS + ["init"]:
    sub = df[(df.tag == str(t)) & (df.layer == "readout")]
    mm = sub.groupby("width")["prev_post_rms"].median()
    axes[2].plot(mm.index, mm.values, "o-" if t != "init" else ":",
                 color=tagcol.get(t, "gray"), label=f"tol{t}" if t != "init" else "init")
axes[2].set_yscale("log"); axes[2].set_title("RMS of the activations feeding the readout")

for ax in axes:
    ax.set_xscale("log", base=2); ax.set_xlabel("width")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

# --- kprop ---------------------------------------------------------------------
md(r"""## 7. kprop error across width × tolerance — and what it tracks (H3)

Runs the REAL algorithm (`run_cumulants`, k=2 approx) on every loaded model against a
Monte-Carlo reference. Left: the failure pattern itself (relative debiased error vs
width, one curve per tolerance, init as control). Right: the **why** — kprop error
scattered against the two structure metrics; if points line up, the structure IS the
failure mode. Needs Python ≥ 3.12 + scipy; skipped otherwise.
""")
code(r"""
KPROP_OK = sys.version_info >= (3, 12)
try:
    import scipy  # noqa
except Exception:
    KPROP_OK = False
MC_SAMPLES = 100_000 if QUICK else 1_000_000

if KPROP_OK:
    from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean
    KCFG = {"k_max": 2, "kind": "simple", "use_avg_metric": False, "factor": False,
            "use_pK": True, "output_d_max": 1, "exact_relu_cov": False}
    kp = []
    for (tag, w, s), m in sorted(MODELS.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        cp = run_cumulants(m, w, KCFG, device=DEVICE)["mean"]
        mc, st = estimate_empirical_mean(model=m, input_dim=w, num_samples=MC_SAMPLES,
                                         batch_size=min(65536, max(8192, (1 << 26) // w)),
                                         device=DEVICE, dtype=torch.float64)
        se = np.asarray(st["mc_stderr"]).reshape(-1)
        meas = float(np.mean((np.asarray(cp).reshape(-1) - np.asarray(mc).reshape(-1))**2))
        rms = math.sqrt(max(meas - float(np.mean(se**2)), 0.0))
        kp.append(dict(tag=str(tag), width=w, seed=s,
                       rel_err=rms / (st["empirical_output_rms"] + 1e-30)))
        print(f"  kprop tag={tag!s:>4} w={w:<5} s={s}  rel_err={kp[-1]['rel_err']:.3e}")
    dfk = pd.DataFrame(kp)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for t in sorted(dfk.tag.unique()):
        mm = dfk[dfk.tag == t].groupby("width")["rel_err"].median()
        axes[0].plot(mm.index, mm.values, "o-" if t != "init" else ":",
                     color=tagcol.get(t, "gray") if t != "init" else "gray", label=f"tol{t}" if t != "init" else "init")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("width"); axes[0].set_ylabel("kprop rel. debiased error")
    axes[0].set_title("the failure: error vs width, per tolerance"); axes[0].legend(fontsize=8)

    j = dfk.merge(df[(df.layer == "hidden2")],
                  on=["tag", "width", "seed"], how="left", suffixes=("", "_m"))
    for ax, xcol, ttl in [(axes[1], "prev_dead_frac", "vs dead fraction (feeding hidden2)"),
                          (axes[2], "W_cos_neg_mu", "vs -mu alignment of hidden2 rows")]:
        sc = ax.scatter(j[xcol], j.rel_err, c=np.log2(j.width), cmap="viridis", s=22)
        ax.set_yscale("log"); ax.set_xlabel(xcol); ax.set_title("kprop error " + ttl, fontsize=9)
        plt.colorbar(sc, ax=ax, label="log2 width")
    for ax in axes: ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
else:
    print("kprop section skipped: needs Python >= 3.12 and scipy "
          f"(this kernel: {sys.version.split()[0]})")
""")

# --- Interpretation -------------------------------------------------------------
md(r"""## 8. How to read the answer

**H1 confirmed** looks like: `dW_top_energy ≈ 1` and `dW_align_v1_mu ≈ 1` for the
late layers (hidden2 ± readout) — training added a rank-1 −μ spike — with
`W_cos_neg_mu > 0` rising above the init baseline.

**H2 confirmed** looks like: §3's `rel_move` per extra tolerance decade is ≪ the
tol5 `rel_change`, and §6's gradient scale collapses by orders of magnitude vs init —
the weights have settled because dead ReLUs / zeroed activations stop backprop. More
required accuracy then just *deepens the same structure* instead of finding new ones.

**H3 / the width question** — why kprop keeps working at low width:
- if §6 shows **dead fractions and −μ alignment grow with width** (wide nets can
  afford to kill units; the rank-1/width footprint of the spike shrinks, so the loss
  barely notices), the wide activation state is maximally non-Gaussian
  (point mass at 0) → the single-Gaussian k=2 state misses it → §7's error rises
  with width *and* its scatter panels line up with the structure metrics;
- narrow nets reaching the SAME loss 1e-5 must instead keep most units alive and
  spread the suppression across many directions (`dW_stable_rank` high, dead
  fraction low) → activations stay near-Gaussian → kprop keeps working;
- if instead the §7 scatter does NOT track structure, the failure is in the k=2
  moment closure itself (training-induced cross-unit correlations) — the remedy is
  k≥3, not a different covariance.

Caveats: μ is an MC estimate (raise `MU_SAMPLES` if `cos` values look noisy);
`grad_flow` uses the float64 analysis copy (scale, not exact optimizer steps);
tol6/tol7 conclusions only exist if `RUN_EXTEND=True` ran.
""")

nb.save(os.path.join(os.path.dirname(__file__), "kprop_failure_analysis_colab.ipynb"))

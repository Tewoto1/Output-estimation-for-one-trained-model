"""Generates frozen_readout_weight_structure_colab.ipynb (valid nbformat-4 JSON).

Q2 -- "When the final (readout) weights are frozen and the net is trained to output 0,
what happens to the pre-readout weight matrix?"  Three candidate stories:

  (a) the entries just become MORE NEGATIVE,
  (b) the rows ALIGN to the direction of  -mean(previous-layer post-activation)  ( -mu ),
  (c) and is THAT what the emergent low-rank structure actually is?

The generated notebook trains (or RELOADS -- see experiments.get_or_train) small
depth-2 frozen-identity-readout MLPs across widths, then dissects the last hidden
weight matrix W. All knobs, naming, and the recycling rule come from experiments.py;
the metrics come from analysis.Tools.weight_structure (nothing is re-defined inline).

Key fix vs the earlier run: depth 2, ordinary batch sizes, and Adam so the nets
actually reach ~0 output. The earlier depth-3 SGD runs stalled at loss ~1e-3 and
show NO structure -- that is undertraining, not absence of the effect.

Run:  python "experiments/noiseless_and_frozen_readout/build_frozen_readout_structure_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()

# =============================================================================
nb.md(r"""# Frozen readout → what is the emergent weight structure? (Q2)

We freeze the readout to the **identity** (so the network output *is* the last hidden
post-ReLU activations) and train the inner layers to send that to **0** over Gaussian inputs.
With no bias, to push `output = ReLU(z_last)` to ~0 the last pre-activation `z_last = W·a_prev`
must be driven **≤ 0**, where `a_prev = ReLU(...) ≥ 0` has a **positive mean** `μ = E[a_prev]`.

The cheapest way to make `w_i · a_prev ≤ 0` for a non-negative `a_prev` is to point each row
`w_i` **against** the dominant direction of `a_prev`, i.e. along **−μ**. If every row does
that, `W ≈ −1·μᵀ` — a **rank-1, −μ-aligned** matrix. So three phrasings of the hypothesis are
really one thing:

| story | measurement | prediction |
|---|---|---|
| (a) "weights get more negative" | `mean(W)` | `< 0` |
| (b) "rows align to −μ" | `mean_i cos(w_i, −μ)` | `> 0`, and `proj = W·μ̂ < 0` |
| (c) "that's the low-rank structure" | `|cos(v₁, μ)|`, μ-energy vs top-σ energy, stable rank | `v₁ ≈ μ`, energies coincide, rank ↓ |

**Controls:** the **first** layer `W₀` (whose input `x` has mean 0) should show none of this;
a **trainable** readout is compared against the frozen one; matched **random init** is the baseline.
""")

nb.md(r"""## 0. Setup — locate the repo
Point at the repo root (must contain `model/`). Needs only `torch`/`numpy`/`pandas`/`matplotlib`
— **kprop is not used here.**
""")
nb.code(BOOTSTRAP_CELL)

nb.md(r"""## 1. Config — **the knobs live HERE** (probe in place)
This cell defines the sweep; edit and re-run it directly (`experiments.py` only keeps
the classic defaults plus naming/recycling machinery). **Precision policy:** any fresh
*training* runs in float32 (the Trainer enforces `TrainConfig.dtype`; fast on GPU),
while the *analysis* below (eigendecompositions, μ-alignment) is done in **float64**
— models are cast to double after loading. `QUICK` defaults to True on a CPU-only
machine (smoke-test sweep). `LOAD_EXISTING=True` is the repo's recycling rule: a
same-named checkpoint on disk is loaded, not re-trained.
""")
nb.code(r"""
import torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)   # analysis dtype (training is float32 regardless)

import experiments as E
from tasks import ZeroTask

DEVICE = str(E.DEVICE)
QUICK  = E.QUICK               # True on a CPU-only machine
DEPTH  = 2
WIDTHS = [32, 64, 128] if QUICK else [16, 32, 64, 128, 256, 512]   # classic: E.WIDTHS
SEEDS  = [0]                                                       # classic: E.SEEDS
LOAD_EXISTING = True
CKPT_DIR = "checkpoints/noiseless_Layerless"   # THIS notebook's checkpoint folder
print("depth", DEPTH, "| widths", WIDTHS, "| device", DEVICE, "| ckpts ->", CKPT_DIR)
""")

nb.md(r"""## 2. Train / reload — `experiments.get_or_train`
`readout-frozen_identity_*` vs `readout-trainable_*`; same names as the existing
checkpoints, so previous runs are recycled. The frozen-identity readout is built by
`E.build_frozen_identity`; the Trainer skips frozen params automatically.
""")
nb.code(r"""
def get_model(cond, w, seed):
    name = E.run_name(f"readout-{cond}", depth=DEPTH, width=w, seed=seed)
    build = (lambda: E.build_frozen_identity(w, DEPTH, seed, device=DEVICE)) \
            if cond == "frozen_identity" else \
            (lambda: E.build_mlp(w, DEPTH, output_dim=w, seed=seed, device=DEVICE))
    m, payload, loaded = E.get_or_train(
        E.ckpt_path(CKPT_DIR, name), build,
        task=ZeroTask(input_dim=w, output_dim=w),
        train_cfg=E.default_train_cfg(w, seed=seed, device=DEVICE),
        extra_meta={"condition": cond, "experiment": "B_readout_d2"},
        load_existing=LOAD_EXISTING, map_location=DEVICE, progress=False)
    return m.to(device=DEVICE, dtype=torch.float64), E.final_loss(payload), loaded
print("ready")
""")

nb.md(r"""## 3. Dissect across widths — metrics from `analysis.Tools.weight_structure`
For each width: frozen-identity vs trainable readout (+ matched random-init baseline).
""")
nb.code(r"""
from analysis import mean_prev_post, weight_structure_metrics, W_last, W_first

rows = []
for w in WIDTHS:
    for seed in SEEDS:
        rnd = E.build_frozen_identity(w, DEPTH, seed, device=DEVICE)   # untrained baseline
        rows.append(dict(cond="random_init", loss=float("nan"),
                         **weight_structure_metrics(W_last(rnd), mean_prev_post(rnd))))
        for cond in ("frozen_identity", "trainable"):
            m, loss, loaded = get_model(cond, w, seed)
            rows.append(dict(cond=("frozen" if cond == "frozen_identity" else "trainable"),
                             loss=loss, **weight_structure_metrics(W_last(m), mean_prev_post(m))))
            print(f"  w{w:>4} {cond:15s} {'[loaded]' if loaded else '[trained]'}: loss={loss:.1e}  "
                  f"cos(-μ)={rows[-1]['cos_neg_mu']:+.3f}  |cos(v1,μ)|={rows[-1]['align_v1_mu']:.3f}  "
                  f"stable_rank={rows[-1]['stable_rank']:.1f}", flush=True)
df = pd.DataFrame(rows)
df.sort_values(["width", "cond"])
""")

nb.md(r"""## 4. Plots — the three stories
""")
nb.code(r"""
col = {"frozen": "crimson", "trainable": "darkorange", "random_init": "gray"}
mk  = {"frozen": "o-", "trainable": "s--", "random_init": "^:"}
fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
def P(a, y, title, ylab, logy=False, hline=None):
    for c in ["frozen", "trainable", "random_init"]:
        s = df[df.cond == c].sort_values("width"); a.plot(s.width, s[y], mk[c], color=col[c], label=c)
    a.set(xscale="log", xlabel="width", ylabel=ylab, title=title); a.grid(True, which="both", alpha=.3); a.legend(fontsize=8)
    if logy: a.set_yscale("log")
    if hline is not None: a.axhline(hline, color="k", lw=.6, ls=":")
P(ax[0], "align_v1_mu", "(c) top singular vector = μ?\n|cos(v₁, μ)|", "|cos(v₁, μ)|", hline=1)
P(ax[1], "cos_neg_mu", "(b) rows align to −μ\nmeanᵢ cos(wᵢ, −μ)", "alignment", hline=0)
P(ax[2], "stable_rank", "(c) low-rank collapse\n‖W‖_F²/σ₁²", "stable rank", logy=True)
for c in ["frozen", "trainable", "random_init"]:
    s = df[df.cond == c].sort_values("width"); ax[3].plot(s.width, s.rank1_energy_mu, mk[c], color=col[c], label=f"{c}: μ-dir")
sf = df[df.cond == 'frozen'].sort_values('width'); ax[3].plot(sf.width, sf.top_sv_energy, "x", color="k", ms=9, label="frozen: top-σ")
ax[3].set(xscale="log", xlabel="width", ylabel="fraction of ‖W‖_F²", title="(c) μ-energy ≈ top-σ energy\n(μ is the principal component)")
ax[3].grid(True, which="both", alpha=.3); ax[3].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

nb.md(r"""## 5. Control: the first layer should show NONE of this
`W₀`'s input is `x ~ N(0, I)` (mean **0**), so there is no positive-mean direction to oppose.
Expect `mean(W₀) ≈ 0` and a high stable rank (≈ random).
""")
nb.code(r"""
print(f"{'width':>6} {'mean(W_last)':>13} {'mean(W0)':>10} {'stable_rank(W_last)':>20} {'stable_rank(W0)':>16}")
for w in WIDTHS:
    m, _, _ = get_model("frozen_identity", w, SEEDS[0])
    Wl = np.asarray(W_last(m), float); W0 = np.asarray(W_first(m), float)
    sv = lambda M: (np.linalg.svd(M, compute_uv=False) ** 2)
    s1 = sv(Wl); s0 = sv(W0)
    print(f"{w:>6} {Wl.mean():>13.4f} {W0.mean():>10.4f} {s1.sum()/s1.max():>20.1f} {s0.sum()/s0.max():>16.1f}")
""")

nb.md(r"""## 6. How to read it

- **All three stories are the same phenomenon.** Training the pre-readout layer to send a
  **non-negative, positive-mean** activation `a_prev` to ~0 pushes each row `w_i` to oppose the
  mean direction `μ`. That simultaneously (a) makes entries **negative** (`mean(W) < 0`),
  (b) **aligns rows to −μ** (`cos(wᵢ,−μ) > 0`, `W·μ̂ < 0`), and (c) makes **μ the dominant
  singular direction** (`|cos(v₁, μ)| → 1`, μ-energy ≈ top-σ energy), i.e. the leading low-rank
  component **is** the −μ structure. It is a **rank-1 −μ spike on a higher-rank remainder**, not a
  pure rank-1 matrix — the μ-direction carries the largest single share of the energy
  (and that share shrinks with width).

- **First-layer control** (§5): `W₀` (input mean 0) stays ≈ zero-mean and high-rank ⇒ the effect
  is **specific to the layer feeding the positive-mean ReLU**, not a generic training artifact.

- **Frozen vs trainable** (depth 2): nearly identical ⇒ at this depth the −μ low-rank structure
  is driven by *converging the pre-readout layer to ~0 output*, **not** by the freezing itself
  (the trainable readout only partially shrinks; the inner layer still does the work).

- **Convergence matters, not depth per se.** The earlier depth-3 SGD runs that stalled at
  loss ≈ 1e-3 show essentially **no** structure (`cos(−μ)≈0`, `|cos(v₁,μ)|≈0.05`); the depth-2
  Adam runs reach loss ≈ 1e-7 and show it strongly. The structure appears **as the output is
  actually driven to ~0** — so finish training before reading the geometry.
""")

nb.save(os.path.join(os.path.dirname(__file__), "frozen_readout_weight_structure_colab.ipynb"))

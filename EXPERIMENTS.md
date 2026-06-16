# EXPERIMENTS.md — how to run, change, and add experiments

The working manual for this repo. If you are about to probe a training parameter,
start here.

## 1. Where every setting is defined

| Setting | Lives in | Examples |
|---|---|---|
| **Device & precision policy** | `experiments.py` (top of file) | `DEVICE`, `DTYPE` (float32), `QUICK` (True on CPU-only) |
| **Classic default grids** (widths, depths, seeds, lr, per-width batch/steps) | `experiments.py` (top of file) | `WIDTHS`, `DEPTHS`, `SEEDS`, `LR`, `batch_steps(w)` |
| **Architecture** | `ModelConfig` in `model/mlp.py` | `hidden_dim`, `depth`, `bias`, `activation`, `seed` |
| **Optimization** | `TrainConfig` in `training/trainer.py` | `steps`, `batch_size`, `lr`, `optimizer`, `loss_tol`, `dtype`, `checkpoint_mode` |
| **Task / data** | one file per task in `tasks/` | `ZeroTask`, `HalfspaceTask`, `DistillTask` |
| **Checkpoint NAMING & recycling** | `experiments.py` (`run_name`, `ckpt_path`, `get_or_train`) | `readout-frozen_identity_d2_w64_seed0_final.pt` |
| **Checkpoint FOLDER + per-experiment knobs** | each notebook's config cell | `CKPT_DIR = "checkpoints/kprop_tol_checkpoints"`, `WIDTHS`, `LOSS_TOL` |
| **Predictor (kprop) knobs** | `Mecha_preds/cumulants/adapter.py` | `k_max`, `exact_relu_cov` |
| **Structured (spike-aware) kprop knobs** | `Mecha_preds/cumulants/skprop/adapter.py` | `n_nodes`, `q_max`, `margin`, `directions`, `deep`, `deep_directions` |
| **Symbolic hidden-mode kprop knobs** | `Mecha_preds/cumulants/shkprop/adapter.py` | `latent`, `direction`, `hidden_degree_initial`, `hidden_degree_max`, `hidden_tail_tol`, `full_covariance` |
| **Analysis tools** | `analysis/Tools/` | `weight_spectrum`, `weight_structure_metrics`, … |

Rule of thumb: **`experiments.py` holds the classic defaults and the machinery
(naming, recycling, builders); each notebook defines its OWN knobs and its OWN
`CKPT_DIR` in its config cell** — probe there in place, no need to touch
`experiments.py` (or push it) for a one-off sweep. If a setting changes regime
semantics (like the kprop loss tolerance), encode it in the run name (`_tol5`) so
recycling can never silently mix regimes.

`training/run.py` (the CLI) takes its argparse defaults directly from `TrainConfig()`,
so changing a default in `trainer.py` changes the CLI default too — there is no second
copy to keep in sync.

## 2. Changing a setting — real examples

**Change the learning rate for everything:** edit `LR = 1e-3` in `experiments.py`.
For one run only: `python -m training.run --task zero --lr 3e-4 ...` or
`E.default_train_cfg(width, lr=3e-4)` in a notebook.

**Change the width sweep:** edit `WIDTHS = [16, 32, 64, 128, 256]` in `experiments.py`
(`QUICK_WIDTHS` is the smoke-test sweep notebooks use when `QUICK=True`).

**Change steps/batch for wide nets:** edit `batch_steps(width)` in `experiments.py` —
it returns `(batch_size, steps)` per width and is the single per-width budget rule.

**Turn biases on:** `--bias` on the CLI, or `ModelConfig(..., bias=True, final_bias=True)`.
They are OFF by default on purpose (half-space boundaries through the origin).

**Train to an exact loss instead of a step count:** `TrainConfig(loss_tol=1e-8)` —
early-stops once the step loss drops below it (checked every `tol_check_every=50`
steps to avoid per-step GPU→CPU syncs). The kprop scaling study uses this as its
*regime*: train until MSE < 1e-5 with a step cap as a safety net, instead of a fixed
budget. Those knobs (`LOSS_TOL`, `MAX_STEPS`, widths) live in the kprop notebook's
config cell.

**Train in float64:** `TrainConfig(dtype="float64")` or `--dtype float64` on the
CLIs. Default is float32 everywhere (see §6 precision policy).

**Train many seeds at once:** `training.train_ensemble` / `E.get_or_train_many` —
all missing runs of one architecture train simultaneously in a single vmapped loop
(exactly equivalent to N independent Adam runs; ~N× faster on GPU).

## 3. Running experiments

**CLI** (grid over depth × width × seed, one checkpoint per cell):

```bash
python -m training.run --task zero      --widths 64 128 256 --depths 2 3 --steps 8000
python -m training.run --task halfspace --widths 128 --depths 3 --offset-std 1.0
python -m training.run --task distill   --widths 128 --depths 3 --teacher-seed 1
```

**Programmatic / notebook** (this is the pattern every notebook uses):

```python
import experiments as E
from tasks import ZeroTask

CKPT_DIR = "checkpoints/noiseless_Layerless"    # the notebook picks its own folder
w, d, seed = 64, 2, 0
model, payload, loaded = E.get_or_train(
    E.ckpt_path(CKPT_DIR, E.run_name("readout-trainable", depth=d, width=w, seed=seed)),
    build=lambda: E.build_mlp(w, d, output_dim=w, seed=seed),
    task=ZeroTask(input_dim=w, output_dim=w),
    train_cfg=E.default_train_cfg(w, seed=seed),
)
print("loaded from disk:" if loaded else "trained fresh:", E.final_loss(payload))
```

**kprop comparison sweep** (trains + predicts + writes CSV/plots):

```bash
python -m Mecha_preds.cumulants.run_comparison --task zero --input-dim 64 \
    --hidden-depth 3 --widths 64 128 256 --seeds 0 1 2 --k-max 3 --outdir results/zero_kprop
```

## 4. Checkpoint recycling — the rule of this repo

**Never retrain what is already on disk.** Before any training, check the checkpoint
folders; `E.get_or_train(path, ...)` does this for you (loads if `path` exists, trains
and saves there otherwise). To see what exists:

```python
import experiments as E
for c in E.list_checkpoints("checkpoints/noiseless_Layerless"):   # pass YOUR notebook's dir(s)
    print(c)   # {'prefix': 'readout-frozen_identity', 'depth': 2, 'width': 64, 'seed': 0, ...}
```

**Naming convention** (parsed by `E.parse_ckpt_name`):

```
<prefix>_d<depth>_w<width>[_<extra><val>...]_seed<seed>_<tag>.pt
readout-frozen_identity_d2_w64_seed0_final.pt
meanfield_d3_w128_r32_bs4096_seed0_final.pt
```

**Current checkpoint folders** (each owned by its notebook's `CKPT_DIR` — there is
no global registry):

| Folder | Owner / contents |
|---|---|
| `checkpoints/noiseless_Layerless/` | frozen-readout notebook — frozen/trainable readout (d2 width sweep) + meanfield grids |
| `checkpoints/weight_analysis_checkpoints/` | mech-interp notebook — halfspace / max / zerobias dissection models |
| `checkpoints/kprop_tol_checkpoints/` | kprop scaling notebook — train-to-tolerance regime (`_tol5` = MSE < 1e-5, d3, widths 16–2048) |
| `checkpoints/shifted_mean_kprop/` | shifted-mean skprop notebook — random `W = W' + B` models (`B = -(1/√n)/-(1/n) 11ᵀ`, d3, widths 32–512) + a per-config `results_*.pt` cache of MC/kprop predictions (recycled, downloadable) |
| `checkpoints/symbolic_hidden_mode_kprop/` | symbolic hidden-mode kprop notebook — square nets **trained to 0 with `loss_tol=1e-7`** (`shkprop-zero_d2_w{16..256}_seed{0,1,2}`) + a per-config `results_*.pt` cache of 40M-point MC / symbolic predictions (recycled, downloadable) |

Checkpoints are self-describing `.pt` files: `model_config`, `state_dict`, `step`,
`history`, `final_loss`, `train_config` (+ any `extra_meta`). Load with
`MLP.load(path)` → `(model, payload)`.

⚠️ The depth-3 frozen/trainable checkpoints in `noiseless_Layerless` stalled at loss
~1e-3 (undertrained — no weight structure). Use the depth-2 set for Q2 conclusions;
see §6 of the frozen-readout notebook.

## 5. Adding a new experiment

1. **New task?** Add one file in `tasks/` subclassing `tasks.base.Task`
   (implement `sample_batch`; override `MSE_loss` only for a non-MSE objective),
   export it in `tasks/__init__.py`, and add a branch in `training/run.py::build_task`
   if you want it on the CLI.
2. **New checkpoint family?** Pick a `CKPT_DIR` under `checkpoints/` in your
   notebook's config cell and a `run_name` prefix; keep names parseable by
   `E.parse_ckpt_name` and add the folder to the table in §4 above.
3. **New notebook?** Notebooks are *generated* from `build_*.py` scripts (edit the
   script, re-run it — keeps them reproducible & diffable). Start from
   `colab_notebooks/noiseless_and_frozen_readout/build_frozen_readout_structure_notebook.py`,
   the reference example. Builders share `colab_notebooks/_nb.py`
   (`NotebookBuilder` + the standard `BOOTSTRAP_CELL` repo-locator); don't re-define
   cell helpers or bootstrap code.
4. **New reusable metric?** Put it in `analysis/Tools/` and export it from
   `analysis/__init__.py` — never define a metric inline in a notebook if a second
   notebook might want it.
5. **Spiked/meaned matrices breaking kprop?** Use structured power KPROP:
   `from Mecha_preds.cumulants.skprop import run_structured_cumulants` — detects the
   low-rank latent (or takes `directions=` explicitly, e.g. from `ΔW = W - W_init`),
   conditions on Gauss–Hermite nodes, and runs the ordinary power-cumulant kprop per
   node. With no spike detected (q=0) it returns EXACTLY the vanilla prediction, so
   it is safe to use unconditionally. Validation + usage examples:
   `colab_notebooks/structured_kprop/` (toy A/B/C/D scaling laws, planted-spike sweep,
   trained-checkpoint study).
6. **Want the hidden mode resolved without running N quadrature nodes?** Use symbolic
   hidden-mode KPROP: `from Mecha_preds.cumulants.shkprop import run_symbolic_cumulants`
   — carries the scalar latent `h` as polynomial jets (`config={"latent": "ones"}` for
   `h = 1ᵀX/√n`, `"none"` for q=0 = ordinary k=2 kprop), exact on linear layers, ReLU
   residual-Gaussian closure only. Torch-only (no kprop/scipy dependency); ships a numpy
   oracle (`shkprop.reference`) the notebook cross-checks against. Validation + the
   trained-to-0 (1e-7, 40M-MC) sweep: `colab_notebooks/symbolic_hidden_mode_kprop/`.

## 6. Gotchas

- **Precision policy — float32 for compute, float64 for measurement.** Training and
  inference run in float32 (`TrainConfig.dtype` default; the Trainer casts the model
  and every batch, enables TF32 matmuls on CUDA, and uses fused Adam — this is what
  makes the wide/train-to-tolerance sweeps run in human time on a GPU). float64
  remains exactly where accuracy is the point: `run_cumulants` builds a float64
  model copy internally (kprop always propagates in double), MC accumulators in
  `estimate_empirical_mean` are float64, and analysis notebooks cast loaded models
  to double before eigendecompositions. Opt back into all-double with
  `TrainConfig(dtype="float64")` / `--dtype float64`.
- **Device is auto-detected** (`E.DEVICE`: cuda → mps → cpu) and `E.QUICK` is True
  on a CPU-only machine — notebooks read it to default to the smoke-test sweep on
  CPU and the full sweep on GPU.
- **Don't sync the GPU per step.** `float(loss.cpu())` every step serializes
  training; the Trainer only syncs on `log_every`/`tol_check_every` boundaries.
  Keep custom loops to the same rule.
- **kprop needs Python ≥ 3.12** (the vendored code uses newer syntax). Training and
  analysis work on older versions.
- **Run from the repo root** — everything is a plain package, no install. Generated
  notebooks locate the root via their bootstrap cell.
- **`results/` is git-ignored** (regenerable); `checkpoints/*.pt` ARE tracked on
  purpose — they are the "one learned case".
- The Trainer skips frozen params (`requires_grad=False`) automatically — freezing a
  layer requires no custom loop.

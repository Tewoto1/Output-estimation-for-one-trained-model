# One trained case

Study **one learned case** of a small MLP and look for a **usable mechanistic
predictor** of its output — an algorithm that, given the trained weights, predicts a
statistic of the network's output (e.g. its mean over Gaussian inputs) *without*
Monte-Carlo sampling. The benchmark to beat / compare against is **cumulant
propagation** ("kprop"), so that algorithm is vendored here and runnable directly on
the models we train.

**→ Running / changing / adding experiments: see [EXPERIMENTS.md](EXPERIMENTS.md).**
That file is the working manual; this one is the map.

```
experiments.py    ALL experiment knobs: sweep grids, per-width budgets, checkpoint
                  naming + the get_or_train recycling rule. Notebooks import this.
model/            the MLP under study (one class, all activations, checkpoint load/save)
checkpoints/      trained .pt files (self-describing: config + weights + history)
  noiseless_Layerless/          frozen/trainable-readout + meanfield grids
  kprop_checkpoints/            trained-to-0 d3 models (widths 16-2048, seeds 3-6)
tasks/            task definitions + losses: train-to-zero, single half-space, distillation
training/         the training loop (Trainer/TrainConfig) + a grid-runner CLI
analysis/         circuit tools (analysis/Tools/, re-exported at package level)
Mecha_preds/      mechanistic predictors
  cumulants/        cumulant propagation as a predictor (+ exact ReLU-covariance variant)
    relu_integrals.py   CANONICAL torch-free Gaussian-ReLU integrals (relu_moments_1d,
                        exact_relu_covariance) shared by every torch-free predictor
    _cumulant_math.py   shared Hermite He_p + moment<->cumulant conversions
    kprop/            the vendored kprop algorithm (from the ARC paper repo)
    swkprop/          shifted-weight kprop: all-ones special direction, √n weight shift
    spikekprop/       spike kprop: ANY unit spike direction v, O(1)-eigenvalue spike M=W'+θvvᵀ
  binned_kprop/     coordinate-spike binned kprop (K=2): HMM over the spiked coord e₁ +
                    conditional bulk kprop per bin (M=W+e₁e₁ᵀ); num_bins hyperparam.
                    Verified width law MSE~n⁻². See Mecha_preds/binned_kprop/README.md
  analytic_kprop/   ANALYTIC AFFINE K=2 (same M=W+e₁e₁ᵀ class): bulk-given-spike law
                    compressed to ONE affine family N(μ₀+μ₁y, Σ₀+Σ₁y); the spike is
                    discretized only transiently into num_nodes closed-form quadrature
                    cells per layer (O(1) congruences/layer, not O(num_bins)). Spec:
                    writeups/analytic_affine_kprop.pdf; see Mecha_preds/analytic_kprop/README.md
experiments/  generated notebooks + their build_*.py generators (+ shared _nb.py)
utils.py          device / seeding / numpy helpers
```

Everything is a plain package — **run from the repo root** (no install needed). Module
entry points are `python -m training.run ...` and
`python -m Mecha_preds.cumulants.run_comparison ...`; notebooks add the repo root to
`sys.path` in their bootstrap cell.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(Training/analysis work on Python ≥ 3.10; the vendored kprop needs ≥ 3.12.)

## Quickstart — load a trained model and predict its output mean

```python
from model import MLP
from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, compare_means

model, payload = MLP.load("checkpoints/noiseless_Layerless/readout-trainable_d2_w64_seed0_final.pt")
print(model.cfg, "final loss:", payload["history"][-1][1])

pred = run_cumulants(model)["mean"]                 # cumulant propagation, default k_max=3
mc, stats = estimate_empirical_mean(model=model, input_dim=model.cfg.input_dim,
                                    num_samples=200_000)   # Monte-Carlo reference
print("relative error:", compare_means(pred, mc, stats)["relative_error_mean"])
```

## Train

One CLI trains any task over a depth × width × seed grid (see
`python -m training.run --help`; defaults come from `TrainConfig`):

```bash
python -m training.run --task zero --widths 64 128 256 --depths 2 3 --steps 8000
```

Programmatically, prefer the recycling helper — it loads an existing checkpoint
instead of retraining (the rule of this repo):

```python
import experiments as E
from tasks import ZeroTask

CKPT_DIR = "checkpoints/noiseless_Layerless"   # each notebook picks its own folder
model, payload, loaded = E.get_or_train(
    E.ckpt_path(CKPT_DIR, E.run_name("readout-trainable", depth=2, width=64)),
    build=lambda: E.build_mlp(64, 2, output_dim=64),
    task=ZeroTask(input_dim=64, output_dim=64),
    train_cfg=E.default_train_cfg(64),
)
```

Conventions: `input_dim == width` by default (a square first layer W₁), **biases OFF**
by default (ReLU half-spaces pass through the origin; layer-0 pre-activations are
exactly zero-mean), and no weight decay — so the only pressure is the task loss and the
emergent geometry is the object of study. Pass `--bias` / `ModelConfig(bias=True)` to
restore biases.

## Analysis

Each tool runs **individually** and returns a dict — there is no master "report" step.
All tools live in `analysis/Tools/` and work on any `model.MLP` + any input batch:

```python
import torch
from model import MLP
from analysis import (weight_spectrum, activation_pca, neuron_importance,
                      ablate, activation_patch, direct_contributions, output_lens,
                      weight_structure_metrics, mean_prev_post, W_last)

m, _ = MLP.load("checkpoints/noiseless_Layerless/readout-frozen_identity_d2_w64_seed0_final.pt")
x = torch.randn(4096, m.cfg.input_dim)

weight_spectrum(m)["readout"]["effective_rank"]    # per-weight SVD effective rank
activation_pca(m, x, layer=-1)["top_explained"]    # PCA of last hidden activations
neuron_importance(m, x, layer=0)["top_k"]          # single-neuron knockout ranking
direct_contributions(m, x)["cancellation_ratio"]   # readout decomposition wᵢ·zᵢ
output_lens(m, x)                                  # read each layer through the readout
weight_structure_metrics(W_last(m), mean_prev_post(m))  # Q2: rank-1 −μ spike metrics
```

## Mechanistic predictors — cumulant propagation

`Mecha_preds/cumulants/` wraps the real kprop algorithm (vendored, unmodified, under
`kprop/`) so it runs on a study `model.MLP`:

```python
from Mecha_preds.cumulants import run_cumulants
pred       = run_cumulants(model, config={"k_max": 3})["mean"]                       # normal kprop
pred_k2    = run_cumulants(model, config={"k_max": 2})["mean"]                       # k=2 (approx ReLU cov)
pred_exact = run_cumulants(model, config={"k_max": 2, "exact_relu_cov": True})["mean"]  # EXACT ReLU covariance
```

`exact_relu_cov=True` (ReLU, `k_max==2` only) uses the exact bivariate-Gaussian ReLU
covariance instead of the leading-order gain approximation. For any other
`k_max`/activation the normal harmonic kprop runs.

For a **rank-one spike** `M = W' + θ vvᵀ` in the weights there are two split-basis
predictors that keep the spike-direction cumulants ordinary kprop truncates away:

```python
from Mecha_preds.cumulants.swkprop  import run_sw_kprop      # all-ones, √n shift (eigenvalue √n)
from Mecha_preds.cumulants.spikekprop import run_spike_kprop  # any direction, O(1) eigenvalue
pred = run_spike_kprop(model, spike_dir="e1", config={"R": 4, "n_nodes": 61})["mean"]
```

`spikekprop` generalizes `swkprop` to an arbitrary unit spike direction `v` (`"e1"` localized,
`"ones"` flat, or a vector) for the `|θ| ≤ n^{o(1)}` regime, retaining `C(v,…,v)=κ_p(S)` to
order `R`. `spike_dir="ones"` reproduces `swkprop`.

Sweep cumulant-vs-Monte-Carlo error across widths (trains via the unified loop, writes
CSV + plots to `--outdir`):

```bash
python -m Mecha_preds.cumulants.run_comparison \
    --task zero --input-dim 64 --hidden-depth 3 \
    --widths 64 128 256 --seeds 0 1 2 --train-steps 5000 --k-max 3 \
    --outdir results/zero_kprop
```

## Notebooks (`experiments/`)

Each notebook is **generated** from the `build_*.py` script next to it (edit the
script, re-run it — keeps notebooks reproducible and diffable). Builders share
`experiments/_nb.py`. Each notebook defines its **own knobs and its own
`CKPT_DIR`** in its config cell — probe there in place; `experiments.py` keeps the
classic defaults plus the naming/recycling machinery.

- **`trained_to_0_cumulants_test/exact_relu_k2_width_scaling_colab.ipynb`** — the
  **baseline** scaling experiment: does the exact ReLU covariance fix cumulant
  propagation on trained-to-zero models? Compare future experiments against it.
- **`noiseless_and_frozen_readout/frozen_readout_weight_structure_colab.ipynb`** —
  Q2: with a frozen (identity) readout trained to output 0, the pre-readout matrix
  develops a rank-1 spike aligned to −μ. **The reference example of the repo's
  notebook pattern** (config from `experiments.py`, checkpoint recycling, shared
  metrics).
- **`mech_interp_on_trained_to_0/weight_structure_vs_randomness.ipynb`** — the
  "structure + randomness" decomposition of trained hidden weights
  (covariance-adjusted −μ drift + Gaussian residual), with task models as controls.

## Notes

- **Precision**: training/inference run in **float32** (repo policy — GPU tensor
  cores, fused Adam, TF32 matmuls; `TrainConfig(dtype="float64")` opts out).
  Accuracy-critical paths stay double: `run_cumulants` builds a float64 copy of the
  model internally, MC accumulators are float64, analysis eigendecompositions cast
  to double.
- **Parallel training**: `training.train_ensemble` / `E.get_or_train_many` train
  all seeds of one architecture in a single vmapped loop (exactly equivalent to N
  independent runs, ~N× faster on GPU).
- Checkpoints store `model_config`, `state_dict`, `step`, `history`, `final_loss`,
  and `train_config`; load with `MLP.load(path)` → `(model, payload)`.
- The `kprop/` library is vendored from the ARC paper *"Estimating the expected output
  of wide random MLPs more efficiently than sampling"* and is MIT-licensed (see
  `LICENSE`); only the harmonic kprop path + the exact ReLU-covariance step are kept.

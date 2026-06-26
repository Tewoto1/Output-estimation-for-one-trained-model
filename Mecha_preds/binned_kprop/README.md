# binned_kprop — coordinate-spike binned cumulant propagation (K = 2)

A mechanistic predictor of `E[model(X)]` (`X ~ N(0, I)`) for a ReLU MLP whose hidden
matrices carry a **coordinate** spike on a single axis `e₁`:

```
M = W + e₁ e₁ᵀ ,     W_ij ~ N(0, 1/n)
```

## What it's for

For a **flat** spike (`(1/√n) 11ᵀ`) ordinary total-order kprop already works — the
spike direction gets a flat-loop `1/n` discount (that's the `spikekprop` "ones" case).
A **coordinate** spike has no such discount: the linear step keeps a large residue of
coordinate 0 (`A⁺ = γA + rᵀB`), so the cumulants on coordinate 0 are `O(1)` at every
order and must **not** be carried as ordinary bulk cumulant-tensor entries.

This module represents the spike coordinate **explicitly**: a hidden-Markov model over
`num_bins` bins of `A` (a scalar transition kernel between layers), with ordinary **K=2**
cumulant propagation of the bulk `B ⟂ e₁` *conditional on each bin*. Each bin stores a
conditional bulk mean + covariance.

- **This is the K = 2 implementation.** `num_bins` is the adjustable hyperparameter.
- `K > 2` (conditional cumulant *tensors* per bin) hooks into the ordinary harmonic
  kprop — `kprop_hook` imports and calls `Mecha_preds.cumulants.kprop`.

Full derivation (algebra of every step + the Wasserstein bin-placement) is in
[ALGORITHM.md](ALGORITHM.md).

## What's inside

```
__init__.py     public API (re-exports the K=2 core + the MLP adapter)
core.py         the K=2 algorithm, numpy/scipy, TORCH-FREE:
                  BinnedK2State, normal_interval_stats, linear_step_k2, relu_step_k2,
                  gaussian_initial_state, run_binned_kprop_k2, unconditional_mean[_cov]
_relu.py        Gaussian-ReLU backend (reuses ..cumulants.relu_integrals; path-load
                  fallback so the K=2 core imports with only numpy+scipy)
kprop_hook.py   bridge to ORDINARY harmonic kprop (needs torch):
                  bulk_relu_kprop (K=2 bulk-ReLU backend, bulk_relu="kprop"),
                  bulk_relu_kprop_tower + BinnedKState + relu_step_k_general (K>2 hook)
adapter.py      run_binned_kprop(model, ...) — drop-in like run_cumulants/run_spike_kprop
selftest.py     numpy MC verification of the K=2 core (python -m ...selftest)
scaling.py      the SCALING-LAW-WITH-WIDTH test: MSE ~ n^-2 (python -m ...scaling)
benchmark.py    practical num_bins/width TUNING harness vs fast MC (python -m ...benchmark)
```

## How to run

Run from the repo root (no install needed). Predict a model's output mean:

```python
from model import MLP
from Mecha_preds.binned_kprop import run_binned_kprop

model, _ = MLP.load("checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt")
pred = run_binned_kprop(model, config={"num_bins": 31})["mean"]   # K=2; 31 spike bins
```

`config` keys: `num_bins` (the hyperparameter), `num_bins_post`, `bulk_relu`
(`"exact"` | `"gain"` | `"kprop"`), `input_std`. Pass `add_spike=True` to add
`e₁ e₁ᵀ` to each square hidden layer when the spike isn't already baked into the weights
(default `False` — assume it is, like `spikekprop`). Requires square hidden layers
(`input_dim == hidden_dim`), which the train-to-zero models satisfy.

Tests:

```bash
python -m Mecha_preds.binned_kprop.selftest     # core vs Monte-Carlo (torch-free)
python -m Mecha_preds.binned_kprop.scaling      # K=2 width scaling law: MSE ~ n^-2
python -m Mecha_preds.binned_kprop.benchmark    # tune num_bins at a larger width (depth 3)
```

Tuning `num_bins` at a larger width in practice (depth 3, 4M MC samples, batch 200k):

```bash
python -m Mecha_preds.binned_kprop.benchmark --widths 256 512 --num-bins 11 21 41
```

prints rel-err / MC-z / wall-time per `num_bins` and a recommended `num_bins` (the
smallest within 1.2× of the best accuracy at each width — larger widths need fewer bins).

## Scaling law (what `scaling.py` verifies)

A budget-`k_max` predictor has output error `O(n^{-k_max})`, so at **K = 2** the
relative **MSE ~ n⁻²** (RMS error ~ n⁻¹) as the width `n` grows. Representing the spike
coordinate with enough bins is exactly what recovers this rate — the naive `num_bins=1`
closure (collapse `A` to its mean, eat the ReLU Jensen gap) only reaches `n⁻¹` MSE.
Seed-averaged over random coordinate-spiked nets, depth 2, `num_bins=31`:

```
   n      MSE(binned)   MSE(1 bin)   RMS advantage
   32     2.5e-04       9.7e-02       19.5x
   64     6.9e-05       4.2e-02       24.8x
  128     1.9e-05       1.7e-02       30.2x
  fit:    MSE ~ n^-1.88     MSE ~ n^-1.25
          (≈ n^-2, K=2 rate) (naive ≈ n^-1)
```

Use **enough bins** to see the `n⁻²` rate: with too few, the spike-discretization error
floors the total error and flattens the slope (the fixed-width refinement curve in
`scaling.py` shows error dropping sharply with `num_bins`, then converging to the bulk
K=2-closure floor).

## Notes

- The **K=2 core** (`core.py`) is numpy/scipy and torch-free.
- The **kprop hook / adapter** use torch; the hook calls `..cumulants.kprop`, which
  needs Python ≥ 3.12 (or the repo's `_kprop_compat` shim, applied automatically).
- Known K=2 limitations (per the implementation spec): a Gaussian closure inside each
  bin, bins collapse within-bin `A` variation to a representative, and the cross-bin
  merge is at order 2 (mean + covariance). The per-bin propagation can run at any
  `k_max` via the kprop hook.
```

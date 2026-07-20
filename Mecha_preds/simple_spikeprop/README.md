# simple_spikeprop — constant bulk K=2 + exact 1-d spike recursion (10th predictor)

The minimal member of the coordinate-spike family (`M = W + e₁e₁ᵀ`, same model class
as `binned_kprop` / `analytic_kprop`). Two ingredients, nothing else:

1. **Spike coordinate: full 1-d law on a grid**, driven per-network by the exact
   scalar channel recursion (the affine_knee recursion)

   `p_{ℓ+1} = φ_ω ∗ (c·ReLU + μ)_# p_ℓ`,  `μ = wᵀm (+ b₀)`,  `ω² = wᵀΣw`,

   with `c = M[0,0]`, `w = M[0,1:]` read off the actual weights and `(m, Σ)` the
   tracked bulk moments. The ReLU folds the law exactly on the grid (negative mass →
   zero atom, positive nodes verbatim); the linear step is an exact mixture
   convolution. The law must stay nonparametric: atom + branch is an O(1) structure,
   and a two-moment surrogate would cost O(n^{-1/2}) at the output.

2. **Bulk: ONE unconditional Gaussian `N(m, Σ)`** — mean and covariance held
   *constant in the spike value* (the "CONST" surrogate: no bins, no affine family,
   no conditioning). ReLU via the exact rank-2 Gaussian-ReLU map
   (`Mecha_preds._utils.exact_relu_covariance`); linear step with the variance swap
   re-aggregated on **both** sides:

   `m' = u·E[S] + V m`,  `Σ' = V Σ Vᵀ + Var(S)·uuᵀ`  (spike → bulk),
   `ω² = wᵀΣw` (bulk → spike).

## Why this should still get MSE ~ n⁻²

From the CONST-vs-AFFINE error accounting: the dropped structures — the bulk↔spike
cross-covariance (ρ/m₁ route) and all conditional-on-the-spike dependence — have
O(n^{-1/2}) amplitude per coordinate but are mass-centered, so they enter
mass-integrated propagated **means** only at second order, O(1/n) per coordinate.
O(n^{-1/2}) readout rows then give an output-mean error O(1/n) → **output MSE ~ n⁻²,
the same order as binned/analytic, with a larger constant** (the entire
linear-response component stays in the residual). This package is the deliberate
ablation floor of the family: CONST (here) ≤ AFFINE (`analytic_kprop`) ≤ BINNED.

## Usage

```python
from Mecha_preds.simple_spikeprop import run_simple_spikeprop
pred = run_simple_spikeprop(model)["mean"]                     # spike baked in
pred = run_simple_spikeprop(model, add_spike=True)["mean"]     # add e1 spike to raw W
```

Torch-free core: `run_simple_spikeprop_core(weights, input_dim)` on `(W, b)` numpy
pairs. No structural hyperparameter; `num_grid` (2001) / `span` (8σ) are quadrature
knobs only and converge fast (grid error ~3e-6 rel at defaults). Cost per layer =
one `VΣVᵀ` + one exact d×d ReLU covariance — what binned pays **per bin**.

## Verification (`python -m Mecha_preds.simple_spikeprop.selftest`, 7/7)

* channel push / ReLU fold match closed forms to grid accuracy; depth-1 is exact
  (rel err 2.8e-6, grid-limited) since marginals never see the dropped cross-cov.
* depth-3, n=48 vs 4M MC: simple 6.4e-2 ≤ binned[1 bin] 11.7e-2; binned[21] 3.2e-2 —
  exactly between the crude and full conditional predictors.
* smoke width scaling (d3, 3 seeds, rel err medians): n=80 → 160 gives
  0.0253 → 0.0125 (×0.49 ≈ 1/n, i.e. MSE ~ n⁻²); n=40 is pre-asymptotic (0.0317).
  The proper split-half cross-MSE width law is a follow-up experiment.

## Limitations

* Coordinate spike only (`e₁`, θ ~ O(1)). For the all-ones ±(1/√n)·11ᵀ shift use
  `swkprop`/`spikekprop` (basis rotation). For strongly supercritical spikes
  (|c| ≫ 1) the fixed grid stretches — see the two-scale (location/scale + fold
  window) prescription in the knee notes before trusting it there.
* `"cov"` in the result is the model-implied output covariance with the spike-bulk
  cross block identically 0 (that IS the approximation) — indicative only; `"mean"`
  is the prediction the width law is about.

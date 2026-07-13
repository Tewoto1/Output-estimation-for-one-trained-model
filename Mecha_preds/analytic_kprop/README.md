# analytic_kprop — analytic affine-conditioned K = 2 propagation (coordinate spike)

A mechanistic predictor of `E[model(X)]` (`X ~ N(0, I)`) for a ReLU MLP whose hidden
matrices carry a **coordinate** spike on a single axis `e₁`:

```
M = W + e₁ e₁ᵀ ,     W_ij ~ N(0, 1/n)
```

Implements **Algorithm 7.2** (finite-grid, exact-cell scalar backend) of the write-up
[`writeups/analytic_affine_kprop.pdf`](../../writeups/analytic_affine_kprop.pdf) — the
analytic alternative to the companion [`binned_kprop`](../binned_kprop/README.md)
(paper appendix C relates the two).

## The idea

Same spike–bulk split as the binned method (`X = A e₁ + B`), but the bulk-given-spike
law between layers is compressed to **one affine family** in the signed spike
pre-activation `y` instead of one Gaussian per bin:

```
C | Y = y  ⇝  N( μ₀ + μ₁ y ,  Σ₀ + Σ₁ y )        (paper eq 11)
```

The spike direction is **approximated as discrete only transiently**: per layer,
`num_nodes` quadrature cells (edge exactly at 0) are laid over the scalar law of `Y`,
which is a *known* Gaussian mixture (one component per retained post-ReLU node, paper
eq 51) — so every cell mass / centroid / conditioned moment is **closed form**
(truncated-normal identities, eqs 63–68; the "bash the probability from the assumed
distribution" step). No per-node bulk state is carried before ReLU.

One layer = linear + exact Bayesian reconditioning on the new spike coordinate
(eqs 66–72) → **weighted-LS affine re-projection** (eqs 86–87, with the
moment-conservative intercept `Σ₀ += R_m` of eq 90 by default) → **exact
Gaussian–ReLU moments per node** (eqs 98–99, the repo's shared exact bivariate
kernel) → positive nodes kept as atoms, all `y ≤ 0` nodes merged **exactly** into the
zero atom (eqs 40–42). The input layer is exact (no input discretization; the
Gaussian input is one component with within-component spike variance `t2 = 1`).

**Cost.** All downstream uses of the per-component congruence `V Sᵢ Vᵀ` are linear
with scalar weights, so the layer needs only **two aggregated congruences** (see
`_covariance_sums`) — `O(1)·d³` per layer vs the binned method's `O(num_bins)·d³` —
plus `O(m·J)` closed-form scalars and the shared `O(num_nodes·d²)` ReLU-kernel
special functions. The advantage shows at larger widths (CPU crossover ≈ n ≳ 200);
at small widths the fixed scalar machinery dominates and binned is faster.

**Error budget** (paper thm 10.1): conditional K=2 closure + affine re-projection
(residuals `E_m`, `E_S`, `tr R_m` logged per layer) + the 1-D quantization term
(eq 134, logged as `scalar_distortion`) which the `num_nodes` knob controls.

## What's inside

| file | contents |
|---|---|
| `core.py` | torch-free numpy/scipy core: `AnalyticState`, `analytic_layer_update`, `run_analytic_kprop_k2` |
| `adapter.py` | `run_analytic_kprop(model, config={"num_nodes": 40})` on a `model.MLP` (torch only here) |
| `selftest.py` | `python -m Mecha_preds.analytic_kprop.selftest` — exact identities (machine precision), depth-1 closed form, MC end-to-end, binned parity |

Shared kernels: ReLU integrals + PSD utils from `Mecha_preds/_utils.py`;
truncated-normal cell stats and W2 (Lloyd-Max) mixture grids from
`binned_kprop/binning.py`; per-node bulk-ReLU backend dispatch from
`binned_kprop/core.py`.

## Usage

```python
from model import MLP
from Mecha_preds.analytic_kprop import run_analytic_kprop
model, _ = MLP.load("checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt")
pred = run_analytic_kprop(model, config={"num_nodes": 40})["mean"]
```

Key knobs (`default_analytic_kprop_config`): `num_nodes` (total signed cells per
layer — the hyperparameter; the §3 experiments show a knee at ~6–10, so 40 is
generous), `num_nodes_neg`/`num_nodes_pos` (sign-split override; default allocates
by mixture mass), `grid` (`"w2"` Lloyd-Max on the exact mixture | `"uniform"`),
`bulk_relu` (`"exact"` | `"gain"` | `"kprop"`), `cov_intercept` (`"mc"` | `"ls"`),
`diagnostics` (adds the per-cell `E_S` residual; costs `J` congruences/layer).

Experiments + validation: `colab_notebooks/analytic_kprop/` (MC references cached in
`checkpoints/analytic_kprop/`, shared with `checkpoints/binned_kprop/` when configs
match). Scalar-law variants of paper §7.3 (mixture-integral, atomic-node,
single-Gaussian baseline) are future knobs.

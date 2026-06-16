# clippedProp — structured propagation with a clamped-Gaussian all-ones channel

A mechanistic predictor (sibling of `cumulants` / `cumulants.skprop` / `cumulants.shkprop`) that
propagates the **structured law**

```
X = s·u + z,   u = 1_d / √d,   P = I − u uᵀ
   s = max(lo, g),  g ~ N(m, v)          # scalar latent on the all-ones direction (clamped Gaussian)
   z ~ N(μ_z, Σ_z),  μ_z, Σ_z ∈ u⊥        # perpendicular Gaussian
   c_s = Cov(z, s) ∈ u⊥                   # cross-covariance (optional, tracked)
```

through a ReLU MLP. The all-ones (mean-shift) component is carried **explicitly** as a clamped
("rectified") Gaussian scalar; everything else is a cross-correlated Gaussian in `u⊥`. The point: a
coherent `O(1)` mean shift — exactly what breaks vanilla `k=2` cumulant propagation — lives entirely
in the scalar channel `s`, and the ReLU step *conditions on it* so the perpendicular residual is
incoherent again and the ordinary Gaussian-ReLU covariance is accurate.

This file is the spec ↔ implementation map so you can check the math against the code.

---

## 1. State  →  `state.py` (`ClippedState`)

The state is exactly the spec's:

| spec field | code | notes |
|---|---|---|
| `u = ones / √d` | `ClippedState.u` (property) | not materialized; built on demand |
| `P = I − u uᵀ` | `proj_vec`, `proj_mat` | applied as `x − (uᵀx)u`, never stored |
| scalar `s`: `p0`, truncated-Gaussian, mean, variance | `m, v, lo` + `scalar_moments()` | `scalar_moments()` returns `(E[s], Var[s], p0)` |
| `z`: `mu_z`, `Sigma_z` | `mu_z`, `Sigma_z` | kept in `u⊥` (`proj_*` applied on build) |
| cross-cov `c_z = Cov(z, s)` | `c_s` | stored as the *clamped* cross-cov `Cov(z, s)` |

The underlying latent `(g, z)` is jointly Gaussian; `s` is the clamp of the scalar latent `g`. Storing
`Cov(z, s)` (clamped) keeps the linear-layer covariance assembly direct; the ReLU step converts it to
`Cov(z, g) = c_s / β` (see §2) for Gaussian conditioning.

Two reconstructions glue everything together:

- **`mean_cov()`** — `(m, v, lo, μ_z, Σ_z, c_s) → (μ_X, Σ_X)`:
  ```
  μ_X    = E[s]·u + μ_z
  Σ_X    = Var[s]·u uᵀ + u c_sᵀ + c_s uᵀ + Σ_z
  ```
- **`from_full_moments(μ_Y, Σ_Y, lo)`** — the **re-split + refit** closure applied after every layer.
  Splits onto the (possibly new-dimension) all-ones direction and refits the scalar:
  ```
  E[s'] = u'ᵀ μ_Y       Var[s'] = u'ᵀ Σ_Y u'
  μ_z'  = P' μ_Y        Σ_z'    = P' Σ_Y P'        c_s' = P' Σ_Y u'
  (m', v') = fit_rect_gauss(E[s'], Var[s'], lo)
  ```
  Because the clamped-Gaussian fit matches the scalar's first two moments **exactly**, `mean_cov()` is
  an exact left-inverse of `from_full_moments()` for `(μ, Σ)` — the clamp shape only affects the *next*
  ReLU's conditioning, never the reconstructed moments.

Constructors: `from_isotropic(d)` (`X ~ N(0,I)`), `from_gaussian(μ, Σ)` (arbitrary Gaussian; scalar is
plain, `lo=−∞`), `from_structured(d, m, v, lo=0, μ_z, Σ_z, c_s)` (the user's structured input directly).

---

## 2. Clamped-Gaussian scalar  →  `scalar.py`

`s = max(lo, g)`, `g ~ N(m, v)`. `lo = 0` ⇒ rectified (point mass `p0 = Φ(−(m−lo)/√v)` + positive tail);
`lo = −∞` ⇒ plain Gaussian (the family contains the Gaussian, so the state representation is uniform).

| map | function | formula |
|---|---|---|
| forward `(m,v,lo) → (E[s], Var[s], p0)` | `rect_gauss_moments` | `E[s]=lo + (m−lo)Φ(α)+√v φ(α)`, `α=(m−lo)/√v` |
| inverse `(E[s],Var[s],lo) → (m,v)` ("refit") | `fit_rect_gauss` | bisect `α` on `R=Var/E²` (a function of `α` alone, monotone); plain-Gaussian / point-mass limits in closed form |
| `β = Cov(g,s)/v` | `clipped_cross_beta` | `lo=−∞ ⇒ 1`; `lo=0 ⇒ (E[s²]−m·E[s])/v`; general `lo` via 1-D Gauss-Hermite |

`β` converts the stored clamped cross-cov to the conditioning cross-cov: `c_g = Cov(z,g) = c_s / β`.

---

## 3. The three layer maps  →  `layers.py`

### Linear `X' = W X + b`  (`linear_layer`)
Propagate the full moments, then re-split + refit in the **output** space:
```
μ' = W μ_X + b           Σ' = W Σ_X Wᵀ
return ClippedState.from_full_moments(μ', Σ', lo)
```
`lo` defaults to `−∞` (a pre-activation `u'ᵀ(WX+b)` is a signed scalar → plain Gaussian). The readout
uses `linear_output_moments` (returns raw `(μ,Σ)`, **no** closure) — clippedProp is *exact* at the final
linear map, `E[out] = W·E[a] + b`.

### Mean-subtraction `X ← P X`  (`mean_subtraction_layer`)
The scalar channel becomes exactly `0` (`uᵀ P x = 0`); `P(s u + z) = z`, so `μ_z, Σ_z` are unchanged and
`c_s` vanishes. (This is the spec's "scalar direction becomes exactly zero, perpendicular covariance
becomes `P Σ P`".)

### ReLU `Y = relu(X)`  (`relu_layer`)
Condition on the scalar latent `g` and integrate by Gauss-Hermite (`n_nodes`):
```
c_g = c_s / β                                   # β = clipped_cross_beta(m,v,lo)
Σ_cond = Σ_z − c_g c_gᵀ / v                     # conditional perp cov (shared across nodes)
for each GH node g_k ~ N(m,v):
    s_k        = max(lo, g_k)                    # clamp the scalar value
    a_k        = s_k·u + μ_z + ((g_k−m)/v)·c_g   # conditional per-coordinate mean (X|g_k is Gaussian)
    (m_k, C_k) = relu_kernel(a_k, Σ_cond)        # Gaussian-ReLU moments at this node
μ_Y = Σ_k w_k m_k
Σ_Y = Σ_k w_k (C_k + m_k m_kᵀ) − μ_Y μ_Yᵀ        # law of total covariance
return ClippedState.from_full_moments(μ_Y, Σ_Y, lo=0)   # post-ReLU scalar is rectified at 0
```
Per-node kernels:
- **`relu_cov="exact"`** — reuses the project's verified `cumulants.kprop.exact_relu_covariance_np`
  (exact bivariate-Gaussian ReLU covariance via Owen's T). scipy/CPU, `O(d²)` per node.
- **`relu_cov="gain"`** — exact ReLU mean+variance; off-diagonal `Σ_ij ← Σ_ij·Φ(α_i)Φ(α_j)` (the
  leading-order gain, no bivariate CDF). Cheaper, no scipy.

A `cross_guard` rescales `c_g` if needed so the conditional diagonal stays non-negative (numerical safety).

---

## 4. Forward & adapter  →  `propagate.py`, `adapter.py`

- **`clipped_mlp_forward(model, init_state=…, n_nodes, relu_cov, clip_after_linear, mean_subtract_after, want_cov)`**
  walks a study `model.MLP`: each hidden block = `linear_layer` → (optional `mean_subtraction_layer`) →
  `relu_layer`; then `linear_output_moments` for the readout. Returns `{"mean", "cov", "final_state"}`.
- **`run_clipped(model, config=…)`** — drop-in twin of `cumulants.run_cumulants`: assumes `X ~ N(0,I)`,
  float64, returns `{"raw_output", "mean" (np.ndarray), "metadata"}`. Config keys: `n_nodes` (21),
  `relu_cov` ("exact"), `clip_after_linear` (False), `mean_subtract_after` (()), `want_cov` (False).

```python
from model import MLP
from Mecha_preds.clippedProp import run_clipped, clipped_mlp_forward, ClippedState

pred = run_clipped(model, config={"n_nodes": 21, "relu_cov": "exact"})["mean"]   # X ~ N(0,I)

# structured / clamped-mean input:
st   = ClippedState.from_structured(d, m=1.0, v=1.0, lo=0.0)        # s = max(0, N(1,1)) on u
pred = clipped_mlp_forward(model, init_state=st, n_nodes=21)["mean"]
```

---

## 5. File correspondence (summary)

| file | contents |
|---|---|
| `scalar.py`   | clamped-Gaussian moment maps: `rect_gauss_moments`, `fit_rect_gauss`, `clipped_cross_beta` |
| `state.py`    | `ClippedState` (state + `mean_cov` + `from_full_moments` split/refit + constructors) |
| `layers.py`   | `linear_layer`, `linear_output_moments`, `mean_subtraction_layer`, `relu_layer` (+ exact/gain kernels) |
| `propagate.py`| `clipped_mlp_forward` — forward through a `model.MLP` |
| `adapter.py`  | `run_clipped`, `default_clipped_config`, `clipped_config_summary` |
| `__init__.py` | public exports |
| `README.md`   | this document |
| `../../colab_notebooks/clipped_prop/` | the test-suite + validation notebook (+ its build script) |

---

## 6. Validation (see the notebook `colab_notebooks/clipped_prop/`)

**§1 unit math — all pass to machine precision:**

| check | measured error |
|---|---|
| rectified-Gaussian moment round-trip (`fit_rect_gauss` ∘ `rect_gauss_moments`) | ~6e-13 |
| `clipped_cross_beta` closed form vs Gauss-Hermite | 0 / <1e-6 |
| single ReLU layer vs exact bivariate kernel (21 nodes) | ~3e-15 |
| mean-subtraction (scalar→0, perp preserved) | ≤2e-15 |
| linear readout mean/cov exactness | 0 |

The single-ReLU error decays spectrally to machine precision by ~15–21 Gauss-Hermite nodes.

**§2–§3 design regime (structured input → standard random MLP), rel-`L₂` vs Monte-Carlo:**
- shifted-Gaussian input `X ~ N(c·1, I)`: **~1–6e-3** (improves with width; near the MC floor).
- clamped-Gaussian input `X = max(0,N(μ_s,σ_s))·u + z`: **~7e-3**.

This is where clippedProp is the right tool: it propagates a coherent shifted/clamped **input** mean to
~MC accuracy.

## 7. Scope / limitations (§4 stress test)

For **weight-shifted** models `W = W' − (1/√n) 11ᵀ` with `X ~ N(0,I)`, each ReLU re-injects an `O(√n)`
shift into the next all-ones channel, which becomes strongly **skewed and one-sided** (`uᵀa ≥ 0` forces
the next pre-activation scalar to be essentially bounded). A *two-moment* scalar closure (Gaussian or
clamped) cannot represent that skew, so clippedProp — like vanilla and exact-cov `k=2` cumulant
propagation — keeps an `O(1)` error there. The first hidden layer is still accurate (~2e-3); the error
enters at deeper layers. The tool built for that regime is **`skprop`**, which conditions on the input
latent *once* and carries the conditional exactly through every linear layer (no per-layer scalar refit).
clippedProp instead targets coherent structure carried in the **state/input**, which it handles natively.

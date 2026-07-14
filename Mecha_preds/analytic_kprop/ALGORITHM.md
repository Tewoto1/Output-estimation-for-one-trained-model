# analytic_kprop — the algorithm, both variants, and asymptotic runtime

Write-up of the implementation in `core.py`: the **pre-activation fit** (`fit="pre"`,
the paper's Algorithm 7.2 with the exact-cell backend) and the **post-activation
fit** (`fit="post"`, the cheap variant that transforms the affine family through
the linear map instead of per-node moments). Formal derivations of the shared
machinery are in the spec, [`writeups/analytic_affine_kprop.pdf`](../../writeups/analytic_affine_kprop.pdf)
(equation numbers below refer to it); this document is the *implementation-level*
account plus the complexity analysis.

**Notation.** Width `n`, bulk dimension `d = n − 1`, depth `L` (hidden layers),
node/cell budget `J = num_nodes`, mixture components `m` (= retained post-ReLU
nodes of the previous layer, `m ≈ J/2 + 1 ≤ J + 1`), worker threads `W`.
Layer block (eq 4): for `M = W + e₁e₁ᵀ` and bias `b`,

```
γ = M₀₀,   r = M₀,₁: ,   u = M₁:,₀ ,   V = M₁:,₁: ,   β = b₀,   η = b₁:
X = A e₁ + B ,   Y = γA + rᵀB + β ,   C = uA + VB + η .
```

Both variants share the same **conditional K = 2 closure**: given the spike, the
bulk is summarized by its first two conditional moments and re-Gaussianized where
the algorithm says so; both use the same exact machinery for everything scalar.

---

## 1. Shared machinery (identical in both variants)

**Scalar mixture (eq 51).** Whatever the bulk representation, each component `i`
of the previous state contributes one Gaussian to the new spike pre-activation:
`Y | i ~ N(m_{Y,i}, s²_{Y,i})`, so `ν_Y = Σᵢ pᵢ N(m_{Y,i}, s²_{Y,i})` is known in
closed form. Only how `(m_{Y,i}, s²_{Y,i}, m_{C,i}, gᵢ, S_{C,i})` are *obtained*
differs between the variants.

**Grid.** `J` signed cells with an edge exactly at 0 (checklist 2), split across
the sign proportionally to mixture mass, placed by the Lloyd-Max W2 quantizer of
the exact mixture (`make_cells`, reusing `binned_kprop.binning`; `grid="uniform"`
is the cheap alternative). Cost: `O(I·m·J)` scalars, `I ≤ 1000` vectorized Lloyd
iterations (usually far fewer; measured ≈ 0.1 s independent of `d`).

**Pair stats (eqs 63–66).** For every (component `i`, cell `j`): mass
`Q_{ij} = Φ(β_{ij}) − Φ(α_{ij})`, truncated mean shift `δ_{ij}`, within-cell
variance `v_{ij}` — closed-form truncated-normal identities, fully broadcast as
`(m, J)` arrays (`_pair_stats`). Deterministic components (`s²_Y < min_prob`) go
wholly to their containing cell (eq 73). Cost: `O(mJ)` special functions.

**Exact cell reconditioning (eqs 66–72).** Within component `i`, `(Y, C)` is
jointly Gaussian, so conditioning on `Y ∈ cell j` is exact Gaussian regression +
truncated-normal moments:

```
y_{i→j} = m_{Y,i} + δ_{ij}
m_{i→j} = m_{C,i} + (gᵢ/s²_{Y,i}) δ_{ij}
S_{i→j} = S_{C,i} − gᵢgᵢᵀ/s²_{Y,i} + gᵢgᵢᵀ v_{ij}/s⁴_{Y,i}
```

merged per cell with posterior weights `η_{i|j} = pᵢQ_{ij}/w_j`:
`w_j = Σᵢ pᵢQ_{ij}`, `y_j = Σᵢ η_{i|j} y_{i→j}` (exact centroids — the readout
mean of the spike is exact for ANY grid), `m̂_j`, `Ŝ_j` by total expectation /
total covariance. `Ŝ_j` is a mixture covariance ⇒ **PSD by construction**.

**Exact Gaussian–ReLU (eqs 98–99 / section 3).** At each retained cell the bulk
slice Gaussian is pushed through ReLU with the repo's shared exact kernel
(`_utils.exact_relu_covariance`: univariate rectified moments + exact bivariate
Φ₂ via Owen's T for every off-diagonal pair). Cost: `c_sf · d²` per node where
`c_sf` is a *special-function* constant (≈ 100–300 ns/entry — two `owens_t`, several
`Φ`/`φ` passes per matrix entry) — orders of magnitude above a flop. Threaded over
nodes (`workers`; scipy ufuncs and LAPACK release the GIL).

**Zero atom (eqs 40–42).** All cells with `y_j ≤ 0` merge exactly into one atom
at `a = 0` by total expectation/total covariance of their *post-ReLU* moments
(never by pre-merging Gaussians — eq 33).

**Readout (eq 126).** `E[f(X)] = W_ro · (Σₖ pₖ aₖ ; Σₖ pₖ mₖ) + b_ro`.

---

## 2. Variant `fit="pre"` (paper Algorithm 7.2)

**State** (`AnalyticState`, eq 97): atomic post-ReLU nodes with *exact nonlinear*
per-node bulk moments

```
{ pᵢ, aᵢ, mᵢ ∈ ℝᵈ, Sᵢ ∈ ℝᵈˣᵈ }ᵢ₌₀..ₘ₋₁            (node 0 = zero atom)
```

plus `t2ᵢ` = within-component spike variance, nonzero only for the exact Gaussian
input component (a = 0, t2 = 1) — layer 1 needs no input discretization and its
affine state is exact (section 9; selftest [4]).

**One layer:**

1. *Component params* (eqs 60–61, generalized by `t2`): per component,
   `m_{Y,i} = γaᵢ + rᵀmᵢ + β`, `s²_{Y,i} = γ²t2ᵢ + rᵀSᵢr`,
   `m_{C,i} = uaᵢ + Vmᵢ + η`, `gᵢ = γt2ᵢu + VSᵢr`.
   `S_{C,i} = t2ᵢuuᵀ + VSᵢVᵀ` is **never formed per component** — see step 3.
2. *Grid, pair stats, `y_j`, `m̂_j`* — shared machinery; `m̂_j` via two `(m,J)×(m,d)`
   contractions, `O(mJd)`.
3. *Affine re-projection* (eqs 86–87). The fit needs only the two grid-weighted
   sums `T₀ = Σⱼ w_jŜ_j`, `T₁ = Σⱼ w_j y_jŜ_j`. Expanding `Ŝ_j` over components,
   every use of `S_{C,i}` is linear with scalar weights, so
   `Σᵢ cᵢ S_{C,i} = V(Σᵢ cᵢ Sᵢ)Vᵀ + (Σᵢcᵢt2ᵢ)uuᵀ` — **two aggregated congruences
   total** (`_covariance_sums`; torch-offloadable via `device`), plus `O(md²)`
   for the aggregation, rank-1 `gᵢgᵢᵀ` corrections and mean outer products.
   Mean fit: `μ₁ = Σw(y−ȳ)m̂/v_Y`, `μ₀ = Σwm̂ − μ₁ȳ`; covariance fit
   `Σ₁ = (T₁−ȳT₀)/v_Y`, `Σ₀ = T₀ − ȳΣ₁ (+ R_m` if `cov_intercept="mc"`, eq 90 —
   conserves the unconditional bulk covariance; identity checked to 1e-16,
   selftest [3]).
4. *PSD (eq 93).* `Σ(y) = Σ₀ + yΣ₁` is affine in `y`, so every interior node is a
   convex combination of the two extreme nodes: **two endpoint Cholesky
   factorizations certify the whole grid**; only Cholesky-failing nodes eigen-clip
   (`project_to_psd`, logged).
5. *ReLU at nodes* (threaded): per node assemble the dense slice
   `symmetrize(Σ₀ + y_jΣ₁)` and call the exact kernel → `(r_j, R_j)`, stored as
   the next state's `(mᵢ, Sᵢ)` stack. **No affine fit on the post-ReLU functions**
   (checklist 7) — nonlinear dependence is retained until the next reconditioning.
6. *Zero-atom merge*, renormalize, log (`E_m`, `E_S` (diagnostics), `tr R_m`,
   scalar distortion eq 134, PSD clip, per-phase times `stats["t_*"]`).

**Approximations:** conditional K = 2 closure + the *pre-activation* affine
re-projection + `O(J⁻²)`-ish scalar quantization. Exactness anchors: layer 1
exact; cell masses/centroids exact; atom merge exact; depth-1 output mean exact
up to quadrature.

## 3. Variant `fit="post"`

**Motivation.** Linearity preserves affinity *exactly*, so if the affine family is
fitted **after** ReLU, the linear step transforms four objects instead of `m`
per-node moments — and the state needs no `(m, d, d)` stack at all.

**State** (`PostAffineState`):

```
{ pₖ, aₖ }ₖ  +  m(a) = m₀ + m₁a ,  S(a) = W₀ + W₁a      (positive branch)
             +  optional EXACT zero atom (m_at, S_at)     [atom="exact", default]
```

The zero atom is the merge of *all* negative cells, so assuming it obeys the same
linearity is a genuinely different hypothesis — `atom="fit"` folds it into the
family as a data point at `a = 0` (toggle; empirically tied at small widths).

**One layer:**

1. *Family transform* (the whole "linear step"):
   `c₀ = Vm₀ + η`, `c₁ = u + Vm₁`, `g₀ = V(W₀r)`, `g₁ = V(W₁r)`,
   `s₀ = rᵀW₀r`, `s₁ = rᵀW₁r`, `G₀ = VW₀Vᵀ`, `G₁ = VW₁Vᵀ`
   (+ the same for the exact atom) — **2–3 congruences + 4 matvecs, total**.
   Component params are then *closed-form affine in `aᵢ`*:
   `m_{Y,i} = (β + rᵀm₀) + (γ + rᵀm₁)aᵢ`, `s²_{Y,i} = s₀ + s₁aᵢ (+ γ²t2ᵢ)`, etc.
2. *Grid + pair stats* — shared machinery, unchanged.
3. *Cell moments in a ≤7-vector basis.* Every within-component vector lives in
   `span B`, `B = [c₀, c₁, m_{C,at}, g₀, g₁, g_at, u] ∈ ℝᵈˣ⁷`:
   `m_{i→j} = B·w_{ij}` with scalar coords `w_{ij}` built from `(aᵢ, δ_{ij}/s²ᵢ)`.
   Hence `m̂_j = B·w̄_j` and

   ```
   Ŝ_j = α_j G₀ + β_j G₁ + ω_j G_at + B C_j Bᵀ ,
   C_j = Σᵢ η_{i|j}(w−w̄)(w−w̄)ᵀ + Σᵢ η_{i|j} hv_{ij} ĝᵢĝᵢᵀ + t2-term ,   (7×7)
   ```

   with `α, β, ω, C_j` all `O(mJ)` scalar sums. Nothing `d`-dimensional is stored
   per (i, j); the dense `Ŝ_j` is materialized one node at a time inside the ReLU
   loop (`O(d²)` assembly per node).
4. *ReLU at nodes* (slot-threaded): worker slot `k` owns nodes `j ≡ k (mod W)` and
   streams into its own accumulators `Σw_jR_j`, `Σw_jy_jR_j`, `Σw_jR_j` (negative
   side), `Σw_j‖R_j‖²` — **no `(J, d, d)` storage**; final slot-sum in fixed order
   (fp-identical to serial only at `W = 1`; allclose otherwise — selftest [10]).
   `Ŝ_j` is a mixture covariance ⇒ PSD by construction: no factorization, just an
   `O(d)` diagonal floor against roundoff (the exact kernel is defensive about
   `|ρ| → 1`).
5. *Zero-atom merge* — exact, from stored `r_j` rows + the streamed negative-side
   accumulator.
6. *Post-ReLU affine fit* (this variant's defining projection): weighted LS of
   `m(a)` on `{(y_j, r_j)}` and of `S(a)` on the streamed `R`-sums over positive
   nodes (+ the atom point when `atom="fit"`), with the same `"mc"`
   moment-conservative intercept `W₀ += R_m`. LS orthogonality conserves the
   weighted mean exactly ⇒ depth-1 output mean is pure quadrature (4e-8 at 40
   nodes; selftest [10]). `E_m`, `E_S` computed streaming.

**Approximations:** conditional K = 2 closure + the *post-activation* affine
projection — exactly the projection the paper's checklist item 7 avoids, because
`r(a) = m_ρ(μ(a), Σ(a))` is nonlinear in `a` (Φ factors, strongest near `a ≈ 0` —
hence the atom toggle). Empirically at parity with `fit="pre"` or slightly better
at n = 48–1024; the reconditioned *pre*-activation is never projected at all
(cells feed ReLU exactly), which is why depth-1 is cleaner than in `fit="pre"`.

---

## 4. Asymptotic runtime and memory

Per hidden layer, with `m ≈ J` and `c_sf` the special-function cost constant
(`c_sf ≫` 1 flop; empirically the whole game):

| phase | `fit="pre"` | `fit="post"` |
|---|---|---|
| linear/family transform | `O(md²)` aggregation | `O(d²)` matvecs |
| congruences `V·Vᵀ` | **2** → `O(d³)` | **2–3** → `O(d³)` |
| grid (Lloyd-Max) | `O(I·mJ)` | `O(I·mJ)` |
| pair stats | `O(mJ)` sf | `O(mJ)` sf |
| cell moments + fit | `O(mJd + md² + Jd²)` | `O(mJ·7² + Jd²)` assembly |
| PSD | 2 Cholesky `O(d³)` (+ rare eigh) | none (`O(d)` diag floor/node) |
| exact ReLU kernel | `J·c_sf·d²` | `J·c_sf·d²` |
| atom merge / new state | `O(Jd²)` | `O(Jd²)` |
| **total / layer** | `O(J·c_sf·d² + d³ + md²)` | `O(J·c_sf·d² + d³)` |
| **state memory** | `O(J·d²)` *(the stack)* | `O(d²)` *(+ `O(W·d²)` transient)* |
| diagnostics (`E_S`) | + `J` congruences `O(Jd³)` | streaming, ~free |

Whole network: × `L`. Both variants are asymptotically `Θ(L·J·c_sf·d²)` in time —
**the exact bivariate ReLU kernel dominates everything** (measured ≈ 95% of
runtime at n = 1024; all propagation machinery together ≈ 0.3 s). The `d³`
congruence terms have small constants and only match the kernel term when
`d ≳ J·c_sf/2` (far beyond these widths on CPU; `device="cuda"` offloads them
anyway). The variants differ asymptotically in **memory** — `O(Jd²)` vs `O(d²)` —
and in constants (post also skips per-node factorizations and the `(J,d,d)`
memory traffic, measured ~1.7× faster at n = 1024 and able to run n = 2048 in a
3.9 GB sandbox where pre is OOM-killed; at n = 4096, J = 80: pre ≈ 34 GB/run vs
post ≈ 1–3 GB/run, which is what lets the A100 script keep 10 seeds fully
parallel).

For reference, the **binned** companion at bin budget `B`: `O(B·c_sf·d² + B·d³)`
per layer (one congruence *per bin*) + `O(B²)` transition scalars, memory
`O(B·d²)` — the analytic variants remove the `B·d³` term (pre) and then the
`B·d²` memory (post).

Measured (depth 2, budget 40, 4-core box; `stats["t_*"]`):

| n | pre total | post total | pre `t_relu` | post `t_relu` | pre peak RAM | post peak RAM |
|---|---|---|---|---|---|---|
| 256 | 0.62 s | 0.33 s | 0.51 | 0.24 | +0.11 GB | ~0 |
| 512 | 2.15 s | 1.60 s | 1.97 | 1.48 | +0.34 GB | ~0 |
| 1024 | 8.52 s | 5.03 s | 8.11 | 4.78 | +0.80 GB | ~0 |
| 2048 | *OOM* | 18.2 s | — | — | ≳10 GB | ≈2 GB |

Consequently: to make either variant materially faster, optimize the shared
kernel (`_utils.exact_relu_covariance`; e.g. upper-triangle-only evaluation,
~1.7×), or drop `num_nodes` toward the accuracy knee (~6–10), or use the
`bulk_relu="gain"` backend — not the propagation.

---

## 5. Where things live

| piece | code |
|---|---|
| pre state / layer | `AnalyticState`, `analytic_layer_update` |
| post state / layer | `PostAffineState`, `analytic_layer_update_post` |
| covariance-sum trick (pre) | `_covariance_sums` (+ `percell_bulk_moments` reference) |
| pair stats / grid | `_pair_stats`, `make_cells`, `split_node_budget` |
| drivers / knobs | `run_analytic_kprop_k2(fit=, atom=, workers=, device=, …)`, `adapter.py` |
| verification | `selftest.py` (10 groups: machine-precision identities, depth-1 closed forms, MC, threading/device parity, post variant) |
| scale experiment | `experiments/analytic_kprop/binning_scaling_experiment.py` (`--fit pre|post|both`) |
